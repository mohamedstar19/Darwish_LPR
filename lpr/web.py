"""FastAPI dashboard + JSON API (HTTP Basic auth on every route)."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from .config import AppConfig
from .pipeline import Engine
from .plate_text import parse
from .storage import Storage

STATIC_DIR = Path(__file__).parent / "static"


class WatchIn(BaseModel):
    plate: str = Field(min_length=1, max_length=40)
    kind: Literal["alert", "allow"] = "alert"
    label: str = Field(default="", max_length=100)


class _Hub:
    """Fan out new events from worker threads to connected SSE clients."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: set[asyncio.Queue] = set()

    def publish(self, record: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        for q in list(self.queues):
            loop.call_soon_threadsafe(_put_nowait, q, record)


def _put_nowait(q: asyncio.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass  # slow client: drop, it will catch up on the next refresh


def create_app(cfg: AppConfig, engine: Engine | None, storage: Storage) -> FastAPI:
    if not cfg.web.password:
        raise ValueError("web.password must be set (plate data is personal data)")

    security = HTTPBasic(realm="Darwish LPR")

    def auth(creds: HTTPBasicCredentials = Depends(security)) -> str:
        user_ok = secrets.compare_digest(creds.username.encode(), cfg.web.username.encode())
        pass_ok = secrets.compare_digest(creds.password.encode(), cfg.web.password.encode())
        if not (user_ok and pass_ok):
            raise HTTPException(401, "Invalid credentials", headers={"WWW-Authenticate": 'Basic realm="Darwish LPR"'})
        return creds.username

    hub = _Hub()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if engine is not None:
            engine.subscribe(hub.publish)
        yield
        hub.loop = None

    app = FastAPI(title="Darwish LPR", docs_url=None, redoc_url=None, openapi_url=None,
                  dependencies=[Depends(auth)], lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    def static(name: str):
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/cameras")
    def cameras():
        if engine is None:
            return []
        return [w.status() for w in engine.workers.values()]

    @app.get("/api/stats")
    def stats():
        day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        return {"today": storage.stats(day_start), "device": engine.device if engine else None}

    @app.get("/api/events")
    def events(q: str = Query("", max_length=40), camera: str | None = Query(None, max_length=40),
               since: float | None = None, until: float | None = None,
               before_id: int | None = Query(None, ge=1), limit: int = Query(100, ge=1, le=1000)):
        return storage.list_events(q, camera, since, until, before_id, limit)

    @app.get("/api/events.csv")
    def events_csv(q: str = Query("", max_length=40), camera: str | None = Query(None, max_length=40),
                   since: float | None = None, until: float | None = None):
        rows = storage.list_events(q, camera, since, until, None, 1000)
        buf = io.StringIO()
        buf.write("﻿")  # BOM so Excel shows Arabic correctly
        writer = csv.writer(buf)
        writer.writerow(["id", "time", "camera", "plate", "vehicle", "confidence", "reads", "watchlist"])
        for r in rows:
            writer.writerow([
                r["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])), r["camera_name"],
                r["plate_display"], r["vehicle_type"], r["confidence"], r["reads"], r["watch_kind"] or "",
            ])
        return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="plates.csv"'})

    @app.get("/api/events/stream")
    async def event_stream(request: Request):
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        hub.queues.add(q)

        async def gen():
            try:
                yield ": connected\n\n"
                while not await request.is_disconnected():
                    try:
                        record = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"
            finally:
                hub.queues.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/cameras/{camera_id}/snapshot.jpg")
    def camera_snapshot(camera_id: str):
        worker = engine.workers.get(camera_id) if engine else None
        if worker is None:
            raise HTTPException(404, "Unknown camera")
        jpg = worker.preview_jpeg()
        if jpg is None:
            raise HTTPException(503, "No frame yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/live.mjpg")
    async def camera_live(camera_id: str, request: Request):
        worker = engine.workers.get(camera_id) if engine else None
        if worker is None:
            raise HTTPException(404, "Unknown camera")

        async def gen():
            last = None
            while not await request.is_disconnected():
                jpg = worker.preview_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.2)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshots/{rel:path}")
    def snapshot(rel: str):
        path = storage.snapshot_path(rel)
        if path is None:
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/watchlist")
    def watchlist():
        return storage.watchlist()

    @app.post("/api/watchlist", status_code=201)
    def watch_add(item: WatchIn):
        plate = parse(item.plate, cfg.ocr.rules)
        if plate is None:
            raise HTTPException(422, "رقم اللوحة غير صحيح — اكتب الحروف والأرقام، مثال: ب ط ص ١٢٣٤")
        return storage.watch_add(plate.key, plate.display, item.kind, item.label.strip())

    @app.delete("/api/watchlist/{watch_id}", status_code=204)
    def watch_delete(watch_id: int):
        if not storage.watch_delete(watch_id):
            raise HTTPException(404)
        return Response(status_code=204)

    return app
