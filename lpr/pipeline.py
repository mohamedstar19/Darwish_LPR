"""Per-camera processing loop and the engine that runs all cameras."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections import deque
from typing import Callable

import cv2
import numpy as np

from .camera import FileSource, LiveSource
from .config import AppConfig, CameraConfig, mask_url, resolve_device, resolve_model
from .detector import PlateLocator, Vehicle, VehicleTracker
from .ocr import PlateReader
from .storage import Storage
from .voting import PlateEvent, PlateVoter

log = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]


class CameraWorker:
    def __init__(self, cam: CameraConfig, cfg: AppConfig, device: str, reader: PlateReader,
                 on_event: Callable[[CameraConfig, PlateEvent], None]):
        self.cam = cam
        self.cfg = cfg
        self.reader = reader
        self.on_event = on_event
        det = cfg.detection
        self.vehicles = VehicleTracker(det.vehicle_model, device, det.vehicle_conf, det.imgsz, det.tracker)
        self.plates = PlateLocator(str(resolve_model(det.plate_model)), device, det.plate_conf, det.plate_imgsz)
        v = cfg.voting
        self.voter = PlateVoter(v.min_reads, v.confirm_reads, v.track_timeout_sec, v.dedup_window_sec)
        if cam.is_file:
            self.source = FileSource(str(cam.source), cam.process_fps)
        else:
            self.source = LiveSource(cam.source, name=cam.id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"worker-{cam.id}", daemon=True)
        self._preview_lock = threading.Lock()
        self._preview: bytes | None = None
        self._labels: dict[int, str] = {}  # track_id -> last plate shown on preview
        self._times: deque[float] = deque(maxlen=30)
        self.processed_frames = 0
        self.last_error = ""

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.source.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)  # let the worker leave read() before releasing the capture
        self.source.stop()

    def join(self) -> None:
        self._thread.join()

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive()

    # ---- status ----------------------------------------------------------

    def status(self) -> dict:
        fps = 0.0
        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            fps = (len(self._times) - 1) / span if span > 0 else 0.0
        return {
            "id": self.cam.id,
            "name": self.cam.name,
            "source": mask_url(self.cam.source),
            "online": bool(self.source.online),
            "resolution": f"{self.source.width}x{self.source.height}" if self.source.width else "",
            "processing_fps": round(fps, 1),
            "target_fps": self.cam.process_fps,
            "last_frame_age": round(time.time() - self.source.last_frame_ts, 1) if self.source.last_frame_ts else None,
            "active_tracks": self.voter.active_tracks(),
            "error": self.source.last_error or self.last_error,
            "live_reads": sorted(set(self._labels.values())),
        }

    def preview_jpeg(self) -> bytes | None:
        with self._preview_lock:
            return self._preview

    # ---- processing ------------------------------------------------------

    def _run(self) -> None:
        interval = 1.0 / self.cam.process_fps
        is_file = isinstance(self.source, FileSource)
        while not self._stop.is_set():
            started = time.time()
            frame = self.source.read(timeout=1.0)
            if frame is None:
                if is_file and self.source.finished:
                    self._emit(self.voter.collect(time.time(), flush=True))
                    log.info("[%s] end of file", self.cam.id)
                    return
                self._emit(self.voter.collect(time.time()))
                continue
            try:
                self._process(frame.image, frame.ts)
                self.last_error = ""
            except Exception as exc:  # keep the camera alive on unexpected errors
                self.last_error = str(exc)[:200]
                log.exception("[%s] processing error", self.cam.id)
            self._times.append(time.time())
            self.processed_frames += 1
            if not is_file:
                self._stop.wait(max(0.0, interval - (time.time() - started)))

    def _in_roi(self, v: Vehicle, w: int, h: int) -> bool:
        if self.cam.roi is None:
            return True
        x1, y1, x2, y2 = self.cam.roi
        cx = (v.box.x1 + v.box.x2) / 2 / w
        cy = (v.box.y1 + v.box.y2) / 2 / h
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _process(self, image: np.ndarray, ts: float) -> None:
        h, w = image.shape[:2]
        vehicles = self.vehicles.track(image)
        drawn: list[tuple[Vehicle, list, bool]] = []
        for v in vehicles:
            self.voter.seen(self.cam.id, v.track_id, v.kind, ts)
            usable = v.box.w >= self.cam.min_vehicle_width and self._in_roi(v, w, h)
            plate_boxes = []
            if usable:
                vehicle_img = v.box.crop(image)
                for pbox, _pconf in self.plates.locate(vehicle_img):
                    plate_boxes.append(pbox)
                    plate_img = pbox.crop(vehicle_img)
                    read = self.reader.read(plate_img)
                    if read is not None:
                        self.voter.add(self.cam.id, v.track_id, v.kind, read.plate, read.conf, ts,
                                       vehicle_img=vehicle_img.copy(), plate_img=plate_img.copy())
                        self._labels[v.track_id] = read.plate.display
            drawn.append((v, plate_boxes, usable))
        self._emit(self.voter.collect(ts))
        self._update_preview(image, drawn)

    def _emit(self, events: list[PlateEvent]) -> None:
        for ev in events:
            try:
                self.on_event(self.cam, ev)
            except Exception:
                log.exception("[%s] failed to store event", self.cam.id)

    def _update_preview(self, image: np.ndarray, drawn) -> None:
        # Drawing happens on a copy, after OCR, so annotations never get read as plates.
        canvas = image.copy()
        h, w = canvas.shape[:2]
        if self.cam.roi is not None:
            x1, y1, x2, y2 = self.cam.roi
            cv2.rectangle(canvas, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 200, 255), 2)
        live_ids = set()
        for v, plate_boxes, usable in drawn:
            live_ids.add(v.track_id)
            color = (60, 200, 60) if usable else (160, 160, 160)
            cv2.rectangle(canvas, (v.box.x1, v.box.y1), (v.box.x2, v.box.y2), color, 2)
            cv2.putText(canvas, f"#{v.track_id} {v.kind}", (v.box.x1, max(15, v.box.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for p in plate_boxes:
                cv2.rectangle(canvas, (v.box.x1 + p.x1, v.box.y1 + p.y1), (v.box.x1 + p.x2, v.box.y1 + p.y2),
                              (0, 0, 255), 2)
        # Plate text is shown in the web UI (OpenCV fonts cannot render Arabic).
        self._labels = {k: val for k, val in self._labels.items() if k in live_ids}
        if w > 1280:
            canvas = cv2.resize(canvas, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._preview_lock:
                self._preview = buf.tobytes()


class Engine:
    """Owns the shared OCR model, storage and all camera workers."""

    def __init__(self, cfg: AppConfig, storage: Storage):
        self.cfg = cfg
        self.storage = storage
        self.device = resolve_device(cfg.device)
        self._subscribers: list[EventCallback] = []
        self._sub_lock = threading.Lock()
        log.info("Loading models on %s", self.device)
        for path in (cfg.detection.plate_model, cfg.ocr.char_model):
            if not resolve_model(path).is_file():
                raise FileNotFoundError(f"Model not found: {path} (see docs/models.md)")
        self.reader = PlateReader(str(resolve_model(cfg.ocr.char_model)), self.device, cfg.ocr.char_conf,
                                  cfg.ocr.rules, cfg.ocr.imgsz)
        self.workers: dict[str, CameraWorker] = {}
        for cam in cfg.cameras:
            if cam.enabled:
                self.workers[cam.id] = CameraWorker(cam, cfg, self.device, self.reader, self._handle_event)
        self._stop = threading.Event()
        self._janitor = threading.Thread(target=self._retention_loop, name="retention", daemon=True)

    def subscribe(self, cb: EventCallback) -> None:
        with self._sub_lock:
            self._subscribers.append(cb)

    def start(self) -> None:
        for w in self.workers.values():
            w.start()
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        for w in self.workers.values():
            w.stop()

    def _retention_loop(self) -> None:
        while not self._stop.is_set():
            try:
                n = self.storage.purge_older_than(self.cfg.storage.retention_days)
                if n:
                    log.info("Retention: deleted %d old events", n)
            except Exception:
                log.exception("Retention cleanup failed")
            self._stop.wait(3600)

    def _handle_event(self, cam: CameraConfig, ev: PlateEvent) -> None:
        watch = self.storage.watch_lookup(ev.plate.key)
        vehicle_rel = self.storage.save_snapshot(ev.vehicle_img, cam.id, ev.ts, ev.track_id, "vehicle")
        plate_rel = self.storage.save_snapshot(ev.plate_img, cam.id, ev.ts, ev.track_id, "plate")
        record = self.storage.add_event(
            ts=ev.ts,
            camera_id=cam.id,
            camera_name=cam.name,
            plate_key=ev.plate.key,
            plate_display=ev.plate.display,
            letters=ev.plate.letters,
            digits=ev.plate.digits,
            confidence=ev.confidence,
            reads=ev.reads,
            vehicle_type=ev.vehicle_type,
            track_id=ev.track_id,
            vehicle_img=vehicle_rel,
            plate_img=plate_rel,
            watch_kind=watch["kind"] if watch else None,
            watch_label=watch["label"] if watch else None,
        )
        log.info("[%s] plate %s (%s, %d reads, conf %.2f)%s", cam.id, ev.plate.display, ev.vehicle_type,
                 ev.reads, ev.confidence, f" WATCHLIST:{watch['kind']}" if watch else "")
        with self._sub_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(record)
            except Exception:
                log.exception("Event subscriber failed")
        alerts = self.cfg.alerts
        if alerts.webhook_url and (alerts.webhook_all_events or (watch and watch["kind"] == "alert")):
            threading.Thread(target=_post_webhook, args=(alerts.webhook_url, record), daemon=True).start()


def _post_webhook(url: str, record: dict) -> None:
    body = json.dumps({"type": "plate", "event": record}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as exc:
        log.warning("Webhook to %s failed: %s", mask_url(url), exc)
