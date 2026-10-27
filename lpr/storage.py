"""SQLite storage for plate events, the watchlist and snapshot images."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .plate_text import search_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    plate_key TEXT NOT NULL,
    plate_display TEXT NOT NULL,
    letters TEXT NOT NULL,
    digits TEXT NOT NULL,
    confidence REAL NOT NULL,
    reads INTEGER NOT NULL,
    vehicle_type TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle_img TEXT,
    plate_img TEXT,
    watch_kind TEXT,
    watch_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_key);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, ts);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_key TEXT NOT NULL UNIQUE,
    plate_display TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert', 'allow')),
    label TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""

EVENT_COLUMNS = (
    "id", "ts", "camera_id", "camera_name", "plate_key", "plate_display", "letters", "digits",
    "confidence", "reads", "vehicle_type", "track_id", "vehicle_img", "plate_img", "watch_kind", "watch_label",
)


class Storage:
    def __init__(self, db_path: str, snapshots_dir: str):
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- snapshots -------------------------------------------------------

    def save_snapshot(self, image: np.ndarray | None, camera_id: str, ts: float, track_id: int, kind: str) -> str | None:
        if image is None or image.size == 0:
            return None
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        stamp = datetime.fromtimestamp(ts).strftime("%H%M%S_%f")[:-3]
        rel = Path(day) / f"{camera_id}_{stamp}_{track_id}_{kind}.jpg"
        path = self.snapshots_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return rel.as_posix()

    def snapshot_path(self, rel: str) -> Path | None:
        """Resolve a stored snapshot path, refusing anything outside the snapshot dir."""
        try:
            path = (self.snapshots_dir / rel).resolve()
        except (OSError, ValueError):
            return None
        if self.snapshots_dir not in path.parents or not path.is_file():
            return None
        return path

    # ---- events ----------------------------------------------------------

    def add_event(self, **values) -> dict:
        cols = [c for c in EVENT_COLUMNS if c != "id"]
        missing = [c for c in cols if c not in values]
        if missing:
            raise ValueError(f"missing event fields: {missing}")
        with self._lock:
            cur = self._db.execute(
                f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [values[c] for c in cols],
            )
            self._db.commit()
            event_id = cur.lastrowid
        return {"id": event_id, **{c: values[c] for c in cols}}

    def list_events(self, query: str = "", camera_id: str | None = None, since: float | None = None,
                    until: float | None = None, before_id: int | None = None, limit: int = 100) -> list[dict]:
        where, args = [], []
        key = search_key(query)
        if key:
            where.append("plate_key LIKE ?")
            args.append(f"%{key}%")
        if camera_id:
            where.append("camera_id = ?")
            args.append(camera_id)
        if since is not None:
            where.append("ts >= ?")
            args.append(since)
        if until is not None:
            where.append("ts < ?")
            args.append(until)
        if before_id is not None:
            where.append("id < ?")
            args.append(before_id)
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def stats(self, since: float) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM events WHERE ts >= ?", (since,)).fetchone()[0]
            unique = self._db.execute("SELECT COUNT(DISTINCT plate_key) FROM events WHERE ts >= ?", (since,)).fetchone()[0]
            alerts = self._db.execute(
                "SELECT COUNT(*) FROM events WHERE ts >= ? AND watch_kind = 'alert'", (since,)).fetchone()[0]
        return {"events": total, "unique_plates": unique, "alerts": alerts}

    # ---- watchlist -------------------------------------------------------

    def watchlist(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM watchlist ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def watch_lookup(self, plate_key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM watchlist WHERE plate_key = ?", (plate_key,)).fetchone()
        return dict(row) if row else None

    def watch_add(self, plate_key: str, plate_display: str, kind: str, label: str) -> dict:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO watchlist (plate_key, plate_display, kind, label, created_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(plate_key) DO UPDATE SET kind = excluded.kind, label = excluded.label",
                (plate_key, plate_display, kind, label, now),
            )
            self._db.commit()
            row = self._db.execute("SELECT * FROM watchlist WHERE plate_key = ?", (plate_key,)).fetchone()
        return dict(row)

    def watch_delete(self, watch_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM watchlist WHERE id = ?", (watch_id,))
            self._db.commit()
        return cur.rowcount > 0

    # ---- retention -------------------------------------------------------

    def purge_older_than(self, days: int) -> int:
        """Delete events and their snapshots older than `days`. Returns events deleted."""
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        with self._lock:
            rows = self._db.execute(
                "SELECT vehicle_img, plate_img FROM events WHERE ts < ?", (cutoff,)).fetchall()
            self._db.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._db.commit()
        for row in rows:
            for rel in (row["vehicle_img"], row["plate_img"]):
                path = self.snapshot_path(rel) if rel else None
                if path is not None:
                    path.unlink(missing_ok=True)
        for day_dir in self.snapshots_dir.iterdir():
            if day_dir.is_dir() and not any(day_dir.iterdir()):
                day_dir.rmdir()
        return len(rows)
