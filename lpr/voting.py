"""Combine many noisy per-frame reads of one tracked vehicle into one plate event."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plate_text import Plate


@dataclass
class _Candidate:
    plate: Plate
    reads: int = 0
    score: float = 0.0
    best_conf: float = 0.0
    best_vehicle: np.ndarray | None = None
    best_plate: np.ndarray | None = None
    first_ts: float = 0.0


@dataclass
class _Track:
    camera_id: str
    track_id: int
    kind: str
    last_seen: float
    candidates: dict[str, _Candidate] = field(default_factory=dict)
    emitted: bool = False


@dataclass
class PlateEvent:
    camera_id: str
    track_id: int
    vehicle_type: str
    plate: Plate
    reads: int
    confidence: float  # mean OCR confidence of the winning reads
    ts: float  # time of the first read of the winning plate
    vehicle_img: np.ndarray | None
    plate_img: np.ndarray | None


class PlateVoter:
    def __init__(self, min_reads: int = 2, confirm_reads: int = 4, track_timeout_sec: float = 2.0,
                 dedup_window_sec: float = 60.0):
        self.min_reads = min_reads
        self.confirm_reads = confirm_reads
        self.track_timeout = track_timeout_sec
        self.dedup_window = dedup_window_sec
        self._tracks: dict[tuple[str, int], _Track] = {}
        self._last_emit: dict[tuple[str, str], float] = {}

    def seen(self, camera_id: str, track_id: int, kind: str, ts: float) -> None:
        """Mark a track as still visible (even when no plate was read this frame)."""
        key = (camera_id, track_id)
        tr = self._tracks.get(key)
        if tr is None:
            self._tracks[key] = _Track(camera_id, track_id, kind, ts)
        else:
            tr.last_seen = max(tr.last_seen, ts)

    def add(self, camera_id: str, track_id: int, kind: str, plate: Plate, conf: float, ts: float,
            vehicle_img: np.ndarray | None = None, plate_img: np.ndarray | None = None) -> None:
        self.seen(camera_id, track_id, kind, ts)
        tr = self._tracks[(camera_id, track_id)]
        cand = tr.candidates.get(plate.key)
        if cand is None:
            cand = tr.candidates[plate.key] = _Candidate(plate=plate, first_ts=ts)
        cand.reads += 1
        cand.score += conf
        if conf >= cand.best_conf:
            cand.best_conf = conf
            cand.best_vehicle = vehicle_img
            cand.best_plate = plate_img

    def collect(self, now: float, flush: bool = False) -> list[PlateEvent]:
        """Return finished plate events.

        A track emits early once one plate reaches `confirm_reads`; otherwise it
        emits when the vehicle leaves (no update for `track_timeout_sec`) if the
        winner has at least `min_reads`. `flush=True` ends every track (end of file).
        """
        events = []
        for key, tr in list(self._tracks.items()):
            ended = flush or now - tr.last_seen > self.track_timeout
            if not tr.emitted and tr.candidates:
                winner = max(tr.candidates.values(), key=lambda c: (c.score, c.reads))
                ready = winner.reads >= self.confirm_reads or (ended and winner.reads >= self.min_reads)
                if ready:
                    tr.emitted = True
                    event = self._make_event(tr, winner)
                    if event is not None:
                        events.append(event)
            if ended:
                del self._tracks[key]
        self._prune(now)
        return events

    def _make_event(self, tr: _Track, c: _Candidate) -> PlateEvent | None:
        dkey = (tr.camera_id, c.plate.key)
        last = self._last_emit.get(dkey)
        if last is not None and c.first_ts - last < self.dedup_window:
            self._last_emit[dkey] = c.first_ts
            return None
        self._last_emit[dkey] = c.first_ts
        return PlateEvent(
            camera_id=tr.camera_id,
            track_id=tr.track_id,
            vehicle_type=tr.kind,
            plate=c.plate,
            reads=c.reads,
            confidence=round(c.score / c.reads, 3),
            ts=c.first_ts,
            vehicle_img=c.best_vehicle,
            plate_img=c.best_plate,
        )

    def _prune(self, now: float) -> None:
        horizon = now - max(self.dedup_window, 1.0) * 2
        for k, ts in list(self._last_emit.items()):
            if ts < horizon:
                del self._last_emit[k]

    def active_tracks(self) -> int:
        return len(self._tracks)
