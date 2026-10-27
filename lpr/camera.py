"""Frame sources: live RTSP/USB cameras (latest frame, auto-reconnect) and video files."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

# RTSP over TCP avoids the smeared/grey frames UDP gives on busy networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .config import mask_url  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    ts: float  # unix time the frame was captured (or its position in a file)
    index: int


def open_capture(source: str | int, timeout_ms: int = 10000) -> cv2.VideoCapture:
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    params = []
    for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop = getattr(cv2, name, None)
        if prop is not None:
            params += [prop, timeout_ms]
    if params:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cap


class LiveSource:
    """Reads a live stream in its own thread and keeps only the newest frame.

    Inference is slower than the camera frame rate, so old frames are dropped
    instead of queueing up (which would add ever-growing latency).
    """

    def __init__(self, source: str | int, name: str = "", reconnect_delay: float = 5.0):
        self.source = source
        self.name = name or mask_url(source)
        self.reconnect_delay = reconnect_delay
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Frame | None = None
        self._consumed_index = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self.online = False
        self.width = 0
        self.height = 0
        self.stream_fps = 0.0
        self.last_error = ""
        self.last_frame_ts = 0.0

    def start(self) -> "LiveSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            cap = open_capture(self.source)
            if not cap.isOpened():
                self.online = False
                self.last_error = "cannot open stream"
                log.warning("[%s] cannot open %s, retrying in %.0fs", self.name, mask_url(self.source), self.reconnect_delay)
                cap.release()
                self._stop.wait(self.reconnect_delay)
                continue
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.stream_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            log.info("[%s] connected %dx%d @ %.1f fps", self.name, self.width, self.height, self.stream_fps)
            failures = 0
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    failures += 1
                    if failures >= 25:
                        break
                    time.sleep(0.04)
                    continue
                failures = 0
                index += 1
                now = time.time()
                with self._cond:
                    self._frame = Frame(image=image, ts=now, index=index)
                    self.online = True
                    self.last_error = ""
                    self.last_frame_ts = now
                    self._cond.notify_all()
            cap.release()
            if not self._stop.is_set():
                self.online = False
                self.last_error = "stream lost"
                log.warning("[%s] stream lost, reconnecting in %.0fs", self.name, self.reconnect_delay)
                self._stop.wait(self.reconnect_delay)

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Return the newest frame not returned before, waiting up to `timeout`."""
        deadline = time.time() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._frame is not None and self._frame.index != self._consumed_index:
                    self._consumed_index = self._frame.index
                    return self._frame
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None

    @property
    def finished(self) -> bool:
        return False


class FileSource:
    """Plays a video file frame by frame, skipping frames to match process_fps."""

    def __init__(self, path: str, process_fps: float, start_ts: float | None = None):
        self.source = path
        self.name = os.path.basename(path)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.stream_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.step = max(1, round(self.stream_fps / process_fps))
        self.start_ts = time.time() if start_ts is None else start_ts
        self._index = 0
        self._done = False
        self.online = True
        self.last_error = ""
        self.last_frame_ts = 0.0

    def start(self) -> "FileSource":
        return self

    def stop(self) -> None:
        self.cap.release()

    def read(self, timeout: float = 1.0) -> Frame | None:
        if self._done:
            return None
        image = None
        for _ in range(self.step):
            ok, img = self.cap.read()
            if not ok:
                self._done = True
                self.online = False
                break
            self._index += 1
            image = img
        if image is None:
            return None
        ts = self.start_ts + self._index / self.stream_fps
        self.last_frame_ts = ts
        return Frame(image=image, ts=ts, index=self._index)

    @property
    def finished(self) -> bool:
        return self._done
