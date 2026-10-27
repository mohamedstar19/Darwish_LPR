"""Read an Egyptian plate with a YOLO model that detects each letter and digit.

General OCR engines (EasyOCR) scored ~2% exact reads on real Egyptian plates in
our tests, so the plate is read as 27 object classes instead (17 letters + 10
digits) and the characters are put in order by position.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from statistics import median

import numpy as np

from .plate_text import ARABIC_DIGITS, PLATE_LETTERS, Plate, PlateRules, normalize_text, validate


@dataclass
class Char:
    text: str  # one canonical letter or Arabic-Indic digit
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class PlateRead:
    plate: Plate
    conf: float


def assemble(chars: list[Char], rules: PlateRules) -> PlateRead | None:
    """Turn detected characters into a plate.

    Only the main row is kept (the "مصر EGYPT" band and stray detections sit
    above or below it and are smaller). Letters read right-to-left, digits
    left-to-right, as printed on Egyptian plates.
    """
    chars = [c for c in chars if c.text]
    if not chars:
        return None
    med_h = median(c.h for c in chars)
    chars = [c for c in chars if c.h >= 0.55 * med_h]
    med_cy = median(c.cy for c in chars)
    row = [c for c in chars if abs(c.cy - med_cy) <= 0.6 * med_h]
    letters = [c for c in row if c.text in PLATE_LETTERS]
    digits = [c for c in row if c.text in ARABIC_DIGITS]
    plate = validate(
        "".join(c.text for c in sorted(letters, key=lambda c: -c.cx)),
        "".join(c.text for c in sorted(digits, key=lambda c: c.cx)),
        rules,
    )
    if plate is None:
        return None
    used = letters + digits
    return PlateRead(plate=plate, conf=float(np.mean([c.conf for c in used])))


class PlateReader:
    """Character-detector reader. One instance is shared by all cameras."""

    def __init__(self, model_path: str, device: str, conf: float, rules: PlateRules, imgsz: int = 256):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.names = {int(k): normalize_text(v) for k, v in self.model.names.items()}
        self.device = device
        self.conf = conf
        self.rules = rules
        self.imgsz = imgsz
        self._lock = threading.Lock()

    def chars(self, plate_bgr: np.ndarray) -> list[Char]:
        if plate_bgr is None or plate_bgr.size == 0 or min(plate_bgr.shape[:2]) < 8:
            return []
        with self._lock:
            result = self.model.predict(plate_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                        agnostic_nms=True, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for (x1, y1, x2, y2), c, cf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int),
                                           boxes.conf.cpu().numpy()):
            out.append(Char(self.names.get(int(c), ""), float(cf), float(x1), float(y1), float(x2), float(y2)))
        return out

    def read(self, plate_bgr: np.ndarray) -> PlateRead | None:
        return assemble(self.chars(plate_bgr), self.rules)
