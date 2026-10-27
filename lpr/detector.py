"""Vehicle detection + tracking, and plate localisation inside a vehicle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO class ids used by the stock YOLO models.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[max(0, self.y1):max(0, self.y2), max(0, self.x1):max(0, self.x2)]

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(self.x1, width)), max(0, min(self.y1, height)),
            max(0, min(self.x2, width)), max(0, min(self.y2, height)),
        )


@dataclass
class Vehicle:
    track_id: int
    kind: str
    conf: float
    box: Box


class VehicleTracker:
    """One instance per camera: ByteTrack state must not be shared between streams."""

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int, tracker: str):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker

    def track(self, frame: np.ndarray) -> list[Vehicle]:
        result = self.model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        h, w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        out = []
        for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, cls, confs):
            kind = VEHICLE_CLASSES.get(int(c))
            if kind is None:
                continue
            out.append(Vehicle(int(tid), kind, float(cf), Box(x1, y1, x2, y2).clip(w, h)))
        return out


class PlateLocator:
    """Find plates inside a vehicle crop with a YOLO plate detector."""

    def __init__(self, model_path: str, device: str, conf: float, imgsz: int = 320):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def locate(self, vehicle_img: np.ndarray, max_plates: int = 1) -> list[tuple[Box, float]]:
        h, w = vehicle_img.shape[:2]
        result = self.model.predict(vehicle_img, conf=self.conf, device=self.device, imgsz=self.imgsz,
                                    verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        out = []
        for (x1, y1, x2, y2), cf in zip(result.boxes.xyxy.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            box = Box(x1, y1, x2, y2).clip(w, h)
            if box.w >= 16 and box.h >= 8:
                out.append((box, float(cf)))
        out.sort(key=lambda bc: -bc[1])
        return out[:max_plates]
