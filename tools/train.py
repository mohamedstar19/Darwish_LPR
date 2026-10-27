"""Train the plate detector or the character reader.

    python tools/train.py plates --data datasets/yolo/plates/data.yaml
    python tools/train.py chars  --data datasets/yolo/chars/data.yaml

Add your own camera images to the same folders (YOLO format) and retrain to
adapt the models to your cameras, angles and lighting.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PRESETS = {
    # Never flip characters horizontally: it changes their shape and order.
    "chars": dict(imgsz=256, epochs=80, batch=32, fliplr=0.0, mosaic=0.5, degrees=3, scale=0.3,
                  translate=0.05, hsv_v=0.5, patience=20, out="models/egypt_chars.pt"),
    "plates": dict(imgsz=416, epochs=30, batch=16, patience=10, out="models/egypt_plate.pt"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=PRESETS)
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="yolov8n.pt", help="starting weights (or an existing model to fine-tune)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", default=None, help="0 = first GPU, cpu = CPU")
    a = p.parse_args()

    from ultralytics import YOLO

    opts = dict(PRESETS[a.task])
    out = Path(opts.pop("out"))
    if a.epochs:
        opts["epochs"] = a.epochs
    model = YOLO(a.base)
    model.train(data=a.data, device=a.device, name=a.task, exist_ok=True, **opts)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model.trainer.best, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
