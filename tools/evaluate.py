"""Measure end-to-end accuracy on the EALPR validation vehicles.

    python tools/evaluate.py datasets/EALPR

Runs plate detection + character reading on each validation vehicle image (cars
neither model trained on) and compares with the labelled plate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from lpr.config import OcrConfig, DetectionConfig, resolve_device, resolve_model  # noqa: E402
from lpr.detector import PlateLocator  # noqa: E402
from lpr.ocr import Char, PlateReader, assemble  # noqa: E402
from lpr.plate_text import normalize_text  # noqa: E402
from prepare_ealpr import is_val  # noqa: E402


def truth(label: Path, names: list[str]) -> str | None:
    chars = []
    for line in label.read_text().split("\n"):
        if line.strip():
            c, x, y, w, h = line.split()
            x, y, w, h = map(float, (x, y, w, h))
            chars.append(Char(normalize_text(names[int(c)]), 1.0, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    read = assemble(chars, OcrConfig().rules)
    return read.plate.key if read else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ealpr", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--show", type=int, default=15, help="print this many mistakes")
    a = p.parse_args()

    det, ocr = DetectionConfig(), OcrConfig()
    device = resolve_device(a.device)
    locator = PlateLocator(str(resolve_model(det.plate_model)), device, det.plate_conf, det.plate_imgsz)
    reader = PlateReader(str(resolve_model(ocr.char_model)), device, ocr.char_conf, ocr.rules, ocr.imgsz)

    lab_dir = a.ealpr / "EALPR- LP characters dataset" / "Characters Labeling"
    names = (lab_dir / "classes.txt").read_text(encoding="utf-8").split()
    veh_dir = a.ealpr / "EALPR Vechicles dataset" / "Vehicles"

    n = found = exact = no_truth = 0
    char_ok = char_total = 0
    mistakes = []
    t0 = time.time()
    for label in sorted(lab_dir.glob("*_license_plate_1.txt")):
        vid = label.stem.split("_")[0]
        if not is_val(vid):
            continue
        img_path = veh_dir / f"{vid}.jpg"
        gt = truth(label, names)
        if gt is None or not img_path.exists():
            no_truth += 1
            continue
        img = cv2.imread(str(img_path))
        n += 1
        boxes = locator.locate(img)
        if boxes:
            found += 1
        read = reader.read(boxes[0][0].crop(img)) if boxes else None
        pred = read.plate.key if read else ""
        exact += pred == gt
        char_total += len(gt)
        char_ok += sum(1 for x, y in zip(pred, gt) if x == y) if len(pred) == len(gt) else 0
        if pred != gt and len(mistakes) < a.show:
            mistakes.append(f"  {vid}: truth {gt}  read {pred or '—'}")
    dt = (time.time() - t0) / max(n, 1)
    print(f"validation vehicles: {n} (skipped {no_truth} without a full label)")
    print(f"plate found:        {found / n:.1%}")
    print(f"plate read exactly: {exact / n:.1%}   (single image, no multi-frame voting)")
    print(f"characters correct: {char_ok / max(char_total, 1):.1%}")
    print(f"speed:              {dt * 1000:.0f} ms per vehicle on {device}")
    if mistakes:
        print("mistakes:")
        print("\n".join(mistakes))


if __name__ == "__main__":
    main()
