"""Convert the EALPR dataset into two YOLO datasets.

    git clone --depth 1 https://github.com/ahmedramadan96/EALPR datasets/EALPR
    python tools/prepare_ealpr.py datasets/EALPR datasets/yolo

Creates:
    datasets/yolo/plates/  vehicle images, 1 class (plate)       -> plate model
    datasets/yolo/chars/   plate crops, 27 classes (letters+digits) -> char model

The split is deterministic and by vehicle (15% validation), so tools/evaluate.py\nmeasures both models on cars neither has seen.
Cite: Youssef et al., "A New Benchmark Dataset for Egyptian License Plate
Detection and Recognition", ACIRS 2022.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def vehicle_id(stem: str) -> str:
    """"0001_license_plate_1" and "0001" both belong to vehicle "0001"."""
    return stem.split("_")[0]


def is_val(stem: str, val_frac: float = 0.15, seed: int = 42) -> bool:
    """Split by vehicle so a car's plate is never in training when the car is in validation."""
    return random.Random(f"{seed}-{vehicle_id(stem)}").random() < val_frac


def build(out: Path, name: str, img_dir: Path, lab_dir: Path, names: list[str]) -> None:
    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    ids = [p.stem for p in lab_dir.glob("*.txt")
           if p.name != "classes.txt" and p.stem in images and p.stat().st_size > 0]
    val = {i for i in ids if is_val(i)}
    for i in ids:
        part = "val" if i in val else "train"
        (out / name / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / name / "labels" / part).mkdir(parents=True, exist_ok=True)
        shutil.copy(images[i], out / name / "images" / part / images[i].name)
        shutil.copy(lab_dir / f"{i}.txt", out / name / "labels" / part / f"{i}.txt")
    lines = [f"path: {(out / name).resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f'  {k}: "{n}"' for k, n in enumerate(names)]
    (out / name / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(ids)} images ({len(val)} val) -> {out / name / 'data.yaml'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ealpr", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    veh = a.ealpr / "EALPR Vechicles dataset"  # (sic) folder name in the dataset
    chars = a.ealpr / "EALPR- LP characters dataset"
    build(a.out, "plates", veh / "Vehicles", veh / "Vehicles Labeling", ["plate"])
    names = (chars / "Characters Labeling" / "classes.txt").read_text(encoding="utf-8").split()
    build(a.out, "chars", a.ealpr / "EALPR- Plates dataset", chars / "Characters Labeling", names)


if __name__ == "__main__":
    main()
