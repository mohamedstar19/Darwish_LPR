"""Command line entry point.

    python -m lpr run        # start cameras + dashboard
    python -m lpr scan-nvr   # find which Hikvision NVR channels answer
    python -m lpr probe      # grab one frame from every configured camera
    python -m lpr video FILE # process a recorded video (no cameras needed)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

from .config import CameraConfig, ConfigError, load_config, mask_url


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("ultralytics", "PIL", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(args) -> int:
    import uvicorn

    from .pipeline import Engine
    from .storage import Storage
    from .web import create_app

    cfg = load_config(args.config)
    if not cfg.cameras:
        print("No cameras in config. Add cameras (see config.example.yaml).", file=sys.stderr)
        return 2
    if not cfg.web.password:
        print("Set web.password in config.yaml (e.g. password: ${LPR_WEB_PASSWORD}).", file=sys.stderr)
        return 2
    storage = Storage(cfg.storage.db_path, cfg.storage.snapshots_dir)
    engine = Engine(cfg, storage)
    app = create_app(cfg, engine, storage)
    engine.start()
    logging.getLogger("lpr").info("Dashboard: http://%s:%d", cfg.web.host, cfg.web.port)
    try:
        uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="warning")
    finally:
        engine.stop()
        storage.close()
    return 0


def _grab(source, timeout: float = 15.0):
    from .camera import LiveSource

    src = LiveSource(source, reconnect_delay=timeout).start()
    try:
        frame = src.read(timeout=timeout)
        return frame, src
    finally:
        src.stop()


def cmd_probe(args) -> int:
    import cv2

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for cam in cfg.cameras:
        if cam.is_file:
            print(f"- {cam.id}: file source {cam.source} (skipped)")
            continue
        frame, src = _grab(cam.source)
        if frame is None:
            failed += 1
            print(f"✗ {cam.id} ({cam.name}): no frame from {mask_url(cam.source)}")
            continue
        h, w = frame.image.shape[:2]
        path = out_dir / f"{cam.id}.jpg"
        cv2.imwrite(str(path), frame.image)
        print(f"✓ {cam.id} ({cam.name}): {w}x{h} @ {src.stream_fps:.0f} fps -> {path}")
    return 1 if failed else 0


def cmd_scan_nvr(args) -> int:
    cfg = load_config(args.config)
    if cfg.nvr is None:
        print("Add an 'nvr' section to config.yaml first.", file=sys.stderr)
        return 2
    found = 0
    for ch in range(1, args.channels + 1):
        url = cfg.nvr.channel_url(ch, args.stream)
        frame, src = _grab(url, timeout=args.timeout)
        if frame is None:
            print(f"  channel {ch:>2}: —")
            continue
        found += 1
        h, w = frame.image.shape[:2]
        print(f"✓ channel {ch:>2}: {w}x{h} @ {src.stream_fps:.0f} fps   ({mask_url(url)})")
    print(f"\n{found} channel(s) answered. Add them under 'cameras' with 'channel: N'.")
    return 0 if found else 1


def cmd_video(args) -> int:
    from .pipeline import Engine
    from .storage import Storage

    cfg = load_config(args.config)
    cam = CameraConfig(id=args.camera_id, name=args.name or Path(args.file).name, source=str(args.file),
                       process_fps=args.fps, min_vehicle_width=args.min_width)
    cfg = dataclasses.replace(cfg, cameras=[cam])
    storage = Storage(cfg.storage.db_path, cfg.storage.snapshots_dir)
    engine = Engine(cfg, storage)
    results = []

    def show(rec: dict) -> None:
        results.append(rec)
        t = time.strftime("%H:%M:%S", time.localtime(rec["ts"]))
        print(f"{t}  {rec['plate_display']:<16} {rec['vehicle_type']:<10} conf={rec['confidence']:.2f} reads={rec['reads']}")

    engine.subscribe(show)
    started = time.time()
    engine.start()
    try:
        for w in engine.workers.values():
            w.join()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        storage.close()
    print(f"\n{len(results)} plate(s) in {time.time() - started:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {args.json}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m lpr", description="Darwish LPR — Egyptian licence plate reader")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run", help="start cameras and the web dashboard")

    sp = sub.add_parser("probe", help="grab one frame from every camera")
    sp.add_argument("--out", default="data/probe")

    sp = sub.add_parser("scan-nvr", help="find which NVR channels answer over RTSP")
    sp.add_argument("--channels", type=int, default=16)
    sp.add_argument("--stream", choices=["main", "sub"], default="sub")
    sp.add_argument("--timeout", type=float, default=8.0)

    sp = sub.add_parser("video", help="process a recorded video file")
    sp.add_argument("file")
    sp.add_argument("--fps", type=float, default=8.0, help="frames per second to analyse")
    sp.add_argument("--min-width", type=int, default=80, help="ignore vehicles narrower than this (px)")
    sp.add_argument("--camera-id", default="video")
    sp.add_argument("--name", default=None)
    sp.add_argument("--json", default=None, help="also save results to this JSON file")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    handlers = {"run": cmd_run, "probe": cmd_probe, "scan-nvr": cmd_scan_nvr, "video": cmd_video, None: cmd_run}
    try:
        return handlers[args.cmd](args)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
