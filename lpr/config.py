"""Load and validate config.yaml (with ${ENV_VAR} substitution)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml

from .plate_text import PlateRules

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    pass


def _expand_env(value):
    if isinstance(value, str):
        def repl(m):
            name = m.group(1)
            if name not in os.environ:
                raise ConfigError(f"Environment variable {name} is not set (used in config)")
            return os.environ[name]
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def mask_url(url: str) -> str:
    """Hide credentials in a stream URL before logging or showing it."""
    return re.sub(r"(//)[^@/]+@", r"\1***@", str(url))


@dataclass
class NvrConfig:
    host: str
    username: str
    password: str
    port: int = 554
    stream: str = "main"  # main | sub

    def channel_url(self, channel: int, stream: str | None = None) -> str:
        """Hikvision RTSP URL: channel 1 main stream -> /Streaming/Channels/101."""
        stream = stream or self.stream
        suffix = {"main": "01", "sub": "02"}.get(stream)
        if suffix is None:
            raise ConfigError(f"nvr.stream must be 'main' or 'sub', got {stream!r}")
        user = quote(self.username, safe="")
        pwd = quote(self.password, safe="")
        return f"rtsp://{user}:{pwd}@{self.host}:{self.port}/Streaming/Channels/{int(channel)}{suffix}"


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str | int  # rtsp/http URL, file path, or USB index
    process_fps: float = 5.0
    roi: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2 as 0..1 fractions
    min_vehicle_width: int = 80
    enabled: bool = True

    @property
    def is_file(self) -> bool:
        if isinstance(self.source, int):
            return False
        return "://" not in self.source


@dataclass
class DetectionConfig:
    vehicle_model: str = "yolov8n.pt"
    vehicle_conf: float = 0.35
    imgsz: int = 640
    plate_model: str = "models/egypt_plate.pt"  # finds the plate inside a vehicle
    plate_conf: float = 0.4
    plate_imgsz: int = 416
    tracker: str = "bytetrack.yaml"


@dataclass
class OcrConfig:
    char_model: str = "models/egypt_chars.pt"  # detects each letter/digit on the plate
    char_conf: float = 0.35
    imgsz: int = 256
    rules: PlateRules = field(default_factory=PlateRules)


@dataclass
class VotingConfig:
    min_reads: int = 2  # reads needed when the track ends
    confirm_reads: int = 4  # reads that confirm a plate immediately
    track_timeout_sec: float = 2.0
    dedup_window_sec: float = 60.0


@dataclass
class StorageConfig:
    db_path: str = "data/lpr.db"
    snapshots_dir: str = "data/snapshots"
    retention_days: int = 30


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    username: str = "admin"
    password: str = ""


@dataclass
class AlertsConfig:
    webhook_url: str | None = None
    webhook_all_events: bool = False


@dataclass
class AppConfig:
    device: str = "auto"
    cameras: list[CameraConfig] = field(default_factory=list)
    nvr: NvrConfig | None = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    web: WebConfig = field(default_factory=WebConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)


def _build(cls, data: dict | None, section: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{section}' must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown keys in '{section}': {', '.join(sorted(unknown))}")
    return cls(**data)


def _camera(raw: dict, nvr: NvrConfig | None, index: int) -> CameraConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"cameras[{index}] must be a mapping")
    raw = dict(raw)
    cam_id = str(raw.pop("id", "") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", cam_id):
        raise ConfigError(f"cameras[{index}].id must be 1-40 chars of A-Z, a-z, 0-9, _ or -")
    channel = raw.pop("channel", None)
    stream = raw.pop("stream", None)
    source = raw.pop("source", None)
    if channel is not None:
        if nvr is None:
            raise ConfigError(f"camera {cam_id}: 'channel' needs an 'nvr' section")
        if source is not None:
            raise ConfigError(f"camera {cam_id}: use either 'channel' or 'source', not both")
        source = nvr.channel_url(int(channel), stream)
    if source is None:
        raise ConfigError(f"camera {cam_id}: set 'channel' (NVR) or 'source'")
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    roi = raw.pop("roi", None)
    if roi is not None:
        if len(roi) != 4 or not all(0 <= float(v) <= 1 for v in roi):
            raise ConfigError(f"camera {cam_id}: roi must be [x1, y1, x2, y2] between 0 and 1")
        x1, y1, x2, y2 = (float(v) for v in roi)
        if x2 <= x1 or y2 <= y1:
            raise ConfigError(f"camera {cam_id}: roi x2/y2 must be greater than x1/y1")
        roi = (x1, y1, x2, y2)
    name = str(raw.pop("name", cam_id))
    cam = _build(CameraConfig, {"id": cam_id, "name": name, "source": source, "roi": roi, **raw}, f"cameras[{index}]")
    if cam.process_fps <= 0:
        raise ConfigError(f"camera {cam_id}: process_fps must be > 0")
    return cam


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path} (copy config.example.yaml to config.yaml)")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)
    return parse_config(raw)


def parse_config(raw: dict) -> AppConfig:
    allowed = {"device", "cameras", "nvr", "detection", "ocr", "voting", "storage", "web", "alerts"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")

    nvr = _build(NvrConfig, raw["nvr"], "nvr") if raw.get("nvr") else None

    ocr_raw = dict(raw.get("ocr") or {})
    rules = _build(PlateRules, ocr_raw.pop("rules", None), "ocr.rules")
    ocr = _build(OcrConfig, {**ocr_raw, "rules": rules}, "ocr")

    cameras = [_camera(c, nvr, i) for i, c in enumerate(raw.get("cameras") or [])]
    ids = [c.id for c in cameras]
    if len(ids) != len(set(ids)):
        raise ConfigError("Camera ids must be unique")

    device = str(raw.get("device", "auto"))
    if device not in ("auto", "cpu") and not re.fullmatch(r"cuda(:\d+)?", device):
        raise ConfigError("device must be auto, cpu, cuda or cuda:N")

    cfg = AppConfig(
        device=device,
        cameras=cameras,
        nvr=nvr,
        detection=_build(DetectionConfig, raw.get("detection"), "detection"),
        ocr=ocr,
        voting=_build(VotingConfig, raw.get("voting"), "voting"),
        storage=_build(StorageConfig, raw.get("storage"), "storage"),
        web=_build(WebConfig, raw.get("web"), "web"),
        alerts=_build(AlertsConfig, raw.get("alerts"), "alerts"),
    )
    if cfg.voting.min_reads < 1 or cfg.voting.confirm_reads < cfg.voting.min_reads:
        raise ConfigError("voting: need 1 <= min_reads <= confirm_reads")
    return cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_model(path: str) -> Path:
    """Find a model file relative to the working directory, then the project root."""
    p = Path(path)
    if p.is_file() or p.is_absolute():
        return p
    alt = PROJECT_ROOT / p
    return alt if alt.is_file() else p


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
