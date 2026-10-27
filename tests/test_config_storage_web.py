import time

import numpy as np
import pytest

from lpr.config import ConfigError, mask_url, parse_config
from lpr.plate_text import parse
from lpr.storage import Storage


def base(**over):
    raw = {
        "nvr": {"host": "10.0.0.5", "username": "lpr", "password": "p@ss:/1"},
        "cameras": [{"id": "gate", "name": "البوابة", "channel": 3}],
        "web": {"password": "secret"},
    }
    raw.update(over)
    return raw


def test_hikvision_channel_url_and_password_encoding():
    cfg = parse_config(base())
    assert cfg.cameras[0].source == "rtsp://lpr:p%40ss%3A%2F1@10.0.0.5:554/Streaming/Channels/301"
    sub = parse_config(base(cameras=[{"id": "g", "channel": 12, "stream": "sub"}]))
    assert sub.cameras[0].source.endswith("/Streaming/Channels/1202")


def test_mask_url():
    assert mask_url("rtsp://u:p@h:554/x") == "rtsp://***@h:554/x"


def test_config_errors():
    with pytest.raises(ConfigError):
        parse_config(base(cameras=[{"id": "a", "channel": 1}, {"id": "a", "channel": 2}]))
    with pytest.raises(ConfigError):
        parse_config(base(cameras=[{"id": "a", "channel": 1, "roi": [0.5, 0, 0.2, 1]}]))
    with pytest.raises(ConfigError):
        parse_config(base(cameras=[{"id": "bad id", "channel": 1}]))
    with pytest.raises(ConfigError):
        parse_config(base(detection={"typo": 1}))
    with pytest.raises(ConfigError):
        parse_config({"cameras": [{"id": "x", "channel": 1}]})  # channel without nvr


def test_env_substitution(tmp_path, monkeypatch):
    from lpr.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("web:\n  password: ${LPR_TEST_PW}\n", encoding="utf-8")
    monkeypatch.setenv("LPR_TEST_PW", "xyz")
    assert load_config(p).web.password == "xyz"
    monkeypatch.delenv("LPR_TEST_PW")
    with pytest.raises(ConfigError):
        load_config(p)


def _event(storage, plate, ts, cam="gate", img=None):
    p = parse(plate)
    rel = storage.save_snapshot(img, cam, ts, 1, "vehicle")
    return storage.add_event(
        ts=ts, camera_id=cam, camera_name=cam, plate_key=p.key, plate_display=p.display,
        letters=p.letters, digits=p.digits, confidence=0.9, reads=3, vehicle_type="car", track_id=1,
        vehicle_img=rel, plate_img=None, watch_kind=None, watch_label=None,
    )


def test_storage_search_watchlist_and_retention(tmp_path):
    s = Storage(str(tmp_path / "db.sqlite"), str(tmp_path / "snaps"))
    now = time.time()
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    old = _event(s, "ب ط ص ١٢٣٤", now - 40 * 86400, img=img)
    _event(s, "س ع ٥٦٧", now)
    assert [e["plate_display"] for e in s.list_events("12")] == ["ب ط ص ١٢٣٤"]
    assert len(s.list_events("")) == 2
    assert s.snapshot_path(old["vehicle_img"]) is not None
    assert s.snapshot_path("../db.sqlite") is None

    w = s.watch_add("سع٥٦٧", "س ع ٥٦٧", "alert", "مطلوب")
    assert s.watch_lookup("سع٥٦٧")["label"] == "مطلوب"
    assert s.watch_delete(w["id"]) and s.watch_lookup("سع٥٦٧") is None

    assert s.purge_older_than(30) == 1
    assert len(s.list_events("")) == 1
    assert s.snapshot_path(old["vehicle_img"]) is None
    s.close()


def test_web_api_auth_and_watchlist(tmp_path):
    from fastapi.testclient import TestClient

    from lpr.web import create_app

    cfg = parse_config(base())
    s = Storage(str(tmp_path / "db.sqlite"), str(tmp_path / "snaps"))
    _event(s, "ب ط ص ١٢٣٤", time.time())
    app = create_app(cfg, None, s)
    with TestClient(app) as client:
        assert client.get("/api/events").status_code == 401
        assert client.get("/api/events", auth=("admin", "wrong")).status_code == 401
        auth = ("admin", "secret")
        assert client.get("/", auth=auth).status_code == 200
        rows = client.get("/api/events", params={"q": "بط"}, auth=auth).json()
        assert rows[0]["plate_display"] == "ب ط ص ١٢٣٤"
        assert client.post("/api/watchlist", json={"plate": "xyz"}, auth=auth).status_code == 422
        r = client.post("/api/watchlist", json={"plate": "ب ط ص 1234", "label": "test"}, auth=auth)
        assert r.status_code == 201 and r.json()["plate_key"] == "بطص١٢٣٤"
        assert client.delete(f"/api/watchlist/{r.json()['id']}", auth=auth).status_code == 204
        assert client.get("/snapshots/..%2Fdb.sqlite", auth=auth).status_code == 404
        assert client.get("/static/..%2F__init__.py", auth=auth).status_code == 404
        csv = client.get("/api/events.csv", auth=auth)
        assert "ب ط ص ١٢٣٤" in csv.text
    s.close()
