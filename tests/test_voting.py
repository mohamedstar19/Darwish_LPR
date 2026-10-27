from lpr.plate_text import parse
from lpr.voting import PlateVoter

A = parse("ب ط ص ١٢٣٤")
B = parse("ب ط ص ١٢٣")


def test_majority_wins_when_track_ends():
    v = PlateVoter(min_reads=2, confirm_reads=10, track_timeout_sec=1.0)
    v.add("cam", 1, "car", A, 0.8, ts=0.0)
    v.add("cam", 1, "car", B, 0.9, ts=0.1)
    v.add("cam", 1, "car", A, 0.7, ts=0.2)
    assert v.collect(now=0.5) == []  # still visible
    events = v.collect(now=2.0)
    assert len(events) == 1
    assert events[0].plate == A
    assert events[0].reads == 2


def test_single_read_is_not_enough():
    v = PlateVoter(min_reads=2, confirm_reads=4, track_timeout_sec=1.0)
    v.add("cam", 1, "car", A, 0.99, ts=0.0)
    assert v.collect(now=5.0) == []


def test_confirm_emits_early_once():
    v = PlateVoter(min_reads=2, confirm_reads=3, track_timeout_sec=1.0)
    for i in range(3):
        v.add("cam", 7, "car", A, 0.8, ts=i * 0.1)
    events = v.collect(now=0.3)
    assert [e.plate for e in events] == [A]
    v.add("cam", 7, "car", A, 0.8, ts=0.4)
    assert v.collect(now=0.5) == []
    assert v.collect(now=5.0) == []


def test_seen_keeps_track_alive():
    v = PlateVoter(min_reads=2, confirm_reads=9, track_timeout_sec=1.0)
    v.add("cam", 1, "car", A, 0.8, ts=0.0)
    v.add("cam", 1, "car", A, 0.8, ts=0.1)
    v.seen("cam", 1, "car", ts=1.5)
    assert v.collect(now=2.0) == []
    assert len(v.collect(now=3.0)) == 1


def test_dedup_across_track_ids_and_cameras():
    v = PlateVoter(min_reads=1, confirm_reads=1, track_timeout_sec=1.0, dedup_window_sec=60)
    v.add("cam", 1, "car", A, 0.8, ts=0.0)
    assert len(v.collect(now=0.0)) == 1
    v.add("cam", 2, "car", A, 0.8, ts=10.0)  # tracker switched id
    assert v.collect(now=10.0) == []
    v.add("other", 3, "car", A, 0.8, ts=10.0)  # different camera is a new event
    assert len(v.collect(now=10.0)) == 1
    v.add("cam", 4, "car", A, 0.8, ts=200.0)  # window passed
    assert len(v.collect(now=200.0)) == 1


def test_flush_ends_all_tracks():
    v = PlateVoter(min_reads=2, confirm_reads=9, track_timeout_sec=100)
    v.add("cam", 1, "car", A, 0.8, ts=0.0)
    v.add("cam", 1, "car", A, 0.8, ts=0.1)
    assert len(v.collect(now=0.2, flush=True)) == 1
    assert v.active_tracks() == 0
