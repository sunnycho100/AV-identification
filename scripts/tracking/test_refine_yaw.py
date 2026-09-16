"""Self-check for the post-tracking heading refinement.

Three things have to hold. The tracker's own yaw must survive the pass byte for
byte, since downstream code still reads it. Every state must come back with a
yaw_refined field, null where no detection matched and a number where one did.
And the refinement has to actually refine: a track driving toward -x with every
detection yaw near 0 has the right axis and the wrong sign, so it must come back
near pi.

Plain asserts, no pytest.

    .venv/bin/python scripts/tracking/test_refine_yaw.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import refine_yaw
import score_heading as sh

N = 40


def write_clip(root, heading, yaws, drop_dets=(), clip="FAKE"):
    """A tiny outputs tree: one track walking along `heading` at 1 m per frame.

    score and vx drift so no frame reads as coasted, and each detection sits on
    its own state so score_heading's 1.0 m matcher always finds it. Frames listed
    in drop_dets get no detection file at all.
    """
    track = [{"frame": i,
              "x": 60.0 + i * np.cos(heading), "y": i * np.sin(heading), "z": -1.4,
              "yaw": 0.25, "vx": 30.0 + 0.01 * i, "vy": 0.0, "vz": 0.0,
              "speed_mps": 30.0, "score": 0.5 + 0.001 * i}
             for i in range(N)]
    t_dir = Path(root) / "outputs/tracking/camera-data" / f"{clip}_phase1"
    d_dir = Path(root) / "outputs/object_detection/camera-data" / f"{clip}_phase1"
    t_dir.mkdir(parents=True)
    d_dir.mkdir(parents=True)
    (t_dir / "tracks.json").write_text(json.dumps(
        {"meta": {"tag": "synthetic"}, "tracks": {"1": track}}))
    for s, y in zip(track, yaws):
        if s["frame"] in drop_dets:
            continue
        (d_dir / f"{s['frame']:03d}_pred.json").write_text(json.dumps(
            [{"class_name": "car", "score": 0.5, "x": s["x"], "y": s["y"], "z": -1.4,
              "l": 4.3, "w": 1.8, "h": 1.4, "yaw": float(y)}]))
    return t_dir / "tracks.json"


def run(root, heading, drop_dets=()):
    """(refined doc, the states of track 1) for one synthetic clip."""
    yaws = np.radians([2.0 * (-1) ** i for i in range(N)])
    p = write_clip(root, heading, yaws, drop_dets)
    doc = refine_yaw.refine(p, sh.default_det_dir(p), "FAKE")
    assert json.loads(p.read_text()) == doc, "refine must write back in place"
    return doc, doc["tracks"]["1"]


def test_tracker_yaw_is_untouched():
    with tempfile.TemporaryDirectory() as root:
        _, states = run(root, np.pi)
    assert [s["yaw"] for s in states] == [0.25] * N, [s["yaw"] for s in states]


def test_every_state_gets_the_new_fields():
    """The three frames with no detection keep a heading, carried in and flagged."""
    dropped = (5, 6, 7)
    with tempfile.TemporaryDirectory() as root:
        doc, states = run(root, np.pi, drop_dets=dropped)
    assert len(states) == N, len(states)
    assert all("yaw_refined" in s and "yaw_det" in s for s in states), states[0]
    for s in states:
        want_null = s["frame"] in dropped
        assert (s["yaw_det"] is None) == want_null, s
        assert s["yaw_refined"] is not None, s
        assert s["yaw_filled"] == want_null, s
    assert doc["meta"]["tag"] == "synthetic", doc["meta"]
    assert doc["meta"]["yaw_refined"]["window"] == 31, doc["meta"]["yaw_refined"]
    assert doc["meta"]["yaw_refined"]["score_pow"] == 2, doc["meta"]["yaw_refined"]


def test_backwards_track_ends_near_pi():
    """Moving toward -x with every detection yaw near 0: axis right, sign wrong."""
    with tempfile.TemporaryDirectory() as root:
        _, states = run(root, np.pi)
    det = np.array([s["yaw_det"] for s in states])
    out = np.array([s["yaw_refined"] for s in states])
    assert np.abs(sh.wrap(det)).max() < np.radians(5.0), np.degrees(det)
    assert np.abs(sh.wrap(out - np.pi)).max() < np.radians(5.0), np.degrees(out)


def test_forwards_track_keeps_the_detector_sign():
    with tempfile.TemporaryDirectory() as root:
        _, states = run(root, 0.0)
    out = np.array([s["yaw_refined"] for s in states])
    assert np.abs(sh.wrap(out)).max() < np.radians(5.0), np.degrees(out)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()


def test_gaps_are_filled_and_flagged():
    """A state with no matched detection takes its neighbour's heading, flagged."""
    states = [{"frame": 0, "yaw_refined": 0.5, "yaw_filled": False},
              {"frame": 1, "yaw_refined": None, "yaw_filled": False},
              {"frame": 2, "yaw_refined": None, "yaw_filled": False},
              {"frame": 3, "yaw_refined": 1.5, "yaw_filled": False}]
    refine_yaw.fill_gaps(states)
    assert [s["yaw_refined"] for s in states] == [0.5, 0.5, 1.5, 1.5], states
    assert [s["yaw_filled"] for s in states] == [False, True, True, False], states
    none_matched = [{"frame": 0, "yaw_refined": None, "yaw_filled": False}]
    refine_yaw.fill_gaps(none_matched)
    assert none_matched[0]["yaw_refined"] is None
