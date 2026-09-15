"""Self-check for the frozen heading score.

Why. score_heading.py grades every orientation candidate, so a silent change in
it re-scores the whole ledger. These cases pin the parts that a refactor could
break without any clip noticing: the sign and folding conventions, the motion
window, the coasted and stationary exclusions, and the keep rule.

Plain asserts, no pytest.

    .venv/bin/python scripts/evaluation/test_score_heading.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_heading as sh


def straight_track(n=12, yaw=0.0, x0=5.0, step=1.0, dy=0.0):
    """A track walking along +x at one metre per frame, never coasted."""
    return [{"frame": i, "x": x0 + i * step, "y": i * dy, "z": -1.4,
             "yaw": yaw, "vx": step * 30.0 + 0.01 * i, "vy": 0.0,
             "score": 0.5 + 0.001 * i}
            for i in range(n)]


def test_zero_offset():
    r = sh.score_clip({"1": straight_track(yaw=0.0)})
    assert r["n"] == 12, r["n"]
    assert abs(r["median_folded_deg"]) < 1e-9, r["median_folded_deg"]
    assert r["frac_raw_gt90"] == 0.0, r["frac_raw_gt90"]
    assert r["per_bin"]["0_40"]["n"] == 12, r["per_bin"]


def test_known_offset():
    r = sh.score_clip({"1": straight_track(yaw=0.2)})
    assert abs(r["median_folded_deg"] - 11.46) < 0.01, r["median_folded_deg"]
    assert abs(r["mean_folded_deg"] - 11.46) < 0.01, r["mean_folded_deg"]
    assert r["frac_raw_gt45"] == 0.0, r["frac_raw_gt45"]


def test_front_back_flip():
    r = sh.score_clip({"1": straight_track(yaw=math.pi)})
    assert abs(r["median_folded_deg"]) < 1e-9, r["median_folded_deg"]
    assert r["frac_raw_gt90"] == 1.0, r["frac_raw_gt90"]
    assert r["frac_raw_gt45"] == 1.0, r["frac_raw_gt45"]


def test_mirror_invariance():
    a = sh.score_clip({"1": straight_track(yaw=0.2, dy=0.3)})
    mirrored = [dict(s, y=-s["y"], yaw=-s["yaw"])
                for s in straight_track(yaw=0.2, dy=0.3)]
    b = sh.score_clip({"1": mirrored})
    assert a["n"] == b["n"], (a["n"], b["n"])
    assert abs(a["median_folded_deg"] - b["median_folded_deg"]) < 1e-9
    assert abs(a["mean_folded_deg"] - b["mean_folded_deg"]) < 1e-9


def test_stationary_skipped():
    r = sh.score_clip({"1": straight_track(step=0.1)})
    assert r["n"] == 0, r["n"]
    assert r["median_folded_deg"] is None, r["median_folded_deg"]


def test_coasted_excluded():
    st = straight_track()
    for i in (4, 5, 9):                  # AB3DMOT repeats score and vx verbatim
        st[i]["score"] = st[i - 1]["score"]
        st[i]["vx"] = st[i - 1]["vx"]
    r = sh.score_clip({"1": st})
    assert r["n"] == 9, r["n"]


def test_yaw_override():
    st = straight_track(yaw=0.0)
    r = sh.score_clip({"1": st}, yaw_override={"1": {i: 0.2 for i in range(12)}})
    assert abs(r["median_folded_deg"] - 11.46) < 0.01, r["median_folded_deg"]


def write_dets(det_dir, track, yaw, offset=0.0):
    """One detection per frame at the state's position, shifted by `offset` metres
    in y, plus a decoy far enough away that the nearest test has to reject it."""
    for s in track:
        p = Path(det_dir) / f"{s['frame']:03d}_pred.json"
        p.write_text(json.dumps([
            {"class_name": "car", "score": 0.5, "x": s["x"], "y": s["y"] + offset,
             "z": -1.4, "l": 4.3, "w": 1.8, "h": 1.4, "yaw": yaw},
            {"class_name": "car", "score": 0.4, "x": s["x"], "y": s["y"] + 9.0,
             "z": -1.4, "l": 4.3, "w": 1.8, "h": 1.4, "yaw": yaw + 1.0},
        ]))


def test_detection_yaw_is_scored():
    """The tracker's yaw is ignored: AB3DMOT may have flipped it by pi."""
    track = straight_track(yaw=math.pi)          # flipped Kalman yaw
    with tempfile.TemporaryDirectory() as d:
        write_dets(d, track, yaw=0.2)            # what BEVHeight actually said
        r = sh.score_clip({"1": track}, det_dir=d)
    assert r["n"] == 12, r["n"]
    assert r["n_no_detection"] == 0, r["n_no_detection"]
    assert abs(r["median_folded_deg"] - 11.46) < 0.01, r["median_folded_deg"]
    assert r["frac_raw_gt90"] == 0.0, r["frac_raw_gt90"]


def test_unmatched_states_excluded():
    track = straight_track(yaw=0.0)
    with tempfile.TemporaryDirectory() as d:
        write_dets(d, track, yaw=0.2, offset=1.5)     # every detection out of range
        Path(d, "003_pred.json").write_text(json.dumps(
            [{"x": track[3]["x"], "y": track[3]["y"] + 0.5, "yaw": 0.2}]))
        r = sh.score_clip({"1": track}, det_dir=d)
    assert r["n"] == 1, r["n"]
    assert r["n_no_detection"] == 11, r["n_no_detection"]
    assert abs(r["median_folded_deg"] - 11.46) < 0.01, r["median_folded_deg"]


def test_default_det_dir():
    got = sh.default_det_dir("outputs/tracking/camera-data/AV_T_WE_1_phase1/tracks.json")
    assert got.parts[-3:] == ("object_detection", "camera-data", "AV_T_WE_1_phase1"), got


def result(per_clip_medians, flip=0.05):
    per_clip = {k: {"median_folded_deg": v} for k, v in per_clip_medians.items()}
    return {"per_clip": per_clip,
            "mean_median_folded_deg": sum(per_clip_medians.values()) / len(per_clip_medians),
            "mean_frac_raw_gt90": flip}


def test_keep_decision():
    base = result({"a": 10.0, "b": 10.0, "c": 10.0})
    worse_one = result({"a": 5.0, "b": 12.0, "c": 10.0})       # mean better, b worse by 2
    keep, why = sh.keep_decision(base, worse_one)
    assert not keep, why
    assert "b" in why, why

    better = result({"a": 9.0, "b": 9.5, "c": 8.0})
    keep, why = sh.keep_decision(base, better)
    assert keep, why

    flipped = result({"a": 9.0, "b": 9.5, "c": 8.0}, flip=0.06)
    keep, why = sh.keep_decision(base, flipped)
    assert not keep, why


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
