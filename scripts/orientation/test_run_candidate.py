"""Self-check for the candidate runner and the first two candidates.

Why. The runner is the only thing standing between a candidate and the ledger,
so what it must never get wrong is pinned here: a candidate that changes nothing
must score exactly like the baseline, the temporal median must actually reduce
the folded error on a track whose yaws are noisy, and the GPS guard must refuse
a candidate that reaches for the held-out trajectories.

Plain asserts, no pytest.

    .venv/bin/python scripts/orientation/test_run_candidate.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import run_candidate as rc
import score_heading as sh


def fake_clip(root, clip="FAKE", yaws=None, n=24, suffix="phase1"):
    """A tiny outputs tree: one tracks.json and one <frame>_pred.json per frame.

    The track walks along +x at one metre per frame with a constant true yaw of
    zero, so the motion heading is zero everywhere and the detection yaw carries
    the whole error. score and vx drift so no frame reads as coasted.
    """
    yaws = [0.0] * n if yaws is None else yaws
    track = [{"frame": i, "x": 10.0 + i, "y": 0.0, "z": -1.4, "yaw": 0.0,
              "vx": 30.0 + 0.01 * i, "vy": 0.0, "score": 0.5 + 0.001 * i}
             for i in range(n)]
    t_dir = Path(root) / "outputs/tracking/camera-data" / f"{clip}_{suffix}"
    d_dir = Path(root) / "outputs/object_detection/camera-data" / f"{clip}_{suffix}"
    t_dir.mkdir(parents=True)
    d_dir.mkdir(parents=True)
    (t_dir / "tracks.json").write_text(json.dumps({"meta": {}, "tracks": {"1": track}}))
    for s, y in zip(track, yaws):
        (d_dir / f"{s['frame']:03d}_pred.json").write_text(json.dumps(
            [{"class_name": "car", "score": 0.5, "x": s["x"], "y": s["y"], "z": -1.4,
              "l": 4.3, "w": 1.8, "h": 1.4, "yaw": y}]))
    return track, d_dir


def test_suffix_scores_the_rerun_on_the_shared_clip_subset():
    """cfg suffix points the candidate at a rerun; a clip with no rerun is dropped
    from both sides, and the rerun's own yaws are what gets scored."""
    with tempfile.TemporaryDirectory() as root:
        noisy = [0.4 * (-1) ** i for i in range(24)]
        fake_clip(root, clip="BOTH", yaws=noisy)
        fake_clip(root, clip="BOTH", yaws=[0.0] * 24, suffix="alt")
        fake_clip(root, clip="ONLYBASE", yaws=noisy)

        base = rc.collect_clips(root=root, clips=["BOTH", "ONLYBASE"])
        alt = rc.collect_clips(root=root, clips=["BOTH", "ONLYBASE"], suffix="alt")
        assert [c for c, _, _ in base] == ["BOTH", "ONLYBASE"], base
        assert [c for c, _, _ in alt] == ["BOTH"], alt
        assert alt[0][2].name == "BOTH_alt", alt[0][2]

        shared = [f for f in base if f[0] == "BOTH"]
        yaws = rc.run_all(rc.load_candidate("bn_stats_recalib"), alt, {})
        before = rc.score(shared)["mean_median_folded_deg"]
        after = rc.score(alt, yaws)["mean_median_folded_deg"]
    assert abs(before - np.degrees(0.4)) < 1e-6, before
    assert after < 1e-6, after


def test_identity_equals_baseline():
    with tempfile.TemporaryDirectory() as root:
        fake_clip(root, yaws=[0.2 * (-1) ** i for i in range(24)])
        found = rc.collect_clips(root=root, clips=["FAKE", "MISSING"])
        assert [c for c, _, _ in found] == ["FAKE"], found
        base = rc.score(found)
        yaws = rc.run_all(rc.load_candidate("identity"), found, {})
        cand = rc.score(found, yaws)
    assert base["per_clip"]["FAKE"]["n"] == 24, base
    assert cand == base, (base, cand)


def noisy_track(n=40):
    """Detection yaws: the true axis (zero) plus alternating +-0.3 rad, and every
    fifth frame a gross outlier the median is there to reject.

    The outliers are what makes this a test. A pure alternation of +-0.3 cannot
    be improved by any window median: the median of an odd window is the centre
    state's own sign of the noise, so the error comes out identical.
    """
    yaws = [0.3 * (-1) ** i for i in range(n)]
    for i in range(0, n, 5):
        yaws[i] = 1.3
    return yaws


def test_temporal_median_reduces_error():
    with tempfile.TemporaryDirectory() as root:
        fake_clip(root, yaws=noisy_track())
        found = rc.collect_clips(root=root, clips=["FAKE"])
        base = rc.score(found)
        yaws = rc.run_all(rc.load_candidate("temporal_median"), found, {"window": 5})
        cand = rc.score(found, yaws)
    b, c = base["per_clip"]["FAKE"], cand["per_clip"]["FAKE"]
    assert c["n"] == b["n"], (b["n"], c["n"])
    assert c["mean_folded_deg"] < b["mean_folded_deg"] - 5.0, (b, c)
    assert c["median_folded_deg"] <= b["median_folded_deg"] + 1e-9, (b, c)


def test_temporal_median_keeps_the_centre_sign():
    """Folding is per window, so a yaw pointing backwards stays backwards."""
    tm = rc.load_candidate("temporal_median")
    yaw = np.array([0.1, np.pi - 0.1, 0.0, -0.1, np.pi + 0.05])
    out = tm.smooth(yaw, window=5)
    assert abs(sh.fold(out[1] - yaw[1])) < 0.2, out
    assert abs(sh.wrap(out[1])) > np.pi / 2, out        # still the backwards sign
    assert abs(sh.wrap(out[0])) < np.pi / 2, out        # still the forwards sign


def test_gps_guard_refuses():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "cheat.py")
        p.write_text("def run(clip, det_dir, tracks, cfg):\n"
                     "    open('trajectory.csv')\n")
        try:
            rc.check_no_gps(p)
        except SystemExit as e:
            assert "GPS" in str(e), e
        else:
            raise AssertionError("the guard let trajectory.csv through")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
