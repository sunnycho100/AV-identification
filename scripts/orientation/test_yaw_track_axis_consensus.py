"""Self-check for the weighted axis consensus candidate.

Three things have to hold. With every weight equal it must be temporal_median
and nothing else, so any change in the ledger is the weighting and not a second
edit smuggled in beside it. A vote must actually be weighted: the same outlier
must pull the consensus further when its detection score is high than when it is
low. And on a track that really is straight, the whole-track mode must come back
with the axis the track is on.

Plain asserts, no pytest.

    .venv/bin/python scripts/orientation/test_yaw_track_axis_consensus.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import run_candidate as rc
import score_heading as sh
from test_run_candidate import fake_clip, noisy_track

CAND = rc.load_candidate("yaw_track_axis_consensus")


def test_equal_weights_reproduce_temporal_median():
    """score_pow 0 and range_pow 0 make every weight 1, which is the plain median."""
    with tempfile.TemporaryDirectory() as root:
        fake_clip(root, yaws=noisy_track(n=60), n=60)
        found = rc.collect_clips(root=root, clips=["FAKE"])
        base = rc.run_all(rc.load_candidate("temporal_median"), found, {"window": 15})
        mine = rc.run_all(CAND, found, {"mode": "window", "window": 15,
                                        "score_pow": 0, "range_pow": 0})
    assert mine.keys() == base.keys(), (mine.keys(), base.keys())
    for clip, tracks in base.items():
        assert mine[clip].keys() == tracks.keys(), clip
        for tid, per_frame in tracks.items():
            assert mine[clip][tid] == per_frame, (clip, tid)
    assert sum(len(t) for t in base["FAKE"].values()) == 60, base


def state(score, x=50.0):
    return {"frame": 0, "x": x, "y": 0.0, "yaw": 0.0, "vx": 1.0, "score": score}


def test_a_low_score_outlier_pulls_less():
    """Same offsets, same ranges, only the outlier's detection score differs."""
    d = np.radians([0.0, 2.0, 4.0, 6.0, 40.0])
    inliers = [state(0.3)] * 4
    low = CAND.weights(inliers + [state(0.05)])
    high = CAND.weights(inliers + [state(0.9)])
    c_low = np.degrees(CAND.weighted_axis_median(d, low))
    c_high = np.degrees(CAND.weighted_axis_median(d, high))
    assert c_low < c_high, (c_low, c_high)
    assert c_high < 40.0, c_high        # one loud vote is still only one vote


def test_track_mode_finds_the_true_axis():
    """A straight track, symmetric noise, half the frames flipped back to front."""
    rng = np.random.default_rng(0)
    n, truth = 400, 0.6
    yaw = truth + np.radians(rng.normal(0.0, 5.0, n))
    yaw[::2] += np.pi                                   # the detector's sign is unreliable
    track = [state(float(s), float(x)) for s, x in
             zip(rng.uniform(0.2, 0.9, n), rng.uniform(15.0, 90.0, n))]
    out = CAND.track_axis(yaw, CAND.weights(track))
    assert np.abs(np.degrees(sh.fold(out - truth))).max() < 1.0, out[:5]
    assert np.allclose(sh.fold(out - yaw), out - yaw, atol=1e-9)   # sign kept per state


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
