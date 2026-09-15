"""Self-check for the one-bit-per-track sign correction.

Four things have to hold. A track driving away from the camera with all its
yaws pointing forwards must come back pointing backwards, with its folded error
untouched, because the sign step may only add pi. A track driving the other way
must come back byte for byte identical. A track that barely moved must be left
alone rather than signed off three noisy displacements. And the decision must be
one bit for the whole track: on a vehicle that turns 60 degrees mid-track, no
state may end up with a different sign from its neighbours.

Plain asserts, no pytest.

    .venv/bin/python scripts/orientation/test_sign_from_motion.py
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

CAND = rc.load_candidate("sign_from_motion")
SMOOTH = rc.load_candidate("yaw_track_axis_consensus")
CFG = {"mode": "window", "window": 31, "score_pow": 2}


def write_clip(root, xy, yaws, clip="FAKE"):
    """A tiny outputs tree with one track on the given path and detection yaws.

    score and vx drift so no frame reads as coasted, and each detection sits
    exactly on its state so score_heading's 1.0 m matcher always finds it.
    """
    track = [{"frame": i, "x": float(x), "y": float(y), "z": -1.4, "yaw": 0.0,
              "vx": 30.0 + 0.01 * i, "vy": 0.0, "score": 0.5 + 0.001 * i}
             for i, (x, y) in enumerate(xy)]
    t_dir = Path(root) / "outputs/tracking/camera-data" / f"{clip}_phase1"
    d_dir = Path(root) / "outputs/object_detection/camera-data" / f"{clip}_phase1"
    t_dir.mkdir(parents=True)
    d_dir.mkdir(parents=True)
    (t_dir / "tracks.json").write_text(json.dumps({"meta": {}, "tracks": {"1": track}}))
    for s, y in zip(track, yaws):
        (d_dir / f"{s['frame']:03d}_pred.json").write_text(json.dumps(
            [{"class_name": "car", "score": 0.5, "x": s["x"], "y": s["y"], "z": -1.4,
              "l": 4.3, "w": 1.8, "h": 1.4, "yaw": float(y)}]))
    return track


def both_runs(root, xy, yaws):
    """(smoothed, signed, score_smoothed, score_signed) for one synthetic clip."""
    write_clip(root, xy, yaws)
    found = rc.collect_clips(root=root, clips=["FAKE"])
    smoothed = rc.run_all(SMOOTH, found, dict(CFG))
    signed = rc.run_all(CAND, found, {})
    return smoothed, signed, rc.score(found, smoothed), rc.score(found, signed)


def straight(n, heading, start=(60.0, 0.0), step=1.0):
    """n positions walking from `start` at `step` metres per frame along `heading`."""
    return [(start[0] + i * step * np.cos(heading), start[1] + i * step * np.sin(heading))
            for i in range(n)]


def test_track_driving_backwards_gets_pi_added():
    """Moving toward -x with every yaw near 0: the axis is right, the sign is not."""
    n = 40
    yaws = np.radians([2.0 * (-1) ** i for i in range(n)])
    with tempfile.TemporaryDirectory() as root:
        smoothed, signed, s_score, c_score = both_runs(root, straight(n, np.pi), yaws)
    out = np.array(list(signed["FAKE"]["1"].values()))
    assert len(out) == n, len(out)
    assert np.abs(sh.wrap(out - np.pi)).max() < np.radians(5.0), np.degrees(out)
    # the sign step adds pi and nothing else, so the axis error does not move
    a, b = s_score["per_clip"]["FAKE"], c_score["per_clip"]["FAKE"]
    assert a["n"] == b["n"] == n, (a["n"], b["n"])
    assert abs(a["median_folded_deg"] - b["median_folded_deg"]) < 1e-9, (a, b)
    assert abs(a["mean_folded_deg"] - b["mean_folded_deg"]) < 1e-9, (a, b)
    assert a["frac_raw_gt90"] == 1.0 and b["frac_raw_gt90"] == 0.0, (a, b)


def test_track_driving_forwards_is_untouched():
    n = 40
    yaws = np.radians([2.0 * (-1) ** i for i in range(n)])
    with tempfile.TemporaryDirectory() as root:
        smoothed, signed, _, _ = both_runs(root, straight(n, 0.0, start=(20.0, 0.0)), yaws)
    assert signed == smoothed, (signed, smoothed)


def test_a_track_with_three_moving_states_is_left_alone():
    """Parked for nine frames, then one 2 m jump: only 3 states clear the 1.0 m
    threshold, which is under min_moving, so the smoothed yaw stands."""
    xy = [(40.0, 0.0)] * 9 + [(38.0, 0.0)]
    yaws = np.radians([2.0 * (-1) ** i for i in range(len(xy))])
    with tempfile.TemporaryDirectory() as root:
        write_clip(root, xy, yaws)
        found = rc.collect_clips(root=root, clips=["FAKE"])
        track = sh.load_tracks(found[0][1])["1"]
        smoothed = rc.run_all(SMOOTH, found, dict(CFG))
        signed = rc.run_all(CAND, found, {})
    assert travel_count(track) == 3, travel_count(track)
    assert signed == smoothed, (signed, smoothed)
    # and it would have been flipped, had there been enough moving states
    with tempfile.TemporaryDirectory() as root:
        write_clip(root, xy, yaws)
        found = rc.collect_clips(root=root, clips=["FAKE"])
        loose = rc.run_all(CAND, found, {"min_moving": 3})
    assert loose != smoothed, loose


def travel_count(track):
    return CAND.travel_direction(track)[1]


def test_one_bit_per_track_survives_a_turn():
    """A vehicle turning 60 degrees, every box backwards. The whole track flips or
    nothing does: no state may disagree with its neighbours."""
    half = 20
    turn = np.radians(60.0)
    first = straight(half, 0.0, start=(30.0, -20.0))
    second = straight(half, turn, start=first[-1], step=1.0)[1:]
    xy = first + second
    heading = np.r_[np.zeros(half), np.full(len(second), turn)]
    rng = np.random.default_rng(0)
    yaws = heading + np.pi + np.radians(rng.normal(0.0, 3.0, len(xy)))
    with tempfile.TemporaryDirectory() as root:
        smoothed, signed, _, _ = both_runs(root, xy, yaws)
    s, c = smoothed["FAKE"]["1"], signed["FAKE"]["1"]
    frames = sorted(s)
    flipped = np.array([abs(float(sh.wrap(c[f] - s[f]))) > np.pi / 2 for f in frames])
    assert len(flipped) == len(xy), len(flipped)
    assert (flipped[1:] == flipped[:-1]).all(), flipped.astype(int)
    assert flipped.all(), flipped.astype(int)      # backwards boxes, so it fires


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
