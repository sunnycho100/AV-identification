"""Self-check for the two-stage (ByteTrack-style) association.

Two things have to hold. A track whose detection drops below the export
threshold for one frame must keep going on a real measurement, not on the
Kalman prediction alone: score_heading calls a state coasted when the score and
the velocity both repeat the previous frame verbatim, and the same track run
without the weak detection must coast there, or the test is not measuring
anything. And a weak detection standing on its own must never become a track,
however many frames it persists for.

Plain asserts, no pytest.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/test_run_ab3dmot.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import run_ab3dmot as rt
import score_heading as sh

N = 12
GAP = 5          # the frame where the detection goes weak
LOW, HIGH = 0.30, 0.60
SIZE = [1.5, 1.8, 4.3]   # h, w, l


def car(frame, score, x0=40.0):
    """One car moving 1 m per frame along +x, as (dets 1x7, info 1x1)."""
    det = SIZE + [x0 + frame, 0.0, -1.4, 0.0]
    return np.array([det], float), np.array([[score]], float)


def empty():
    return np.empty((0, 7)), np.empty((0, 1))


def run(load, low_thresh):
    tracker = (rt.AB3DMOT(rt.build_cfg(), cat="Car", ID_init=0) if low_thresh is None
               else rt.ByteAB3DMOT(rt.build_cfg(), cat="Car", low_thresh=low_thresh, ID_init=0))
    return rt.track_frames(tracker, list(range(N)), load, 30.0)


def gap_load(frame):
    """The car, weak on one frame."""
    return car(frame, LOW if frame == GAP else HIGH)


def dropped_load(frame):
    """The same car with the weak frame missing entirely: what 0.45 export gives."""
    return empty() if frame == GAP else car(frame, HIGH)


def test_low_score_continues_a_track():
    tracks = run(gap_load, low_thresh=0.20)
    assert len(tracks) == 1, f"expected one track, got {len(tracks)}"
    states = sorted(next(iter(tracks.values())), key=lambda s: s["frame"])
    assert len(states) == N, f"track lost frames: {len(states)} of {N}"
    at_gap = [i for i, s in enumerate(states) if s["frame"] == GAP][0]

    assert abs(states[at_gap]["score"] - LOW) < 1e-9, (
        f"the weak detection was not the measurement at frame {GAP}: "
        f"score {states[at_gap]['score']}")
    assert not sh.coasted(states)[at_gap], f"frame {GAP} still reads as coasted"

    # and the same track without that detection does coast there, so the gap is real
    base = sorted(next(iter(run(dropped_load, low_thresh=None).values())),
                  key=lambda s: s["frame"])
    assert sh.coasted(base)[at_gap], (
        f"baseline did not coast at frame {GAP}; the test proves nothing")
    print(f"ok: frame {GAP} continues on the {LOW} detection (baseline coasts there)")


def test_low_score_alone_never_starts_a_track():
    tracks = run(lambda f: car(f, LOW), low_thresh=0.20)
    assert tracks == {}, f"a low-score detection started {len(tracks)} track(s)"

    # the default path still tracks it: the refusal is the flag, not the data
    assert len(run(lambda f: car(f, LOW), low_thresh=None)) == 1, "baseline lost the car"
    print("ok: a lone low-score detection never births a track")


if __name__ == "__main__":
    test_low_score_continues_a_track()
    test_low_score_alone_never_starts_a_track()
    print("all ok")
