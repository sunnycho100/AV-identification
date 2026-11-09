"""Weighted circular consensus of a track's detection yaws, on the axis only.

Why. temporal_median showed that a track's own detection yaws already carry the
axis the vehicle is on, and that a plain median over a short window recovers it.
This candidate keeps that and asks whether the votes should be equal. They are
not obviously equal: BEVHeight's yaw is worse on the far half of the scene and
worse on the boxes it is least sure about, so a vote weighted by detection score
and by closeness should land on the axis a little better than a vote that counts
a 90 m low-score box the same as a 30 m confident one.

    w = score ** score_pow / max(x, range_floor) ** range_pow

The range floor keeps a near-zero range from taking over the whole window, and
both exponents default to 1 so the weighting is the plain linear one until the
ledger says otherwise. Weights are relative, so the scale of w never matters.

Two modes. "window" is temporal_median plus weights: a centred window of
`window` states votes on the axis of the centre state. "track" gives one axis to
the whole track, every state voting, which is the strongest version of the
assumption that a vehicle holds its axis for as long as it is tracked, and the
one that breaks first on a turn.

Score and range come from the track state rather than from the detection json.
AB3DMOT stores the detection's own score on the state and the state position is
within the 1.0 m match radius of the detection, so the two agree wherever a yaw
was matched at all, and reusing score_heading's matcher keeps the candidate on
exactly the frames the grader scores.

Sign, exactly as temporal_median. The consensus is solved in the folded domain
relative to the centre state's yaw and returned as `centre + d`, so the vote
decides the axis and the centre state keeps its own front/back sign. Position
and motion are never read; tracks are used for membership only.

    cfg: mode (window|track), window (odd frame count, default 15),
         score_pow (default 1), range_pow (default 1), range_floor (default 20)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh

WINDOW = 15
RANGE_FLOOR = 20.0


def weights(track, score_pow=1.0, range_pow=1.0, range_floor=RANGE_FLOOR):
    """Vote weight per state: high score and short range count for more."""
    score = np.array([s["score"] for s in track], float)
    rng = np.array([s["x"] for s in track], float)
    return score ** score_pow / np.maximum(rng, range_floor) ** range_pow


def weighted_axis_median(d, w):
    """The offset in `d` minimising sum_j w_j |fold(d - d_j)|.

    Brute force over the offsets themselves, which is exact for this definition
    and cheap at a window of tens or a track of low hundreds.
    """
    d = np.asarray(d, float)
    cost = (np.asarray(w, float) * np.abs(sh.fold(d[:, None] - d[None, :]))).sum(1)
    return float(d[int(np.argmin(cost))])


def window_axis(yaw, w, window=WINDOW):
    """Centred weighted axis consensus over `window` states, NaN where yaw is NaN."""
    half = window // 2
    out = np.full(len(yaw), np.nan)
    for i, centre in enumerate(yaw):
        if np.isnan(centre):
            continue
        lo, hi = max(0, i - half), i + half + 1
        win, ww = yaw[lo:hi], w[lo:hi]
        ok = ~np.isnan(win)
        out[i] = centre + weighted_axis_median(sh.fold(win[ok] - centre), ww[ok])
    return out


def track_axis(yaw, w):
    """One axis for the whole track, each state expressing it with its own sign."""
    ok = ~np.isnan(yaw)
    if not ok.any():
        return np.full(len(yaw), np.nan)
    ref = yaw[ok][0]
    axis = ref + weighted_axis_median(sh.fold(yaw[ok] - ref), w[ok])
    return yaw + sh.fold(axis - yaw)


def run(clip, det_dir, tracks, cfg):
    mode = str(cfg.get("mode", "window"))
    if mode not in ("window", "track"):
        raise SystemExit(f"mode must be window or track, not {mode!r}")
    window = int(cfg.get("window", WINDOW))
    out = {}
    for tid, track in tracks.items():
        yaw = sh.detection_yaw(track, det_dir)
        w = weights(track, float(cfg.get("score_pow", 1.0)),
                    float(cfg.get("range_pow", 1.0)),
                    float(cfg.get("range_floor", RANGE_FLOOR)))
        y = track_axis(yaw, w) if mode == "track" else window_axis(yaw, w, window)
        out[tid] = {s["frame"]: float(v) for s, v in zip(track, y) if not np.isnan(v)}
    return out
