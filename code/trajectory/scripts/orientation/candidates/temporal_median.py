"""Per-track temporal median of the raw detection yaw.

Why. BEVHeight predicts orientation per frame from appearance alone, so the yaw
of one track rattles frame to frame around the axis the vehicle is actually on.
A median over a short centred window of that track's own detection yaws keeps
the axis and drops the outliers, and the median rather than the mean so a single
badly wrong frame cannot drag the window.

The only input is the detector's yaw over time. Position, velocity and the
motion heading are never read, which is the constraint from the lab lead: the
score grades yaw against motion, so a candidate that looked at motion would be
fitting the grader instead of fixing the detector.

Front/back sign. BEVHeight's sign is unreliable, so a window can hold yaws that
agree on the axis and disagree by pi. Averaging or taking a full-circle median
over those raw would land between the two clusters. Instead every yaw in the
window is first folded to the axis relative to the centre state's yaw,
`d = fold(other - centre)` in [-90, 90) degrees, the median is taken over those
axis offsets, and the centre's own yaw carries the sign back out:
`out = centre + median(d)`. So the window votes on the axis only and the centre
state keeps its own front/back sign. This candidate corrects axis error; flips
stay as the detector left them.

Circular median: the axis offset in the window that minimises the sum of
absolute folded differences to the rest of the window, folded because the
distance that matters here is mod 180 as well. Brute force over the window's own
offsets, which is exact for the definition and cheap at these window sizes.

    cfg: window (odd frame count, default 15)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh

WINDOW = 15


def axis_median(window_yaw, centre):
    """The window's median axis, expressed with the centre state's own sign."""
    d = sh.fold(np.asarray(window_yaw, float) - centre)
    cost = [np.abs(sh.fold(d - x)).sum() for x in d]
    return float(centre + d[int(np.argmin(cost))])


def smooth(yaw, window=WINDOW):
    """Centred axis median of yaw over `window` states, NaN where the yaw is NaN."""
    half = window // 2
    out = np.full(len(yaw), np.nan)
    for i, centre in enumerate(yaw):
        if np.isnan(centre):
            continue
        win = yaw[max(0, i - half): i + half + 1]
        out[i] = axis_median(win[~np.isnan(win)], centre)
    return out


def run(clip, det_dir, tracks, cfg):
    window = int(cfg.get("window", WINDOW))
    out = {}
    for tid, track in tracks.items():
        yaw = smooth(sh.detection_yaw(track, det_dir), window)
        out[tid] = {s["frame"]: float(y) for s, y in zip(track, yaw) if not np.isnan(y)}
    return out
