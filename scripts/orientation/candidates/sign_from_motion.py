"""Axis from appearance, front versus back from the track's direction of travel.

Why. yaw_track_axis_consensus fixes the axis and leaves the sign alone, and the
sign is where the remaining error is: BEVHeight reports the same absolute
direction for both traffic streams, so one stream is right 85 to 95 percent of
the time and the other is backwards 81 to 92 percent of the time, stable within
a track. No amount of smoothing recovers that, because every detection on a
track agrees with every other one.

The lab approved exactly one bit of position information per track. This
candidate spends it and nothing more:

1. The smoothed yaw comes from yaw_track_axis_consensus, unchanged, and that is
   the only thing that sets the axis. cfg passes straight through, so mode,
   window and score_pow are the kept ones (window, 31, score_pow 2).
2. One direction of travel is computed for the whole track: the circular mean of
   the per-state displacement headings, using score_heading.motion_heading's own
   window and its 1.0 m minimum displacement, so only states that actually moved
   vote. A track with fewer than `min_moving` such states gets no sign step at
   all and keeps the smoothed yaw.
3. Every state of the track whose smoothed yaw is more than 90 degrees from that
   one direction gets pi added. Every state of the track is judged against the
   same direction, so the decision is one bit for the track, never per frame: a
   vehicle turning through a junction keeps a consistent sign, and the axis the
   consensus chose is never touched, only its front and back.

What this does not do. It does not steer the yaw toward the motion, it does not
use the per-state heading as the answer, and it does not read position for
anything else. A pure motion heading would be a different candidate and a much
larger claim.

The raw sign is not destroyed. It is still in <frame>_pred.json, and the
consensus run (outputs/orientation/runs/yaw_track_axis_consensus_*/yaws.json)
still holds the smoothed yaw with the detector's own sign. A downstream
classifier can carry both, the appearance sign and the motion sign, and learn
where they disagree.

    cfg: min_moving (default 5), plus everything yaw_track_axis_consensus takes
         (mode, window, score_pow, range_pow, range_floor)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "evaluation"))

import score_heading as sh
import yaw_track_axis_consensus as consensus

WINDOW = 31
SCORE_POW = 2
MIN_MOVING = 5


def travel_direction(track):
    """(direction, n_moving) for the whole track from its displacement headings.

    The circular mean of score_heading.motion_heading, which is already NaN
    wherever the centred window displacement is under its 1.0 m threshold, so
    only states that actually moved vote. (nan, 0) when none did.
    """
    motion = sh.motion_heading(track)
    ok = ~np.isnan(motion)
    if not ok.any():
        return float("nan"), 0
    return (float(np.arctan2(np.sin(motion[ok]).sum(), np.cos(motion[ok]).sum())),
            int(ok.sum()))


def run(clip, det_dir, tracks, cfg):
    cfg = {"mode": "window", "window": WINDOW, "score_pow": SCORE_POW, **cfg}
    out = consensus.run(clip, det_dir, tracks, cfg)
    min_moving = int(cfg.get("min_moving", MIN_MOVING))
    for tid, track in tracks.items():
        per_frame = out.get(tid)
        if not per_frame:
            continue
        travel, n_moving = travel_direction(track)
        if n_moving < min_moving:
            continue
        for frame, yaw in per_frame.items():
            if abs(float(sh.wrap(yaw - travel))) > np.pi / 2:
                per_frame[frame] = float(sh.wrap(yaw + np.pi))
    return out
