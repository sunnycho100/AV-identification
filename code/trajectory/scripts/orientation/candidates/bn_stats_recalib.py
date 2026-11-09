"""Candidate 3: the detector rerun with BatchNorm statistics recomputed on our frames.

The yaws come from a real rerun, not from post-processing, so this module does no
arithmetic: it reports the raw detection yaw of the `_bnrecal` run at every state
of the `_bnrecal` tracks. The checkpoint was produced by
`scripts/orientation/bn_recalib.py` (weights identical, BN buffers re-estimated),
and the rerun lives in `outputs/object_detection/camera-data/<clip>_bnrecal/`
with tracking in `outputs/tracking/camera-data/<clip>_bnrecal/`.

Run it so the runner hands over the recalibrated run rather than phase1:

    python scripts/orientation/run_candidate.py bn_stats_recalib --cfg suffix=bnrecal

Without that suffix this scores the phase1 run and is just the identity, which is
the intended failure mode: it cannot silently grade the wrong detections.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh


def run(clip, det_dir, tracks, cfg):
    if "bnrecal" not in str(det_dir):
        print(f"WARNING {clip}: det_dir is {det_dir}, "
              f"pass --cfg suffix=bnrecal to grade the recalibrated run")
    out = {}
    for tid, track in tracks.items():
        yaw = sh.detection_yaw(track, det_dir)
        out[tid] = {s["frame"]: float(y) for s, y in zip(track, yaw) if not np.isnan(y)}
    return out
