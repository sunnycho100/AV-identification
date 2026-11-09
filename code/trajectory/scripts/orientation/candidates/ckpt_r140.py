"""Candidate: the detector rerun with the 140.8 m DAIR checkpoint instead of 102.4 m.

Same shape as `bn_stats_recalib`: the yaws come from a rerun, so this module only
reports the raw detection yaw of the `_r140` run at every state of the `_r140`
tracks. The reruns already exist for four clips; `HV_T_EW_2` and `AV_V_WE_3` have
none, so the runner drops them from both sides of the keep rule.

    python scripts/orientation/run_candidate.py ckpt_r140 --cfg suffix=r140
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh


def run(clip, det_dir, tracks, cfg):
    if "r140" not in str(det_dir):
        print(f"WARNING {clip}: det_dir is {det_dir}, "
              f"pass --cfg suffix=r140 to grade the 140.8 m run")
    out = {}
    for tid, track in tracks.items():
        yaw = sh.detection_yaw(track, det_dir)
        out[tid] = {s["frame"]: float(y) for s, y in zip(track, yaw) if not np.isnan(y)}
    return out
