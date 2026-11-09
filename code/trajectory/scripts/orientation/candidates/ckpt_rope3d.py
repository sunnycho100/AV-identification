"""Candidate: the detector rerun with the Rope3D checkpoint instead of the DAIR one.

`BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt` with
`experiments/rope3d/bev_height_lss_r50_864_1536_128x128_102.py` (d_bound
[-1.5, 3.0, 180], so a 180-channel height head). Rope3D's kitti conversion puts
the ground at ego z = 0, not DAIR's -1.73, so the rerun uses --no-ground-shift.

Same shape as `ckpt_r140` and `virtual_camera_rescale`: the yaws come from the
rerun, so this module only reports the raw detection yaw of the `_rope3d` run at
every state of the `_rope3d` tracks.

    python scripts/orientation/run_candidate.py ckpt_rope3d --cfg suffix=rope3d
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh


def run(clip, det_dir, tracks, cfg):
    if "rope3d" not in str(det_dir):
        print(f"WARNING {clip}: det_dir is {det_dir}, "
              f"pass --cfg suffix=rope3d to grade the Rope3D run")
    out = {}
    for tid, track in tracks.items():
        yaw = sh.detection_yaw(track, det_dir)
        out[tid] = {s["frame"]: float(y) for s, y in zip(track, yaw) if not np.isnan(y)}
    return out
