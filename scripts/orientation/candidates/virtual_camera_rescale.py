"""Virtual camera rescale: raw detection yaws from the `_vcam` rerun.

The candidate's work happens before the detector, not after it:
`scripts/orientation/virtual_camera.py` centre-crops each frame and upscales it
so the effective fx becomes DAIR's ~2183, and the detector is rerun on those
frames with the rewritten K and the unchanged extrinsic. So there is nothing to
post-process here, this module only hands the scorer the rerun's own yaws.

    run_candidate.py virtual_camera_rescale --cfg suffix=vcam

note: identical in body to `identity`. Kept as its own file so the ledger
row carries the candidate's name and this explanation rather than "identity with
a suffix".
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

import score_heading as sh


def run(clip, det_dir, tracks, cfg):
    out = {}
    for tid, track in tracks.items():
        yaw = sh.detection_yaw(track, det_dir)
        out[tid] = {s["frame"]: float(y) for s, y in zip(track, yaw) if not np.isnan(y)}
    return out
