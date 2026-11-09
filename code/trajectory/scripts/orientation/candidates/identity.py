"""Identity candidate: the raw detection yaw, unchanged.

Why. It has no effect by construction, so its score must equal the baseline
exactly. That makes it the runner's self-check: if identity moves the number,
the plumbing between the candidate interface and the score is wrong, not the
candidate.
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
