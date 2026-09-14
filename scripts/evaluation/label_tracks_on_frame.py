"""Draw every track alive at one frame with its ID, so the instrumented car can be
picked by eye against Camera data/figures/car.png. Image only; GPS is never read.

    .venv/bin/python scripts/evaluation/label_tracks_on_frame.py --clip AV_T_WE_1 --frame 150
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--run", default="phase1")
    a = ap.parse_args()
    det = ROOT / f"outputs/object_detection/camera-data/{a.clip}_{a.run}"
    cal = json.load(open(det / "calibration_used.json"))
    K, l2c = np.array(cal["K"], float), np.array(cal["lidar2cam"], float)
    tracks = json.load(open(ROOT / f"outputs/tracking/camera-data/{a.clip}_{a.run}/tracks.json"))["tracks"]
    img = cv2.imread(str(ROOT / f"data/camera-data/{a.clip}/frames_all/{a.frame:03d}.jpg"))
    for tid, states in tracks.items():
        s = next((s for s in states if s["frame"] == a.frame), None)
        if s is None:
            continue
        pc = l2c @ np.array([s["x"], s["y"], s["z"], 1.0])
        if pc[2] <= 0:
            continue
        u, v, w = K @ pc[:3]
        u, v = int(u / w), int(v / w)
        cv2.circle(img, (u, v), 6, (0, 140, 255), -1)
        cv2.putText(img, tid, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
    out = det.parent / f"{a.clip}_{a.run}_tracks_frame{a.frame:03d}.jpg"
    cv2.imwrite(str(out), img)
    print(out)


if __name__ == "__main__":
    main()
