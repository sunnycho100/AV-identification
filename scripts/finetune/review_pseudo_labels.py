"""Render pseudo-labels next to the raw detections so a human can judge them.

Left: what BEVHeight actually predicted. Right: the label we would train on.
The point of the split is that the difference between the two panels is exactly
what fine-tuning would teach, so if the right panel is not visibly better, the
labels are not worth training on.

    .venv/bin/python scripts/finetune/review_pseudo_labels.py --clip HV_T_EW_1
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image

RAW = (80, 80, 235)      # red: raw prediction
LAB = (70, 220, 70)      # green: pseudo-label


def corners8(size, yaw, ctr):
    l, w, h = size
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                  [0, 0, 0, 0, h, h, h, h]])
    return (R @ b + np.array(ctr).reshape(3, 1)).T


def draw(img, objs, l2c, k34, col, tag):
    n = 0
    for o in objs:
        cc = (l2c @ np.c_[corners8([o["l"], o["w"], o["h"]], o["yaw"],
                                   [o["x"], o["y"], o["z"]]), np.ones(8)].T).T[:, :3]
        if np.sum(cc[:, 2] > 1e-6) < 4:
            continue
        draw_box_3d(img, project_to_image(cc, k34), c=col)
        n += 1
    for th, c in ((5, (0, 0, 0)), (2, (255, 255, 255))):
        cv2.putText(img, f"{tag} ({n})", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, th, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser("Review pseudo-labels against raw predictions")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stills", type=int, default=6)
    args = ap.parse_args()
    clip = args.clip

    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_phase1"
    lab_dir = ROOT / f"outputs/finetune/pseudo_labels/{clip}"
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K = np.array(cal["K"]); l2c = np.array(cal["lidar2cam"])
    k34 = np.zeros((3, 4)); k34[:3, :3] = K

    frames = sorted(int(p.stem.split("_")[0]) for p in lab_dir.glob("*_label.json"))
    out = ROOT / f"outputs/finetune/review/{clip}"
    out.mkdir(parents=True, exist_ok=True)

    for f in frames:
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        if img is None:
            continue
        raw = [o for o in json.loads((det_dir / f"{f:03d}_pred.json").read_text())
               if o["class_name"] == "car" and o["score"] >= 0.3]
        lab = json.loads((lab_dir / f"{f:03d}_label.json").read_text())
        left = draw(img.copy(), raw, l2c, k34, RAW, "RAW PREDICTION")
        right = draw(img.copy(), lab, l2c, k34, LAB, "PSEUDO-LABEL (train on this)")
        cv2.imwrite(str(out / f"{f:03d}.jpg"), np.hstack([left, right]))

    listing = out / "_f.txt"
    listing.write_text("".join(f"file '{f:03d}.jpg'\n" for f in frames))
    mp4 = out / f"{clip}_label_review.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-r", "30", "-f", "concat",
                    "-safe", "0", "-i", str(listing), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-vf", "scale=1920:-2", "-crf", "22",
                    str(mp4)], check=True)
    listing.unlink()

    idx = np.linspace(0, len(frames) - 1, args.stills).round().astype(int)
    picks = [frames[i] for i in idx]
    print(f"{clip}: {len(frames)} review frames -> {mp4}")
    print("stills: " + ", ".join(f"{out}/{f:03d}.jpg" for f in picks))


if __name__ == "__main__":
    main()
