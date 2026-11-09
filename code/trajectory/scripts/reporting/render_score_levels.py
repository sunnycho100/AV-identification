"""Raw 3D detections at several score cut-offs, pretrained (top row) and fine-tuned (bottom row).

No road filter, tracking or smoothing: every box the detector outputs above the cut-off,
cars in green, other classes in yellow, with the car count per panel. Shows whether
missed cars are a confidence question or never detected.

    python scripts/reporting/render_score_levels.py CLIP PRETRAINED_RUN FINETUNED_RUN OUT.mp4
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluators.result2kitti import get_lidar_3d_8points  # noqa: E402

LEVELS = (0.45, 0.30, 0.10)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def draw(img, boxes, K, M, lo):
    n = 0
    for b in boxes:
        if b["score"] < lo:
            continue
        c = np.asarray(get_lidar_3d_8points([b["l"], b["w"], b["h"]], b["yaw"], [b["x"], b["y"], b["z"] + b["h"] / 2]))
        q = (M @ np.c_[c, np.ones(8)].T)[:3]
        if (q[2] <= 1).any():
            continue
        uv = ((K @ q)[:2] / (K @ q)[2]).T.astype(int)
        car = b["class_name"] == "car"
        n += car
        for i, j in EDGES:
            cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 220, 0) if car else (0, 220, 255), 2)
    return n


def main():
    clip, runs, out = sys.argv[1], (sys.argv[2], sys.argv[3]), Path(sys.argv[4])
    cam = ROOT / "outputs/object_detection/camera-data"
    cals = [json.loads((cam / f"{clip}_{r}" / "calibration_used.json").read_text()) for r in runs]
    tmp = out.with_suffix("")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    names = ("Pretrained BEVHeight", "Fine-tuned (retrained head + feature mask)")
    frames = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))
    for f in frames:
        img = cv2.imread(str(f))
        rows = []
        for r, cal, name in zip(runs, cals, names):
            p = cam / f"{clip}_{r}" / f"{f.stem}_pred.json"
            boxes = json.loads(p.read_text()) if p.exists() else []
            K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
            tiles = []
            for lo in LEVELS:
                im = img.copy()
                n = draw(im, boxes, K, M, lo)
                im = cv2.resize(im, (960, 540))
                cv2.rectangle(im, (0, 0), (960, 44), (0, 0, 0), -1)
                cv2.putText(im, f"{name}, score >= {lo:.2f}: {n} cars", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                tiles.append(im)
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(tmp / f"{f.stem}.jpg"), np.vstack(rows))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp / "%03d.jpg"), "-vf", "scale=1920:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(out)], check=True)
    keep = tmp / f"{len(frames) // 2:03d}.jpg"
    shutil.copy(keep, out.with_name(out.stem + "_still.jpg"))
    shutil.rmtree(tmp)
    print(out)


if __name__ == "__main__":
    main()
