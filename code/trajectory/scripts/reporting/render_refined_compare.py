"""Side-by-side video: raw detector heading (left) vs refined heading (right).

Right panel uses the same refinement as the pipeline (refine_yaw.py): a
score-weighted 31-frame axis consensus over each track's own detection yaws,
then one travel-direction sign bit per track. Detections that no track state
claims (within 1 m) are drawn grey on the right, since they get no refinement.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_refined_compare.py \
        --clip AV_T_EW_3 --run trafcam
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/orientation/candidates")]
os.environ["ORIENTATION_NO_GPS"] = "1"
import score_heading as sh          # noqa: E402
import sign_from_motion as sfm      # noqa: E402


def draw(img, p, yaw, K, M, color):
    c, s = math.cos(yaw), math.sin(yaw)
    fwd, left = np.array([c, s]) * p["l"] / 2, np.array([-s, c]) * p["w"] / 2
    ctr = np.array([p["x"], p["y"]])
    pts = np.array([ctr + fwd + left, ctr + fwd - left, ctr - fwd - left, ctr - fwd + left, ctr + fwd])
    z = p["z"]    # road plane in the lidar2cam frame used for this run
    q = (M[:3, :3] @ np.c_[pts, np.full(len(pts), z)].T).T + M[:3, 3]
    uv = (K @ q.T).T
    uv = (uv[:, :2] / uv[:, 2:]).astype(int)
    cv2.polylines(img, [uv[:4].reshape(-1, 1, 2)], True, color, 2)
    cv2.line(img, tuple(uv[0]), tuple(uv[1]), (0, 0, 255), 3)
    cv2.circle(img, tuple(uv[4]), 5, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", default="trafcam")
    args = ap.parse_args()
    tag = f"{args.clip}_{args.run}"
    det_dir = ROOT / "outputs/object_detection/camera-data" / tag
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    # trafcam writes z at the DAIR road height but its lidar2cam has the road at 0
    z_off = cal.get("road_plane_z_in_output", 0.0) if "px_per_m" in cal else 0.0

    tracks = sh.load_tracks(ROOT / "outputs/tracking/camera-data" / tag / "tracks.json")
    refined = sfm.run(args.clip, str(det_dir), tracks, {})

    out_dir = ROOT / "outputs/reporting" / f"{tag}_refined_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted((ROOT / "data/camera-data" / args.clip / "frames_all").glob("*.jpg"))
    for f in frames:
        fr = int(f.stem)
        img = cv2.imread(str(f))
        left, right = img.copy(), img.copy()
        states = [(tid, s) for tid, st in tracks.items() for s in st if s["frame"] == fr]
        for p in json.loads((det_dir / f"{f.stem}_pred.json").read_text()):
            if p["class_name"] != "car":
                continue
            p = dict(p, z=p["z"] - z_off)
            draw(left, p, p["yaw"], K, M, (0, 255, 0))
            near = [(tid, s) for tid, s in states if math.hypot(s["x"] - p["x"], s["y"] - p["y"]) <= 1.0]
            yaw = refined.get(near[0][0], {}).get(fr) if near else None
            draw(right, p, p["yaw"] if yaw is None else yaw, K, M,
                 (160, 160, 160) if yaw is None else (0, 255, 0))
        for im, label in ((left, "raw detector heading"), (right, "track axis consensus + motion sign")):
            cv2.putText(im, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(im, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{f.stem}.jpg"), np.hstack([left, right]))
    mp4 = ROOT / "outputs/reporting" / f"{tag}_raw_vs_refined.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(out_dir / "%03d.jpg"),
                    "-vf", "scale=2560:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)],
                   check=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
