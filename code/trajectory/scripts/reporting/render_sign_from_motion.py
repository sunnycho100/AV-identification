"""Video of what the motion sign step does, frame by frame.

Left: the detector's raw heading. Right: the same boxes after the one-bit fix
(sign_from_motion's rule applied to the raw yaw, no axis smoothing, so only the
sign step is shown). Each tracked car also gets its recent path (white dots)
and the track's overall travel direction (yellow arrow), which is what the
rule compares against.

Box colours on the right:
  green   kept: the box already pointed within 90 deg of the travel direction
  orange  flipped 180 deg by the rule
  grey    no decision: the track has fewer than 5 moving steps

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_sign_from_motion.py \
        --clip AV_T_EW_3 --run phase1
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
sys.path[:0] = [str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/orientation/candidates"),
                str(ROOT / "scripts/reporting")]
os.environ["ORIENTATION_NO_GPS"] = "1"
import score_heading as sh          # noqa: E402
import sign_from_motion as sfm      # noqa: E402
from render_refined_compare import draw   # noqa: E402

GREEN, ORANGE, GREY = (0, 220, 0), (0, 140, 255), (170, 170, 170)


def project(K, M, x, y, z):
    q = K @ (M[:3, :3] @ np.array([x, y, z]) + M[:3, 3])
    return int(q[0] / q[2]), int(q[1] / q[2])


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", default="phase1")
    a = ap.parse_args()
    tag = f"{a.clip}_{a.run}"
    det_dir = ROOT / "outputs/object_detection/camera-data" / tag
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    zr = cal.get("road_plane_z_in_output", -1.73)
    tracks = sh.load_tracks(ROOT / "outputs/tracking/camera-data" / tag / "tracks.json")

    travel = {}
    for tid, t in tracks.items():
        d, n = sfm.travel_direction(t)
        travel[tid] = d if n >= sfm.MIN_MOVING else None
    by_frame = {}
    for tid, t in tracks.items():
        for i, s in enumerate(t):
            by_frame.setdefault(s["frame"], []).append((tid, i, s))

    out_dir = ROOT / "outputs/reporting" / f"{tag}_sign_from_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"kept": 0, "flipped": 0, "none": 0}
    frames = sorted((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg"))
    for f in frames:
        fr = int(f.stem)
        img = cv2.imread(str(f))
        left, right = img.copy(), img.copy()
        states = by_frame.get(fr, [])
        for p in json.loads((det_dir / f"{f.stem}_pred.json").read_text()):
            if p["class_name"] != "car":
                continue
            draw(left, p, p["yaw"], K, M, (255, 255, 255))
            near = [(tid, i, s) for tid, i, s in states if math.hypot(s["x"] - p["x"], s["y"] - p["y"]) <= 1.0]
            tv = travel.get(near[0][0]) if near else None
            if tv is None:
                draw(right, p, p["yaw"], K, M, GREY)
                counts["none"] += 1
            elif abs(float(sh.wrap(p["yaw"] - tv))) > math.pi / 2:
                draw(right, p, float(sh.wrap(p["yaw"] + math.pi)), K, M, ORANGE)
                counts["flipped"] += 1
            else:
                draw(right, p, p["yaw"], K, M, GREEN)
                counts["kept"] += 1
        # recent path and travel direction for every live track
        for tid, i, s in states:
            t = tracks[tid]
            for q in t[max(0, i - 20):i + 1:2]:
                cv2.circle(right, project(K, M, q["x"], q["y"], zr), 3, (255, 255, 255), -1)
            tv = travel.get(tid)
            if tv is not None:
                a0 = project(K, M, s["x"], s["y"], zr)
                a1 = project(K, M, s["x"] + 6 * math.cos(tv), s["y"] + 6 * math.sin(tv), zr)
                cv2.arrowedLine(right, a0, a1, (0, 230, 255), 3, cv2.LINE_AA, tipLength=0.3)
        label(left, "raw detector heading (red edge = the front it predicts)")
        label(right, "after the motion sign: green kept, orange flipped, grey no decision")
        cv2.imwrite(str(out_dir / f"{f.stem}.jpg"), np.hstack([left, right]))
    mp4 = ROOT / "outputs/reporting" / f"{tag}_sign_from_motion.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(out_dir / "%03d.jpg"),
                    "-vf", "scale=2560:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)],
                   check=True)
    print(f"boxes over the clip: {counts}")
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
