"""Visual proof of the marker-based target identification.

For every frame of the identified target track: all tracked cars drawn dim,
the target track's box in orange with its ID, and the hood-marker match in red
with its score wherever the matcher accepted it. If the identification is right,
the red marker sits inside the orange box in every frame where both exist, and
the orange box stays on the same physical car before and after the marker window.

    .venv/bin/python scripts/reporting/render_target_id.py --clip HV_T_EW_1
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

DIM = (140, 140, 140)
TARGET = (0, 140, 255)     # orange
MARKER = (0, 0, 255)       # red


def corners8(size, yaw, ctr):
    l, w, h = size
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                  [0, 0, 0, 0, h, h, h, h]])
    return (R @ b + np.array(ctr).reshape(3, 1)).T


def main():
    ap = argparse.ArgumentParser("Render the marker-identified target")
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()
    clip = args.clip

    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_phase1"
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K = np.array(cal["K"]); l2c = np.array(cal["lidar2cam"])
    k34 = np.zeros((3, 4)); k34[:3, :3] = K

    tgt = json.loads((ROOT / f"outputs/tracking/camera-data/{clip}_phase1/target_track.json").read_text())
    tracks = json.loads((ROOT / f"outputs/tracking/camera-data/{clip}_phase1/tracks.json").read_text())["tracks"]
    marker = json.loads((ROOT / f"outputs/target_id/{clip}/marker_matches.json").read_text())
    runs = [(r["start"], r["end"]) for r in marker["target_runs"]]
    tid = tgt["target_track_id"]

    by_frame = {}
    for t, states in tracks.items():
        for s in states:
            by_frame.setdefault(s["frame"], []).append((t, s))

    def size_for(f, s_):
        pf = det_dir / f"{f:03d}_pred.json"
        if not pf.exists():
            return (4.5, 1.8, 1.5)
        xy = np.array([s_["x"], s_["y"]])
        best, bd = (4.5, 1.8, 1.5), 2.0
        for o in json.loads(pf.read_text()):
            if o["class_name"] != "car" or o["score"] < 0.3:
                continue
            d = np.linalg.norm(np.array([o["x"], o["y"]]) - xy)
            if d < bd:
                best, bd = (o["l"], o["w"], o["h"]), d
        return best

    out_dir = ROOT / f"outputs/reports/target-id/{clip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = tgt["frame_range"]
    header = (f"{clip}: TARGET = track {tid}, identified by hood marker "
              f"({tgt['votes']}/{tgt['marker_frames']} frames), GPS not used")
    rendered = []
    for f in range(f0, f1 + 1):
        img_p = ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        for t, s in by_frame.get(f, []):
            cc = (l2c @ np.c_[corners8(size_for(f, s), s["yaw"],
                                       [s["x"], s["y"], s["z"]]), np.ones(8)].T).T[:, :3]
            if np.sum(cc[:, 2] > 1e-6) < 4:
                continue
            uv = project_to_image(cc, k34)
            if t == tid:
                draw_box_3d(img, uv, c=TARGET)
                draw_box_3d(img, uv, c=TARGET)   # double-stroke: stands out
                u, v = int(uv[:, 0].min()), int(uv[:, 1].min())
                cv2.putText(img, f"TARGET {tid}", (max(0, u), max(20, v - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TARGET, 2, cv2.LINE_AA)
            else:
                draw_box_3d(img, uv, c=DIM)
        rec = marker["per_frame"].get(str(f))
        in_run = any(a <= f <= b for a, b in runs)
        if rec and in_run and rec["score"] >= marker["score_min"]:
            x, y, w, h = int(rec["x"]), int(rec["y"]), int(rec["w"]), int(rec["h"])
            cv2.rectangle(img, (x, y), (x + w, y + h), MARKER, 3)
            cv2.putText(img, f"marker {rec['score']:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MARKER, 2, cv2.LINE_AA)
        for th, col in ((5, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(img, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        col, th, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{f:03d}.jpg"), img)
        rendered.append(f)

    listing = out_dir / "_frames.txt"
    listing.write_text("".join(f"file '{f:03d}.jpg'\n" for f in rendered))
    mp4 = out_dir / f"{clip}_target_id.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-r", "30", "-f", "concat",
                    "-safe", "0", "-i", str(listing), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    listing.unlink()
    print(f"{len(rendered)} frames -> {mp4}")


if __name__ == "__main__":
    main()
