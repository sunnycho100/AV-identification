"""Check video for road_mask.py: what the mask keeps and what it drops.

Outside the road polygon the frame is softly blurred, only
in this video (the detector never sees the blur). Kept boxes in green with the
red front edge, dropped boxes in grey.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_road_mask.py \
        --clip AV_T_EW_3 --run headft
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/reporting"), str(ROOT / "scripts/object_detection")]
from render_refined_compare import draw   # noqa: E402
import road_mask as rm                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    src = ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{a.run}"
    cal = json.loads((src.with_name(src.name + "_road") / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z = cal.get("road_plane_z_in_output", -1.73)
    frames = sorted((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    built = rm.load_built(a.clip)
    road = np.zeros((h, w), np.uint8)
    if built is not None:
        road[built] = 255
    else:
        cv2.fillPoly(road, [np.array(cal["road_mask"]["polygon_px"], np.int32)], 255)
    soft = cv2.GaussianBlur(road, (0, 0), 6).astype(np.float32)[..., None] / 255

    out_dir = ROOT / "outputs/reporting" / f"{a.clip}_{a.run}_road"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        img = cv2.imread(str(f))
        img = (img * soft + cv2.GaussianBlur(img, (0, 0), 9) * (1 - soft)).astype(np.uint8)
        poly = np.array(cal["road_mask"]["polygon_px"], np.int32)
        n_keep = n_drop = 0
        for p in json.loads((src / f"{f.stem}_pred.json").read_text()):
            if p["class_name"] != "car":
                continue
            if rm.keep(p, poly, K, M, z, built):
                draw(img, p, p["yaw"], K, M, (0, 220, 0))
                n_keep += 1
            else:
                draw(img, p, p["yaw"], K, M, (150, 150, 150))
                n_drop += 1
        cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(img, f"{a.run}  frame {f.stem}: {n_keep} kept (green), {n_drop} dropped (grey), "
                    f"blur = not traffic", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{f.stem}.jpg"), img)
    mp4 = out_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(out_dir / "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
