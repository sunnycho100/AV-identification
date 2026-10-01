"""Side-by-side video of two detection runs' raw boxes on the same clip.

Each box is drawn as its road footprint, red edge at the predicted front, no
tracking or smoothing, so what you see is exactly what each model outputs per
frame.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_two_runs.py \
        --clip AV_T_EW_3 --runs base102 headft --labels "102.4 m, pretrained" "102.4 m, head fine-tuned"
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reporting"))
from render_refined_compare import draw   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--labels", nargs=2, required=True)
    a = ap.parse_args()
    dirs = [ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{r}" for r in a.runs]
    cals = [json.loads((d / "calibration_used.json").read_text()) for d in dirs]
    out_dir = ROOT / "outputs/reporting" / f"{a.clip}_{a.runs[0]}_vs_{a.runs[1]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg")):
        img = cv2.imread(str(f))
        panels = []
        for d, cal, label in zip(dirs, cals, a.labels):
            p_img = img.copy()
            K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
            n = 0
            for p in json.loads((d / f"{f.stem}_pred.json").read_text()):
                if p["class_name"] == "car":
                    draw(p_img, p, p["yaw"], K, M, (0, 220, 0))
                    n += 1
            cv2.rectangle(p_img, (0, 0), (p_img.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(p_img, f"{label}   frame {f.stem}, {n} cars", (24, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(p_img)
        cv2.imwrite(str(out_dir / f"{f.stem}.jpg"), np.hstack(panels))
    mp4 = ROOT / "outputs/reporting" / f"{a.clip}_{a.runs[0]}_vs_{a.runs[1]}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(out_dir / "%03d.jpg"),
                    "-vf", "scale=2560:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)],
                   check=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
