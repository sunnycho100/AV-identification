"""Final training labels with their roles (D34), as a video per clip and one still sheet.

Green: label from a smoothed track. Cyan: single-frame label a 2D detection confirms.
Gray: ignored label (second box on a car, or no 2D support). Magenta rectangle: a 2D
detection no label covers, ignored so the missed car is not taught as background.

    python3 scripts/reporting/render_label_roles.py LABEL_SET OUT_DIR
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reporting"))
import match2d as X                      # noqa: E402
import render_untracked_conflicts as R   # noqa: E402

CLIPS = ("HV_T_EW_1", "AV_T_WE_1", "AV_T_EW_7")


def draw(c, f, objs, K, M):
    img = cv2.imread(str(ROOT / f"data/camera-data/{c}/frames_all/{f:03d}.jpg"))
    n = {"label": 0, "ignore": 0}
    for o in objs:
        n[o["role"]] += 1
        if o.get("reason") == "2d_only":
            x0, y0, x1, y1 = map(int, o["box2d"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
            continue
        uv = R.project(o, K, M)
        if uv is None:
            continue
        col = (150, 150, 150) if o["role"] == "ignore" else (0, 200, 0) if R.tracked(o) else (230, 200, 0)
        for i, j in R.E:
            cv2.line(img, tuple(uv[i]), tuple(uv[j]), col, 2)
    cv2.rectangle(img, (0, 0), (1920, 44), (0, 0, 0), -1)
    cv2.putText(img, f"{c} frame {f}: {n['label']} labels, {n['ignore']} ignored | green track, cyan 2D-confirmed, "
                     f"gray ignored, magenta 2D only (ignored)", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    lab_set, out = sys.argv[1], ROOT / sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    stills = []
    for c in CLIPS:
        K, M = X.load(c)
        files = sorted((ROOT / f"outputs/finetune/{lab_set}/{c}").glob("*_label.json"))
        tmp = out / f"_{c}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for k, p in enumerate(files):
            img = draw(c, int(p.name[:3]), json.loads(p.read_text()), K, M)
            cv2.imwrite(str(tmp / f"{k:04d}.jpg"), cv2.resize(img, (1280, 720)))
            if k in (len(files) // 4, len(files) // 2, 3 * len(files) // 4):
                stills.append(cv2.resize(img, (960, 540)))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(tmp / "%04d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(out / f"{c}_label_roles.mp4")], check=True)
        shutil.rmtree(tmp)
    cv2.imwrite(str(out / "label_roles_stills.jpg"), np.vstack([np.hstack(stills[r:r + 3]) for r in range(0, 9, 3)]))
    print(out)


if __name__ == "__main__":
    main()
