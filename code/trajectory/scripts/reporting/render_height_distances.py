"""Annotate the camera-vs-GPS distance for each step of the instrumented car.

For every pair of consecutive contact points: the distance the camera
measures on the road (back-projected with height h and a pitch correction) next
to the distance the GPS says the car moved, and the difference. Left panel uses
the height in the pipeline (16.26 m, VP pitch), right panel the joint fit.
Red = camera longer than GPS, blue = camera shorter; a range-dependent colour
pattern is the bias this is meant to expose.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_height_distances.py
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/calibration"))
import fit_height_from_gps_distances as F   # noqa: E402

SETTINGS = (("in use: h 16.26 m, VP pitch", 16.26, 0.0),
            ("joint fit: h 15.10 m, pitch -0.31 deg", 15.10, math.radians(-0.31)))


def panel(bg, uv, P, K, R, frames, label, h, dp):
    img = bg.copy()
    G = F.ground(uv, K, R, h, dp)
    cam_range = np.hypot(G[:, 0], G[:, 1])
    rows = []
    for i in range(len(uv) - 1):
        dc = float(np.linalg.norm(G[i + 1] - G[i]))
        dg = float(np.linalg.norm(P[i + 1] - P[i]))
        e = dc - dg
        col = (40, 40, 220) if e > 0 else (220, 120, 30)
        a, b = tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[i + 1])
        cv2.line(img, a, b, col, 3, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        side = 1 if i % 2 == 0 else -1
        txt = f"cam {dc:4.1f} / gps {dg:4.1f}  {e:+.2f} m"
        org = (mid[0] + side * 20 if side > 0 else mid[0] - 20 - 11 * len(txt), mid[1] + 5)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        rows.append((float(cam_range[i:i + 2].mean()), e, dg))
    for p in uv:
        cv2.circle(img, tuple(int(v) for v in p), 6, (255, 255, 255), -1)
        cv2.circle(img, tuple(int(v) for v in p), 6, (0, 0, 0), 1)
    rows = np.array(rows)
    near, far = rows[rows[:, 0] < 50], rows[rows[:, 0] >= 50]
    stats = (f"{label} | per step, camera minus GPS: under 50 m {near[:, 1].mean():+.2f} m "
             f"({near[:, 1].sum() / near[:, 2].sum():+.1%}), over 50 m {far[:, 1].mean():+.2f} m "
             f"({far[:, 1].sum() / far[:, 2].sum():+.1%})")
    cv2.rectangle(img, (0, 0), (1920, 44), (0, 0, 0), -1)
    cv2.putText(img, stats, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    out_dir = ROOT / "outputs/reporting"
    for clip in ("HV_T_EW_1", "AV_T_WE_1"):
        clicks = ROOT / "outputs/calibration/contact_clicks" / clip / "auto_contacts.json"
        K, R, uv, D, frames = F.load_inputs(clip, clicks)
        sys.path.insert(0, str(ROOT / "scripts/tracking"))
        import grade_target_vs_gps as g
        gps = g.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv")
        P = np.array([gps[f][:2] for f in frames])
        mid = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{frames[len(frames) // 2]:03d}.jpg"))
        bg = cv2.addWeighted(mid, 0.55, np.full_like(mid, 255), 0.45, 0)   # faded, so the lines read
        panels = [panel(bg, uv, P, K, R, frames, *s) for s in SETTINGS]
        out = out_dir / f"{clip}_height_distance_check.jpg"
        cv2.imwrite(str(out), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
