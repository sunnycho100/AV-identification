"""The two road masks the pipeline uses, one row per camera pose.

Left: the feature mask (before detection). Image cells outside the road band
(y -39 to -1 m, 0 to 2.5 m high, through the h15.1 dp-0.31 calibration) are
zeroed before the BEV projection; shown darkened.
Right: the box filter (after detection). road_mask.png, painted from where the
pretrained detector's moving tracks drove; a box is kept when its ground point
lands inside. Shown green.

    python3 scripts/reporting/render_masks_in_use.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "outputs/calibration/camera-data"
POSES = {"2025 pose (HV_T_EW_1)": ("HV_T_EW_1", 150), "2026 pose (AV_T_EW_6)": ("AV_T_EW_6", 150)}
OUT = ROOT / "outputs/reports/2026-11-04_new_todd/annotated/road_masks_in_use.jpg"


def title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


def row(name, clip, frame):
    img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{frame:03d}.jpg"))
    h, w = img.shape[:2]
    fm = cv2.resize(np.load(CAL / clip / "feature_mask_h25.npy"), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    left = img.copy()
    left[~fm] = (left[~fm] * 0.25).astype(np.uint8)
    road = cv2.imread(str(CAL / clip / "road_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    right = img.copy()
    right[road] = (0.55 * right[road] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, cs, -1, (0, 255, 255), 3)
    return np.hstack([title(left, f"{name}: feature mask, before detection (dark = zeroed)"),
                      title(right, f"{name}: box filter, after detection (green = kept)")])


if __name__ == "__main__":
    sheet = np.vstack([row(n, c, f) for n, (c, f) in POSES.items()])
    cv2.imwrite(str(OUT), cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2)))
    print(OUT)
