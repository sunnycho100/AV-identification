"""Open flag F1: can the road mask come from the 2D detector instead of the 3D detector it filters? (overnight E6)

Same painter as build_road_mask.py (moving tracks of 15 m or more paint a 2.5 m stroke),
same pooling per camera pose, but from the 2D baseline's tracks (Faster R-CNN + ground
plane, through the same tracker). Compared with the current mask (pretrained BEVHeight
tracks) by overlap, and by how many confident round-2 boxes (score 0.45 or more) each
mask would drop on the graded clips.

    python3 scripts/evaluation/road_mask_2d.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/calibration"), str(ROOT / "scripts/object_detection")]
import build_road_mask as B     # noqa: E402
import road_mask as rm          # noqa: E402

POSES = {"2026": ["AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"],
         "2025": ["AV_T_EW_3", "AV_T_WE_1", "AV_T_WE_3", "HV_T_EW_1"]}
CAM = ROOT / "outputs/object_detection/camera-data"
OUT = ROOT / "outputs/evaluation/road_mask_2d"


def build(clips, shape):
    raw = np.zeros(shape, np.uint8)
    for c in clips:
        cal = json.loads((CAM / f"{c}_det2d" / "calibration_used.json").read_text())
        tracks = json.loads((ROOT / f"outputs/trajectories/{c}_d2/tracks.json").read_text())["tracks"].values()
        raw |= B.paint(tracks, np.array(cal["K"]), np.array(cal["lidar2cam"]), cal.get("road_plane_z_in_output", -1.73), shape)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = np.zeros(shape, bool)
    for r in B.regions(raw):
        m |= r
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for pose, clips in POSES.items():
        img = cv2.imread(str(next((ROOT / f"data/camera-data/{clips[0]}/frames_all").glob("*.jpg"))))
        m2 = build(clips, img.shape[:2])
        for c in clips:
            m3 = rm.load_built(c)
            cal = json.loads((CAM / f"{c}_r2full_lo10" / "calibration_used.json").read_text())
            K, M, z = np.array(cal["K"]), np.array(cal["lidar2cam"]), cal.get("road_plane_z_in_output", -1.73)
            n = k3 = k2 = 0
            for p in (CAM / f"{c}_r2full_lo10").glob("*_pred.json"):
                for b in json.loads(p.read_text()):
                    if b["class_name"] != "car" or b["score"] < 0.45 or not (0 < b["x"] < 102.4):
                        continue
                    n += 1
                    k3 += rm.keep(b, None, K, M, z, m3)
                    k2 += rm.keep(b, None, K, M, z, m2)
            res[c] = {"pose": pose, "iou": round(float((m2 & m3).sum() / (m2 | m3).sum()), 3),
                      "area_2d_over_3d": round(float(m2.sum() / m3.sum()), 3),
                      "boxes": n, "kept_current": round(k3 / n, 4), "kept_2d": round(k2 / n, 4)}
            print(c, res[c])
        over = img.copy()
        m3 = rm.load_built(clips[0])
        for m, col in ((m3, (0, 255, 255)), (m2, (255, 0, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, col, 3)
        cv2.putText(over, f"{pose} pose: yellow = current mask (3D tracks), magenta = from 2D tracks", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.imwrite(str(OUT / f"road_mask_compare_{pose}.jpg"), over)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
