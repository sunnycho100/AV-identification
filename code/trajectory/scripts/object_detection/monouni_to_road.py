"""MonoUNI camera-frame boxes to our road frame, in run_bevheight_generic.py's schema.

MonoUNI writes KITTI lines in its camera frame: bottom centre x y z (m) and ry,
the yaw about the camera's y axis (forward = (cos ry, 0, -sin ry)). The camera is
the same for variants A and B (B only zooms), so one rotation serves both:
p_road = R^T (p_cam - t), from the site extrinsic in prep.json, road at z = 0.
yaw is the box's long axis in the road frame (atan2 of R^T forward), the
convention road_mask, the tracker and the renderers read. ry fixes only the
heading's projection on the camera xz plane, so it is lifted back onto the road. big_vehicle is written
as car (BEVHeight's car class includes trucks); the original class is kept.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/object_detection/monouni_to_road.py \
        --pred-dir outputs/monouni/AV_T_EW_3_A/pred --prep outputs/monouni/AV_T_EW_3_A/prep.json \
        --run monouni_A
writes outputs/object_detection/camera-data/<clip>_<run>/NNN_pred.json + calibration_used.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASS = {"car": "car", "big_vehicle": "car", "pedestrian": "pedestrian", "cyclist": "bicycle"}
CLASS_ID = {"car": 0, "pedestrian": 8, "bicycle": 6}


def to_road(line, R, t):
    v = line.split()
    h, w, l, x, y, z, ry, score = (float(s) for s in v[8:16])
    p = R.T @ (np.array([x, y, z]) - t)
    # ry only fixes the heading's projection on the camera xz plane; the camera is
    # pitched ~19 deg, so lift it back onto the road: pick the camera-y component k
    # that makes the direction level in the road frame.
    k = (R[2, 2] * np.sin(ry) - R[0, 2] * np.cos(ry)) / R[1, 2]
    f = R.T @ np.array([np.cos(ry), k, -np.sin(ry)])
    name = CLASS[v[0]]
    return {"class_id": CLASS_ID[name], "class_name": name, "monouni_class": v[0], "score": score,
            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "l": l, "w": w, "h": h,
            "yaw": float(np.arctan2(f[1], f[0])), "box2d": [float(s) for s in v[4:8]]}


def from_road(b, R, t):
    """Inverse, for the selfcheck: a road-frame box as a MonoUNI KITTI line."""
    p = R @ np.array([b["x"], b["y"], b["z"]]) + t
    f = R @ np.array([np.cos(b["yaw"]), np.sin(b["yaw"]), 0.0])
    ry = np.arctan2(-f[2], f[0])
    return f"car 0.0 0 0 0 0 0 0 {b['h']} {b['w']} {b['l']} {p[0]} {p[1]} {p[2]} {ry} 0.9"


def main():
    ap = argparse.ArgumentParser("MonoUNI predictions to road-frame pred.json")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    meta = json.loads(Path(a.prep).read_text())
    R, t = np.array(meta["rotation"]), np.array(meta["translation"])
    out = ROOT / "outputs/object_detection/camera-data" / f"{meta['clip']}_{a.run}"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(a.pred_dir).glob("*.txt")):
        boxes = [to_road(s, R, t) for s in f.read_text().splitlines() if s.strip()]
        if meta.get("crop_xy") and meta["zoom"] != 1.0:     # 2D box back to original pixels
            (x0, y0), s = meta["crop_xy"], meta["zoom"]
            for b in boxes:
                b["box2d"] = [(b["box2d"][0] + x0) / s, (b["box2d"][1] + y0) / s,
                              (b["box2d"][2] + x0) / s, (b["box2d"][3] + y0) / s]
        (out / f"{int(f.stem):03d}_pred.json").write_text(json.dumps(boxes, indent=1))
        n += len(boxes)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    (out / "calibration_used.json").write_text(json.dumps({
        "detector": "MonoUNI (Rope3D checkpoint)", "variant": meta["variant"], "zoom": meta["zoom"],
        "crop_xy": meta["crop_xy"], "K_input": meta["K_used"], "extrinsic_json": meta["extrinsic"],
        "K": meta["K_orig"], "lidar2cam": M.tolist(), "road_plane_z_in_output": 0.0}, indent=2))
    print(f"{meta['clip']} {a.run}: {n} boxes in {len(list(out.glob('*_pred.json')))} frames -> {out}")


def _selfcheck():
    """A box placed in the road frame survives road -> MonoUNI camera line -> road."""
    ext = json.loads((ROOT / "outputs/calibration/camera-data/AV_T_EW_3/metric_extrinsic_h151_dpm031.json").read_text())
    R, t = np.array(ext["rotation"]), np.array(ext["translation"])
    for yaw in (0.0, 0.3, np.pi - 0.05, -2.0):
        b = {"x": 85.0, "y": -12.0, "z": 0.0, "l": 4.5, "w": 1.8, "h": 1.5, "yaw": yaw}
        r = to_road(from_road(b, R, t), R, t)
        assert abs(r["x"] - 85) < 0.01 and abs(r["y"] + 12) < 0.01 and abs(r["z"]) < 0.01, r
        assert abs((r["yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi) < np.radians(0.1), (r["yaw"], yaw)
    print("selfcheck ok (position within 1 cm and yaw within 0.1 deg after a round trip)")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
