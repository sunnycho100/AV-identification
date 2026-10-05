"""Drop detections off the road.

Why: parked cars in the lots beside US 12/18 and the "US 12/18 @ TODD DR"
banner come out as cars. They are not traffic. The image the detector sees is
left alone; this only filters its boxes afterwards.

The road is one band in road coordinates (x along the road, y across it),
ROAD_X by ROAD_Y. On Todd Drive moving cars sit at y -37 to -2 m (16.26 m
extrinsic), parked cars near -49 and the banner at +2. The band is projected
into the image with the run's own K and extrinsic, so a box is kept when its
ground point lands inside that image polygon. The banner's false box sits off
the road, so the band removes it too; masking the banner as a rectangle would
also drop real cars in the nearest lane, which drive under the text.
Judging in pixels keeps the mask the same patch of road whatever height the
calibration uses.

If scripts/calibration/build_road_mask.py has written road_mask.png for the
clip (built from where moving cars drove), that mask is used instead of the
band; ground points off the image are judged at the nearest edge pixel.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/object_detection/road_mask.py \
        --clip AV_T_EW_3 --run headft
writes outputs/object_detection/camera-data/<clip>_<run>_road/.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROAD_X = (5.0, 3000.0)        # far end runs out toward the vanishing point
ROAD_Y = (-39.0, -1.0)          # note: Todd Drive only; another site needs its own band


def road_polygon(K, M, z):
    """The road band as an image polygon (int32 Nx2). The near edge is cut where
    it would fall behind the camera, so the polygon stays valid."""
    xs = np.geomspace(*ROAD_X, 60)     # dense near the camera, sparse far away
    edge = [(x, ROAD_Y[0]) for x in xs] + [(x, ROAD_Y[1]) for x in xs[::-1]]
    pts = []
    for x, y in edge:
        q = M[:3, :3] @ np.array([x, y, z]) + M[:3, 3]
        if q[2] > 0.5:
            u = K @ q
            pts.append(u[:2] / u[2])
    return np.array(pts, np.float32).round().astype(np.int32)


def ground_px(p, K, M, z):
    q = K @ (M[:3, :3] @ np.array([p["x"], p["y"], z]) + M[:3, 3])
    return float(q[0] / q[2]), float(q[1] / q[2])


def keep(p, poly, K, M, z, built=None):
    u, v = ground_px(p, K, M, z)
    if built is not None:
        h, w = built.shape
        return bool(built[min(max(int(v), 0), h - 1), min(max(int(u), 0), w - 1)])
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (u, v), False) >= 0


def load_built(clip):
    """The clip's built mask (bool HxW), or None to fall back to the band."""
    p = ROOT / "outputs/calibration/camera-data" / clip / "road_mask.png"
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 if p.exists() else None


def main():
    ap = argparse.ArgumentParser("Filter detections to the road")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    src = ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{a.run}"
    dst = src.with_name(src.name + "_road")
    dst.mkdir(exist_ok=True)
    cal = json.loads((src / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z = cal.get("road_plane_z_in_output", -1.73)
    poly = road_polygon(K, M, z)
    built = load_built(a.clip)
    cal["road_mask"] = {"source": "road_mask.png" if built is not None else "band",
                        "road_x": ROAD_X, "road_y": ROAD_Y,
                        "polygon_px": poly.tolist()}
    (dst / "calibration_used.json").write_text(json.dumps(cal, indent=2))
    n_in = n_out = 0
    for f in sorted(src.glob("*_pred.json")):
        preds = json.loads(f.read_text())
        kept = [p for p in preds if keep(p, poly, K, M, z, built)]
        n_in, n_out = n_in + len(preds), n_out + len(kept)
        (dst / f.name).write_text(json.dumps(kept, indent=2))
    print(f"{a.clip}_{a.run}: kept {n_out} of {n_in} boxes "
          f"({cal['road_mask']['source']}), wrote {dst}")


def _selfcheck():
    """With the Todd Drive site calibration: a lane car stays, a parked car at
    y -49 and the banner box at y +2 go, a real car under the banner stays."""
    cal = json.loads((ROOT / "outputs/object_detection/camera-data/AV_T_EW_3_headft"
                      / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z = -1.73
    poly = road_polygon(K, M, z)
    car = {"x": 65.0, "y": -8.9}
    assert keep(car, poly, K, M, z)
    assert keep({"x": 27.8, "y": -35.5}, poly, K, M, z)       # the white van, far lane
    assert not keep({"x": 44.3, "y": -48.3}, poly, K, M, z)   # parked car, frame 150
    assert not keep({"x": 24.8, "y": 2.1}, poly, K, M, z)     # banner, frame 270
    assert keep({"x": 22.0, "y": -5.2}, poly, K, M, z)        # car under the banner, frame 60
    print(f"selfcheck ok (polygon {len(poly)} points)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
