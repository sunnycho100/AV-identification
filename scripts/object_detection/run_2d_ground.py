"""Baseline: 2D boxes plus the ground plane, no 3D detector (the SVBRD-LLM style).

A COCO Faster R-CNN (torchvision) finds cars, trucks and buses. The bottom
centre of each 2D box is the car's nearest ground contact; its ray hits the road
(z = 0) at P. The car's centre lies further along that ray's ground direction by
the car's half-extent seen from there, (l/2)|cos phi| + (w/2)|sin phi|, phi
the angle between the ray and the road (x axis). Size is a fixed 4.5 x 1.8 x 1.5 m,
yaw along the road. Writes the run_bevheight_generic.py schema, so road mask,
tracker and smoother run unchanged; this isolates what 3D detection buys us.

Runs on the lab (bevheight env, torchvision 0.10; weights ~160 MB from pytorch.org):
    CUDA_VISIBLE_DEVICES=1 python scripts/object_detection/run_2d_ground.py --clip AV_T_EW_3
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COCO = {3: "car", 6: "bus", 8: "truck"}
L, W, H = 4.5, 1.8, 1.5       # note: one size for all; per-class sizes if trucks matter


def ground_point(u, v, K, R, t):
    """Pixel -> road point (z = 0), or None above the horizon."""
    ray = R.T @ np.linalg.solve(K, [u, v, 1.0])          # direction in road frame
    c = -R.T @ t                                         # camera centre in road frame
    if ray[2] >= 0:
        return None
    return c + ray * (-c[2] / ray[2]), c


def centre(u, v, K, R, t):
    g = ground_point(u, v, K, R, t)
    if g is None:
        return None
    p, c = g
    d = p[:2] - c[:2]
    d /= np.linalg.norm(d)
    phi = np.arctan2(d[1], d[0])
    return p[:2] + d * (L / 2 * abs(np.cos(phi)) + W / 2 * abs(np.sin(phi)))


def main():
    import cv2
    import torch
    import torchvision
    ap = argparse.ArgumentParser("2D boxes to road-frame pred.json")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--run", default="det2d")
    a = ap.parse_args()
    cal = ROOT / "outputs/calibration/camera-data" / a.clip
    fx, fy, cx, cy = json.loads(next(cal.glob("*_anycalib_pinhole_pinhole.json")).read_text())["prediction"]["intrinsics"]
    ext = json.loads((cal / "metric_extrinsic_h151_dpm031.json").read_text())
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    R, t = np.array(ext["rotation"]), np.array(ext["translation"])
    out = ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{a.run}"
    out.mkdir(parents=True, exist_ok=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    n = 0
    for f in sorted((ROOT / f"data/camera-data/{a.clip}/frames_all").glob("*.jpg")):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            r = model([torch.from_numpy(im).permute(2, 0, 1).float().cuda() / 255])[0]
        preds = []
        for b, lab, s in zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy()):
            if int(lab) not in COCO or s < a.score:
                continue
            xy = centre((b[0] + b[2]) / 2, b[3], K, R, t)
            if xy is None:
                continue
            preds.append({"class_id": 0, "class_name": "car", "coco_class": COCO[int(lab)], "score": float(s),
                          "x": float(xy[0]), "y": float(xy[1]), "z": 0.0, "l": L, "w": W, "h": H, "yaw": 0.0,
                          "box2d": [float(v) for v in b]})
        (out / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=1))
        n += len(preds)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    (out / "calibration_used.json").write_text(json.dumps({
        "detector": "torchvision fasterrcnn_resnet50_fpn (COCO) + ground plane", "score_thresh": a.score,
        "extrinsic_json": "metric_extrinsic_h151_dpm031.json", "K": K.tolist(), "lidar2cam": M.tolist(),
        "road_plane_z_in_output": 0.0, "fixed_size_lwh": [L, W, H]}, indent=2))
    print(f"{a.clip} {a.run}: {n} boxes -> {out}")


def _selfcheck():
    """A car's near-bottom pixel maps back to a centre within 0.3 m of the truth."""
    cal = ROOT / "outputs/calibration/camera-data/AV_T_EW_3"
    fx, fy, cx, cy = json.loads((cal / "150_anycalib_pinhole_pinhole.json").read_text())["prediction"]["intrinsics"]
    ext = json.loads((cal / "metric_extrinsic_h151_dpm031.json").read_text())
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    R, t = np.array(ext["rotation"]), np.array(ext["translation"])
    for x, y in ((40.0, -10.0), (90.0, -25.0)):
        corners = [np.array([x + sx * L / 2, y + sy * W / 2, z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, H)]
        uv = np.array([(K @ (R @ p + t))[:2] / (K @ (R @ p + t))[2] for p in corners])
        est = centre((uv[:, 0].min() + uv[:, 0].max()) / 2, uv[:, 1].max(), K, R, t)
        assert np.hypot(est[0] - x, est[1] - y) < 0.3 * (x / 40), (x, y, est)
    print("selfcheck ok (box-bottom ground point lands near the true car centre)")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
