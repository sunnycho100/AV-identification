"""GPS-free calibration for the SVBRD-LLM Austin camera (one PTZ pose, clips SV_*).

No calibration ships with that dataset. K comes from AnyCalib on one frame;
pitch and yaw from the main road's vanishing point, the median over the clips'
median backgrounds (lsd_vanishing_point, roll 0; the pose is shared, single
clips disagree by up to 5 deg when lines are few). Height from vehicle size:
with --boxes, every 2D car box of a provisional run (run_2d_ground.py, tag
det2d) is placed on the road at its bottom centre for a trial height, a
4.6 x 1.85 x 1.5 m car aligned with the main road is projected there, and the
height whose projected box height matches the detected one (median log ratio
0) wins. Lane spacing was tried first and is unusable here: per-clip lane
gaps gave 24 to 137 m. A car-size error of +-5% moves every distance and speed
by about the same; ratios between vehicles (AV vs others) do not depend on it.

Writes outputs/calibration/camera-data/<clip>/ the same files the pipeline reads
for Todd Drive: the AnyCalib json (copied), metric_extrinsic_h151_dpm031.json
(name kept so every script finds it; the note says what it is) and an all-road
road_mask.png (an intersection has no single road band; parked cars are dropped
later because they never move).

    .venv/bin/python scripts/calibration/svbrd_calibrate.py --ref SV_355 --clips SV_355 SV_366 SV_373 SV_382 SV_407 SV_411
    (run_2d_ground.py on the lab, tag det2d)
    .venv/bin/python scripts/calibration/svbrd_calibrate.py --ref SV_355 --clips ... --boxes
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point   # noqa: E402
from scripts.calibration.vp_extrinsic_from_frame import solve_pose        # noqa: E402

NAME = "metric_extrinsic_h151_dpm031.json"


def background(clip):
    fr = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))[::10]
    return np.median(np.stack([cv2.imread(str(f)) for f in fr]), 0).astype(np.uint8)


CAR = (4.6, 1.85, 1.5)


def box_height_px(K, R, t, x, y):
    """Projected 2D box height of a standard car centred at road (x, y), along the road."""
    l, w, h = CAR
    pts = np.array([[x + sx * l / 2, y + sy * w / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)])
    c = (R @ pts.T).T + t
    uv = (K @ c.T).T
    v = uv[:, 1] / uv[:, 2]
    return v.max() - v.min()


def height_from_boxes(K, vp, boxes, hs=np.arange(8.0, 60.0, 0.25)):
    """Height where projected standard-car box heights match the detected ones (median log ratio)."""
    best = None
    for h in hs:
        R, t, _, _ = solve_pose(K, vp, h)
        t = np.asarray(t).ravel()
        c = -R.T @ t
        r = []
        for b in boxes:
            u, v = (b[0] + b[2]) / 2, b[3]
            ray = R.T @ np.linalg.solve(K, [u, v, 1.0])
            if ray[2] >= 0:
                continue
            p = c + ray * (-c[2] / ray[2])
            r.append(np.log(box_height_px(K, R, t, p[0], p[1]) / (b[3] - b[1])))
        m = float(np.median(r))
        if best is None or abs(m) < abs(best[1]):
            best = (float(h), m, len(r))
    return best


def main():
    ap = argparse.ArgumentParser("SVBRD camera calibration, GPS-free")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--height", type=float, default=20.0, help="provisional height until --boxes")
    ap.add_argument("--boxes", action="store_true", help="fit height from the det2d run's car boxes")
    a = ap.parse_args()
    ref = ROOT / "outputs/calibration/camera-data" / a.ref
    kj = ref / "150_anycalib_pinhole_pinhole.json"
    fx, fy, cx, cy = json.loads(kj.read_text())["prediction"]["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    vps = []
    for clip in a.clips:
        vp, _, diag = lsd_vanishing_point(background(clip))
        vps.append(vp)
        print(f"{clip}: VP ({vp[0]:.0f}, {vp[1]:.0f}), residual {diag['residual_median_px']:.2f} px")
    vp = np.median(np.array(vps), 0)
    h, how = a.height, "provisional"
    if a.boxes:
        boxes = []
        for clip in a.clips:
            for f in (ROOT / "outputs/object_detection/camera-data" / f"{clip}_det2d").glob("*_pred.json"):
                boxes += [b["box2d"] for b in json.loads(f.read_text()) if b.get("coco_class") == "car"
                          and b["score"] > 0.8 and b["box2d"][3] < 1075 and b["box2d"][1] > 5]
        h, m, n = height_from_boxes(K, vp, boxes)
        how = f"car-size fit on {n} boxes, median log ratio {m:+.3f}"
    R, t, pitch, yaw = solve_pose(K, vp, h)
    for clip in a.clips:
        out = ROOT / "outputs/calibration/camera-data" / clip
        out.mkdir(parents=True, exist_ok=True)
        if out != ref:
            shutil.copy(kj, out / kj.name)
        (out / NAME).write_text(json.dumps({
            "rotation": np.asarray(R).tolist(), "translation": np.asarray(t).ravel().tolist(),
            "note": f"SVBRD Austin camera, GPS-free: median VP pose over {len(a.clips)} clips, height {how} "
                    f"(file name kept for the pipeline; not 15.1 m)",
            "diagnostics": {"height_m": round(h, 2), "pitch_deg": round(float(pitch), 2),
                            "yaw_deg": round(float(yaw), 2), "vp": vp.tolist(),
                            "vp_per_clip": {c: v.tolist() for c, v in zip(a.clips, vps)}}}, indent=2))
        cv2.imwrite(str(out / "road_mask.png"), np.full((1080, 1920), 255, np.uint8))
    print(f"pose: VP ({vp[0]:.0f}, {vp[1]:.0f}), pitch {float(pitch):.2f} deg, height {h:.2f} m ({how})")

if __name__ == "__main__":
    main()
