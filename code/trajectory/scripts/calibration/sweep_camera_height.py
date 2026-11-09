"""Sweep the assumed camera height and score BEVHeight's heading self-consistency.

Tests whether the unvalidated 15.39 m VP height is the reason car headings scatter
off the road axis. Holds intrinsics, images and model fixed; only scales the
extrinsic translation, which (camera centre is [0,0,h] in the ground frame) is a
pure change of mount height.

Score is the circular concentration of the doubled heading angle,
R = |mean(exp(2i*yaw))|, over detected cars. Doubling folds the 180-degree
ambiguity so oncoming and receding traffic reinforce instead of cancel, and R is
mode-agnostic, so it does not assume where the road axis lies. R near 1 means all
cars point along one axis; R near 0 means scattered.

Unsupervised on purpose: the GPS trajectory CSVs are held-out evaluation ground
truth and must never feed calibration.

    .venv/bin/python scripts/calibration/sweep_camera_height.py \
        --frames-dir /tmp/sweep_frames \
        --anycalib-json outputs/calibration/camera-data/HV_T_EW_2/150_anycalib_pinhole_pinhole.json \
        --extrinsic-json outputs/calibration/camera-data/HV_T_EW_2/metric_extrinsic.json \
        --heights 6 8 10 12 15.39 18
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def concentration(yaws):
    """|mean(exp(2i*yaw))| — 1.0 = one common axis, 0.0 = uniform scatter."""
    if len(yaws) < 2:
        return float("nan")
    return float(abs(np.mean(np.exp(2j * np.asarray(yaws)))))


def lidar2cam_at_height(R, t, height):
    """Rescale translation to move the camera centre to `height` metres."""
    base = np.linalg.norm(t)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t * (height / base)
    return m, base


def main():
    # imported here so --selfcheck runs without the mmdet3d/mmcv stack
    from scripts.adapter.calib_to_bevheight_input import load_K_from_anycalib
    from scripts.object_detection.run_bevheight_generic import run_frame
    from scripts.object_detection.run_bevheight_single import (
        CKPT_PATH, build_model, load_checkpoint)

    ap = argparse.ArgumentParser("camera-height sweep")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--extrinsic-json", required=True)
    ap.add_argument("--heights", type=float, nargs="+", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    if not frames:
        sys.exit(f"No frames in {args.frames_dir}")

    K = load_K_from_anycalib(args.anycalib_json)
    ext = json.loads(Path(args.extrinsic_json).read_text())
    R = np.array(ext["rotation"])
    t = np.array(ext["translation"])

    model = build_model()
    load_checkpoint(model, CKPT_PATH)

    rows = []
    for h in args.heights:
        lidar2cam, base = lidar2cam_at_height(R, t, h)
        yaws, heights_m = [], []
        for f in frames:
            for d in run_frame(model, f, K, lidar2cam):
                if d["class_name"] == "car" and d["score"] >= args.score_thresh:
                    yaws.append(d["yaw"])
                    heights_m.append(d["h"])
        row = {
            "height_m": h,
            "n_cars": len(yaws),
            "concentration": concentration(yaws),
            "median_box_h": float(np.median(heights_m)) if heights_m else None,
            "is_baseline": abs(h - base) < 1e-6,
        }
        rows.append(row)
        print("h=%5.2f m  cars=%4d  R=%.3f  median box h=%.2f m%s" % (
            h, row["n_cars"], row["concentration"],
            row["median_box_h"] or float("nan"),
            "   <- current calibration" if row["is_baseline"] else ""))

    best = max((r for r in rows if r["n_cars"] >= 10),
               key=lambda r: r["concentration"], default=None)
    if best:
        print("\nbest R at h=%.2f m (R=%.3f); current calibration h=%.2f m"
              % (best["height_m"], best["concentration"], np.linalg.norm(t)))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print("wrote", args.out_json)


def _selfcheck():
    """Aligned headings score ~1, scattered ~0, and 180-flips must not cancel."""
    rng = np.random.default_rng(0)
    aligned = rng.normal(0.0, 0.05, 200)
    flipped = np.concatenate([aligned, aligned + np.pi])
    scattered = rng.uniform(-np.pi, np.pi, 2000)
    assert concentration(aligned) > 0.95, concentration(aligned)
    assert concentration(flipped) > 0.95, concentration(flipped)
    assert concentration(scattered) < 0.15, concentration(scattered)

    R = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    t = np.array([0., 0., 10.])
    m, base = lidar2cam_at_height(R, t, 5.0)
    assert abs(base - 10.0) < 1e-9
    assert abs(np.linalg.norm(m[:3, 3]) - 5.0) < 1e-9
    assert np.allclose(m[:3, :3], R)
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
