"""Detect in an ego frame whose x axis follows the camera, then rotate back.

Why. site_extrinsic.py points ground x along the road (the lane vanishing
point), which leaves the camera's optical axis at -26 to -27 deg in the ego
frame on the Todd Drive clips. DAIR's virtuallidar frame has it at about -2 deg,
so the BEV head has only ever seen the camera looking down +x. This reruns the
detector with the ego frame rotated about z by the optical-axis heading, so the
input geometry matches training, and writes every box back in the original road
frame (x, y rotated, yaw + theta). Tracking, refine_yaw and the heading score
then run unchanged on the `_camalign` outputs.

Only the ego frame turns: the camera, K and the road plane are untouched, and z
is unchanged because the rotation is about the vertical.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/orientation/camera_aligned_detect.py \
        --clip AV_T_EW_3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rot_z4(theta):
    c, s = math.cos(theta), math.sin(theta)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def optical_axis_heading(lidar2cam):
    """Heading of the camera's optical axis projected onto the ego ground plane."""
    z = lidar2cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return math.atan2(z[1], z[0])


def align(lidar2cam):
    """(extrinsic of the camera-aligned frame, theta). p_road = Rz(theta) p_aligned."""
    theta = optical_axis_heading(lidar2cam)
    return lidar2cam @ rot_z4(theta), theta


def to_road_frame(preds, theta):
    """Boxes from the aligned frame back to the road frame, in place."""
    c, s = math.cos(theta), math.sin(theta)
    for d in preds:
        x, y = d["x"], d["y"]
        d["x"], d["y"] = c * x - s * y, s * x + c * y
        d["yaw"] = math.atan2(math.sin(d["yaw"] + theta), math.cos(d["yaw"] + theta))
    return preds


def mirror(lidar2cam, K, width):
    """Extrinsic and K for the horizontally flipped image, in the ego frame mirrored
    in y. A world point p seen at pixel u appears at W-1-u in the flipped image and
    at (x, -y, z) in the mirrored frame, so lidar2cam' = Sc lidar2cam Sw."""
    Sc, Sw = np.diag([-1.0, 1, 1, 1]), np.diag([1.0, -1, 1, 1])
    K2 = K.copy()
    K2[0, 2] = width - 1 - K[0, 2]
    return Sc @ lidar2cam @ Sw, K2


def unmirror(preds):
    """Boxes from the mirrored frame back to the road frame, in place."""
    for d in preds:
        d["y"], d["yaw"] = -d["y"], -d["yaw"]
    return preds


def main():
    # heavy imports only here, so --selfcheck runs without torch
    import torch
    from models.bev_height import BEVHeight
    from scripts.adapter.calib_to_bevheight_input import (
        load_K_from_anycalib, load_extrinsic_json)
    from scripts.object_detection import run_bevheight_generic as gen
    from scripts.object_detection.run_bevheight_single import load_checkpoint

    ap = argparse.ArgumentParser("BEVHeight in a camera-aligned ego frame")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--suffix", default="camalign")
    ap.add_argument("--score-thresh", type=float, default=0.45,
                    help="0.45 to match the phase1 runs it is compared against")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="default: run_bevheight_generic's")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rot-deg", type=float, default=None,
                    help="rotate the ego frame by this instead of the optical-axis heading")
    ap.add_argument("--mirror", action="store_true",
                    help="flip the image left-right and mirror the ego frame (no rotation)")
    ap.add_argument("--extrinsic", default="metric_extrinsic_site.json")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else gen.DEFAULT_CKPT
    config = Path(args.config) if args.config else gen.DEFAULT_CONFIG

    cal = ROOT / "outputs/calibration/camera-data" / args.clip
    frames = sorted((ROOT / "data/camera-data" / args.clip / "frames_all").glob("*.jpg"))
    frames = frames[:args.limit] if args.limit else frames
    anycalib = next(cal.glob("*_anycalib_pinhole_pinhole.json"))
    K = load_K_from_anycalib(anycalib)
    road = gen.to_dair_ground(load_extrinsic_json(cal / args.extrinsic))
    K_in = K
    if args.mirror:
        import cv2
        width = cv2.imread(str(frames[0])).shape[1]
        aligned, K_in = mirror(road, K, width)
        theta = 0.0
        mdir = ROOT / "outputs/object_detection/camera-data" / f"{args.clip}_{args.suffix}_frames"
        mdir.mkdir(parents=True, exist_ok=True)
        src = frames
        frames = []
        for f in src:
            cv2.imwrite(str(mdir / f.name), cv2.flip(cv2.imread(str(f)), 1))
            frames.append(mdir / f.name)
        print(f"{args.clip}: image flipped, ego frame mirrored in y, boxes mirrored back")
    else:
        theta = math.radians(args.rot_deg) if args.rot_deg is not None else optical_axis_heading(road)
        aligned = road @ rot_z4(theta)
        print(f"{args.clip}: ego frame rotated {math.degrees(theta):+.2f} deg, boxes written back in the road frame")

    exp = gen.load_exp(config)
    model = BEVHeight(exp.backbone_conf, exp.head_conf)
    model.eval()
    load_checkpoint(model, ckpt)
    model.to(gen.DEVICE)

    out_dir = ROOT / "outputs/object_detection/camera-data" / f"{args.clip}_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_used.json").write_text(json.dumps({
        "checkpoint": str(ckpt), "config": str(config), "mirror": args.mirror,
        "anycalib_json": str(anycalib), "score_thresh": args.score_thresh,
        "ego_rotation_deg": math.degrees(theta), "boxes_written_in": "road frame",
        "ground_shift_applied_m": -gen.DAIR_GROUND_Z,
        "road_plane_z_in_output": gen.DAIR_GROUND_Z,
        "K": K.tolist(), "lidar2cam": road.tolist(), "lidar2cam_model_input": aligned.tolist(),
    }, indent=2))
    with torch.no_grad():
        for f in frames:
            preds = gen.run_frame(model, f, K_in, aligned, exp, args.score_thresh)
            preds = unmirror(preds) if args.mirror else to_road_frame(preds, theta)
            (out_dir / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=2))
            print(f"{f.name}: {sum(d['class_name'] == 'car' for d in preds)} cars")
    print(f"saved to {out_dir}")


def _selfcheck():
    """The rotation must leave every camera-frame point where it was, put the
    optical axis on +x, and round-trip a box back to the road frame exactly."""
    from scripts.calibration.vp_extrinsic_from_frame import solve_pose
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    R, t, _, _ = solve_pose(K, (156.6, 21.9), 16.26)
    road = np.eye(4)
    road[:3, :3], road[:3, 3] = R, t
    aligned, theta = align(road)
    assert abs(math.degrees(theta) + 26.9) < 1.5, math.degrees(theta)  # the measured offset
    assert abs(optical_axis_heading(aligned)) < 1e-9

    p_road = np.array([60.0, -4.0, -1.73, 1.0])
    p_aligned = rot_z4(-theta) @ p_road
    assert np.allclose(road @ p_road, aligned @ p_aligned), "camera point moved"

    yaw_road = math.pi - 0.05                         # an oncoming car
    box = [{"x": p_aligned[0], "y": p_aligned[1], "yaw": yaw_road - theta}]
    back = to_road_frame(box, theta)[0]
    assert np.allclose([back["x"], back["y"]], p_road[:2], atol=1e-9), back
    assert abs(math.remainder(back["yaw"] - yaw_road, 2 * math.pi)) < 1e-9, back
    # mirror: same pixel after the flip, and a box maps back to the road frame
    M2, K2 = mirror(road, K, 1920)
    q, q2 = K @ (road @ p_road)[:3], K2 @ (M2 @ (p_road * [1, -1, 1, 1]))[:3]
    assert np.allclose([1919 - q[0] / q[2], q[1] / q[2]], q2[:2] / q2[2]), (q, q2)
    assert abs(np.linalg.det(M2[:3, :3]) - 1) < 1e-9
    back = unmirror([{"y": -p_road[1], "yaw": -yaw_road}])[0]
    assert np.isclose(back["y"], p_road[1]) and np.isclose(back["yaw"], yaw_road)
    print(f"selfcheck ok (optical axis {math.degrees(theta):+.2f} deg, point and box round-trip)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
