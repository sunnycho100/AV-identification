"""Derive a road-aligned extrinsic from one frame's lane vanishing point.

Uses lsd_vanishing_point's independently fitted road lines (GT-free).
Given AnyCalib K and an assumed camera height, solves pitch and yaw in closed
form (roll = 0) so the ground frame's X axis runs along the road:

  d_cam  = normalize(K^-1 [u_vp, v_vp, 1])       (road direction in camera coords)
  pitch  = atan2(-dy, dz)                        (makes road horizontal in ground frame)
  yaw    = atan2(d_ground_y, d_ground_x)         (rotates ground X onto the road)

Height stays an input (--height): a vanishing point carries no metric scale.
This is still a smoke-level pose - the metric anchor comes later from
GCPs/solvePnP or OpenTrafficCam3D.

Run:
    .venv/bin/python scripts/calibration/vp_extrinsic_from_frame.py \
        --image data/camera-data/AV_T_WE_1/frames/150.jpg \
        --anycalib-json outputs/calibration/camera-data/AV_T_WE_1/150_anycalib_pinhole_pinhole.json \
        --height 12 \
        --out outputs/calibration/camera-data/AV_T_WE_1/vp_extrinsic.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point


def base_rotation(pitch):
    s, c = math.sin(pitch), math.cos(pitch)
    return np.array([[0, -1, 0], [-s, 0, -c], [c, 0, -s]], dtype=float)


def rot_z(a):
    s, c = math.sin(a), math.cos(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def solve_pose(K, vp_uv, height):
    d_cam = np.linalg.inv(K) @ np.array([vp_uv[0], vp_uv[1], 1.0])
    d_cam /= np.linalg.norm(d_cam)
    dx, dy, dz = d_cam
    pitch = math.atan2(-dy, dz)          # road horizontal => ground-z component 0
    r_pitch = base_rotation(pitch)
    d_ground = r_pitch.T @ d_cam         # z-component ~0 by construction
    yaw = math.atan2(d_ground[1], d_ground[0])
    R = r_pitch @ rot_z(yaw)             # ground X now along the road
    C = np.array([0.0, 0.0, height])     # camera above ground origin
    t = -R @ C
    return R, t, math.degrees(pitch), math.degrees(yaw)


def draw_check_overlay(img, K, R, t, vp_uv, lines, out_path):
    """Ground-frame guide lines (parallel to road X) projected onto the image.

    `lines` are the fitted road lines (normal, offset, length) drawn in white; the
    orange ground guides must end up parallel to them.
    """
    h = img.shape[0]
    for n, d, _ in lines:
        if abs(n[0]) < 1e-9:
            continue
        cv2.line(img, (int(d / n[0]), 0), (int((d - n[1] * h) / n[0]), h),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for y_lat in range(-20, 21, 4):
        pts = []
        for x_fwd in np.linspace(5, 150, 60):
            p_cam = R @ np.array([x_fwd, float(y_lat), 0.0]) + t
            if p_cam[2] <= 0.5:
                continue
            uv = K @ p_cam
            pts.append((int(uv[0] / uv[2]), int(uv[1] / uv[2])))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (255, 160, 0), 2)
    cv2.circle(img, (int(vp_uv[0]), int(vp_uv[1])), 10, (0, 0, 255), 3)
    cv2.putText(img, "VP", (int(vp_uv[0]) + 12, int(vp_uv[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser("VP-derived road-aligned extrinsic from one frame")
    ap.add_argument("--image", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--height", type=float, default=12.0,
                    help="assumed camera height in meters (VP gives no scale)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = json.loads(Path(args.anycalib_json).read_text())["prediction"]["intrinsics"]
    fx, fy, cx, cy = pred[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    img = cv2.imread(args.image)
    vp, lines, diag = lsd_vanishing_point(img)
    if vp is None:
        sys.exit(f"no vanishing point found ({diag.get('reject', 'too few road lines')})")
    print(f"VP = ({vp[0]:.1f}, {vp[1]:.1f}) px from {diag['n_lines']} road lines; "
          f"median residual {diag['residual_median_px']:.1f} px, "
          f"ramp {diag['ramp_corr']:+.2f}")

    R, t, pitch_deg, yaw_deg = solve_pose(K, vp, args.height)
    print(f"solved pose: pitch = {pitch_deg:.1f} deg down, "
          f"yaw = {yaw_deg:.1f} deg (road vs camera axis), height = {args.height} m (assumed)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rotation": R.tolist(), "translation": t.tolist(),
        "diagnostics": {"vp_px": [float(vp[0]), float(vp[1])], **diag},
        "note": (f"VP-derived pose from {Path(args.image).name}: "
                 f"pitch={pitch_deg:.2f}deg yaw={yaw_deg:.2f}deg roll=0 "
                 f"height={args.height}m ASSUMED (no metric anchor yet)"),
    }, indent=2))
    print(f"wrote {out}")

    overlay_path = out.with_name(out.stem + "_check.jpg")
    draw_check_overlay(img, K, R, t, vp, lines, overlay_path)
    print(f"wrote {overlay_path} (orange guides must run parallel to the lanes)")


if __name__ == "__main__":
    main()
