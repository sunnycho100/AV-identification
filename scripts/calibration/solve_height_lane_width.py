"""Solve metric camera height from the standard US lane width (3.6576 m).

Steps (all GT/GPS-free):
1. Median-blend the extracted frames (static camera) -> vehicle-free background.
2. Detect lane segments + RANSAC vanishing point on the background
   (reuses road_line_calibrate_v3), solve pitch/yaw (vp_extrinsic_from_frame).
3. Backproject segment midpoints to the ground plane at h=1; cluster their
   lateral (ground-Y) offsets into lane lines; adjacent-cluster spacing scales
   linearly with h, so h = 3.6576 / median_spacing_at_h1.
4. Write the metric extrinsic + a verification overlay with a 3.66 m grid.

Run:
    .venv/bin/python scripts/calibration/solve_height_lane_width.py \
        --frames-dir data/camera-data/AV_T_WE_1/frames \
        --anycalib-json outputs/calibration/camera-data/AV_T_WE_1/150_anycalib_pinhole_pinhole.json \
        --out outputs/calibration/camera-data/AV_T_WE_1/metric_extrinsic.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.calibration.road_line_calibrate_v3 import (
    detect_lane_like_segments, ransac_vanishing_point)
from scripts.calibration.vp_extrinsic_from_frame import solve_pose

LANE_WIDTH_M = 3.6576  # 12 ft US standard


def median_background(frames_dir):
    paths = sorted(Path(frames_dir).glob("*.jpg"))
    stack = np.stack([cv2.imread(str(p)) for p in paths])
    return np.median(stack, axis=0).astype(np.uint8), len(paths)


def ground_point(uv, K, R, t, eps=1e-9):
    """Intersect the pixel ray with the z=0 ground plane; None if ray points up."""
    d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d = R.T @ d_cam                      # ray direction in ground frame
    C = -R.T @ t                         # camera center in ground frame
    if d[2] > -eps:
        return None
    s = -C[2] / d[2]
    return C + s * d


def cluster_1d(values, weights, gap_frac=0.35):
    """Sort and split where the gap exceeds gap_frac * median lane guess."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    diffs = np.diff(v)
    typical = np.median(diffs[diffs > 1e-6]) if len(diffs) else 0
    thresh = max(typical * 1.5, 1e-6) * gap_frac / 0.35
    clusters, cur_v, cur_w = [], [v[0]], [w[0]]
    for val, wt, gap in zip(v[1:], w[1:], diffs):
        if gap > thresh:
            clusters.append(np.average(cur_v, weights=cur_w))
            cur_v, cur_w = [], []
        cur_v.append(val)
        cur_w.append(wt)
    clusters.append(np.average(cur_v, weights=cur_w))
    return np.array(clusters)


def main():
    ap = argparse.ArgumentParser("Metric height from lane width")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lane-width", type=float, default=LANE_WIDTH_M)
    args = ap.parse_args()

    pred = json.loads(Path(args.anycalib_json).read_text())["prediction"]["intrinsics"]
    fx, fy, cx, cy = pred[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    bg, n_frames = median_background(args.frames_dir)
    print(f"background: median of {n_frames} frames")

    segments, _ = detect_lane_like_segments(bg)
    rng = np.random.default_rng(0)
    vp, inlier_mask, _ = ransac_vanishing_point(segments, 1500, 6.0, rng)
    if vp is None:
        sys.exit("no VP on background image")
    inliers = [s for s, k in zip(segments, inlier_mask) if k]
    print(f"background segments: {len(segments)}, VP inliers: {len(inliers)}, "
          f"VP=({vp[0]:.0f},{vp[1]:.0f})")

    R, t1, pitch_deg, yaw_deg = solve_pose(K, vp, height=1.0)  # h=1 reference
    print(f"pose: pitch={pitch_deg:.1f} yaw={yaw_deg:.1f} (roll=0)")

    ys, ws = [], []
    for x1, y1, x2, y2, length in inliers:
        mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        p = ground_point(mid, K, R, t1)
        if p is None or p[0] <= 0:
            continue
        ys.append(p[1])
        ws.append(length)
    if len(ys) < 4:
        sys.exit(f"only {len(ys)} usable lane points - not enough to cluster")

    centers = np.sort(cluster_1d(ys, ws))
    # physical prior: camera height 8-40 m => valid lane spacing at h=1 in
    # [w/40, w/8]. Same-paint-line splits are far below this band.
    s_min, s_max = args.lane_width / 40.0, args.lane_width / 8.0
    # merge over-split clusters (closer than half the minimum valid spacing)
    merged = [centers[0]]
    for c in centers[1:]:
        if c - merged[-1] < s_min / 2:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    merged = np.array(merged)
    # candidate spacings: all pairwise gaps inside the physical band
    diffs = np.abs(merged[:, None] - merged[None, :])[np.triu_indices(len(merged), 1)]
    cands = np.sort(diffs[(diffs >= s_min) & (diffs <= s_max)])
    print(f"lane-line clusters: {len(centers)} -> {len(merged)} after merge; "
          f"candidate spacings in [{s_min:.3f},{s_max:.3f}]: {np.round(cands, 4).tolist()}")
    if len(cands) == 0:
        sys.exit("FAIL: no lane spacing inside the 8-40 m height band - "
                 "bad VP or poor lane visibility; do not trust this pose")
    # smallest well-supported peak = fundamental lane width (multiples are
    # 2-lane gaps). support = candidates within +-15%.
    best = None
    for c in cands:
        group = cands[(cands > 0.85 * c) & (cands < 1.15 * c)]
        support = len(group)
        if support >= 2 or len(cands) == 1:
            best = float(np.median(group))
            break
    if best is None:
        best = float(cands[0])
        print("WARN: no spacing had support >=2; using smallest candidate")
    s1 = best
    height = args.lane_width / s1
    print(f"chosen spacing (h=1): {s1:.4f} -> "
          f"HEIGHT = {args.lane_width}/{s1:.4f} = {height:.2f} m")

    R_final, t_final, _, _ = solve_pose(K, vp, height=height)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rotation": R_final.tolist(), "translation": t_final.tolist(),
        "note": (f"VP pose (pitch={pitch_deg:.2f} yaw={yaw_deg:.2f} roll=0) + "
                 f"METRIC height {height:.2f} m from {args.lane_width} m lane-width anchor; "
                 f"{len(inliers)} VP-inlier segments on median background of {n_frames} frames"),
        "diagnostics": {
            "vp_px": [float(vp[0]), float(vp[1])],
            "pitch_deg": pitch_deg, "yaw_deg": yaw_deg,
            "height_m": height,
            "n_clusters": int(len(centers)),
            "candidate_spacings_h1": cands.tolist(),
            "chosen_spacing_h1": s1,
        },
    }, indent=2))
    print(f"wrote {out}")

    # verification overlay: longitudinal grid lines spaced exactly one lane width
    vis = bg.copy()
    for x1, y1, x2, y2, _ in inliers:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
    for i in range(-8, 9):
        y_lat = i * args.lane_width
        pts = []
        for x_fwd in np.linspace(3, 200, 80):
            p_cam = R_final @ np.array([x_fwd, y_lat, 0.0]) + t_final
            if p_cam[2] <= 0.5:
                continue
            uv = K @ p_cam
            pts.append((int(uv[0] / uv[2]), int(uv[1] / uv[2])))
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, (255, 160, 0), 2)
    cv2.putText(vis, f"grid spacing = {args.lane_width} m | solved height = {height:.2f} m",
                (30, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    check = out.with_name(out.stem + "_check.jpg")
    cv2.imwrite(str(check), vis)
    print(f"wrote {check} (orange grid must land ON adjacent lane lines)")


if __name__ == "__main__":
    main()
