"""Solve metric camera height from the standard US lane width (3.6576 m).

Steps (all GT/GPS-free):
1. Median-blend the extracted frames (static camera) -> vehicle-free background.
2. Fit road lines independently + least-squares vanishing point
   (lsd_vanishing_point), solve pitch/yaw (vp_extrinsic_from_frame).
3. Sweep the camera height; at each one project a grid of ground lines spaced
   one lane apart and score how well it overlays the detected road lines. The
   best-scoring height wins, and the width of the minimum reports confidence.
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

from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
from scripts.calibration.vp_extrinsic_from_frame import solve_pose

LANE_WIDTH_M = 3.6576  # 12 ft US standard
MEASURE_ROW = 780.0    # image row where grid and detected lines are compared


def median_background(frames_dir):
    """Vehicle-free background. Only useful if the camera is truly static: these
    pole-mounted cameras sway, so the blend smears lane edges and measurably
    degrades the line fits (HV_T_EW_2 misfit 17.0 px blended vs 7.8 px single).
    Kept behind --median-background; a single sharp frame is the default."""
    paths = sorted(Path(frames_dir).glob("*.jpg"))
    stack = np.stack([cv2.imread(str(p)) for p in paths])
    return np.median(stack, axis=0).astype(np.uint8), len(paths)


def solve_per_frame(frames_dir, K, lane_width):
    """Solve each frame independently and aggregate.

    One frame is not enough: on HV_T_EW_2 the answer moves 15.95 -> 18.23 m
    between frames of the same clip, because which lines LSD recovers depends on
    passing vehicles and lighting. The spread across frames is the honest error
    bar, so it is reported rather than hidden behind a single pick.
    """
    rows = []
    for path in sorted(Path(frames_dir).glob("*.jpg")):
        img = cv2.imread(str(path))
        vp, lines, diag = lsd_vanishing_point(img)
        if vp is None:
            continue
        line_x = np.array(sorted((d - n[1] * MEASURE_ROW) / n[0]
                                 for n, d, _ in lines if abs(n[0]) > 1e-9))
        if len(line_x) < 4:
            continue
        h, misfit, _ = solve_height(K, vp, line_x, lane_width)
        if h is not None:
            rows.append({"frame": path.name, "vp": [float(vp[0]), float(vp[1])],
                         "height_m": h, "misfit_px": misfit,
                         "n_lines": len(line_x)})
    return rows


def grid_x(K, R, t, lat, row=MEASURE_ROW):
    """Image x where the ground line at lateral offset `lat` crosses `row`.

    A ground line projects to an image line, so two projected points define it
    exactly; no sampling needed.
    """
    pts = []
    for x_fwd in (30.0, 150.0):
        p = R @ np.array([x_fwd, lat, 0.0]) + t
        if p[2] <= 0.5:
            return None
        uv = K @ p
        pts.append((uv[0] / uv[2], uv[1] / uv[2]))
    (u1, v1), (u2, v2) = pts
    if abs(v2 - v1) < 1e-9:
        return None
    return u1 + (u2 - u1) * (row - v1) / (v2 - v1)


def grid_misfit(K, vp, height, line_x, lane_width, n_lanes=14):
    """Symmetric px distance between the projected lane grid and the road lines."""
    R, t, _, _ = solve_pose(K, vp, height)
    return grid_misfit_pose(K, R, t, line_x, lane_width, n_lanes)


def grid_misfit_pose(K, R, t, line_x, lane_width, n_lanes=14):
    """Same score for a pose given directly, so an extrinsic read from a file can
    be graded without going back through its vanishing point.

    Only the grid's *spacing* is under test, so its lateral offset is optimised
    first: a grid that merely sits one lane over must not be penalised.

    Both directions are scored. Measuring only line-to-grid rewards a grid that is
    too dense, since a denser grid always has something near every line; at twice
    the true height the grid projects twice as fine and scored perfectly. Charging
    grid lines that match nothing removes that bias.
    """
    gx = [g for lat in np.arange(-n_lanes * lane_width, n_lanes * lane_width + 1e-6,
                                 lane_width)
          if (g := grid_x(K, R, t, lat)) is not None]
    if len(gx) < 3:
        return np.inf
    gx = np.array(sorted(gx))
    lo, hi = line_x.min(), line_x.max()
    best = np.inf
    for anchor in line_x:                          # slide the grid onto each line
        shifted = gx + (anchor - gx[np.argmin(np.abs(gx - anchor))])
        inside = shifted[(shifted >= lo) & (shifted <= hi)]
        if len(inside) < 2:
            continue
        fwd = np.median([np.min(np.abs(shifted - x)) for x in line_x])
        rev = np.median([np.min(np.abs(line_x - g)) for g in inside])
        best = min(best, float((fwd + rev) / 2))
    return best


def solve_height(K, vp, line_x, lane_width, h_lo=8.0, h_hi=40.0, step=0.25):
    """Height whose lane grid best overlays the detected road lines.

    Scores the whole grid against every line at once, rather than picking one gap
    out of a cluster list, which is what made the old solver return 30.09 m and
    10.81 m for the same camera pose. Returns (height, misfit_px, (lo, hi)) where
    the interval spans every height scoring within 1.5x of the minimum: a wide
    interval means the geometry does not pin the scale down and the number should
    not be trusted on its own.
    """
    hs = np.arange(h_lo, h_hi + 1e-9, step)
    sc = np.array([grid_misfit(K, vp, h, line_x, lane_width) for h in hs])
    if not np.isfinite(sc).any():
        return None, np.inf, (np.nan, np.nan)
    i = int(np.argmin(sc))
    best = hs[i]
    if 0 < i < len(hs) - 1:                        # parabola vertex, sub-step
        den = sc[i - 1] - 2 * sc[i] + sc[i + 1]
        if den != 0:
            best = hs[i] - 0.5 * (sc[i + 1] - sc[i - 1]) / den * step
    near = hs[sc < sc[i] * 1.5]
    return float(best), float(sc[i]), (float(near.min()), float(near.max()))


def main():
    ap = argparse.ArgumentParser("Metric height from lane width")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lane-width", type=float, default=LANE_WIDTH_M)
    ap.add_argument("--median-background", action="store_true",
                    help="blend all frames instead of using one sharp frame; "
                         "only helps if the camera does not sway")
    args = ap.parse_args()

    pred = json.loads(Path(args.anycalib_json).read_text())["prediction"]["intrinsics"]
    fx, fy, cx, cy = pred[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    per = solve_per_frame(args.frames_dir, K, args.lane_width)
    if len(per) < 3:
        sys.exit(f"only {len(per)} frames yielded a solution - not enough to trust")
    hs = np.array([r["height_m"] for r in per])
    q1, q3 = np.percentile(hs, [25, 75])
    print(f"per-frame heights ({len(per)} frames): {np.round(sorted(hs), 2).tolist()}")
    print(f"median {np.median(hs):.2f} m   IQR {q1:.2f}-{q3:.2f} m "
          f"({q3 - q1:.2f} m wide)")
    if q3 - q1 > 1.5:
        print("WARN: heights disagree across frames of the same clip; the lane "
              "grid does not pin the scale down here (needs a depth anchor, "
              "e.g. the 12.19 m dash period)")
    best = min(per, key=lambda r: abs(r["height_m"] - np.median(hs)))
    bg = cv2.imread(str(Path(args.frames_dir) / best["frame"]))
    src = f"{len(per)} frames, median-representative {best['frame']}"
    print(f"source: {src}")

    vp, lines, diag = lsd_vanishing_point(bg)
    if vp is None:
        sys.exit(f"no VP on background image ({diag.get('reject', 'too few road lines')})")
    print(f"background road lines: {diag['n_lines']}, VP=({vp[0]:.1f},{vp[1]:.1f}), "
          f"median residual {diag['residual_median_px']:.1f} px")

    _, _, pitch_deg, yaw_deg = solve_pose(K, vp, height=1.0)   # rotation only
    print(f"pose: pitch={pitch_deg:.1f} yaw={yaw_deg:.1f} (roll=0)")

    line_x = np.array(sorted((d - n[1] * MEASURE_ROW) / n[0]
                             for n, d, _ in lines if abs(n[0]) > 1e-9))
    if len(line_x) < 4:
        sys.exit(f"only {len(line_x)} usable road lines - not enough to fit a grid")

    height = float(np.median(hs))
    misfit = float(np.median([r["misfit_px"] for r in per]))
    h_lo, h_hi = float(q1), float(q3)
    s1 = args.lane_width / height
    print(f"HEIGHT = {height:.2f} m (median of {len(per)} frames), "
          f"median misfit {misfit:.1f} px")

    R_final, t_final, _, _ = solve_pose(K, vp, height=height)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rotation": R_final.tolist(), "translation": t_final.tolist(),
        "note": (f"VP pose (pitch={pitch_deg:.2f} yaw={yaw_deg:.2f} roll=0) + "
                 f"METRIC height {height:.2f} m from {args.lane_width} m lane-width anchor; "
                 f"{diag['n_lines']} independently fitted road lines on {src}"),
        "diagnostics": {
            "vp_px": [float(vp[0]), float(vp[1])], **diag,
            "pitch_deg": pitch_deg, "yaw_deg": yaw_deg,
            "height_m": height,
            "n_road_lines": int(len(line_x)),
            "grid_misfit_px": misfit,
            "height_iqr_m": [h_lo, h_hi],
            "per_frame": per,
            "chosen_spacing_h1": s1,
        },
    }, indent=2))
    print(f"wrote {out}")

    # verification overlay: longitudinal grid lines spaced exactly one lane width
    vis = bg.copy()
    for n, d, _ in lines:
        if abs(n[0]) < 1e-9:
            continue
        h_img = vis.shape[0]
        cv2.line(vis, (int(d / n[0]), 0), (int((d - n[1] * h_img) / n[0]), h_img),
                 (255, 255, 255), 2, cv2.LINE_AA)
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




def _selfcheck():
    """A grid scored against lines generated at a known height must recover it,
    and a wrong height must score clearly worse."""
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    vp, truth_h = (156.6, 21.9), 16.0
    R, t, _, _ = solve_pose(K, vp, truth_h)
    line_x = np.array(sorted(
        g for lat in np.arange(-5 * LANE_WIDTH_M, 5 * LANE_WIDTH_M, LANE_WIDTH_M)
        if (g := grid_x(K, R, t, lat)) is not None))
    assert len(line_x) >= 6, len(line_x)
    got, misfit, (lo, hi) = solve_height(K, vp, line_x, LANE_WIDTH_M)
    assert got is not None and abs(got - truth_h) < 0.15, (got, truth_h)
    assert misfit < 1.0, misfit
    assert grid_misfit(K, vp, truth_h * 1.5, line_x, LANE_WIDTH_M) > misfit * 3
    # doubling the height projects a grid twice as fine; a one-sided score called
    # that a perfect fit, so it must now be rejected too
    assert grid_misfit(K, vp, truth_h * 2.0, line_x, LANE_WIDTH_M) > misfit * 3
    assert lo <= truth_h <= hi and hi - lo < 3.0, (lo, hi)
    print(f"selfcheck ok (recovered {got:.2f} m vs {truth_h} m, misfit {misfit:.2f} px, "
          f"interval {lo:.1f}-{hi:.1f} m)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
