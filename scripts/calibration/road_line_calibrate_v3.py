import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sanitize_tag(text: str):
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    tag = "".join(safe).strip("_")
    return tag or "unknown"


def load_intrinsics_from_prediction(pred_json_path: Path):
    pred = read_json(pred_json_path)
    fx, fy, cx, cy = pred["prediction"]["intrinsics"][:4]
    k = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    model_id = str(pred.get("model_id", "unknown_model"))
    cam_id = str(pred.get("cam_id", "unknown_cam"))
    return k, model_id, cam_id


def load_extrinsic(extrinsic_json_path: Path):
    ext = read_json(extrinsic_json_path)
    r = np.array(ext["rotation"], dtype=np.float64)
    t = np.array(ext["translation"], dtype=np.float64).reshape(3)
    return r, t


def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def delta_rotation(roll_deg, pitch_deg, yaw_deg):
    rx = rot_x(np.deg2rad(roll_deg))
    ry = rot_y(np.deg2rad(pitch_deg))
    rz = rot_z(np.deg2rad(yaw_deg))
    return rz @ ry @ rx


def build_h_ground_to_image(k, r, t):
    h = k @ np.column_stack((r[:, 0], r[:, 1], t))
    return h


def detect_lane_like_segments(image):
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))
    yellow = cv2.inRange(hsv, np.array([10, 50, 80], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(white, yellow)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    lane_edges = cv2.bitwise_and(edges, mask)

    roi = np.zeros_like(lane_edges)
    poly = np.array(
        [[
            (int(0.03 * w), h - 1),
            (int(0.45 * w), int(0.45 * h)),
            (int(0.55 * w), int(0.45 * h)),
            (int(0.97 * w), h - 1),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, poly, 255)
    lane_edges = cv2.bitwise_and(lane_edges, roi)

    lines = cv2.HoughLinesP(
        lane_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=25,
        maxLineGap=20,
    )

    def collect_segments(raw_lines):
        out = []
        if raw_lines is None:
            return out
        for l in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in l]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 22:
                continue
            if abs(dy) < 4:
                continue
            slope_img = abs(dx) / max(abs(dy), 1e-6)
            if slope_img > 5.0:
                continue
            out.append((x1, y1, x2, y2, length))
        return out

    segments = collect_segments(lines)

    if len(segments) < 12:
        roi_edges = cv2.bitwise_and(edges, roi)
        fb_lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=20,
            maxLineGap=25,
        )
        segments = collect_segments(fb_lines)

    return segments, lane_edges


def image_points_to_ground(points_uv, h_img_to_ground):
    pts = np.concatenate([points_uv, np.ones((points_uv.shape[0], 1), dtype=np.float64)], axis=1)
    mapped = (h_img_to_ground @ pts.T).T
    z = mapped[:, 2:3]
    valid = np.abs(z[:, 0]) > 1e-9
    out = np.full((points_uv.shape[0], 2), np.nan, dtype=np.float64)
    out[valid] = mapped[valid, :2] / z[valid]
    return out


def score_calibration(segments, h_img_to_ground):
    if len(segments) == 0:
        return {
            "score": 0.0,
            "valid_segments": 0,
            "mean_abs_dy_dx": 1e9,
            "weighted_length": 0.0,
        }

    dir_scores = []
    weights = []
    slopes = []

    for x1, y1, x2, y2, length in segments:
        uv = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
        ground = image_points_to_ground(uv, h_img_to_ground)
        if np.isnan(ground).any():
            continue

        xg1, yg1 = ground[0]
        xg2, yg2 = ground[1]

        dx = xg2 - xg1
        dy = yg2 - yg1
        if abs(dx) < 1e-6:
            continue

        slope = abs(dy / dx)
        slopes.append(slope)

        dir_score = float(np.exp(-3.0 * slope))
        weight = min(1.0, length / 120.0)

        dir_scores.append(dir_score)
        weights.append(weight)

    if len(dir_scores) == 0:
        return {
            "score": 0.0,
            "valid_segments": 0,
            "mean_abs_dy_dx": 1e9,
            "weighted_length": 0.0,
        }

    weights_np = np.array(weights, dtype=np.float64)
    scores_np = np.array(dir_scores, dtype=np.float64)
    weighted_score = float(np.sum(scores_np * weights_np) / max(np.sum(weights_np), 1e-9))

    coverage = min(1.0, len(dir_scores) / 25.0)
    total_score = 0.8 * weighted_score + 0.2 * coverage

    return {
        "score": float(total_score),
        "valid_segments": int(len(dir_scores)),
        "mean_abs_dy_dx": float(np.mean(slopes)),
        "weighted_length": float(np.sum(weights_np)),
    }


def project_ground_polyline(k, r, t, y_const, x_min=5.0, x_max=80.0, n=100):
    xs = np.linspace(x_min, x_max, n)
    ys = np.full_like(xs, y_const)
    zs = np.zeros_like(xs)
    points = np.stack([xs, ys, zs], axis=1)

    cam = (r @ points.T).T + t.reshape(1, 3)
    zc = cam[:, 2]
    valid = zc > 1e-6
    if np.sum(valid) < 2:
        return None

    cam = cam[valid]
    proj = (k @ cam.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    poly = np.stack([u, v], axis=1)
    return poly


def segment_ground_slope(seg, h_base_inv):
    x1, y1, x2, y2, _ = seg
    uv = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
    ground = image_points_to_ground(uv, h_base_inv)
    if np.isnan(ground).any():
        return None
    xg1, yg1 = ground[0]
    xg2, yg2 = ground[1]
    dx = xg2 - xg1
    dy = yg2 - yg1
    if abs(dx) < 1e-6:
        return None
    return abs(dy / dx), float((yg1 + yg2) * 0.5)


def filter_longitudinal_segments(segments, h_base_inv, lat_thresh):
    kept = []
    rejected = []
    for seg in segments:
        info = segment_ground_slope(seg, h_base_inv)
        if info is None:
            rejected.append(seg)
            continue
        ground_slope, _ = info
        if ground_slope < lat_thresh:
            kept.append(seg)
        else:
            rejected.append(seg)
    return kept, rejected


def filter_with_fallback(segments, h_base_inv, lat_thresh, min_kept=8):
    kept, rejected = filter_longitudinal_segments(segments, h_base_inv, lat_thresh)
    fallback = None
    if len(kept) < min_kept:
        relaxed_thresh = lat_thresh * 2.0
        kept, rejected = filter_longitudinal_segments(segments, h_base_inv, relaxed_thresh)
        fallback = "relaxed_lat_thresh"
        if len(kept) < min_kept:
            kept = list(segments)
            rejected = []
            fallback = "all_segments"
    return kept, rejected, fallback


def segment_homogeneous_line(seg):
    x1, y1, x2, y2, _ = seg
    p1 = np.array([x1, y1, 1.0], dtype=np.float64)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64)
    return np.cross(p1, p2)


def segment_direction(seg):
    x1, y1, x2, y2, _ = seg
    d = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return d / n


def segment_midpoint(seg):
    x1, y1, x2, y2, _ = seg
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def angle_between_dirs(a, b):
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(dot))))


def vp_inlier_mask(segments, vp_xy, angle_deg):
    inliers = []
    for seg in segments:
        seg_dir = segment_direction(seg)
        if seg_dir is None:
            continue
        mid = segment_midpoint(seg)
        to_vp = vp_xy - mid
        vp_norm = np.linalg.norm(to_vp)
        if vp_norm < 1e-6:
            continue
        to_vp_dir = to_vp / vp_norm
        ang = angle_between_dirs(seg_dir, to_vp_dir)
        if ang <= angle_deg or ang >= (180.0 - angle_deg):
            inliers.append(True)
        else:
            inliers.append(False)
    return inliers


def ransac_vanishing_point(segments, n_iters, angle_deg, rng):
    n = len(segments)
    if n < 2:
        return None, [], 0

    lines = [segment_homogeneous_line(seg) for seg in segments]
    best_vp = None
    best_inliers = []
    best_count = -1

    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        vp_h = np.cross(lines[i], lines[j])
        if abs(vp_h[2]) < 1e-6:
            continue
        vp_xy = vp_h[:2] / vp_h[2]
        if not np.isfinite(vp_xy).all():
            continue

        mask = vp_inlier_mask(segments, vp_xy, angle_deg)
        count = int(sum(mask))
        if count > best_count:
            best_count = count
            best_vp = vp_xy
            best_inliers = mask

    return best_vp, best_inliers, best_count


def density_reject_segments(segments, cell_px, dense_count, img_h, img_w):
    if len(segments) == 0:
        return list(segments), [], 0, False

    n_cols = max(1, int(np.ceil(img_w / cell_px)))
    n_rows = max(1, int(np.ceil(img_h / cell_px)))
    cell_counts = np.zeros((n_rows, n_cols), dtype=np.int32)

    mid_cells = []
    for seg in segments:
        mid = segment_midpoint(seg)
        col = int(min(n_cols - 1, max(0, mid[0] // cell_px)))
        row = int(min(n_rows - 1, max(0, mid[1] // cell_px)))
        cell_counts[row, col] += 1
        mid_cells.append((row, col))

    dense_cells = set()
    for row in range(n_rows):
        for col in range(n_cols):
            if cell_counts[row, col] > dense_count:
                dense_cells.add((row, col))

    kept = []
    removed = []
    for seg, (row, col) in zip(segments, mid_cells):
        if (row, col) in dense_cells:
            removed.append(seg)
        else:
            kept.append(seg)

    return kept, removed, len(removed), False


def search_position(value, vmin, vmax, step):
    tol = step * 0.51
    at_min = abs(value - vmin) <= tol
    at_max = abs(value - vmax) <= tol
    if at_min or at_max:
        return "BOUNDARY"
    return "INTERIOR"


def draw_overlay_v3(image, kept_segments, rejected_segments, k, r, t, title_lines, h_base_inv, vp_xy=None):
    out = image.copy()
    fitted_lines_ok = True

    for x1, y1, x2, y2, _ in rejected_segments:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 128), 1, cv2.LINE_AA)

    for x1, y1, x2, y2, _ in kept_segments:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    try:
        cluster_y_tol = 1.75
        clusters = []
        for seg in kept_segments:
            info = segment_ground_slope(seg, h_base_inv)
            if info is None:
                continue
            _, mean_y = info
            placed = False
            for cluster in clusters:
                if abs(cluster["mean_y"] - mean_y) <= cluster_y_tol:
                    cluster["segments"].append(seg)
                    ys = []
                    for s in cluster["segments"]:
                        info_s = segment_ground_slope(s, h_base_inv)
                        if info_s is not None:
                            ys.append(info_s[1])
                    if ys:
                        cluster["mean_y"] = float(np.mean(ys))
                    placed = True
                    break
            if not placed:
                clusters.append({"mean_y": mean_y, "segments": [seg]})

        for cluster in clusters:
            if len(cluster["segments"]) < 2:
                continue
            pts = []
            y_vals = []
            for x1, y1, x2, y2, _ in cluster["segments"]:
                pts.append([x1, y1])
                pts.append([x2, y2])
                y_vals.extend([y1, y2])
            if len(pts) < 2:
                continue
            pts_np = np.array(pts, dtype=np.float32)
            line = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            vx, vy, x0, y0 = [float(v) for v in line[:4]]
            y_min = int(min(y_vals))
            y_max = int(max(y_vals))
            if abs(vy) < 1e-6:
                continue
            t0 = (y_min - y0) / vy
            t1 = (y_max - y0) / vy
            x_a = int(x0 + t0 * vx)
            y_a = y_min
            x_b = int(x0 + t1 * vx)
            y_b = y_max
            cv2.line(out, (x_a, y_a), (x_b, y_b), (0, 255, 0), 3, cv2.LINE_AA)
    except Exception:
        fitted_lines_ok = False

    for y_const in (-9, -6, -3, 0, 3, 6, 9):
        poly = project_ground_polyline(k, r, t, y_const)
        if poly is None:
            continue
        pts = np.round(poly).astype(np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    if vp_xy is not None and np.isfinite(vp_xy).all():
        cx, cy = int(round(vp_xy[0])), int(round(vp_xy[1]))
        cv2.circle(out, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

    y = 26
    for line in title_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22

    return out, fitted_lines_ok


def build_ipm_image(image, k, r, t, x_fwd=(0.0, 80.0), y_lat=(-12.0, 12.0), ppm=8):
    h_img_to_ground = np.linalg.inv(build_h_ground_to_image(k, r, t))

    x_min, x_max = x_fwd
    y_min, y_max = y_lat
    out_w = int(round((x_max - x_min) * ppm))
    out_h = int(round((y_max - y_min) * ppm))
    if out_w < 2 or out_h < 2:
        raise ValueError("IPM output size too small")

    # ground (x forward, y lateral) -> IPM pixel (u right, v down)
    scale = np.array(
        [
            [ppm, 0.0, -x_min * ppm],
            [0.0, ppm, -y_min * ppm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_ground_to_ipm = scale
    h_ipm_to_ground = np.linalg.inv(h_ground_to_ipm)
    h_ipm_to_image = build_h_ground_to_image(k, r, t) @ h_ipm_to_ground

    ipm = cv2.warpPerspective(
        image,
        h_ipm_to_image,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # draw meter grid for sanity check
    for x_m in range(int(x_min), int(x_max) + 1, 10):
        u = int(round((x_m - x_min) * ppm))
        cv2.line(ipm, (u, 0), (u, out_h - 1), (40, 40, 40), 1, cv2.LINE_AA)
    for y_m in range(int(y_min), int(y_max) + 1, 4):
        v = int(round((y_m - y_min) * ppm))
        cv2.line(ipm, (0, v), (out_w - 1, v), (40, 40, 40), 1, cv2.LINE_AA)

    cv2.putText(
        ipm,
        f"IPM top-down | X=[{x_min},{x_max}]m Y=[{y_min},{y_max}]m @ {ppm}px/m",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return ipm


def main():
    parser = argparse.ArgumentParser("Lane-line based road calibration v3 (VP RANSAC + density rejection + IPM)")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument("--pred-json", default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json")
    parser.add_argument("--out-dir", default="outputs/calibration/road_line_v3")
    parser.add_argument("--roll-min", type=float, default=-3.0)
    parser.add_argument("--roll-max", type=float, default=3.0)
    parser.add_argument("--pitch-min", type=float, default=-5.0)
    parser.add_argument("--pitch-max", type=float, default=5.0)
    parser.add_argument("--yaw-min", type=float, default=-5.0)
    parser.add_argument("--yaw-max", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--lat-thresh", type=float, default=1.0)
    parser.add_argument("--vp-iters", type=int, default=1500)
    parser.add_argument("--vp-angle-deg", type=float, default=6.0)
    parser.add_argument("--cell-px", type=int, default=40)
    parser.add_argument("--dense-count", type=int, default=6)
    parser.add_argument("--ipm-ppm", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.data_root)
    image_path = root / "image" / f"{args.sample_id}.jpg"
    extr_path = root / "calib" / "virtuallidar_to_camera" / f"{args.sample_id}.json"
    pred_path = Path(args.pred_json)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    img_h, img_w = image.shape[:2]
    k, model_id, cam_id = load_intrinsics_from_prediction(pred_path)
    model_tag = sanitize_tag(f"{model_id}_{cam_id}")
    r_base, t_base = load_extrinsic(extr_path)

    h_base = build_h_ground_to_image(k, r_base, t_base)
    h_base_inv = np.linalg.inv(h_base)

    segments, _ = detect_lane_like_segments(image)
    total_detected = len(segments)

    # --- Stage 2: VP RANSAC (data-driven, no GT extrinsics) ---
    fallback_vp = False
    vp_px = None
    vp_inlier_count = 0
    vp_rejected = []
    vp_inliers = list(segments)

    try:
        rng = np.random.default_rng(42)
        vp_px, inlier_mask, vp_inlier_count = ransac_vanishing_point(
            segments, args.vp_iters, args.vp_angle_deg, rng
        )
        if vp_px is None or vp_inlier_count < 8:
            fallback_vp = True
            vp_inliers = list(segments)
            vp_rejected = []
            vp_inlier_count = len(vp_inliers)
        else:
            vp_inliers = [seg for seg, ok in zip(segments, inlier_mask) if ok]
            vp_rejected = [seg for seg, ok in zip(segments, inlier_mask) if not ok]
    except Exception:
        fallback_vp = True
        vp_inliers = list(segments)
        vp_rejected = []
        vp_inlier_count = len(vp_inliers)
        vp_px = None

    # --- Stage 3: density rejection on VP inliers ---
    density_removed = 0
    dense_skipped = False
    post_density = list(vp_inliers)
    density_rejected = []

    try:
        kept_dense, density_rejected, density_removed, _ = density_reject_segments(
            vp_inliers, args.cell_px, args.dense_count, img_h, img_w
        )
        if len(kept_dense) < 8:
            dense_skipped = True
            post_density = list(vp_inliers)
            density_removed = 0
            density_rejected = []
        else:
            post_density = kept_dense
    except Exception:
        dense_skipped = True
        post_density = list(vp_inliers)

    # --- Stage 4: v2 lateral ground-slope filter ---
    lat_kept, lat_rejected, lat_fallback = filter_with_fallback(
        post_density, h_base_inv, args.lat_thresh
    )
    final_kept = lat_kept
    final_rejected = list(vp_rejected) + list(density_rejected) + list(lat_rejected)

    baseline = score_calibration(final_kept, h_base_inv)

    best = {
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "r": r_base.copy(),
        "score_info": baseline,
    }

    roll_vals = np.arange(args.roll_min, args.roll_max + 1e-9, args.step)
    pitch_vals = np.arange(args.pitch_min, args.pitch_max + 1e-9, args.step)
    yaw_vals = np.arange(args.yaw_min, args.yaw_max + 1e-9, args.step)

    for roll in roll_vals:
        for pitch in pitch_vals:
            for yaw in yaw_vals:
                dr = delta_rotation(roll, pitch, yaw)
                r_cand = dr @ r_base
                h_cand = build_h_ground_to_image(k, r_cand, t_base)
                if abs(np.linalg.det(h_cand)) < 1e-9:
                    continue
                h_cand_inv = np.linalg.inv(h_cand)
                cand = score_calibration(final_kept, h_cand_inv)
                if cand["score"] > best["score_info"]["score"]:
                    best = {
                        "roll_deg": float(roll),
                        "pitch_deg": float(pitch),
                        "yaw_deg": float(yaw),
                        "r": r_cand,
                        "score_info": cand,
                    }

    vp_for_draw = vp_px if vp_px is not None else None

    left, _ = draw_overlay_v3(
        image,
        final_kept,
        final_rejected,
        k,
        r_base,
        t_base,
        [
            "ROAD-LINE CALIBRATION v3",
            f"MODEL: {model_id} | CAM: {cam_id}",
            "LEFT: baseline | white=kept red=rejected green=fitted blue=guides",
            f"detected={total_detected} VP-inliers={vp_inlier_count} density-removed={density_removed}",
            f"final kept={len(final_kept)} score={baseline['score']:.4f}",
            f"mean |dY/dX|={baseline['mean_abs_dy_dx']:.4f}",
        ],
        h_base_inv,
        vp_for_draw,
    )
    right, fitted_lines_ok = draw_overlay_v3(
        image,
        final_kept,
        final_rejected,
        k,
        best["r"],
        t_base,
        [
            "ROAD-LINE CALIBRATION v3",
            f"MODEL: {model_id} | CAM: {cam_id}",
            "RIGHT: refined | yellow dot = vanishing point",
            f"detected={total_detected} VP-inliers={vp_inlier_count} density-removed={density_removed}",
            f"final kept={len(final_kept)} score={best['score_info']['score']:.4f}",
            f"mean |dY/dX|={best['score_info']['mean_abs_dy_dx']:.4f}",
            f"dRoll={best['roll_deg']:+.2f} dPitch={best['pitch_deg']:+.2f} dYaw={best['yaw_deg']:+.2f}",
        ],
        h_base_inv,
        vp_for_draw,
    )

    compare = np.concatenate([left, right], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{args.sample_id}_road_line_v3_compare.jpg"
    out_ipm = out_dir / f"{args.sample_id}_road_line_v3_ipm.jpg"
    out_json = out_dir / f"{args.sample_id}_road_line_v3_result.json"

    cv2.imwrite(str(out_img), compare)

    personal_dir = Path("personal-documents/calibration/images")
    personal_dir.mkdir(parents=True, exist_ok=True)
    personal_compare = personal_dir / f"{args.sample_id}_road_line_v3_compare.jpg"
    personal_ipm = personal_dir / f"{args.sample_id}_road_line_v3_ipm.jpg"
    shutil.copy2(out_img, personal_compare)

    ipm_ok = False
    ipm_error = None
    try:
        ipm = build_ipm_image(image, k, best["r"], t_base, ppm=args.ipm_ppm)
        cv2.imwrite(str(out_ipm), ipm)
        shutil.copy2(out_ipm, personal_ipm)
        ipm_ok = True
    except Exception as exc:
        ipm_error = str(exc)

    roll_pos = search_position(best["roll_deg"], args.roll_min, args.roll_max, args.step)
    pitch_pos = search_position(best["pitch_deg"], args.pitch_min, args.pitch_max, args.step)
    yaw_pos = search_position(best["yaw_deg"], args.yaw_min, args.yaw_max, args.step)

    result = {
        "sample_id": args.sample_id,
        "task": "road_line_calibration_v3",
        "model_id": model_id,
        "cam_id": cam_id,
        "model_tag": model_tag,
        "method": "VP RANSAC + density rejection + lateral filter + orientation grid search + IPM",
        "pred_intrinsics_json": str(pred_path),
        "baseline": {
            "score": baseline["score"],
            "mean_abs_dy_dx": baseline["mean_abs_dy_dx"],
            "valid_segments": baseline["valid_segments"],
            "final_kept_segments": len(final_kept),
            "rotation": r_base.tolist(),
            "translation": t_base.tolist(),
        },
        "refined": {
            "score": best["score_info"]["score"],
            "mean_abs_dy_dx": best["score_info"]["mean_abs_dy_dx"],
            "valid_segments": best["score_info"]["valid_segments"],
            "delta_roll_deg": best["roll_deg"],
            "delta_pitch_deg": best["pitch_deg"],
            "delta_yaw_deg": best["yaw_deg"],
            "roll_search_position": roll_pos,
            "pitch_search_position": pitch_pos,
            "yaw_search_position": yaw_pos,
            "rotation": best["r"].tolist(),
            "translation": t_base.tolist(),
        },
        "segment_filter": {
            "total_detected": total_detected,
            "vp_inliers": vp_inlier_count,
            "density_removed": density_removed,
            "final_kept": len(final_kept),
            "final_rejected": len(final_rejected),
            "fallback_vp": fallback_vp,
            "dense_skipped": dense_skipped,
            "lat_fallback": lat_fallback,
        },
        "vanishing_point": {
            "pixel_u": float(vp_px[0]) if vp_px is not None else None,
            "pixel_v": float(vp_px[1]) if vp_px is not None else None,
            "vp_iters": args.vp_iters,
            "vp_angle_deg": args.vp_angle_deg,
        },
        "density_filter": {
            "cell_px": args.cell_px,
            "dense_count": args.dense_count,
            "dense_skipped": dense_skipped,
            "removed": density_removed,
        },
        "lateral_filter": {
            "lat_thresh": args.lat_thresh,
            "fallback": lat_fallback,
        },
        "search": {
            "roll_range": [args.roll_min, args.roll_max],
            "pitch_range": [args.pitch_min, args.pitch_max],
            "yaw_range": [args.yaw_min, args.yaw_max],
            "step": args.step,
        },
        "ipm": {
            "ok": ipm_ok,
            "error": ipm_error,
            "ppm": args.ipm_ppm,
            "x_range_m": [0.0, 80.0],
            "y_range_m": [-12.0, 12.0],
        },
        "visual_legend": {
            "white": "final kept clean segments",
            "red_dim": "rejected segments (VP outlier, dense, lateral)",
            "green": "fitted per-lane lines from kept clusters",
            "blue": "calibration ground guide lines",
            "yellow_dot": "detected vanishing point",
            "fitted_lines_drawn": fitted_lines_ok,
        },
        "artifacts": {
            "compare_image": str(out_img.resolve()),
            "ipm_image": str(out_ipm.resolve()) if ipm_ok else None,
            "result_json": str(out_json.resolve()),
            "personal_compare": str(personal_compare.resolve()),
            "personal_ipm": str(personal_ipm.resolve()) if ipm_ok else None,
        },
    }

    write_json(out_json, result)

    print(f"Wrote compare: {out_img.resolve()}")
    if ipm_ok:
        print(f"Wrote IPM: {out_ipm.resolve()}")
    print(f"Wrote result: {out_json.resolve()}")
    print(f"Copied compare to: {personal_compare.resolve()}")
    if ipm_ok:
        print(f"Copied IPM to: {personal_ipm.resolve()}")
    print(
        json.dumps(
            {
                "total_detected": total_detected,
                "vp_inliers": vp_inlier_count,
                "density_removed": density_removed,
                "final_kept": len(final_kept),
                "fallback_vp": fallback_vp,
                "dense_skipped": dense_skipped,
                "lat_fallback": lat_fallback,
                "vp_pixel_u": float(vp_px[0]) if vp_px is not None else None,
                "vp_pixel_v": float(vp_px[1]) if vp_px is not None else None,
                "baseline_score": baseline["score"],
                "refined_score": best["score_info"]["score"],
                "baseline_mean_abs_dy_dx": baseline["mean_abs_dy_dx"],
                "refined_mean_abs_dy_dx": best["score_info"]["mean_abs_dy_dx"],
                "delta_roll_deg": best["roll_deg"],
                "delta_pitch_deg": best["pitch_deg"],
                "delta_yaw_deg": best["yaw_deg"],
                "roll_search_position": roll_pos,
                "pitch_search_position": pitch_pos,
                "yaw_search_position": yaw_pos,
                "ipm_ok": ipm_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
