import argparse
import json
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
    # ground plane is lidar z=0, so camera projection is K * [r1 r2 t]
    h = k @ np.column_stack((r[:, 0], r[:, 1], t))
    return h


def detect_lane_like_segments(image):
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # white mask
    white = cv2.inRange(hsv, np.array([0, 0, 180], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))
    # yellow-ish mask
    yellow = cv2.inRange(hsv, np.array([10, 50, 80], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(white, yellow)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    # combine color prior and edges
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
        # Fallback: use pure edge ROI to avoid missing faded lane markings.
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

        # lane lines in ground should be near longitudinal direction: dY/dX close to 0
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

    # reward having usable segments
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


def draw_overlay(image, segments, k, r, t, title_lines):
    out = image.copy()

    # white: detected lane/road-edge segments
    for x1, y1, x2, y2, _ in segments:
        # Draw a dark underlay first so white lines stay visible on bright road markings.
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    # blue: calibration-derived road guide lines
    for y_const in (-9, -6, -3, 0, 3, 6, 9):
        poly = project_ground_polyline(k, r, t, y_const)
        if poly is None:
            continue
        pts = np.round(poly).astype(np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    y = 26
    for line in title_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    return out


def main():
    parser = argparse.ArgumentParser("Lane-line based road calibration on one sample")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument("--pred-json", default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json")
    parser.add_argument("--out-dir", default="outputs/calibration/road_line_calib_single")
    parser.add_argument("--roll-min", type=float, default=-3.0)
    parser.add_argument("--roll-max", type=float, default=3.0)
    parser.add_argument("--pitch-min", type=float, default=-3.0)
    parser.add_argument("--pitch-max", type=float, default=3.0)
    parser.add_argument("--yaw-min", type=float, default=-3.0)
    parser.add_argument("--yaw-max", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.data_root)
    image_path = root / "image" / f"{args.sample_id}.jpg"
    extr_path = root / "calib" / "virtuallidar_to_camera" / f"{args.sample_id}.json"
    pred_path = Path(args.pred_json)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    k, model_id, cam_id = load_intrinsics_from_prediction(pred_path)
    model_tag = sanitize_tag(f"{model_id}_{cam_id}")
    r_base, t_base = load_extrinsic(extr_path)

    segments, _ = detect_lane_like_segments(image)

    h_base = build_h_ground_to_image(k, r_base, t_base)
    h_base_inv = np.linalg.inv(h_base)
    baseline = score_calibration(segments, h_base_inv)

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
                cand = score_calibration(segments, h_cand_inv)
                if cand["score"] > best["score_info"]["score"]:
                    best = {
                        "roll_deg": float(roll),
                        "pitch_deg": float(pitch),
                        "yaw_deg": float(yaw),
                        "r": r_cand,
                        "score_info": cand,
                    }

    left = draw_overlay(
        image,
        segments,
        k,
        r_base,
        t_base,
        [
            "TASK: ROAD-LINE CALIBRATION",
            f"MODEL: {model_id} | CAM: {cam_id}",
            "LEFT: baseline road guides (blue) + detected lines (white)",
            f"score={baseline['score']:.4f} valid={baseline['valid_segments']}",
            f"detected={len(segments)}",
            f"mean |dY/dX|={baseline['mean_abs_dy_dx']:.4f}",
        ],
    )
    right = draw_overlay(
        image,
        segments,
        k,
        best["r"],
        t_base,
        [
            "TASK: ROAD-LINE CALIBRATION",
            f"MODEL: {model_id} | CAM: {cam_id}",
            "RIGHT: refined road guides (blue) + detected lines (white)",
            f"score={best['score_info']['score']:.4f} valid={best['score_info']['valid_segments']}",
            f"detected={len(segments)}",
            f"mean |dY/dX|={best['score_info']['mean_abs_dy_dx']:.4f}",
            f"dRoll={best['roll_deg']:+.2f} dPitch={best['pitch_deg']:+.2f} dYaw={best['yaw_deg']:+.2f}",
        ],
    )

    compare = np.concatenate([left, right], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{args.sample_id}_road_line_calib_compare.jpg"
    out_json = out_dir / f"{args.sample_id}_road_line_calib_result.json"

    cv2.imwrite(str(out_img), compare)

    result = {
        "sample_id": args.sample_id,
        "task": "road_line_calibration",
        "model_id": model_id,
        "cam_id": cam_id,
        "model_tag": model_tag,
        "method": "lane-line based orientation refinement using ground-plane alignment score",
        "pred_intrinsics_json": str(pred_path),
        "baseline": {
            "score": baseline["score"],
            "valid_segments": baseline["valid_segments"],
            "mean_abs_dy_dx": baseline["mean_abs_dy_dx"],
            "weighted_length": baseline["weighted_length"],
            "rotation": r_base.tolist(),
            "translation": t_base.tolist(),
        },
        "refined": {
            "score": best["score_info"]["score"],
            "valid_segments": best["score_info"]["valid_segments"],
            "mean_abs_dy_dx": best["score_info"]["mean_abs_dy_dx"],
            "weighted_length": best["score_info"]["weighted_length"],
            "delta_roll_deg": best["roll_deg"],
            "delta_pitch_deg": best["pitch_deg"],
            "delta_yaw_deg": best["yaw_deg"],
            "rotation": best["r"].tolist(),
            "translation": t_base.tolist(),
        },
        "search": {
            "roll_range": [args.roll_min, args.roll_max],
            "pitch_range": [args.pitch_min, args.pitch_max],
            "yaw_range": [args.yaw_min, args.yaw_max],
            "step": args.step,
        },
        "visual_legend": {
            "white": "detected lane/road-edge lines",
            "blue": "calibration-derived road guide lines",
        },
        "artifacts": {
            "compare_image": str(out_img),
            "result_json": str(out_json),
        },
    }

    write_json(out_json, result)

    print(f"Wrote image: {out_img}")
    print(f"Wrote result: {out_json}")
    print(
        json.dumps(
            {
                "baseline_score": baseline["score"],
                "refined_score": best["score_info"]["score"],
                "detected_segments": len(segments),
                "baseline_mean_abs_dy_dx": baseline["mean_abs_dy_dx"],
                "refined_mean_abs_dy_dx": best["score_info"]["mean_abs_dy_dx"],
                "delta_roll_deg": best["roll_deg"],
                "delta_pitch_deg": best["pitch_deg"],
                "delta_yaw_deg": best["yaw_deg"],
                "valid_segments": best["score_info"]["valid_segments"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
