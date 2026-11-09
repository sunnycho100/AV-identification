import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROAD_CLASSES = {
    "car",
    "van",
    "truck",
    "bus",
    "pedestrian",
    "cyclist",
    "motorcyclist",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_k34_from_anycalib(pred_json_path: Path):
    pred_json = read_json(pred_json_path)
    vals = np.array(pred_json["prediction"]["intrinsics"], dtype=np.float64).reshape(4)
    k = np.array(
        [[vals[0], 0.0, vals[2]], [0.0, vals[1], vals[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k34 = np.concatenate([k, np.zeros((3, 1), dtype=np.float64)], axis=1)
    return k, k34


def build_tr_velo_to_cam(extrinsic_json):
    r = np.array(extrinsic_json["rotation"], dtype=np.float64)
    t = np.array(extrinsic_json["translation"], dtype=np.float64).reshape(3)
    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = r
    tr[:3, 3] = t
    return tr


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


def get_lidar_3d_8points(obj_size_lwh, yaw_lidar, center_lidar):
    x, y, z = [float(v) for v in center_lidar]
    l, w, h = obj_size_lwh

    lidar_r = np.array(
        [[np.cos(yaw_lidar), -np.sin(yaw_lidar), 0.0], [np.sin(yaw_lidar), np.cos(yaw_lidar), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    corners = np.array(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h],
        ],
        dtype=np.float64,
    )
    corners = lidar_r @ corners
    corners += np.array([[x], [y], [z]], dtype=np.float64)
    return corners.T


def project_box(corners_lidar, tr_velo_to_cam, k34, image_w, image_h):
    corners_h = np.concatenate([corners_lidar, np.ones((corners_lidar.shape[0], 1), dtype=np.float64)], axis=1)
    corners_cam = (tr_velo_to_cam @ corners_h.T).T
    proj = (k34 @ corners_cam.T).T

    z = proj[:, 2]
    if np.sum(z > 1e-6) < 4:
        return None

    u = proj[:, 0] / np.maximum(z, 1e-6)
    v = proj[:, 1] / np.maximum(z, 1e-6)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        return None

    xmin, ymin = float(np.min(u)), float(np.min(v))
    xmax, ymax = float(np.max(u)), float(np.max(v))

    xmin = max(0.0, min(float(image_w), xmin))
    xmax = max(0.0, min(float(image_w), xmax))
    ymin = max(0.0, min(float(image_h), ymin))
    ymax = max(0.0, min(float(image_h), ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    return [xmin, ymin, xmax, ymax]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def collect_ann_boxes(anns, max_boxes=80):
    out = []
    for ann in anns:
        cls = str(ann.get("type", "")).lower()
        if cls not in ROAD_CLASSES:
            continue
        dims = ann.get("3d_dimensions", {})
        loc = ann.get("3d_location", {})
        box2d = ann.get("2d_box", {})
        try:
            h = float(dims["h"])
            w = float(dims["w"])
            l = float(dims["l"])
            x = float(loc["x"])
            y = float(loc["y"])
            z = float(loc["z"])
            yaw = float(ann["rotation"])
            label_box = [
                float(box2d["xmin"]),
                float(box2d["ymin"]),
                float(box2d["xmax"]),
                float(box2d["ymax"]),
            ]
        except (KeyError, ValueError, TypeError):
            continue

        center = [x, y, z + h / 2.0]
        corners = get_lidar_3d_8points([l, w, h], yaw, center)
        out.append((label_box, corners, cls))
        if len(out) >= max_boxes:
            break
    return out


def evaluate(tr_velo_to_cam, ann_boxes, k34, image_w, image_h):
    ious = []
    bottom_errs = []
    projected = []
    for label_box, corners, cls in ann_boxes:
        proj = project_box(corners, tr_velo_to_cam, k34, image_w, image_h)
        if proj is None:
            continue
        ious.append(iou_xyxy(label_box, proj))
        bottom_errs.append(abs(proj[3] - label_box[3]))
        projected.append((label_box, proj, cls))

    if not ious:
        return {
            "used_boxes": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "mean_bottom_err_px": 1e9,
            "median_bottom_err_px": 1e9,
            "projected": projected,
        }

    return {
        "used_boxes": len(ious),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_bottom_err_px": float(np.mean(bottom_errs)),
        "median_bottom_err_px": float(np.median(bottom_errs)),
        "projected": projected,
    }


def draw_box(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def put_text_block(img, lines, x=12, y0=24, color=(255, 255, 255)):
    y = y0
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 22


def main():
    parser = argparse.ArgumentParser("Road-oriented calibration refinement on a single frame")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument(
        "--pred-json",
        default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json",
    )
    parser.add_argument("--out-dir", default="outputs/calibration/road_calib_single")
    parser.add_argument("--max-boxes", type=int, default=80)
    parser.add_argument("--roll-min", type=float, default=-2.0)
    parser.add_argument("--roll-max", type=float, default=2.0)
    parser.add_argument("--pitch-min", type=float, default=-3.0)
    parser.add_argument("--pitch-max", type=float, default=3.0)
    parser.add_argument("--yaw-min", type=float, default=-2.0)
    parser.add_argument("--yaw-max", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.data_root)
    image_path = root / "image" / f"{args.sample_id}.jpg"
    label_path = root / "label" / "camera" / f"{args.sample_id}.json"
    extr_path = root / "calib" / "virtuallidar_to_camera" / f"{args.sample_id}.json"
    pred_path = Path(args.pred_json)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    h, w = image.shape[:2]

    anns = read_json(label_path)
    ext_json = read_json(extr_path)
    _, k34 = build_k34_from_anycalib(pred_path)

    base_tr = build_tr_velo_to_cam(ext_json)
    base_r = base_tr[:3, :3].copy()

    ann_boxes = collect_ann_boxes(anns, max_boxes=args.max_boxes)
    if not ann_boxes:
        raise RuntimeError("No usable road objects found for calibration objective.")

    baseline_eval = evaluate(base_tr, ann_boxes, k34, w, h)

    roll_vals = np.arange(args.roll_min, args.roll_max + 1e-9, args.step)
    pitch_vals = np.arange(args.pitch_min, args.pitch_max + 1e-9, args.step)
    yaw_vals = np.arange(args.yaw_min, args.yaw_max + 1e-9, args.step)

    best = {
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "eval": baseline_eval,
        "score": (baseline_eval["median_bottom_err_px"], -baseline_eval["mean_iou"]),
        "tr": base_tr.copy(),
    }

    for roll in roll_vals:
        for pitch in pitch_vals:
            for yaw in yaw_vals:
                d_r = delta_rotation(roll, pitch, yaw)
                tr = base_tr.copy()
                tr[:3, :3] = d_r @ base_r
                ev = evaluate(tr, ann_boxes, k34, w, h)
                score = (ev["median_bottom_err_px"], -ev["mean_iou"])
                if score < best["score"]:
                    best = {
                        "roll_deg": float(roll),
                        "pitch_deg": float(pitch),
                        "yaw_deg": float(yaw),
                        "eval": ev,
                        "score": score,
                        "tr": tr,
                    }

    # visualization with white + blue boxes in both panels
    left = image.copy()
    right = image.copy()

    for label_box, proj_box, _ in baseline_eval["projected"]:
        draw_box(left, label_box, (255, 255, 255), 2)
        draw_box(left, proj_box, (255, 0, 0), 2)

    for label_box, proj_box, _ in best["eval"]["projected"]:
        draw_box(right, label_box, (255, 255, 255), 2)
        draw_box(right, proj_box, (255, 0, 0), 2)

    put_text_block(
        left,
        [
            "LEFT: baseline extrinsic (blue) vs label2D (white)",
            f"used={baseline_eval['used_boxes']} meanIoU={baseline_eval['mean_iou']:.4f}",
            f"median bottom err={baseline_eval['median_bottom_err_px']:.2f}px",
        ],
    )
    put_text_block(
        right,
        [
            "RIGHT: road-calibrated extrinsic (blue) vs label2D (white)",
            f"used={best['eval']['used_boxes']} meanIoU={best['eval']['mean_iou']:.4f}",
            f"median bottom err={best['eval']['median_bottom_err_px']:.2f}px",
            f"dRoll={best['roll_deg']:.2f} dPitch={best['pitch_deg']:.2f} dYaw={best['yaw_deg']:.2f}",
        ],
    )

    concat = np.concatenate([left, right], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_img = out_dir / f"{args.sample_id}_road_calib_compare.jpg"
    out_json = out_dir / f"{args.sample_id}_road_calib_result.json"

    cv2.imwrite(str(out_img), concat)

    result = {
        "sample_id": args.sample_id,
        "objective": "minimize median bottom-edge error, tie-break by mean IoU",
        "pred_intrinsics_json": str(pred_path),
        "baseline": {
            "used_boxes": baseline_eval["used_boxes"],
            "mean_iou": baseline_eval["mean_iou"],
            "median_iou": baseline_eval["median_iou"],
            "mean_bottom_err_px": baseline_eval["mean_bottom_err_px"],
            "median_bottom_err_px": baseline_eval["median_bottom_err_px"],
            "rotation": base_tr[:3, :3].tolist(),
            "translation": base_tr[:3, 3].tolist(),
        },
        "refined": {
            "used_boxes": best["eval"]["used_boxes"],
            "mean_iou": best["eval"]["mean_iou"],
            "median_iou": best["eval"]["median_iou"],
            "mean_bottom_err_px": best["eval"]["mean_bottom_err_px"],
            "median_bottom_err_px": best["eval"]["median_bottom_err_px"],
            "delta_roll_deg": best["roll_deg"],
            "delta_pitch_deg": best["pitch_deg"],
            "delta_yaw_deg": best["yaw_deg"],
            "rotation": best["tr"][:3, :3].tolist(),
            "translation": best["tr"][:3, 3].tolist(),
        },
        "search": {
            "roll_range": [args.roll_min, args.roll_max],
            "pitch_range": [args.pitch_min, args.pitch_max],
            "yaw_range": [args.yaw_min, args.yaw_max],
            "step": args.step,
        },
        "artifacts": {
            "compare_image": str(out_img),
            "result_json": str(out_json),
        },
    }

    write_json(out_json, result)

    print(f"Wrote image: {out_img}")
    print(f"Wrote result: {out_json}")
    print(json.dumps({
        "baseline_mean_iou": baseline_eval["mean_iou"],
        "refined_mean_iou": best["eval"]["mean_iou"],
        "baseline_median_bottom_err_px": baseline_eval["median_bottom_err_px"],
        "refined_median_bottom_err_px": best["eval"]["median_bottom_err_px"],
        "delta_roll_deg": best["roll_deg"],
        "delta_pitch_deg": best["pitch_deg"],
        "delta_yaw_deg": best["yaw_deg"],
    }, indent=2))


if __name__ == "__main__":
    main()
