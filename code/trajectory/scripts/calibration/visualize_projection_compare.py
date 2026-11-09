import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_tag(text: str):
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    tag = "".join(safe).strip("_")
    return tag or "unknown"


def get_tr_velo_to_cam(extrinsic_json):
    r_velo2cam = np.array(extrinsic_json["rotation"], dtype=np.float64)
    t_velo2cam = np.array(extrinsic_json["translation"], dtype=np.float64).reshape(3)
    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = r_velo2cam
    tr[:3, 3] = t_velo2cam
    return tr


def get_lidar_3d_8points(obj_size_lwh, yaw_lidar, center_lidar):
    cx, cy, cz = float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])
    l, w, h = obj_size_lwh

    lidar_r = np.array(
        [
            [np.cos(yaw_lidar), -np.sin(yaw_lidar), 0.0],
            [np.sin(yaw_lidar), np.cos(yaw_lidar), 0.0],
            [0.0, 0.0, 1.0],
        ],
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
    corners += np.array([[cx], [cy], [cz]], dtype=np.float64)
    return corners.T


def project_box(corners_lidar, tr_velo_to_cam, k34, image_w, image_h):
    corners_h = np.concatenate([corners_lidar, np.ones((corners_lidar.shape[0], 1), dtype=np.float64)], axis=1)
    corners_cam = (tr_velo_to_cam @ corners_h.T).T
    proj = (k34 @ corners_cam.T).T

    eps = 1e-6
    z = proj[:, 2]
    valid_depth = np.sum(z > eps) >= 4
    if not valid_depth:
        return None

    u = proj[:, 0] / np.maximum(z, eps)
    v = proj[:, 1] / np.maximum(z, eps)

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
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def draw_box(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def put_legend(img, rows):
    y = 24
    for text, color in rows:
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24


def main():
    parser = argparse.ArgumentParser("Visualize GT vs predicted intrinsics projection quality")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument(
        "--pred-json",
        default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json",
    )
    parser.add_argument("--max-boxes", type=int, default=40)
    parser.add_argument("--out-dir", default="outputs/calibration/anycalib_single")
    args = parser.parse_args()

    root = Path(args.data_root)
    image_path = root / "image" / f"{args.sample_id}.jpg"
    label_path = root / "label" / "camera" / f"{args.sample_id}.json"
    intrinsic_path = root / "calib" / "camera_intrinsic" / f"{args.sample_id}.json"
    extrinsic_path = root / "calib" / "virtuallidar_to_camera" / f"{args.sample_id}.json"
    pred_path = Path(args.pred_json)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    image_h, image_w = image.shape[:2]

    anns = read_json(label_path)
    intr_gt_json = read_json(intrinsic_path)
    ext_json = read_json(extrinsic_path)
    pred_json = read_json(pred_path)
    model_id = str(pred_json.get("model_id", "unknown_model"))
    cam_id = str(pred_json.get("cam_id", "unknown_cam"))
    model_tag = sanitize_tag(f"{model_id}_{cam_id}")

    k_gt = np.array(intr_gt_json["cam_K"], dtype=np.float64).reshape(3, 3)
    k_pred = np.array(pred_json["prediction"]["intrinsics"], dtype=np.float64).reshape(4)
    k_pred = np.array(
        [
            [k_pred[0], 0.0, k_pred[2]],
            [0.0, k_pred[1], k_pred[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    k_gt_34 = np.concatenate([k_gt, np.zeros((3, 1), dtype=np.float64)], axis=1)
    k_pred_34 = np.concatenate([k_pred, np.zeros((3, 1), dtype=np.float64)], axis=1)
    tr_velo_to_cam = get_tr_velo_to_cam(ext_json)

    left = image.copy()
    right = image.copy()

    ious_gt = []
    ious_pred = []
    used = 0

    for ann in anns:
        if used >= args.max_boxes:
            break

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

        proj_gt = project_box(corners, tr_velo_to_cam, k_gt_34, image_w, image_h)
        proj_pred = project_box(corners, tr_velo_to_cam, k_pred_34, image_w, image_h)
        if proj_gt is None or proj_pred is None:
            continue

        draw_box(left, label_box, (255, 255, 255), 2)
        draw_box(left, proj_gt, (255, 0, 0), 2)

        draw_box(right, label_box, (255, 255, 255), 2)
        draw_box(right, proj_pred, (0, 0, 255), 2)

        ious_gt.append(iou_xyxy(label_box, proj_gt))
        ious_pred.append(iou_xyxy(label_box, proj_pred))
        used += 1

    put_legend(
        left,
        [
            ("TASK: OBJECT CALIBRATION", (255, 255, 255)),
            (f"MODEL: {model_id} | CAM: {cam_id}", (255, 255, 255)),
            ("LEFT: label2D (white) vs GT calibration projection (blue)", (255, 255, 255)),
            (f"used_boxes={used}", (255, 255, 255)),
        ],
    )
    put_legend(
        right,
        [
            ("TASK: OBJECT CALIBRATION", (255, 255, 255)),
            (f"MODEL: {model_id} | CAM: {cam_id}", (255, 255, 255)),
            ("RIGHT: label2D (white) vs AnyCalib projection (red)", (255, 255, 255)),
            (f"used_boxes={used}", (255, 255, 255)),
        ],
    )

    concat = np.concatenate([left, right], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{args.sample_id}_projection_compare.jpg"
    out_json = out_dir / f"{args.sample_id}_projection_compare_metrics.json"

    cv2.imwrite(str(out_img), concat)

    metrics = {
        "sample_id": args.sample_id,
        "task": "object_calibration_projection_compare",
        "model_id": model_id,
        "cam_id": cam_id,
        "model_tag": model_tag,
        "used_boxes": used,
        "mean_iou_label_vs_gt_projection": float(np.mean(ious_gt)) if ious_gt else 0.0,
        "mean_iou_label_vs_anycalib_projection": float(np.mean(ious_pred)) if ious_pred else 0.0,
        "median_iou_label_vs_gt_projection": float(np.median(ious_gt)) if ious_gt else 0.0,
        "median_iou_label_vs_anycalib_projection": float(np.median(ious_pred)) if ious_pred else 0.0,
        "pred_json": str(pred_path),
        "output_image": str(out_img),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote image: {out_img}")
    print(f"Wrote metrics: {out_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
