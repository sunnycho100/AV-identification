"""Run BEVHeight (CPU) on arbitrary image frames with file-based calibration.

The non-DAIR runner: takes a directory of frames, an AnyCalib intrinsics JSON,
and a generic extrinsic JSON ({"rotation": 3x3, "translation": [3]},
ground-frame -> camera). Reuses the model/config/render code from
run_bevheight_single.py.

Smoke test (Camera data clip AV_T_WE_1, mock extrinsic):
    .venv/bin/python scripts/object_detection/run_bevheight_generic.py \
        --frames-dir data/camera-data/AV_T_WE_1/frames \
        --anycalib-json outputs/calibration/camera-data/AV_T_WE_1/150_anycalib_pinhole_pinhole.json \
        --extrinsic-json outputs/calibration/camera-data/AV_T_WE_1/mock_extrinsic.json \
        --out-dir outputs/object_detection/camera-data/AV_T_WE_1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluators.result2kitti import get_lidar_3d_8points
from scripts.adapter.calib_to_bevheight_input import (
    build_mats_dict, load_K_from_anycalib, load_extrinsic_json)
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
from scripts.object_detection.run_bevheight_single import (
    CKPT_PATH, SCORE_THRESH, build_model, filter_and_pack, final_dim,
    img_conf, load_checkpoint)


def run_frame(model, image_path, K, lidar2cam):
    img_tensor, mats_dict, img_meta = build_mats_dict(
        str(image_path), K, lidar2cam, final_dim, img_conf)
    img_meta["box_type_3d"] = LiDARInstance3DBoxes
    with torch.no_grad():
        preds = model(img_tensor, mats_dict)
        results = model.get_bboxes(preds, [img_meta])
    boxes = results[0][0].tensor.cpu().numpy()
    scores = results[0][1].cpu().numpy()
    labels = results[0][2].cpu().numpy()
    return filter_and_pack(boxes, scores, labels)


def render_annotated(image_path, preds, K, lidar2cam, out_path):
    img = cv2.imread(str(image_path))
    k34 = np.zeros((3, 4), dtype=np.float64)
    k34[:3, :3] = K
    for det in preds:
        l, w, h = det["l"], det["w"], det["h"]
        center = [det["x"], det["y"], det["z"] + h / 2.0]
        corners = get_lidar_3d_8points([l, w, h], det["yaw"], center)
        corners_cam = (lidar2cam @ np.concatenate(
            [corners, np.ones((8, 1))], axis=1).T).T[:, :3]
        if np.sum(corners_cam[:, 2] > 1e-6) < 4:
            continue
        pts_2d = project_to_image(corners_cam, k34)
        draw_box_3d(img, pts_2d, c=(0, 255, 0))
        u_min = int(np.min(pts_2d[:, 0]))
        v_min = int(np.min(pts_2d[:, 1]))
        cv2.putText(img, f"{det['class_name']} {det['score']:.2f}",
                    (max(0, u_min), max(12, v_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser("BEVHeight on generic frames + file-based calibration")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--extrinsic-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="max frames to process")
    args = ap.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg")) + \
        sorted(Path(args.frames_dir).glob("*.png"))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"No frames in {args.frames_dir}")

    K = load_K_from_anycalib(args.anycalib_json)
    lidar2cam = load_extrinsic_json(args.extrinsic_json)

    model = build_model()
    info = load_checkpoint(model, CKPT_PATH)
    print(f"checkpoint loaded: {info['matched']} keys matched, "
          f"{len(info['missing'])} missing")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        preds = run_frame(model, f, K, lidar2cam)
        (out_dir / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=2))
        render_annotated(f, preds, K, lidar2cam, out_dir / f"{f.stem}_annotated.jpg")
        cars = [d for d in preds if d["class_name"] == "car"]
        print(f"{f.name}: {len(preds)} detections (score>={SCORE_THRESH}), "
              f"{len(cars)} cars")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
