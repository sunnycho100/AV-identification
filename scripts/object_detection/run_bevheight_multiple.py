"""Run BEVHeight 3D detection on a consecutive DAIR frame range (CPU).

Uses our AnyCalib intrinsics (shared across frames) and per-frame GT extrinsics.
Outputs land in outputs/object_detection/our_intrinsics/.

Default range 000000–000022 (23 consecutive frames in v2x-i).

  /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/object_detection/run_bevheight_multiple.py
  /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/object_detection/run_bevheight_multiple.py --start 0 --end 22
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluators.result2kitti import get_lidar_3d_8points
from scripts.adapter.calib_to_bevheight_input import build_mats_dict, load_K_from_anycalib
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
from scripts.object_detection.run_bevheight_single import (
    CLASSES,
    SCORE_THRESH,
    build_model,
    filter_and_pack,
    final_dim,
    img_conf,
    load_checkpoint,
    load_gt_extrinsic,
)

CKPT_PATH = ROOT / "checkpoints/BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt"
OUT_DIR = ROOT / "outputs/object_detection/our_intrinsics"


def frame_id(i: int) -> str:
    return f"{i:06d}"


def run_inference(
    model,
    image_path: Path,
    K: np.ndarray,
    lidar2cam: np.ndarray,
) -> tuple:
    img_tensor, mats_dict, img_meta = build_mats_dict(
        str(image_path), K, lidar2cam, final_dim, img_conf,
    )
    img_meta["box_type_3d"] = LiDARInstance3DBoxes
    img_metas = [img_meta]

    with torch.no_grad():
        preds = model(img_tensor, mats_dict)
        results = model.get_bboxes(preds, img_metas)

    boxes = results[0][0].tensor.cpu().numpy()
    scores = results[0][1].cpu().numpy()
    labels = results[0][2].cpu().numpy()
    return boxes, scores, labels


def render_annotated(
    preds: list,
    K: np.ndarray,
    lidar2cam: np.ndarray,
    image_path: Path,
    out_path: Path,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    k34 = np.zeros((3, 4), dtype=np.float64)
    k34[:3, :3] = K

    for det in preds:
        l, w, h = det["l"], det["w"], det["h"]
        center = [det["x"], det["y"], det["z"] + h / 2.0]
        corners = get_lidar_3d_8points([l, w, h], det["yaw"], center)
        corners_cam = (lidar2cam @ np.concatenate(
            [corners, np.ones((8, 1), dtype=np.float64)], axis=1,
        ).T).T[:, :3]
        if np.sum(corners_cam[:, 2] > 1e-6) < 4:
            continue
        pts_2d = project_to_image(corners_cam, k34)
        draw_box_3d(img, pts_2d, c=(0, 255, 0))
        u_min = int(np.min(pts_2d[:, 0]))
        v_min = int(np.min(pts_2d[:, 1]))
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.putText(
            img, label, (max(0, u_min), max(12, v_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img)


def process_frame(
    model,
    data_root: Path,
    fid: str,
    K: np.ndarray,
    out_dir: Path,
) -> int:
    image_path = data_root / "image" / f"{fid}.jpg"
    extrinsic_path = data_root / "calib/virtuallidar_to_camera" / f"{fid}.json"
    for p in (image_path, extrinsic_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    lidar2cam = load_gt_extrinsic(extrinsic_path)
    boxes, scores, labels = run_inference(model, image_path, K, lidar2cam)
    preds = filter_and_pack(boxes, scores, labels)

    pred_json = out_dir / f"{fid}_pred.json"
    pred_json.write_text(json.dumps(preds, indent=2))
    annotated = out_dir / f"{fid}_annotated.jpg"
    render_annotated(preds, K, lidar2cam, image_path, annotated)

    print(f"  {fid}: {len(preds)} detections -> {pred_json.name}, {annotated.name}")
    return len(preds)


def main():
    parser = argparse.ArgumentParser(description="BEVHeight batch inference (our intrinsics)")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument("--start", type=int, default=0, help="first frame index (inclusive)")
    parser.add_argument("--end", type=int, default=22, help="last frame index (inclusive)")
    parser.add_argument(
        "--anycalib-json",
        default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json",
    )
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be <= --end")

    data_root = ROOT / args.data_root
    frame_ids = [frame_id(i) for i in range(args.start, args.end + 1)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    K = load_K_from_anycalib(str(ROOT / args.anycalib_json))
    model = build_model()
    load_info = load_checkpoint(model, CKPT_PATH)

    print("=== BEVHeight batch (our intrinsics) ===")
    print(f"  frames: {frame_ids[0]} .. {frame_ids[-1]} ({len(frame_ids)} total)")
    print(f"  data_root: {data_root}")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  checkpoint matched keys: {load_info['matched']}")
    print(f"  score_thresh: {SCORE_THRESH}")

    t0 = time.time()
    counts = []
    for fid in frame_ids:
        counts.append(process_frame(model, data_root, fid, K, OUT_DIR))

    elapsed = time.time() - t0
    print(f"\nDone: {len(frame_ids)} frames, {sum(counts)} total detections, {elapsed:.1f}s")
    print(f"Outputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
