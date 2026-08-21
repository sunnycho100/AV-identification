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

# BEVHeight was trained on DAIR, whose virtuallidar frame puts the road surface
# at z = -1.73, not 0 (GT car bottoms median -1.73 over 300 frames). The height
# frustum spans d_bound = [-2, 0], so the model can only ever place a point at
# ego z in (-2, 0]. Our own extrinsics put the road at z = 0, which left every
# real car pixel (z 0..+2) outside the representable range: the model clamped
# them to (-2, 0] and the exact-geometry lift pushed each box ~6 m down-range.
# Raising the ego origin by 1.73 m reproduces the training convention. Only z
# moves, so x/y (and therefore trajectories) are unchanged by this.
DAIR_GROUND_Z = -1.73


def to_dair_ground(lidar2cam):
    """Raise the ego origin so the road sits at DAIR_GROUND_Z instead of z=0."""
    R, t = lidar2cam[:3, :3], lidar2cam[:3, 3]
    out = lidar2cam.copy()
    out[:3, 3] = t + R @ np.array([0.0, 0.0, -DAIR_GROUND_Z])
    return out


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
    ap.add_argument("--no-ground-shift", action="store_true",
                    help="feed the extrinsic as-is; use for extrinsics that "
                         "already follow the DAIR convention (e.g. DAIR's own)")
    args = ap.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg")) + \
        sorted(Path(args.frames_dir).glob("*.png"))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"No frames in {args.frames_dir}")

    K = load_K_from_anycalib(args.anycalib_json)
    lidar2cam = load_extrinsic_json(args.extrinsic_json)
    if not args.no_ground_shift:
        lidar2cam = to_dair_ground(lidar2cam)
        print(f"ego origin raised {-DAIR_GROUND_Z} m: road now at z={DAIR_GROUND_Z} "
              f"(DAIR training convention)")

    model = build_model()
    info = load_checkpoint(model, CKPT_PATH)
    print(f"checkpoint loaded: {info['matched']} keys matched, "
          f"{len(info['missing'])} missing")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the per-frame files stay a bare list of detections (run_ab3dmot reads them
    # that way), so provenance goes in a sidecar
    (out_dir / "calibration_used.json").write_text(json.dumps({
        "anycalib_json": str(args.anycalib_json),
        "extrinsic_json": str(args.extrinsic_json),
        "ground_shift_applied_m": 0.0 if args.no_ground_shift else -DAIR_GROUND_Z,
        "road_plane_z_in_output": 0.0 if args.no_ground_shift else DAIR_GROUND_Z,
        "K": K.tolist(), "lidar2cam": lidar2cam.tolist(),
    }, indent=2))
    for f in frames:
        preds = run_frame(model, f, K, lidar2cam)
        (out_dir / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=2))
        render_annotated(f, preds, K, lidar2cam, out_dir / f"{f.stem}_annotated.jpg")
        cars = [d for d in preds if d["class_name"] == "car"]
        print(f"{f.name}: {len(preds)} detections (score>={SCORE_THRESH}), "
              f"{len(cars)} cars")
    print(f"saved to {out_dir}")


def _selfcheck():
    """The shift must move only z: a road point lands on the DAIR ground plane,
    its x/y are untouched, and the camera keeps its real height above the road."""
    from scripts.calibration.vp_extrinsic_from_frame import solve_pose
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    height = 16.31
    R, t, _, _ = solve_pose(K, (156.6, 21.9), height)
    old = np.eye(4)
    old[:3, :3], old[:3, 3] = R, t
    new = to_dair_ground(old)

    road = np.array([37.0, -5.5, 0.0])                  # a point on the road
    cam_from_old = R @ road + t
    # same physical point, re-expressed in the raised frame
    road_new = road + np.array([0.0, 0.0, DAIR_GROUND_Z])
    cam_from_new = new[:3, :3] @ road_new + new[:3, 3]
    assert np.allclose(cam_from_old, cam_from_new, atol=1e-9), (cam_from_old, cam_from_new)
    assert np.allclose(road_new[:2], road[:2]), "x/y must not move"
    assert abs(road_new[2] - DAIR_GROUND_Z) < 1e-9, road_new

    cam_centre_new = -new[:3, :3].T @ new[:3, 3]
    assert abs((cam_centre_new[2] - DAIR_GROUND_Z) - height) < 1e-9, cam_centre_new
    # what the adapter will report: distance to the new z=0 plane, not to the road
    assert abs(cam_centre_new[2] - (height + DAIR_GROUND_Z)) < 1e-9, cam_centre_new
    assert np.allclose(to_dair_ground(old)[:3, :3], R), "rotation must not change"
    print(f"selfcheck ok (road -> z={DAIR_GROUND_Z}, x/y fixed, camera still "
          f"{height} m above the road, reference_height becomes "
          f"{height + DAIR_GROUND_Z:.2f} m)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
