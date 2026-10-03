"""Run BEVHeight (CPU) on arbitrary image frames with file-based calibration.

The non-DAIR runner: takes a directory of frames, an AnyCalib intrinsics JSON,
and a generic extrinsic JSON ({"rotation": 3x3, "translation": [3]},
ground-frame -> camera). Reuses the model/config/render code from
run_bevheight_single.py.

The pipeline default is the 140.8 m DAIR checkpoint at score >= 0.45. Its BEV
grid reaches the far end of this footage, where the 102.4 m one runs out of
range and starts placing boxes on empty pavement; at the same threshold it
finds about a third more cars per frame (7.0 against 5.1 on the five Todd Drive
clips) and a sixth as many pedestrian and bicycle false positives on the
freeway. --ckpt and --config together still reach the 102.4 m checkpoint
(bev_height_lss_r50_864_1536_128x128_102.py) and Rope3D.

Smoke test (Camera data clip AV_T_WE_1, mock extrinsic):
    .venv/bin/python scripts/object_detection/run_bevheight_generic.py \
        --frames-dir data/camera-data/AV_T_WE_1/frames \
        --anycalib-json outputs/calibration/camera-data/AV_T_WE_1/150_anycalib_pinhole_pinhole.json \
        --extrinsic-json outputs/calibration/camera-data/AV_T_WE_1/mock_extrinsic.json \
        --out-dir outputs/object_detection/camera-data/AV_T_WE_1
"""
import argparse
import importlib.util
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
from models.bev_height import BEVHeight
from scripts.adapter.calib_to_bevheight_input import (
    build_mats_dict, load_K_from_anycalib, load_extrinsic_json)
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
from scripts.object_detection.run_bevheight_single import (
    SCORE_THRESH, filter_and_pack, load_checkpoint)

DEFAULT_CKPT = ROOT / "checkpoints/BEVHeight_R50_128_140.8_75.22_49_epochs.ckpt"
DEFAULT_CONFIG = ROOT / "experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_140.py"


def load_exp(path):
    """The experiment module, so the model is built to match the checkpoint.

    Rope3D's config differs from DAIR's in d_bound ([-1.5, 3.0, 180] against
    [-2.0, 0.0, 90]), which changes the height head to 180 channels. Its
    checkpoint only loads against its own config.
    """
    spec = importlib.util.spec_from_file_location("bev_exp_cfg", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    # ponytail: the lab server's torch 1.9 cusolver cannot create a handle
    # (CUSOLVER_STATUS_INTERNAL_ERROR on a bare 4x4 inverse). The model only
    # inverts small calibration matrices, so route those through the CPU. Drop
    # this shim once the server env is rebuilt on a torch with a working cusolver.
    _inverse = torch.Tensor.inverse

    def _cpu_inverse(t):
        return _inverse(t.cpu()).to(t.device) if t.is_cuda else _inverse(t)

    torch.Tensor.inverse = _cpu_inverse
    torch.inverse = _cpu_inverse


def run_frame(model, image_path, K, lidar2cam, exp, score_thresh=SCORE_THRESH):
    img_tensor, mats_dict, img_meta = build_mats_dict(
        str(image_path), K, lidar2cam, exp.final_dim, exp.img_conf)
    img_tensor = img_tensor.to(DEVICE)
    mats_dict = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in mats_dict.items()}
    img_meta["box_type_3d"] = LiDARInstance3DBoxes
    with torch.no_grad():
        preds = model(img_tensor, mats_dict)
        results = model.get_bboxes(preds, [img_meta])
    boxes = results[0][0].tensor.cpu().numpy()
    scores = results[0][1].cpu().numpy()
    labels = results[0][2].cpu().numpy()
    preds = filter_and_pack(boxes, scores, labels, score_thresh)
    for det in preds:  # name classes from the config actually in use
        det["class_name"] = exp.CLASSES[det["class_id"]]
    return preds


def render_annotated(image_path, preds, K, lidar2cam, out_path):
    img = cv2.imread(str(image_path))
    k34 = np.zeros((3, 4), dtype=np.float64)
    k34[:3, :3] = K
    for det in preds:
        l, w, h = det["l"], det["w"], det["h"]
        center = [det["x"], det["y"], det["z"] + h / 2.0]
        # get_lidar_3d_8points takes [l, w, h]: length along the heading (x before
        # the yaw rotation), width across it.
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
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="checkpoint to load (default: the DAIR 140.8 m one)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="experiment config whose model definition matches --ckpt "
                         "(e.g. experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py "
                         "for the 102.4 m checkpoint)")
    ap.add_argument("--score-thresh", type=float, default=SCORE_THRESH,
                    help=f"keep detections scoring at least this (default {SCORE_THRESH}, "
                         "upstream's export threshold)")
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

    exp = load_exp(args.config)
    model = BEVHeight(exp.backbone_conf, exp.head_conf)
    model.eval()
    ckpt_path = Path(args.ckpt)
    info = load_checkpoint(model, ckpt_path)
    model.to(DEVICE)
    print(f"config {Path(args.config).name}: d_bound {exp.backbone_conf['d_bound']}")
    print(f"checkpoint loaded: {info['matched']} keys matched, "
          f"{len(info['missing'])} missing, {len(info['unexpected'])} unexpected; "
          f"device {DEVICE}")
    for tag in ("missing", "unexpected"):
        if info[tag]:
            print(f"  {tag}: {info[tag]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the per-frame files stay a bare list of detections (run_ab3dmot reads them
    # that way), so provenance goes in a sidecar
    (out_dir / "calibration_used.json").write_text(json.dumps({
        "checkpoint": ckpt_path.name,
        "config": str(args.config),
        "anycalib_json": str(args.anycalib_json),
        "extrinsic_json": str(args.extrinsic_json),
        "score_thresh": args.score_thresh,
        "ground_shift_applied_m": 0.0 if args.no_ground_shift else -DAIR_GROUND_Z,
        "road_plane_z_in_output": 0.0 if args.no_ground_shift else DAIR_GROUND_Z,
        "K": K.tolist(), "lidar2cam": lidar2cam.tolist(),
    }, indent=2))
    for f in frames:
        preds = run_frame(model, f, K, lidar2cam, exp, args.score_thresh)
        (out_dir / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=2))
        render_annotated(f, preds, K, lidar2cam, out_dir / f"{f.stem}_annotated.jpg")
        cars = [d for d in preds if d["class_name"] == "car"]
        print(f"{f.name}: {len(preds)} detections (score>={args.score_thresh}), "
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
