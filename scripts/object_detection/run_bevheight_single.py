"""Run BEVHeight 3D detection on a single DAIR frame (CPU), twice with GT vs AnyCalib intrinsics."""
import importlib.util
import json
import shutil
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
from scripts.adapter.calib_to_bevheight_input import build_mats_dict, load_K_from_anycalib
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image

# --- exp configs (import module with hyphenated path via importlib) ---
_exp_path = ROOT / "experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py"
_spec = importlib.util.spec_from_file_location("bev_exp", _exp_path)
bev_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bev_exp)

final_dim = bev_exp.final_dim
img_conf = bev_exp.img_conf
backbone_conf = bev_exp.backbone_conf
head_conf = bev_exp.head_conf
CLASSES = bev_exp.CLASSES

SCORE_THRESH = 0.3
FRAME_ID = "000000"
IMAGE_PATH = ROOT / "data/dair-v2x-i/image/000000.jpg"
GT_EXTRINSIC_PATH = ROOT / "data/dair-v2x-i/calib/virtuallidar_to_camera/000000.json"
GT_INTRINSIC_PATH = ROOT / "data/dair-v2x-i/calib/camera_intrinsic/000000.json"
ANYCALIB_PATH = ROOT / "outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json"
CKPT_PATH = ROOT / "checkpoints/BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt"
OUT_ROOT = ROOT / "outputs/object_detection"
PERSONAL_ROOT = ROOT / "personal-documents/object-detection"


def load_gt_intrinsic(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.array(data["cam_K"], dtype=np.float64).reshape(3, 3)


def load_gt_extrinsic(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    rot = np.array(data["rotation"], dtype=np.float64)
    t = np.array(data["translation"], dtype=np.float64).reshape(3)
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = rot
    ext[:3, 3] = t
    return ext


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    stripped = {}
    for k, v in state.items():
        key = k[6:] if k.startswith("model.") else k
        stripped[key] = v
    incompatible = model.load_state_dict(stripped, strict=False)
    return {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "matched": len(stripped) - len(incompatible.unexpected_keys),
    }


def build_model() -> BEVHeight:
    model = BEVHeight(backbone_conf, head_conf)
    model.eval()
    return model


def run_inference(model: BEVHeight, K: np.ndarray, lidar2cam: np.ndarray) -> tuple:
    img_tensor, mats_dict, img_meta = build_mats_dict(
        str(IMAGE_PATH), K, lidar2cam, final_dim, img_conf,
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


def filter_and_pack(boxes, scores, labels):
    kept = []
    for i in range(len(scores)):
        if scores[i] < SCORE_THRESH:
            continue
        box = boxes[i]
        x, y, z = box[0], box[1], box[2]
        # Model dim order is [width, length, height]: verified empirically (a car
        # detection has box[4]~4.3 = length, box[3]~1.7 = width). Label correctly
        # so "l" always holds the longer along-heading dimension.
        width, length, height = box[3], box[4], box[5]
        yaw = box[6]
        kept.append({
            "class_id": int(labels[i]),
            "class_name": CLASSES[int(labels[i])],
            "score": float(scores[i]),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "l": float(length),
            "w": float(width),
            "h": float(height),
            "yaw": float(yaw),
        })
    return kept


def render_annotated(preds: list, K: np.ndarray, lidar2cam: np.ndarray, out_path: Path):
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    h_img, w_img = img.shape[:2]

    k34 = np.zeros((3, 4), dtype=np.float64)
    k34[:3, :3] = K

    for det in preds:
        l, w, h = det["l"], det["w"], det["h"]
        # get_lidar_3d_8points expects [length, width, height] (length along the
        # heading axis) and internally lowers the center by h/2, so pass z + h/2
        # (model z is the box bottom) to seat the box correctly in height.
        center = [det["x"], det["y"], det["z"] + h / 2.0]
        yaw = det["yaw"]
        corners = get_lidar_3d_8points([l, w, h], yaw, center)
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
        cv2.putText(img, label, (max(0, u_min), max(12, v_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def run_tag(tag: str, K: np.ndarray, lidar2cam: np.ndarray, model: BEVHeight):
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes, scores, labels = run_inference(model, K, lidar2cam)
    preds = filter_and_pack(boxes, scores, labels)

    pred_json = out_dir / f"{FRAME_ID}_pred.json"
    pred_json.write_text(json.dumps(preds, indent=2))

    annotated = out_dir / f"{FRAME_ID}_annotated.jpg"
    render_annotated(preds, K, lidar2cam, annotated)

    print(f"\n=== {tag} ===")
    print(f"  detections (score>={SCORE_THRESH}): {len(preds)}")
    for det in preds[:3]:
        print(f"    {det['class_name']} score={det['score']:.3f} "
              f"pos=({det['x']:.2f},{det['y']:.2f},{det['z']:.2f})")
    print(f"  saved: {pred_json}")
    print(f"  saved: {annotated}")
    return preds


def copy_deliverables():
    PERSONAL_ROOT.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("gt_intrinsics", f"{FRAME_ID}_gt_intrinsics"),
        ("our_intrinsics", f"{FRAME_ID}_our_intrinsics"),
    ]
    for tag, prefix in mapping:
        src_dir = OUT_ROOT / tag
        shutil.copy2(src_dir / f"{FRAME_ID}_pred.json",
                      PERSONAL_ROOT / f"{prefix}_pred.json")
        shutil.copy2(src_dir / f"{FRAME_ID}_annotated.jpg",
                      PERSONAL_ROOT / f"{prefix}_annotated.jpg")


def main():
    lidar2cam = load_gt_extrinsic(GT_EXTRINSIC_PATH)
    K_gt = load_gt_intrinsic(GT_INTRINSIC_PATH)
    K_our = load_K_from_anycalib(str(ANYCALIB_PATH))

    model = build_model()
    load_info = load_checkpoint(model, CKPT_PATH)
    print("=== Checkpoint load ===")
    print(f"  checkpoint: {CKPT_PATH}")
    print(f"  keys in state_dict (after strip): {load_info['matched'] + len(load_info['missing'])}")
    print(f"  matched (loaded): {load_info['matched']}")
    print(f"  missing keys: {len(load_info['missing'])}")
    if load_info["missing"]:
        print(f"    first few: {load_info['missing'][:5]}")
    print(f"  unexpected keys: {len(load_info['unexpected'])}")
    if load_info["unexpected"]:
        print(f"    first few: {load_info['unexpected'][:5]}")

    preds_gt = run_tag("gt_intrinsics", K_gt, lidar2cam, model)
    preds_our = run_tag("our_intrinsics", K_our, lidar2cam, model)

    copy_deliverables()
    print(f"\nCopied deliverables to {PERSONAL_ROOT}")
    print(f"  count gt_intrinsics: {len(preds_gt)}, our_intrinsics: {len(preds_our)}")


if __name__ == "__main__":
    main()
