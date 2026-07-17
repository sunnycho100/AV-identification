"""Adapter: (image + our calibration) -> standardized BEVHeight `mats_dict` input.

This is the clean seam where our own calibration (AnyCalib intrinsics +
road-line refined extrinsics) plugs into the BEVHeight detector, for the generic
case of a dataset that does NOT ship the DAIR info-pkl format.

It reproduces the exact `mats_dict` schema that `dataset/nusc_mv_det_dataset.py`
`collate_fn` emits (verified against that file), but standalone: it depends only
on numpy / cv2 / torch, not on the nuScenes dataset class or the mm* stack, so it
also works for a new dataset.

The small geometry helpers (equation_plane / denorm / sensor2virtual /
reference_height / img_transform) are copied verbatim from
`dataset/nusc_mv_det_dataset.py` so the output is numerically identical.

Run (demo on DAIR frame 000000 using OUR calibration outputs):
    python scripts/adapter/calib_to_bevheight_input.py
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# --- geometry helpers (verbatim from dataset/nusc_mv_det_dataset.py) ---

def equation_plane(points):
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1, b1, c1 = x2 - x1, y2 - y1, z2 - z1
    a2, b2, c2 = x3 - x1, y3 - y1, z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (-a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])


def get_denorm(ego2sensor):
    # ground plane (lidar z=0) expressed in camera coords -> plane equation
    ground = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground = np.concatenate((ground, np.ones((ground.shape[0], 1))), axis=1)
    ground_cam = np.matmul(ego2sensor, ground.T).T
    return -1 * equation_plane(ground_cam)


def get_sensor2virtual(denorm):
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.linalg.norm(target_vector)
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector)
    n_vector = (n_vector / np.linalg.norm(n_vector)).astype(np.float32)
    rot_mat, _ = cv2.Rodrigues(n_vector * sita)
    sensor2virtual = np.eye(4)
    sensor2virtual[:3, :3] = rot_mat.astype(np.float32)
    return sensor2virtual.astype(np.float32)


def get_reference_height(denorm):
    return (np.abs(denorm[3]) / np.linalg.norm(denorm[:3])).astype(np.float32)


def resize_img_and_ida(img, src_wh, final_dim):
    """Uniform resize to final_dim (fH, fW); returns (np img, ida_mat 4x4).

    Matches dataset img_transform with crop=(0,0,fW,fH), no flip/rotate.
    """
    W, H = src_wh
    fH, fW = final_dim
    resize = min(fW / W, fH / H)
    img = img.resize((int(W * resize), int(H * resize)))
    img = img.crop((0, 0, fW, fH))
    ida = torch.eye(4)
    ida[:2, :2] *= resize
    return img, ida


def build_mats_dict(image_path, K, lidar2cam, final_dim, img_conf):
    """Assemble the standardized BEVHeight input from our calibration.

    K: 3x3 intrinsics. lidar2cam: 4x4 extrinsic (ego/lidar -> camera).
    Returns (img_tensor[B,sweeps,cams,3,fH,fW], mats_dict, img_metas).
    """
    lidar2cam = np.asarray(lidar2cam, dtype=np.float64)
    cam2lidar = np.linalg.inv(lidar2cam)              # sensor -> ego

    denorm = get_denorm(lidar2cam)                    # ground plane in cam frame
    sensor2virtual = get_sensor2virtual(denorm)
    ref_height = float(get_reference_height(denorm))

    intrin = np.eye(4)
    intrin[:3, :3] = K

    img = Image.open(image_path).convert("RGB")
    src_wh = img.size
    img, ida = resize_img_and_ida(img, src_wh, final_dim)
    img = (np.array(img, dtype=np.float32) - np.array(img_conf["img_mean"])) / \
        np.array(img_conf["img_std"])
    img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (3,fH,fW)

    def nest(m):  # (4,4) -> (B=1, sweeps=1, cams=1, 4, 4)
        return torch.tensor(np.asarray(m), dtype=torch.float32).reshape(1, 1, 1, 4, 4)

    mats_dict = {
        "sensor2ego_mats": nest(cam2lidar),
        "intrin_mats": nest(intrin),
        "ida_mats": ida.reshape(1, 1, 1, 4, 4).float(),
        "sensor2sensor_mats": torch.eye(4).reshape(1, 1, 1, 4, 4),  # single sweep
        "sensor2virtual_mats": nest(sensor2virtual),
        "reference_heights": torch.tensor([[[ref_height]]], dtype=torch.float32),
        "bda_mat": torch.eye(4).reshape(1, 4, 4),
    }
    img_tensor = img_t.reshape(1, 1, 1, *img_t.shape)  # (B,sweeps,cams,3,fH,fW)
    img_metas = {"denorm": denorm.tolist(), "reference_height": ref_height}
    return img_tensor, mats_dict, img_metas


def load_K_from_anycalib(path):
    pred = json.loads(Path(path).read_text())["prediction"]["intrinsics"]
    fx, fy, cx, cy = pred[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_extrinsic_json(path):
    """Generic pose loader: one JSON with {"rotation": 3x3, "translation": [3]}
    (ground-frame -> camera, same schema as DAIR's virtuallidar_to_camera).
    This is the non-DAIR path for new datasets."""
    data = json.loads(Path(path).read_text())
    ext = np.eye(4)
    ext[:3, :3] = np.asarray(data["rotation"], dtype=np.float64)
    ext[:3, 3] = np.asarray(data["translation"], dtype=np.float64).reshape(3)
    return ext


def load_extrinsic(rotation_json, translation_json):
    """rotation from our road-line result (refined), translation from GT."""
    rot = np.asarray(json.loads(Path(rotation_json).read_text())["refined"]["rotation"], dtype=np.float64)
    t = np.asarray(json.loads(Path(translation_json).read_text())["translation"], dtype=np.float64).reshape(3)
    ext = np.eye(4)
    ext[:3, :3] = rot
    ext[:3, 3] = t
    return ext


def main():
    p = argparse.ArgumentParser("Our calibration -> BEVHeight mats_dict adapter")
    p.add_argument("--image", default="data/dair-v2x-i/image/000000.jpg")
    p.add_argument("--anycalib-json",
                   default="outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json")
    p.add_argument("--roadline-json",
                   default="outputs/calibration/road_line_v3/000000_road_line_v3_result.json")
    p.add_argument("--gt-extrinsic",
                   default="data/dair-v2x-i/calib/virtuallidar_to_camera/000000.json")
    p.add_argument("--out", default="outputs/adapter/000000_bevheight_input.pt")
    args = p.parse_args()

    final_dim = (864, 1536)
    img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                    img_std=[58.395, 57.12, 57.375])

    K = load_K_from_anycalib(args.anycalib_json)
    lidar2cam = load_extrinsic(args.roadline_json, args.gt_extrinsic)

    img_t, mats_dict, img_metas = build_mats_dict(
        args.image, K, lidar2cam, final_dim, img_conf)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"img": img_t, "mats_dict": mats_dict, "img_metas": img_metas}, out)

    print("=== Standardized BEVHeight input (from OUR calibration) ===")
    print(f"image tensor : {tuple(img_t.shape)}  (B, sweeps, cams, C, H, W)")
    for k, v in mats_dict.items():
        print(f"  {k:22s}: {tuple(v.shape)}")
    print("\nintrinsics K (top-left 3x3 of intrin_mats):")
    print(np.round(mats_dict["intrin_mats"][0, 0, 0, :3, :3].numpy(), 2))
    print(f"\nreference_height (camera height above ground, m): "
          f"{mats_dict['reference_heights'].item():.3f}")
    print(f"denorm (ground plane in cam frame): "
          f"{np.round(img_metas['denorm'], 4)}")
    print(f"\nSaved standardized input -> {out}")


if __name__ == "__main__":
    main()
