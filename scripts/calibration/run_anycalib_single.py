import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anycalib import AnyCalib


def load_gt_intrinsics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cam_k = np.array(data["cam_K"], dtype=np.float64).reshape(3, 3)
    gt = {
        "fx": float(cam_k[0, 0]),
        "fy": float(cam_k[1, 1]),
        "cx": float(cam_k[0, 2]),
        "cy": float(cam_k[1, 2]),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
        "distortion_model": data.get("distortion_model", "unknown"),
        "cam_D": data.get("cam_D", []),
    }
    return gt


def load_gt_extrinsics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rot = np.array(data["rotation"], dtype=np.float64).tolist()
    trans = np.array(data["translation"], dtype=np.float64).reshape(-1).tolist()
    return {"rotation": rot, "translation": trans}


def to_numpy_intrinsics(raw_intrinsics):
    if isinstance(raw_intrinsics, list):
        raw_intrinsics = raw_intrinsics[0]
    if torch.is_tensor(raw_intrinsics):
        raw_intrinsics = raw_intrinsics.detach().cpu().numpy()
    return np.array(raw_intrinsics, dtype=np.float64).reshape(-1)


def run_inference(image_path: Path, model_id: str, cam_id: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = torch.tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0

    model = AnyCalib(model_id=model_id).to(dev)
    output = model.predict(tensor, cam_id=cam_id)

    intrinsics = to_numpy_intrinsics(output["intrinsics"])

    fov_field = output["fov_field"]
    if torch.is_tensor(fov_field):
        fov_field = fov_field.detach().cpu().numpy()

    rays = output["rays"]
    if torch.is_tensor(rays):
        rays = rays.detach().cpu().numpy()

    pred_size = output["pred_size"]
    if isinstance(pred_size, torch.Size):
        pred_size = tuple(pred_size)

    return {
        "device": str(dev),
        "intrinsics": intrinsics,
        "fov_shape": list(np.array(fov_field).shape),
        "rays_shape": list(np.array(rays).shape),
        "pred_size": list(pred_size),
    }


def validity_checks(pred_intrinsics, gt_intrinsics, cam_id):
    checks = []

    checks.append({
        "name": "intrinsics_finite",
        "passed": bool(np.isfinite(pred_intrinsics).all()),
        "value": pred_intrinsics.tolist(),
    })

    if pred_intrinsics.size >= 1:
        checks.append({
            "name": "focal_positive",
            "passed": bool(pred_intrinsics[0] > 0),
            "value": float(pred_intrinsics[0]),
        })

    if cam_id.startswith("pinhole") and pred_intrinsics.size >= 4:
        fx, fy, cx, cy = pred_intrinsics[:4]
        checks.append({
            "name": "fx_fy_positive",
            "passed": bool(fx > 0 and fy > 0),
            "value": {"fx": float(fx), "fy": float(fy)},
        })
        width = gt_intrinsics["width"]
        height = gt_intrinsics["height"]
        checks.append({
            "name": "principal_point_in_image",
            "passed": bool(0 <= cx <= width and 0 <= cy <= height),
            "value": {"cx": float(cx), "cy": float(cy), "width": width, "height": height},
        })

        fx_err = abs(float(fx) - gt_intrinsics["fx"]) / max(gt_intrinsics["fx"], 1e-6)
        fy_err = abs(float(fy) - gt_intrinsics["fy"]) / max(gt_intrinsics["fy"], 1e-6)
        cx_err = abs(float(cx) - gt_intrinsics["cx"])
        cy_err = abs(float(cy) - gt_intrinsics["cy"])
        checks.append({
            "name": "gt_alignment_summary",
            "passed": bool(np.isfinite([fx_err, fy_err, cx_err, cy_err]).all()),
            "value": {
                "fx_rel_error": fx_err,
                "fy_rel_error": fy_err,
                "cx_abs_error_px": cx_err,
                "cy_abs_error_px": cy_err,
            },
        })

    return checks


def main():
    parser = argparse.ArgumentParser("Run AnyCalib on one image; validate vs DAIR GT when available")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument("--image", default=None,
                        help="explicit image path (no-GT mode: skips DAIR calib files entirely)")
    parser.add_argument("--model-id", default="anycalib_pinhole")
    parser.add_argument("--cam-id", default="pinhole")
    parser.add_argument("--out-dir", default="outputs/calibration/anycalib_single")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        sample_id = image_path.stem
        gt_intrinsics = None
        gt_extrinsics = None
    else:
        root = Path(args.data_root)
        sample_id = args.sample_id
        image_path = root / "image" / f"{sample_id}.jpg"
        intrinsic_path = root / "calib" / "camera_intrinsic" / f"{sample_id}.json"
        extrinsic_path = root / "calib" / "virtuallidar_to_camera" / f"{sample_id}.json"
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")
        if not extrinsic_path.exists():
            raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_path}")
        gt_intrinsics = load_gt_intrinsics(intrinsic_path)
        gt_extrinsics = load_gt_extrinsics(extrinsic_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pred = run_inference(image_path, args.model_id, args.cam_id)
    if gt_intrinsics is None:
        with Image.open(image_path) as im:
            w, h = im.size
        no_gt = {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "width": w, "height": h}
        checks = [c for c in validity_checks(pred["intrinsics"], no_gt, args.cam_id)
                  if c["name"] != "gt_alignment_summary"]
    else:
        checks = validity_checks(pred["intrinsics"], gt_intrinsics, args.cam_id)
    all_passed = all(item["passed"] for item in checks)

    result = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "model_id": args.model_id,
        "cam_id": args.cam_id,
        "prediction": {
            "device": pred["device"],
            "intrinsics": pred["intrinsics"].tolist(),
            "fov_shape": pred["fov_shape"],
            "rays_shape": pred["rays_shape"],
            "pred_size": pred["pred_size"],
        },
        "ground_truth": None if gt_intrinsics is None else {
            "intrinsics": gt_intrinsics,
            "extrinsics": gt_extrinsics,
        },
        "validity_checks": checks,
        "all_checks_passed": all_passed,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_{args.model_id}_{args.cam_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"all_checks_passed={all_passed}")


if __name__ == "__main__":
    main()
