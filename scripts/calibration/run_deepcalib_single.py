# =============================================================================
# NOTE / KNOWN LIMITATIONS OF DEEPCALIB FOR THIS PROJECT (DAIR-V2X-I roadside)
# -----------------------------------------------------------------------------
# Read this before trusting any output from this script again.
#
# 1) OUT-OF-DISTRIBUTION INPUT (main reason results were unreliable).
#    DeepCalib was trained on wide-angle / distorted photos. Our roadside
#    camera view is much narrower (normal-ish FOV). On sample 000000 the model
#    "maxed out": focal_pred_299 = 501 (the top of its 40-500 range) and
#    distortion = 0. That is a degenerate/saturated guess, not a real reading.
#
# 2) SINGLE FOCAL -> FAKE fx, fy (artifact of OUR conversion, not the camera).
#    DeepCalib predicts only ONE focal number plus one distortion value.
#    A pinhole needs two focals (fx, fy). In map_focal_to_intrinsics() below we
#    synthesize them by scaling that single focal by DIFFERENT constants:
#        fx = focal_299 * (image_width  / 299)
#        fy = focal_299 * (image_height / 299)
#    Because width != height (e.g. 1920 vs 1080), fx and fy come out unequal
#    (e.g. 3217 vs 1809) even though a real square-pixel camera has fx ~= fy.
#    The principal point is also FORCED to image center (cx=W/2, cy=H/2), not
#    measured. So fx/fy/cx/cy from this script are an approximation, not truth.
#
# RECOMMENDATION IF REUSING DEEPCALIB:
#    - Prefer AnyCalib (predicts the full intrinsics directly, fit our images
#      much better), or just use the DAIR ground-truth intrinsics that ship
#      with the dataset.
#    - If you must use DeepCalib, do NOT treat the fx != fy split as real, and
#      remember the focal can saturate on narrow-FOV roadside scenes.
# =============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_intrinsics(path: Path):
    data = read_json(path)
    cam_k = np.array(data["cam_K"], dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(cam_k[0, 0]),
        "fy": float(cam_k[1, 1]),
        "cx": float(cam_k[0, 2]),
        "cy": float(cam_k[1, 2]),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
        "distortion_model": data.get("distortion_model", "unknown"),
        "cam_D": data.get("cam_D", []),
    }


def load_gt_extrinsics(path: Path):
    data = read_json(path)
    rot = np.array(data["rotation"], dtype=np.float64).tolist()
    trans = np.array(data["translation"], dtype=np.float64).reshape(-1).tolist()
    return {"rotation": rot, "translation": trans}


def build_deepcalib_model(tf):
    input_shape = (299, 299, 3)
    main_input = tf.keras.layers.Input(shape=input_shape, dtype="float32", name="main_input")

    backbone = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_tensor=main_input,
        input_shape=input_shape,
    )

    phi_features = backbone.output
    phi_flattened = tf.keras.layers.Flatten(name="phi-flattened")(phi_features)
    final_output_focal = tf.keras.layers.Dense(1, activation="sigmoid", name="output_focal")(phi_flattened)
    final_output_distortion = tf.keras.layers.Dense(1, activation="sigmoid", name="output_distortion")(phi_flattened)

    model = tf.keras.Model(inputs=main_input, outputs=[final_output_focal, final_output_distortion])
    return model


def preprocess_image_for_deepcalib(image_path: Path, tf):
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    image = tf.image.resize(image, (299, 299), method="bilinear").numpy()

    # Keep the original pre-processing logic from DeepCalib scripts.
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def map_focal_to_intrinsics(focal_299, image_w, image_h):
    fx = float(focal_299 * (float(image_w) / 299.0))
    fy = float(focal_299 * (float(image_h) / 299.0))
    cx = float(image_w) / 2.0
    cy = float(image_h) / 2.0
    return [fx, fy, cx, cy]


def run_inference(image_path: Path, weights_path: Path):
    import tensorflow as tf

    model = build_deepcalib_model(tf)
    model.load_weights(str(weights_path))

    inp = preprocess_image_for_deepcalib(image_path, tf)
    pred_focal, pred_dist = model.predict(inp, verbose=0)

    focal_start = 40.0
    focal_end = 500.0
    focal_299 = float(pred_focal[0][0] * ((focal_end + 1.0) - focal_start) + focal_start)
    distortion = float(pred_dist[0][0] * 1.2)

    image = Image.open(image_path)
    image_w, image_h = image.size

    intrinsics = map_focal_to_intrinsics(focal_299, image_w, image_h)

    return {
        "runtime": {
            "framework": "tensorflow-macos",
            "tf_version": str(tf.__version__),
        },
        "focal_pred_299": focal_299,
        "distortion_pred": distortion,
        "intrinsics": intrinsics,
        "image_size": [int(image_w), int(image_h)],
    }


def validity_checks(pred_intrinsics, gt_intrinsics):
    fx, fy, cx, cy = pred_intrinsics
    checks = []

    checks.append(
        {
            "name": "intrinsics_finite",
            "passed": bool(np.isfinite(pred_intrinsics).all()),
            "value": [float(v) for v in pred_intrinsics],
        }
    )
    checks.append(
        {
            "name": "fx_fy_positive",
            "passed": bool(fx > 0 and fy > 0),
            "value": {"fx": float(fx), "fy": float(fy)},
        }
    )

    width = gt_intrinsics["width"]
    height = gt_intrinsics["height"]
    checks.append(
        {
            "name": "principal_point_in_image",
            "passed": bool(0 <= cx <= width and 0 <= cy <= height),
            "value": {"cx": float(cx), "cy": float(cy), "width": width, "height": height},
        }
    )

    fx_err = abs(float(fx) - gt_intrinsics["fx"]) / max(gt_intrinsics["fx"], 1e-6)
    fy_err = abs(float(fy) - gt_intrinsics["fy"]) / max(gt_intrinsics["fy"], 1e-6)
    cx_err = abs(float(cx) - gt_intrinsics["cx"])
    cy_err = abs(float(cy) - gt_intrinsics["cy"])
    checks.append(
        {
            "name": "gt_alignment_summary",
            "passed": bool(np.isfinite([fx_err, fy_err, cx_err, cy_err]).all()),
            "value": {
                "fx_rel_error": float(fx_err),
                "fy_rel_error": float(fy_err),
                "cx_abs_error_px": float(cx_err),
                "cy_abs_error_px": float(cy_err),
            },
        }
    )

    return checks


def main():
    parser = argparse.ArgumentParser("Run DeepCalib Single-Net regression on one DAIR frame")
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--data-root", default="data/dair-v2x-i")
    parser.add_argument(
        "--weights",
        default="external/DeepCalib/weights/Regression/Single_net/weights_10_0.02.h5",
    )
    parser.add_argument("--out-dir", default="outputs/calibration/deepcalib_single")
    args = parser.parse_args()

    root = Path(args.data_root)
    image_path = root / "image" / f"{args.sample_id}.jpg"
    intrinsic_path = root / "calib" / "camera_intrinsic" / f"{args.sample_id}.json"
    extrinsic_path = root / "calib" / "virtuallidar_to_camera" / f"{args.sample_id}.json"
    weights_path = Path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")
    if not extrinsic_path.exists():
        raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DeepCalib weights not found: {weights_path}")

    gt_intrinsics = load_gt_intrinsics(intrinsic_path)
    gt_extrinsics = load_gt_extrinsics(extrinsic_path)

    pred = run_inference(image_path, weights_path)
    pred_intrinsics = np.array(pred["intrinsics"], dtype=np.float64)
    checks = validity_checks(pred_intrinsics, gt_intrinsics)
    all_passed = all(item["passed"] for item in checks)

    result = {
        "sample_id": args.sample_id,
        "image_path": str(image_path),
        "model_id": "deepcalib_single_net_regression",
        "cam_id": "pinhole",
        "prediction": {
            "runtime": pred["runtime"],
            "weights_path": str(weights_path),
            "focal_pred_299": pred["focal_pred_299"],
            "distortion_pred": pred["distortion_pred"],
            "intrinsics": [float(v) for v in pred_intrinsics.tolist()],
            "mapping_note": "DeepCalib predicts one focal + distortion. Converted to pinhole [fx,fy,cx,cy] via focal scaling from 299 to original image width/height and principal point at image center.",
        },
        "ground_truth": {
            "intrinsics": gt_intrinsics,
            "extrinsics": gt_extrinsics,
        },
        "validity_checks": checks,
        "all_checks_passed": bool(all_passed),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}_deepcalib_pinhole.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"all_checks_passed={all_passed}")
    print(json.dumps({
        "focal_pred_299": pred["focal_pred_299"],
        "distortion_pred": pred["distortion_pred"],
        "intrinsics": [float(v) for v in pred_intrinsics.tolist()],
    }, indent=2))


if __name__ == "__main__":
    main()
