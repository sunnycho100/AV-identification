"""Virtual camera: centre-crop and upscale frames so fx matches DAIR's.

Why. BEVHeight was trained at fx 2183 (DAIR); our cameras run 1493 to 1698 px at
1920x1080, so a car at 50 m is 1.3 to 1.5x smaller than anything the yaw head
saw. Cropping the centre 1920/c x 1080/c (c = target_fx / fx) and resizing back
to 1920x1080 is a pure intrinsic change: the camera centre and rotation are
untouched, so the extrinsic carries over verbatim and only K moves.

    fx' = fx * sx,  cx' = (cx - x0) * sx      sx = 1920 / crop_w
    fy' = fy * sy,  cy' = (cy - y0) * sy      sy = 1080 / crop_h

note: one centre crop, not the 2 to 3 overlapping tiles the source note
proposes. Vehicles outside the crop are lost and edge recall drops; that is the
accepted cost for a first signal. Add tiling + NMS merge only if the folded
error actually moves.

    python scripts/orientation/virtual_camera.py \
        --frames-dir data/camera-data/AV_T_EW_3/frames_all \
        --anycalib-json outputs/calibration/camera-data/AV_T_EW_3/150_anycalib_pinhole_pinhole.json \
        --out-frames-dir data/camera-data/AV_T_EW_3/frames_vcam \
        --out-anycalib-json outputs/calibration/camera-data/AV_T_EW_3/150_anycalib_vcam.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def crop_box(width, height, fx, target_fx):
    """Centre crop (x0, y0, w, h) whose upscale to (width, height) gives fx~=target_fx."""
    c = target_fx / fx
    if c <= 1.0:
        raise SystemExit(f"fx {fx:.1f} already >= target {target_fx}: nothing to do")
    cw, ch = round(width / c), round(height / c)
    return (width - cw) // 2, (height - ch) // 2, cw, ch


def new_K(K, box, width, height):
    """K after cropping to box and resizing that crop back to (width, height)."""
    x0, y0, cw, ch = box
    sx, sy = width / cw, height / ch
    out = K.copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (K[0, 2] - x0) * sx
    out[1, 2] = (K[1, 2] - y0) * sy
    return out


def warp(img, box, width, height):
    x0, y0, cw, ch = box
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (width, height),
                      interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser("Virtual camera: crop and rescale to a target focal")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--target-fx", type=float, default=2183.0)
    ap.add_argument("--out-frames-dir", required=True)
    ap.add_argument("--out-anycalib-json", required=True)
    args = ap.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg")) + \
        sorted(Path(args.frames_dir).glob("*.png"))
    if not frames:
        sys.exit(f"No frames in {args.frames_dir}")

    calib = json.loads(Path(args.anycalib_json).read_text())
    fx, fy, cx, cy = calib["prediction"]["intrinsics"][:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    h, w = cv2.imread(str(frames[0])).shape[:2]
    box = crop_box(w, h, fx, args.target_fx)
    K2 = new_K(K, box, w, h)

    out_dir = Path(args.out_frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        cv2.imwrite(str(out_dir / f.name), warp(cv2.imread(str(f)), box, w, h),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    calib["prediction"]["intrinsics"] = [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]] + \
        calib["prediction"]["intrinsics"][4:]
    calib["virtual_camera"] = {"source_anycalib_json": str(args.anycalib_json),
                               "target_fx": args.target_fx,
                               "crop_xywh": list(box), "output_wh": [w, h],
                               "source_intrinsics": [fx, fy, cx, cy]}
    out_json = Path(args.out_anycalib_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(calib, indent=2))

    print(f"{len(frames)} frames -> {out_dir}  crop {box} -> {w}x{h}")
    print(f"fx {fx:.1f} -> {K2[0,0]:.1f}, fy {fy:.1f} -> {K2[1,1]:.1f}, "
          f"cx {cx:.1f} -> {K2[0,2]:.1f}, cy {cy:.1f} -> {K2[1,2]:.1f}")
    print(f"wrote {out_json}")


def _selfcheck():
    """A 3D point projected through the original K must land on the same physical
    pixel of the cropped-and-resized image when projected through the new K."""
    W, H = 1920, 1080
    K = np.array([[1493.33, 0, 960.51], [0, 1481.09, 541.46], [0, 0, 1]])
    box = crop_box(W, H, K[0, 0], 2183.0)
    K2 = new_K(K, box, W, H)
    x0, y0, cw, ch = box
    sx, sy = W / cw, H / ch

    assert abs(K2[0, 0] - 2183.0) < 5.0, K2[0, 0]          # focal hit the target
    for p in ([2.0, 1.0, 40.0], [-6.0, -2.5, 55.0], [8.0, 0.5, 90.0],
              [0.0, 0.0, 30.0], [-1.0, 3.0, 70.0]):
        uv = K @ np.array(p)
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        assert x0 <= u < x0 + cw and y0 <= v < y0 + ch, f"{p} outside the crop"
        # where that pixel moves under crop-then-resize
        u_exp, v_exp = (u - x0) * sx, (v - y0) * sy
        uv2 = K2 @ np.array(p)
        u2, v2 = uv2[0] / uv2[2], uv2[1] / uv2[2]
        assert abs(u2 - u_exp) < 1e-6 and abs(v2 - v_exp) < 1e-6, (u2, u_exp, v2, v_exp)

    # and the same thing on real pixels: warp a synthetic image, find the marker
    img = np.zeros((H, W, 3), np.uint8)
    uv = K @ np.array([2.0, 1.0, 40.0])
    u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
    img[v, u] = 255
    out = warp(img, box, W, H)
    ys, xs = np.nonzero(out[:, :, 0] > 40)
    uv2 = K2 @ np.array([2.0, 1.0, 40.0])
    u2, v2 = uv2[0] / uv2[2], uv2[1] / uv2[2]
    assert abs(xs.mean() - u2) < 2.0 and abs(ys.mean() - v2) < 2.0, \
        (xs.mean(), u2, ys.mean(), v2)
    print(f"selfcheck ok (crop {box}, fx {K[0,0]:.1f} -> {K2[0,0]:.1f}, "
          f"projection round-trips to under 1e-6 px analytically and 2 px on pixels)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
