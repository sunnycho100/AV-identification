"""Write one clip's frames as a MonoUNI (Rope3D layout) dataset root.

MonoUNI reads per image: image_2/<id>.jpg, calib/<id>.txt (P2) and
denorm/<id>.txt (ground plane a b c d in camera frame, unit normal pointing up,
so b < 0; only a b c are used, for the pitch). Our K comes from AnyCalib and the
plane from the site extrinsic (road at z = 0), so every frame gets the same two
files. Label folders are created empty because its dataset class asserts they exist.

Variants:
  A  frames as they are (our focal ~1490 px; MonoUNI was trained on 2100-2800).
  B  virtual zoom: scale by --zoom so the focal lands in the trained range, then
     crop a 1920x1080 window on the far road. Same camera rotation, so 3D boxes
     come out in the same camera frame; only K changes (K' = s K, shifted by the crop).

    python scripts/object_detection/monouni_prep.py --clip AV_T_EW_3 --variant B \
        --out /home/data/scho242/monouni/AV_T_EW_3_B
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W, H = 1920, 1080
EXTRINSIC = "metric_extrinsic_h151_dpm031.json"


def load_cal(clip):
    cal = ROOT / "outputs/calibration/camera-data" / clip
    fx, fy, cx, cy = json.loads(next(cal.glob("*_anycalib_pinhole_pinhole.json")).read_text())["prediction"]["intrinsics"]
    ext = json.loads((cal / EXTRINSIC).read_text())
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, np.array(ext["rotation"]), np.array(ext["translation"])


def denorm(R, t):
    """Road plane z = 0 in camera frame: n . p + d = 0, n = road up in camera axes."""
    n = R @ np.array([0, 0, 1.0])
    return np.append(n, -n @ t)


def zoom_window(K, R, t, s, x_far=(40.0, 400.0), y_mid=-20.0):
    """Crop origin (x0, y0) in the s-scaled image: centred on the road beyond 40 m."""
    pts = [K @ (R @ np.array([x, y_mid, 0.0]) + t) for x in np.linspace(*x_far, 20)]
    uv = np.array([p[:2] / p[2] for p in pts if p[2] > 0]) * s
    c = (uv.min(0) + uv.max(0)) / 2
    x0 = int(np.clip(c[0] - W / 2, 0, s * W - W))
    y0 = int(np.clip(c[1] - H / 2, 0, s * H - H))
    return x0, y0


def prep(clip, variant, out, zoom):
    K, R, t = load_cal(clip)
    out = Path(out)
    for d in ("image_2", "calib", "denorm", "ImageSets", "label_2_4cls_for_train",
              "label_2_4cls_filter_with_roi_for_eval"):
        (out / d).mkdir(parents=True, exist_ok=True)
    frames = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))
    Kv, crop = K.copy(), (0, 0)
    if variant == "B":
        crop = zoom_window(K, R, t, zoom)
        Kv = np.diag([zoom, zoom, 1.0]) @ K
        Kv[0, 2] -= crop[0]
        Kv[1, 2] -= crop[1]
    P2 = " ".join(f"{v:.6f}" for v in np.hstack([Kv, np.zeros((3, 1))]).ravel())
    dn = " ".join(f"{v:.9f}" for v in denorm(R, t))
    ids = []
    for f in frames:
        i = f.stem
        ids.append(i)
        dst = out / "image_2" / f"{i}.jpg"
        if variant == "A":
            if not dst.exists():
                os.symlink(f.resolve(), dst)
        else:
            im = cv2.resize(cv2.imread(str(f)), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst), im[crop[1]:crop[1] + H, crop[0]:crop[0] + W], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "calib" / f"{i}.txt").write_text(f"P2: {P2}\n")
        (out / "denorm" / f"{i}.txt").write_text(dn + "\n")
        for d in ("label_2_4cls_for_train", "label_2_4cls_filter_with_roi_for_eval"):
            (out / d / f"{i}.txt").touch()
    for s in ("train", "val"):
        (out / "ImageSets" / f"{s}.txt").write_text("\n".join(ids) + "\n")
    meta = {"clip": clip, "variant": variant, "zoom": zoom if variant == "B" else 1.0, "crop_xy": crop,
            "K_used": Kv.tolist(), "K_orig": K.tolist(), "rotation": R.tolist(), "translation": t.tolist(),
            "denorm": dn, "extrinsic": EXTRINSIC, "frames": len(ids)}
    (out / "prep.json").write_text(json.dumps(meta, indent=2))
    pitch = np.degrees(np.arctan(float(dn.split()[2]) / float(dn.split()[1])))
    print(f"{clip} {variant}: {len(ids)} frames, f {Kv[0, 0]:.0f} px, crop {crop}, "
          f"MonoUNI pitch {pitch:.2f} deg, plane d {float(dn.split()[3]):.2f} m -> {out}")


def _selfcheck():
    """Denorm matches MonoUNI's own parser (pitch = atan(c / b)), b < 0, and d is the
    camera height; a road point projects to the same pixel through K and through
    the zoomed K' once the crop and scale are undone."""
    K, R, t = load_cal("AV_T_EW_3")
    a, b, c, d = denorm(R, t)
    assert b < 0 and abs(np.linalg.norm([a, b, c]) - 1) < 1e-9
    assert abs(d - 15.1) < 0.05, d
    pitch = np.degrees(np.arctan(c / b))
    cam_z = R[2]                                           # optical axis in road frame
    assert abs(pitch - np.degrees(np.arcsin(-cam_z[2]))) < 0.05, (pitch, cam_z)
    s, (x0, y0) = 1.48, zoom_window(K, R, t, 1.48)
    Kv = np.diag([s, s, 1.0]) @ K
    Kv[0, 2] -= x0
    Kv[1, 2] -= y0
    p = R @ np.array([120.0, -15.0, 0.0]) + t
    u, uv = (K @ p)[:2] / p[2], (Kv @ p)[:2] / p[2]
    assert np.allclose((uv + [x0, y0]) / s, u, atol=1e-6)
    assert 0 <= uv[0] < W and 0 <= uv[1] < H, uv
    print(f"selfcheck ok (pitch {pitch:.2f} deg, height {d:.2f} m, 120 m road point inside the zoomed crop)")


def main():
    ap = argparse.ArgumentParser("Frames to a MonoUNI dataset root")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--variant", choices=("A", "B"), default="A")
    ap.add_argument("--zoom", type=float, default=1.48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prep(a.clip, a.variant, a.out, a.zoom)


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
