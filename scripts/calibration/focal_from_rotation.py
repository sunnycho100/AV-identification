"""Focal length of a re-pointed PTZ camera from a reference view of the same camera.

The camera stays on the same pole, so two views differ only by a rotation and a
zoom: x2 ~ K2 R K1^-1 x1 for every static point, not just a plane. Given K1 (the
reference view, already calibrated) the homography H between the two median
backgrounds fixes K2: R = K2^-1 H K1 must be a rotation, so f2 is the focal length
that makes it orthonormal. Principal points stay at the AnyCalib values (both
within 2 px of the image centre). Moving vehicles are removed by the per-clip
median background; the 511 logo and the caption are masked out.

    .venv/bin/python scripts/calibration/focal_from_rotation.py \
        --ref AV_T_WE_1 --clip AV_T_EW_6 --out outputs/calibration/camera-data/AV_T_EW_6/focal_from_rotation.json
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def background(clip, step=10):
    frames = sorted(glob.glob(str(ROOT / f"data/camera-data/{clip}/frames_all/*.jpg")))[::step]
    return np.median(np.stack([cv2.imread(f) for f in frames]), 0).astype(np.uint8)


def intrinsics(clip):
    k = json.loads(next((ROOT / f"outputs/calibration/camera-data/{clip}").glob("*_anycalib_pinhole_pinhole.json")).read_text())
    fx, fy, cx, cy = k["prediction"]["intrinsics"][:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def static_mask(img):
    m = np.full(img.shape[:2], 255, np.uint8)
    h, w = m.shape
    m[int(0.85 * h):, :int(0.40 * w)] = 0        # "US 12/18 @ TODD DR" caption
    m[int(0.75 * h):, int(0.85 * w):] = 0        # 511 logo
    m[:int(0.08 * h), int(0.20 * w):int(0.25 * w)] = 0   # "SE" tag
    return m


def orthonormality(H, K1, f, cx, cy):
    K2 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R = np.linalg.inv(K2) @ H @ K1
    R = R / np.cbrt(np.linalg.det(R))
    return float(np.linalg.norm(R.T @ R - np.eye(3))), R


def solve(H, K1, cx, cy, lo=1000, hi=2600):
    fs = np.arange(lo, hi, 0.5)
    err = [orthonormality(H, K1, f, cx, cy)[0] for f in fs]
    i = int(np.argmin(err))
    return float(fs[i]), float(err[i]), orthonormality(H, K1, fs[i], cx, cy)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="calibrated reference clip (same camera)")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-extrinsic", default=None,
                    help="also write this clip's pose as the reference pose followed by the measured rotation")
    ap.add_argument("--height", type=float, default=None, help="camera height for the composed pose")
    ap.add_argument("--extrinsic-name", default="metric_extrinsic_fromref.json")
    a = ap.parse_args()
    b1, b2 = background(a.ref), background(a.clip)
    g1, g2 = (cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in (b1, b2))
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(g1, static_mask(b1))
    k2, d2 = sift.detectAndCompute(g2, static_mask(b2))
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p1[inl][None], H)[0]
    reproj = float(np.median(np.linalg.norm(proj - p2[inl], axis=1)))
    K1 = intrinsics(a.ref)
    K2a = intrinsics(a.clip)
    f, err, R = solve(H, K1, K2a[0, 2], K2a[1, 2])
    # spread: refit on random halves of the inliers
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(inl)
    fs = []
    for _ in range(50):
        s = rng.choice(idx, size=len(idx) // 2, replace=False)
        Hs, _ = cv2.findHomography(p1[s], p2[s], 0)
        fs.append(solve(Hs, K1, K2a[0, 2], K2a[1, 2])[0])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    out = {"ref": a.ref, "clip": a.clip, "f_ref": float(K1[0, 0]), "f": f, "f_spread_halves": [float(np.percentile(fs, 5)), float(np.percentile(fs, 95))],
           "zoom_ratio": f / float(K1[0, 0]), "orthonormality_err": err, "matches": int(len(good)), "inliers": int(inl.sum()),
           "median_reproj_px": reproj, "rotation_deg": ang, "anycalib_f": float(K2a[0, 0]), "H": H.tolist()}
    if a.ref_extrinsic:
        # zoom unchanged: both views use the reference K, and the rotation is the
        # nearest rotation to K^-1 H K
        U, _, Vt = np.linalg.svd(np.linalg.inv(K1) @ H @ K1)
        R_rel = U @ Vt * np.sign(np.linalg.det(U @ Vt))
        ref = json.loads(Path(a.ref_extrinsic).read_text())
        R_new = R_rel @ np.array(ref["rotation"], float)
        t_new = -R_new @ np.array([0.0, 0.0, a.height])
        ext = Path(a.out).parent / a.extrinsic_name
        ext.write_text(json.dumps({"rotation": R_new.tolist(), "translation": t_new.tolist(),
            "note": f"reference pose {a.ref_extrinsic} followed by the rotation measured from the static background "
                    f"(homography of median backgrounds, {int(inl.sum())} inliers), reference K kept (zoom ratio {f / K1[0, 0]:.4f}), height {a.height} m",
            "diagnostics": {"height_m": a.height, "rotation_from_ref_deg": ang, "zoom_ratio": f / float(K1[0, 0]),
                            "median_reproj_px": reproj, "reference_extrinsic": a.ref_extrinsic}}, indent=2))
        out["extrinsic_written"] = str(ext)
    Path(a.out).write_text(json.dumps(out, indent=1))
    vis = cv2.drawMatches(b1, k1, b2, k2, [m for m, ok in zip(good, inl) if ok][::20], None, matchColor=(0, 255, 0), flags=2)
    cv2.imwrite(a.out.rsplit(".", 1)[0] + "_matches.jpg", vis)
    print(json.dumps(out))


def selfcheck():
    """Synthetic: a rotation plus a zoom from 1556 to 1600 px is recovered."""
    K1 = np.array([[1556, 0, 960], [0, 1556, 540], [0, 0, 1.0]])
    K2 = np.array([[1600, 0, 960], [0, 1600, 540], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.array([0.02, -0.06, 0.01]))
    H = K2 @ R @ np.linalg.inv(K1)
    f, err, _ = solve(H, K1, 960, 540)
    assert abs(f - 1600) <= 1.0 and err < 1e-3, (f, err)
    print("selfcheck ok: f", f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        selfcheck()
    else:
        main()
