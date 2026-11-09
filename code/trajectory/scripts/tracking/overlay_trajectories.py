"""Draw each tracked vehicle's trajectory ON the camera image (not top-down).

For every track, projects its sequence of 3D positions into the image plane and
draws a trail: hollow circle = where it started, line = its path, filled dot +
ID = where it ended (its position in this frame). A still car shows as a dot with
no tail; a moving car shows a visible trail "from here to there".

Drawn on the segment's LAST frame by default (camera is fixed, so all trail points
project consistently onto it).

  /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/overlay_trajectories.py --label seg_00_05
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[2]
TRK_ROOT = ROOT / "outputs/tracking/our_intrinsics"
DATA_ROOT = ROOT / "data/dair-v2x-i"
ANYCALIB_K = ROOT / "outputs/calibration/anycalib_single/000000_anycalib_pinhole_pinhole.json"


def load_K():
    fx, fy, cx, cy = json.loads(ANYCALIB_K.read_text())["prediction"]["intrinsics"][:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_lidar2cam(fid):
    d = json.loads((DATA_ROOT / "calib/virtuallidar_to_camera" / f"{fid:06d}.json").read_text())
    ext = np.eye(4)
    ext[:3, :3] = np.array(d["rotation"], dtype=np.float64)
    ext[:3, 3] = np.array(d["translation"], dtype=np.float64).reshape(3)
    return ext


def project(xyz, lidar2cam, K):
    cam = lidar2cam @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if cam[2] <= 0.1:  # behind camera
        return None
    uv = K @ cam[:3]
    return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))


def bgr(rgba):
    return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))


def overlay_segment(label, min_len):
    data = json.loads((TRK_ROOT / label / "tracks.json").read_text())
    tracks, meta = data["tracks"], data["meta"]
    draw_frame = meta["frames"][1]  # last frame of the segment

    img = cv2.imread(str(DATA_ROOT / "image" / f"{draw_frame:06d}.jpg"))
    if img is None:
        raise FileNotFoundError(f"missing image for frame {draw_frame:06d}")
    K = load_K()
    lidar2cam = load_lidar2cam(draw_frame)

    kept = {tid: v for tid, v in tracks.items() if len(v) >= min_len}
    colors = cm.get_cmap("tab20")(np.linspace(0, 1, max(len(kept), 1)))

    n_moved, n_still = 0, 0
    for (tid, pts), rgba in zip(sorted(kept.items(), key=lambda kv: int(kv[0])), colors):
        c = bgr(rgba)
        px = [project((p["x"], p["y"], p["z"]), lidar2cam, K) for p in pts]
        px = [p for p in px if p is not None]
        if len(px) < 2:
            continue
        # net pixel displacement -> moved vs still
        moved = np.hypot(px[-1][0] - px[0][0], px[-1][1] - px[0][1]) > 12
        n_moved += moved
        n_still += not moved
        # trail
        for a, b in zip(px[:-1], px[1:]):
            cv2.line(img, a, b, c, 2, cv2.LINE_AA)
        cv2.circle(img, px[0], 5, c, 1, cv2.LINE_AA)       # start: hollow
        cv2.circle(img, px[-1], 5, c, -1, cv2.LINE_AA)     # end: filled
        cv2.putText(img, str(tid), (px[-1][0] + 6, px[-1][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    legend = f"frame {draw_frame:06d} | {len(kept)} tracks (>={min_len} pts): {n_moved} moved, {n_still} still"
    cv2.rectangle(img, (0, 0), (max(620, 8 * len(legend)), 26), (0, 0, 0), -1)
    cv2.putText(img, legend, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = TRK_ROOT / label / "trajectories_overlay.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"  {label}: {n_moved} moved + {n_still} still on frame {draw_frame:06d} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Overlay trajectories on the camera image")
    ap.add_argument("--label", default=None, help="segment subfolder; omit for all segments")
    ap.add_argument("--min-len", type=int, default=3, help="min track length to draw")
    args = ap.parse_args()

    labels = [args.label] if args.label else [p.parent.name for p in TRK_ROOT.glob("*/tracks.json")]
    for label in sorted(labels):
        overlay_segment(label, args.min_len)


if __name__ == "__main__":
    main()
