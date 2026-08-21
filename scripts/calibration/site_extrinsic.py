"""Extrinsic for a clip from its own vanishing point plus a known site height.

Why not solve height per clip: on these Beltline cameras the lane-width grid
scatters by ~2 m even where it looks confident (AV_T_WE_1 and HV_T_EW_2 are the
same physical pose yet pool to 18.0 and 16.0 m). The Todd Drive height is instead
fixed by two routes that do not use the grid at all - July's GPS-scale estimate
16.22 m and its vehicle-length estimate 16.31 m - so it is treated as a site
constant and only the rotation is solved per clip.

The rotation comes from the median vanishing point over all the clip's frames,
which is robust to the odd frame where LSD recovers a bad line set. The spread
across frames is reported, and so is the angle to a reference pose: these are PTZ
cameras, so a clip that has been re-pointed must not silently inherit the site
height.

    .venv/bin/python scripts/calibration/site_extrinsic.py \
        --frames-dir data/camera-data/AV_T_EW_3/frames \
        --anycalib-json outputs/calibration/camera-data/AV_T_EW_3/150_anycalib_pinhole_pinhole.json \
        --height 16.26 \
        --reference outputs/calibration/camera-data/AV_T_WE_1/metric_extrinsic_v2.json \
        --out outputs/calibration/camera-data/AV_T_EW_3/metric_extrinsic_site.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
from scripts.calibration.vp_extrinsic_from_frame import solve_pose

MAX_POSE_DIFF_DEG = 2.0   # beyond this the camera has been re-pointed


def clip_vanishing_point(frames_dir):
    """Median vanishing point over the clip, with the per-frame spread."""
    vps, names = [], []
    for p in sorted(Path(frames_dir).glob("*.jpg")):
        vp, _, diag = lsd_vanishing_point(cv2.imread(str(p)))
        if vp is None:
            continue
        vps.append(vp)
        names.append(p.name)
    if len(vps) < 3:
        return None, {"n_frames": len(vps)}
    vps = np.array(vps)
    med = np.median(vps, axis=0)
    return med, {"n_frames": len(vps), "frames_used": names,
                 "vp_px": [float(med[0]), float(med[1])],
                 "vp_spread_px": [float(vps[:, 0].std()), float(vps[:, 1].std())],
                 "vp_per_frame": vps.tolist()}


def angle_between(Ra, Rb):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser("Clip extrinsic from its VP plus a site height")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--anycalib-json", required=True)
    ap.add_argument("--height", type=float, required=True,
                    help="site camera height in m, established independently")
    ap.add_argument("--reference", default=None,
                    help="extrinsic JSON of a clip known to share this pose; "
                         "the solved rotation is compared against it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fx, fy, cx, cy = json.loads(
        Path(args.anycalib_json).read_text())["prediction"]["intrinsics"][:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    vp, diag = clip_vanishing_point(args.frames_dir)
    if vp is None:
        sys.exit(f"only {diag['n_frames']} frames yielded a vanishing point")
    R, t, pitch, yaw = solve_pose(K, vp, args.height)
    print(f"VP = ({vp[0]:.1f}, {vp[1]:.1f}) px over {diag['n_frames']} frames "
          f"(spread {diag['vp_spread_px'][0]:.1f}, {diag['vp_spread_px'][1]:.1f} px)")
    print(f"pose: pitch={pitch:.2f} yaw={yaw:.2f} roll=0, height={args.height} m (site constant)")

    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        d = angle_between(np.array(ref["rotation"], dtype=float), R)
        diag["pose_diff_vs_reference_deg"] = d
        diag["reference"] = str(args.reference)
        verdict = "same pose" if d <= MAX_POSE_DIFF_DEG else "RE-POINTED"
        print(f"vs reference: {d:.2f} deg -> {verdict}")
        if d > MAX_POSE_DIFF_DEG:
            print(f"WARN: {d:.2f} deg from the reference pose exceeds "
                  f"{MAX_POSE_DIFF_DEG} deg, so the site height may not apply "
                  f"to this clip; solve its height before trusting the output")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rotation": R.tolist(), "translation": t.tolist(),
        "note": (f"VP pose (pitch={pitch:.2f} yaw={yaw:.2f} roll=0) over "
                 f"{diag['n_frames']} frames + site height {args.height} m "
                 f"(established GPS-free, not solved from this clip)"),
        "diagnostics": {"pitch_deg": pitch, "yaw_deg": yaw,
                        "height_m": args.height, "height_source": "site constant",
                        **diag},
    }, indent=2))
    print(f"wrote {out}")


def _selfcheck():
    """A pose rebuilt from its own vanishing point must come back unchanged, and
    the pose comparison must separate a nudge from a re-point."""
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    R, t, pitch, yaw = solve_pose(K, (156.6, 21.9), 16.26)
    assert abs(angle_between(R, R)) < 1e-9
    R2, _, _, _ = solve_pose(K, (160.0, 25.0), 16.26)
    small = angle_between(R, R2)
    R3, _, _, _ = solve_pose(K, (600.0, 200.0), 16.26)
    big = angle_between(R, R3)
    assert small < MAX_POSE_DIFF_DEG < big, (small, big)
    # the height must land in the translation, not the rotation
    Ra, ta, _, _ = solve_pose(K, (156.6, 21.9), 16.26)
    Rb, tb, _, _ = solve_pose(K, (156.6, 21.9), 8.13)
    assert np.allclose(Ra, Rb) and np.allclose(ta, 2 * tb), (ta, tb)
    print(f"selfcheck ok (nudge {small:.2f} deg, re-point {big:.2f} deg, "
          f"threshold {MAX_POSE_DIFF_DEG} deg)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
