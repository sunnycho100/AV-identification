"""Grade the marker-identified target track against the GPS log.

The target was chosen from the hood marker alone (identify_target_by_marker.py),
so this is a test of a decision already made rather than a search for the track
that scores best. For contrast the best-scoring track is also reported: if the two
differ, the older "pick the closest match" number was measuring the wrong car.

Three numbers, in order of what they actually tell us:

1. Similarity-fit SCALE. Rotation, translation and scale are fitted; a metrically
   correct calibration returns 1.00. Alignment cannot absorb this, which is why it
   is the headline and RMSE is not.
2. Speed error, which needs no alignment at all and is independently sensitive to
   the same scale question.
3. Rigid-fit RMSE (rotation and translation only). Path shape, not size.

GPS is evaluation-only and is never fed back into calibration or tracking.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_gps(csv_path):
    per = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["nearest_frame_index"]:
                continue
            per.setdefault(int(row["nearest_frame_index"]), []).append(
                (float(row["processed_x"]), float(row["processed_y"]),
                 math.hypot(float(row["processed_vx"]), float(row["processed_vy"]))))
    return {k: tuple(np.mean(v, axis=0)) for k, v in per.items()}


def rigid_align(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def similarity_align(A, B):
    """Umeyama with scale. Returns (R, t, s) minimising ||s*R*A + t - B||."""
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    H = Ac.T @ Bc / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = Vt.T @ S @ U.T
    var = (Ac ** 2).sum() / len(A)
    s = float((D * np.diag(S)).sum() / var)
    return R, cb - s * R @ ca, s


def score(states, gps, skip=()):
    common = [(s, gps[s["frame"]]) for s in states
              if s["frame"] in gps and not any(a <= s["frame"] <= b for a, b in skip)]
    if len(common) < 10:
        return None
    A = np.array([[g[0], g[1]] for _, g in common])       # GPS local frame
    B = np.array([[s["x"], s["y"]] for s, _ in common])   # camera ground frame
    R, t = rigid_align(A, B)
    rmse = float(np.sqrt((((A @ R.T + t) - B) ** 2).sum(1).mean()))
    Rs, ts, s_ = similarity_align(A, B)
    rmse_s = float(np.sqrt((((s_ * (A @ Rs.T)) + ts - B) ** 2).sum(1).mean()))
    v_cam = np.array([s["speed_mps"] for s, _ in common])
    v_gps = np.array([g[2] for _, g in common])

    # Two further scale estimates. The similarity fit's scale tracks the spread of
    # the whole point set, so a range-dependent depth warp contaminates it; the
    # chord (endpoint-to-endpoint) ratio and the displacement speed do not agree
    # with it unless the path is linearly scaled. Reporting all three exposes the
    # warp instead of hiding it in one number.
    frames = [s_["frame"] for s_, _ in common]
    dt = (frames[-1] - frames[0]) / 30.0
    chord = float(np.linalg.norm(B[-1] - B[0]) / max(np.linalg.norm(A[-1] - A[0]), 1e-9))
    disp_speed = float(np.linalg.norm(B[-1] - B[0]) / max(dt, 1e-9))

    # depth error along the viewing ray, binned by range: the residual structure
    resid = B - (A @ R.T + t)
    rng = np.linalg.norm(B, axis=1)
    depth_err = (resid * (B / rng[:, None])).sum(1)
    bands = {}
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 120)]:
        m = (rng >= lo) & (rng < hi)
        if m.sum() >= 5:
            bands[f"{lo}-{hi}m"] = {"n": int(m.sum()),
                                    "depth_err_median_m": float(np.median(depth_err[m]))}
    return {"n": len(common), "scale": s_, "scale_chord": chord,
            "rmse_rigid_m": rmse, "rmse_similarity_m": rmse_s,
            "speed_cam_mps": float(v_cam.mean()), "speed_gps_mps": float(v_gps.mean()),
            "speed_disp_mps": disp_speed,
            "speed_mae_mps": float(np.abs(v_cam - v_gps).mean()),
            "speed_ratio": float(v_cam.mean() / v_gps.mean()),
            "depth_err_range_corr": float(np.corrcoef(depth_err, rng)[0, 1]),
            "depth_err_by_range": bands}


def main():
    ap = argparse.ArgumentParser("Grade the marker-identified track against GPS")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--skip-frames", default=None,
                    help="inclusive ranges to exclude, e.g. 181-268 for a frozen "
                         "stretch of source video")
    args = ap.parse_args()
    clip = args.clip
    skip = []
    if args.skip_frames:
        for part in args.skip_frames.split(","):
            a, b = part.split("-")
            skip.append((int(a), int(b)))

    tdir = Path(f"outputs/tracking/camera-data/{clip}_phase1")
    tgt = json.loads((tdir / "target_track.json").read_text())
    gps = load_gps(ROOT / f"Camera data/{clip}_trajectory.csv")

    r = score(tgt["states"], gps, skip)
    if r is None:
        sys.exit("target track and GPS do not overlap enough to grade")

    print(f"{clip}: target = track {tgt['target_track_id']} "
          f"(chosen from the marker, {tgt['votes']}/{tgt['marker_frames']} votes)")
    print(f"  overlapping frames with GPS : {r['n']}")
    print(f"  SCALE similarity / chord    : {r['scale']:.4f} / {r['scale_chord']:.4f}   "
          f"(1.0000 = metrically correct; disagreement = range-dependent depth warp)")
    print(f"  speed  camera vs GPS        : {r['speed_cam_mps']:.2f} vs "
          f"{r['speed_gps_mps']:.2f} m/s   ratio {r['speed_ratio']:.4f}, "
          f"MAE {r['speed_mae_mps']:.2f}")
    print(f"  RMSE rigid / similarity     : {r['rmse_rigid_m']:.2f} m / "
          f"{r['rmse_similarity_m']:.2f} m")
    print(f"  depth error vs range        : corr {r['depth_err_range_corr']:+.2f}  " +
          "  ".join(f"{k} {v['depth_err_median_m']:+.2f}m"
                    for k, v in r["depth_err_by_range"].items()))

    # what the old "pick the best-matching track" rule would have chosen
    tracks = json.loads((tdir / "tracks.json").read_text())["tracks"]
    best, best_r = None, None
    for tid, st in tracks.items():
        rr = score(st, gps, skip)
        if rr and rr["n"] >= 25 and (best_r is None or rr["rmse_rigid_m"] < best_r["rmse_rigid_m"]):
            best, best_r = tid, rr
    if best is not None:
        agree = (best == tgt["target_track_id"])
        print(f"\n  best-scoring track by RMSE  : {best} "
              f"({best_r['rmse_rigid_m']:.2f} m, scale {best_r['scale']:.4f})")
        print(f"  agrees with the marker?     : "
              f"{'YES - independent identification confirmed' if agree else 'NO'}")
        if not agree:
            print("  the older method was grading a different vehicle than the "
                  "instrumented one")

    (tdir / "gps_grading_target.json").write_text(json.dumps(
        {"clip": clip, "target_track_id": tgt["target_track_id"],
         "identified_from": tgt["identified_from"], "skipped_frames": skip,
         "target": r, "best_scoring_track": best, "best_scoring": best_r}, indent=2))
    print(f"\nwrote {tdir/'gps_grading_target.json'}")


if __name__ == "__main__":
    main()
