"""Association-quality stats + target-vehicle grading against the GPS CSV.

Items 3+4 of the 7-15 overnight loop.
- Track stats: length distribution, moving/static split, top-down plot.
- Target grading: among moving tracks, find the one whose path best matches
  the GPS trajectory under a RIGID 2D alignment (rotation+translation, NO
  scale - metric scale is exactly what the calibration must get right), then
  report position RMSE and speed error. GPS is used for EVALUATION ONLY.

Run:
    .venv/bin/python scripts/tracking/analyze_tracks_vs_gps.py \
        --tracks outputs/tracking/camera-data/AV_T_WE_1/tracks.json \
        --gps-csv "Camera data/AV_T_WE_1_trajectory.csv" \
        --out-dir outputs/tracking/camera-data/AV_T_WE_1
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_gps_per_frame(csv_path):
    """frame -> (x_east, y_south, speed) from nearest_frame_index mapping."""
    per_frame = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["nearest_frame_index"]:
                continue
            fr = int(row["nearest_frame_index"])
            x = float(row["processed_x"])
            y = float(row["processed_y"])
            spd = math.hypot(float(row["processed_vx"]), float(row["processed_vy"]))
            per_frame.setdefault(fr, []).append((x, y, spd))
    return {fr: tuple(np.mean(v, axis=0)) for fr, v in per_frame.items()}


def rigid_align(A, B):
    """R,t minimizing ||R@A + t - B|| (no scale). A,B: Nx2."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gps-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-len", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.tracks).read_text())
    tracks = data["tracks"]
    fps = data["meta"]["frame_rate_hz"]

    # ---- item 3: association quality ----
    lengths = {tid: len(v) for tid, v in tracks.items()}
    mean_speeds = {tid: float(np.mean([s["speed_mps"] for s in v[1:]] or [0]))
                   for tid, v in tracks.items()}
    moving = [tid for tid in tracks if mean_speeds[tid] > 5.0]
    static = [tid for tid in tracks if mean_speeds[tid] <= 5.0]
    long_moving = [tid for tid in moving if lengths[tid] >= args.min_len]
    stats = {
        "n_tracks": len(tracks),
        "n_moving": len(moving),
        "n_static": len(static),
        "n_moving_ge_min_len": len(long_moving),
        "len_percentiles_moving": np.percentile(
            [lengths[t] for t in moving], [25, 50, 75, 90, 100]).tolist() if moving else [],
        "fps": fps,
    }
    print(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist([lengths[t] for t in moving], bins=25, color="steelblue")
    axes[0].set_title(f"Moving-track lengths (n={len(moving)}, {fps:.0f} Hz)")
    axes[0].set_xlabel("frames"); axes[0].set_ylabel("tracks")
    for tid in moving:
        v = tracks[tid]
        axes[1].plot([s["x"] for s in v], [s["y"] for s in v], lw=1)
    axes[1].set_title("Top-down moving tracks (ground frame, m)")
    axes[1].set_xlabel("x fwd (m)"); axes[1].set_ylabel("y lat (m)")
    axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "association_quality.png", dpi=110)

    # ---- item 4: grade best-matching track vs GPS ----
    gps = load_gps_per_frame(args.gps_csv)
    results = []
    for tid in long_moving:
        v = tracks[tid]
        common = [(s, gps[s["frame"]]) for s in v if s["frame"] in gps]
        if len(common) < args.min_len:
            continue
        A = np.array([[g[0], g[1]] for _, g in common])       # GPS east/south
        B = np.array([[s["x"], s["y"]] for s, _ in common])   # track ground frame
        R, t = rigid_align(A, B)
        resid = (A @ R.T + t) - B
        rmse = float(np.sqrt((resid ** 2).sum(1).mean()))
        spd_track = np.array([s["speed_mps"] for s, _ in common])
        spd_gps = np.array([g[2] for _, g in common])
        results.append({
            "track_id": tid, "n_common": len(common), "rmse_m": rmse,
            "speed_mae_mps": float(np.abs(spd_track - spd_gps).mean()),
            "mean_speed_track": float(spd_track.mean()),
            "mean_speed_gps": float(spd_gps.mean()),
        })
    if not results:
        print("NO candidate track overlaps the GPS window - grading failed")
        return
    results.sort(key=lambda r: r["rmse_m"])
    best = results[0]
    print("\nbest-matching track (rigid align, NO scale):")
    print(json.dumps(best, indent=2))
    (out_dir / "gps_grading.json").write_text(json.dumps(
        {"stats": stats, "best": best, "all_candidates": results[:10]}, indent=2))

    # figure: aligned paths + speed curves
    tid = best["track_id"]
    v = tracks[tid]
    common = [(s, gps[s["frame"]]) for s in v if s["frame"] in gps]
    A = np.array([[g[0], g[1]] for _, g in common])
    B = np.array([[s["x"], s["y"]] for s, _ in common])
    R, t = rigid_align(A, B)
    A_al = A @ R.T + t
    frames = [s["frame"] for s, _ in common]
    spd_track = [s["speed_mps"] for s, _ in common]
    spd_gps = [g[2] for _, g in common]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(A_al[:, 0], A_al[:, 1], "g-", lw=3, label="GPS (rigid-aligned)")
    axes[0].plot(B[:, 0], B[:, 1], "r--", lw=2, label=f"camera track {tid}")
    axes[0].set_title(f"Path: RMSE={best['rmse_m']:.2f} m over {best['n_common']} frames")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].legend(); axes[0].set_aspect("equal")
    axes[1].plot(frames, spd_gps, "g-", lw=3, label="GPS speed")
    axes[1].plot(frames, spd_track, "r--", lw=2, label="camera track speed")
    axes[1].set_title(f"Speed: MAE={best['speed_mae_mps']:.2f} m/s "
                      f"(GPS {best['mean_speed_gps']:.1f} vs cam {best['mean_speed_track']:.1f})")
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("m/s"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gps_vs_track.png", dpi=110)
    print(f"figures -> {out_dir}/association_quality.png, gps_vs_track.png")


if __name__ == "__main__":
    main()
