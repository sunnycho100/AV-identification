"""Item 11: RTS-smooth a camera track and compare raw vs smoothed against GPS.

Constant-velocity Kalman filter over the track's (x, y) at the clip frame
rate, then Rauch-Tung-Striebel backward smoothing (filterpy). Speed from the
smoothed velocity state. GPS remains evaluation-only.

Run (miniforge env - needs filterpy):
    NUMBA_DISABLE_JIT=1 /Users/sunghwan_cho/miniforge/bin/python3.12 \
        scripts/tracking/rts_smooth_track.py \
        --tracks outputs/tracking/camera-data/AV_T_WE_1/tracks.json \
        --track-id 141 \
        --gps-csv "Camera data/AV_T_WE_1_trajectory.csv" \
        --out-dir outputs/tracking/camera-data/AV_T_WE_1
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from filterpy.kalman import KalmanFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_kf(dt):
    kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, y, vx, vy]
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]])
    kf.R *= 0.25          # ~0.5 m detection noise
    kf.Q = np.diag([0.01, 0.01, 0.5, 0.5]) * dt
    kf.P = np.diag([1, 1, 100, 100.])
    return kf


def gps_speed_per_frame(csv_path):
    import csv as csvmod
    per = {}
    with open(csv_path) as f:
        for row in csvmod.DictReader(f):
            if not row["nearest_frame_index"]:
                continue
            fr = int(row["nearest_frame_index"])
            per.setdefault(fr, []).append(
                math.hypot(float(row["processed_vx"]), float(row["processed_vy"])))
    return {fr: float(np.mean(v)) for fr, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--track-id", required=True)
    ap.add_argument("--gps-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.tracks).read_text())
    fps = data["meta"]["frame_rate_hz"]
    v = data["tracks"][args.track_id]
    frames = [s["frame"] for s in v]
    xy = np.array([[s["x"], s["y"]] for s in v])
    raw_speed = np.array([s["speed_mps"] for s in v])

    kf = build_kf(1.0 / fps)
    kf.x = np.array([xy[0, 0], xy[0, 1], 0, 0.])
    means, covs = [], []
    for z in xy:
        kf.predict()
        kf.update(z)
        means.append(kf.x.copy())
        covs.append(kf.P.copy())
    means, covs = np.array(means), np.array(covs)
    sm, _, _, _ = kf.rts_smoother(means, covs)
    smooth_speed = np.hypot(sm[:, 2], sm[:, 3])

    gps = gps_speed_per_frame(args.gps_csv)
    idx = [i for i, fr in enumerate(frames) if fr in gps]
    gs = np.array([gps[frames[i]] for i in idx])
    raw_mae = float(np.abs(raw_speed[idx] - gs).mean())
    sm_mae = float(np.abs(smooth_speed[idx] - gs).mean())
    # steady-state (skip 5-frame warmup)
    ss = [i for i in idx if i >= 5]
    gss = np.array([gps[frames[i]] for i in ss])
    raw_ss = float(np.abs(raw_speed[ss] - gss).mean())
    sm_ss = float(np.abs(smooth_speed[ss] - gss).mean())
    out = {
        "track_id": args.track_id, "n_frames": len(v), "fps": fps,
        "speed_mae_raw": raw_mae, "speed_mae_rts": sm_mae,
        "speed_mae_raw_steady": raw_ss, "speed_mae_rts_steady": sm_ss,
        "mean_speed_gps": float(gs.mean()),
        "mean_speed_raw": float(raw_speed[idx].mean()),
        "mean_speed_rts": float(smooth_speed[idx].mean()),
    }
    print(json.dumps(out, indent=2))
    out_dir = Path(args.out_dir)
    (out_dir / "rts_comparison.json").write_text(json.dumps(out, indent=2))

    plt.figure(figsize=(10, 5))
    plt.plot([frames[i] for i in idx], gs, "g-", lw=3, label="GPS")
    plt.plot(frames, raw_speed, "r--", lw=1.5, label="raw AB3DMOT speed")
    plt.plot(frames, smooth_speed, "b-", lw=2, label="RTS-smoothed speed")
    plt.ylim(0, 50)
    plt.xlabel("frame"); plt.ylabel("m/s")
    plt.title(f"Track {args.track_id}: speed raw MAE {raw_mae:.2f} vs RTS {sm_mae:.2f} m/s "
              f"(steady-state {raw_ss:.2f} vs {sm_ss:.2f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "rts_speed_comparison.png", dpi=110)
    print(f"figure -> {out_dir}/rts_speed_comparison.png")


if __name__ == "__main__":
    main()
