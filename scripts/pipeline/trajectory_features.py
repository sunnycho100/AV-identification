"""Per-vehicle driving measures from extract_trajectories.py output.

One row per vehicle observed for at least MIN_DURATION_S while moving:
  speed         mean, std, min, max (RTS-smoothed, m/s)
  acceleration  along the direction of travel, central difference of speed over
                DIFF_WIN_S; rms and max |a|
  jerk          same difference applied to acceleration; rms
  lane changes  lane id switches that hold for at least HOLD_S
  lane keeping  std of the lateral offset from the lane centre, on segments
                without a lane change
  following     median and minimum time headway to the leader in the same
                lane, and the share of time with a leader closer than 3 s
Every derivative depends on the smoothing and on DIFF_WIN_S, so these settings
are written next to the table and must be identical for both vehicle groups.
The instrumented vehicle (instrumented.json) is flagged.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/pipeline/trajectory_features.py \
        --runs AV_T_EW_3_ft102 HV_T_EW_1_ft102
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 30.0
MIN_DURATION_S, MOVING_MPS = 2.0, 3.0
DIFF_WIN_S, HOLD_S, FOLLOW_S = 0.5, 0.5, 3.0


def diff(v, fps=FPS, win=DIFF_WIN_S, t=None):
    """Central difference over win seconds (clamped at the ends). With t (seconds
    per sample) the step is divided by the real elapsed time, not samples / fps."""
    h = max(1, int(round(win * fps / 2)))
    i = np.arange(len(v))
    lo, hi = np.clip(i - h, 0, len(v) - 1), np.clip(i + h, 0, len(v) - 1)
    if t is None:
        return (v[hi] - v[lo]) / np.maximum(hi - lo, 1) * fps
    return (v[hi] - v[lo]) / np.maximum(t[hi] - t[lo], 1.0 / fps)


def lane_changes(lanes, fps=FPS, hold=HOLD_S):
    """Switches between lanes that each last at least `hold` seconds."""
    runs = []
    for lane in lanes:
        if runs and runs[-1][0] == lane:
            runs[-1][1] += 1
        else:
            runs.append([lane, 1])
    stable = [r[0] for r in runs if r[0] is not None and r[1] >= hold * fps]
    return sum(1 for a, b in zip(stable, stable[1:]) if a != b)


def features(rows):
    rows = [r for i, r in enumerate(rows) if i == 0 or r["t_s"] > rows[i - 1]["t_s"]]   # frozen copies
    t = np.array([r["t_s"] for r in rows], float)
    v = np.array([r["speed_mps"] for r in rows], float)
    if t[-1] - t[0] < MIN_DURATION_S or np.median(v) < MOVING_MPS:
        return None
    a = diff(v, t=t)
    j = diff(a, t=t)
    lanes = [None if r["lane"] in ("", None) else int(r["lane"]) for r in rows]
    lat = np.array([np.nan if r["lateral_m"] in ("", None) else float(r["lateral_m"]) for r in rows])
    n_lc = lane_changes(lanes)
    hw = np.array([float(r["time_headway_s"]) for r in rows if r["time_headway_s"] not in ("", None)])
    return {
        "duration_s": round(t[-1] - t[0], 2), "frames": len(rows),
        "detected_share": round(np.mean([str(r["detected"]) == "True" for r in rows]), 3),
        "mean_along_m": round(float(np.mean([float(r["along_m"]) for r in rows])), 1),
        "speed_mean": round(v.mean(), 2), "speed_std": round(v.std(), 3),
        "speed_min": round(v.min(), 2), "speed_max": round(v.max(), 2),
        "accel_rms": round(float(np.sqrt(np.mean(a ** 2))), 3), "accel_max_abs": round(float(np.abs(a).max()), 3),
        "jerk_rms": round(float(np.sqrt(np.mean(j ** 2))), 3),
        "lane_changes": n_lc,
        "lateral_std_m": round(float(np.nanstd(lat)), 3) if n_lc == 0 and np.isfinite(lat).any() else None,
        "headway_median_s": round(float(np.median(hw)), 2) if len(hw) else None,
        "headway_min_s": round(float(hw.min()), 2) if len(hw) else None,
        "following_share": round(float(np.mean(hw < FOLLOW_S)) * len(hw) / len(rows), 3) if len(hw) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser("Per-vehicle driving measures")
    ap.add_argument("--runs", nargs="+", required=True, help="folders under outputs/trajectories")
    ap.add_argument("--out", default="outputs/trajectories/features.csv")
    a = ap.parse_args()
    table = []
    for run in a.runs:
        d = ROOT / "outputs/trajectories" / run
        inst = json.loads((d / "instrumented.json").read_text())["vehicle_id"] if (d / "instrumented.json").exists() else None
        for p in sorted(d.glob("vehicle_*.csv")):
            rows = list(csv.DictReader(open(p)))
            for r in rows:
                r["t_s"], r["speed_mps"] = float(r["t_s"]), float(r["speed_mps"])
            f = features(rows)
            if f:
                vid = p.stem.split("_", 1)[1]
                table.append({"run": run, "vehicle": vid, "instrumented": vid == inst,
                              "direction": rows[0]["direction"], **f})
    out = ROOT / a.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    out.with_suffix(".settings.json").write_text(json.dumps({
        "runs": a.runs, "fps": FPS, "min_duration_s": MIN_DURATION_S, "moving_mps": MOVING_MPS,
        "diff_window_s": DIFF_WIN_S, "lane_change_hold_s": HOLD_S, "following_s": FOLLOW_S,
        "smoothing": "RTS constant-velocity (postprocess_tracks.py), applied identically to every vehicle"}, indent=2))
    print(f"{len(table)} vehicles from {len(a.runs)} runs -> {out}")
    for r in table:
        if r["instrumented"]:
            print("instrumented:", {k: r[k] for k in ("run", "vehicle", "duration_s", "speed_mean", "accel_rms",
                                                      "jerk_rms", "lane_changes", "lateral_std_m", "headway_median_s")})


def _selfcheck():
    """Constant speed has no acceleration; a ramp is measured; a held lane change
    counts once and boundary flicker does not."""
    v = np.full(90, 28.0)
    assert np.abs(diff(v)).max() < 1e-9
    ramp = 20 + 1.5 * np.arange(90) / FPS
    assert np.allclose(diff(ramp)[10:-10], 1.5)
    assert lane_changes([1] * 30 + [2] * 30) == 1
    assert lane_changes([1] * 30 + [2, 1, 2, 1] + [1] * 30) == 0
    t31 = np.arange(90) / 31.0
    assert np.allclose(diff(20 + 1.5 * t31, t=t31)[10:-10], 1.5)
    print("selfcheck ok (zero, ramp 1.5 m/s^2 on 30 and 31 fps clocks, one held lane change, flicker ignored)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
