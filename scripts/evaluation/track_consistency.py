"""GPS-free frame-to-frame consistency of the 3D tracks, by detection score band (D20 to D22).

For every car on the road, not only the lab car:
  resid_m / lat_resid_m  distance (and its across-road part) from each detected box
                         to the smoothed trajectory, split into boxes at 0.45 or above
                         and boxes under 0.45 (low-score continuation)
  wobble_m               across-road wander of a smoothed track: std of y after a
                         straight-line fit over time, median over tracks of 30+ states
  hard_accel             share of smoothed states with |acceleration| above 4 m/s^2
                         (0.5 s central difference)
  yaw_jump               share of consecutive detected states whose box heading moves
                         more than 30 deg in one frame (flips included), by band of the
                         later box

    python scripts/evaluation/track_consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS = {"2025": ["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"],
         "2026": ["AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]}
RUNS = ["base", "lo10", "lo30", "lo30k4"]
HIGH, A_MAX, YAW_JUMP, MIN_STATES = 0.45, 4.0, np.radians(30), 30


def tracks(clip, run):
    tag = ("abl_s4" if clip in CLIPS["2025"] else "iv_s4") if run == "base" else run
    return json.loads((ROOT / "outputs/trajectories" / f"{clip}_{tag}" / "tracks.json").read_text())["tracks"].values()


def measure(clips, run):
    res = {"high": [], "low": []}
    lat = {"high": [], "low": []}
    yaw = {"high": [], "low": []}
    wob, acc = [], []
    for clip in clips:
        for st in tracks(clip, run):
            det = [s for s in st if "x_det" in s]
            for s in det:
                b = "high" if s["score"] >= HIGH else "low"
                res[b].append(np.hypot(s["x_det"] - s["x"], s["y_det"] - s["y"]))
                lat[b].append(abs(s["y_det"] - s["y"]))
            for p, s in zip(det, det[1:]):
                if s["frame"] == p["frame"] + 1:
                    d = abs((s["yaw"] - p["yaw"] + np.pi) % (2 * np.pi) - np.pi)
                    yaw["high" if s["score"] >= HIGH else "low"].append(d > YAW_JUMP)
            if len(st) >= MIN_STATES:
                t = np.array([s["t_s"] for s in st])
                y = np.array([s["y"] for s in st])
                wob.append(np.std(y - np.polyval(np.polyfit(t, y, 1), t)))
                v = np.array([s["speed_mps"] for s in st])
                k = 7                                                   # about 0.25 s at 30 fps
                if len(v) > 2 * k:
                    a = (v[2 * k:] - v[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
                    acc.extend(np.abs(a) > A_MAX)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    return {"boxes_high": len(res["high"]), "boxes_low": len(res["low"]),
            "resid_med_high": q(res["high"], 50), "resid_med_low": q(res["low"], 50),
            "resid_p95_high": q(res["high"], 95), "resid_p95_low": q(res["low"], 95),
            "lat_p95_high": q(lat["high"], 95), "lat_p95_low": q(lat["low"], 95),
            "yaw_jump_high": float(np.mean(yaw["high"])) if yaw["high"] else float("nan"),
            "yaw_jump_low": float(np.mean(yaw["low"])) if yaw["low"] else float("nan"),
            "wobble_m": float(np.median(wob)), "hard_accel": float(np.mean(acc))}


def main():
    out = {f"{year}|{run}": measure(clips, run) for year, clips in CLIPS.items() for run in RUNS}
    out.update({f"HV_T_EW_6|{run}": measure(["HV_T_EW_6"], run) for run in RUNS})
    path = ROOT / "outputs/evaluation/lowscore/track_consistency.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:16s} boxes {v['boxes_high']}+{v['boxes_low']} | resid med {v['resid_med_high']:.2f}/{v['resid_med_low']:.2f} "
              f"p95 {v['resid_p95_high']:.2f}/{v['resid_p95_low']:.2f} lat p95 {v['lat_p95_high']:.2f}/{v['lat_p95_low']:.2f} m | "
              f"yaw jumps {v['yaw_jump_high']:.1%}/{v['yaw_jump_low']:.1%} | wobble {v['wobble_m']:.3f} m | hard accel {v['hard_accel']:.1%}")
    print("->", path)


if __name__ == "__main__":
    main()
