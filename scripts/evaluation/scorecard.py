"""GPS-free scorecard for one pipeline version, run after every change.

Per run (outputs/trajectories/<clip>_<tag>):
  frag_1s       share of states in tracks of at least 1 s (run.json)
  accel_steady  median rms acceleration of cars without a lane change, m/s^2
                (lower is smoother; real highway cruising is ~0.1-0.2)
  lat_std       median lateral std within a lane, m
  follow_mph    median |follower - leader| speed at spacing under 40 m, mph
                (cars following closely move at nearly the same speed)
  frozen        states inside frozen video frames (frame_times.json), must be 0
  resid_m       rms of detection minus smoothed position along the road, m
                (guards against over-smoothing: should stay near detection noise)
  resid_ac1     lag-1 autocorrelation of that residual within a track
                (near 0: the smoother follows the car; large: it lags behind)
Writes nothing; prints one row per run and the mean, as JSON if --json.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/scorecard.py --tag ft102
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/pipeline")]
import trajectory_features as tf   # noqa: E402
import frame_times as ft           # noqa: E402

CLIPS = ("AV_T_EW_3", "HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "HV_T_EW_2")
MPH = 2.23694


def score(clip, tag):
    d = ROOT / "outputs/trajectories" / f"{clip}_{tag}"
    run = json.loads((d / "run.json").read_text())
    V = {}
    for p in d.glob("vehicle_*.csv"):
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            r["t_s"], r["speed_mps"] = float(r["t_s"]), float(r["speed_mps"])
        V[p.stem.split("_", 1)[1]] = rows
    feats = [f for f in (tf.features(r) for r in V.values()) if f]
    steady = [f["accel_rms"] for f in feats if f["lane_changes"] == 0]
    lat = [f["lateral_std_m"] for f in feats if f["lateral_std_m"] is not None]
    by = {vid: {int(r["frame"]): r for r in rows} for vid, rows in V.items()}
    gaps = [abs(float(r["speed_mps"]) - float(by[r["lead_id"]][f]["speed_mps"])) * MPH
            for vid, R in by.items() for f, r in R.items()
            if r["lead_id"] and r["spacing_m"] and float(r["spacing_m"]) < 40
            and r["lead_id"] in by and f in by[r["lead_id"]]]
    T = json.loads((d / "tracks.json").read_text())["tracks"]
    res, ac = [], []
    for t in T.values():
        r = np.array([s["x_det"] - s["x"] for s in t if "x_det" in s])
        if len(r) >= 10:
            res.extend(r)
            ac.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
    times = ft.load(clip) or {}
    frozen = sum(1 for R in by.values() for f in R if times.get(f, (0, False))[1])
    return {"clip": clip, "vehicles": len(feats),
            "frag_1s": round(run["fragmentation"]["states_in_tracks_1s_plus"], 3),
            "accel_steady": round(float(np.median(steady)), 3) if steady else None,
            "lat_std": round(float(np.median(lat)), 3) if lat else None,
            "follow_mph": round(float(np.median(gaps)), 2) if gaps else None,
            "frozen": frozen,
            "resid_m": round(float(np.sqrt(np.mean(np.square(res)))), 3) if res else None,
            "resid_ac1": round(float(np.median(ac)), 3) if ac else None}


def main():
    ap = argparse.ArgumentParser("GPS-free scorecard")
    ap.add_argument("--tag", default="ft102")
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = [score(c, a.tag) for c in a.clips]
    keys = ("frag_1s", "accel_steady", "lat_std", "follow_mph", "resid_m", "resid_ac1")
    mean = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) for k in keys}
    mean["frozen"] = sum(r["frozen"] for r in rows)
    if a.json:
        print(json.dumps({"tag": a.tag, "rows": rows, "mean": mean}))
        return
    for r in rows:
        print(r)
    print("mean", mean)


if __name__ == "__main__":
    main()
