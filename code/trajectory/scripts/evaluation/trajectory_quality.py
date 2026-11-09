"""GPS-free quality of every extracted trajectory, not only the GPS car (overnight E4).

GPS covers one car per pass; these checks cover all of them, in the spirit of the
internal-consistency analysis of Punzo et al. (2011) for NGSIM:
  jitter       RMS of raw detections about the smoothed path, along and across the road,
               per track, by range: how steady the detector is
  hard accel   share of smoothed states with |a| > 4 m/s^2 (physically rare on this road)
  jerk flips   sign changes of jerk per second (Punzo's jerk analysis)
  coverage     vehicle-seconds and tracks of 1 s or more
Tracks shorter than 30 states are skipped.

    python scripts/evaluation/trajectory_quality.py TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/evaluation")]
import heldout_eval as he       # noqa: E402

BINS = ((0, 40), (40, 70), (70, 105))


def track_stats(st):
    t = np.array([s["t_s"] for s in st]); sp = np.array([s["speed_mps"] for s in st])
    x = np.array([s["x"] for s in st]); y = np.array([s["y"] for s in st])
    det = np.array([s.get("x_det") is not None for s in st])
    xd = np.array([s["x_det"] if s.get("x_det") is not None else np.nan for s in st])
    yd = np.array([s["y_det"] if s.get("y_det") is not None else np.nan for s in st])
    a = np.gradient(sp, t); j = np.gradient(a, t)
    flips = np.sum(np.diff(np.sign(j[np.abs(j) > 1e-6])) != 0) / max(t[-1] - t[0], 1e-6)
    return {"n": len(st), "range": float(np.median(x)), "along": float(np.sqrt(np.nanmean((xd - x)[det] ** 2))) if det.any() else np.nan,
            "across": float(np.sqrt(np.nanmean((yd - y)[det] ** 2))) if det.any() else np.nan,
            "hard": float(np.mean(np.abs(a) > 4.0)), "flips": float(flips)}


def main(tags):
    out = {}
    for tag in tags:
        rows = []
        for c in he.HELD:
            p = ROOT / f"outputs/trajectories/{c}_{tag}/tracks.json"
            if p.exists():
                rows += [track_stats(st) for st in json.loads(p.read_text())["tracks"].values() if len(st) >= 30]
        r = {"tracks": len(rows), "vehicle_seconds": round(sum(x["n"] for x in rows) / 30, 1),
             "hard_accel_share": round(float(np.mean([x["hard"] for x in rows])), 4),
             "jerk_flips_per_s": round(float(np.median([x["flips"] for x in rows])), 2)}
        for lo, hi in BINS:
            b = [x for x in rows if lo <= x["range"] < hi]
            r[f"jitter_{lo}_{hi}m"] = (len(b), round(float(np.nanmedian([x["along"] for x in b])), 3) if b else None,
                                      round(float(np.nanmedian([x["across"] for x in b])), 3) if b else None)
        out[tag] = r
        print(tag, r)
    (ROOT / "outputs/evaluation/round2/trajectory_quality.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
