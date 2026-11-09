"""Round-2 heads compared on the frames both runs track (D34), so added harder frames do not count against a head.

Graded as lowfilter_study.grade (file clock), but each pair's errors are computed on
the frames observed in both runs. 95% intervals resample the held-out passes (seed 0).

    python scripts/evaluation/round2_same_frames.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import ablation as ab           # noqa: E402
import heldout_eval as he       # noqa: E402
import lowscore_sweep as ls     # noqa: E402
import new_todd_eval as nt      # noqa: E402

PAIRS = {"hb30": [("r2fullhb", "hb30"), ("r2fullhb", "r2no2dhb"), ("r2fullhb", "r2noighb")],
         "S4": [("r2fulls4", "nv45"), ("r2fulls4", "r2no2ds4"), ("r2fulls4", "r2noigs4")]}
KEYS = {"speed_mps": lambda m: m["speed_mae_mph"] * 0.44704, "lateral_m": lambda m: m["across_rms_m"],
        "box_heading_deg": lambda m: m["box_heading_deg"]}


def load(clip, t, T, gps):
    inst = ls.TRAJ / f"{clip}_{t}" / "instrumented.json"
    if not inst.exists() or json.loads(inst.read_text()).get("vehicle_id") is None:
        return None
    d = ab.load(clip, t, T)
    if d is not None and clip not in he.Y2025:
        d, _ = nt.realign(d, gps, T, 0.0)
    return d


def main():
    from gps_to_image import load_clip, load_site
    T = load_site()
    rng = np.random.default_rng(0)
    out = {}
    for trk, pairs in PAIRS.items():
        for a, b in pairs:
            rows = []
            for c in he.HELD:
                gps = load_clip(c, he.tag(c, "S4"))["gps"]
                g = gps if c in he.Y2025 else nt.shifted(gps, 0.0)
                da, db = load(c, a, T, gps), load(c, b, T, gps)
                if da is None or db is None:
                    continue
                common = sorted(set(da["frame"][da["observed"]].tolist()) & set(db["frame"][db["observed"]].tolist()))
                if len(common) < 10:
                    continue
                ma, mb = (ab.agg(p, np.arange(len(common)), np.isfinite(p["acc"]))
                          for p in (ab.per_frame(da, common, g, T), ab.per_frame(db, common, g, T)))
                rows.append({"clip": c, "frames": len(common), **{f"{k}_a": f(ma) for k, f in KEYS.items()},
                             **{f"{k}_b": f(mb) for k, f in KEYS.items()}})
            res = {"passes": len(rows), "frames": sum(r["frames"] for r in rows)}
            for k in KEYS:
                d = np.array([r[f"{k}_a"] - r[f"{k}_b"] for r in rows])
                boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                res[k] = {"a": round(float(np.mean([r[f"{k}_a"] for r in rows])), 3), "b": round(float(np.mean([r[f"{k}_b"] for r in rows])), 3),
                          "diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)]}
            out[f"{trk}: {a} vs {b}"] = res
            print(f"{trk}: {a} vs {b} | {res['passes']} passes, {res['frames']} common frames | " +
                  " | ".join(f"{k} {v['a']} vs {v['b']}, diff {v['diff']:+} {v['ci95']}" for k, v in res.items() if isinstance(v, dict)))
    (ROOT / "outputs/evaluation/round2/same_frames.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
