"""Low-score box filters (decision D29) on the retrained head (D27).

Variants, all on the retrained head's 0.10 export (nvfeat25_lo10):
  nv45       S4: continuation at 0.45 (no low-score boxes)
  nv30k4     continuation at 0.30, weak boxes at 4x measurement noise (D21)
  nv30k4o    + drop low-score boxes on the burnt-in caption, logo or tag
  nv30k4o2   + keep a low-score box only if a 2D car, truck or bus box overlaps it (IoU 0.3)
  nv30k4o2g  + a low-score box joins a track only within 1.5 m of the track's prediction
Choice on the tuning clips (rule in D29); held-out passes report every variant.

    python scripts/evaluation/lowfilter_study.py run CLIP [CLIP ...]
    python scripts/evaluation/lowfilter_study.py score
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

DET = "nvfeat25_lo10"
VARIANTS = {"nv45": (0.45, 1, "", 0.0), "nv30k4": (0.30, 4, "", 0.0), "nv30k4o": (0.30, 4, "overlay", 0.0),
            "nv30k4o2": (0.30, 4, "overlay2d", 0.0), "nv30k4o2g": (0.30, 4, "overlay2d", 1.5)}
OUT = ROOT / "outputs/evaluation/lowfilter"


def run(clips):
    for clip in clips:
        for t, (L, k, filt, gate) in VARIANTS.items():
            if not (ls.TRAJ / f"{clip}_{t}" / "tracks.json").exists():
                ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", t, "--det-tag", DET,
                       "--track-low-thresh", L, "--low-noise", k, "--low-filter", filt, "--low-gate-m", gate,
                       "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1])
            print(clip, t, "car", ls.mark(clip, t))


def grade(clip, T, tau):
    from gps_to_image import load_clip
    gps = load_clip(clip, he.tag(clip, "S4"))["gps"]
    g = gps if clip in he.Y2025 else nt.shifted(gps, tau.get(clip, 0.0))
    data = {}
    for t in VARIANTS:
        inst = ls.TRAJ / f"{clip}_{t}" / "instrumented.json"
        if inst.exists() and json.loads(inst.read_text()).get("vehicle_id") is not None:
            d = ab.load(clip, t, T)
            if d is not None and clip not in he.Y2025:
                d, _ = nt.realign(d, gps, T, tau.get(clip, 0.0))
            data[t] = d
    obs = {t: sorted(set(d["frame"][d["observed"]].tolist())) for t, d in data.items() if d is not None}
    union = set().union(*map(set, obs.values())) if obs else set()
    out = {}
    for t in VARIANTS:
        r = {"coverage": round(len(obs.get(t, [])) / max(len(union), 1), 3), "seconds": round(len(obs.get(t, [])) / 30, 1)}
        if len(obs.get(t, [])) >= 10:
            p = ab.per_frame(data[t], obs[t], g, T)
            m = ab.agg(p, np.arange(len(obs[t])), np.isfinite(p["acc"]))
            r.update(lateral_m=m["across_rms_m"], speed_mps=m["speed_mae_mph"] * 0.44704, accel=m["accel_rms_diff"],
                     box_heading_deg=m["box_heading_deg"])
        out[t] = r
    return out


def score():
    from gps_to_image import load_site
    T = load_site()
    tau = {c: (v.get("tau_used_s") or 0.0) for c, v in
           json.loads((ROOT / "outputs/evaluation/new_todd/results.json").read_text()).items()}
    res = {c: grade(c, T, tau) for c in he.TUNING + he.HELD}
    base = lambda c, t: res[c].get(t, {})
    passes = lambda t: all(
        "speed_mps" in base(c, t) and "speed_mps" in base(c, "nv45")
        and base(c, t)["speed_mps"] <= 1.1 * base(c, "nv45")["speed_mps"]
        and base(c, t)["lateral_m"] <= 1.1 * base(c, "nv45")["lateral_m"]
        and base(c, t)["coverage"] >= base(c, "nv45")["coverage"] for c in he.TUNING)
    keeps = lambda t: all(base(c, t)["coverage"] >= 0.9 * base(c, "nv30k4")["coverage"] for c in he.TUNING)
    chosen = next((t for t in ("nv30k4o2g", "nv30k4o2", "nv30k4o") if passes(t) and keeps(t)), None) \
        or ("nv30k4" if passes("nv30k4") else "nv45")
    res["chosen"] = chosen
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    f = lambda r, k, n=2: f"{r[k]:.{n}f}" if k in r else "-"
    print("| Clip | Variant | Seconds | Coverage | Lateral (m) | Speed (m/s) | Accel. (m/s^2) | Box heading (deg) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in he.TUNING + he.HELD:
        for t in VARIANTS:
            r = res[c][t]
            print(f"| {c} | {t} | {r['seconds']} | {r['coverage']:.2f} | {f(r, 'lateral_m')} | {f(r, 'speed_mps')} | {f(r, 'accel')} | {f(r, 'box_heading_deg', 1)} |")
    print("\nheld-out means over graded passes (passes graded):")
    for t in VARIANTS:
        R = [res[c][t] for c in he.HELD]
        G = [r for r in R if "speed_mps" in r]
        print(f"  {t:10s} coverage {np.mean([r['coverage'] for r in R]):.2f} | graded {len(G)} | lateral {np.mean([r['lateral_m'] for r in G]):.3f} "
              f"speed {np.mean([r['speed_mps'] for r in G]):.3f} accel {np.mean([r['accel'] for r in G]):.3f}")
    print("chosen on the tuning clips:", chosen)


if __name__ == "__main__":
    run(sys.argv[2:]) if sys.argv[1] == "run" else score()
