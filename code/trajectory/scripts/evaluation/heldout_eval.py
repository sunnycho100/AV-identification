"""Held-out evaluation, one row per pass (decisions D23, D24).

Variants on every Todd SE clip with GPS: the pretrained detector (S0), the external
baseline (COCO Faster R-CNN 2D boxes on the ground plane, run through the same road
filter, tracker, stitching and smoother as S4), the full pipeline (S4) and S4 with
low-score continuation. Each variant is graded on its own frames, coverage over the
frames any variant tracks. 2025 clips use the lab timing of the ablation; 2026 clips
the D14 time shift (taken from the S4 track, new_todd_eval results). Tuning clips are
listed but kept out of the summary, whose intervals come from resampling passes.

    python scripts/evaluation/heldout_eval.py run      # 2D-baseline trajectories and car marks
    python scripts/evaluation/heldout_eval.py score [retrained] [fileclock]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import ablation as ab           # noqa: E402
import lowscore_sweep as ls     # noqa: E402
import new_todd_eval as nt      # noqa: E402

TUNING = ["HV_T_EW_1", "AV_T_WE_1"]
HELD = ["AV_T_EW_3", "AV_T_WE_3", "AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]
Y2025 = {"HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"}
VARIANTS = ["S0 pretrained", "2D + ground plane", "S4", "S4 + low-score"]
OUT = ROOT / "outputs/evaluation/heldout"
N_BOOT = 10000


TAGS = {"S0 pretrained": ("abl_s0", "iv_s0"), "2D + ground plane": ("d2", "d2"), "S4": ("abl_s4", "iv_s4"),
        "S4 + low-score": ("lo30k4", "lo30k4")}                     # (2025 tag, 2026 tag)
RETRAINED = {"S4": ("nv45", "nv45"), "S4 + low-score": ("nv30k4", "nv30k4")}   # head trained without the test clip (D27)
FILE_CLOCK = False      # True: 2026 GPS on the file clock (no fitted shift), the primary timing from 2026-11-07


def tag(clip, variant):
    return TAGS[variant][0 if clip in Y2025 else 1]


def taus():
    if FILE_CLOCK:
        return {}
    return {c: (v.get("tau_used_s") or 0.0) for c, v in
            json.loads((ROOT / "outputs/evaluation/new_todd/results.json").read_text()).items()}


def configure(argv):
    """score [retrained] [fileclock]: switch the S4 tags and the timing; returns the output file name."""
    global FILE_CLOCK
    if "retrained" in argv:
        TAGS.update(RETRAINED)
    FILE_CLOCK = "fileclock" in argv
    return "results" + "".join(f"_{a}" for a in ("retrained", "fileclock") if a in argv) + ".json"


def run():
    for clip in TUNING + HELD:
        if not (ls.TRAJ / f"{clip}_d2" / "tracks.json").exists():
            ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", "d2", "--det-tag", "det2d",
                   "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1])
        print(clip, "d2 car", ls.mark(clip, "d2"))


def score(name="results.json"):
    from gps_to_image import load_clip, load_site
    T = load_site()
    tau = taus()
    rows, pairs = [], []
    for clip in TUNING + HELD:
        gps = load_clip(clip, tag(clip, "S4"))["gps"]
        g = gps if clip in Y2025 else nt.shifted(gps, tau.get(clip, 0.0))
        data = {}
        for v in VARIANTS:
            inst = ls.TRAJ / f"{clip}_{tag(clip, v)}" / "instrumented.json"
            if not inst.exists() or json.loads(inst.read_text()).get("vehicle_id") is None:
                data[v] = None
                continue
            d = ab.load(clip, tag(clip, v), T)
            if d is not None and clip not in Y2025:
                d, _ = nt.realign(d, gps, T, tau.get(clip, 0.0))
            data[v] = d
        obs = {v: set() if d is None else set(d["frame"][d["observed"]].tolist()) for v, d in data.items()}
        union = set().union(*obs.values())
        for v in ("S4", "S4 + low-score"):              # same frames as the 2D baseline, so range is matched
            both = sorted(obs[v] & obs["2D + ground plane"])
            if clip in HELD and len(both) >= 10:
                for w in (v, "2D + ground plane"):
                    p = ab.per_frame(data[w], both, g, T)
                    m = ab.agg(p, np.arange(len(both)), np.isfinite(p["acc"]))
                    pairs.append({"clip": clip, "pair": f"{v} vs 2D", "variant": w, "frames": len(both),
                                  "lateral_m": m["across_rms_m"], "speed_mps": m["speed_mae_mph"] * 0.44704, "accel": m["accel_rms_diff"]})
        for v, d in data.items():
            row = {"clip": clip, "role": "tuning" if clip in TUNING else "held out", "variant": v,
                   "seconds": round(len(obs[v]) / 30, 1), "coverage": round(len(obs[v]) / max(len(union), 1), 3)}
            if d is not None and len(obs[v]) >= 10:
                own = sorted(obs[v])
                p = ab.per_frame(d, own, g, T)
                m = ab.agg(p, np.arange(len(own)), np.isfinite(p["acc"]))
                row.update(lateral_m=m["across_rms_m"], speed_mps=m["speed_mae_mph"] * 0.44704, accel=m["accel_rms_diff"],
                           box_heading_deg=None if v == "2D + ground plane" else m["box_heading_deg"])   # 2D boxes: fixed yaw
            rows.append(row)
    rng = np.random.default_rng(0)

    def summarise(passes):
        out = {}
        for v in VARIANTS:
            for k in ("coverage", "lateral_m", "speed_mps", "accel"):
                x = np.array([r[k] for r in rows if r["variant"] == v and r["clip"] in passes and k in r])
                if len(x):
                    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
                    out[f"{v}|{k}"] = {"passes": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                                       "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        return out
    graded = lambda c: all("speed_mps" in r for r in rows if r["clip"] == c)
    res = {"rows": rows, "pairs_same_frames": pairs, "summary_all_held_out": summarise(HELD),
           "common_passes": [c for c in HELD if graded(c)], "summary_common": summarise([c for c in HELD if graded(c)])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(res, indent=1))
    f = lambda r, k, n=2: f"{r[k]:.{n}f}" if r.get(k) is not None else "-"
    print("| Clip | Role | Variant | Seconds | Coverage | Lateral (m) | Speed (m/s) | Accel. (m/s^2) | Box heading (deg) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['clip']} | {r['role']} | {r['variant']} | {r['seconds']} | {r['coverage']:.2f} | {f(r, 'lateral_m')} | "
              f"{f(r, 'speed_mps')} | {f(r, 'accel')} | {f(r, 'box_heading_deg', 1)} |")
    for part in ("summary_all_held_out", "summary_common"):
        print(f"\n{part} (passes: {HELD if part.endswith('out') else res['common_passes']})")
        for k, s in res[part].items():
            print(f"  {k}: mean {s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}], median {s['median']:.3f}, n {s['passes']}")
    print("\nSame frames as the 2D baseline (mean over passes, per pass in results.json)")
    for pair in ("S4 vs 2D", "S4 + low-score vs 2D"):
        for v in (pair.split(" vs ")[0], "2D + ground plane"):
            x = [r for r in pairs if r["pair"] == pair and r["variant"] == v]
            print(f"  {pair:22s} {v:18s} passes {len(x)}, frames {sum(r['frames'] for r in x)}: lateral {np.mean([r['lateral_m'] for r in x]):.3f} m, "
                  f"speed {np.mean([r['speed_mps'] for r in x]):.3f} m/s, accel {np.mean([r['accel'] for r in x]):.3f} m/s^2")
    print("->", OUT / name)


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score(configure(sys.argv[2:]))
