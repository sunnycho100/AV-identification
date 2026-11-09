"""Evaluation only: the 2026 Todd SE clips against their GPS (plan 2026-11-04, Phase 6).

Same metrics as ablation.py (per_frame, agg, 1 s paired block bootstrap), on the
image-identified red Lincoln, for the full pipeline (iv_s4) and the pretrained
baseline (iv_s0), scored on the frames both variants observe.

The 2026 GPS claims a shared clock with the video (video_time_sec). This script
checks the claim before using it:
  along_offset_m   mean along-road difference camera minus GPS at video_time_sec
  clock_offset_s   the time shift that removes it, offset / mean speed (positive:
                   the GPS position belongs to a later video time)
  drift_s          clock offset in the second half of the pass minus the first;
                   near 0 means one constant offset, not a drifting clock
Because the claimed clock turns out to be off by seconds, a different amount per
clip in position (clock_offset_s), the speed profiles were checked: where the
speed varies they match the GPS within about a second, so the file clock is
roughly right and the along-road GPS positions are what is off (15 to 20 m, and
the GPS position advances slower than its own speed). One time shift per clip is
applied: the speed-profile fit where the speed varies enough to fix it, else the
file clock. The fit uses the full pipeline's track and is applied to both
variants. Positions are compared after removing one constant offset, as in
2025, so along-road accuracy is not graded. Coverage is as in the paper: frames
this variant observes the car over frames any variant does. in_view_frames
(shifted GPS inside the usable view) is kept as a diagnostic only.

    python scripts/evaluation/new_todd_eval.py                 # iv_s0, iv_s4
    python scripts/evaluation/new_todd_eval.py iv_s0 iv_s4 lo20   # extra runs, same time shift
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import ablation as ab           # noqa: E402
import relative_trajectory as rt  # noqa: E402
from gps_to_image import load_site, load_clip, gps_road  # noqa: E402

CLIPS = ["AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]
TAGS = ["iv_s0", "iv_s4"]
OUT = ROOT / "outputs/evaluation/new_todd"
TAU_MIN_RANGE = 1.5     # m/s: speed-fit curve range needed to trust its time shift


def timing(d):
    """Zero-offset along-road error and the clock shift it implies, whole pass and halves."""
    a = d["q"][:, 0] - d["g"][:, 0]
    sp = np.hypot(*d["v"].T)
    sgn = np.sign(np.median(d["v"][:, 0]))        # +1 driving away from the camera
    shift = lambda k: float(-np.mean(a[k]) / (sgn * np.mean(sp[k])))
    n = len(a)
    h1, h2 = np.arange(n) < n // 2, np.arange(n) >= n // 2
    return {"along_offset_m": float(np.mean(a)), "clock_offset_s": shift(np.ones(n, bool)),
            "drift_s": shift(h2) - shift(h1) if n >= 20 else None,
            "across_offset_m": float(np.mean(d["q"][:, 1] - d["g"][:, 1]))}


def shifted(gps, tau):
    g = dict(gps)
    g["t"] = gps["t"] - tau          # GPS at video time t is the original GPS at t + tau
    return g


def fit_tau(d, gps, T, grid=np.arange(-8, 8.001, 0.05)):
    """Time shift that best matches the camera speed profile to the GPS speed."""
    sp = np.hypot(*d["v"].T)
    err = []
    for tau in grid:
        g = shifted(gps, tau)
        ok = (d["t"] >= g["t"][0]) & (d["t"] <= g["t"][-1])
        if ok.sum() < 10:
            err.append(np.inf)
            continue
        _, _, vg = gps_road(g, T, d["t"][ok])
        err.append(np.mean(np.abs(sp[ok] - vg)))
    i = int(np.argmin(err))
    flat = float(np.ptp([e for e in err if np.isfinite(e)])) if np.isfinite(err).any() else 0.0
    return float(grid[i]), float(err[i]), flat


def realign(d, gps, T, tau):
    g = shifted(gps, tau)
    ok = (d["t"] >= g["t"][0]) & (d["t"] <= g["t"][-1])
    d = {k: (v[ok] if isinstance(v, np.ndarray) and len(v) == len(ok) else v) for k, v in d.items()}
    d["g"], d["psi"], d["vg"] = gps_road(g, T, d["t"])
    return d, g


def in_view(clip, gps, T, x_lim=(5.0, 110.0), y_lim=(-39.0, -1.0)):
    c = load_clip(clip, "iv_s4")
    t = c["t"][(c["t"] >= gps["t"][0]) & (c["t"] <= gps["t"][-1])]
    xy, _, _ = gps_road(gps, T, t)
    return int(np.sum((xy[:, 0] >= x_lim[0]) & (xy[:, 0] <= x_lim[1]) & (xy[:, 1] >= y_lim[0]) & (xy[:, 1] <= y_lim[1])))


def main():
    tags = sys.argv[1:] or TAGS
    OUT.mkdir(parents=True, exist_ok=True)
    T = load_site()
    rng = np.random.default_rng(0)
    res = {}
    for clip in CLIPS:
        gps = load_clip(clip, "iv_s4")["gps"]
        window = int(np.sum((load_clip(clip, "iv_s4")["t"] >= gps["t"][0]) & (load_clip(clip, "iv_s4")["t"] <= gps["t"][-1])))
        data = {}
        for tag in tags:
            inst = ROOT / "outputs/trajectories" / f"{clip}_{tag}" / "instrumented.json"
            vid = json.loads(inst.read_text()).get("vehicle_id") if inst.exists() else None
            data[tag] = ab.load(clip, tag, T) if vid is not None else None
        row = {"gps_window_frames": window}
        ref = next((data[t] for t in ["iv_s4", "iv_s0"] + tags if data.get(t) is not None), None)   # S4 sets the shift when it has the car
        tau_s, tau_err, tau_flat = fit_tau(ref, gps, T) if ref is not None and len(ref["frame"]) >= 10 else (None, None, None)
        row["tau_speed_s"], row["tau_speed_mae_mps"], row["tau_curve_range_mps"] = tau_s, tau_err, tau_flat
        # Speed profiles match the GPS within about a second where the speed varies, while the
        # positions then differ by 15 to 20 m along the road: the file clock is roughly right and
        # the along-road positions are not usable. Rule: speed-fit shift where the speed varies
        # enough to fix it (curve range >= TAU_MIN_RANGE), else the file clock.
        tau = (tau_s if tau_flat is not None and tau_flat >= TAU_MIN_RANGE else 0.0) if ref is not None else None
        row["tau_used_s"] = tau
        gps_al = shifted(gps, tau) if tau is not None else gps
        row["in_view_frames"] = in_view(clip, gps_al, T)
        for tag, d in data.items():
            row[tag] = {"tracked_frames": 0 if d is None else int(len(d["frame"]))}
            if d is not None and len(d["frame"]) >= 10:
                row[tag]["timing"] = timing(d)
                data[tag], _ = realign(d, gps, T, tau)
        # coverage as in the paper: frames this variant observes the car, over frames any variant does
        obs = {t: (set() if d is None else set(d["frame"][d["observed"]])) for t, d in data.items()}
        union = set().union(*obs.values())
        for tag in data:
            row[tag]["coverage"] = round(len(obs[tag]) / max(len(union), 1), 3)
        row["union_frames"] = len(union)
        have = [t for t, d in data.items() if d is not None and len(d["frame"]) >= 10]
        common = sorted(set.intersection(*(set(data[t]["frame"][data[t]["observed"]]) for t in have))) if have else []
        row["common_frames"] = len(common)
        if len(common) >= 10:
            pf = {t: ab.per_frame(data[t], common, gps_al, T) for t in have}
            t0 = next(iter(pf.values()))["t"]
            acc_ok = np.all([np.isfinite(p["acc"]) for p in pf.values()], 0)
            samples = ab.block_samples(t0, rng) if len(t0) >= 30 else []
            full = np.arange(len(t0))
            for t in have:
                m = ab.agg(pf[t], full, acc_ok)
                boots = [ab.agg(pf[t], i, acc_ok) for i in samples]
                ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
                      for k in m} if boots else None
                row[t]["metrics_common"] = m
                row[t]["ci95"] = ci
        for t in have:
            own = sorted(set(data[t]["frame"][data[t]["observed"]]))
            if len(own) >= 10:
                p = ab.per_frame(data[t], own, gps_al, T)
                row[t]["metrics_own"] = ab.agg(p, np.arange(len(own)), np.isfinite(p["acc"]))
                row[t]["metrics_own"]["speed_ratio"] = float(np.mean(p["sp"]) / np.mean(p["vg"]))
        res[clip] = row
        s4 = row.get("iv_s4", {})
        m = s4.get("metrics_own") or {}
        tm = s4.get("timing") or {}
        print(f"{clip}: in view {row['in_view_frames']} fr, S4 tracked {s4.get('tracked_frames')} (coverage {s4.get('coverage')} of {row['union_frames']}), shift {row['tau_used_s'] if row['tau_used_s'] is None else round(row['tau_used_s'], 2)} s, "
              f"common {row['common_frames']} | lateral {m.get('across_rms_m', float('nan')):.2f} m, "
              f"speed {m.get('speed_mae_mph', float('nan')) * 0.44704:.2f} m/s, accel {m.get('accel_rms_diff', float('nan')):.2f} m/s2, "
              f"box {m.get('box_heading_deg', float('nan')):.1f} deg, speed ratio {m.get('speed_ratio', float('nan')):.3f} | clock offset {tm.get('clock_offset_s', float('nan')):+.2f} s "
              f"(drift {tm.get('drift_s') if tm.get('drift_s') is None else round(tm['drift_s'], 2)} s, along {tm.get('along_offset_m', float('nan')):+.1f} m)")
    out = OUT / ("results.json" if tags == TAGS else f"results_{'_'.join(tags)}.json")
    out.write_text(json.dumps(res, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
