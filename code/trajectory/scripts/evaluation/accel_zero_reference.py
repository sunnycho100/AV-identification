"""Zero-acceleration reference for the acceleration error (2026-11-07).

On the frames where the 2D baseline and S4 with low-score continuation both see the car,
compares each one's acceleration error with the error of always predicting zero (the
rms of the GPS acceleration itself). Same 0.5 s window as ablation.accel_at.

    python scripts/evaluation/accel_zero_reference.py
"""
import json, sys
import numpy as np
sys.path[:0] = ["scripts/evaluation", "."]
import heldout_eval as he, ablation as ab, new_todd_eval as nt
from gps_to_image import load_clip, load_site, gps_road
T = load_site()
tau = {c: (v.get("tau_used_s") or 0.0) for c, v in json.load(open("outputs/evaluation/new_todd/results.json")).items()}
tot = {"zero": [], "2D": [], "3D": []}
for clip in he.HELD:
    gps = load_clip(clip, he.tag(clip, "S4"))["gps"]
    g = gps if clip in he.Y2025 else nt.shifted(gps, tau.get(clip, 0.0))
    d = {}
    for v in ("2D + ground plane", "S4 + low-score"):
        x = ab.load(clip, he.tag(clip, v), T)
        if clip not in he.Y2025: x, _ = nt.realign(x, gps, T, tau.get(clip, 0.0))
        d[v] = x
    both = sorted(set(d["2D + ground plane"]["frame"][d["2D + ground plane"]["observed"]]) & set(d["S4 + low-score"]["frame"][d["S4 + low-score"]["observed"]]))
    if len(both) < 10: continue
    p2 = ab.per_frame(d["2D + ground plane"], both, g, T); p3 = ab.per_frame(d["S4 + low-score"], both, g, T)
    t = p3["t"]; _, _, lo = gps_road(g, T, t - 0.25); _, _, hi = gps_road(g, T, t + 0.25); ag = (hi - lo) / 0.5
    ok = np.isfinite(p2["acc"]) & np.isfinite(p3["acc"])
    r = lambda x: float(np.sqrt(np.mean(x[ok] ** 2)))
    print(f"{clip}: GPS accel rms (always-zero error) {r(ag):.2f} | 2D error {r(p2['acc']):.2f} | 3D error {r(p3['acc']):.2f} m/s^2 | GPS accel range {ag[ok].min():+.2f} to {ag[ok].max():+.2f}")
    for k, x in (("zero", ag), ("2D", p2["acc"]), ("3D", p3["acc"])): tot[k].append(r(x))
print({k: round(float(np.mean(v)), 2) for k, v in tot.items()})
