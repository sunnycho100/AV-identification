"""Trial: per-clip time offset as a nuisance parameter, lever arm from the clips.

The four site T clips place the camera 75 m apart along the road when each is
fitted alone. The remaining explanation is per-clip alignment of the lab's
estimated video timeline. This trial fits one time offset dt_c per clip together
with the shared camera position, bearing and range scale, and optionally the
sideways lever arm l_y.

Caveats, stated up front:
  * dt_c is fitted on GPS positions, so it is chosen to minimise GPS error on the
    same clip. It can only shift the car along its travel direction.
  * At constant speed dt_c and the forward lever arm l_x are the same thing, so
    l_x is fixed at 0 and dt_c carries it (a 1 m arm is 0.04 s). l_y flips sign
    with drive direction and is estimable.
  * Held-out grading: shared parameters from the other clips, dt for the held-out
    clip fitted on itself. Across-road error and scale are honest held-out numbers;
    the along-road number is shape-only after a 1-parameter shift.

    .venv/bin/python scripts/evaluation/trial_time_offsets.py --out outputs/evaluation/trial_time_offsets.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_error_model import (ROOT, P_S, evaluate, gps_at, hold_out_reason, initial_theta,
                              kinematics, load_clips, predict, rms, standalone, wrap)


def shifted(c, dt):
    """Clip re-associated with GPS at t + dt. Frames whose shifted time falls outside
    the GPS rows are marked invalid instead of being clamped to the last row."""
    t = c["t"] + dt
    g, psi, v = gps_at(c["gps"], t)
    outside = (t < c["gps"]["t"][0]) | (t > c["gps"]["t"][-1])
    return {**c, "g": g, "psi": psi, "v": v, "invalid": c["invalid"] | outside}


def coverage(c, dt):
    t = c["t"][~c["invalid"]] + dt
    return float(((t >= c["gps"]["t"][0]) & (t <= c["gps"]["t"][-1])).mean())


def dt_bounds(c, limit=4.0):
    """Offsets that keep every valid frame inside the GPS rows (no clamping)."""
    t = c["t"][~c["invalid"]]
    return max(-limit, c["gps"]["t"][0] - t.min()), min(limit, c["gps"]["t"][-1] - t.max())


def theta7(Ce, Cn, beta, s, ly):
    return np.array([Ce, Cn, beta, 0.0, s, 0.0, ly])


def one(th, c, dt):
    # fixed mask so the residual vector keeps its length; dt stays within dt_bounds
    pc, pg = predict(th, shifted(c, dt))
    return (pc - pg)[~c["invalid"]].ravel()


def fit_shared_dt(clips, fit_ly, per_clip_s=False):
    """Parameters: C_e, C_n, beta, s (one, or one per clip), [l_y], dt per clip."""
    n = len(clips)
    ns = n if per_clip_s else 1
    off = 3 + ns + (1 if fit_ly else 0)
    th0 = initial_theta(clips)
    p0 = np.r_[th0[0], th0[1], th0[2], np.ones(ns), [0.0] * (1 if fit_ly else 0), np.zeros(n)]
    lo, hi = np.full(p0.size, -np.inf), np.full(p0.size, np.inf)
    lo[3:3 + ns], hi[3:3 + ns] = 0.5, 2.0
    for i, c in enumerate(clips):
        lo[off + i], hi[off + i] = dt_bounds(c)

    def unpack(p):
        ly = p[3 + ns] if fit_ly else 0.0
        return [theta7(p[0], p[1], p[2], p[3 + (i if per_clip_s else 0)], ly) for i in range(n)], p[off:]

    def resid(p):
        ths, dts = unpack(p)
        return np.concatenate([one(th, c, dt) for th, c, dt in zip(ths, clips, dts)])

    p = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0).x
    ths, dts = unpack(p)
    pinned = {c["name"]: bool(min(abs(dt - b) for b in dt_bounds(c)) < 0.02)
              for c, dt in zip(clips, dts)}
    return {"theta": ths[0], "s_per_clip": {c["name"]: float(th[P_S]) for th, c in zip(ths, clips)},
            "dt": dict(zip([c["name"] for c in clips], dts.tolist())),
            "dt_bounds": {c["name"]: list(dt_bounds(c)) for c in clips}, "pinned": pinned}


def fit_dt_only(th, c):
    """1-D offset for a held-out clip: coarse scan over offsets that keep at least
    80% of the track inside the GPS rows, then a local refinement."""
    lo, hi = dt_bounds(c)
    if hi <= lo:
        return None, "no offset keeps the track inside the GPS rows"
    grid = np.arange(lo, hi + 1e-9, 0.05)
    cost = [np.mean(np.square(one(th, c, dt))) for dt in grid]
    dt0 = float(grid[int(np.argmin(cost))])
    r = least_squares(lambda p: one(th, c, p[0]), [dt0], bounds=([lo], [hi]))
    dt = float(r.x[0])
    note = None
    if min(abs(dt - lo), abs(dt - hi)) < 0.02:
        note = f"offset pinned at the GPS coverage limit [{lo:+.2f}, {hi:+.2f}] s; the required shift exceeds the data"
    return dt, note


def dt_from_speed(c, window=31):
    """Independent check: shift that best matches the camera speed profile to the
    GPS speed profile. Only informative when speed varies within the clip."""
    dt_frame = float(np.median(np.diff(c["t"])))
    vel = savgol_filter(c["q"], window, 2, deriv=1, delta=dt_frame, axis=0)
    v_cam = np.linalg.norm(vel, axis=1)
    m = ~c["invalid"]
    g = c["gps"]
    best = None
    for dt in np.arange(-4, 4.0001, 1 / 60):
        t = c["t"][m] + dt
        if (t < g["t"][0]).mean() > 0.1 or (t > g["t"][-1]).mean() > 0.1:
            continue
        v = np.interp(t, g["t"], g["v"])
        err = rms(v_cam[m] * (v.mean() / v_cam[m].mean()) - v)
        if best is None or err < best[0]:
            best = (err, float(dt))
    v0 = np.interp(c["t"][m], g["t"], g["v"])
    return {"dt_s": best[1], "rms_at_best": best[0],
            "rms_at_zero": rms(v_cam[m] * (v0.mean() / v_cam[m].mean()) - v0),
            "gps_speed_range": [float(g["v"].min()), float(g["v"].max())]}


def partial_loco(clips, fit_ly):
    out = {}
    for c in clips:
        others = [o for o in clips if o is not c]
        why = hold_out_reason(c, others)
        if why:
            out[c["name"]] = {"skipped": why}
            continue
        th = fit_shared_dt(others, fit_ly)["theta"]
        dt, note = fit_dt_only(th, c)
        if dt is None:
            out[c["name"]] = {"not_evaluable": note}
            continue
        cs = shifted(c, dt)
        out[c["name"]] = {"dt_fitted_on_this_clip_s": dt, "note": note,
                          "s_from_others": float(th[P_S]), "l_y_from_others": float(th[6]),
                          "held_out": evaluate(th, cs, ~cs["invalid"]),
                          "fit_free": kinematics(c)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "scripts/evaluation/site_T.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    clips, _ = load_clips(cfg)
    report = {"clips": [c["name"] for c in clips],
              "T0_standalone": {c["name"]: standalone(c, cfg["lat0"], cfg["lon0"]) for c in clips},
              "speed_profile_dt": {c["name"]: dt_from_speed(c) for c in clips}}
    for tag, fit_ly, pcs in (("T1_shared_plus_dt", False, False),
                             ("T2_shared_plus_dt_plus_ly", True, False),
                             ("T3_per_clip_s_plus_dt_plus_ly", True, True)):
        j = fit_shared_dt(clips, fit_ly, pcs)
        ins = {}
        for c in clips:
            th = j["theta"].copy()
            th[P_S] = j["s_per_clip"][c["name"]]
            cs = shifted(c, j["dt"][c["name"]])
            ins[c["name"]] = evaluate(th, cs, ~cs["invalid"])
        report[tag] = {"dt_s": j["dt"], "dt_bounds_s": j["dt_bounds"], "dt_pinned": j["pinned"],
                       "theta": {"C_e": j["theta"][0], "C_n": j["theta"][1],
                                 "beta_deg": float(np.degrees(wrap(j["theta"][2]))),
                                 "s": j["s_per_clip"], "l_y": j["theta"][6]},
                       "in_sample": ins,
                       "partial_leave_one_out": None if pcs else partial_loco(clips, fit_ly)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
