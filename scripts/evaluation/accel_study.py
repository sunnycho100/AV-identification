"""Acceleration evaluation (decision D25): skill against always-zero, events, CA smoother.

skill = 1 - rms error / rms of GPS acceleration, where the rms of GPS acceleration is
the error of always predicting zero. Event frames have GPS acceleration magnitude at
least EVENT (0.3 m/s^2). Acceleration is the 0.5 s speed difference of ablation.accel_at
for every variant, so only the trajectory changes. Smoothers: the current constant
velocity RTS (q_vel 0.1) and a constant-acceleration RTS with jerk noise in Q_JERK, on
3D (S4 with low-score continuation) and 2D (Faster R-CNN plus ground plane). q_jerk is
chosen per detector on the tuning clips; the held-out passes only report.

    python scripts/evaluation/accel_study.py run CLIP [CLIP ...]
    python scripts/evaluation/accel_study.py score
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

EVENT = 0.3
Q_JERK = [0.1, 1.0, 10.0]
DET = {"3D": ("lo30k4", "ft102feat25_lo10", 0.30, 4), "2D": ("d2", "det2d", 0.45, 1)}   # CV tag, det tag, L, k
OUT = ROOT / "outputs/evaluation/accel"


def tags(det):
    cv = DET[det][0]
    return [cv] + [f"{cv}ca{q:g}" for q in Q_JERK]


def run(clips):
    for clip in clips:
        for det, (cv, det_tag, L, k) in DET.items():
            for q in Q_JERK:
                t = f"{cv}ca{q:g}"
                if not (ls.TRAJ / f"{clip}_{t}" / "tracks.json").exists():
                    ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", t, "--det-tag", det_tag,
                           "--track-low-thresh", L, "--low-noise", k, "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1, "--q-jerk", q])
                print(clip, t, "car", ls.mark(clip, t))


def accel_metrics(d, frames, g, T):
    """Error, always-zero error, skill, and event metrics on the given frames."""
    from gps_to_image import gps_road
    p = ab.per_frame(d, frames, g, T)
    t = p["t"]
    _, _, lo = gps_road(g, T, t - ab.ACC_HALF_S)
    _, _, hi = gps_road(g, T, t + ab.ACC_HALF_S)
    a_g = (hi - lo) / (2 * ab.ACC_HALF_S)
    a_c = p["acc"] + a_g
    ok = np.isfinite(p["acc"])
    rms = lambda x, m: float(np.sqrt(np.mean(x[m] ** 2))) if m.sum() else float("nan")
    out = {"frames": int(ok.sum()), "err": rms(p["acc"], ok), "zero": rms(a_g, ok), "speed_mps": float(np.mean(p["speed"])) * 0.44704}
    out["skill"] = 1 - out["err"] / out["zero"]
    ev = ok & (np.abs(a_g) >= EVENT)
    out["event_frames"] = int(ev.sum())
    if ev.sum() >= 10:
        out.update(event_err=rms(p["acc"], ev), event_zero=rms(a_g, ev))
        out["event_skill"] = 1 - out["event_err"] / out["event_zero"]
        i = np.flatnonzero(ev)
        j = i[np.argmax(np.abs(a_g[i]))]                               # strongest GPS event
        out["peak_gps"], out["peak_cam"] = float(a_g[j]), float(a_c[j])
        tt, ac, ag = t[ok], a_c[ok], a_g[ok]                           # lag: camera shifted by tau best matches GPS
        lags = np.arange(-1.0, 1.0001, 1 / 30)
        cc = [np.corrcoef(np.interp(tt + L, tt, ac), ag)[0, 1] for L in lags]
        out["lag_s"] = float(lags[int(np.nanargmax(cc))])
    return out


def score(name="results.json"):
    from gps_to_image import load_clip, load_site
    T = load_site()
    tau = he.taus()
    res = {"own": {}, "same_frames": {}, "q_vel_recheck": {}}
    for clip in he.TUNING + he.HELD:
        gps = load_clip(clip, he.tag(clip, "S4"))["gps"]
        g = gps if clip in he.Y2025 else nt.shifted(gps, tau.get(clip, 0.0))
        data = {}
        for det in DET:
            for t in tags(det):
                inst = ls.TRAJ / f"{clip}_{t}" / "instrumented.json"
                if not inst.exists() or json.loads(inst.read_text()).get("vehicle_id") is None:
                    continue
                d = ab.load(clip, t, T)
                if d is not None and clip not in he.Y2025:
                    d, _ = nt.realign(d, gps, T, tau.get(clip, 0.0))
                if d is not None:
                    data[t] = d
        obs = {t: sorted(set(d["frame"][d["observed"]].tolist())) for t, d in data.items()}
        for t, d in data.items():
            if len(obs[t]) >= 10:
                res["own"][f"{clip}|{t}"] = accel_metrics(d, obs[t], g, T)
        for i, (t3, t2) in enumerate(zip(tags("3D"), tags("2D"))):        # same smoother setting, same frames
            if t3 in obs and t2 in obs:
                both = sorted(set(obs[t3]) & set(obs[t2]))
                if len(both) >= 10:
                    res["same_frames"][f"{clip}|{t3}"] = accel_metrics(data[t3], both, g, T)
                    res["same_frames"][f"{clip}|{t2}"] = accel_metrics(data[t2], both, g, T)
        if clip in he.TUNING and DET["3D"][0] == "lo30k4":               # existing constant-velocity q sweep (original head)
            for q in ab.Q_GRID:
                t = f"abl_q{q:g}"
                d = ab.load(clip, t, T)
                if d is not None:
                    fr = sorted(set(d["frame"][d["observed"]].tolist()))
                    res["q_vel_recheck"][f"{clip}|{q:g}"] = accel_metrics(d, fr, g, T)
    # choice of q_jerk per detector on the tuning clips (rule fixed in D25)
    choice = {}
    for det in DET:
        cv = DET[det][0]
        best, best_skill = cv, np.mean([res["own"][f"{c}|{cv}"]["skill"] for c in he.TUNING])
        for t in tags(det)[1:]:
            ok = all(f"{c}|{t}" in res["own"] for c in he.TUNING)
            if not ok:
                continue
            sk = np.mean([res["own"][f"{c}|{t}"]["skill"] for c in he.TUNING])
            sp_ok = all(res["own"][f"{c}|{t}"]["speed_mps"] <= 1.1 * res["own"][f"{c}|{cv}"]["speed_mps"] for c in he.TUNING)
            if sp_ok and sk > best_skill:
                best, best_skill = t, sk
        choice[det] = {"chosen": best, "tuning_skill": float(best_skill)}
    res["choice"] = choice
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(res, indent=1))
    print("choice on tuning clips:", choice)
    print("\nq_vel re-check (tuning clips, constant velocity): skill, err, zero")
    for k, m in res["q_vel_recheck"].items():
        print(f"  {k}: skill {m['skill']:+.2f}  err {m['err']:.2f}  zero {m['zero']:.2f}  speed {m['speed_mps']:.2f}")
    print("\nheld out, same frames per detector pair: skill | event skill | peak gps/cam | lag")
    for k, m in res["same_frames"].items():
        if k.split("|")[0] in he.HELD:
            ev = (f"{m['event_skill']:+.2f} ({m['event_frames']} fr) | {m['peak_gps']:+.2f}/{m['peak_cam']:+.2f} | {m['lag_s']:+.2f} s"
                  if "event_skill" in m else "-")
            print(f"  {k:24s} skill {m['skill']:+.2f}  err {m['err']:.2f}  zero {m['zero']:.2f} | {ev}")
    print("->", OUT / name)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:                                       # score [retrained] [fileclock]
        name = he.configure(sys.argv[2:])
        if "retrained" in sys.argv:
            DET["3D"] = ("nv30k4", "nvfeat25_lo10", 0.30, 4)
            Q_JERK.clear()                      # constant acceleration was rejected (D25)
        score(name)
