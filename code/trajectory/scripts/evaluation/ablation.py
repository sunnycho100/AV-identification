"""Evaluation only: how much each pipeline component improves the lab car's trajectory.

Additive steps (each adds one component to the one before):
  S0 pretrained BEVHeight + tracking   raw detections, speed by central difference
  S1 + feature mask                    (mask at inference, 2.5 m)
  S2 + fine-tune on stabilized boxes   (ft102 head, same mask)
  S3 + jump removal only               (postprocess_tracks.remove_jumps)
  S4 + light smoothing                 (RTS at the q_vel picked on the tuning clips)
Leave-one-out from S4: no mask, no fine-tune, no jump removal (no smoothing = S3).
Reference: the pipeline as it was before (tracker output, old jump split, q_vel 0.1).

Metrics, fixed before any run (2026-10-11), lab car vs its GPS, lab timing, one
constant offset removed (relative_trajectory.compare), on the frames every main
variant covers:
  across_rms_m       across-road position after the offset is removed
  speed_mae_mph      speed error
  heading_mae_deg    direction of travel vs GPS heading
  box_heading_deg    the box's own heading (yaw) vs GPS heading, 0-180 deg
  accel_rms_diff     acceleration difference vs GPS (0.5 s central difference):
                     too little smoothing adds noise, too much erases real changes
  speed_std_ratio    camera speed std / GPS speed std (1 = variation kept;
                     >1 noise, <1 flattened); HV_T_EW_1 is the HV clip with GPS
  coverage           share of GPS-window frames where the car has a state
with 95% intervals from a moving-block bootstrap (1 s blocks). False objects per
detector come from false_objects.py (all 5 clips).

Clip roles: HV_T_EW_1 and AV_T_WE_1 tuned the calibration, the fine-tune labels and
the smoothing (tuning); AV_T_WE_3 was not trained or tuned on, but the mask height
and road filter used all clips (held out, partly); AV_T_EW_3 is the test clip.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/ablation.py run --variants q_sweep
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/ablation.py pick-q
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/ablation.py run --variants main --q <picked>
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/ablation.py score
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import relative_trajectory as rt   # noqa: E402
import false_objects as fo         # noqa: E402

CLIPS = rt.CLIPS
ROLE = {"HV_T_EW_1": "tuning", "AV_T_WE_1": "tuning", "AV_T_WE_3": "held out (partly)", "AV_T_EW_3": "test"}
TUNING = ["HV_T_EW_1", "AV_T_WE_1"]
Q_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
Q_TIE = 0.10                 # lighter smoothing wins when within 10% of the best
BLOCK, N_BOOT = 30, 1000
OUT = ROOT / "outputs/evaluation/ablation"


def variants(q):
    """name -> (label, det_tag, jumps, smooth, q_vel)."""
    return {
        "abl_s0": ("S0 pretrained + tracking", "base102_h151", "none", "none", None),
        "abl_s1": ("S1 + feature mask", "basefeat25", "none", "none", None),
        "abl_s2": ("S2 + fine-tune", "ft102feat25", "none", "none", None),
        "abl_s3": ("S3 + jump removal", "ft102feat25", "remove", "none", None),
        "abl_s4": ("S4 + light smoothing", "ft102feat25", "remove", "rts", q),
        "abl_no_mask": ("S4 without feature mask", "ft102", "remove", "rts", q),
        "abl_no_ft": ("S4 without fine-tune", "basefeat25", "remove", "rts", q),
        "abl_no_jump": ("S4 without jump removal", "ft102feat25", "none", "rts", q),
    }


MAIN = ["abl_s0", "abl_s1", "abl_s2", "abl_s3", "abl_s4"]
LOO = ["abl_no_mask", "abl_no_ft", "abl_no_jump"]
REF = {"ft102": ("Previous pipeline (tracker output, q 0.1)", "ft102")}


def q_variants():
    return {f"abl_q{q:g}": (f"q_vel {q:g}", "ft102feat25", "remove", "rts", q) for q in Q_GRID}


def run_variant(name, v, clips):
    _, det, jumps, smooth, q = v
    py = sys.executable
    for clip in clips:
        cmd = [py, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", name, "--det-tag", det,
               "--jumps", jumps, "--smooth", smooth] + (["--q-vel", str(q)] if q is not None else [])
        mark = [py, "scripts/pipeline/mark_instrumented.py", "--clip", clip, "--tag", name]
        for c in [cmd, mark] if clip in CLIPS else [cmd]:    # HV_T_EW_2: no car to mark, false objects only
            r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, NUMBA_DISABLE_JIT="1"))
            if r.returncode:
                raise SystemExit(f"{name} {clip} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        print(f"{name} {clip}: {r.stdout.strip().splitlines()[-1]}")


ACC_HALF_S, ACC_TOL_S = 0.25, 0.05     # acceleration over +-0.25 s, endpoints within 0.05 s of that
SAME_CAR_M = 2.0


def accel_at(t_all, sp_all, t):
    """Speed change over [t - 0.25 s, t + 0.25 s] from the variant's own states, NaN
    when no state lies within ACC_TOL_S of either end (a gap)."""
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        lo, hi = np.argmin(np.abs(t_all - (ti - ACC_HALF_S))), np.argmin(np.abs(t_all - (ti + ACC_HALF_S)))
        if abs(t_all[lo] - (ti - ACC_HALF_S)) <= ACC_TOL_S and abs(t_all[hi] - (ti + ACC_HALF_S)) <= ACC_TOL_S:
            out[i] = (sp_all[hi] - sp_all[lo]) / (t_all[hi] - t_all[lo])
    return out


def per_frame(d, frames, gps, T):
    """Per-frame errors of one variant on the given frames (offset removed once)."""
    from site_error_model import wrap
    from gps_to_image import gps_road
    k = np.isin(d["frame"], frames)
    q, v, g, psi, vg, t, yaw = (d[x][k] for x in ("q", "v", "g", "psi", "vg", "t", "yaw"))
    r = (q - g) - (q - g).mean(0)
    sp = np.hypot(v[:, 0], v[:, 1])
    _, _, g_lo = gps_road(gps, T, t - ACC_HALF_S)
    _, _, g_hi = gps_road(gps, T, t + ACC_HALF_S)
    sp_all = np.hypot(d["v"][:, 0], d["v"][:, 1])
    return {"across": r[:, 1], "speed": np.abs(sp - vg) * rt.MPH,
            "dir": np.degrees(np.abs(wrap(np.arctan2(v[:, 1], v[:, 0]) - psi))),
            "box": np.degrees(np.abs(wrap(yaw - psi))),
            "acc": accel_at(d["t"], sp_all, t) - (g_hi - g_lo) / (2 * ACC_HALF_S),
            "sp": sp, "vg": vg, "t": t, "q": q}


def agg(pf, idx, acc_ok):
    """The fixed metrics over frame indices idx (may repeat, from the bootstrap)."""
    ia = idx[acc_ok[idx]]
    return {"across_rms_m": float(np.sqrt(np.mean(pf["across"][idx] ** 2))),
            "speed_mae_mph": float(np.mean(pf["speed"][idx])),
            "heading_mae_deg": float(np.mean(pf["dir"][idx])),
            "box_heading_deg": float(np.mean(pf["box"][idx])),
            "accel_rms_diff": float(np.sqrt(np.mean(pf["acc"][ia] ** 2))) if len(ia) else float("nan"),
            "speed_std_ratio": float(np.std(pf["sp"][idx]) / max(np.std(pf["vg"][idx]), 1e-6))}


def block_samples(t, rng, n_boot=N_BOOT, block_s=1.0):
    """Paired circular block bootstrap in time: 1 s blocks of the common frames
    (wrapping past the end, so the last second is sampled as often as the rest),
    concatenated to the original length. The same samples serve every variant."""
    n, span = len(t), t[-1] - t[0] + np.median(np.diff(t))
    out = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s0 = rng.integers(n)
            rel = (t - t[s0]) % span
            blk = np.flatnonzero(rel < block_s)
            idx.extend(blk[np.argsort(rel[blk])])
        out.append(np.array(idx[:n]))
    return out


def load(clip, tag, T):
    """relative_trajectory.load plus the box yaw and whether each state was observed."""
    d = rt.load(clip, tag, T)
    if d is None:
        return None
    tr = json.loads((ROOT / "outputs/trajectories" / f"{clip}_{tag}" / "tracks.json").read_text())["tracks"][d["id"]]
    by = {s["frame"]: s for s in tr}
    d["yaw"] = np.array([by[f]["yaw"] for f in d["frame"]])
    d["observed"] = np.array(["x_det" in by[f] for f in d["frame"]])
    return d


def score_set(tags, clips, T, deltas=()):
    """Metrics per clip and variant on the frames all variants observe, with paired
    95% intervals; deltas = (a, b) pairs reported as a - b with their interval."""
    from gps_to_image import load_clip
    rng = np.random.default_rng(0)
    out = {}
    for clip in clips:
        data = {tag: load(clip, tag, T) for tag in tags}
        if any(v is None for v in data.values()):
            continue
        obs = {tag: set(d["frame"][d["observed"]].tolist()) for tag, d in data.items()}
        frames = sorted(set.intersection(*obs.values()))
        union = set.union(*obs.values())
        gps = load_clip(clip, tags[0])["gps"]
        pf = {tag: per_frame(d, frames, gps, T) for tag, d in data.items()}
        acc_ok = np.all([np.isfinite(p["acc"]) for p in pf.values()], axis=0)
        ref_q = next(iter(pf.values()))["q"]
        same = {tag: float(np.median(np.hypot(*(p["q"] - ref_q).T))) for tag, p in pf.items()}
        if max(same.values()) > SAME_CAR_M:
            raise SystemExit(f"{clip}: variants picked different cars {same}")
        t = next(iter(pf.values()))["t"]
        samples = block_samples(t, rng) if len(t) >= 30 else []
        full = np.arange(len(t))
        boots = {tag: [agg(p, i, acc_ok) for i in samples] for tag, p in pf.items()}
        ci = lambda vals: [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))] if len(vals) else None
        for tag, p in pf.items():
            m = agg(p, full, acc_ok)
            out[f"{clip}|{tag}"] = {"clip": clip, "role": ROLE[clip], "tag": tag, "car": data[tag]["id"],
                                    "frames": len(t), "accel_frames": int(acc_ok.sum()),
                                    "coverage": round(len(obs[tag]) / max(len(union), 1), 3),
                                    "same_car_median_m": round(same[tag], 3), "metrics": m,
                                    "ci95": {k: ci([b[k] for b in boots[tag]]) for k in m}}
        for a_, b_ in deltas:
            ma, mb = agg(pf[a_], full, acc_ok), agg(pf[b_], full, acc_ok)
            out[f"{clip}|{a_}-{b_}"] = {"clip": clip, "a": a_, "b": b_,
                                        "delta": {k: ma[k] - mb[k] for k in ma},
                                        "ci95": {k: ci([x[k] - y[k] for x, y in zip(boots[a_], boots[b_])]) for k in ma}}
    return out


def pick_q(T):
    tags = list(q_variants())
    res = score_set(tags, TUNING, T)
    mean = {tag: float(np.mean([res[f"{c}|{tag}"]["metrics"]["accel_rms_diff"] for c in TUNING])) for tag in tags}
    best = min(mean.values())
    ok = [tag for tag in tags if mean[tag] <= best * (1 + Q_TIE)]
    pick = max(ok, key=lambda tag: q_variants()[tag][4])
    rep = {"criterion": f"mean accel_rms_diff over {TUNING}; lightest q within {Q_TIE:.0%} of the best",
           "mean_accel_rms_diff": mean, "per_clip": res, "picked": pick, "q_vel": q_variants()[pick][4]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "q_pick.json").write_text(json.dumps(rep, indent=2))
    for tag in tags:
        sr = [res[f"{c}|{tag}"]["metrics"]["speed_std_ratio"] for c in TUNING]
        print(f"{tag:10s} accel diff {mean[tag]:.3f}   speed std ratio HV_T_EW_1 {sr[0]:.2f}  AV_T_WE_1 {sr[1]:.2f}")
    print(f"picked {pick} (q_vel {rep['q_vel']})")
    return rep["q_vel"]


V_IMPOSSIBLE = 45.0          # m/s (100 mph), the pipeline's own V_MAX


def impossible(tag):
    """Supplementary (added after the first results, because the lab car never jumps):
    over every vehicle in all 5 clips, states faster than V_IMPOSSIBLE."""
    n = bad = 0
    for clip in fo.CLIPS:
        for t in json.loads((ROOT / "outputs/trajectories" / f"{clip}_{tag}" / "tracks.json").read_text())["tracks"].values():
            n += len(t)
            bad += sum(s["speed_mps"] > V_IMPOSSIBLE for s in t)
    return {"states": n, "over_100_mph": bad, "share": round(bad / max(n, 1), 4)}


def score(T):
    q = json.loads((OUT / "q_pick.json").read_text())["q_vel"]
    v = variants(q)
    res = {"q_vel": q, "labels": {k: x[0] for k, x in v.items()} | {k: x[0] for k, x in REF.items()},
           "main": score_set(MAIN, CLIPS, T, deltas=list(zip(MAIN[1:], MAIN[:-1]))),
           "loo": score_set(["abl_s4"] + LOO, CLIPS, T, deltas=[(x, "abl_s4") for x in LOO]),
           "reference": score_set(["abl_s4", "ft102"], CLIPS, T, deltas=[("ft102", "abl_s4")])}
    res["false_objects"] = {}
    for tag in dict.fromkeys(x[1] for x in v.values()):
        pts = {c: fo.false_points(c, tag) for c in fo.CLIPS}
        res["false_objects"][tag] = {"objects": sum(fo.objects(p) if p else 0 for p in pts.values()),
                                     "boxes": sum(len(p) for p in pts.values()),
                                     "per_clip": {c: fo.objects(p) if p else 0 for c, p in pts.items()}}
    res["impossible_states"] = {tag: impossible(tag) for tag in MAIN + LOO + list(REF)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(table(res))
    (OUT / "table.md").write_text(table(res) + "\n")


COLS = [("across_rms_m", "Across (m)", 2), ("speed_mae_mph", "Speed err (mph)", 2),
        ("heading_mae_deg", "Travel dir (deg)", 2), ("box_heading_deg", "Box heading (deg)", 1),
        ("accel_rms_diff", "Accel diff (m/s2)", 2), ("speed_std_ratio", "Speed var ratio", 2)]


def table(res):
    fmt = lambda m, ci, p: f"{m:.{p}f}" + (f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]" if ci else "")
    lines = []
    for part, title in (("main", "Additive steps"), ("loo", "Leave one out (from S4)"), ("reference", "Reference")):
        rows = [r for r in res[part].values() if "tag" in r]
        dl = [r for r in res[part].values() if "a" in r]
        lines += [f"\n### {title}\n", "| Clip | Role | Step | Frames | Coverage | " + " | ".join(c[1] for c in COLS) + " |",
                  "|" + "---|" * (5 + len(COLS))]
        for r in rows:
            lines.append(f"| {r['clip']} | {r['role']} | {res['labels'][r['tag']]} | {r['frames']} | {r['coverage']:.2f} | "
                         + " | ".join(fmt(r["metrics"][k], r["ci95"].get(k), p) for k, _, p in COLS) + " |")
        lines += [f"\n{title}: change and 95% interval (negative = the first one is lower)\n",
                  "| Clip | Change | " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * (2 + len(COLS))]
        for r in dl:
            lines.append(f"| {r['clip']} | {res['labels'][r['a']]} minus {res['labels'][r['b']]} | "
                         + " | ".join(fmt(r["delta"][k], r["ci95"].get(k), p) for k, _, p in COLS) + " |")
    lines += ["\n### False objects (all 5 clips, per detector)\n", "| Detector | Objects | Boxes |", "|---|---|---|"]
    lines += [f"| {k} | {v['objects']} | {v['boxes']} |" for k, v in res["false_objects"].items()]
    lines += ["\n### Impossible states, all vehicles in all 5 clips (supplementary, added after the first results)\n",
              "| Step | States | Over 100 mph | Share |", "|---|---|---|---|"]
    lines += [f"| {res['labels'][k]} | {v['states']} | {v['over_100_mph']} | {v['share']:.2%} |"
              for k, v in res["impossible_states"].items()]
    return "\n".join(lines)


def _selfcheck():
    """Perfect track: zero errors, speed variation ratio 1; noisy speed: ratio > 1;
    acceleration refuses windows across a gap; paired blocks keep time order."""
    t = np.arange(120) / 30
    sp = 27 + 0.5 * t
    a = accel_at(t, sp, t)
    assert np.allclose(a[6:-6], 0.5) and np.isnan(a[:6]).all(), a[:10]
    tg = np.delete(t, range(50, 70))                       # a 0.67 s gap
    ag = accel_at(tg, 27 + 0.5 * tg, tg)
    assert np.isnan(ag[(tg > 1.45) & (tg < 2.5)]).all() and np.allclose(ag[10:40], 0.5)
    pf = {"across": np.zeros(120), "speed": np.zeros(120), "dir": np.zeros(120), "box": np.full(120, 0.5),
          "acc": np.zeros(120), "sp": sp, "vg": sp}
    m = agg(pf, np.arange(120), np.ones(120, bool))
    assert m["across_rms_m"] == 0 and m["speed_std_ratio"] == 1 and m["box_heading_deg"] == 0.5, m
    noisy = dict(pf, sp=sp + np.random.default_rng(0).normal(0, 3, 120))
    assert agg(noisy, np.arange(120), np.ones(120, bool))["speed_std_ratio"] > 3
    bs = block_samples(t, np.random.default_rng(0), n_boot=5)
    assert all(len(b) == 120 and np.all(np.isin(np.diff(b[:30]), [1, -119])) for b in bs)
    print("selfcheck ok")


def main():
    from gps_to_image import load_site
    if "--selfcheck" in sys.argv:
        return _selfcheck()
    ap = argparse.ArgumentParser("Pipeline ablation against GPS")
    ap.add_argument("cmd", choices=["run", "pick-q", "score"])
    ap.add_argument("--variants", default="main", choices=["main", "q_sweep"])
    ap.add_argument("--q", type=float)
    a = ap.parse_args()
    if a.cmd == "run":
        vs = q_variants() if a.variants == "q_sweep" else variants(a.q)
        clips = TUNING if a.variants == "q_sweep" else list(dict.fromkeys(CLIPS + fo.CLIPS))
        for name, v in vs.items():
            run_variant(name, v, clips)
    elif a.cmd == "pick-q":
        pick_q(load_site())
    else:
        score(load_site())


if __name__ == "__main__":
    main()
