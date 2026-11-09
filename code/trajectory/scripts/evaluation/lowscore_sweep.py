"""Low-score track continuation sweep (decision D20, ByteTrack-style second pass).

The S4 head's detections exported at score >= 0.10 (tag ft102feat25_lo10) are
tracked with birth at 0.45 and continuation at L for every L in GRID; L 0.45 uses
no box under 0.45 and must reproduce the existing S4 runs (check). Everything else
is the S4 pipeline. The car is marked from the image as before: marker votes, the
2025 reference track, or the recorded hand picks. Only the 2025 tuning clips choose
L (rule in the decision list); the other clips are reported for every L.

    python scripts/evaluation/lowscore_sweep.py run CLIP [CLIP ...]
    python scripts/evaluation/lowscore_sweep.py check
    python scripts/evaluation/lowscore_sweep.py score
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]

GRID = [0.10, 0.15, 0.20, 0.30, 0.45]
CONFIGS = [(L, 1) for L in GRID] + [(L, k) for L in (0.15, 0.20, 0.30) for k in (4, 16)]   # (L, noise x k): D20, D21
DET = "ft102feat25_lo10"
C2025 = ["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"]
TUNING = ["HV_T_EW_1", "AV_T_WE_1"]
C2026 = ["AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6", "AV_T_EW_7"]
TRAJ = ROOT / "outputs/trajectories"
OUT = ROOT / "outputs/evaluation/lowscore"
TIE = 0.10      # within 10% of the baseline counts as no worse


def tag(L, k=1):
    return f"lo{round(L * 100):02d}" + (f"k{k}" if k != 1 else "")


def base(clip):
    return "abl_s4" if clip in C2025 else "iv_s4"


def sh(cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, NUMBA_DISABLE_JIT="1"))
    if r.returncode and check:
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout


def mark(clip, t):
    """Mark the car on run t from the image, the way the base run was marked (picks or marker votes)."""
    inst = TRAJ / f"{clip}_{base(clip)}" / "instrumented.json"
    ref = json.loads(inst.read_text()) if inst.exists() else None
    if ref is None and not (ROOT / f"outputs/target_id/{clip}/marker_matches.json").exists():
        return None                           # nothing to mark from (AV_T_EW_7: no GPS)
    if ref is not None and "picks" in ref:
        picks = [x for p in ref["picks"] for x in ("--pick", ",".join(map(str, p)))]
        sh([sys.executable, "scripts/pipeline/mark_instrumented_manual.py", "--clip", clip, "--tag", t] + picks, check=False)
    else:
        sh([sys.executable, "scripts/pipeline/mark_instrumented.py", "--clip", clip, "--tag", t], check=False)   # exits 1 when no car is found
    inst_t = TRAJ / f"{clip}_{t}" / "instrumented.json"
    return json.loads(inst_t.read_text()).get("vehicle_id") if inst_t.exists() else None


def run(clip):
    for L, k in CONFIGS:
        if (TRAJ / f"{clip}_{tag(L, k)}" / "tracks.json").exists():
            continue
        sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", tag(L, k), "--det-tag", DET,
            "--track-low-thresh", L, "--low-noise", k, "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1])
        print(clip, tag(L, k), "car", mark(clip, tag(L, k)))


def check():
    """Boxes at 0.45 or above equal the 0.45 export, and L 0.45 reproduces the S4 tracks."""
    det = ROOT / "outputs/object_detection/camera-data"
    for clip in C2025 + C2026:
        lo, hi = det / f"{clip}_{DET}", det / f"{clip}_ft102feat25"
        worst, n = 0.0, 0
        for p in sorted(hi.glob("*_pred.json")):
            a = sorted((b for b in json.loads(p.read_text())), key=lambda b: -b["score"])
            b = sorted((b for b in json.loads((lo / p.name).read_text()) if b["score"] >= 0.45), key=lambda b: -b["score"])
            if len(a) != len(b):
                worst = np.inf
                break
            for x, y in zip(a, b):
                worst = max(worst, max(abs(x[k] - y[k]) for k in ("x", "y", "z", "l", "w", "h", "yaw", "score")))
            n += len(a)
        t0 = json.loads((TRAJ / f"{clip}_{base(clip)}" / "tracks.json").read_text())["tracks"]
        t1 = json.loads((TRAJ / f"{clip}_{tag(0.45)}" / "tracks.json").read_text())["tracks"]
        same = t0.keys() == t1.keys() and all(
            len(t0[k]) == len(t1[k]) and all(abs(s["x"] - u["x"]) + abs(s["y"] - u["y"]) < 1e-3 for s, u in zip(t0[k], t1[k]))
            for k in t0)
        print(f"{clip}: {n} boxes at 0.45+, max difference {worst:.2g}; L 0.45 tracks identical to {base(clip)}: {same}")


def observed_per_frame(clip, t):
    tr = json.loads((TRAJ / f"{clip}_{t}" / "tracks.json").read_text())["tracks"]
    n = sum("x_det" in s for st in tr.values() for s in st)
    nf = len(list((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg")))
    return round(n / nf, 2), len(tr)


def score():
    import ablation as ab
    from gps_to_image import load_clip, load_site
    T = load_site()
    OUT.mkdir(parents=True, exist_ok=True)
    tags = ["abl_s4"] + [tag(L, k) for L, k in CONFIGS]
    res = {}
    for clip in C2025:
        data = {t: ab.load(clip, t, T) for t in tags}
        obs = {t: set(d["frame"][d["observed"]].tolist()) for t, d in data.items() if d is not None}
        union = set().union(*obs.values())
        gps = load_clip(clip, "abl_s4")["gps"]
        for t, d in data.items():
            if d is None:
                continue
            own = sorted(obs[t])
            p = ab.per_frame(d, own, gps, T)
            m = ab.agg(p, np.arange(len(own)), np.isfinite(p["acc"]))
            res[f"{clip}|{t}"] = {"car": d["id"], "frames": len(own), "coverage": round(len(own) / max(len(union), 1), 3),
                                  "lateral_m": m["across_rms_m"], "speed_mps": m["speed_mae_mph"] * 0.44704,
                                  "accel": m["accel_rms_diff"], "box_heading_deg": m["box_heading_deg"]}
    for clip in C2025 + C2026:
        for t in [base(clip)] + [tag(L, k) for L, k in CONFIGS]:
            if (TRAJ / f"{clip}_{t}" / "tracks.json").exists():
                res.setdefault(f"{clip}|{t}", {})["observed_per_frame"], res[f"{clip}|{t}"]["tracks"] = observed_per_frame(clip, t)
    # the selection rules (D20: k 1; D21: k > 1), tuning clips only; lowest L, then smaller k
    ok = lambda t: all(res[f"{c}|{t}"]["speed_mps"] <= (1 + TIE) * res[f"{c}|abl_s4"]["speed_mps"]
                       and res[f"{c}|{t}"]["lateral_m"] <= (1 + TIE) * res[f"{c}|abl_s4"]["lateral_m"]
                       and res[f"{c}|{t}"]["coverage"] >= res[f"{c}|abl_s4"]["coverage"] for c in TUNING)
    for rule, fam in (("D20", [(L, 1) for L in GRID[:-1]]), ("D21", [c for c in CONFIGS if c[1] > 1])):
        passing = sorted(c for c in fam if ok(tag(*c)))
        res[rule] = {"passing": [tag(*c) for c in passing], "chosen": tag(*passing[0]) if passing else tag(0.45)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        for c in sys.argv[2:]:
            run(c)
    else:
        {"check": check, "score": score}[cmd]()
