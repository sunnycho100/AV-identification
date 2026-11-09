"""Round-2 self-training heads (D34) against round 1, graded like D31: file clock, held-out passes.

Each head is tracked two ways: S4 (score 0.45, no continuation) and hb30 (the adopted
setting: continuation from 0.30, weak boxes start tracks only with 2D agreement).
Round 1 reuses its existing runs (nv45, hb30). Pass rule (pre-registered, D34 plan and
setup): under hb30, round 2 full tracks at least as many held-out seconds as round 1,
with paired mean speed and lateral errors within 1.1 times round 1's.

    python scripts/evaluation/round2_study.py fetch
    python scripts/evaluation/round2_study.py run CLIP [CLIP ...]
    python scripts/evaluation/round2_study.py score
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import heldout_eval as he       # noqa: E402
import lowfilter_study as lf    # noqa: E402
import lowscore_sweep as ls     # noqa: E402

HEADS = {"r1": "nvfeat25_lo10", "r2full": "r2full_lo10", "r2noig": "r2noignore_lo10", "r2no2d": "r2no2d_lo10"}
TRACK = {"s4": (0.45, 1, []), "hb": (0.30, 4, ["--low-filter", "confirm", "--confirm-floor", 0.30])}
EXISTING = {("r1", "s4"): "nv45", ("r1", "hb"): "hb30"}
OUT = ROOT / "outputs/evaluation/round2"
CAM = ROOT / "outputs/object_detection/camera-data"


def tag(h, t):
    return EXISTING.get((h, t), f"{h}{t}")


def fetch():
    """Pull each finished server detection (marked .done there) that is not here yet."""
    srv = "roadside-camera/BEVHeights/outputs/object_detection/camera-data"
    done = subprocess.run(["ssh", "cee", f"cd {srv} && ls -d *_r2*_lo10/.done"], capture_output=True, text=True).stdout.split()
    for d in (p.split("/")[0] for p in done):
        if not (CAM / d / ".fetched").exists():
            subprocess.run(["rsync", "-az", "--exclude", ".done", f"cee:{srv}/{d}", str(CAM)], check=True)
            (CAM / d / ".fetched").touch()
            print("fetched", d)


def run(clips):
    for clip in clips:
        for h, det in HEADS.items():
            if h != "r1" and not (CAM / f"{clip}_{det}" / ".fetched").exists():
                continue                                            # head not detected on this clip yet
            for t, (L, k, extra) in TRACK.items():
                g = tag(h, t)
                if not (ls.TRAJ / f"{clip}_{g}" / "tracks.json").exists():
                    ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", g, "--det-tag", det,
                           "--track-low-thresh", L, "--low-noise", k, "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1] + extra)
                print(clip, g, "car", ls.mark(clip, g))


def score():
    from gps_to_image import load_site
    T = load_site()
    lf.VARIANTS = {tag(h, t): None for t in TRACK for h in HEADS}
    res = {c: lf.grade(c, T, {}) for c in he.TUNING + he.HELD}            # {} = file clock (D30)
    summary = {}
    for t in TRACK:
        base = tag("r1", t)
        for h in HEADS:
            g = tag(h, t)
            both = [c for c in he.HELD if "speed_mps" in res[c][g] and "speed_mps" in res[c][base]]
            summary[g] = {"seconds": round(sum(res[c][g]["seconds"] for c in he.HELD), 1),
                          "graded_with_r1": len(both),
                          **{k: round(float(np.mean([res[c][g][k] for c in both])), 3) if both else None
                             for k in ("speed_mps", "lateral_m", "accel", "box_heading_deg")},
                          **{f"r1_{k}": round(float(np.mean([res[c][base][k] for c in both])), 3) if both else None
                             for k in ("speed_mps", "lateral_m")}}
    s, r = summary[tag("r2full", "hb")], summary[tag("r1", "hb")]
    passed = (s["seconds"] >= r["seconds"] and s["speed_mps"] <= 1.1 * s["r1_speed_mps"]
              and s["lateral_m"] <= 1.1 * s["r1_lateral_m"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"per_clip": res, "held_out": summary, "round2_adopted": passed}, indent=1))
    print("| Tracking | Head | Held-out seconds | Passes graded (paired) | Speed (m/s) | Lateral (m) | Accel. (m/s^2) | Box heading (deg) | Round 1 speed, lateral (same passes) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in TRACK:
        for h in HEADS:
            v = summary[tag(h, t)]
            print(f"| {t} | {h} | {v['seconds']} | {v['graded_with_r1']} | {v['speed_mps']} | {v['lateral_m']} | {v['accel']} | "
                  f"{v['box_heading_deg']} | {v['r1_speed_mps']}, {v['r1_lateral_m']} |")
    print("round 2 adopted under the pre-registered rule:", passed)


if __name__ == "__main__":
    {"fetch": lambda: fetch(), "run": lambda: run(sys.argv[2:]), "score": lambda: score()}[sys.argv[1]]()
