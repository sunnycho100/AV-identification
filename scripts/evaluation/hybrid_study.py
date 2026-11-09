"""2D and 3D together (decisions D31, D32), retrained head, file clock.

D31, 2D proposes and 3D confirms and measures: weak 3D boxes that a Faster R-CNN box
overlaps may start tracks (hb30: from 0.30, hb10: from 0.10); position, size and heading
stay 3D. D32, 2D fills gaps: 2D ground-plane boxes join as continuation-only candidates
(score 0.35, 4x noise, fixed 4.5 x 1.8 x 1.5 m) at all ranges (gf) or beyond 40 m (gf40).
Graded like lowfilter_study against S4 (nv45) and the plain continuation (nv30k4).

    python scripts/evaluation/hybrid_study.py run CLIP [CLIP ...]
    python scripts/evaluation/hybrid_study.py score
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import heldout_eval as he       # noqa: E402
import lowfilter_study as lf    # noqa: E402
import lowscore_sweep as ls     # noqa: E402

CAM = ROOT / "outputs/object_detection/camera-data"
VARIANTS = {"hb30": ("nvfeat25_lo10", 0.30, 4, ["--low-filter", "confirm", "--confirm-floor", 0.30]),
            "hb10": ("nvfeat25_lo10", 0.30, 4, ["--low-filter", "confirm", "--confirm-floor", 0.10]),
            "gf": ("nvplus2d_lo10", 0.30, 4, []), "gf40": ("nvplus2d40_lo10", 0.30, 4, [])}
MERGED = {"nvplus2d_lo10": 0.0, "nvplus2d40_lo10": 40.0}     # 2D boxes from this distance along the road
OUT = ROOT / "outputs/evaluation/hybrid"


def merge(clip, tag, min_x):
    """3D boxes plus 2D ground-plane boxes as weak candidates, in the 3D run's frame."""
    out, src, d2 = CAM / f"{clip}_{tag}", CAM / f"{clip}_nvfeat25_lo10", CAM / f"{clip}_det2d"
    if (out / "calibration_used.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads((src / "calibration_used.json").read_text())
    z = cal.get("road_plane_z_in_output", -1.73)
    for p in src.glob("*_pred.json"):
        boxes = json.loads(p.read_text())
        q = d2 / p.name
        extra = [{"class_id": 0, "class_name": "car", "score": 0.35, "x": b["x"], "y": b["y"], "z": z, "l": 4.5, "w": 1.8,
                  "h": 1.5, "yaw": 0.0, "from2d": True} for b in (json.loads(q.read_text()) if q.exists() else []) if b["x"] >= min_x]
        (out / p.name).write_text(json.dumps(boxes + extra))
    shutil.copy(src / "calibration_used.json", out / "calibration_used.json")


def run(clips):
    for clip in clips:
        for tag, min_x in MERGED.items():
            merge(clip, tag, min_x)
        for t, (det, L, k, extra) in VARIANTS.items():
            if not (ls.TRAJ / f"{clip}_{t}" / "tracks.json").exists():
                ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", t, "--det-tag", det,
                       "--track-low-thresh", L, "--low-noise", k, "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1] + extra)
            print(clip, t, "car", ls.mark(clip, t))


def score():
    from gps_to_image import load_site
    T = load_site()
    lf.VARIANTS = {t: None for t in ("nv45", "nv30k4", *VARIANTS)}
    res = {c: lf.grade(c, T, {}) for c in he.TUNING + he.HELD}            # {} = file clock (D30)
    b = lambda c, t: res[c].get(t, {})
    passes = lambda t: all("speed_mps" in b(c, t) and "speed_mps" in b(c, "nv45")
                           and b(c, t)["speed_mps"] <= 1.1 * b(c, "nv45")["speed_mps"]
                           and b(c, t)["lateral_m"] <= 1.1 * b(c, "nv45")["lateral_m"]
                           and b(c, t)["coverage"] >= b(c, "nv45")["coverage"] for c in he.TUNING)
    res["passes_rule"] = {t: passes(t) for t in lf.VARIANTS}
    res["D31_chosen"] = next((t for t in ("hb10", "hb30") if passes(t)), None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    f = lambda r, k, n=2: f"{r[k]:.{n}f}" if k in r else "-"
    print("| Clip | Variant | Seconds | Coverage | Lateral (m) | Speed (m/s) | Accel. (m/s^2) | Box heading (deg) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in he.TUNING + he.HELD:
        for t in lf.VARIANTS:
            r = res[c][t]
            print(f"| {c} | {t} | {r['seconds']} | {r['coverage']:.2f} | {f(r, 'lateral_m')} | {f(r, 'speed_mps')} | {f(r, 'accel')} | {f(r, 'box_heading_deg', 1)} |")
    print("\nheld-out means (file clock):")
    for t in lf.VARIANTS:
        R = [res[c][t] for c in he.HELD]
        G = [r for r in R if "speed_mps" in r]
        print(f"  {t:7s} coverage {np.mean([r['coverage'] for r in R]):.2f} seconds {sum(r['seconds'] for r in R):.1f} | graded {len(G)} | "
              f"lateral {np.mean([r['lateral_m'] for r in G]):.3f} speed {np.mean([r['speed_mps'] for r in G]):.3f} accel {np.mean([r['accel'] for r in G]):.3f}")
    print("passes the tuning rule:", res["passes_rule"], "| D31 chosen:", res["D31_chosen"])


if __name__ == "__main__":
    run(sys.argv[2:]) if sys.argv[1] == "run" else score()
