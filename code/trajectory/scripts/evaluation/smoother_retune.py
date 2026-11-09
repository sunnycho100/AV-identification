"""Does the RTS smoother's process noise (q_vel 0.1, set for round 1) still suit the round-2 head? (overnight E5)

Rule fixed before running: pick q_vel by mean speed error on the two tuning clips (also
round-2 training clips, as in D31), provided their tracked seconds stay within 10% of
q_vel 0.1; held-out numbers are reported for every q, but only the tuning choice counts.

    python scripts/evaluation/smoother_retune.py run CLIP [CLIP ...]
    python scripts/evaluation/smoother_retune.py score
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import heldout_eval as he       # noqa: E402
import lowfilter_study as lf    # noqa: E402
import lowscore_sweep as ls     # noqa: E402

Q = {"r2fullhb": 0.1, "r2fullhbq03": 0.03, "r2fullhbq3": 0.3, "r2fullhbq1": 1.0}


def run(clips):
    for clip in clips:
        for t, q in Q.items():
            if not (ls.TRAJ / f"{clip}_{t}" / "tracks.json").exists():
                ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", clip, "--tag", t, "--det-tag", "r2full_lo10",
                       "--track-low-thresh", 0.30, "--low-noise", 4, "--jumps", "remove", "--smooth", "rts", "--q-vel", q,
                       "--low-filter", "confirm", "--confirm-floor", 0.30])
            print(clip, t, "car", ls.mark(clip, t))


def score():
    from gps_to_image import load_site
    T = load_site()
    lf.VARIANTS = {t: None for t in Q}
    res = {c: lf.grade(c, T, {}) for c in he.TUNING + he.HELD}
    out = {}
    for t in Q:
        for name, clips in (("tuning", he.TUNING), ("held_out", he.HELD)):
            g = [c for c in clips if "speed_mps" in res[c][t]]
            out[f"{t} {name}"] = {"q_vel": Q[t], "seconds": round(sum(res[c][t]["seconds"] for c in clips), 1),
                                  **{k: round(float(np.mean([res[c][t][k] for c in g])), 3) for k in ("speed_mps", "lateral_m", "accel")}}
    ok = [t for t in Q if out[f"{t} tuning"]["seconds"] >= 0.9 * out["r2fullhb tuning"]["seconds"]]
    chosen = min(ok, key=lambda t: out[f"{t} tuning"]["speed_mps"])
    out["chosen_by_tuning"] = chosen
    (ROOT / "outputs/evaluation/round2/smoother_retune.json").write_text(json.dumps({"summary": out, "per_clip": res}, indent=1))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    run(sys.argv[2:]) if sys.argv[1] == "run" else score()
