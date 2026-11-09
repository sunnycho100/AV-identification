"""Per-pass bias versus noise of the GPS car's errors (overnight E2, Bland-Altman style).

Signed speed error (detector minus GPS) and the absolute lateral and along-road offsets
that the RMS grades remove. A consistent sign across passes points to calibration
(scale, height, roll); scatter around zero points to detection noise.

    python scripts/evaluation/bias_check.py TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import ablation as ab           # noqa: E402
import heldout_eval as he       # noqa: E402
import round2_same_frames as sf  # noqa: E402


def main(tags):
    from gps_to_image import load_clip, load_site
    T = load_site()
    out = {}
    for tag in tags:
        rows = []
        for c in he.TUNING + he.HELD:
            gps = load_clip(c, he.tag(c, "S4"))["gps"]
            d = sf.load(c, tag, T, gps)
            if d is None:
                continue
            obs = sorted(set(d["frame"][d["observed"]].tolist()))
            if len(obs) < 10:
                continue
            k = np.isin(d["frame"], obs)
            q, g, v, vg = d["q"][k], d["g"][k], d["v"][k], d["vg"][k]
            sp = np.hypot(v[:, 0], v[:, 1])
            off = (q - g).mean(0)
            rows.append({"clip": c, "frames": len(obs), "speed_bias": float(np.mean(sp - vg)), "speed_sd": float(np.std(sp - vg)),
                         "rel_bias_pct": float(100 * np.mean(sp - vg) / np.mean(vg)), "along_off_m": float(off[0]), "across_off_m": float(off[1])})
        out[tag] = rows
        print(tag)
        for r in rows:
            print(f"   {r['clip']:10s} {r['frames']:4d} fr | speed bias {r['speed_bias']:+.2f} m/s ({r['rel_bias_pct']:+.1f}%), sd {r['speed_sd']:.2f} | "
                  f"offset along {r['along_off_m']:+.2f} m, across {r['across_off_m']:+.2f} m")
        rb = [r["rel_bias_pct"] for r in rows if r["clip"] in he.HELD]
        print(f"   held-out: mean relative speed bias {np.mean(rb):+.1f}% (passes positive {sum(x > 0 for x in rb)}/{len(rb)})")
    (ROOT / "outputs/evaluation/round2/bias_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
