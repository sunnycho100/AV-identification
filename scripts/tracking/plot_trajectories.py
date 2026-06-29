"""Draw top-down (BEV) trajectories from AB3DMOT tracks.json.

The trajectory is what we actually want: each tracked vehicle's ID + its path of
(x, y) ground positions over the segment. Absolute time is irrelevant for the path
shape (constant Δt assumed), so this plots position only — lanes read as horizontal
bands, travel as left-right motion.

  /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/plot_trajectories.py --label seg_00_05
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRK_ROOT = ROOT / "outputs/tracking/our_intrinsics"


def plot_segment(label, min_len):
    tracks_path = TRK_ROOT / label / "tracks.json"
    data = json.loads(tracks_path.read_text())
    tracks = data["tracks"]
    meta = data["meta"]

    # keep tracks with enough points to form a real path
    kept = {tid: v for tid, v in tracks.items() if len(v) >= min_len}
    colors = cm.get_cmap("tab20")(np.linspace(0, 1, max(len(kept), 1)))

    fig, ax = plt.subplots(figsize=(11, 7))
    for (tid, pts), color in zip(sorted(kept.items(), key=lambda kv: int(kv[0])), colors):
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.9)
        ax.plot(xs[0], ys[0], "o", color=color, ms=5)          # start
        # arrow on the final step to show travel direction
        if len(xs) >= 2 and (xs[-1] != xs[-2] or ys[-1] != ys[-2]):
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
        ax.text(xs[0], ys[0], f" {tid}", color=color, fontsize=7, va="center")

    ax.set_xlabel("x — forward distance from camera (m)")
    ax.set_ylabel("y — lateral position (m)")
    ax.set_title(f"Vehicle trajectories — {label}  "
                 f"(frames {meta['frames'][0]:06d}–{meta['frames'][1]:06d}, "
                 f"{len(kept)}/{len(tracks)} tracks with ≥{min_len} points)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    out_path = TRK_ROOT / label / "trajectories.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label}: drew {len(kept)} trajectories (of {len(tracks)}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot BEV trajectories from tracks.json")
    ap.add_argument("--label", default=None, help="segment subfolder; omit to plot all segments")
    ap.add_argument("--min-len", type=int, default=3, help="min track length to draw")
    args = ap.parse_args()

    labels = [args.label] if args.label else [
        p.parent.name for p in TRK_ROOT.glob("*/tracks.json")
    ]
    for label in sorted(labels):
        plot_segment(label, args.min_len)


if __name__ == "__main__":
    main()
