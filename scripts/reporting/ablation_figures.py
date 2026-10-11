"""Figures for the pipeline ablation (outputs/evaluation/ablation/*.json -> outputs/reports/ablation/).

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/ablation_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
A = ROOT / "outputs/evaluation/ablation"
OUT = ROOT / "outputs/reports/ablation"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # categorical slots 1-4, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLIPS = ["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"]
CLIP_LABEL = {"HV_T_EW_1": "HV_T_EW_1 (tuning)", "AV_T_WE_1": "AV_T_WE_1 (tuning)",
              "AV_T_WE_3": "AV_T_WE_3 (held out)", "AV_T_EW_3": "AV_T_EW_3 (test)"}
STEPS = ["abl_s0", "abl_s1", "abl_s2", "abl_s3", "abl_s4"]
STEP_SHORT = ["Pretrained\n+ tracking", "+ feature\nmask", "+ fine-tune", "+ jump\nremoval", "+ light\nsmoothing"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def steps_figure(res):
    panels = [("speed_mae_mph", "Speed error (mph)", True), ("accel_rms_diff", "Acceleration error (m/s²)", True),
              ("across_rms_m", "Across-road error (m)", False), ("box_heading_deg", "Box heading error (deg)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), facecolor=SURF)
    x = np.arange(len(STEPS))
    for ax, (key, title, log) in zip(axes, panels):
        style(ax)
        for c, col in zip(CLIPS, SERIES):
            y = [res["main"][f"{c}|{s}"]["metrics"][key] for s in STEPS]
            ax.plot(x, y, "-o", color=col, lw=2, ms=7, mec=SURF, mew=1.5,
                    label=CLIP_LABEL[c], zorder=3 if c == "AV_T_EW_3" else 2)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x, STEP_SHORT, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    fig.suptitle("Each step adds one component (lab car vs GPS, one constant offset removed)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_steps.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def traces_figure():
    import ablation as ab
    from gps_to_image import load_site
    T = load_site()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), facecolor=SURF, sharey=False)
    for ax, clip in zip(axes, ["HV_T_EW_1", "AV_T_EW_3"]):
        style(ax)
        raw, sm = ab.load(clip, "abl_s3", T), ab.load(clip, "abl_s4", T)
        ax.plot(raw["t"], np.hypot(*raw["v"].T) * 2.23694, color="#9a9893", lw=1, label="Before smoothing (jumps removed)")
        ax.plot(sm["t"], np.hypot(*sm["v"].T) * 2.23694, color=SERIES[0], lw=2.2, label="After light smoothing")
        ax.plot(sm["t"], sm["vg"] * 2.23694, "--", color=INK, lw=1.6, label="GPS")
        lo, hi = np.percentile(sm["vg"] * 2.23694, [0, 100])
        ax.set_ylim(lo - 25, hi + 25)
        ax.set_xlabel("video time (s)", color=INK2, fontsize=9)
        ax.set_ylabel("speed (mph)", color=INK2, fontsize=9)
        ax.set_title(CLIP_LABEL[clip], color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Lab car speed: raw box-to-box speed jumps; light smoothing follows GPS, including HV_T_EW_1's speed-up",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "speed_traces.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def q_figure(qp):
    tags = list(qp["mean_accel_rms_diff"])
    qs = [float(t.split("q")[-1]) for t in tags]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURF)
    for ax in axes:
        style(ax)
        ax.set_xscale("log")
        ax.axvline(qp["q_vel"], color=INK2, lw=1, ls=":")
    axes[0].plot(qs, [qp["mean_accel_rms_diff"][t] for t in tags], "-o", color=SERIES[0], lw=2, ms=7, mec=SURF)
    axes[0].set_title("Acceleration error vs GPS (mean of tuning clips)", color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("m/s²", color=INK2, fontsize=9)
    for c, col in zip(["HV_T_EW_1", "AV_T_WE_1"], SERIES):
        axes[1].plot(qs, [qp["per_clip"][f"{c}|{t}"]["metrics"]["speed_std_ratio"] for t in tags], "-o", color=col,
                     lw=2, ms=7, mec=SURF, label=CLIP_LABEL[c])
    axes[1].axhline(1, color=INK, lw=1)
    axes[1].set_title("Speed variation, camera / GPS (1 = kept)", color=INK, fontsize=10.5, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    for ax in axes:
        ax.set_xlabel("smoothing setting q_vel (higher = lighter)", color=INK2, fontsize=9)
    axes[0].annotate(f"picked {qp['q_vel']:g}", (qp["q_vel"], qp["mean_accel_rms_diff"][qp["picked"]]),
                     xytext=(12, 18), textcoords="offset points", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "smoothing_setting.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def false_figure(res):
    order = [("base102_h151", "Pretrained"), ("basefeat25", "Pretrained\n+ feature mask"),
             ("ft102", "Fine-tuned"), ("ft102feat25", "Fine-tuned\n+ feature mask")]
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURF)
    style(ax)
    v = [res["false_objects"][k]["objects"] for k, _ in order]
    bars = ax.bar(range(4), v, width=0.55, color=SERIES[0], edgecolor=SURF, linewidth=2)
    for b, n in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, n + 0.5, str(n), ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(4), [l for _, l in order], fontsize=9)
    ax.set_title("False objects, all 5 clips (one object counted once)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "false_objects.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((A / "results.json").read_text())
    steps_figure(res)
    traces_figure()
    q_figure(json.loads((A / "q_pick.json").read_text()))
    false_figure(res)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
