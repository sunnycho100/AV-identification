"""Extra images for the ablation slides (outputs/reports/ablation/slides/).

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/ablation_slide_images.py
"""
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/reporting"), str(ROOT / "scripts/tracking")]
from evaluators.result2kitti import get_lidar_3d_8points          # noqa: E402
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image  # noqa: E402
import ablation as ab                                               # noqa: E402
import false_objects as fo                                          # noqa: E402
from ablation_figures import style, SERIES, INK, INK2, SURF, CLIPS, CLIP_LABEL  # noqa: E402

OUT = ROOT / "outputs/reports/ablation/slides"
DET = ROOT / "outputs/object_detection/camera-data"
RES = json.loads((ROOT / "outputs/evaluation/ablation/results.json").read_text())
RED = "#C00000"
MPH = 2.23694


def cal(clip, tag):
    c = json.loads((DET / f"{clip}_{tag}/calibration_used.json").read_text())
    return np.array(c["K"]), np.array(c["lidar2cam"])


def draw(img, o, K, M, col, arrow=True):
    b = np.asarray(get_lidar_3d_8points([o["l"], o["w"], o["h"]], o["yaw"], [o["x"], o["y"], o["z"] + o["h"] / 2]))
    cam = np.asarray((M @ np.c_[b, np.ones(8)].T).T[:, :3])
    if (cam[:, 2] <= 0).any():
        return
    draw_box_3d(img, project_to_image(cam, np.c_[K, np.zeros(3)]), c=col)
    if arrow:
        a = np.array([[o["x"], o["y"], o["z"]], [o["x"] + (o["l"] / 2 + 1.5) * math.cos(o["yaw"]),
                                                  o["y"] + (o["l"] / 2 + 1.5) * math.sin(o["yaw"]), o["z"]]])
        ca = (M @ np.c_[a, np.ones(2)].T)[:3]
        uv = (K @ ca)[:2] / ca[2]
        cv2.arrowedLine(img, tuple(uv[:, 0].astype(int)), tuple(uv[:, 1].astype(int)), (0, 0, 255), 3, tipLength=0.35)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(img, text, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def step_frames(clip="AV_T_EW_3", f=250):
    """Same frame through the detector steps: boxes as each step outputs them, cropped
    to where the boxes are (the same crop for every step)."""
    imgs, uvs = [], []
    for tag in ("base102_h151", "basefeat25", "ft102feat25"):
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        K, M = cal(clip, tag)
        for o in json.loads((DET / f"{clip}_{tag}/{f:03d}_pred.json").read_text()):
            if o["class_name"] != "car":
                continue
            draw(img, o, K, M, (0, 220, 0))
            q = M[:3, :3] @ np.array([o["x"], o["y"], o["z"]]) + M[:3, 3]
            uvs.append((K @ q)[:2] / q[2])
        imgs.append(img)
    uv = np.array(uvs)
    x0, y0 = np.clip(uv.min(0) - [160, 120], 0, None).astype(int)
    x1, y1 = np.minimum(uv.max(0) + [160, 120], [1920, 1080]).astype(int)
    if (x1 - x0) < 1.6 * (y1 - y0):                       # keep a wide crop
        cx, w = (x0 + x1) // 2, int(1.6 * (y1 - y0))
        x0, x1 = max(cx - w // 2, 0), min(cx + w // 2, 1920)
    tiles = [label(cv2.resize(i[y0:y1, x0:x1], None, fx=1.5, fy=1.5), n)
             for i, n in zip(imgs, ("Pretrained", "+ feature mask", "+ fine-tune"))]
    sep = np.full((tiles[0].shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / "step_frames.jpg"), np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for t, n in zip(tiles, ("pretrained", "mask", "finetune")):
        cv2.imwrite(str(OUT / f"frame_{n}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 88])


def false_map(clip="HV_T_EW_1", f=150):
    """Every false detection of the clip, drawn on one frame: pretrained vs fine-tuned with the mask."""
    tiles = []
    for tag, name in (("base102_h151", "Pretrained"), ("ft102feat25", "Fine-tuned + feature mask")):
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        K, M = cal(clip, tag)
        pts = fo.false_points(clip, tag)
        for x, y in pts:
            q = M[:3, :3] @ np.array([x, y, -1.73]) + M[:3, 3]
            if q[2] > 0:
                u, v = (K @ q)[:2] / q[2]
                cv2.circle(img, (int(u), int(v)), 9, (0, 0, 220), -1)
                cv2.circle(img, (int(u), int(v)), 9, (255, 255, 255), 2)
        n = fo.objects(pts) if pts else 0
        tiles.append(label(img, f"{name}: {len(pts)} false boxes, {n} objects"))
    sep = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
    out = np.hstack([tiles[0], sep, tiles[1]])
    cv2.imwrite(str(OUT / "false_map.jpg"), cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2)), [cv2.IMWRITE_JPEG_QUALITY, 88])


def jump_example():
    """A real track where jump removal dropped detections: position and speed before and after."""
    import postprocess_tracks as pp
    best = None
    for clip in fo.CLIPS:
        t3 = json.loads((ROOT / f"outputs/trajectories/{clip}_abl_s2/tracks.json").read_text())["tracks"]
        for k, s in t3.items():
            if len(s) < 40:
                continue
            kept = pp.remove_jumps(s, 30)
            n_drop = len(s) - sum(len(p) for p in kept)
            sp = max(x["speed_mps"] for x in s)
            if 0 < n_drop <= 6 and len(kept) == 1 and (best is None or sp > best[0]):
                best = (sp, clip, k, s, kept[0])
    _, clip, k, s, kept = best
    keep = {x["frame"] for x in kept}
    t = np.array([x["t_s"] for x in s])
    xy = np.array([[x["x"], x["y"]] for x in s])
    drop = np.array([x["frame"] not in keep for x in s])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF)
    for a in ax:
        style(a)
    ax[0].plot(t, xy[:, 1], "-o", color="#9a9893", ms=4, lw=1, label="detections")
    ax[0].plot(t[drop], xy[drop, 1], "o", color=RED, ms=10, mfc="none", mew=2.5, label="removed as a jump")
    ax[0].set(xlabel="video time (s)", ylabel="across-road position (m)")
    ax[0].set_title("Across-road position of one car", color=INK, loc="left", fontsize=11)
    sp_before = np.array([x["speed_mps"] for x in pp.finite_velocity(s)]) * MPH
    sp_after = np.array([x["speed_mps"] for x in pp.finite_velocity(kept)]) * MPH
    ax[1].plot(t, sp_before, color="#9a9893", lw=1.2, label="before")
    ax[1].plot([x["t_s"] for x in kept], sp_after, color=SERIES[0], lw=2, label="after jump removal")
    ax[1].set(xlabel="video time (s)", ylabel="box-to-box speed (mph)")
    ax[1].set_title("Speed from consecutive boxes", color=INK, loc="left", fontsize=11)
    for a in ax:
        a.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"{clip}, vehicle {k}: {int(drop.sum())} detections removed", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "jump_example.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def loo_bars():
    """Test clip: how much worse (or better) S4 gets when one component is removed."""
    d = {r["a"]: r for r in RES["loo"].values() if "a" in r and r["clip"] == "AV_T_EW_3"}
    names = {"abl_no_mask": "without feature mask", "abl_no_ft": "without fine-tune", "abl_no_jump": "without jump removal"}
    panels = [("speed_mae_mph", "Speed error change (mph)"), ("across_rms_m", "Across-road change (m)"),
              ("box_heading_deg", "Box heading change (deg)")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), facecolor=SURF)
    for a, (key, title) in zip(ax, panels):
        style(a)
        a.grid(axis="x", color="#e4e3df")
        a.grid(axis="y", visible=False)
        v = [d[n]["delta"][key] for n in names]
        ci = [d[n]["ci95"][key] for n in names]
        y = np.arange(len(names))
        a.barh(y, v, color=[SERIES[1] if x > 0 else SERIES[0] for x in v], height=0.5)
        a.errorbar(v, y, xerr=[[x - c[0] for x, c in zip(v, ci)], [c[1] - x for x, c in zip(v, ci)]], fmt="none",
                   ecolor=INK, capsize=4, lw=1.2)
        a.axvline(0, color=INK, lw=1)
        a.set_yticks(y, list(names.values()) if key == panels[0][0] else [""] * 3, fontsize=10)
        a.set_title(title, color=INK, loc="left", fontsize=11)
        a.invert_yaxis()
    fig.suptitle("Test clip: change when one component is removed from the full pipeline (orange = worse)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "leave_one_out.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def coverage():
    steps = ab.MAIN
    names = ["Pretrained", "+ mask", "+ fine-tune", "+ jumps", "+ smoothing"]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURF)
    style(ax)
    w = 0.16
    for i, s in enumerate(steps):
        v = [RES["main"][f"{c}|{s}"]["coverage"] * 100 for c in CLIPS]
        ax.bar(np.arange(4) + (i - 2) * w, v, width=w - 0.02, color=["#c9c7c0", "#a9b8cc", SERIES[0], "#5f93d8", "#14213D"][i],
               label=names[i])
    ax.set_xticks(range(4), [CLIP_LABEL[c] for c in CLIPS], fontsize=9.5)
    ax.set_ylabel("% of frames the lab car is observed", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32), labelcolor=INK2)
    ax.set_title("Lab car coverage: fine-tuning finds the car in more frames", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def impossible():
    order = [("abl_s0", "Pretrained"), ("abl_s1", "+ mask"), ("abl_s2", "+ fine-tune"), ("abl_s3", "+ jump removal"),
             ("abl_s4", "+ smoothing")]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURF)
    style(ax)
    v = [RES["impossible_states"][k]["over_100_mph"] for k, _ in order]
    bars = ax.bar(range(5), v, width=0.55, color=[SERIES[0]] * 3 + [SERIES[1], "#14213D"], edgecolor=SURF, lw=2)
    for b, n in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, n + 2, str(n), ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(5), [n for _, n in order], fontsize=9.5)
    ax.set_title("States faster than 100 mph, all vehicles, all 5 clips", color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "impossible_states.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def hv_variation():
    from gps_to_image import load_site
    T = load_site()
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURF)
    style(ax)
    for (tag, name), col in zip((("abl_q0.03", "q 0.03 (stronger)"), ("abl_q0.1", "q 0.1 (picked)"),
                                 ("abl_q1", "q 1 (lighter)")), ("#9a9893", SERIES[0], SERIES[1])):
        d = ab.load("HV_T_EW_1", tag, T)
        ax.plot(d["t"], np.hypot(*d["v"].T) * MPH, color=col, lw=2.4 if "picked" in name else 1.6, label=name)
    ax.plot(d["t"], d["vg"] * MPH, "--", color=INK, lw=1.8, label="GPS")
    ax.set(xlabel="video time (s)", ylabel="speed (mph)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=4, loc="upper left")
    ax.set_title("HV_T_EW_1 (human driving): the picked setting keeps the real speed-up", color=INK, loc="left",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "hv_speed_variation.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def per_clip_table():
    cols = [("speed_mae_mph", "Speed err\n(mph)", 2), ("across_rms_m", "Across\n(m)", 2),
            ("box_heading_deg", "Box heading\n(deg)", 1), ("accel_rms_diff", "Accel err\n(m/s²)", 2)]
    rows, labels = [], []
    for c in CLIPS:
        for s, n in zip(ab.MAIN, ["Pretrained", "+ mask", "+ fine-tune", "+ jumps", "+ smoothing"]):
            r = RES["main"][f"{c}|{s}"]
            rows.append([f"{r['metrics'][k]:.{p}f}" for k, _, p in cols])
            labels.append(f"{c}  {n}")
    fig, ax = plt.subplots(figsize=(9, 7.4), facecolor="white")
    ax.axis("off")
    tb = ax.table(cellText=rows, rowLabels=labels, colLabels=[c[1] for c in cols], loc="center", cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if r == 0:
            cell.set_facecolor(RED)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.6)
        elif (r - 1) % 5 == 4:
            cell.set_facecolor("#F6E5E5")
    fig.tight_layout()
    fig.savefig(OUT / "per_clip_table.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or ["step_frames", "false_map", "jump_example", "loo_bars", "coverage", "impossible",
                                   "hv_variation", "per_clip_table"]):
        globals()[name]()
    print("->", OUT)


if __name__ == "__main__":
    main()
