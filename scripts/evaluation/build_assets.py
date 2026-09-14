"""Presentation assets for the advisor update (140.8 m checkpoint run).

    .venv/bin/python scripts/evaluation/build_assets.py

Writes PNGs to outputs/evaluation/assets/.
"""
import json, sys
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_error_model import ROOT, load_clips, predict, rot, P_S
from trial_time_offsets import shifted, fit_shared_dt

OUT = ROOT / "outputs/evaluation/assets"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, PURPLE, ORANGE, INK = "#2C6BE0", "#8E4CB8", "#D07A16", "#1A2430"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#D5DCE3", "grid.linewidth": 0.7})
cfg = json.load(open(ROOT / "scripts/evaluation/site_T_r140.json"))
clips, _ = load_clips(cfg); by = {c["name"]: c for c in clips}
trial = json.load(open(ROOT / "outputs/evaluation/trial_time_offsets_r140.json"))
T1 = trial["T1_shared_plus_dt"]
s_sh = list(T1["theta"]["s"].values())[0]
theta = np.array([T1["theta"]["C_e"], T1["theta"]["C_n"], np.radians(T1["theta"]["beta_deg"]), 0, s_sh, 0, 0])
C, beta = theta[:2], theta[2]
to_cam = lambda G: ((G - C) @ rot(beta)) / s_sh
order = ["AV_T_EW_3", "HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3"]

# ---------- 1. GPS projected into the image (headline)
def project(K, l2c, x, y, z):
    pc = l2c @ np.array([x, y, z, 1.0]); u, v, w = K @ pc[:3]; return int(u / w), int(v / w)

def headline(name, frame, run="r140"):
    c = by[name]; dt = T1["dt_s"][name]
    det = ROOT / f"outputs/object_detection/camera-data/{name}_{run}"
    cal = json.load(open(det / "calibration_used.json")); K, l2c = np.array(cal["K"]), np.array(cal["lidar2cam"])
    img = cv2.imread(str(det / f"{frame:03d}_annotated.jpg"))
    i = int(np.argmin(np.abs(c["t"] * 30 - frame)))   # track sample at this frame (30 fps clips)
    z = float(np.median([s for s in [c["q"][i, 0] * 0 - 1.5]]))
    q_cam = c["q"][i]; q_lab = to_cam(c["g"])[i]; q_fix = to_cam(shifted(c, dt)["g"])[i]
    pts = {"camera track": (q_cam, BLUE), "GPS at lab timing": (q_lab, PURPLE), f"GPS shifted {dt:+.2f} s": (q_fix, ORANGE)}
    def bgr(h): return tuple(int(h[k:k + 2], 16) for k in (5, 3, 1))
    uvs = {}
    for label, (q, col) in pts.items():
        u, v = project(K, l2c, q[0], q[1], -1.5); uvs[label] = (u, v)
        cv2.circle(img, (u, v), 14, bgr(col), -1); cv2.circle(img, (u, v), 14, (255, 255, 255), 2)
    (u1, v1), (u2, v2) = uvs["GPS at lab timing"], uvs[f"GPS shifted {dt:+.2f} s"]
    cv2.arrowedLine(img, (u1, v1), (u2, v2), bgr(PURPLE), 3, tipLength=0.08)
    dist = np.hypot(*(q_lab - q_fix))
    cv2.putText(img, f"{dist:.0f} m = {abs(dt):.1f} s at {c['v'][i]:.0f} m/s", ((u1 + u2) // 2 + 15, (v1 + v2) // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5); cv2.putText(img, f"{dist:.0f} m = {abs(dt):.1f} s at {c['v'][i]:.0f} m/s", ((u1 + u2) // 2 + 15, (v1 + v2) // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr(INK), 2)
    y0 = 40
    for label, (_, col) in pts.items():
        cv2.circle(img, (40, y0), 12, bgr(col), -1); cv2.putText(img, label, (62, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4); cv2.putText(img, label, (62, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bgr(INK), 2); y0 += 36
    cv2.putText(img, f"{name}  frame {frame}  140.8 m checkpoint", (40, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4); cv2.putText(img, f"{name}  frame {frame}  140.8 m checkpoint", (40, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr(INK), 2)
    p = OUT / f"1_gps_in_image_{name}.png"; cv2.imwrite(str(p), cv2.resize(img, (1600, 900))); return p

a1 = headline("HV_T_EW_1", 190); a1b = headline("AV_T_WE_1", 110)

# ---------- 2. bird's-eye before/after, four panels
fig, axes = plt.subplots(2, 2, figsize=(14, 6.4), sharex=True, sharey=True)
for ax, name in zip(axes.ravel(), order):
    c = by[name]; m = ~c["invalid"]; dt = T1["dt_s"][name]
    ax.scatter(c["q"][m, 0], c["q"][m, 1], s=10, color=BLUE, zorder=3)
    g0 = to_cam(c["g"]); ax.scatter(g0[m, 0], g0[m, 1], s=18, facecolors="none", edgecolors=PURPLE, linewidths=1.1, zorder=2)
    g1 = to_cam(shifted(c, dt)["g"]); ax.scatter(g1[m, 0], g1[m, 1], s=8, color=ORANGE, zorder=4)
    ax.plot(0, 0, marker="^", color=INK, ms=9, zorder=5)
    arrow = "toward camera" if name.split("_")[2] == "EW" else "away from camera"
    pin = ", at data limit" if T1["dt_pinned"][name] else ""
    ax.set_title(f"{name}, {arrow}   lab timing off by {dt:+.2f} s{pin}", fontsize=11, loc="left")
    ax.set_aspect("equal"); ax.set_xlim(-5, 140); ax.set_ylim(-34, 4)
for ax in axes[1]: ax.set_xlabel("range along the road from the camera (m)")
for ax in axes[:, 0]: ax.set_ylabel("left of camera axis (m)")
fig.legend([plt.Line2D([], [], marker="o", ls="", color=BLUE), plt.Line2D([], [], marker="o", ls="", mfc="none", mec=PURPLE), plt.Line2D([], [], marker="o", ls="", color=ORANGE)],
           ["camera track (BEVHeight 140.8 m)", "GPS car at the lab's video timing", "GPS car after the fitted time offset"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("One shared camera position, bearing and range scale for all four clips", x=0.02, ha="left", fontsize=13)
fig.subplots_adjust(hspace=0.45, wspace=0.06, top=0.9, bottom=0.14)
fig.savefig(OUT / "2_birds_eye_before_after.png", dpi=115, facecolor="white", bbox_inches="tight"); plt.close(fig)

# ---------- 3. box orientation, 102.4 vs 140.8, same frame crops
def crop_pair(name, frame, size=(700, 400)):
    """Crops centred on the target's projected position in the 140.8 m run."""
    c = by[name]; i = int(np.argmin(np.abs(c["t"] * 30 - frame)))
    cal = json.load(open(ROOT / f"outputs/object_detection/camera-data/{name}_r140/calibration_used.json"))
    u, v = project(np.array(cal["K"]), np.array(cal["lidar2cam"]), c["q"][i, 0], c["q"][i, 1], -1.0)
    half_w = 350 if c["q"][i, 0] < 60 else 250
    half_h = int(half_w * size[1] / size[0])
    x0, y0 = max(0, min(1920 - 2 * half_w, u - half_w)), max(0, min(1080 - 2 * half_h, v - half_h))
    box = (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h)
    ims = []
    for run, label in (("phase1", "102.4 m checkpoint"), ("r140", "140.8 m checkpoint")):
        im = cv2.imread(str(ROOT / f"outputs/object_detection/camera-data/{name}_{run}/{frame:03d}_annotated.jpg"))
        cr = cv2.resize(im[box[1]:box[3], box[0]:box[2]], size)
        cv2.putText(cr, f"{name} frame {frame}, {label}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4); cv2.putText(cr, f"{name} frame {frame}, {label}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        ims.append(cr)
    return np.hstack(ims)
rows = [crop_pair("HV_T_EW_1", 230), crop_pair("AV_T_EW_3", 250), crop_pair("AV_T_WE_1", 90)]
cv2.imwrite(str(OUT / "3_box_orientation_102_vs_140.png"), np.vstack(rows))

# ---------- 4. held-out residual vs range (140.8)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharex=True)
cols = {"HV_T_EW_1": BLUE, "AV_T_WE_1": ORANGE, "AV_T_WE_3": PURPLE}
for name, v in T1["partial_leave_one_out"].items():
    if "held_out" not in v or v.get("note"): continue
    c = by[name]; others = [o for o in clips if o is not c]
    th = fit_shared_dt(others, False)["theta"]; cs = shifted(c, v["dt_fitted_on_this_clip_s"]); m = ~cs["invalid"]
    pc, pg = predict(th, cs); r = (pc - pg)[m]; d = np.stack([np.cos(cs["psi"]), np.sin(cs["psi"])], 1)[m]
    along = (r * d).sum(1); across = (r * np.stack([-d[:, 1], d[:, 0]], 1)).sum(1); x = c["q"][m, 0]
    axes[0].plot(x, along, marker="o", ms=3.5, lw=1, color=cols[name], label=f"{name} (rms {v['held_out']['along_rms_m']:.2f} m)")
    axes[1].plot(x, across, marker="o", ms=3.5, lw=1, color=cols[name], label=f"{name} (rms {v['held_out']['across_rms_m']:.2f} m)")
for ax, t in zip(axes, ["along the road, after the 1-parameter time shift", "across the road, fully held out"]):
    ax.axhline(0, color=INK, lw=0.8, alpha=0.6); ax.set_title(t, fontsize=12, loc="left"); ax.set_xlabel("range from camera (m)"); ax.set_ylim(-3, 3); ax.legend(frameon=False, fontsize=10)
axes[0].set_ylabel("camera minus GPS (m)")
fig.suptitle("Held-out error: camera position, bearing and scale taken from the other three clips (140.8 m checkpoint)", x=0.02, ha="left", fontsize=13)
fig.savefig(OUT / "4_held_out_residual_vs_range.png", dpi=115, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
