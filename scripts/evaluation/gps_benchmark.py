"""Evaluation only: one GPS benchmark per clip, timing-free and timing-dependent parts apart.

The car graded is the one picked from the image (instrumented.json); GPS never
chooses it and nothing here feeds back into the pipeline. GPS goes into the road
frame with the site transform from gps_to_image.py (set on the calibration clips
only), so no per-clip rotation or translation is fitted.

Timing-free (hold whatever the GPS-video offset is, on a straight road at steady
speed):
  speed ratio      mean camera speed (tracks.json speed_mps) / mean GPS speed
  lateral          camera y minus GPS y in the road frame: bias and rms, and each
                   one's offset from the nearest lane centre (lanes.json)
  heading          camera velocity direction minus GPS heading, both in the road frame
  accel noise      camera and GPS acceleration, same 0.5 s central difference as
                   trajectory_features.diff, on true frame times
Timing-dependent:
  along            camera x minus GPS x. At steady speed a time offset tau and an
                   along shift v*tau are the same, so one constant tau is fitted
                   (camera frame at t matched to GPS at lab time t + tau) and the
                   along residual after it is reported with tau. |tau| > 0.5 s is
                   flagged. tau is relative to the calibration clips' mean timing,
                   which sets the along anchor.
  speed MAE, accel difference   depend on timing once the speed changes, so they
                   are given at the lab timing and at the fitted tau.
Frame times count from the video's first frame. A clip whose first frame has a
later presentation time (AV_T_WE_3 starts at 0.967 s) also gets the offset on
the original timeline, in case the cut kept the source timestamps.
A clip with no image-identified car (HV_T_EW_2) is not graded; instead the GPS
car's time in view and inside the video hole is given for a few offsets.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/gps_benchmark.py
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/gps_benchmark.py --selfcheck
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import gps_to_image as g2i  # noqa: E402
from site_error_model import wrap  # noqa: E402

CLIPS = ("AV_T_EW_3", "AV_T_WE_3", "HV_T_EW_2", "HV_T_EW_1", "AV_T_WE_1")
ROLE = {"AV_T_EW_3": "held out", "AV_T_WE_3": "held out (local video is a 6 s cut)",
        "HV_T_EW_2": "held out (3 s video hole)",
        "HV_T_EW_1": "calibration (set height, pitch and the GPS transform)",
        "AV_T_WE_1": "calibration (set height, pitch and the GPS transform)"}
TAU_FLAG_S, TAU_MAX_S, MIN_FRAMES = 0.5, 4.0, 30
VIEW_X = (10.0, 140.0)     # along-road range where the detector can see a car


def diff_t(v, t, win=0.5, fps=30.0):
    """trajectory_features.diff on true times: +-round(win*fps/2) samples, clamped at the ends."""
    h = max(1, int(round(win * fps / 2)))
    i = np.arange(len(v))
    lo, hi = np.clip(i - h, 0, len(v) - 1), np.clip(i + h, 0, len(v) - 1)
    return (v[hi] - v[lo]) / np.maximum(t[hi] - t[lo], 1e-9)


def metrics(car, gps, T, tau, lanes=None):
    t = car["t"]
    m = (t + tau >= gps["t"][0]) & (t + tau <= gps["t"][-1])
    if m.sum() < MIN_FRAMES:
        return None
    q, vq, sc, t = car["q"][m], car["v"][m], car["speed"][m], t[m]
    xy, hd, sg = g2i.gps_road(gps, T, t + tau)
    d = q - xy
    dh = np.degrees(wrap(np.arctan2(vq[:, 1], vq[:, 0]) - hd))
    ac, ag = diff_t(sc, t), diff_t(sg, t)
    r = lambda x, k=3: round(float(x), k)
    out = {"frames": int(m.sum()), "t_range_s": [r(t[0], 2), r(t[-1], 2)],
           "speed_ratio": r(sc.mean() / sg.mean(), 4), "speed_mae_mps": r(np.abs(sc - sg).mean()),
           "lateral_bias_m": r(d[:, 1].mean()), "lateral_rms_m": r(np.sqrt((d[:, 1] ** 2).mean())),
           "lateral_std_m": r(d[:, 1].std()),
           "heading_bias_deg": r(dh.mean(), 2), "heading_mae_deg": r(np.abs(dh).mean(), 2),
           "accel_rms_camera": r(np.sqrt((ac ** 2).mean())), "accel_rms_gps": r(np.sqrt((ag ** 2).mean())),
           "accel_rms_diff": r(np.sqrt(((ac - ag) ** 2).mean())),
           "along_bias_m": r(d[:, 0].mean()), "along_rms_m": r(np.sqrt((d[:, 0] ** 2).mean())),
           "along_std_m": r(d[:, 0].std()), "speed_mps": r(sg.mean(), 2)}
    if lanes:
        cent = np.array([c for cs in lanes.values() for c in cs])
        for k, y in (("camera", q[:, 1]), ("gps", xy[:, 1])):
            c = cent[np.argmin(np.abs(cent - np.median(y)))]
            out[f"lane_offset_{k}_m"] = r(np.median(y) - c)
            out[f"lane_centre_{k}_m"] = r(c, 2)
    return out


def best_tau(car, gps, T, step=1 / 125):
    """Constant offset minimising along rms (needs MIN_FRAMES of overlap)."""
    best = (math.inf, None)
    for tau in np.arange(-TAU_MAX_S, TAU_MAX_S + step / 2, step):
        mm = metrics(car, gps, T, tau)
        if mm and mm["along_rms_m"] < best[0]:
            best = (mm["along_rms_m"], float(tau))
    return best[1]


def visibility(c, T, tau=0.0):
    """Where the GPS car is over the video span at the lab timing, on a 1/30 s time
    grid (frozen copies carry no time of their own, so the grid is used instead of
    the frames): seconds the GPS car is inside VIEW_X and in the picture, and how
    many of those fall in a hole (no real frame within 0.05 s)."""
    real = c["t"][~c["copy"]]
    t = np.arange(0, real[-1] + 1e-9, 1 / 30)
    ok = (t + tau >= c["gps"]["t"][0]) & (t + tau <= c["gps"]["t"][-1])
    xy, _, _ = g2i.gps_road(c["gps"], T, t + tau)
    px = g2i.to_px(np.c_[xy, np.full(len(t), g2i.ROAD_Z)], c["K"], c["l2c"])
    with np.errstate(invalid="ignore"):
        inview = ok & (xy[:, 0] > VIEW_X[0]) & (xy[:, 0] < VIEW_X[1]) & np.isfinite(px).all(1) \
            & (px[:, 0] >= 0) & (px[:, 0] < 1920) & (px[:, 1] >= 0) & (px[:, 1] < 1080)
    hole = np.min(np.abs(t[:, None] - real[None]), axis=1) > 0.05
    f = np.flatnonzero(inview)
    return {"video_span_s": round(float(real[-1]), 2), "seconds_in_view": round(inview.sum() / 30, 2),
            "seconds_in_view_inside_hole": round((inview & hole).sum() / 30, 2),
            "hole_s": round(hole.sum() / 30, 2),
            "in_view_from_to_s": [round(float(t[f[0]]), 2), round(float(t[f[-1]]), 2)] if len(f) else None,
            "gps_x_at_video_start_end_m": [round(float(xy[0, 0]), 1), round(float(xy[-1, 0]), 1)],
            "gps_y_median_m": round(float(np.median(xy[ok, 1])), 2),
            "seconds_to_cross_view": round((VIEW_X[1] - VIEW_X[0]) / float(np.median(c["gps"]["v"])), 2)}


def bench(clip, T):
    c = g2i.load_clip(clip)
    lanes = json.loads((ROOT / "outputs/trajectories" / f"{clip}_ft102" / "lanes.json").read_text())["centres_by_direction"]
    out = {"clip": clip, "role": ROLE.get(clip, "held out"),
           "site_transform": {"from_clips": T["clips"], "road_bearing_enu_deg": round(math.degrees(T["phi"]), 3)},
           "visibility_lab_timing": visibility(c, T)}
    if c["car"] is None:
        out["note"] = "no image-identified car (instrumented.json missing); nothing graded"
        out["visibility_by_offset"] = {f"{tau:+.1f}": visibility(c, T, tau) for tau in (-1.0, 0.0, 1.0, 2.0, 2.5)}
        return out
    out["vehicle"] = c["car"]["id"]
    tau = best_tau(c["car"], c["gps"], T)
    lab = metrics(c["car"], c["gps"], T, 0.0, lanes)
    fit = metrics(c["car"], c["gps"], T, tau, lanes) if tau is not None else None
    tf = ("speed_ratio", "lateral_bias_m", "lateral_rms_m", "lateral_std_m", "heading_bias_deg",
          "heading_mae_deg", "accel_rms_camera", "accel_rms_gps", "lane_offset_camera_m",
          "lane_offset_gps_m", "lane_centre_camera_m", "lane_centre_gps_m")
    src = lab or fit
    out["timing_free"] = {k: src[k] for k in tf if k in src}
    out["timing_free"]["evaluated_at"] = "lab timing" if lab else "fitted offset (lab timing overlaps too few frames)"
    pts0 = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                                 "-read_intervals", "%+#1", "-of", "csv=p=0", str(ROOT / "Camera data" / f"{clip}.mp4")],
                                capture_output=True, text=True).stdout.split()[0].strip(","))
    out["timing"] = {
        "best_offset_s": None if tau is None else round(tau, 3),
        "video_first_pts_s": round(pts0, 3),   # frame_times.json counts from here
        "best_offset_on_original_pts_s": None if tau is None else round(tau - pts0, 3),
        "offset_flag": None if tau is None else abs(tau) > TAU_FLAG_S,
        "offset_as_along_shift_m": None if tau is None else round(tau * fit["speed_mps"], 1),
        "lab_timing": None if lab is None else {k: lab[k] for k in
                                                ("frames", "along_bias_m", "along_rms_m", "speed_mae_mps", "accel_rms_diff")},
        "at_best_offset": None if fit is None else {k: fit[k] for k in
                                                    ("frames", "t_range_s", "along_bias_m", "along_rms_m", "along_std_m",
                                                     "speed_ratio", "speed_mae_mps", "accel_rms_diff", "lateral_bias_m")}}
    return out


ROW = "| {clip} | {role} | {veh} | {sr} | {lat} | {lane} | {hd} | {acc} | {tau} | {al0} | {al} |"


def table(results):
    lines = ["| Clip | Role | Car | Speed ratio | Lateral bias / rms (m) | Lane offset cam / GPS (m) | Heading bias / MAE (deg) "
             "| Accel rms cam / GPS / diff | Best offset (s) | Along at lab timing bias / rms (m) | Along after offset rms (m) |",
             "|" + "---|" * 11]
    for r in results:
        if "timing_free" not in r:
            v = r["visibility_lab_timing"]
            lines.append(ROW.format(clip=r["clip"], role=r["role"], veh="none", sr="", lat="", lane="", hd="", acc="",
                                    tau="", al0=f"GPS car in view {v['seconds_in_view']} s, {v['seconds_in_view_inside_hole']} s of it in the video hole", al=""))
            continue
        f, tm = r["timing_free"], r["timing"]
        lab, fit = tm["lab_timing"] or {}, tm["at_best_offset"] or {}
        lines.append(ROW.format(
            clip=r["clip"], role=r["role"].split(" (")[0], veh=r["vehicle"], sr=f["speed_ratio"],
            lat=f"{f['lateral_bias_m']:+.2f} / {f['lateral_rms_m']:.2f}",
            lane=f"{f.get('lane_offset_camera_m', 0):+.2f} / {f.get('lane_offset_gps_m', 0):+.2f}",
            hd=f"{f['heading_bias_deg']:+.2f} / {f['heading_mae_deg']:.2f}",
            acc=f"{f['accel_rms_camera']:.2f} / {f['accel_rms_gps']:.2f} / {fit.get('accel_rms_diff', float('nan')):.2f}",
            tau=f"{tm['best_offset_s']:+.2f}" + (" FLAG" if tm["offset_flag"] else ""),
            al0=f"{lab.get('along_bias_m', float('nan')):+.1f} / {lab.get('along_rms_m', float('nan')):.1f}",
            al=f"{fit.get('along_rms_m', float('nan')):.2f}"))
    return "\n".join(lines)


def _selfcheck():
    """Synthetic held-out car: straight lane, 28 m/s, GPS lab times 0.6 s late, camera
    0.3 m left of the GPS and 1% fast. Recovered: tau 0.6 (flagged), lateral +0.3,
    speed ratio 1.01, along residual ~0 after the offset, heading 0."""
    phi, C, v, tau = math.radians(-2.7), np.array([-230.0, 16.0]), 28.0, 0.6
    T = {"phi": phi, "C": C}
    tg = np.arange(0, 10, 1 / 125)
    xg = 10 + v * tg
    g = C + np.c_[xg, np.full_like(xg, -24.5)] @ g2i.rot(phi).T
    gps = {"t": tg + tau, "e": g[:, 0], "n": g[:, 1], "hx": np.full_like(tg, math.cos(phi)),
           "hn": np.full_like(tg, math.sin(phi)), "v": np.full_like(tg, v)}
    t = np.arange(30, 150) / 30.0
    car = {"t": t, "q": np.c_[10 + v * t, np.full_like(t, -24.2)], "v": np.c_[np.full_like(t, v), 0 * t],
           "speed": np.full_like(t, 1.01 * v)}
    got = best_tau(car, gps, T)
    assert abs(got - tau) < 0.01, got
    m = metrics(car, gps, T, got, {"1": [-24.5, -27.5]})
    assert abs(m["lateral_bias_m"] - 0.3) < 1e-3 and abs(m["speed_ratio"] - 1.01) < 1e-3, m
    assert m["along_rms_m"] < 0.1 and abs(m["heading_bias_deg"]) < 1e-6, m
    assert abs(m["lane_offset_camera_m"] - 0.3) < 1e-3 and m["lane_offset_gps_m"] == 0, m
    m0 = metrics(car, gps, T, 0.0)
    assert abs(m0["along_bias_m"] - v * tau) < 0.05, m0          # lab timing: the offset shows as v*tau
    assert abs(m0["lateral_bias_m"] - 0.3) < 1e-3                  # and lateral does not care
    tt = np.arange(60) / 31.0                                      # non-30 fps frame times
    assert np.allclose(diff_t(20 + 1.5 * tt, tt), 1.5)
    print(f"selfcheck ok (tau {got:.3f} s recovered, lateral, speed ratio, lane, accel)")


def main():
    ap = argparse.ArgumentParser("GPS benchmark per clip")
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    a = ap.parse_args()
    T = g2i.load_site()
    out = ROOT / "outputs/evaluation/gps_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for clip in a.clips:
        r = bench(clip, T)
        r["site_transform"]["calibration_clip_offsets_road_m"] = {k: [round(x, 2) for x in v]
                                                                  for k, v in T["per_clip_offset_road_m"].items()}
        (out / f"{clip}.json").write_text(json.dumps(r, indent=2))
        res.append(r)
    md = table(res)
    (out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
