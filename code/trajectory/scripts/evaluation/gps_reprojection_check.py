"""Where does the box sit relative to the car, in metres and in pixels?

The comparison video shows boxes that lag or lead the car along the road and
sometimes sit off it. Three explanations give the same visual symptom:

  a range scale error   the box is wrong by a fraction of its range, so the
                        along-road offset grows with distance (this is what a
                        camera height error looks like: to first order the
                        fitted range scale s is h_true / h_used)
  a time offset         the box is wrong by speed times the offset, so the
                        along-road offset is constant in metres at every range
  detector noise        no bias at all, only spread

This script separates them. Per clip it fits the standalone site model from
site_error_model (camera position C, bearing beta, range scale s, lever arm
fixed at zero), then puts the GPS samples into the camera ground frame with the
scale DELIBERATELY NOT APPLIED:

    q_gps = Rot(beta)^T (g - C)          s = 1

so q_gps is where the car truly is in the camera frame given that camera
position and bearing, and the box minus q_gps carries the whole range error.
The fitted s is reported separately because it is the height ratio, not part of
the conversion. Note that C absorbs any constant along-road shift, so the
constant term is not identifiable here and only the per-range-bin STRUCTURE is
evidence: a scale error separates the bins, a time offset does not.

Sign convention. along is positive when the box is ahead of the car along its
travel direction, across positive to the left of travel, the same convention as
site_error_model.evaluate. Pixel offsets are box minus GPS: du positive is
right in the image, dv positive is down.

Frames. The target track only, the instrumented vehicle, identified from the
image alone. Frames the tracker coasted and frames whose JPEG duplicates the
previous one are dropped, and a frame is used only when a RAW detection sits
within 1 m of the track state, so the numbers describe what the detector
produced rather than what the Kalman filter interpolated.

Caveat. One instrumented car per clip, about 100 usable frames each, two clips.
This measures the bias on one vehicle on one lane, not a population.

GPS is read here and only for grading. It never feeds calibration or tracking.

    .venv/bin/python scripts/evaluation/gps_reprojection_check.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import site_error_model as sem

CLIPS = ("AV_T_EW_3", "HV_T_EW_1")
RUN = "phase1"                    # the current phase1 runs are the 140.8 m checkpoint
BINS = (("0_40", 0.0, 40.0), ("40_60", 40.0, 60.0), ("60_inf", 60.0, np.inf))
ROAD_Z = -1.73                    # the training data's ground plane, as exported
MATCH_M = 1.0                     # same raw-detection match radius as score_heading
LAT0, LON0, HEIGHT_USED_M = 43.0353, -89.4215, 16.26   # from site_T.json
OUT_DIR = ROOT / "outputs/evaluation/reprojection"
N_RENDER = 4


def to_camera_frame(g, C, beta):
    """GPS ENU to the camera ground frame, scale NOT applied (s = 1)."""
    return (np.asarray(g, float) - np.asarray(C, float)) @ sem.rot(beta)


def along_across(q_box, q_gps, psi, beta):
    """Box minus GPS, resolved on the car's travel direction in the camera frame."""
    d = np.stack([np.cos(psi - beta), np.sin(psi - beta)], 1)
    r = q_box - q_gps
    return (r * d).sum(1), (r * np.stack([-d[:, 1], d[:, 0]], 1)).sum(1)


def to_pixel(xy, K, l2c, z=ROAD_Z):
    """Ground-frame points at the road plane to image pixels."""
    xy = np.asarray(xy, float)
    p = np.c_[xy, np.full(len(xy), z), np.ones(len(xy))]
    cam = (np.asarray(l2c) @ p.T).T[:, :3]
    uv = (np.asarray(K) @ cam.T).T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def binstats(v, rng):
    """median and IQR per range bin; bins are on range from the camera."""
    out = {}
    for name, lo, hi in BINS:
        m = (rng >= lo) & (rng < hi)
        if not m.sum():
            out[name] = {"n": 0}
            continue
        x = v[m]
        q1, q3 = np.percentile(x, [25, 75])
        out[name] = {"n": int(m.sum()), "median": round(float(np.median(x)), 3),
                     "iqr": round(float(q3 - q1), 3),
                     "q25": round(float(q1), 3), "q75": round(float(q3), 3)}
    return out


def load_clip(clip):
    """Target track, matched raw detections, GPS, and the usable frame mask."""
    gps = sem.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv", LAT0, LON0)
    trk = ROOT / f"outputs/tracking/camera-data/{clip}_{RUN}/target_track.json"
    fr, xy, yaw, coasted = sem.load_track(trk)
    frozen = sem.frozen_frames(ROOT / "data/camera-data" / clip / "frames_all", fr)
    c = sem.build_clip(clip, "P0", gps, fr, xy, yaw, coasted | frozen)

    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_{RUN}"
    dets = []
    for f, (x, y) in zip(fr.astype(int), xy):
        p, best = det_dir / f"{f:03d}_pred.json", None
        if p.exists():
            for d in json.load(open(p)):
                r = float(np.hypot(d["x"] - x, d["y"] - y))
                if r <= MATCH_M and (best is None or r < best[0]):
                    best = (r, d)
        dets.append(None if best is None else best[1])
    inside = (c["t"] >= gps["t"].min()) & (c["t"] <= gps["t"].max())
    keep = ~c["invalid"] & inside & np.array([d is not None for d in dets])
    return c, det_dir, dets, keep, fr.astype(int)


def analyse(clip):
    c, det_dir, dets, keep, frames = load_clip(clip)
    st = sem.standalone(c, LAT0, LON0)
    C, beta = np.array([st["C_e"], st["C_n"]]), np.radians(st["beta_deg"])
    s = st["s"]

    q_box = np.array([[d["x"], d["y"]] for d in np.array(dets, object)[keep]], float)
    q_gps = to_camera_frame(c["g"][keep], C, beta)
    psi, v = c["psi"][keep], c["v"][keep]
    rng = np.hypot(q_box[:, 0], q_box[:, 1])

    along, across = along_across(q_box, q_gps, psi, beta)
    # (5) the same offsets with the box pulled toward the camera by the fitted
    # scale, which is what rerunning the detector at the implied height would do.
    # s < 1 here, so q * s is the correcting direction; q / s is reported too
    # because the task named it and it is the opposite direction.
    a_times, _ = along_across(q_box * s, q_gps, psi, beta)
    a_over, _ = along_across(q_box / s, q_gps, psi, beta)

    cal = json.load(open(det_dir / "calibration_used.json"))
    K, l2c = np.array(cal["K"]), np.array(cal["lidar2cam"])
    uv_box, _ = to_pixel(q_box, K, l2c)
    uv_gps, _ = to_pixel(q_gps, K, l2c)
    duv = uv_box - uv_gps
    # how many pixels one metre along the road is worth here: the bridge between
    # the metre tables and what the video shows
    d1 = np.stack([np.cos(psi - beta), np.sin(psi - beta)], 1)
    uv_1m, _ = to_pixel(q_gps + d1, K, l2c)
    px_per_m = np.linalg.norm(uv_1m - uv_gps, axis=1)

    rows = {"frames": frames[keep], "q_box": q_box, "q_gps": q_gps, "rng": rng,
            "along": along, "across": across, "duv": duv,
            "dets": [d for d, k in zip(dets, keep) if k], "K": K, "l2c": l2c}
    res = {
        "run": RUN, "checkpoint": cal["checkpoint"],
        "n_track_frames": int(len(frames)), "n_used": int(keep.sum()),
        "n_dropped_coasted_or_frozen_or_unmatched": int((~keep).sum()),
        "standalone_fit": {k: st[k] for k in
                           ("C_e", "C_n", "camera_lat", "camera_lon", "beta_deg",
                            "s", "rmse_m", "along_rms_m", "across_rms_m")},
        "implied_camera_height_m": round(s * HEIGHT_USED_M, 3),
        "height_used_m": HEIGHT_USED_M,
        "mean_speed_mps": round(float(v.mean()), 3),
        "range_span_m": [round(float(rng.min()), 1), round(float(rng.max()), 1)],
        "metres": {"along": binstats(along, rng), "across": binstats(across, rng)},
        "metres_scaled_q_times_s": {"along": binstats(a_times, rng)},
        "metres_scaled_q_over_s": {"along": binstats(a_over, rng)},
        "pixels": {"du": binstats(duv[:, 0], rng), "dv": binstats(duv[:, 1], rng),
                   "px_per_m_along": binstats(px_per_m, rng)},
        "along_profile_10m": {f"{lo}_{lo + 10}": round(float(np.median(along[m])), 2)
                              for lo in range(0, 140, 10)
                              if (m := (rng >= lo) & (rng < lo + 10)).sum() >= 3},
        "overall": {
            "along_median_m": round(float(np.median(along)), 3),
            "along_median_m_q_times_s": round(float(np.median(a_times)), 3),
            "along_abs_median_m": round(float(np.median(np.abs(along))), 3),
            "along_abs_median_m_q_times_s": round(float(np.median(np.abs(a_times))), 3),
            "across_median_m": round(float(np.median(across)), 3),
            "du_median_px": round(float(np.median(duv[:, 0])), 2),
            "dv_median_px": round(float(np.median(duv[:, 1])), 2),
            "along_vs_range_slope_m_per_m": round(
                float(np.polyfit(rng, along, 1)[0]), 5),
            "implied_time_offset_s_if_constant": round(
                float(np.median(along) / v.mean()), 4)},
    }
    return res, rows


def render(clip, rows, out_dir):
    """4 frames spread across the track: the raw box, and the GPS point as a disc."""
    import cv2
    # note: evaluators.result2kitti has the same corner helper but importing it
    # pulls in numba, which .venv does not have. Five lines instead of a GPU stack.
    from scripts.data_converter.visual_utils import draw_box_3d, project_to_image

    def corners_of(d):
        l, w, h = d["l"], d["w"], d["h"]
        c, s = np.cos(d["yaw"]), np.sin(d["yaw"])
        u = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        xy = u @ np.array([[c, s], [-s, c]]) + np.array([d["x"], d["y"]])
        return np.c_[np.tile(xy, (2, 1)), np.r_[np.full(4, d["z"]), np.full(4, d["z"] + h)]]

    K34 = np.zeros((3, 4))
    K34[:3, :3] = rows["K"]
    l2c = rows["l2c"]
    uv_b, _ = to_pixel(rows["q_box"], rows["K"], l2c)
    uv_g, _ = to_pixel(rows["q_gps"], rows["K"], l2c)
    inside = np.all((uv_b > 60) & (uv_g > 60) & (uv_b < [1860, 1000])
                    & (uv_g < [1860, 1000]), axis=1)
    ok = np.flatnonzero(inside)
    idx = ok[np.linspace(0, len(ok) - 1, N_RENDER).round().astype(int)]
    made = []
    for i in idx:
        f = int(rows["frames"][i])
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        if img is None:
            continue
        d = rows["dets"][i]
        corners = corners_of(d)
        cc = (l2c @ np.c_[corners, np.ones(8)].T).T[:, :3]
        if (cc[:, 2] > 1e-6).sum() >= 4:
            draw_box_3d(img, project_to_image(cc, K34), c=(0, 220, 0))
        (ub, vb), = to_pixel(rows["q_box"][i:i + 1], rows["K"], l2c)[0]
        (ug, vg), = to_pixel(rows["q_gps"][i:i + 1], rows["K"], l2c)[0]
        cv2.circle(img, (int(round(ug)), int(round(vg))), 7, (0, 0, 255), -1)
        cv2.line(img, (int(ub), int(vb)), (int(ug), int(vg)), (255, 255, 0), 2)
        cv2.circle(img, (int(round(ub)), int(round(vb))), 4, (0, 220, 0), -1)

        # zoom inset, otherwise a few pixels at 120 m are invisible
        h = 140
        cx, cy = int((ub + ug) / 2), int((vb + vg) / 2)
        x0, y0 = max(0, cx - h), max(0, cy - h)
        crop = img[y0:y0 + 2 * h, x0:x0 + 2 * h]
        if crop.size:
            z = cv2.resize(crop, (3 * crop.shape[1], 3 * crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            z = z[:520, :520]
            img[560:560 + z.shape[0], 1920 - 20 - z.shape[1]:1920 - 20] = z
            cv2.rectangle(img, (1920 - 20 - z.shape[1], 560),
                          (1920 - 20, 560 + z.shape[0]), (255, 255, 255), 2)
        cv2.rectangle(img, (0, 0), (1920, 120), (0, 0, 0), -1)
        cv2.putText(img, f"{clip} frame {f}   green box = detection, "
                         f"red disc = GPS on the road plane   (inset 3x)",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"range {rows['rng'][i]:.1f} m   along {rows['along'][i]:+.2f} m   "
                         f"across {rows['across'][i]:+.2f} m   "
                         f"du {rows['duv'][i, 0]:+.1f} px   dv {rows['duv'][i, 1]:+.1f} px",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        p = out_dir / f"{clip}_{f:03d}.jpg"
        cv2.imwrite(str(p), img)
        made.append(str(p.relative_to(ROOT)))
    return made


def table(rep):
    w = lambda b: "  n/a " if b["n"] == 0 else f"{b['median']:+6.2f} ({b['iqr']:.2f})"
    lines = []
    for clip, r in rep["clips"].items():
        f = r["standalone_fit"]
        lines.append(
            f"\n{clip}  n={r['n_used']} of {r['n_track_frames']} frames  "
            f"range {r['range_span_m'][0]}-{r['range_span_m'][1]} m  "
            f"speed {r['mean_speed_mps']:.1f} m/s")
        lines.append(f"  standalone fit: beta {f['beta_deg']:+.2f} deg   s {f['s']:.4f}"
                     f"   implied height {r['implied_camera_height_m']:.2f} m "
                     f"(used {r['height_used_m']})   rmse {f['rmse_m']:.2f} m")
        lines.append(f"  {'quantity':<34}" + "".join(f"{b:>16}" for b, _, _ in BINS))
        for label, d in (("px per m along road", r["pixels"]["px_per_m_along"]),
                         ("along m, median (IQR)", r["metres"]["along"]),
                         ("across m", r["metres"]["across"]),
                         ("along m after q*s", r["metres_scaled_q_times_s"]["along"]),
                         ("along m after q/s", r["metres_scaled_q_over_s"]["along"]),
                         ("du px", r["pixels"]["du"]),
                         ("dv px", r["pixels"]["dv"])):
            lines.append(f"  {label:<34}" + "".join(f"{w(d[b]):>16}" for b, _, _ in BINS))
        o = r["overall"]
        lines.append(f"  along vs range slope {o['along_vs_range_slope_m_per_m']:+.4f} m/m"
                     f"   |along| median {o['along_abs_median_m']:.2f} m -> "
                     f"{o['along_abs_median_m_q_times_s']:.2f} m after q*s"
                     f"   constant-offset reading {o['implied_time_offset_s_if_constant']:+.3f} s")
    return "\n".join(lines)


def _self_check():
    C, beta, s = np.array([10.0, -4.0]), 0.7, 0.96
    q = np.array([[30.0, 1.0], [80.0, -2.0], [120.0, 0.5]])
    g = C + s * (q @ sem.rot(beta).T)                    # GPS built from the model
    q_gps = to_camera_frame(g, C, beta)
    assert np.allclose(q_gps, s * q), q_gps               # s = 1 inversion, so s*q

    psi = np.full(3, beta)                               # car travels along camera +x
    a, x = along_across(q_gps + np.array([2.0, 0.0]), q_gps, psi, beta)
    assert np.allclose(a, 2.0) and np.allclose(x, 0.0), (a, x)
    a, x = along_across(q_gps + np.array([0.0, 1.0]), q_gps, psi, beta)
    assert np.allclose(a, 0.0) and np.allclose(x, 1.0), (a, x)

    # the box reads 1/s too far, so q*s removes the along error and q/s doubles it
    a0, _ = along_across(q, q_gps, psi, beta)
    a1, _ = along_across(q * s, q_gps, psi, beta)
    a2, _ = along_across(q / s, q_gps, psi, beta)
    assert np.all(a0 > 0) and np.allclose(a1, 0.0, atol=1e-9)
    assert np.all(np.abs(a2) > np.abs(a0)), (a0, a2)

    b = binstats(np.array([1.0, 2.0, 3.0, 9.0]), np.array([10.0, 20.0, 50.0, 70.0]))
    assert b["0_40"]["n"] == 2 and b["0_40"]["median"] == 1.5 and b["0_40"]["iqr"] == 0.5
    assert b["60_inf"]["n"] == 1 and b["60_inf"]["median"] == 9.0

    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    l2c = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    uv, z = to_pixel(np.array([[50.0, 0.0], [50.0, 1.0]]), K, l2c)
    assert np.allclose(z, 50.0)
    assert np.allclose(uv[0], [960.0, 540 + 1000 * 1.73 / 50])
    assert np.allclose(uv[1] - uv[0], [-20.0, 0.0])      # +1 m left is 20 px left


def main():
    ap = argparse.ArgumentParser("Box vs GPS, in metres and pixels, per range bin")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--out", default=str(OUT_DIR / "summary.json"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    _self_check()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"run": RUN, "road_plane_z": ROAD_Z, "match_radius_m": MATCH_M,
           "scale_applied_to_gps": 1.0,
           "caveat": "one instrumented car per clip, about 100 usable frames each",
           "clips": {}}
    for clip in args.clips:
        r, rows = analyse(clip)
        if not args.no_render:
            r["renders"] = render(clip, rows, out_dir)
        rep["clips"][clip] = r
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(table(rep))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
