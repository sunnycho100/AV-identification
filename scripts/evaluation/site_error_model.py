"""Site error model: grade camera tracks against GPS with shared physical parameters.

Why. scripts/tracking/grade_target_vs_gps.py fits a rotation and translation per
clip and reports the residual. Those are what a georeferenced calibration should
supply, so the per-clip fit hides calibration error, time-sync error and wrong
target picks, and grades path shape only. It also cannot represent the mirror
between the GPS frame (east, south) and the camera frame (forward, left):
harmless on straight paths, wrong on curves. It stays in the repo as the
path-alignment baseline.

This model has NO per-clip parameters. Every parameter is a physical quantity
shared across all clips of a site and each clip is graded with parameters
fitted on the OTHER clips (leave-one-clip-out):

    theta = [C_e, C_n, beta_P0, beta_P1, s, l_x, l_y]
    P_i = C + s * Rot(beta_p) q_i          camera prediction of the box centre
    G_i = g_i + Rot(psi_i) l               GPS prediction of the box centre
    r_i = P_i - G_i

  C       camera ground point, site ENU (m)
  beta_p  bearing of the camera-frame x axis, one per PTZ pose; clips that pass the
          2-degree pose gate in site_extrinsic.py share a pose
  s       range scale about the camera. To first order s = h_true / h_used, but
          each clip's own AnyCalib focal length also scales range, so s is a
          candidate explanation to test against the per-clip fx, not a height.
  l       GPS antenna to box centre, vehicle body frame (forward, left)
  q_i     camera-frame track position (x forward, y left)
  g_i     GPS position in site ENU at the frame time; psi_i vehicle heading,
          ENU, counter-clockwise from east

Time offset is not a parameter: on constant-speed clips it is indistinguishable
from l_x. The frame-to-GPS association comes from the lab's estimated video
timeline and is NOT independently verified here; a per-clip offset shows up as
an along-road disagreement in the per-clip standalone camera positions.

Frames the tracker coasted (no detection) and frames whose image is a duplicate
of the previous one (constant-rate extraction over dropped frames) are excluded
from every fit and reported as invalid.

A hold-out is graded only when the other clips cover its pose and its drive
direction; otherwise the lever arm (or the bearing) would be silently absorbed
and the score would be meaningless. Such hold-outs are reported as skipped.

GPS is read only here and only for grading. It never feeds calibration or the
choice of target track.

    .venv/bin/python scripts/evaluation/site_error_model.py \
        --out outputs/evaluation/site_T_report.json
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
R_EARTH = 6378137.0
FPS = 30.0
POSES = ("P0", "P1")
NAMES = ["C_e", "C_n", "beta_P0", "beta_P1", "s", "l_x", "l_y"]
P_S, P_LX = 4, 5
UNPICKED = ("PICK_FROM_IMAGE", "UNIDENTIFIED")


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fold(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


WGS84_E2 = 6.69437999014e-3   # first eccentricity squared


def radii(lat0):
    """WGS84 meridian (north) and prime-vertical (east) radii of curvature at lat0, m.
    Using the equatorial radius for both, as this file did before 2026-10-06, made
    east distances 0.16% short and north 0.20% long at Madison (0.2 m rms, 0.5 m at
    300 m against pymap3d.geodetic2enu)."""
    w = 1 - WGS84_E2 * math.sin(math.radians(lat0)) ** 2
    return R_EARTH * (1 - WGS84_E2) / w ** 1.5, R_EARTH / math.sqrt(w)


def enu_from_latlon(lat, lon, lat0, lon0):
    """Local tangent plane in metres, ellipsoidal radii at lat0. Over this 300 m site it
    matches pymap3d.geodetic2enu within 1 cm (height is ignored: 5 m moves it 0.2 mm)."""
    M, N = radii(lat0)
    e = N * np.radians(np.asarray(lon) - lon0) * math.cos(math.radians(lat0))
    n = M * np.radians(np.asarray(lat) - lat0)
    return e, n


def load_gps(csv_path, lat0, lon0):
    # keep every row with a video time, including rows outside the video window
    # (the CSV is cropped to the window plus about half a second)
    rows = [r for r in csv.DictReader(open(csv_path)) if r["video_time_sec"]]
    col = lambda k: np.array([float(r[k]) for r in rows])
    e, n = enu_from_latlon(col("latitude"), col("longitude"), lat0, lon0)
    # The lab's frame times use each clip's true frame rate (30, 31 or 29.97 fps);
    # recover it from the CSV instead of assuming 30.
    fr = [(int(r["nearest_frame_index"]), float(r["nearest_frame_time_sec"]))
          for r in rows if r["nearest_frame_index"] and int(r["nearest_frame_index"]) > 0]
    fps = float(np.median([i / t for i, t in fr])) if fr else FPS
    # processed_heading_y is the SOUTH component; negate to make the frame right-handed
    return {"t": col("video_time_sec"), "e": e, "n": n, "fps": fps,
            "hx": col("processed_heading_x"), "hn": -col("processed_heading_y"),
            "v": np.hypot(col("east_velocity"), col("north_velocity"))}


def gps_at(gps, t):
    """Position (N,2), heading psi (N,), speed (N,) interpolated at video times t."""
    i = lambda k: np.interp(t, gps["t"], gps[k])
    return np.stack([i("e"), i("n")], 1), np.arctan2(i("hn"), i("hx")), i("v")


def load_track(path, track_id=None):
    d = json.load(open(path))
    if "states" in d:
        st = d["states"]
    else:   # one id, or a list of ids when the tracker split the target's track
        ids = track_id if isinstance(track_id, list) else [track_id]
        st = sorted((s for i in ids for s in d["tracks"][str(i)]), key=lambda s: s["frame"])
    fr = np.array([s["frame"] for s in st], float)
    xy = np.array([[s["x"], s["y"]] for s in st], float)
    yaw = np.array([s["yaw"] for s in st], float)
    sc = np.array([s["score"] for s in st], float)
    vx = np.array([s["vx"] for s in st], float)
    # AB3DMOT repeats the previous score and velocity verbatim on frames with no detection
    coasted = np.r_[False, (sc[1:] == sc[:-1]) & (vx[1:] == vx[:-1])]
    return fr, xy, yaw, coasted


def frozen_frames(frames_dir, fr):
    """Frames whose JPEG is byte-identical to the previous frame: constant-rate
    extraction duplicates a frame wherever the stream dropped one, so the detector
    sees the same image twice and the track stalls for a frame."""
    fr = fr.astype(int)
    digest = {}
    for k in np.r_[fr, fr - 1]:
        p = Path(frames_dir) / f"{k:03d}.jpg"
        if p.exists():
            digest[k] = hashlib.md5(p.read_bytes()).hexdigest()
    return np.array([digest.get(k) is not None and digest.get(k) == digest.get(k - 1)
                     for k in fr])


def build_clip(name, pose, gps, fr, xy, yaw, invalid, fx=None):
    t = fr / gps.get("fps", FPS)
    g, psi, v = gps_at(gps, t)
    return {"name": name, "pose": pose, "t": t, "q": xy, "yaw": yaw,
            "g": g, "psi": psi, "v": v, "invalid": invalid, "fx": fx, "gps": gps}


def predict(theta, c):
    beta = theta[2 + POSES.index(c["pose"])]
    p_cam = theta[:2] + theta[P_S] * (c["q"] @ rot(beta).T)
    lx, ly = theta[P_LX], theta[P_LX + 1]
    cp, sp = np.cos(c["psi"]), np.sin(c["psi"])
    p_gps = c["g"] + np.stack([cp * lx - sp * ly, sp * lx + cp * ly], 1)
    return p_cam, p_gps


def residuals(theta, clips):
    out = []
    for c in clips:
        p_cam, p_gps = predict(theta, c)
        out.append((p_cam - p_gps)[~c["invalid"]].ravel())
    return np.concatenate(out)


def heading_of(c):
    return float(np.median(c["psi"]))


def same_direction(a, b):
    return abs(wrap(heading_of(a) - heading_of(b))) < np.pi / 2


def free_mask(clips):
    """Which parameters the data can constrain. A pose with no clips has no beta.
    The lever arm separates from the camera position only when the vehicle
    headings span both drive directions: the heading flip changes the sign of
    Rot(psi) l while C stays put."""
    poses = {c["pose"] for c in clips}
    m = np.ones(7, bool)
    for pi_, pz in enumerate(POSES):
        m[2 + pi_] = pz in poses
    m[P_LX:] = any(not same_direction(a, b) for a in clips for b in clips)
    return m


def initial_theta(clips):
    theta = np.zeros(7)
    theta[P_S] = 1.0
    for pi_, pz in enumerate(POSES):
        cs = [c for c in clips if c["pose"] == pz]
        if not cs:
            continue
        c0 = cs[0]                            # one clip fixes the branch; all share the pose
        psi = heading_of(c0)
        best = None
        for beta in (psi, psi + np.pi):      # camera x axis runs along or against travel
            th = theta.copy()
            th[2 + pi_] = beta
            th[:2] = c0["g"].mean(0) - (c0["q"] @ rot(beta).T).mean(0)
            cost = float(np.sum(residuals(th, [c0]) ** 2))
            if best is None or cost < best[0]:
                best = (cost, th)
        theta = best[1]
    return theta


def numeric_jac(f, x, eps=1e-6):
    """Central-difference Jacobian of the UNWEIGHTED residuals. least_squares
    returns a row-weighted Jacobian under a robust loss; the weighting cannot
    change the null space but does change the singular values."""
    f0 = f(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        J[:, i] = (f(x + d) - f(x - d)) / (2 * eps)
    return J


def fit(clips, theta0=None, free=None):
    """Robust least squares over the parameters marked free; the rest stay at theta0.
    Returns singular values of the plain Jacobian over the free parameters and the
    least-observable direction, so a degenerate fit is visible instead of drifting."""
    theta0 = initial_theta(clips) if theta0 is None else np.asarray(theta0, float)
    free = np.ones(7, bool) if free is None else np.asarray(free, bool)

    def full(p):
        th = theta0.copy()
        th[free] = p
        return th

    f = lambda p: residuals(full(p), clips)
    # s > 0 removes the exact symmetry (s, beta) == (-s, beta + pi)
    lo, hi = np.full(7, -np.inf), np.full(7, np.inf)
    lo[P_S], hi[P_S] = 0.5, 2.0
    res = least_squares(f, theta0[free], bounds=(lo[free], hi[free]),
                        loss="soft_l1", f_scale=1.0)
    _, sv, vt = np.linalg.svd(numeric_jac(f, res.x))
    null = np.zeros(7)
    null[free] = vt[-1]
    return {"theta": full(res.x), "sv": sv, "null_dir": null, "free": free,
            "n": res.fun.size // 2}


def evaluate(theta, c, mask):
    """Residual statistics on the frames selected by mask. Sign: camera minus GPS,
    along positive means the camera places the car ahead along its travel direction,
    across positive means to the left of travel."""
    p_cam, p_gps = predict(theta, c)
    r = (p_cam - p_gps)[mask]
    d = np.stack([np.cos(c["psi"]), np.sin(c["psi"])], 1)[mask]
    along = (r * d).sum(1)
    across = (r * np.stack([-d[:, 1], d[:, 0]], 1)).sum(1)
    e = np.linalg.norm(r, axis=1)
    beta = theta[2 + POSES.index(c["pose"])]
    dh = wrap(c["yaw"][mask] + beta - c["psi"][mask])
    return {"n": int(mask.sum()),
            "rmse_m": rms(e), "median_m": float(np.median(e)),
            "p95_m": float(np.percentile(e, 95)),
            "along_bias_m": float(along.mean()), "along_rms_m": rms(along),
            "across_bias_m": float(across.mean()), "across_rms_m": rms(across),
            "heading_bias_deg": float(np.degrees(np.median(dh))),
            "heading_fold_mae_deg": float(np.degrees(np.abs(fold(dh)).mean())),
            "n_heading_flipped": int((np.abs(dh) > np.pi / 2).sum())}


def kinematics(c, window=31):
    """Speed and longitudinal acceleration from the camera track alone: no
    alignment, no fitted parameters. speed_ratio is a direct estimate of s.
    Per-frame box positions are noisy, so the derivative window is 1 s; the
    acceleration number is mostly a noise floor on these constant-speed clips."""
    dt = float(np.median(np.diff(c["t"])))
    if np.any(np.diff(c["t"]) > 1.5 * dt):
        return {"skipped": "track has frame gaps; the derivative filter needs uniform spacing"}
    window = window if len(c["q"]) >= window else 15
    if len(c["q"]) < window:
        return {"skipped": "track shorter than the derivative window"}
    vel = savgol_filter(c["q"], window, 2, deriv=1, delta=dt, axis=0)
    acc = savgol_filter(c["q"], window, 2, deriv=2, delta=dt, axis=0)
    v_cam = np.linalg.norm(vel, axis=1)
    a_cam = (vel * acc).sum(1) / np.maximum(v_cam, 1e-6)
    a_gps = np.gradient(c["v"], c["t"])
    m = ~c["invalid"]
    return {"speed_ratio_gps_over_cam": float(c["v"][m].mean() / v_cam[m].mean()),
            "speed_rmse_mps": rms(v_cam[m] - c["v"][m]),
            "accel_rmse_mps2": rms(a_cam[m] - a_gps[m])}


def standalone(c, lat0, lon0):
    """One clip alone: C, beta, s with the lever arm fixed at zero. NOT a grade (it
    fits the clip it scores). A consistency check: every clip from one camera must
    put C at the same place. Disagreement along the road with across-road agreement
    means a time-sync error in that clip's CSV or the wrong target vehicle in the
    same lane; the camera coordinates are returned so a map can arbitrate."""
    m = free_mask([c])
    m[P_LX:] = False
    th = fit([c], free=m)["theta"]
    M, N = radii(lat0)
    lat = lat0 + math.degrees(th[1] / M)
    lon = lon0 + math.degrees(th[0] / (N * math.cos(math.radians(lat0))))
    ev = evaluate(th, c, ~c["invalid"])
    return {"C_e": float(th[0]), "C_n": float(th[1]),
            "camera_lat": float(lat), "camera_lon": float(lon),
            "beta_deg": float(np.degrees(wrap(th[2 + POSES.index(c["pose"])]))),
            "s": float(th[P_S]), "fx_used": c.get("fx"),
            "n": ev["n"], "n_invalid_frames": int(c["invalid"].sum()), "rmse_m": ev["rmse_m"],
            "along_rms_m": ev["along_rms_m"], "across_rms_m": ev["across_rms_m"],
            "heading_bias_deg": ev["heading_bias_deg"],
            "speed_ratio_gps_over_cam": kinematics(c).get("speed_ratio_gps_over_cam")}


def hold_out_reason(c, others):
    if not others:
        return "no other clip"
    if not any(o["pose"] == c["pose"] for o in others):
        return "no training clip with this camera pose"
    if not any(same_direction(c, o) for o in others):
        return "no training clip in this drive direction; the lever arm would be absorbed"
    return None


def loco(clips):
    out = {}
    for c in clips:
        others = [o for o in clips if o is not c]
        why = hold_out_reason(c, others)
        if why:
            out[c["name"]] = {"skipped": why}
            continue
        f = fit(others, free=free_mask(others))
        out[c["name"]] = {
            "theta_from_other_clips": dict(zip(NAMES, f["theta"].tolist())),
            "free_parameters": [n for n, k in zip(NAMES, f["free"]) if k],
            "sv_min_over_max": float(f["sv"][-1] / f["sv"][0]),
            "all_frames": evaluate(f["theta"], c, np.ones(len(c["q"]), bool)),
            "excluding_invalid": evaluate(f["theta"], c, ~c["invalid"]),
            "fit_free": kinematics(c)}
    return out


def load_clips(cfg):
    clips, excluded = [], {}
    for c in cfg["clips"]:
        if c.get("track_id") in UNPICKED:
            excluded[c["name"]] = c.get("identified_from", "no target track chosen")
            continue
        gps = load_gps(ROOT / "Camera data" / f"{c['name']}_trajectory.csv",
                       cfg["lat0"], cfg["lon0"])
        fx = None
        run = c["track"].split("/")[-2].replace(c["name"] + "_", "", 1)   # e.g. phase1, r140
        cal = ROOT / f"outputs/object_detection/camera-data/{c['name']}_{run}/calibration_used.json"
        if cal.exists():
            fx = float(json.load(open(cal))["K"][0][0])
        fr, xy, yaw, coasted = load_track(ROOT / c["track"], c.get("track_id"))
        frozen = frozen_frames(ROOT / "data/camera-data" / c["name"] / "frames_all", fr)
        clips.append(build_clip(c["name"], c["pose"], gps, fr, xy, yaw,
                                coasted | frozen, fx=fx))
    return clips, excluded


def main():
    ap = argparse.ArgumentParser("Shared-parameter site error model, leave-one-clip-out")
    ap.add_argument("--config", default=str(ROOT / "scripts/evaluation/site_T.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    clips, excluded = load_clips(cfg)
    free = free_mask(clips)
    f = fit(clips, free=free)
    report = {"site": cfg["site"], "clips": [c["name"] for c in clips],
              "excluded": excluded,
              "per_clip_standalone": {c["name"]: standalone(c, cfg["lat0"], cfg["lon0"])
                                      for c in clips},
              "free_parameters": [n for n, k in zip(NAMES, free) if k],
              "theta_all_clips": dict(zip(NAMES, f["theta"].tolist())),
              "scale_times_height_used_m": float(f["theta"][P_S] * cfg["height_used_m"]),
              "jacobian_singular_values": f["sv"].tolist(),
              "least_observable_direction": dict(zip(NAMES, f["null_dir"].tolist())),
              "in_sample": {c["name"]: evaluate(f["theta"], c, ~c["invalid"]) for c in clips},
              "leave_one_clip_out": loco(clips)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
