"""Camera height (and a small tilt correction) from GPS distances between frames.

Why. The vanishing point fixes pitch only to about 0.25 deg (the lines disagree
by ~12 px depending on which carriageway they come from), and no GPS-free route
fixes height reliably at every site. The instrumented car gives metric
distances: for two frames i, j the GPS says how far it moved, D_ij, and the image
says where it touched the road, p_i and p_j. With K and the VP rotation, a road
pixel back-projects to h * g(p), so the image distance scales with h.

Distances between frames are used, never absolute positions, so the fit does not
depend on the camera's map position, the road bearing, a constant GPS antenna
offset, or (at steady speed) the per-clip time offset. It uses GPS for scale
only, which makes this a calibration input: declare the clips it is fitted on
and grade on the others.

Unknowns: height h, pitch correction dp, and optionally roll dr (about the
optical axis). Residual per pair: |G_i - G_j| - D_ij, weighted by the pair's
expected metric error from 1 px of click error, so far frames count less.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/calibration/fit_height_from_gps_distances.py \
        --clip HV_T_EW_1 --clicks ~/Downloads/contact_clicks_HV_T_EW_1.json
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
CLICK_PX = 1.0     # assumed click error in full-frame pixels


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ground(uv, K, R, h, dp=0.0, dr=0.0):
    """Road-plane (x, y) under pixels uv (N, 2) for a camera h above the road.

    dp tilts about the camera's x axis, dr rolls about its optical axis; both are
    applied in the camera frame on top of R (ground -> camera).
    """
    Rc = rot_z(dr) @ rot_x(dp) @ R
    rays = np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T      # camera frame
    d = Rc.T @ rays                                               # ground frame
    s = -h / d[2]
    return np.c_[s * d[0], s * d[1]]


def pair_metric_sigma(uv, K, R, h):
    """Metric error of each pair distance from CLICK_PX of error at both ends."""
    g0 = ground(uv, K, R, h)
    sig = np.zeros(len(uv))
    for k in (np.array([CLICK_PX, 0.0]), np.array([0.0, CLICK_PX])):
        sig = np.maximum(sig, np.linalg.norm(ground(uv + k, K, R, h) - g0, axis=1))
    return sig


def fit(uv, D_pairs, K, R, h0=16.0, with_roll=False):
    """Least squares over all pairs. Returns dict with estimates and 1-sigma."""
    pairs = np.array(list(itertools.combinations(range(len(uv)), 2)))
    D = np.array([D_pairs[i, j] for i, j in pairs])
    sig_pt = pair_metric_sigma(uv, K, R, h0)
    w = 1.0 / np.hypot(sig_pt[pairs[:, 0]], sig_pt[pairs[:, 1]])

    def resid(x):
        h, dp = x[0], x[1]
        dr = x[2] if with_roll else 0.0
        G = ground(uv, K, R, h, dp, dr)
        return w * (np.linalg.norm(G[pairs[:, 0]] - G[pairs[:, 1]], axis=1) - D)

    x0 = [h0, 0.0] + ([0.0] if with_roll else [])
    r = least_squares(resid, x0, x_scale=[1.0, 0.01] + ([0.01] if with_roll else []))
    J = r.jac
    dof = max(1, len(D) - len(x0))
    s2 = float(r.fun @ r.fun) / dof
    cov = np.linalg.pinv(J.T @ J) * s2
    sd = np.sqrt(np.diag(cov))
    out = {"h_m": float(r.x[0]), "h_sd_m": float(sd[0]),
           "dpitch_deg": math.degrees(r.x[1]), "dpitch_sd_deg": math.degrees(sd[1]),
           "n_frames": int(len(uv)), "n_pairs": int(len(D)),
           "rms_pair_error_m": float(np.sqrt(np.mean((r.fun / w) ** 2))), "success": bool(r.success)}
    if with_roll:
        out.update(droll_deg=math.degrees(r.x[2]), droll_sd_deg=math.degrees(sd[2]))
    return out


def fit_joint(sets, shared_pitch=True, h0=16.0):
    """One height across clips; pitch correction shared or one per clip.

    sets: [(uv, D, K, R)], each clip keeping its own K and VP rotation (the PTZ
    zoom and tilt differ slightly per clip). Pairs never cross clips. A shared
    correction assumes the VP errs the same way on every clip; per-clip
    corrections let each clip keep its own height-pitch trade-off.
    """
    blocks = []
    for uv, D, K, R in sets:
        pairs = np.array(list(itertools.combinations(range(len(uv)), 2)))
        sig = pair_metric_sigma(uv, K, R, h0)
        blocks.append((uv, K, R, pairs, np.array([D[i, j] for i, j in pairs]),
                       1.0 / np.hypot(sig[pairs[:, 0]], sig[pairs[:, 1]])))
    n_dp = 1 if shared_pitch else len(sets)

    def resid(x):
        out = []
        for k, (uv, K, R, pairs, D, w) in enumerate(blocks):
            G = ground(uv, K, R, x[0], x[1 + (0 if shared_pitch else k)])
            out.append(w * (np.linalg.norm(G[pairs[:, 0]] - G[pairs[:, 1]], axis=1) - D))
        return np.concatenate(out)

    r = least_squares(resid, [h0] + [0.0] * n_dp, x_scale=[1.0] + [0.01] * n_dp)
    s2 = float(r.fun @ r.fun) / max(1, len(r.fun) - len(r.x))
    sd = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac) * s2))
    w_all = np.concatenate([b[5] for b in blocks])
    return {"h_m": float(r.x[0]), "h_sd_m": float(sd[0]),
            "dpitch_deg": [math.degrees(v) for v in r.x[1:]],
            "dpitch_sd_deg": [math.degrees(v) for v in sd[1:]],
            "rms_pair_error_m": float(np.sqrt(np.mean((r.fun / w_all) ** 2))),
            "n_pairs": int(len(r.fun)), "success": bool(r.success)}


def load_inputs(clip, clicks_path):
    sys.path.insert(0, str(ROOT / "scripts/tracking"))
    import grade_target_vs_gps as g
    cal = ROOT / "outputs/calibration/camera-data" / clip
    k = json.loads(next(cal.glob("*_anycalib_pinhole_pinhole.json")).read_text())["prediction"]["intrinsics"][:4]
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]])
    R = np.array(json.loads((cal / "metric_extrinsic_site.json").read_text())["rotation"])
    gps = g.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv")
    clicks = [c for c in json.loads(Path(clicks_path).expanduser().read_text())["clicks"]
              if not c.get("skipped") and c["frame"] in gps]
    uv = np.array([[c["u"], c["v"]] for c in clicks], float)
    P = np.array([gps[c["frame"]][:2] for c in clicks], float)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)       # GPS east/south metres; distances only
    return K, R, uv, D, [c["frame"] for c in clicks]


def main():
    ap = argparse.ArgumentParser("Camera height from GPS distances between frames")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K, R, uv, D, frames = load_inputs(args.clip, args.clicks)
    res = {"clip": args.clip, "frames": frames,
           "height_pitch": fit(uv, D, K, R), "height_pitch_roll": fit(uv, D, K, R, with_roll=True),
           "note": "GPS used for scale only (pair distances); declare this clip as a calibration clip"}
    # leave-one-frame-out spread, a cheap check that no single click drives the answer
    loo = []
    for i in range(len(uv)):
        keep = [j for j in range(len(uv)) if j != i]
        loo.append(fit(uv[keep], D[np.ix_(keep, keep)], K, R)["h_m"])
    res["leave_one_frame_out_h_m"] = [min(loo), max(loo)]
    print(json.dumps(res, indent=2))
    out = Path(args.out) if args.out else ROOT / "outputs/calibration/camera-data" / args.clip / "height_from_gps_distances.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


def _selfcheck():
    """Synthetic car driving past a known camera: the fit must recover h and dp."""
    sys.path.insert(0, str(ROOT))
    from scripts.calibration.vp_extrinsic_from_frame import solve_pose
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    R, _, _, _ = solve_pose(K, (156.6, 21.9), 16.26)
    h_true, dp_true = 15.6, math.radians(0.25)
    xs = np.linspace(22, 90, 16)
    pts = np.c_[xs, np.full_like(xs, -5.4) + 0.02 * xs, np.zeros_like(xs)]   # a gently drifting lane
    Rt = rot_x(dp_true) @ R
    cam = (Rt @ (pts - np.array([0, 0, h_true])).T)
    uv = (K @ cam).T
    uv = uv[:, :2] / uv[:, 2:]
    rng = np.random.default_rng(0)
    uv_noisy = uv + rng.normal(0, 0.5, uv.shape)
    # the GPS frame is rotated and shifted relative to ours: distances must not care
    th = 1.3
    P = pts[:, :2] @ np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]).T + [500, -80]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    r = fit(uv, D, K, R)
    assert abs(r["h_m"] - h_true) < 1e-4 and abs(r["dpitch_deg"] - 0.25) < 1e-4, r
    r = fit(uv_noisy, D, K, R)
    assert abs(r["h_m"] - h_true) < 3 * r["h_sd_m"] + 0.05, r
    # an elevated point (the hood, 1.3 m up) instead of the road contact must bias h
    up = (K @ (Rt @ (pts + [0, 0, 1.3] - np.array([0, 0, h_true])).T)).T
    rb = fit(up[:, :2] / up[:, 2:], D, K, R)
    print(f"selfcheck ok (exact recovery; noisy h {r['h_m']:.2f} +- {r['h_sd_m']:.2f} m; "
          f"clicking the hood instead of the road gives h {rb['h_m']:.2f} m)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
