"""Sweep camera roll and measure what a roll=0 calibration costs on the ground.

The production extrinsic (vp_extrinsic_from_frame.solve_pose) fixes roll at 0 by
construction: one vanishing point gives two constraints and the rotation has
three unknowns, so roll is the unknown nobody solved. This asks how much that
costs, geometrically, before any detector is involved.

Two arms, because a naive roll injection secretly changes the road direction too:

  Arm A "naive"   R = rot_z(roll) @ R_site           pitch/yaw untouched, VP moves
  Arm B "vp_refit" R = rot_z(roll) @ solve_pose(rot_z(-roll) @ d_road)
                                                     pitch/yaw refit so the model
                                                     reproduces the clip's measured VP

Arm B is the honest one. Roll trades against pitch/yaw, and only Arm B holds the
one thing the calibration actually measured (the road vanishing point) fixed, so
the residual error is roll and nothing else. rot_z here is a left multiply in
camera coordinates, i.e. a rotation about the optical axis, and the camera centre
stays at [0, 0, h] so t = -R @ [0, 0, h] in both arms.

The geometric metric is: forward-project a ground point through the TRUE model
(roll = r), back-project the pixel through the roll = 0 model onto z = 0, and
measure how far the recovered point moved. Heading uses a 4 m baseline through
the same point in three ground directions (road-parallel, 45 deg, cross).

Do NOT score this with sweep_camera_height's heading concentration. Roll barely
moves road-parallel heading (0.32 deg per deg naive, exactly 0 after the VP
refit), so concentration is blind to precisely the error this sweep is about.

Range: BEVHeight's training augmentation draws roll from N(0, 2.0) and DAIR's
shipped extrinsics imply |roll| <= 1.34 deg, so +-3 deg is the in-distribution
band and +-5 deg are out-of-distribution controls.

Run (emits extrinsics + prints the table, seconds, no model needed):

    .venv/bin/python scripts/calibration/sweep_roll.py
    .venv/bin/python scripts/calibration/sweep_roll.py --selfcheck

Detector arm, later: every emitted file is a drop-in metric_extrinsic_site.json,
so point the runner at one per roll value, e.g.

    .venv/bin/python scripts/object_detection/run_bevheight_generic.py \
        --frames-dir data/camera-data/AV_T_WE_1/frames \
        --anycalib-json outputs/calibration/camera-data/AV_T_WE_1/150_anycalib_pinhole_pinhole.json \
        --extrinsic-json outputs/calibration/roll_sweep/AV_T_WE_1/roll_+1/extrinsic_vp_refit.json \
        --out-dir outputs/object_detection/roll_sweep/AV_T_WE_1/roll_+1

The primary metric for THAT stage is GPS-trajectory RMSE against the held-out
trajectory CSVs, not heading concentration. Concentration stays a sanity read
only; a roll error that leaves every car on the road axis still puts every car
in the wrong lane, which is what the RMSE catches and R does not.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.calibration.vp_extrinsic_from_frame import rot_z, solve_pose

ROLLS = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
DIRECTIONS = {"road_parallel": 0.0, "diag_45": 45.0, "cross": 90.0}
BASELINE_M = 4.0          # car-length baseline for the heading estimate
RANGE_M = (20.0, 90.0)
LATERAL_M = 16.0          # half-width of the modelled roadway; see ground_grid


def extrinsic_at_roll(K, R_site, vp_uv, height, roll_deg, refit_vp):
    """Extrinsic with `roll_deg` about the optical axis, both arms.

    refit_vp=False (Arm A): pitch/yaw untouched, so the modelled VP moves.
    refit_vp=True  (Arm B): pitch/yaw resolved so R[:, 0] is still the measured
    road direction, isolating roll from the pitch/yaw it trades against.
    """
    Rz = rot_z(math.radians(roll_deg))
    if refit_vp:
        d = np.linalg.inv(K) @ np.array([vp_uv[0], vp_uv[1], 1.0])
        uv = K @ (Rz.T @ d)                      # VP as seen by the unrolled camera
        R, _, pitch, yaw = solve_pose(K, uv[:2] / uv[2], height)
        R = Rz @ R
    else:
        R, pitch, yaw = Rz @ R_site, None, None
    t = -R @ np.array([0.0, 0.0, height])
    return R, t, pitch, yaw


def project(K, R, t, P):
    """Ground points (N,3) -> pixels (N,2)."""
    pc = P @ R.T + t
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def backproject(K, R, t, uv):
    """Pixels (N,2) -> the ground plane z=0 of this model's frame."""
    d = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T
    dg = d @ R                                   # R.T @ d, rowwise
    C = -R.T @ t                                 # camera centre, = [0, 0, h]
    lam = -C[2] / dg[:, 2]
    return C + lam[:, None] * dg


def ground_grid(K, R, t, img_wh, lateral_m=None):
    """Ground points 20-90 m down the road that actually land in the image.

    Lateral half-width matters: roll error is smallest on the road axis and grows
    off it, so a narrow band understates it and a wide one overstates it. 16 m
    covers a divided beltline's lanes plus shoulders. The reported medians move
    roughly 0.30 m (axis only) to 0.70 m (+-25 m) per degree of roll, so quote
    the band, not just the point estimate.
    """
    lat = LATERAL_M if lateral_m is None else lateral_m
    x, y = np.meshgrid(np.arange(RANGE_M[0], RANGE_M[1] + 1, 5.0),
                       np.arange(-lat, lat + 0.1, 4.0))
    P = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    uv, z = project(K, R, t, P)
    ok = ((z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0])
          & (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1]))
    return P[ok]


def errors(K, R_true, t_true, R0, t0, P, heading_deg):
    """Position and heading error of the roll=0 reading of a true-roll world."""
    a = math.radians(heading_deg)
    u = np.array([math.cos(a), math.sin(a), 0.0])

    def read(pts):
        uv, z = project(K, R_true, t_true, pts)
        return backproject(K, R0, t0, uv), z

    mid, z_mid = read(P)
    back, z_back = read(P - 0.5 * BASELINE_M * u)
    fwd, z_fwd = read(P + 0.5 * BASELINE_M * u)
    ok = (z_mid > 0.5) & (z_back > 0.5) & (z_fwd > 0.5)

    pos_err = np.linalg.norm(mid[ok] - P[ok], axis=1)
    seg = fwd[ok] - back[ok]
    err = np.arctan2(seg[:, 1], seg[:, 0]) - a
    head_err = np.degrees(np.abs(np.arctan2(np.sin(err), np.cos(err))))
    return pos_err, head_err


def load_clip(calib_used_path):
    """(clip, K, R, t, vp_px, height) from a phase1 calibration_used.json."""
    used = json.loads(Path(calib_used_path).read_text())
    ext = json.loads((ROOT / used["extrinsic_json"]).read_text())
    return {
        "clip": Path(calib_used_path).parent.name.replace("_phase1", ""),
        "K": np.array(used["K"], dtype=float),
        "R": np.array(ext["rotation"], dtype=float),
        "t": np.array(ext["translation"], dtype=float),
        "vp": ext["diagnostics"]["vp_px"],
        "height": ext["diagnostics"]["height_m"],
        "extrinsic_json": used["extrinsic_json"],
    }


def emit(out_dir, clip, roll, arm, R, t, pitch, yaw, height, src):
    d = out_dir / clip["clip"] / f"roll_{roll:+g}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"extrinsic_{arm}.json"
    pose = (f"pitch={pitch:.2f} yaw={yaw:.2f}" if pitch is not None
            else "pitch/yaw held at the site values")
    p.write_text(json.dumps({
        "rotation": R.tolist(), "translation": t.tolist(),
        "note": (f"roll sweep arm '{arm}': roll={roll:+g} deg about the optical "
                 f"axis, {pose}, height {height} m (site constant). "
                 f"Perturbation of {src}; NOT a measured pose."),
        "diagnostics": {"roll_deg": roll, "arm": arm,
                        "pitch_deg": pitch, "yaw_deg": yaw,
                        "height_m": height, "height_source": "site constant",
                        "vp_px": clip["vp"], "source_extrinsic": src},
    }, indent=2))
    return p


def main():
    ap = argparse.ArgumentParser("roll sweep: emit extrinsics + geometric cost")
    ap.add_argument("--calibration-used", nargs="+", default=None,
                    help="phase1 calibration_used.json files (default: all)")
    ap.add_argument("--rolls", type=float, nargs="+", default=ROLLS)
    ap.add_argument("--lateral-m", type=float, default=LATERAL_M,
                    help="half-width of the evaluated roadway band")
    ap.add_argument("--out-dir", default="outputs/calibration/roll_sweep")
    args = ap.parse_args()

    paths = args.calibration_used or sorted(
        (ROOT / "outputs/object_detection/camera-data").glob("*_phase1/calibration_used.json"))
    if not paths:
        sys.exit("no calibration_used.json found")
    out_dir = ROOT / args.out_dir
    rows, written = [], []

    for path in paths:
        c = load_clip(path)
        K, R0, t0, h = c["K"], c["R"], c["t"], c["height"]
        img_wh = (2 * K[0, 2], 2 * K[1, 2])
        P = ground_grid(K, R0, t0, img_wh, args.lateral_m)
        for roll in args.rolls:
            for arm, refit in (("naive", False), ("vp_refit", True)):
                R, t, pitch, yaw = extrinsic_at_roll(K, R0, c["vp"], h, roll, refit)
                written.append(emit(out_dir, c, roll, arm, R, t, pitch, yaw, h,
                                    c["extrinsic_json"]))
                for name, ang in DIRECTIONS.items():
                    pe, he = errors(K, R, t, R0, t0, P, ang)
                    rows.append({
                        "clip": c["clip"], "arm": arm, "roll_deg": roll,
                        "direction": name, "n_points": len(pe),
                        "pos_err_median_m": float(np.median(pe)),
                        "pos_err_max_m": float(np.max(pe)),
                        "head_err_median_deg": float(np.median(he)),
                        "head_err_max_deg": float(np.max(he)),
                    })

    csv_path = out_dir / "geometric_error.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print_markdown(rows, sorted({r["clip"] for r in rows}), args.lateral_m)
    print(f"\nwrote {len(written)} extrinsics under {out_dir}")
    print(f"wrote {csv_path}")


def print_markdown(rows, clips, lateral_m):
    print(f"Pooled over {len(clips)} clips ({', '.join(clips)}); ground points "
          f"{RANGE_M[0]:.0f}-{RANGE_M[1]:.0f} m down the road, +-{lateral_m:.0f} m "
          f"laterally, heading baseline {BASELINE_M} m.")
    print("Error = roll=0 model's reading of a world whose true roll is `roll`.\n")
    for arm, label in (("naive", "Arm A - naive (pitch/yaw fixed, VP moves)"),
                       ("vp_refit", "Arm B - honest (pitch/yaw refit to the measured VP)")):
        print(f"### {label}\n")
        print("| roll (deg) | pos err med (m) | pos err max (m) | "
              + " | ".join(f"{d} head med/max (deg)" for d in DIRECTIONS) + " |")
        print("|---" * (3 + len(DIRECTIONS)) + "|")
        for roll in sorted({r["roll_deg"] for r in rows}):
            sel = [r for r in rows if r["arm"] == arm and r["roll_deg"] == roll]
            pos = [r for r in sel if r["direction"] == "road_parallel"]
            cells = []
            for d in DIRECTIONS:
                s = [r for r in sel if r["direction"] == d]
                cells.append("%.2f / %.2f" % (np.median([r["head_err_median_deg"] for r in s]),
                                              max(r["head_err_max_deg"] for r in s)))
            print("| %+g | %.3f | %.3f | %s |" % (
                roll,
                np.median([r["pos_err_median_m"] for r in pos]),
                max(r["pos_err_max_m"] for r in pos),
                " | ".join(cells)))
        print()


def _selfcheck():
    """roll=0 must cost nothing, and Arm B must reproduce the VP exactly."""
    K = np.array([[1556.02, 0, 961.74], [0, 1538.91, 540.55], [0, 0, 1]])
    vp, h = (154.653, 21.013), 16.26
    R0, t0, _, _ = solve_pose(K, vp, h)

    # the site extrinsic itself is a roll=0 pose: it must reproduce its own VP
    d = np.linalg.inv(K) @ np.array([vp[0], vp[1], 1.0])
    assert np.allclose(R0[:, 0], d / np.linalg.norm(d), atol=1e-12)

    P = ground_grid(K, R0, t0, (2 * K[0, 2], 2 * K[1, 2]))
    assert len(P) > 20, len(P)

    for refit in (False, True):
        # roll = 0 is the identity perturbation, in both arms
        R, t, _, _ = extrinsic_at_roll(K, R0, vp, h, 0.0, refit)
        assert np.allclose(R, R0, atol=1e-12) and np.allclose(t, t0, atol=1e-9)
        for ang in DIRECTIONS.values():
            pe, he = errors(K, R, t, R0, t0, P, ang)
            assert pe.max() < 1e-9 and he.max() < 1e-9, (refit, ang, pe.max(), he.max())

    for roll in (-5.0, -1.0, 0.5, 3.0):
        Rz = rot_z(math.radians(roll))
        # Arm A is exactly the left multiply, nothing else
        Ra, ta, _, _ = extrinsic_at_roll(K, R0, vp, h, roll, False)
        assert np.allclose(Ra, Rz @ R0, atol=1e-12)
        assert np.allclose(ta, -Ra @ np.array([0, 0, h]), atol=1e-12)

        # Arm B reproduces the measured VP exactly, and so leaves road-parallel
        # heading untouched, which is why it is the arm that isolates roll
        Rb, tb, pitch, yaw = extrinsic_at_roll(K, R0, vp, h, roll, True)
        assert np.allclose(Rb[:, 0], d / np.linalg.norm(d), atol=1e-12), roll
        uv, z = project(K, Rb, tb, np.array([[1e7, 0.0, 0.0]]))  # far along the road
        assert z[0] > 0 and np.allclose(uv[0], vp, atol=1e-2), (roll, uv[0])
        assert np.allclose(np.linalg.norm(tb), h, atol=1e-9)
        _, he = errors(K, Rb, tb, R0, t0, P, 0.0)
        assert he.max() < 1e-6, (roll, he.max())
        # ...while Arm A does move it
        _, he_a = errors(K, Ra, ta, R0, t0, P, 0.0)
        assert he_a.max() > 0.1 * abs(roll), (roll, he_a.max())

    # projection and back-projection are inverses on the ground plane
    uv, _ = project(K, R0, t0, P)
    assert np.abs(backproject(K, R0, t0, uv) - P).max() < 1e-8

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
