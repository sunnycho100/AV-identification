"""Trajectory accuracy vs the GPS ground truth: position RMSE and heading error.

What this adds over scripts/tracking/grade_target_vs_gps.py:

- Frame-to-GT matching is nearest-timestamp, one GT sample per image frame (the
  advisor's instruction: no interpolation). grade_target_vs_gps averages every
  GT row that falls on a frame, which is fine for position but wrong for angles.
- Heading. Raw per-frame DETECTION yaw is the primary signal; track yaw is
  reported second because AB3DMOT's orientation_correction flips track yaw by
  +-pi whenever it disagrees with the detection by more than 90 deg.
- Both raw and mod-180 circular error, so a 180 deg flip is visible instead of
  quietly inflating the mean.

Alignment is IMPORTED from scripts/tracking/grade_target_vs_gps.py (rigid, no
scale) - the same fit behind the 0.90-0.99 m RMSE figures already reported.

GPS is evaluation-only and never feeds calibration or tracking.

Run:
    .venv/bin/python scripts/reporting/compare_trajectories.py --clip HV_T_EW_1
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tracking"))
from grade_target_vs_gps import rigid_align  # noqa: E402  (reuse, do not re-derive)

# The Aug-21 Box export carries processed_heading_deg: vehicle heading measured
# from +x (east) toward +y (south), same frame as processed_x/y. Try that first.
# `azimuth` is also present but is the raw INS bearing in a different reference,
# so it must never win the auto-detect.
HEADING_COLS = ("processed_heading_deg", "vehicle_heading", "heading",
                "heading_deg", "heading_rad", "vehicle_heading_deg", "course")


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wrap_half_pi(a):
    """Fold onto [-90, 90): a 180 deg flip becomes zero error."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def load_gt(csv_path, heading_col=None):
    """frame -> nearest-in-time GT sample. No interpolation, no averaging.

    GT logs at ~125 Hz, video at 30 Hz, so several GT rows land on one frame;
    the CSV already carries nearest_frame_time_error_sec per row, so the
    nearest sample is just the row with the smallest |error| for that frame.
    """
    rows = list(csv.DictReader(open(csv_path)))
    cols = rows[0].keys()
    if heading_col is None:
        heading_col = next((c for c in HEADING_COLS if c in cols), None)
    elif heading_col not in cols:
        sys.exit(f"--gt-heading-col {heading_col!r} not in {csv_path}")

    best = {}
    for r in rows:
        if not r["nearest_frame_index"]:
            continue
        fr = int(r["nearest_frame_index"])
        dt = abs(float(r["nearest_frame_time_error_sec"]))
        if fr in best and best[fr]["dt"] <= dt:
            continue
        vx, vy = float(r["processed_vx"]), float(r["processed_vy"])
        if heading_col:
            h = float(r[heading_col])
            # Decide units by NAME, not magnitude: a heading of 2.5 deg is a
            # legal radian value too, so the magnitude test silently mangles
            # any clip whose heading sits near zero (AV_T_WE_1 runs 2-3 deg).
            if heading_col.endswith("_deg") or heading_col in ("azimuth", "course"):
                h = math.radians(h)
            elif not heading_col.endswith("_rad") and abs(h) > 2 * math.pi:
                h = math.radians(h)
        else:
            # ponytail: fallback while the heading column is missing. Course over
            # ground from the GT velocity, in the same (x east, y south) frame as
            # processed_x/y. Not the same as vehicle heading under any slip, so it
            # is labelled as a fallback in the output.
            h = math.atan2(vy, vx)
        best[fr] = {"dt": dt, "t_gt": float(r["time_sec"]),
                    "t_frame": float(r["nearest_frame_time_sec"]),
                    "x": float(r["processed_x"]), "y": float(r["processed_y"]),
                    "speed": math.hypot(vx, vy), "heading": h}
    return best, (heading_col or "course_from_velocity(FALLBACK)")


def load_target_states(track_dir):
    """The marker-identified target track, or the longest moving track."""
    tp = track_dir / "target_track.json"
    if tp.exists():
        d = json.loads(tp.read_text())
        return d["states"], f"track {d['target_track_id']} ({d['identified_from']})"
    tracks = json.loads((track_dir / "tracks.json").read_text())["tracks"]
    tid = max(tracks, key=lambda t: (np.mean([s["speed_mps"] for s in tracks[t][1:]] or [0]) > 5.0,
                                     len(tracks[t])))
    return tracks[tid], f"track {tid} (longest moving - NO marker identification)"


def load_raw_dets(det_dir, states, gate_m=3.0):
    """frame -> the raw detection nearest that frame's track state.

    Raw detection yaw is the primary heading signal; the track's yaw has been
    through AB3DMOT's +-pi orientation_correction.
    """
    out = {}
    for s in states:
        p = det_dir / f"{s['frame']:03d}_pred.json"
        if not p.exists():
            continue
        dets = [d for d in json.loads(p.read_text()) if d.get("class_name") == "car"]
        if not dets:
            continue
        d = min(dets, key=lambda d: math.hypot(d["x"] - s["x"], d["y"] - s["y"]))
        if math.hypot(d["x"] - s["x"], d["y"] - s["y"]) <= gate_m:
            out[s["frame"]] = d
    return out


def compare(A, B, h_gt=None, h_cam=None):
    """A = GT xy (GPS frame), B = camera xy. Rigid-align A onto B, then score.

    Heading: rigid_align gives R mapping GT frame -> camera frame, so the GT
    heading expressed in the camera frame is h_gt + angle(R).
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    R, t = rigid_align(A, B)
    resid = (A @ R.T + t) - B
    out = {"n": len(A),
           "rmse_m": float(np.sqrt((resid ** 2).sum(1).mean())),
           "median_err_m": float(np.median(np.linalg.norm(resid, axis=1))),
           "max_err_m": float(np.linalg.norm(resid, axis=1).max()),
           "frame_rotation_deg": float(math.degrees(math.atan2(R[1, 0], R[0, 0])))}
    if h_gt is not None and h_cam is not None:
        e = wrap_pi(np.asarray(h_cam) - (np.asarray(h_gt) + math.atan2(R[1, 0], R[0, 0])))
        e180 = wrap_half_pi(e)
        out.update({
            "heading_rmse_deg": float(np.degrees(np.sqrt((e ** 2).mean()))),
            "heading_mae_deg": float(np.degrees(np.abs(e).mean())),
            "heading_median_bias_deg": float(np.degrees(np.median(e))),
            "heading_mod180_rmse_deg": float(np.degrees(np.sqrt((e180 ** 2).mean()))),
            "heading_mod180_mae_deg": float(np.degrees(np.abs(e180).mean())),
            "n_flipped_gt90deg": int((np.abs(e) > np.pi / 2).sum()),
        })
        out["_per_frame_heading_err_deg"] = np.degrees(e)
    return out


def _self_check():
    """Synthetic trajectory in -> zero error out."""
    th = np.linspace(0, 1.2, 60)
    A = np.stack([40 * th, 3 * np.sin(th)], 1)                 # GT path
    h = np.arctan2(np.gradient(A[:, 1]), np.gradient(A[:, 0]))  # GT heading
    ang, T = 0.7, np.array([12.0, -5.0])
    Rk = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    B = A @ Rk.T + T                                            # same path, other frame
    r = compare(A, B, h, h + ang)
    assert r["rmse_m"] < 1e-9, r["rmse_m"]
    assert r["heading_rmse_deg"] < 1e-6, r["heading_rmse_deg"]
    assert abs(r["frame_rotation_deg"] - math.degrees(ang)) < 1e-6
    # a 180 deg flip must show up raw and vanish mod-180
    rf = compare(A, B, h, h + ang + math.pi)
    assert rf["heading_rmse_deg"] > 179 and rf["heading_mod180_rmse_deg"] < 1e-6
    assert rf["n_flipped_gt90deg"] == len(A)


def main():
    ap = argparse.ArgumentParser("GT trajectory vs detection/track: position + heading")
    ap.add_argument("--clip", required=True, help="e.g. HV_T_EW_1")
    ap.add_argument("--gps-csv", default=None)
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--track-dir", default=None)
    ap.add_argument("--gt-heading-col", default=None,
                    help="name of the vehicle-heading column in the new Box export")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "reporting"))
    args = ap.parse_args()

    _self_check()
    clip = args.clip
    gps_csv = Path(args.gps_csv or ROOT / f"Camera data/{clip}_trajectory.csv")
    det_dir = Path(args.det_dir or ROOT / f"outputs/object_detection/camera-data/{clip}_phase1")
    trk_dir = Path(args.track_dir or ROOT / f"outputs/tracking/camera-data/{clip}_phase1")
    for p in (gps_csv, trk_dir):
        if not p.exists():
            sys.exit(f"missing: {p}")

    gt, heading_src = load_gt(gps_csv, args.gt_heading_col)
    states, target_desc = load_target_states(trk_dir)
    dets = load_raw_dets(det_dir, states) if det_dir.exists() else {}

    rows = []
    for s in states:
        g = gt.get(s["frame"])
        if g is None:
            continue
        d = dets.get(s["frame"])
        rows.append({"frame": s["frame"], "gt_time_sec": g["t_gt"],
                     "frame_time_sec": g["t_frame"], "match_dt_sec": g["dt"],
                     "gt_x": g["x"], "gt_y": g["y"], "gt_speed_mps": g["speed"],
                     "gt_heading_rad": g["heading"],
                     "trk_x": s["x"], "trk_y": s["y"], "trk_yaw_rad": s["yaw"],
                     "trk_speed_mps": s["speed_mps"],
                     "det_x": d["x"] if d else "", "det_y": d["y"] if d else "",
                     "det_yaw_rad": d["yaw"] if d else ""})
    if len(rows) < 10:
        sys.exit(f"only {len(rows)} frames overlap GT - nothing to grade")

    A = [[r["gt_x"], r["gt_y"]] for r in rows]
    h_gt = [r["gt_heading_rad"] for r in rows]
    trk = compare(A, [[r["trk_x"], r["trk_y"]] for r in rows],
                  h_gt, [r["trk_yaw_rad"] for r in rows])
    dr = [r for r in rows if r["det_x"] != ""]
    det = compare([[r["gt_x"], r["gt_y"]] for r in dr],
                  [[r["det_x"], r["det_y"]] for r in dr],
                  [r["gt_heading_rad"] for r in dr],
                  [r["det_yaw_rad"] for r in dr]) if len(dr) >= 10 else None

    for r, res in ((rows, trk), (dr, det)):
        if res:
            for row, e in zip(r, res.pop("_per_frame_heading_err_deg")):
                row["trk_heading_err_deg" if res is trk else "det_heading_err_deg"] = e

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{clip}_trajectory_compare.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame"] + [c for c in fields if c != "frame"])
        w.writeheader()
        w.writerows(rows)

    dts = [r["match_dt_sec"] for r in rows]
    spread = float(np.degrees(np.ptp(wrap_pi(h_gt))))
    print(f"\n{clip}   target: {target_desc}")
    print(f"  GT csv        : {gps_csv}")
    print(f"  GT heading    : {heading_src}   (spread over clip {spread:.2f} deg)")
    print(f"  frames matched: {len(rows)} (nearest-timestamp, median |dt| "
          f"{np.median(dts)*1000:.1f} ms, max {max(dts)*1000:.1f} ms)")
    if spread < 5.0:
        print("  WARNING: the GT heading barely changes on this clip, so a constant "
              "convention offset (incl. a 180 deg nose/tail or a frame-handedness "
              "flip) cannot be separated from real heading error. Read the mod-180 "
              "numbers as the accuracy and the median bias as the convention.")
    for name, res in (("RAW DETECTIONS (primary)", det), ("TRACK (secondary)", trk)):
        if res is None:
            print(f"\n  {name}: too few gated detections ({len(dr)})")
            continue
        print(f"\n  {name}  n={res['n']}")
        print(f"    position RMSE / median / max : {res['rmse_m']:.2f} / "
              f"{res['median_err_m']:.2f} / {res['max_err_m']:.2f} m")
        print(f"    GT->camera frame rotation    : {res['frame_rotation_deg']:+.2f} deg")
        print(f"    heading RMSE / MAE           : {res['heading_rmse_deg']:.2f} / "
              f"{res['heading_mae_deg']:.2f} deg   (median bias "
              f"{res['heading_median_bias_deg']:+.2f})")
        print(f"    heading mod-180 RMSE / MAE   : {res['heading_mod180_rmse_deg']:.2f} / "
              f"{res['heading_mod180_mae_deg']:.2f} deg   "
              f"({res['n_flipped_gt90deg']}/{res['n']} frames off by >90 deg)")
    print("\n  known biases, NOT corrected here: ~7% speed underestimate from the "
          "camera-height scale; range-dependent depth bias (~1.2 m near, ~1 m far).")
    print(f"\nwrote {csv_out}")


if __name__ == "__main__":
    main()
