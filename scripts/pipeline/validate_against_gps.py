"""Evaluation only: the instrumented vehicle's extracted trajectory against its GPS.

Reads the pipeline's final output (outputs/trajectories/<clip>_<tag>) and the
vehicle marked by mark_instrumented.py, so the vehicle choice was made before
GPS is opened. Reports, on frames with both:
  position   rigid-fit RMSE and along-ray error by range band
             (grade_target_vs_gps.score; a constant offset is absorbed by the fit)
  speed      mean ratio camera/GPS and mean absolute error
  heading    angle between the camera and GPS velocity directions after the
             rigid fit's rotation
  accel      along-track acceleration from both speeds with the same 0.5 s
             central difference (trajectory_features.diff); rms of each and of
             their difference
Clips whose GPS set the calibration (HV_T_EW_1, AV_T_WE_1) are marked as such;
their numbers are not a held-out test. AV_T_WE_3's video is a shorter cut than
the one its GPS was aligned to.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/pipeline/validate_against_gps.py \
        --clips AV_T_EW_3 HV_T_EW_2
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/tracking"), str(ROOT / "scripts/pipeline")]
import grade_target_vs_gps as gt        # noqa: E402
from trajectory_features import diff    # noqa: E402

ROLE = {"HV_T_EW_1": "used to fit the calibration", "AV_T_WE_1": "used to fit the calibration",
        "AV_T_WE_3": "held out, video cut shorter than the GPS alignment"}


def validate(clip, tag):
    d = ROOT / "outputs/trajectories" / f"{clip}_{tag}"
    vid = json.loads((d / "instrumented.json").read_text())["vehicle_id"]
    states = json.loads((d / "tracks.json").read_text())["tracks"][vid]
    gps = gt.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv")
    r = gt.score(states, gps)
    if r is None:
        return {"clip": clip, "vehicle": vid, "note": "under 10 frames with GPS"}
    common = [s for s in states if s["frame"] in gps]
    A = np.array([gps[s["frame"]][:2] for s in common])
    B = np.array([[s["x"], s["y"]] for s in common])
    R, _ = gt.rigid_align(A, B)
    vg = np.gradient(A @ R.T, axis=0)
    vc = np.array([[s["vx"], s["vy"]] for s in common])
    ang = np.degrees(np.abs(np.arctan2(vc[:, 1], vc[:, 0]) - np.arctan2(vg[:, 1], vg[:, 0])))
    ang = np.minimum(ang, 360 - ang)
    sc = np.array([s["speed_mps"] for s in common])
    sg = np.array([gps[s["frame"]][2] for s in common])
    ac, ag = diff(sc), diff(sg)
    return {"clip": clip, "role": ROLE.get(clip, "held out"), "vehicle": vid, "frames": len(common),
            "rmse_rigid_m": round(r["rmse_rigid_m"], 2), "along_ray_by_range_m":
            {k: round(v["depth_err_median_m"], 2) for k, v in r["depth_err_by_range"].items()},
            "speed_ratio": round(float(sc.mean() / sg.mean()), 3),
            "speed_mae_mps": round(float(np.abs(sc - sg).mean()), 2),
            "heading_err_median_deg": round(float(np.median(ang)), 2),
            "accel_rms_camera": round(float(np.sqrt(np.mean(ac ** 2))), 3),
            "accel_rms_gps": round(float(np.sqrt(np.mean(ag ** 2))), 3),
            "accel_rms_diff": round(float(np.sqrt(np.mean((ac - ag) ** 2))), 3)}


def main():
    ap = argparse.ArgumentParser("Instrumented vehicle vs GPS")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--tag", default="ft102")
    ap.add_argument("--out", default="outputs/trajectories/gps_validation.json")
    a = ap.parse_args()
    res = [validate(c, a.tag) for c in a.clips]
    (ROOT / a.out).write_text(json.dumps(res, indent=2))
    for r in res:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
