"""Grade two runs of the same clips against GPS on the image-identified target.

Why. The camera height changed (16.26 m in _phase1, 15.6 m in _h156, from the
GPS-distance fit and OTC3D). Only the height differs between the runs, so the
along-road scale should change and nothing else. This reports, per clip, the
numbers grade_target_vs_gps already computes: speed ratio (no alignment, not
sensitive to the time offset), rigid RMSE, and along-ray error by range band.

The target is never chosen by GPS. Each clip has a reference track picked from
the image (hood marker or a manual pick); in each run the target is the track
that follows that reference most often, matched in image pixels so the height
change cannot break the match.

HV_T_EW_1 and AV_T_WE_1 were used to fit the height, so only AV_T_EW_3 and
AV_T_WE_3 are fair tests. AV_T_WE_3's local video is a shorter cut than the one
its GPS was aligned to, so its GPS numbers are flagged.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/compare_height_runs.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tracking"))
import grade_target_vs_gps as g   # noqa: E402

# clip -> (run holding the reference track, track id or "target", role)
REFERENCE = {
    "AV_T_EW_3": ("phase1", "target", "held out"),
    "AV_T_WE_3": ("phase1_102", "17", "held out, GPS cut mismatch"),
    "HV_T_EW_1": ("phase1", "target", "used to fit the height"),
    "AV_T_WE_1": ("phase1_102", "28", "used to fit the height"),
}
RUNS = (("phase1", "16.26 m"), ("h156", "15.6 m"))
MATCH_PX = 40.0


def load_track(clip, run, tid):
    d = ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}"
    if tid == "target":
        return json.loads((d / "target_track.json").read_text())["states"]
    return json.loads((d / "tracks.json").read_text())["tracks"][tid]


def to_px(clip, run, states):
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{clip}_{run}"
                      / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    out = {}
    for s in states:
        q = M[:3, :3] @ np.array([s["x"], s["y"], s["z"]]) + M[:3, 3]
        uv = K @ q
        out[s["frame"]] = uv[:2] / uv[2]
    return out


def pick_target(clip, run, ref_px):
    """The track in `run` that sits within MATCH_PX of the reference most often."""
    tracks = json.loads((ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}"
                         / "tracks.json").read_text())["tracks"]
    best, votes = None, 0
    for tid, st in tracks.items():
        px = to_px(clip, run, st)
        v = sum(1 for f, p in px.items() if f in ref_px and np.linalg.norm(p - ref_px[f]) < MATCH_PX)
        if v > votes:
            best, votes = tid, v
    return best, votes


def main():
    rows = []
    for clip, (ref_run, ref_tid, role) in REFERENCE.items():
        ref = load_track(clip, ref_run, ref_tid)
        ref_px = to_px(clip, ref_run, ref)
        gps = g.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv")
        for run, label in RUNS:
            if not (ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}" / "tracks.json").exists():
                continue
            tid, votes = pick_target(clip, run, ref_px)
            st = json.loads((ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}"
                             / "tracks.json").read_text())["tracks"][tid]
            r = g.score(st, gps)
            if r is None:
                continue
            rows.append({"clip": clip, "role": role, "run": run, "height": label, "track": tid,
                         "votes": votes, **{k: r[k] for k in ("n", "speed_ratio", "scale", "scale_chord",
                                                              "rmse_rigid_m", "depth_err_by_range")}})
    for r in rows:
        bands = "  ".join(f"{k} {v['depth_err_median_m']:+.2f}" for k, v in r["depth_err_by_range"].items())
        print(f"{r['clip']:10s} {r['height']:8s} n={r['n']:3d} speed ratio {r['speed_ratio']:.3f}  "
              f"chord {r['scale_chord']:.3f}  rigid RMSE {r['rmse_rigid_m']:.2f} m | along-ray error: {bands}"
              f"   [{r['role']}]")
    out = ROOT / "outputs/evaluation/height_156_vs_1626.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
