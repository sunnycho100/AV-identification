"""Mark the instrumented vehicle from hand-picked image positions (rear-view clips).

For clips where neither the hood marker nor a reference track can be used, the car
is located by eye in a few frames (roof rack and rear sign), BEFORE any GPS file is
opened. Each pick votes for the track whose projected 3D box contains it, the same
rule identify_target_by_marker.py applies to marker boxes.

    python scripts/pipeline/mark_instrumented_manual.py --clip HV_T_WE_6 --tag iv_s4 \
        --pick 91,1575,720 --pick 99,1350,624 --pick 107,1185,531 --pick 115,1074,465
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def box_hull(s, K, M, z_road):
    l, w, h, yaw = s.get("l", 4.6), s.get("w", 1.9), s.get("h", 1.5), s.get("yaw", 0.0)
    c, sn = np.cos(yaw), np.sin(yaw)
    pts = []
    for dx in (-l / 2, l / 2):
        for dy in (-w / 2, w / 2):
            for dz in (0.0, h):
                x = s["x"] + c * dx - sn * dy
                y = s["y"] + sn * dx + c * dy
                q = M[:3, :3] @ np.array([x, y, z_road + dz]) + M[:3, 3]
                if q[2] <= 1:
                    return None
                u = K @ q
                pts.append(u[:2] / u[2])
    return cv2.convexHull(np.array(pts, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pick", action="append", required=True, help="frame,u,v")
    a = ap.parse_args()
    run = ROOT / "outputs/trajectories" / f"{a.clip}_{a.tag}"
    det_tag = json.loads((run / "run.json").read_text())["settings"]["det_tag"]
    cal = json.loads((ROOT / f"outputs/object_detection/camera-data/{a.clip}_{det_tag}_road/calibration_used.json").read_text())
    K, M, z = np.array(cal["K"]), np.array(cal["lidar2cam"]), cal.get("road_plane_z_in_output", -1.73)
    tracks = json.loads((run / "tracks.json").read_text())["tracks"]
    picks = [tuple(int(v) for v in p.split(",")) for p in a.pick]
    votes = {}
    for f, u, v in picks:
        for tid, st in tracks.items():
            s = next((s for s in st if s["frame"] == f), None)
            hull = box_hull(s, K, M, z) if s else None
            if hull is not None and cv2.pointPolygonTest(hull, (float(u), float(v)), False) >= 0:
                votes[tid] = votes.get(tid, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    res = {"clip": a.clip, "identified_from": "image only, GPS not used",
           "method": "manual picks (roof rack and rear sign), projected-box vote",
           "picks": picks, "vehicle_id": ranked[0][0] if ranked else None,
           "votes": ranked[0][1] if ranked else 0, "runner_up_votes": ranked[1][1] if len(ranked) > 1 else 0}
    if res["vehicle_id"] is None:
        print(f"{a.clip} {a.tag}: no track contains the picks (vehicle not tracked in those frames)")
    (run / "instrumented.json").write_text(json.dumps(res, indent=2))
    print(f"{a.clip} {a.tag}: instrumented vehicle = {res['vehicle_id']} (votes {res['votes']} of {len(picks)}, runner-up {res['runner_up_votes']})")


if __name__ == "__main__":
    main()
