"""Count false objects per detector run: detections off the highway or removed by
the road filter, with one object seen in many frames counted once.

A detection is false when its ground point is off the road band (y < -39 m: the
parking lot; y > -1 m: poles, signs and the text overlay at the left edge) or the
road filter (road_mask.py, from where moving cars drove) removed it. Frozen
copy frames are skipped. False detections of one clip within MERGE_M of each
other are one object (single linkage), so a sign detected in 150 frames is 1.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/false_objects.py \
        --tags ft102 ft102mask ft102feat25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ["HV_T_EW_1", "HV_T_EW_2", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"]
ROAD_Y = (-39.0, -1.0)
MERGE_M = 6.0


def objects(points, r=MERGE_M):
    """Number of single-linkage clusters of (x, y) points within r metres."""
    p = np.asarray(points, float).reshape(-1, 2)
    parent = list(range(len(p)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(p)):
        for j in np.where(np.hypot(*(p[i + 1:] - p[i]).T) < r)[0]:
            parent[root(i)] = root(i + 1 + int(j))
    return len({root(i) for i in range(len(p))})


def false_points(clip, tag):
    det = ROOT / "outputs/object_detection/camera-data" / f"{clip}_{tag}"
    road = det.with_name(det.name + "_road")
    ft = ROOT / f"data/camera-data/{clip}/frame_times.json"
    copy = {f["frame"] for f in json.loads(ft.read_text())["frames"] if f["copy"]} if ft.exists() else set()
    pts, boxes = [], 0
    for p in sorted(det.glob("*_pred.json")):
        if int(p.stem[:3]) in copy:
            continue
        keep = {(round(o["x"], 3), round(o["y"], 3)) for o in json.loads((road / p.name).read_text())}
        for o in json.loads(p.read_text()):
            if not ROAD_Y[0] <= o["y"] <= ROAD_Y[1] or (round(o["x"], 3), round(o["y"], 3)) not in keep:
                pts.append((o["x"], o["y"]))
    return pts


def main():
    ap = argparse.ArgumentParser("False objects per run")
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    out = {}
    for tag in a.tags:
        per = {}
        for clip in CLIPS:
            pts = false_points(clip, tag)
            vehicles = json.loads((ROOT / f"outputs/trajectories/{clip}_{tag}/run.json").read_text())["vehicles"]
            per[clip] = {"false_objects": objects(pts) if pts else 0, "false_boxes": len(pts), "vehicles": vehicles}
        tot = {k: sum(v[k] for v in per.values()) for k in ("false_objects", "false_boxes", "vehicles")}
        out[tag] = {"clips": per, "total": tot}
        print(f"{tag:12s} false objects {tot['false_objects']:3d}  (boxes {tot['false_boxes']:4d})  vehicles {tot['vehicles']}  "
              + "  ".join(f"{c} {v['false_objects']}" for c, v in per.items()))
    (ROOT / "outputs/evaluation/false_objects.json").write_text(json.dumps(out, indent=2))


def _selfcheck():
    """Points along one sign are one object; two far apart spots are two; empty is zero."""
    assert objects([(21, -2), (21.5, -2.2), (22, -1.8), (24, -2)]) == 1
    assert objects([(21, -2), (60, -45)]) == 2
    assert objects([(0, 0), (5, 0), (10, 0)]) == 1          # chained within 6 m
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
