"""Give every training label a role from image agreement with a 2D detector (decision D34).

role "label":  taught as a car. Tracked labels, and single-frame boxes that own a 2D box.
role "ignore": no loss either way (ST3D's ignored margin, KITTI's DontCare), for
  dup     a second box on a car another label fits better in the image
  no_2d   a single-frame box no 2D detection supports (trucks, edge-cut cars, junk)
  2d_only a 2D detection no label covers: a car the 3D labels missed, so it must not
          be taught as background. Placed at the 2D box's ground point.

Nothing is deleted on the 2D detector's word: it merges far queued cars and misses
cars cut by the frame edge (D33 same-car test), so a wrong flag must cost nothing.

    python3 scripts/finetune/mark_label_roles.py SRC DST    e.g. pseudo_labels_r2hb_v2 pseudo_labels_r2hb_v3
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reporting"))
import match2d as X                      # noqa: E402
import render_untracked_conflicts as R   # noqa: E402
sys.path.insert(0, str(ROOT / "scripts/object_detection"))
import road_mask as rm                   # noqa: E402

CLIPS = ("HV_T_EW_1", "AV_T_WE_1", "AV_T_EW_7")
GRID_X, GRID_Y = (0.0, 102.4), 51.2      # the BEV grid; ignore zones outside it do nothing


def on_road(box2d, road):
    """The 2D box's bottom-centre pixel lies on the clip's road mask (no ground projection needed)."""
    if road is None:
        return True
    h, w = road.shape
    u, v = int((box2d[0] + box2d[2]) / 2), int(box2d[3])
    return bool(road[min(max(v, 0), h - 1), min(max(u, 0), w - 1)])


def roles(labels, boxes2d, K, M, road_z, road=None):
    """Labels with a role, plus ignore objects for 2D detections no label covers (on the road only)."""
    st = X.match(labels, boxes2d, K, M)
    out = []
    for b, (s, _) in zip(labels, st):
        if s == "dup":
            out.append({**b, "role": "ignore", "reason": "dup"})
        elif s == "nosupport" and not R.tracked(b):
            out.append({**b, "role": "ignore", "reason": "no_2d"})
        else:
            out.append({**b, "role": "label"})
    rects = [X.rect(b, K, M) for b in labels]
    for d in boxes2d:
        if any(r and X.iou(r, d["box2d"]) >= X.MIN_IOU for r in rects):
            continue
        if GRID_X[0] < d["x"] < GRID_X[1] and abs(d["y"]) < GRID_Y and on_road(d["box2d"], road):
            out.append({"class_name": "car", "x": d["x"], "y": d["y"], "z": road_z, "l": 4.5, "w": 1.8, "h": 1.5,
                        "yaw": 0.0, "score": d["score"], "box2d": d["box2d"], "track_id": None,
                        "role": "ignore", "reason": "2d_only"})
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    summary = {}
    for c in CLIPS:
        K, M = X.load(c)
        road_z = json.loads((ROOT / f"outputs/object_detection/camera-data/{c}_r2lab/calibration_used.json").read_text()).get(
            "road_plane_z_in_output", -1.73)
        road = rm.load_built(c)
        out_dir = ROOT / f"outputs/finetune/{dst}/{c}"
        out_dir.mkdir(parents=True, exist_ok=True)
        C = collections.Counter()
        for p in sorted(glob.glob(str(ROOT / f"outputs/finetune/{src}/{c}/*_label.json"))):
            f = int(Path(p).name[:3])
            objs = roles(json.loads(Path(p).read_text()), X.boxes2d(c, f), K, M, road_z, road)
            C.update((o["role"], o.get("reason") or ("tracked" if R.tracked(o) else "single_2d")) for o in objs)
            (out_dir / Path(p).name).write_text(json.dumps(objs))
        summary[c] = {f"{r}/{why}": n for (r, why), n in sorted(C.items())}
        print(c, summary[c])
    (ROOT / f"outputs/finetune/{dst}/manifest.json").write_text(json.dumps({
        "source": src, "decision": "D34", "min_iou": X.MIN_IOU, "min_2d_score": X.MIN_SCORE,
        "merge_2d_iou": X.SAME_2D, "counts": summary}, indent=1))


def _selfcheck():
    """Two labels on one 2D box: the worse fit is ignored. An uncovered 2D box becomes an ignore zone."""
    rects = {1: (0, 0, 10, 10), 2: (2, 2, 12, 12), 3: (50, 50, 60, 60)}
    real = X.rect
    X.rect = lambda b, K, M: rects[b["id"]]
    try:
        labs = [{"id": 1, "yaw_source": "lane_flow"}, {"id": 2, "yaw_source": "track_motion"},
                {"id": 3, "yaw_source": "lane_flow"}]
        d2 = [{"box2d": (0, 0, 10, 10), "x": 30.0, "y": 0.0, "score": 0.9},
              {"box2d": (100, 100, 110, 110), "x": 40.0, "y": -5.0, "score": 0.8}]
        o = roles(labs, d2, None, None, -1.7)
        assert [x.get("reason") for x in o[:3]] == [None, "dup", "no_2d"], o
        assert o[3]["reason"] == "2d_only" and o[3]["x"] == 40.0 and len(o) == 4, o
    finally:
        X.rect = real
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
    main()
