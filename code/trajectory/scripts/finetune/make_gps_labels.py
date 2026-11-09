"""Fine-tuning labels from the instrumented car's GPS (Hang's masked-loss idea).

We have ground truth for one car per clip. Every frame where that car's GPS
falls inside the detector's 102.4 m grid gets one "gps" label; every other
vehicle in the frame is listed separately so the training loss can treat it
two ways (train_finetune.py --mode):

  gps_only  other vehicles are IGNORE regions: the heatmap loss is switched
            off around them, so the network is neither told they are cars nor
            told they are background. Only the GPS car supervises.
  hybrid    tracked other vehicles become pseudo-labels at a lower weight;
            untracked detections stay ignore regions.

Roles written per object:
  gps     the instrumented car at GPS position, heading and velocity, mapped
          into the detector frame (gps_to_image.py site transform) at the
          per-clip time offset fitted by gps_benchmark.py. z = road plane.
          Size is a fixed 4.6 x 1.9 x 1.5 m guess, so training masks its size.
  other   a cleaned ft102 trajectory state (Kalman + RTS smoothed), heading
          from its motion, size from the matching detection.
  ignore  any other ft102 car detection (score >= 0.3) with no track state.

Along the road the GPS position is NOT independent: the time offset was fitted
by matching the detector's own track, so along-road position is circular. Only
the across-road position, heading, speed and the road-plane z are real new
information. train_finetune.py drops the along-road offset from the GPS car's
regression for that reason.

Clips: HV_T_EW_1, AV_T_WE_1, AV_T_WE_3 only. AV_T_EW_3 is the test clip and
its GPS never enters training. HV_T_EW_2 has no identifiable instrumented car.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/finetune/make_gps_labels.py
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/finetune/make_gps_labels.py --review
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
from gps_to_image import L, W, H, ROAD_Z, gps_road, load_clip, load_site  # noqa: E402

CLIPS = ["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3"]
TEST_CLIP = "AV_T_EW_3"
TAG = "ft102"
X_RANGE, Y_HALF = (0.0, 102.4), 51.2          # detector grid (bev_height ..._102.py)
LAST_CLEAN_FRAME = {"AV_T_WE_3": 99}          # the source recording lost frames; cars smear from ~100 on
SAME_CAR = (4.0, 2.0)                         # m along, across: a state this close is the GPS car itself
OUT = ROOT / "outputs/finetune/gps_labels"


def in_image(x, y, yaw, K, l2c, w=1920, h=1080):
    """All 8 corners of the GPS car's box are in front of the camera and inside the
    picture. Near the camera the grid starts under the lens, where the car is not
    visible at all, and a label there would teach a car on empty pixels."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(x + a * L / 2 * c - b * W / 2 * s, y + a * L / 2 * s + b * W / 2 * c, ROAD_Z + z)
           for a in (-1, 1) for b in (-1, 1) for z in (0, H)]
    cam = (l2c @ np.c_[np.array(pts), np.ones(8)].T)[:3]
    if (cam[2] <= 1e-6).any():
        return False
    uv = (K @ cam)[:2] / cam[2]
    return bool((uv[0] >= 0).all() and (uv[0] < w).all() and (uv[1] >= 0).all() and (uv[1] < h).all())


def offsets():
    return {c: json.loads((ROOT / "outputs/evaluation/gps_benchmark" / f"{c}.json").read_text())["timing"]["best_offset_s"]
            for c in CLIPS}


def build(clip, T, tau):
    assert clip != TEST_CLIP
    c = load_clip(clip, TAG)
    det_dir = ROOT / "outputs/object_detection/camera-data" / f"{clip}_{TAG}"
    states = {}
    for tid, st in c["tracks"].items():
        for s in st:
            states.setdefault(s["frame"], []).append((tid, s))
    inst = c["car"]["id"] if c["car"] else None       # the image-identified lab car, never an "other"
    g0, g1 = c["gps"]["t"][0], c["gps"]["t"][-1]
    frames, n_other, n_ignore = {}, 0, 0
    for i, t in enumerate(c["t"]):
        if c["copy"][i] or i > LAST_CLEAN_FRAME.get(clip, 10 ** 6) or not (g0 <= t + tau <= g1):
            continue
        xy, hd, v = gps_road(c["gps"], T, np.array([t + tau]))
        x, y, yaw, sp = float(xy[0, 0]), float(xy[0, 1]), float(hd[0]), float(v[0])
        if not (X_RANGE[0] <= x < X_RANGE[1] and abs(y) < Y_HALF) or not in_image(x, y, yaw, c["K"], c["l2c"]):
            continue
        objs = [{"role": "gps", "x": x, "y": y, "z": ROAD_Z, "l": L, "w": W, "h": H, "yaw": yaw,
                 "vx": sp * math.cos(yaw), "vy": sp * math.sin(yaw), "t_video_s": float(t), "tau_s": tau}]

        def is_gps_car(px, py):
            return abs(px - x) < SAME_CAR[0] and abs(py - y) < SAME_CAR[1]

        dets = [o for o in json.loads((det_dir / f"{i:03d}_pred.json").read_text())
                if o["class_name"] == "car" and o["score"] >= 0.3]
        used = set()
        for tid, s in states.get(i, []):
            if tid == inst or is_gps_car(s["x"], s["y"]) or "x_det" not in s:
                continue                      # the GPS car itself, or a coasted state with no image evidence
            j = min(range(len(dets)), key=lambda k: math.hypot(dets[k]["x"] - s["x_det"], dets[k]["y"] - s["y_det"]),
                    default=None)
            d = dets[j] if j is not None and math.hypot(dets[j]["x"] - s["x_det"], dets[j]["y"] - s["y_det"]) < 1.0 else None
            if d is None:
                continue
            used.add(j)
            yaw_s = math.atan2(s["vy"], s["vx"]) if s["speed_mps"] > 1 else s["yaw"]
            objs.append({"role": "other", "track_id": tid, "x": s["x"], "y": s["y"], "z": ROAD_Z,
                         "l": d["l"], "w": d["w"], "h": d["h"], "yaw": yaw_s, "vx": s["vx"], "vy": s["vy"]})
            n_other += 1
        for k, d in enumerate(dets):
            if k in used or is_gps_car(d["x"], d["y"]):
                continue
            objs.append({"role": "ignore", "x": d["x"], "y": d["y"], "z": ROAD_Z, "l": d["l"], "w": d["w"],
                         "h": d["h"], "yaw": d["yaw"], "vx": 0.0, "vy": 0.0})
            n_ignore += 1
        frames[i] = objs
    out = OUT / clip
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*_label.json"):
        f.unlink()
    for i, objs in frames.items():
        (out / f"{i:03d}_label.json").write_text(json.dumps(objs, indent=1))
    gx = [o[0]["x"] for o in frames.values()]
    man = {"clip": clip, "tau_s": tau, "frames": len(frames), "frame_range": [min(frames), max(frames)] if frames else None,
           "gps_x_range_m": [round(min(gx), 1), round(max(gx), 1)] if gx else None,
           "other_labels": n_other, "ignore_regions": n_ignore, "instrumented_track": c["car"]["id"] if c["car"] else None,
           "provenance": "gps: lab GPS at fitted per-clip offset, site transform from HV_T_EW_1 + AV_T_WE_1; "
                         "other: ft102 smoothed tracks; ignore: ft102 detections. AV_T_EW_3 not used."}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"{clip}: {man['frames']} frames {man['frame_range']}, GPS car x {man['gps_x_range_m']} m, "
          f"{n_other} other-car labels, {n_ignore} ignore regions")
    return man


def review(clip, n=6):
    """Contact sheet: GPS label red, other-car labels green, ignore regions yellow."""
    import cv2
    from evaluators.result2kitti import get_lidar_3d_8points
    from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{clip}_{TAG}" / "calibration_used.json").read_text())
    K, l2c = np.array(cal["K"]), np.array(cal["lidar2cam"])
    k34 = np.c_[K, np.zeros(3)]
    labs = sorted((OUT / clip).glob("*_label.json"))
    pick = [labs[int(k)] for k in np.linspace(0, len(labs) - 1, n)]
    col = {"gps": (0, 0, 255), "other": (0, 220, 0), "ignore": (0, 220, 255)}
    tiles = []
    for p in pick:
        f = int(p.stem.split("_")[0])
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        objs = json.loads(p.read_text())
        for o in sorted(objs, key=lambda o: o["role"] == "gps"):
            b = get_lidar_3d_8points([o["l"], o["w"], o["h"]], o["yaw"], [o["x"], o["y"], o["z"] + o["h"] / 2])
            cc = (l2c @ np.c_[b, np.ones(8)].T).T[:, :3]
            if np.sum(cc[:, 2] > 1e-6) < 8:
                continue
            draw_box_3d(img, project_to_image(cc, k34), c=col[o["role"]])
        g = objs[0]
        txt = f"{clip} frame {f}  GPS car x={g['x']:.1f} y={g['y']:.1f} m  {g['vx'] and math.hypot(g['vx'], g['vy']) * 2.23694:.0f} mph"
        cv2.putText(img, txt, (20, 50), 0, 1.2, (0, 0, 0), 6)
        cv2.putText(img, txt, (20, 50), 0, 1.2, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (960, 540)))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    out = ROOT / "outputs/reports/gps_labels"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{clip}_labels.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("   review ->", out / f"{clip}_labels.jpg")


def _selfcheck():
    """Labels exist only for training clips, the GPS car comes first in every frame,
    lies inside the grid, and no other-car label sits on top of it."""
    for clip in CLIPS:
        for p in (OUT / clip).glob("*_label.json"):
            objs = json.loads(p.read_text())
            g = objs[0]
            assert g["role"] == "gps" and sum(o["role"] == "gps" for o in objs) == 1, p
            assert X_RANGE[0] <= g["x"] <= X_RANGE[1], p
            assert int(p.stem[:3]) <= LAST_CLEAN_FRAME.get(clip, 10 ** 6), p
            for o in objs[1:]:
                assert not (abs(o["x"] - g["x"]) < SAME_CAR[0] and abs(o["y"] - g["y"]) < SAME_CAR[1]), p
    assert not (OUT / TEST_CLIP).exists(), "test clip must never get GPS labels"
    print("selfcheck ok (one GPS car per frame, in range, no duplicate, no test-clip labels)")


def main():
    ap = argparse.ArgumentParser("GPS labels for masked-loss fine-tuning")
    ap.add_argument("--review", action="store_true", help="also write contact sheets")
    a = ap.parse_args()
    T, taus = load_site(), offsets()
    mans = [build(c, T, taus[c]) for c in CLIPS]
    (OUT / "summary.json").write_text(json.dumps(mans, indent=2))
    _selfcheck()
    if a.review:
        for c in CLIPS:
            review(c)


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
