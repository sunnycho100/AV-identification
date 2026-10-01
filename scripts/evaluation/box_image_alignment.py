"""How well do 3D boxes (labels or detections) land on the cars in the image?

No GPS. The clip's median frame is the empty road; what differs from it is
moving cars (and their shadows). For each box, the 3D box is projected to a 2D
rectangle and matched to the foreground blob it overlaps most. Only one-to-one
matches count (one box per blob, one blob per box), so merged cars are skipped.

Reported per range band:
  iou        2D overlap of the projected box with its blob
  ray_m      where the box meets the road minus where the car meets the road,
             along the line of sight in metres (+ = box farther than the car)
  side_px    horizontal offset of the box's 2D centre from the blob's, pixels
The road contact of the car is the blob's lowest row (median column there),
back-projected to the road plane; the box's is the lowest of its 4 bottom
corners. Both are the point nearest the camera, so a range error in the box
shows up directly as ray_m.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/box_image_alignment.py \
        --clip HV_T_EW_1 --labels
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/box_image_alignment.py \
        --clip AV_T_EW_3 --run h151p
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANDS = [(0, 40), (40, 60), (60, 80), (80, 140)]


def corners8(l, w, h, yaw, x, y, z):
    """Box corners, length along the heading, z = box bottom."""
    c, s = math.cos(yaw), math.sin(yaw)
    bx = np.array([l, l, -l, -l] * 2) / 2
    by = np.array([w, -w, -w, w] * 2) / 2
    bz = np.array([0] * 4 + [h] * 4, float)
    return np.c_[x + c * bx - s * by, y + s * bx + c * by, z + bz]


def project(K, M, pts):
    q = (M[:3, :3] @ pts.T).T + M[:3, 3]
    uv = (K @ q.T).T
    return uv[:, :2] / uv[:, 2:], q[:, 2]


def to_ground(K, M, u, v, z):
    """Pixel to the road plane z, in the box frame."""
    R, t = M[:3, :3], M[:3, 3]
    ray = R.T @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    cam = -R.T @ t
    s = (z - cam[2]) / ray[2]
    return cam + s * ray, cam


def background(frames):
    pick = frames[::max(1, len(frames) // 60)]
    return np.median(np.stack([cv2.imread(str(f)) for f in pick]), axis=0).astype(np.uint8)


def blobs(img, bg):
    d = cv2.absdiff(img, bg).max(axis=2)
    m = (d > 35).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(lab == i, st[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 150]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def measure(boxes, img, bg, K, M, z_road):
    H, W = img.shape[:2]
    bl = blobs(img, bg)
    rects = [(st[0], st[1], st[0] + st[2], st[1] + st[3]) for _, st in bl]
    cand = []
    for i, b in enumerate(boxes):
        c = corners8(b["l"], b["w"], b["h"], b["yaw"], b["x"], b["y"], b["z"])
        uv, depth = project(K, M, c)
        if (depth <= 1).any():
            continue
        r = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        if r[0] < 5 or r[1] < 5 or r[2] > W - 5 or r[3] > H - 5:
            continue            # cut by the frame edge: the blob is cut too
        ious = [iou(r, q) for q in rects]
        if ious and max(ious) > 0.1:
            cand.append((i, int(np.argmax(ious)), max(ious), r, uv))
    # one-to-one only
    by_blob = {}
    for c in cand:
        by_blob.setdefault(c[1], []).append(c)
    out = []
    for j, cs in by_blob.items():
        if len(cs) != 1:
            continue
        i, _, ov, r, uv = cs[0]
        if sum(1 for c in cand if c[0] == i) != 1:
            continue
        mask = bl[j][0]
        ys, xs = np.nonzero(mask)
        vb = ys.max()
        ub = float(np.median(xs[ys >= vb - 2]))
        car_pt, cam = to_ground(K, M, ub, vb + 0.5, z_road)
        bottom = uv[:4]
        k = int(np.argmax(bottom[:, 1]))
        box_pt, _ = to_ground(K, M, bottom[k, 0], bottom[k, 1], z_road)
        rng = float(np.hypot(*(box_pt[:2] - cam[:2])))
        ray_m = rng - float(np.hypot(*(car_pt[:2] - cam[:2])))
        side_px = (r[0] + r[2]) / 2 - (rects[j][0] + rects[j][2]) / 2
        out.append({"src": boxes[i].get("yaw_source", "model"), "range": float(np.hypot(boxes[i]["x"], boxes[i]["y"])),
                    "iou": ov, "ray_m": ray_m, "side_px": side_px})
    return out


def summarize(rows, title):
    print(title)
    for lo, hi in BANDS:
        r = [x for x in rows if lo <= x["range"] < hi]
        if len(r) < 5:
            continue
        a = lambda k: np.median([x[k] for x in r])
        print(f"  {lo:3d}-{hi:3d} m  n {len(r):4d}  iou {a('iou'):.2f}  ray_m {a('ray_m'):+.2f}  "
              f"side_px {a('side_px'):+.1f}")


def main():
    ap = argparse.ArgumentParser("Box-to-car alignment in the image")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", help="detection run suffix, e.g. h151p")
    ap.add_argument("--labels", action="store_true", help="grade the fine-tuning pseudo-labels instead")
    a = ap.parse_args()
    det = ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{a.run or 'phase1'}"
    cal = json.loads((det / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z_road = cal.get("road_plane_z_in_output", -1.73)
    frames = sorted((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg"))
    bg = background(frames)
    rows = []
    for f in frames[::3]:
        if a.labels:
            p = ROOT / "outputs/finetune/pseudo_labels" / a.clip / f"{f.stem}_label.json"
            boxes = json.loads(p.read_text()) if p.exists() else []
        else:
            boxes = [b for b in json.loads((det / f"{f.stem}_pred.json").read_text()) if b["class_name"] == "car"]
        rows += measure(boxes, cv2.imread(str(f)), bg, K, M, z_road)
    name = f"{a.clip} {'labels' if a.labels else a.run}"
    summarize(rows, f"{name}: all")
    if a.labels:
        summarize([r for r in rows if r["src"] == "track_motion"], f"{name}: corrected from track")


def _selfcheck():
    """A box placed exactly on a synthetic car scores ray_m ~0; moved 2 m away, ~+2."""
    K = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]])
    th = math.radians(20)
    R = np.array([[0, -1, 0], [-math.sin(th), 0, -math.cos(th)], [math.cos(th), 0, -math.sin(th)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ np.array([0, 0, 15.0])
    car = {"l": 4.5, "w": 1.8, "h": 1.5, "yaw": 0.0, "x": 50.0, "y": 0.0, "z": 0.0}
    img = np.zeros((1080, 1920, 3), np.uint8)
    uv, _ = project(K, M, corners8(**{k: car[k] for k in ("l", "w", "h", "yaw", "x", "y", "z")}))
    cv2.fillConvexPoly(img, cv2.convexHull(uv.astype(np.int32)), (255, 255, 255))
    bg = np.zeros_like(img)
    on = measure([car], img, bg, K, M, 0.0)[0]
    off = measure([dict(car, x=52.0)], img, bg, K, M, 0.0)[0]
    assert abs(on["ray_m"]) < 0.3, on
    assert 1.5 < off["ray_m"] < 2.5, off
    print(f"selfcheck ok (on the car {on['ray_m']:+.2f} m, moved 2 m {off['ray_m']:+.2f} m)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
