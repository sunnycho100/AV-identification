"""Side-by-side 'here -> there' for one tracked vehicle — anchored on REAL detections.

The key fix: the frames are not consecutive, so a track can be Kalman-coasted through
frames where BEVHeight detected nothing. Drawing the endpoint on such a frame puts the
marker on empty road. So here we only use frames where the vehicle has an ACTUAL
detection box: each track point is matched back to a real detection in that frame's
pred.json, coasted frames are dropped, and the real BEVHeight 3D box is drawn so you
can see the car at both the start and end panels.

  /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/highlight_vehicle.py --label seg_11_22
  ...                                                        --label seg_11_22 --id 15
"""
import argparse
import json
import math

import cv2
import numpy as np

from overlay_trajectories import ROOT, TRK_ROOT, DATA_ROOT, load_K, load_lidar2cam, project

DET_DIR = ROOT / "outputs/object_detection/our_intrinsics"

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 0, 255)     # bold red: marker + trail
BOX_COLOR = (0, 255, 0)  # green: the real BEVHeight detection box
MATCH_THRESH_M = 2.5     # a track point matches a detection if within this (x,y) distance


def get_lidar_3d_8points(obj_size, yaw, center):
    # replicated from evaluators/result2kitti.py (avoids that module's heavy imports)
    l, w, h = obj_size
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2]) - h / 2
    r = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    corners = np.array([
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0, 0, 0, 0, h, h, h, h]])
    return (r @ corners + np.array([[cx], [cy], [cz]])).T  # 8x3


def load_dets(frame):
    path = DET_DIR / f"{frame:06d}_pred.json"
    objs = json.loads(path.read_text())
    return [o for o in objs if o["class_name"] == "car"]


def match_detection(point, dets):
    """Return the car detection nearest to the track point in (x,y), or None if none within thresh."""
    best, bestd = None, MATCH_THRESH_M
    for o in dets:
        d = math.hypot(o["x"] - point["x"], o["y"] - point["y"])
        if d < bestd:
            best, bestd = o, d
    return best


def real_appearances(pts):
    """Track points that correspond to an actual detection box, with that box attached."""
    out = []
    for p in pts:
        det = match_detection(p, load_dets(p["frame"]))
        if det is not None:
            out.append((p["frame"], det))
    return out


def draw_box(img, det, lidar2cam, K):
    center = [det["x"], det["y"], det["z"] + det["h"] / 2.0]
    corners = get_lidar_3d_8points([det["l"], det["w"], det["h"]], det["yaw"], center)
    cam = (lidar2cam @ np.c_[corners, np.ones(8)].T).T[:, :3]
    if np.sum(cam[:, 2] > 1e-6) < 4:
        return None
    pts2d = (K @ cam.T).T
    pts2d = pts2d[:, :2] / pts2d[:, 2:3]
    face = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for f in face:
        for j in range(4):
            a = tuple(np.round(pts2d[f[j]]).astype(int))
            b = tuple(np.round(pts2d[f[(j + 1) % 4]]).astype(int))
            cv2.line(img, a, b, BOX_COLOR, 2, cv2.LINE_AA)
    return tuple(np.round(pts2d.mean(axis=0)).astype(int))  # box center in pixels


def inset(img, pt, w=220, h=160, zoom=2):
    H, W = img.shape[:2]
    x0, y0 = max(0, min(pt[0] - w // 2, W - w)), max(0, min(pt[1] - h // 2, H - h))
    return cv2.resize(img[y0:y0 + h, x0:x0 + w].copy(), (w * zoom, h * zoom))


def draw_panel(frame, real_pts, sel_det, tid, tag, K):
    img = cv2.imread(str(DATA_ROOT / "image" / f"{frame:06d}.jpg"))
    l2c = load_lidar2cam(frame)
    # honest trail: only through real-detection positions
    trail = [project((d["x"], d["y"], d["z"]), l2c, K) for _, d in real_pts]
    trail = [t for t in trail if t]
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(img, a, b, COLOR, 2, cv2.LINE_AA)
    center_px = draw_box(img, sel_det, l2c, K)
    pt = center_px or project((sel_det["x"], sel_det["y"], sel_det["z"]), l2c, K)
    cv2.drawMarker(img, pt, COLOR, cv2.MARKER_STAR, 30, 2, cv2.LINE_AA)
    ins = inset(img, pt)
    ih, iw = ins.shape[:2]
    img[34:34 + ih, img.shape[1] - iw:] = ins
    cv2.rectangle(img, (img.shape[1] - iw, 34), (img.shape[1] - 1, 34 + ih), COLOR, 2)
    cv2.rectangle(img, (0, 0), (760, 32), (0, 0, 0), -1)
    cv2.putText(img, f"ID {tid} | frame {frame:06d} ({tag}) | score {sel_det['score']:.2f}",
                (8, 23), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def highlight(label, tid):
    tracks = json.loads((TRK_ROOT / label / "tracks.json").read_text())["tracks"]
    real = real_appearances(tracks[str(tid)])
    if len(real) < 2:
        print(f"  ID {tid}: only {len(real)} real detection(s) — cannot show here->there; skipped")
        return False
    (f0, d0), (f1, d1) = real[0], real[-1]
    move = math.hypot(d1["x"] - d0["x"], d1["y"] - d0["y"])
    K = load_K()
    left = draw_panel(f0, real, d0, tid, "start", K)
    right = draw_panel(f1, real, d1, tid, "end", K)
    combined = np.hstack([left, right])
    cv2.putText(combined, f"net move {move:.1f} m | {len(real)} real detections "
                f"(frames {f0:06d} - {f1:06d})", (8, combined.shape[0] - 16),
                FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    out = TRK_ROOT / label / f"highlight_id{tid}.jpg"
    cv2.imwrite(str(out), combined)
    print(f"  ID {tid}: {len(real)} real dets, frames {f0:06d}->{f1:06d}, net {move:.1f} m -> {out.name}")
    return True


def rank_candidates(label):
    """Rank tracks by (real appearances, real displacement) so we pick ones that truly move and are seen."""
    tracks = json.loads((TRK_ROOT / label / "tracks.json").read_text())["tracks"]
    rows = []
    for tid, pts in tracks.items():
        real = real_appearances(pts)
        if len(real) < 2:
            continue
        move = math.hypot(real[-1][1]["x"] - real[0][1]["x"], real[-1][1]["y"] - real[0][1]["y"])
        rows.append((int(tid), len(real), round(move, 1)))
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Side-by-side highlight anchored on real detections")
    ap.add_argument("--label", required=True)
    ap.add_argument("--id", type=int, default=None, help="specific track ID")
    ap.add_argument("--top", type=int, default=5, help="if no --id, generate this many top candidates")
    args = ap.parse_args()

    if args.id is not None:
        highlight(args.label, args.id)
        return

    ranked = rank_candidates(args.label)
    print(f"{args.label}: candidates (id, real_detections, net_move_m):")
    for tid, n, mv in ranked[:12]:
        print(f"  ID {tid:>3}: {n} real dets, {mv} m")
    print("generating top movers:")
    movers = [r for r in ranked if r[2] >= 2.0][:args.top]
    for tid, _, _ in movers:
        highlight(args.label, tid)
    # also include the most-seen near-still vehicle (a clean 'stayed put' example)
    still = [r for r in ranked if r[2] < 1.0]
    if still:
        highlight(args.label, still[0][0])


if __name__ == "__main__":
    main()
