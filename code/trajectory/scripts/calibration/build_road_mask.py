"""Road mask for one camera, built from where moving cars actually drove.

Why: road_mask.py's hand-picked band works for Todd Drive but has to be chosen
by eye for every new camera. Here the road is whatever moving traffic covered:
every track that travels at least MIN_TRAVEL_M paints a stroke LANE_HALF_M wide
(in metres, so the stroke narrows with distance) along its path in the image.
Parked cars never travel, so they never paint. Separate roads (Verona's road
under the bridge) come out as separate regions, numbered in the overlay, and
--keep chooses which ones count. Same tracks + same settings = same mask.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/calibration/build_road_mask.py \
        --clip AV_T_EW_3 --from AV_T_EW_3 AV_T_WE_1 AV_T_WE_3 HV_T_EW_1 HV_T_EW_2
--from pools tracks from other clips of the same camera (each projected with its
own calibration): far-away cars rarely hold a track long enough in one clip.
writes outputs/calibration/camera-data/<clip>/road_mask.png (255 = road) and
road_mask_overlay.jpg for review.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MIN_TRAVEL_M = 15.0     # start to end; parked cars jitter well under this
MIN_STATES = 15
LANE_HALF_M = 2.5       # stroke half width: half a lane plus box noise
MIN_REGION_PX = 5000    # drop specks from a single stray track


def paint(tracks, K, M, z, shape):
    mask = np.zeros(shape, np.uint8)
    f = K[0, 0]
    for t in tracks:
        xy = np.array([[s["x"], s["y"]] for s in t], float)
        if len(t) < MIN_STATES or np.linalg.norm(xy[-1] - xy[0]) < MIN_TRAVEL_M:
            continue
        cam = (M[:3, :3] @ np.c_[xy, np.full(len(xy), z)].T).T + M[:3, 3]
        ok = cam[:, 2] > 1.0
        px = (K @ cam[ok].T).T
        px = px[:, :2] / px[:, 2:]
        r = np.maximum(1, f * LANE_HALF_M / cam[ok, 2]).astype(int)
        for i in range(len(px)):
            p = tuple(int(v) for v in px[i])
            cv2.circle(mask, p, int(r[i]), 255, -1)
            if i:
                cv2.line(mask, tuple(int(v) for v in px[i - 1]), p, 255, int(2 * min(r[i], r[i - 1])))
    return mask


def regions(mask):
    """Connected road regions, largest first, specks dropped."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_PX]
    ids.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [(lab == i) for i in ids]


def main():
    ap = argparse.ArgumentParser("Build a camera's road mask from moving tracks")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--from", dest="sources", nargs="+", help="clips of the same camera to pool "
                    "(default: --clip only)")
    ap.add_argument("--run", default="phase1", help="detection and tracking run to build from")
    ap.add_argument("--keep", default="all", help="region numbers to keep, e.g. 1,2 (default all)")
    a = ap.parse_args()
    frame = next((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg"))
    img = cv2.imread(str(frame))
    sources = a.sources or [a.clip]
    raw = np.zeros(img.shape[:2], np.uint8)
    for c in sources:
        cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{c}_{a.run}"
                          / "calibration_used.json").read_text())
        tracks = json.loads((ROOT / "outputs/tracking/camera-data" / f"{c}_{a.run}"
                             / "tracks.json").read_text())["tracks"].values()
        raw |= paint(tracks, np.array(cal["K"]), np.array(cal["lidar2cam"]),
                     cal.get("road_plane_z_in_output", -1.73), img.shape[:2])
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    regs = regions(raw)
    keep = range(1, len(regs) + 1) if a.keep == "all" else [int(k) for k in a.keep.split(",")]
    mask = np.zeros(img.shape[:2], np.uint8)
    for k in keep:
        mask[regs[k - 1]] = 255

    out = ROOT / "outputs/calibration/camera-data" / a.clip
    cv2.imwrite(str(out / "road_mask.png"), mask)
    over = img.copy()
    over[mask == 0] = (over[mask == 0] * 0.45 + np.array([0, 0, 160]) * 0.55).astype(np.uint8)
    for k, r in enumerate(regs, 1):
        cs, _ = cv2.findContours(r.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cs, -1, (0, 255, 255) if k in keep else (160, 160, 160), 3)
        ys, xs = np.nonzero(r)
        cv2.putText(over, str(k), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 5, cv2.LINE_AA)
    cv2.imwrite(str(out / "road_mask_overlay.jpg"), over)
    (out / "road_mask.json").write_text(json.dumps({
        "built_from": [f"{c}_{a.run}" for c in sources], "regions": len(regs), "kept": list(keep),
        "min_travel_m": MIN_TRAVEL_M, "min_states": MIN_STATES, "lane_half_m": LANE_HALF_M}, indent=2))
    print(f"{len(regs)} regions, kept {list(keep)}; wrote {out / 'road_mask.png'}")


def _selfcheck():
    """A moving car paints its lane, a parked one paints nothing."""
    K = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]          # x forward, camera at the origin
    M[:3, 3] = [0, 10, 0]                                      # 10 m above the road
    mover = [{"x": 20 + i, "y": -2.0} for i in range(30)]
    parked = [{"x": 30 + 0.05 * (i % 2), "y": 6.0} for i in range(60)]
    m = paint([mover, parked], K, M, 0.0, (1080, 1920))
    lane = K @ (M[:3, :3] @ np.array([35, -2.0, 0]) + M[:3, 3])
    lot = K @ (M[:3, :3] @ np.array([30, 6.0, 0]) + M[:3, 3])
    assert m[int(lane[1] / lane[2]), int(lane[0] / lane[2])] == 255
    assert m[int(lot[1] / lot[2]), int(lot[0] / lot[2])] == 0
    print("selfcheck ok (moving track painted, parked car not)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
