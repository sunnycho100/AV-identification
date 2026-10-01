"""Correct pseudo-label positions against the car outlines in the image.

Why: BEVHeight places boxes about 1 m too close to the camera along the line
of sight (scripts/evaluation/box_image_alignment.py; the same with box height
free, so not a bumper or box-height artefact). The pseudo-labels inherit it,
and fine-tuning anything that learns position would bake it in. GPS cannot see
it, because every GPS grade first fits a free rigid transform.

How, image only, no GPS:
  1. Moving cars are what differs from the clip's median frame (the empty road).
  2. Each label that matches exactly one car outline (and vice versa) is slid
     along the line of sight, SHIFTS_M, to the shift where its projected 3D box
     overlaps the outline best.
  3. A track's shift is the median over its matched frames (MIN_FITS or more),
     applied to all its labels. Labels with no usable track get the clip's
     median shift: the bias is close to constant with range.
  4. Labels off the road (road_mask.png from build_road_mask.py) are dropped:
     parked cars, the location banner, snow.
  5. Labels kept from the model (no usable track, mostly far cars whose
     tracks break into pieces under 15 frames) still carry the pretrained
     model's heading: 5-11 deg off the lane at the median and backwards for
     most oncoming cars. They take the heading of the track-corrected labels in
     the same stretch of carriageway (LANE_DY_M across, LANE_DX_M along, any frame)
     when those clearly agree; otherwise they are left as they are.
  6. Two labels whose road footprints overlap in one frame are one vehicle
     seen twice (e.g. a tow truck and the car on it); only the best is kept:
     own track, then lane flow, then model, then score.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/finetune/correct_labels.py --clip HV_T_EW_1
writes outputs/finetune/pseudo_labels_v2/<clip>/ (same file layout as v1) with
each label's applied shift_m and its source.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/object_detection")]
import box_image_alignment as B   # noqa: E402
import road_mask as rm            # noqa: E402

SHIFTS_M = np.arange(-3.0, 3.01, 0.1)
MIN_FITS = 5          # matched frames before a track's own shift is trusted
MIN_IOU = 0.4         # a fit below this overlap is not a clean match
LANE_DY_M, LANE_DX_M = 5.0, 30.0    # one carriageway: all its lanes run the same way, the two are ~6 m apart
MIN_LANE_VOTES, MIN_LANE_AGREE = 10, 0.9
RANK = {"track_motion": 0, "lane_flow": 1}


def ray_dir(b, cam):
    d = np.array([b["x"] - cam[0], b["y"] - cam[1]])
    return d / np.linalg.norm(d)


def silhouette(K, M, b, shift, cam, shape):
    d = ray_dir(b, cam)
    c = B.corners8(b["l"], b["w"], b["h"], b["yaw"], b["x"] + shift * d[0], b["y"] + shift * d[1], b["z"])
    uv, _ = B.project(K, M, c)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 1)
    return m


def best_shift(K, M, b, blob, cam):
    """(shift, overlap) that best fits the box silhouette to the car outline."""
    scores = []
    for s in SHIFTS_M:
        m = silhouette(K, M, b, s, cam, blob.shape)
        scores.append((m & blob).sum() / max((m | blob).sum(), 1))
    k = int(np.argmax(scores))
    return float(SHIFTS_M[k]), float(scores[k])


def lane_heading(b, moving):
    """Heading of tracked traffic where b is (circular mean of the
    track-corrected labels there), or None without clear evidence."""
    near = moving[(np.abs(moving[:, 1] - b["y"]) < LANE_DY_M) & (np.abs(moving[:, 0] - b["x"]) < LANE_DX_M)]
    if len(near) < MIN_LANE_VOTES:
        return None
    mean = np.arctan2(np.sin(near[:, 2]).mean(), np.cos(near[:, 2]).mean())
    if np.mean(np.cos(near[:, 2] - mean) > 0) < MIN_LANE_AGREE:
        return None          # traffic in both directions here: no single heading
    return float(mean)


def footprints_overlap(a, b):
    ra, rb = (((p["x"], p["y"]), (p["l"], p["w"]), float(np.degrees(p["yaw"]))) for p in (a, b))
    return cv2.rotatedRectangleIntersection(ra, rb)[0] != cv2.INTERSECT_NONE


def dedupe(labels):
    """Keep the best of any labels whose footprints overlap; returns (kept, n_dropped)."""
    order = sorted(labels, key=lambda b: (RANK.get(b.get("yaw_source"), 2), -b.get("score", 0)))
    kept = []
    for b in order:
        if not any(footprints_overlap(b, k) for k in kept):
            kept.append(b)
    return kept, len(labels) - len(kept)


def one_to_one(labels, img, bg, K, M):
    """(label index, blob mask) for labels that match exactly one outline."""
    H, W = img.shape[:2]
    bl = B.blobs(img, bg)
    rects = [(s[0], s[1], s[0] + s[2], s[1] + s[3]) for _, s in bl]
    hits = []
    for i, b in enumerate(labels):
        uv, depth = B.project(K, M, B.corners8(b["l"], b["w"], b["h"], b["yaw"], b["x"], b["y"], b["z"]))
        if (depth <= 1).any():
            continue
        r = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        if r[0] < 5 or r[1] < 5 or r[2] > W - 5 or r[3] > H - 5:
            continue
        ious = [B.iou(r, q) for q in rects]
        if ious and max(ious) > 0.1:
            hits.append((i, int(np.argmax(ious))))
    per_blob = {}
    for _, j in hits:
        per_blob[j] = per_blob.get(j, 0) + 1
    return [(i, bl[j][0].astype(np.uint8)) for i, j in hits if per_blob[j] == 1]


def main():
    ap = argparse.ArgumentParser("Correct pseudo-label positions from the image")
    ap.add_argument("--clip", required=True)
    a = ap.parse_args()
    src = ROOT / "outputs/finetune/pseudo_labels" / a.clip
    dst = ROOT / "outputs/finetune/pseudo_labels_v2" / a.clip
    dst.mkdir(parents=True, exist_ok=True)
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_phase1"
                      / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z = cal.get("road_plane_z_in_output", -1.73)
    cam = -M[:3, :3].T @ M[:3, 3]
    road = rm.load_built(a.clip)
    if road is None:
        raise SystemExit(f"no road_mask.png for {a.clip}: run scripts/calibration/build_road_mask.py first")
    frames = sorted((ROOT / "data/camera-data" / a.clip / "frames_all").glob("*.jpg"))
    bg = B.background(frames)
    labels = {f.stem: json.loads(p.read_text()) for f in frames
              if (p := src / f"{f.stem}_label.json").exists()}

    fits, per_track = [], {}
    for f in frames:
        if f.stem not in labels:
            continue
        for i, blob in one_to_one(labels[f.stem], cv2.imread(str(f)), bg, K, M):
            s, ov = best_shift(K, M, labels[f.stem][i], blob, cam)
            if ov >= MIN_IOU:
                fits.append(s)
                tid = labels[f.stem][i].get("track_id")
                if tid is not None:
                    per_track.setdefault(tid, []).append(s)
    clip_shift = float(np.median(fits))
    track_shift = {t: float(np.median(v)) for t, v in per_track.items() if len(v) >= MIN_FITS}

    moving = np.array([[b["x"], b["y"], b["yaw"]] for labs in labels.values() for b in labs
                       if b.get("yaw_source") == "track_motion"])
    n_in = n_out = n_off = n_flip = n_dup = 0
    for stem, labs in labels.items():
        out = []
        for b in labs:
            n_in += 1
            if not rm.keep(b, None, K, M, z, road):
                n_off += 1
                continue
            tid = b.get("track_id")
            s, how = (track_shift[tid], "track") if tid in track_shift else (clip_shift, "clip_median")
            d = ray_dir(b, cam)
            nb = {**b, "x": b["x"] + s * d[0], "y": b["y"] + s * d[1],
                  "shift_m": s, "pos_source": f"image_{how}"}
            if b.get("yaw_source") != "track_motion":
                lane = lane_heading(b, moving)
                if lane is not None:
                    nb["yaw"], nb["yaw_source"] = lane, "lane_flow"
                    n_flip += 1
            out.append(nb)
        out, d = dedupe(out)
        n_dup += d
        n_out += len(out)
        (dst / f"{stem}_label.json").write_text(json.dumps(out, indent=2))
    manifest = {"clip": a.clip, "from": str(src.relative_to(ROOT)), "labels_in": n_in, "labels_out": n_out,
                "dropped_off_road": n_off, "headings_from_lane_flow": n_flip, "duplicates_dropped": n_dup, "fits": len(fits), "clip_median_shift_m": clip_shift,
                "tracks_with_own_shift": len(track_shift),
                "track_shift_spread_m": float(np.subtract(*np.percentile(list(track_shift.values()), [75, 25])))
                if track_shift else None,
                "provenance": "v1 labels; position slid along the line of sight to fit the car outline "
                              "(background subtraction); off-road labels dropped; model-kept headings set from the "
                              "lane's tracked traffic. GPS not used."}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _selfcheck():
    """A label placed 1 m too close to a synthetic car is found to need +1 m."""
    K = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]])
    th = np.radians(20)
    R = np.array([[0, -1, 0], [-np.sin(th), 0, -np.cos(th)], [np.cos(th), 0, -np.sin(th)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ np.array([0, 0, 15.0])
    cam = -R.T @ M[:3, 3]
    car = {"l": 4.5, "w": 1.8, "h": 1.5, "yaw": 0.0, "x": 50.0, "y": -3.0, "z": 0.0}
    blob = silhouette(K, M, car, 0.0, cam, (1080, 1920))
    d = ray_dir(car, cam)
    label = dict(car, x=car["x"] - d[0], y=car["y"] - d[1])
    s, ov = best_shift(K, M, label, blob, cam)
    assert abs(s - 1.0) <= 0.15 and ov > 0.9, (s, ov)
    moving = np.array([[60.0 + i, -8.0, np.pi] for i in range(20)])          # oncoming lane
    assert abs(abs(lane_heading({"x": 62.0, "y": -8.5}, moving)) - np.pi) < 1e-6
    assert lane_heading({"x": 62.0, "y": -30.0}, moving) is None                # no evidence there
    box = {"l": 4.3, "w": 1.8, "yaw": 0.0}
    kept, d = dedupe([{**box, "x": 50.0, "y": -36.0, "yaw_source": "model", "score": 0.9},
                      {**box, "x": 53.0, "y": -36.6, "yaw_source": "lane_flow", "score": 0.5},   # 3 m apart, overlapping
                      {**box, "x": 60.0, "y": -36.0, "yaw_source": "model", "score": 0.5}])
    assert d == 1 and [k["yaw_source"] for k in kept] == ["lane_flow", "model"], kept
    print(f"selfcheck ok (label 1 m too close -> shift {s:+.1f} m, overlap {ov:.2f})")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
