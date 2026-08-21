"""Turn tracked detections into training labels for fine-tuning.

The model's own output cannot be used as its own supervision unchanged: it would
simply relearn its current mistakes. So every field a label carries is either
kept because we measured it to be good, or replaced from a source independent of
the network:

  position  kept       graded against GPS at ~1 m, the part that already works
  yaw       REPLACED   from the track's direction of travel, not the box. The box
                       heading is the known weak spot (~10% of boxes >45 deg off);
                       displacement between frames is geometry, not appearance.
  z         REPLACED   snapped to the road plane, which calibration fixes exactly
  l,w,h     REPLACED   per-track median, killing frame-to-frame size jitter

Selection is deliberately strict. A pseudo-label set should be small and clean
rather than large and noisy, because every wrong label is actively taught.

RECALL IS PART OF THE LABEL, NOT JUST A LIMITATION. An early version emitted only
the vehicles that passed the track gates, roughly 5 per frame against the ~26 the
model actually detects. Training on that collapsed the model to detecting nothing
in 24 steps, because every correctly-found car missing from the label set is read
as a negative. So the label set now contains EVERY confident detection, and the
track-derived corrections are applied to the subset that has a trustworthy track.
Cars without one keep their original box: not ideal, but far better than teaching
the model they are not cars.

Known limitation, state it when reporting: this can only refine vehicles the
model already finds. Cars it misses entirely stay missing, so pseudo-labelling
cannot fix recall.

    .venv/bin/python scripts/finetune/make_pseudo_labels.py --clip HV_T_EW_1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MIN_TRACK_FRAMES = 15     # shorter tracks have unreliable direction estimates
MIN_MEAN_SPEED = 5.0      # m/s; parked cars and static false positives are out
MIN_MEDIAN_SCORE = 0.40   # above the 0.3 detection floor: labels should be confident
YAW_FIT_DEGREE = 2        # quadratic path fit: straight enough to reject position
                          # noise, flexible enough to keep a real lane change
MAX_YAW_JITTER_DEG = 25.0 # a real vehicle's heading cannot swing more than this
                          # across a clean track; if it does, the track is a mess
MIN_DET_COVERAGE = 0.8    # AB3DMOT coasts through gaps (max_age 6). Those states
                          # have no image evidence, so they are dropped rather than
                          # labelled, but they must not disqualify the whole track.
ROAD_Z_TOL = 1.0          # m; a box floating this far off the road is not trusted


def track_yaws(road_positions):
    """Heading at every state, from the tangent of a smooth path fitted to the track.

    Differencing raw positions over a short window looks obvious but amplifies
    position noise badly: on the marker-identified target car it reported 26.4 deg
    of heading swing where the geometry allows only 1.3 deg (1.9 m of lateral drift
    over 83 m). Fitting x(t) and y(t) as low-order polynomials and taking the
    tangent gives 3.1 deg on the same track. Degree 2 is deliberate: a straight line
    could not represent a lane change, and a higher order would start chasing the
    noise again.
    """
    n = len(road_positions)
    if n < 5:
        return None
    t = np.arange(n)
    deg = min(YAW_FIT_DEGREE, n - 1)
    dx = np.polyval(np.polyder(np.polyfit(t, road_positions[:, 0], deg)), t)
    dy = np.polyval(np.polyder(np.polyfit(t, road_positions[:, 1], deg)), t)
    if np.median(np.hypot(dx, dy)) < 1e-3:
        return None
    return np.arctan2(dy, dx)


def main():
    ap = argparse.ArgumentParser("Build pseudo-labels from tracks")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    clip = args.clip

    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_phase1"
    trk = ROOT / f"outputs/tracking/camera-data/{clip}_phase1/tracks.json"
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    road_z = cal["road_plane_z_in_output"]
    tracks = json.loads(trk.read_text())["tracks"]

    # per-frame detection sizes, so a label's dimensions come from real detections
    sizes = {}
    for pf in sorted(det_dir.glob("*_pred.json")):
        f = int(pf.stem.split("_")[0])
        sizes[f] = [(np.array([o["x"], o["y"]]), (o["l"], o["w"], o["h"]), o["score"])
                    for o in json.loads(pf.read_text())
                    if o["class_name"] == "car" and o["score"] >= 0.3]

    def det_for(f, s):
        best, bd = None, 2.0
        for xy, lwh, sc in sizes.get(f, []):
            d = float(np.linalg.norm(xy - np.array([s["x"], s["y"]])))
            if d < bd:
                best, bd = (lwh, sc), d
        return best

    kept, rejected = {}, {"short": 0, "slow": 0, "low_score": 0,
                          "yaw_jitter": 0, "off_road": 0, "no_detection": 0}
    n_labels = 0
    for tid, states in tracks.items():
        if len(states) < MIN_TRACK_FRAMES:
            rejected["short"] += 1
            continue
        if float(np.mean([s["speed_mps"] for s in states])) < MIN_MEAN_SPEED:
            rejected["slow"] += 1
            continue
        matched_all = [(i, s, det_for(s["frame"], s)) for i, s in enumerate(states)]
        matched = [(i, s, m) for i, s, m in matched_all if m is not None]
        if len(matched) < MIN_DET_COVERAGE * len(states):
            rejected["no_detection"] += 1
            continue
        scores = [m[1] for _, _, m in matched]
        if float(np.median(scores)) < MIN_MEDIAN_SCORE:
            rejected["low_score"] += 1
            continue
        if abs(float(np.median([s["z"] for s in states])) - road_z) > ROAD_Z_TOL:
            rejected["off_road"] += 1
            continue

        pos = np.array([[s["x"], s["y"]] for s in states])
        yaws = track_yaws(pos)
        if yaws is None:
            rejected["slow"] += 1
            continue
        # heading must be coherent along the track, else the track itself is bad
        unwrapped = np.unwrap(yaws)
        if float(np.degrees(unwrapped.max() - unwrapped.min())) > MAX_YAW_JITTER_DEG:
            rejected["yaw_jitter"] += 1
            continue

        dims = np.median(np.array([m[0] for _, _, m in matched]), axis=0)
        for i, s, m in matched:
            kept.setdefault(s["frame"], []).append({
                "track_id": tid, "class_name": "car",
                "x": s["x"], "y": s["y"], "z": road_z,      # snapped to the road
                "l": float(dims[0]), "w": float(dims[1]), "h": float(dims[2]),
                "yaw": float(yaws[i]),                       # from motion
                # the head regresses velocity too (code weight 0.5), so labels
                # must carry it or training would teach "every car is stationary"
                "vx": float(s["vx"]), "vy": float(s["vy"]),
                "score": float(m[1]),
                "yaw_source": "track_motion", "z_source": "road_plane",
            })
            n_labels += 1

    # Every confident detection becomes a label; corrected where a good track
    # exists, original otherwise. Keeps the label set's recall equal to the
    # model's, which is what stops the collapse.
    corrected = {(f, round(o["x"], 3), round(o["y"], 3)): o
                 for f, objs in kept.items() for o in objs}
    full = {}
    n_corr = n_orig = 0
    for pf in sorted(det_dir.glob("*_pred.json")):
        f = int(pf.stem.split("_")[0])
        rows = []
        for o in json.loads(pf.read_text()):
            if o["class_name"] != "car" or o["score"] < 0.3:
                continue
            # nearest match, not exact: track states are Kalman-filtered so they
            # never equal the detection they came from
            hit, hd = None, 2.0
            for lab in kept.get(f, []):
                d = float(np.hypot(lab["x"] - o["x"], lab["y"] - o["y"]))
                if d < hd:
                    hit, hd = lab, d
            if hit is not None:
                rows.append(hit); n_corr += 1
            else:
                rows.append({**o, "track_id": None, "vx": 0.0, "vy": 0.0,
                             "yaw_source": "model", "z_source": "model"})
                n_orig += 1
        if rows:
            full[f] = rows
    kept = full
    n_labels = n_corr + n_orig

    out = Path(args.out or f"outputs/finetune/pseudo_labels/{clip}")
    out.mkdir(parents=True, exist_ok=True)
    for f, objs in kept.items():
        (out / f"{f:03d}_label.json").write_text(json.dumps(objs, indent=2))
    n_tracks = len({o["track_id"] for objs in kept.values() for o in objs} - {None})
    manifest = {
        "clip": clip, "road_plane_z": road_z,
        "frames_with_labels": len(kept), "labels": n_labels, "tracks_used": n_tracks,
        "tracks_total": len(tracks), "rejected": rejected,
        "gates": {"min_track_frames": MIN_TRACK_FRAMES, "min_mean_speed": MIN_MEAN_SPEED,
                  "min_median_score": MIN_MEDIAN_SCORE,
                  "max_yaw_jitter_deg": MAX_YAW_JITTER_DEG},
        "provenance": "position from track; yaw from motion; z from road plane; "
                      "dims from per-track median. GPS not used.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["labels_corrected"] = n_corr
    manifest["labels_kept_as_model"] = n_orig
    print(f"{clip}: {n_labels} labels on {len(kept)} frames "
          f"({n_corr} track-corrected, {n_orig} kept as model output) "
          f"from {n_tracks}/{len(tracks)} tracks")
    print(f"   rejected: {rejected}")
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
