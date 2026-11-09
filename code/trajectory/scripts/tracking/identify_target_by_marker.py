"""Identify WHICH tracked vehicle is the instrumented car, using only the image.

The GPS log is never opened here. That is the whole point: `analyze_tracks_vs_gps`
currently picks whichever track best matches the GPS and then reports that track's
error, which is selection on the very metric being reported and biases it
optimistically. Choosing the target from the hood marker instead makes the later
GPS comparison an honest test of a decision already made.

Method: the marker matcher gives a pixel box per frame over its coherent run. Each
tracked vehicle's 3D box is projected into the same frame. The track whose box
contains the marker most often across the run wins, by vote, so one bad frame
cannot decide it.

    .venv/bin/python scripts/tracking/identify_target_by_marker.py --clip HV_T_EW_1
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_converter.visual_utils import project_to_image

MARGIN_PX = 25.0   # a marker just outside a box still counts: the hood marker sits
                   # at the very front face, where box edges are least accurate


def corners8(size, yaw, bottom_centre):
    l, w, h = size
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                  [0, 0, 0, 0, h, h, h, h]])
    return (R @ b + np.array(bottom_centre).reshape(3, 1)).T


def marker_centres(marker_json):
    """Pixel centre of the marker for every frame inside an accepted target run."""
    d = json.loads(Path(marker_json).read_text())
    runs = d.get("target_runs", [])
    if not runs:
        return {}, d
    spans = [(r["start"], r["end"]) for r in runs]
    out = {}
    for fid, rec in d["per_frame"].items():
        f = int(fid)
        if not any(a <= f <= b for a, b in spans):
            continue
        if rec.get("score", 0) < d.get("score_min", 0.8):
            continue
        out[f] = (rec["x"] + rec["w"] / 2.0, rec["y"] + rec["h"] / 2.0)
    return out, d


def main():
    ap = argparse.ArgumentParser("Pick the target track from the hood marker")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--marker-json", default=None)
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clip = args.clip
    tracks_p = Path(args.tracks or f"outputs/tracking/camera-data/{clip}_phase1/tracks.json")
    marker_p = Path(args.marker_json or f"outputs/target_id/{clip}/marker_matches.json")
    det_dir = Path(args.det_dir or f"outputs/object_detection/camera-data/{clip}_phase1")

    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K = np.array(cal["K"]); l2c = np.array(cal["lidar2cam"])
    k34 = np.zeros((3, 4)); k34[:3, :3] = K

    centres, mdiag = marker_centres(marker_p)
    if not centres:
        sys.exit(f"no accepted marker run in {marker_p}")
    tracks = json.loads(tracks_p.read_text())["tracks"]

    # Track states carry position but not box size, so each state is matched back
    # to the detection it came from to recover l/w/h. Falling back to a nominal
    # car would quietly change the containment test the vote depends on.
    dims = {}
    for f in sorted(centres):
        pf = det_dir / f"{f:03d}_pred.json"
        if not pf.exists():
            continue
        dims[f] = [(np.array([o["x"], o["y"]]), (o["l"], o["w"], o["h"]))
                   for o in json.loads(pf.read_text())
                   if o["class_name"] == "car" and o["score"] >= 0.3]

    def size_for(f, s_):
        cands = dims.get(f, [])
        if not cands:
            return None
        xy = np.array([s_["x"], s_["y"]])
        d = [np.linalg.norm(c[0] - xy) for c in cands]
        i = int(np.argmin(d))
        return cands[i][1] if d[i] < 2.0 else None

    # index track states by frame for a single pass over the marker's frames
    by_frame = {}
    for tid, states in tracks.items():
        for s in states:
            by_frame.setdefault(s["frame"], []).append((tid, s))

    votes, dists = {}, {}
    for f, (mx, my) in sorted(centres.items()):
        for tid, s in by_frame.get(f, []):
            size = size_for(f, s)
            if size is None:
                continue
            pts = corners8(list(size), s["yaw"], [s["x"], s["y"], s["z"]])
            cc = (l2c @ np.c_[pts, np.ones(8)].T).T[:, :3]
            if np.sum(cc[:, 2] > 1e-6) < 4:
                continue
            uv = project_to_image(cc, k34)
            hull = cv2.convexHull(uv.astype(np.int32))
            d = cv2.pointPolygonTest(hull, (float(mx), float(my)), True)
            if d >= -MARGIN_PX:                       # inside, or just outside
                votes[tid] = votes.get(tid, 0) + 1
                dists.setdefault(tid, []).append(abs(min(d, 0.0)))

    if not votes:
        sys.exit("marker never fell inside any tracked box: no target identified")
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    best, n = ranked[0]
    total = len(centres)
    runner = ranked[1][1] if len(ranked) > 1 else 0

    print(f"marker present in {total} frames (run "
          f"{mdiag['target_runs'][0]['start']}-{mdiag['target_runs'][0]['end']}, "
          f"peak {mdiag['target_runs'][0]['peak_score']})")
    for tid, v in ranked[:5]:
        print(f"   track {tid:>4}: {v:3d} votes ({v/total:5.1%} of marker frames)")
    print(f"\nTARGET = track {best}  ({n}/{total} = {n/total:.1%}, "
          f"runner-up {runner})")
    if n < 0.5 * total:
        print("WARN: winner holds under half the marker frames; identification is weak")
    if runner and n < 2 * runner:
        print("WARN: winner and runner-up are close; the marker may be straddling "
              "two boxes (duplicate detections on the same car)")

    states = tracks[best]
    out = Path(args.out or f"outputs/tracking/camera-data/{clip}_phase1/target_track.json")
    out.write_text(json.dumps({
        "clip": clip, "target_track_id": best,
        "marker_frames": total, "votes": n, "runner_up_votes": runner,
        "identified_from": "hood marker template match (image only, no GPS)",
        "track_frames": len(states),
        "frame_range": [states[0]["frame"], states[-1]["frame"]],
        "states": states,
    }, indent=2))
    print(f"wrote {out}  ({len(states)} frames, "
          f"{states[0]['frame']}-{states[-1]['frame']})")


if __name__ == "__main__":
    main()
