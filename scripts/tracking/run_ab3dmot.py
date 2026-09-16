"""Run AB3DMOT on BEVHeight per-frame detections to get ID-linked vehicle trajectories.

Reads outputs/object_detection/<tag>/<frame>_pred.json (our BEVHeight output), feeds them
frame-by-frame into AB3DMOT's tracker core, and writes per-track trajectories with the
Kalman-filter velocity as the v0 motion feature.

Bypasses AB3DMOT's main.py / utils.py / io.py (wired to KITTI/nuScenes layouts). Uses only
the tracker core (model.AB3DMOT). Ego-motion compensation is OFF: fixed roadside camera.

Env: miniforge 3.12 (numpy/scipy/numba present; filterpy pip-installed).
Run: /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/run_ab3dmot.py
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AB3DMOT_DIR = ROOT / "third_party" / "AB3DMOT"
STUBS_DIR = Path(__file__).resolve().parent / "xinshuo_stubs"

# stubs first so AB3DMOT_libs imports resolve without the real Xinshuo_PyToolbox
sys.path.insert(0, str(STUBS_DIR))
sys.path.insert(0, str(AB3DMOT_DIR))

from AB3DMOT_libs.matching import data_association  # noqa: E402
from AB3DMOT_libs.model import AB3DMOT  # noqa: E402

# --- what to track ---
TAG = "our_intrinsics"
DEFAULT_FPS = 12.0  # DAIR-V2X-I 12 Hz; new datasets pass --fps (Camera data clips are 30)
TRACK_CLASS = "car"   # v0: vehicles only (the AV-vs-human question)
HIGH_SCORE = 0.45     # the detector's export threshold: anything at least this is a
                      # normal detection, below it is a ByteTrack-style leftover
DET_DIR = ROOT / "outputs/object_detection" / TAG
OUT_DIR = ROOT / "outputs/tracking" / TAG


def build_cfg():
    # minimal cfg the AB3DMOT core reads. nuScenes/centerpoint preset fits BEVHeight's
    # CenterPoint-style head (Car: greedy + giou_3d, min_hits=1, max_age=2).
    return SimpleNamespace(
        dataset="nuScenes",
        det_name="centerpoint",
        ego_com=False,   # fixed roadside camera: no ego motion
        vis=False,
        affi_pro=False,  # skip affinity post-processing
    )


class ByteAB3DMOT(AB3DMOT):
    """AB3DMOT with ByteTrack's second association pass.

    On this footage 38 percent of track states have no detection within 1.0 m at
    the 0.45 export threshold, and about 70 percent of those gaps do have one at
    0.20. Those weak detections are real but not trustworthy enough to start a
    track on, so they are offered only to the tracks the first pass left
    unmatched and never reach birth().

    Same order as AB3DMOT.track (predict, associate, update, birth, output) with
    the second association inserted before birth. Ego motion compensation,
    visualization and affinity post-processing are off in build_cfg(), so they
    are not repeated here.
    """

    def __init__(self, cfg, cat, low_thresh, **kwargs):
        super().__init__(cfg, cat, **kwargs)
        self.low_thresh = low_thresh

    def track(self, dets_all, frame, seq_name):
        dets, info = dets_all["dets"], dets_all["info"]
        self.frame_count += 1
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        score = info[:, 0] if len(info) else np.zeros(0)
        high = score >= HIGH_SCORE
        low = (score >= self.low_thresh) & ~high
        dets_high = self.process_dets(dets[high])
        trks = self.prediction()
        matched, unmatched_dets, unmatched_trks, _, _ = data_association(
            dets_high, trks, self.metric, self.thres, self.algm)
        self.update(matched, unmatched_trks, dets_high, info[high])

        if low.any() and len(unmatched_trks):
            dets_low = self.process_dets(dets[low])
            m, _, _, _, _ = data_association(
                dets_low, [trks[t] for t in unmatched_trks],
                self.metric, self.thres, self.algm)
            # association ran on a subset of the tracks, so map back to self.trackers
            m = np.array([(d, unmatched_trks[t]) for d, t in m], dtype=int).reshape(-1, 2)
            rest = [t for t in range(len(self.trackers)) if t not in set(m[:, 1])]
            self.update(m, rest, dets_low, info[low])

        self.birth(dets_high, info[high], unmatched_dets)   # low scores never start a track
        results = self.output()
        results = [np.concatenate(results)] if results else [np.empty((0, 15))]
        self.id_now_output = results[0][:, 7].tolist()
        return results, None


def load_frame_dets(frame_id, path=None):
    """Return (dets Nx7 [h,w,l,x,y,z,theta], info Nx1 [score]) for one frame, cars only."""
    if path is None:
        path = DET_DIR / f"{frame_id:06d}_pred.json"
    objs = json.loads(path.read_text())
    dets, info = [], []
    for o in objs:
        if o["class_name"] != TRACK_CLASS:
            continue
        dets.append([o["h"], o["w"], o["l"], o["x"], o["y"], o["z"], o["yaw"]])
        info.append([o["score"]])
    if not dets:
        return np.empty((0, 7)), np.empty((0, 1))
    return np.array(dets, dtype=float), np.array(info, dtype=float)


def track_frames(tracker, frame_ids, load, frame_rate_hz):
    """{track_id: [per-frame state]} from feeding every frame through the tracker.

    load(frame) -> (dets Nx7, info Nx1).
    """
    trajectories = {}
    for frame in frame_ids:
        dets, info = load(frame)
        results, _ = tracker.track({"dets": dets, "info": info}, frame, TAG)
        arr = results[0]  # rows: [h,w,l,x,y,z,theta, ID, score]

        # per-frame velocity lives in the live Kalman state, not in the output array
        vmap = {int(t.id): np.asarray(t.get_velocity()).reshape(-1) for t in tracker.trackers}

        for row in arr:
            tid = int(row[7])
            x, y, z = float(row[3]), float(row[4]), float(row[5])
            score = float(row[8]) if row.shape[0] > 8 else None
            vx, vy, vz = (vmap.get(tid, [math.nan] * 3))[:3]
            trajectories.setdefault(tid, []).append({
                "frame": frame,
                "x": x, "y": y, "z": z,
                "yaw": float(row[6]),
                # velocity is meters-per-frame in the Kalman state; *frame_rate_hz for m/s
                "vx": float(vx), "vy": float(vy), "vz": float(vz),
                "speed_mps": float(math.hypot(vx, vy) * frame_rate_hz),
                "score": score,
            })
    return trajectories


def main():
    ap = argparse.ArgumentParser(description="AB3DMOT on BEVHeight detections")
    ap.add_argument("--start", type=int, default=None, help="first frame index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame index (inclusive)")
    ap.add_argument("--label", default=None,
                    help="output subfolder under outputs/tracking/<tag>/ (keeps segments separate)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help=f"video frame rate in Hz for velocity scaling (default {DEFAULT_FPS})")
    ap.add_argument("--det-dir", default=None,
                    help="detections dir (default: outputs/object_detection/<TAG>)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: outputs/tracking/<TAG>)")
    ap.add_argument("--low-thresh", type=float, default=None,
                    help=f"turn on two-stage association: detections scoring "
                         f"[this, {HIGH_SCORE}) may continue an existing track but "
                         "never start one (default off)")
    ap.add_argument("--max-age", type=int, default=None,
                    help="frames a track survives without a match (preset 2 was "
                         "tuned for 10 Hz; try ~6 at 30 Hz)")
    args = ap.parse_args()
    frame_rate_hz = args.fps
    global DET_DIR, OUT_DIR
    if args.det_dir:
        DET_DIR = Path(args.det_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    frame_files = sorted(DET_DIR.glob("*_pred.json"))
    if not frame_files:
        sys.exit(f"No detections in {DET_DIR}")
    frame_ids = [int(f.stem.split("_")[0]) for f in frame_files]
    frame_paths = dict(zip(frame_ids, frame_files))
    if args.start is not None:
        frame_ids = [f for f in frame_ids if f >= args.start]
    if args.end is not None:
        frame_ids = [f for f in frame_ids if f <= args.end]
    if not frame_ids:
        sys.exit(f"No detection frames in range [{args.start}, {args.end}]")

    out_dir = OUT_DIR / args.label if args.label else OUT_DIR
    print(f"Tracking '{TRACK_CLASS}' over {len(frame_ids)} frames "
          f"({frame_ids[0]:06d}..{frame_ids[-1]:06d}) from {DET_DIR}")

    if args.low_thresh is None:
        tracker = AB3DMOT(build_cfg(), cat="Car", ID_init=0)
    else:
        tracker = ByteAB3DMOT(build_cfg(), cat="Car", low_thresh=args.low_thresh, ID_init=0)
    if args.max_age is not None:
        tracker.max_age = args.max_age

    trajectories = track_frames(
        tracker, frame_ids,
        lambda f: load_frame_dets(f, frame_paths.get(f)), frame_rate_hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracks.json"
    out_path.write_text(json.dumps({
        "meta": {"tag": TAG, "frame_rate_hz": frame_rate_hz, "class": TRACK_CLASS,
                 "low_thresh": args.low_thresh,
                 "frames": [frame_ids[0], frame_ids[-1]], "num_tracks": len(trajectories)},
        "tracks": trajectories,
    }, indent=2))

    # --- summary ---
    lengths = sorted((len(v) for v in trajectories.values()), reverse=True)
    multi = [L for L in lengths if L >= 2]
    print(f"\n{len(trajectories)} tracks total; {len(multi)} span >=2 frames")
    print(f"track lengths (top 10): {lengths[:10]}")
    if trajectories:
        longest_id = max(trajectories, key=lambda k: len(trajectories[k]))
        traj = trajectories[longest_id]
        print(f"\nlongest track: ID {longest_id}, {len(traj)} frames")
        for s in traj[:4]:
            print(f"  f{s['frame']:>3}  pos=({s['x']:6.1f},{s['y']:6.1f})  "
                  f"speed={s['speed_mps']:5.1f} m/s")
    print(f"\nsaved: {out_path}")

    # --- self-check: ID persistence and finite velocities ---
    assert lengths and lengths[0] >= 2, "no track spanned >=2 frames; association failed"
    for v in trajectories.values():
        if len(v) >= 2:
            assert all(math.isfinite(s["speed_mps"]) for s in v[1:]), "non-finite velocity"
    print("self-check OK: IDs persist across frames, velocities finite")


if __name__ == "__main__":
    main()
