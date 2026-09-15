"""Frozen heading score: per-frame box yaw against the track's direction of motion.

Why. BEVHeight predicts orientation per frame from appearance alone, and on this
roadside footage roughly 10% of car boxes sit more than 45 degrees off the
direction of travel. Every candidate fix in the orientation research loop is
graded here, so this file is written once, self-checked, and not edited while
candidates run. Any change to it re-scores every logged candidate.

There is no orientation ground truth on these clips, so the reference is the
track's own motion: the direction of its displacement over a centred window of
5 frames (0.17 s). That reference is only meaningful when the vehicle actually
moved, so frames whose window displacement is under 1.0 m are skipped, and
frames the tracker coasted (AB3DMOT repeats the previous score and velocity
verbatim when there is no detection) are excluded as well.

The scored angle is the RAW detection yaw from <frame>_pred.json, not the yaw in
tracks.json. The tracker's yaw is a Kalman state that AB3DMOT's
orientation_correction flips by +-pi whenever it disagrees with the detection by
more than 90 degrees, so grading it measures the tracker and the detector
together. The track is kept for identity and for the motion heading only: each
state takes the yaw of the detection nearest to it in (x, y) within 1.0 m, and
states with no such detection are excluded and counted as n_no_detection.

Two errors are reported per frame. Raw, wrap(yaw - motion) in (-pi, pi], catches
the front-to-back flip. Folded, the same angle mod 180 degrees in
[-pi/2, pi/2), catches axis error alone, so a box pointing backwards along the
correct axis scores zero folded and one flip. The primary number is the median
absolute folded error in degrees, averaged over clips; the secondary is the
fraction of raw errors over 90 degrees. Both are also reported per range bin
(0 to 40 m, 40 to 60 m, over 60 m) because the error concentrates at 40 to 60 m.

A candidate supplies replacement yaws as {clip: {track_id: {frame: yaw}}} and is
scored on exactly the frames the baseline was scored on.

GPS is never read here. Nothing under "Camera data/" is opened: the score has to
stay usable on the clips that have no GPS at all.

    .venv/bin/python scripts/evaluation/score_heading.py \
        --tracks outputs/tracking/camera-data/AV_T_EW_3_phase1/tracks.json \
        --out outputs/orientation/baseline.json
"""
import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

WINDOW = 5
MIN_MOVE_M = 1.0
MATCH_M = 1.0
BINS = (("0_40", 0.0, 40.0), ("40_60", 40.0, 60.0), ("60_inf", 60.0, np.inf))


def wrap(a):
    """Angle into (-pi, pi]."""
    return -((-np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi)


def fold(a):
    """Angle mod 180 degrees, into [-pi/2, pi/2)."""
    return (np.asarray(a, float) + np.pi / 2) % np.pi - np.pi / 2


def load_tracks(path):
    """tracks.json to {track_id: [state, ...]} with each track ordered by frame."""
    path = Path(path)
    if "Camera data" in path.parts:
        raise SystemExit("the heading score never reads Camera data/ (GPS is held out)")
    d = json.load(open(path))
    return {str(tid): sorted(states, key=lambda s: s["frame"])
            for tid, states in d["tracks"].items()}


@lru_cache(maxsize=None)
def load_dets(det_dir, frame):
    """One frame of raw detections as (N, 3): x, y, yaw. Missing frame is empty."""
    p = Path(det_dir) / f"{int(frame):03d}_pred.json"
    if not p.exists():
        return np.zeros((0, 3))
    d = json.load(open(p))
    return np.array([[s["x"], s["y"], s["yaw"]] for s in d], float).reshape(-1, 3)


def detection_yaw(track, det_dir, max_dist_m=MATCH_M):
    """Raw detection yaw at each state, NaN where nothing is within max_dist_m.

    The track carries the Kalman yaw, which AB3DMOT may have flipped by pi; the
    detection at the same place carries what BEVHeight actually predicted.
    """
    out = np.full(len(track), np.nan)
    for i, s in enumerate(track):
        dets = load_dets(str(det_dir), s["frame"])
        if not len(dets):
            continue
        d = np.hypot(dets[:, 0] - s["x"], dets[:, 1] - s["y"])
        j = int(np.argmin(d))
        if d[j] <= max_dist_m:
            out[i] = dets[j, 2]
    return out


def default_det_dir(tracks_path):
    """outputs/tracking/<...>/tracks.json to outputs/object_detection/<...>/."""
    parts = list(Path(tracks_path).resolve().parent.parts)
    if "tracking" not in parts:
        raise SystemExit(f"cannot guess a detection directory for {tracks_path}")
    parts[parts.index("tracking")] = "object_detection"
    return Path(*parts)


def coasted(track):
    """Frames with no detection: AB3DMOT repeats the previous score and vx verbatim."""
    sc = np.array([s["score"] for s in track], float)
    vx = np.array([s["vx"] for s in track], float)
    return np.r_[False, (sc[1:] == sc[:-1]) & (vx[1:] == vx[:-1])]


def motion_heading(track, window=WINDOW):
    """Direction of travel per frame from the track alone, NaN where unusable.

    Centred displacement over `window` frames, clamped to the available span at
    the track ends and refused under 3 states or under MIN_MOVE_M of motion.
    Camera ground frame: x forward, y left, so atan2(dy, dx) shares the yaw axis.
    """
    x = np.array([s["x"] for s in track], float)
    y = np.array([s["y"] for s in track], float)
    half = window // 2
    out = np.full(len(track), np.nan)
    for i in range(len(track)):
        lo, hi = max(0, i - half), min(len(track) - 1, i + half)
        if hi - lo + 1 < 3:
            continue
        dx, dy = x[hi] - x[lo], y[hi] - y[lo]
        if math.hypot(dx, dy) < MIN_MOVE_M:
            continue
        out[i] = math.atan2(dy, dx)
    return out


def stats(raw):
    """Error summary over a set of frames. Empty stays None rather than NaN."""
    raw = np.asarray(raw, float)
    if raw.size == 0:
        return {"n": 0, "median_folded_deg": None, "mean_folded_deg": None,
                "frac_raw_gt45": None, "frac_raw_gt90": None}
    folded = np.degrees(np.abs(fold(raw)))
    absolute = np.degrees(np.abs(raw))
    return {"n": int(raw.size),
            "median_folded_deg": float(np.median(folded)),
            "mean_folded_deg": float(folded.mean()),
            "frac_raw_gt45": float((absolute > 45.0).mean()),
            "frac_raw_gt90": float((absolute > 90.0).mean())}


def score_clip(tracks, yaw_override=None, det_dir=None):
    """Grade one clip.

    det_dir supplies the raw detection yaws; without it the track's own yaw is
    scored, which is only meaningful for synthetic tracks. yaw_override replaces
    the yaw of matching states and is scored on the same frames as the baseline.
    """
    override = {str(tid): {int(f): float(v) for f, v in per_frame.items()}
                for tid, per_frame in (yaw_override or {}).items()}
    raw, rng, unmatched = [], [], 0
    for tid, track in tracks.items():
        yaw = np.array([s["yaw"] for s in track], float)
        motion = motion_heading(track)
        keep = ~coasted(track) & ~np.isnan(motion)
        if det_dir is not None:
            ydet = detection_yaw(track, det_dir)
            matched = ~np.isnan(ydet)
            unmatched += int((keep & ~matched).sum())
            yaw = np.where(matched, ydet, yaw)
            keep &= matched
        per_frame = override.get(str(tid), {})
        if per_frame:
            yaw = np.array([per_frame.get(s["frame"], y) for s, y in zip(track, yaw)], float)
        raw.append(wrap(yaw[keep] - motion[keep]))
        rng.append(np.array([s["x"] for s in track], float)[keep])
    raw = np.concatenate(raw) if raw else np.zeros(0)
    rng = np.concatenate(rng) if rng else np.zeros(0)
    out = stats(raw)
    out["n_no_detection"] = unmatched
    out["per_bin"] = {}
    for name, lo, hi in BINS:
        s = stats(raw[(rng >= lo) & (rng < hi)])
        out["per_bin"][name] = {k: s[k] for k in ("n", "median_folded_deg", "frac_raw_gt45")}
    return out


def score_clips(clips):
    """clips is a list of (name, tracks, yaw_override[, det_dir]). Clips with no
    scored frame are reported but left out of the means."""
    per_clip = {c[0]: score_clip(c[1], c[2], c[3] if len(c) > 3 else None) for c in clips}
    scored = [r for r in per_clip.values() if r["n"]]
    mean = lambda k: float(np.mean([r[k] for r in scored])) if scored else None
    return {"per_clip": per_clip,
            "mean_median_folded_deg": mean("median_folded_deg"),
            "mean_frac_raw_gt90": mean("frac_raw_gt90")}


def keep_decision(baseline_result, candidate_result, per_clip_margin_deg=1.0):
    """Generic first: the mean must improve, no clip may pay more than the margin
    for it, and the flip fraction may not rise. Returns (kept, reason)."""
    b, c = baseline_result, candidate_result
    if c["mean_median_folded_deg"] is None or b["mean_median_folded_deg"] is None:
        return False, "no scored frames"
    if c["mean_median_folded_deg"] >= b["mean_median_folded_deg"]:
        return False, (f"mean median folded did not improve: "
                       f"{b['mean_median_folded_deg']:.3f} to {c['mean_median_folded_deg']:.3f} deg")
    for name, bc in b["per_clip"].items():
        cc = c["per_clip"].get(name)
        if cc is None:
            return False, f"clip {name} missing from the candidate"
        delta = cc["median_folded_deg"] - bc["median_folded_deg"]
        if delta > per_clip_margin_deg:
            return False, f"clip {name} worsened by {delta:.3f} deg (margin {per_clip_margin_deg})"
    if c["mean_frac_raw_gt90"] > b["mean_frac_raw_gt90"]:
        return False, (f"flip fraction rose: {b['mean_frac_raw_gt90']:.4f} to "
                       f"{c['mean_frac_raw_gt90']:.4f}")
    return True, (f"mean median folded {b['mean_median_folded_deg']:.3f} to "
                  f"{c['mean_median_folded_deg']:.3f} deg, no clip over the margin")


def main():
    ap = argparse.ArgumentParser("Heading score against track motion")
    ap.add_argument("--tracks", nargs="+", required=True, help="one tracks.json per clip")
    ap.add_argument("--det-dir", nargs="+", help="detection directory per tracks file "
                    "(default: the same path with tracking replaced by object_detection)")
    ap.add_argument("--yaw-json", help="candidate yaws, {clip: {track_id: {frame: yaw}}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.det_dir and len(args.det_dir) != len(args.tracks):
        raise SystemExit("--det-dir needs one directory per --tracks file")
    override = json.load(open(args.yaw_json)) if args.yaw_json else {}
    clips = []
    for i, p in enumerate(args.tracks):
        name = Path(p).resolve().parent.name
        det = Path(args.det_dir[i]) if args.det_dir else default_det_dir(p)
        clips.append((name, load_tracks(p), override.get(name), det))
    report = score_clips(clips)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
