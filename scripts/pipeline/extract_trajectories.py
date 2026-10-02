"""One command from a clip's frames to one trajectory file per vehicle.

Stages, each written to its own folder so any one can be inspected or rerun:
  1. detect     BEVHeight, 102.4 m checkpoint fine-tuned on corrected labels,
                15.1 m calibration (skipped when the detections already exist,
                e.g. copied back from the lab GPU)
  2. road       drop boxes off the road (road_mask.py, mask built from traffic)
  3. track      AB3DMOT, birth and continuation at score 0.45, max age 6, 30 fps
  4. stitch     join fragments of one vehicle (stitch_tracks.py, I-24 MOTION)
  5. smooth     trim coasted tails, RTS-smooth position and velocity
                (postprocess_tracks.py), then drop duplicate tracks: a track
                that sits within DUP_DX_M along and DUP_DY_M across a longer
                one for at least half its frames is the same vehicle twice
  6. lanes      lane centres from where moving traffic drives, lane per state,
                along-road and lateral position, leader in the same lane and
                the spacing and time headway to it
Output: outputs/trajectories/<clip>_<tag>/vehicle_<id>.csv plus lanes.json and
run.json with every setting, so both vehicle groups can be shown to have gone
through identical processing.

Road frame: x runs along the road (the calibration aligns it with the lane
vanishing point), y across it, so "along" is x and the lateral offset is y
minus the lane centre. Spacing is centre to centre (space headway); a leader
closer than MIN_SPACING_M is impossible for two cars and is ignored.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/pipeline/extract_trajectories.py --clip AV_T_EW_3
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("scripts/tracking", "scripts/object_detection", "scripts/evaluation")]
import postprocess_tracks as pp   # noqa: E402
import stitch_tracks as st        # noqa: E402
from score_heading import coasted  # noqa: E402

PY = sys.executable
DEFAULTS = {
    "tag": "ft102",
    "ckpt": "outputs/finetune/run3_v2labels/head_ft_ep5.ckpt",
    "config": "experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py",
    "extrinsic": "metric_extrinsic_h151_dpm031.json",
    "score_thresh": 0.45, "track_low_thresh": 0.45, "max_age": 6, "fps": 30,
}
LANE_BIN_M, MIN_LANE_SEP_M, MOVING_MPS = 0.25, 2.5, 3.0
MIN_LANE_STATES, MAX_LANE_OFFSET_M = 30, 2.0   # a quiet lane still counts; off every lane = no lane
DUP_DX_M, DUP_DY_M, MIN_SPACING_M = 3.0, 1.2, 4.0


def run(cmd):
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def drop_duplicates(tracks):
    """Remove tracks that mostly ride on top of a longer track."""
    order = sorted(tracks, key=lambda k: -len(tracks[k]))
    kept = {}
    for k in order:
        t = {s["frame"]: s for s in tracks[k]}
        dup = False
        for o in kept.values():
            same = [abs(t[f]["x"] - o[f]["x"]) < DUP_DX_M and abs(t[f]["y"] - o[f]["y"]) < DUP_DY_M
                    for f in t.keys() & o.keys()]
            if sum(same) >= 0.5 * len(t):
                dup = True
                break
        if not dup:
            kept[k] = t
    return {k: tracks[k] for k in kept}


def lane_centres(tracks):
    """Lane centres per direction (+1 along x, -1 against): peaks of the
    histogram of lateral position over moving states."""
    out = {}
    for d in (1, -1):
        ys = np.array([s["y"] for t in tracks.values() for s in t
                       if s["speed_mps"] > MOVING_MPS and np.sign(s["vx"]) == d])
        if len(ys) < 50:
            continue
        edges = np.arange(ys.min() - 1, ys.max() + 1 + LANE_BIN_M, LANE_BIN_M)
        h, _ = np.histogram(ys, edges)
        h = np.convolve(h, np.ones(5) / 5, mode="same")
        mids = (edges[:-1] + edges[1:]) / 2
        peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1]
                 and np.sum(np.abs(ys - mids[i]) < MIN_LANE_SEP_M / 2) >= MIN_LANE_STATES]
        kept = []
        for i in sorted(peaks, key=lambda i: -h[i]):
            if all(abs(mids[i] - mids[j]) >= MIN_LANE_SEP_M for j in kept):
                kept.append(i)
        out[d] = sorted(float(mids[i]) for i in kept)
    return out


def lane_of(y, d, centres):
    c = centres.get(d)
    if not c:
        return None, None
    k = int(np.argmin([abs(y - v) for v in c]))
    if abs(y - c[k]) > MAX_LANE_OFFSET_M:
        return None, None
    return k + 1, y - c[k]


def export(tracks, centres, out_dir, fps):
    """Per-vehicle CSV with lane, along-road position and the leader."""
    by_frame = {}
    for tid, t in tracks.items():
        for s in t:
            by_frame.setdefault(s["frame"], []).append((tid, s))
    rows_per = {}
    for tid, t in tracks.items():
        det = ~coasted(t)
        rows = []
        for s, has_det in zip(t, det):
            d = int(np.sign(s["vx"])) or 1
            lane, lat = lane_of(s["y"], d, centres)
            lead, spacing = None, None
            for oid, o in by_frame.get(s["frame"], []):
                if lane is None or oid == tid or (int(np.sign(o["vx"])) or 1) != d \
                        or lane_of(o["y"], d, centres)[0] != lane:
                    continue
                ahead = (o["x"] - s["x"]) * d
                if ahead > MIN_SPACING_M and (spacing is None or ahead < spacing):
                    lead, spacing = oid, ahead
            moving = s["speed_mps"] > MOVING_MPS
            rows.append({
                "frame": s["frame"], "t_s": round(s["frame"] / fps, 4),
                "x_m": round(s["x"], 3), "y_m": round(s["y"], 3),
                "along_m": round(s["x"], 3), "lane": lane, "direction": d,
                "lateral_m": None if lat is None else round(lat, 3),
                "speed_mps": round(s["speed_mps"], 3),
                "heading_deg": round(math.degrees(math.atan2(s["vy"], s["vx"])), 2) if moving else None,
                "yaw_det_deg": round(math.degrees(s["yaw"]), 2),
                "detected": bool(has_det),
                "lead_id": lead, "spacing_m": None if spacing is None else round(spacing, 2),
                "time_headway_s": round(spacing / s["speed_mps"], 3) if spacing and moving else None})
        rows_per[tid] = rows
        with open(out_dir / f"vehicle_{tid}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows_per


def main():
    ap = argparse.ArgumentParser("Clip frames to per-vehicle trajectories")
    ap.add_argument("--clip", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--detect-only", action="store_true")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    run_name = f"{a.clip}_{a.tag}"
    det = ROOT / "outputs/object_detection/camera-data" / run_name
    cal = ROOT / "outputs/calibration/camera-data" / a.clip

    if not (det / "calibration_used.json").exists():
        print("1. detect")
        run([PY, "scripts/object_detection/run_bevheight_generic.py",
             "--frames-dir", f"data/camera-data/{a.clip}/frames_all",
             "--anycalib-json", next(cal.glob("*_anycalib_pinhole_pinhole.json")),
             "--extrinsic-json", cal / a.extrinsic, "--out-dir", det,
             "--ckpt", a.ckpt, "--config", a.config, "--score-thresh", a.score_thresh])
    else:
        print("1. detect: found existing detections")
    if a.detect_only:
        return
    print("2. road mask")
    run([PY, "scripts/object_detection/road_mask.py", "--clip", a.clip, "--run", a.tag])
    print("3. track")
    trk = ROOT / "outputs/tracking/camera-data" / f"{run_name}_road"
    run([PY, "scripts/tracking/run_ab3dmot.py", "--det-dir", det.with_name(det.name + "_road"),
         "--out-dir", trk, "--low-thresh", a.track_low_thresh, "--max-age", a.max_age, "--fps", a.fps])
    d = json.loads((trk / "tracks.json").read_text())
    print("4. stitch")
    tracks, joined = st.stitch(d["tracks"], a.fps)
    print("5. smooth")
    tracks = {k: pp.clean(v, a.fps) for k, v in tracks.items()}
    tracks = {k: v for k, v in tracks.items() if len(v) >= 3}
    n_before = len(tracks)
    tracks = drop_duplicates(tracks)
    n_dup = n_before - len(tracks)
    print("6. lanes and leaders")
    centres = lane_centres(tracks)
    out = ROOT / "outputs/trajectories" / run_name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("vehicle_*.csv"):
        p.unlink()
    rows = export(tracks, centres, out, a.fps)
    (out / "lanes.json").write_text(json.dumps({"centres_by_direction": {str(k): v for k, v in centres.items()},
                                                "note": "direction +1 travels along +x, -1 against"}, indent=2))
    (out / "tracks.json").write_text(json.dumps({"meta": d["meta"], "tracks": tracks}))
    frag = st.fragmentation(tracks, a.fps)
    (out / "run.json").write_text(json.dumps({
        "clip": a.clip, "settings": cfg, "stitch": {"joins": len(joined), "time_win_s": st.TIME_WIN_S,
        "side_limit": f"{st.SIDE_NOISE_M} m + {st.SIDE_SPEED_MPS} m/s x gap", "thresh": st.STITCH_THRESH},
        "smoothing": "RTS, constant-velocity model (rts_smooth_track.build_kf), coasted tails trimmed",
        "lanes": {"bin_m": LANE_BIN_M, "min_sep_m": MIN_LANE_SEP_M, "moving_mps": MOVING_MPS,
                  "min_states": MIN_LANE_STATES, "max_offset_m": MAX_LANE_OFFSET_M},
        "duplicates_dropped": n_dup, "dup_limits_m": [DUP_DX_M, DUP_DY_M], "min_spacing_m": MIN_SPACING_M,
        "vehicles": len(rows), "fragmentation": frag}, indent=2))
    print(f"{run_name}: {len(rows)} vehicles, {len(joined)} joins, {n_dup} duplicates dropped, lanes "
          f"{ {k: [round(v, 1) for v in c] for k, c in centres.items()} } -> {out}")


def _selfcheck():
    """Two lanes each way are found; a follower gets its leader and spacing."""
    rng = np.random.default_rng(0)
    tracks = {}
    for i, (y, v) in enumerate([(-6, -28), (-9.6, -28), (-26, 28), (-29.6, 28)] * 3):
        x0 = 30 + 15 * (i // 4) if v > 0 else 110 - 15 * (i // 4)
        tracks[str(i)] = [{"frame": f, "x": x0 + v * f / 30, "y": y + rng.normal(0, 0.2), "vx": v, "vy": 0.0,
                           "speed_mps": abs(v), "yaw": 0.0, "score": 0.5 + f * 1e-4} for f in range(60)]
    c = lane_centres(tracks)
    assert len(c[1]) == 2 and len(c[-1]) == 2, c
    assert all(min(abs(t - v) for v in c[1]) < 0.3 for t in (-26, -29.6)), c
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rows = export(tracks, c, Path(tmp), 30)
    follower = rows["2"][10]          # lane y=-26 heading +x, starts behind vehicle 6
    assert follower["lead_id"] == "6" and abs(follower["spacing_m"] - 15) < 0.5, follower
    twin = {**tracks, "dup": [dict(s, x=s["x"] + 1.0) for s in tracks["2"][:30]]}
    assert "dup" not in drop_duplicates(twin) and len(drop_duplicates(twin)) == len(tracks)
    print(f"selfcheck ok (lanes {c}, follower spacing {follower['spacing_m']} m)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
