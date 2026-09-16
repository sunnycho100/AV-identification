"""Two-panel mp4 that shows heading accuracy by colour, not by eye.

--source det draws the raw per-frame detections, so a frame where the detector
fired nothing shows no box: that is the gap you see on a car that is plainly
there. --source track draws the tracker's own state for every frame of every
track instead, with the box size fixed to the track's median and the heading
from yaw_refined, so a coasted frame still gets a box (drawn thin). The pipeline
consumes the track, not the per-frame detections, so --source track is what the
classifier actually sees.

Top: the 102.4 m checkpoint, raw detections (the pipeline before today).
Bottom: the 140.8 m checkpoint with the kept post-processing (score-weighted axis
consensus plus one travel-direction sign bit per track).

Every box is coloured by how far its heading is from the direction the tracked car
actually moved (the same reference the frozen score uses):
  green   within 15 degrees          yellow  15 to 45 degrees
  red     more than 45 degrees off   magenta pointing backwards (more than 90)
  grey    no motion reference (stationary, coasted, or untracked)
The arrow shows the heading; a running tally per panel counts frames so far.

    NUMBA_DISABLE_JIT=1 /Users/sunghwan_cho/miniforge/bin/python3.12 \
        scripts/reporting/render_compare_video.py --clip AV_T_EW_3
"""
import argparse, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from evaluators.result2kitti import get_lidar_3d_8points
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
import score_heading as sh

# ponytail: run names follow the disk while phase1 is being regenerated; the old
# 102.4 run is phase1_102 once moved, phase1 before that, and r140 is the 140.8 run.
OLD = "phase1_102" if (ROOT / "outputs/tracking/camera-data/AV_T_EW_3_phase1_102").exists() else "phase1"
PANELS = [("102.4 m checkpoint, raw boxes", OLD, None),
          ("140.8 m checkpoint + consensus + sign bit", "phase1", "sign_from_motion_dbe42883")]
COL = {"good": (0, 200, 0), "ok": (0, 220, 255), "bad": (0, 0, 255), "flip": (255, 0, 255), "none": (150, 150, 150)}


def grade(err):
    if err is None:
        return "none"
    if abs(err) > np.pi / 2:
        return "flip"
    f = abs(sh.fold(err))
    return "good" if f < np.radians(15) else "ok" if f < np.radians(45) else "bad"


def track_boxes(clip, run):
    """{frame: [(box dict, motion heading or None, coasted)]} from the tracks.

    Box size is the track's median over the frames where a detection matched, so
    a coasted frame inherits a plausible size instead of none. Heading is
    yaw_refined when refine_yaw.py has run, else the tracker's yaw.
    """
    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_{run}"
    tracks = sh.load_tracks(ROOT / f"outputs/tracking/camera-data/{clip}_{run}/tracks.json")
    dets = {int(p.stem[:-5]): json.load(open(p)) for p in det_dir.glob("*_pred.json")}
    out = {f: [] for f in dets}
    for track in tracks.values():
        motion, coasted = sh.motion_heading(track), sh.coasted(track)
        dims = []
        for s in track:
            d = dets.get(s["frame"], [])
            if not d:
                continue
            xy = np.array([[o["x"], o["y"]] for o in d])
            k = int(np.argmin(np.hypot(xy[:, 0] - s["x"], xy[:, 1] - s["y"])))
            if np.hypot(xy[k, 0] - s["x"], xy[k, 1] - s["y"]) <= sh.MATCH_M:
                dims.append([d[k]["l"], d[k]["w"], d[k]["h"]])
        if not dims:
            continue
        l, w, h = np.median(np.array(dims), axis=0)
        for i, s in enumerate(track):
            yaw = s.get("yaw_refined")
            if yaw is None:
                yaw = s["yaw"]
            box = {"x": s["x"], "y": s["y"], "z": s["z"], "l": l, "w": w, "h": h, "yaw": yaw}
            m = None if np.isnan(motion[i]) else float(motion[i])
            out.setdefault(s["frame"], []).append((box, m, bool(coasted[i])))
    return det_dir, out


def per_frame_headings(clip, run, yaw_run):
    """{frame: {det index: (yaw used, motion heading or None)}} for one panel."""
    det_dir = ROOT / f"outputs/object_detection/camera-data/{clip}_{run}"
    tracks = sh.load_tracks(ROOT / f"outputs/tracking/camera-data/{clip}_{run}/tracks.json")
    over = json.load(open(ROOT / "outputs/orientation/runs" / yaw_run / "yaws.json")).get(clip, {}) if yaw_run else {}
    dets = {int(p.stem[:-5]): json.load(open(p)) for p in det_dir.glob("*_pred.json")}
    out = {f: {} for f in dets}
    for tid, track in tracks.items():
        motion = sh.motion_heading(track)
        coasted = sh.coasted(track)
        for i, s in enumerate(track):
            f = s["frame"]
            d = dets.get(f, [])
            if not d:
                continue
            xy = np.array([[o["x"], o["y"]] for o in d])
            k = int(np.argmin(np.hypot(xy[:, 0] - s["x"], xy[:, 1] - s["y"])))
            if np.hypot(xy[k, 0] - s["x"], xy[k, 1] - s["y"]) > sh.MATCH_M:
                continue
            yaw = over.get(tid, {}).get(str(f), d[k]["yaw"]) if yaw_run else d[k]["yaw"]
            m = None if coasted[i] or np.isnan(motion[i]) else float(motion[i])
            out[f][k] = (yaw, m)
    return det_dir, dets, out


def draw_panel(img, items, k34, l2c, label, tally):
    """items: [(box dict, motion heading or None, thin)] already resolved."""
    for d, m, thin in items:
        yaw = d["yaw"]
        g = grade(None if m is None else sh.wrap(yaw - m))
        if g != "none":
            tally[g] = tally.get(g, 0) + 1
        l, w, h = d["l"], d["w"], d["h"]
        centre = [d["x"], d["y"], d["z"] + h / 2]
        nose = [d["x"] + 0.9 * l * np.cos(yaw), d["y"] + 0.9 * l * np.sin(yaw), centre[2]]
        pts3 = np.vstack([get_lidar_3d_8points([l, w, h], yaw, centre), centre, nose])
        cc = (l2c @ np.concatenate([pts3, np.ones((10, 1))], 1).T).T[:, :3]
        if np.sum(cc[:8, 2] > 1e-6) < 4 or cc[8, 2] <= 0 or cc[9, 2] <= 0:
            continue
        pts = project_to_image(cc, k34)
        col = tuple(int(0.55 * c) for c in COL[g]) if thin else COL[g]
        draw_box_3d(img, pts[:8], c=col)
        cv2.arrowedLine(img, (int(pts[8, 0]), int(pts[8, 1])), (int(pts[9, 0]), int(pts[9, 1])),
                        col, 1 if thin else 3, tipLength=0.4)
    n = sum(tally.get(k, 0) for k in ("good", "ok", "bad", "flip"))
    pct = lambda k: f"{100 * tally.get(k, 0) / n:.0f}%" if n else "-"
    cv2.rectangle(img, (0, 0), (1920, 110), (0, 0, 0), -1)
    cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(img, f"so far: within 15 deg {pct('good')}   15 to 45 {pct('ok')}   over 45 {pct('bad')}   backwards {pct('flip')}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--source", choices=("det", "track"), default="det")
    a = ap.parse_args()
    panels = []
    for label, run, yaw_run in PANELS:
        if a.source == "track":
            det_dir, per_frame = track_boxes(a.clip, run)
            label += ", tracks"
        else:
            det_dir, dets, heads = per_frame_headings(a.clip, run, yaw_run)
            per_frame = {f: [(d, heads.get(f, {}).get(i, (d["yaw"], None))[1], False)
                             for i, d in enumerate(v) if d.get("class_name", "car") == "car"
                             for d in [dict(d, yaw=heads.get(f, {}).get(i, (d["yaw"], None))[0])]]
                         for f, v in dets.items()}
        cal = json.load(open(det_dir / "calibration_used.json"))
        k34 = np.zeros((3, 4)); k34[:3, :3] = np.array(cal["K"])
        panels.append((label, per_frame, k34, np.array(cal["lidar2cam"]), {}))
    out_dir = ROOT / "outputs/evaluation/assets" / f"compare_{a.clip}{'_tracks' if a.source == 'track' else ''}"; out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(panels[0][1]):
        src = cv2.imread(str(ROOT / f"data/camera-data/{a.clip}/frames_all/{f:03d}.jpg"))
        tiles = [draw_panel(src.copy(), per_frame.get(f, []), k34, l2c, label, tally)
                 for label, per_frame, k34, l2c, tally in panels]
        cv2.imwrite(str(out_dir / f"{f:03d}.jpg"), np.vstack(tiles))
    mp4 = ROOT / "outputs/evaluation/assets" / f"compare_{a.clip}{'_tracks' if a.source == 'track' else ''}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(out_dir / "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    print(mp4)


if __name__ == "__main__":
    main()
