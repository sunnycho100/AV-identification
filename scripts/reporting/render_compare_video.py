"""Four-panel mp4 to compare detection runs and the orientation post-processing by eye.

Panels: 102.4 m checkpoint raw, 140.8 m raw, and each with the kept post-processing
(score-weighted axis consensus plus the travel-direction sign bit) drawn from a
run's yaws.json. Boxes are the raw detections; on the post-processed panels the
yaw of the detection nearest each track state (within 1 m) is replaced by the run's
yaw. An arrow from the box centre shows the heading so flips are visible.

    .venv/bin/python scripts/reporting/render_compare_video.py --clip AV_T_EW_3
"""
import argparse, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluators.result2kitti import get_lidar_3d_8points
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image

PANELS = [("102.4 m raw", "phase1", None),
          ("140.8 m raw", "r140", None),
          ("102.4 m + consensus + sign", "phase1", "sign_from_motion"),
          ("140.8 m + consensus + sign", "r140", "sign_from_motion_dbe42883")]


def load_run(clip, run):
    det = ROOT / f"outputs/object_detection/camera-data/{clip}_{run}"
    cal = json.load(open(det / "calibration_used.json"))
    k34 = np.zeros((3, 4)); k34[:3, :3] = np.array(cal["K"])
    tracks = json.load(open(ROOT / f"outputs/tracking/camera-data/{clip}_{run}/tracks.json"))["tracks"]
    return det, k34, np.array(cal["lidar2cam"]), tracks


def yaw_overrides(clip, run_dir, tracks, det, frame):
    """{det index: yaw} for this frame from a run's yaws.json via nearest detection."""
    yaws = json.load(open(ROOT / "outputs/orientation/runs" / run_dir / "yaws.json")).get(clip, {})
    dets = json.load(open(det / f"{frame:03d}_pred.json"))
    xy = np.array([[d["x"], d["y"]] for d in dets]) if dets else np.zeros((0, 2))
    out = {}
    for tid, per_frame in yaws.items():
        if str(frame) not in per_frame:
            continue
        st = next((s for s in tracks[tid] if s["frame"] == frame), None)
        if st is None or not len(xy):
            continue
        i = int(np.argmin(np.hypot(xy[:, 0] - st["x"], xy[:, 1] - st["y"])))
        if np.hypot(xy[i, 0] - st["x"], xy[i, 1] - st["y"]) <= 1.0:
            out[i] = per_frame[str(frame)]
    return out


def draw_panel(img, dets, k34, l2c, overrides, label):
    for i, d in enumerate(dets):
        yaw = overrides.get(i, d["yaw"])
        car = d.get("class_name", "car") == "car"
        # cars green (orange once post-processed), every other class grey
        c = (160, 160, 160) if not car else (0, 200, 255) if i in overrides else (0, 255, 0)
        l, w, h = d["l"], d["w"], d["h"]
        centre = [d["x"], d["y"], d["z"] + h / 2]
        corners = get_lidar_3d_8points([l, w, h], yaw, centre)
        nose = [d["x"] + 0.8 * l * np.cos(yaw), d["y"] + 0.8 * l * np.sin(yaw), d["z"] + h / 2]
        pts3 = np.vstack([corners, centre, nose])
        cc = (l2c @ np.concatenate([pts3, np.ones((10, 1))], 1).T).T[:, :3]
        if np.sum(cc[:8, 2] > 1e-6) < 4 or cc[8, 2] <= 0 or cc[9, 2] <= 0:
            continue
        pts = project_to_image(cc, k34)
        draw_box_3d(img, pts[:8], c=c)
        cv2.arrowedLine(img, (int(pts[8, 0]), int(pts[8, 1])), (int(pts[9, 0]), int(pts[9, 1])), (0, 0, 255), 2, tipLength=0.4)
    n = sum(1 for d in dets if d.get("class_name", "car") == "car")
    cv2.putText(img, f"{label}   cars {n}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return cv2.resize(img, (960, 540))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    runs = {run: load_run(a.clip, run) for run in {p[1] for p in PANELS}}
    out_dir = ROOT / "outputs/evaluation/assets" / f"compare_{a.clip}"; out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(int(p.stem[:-5]) for p in runs["phase1"][0].glob("*_pred.json"))
    for f in frames:
        src = cv2.imread(str(ROOT / f"data/camera-data/{a.clip}/frames_all/{f:03d}.jpg"))
        tiles = []
        for label, run, yaw_run in PANELS:
            det, k34, l2c, tracks = runs[run]
            dets = json.load(open(det / f"{f:03d}_pred.json"))
            ov = yaw_overrides(a.clip, yaw_run, tracks, det, f) if yaw_run else {}
            tiles.append(draw_panel(src.copy(), dets, k34, l2c, ov, label))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(out_dir / f"{f:03d}.jpg"), grid)
    mp4 = ROOT / "outputs/evaluation/assets" / f"compare_{a.clip}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(out_dir / "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    print(mp4)


if __name__ == "__main__":
    main()
