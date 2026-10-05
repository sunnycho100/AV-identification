"""Draw the final extracted trajectories (extract_trajectories.py output) on the
clip's frames and write an mp4: one box per vehicle with its id and speed in mph, the
instrumented vehicle (instrumented.json, or --mark) in orange.

Track states carry position, heading and speed but no box size, so every box is
drawn at a typical car size (ponytail: fixed 4.6 x 1.9 x 1.5 m; trucks look short).

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_trajectories.py --clip AV_T_WE_3
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluators.result2kitti import get_lidar_3d_8points
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image

L, W, H = 4.6, 1.9, 1.5
ORANGE, GREEN = (0, 140, 255), (0, 220, 0)
MPH = 2.23694


def draw_states(img, states, k34, l2c, mark):
    """Boxes with id and mph for one frame's (track id, state) pairs; mark in orange."""
    for tid, s in states:
        yaw = math.atan2(s["vy"], s["vx"]) if s["speed_mps"] > 1 else s["yaw"]
        c = get_lidar_3d_8points([L, W, H], yaw, [s["x"], s["y"], s["z"] + H / 2])
        cc = (l2c @ np.c_[c, np.ones(8)].T).T[:, :3]
        if np.sum(cc[:, 2] > 1e-6) < 4:
            continue
        pts = project_to_image(cc, k34)
        col = ORANGE if tid == mark else GREEN
        draw_box_3d(img, pts, c=col)
        u, v = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
        label = f"{'GPS? ' if tid == mark else ''}#{tid}  {s['speed_mps'] * MPH:.0f} mph"
        cv2.putText(img, label, (u, v), 0, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (u, v), 0, 0.6, col, 2)


def main():
    ap = argparse.ArgumentParser("Render extracted trajectories")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tag", default="ft102")
    ap.add_argument("--mark", default=None, help="vehicle id to highlight (default: instrumented.json)")
    a = ap.parse_args()
    run = f"{a.clip}_{a.tag}"
    traj = ROOT / "outputs/trajectories" / run
    cal = json.load(open(ROOT / "outputs/object_detection/camera-data" / run / "calibration_used.json"))
    k34 = np.zeros((3, 4)); k34[:3, :3] = np.array(cal["K"]); l2c = np.array(cal["lidar2cam"])
    tracks = json.load(open(traj / "tracks.json"))["tracks"]
    inst = traj / "instrumented.json"
    mark = a.mark or (json.load(open(inst))["vehicle_id"] if inst.exists() else None)
    by_frame = {}
    for tid, states in tracks.items():
        for s in states:
            by_frame.setdefault(s["frame"], []).append((tid, s))
    out = ROOT / "outputs/reports/trajectories" / run
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((ROOT / f"data/camera-data/{a.clip}/frames_all").glob("*.jpg"))
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        draw_states(img, by_frame.get(i, []), k34, l2c, mark)
        cv2.putText(img, f"{run} frame {i}  marked car: {mark or 'none'}", (20, 40), 0, 1.0, (255, 255, 255), 3)
        cv2.imwrite(str(out / f"{i:03d}.jpg"), img)
    mp4 = out / f"{run}_trajectories.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(out / "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1280:-2", str(mp4)], check=True)
    print(f"{len(frames)} frames, {len(tracks)} vehicles, marked {mark} -> {mp4}")


if __name__ == "__main__":
    main()
