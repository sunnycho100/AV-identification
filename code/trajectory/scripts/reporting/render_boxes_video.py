"""Re-draw a detection run's boxes on its frames and write an mp4.

--swap-dims draws with the earlier convention (length and width exchanged in
the corner builder), which puts the long axis of every box across the car
instead of along it. That drawing is geometrically wrong but was the one used
in the August comparison with Bofeng's run, so it is kept available on purpose.

    .venv/bin/python scripts/reporting/render_boxes_video.py --clip HV_T_EW_1 --run r140 --swap-dims
"""
import argparse, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluators.result2kitti import get_lidar_3d_8points
from scripts.data_converter.visual_utils import draw_box_3d, project_to_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--run", default="r140")
    ap.add_argument("--swap-dims", action="store_true"); ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    det = ROOT / f"outputs/object_detection/camera-data/{a.clip}_{a.run}"
    cal = json.load(open(det / "calibration_used.json")); K, l2c = np.array(cal["K"]), np.array(cal["lidar2cam"])
    k34 = np.zeros((3, 4)); k34[:3, :3] = K
    tag = "perpendicular" if a.swap_dims else "aligned"
    out = ROOT / f"outputs/evaluation/assets/frames_{a.clip}_{a.run}_{tag}"; out.mkdir(parents=True, exist_ok=True)
    for p in sorted(det.glob("*_pred.json")):
        stem = p.stem[:-5]
        img = cv2.imread(str(ROOT / f"data/camera-data/{a.clip}/frames_all/{stem}.jpg"))
        for d in json.load(open(p)):
            l, w, h = d["l"], d["w"], d["h"]
            dims = [w, l, h] if a.swap_dims else [l, w, h]
            corners = get_lidar_3d_8points(dims, d["yaw"], [d["x"], d["y"], d["z"] + h / 2])
            cc = (l2c @ np.concatenate([corners, np.ones((8, 1))], 1).T).T[:, :3]
            if np.sum(cc[:, 2] > 1e-6) < 4: continue
            pts = project_to_image(cc, k34); draw_box_3d(img, pts, c=(0, 255, 0))
        cv2.imwrite(str(out / f"{stem}.jpg"), img)
    mp4 = ROOT / f"outputs/evaluation/assets/{a.clip}_{a.run}_{tag}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-pattern_type", "glob", "-i", str(out / "*.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    print(mp4)


if __name__ == "__main__":
    main()
