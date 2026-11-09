"""Side-by-side video of two detectors' final trajectories on one clip.

Each panel is render_trajectories.py's drawing (tracked, smoothed, id and mph,
the image-identified GPS car in orange) for one tag, plus range lines across
the road at 50, 100 and 140 m so coverage by distance is visible.
Writes outputs/reports/monouni_compare/<clip>_<tagA>_vs_<tagB>.mp4 and a contact
sheet jpg of 5 frames.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/reporting/render_compare_detectors.py \
        --clip AV_T_EW_3 --tags ft102 monouni_B --labels "BEVHeight ft102" "MonoUNI zoom"
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reporting"))
from render_trajectories import draw_states   # noqa: E402

RANGES = (50, 100, 140)


def load(clip, tag):
    run = f"{clip}_{tag}"
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / run / "calibration_used.json").read_text())
    k34 = np.zeros((3, 4))
    k34[:3, :3] = np.array(cal["K"])
    traj = ROOT / "outputs/trajectories" / run
    inst = traj / "instrumented.json"
    by = {}
    for tid, st in json.loads((traj / "tracks.json").read_text())["tracks"].items():
        for s in st:
            by.setdefault(s["frame"], []).append((tid, s))
    return {"k34": k34, "l2c": np.array(cal["lidar2cam"]), "z": cal.get("road_plane_z_in_output", -1.73),
            "by": by, "mark": json.loads(inst.read_text())["vehicle_id"] if inst.exists() else None}


def range_lines(img, r):
    for x in RANGES:
        p = np.array([[x, y, r["z"], 1.0] for y in np.linspace(-39, -1, 20)])
        c = (r["l2c"] @ p.T)[:3]
        uv = (r["k34"][:, :3] @ c)[:2] / c[2]
        pts = uv.T.round().astype(np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 0), 2)
        cv2.putText(img, f"{x} m", tuple(pts[-1] + [8, 0]), 0, 0.7, (255, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser("Two detectors side by side")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tags", nargs=2, required=True)
    ap.add_argument("--labels", nargs=2, required=True)
    a = ap.parse_args()
    runs = [load(a.clip, t) for t in a.tags]
    out = ROOT / "outputs/reports/monouni_compare"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.clip}_{a.tags[0]}_vs_{a.tags[1]}"
    frames = sorted((ROOT / f"data/camera-data/{a.clip}/frames_all").glob("*.jpg"))
    sheet = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            panels = []
            for r, lab in zip(runs, a.labels):
                img = cv2.imread(str(f))
                range_lines(img, r)
                draw_states(img, r["by"].get(i, []), r["k34"], r["l2c"], r["mark"])
                n = len(r["by"].get(i, []))
                cv2.putText(img, f"{lab}   {n} tracked   frame {i}", (20, 50), 0, 1.3, (0, 0, 0), 6)
                cv2.putText(img, f"{lab}   {n} tracked   frame {i}", (20, 50), 0, 1.3, (255, 255, 255), 2)
                panels.append(cv2.resize(img, (960, 540)))
            im = np.hstack(panels)
            cv2.imwrite(f"{tmp}/{i:03d}.jpg", im)
            if i in np.linspace(0, len(frames) - 1, 5).astype(int):
                sheet.append(im)
        mp4 = out / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{tmp}/%03d.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    cv2.imwrite(str(out / f"{name}_sheet.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(frames)} frames -> {mp4}")


if __name__ == "__main__":
    main()
