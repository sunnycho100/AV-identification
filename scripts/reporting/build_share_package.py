"""Assemble the per-clip visualisation package to share with collaborators.

Per clip: an mp4 of the annotated frames, a contact sheet sampling the clip, and
the numbers that say whether the run is sane (detections per frame, where box
bottoms sit relative to the road, how much of the output is off the roadway).

Frame numbers from ffmpeg's -frame_pts are irregular because the encoder drops
frames, so the video is built from a sorted file list rather than a %03d pattern,
which would silently stop at the first gap.

    .venv/bin/python scripts/reporting/build_share_package.py \
        --out-dir outputs/reports/2026-08-11-box-visualizations
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTACT_TILES = 6
SCORE_MIN = 0.3


def clip_stats(det_dir):
    """Detection counts and 3D placement, read straight from the predictions."""
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    road_z = cal["road_plane_z_in_output"]
    per_frame, above, lateral = [], [], []
    for pf in sorted(det_dir.glob("*_pred.json")):
        cars = [d for d in json.loads(pf.read_text())
                if d["class_name"] == "car" and d["score"] >= SCORE_MIN]
        per_frame.append(len(cars))
        for d in cars:
            above.append(d["z"] - road_z)
            lateral.append(d["y"])
    if not per_frame:
        return None
    lateral = np.array(lateral)
    return {
        "frames": len(per_frame),
        "cars_per_frame_median": float(np.median(per_frame)),
        "cars_total": int(sum(per_frame)),
        "box_bottom_above_road_m": float(np.median(above)),
        "lateral_p10_m": float(np.percentile(lateral, 10)),
        "lateral_p90_m": float(np.percentile(lateral, 90)),
        "camera_height_m": float((-np.array(cal["lidar2cam"])[:3, :3].T
                                  @ np.array(cal["lidar2cam"])[:3, 3])[2] - road_z),
    }


def contact_sheet(det_dir, out_path, cols=3):
    shots = sorted(det_dir.glob("*_annotated.jpg"))
    if not shots:
        return False
    idx = np.linspace(0, len(shots) - 1, CONTACT_TILES).round().astype(int)
    tiles = []
    for i in idx:
        img = cv2.imread(str(shots[i]))
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        cv2.putText(img, shots[i].name.replace("_annotated.jpg", ""), (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, shots[i].name.replace("_annotated.jpg", ""), (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))
    return True


def make_video(det_dir, out_path, fps=30):
    shots = sorted(det_dir.glob("*_annotated.jpg"))
    if not shots:
        return False
    listing = det_dir / "_frames.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in shots))
    cmd = ["ffmpeg", "-y", "-v", "error", "-r", str(fps), "-f", "concat",
           "-safe", "0", "-i", str(listing), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    listing.unlink(missing_ok=True)
    return ok


def main():
    ap = argparse.ArgumentParser("Build the share package")
    ap.add_argument("--clips", nargs="+", default=[
        "AV_T_WE_1", "HV_T_EW_2", "AV_T_EW_3", "HV_T_EW_1", "AV_T_WE_3"])
    ap.add_argument("--suffix", default="_phase1")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for clip in args.clips:
        det = ROOT / "outputs/object_detection/camera-data" / f"{clip}{args.suffix}"
        if not det.exists():
            print(f"{clip}: no detection dir, skipped")
            continue
        st = clip_stats(det)
        if st is None:
            print(f"{clip}: no predictions, skipped")
            continue
        # keep the recorded clip name, but mark it so these are never mistaken
        # for the original footage the collaborators already hold
        vid = make_video(det, out / f"{clip}_detections.mp4")
        sheet = contact_sheet(det, out / f"{clip}_contact.jpg")
        table[clip] = st
        print(f"{clip:<12} {st['frames']:>4} frames  "
              f"{st['cars_per_frame_median']:>4.1f} cars/frame  "
              f"bottoms {st['box_bottom_above_road_m']:+.2f} m vs road  "
              f"mp4={'ok' if vid else 'FAIL'} sheet={'ok' if sheet else 'FAIL'}")
    (out / "stats.json").write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
