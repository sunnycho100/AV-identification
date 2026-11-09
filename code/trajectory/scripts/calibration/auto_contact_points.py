"""Road contact points of the instrumented car, from the image alone.

The clip's median frame is the empty road; subtracting it leaves the moving
cars. The blob nearest a reference track (picked from the image, never by GPS)
is the target, and its lowest pixel row, horizontally its median, is where the
car meets the road (front bumper when it approaches, rear when it leaves).
The reference track only selects which blob; its position is never used.

Writes outputs/calibration/contact_clicks/<clip>/auto_contacts.json and a
contact sheet for review. Frames whose blob touches the overlay text or the
image edge should be marked skipped by hand after looking at the sheet.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/calibration/auto_contact_points.py \
        --clip AV_T_EW_3 --ref-run phase1 --ref-track target
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ref-run", required=True)
    ap.add_argument("--ref-track", required=True, help='"target" for target_track.json, else a track id')
    ap.add_argument("--max-range", type=float, default=90.0)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    clip = a.clip
    D = ROOT / "outputs/calibration/contact_clicks" / clip
    D.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(ROOT / f"data/camera-data/{clip}/frames_all/*.jpg")))
    bg = np.median(np.stack([cv2.imread(p) for p in paths[::3]]), axis=0).astype(np.uint8)
    run_dir = ROOT / "outputs/tracking/camera-data" / f"{clip}_{a.ref_run}"
    st = (json.loads((run_dir / "target_track.json").read_text())["states"] if a.ref_track == "target"
          else json.loads((run_dir / "tracks.json").read_text())["tracks"][a.ref_track])
    st = sorted([s for s in st if s["x"] <= a.max_range], key=lambda s: s["frame"])
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{clip}_{a.ref_run}"
                      / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    zr = cal.get("road_plane_z_in_output", -1.73)

    def proj(p):
        q = K @ (M[:3, :3] @ np.array(p) + M[:3, 3])
        return q[:2] / q[2]

    clicks, tiles = [], []
    for s in [st[int(i)] for i in np.linspace(0, len(st) - 1, a.n)]:
        f = s["frame"]
        img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
        body = proj([s["x"], s["y"], zr + 1.0])
        m = (cv2.absdiff(img, bg).max(axis=2) > 35).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        _, lab, _, _ = cv2.connectedComponentsWithStats(m)
        ys, xs = np.nonzero(lab > 0)
        j = int(np.argmin((xs - body[0]) ** 2 + (ys - body[1]) ** 2))
        k = lab[ys[j], xs[j]]
        by, bx = np.nonzero(lab == k)
        ymax = by.max()
        u, v = float(np.median(bx[by >= ymax - 2])), float(ymax)
        touches_edge = bool(ymax >= img.shape[0] - 2 or bx.min() <= 1 or bx.max() >= img.shape[1] - 2)
        clicks.append({"frame": f, "u": u, "v": v, "blob_px": int(len(by)), "range_m": round(s["x"], 1),
                       "skipped": touches_edge})
        x0 = max(0, min(img.shape[1] - 240, int(u - 120)))
        y0 = max(0, min(img.shape[0] - 200, int(v - 150)))
        tile = img[y0:y0 + 200, x0:x0 + 240].copy()
        ov = np.zeros_like(tile)
        ov[(lab == k)[y0:y0 + 200, x0:x0 + 240]] = (255, 120, 0)
        tile = cv2.addWeighted(tile, 1, ov, 0.35, 0)
        cv2.circle(tile, (int(u - x0), int(v - y0)), 5, (0, 0, 255), -1)
        cv2.putText(tile, f"{f} {s['x']:.0f}m{' SKIP' if touches_edge else ''}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(cv2.resize(tile, (360, 300), interpolation=cv2.INTER_CUBIC))
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(D / "auto_contacts_sheet.jpg"), np.vstack(rows))
    (D / "auto_contacts.json").write_text(json.dumps({
        "clip": clip, "point": "lowest point of the background-subtracted target blob (auto)",
        "reference": f"{a.ref_run} track {a.ref_track} (image-identified, selects the blob only)",
        "clicks": clicks}, indent=1))
    print(f"{clip}: {len(clicks)} frames, {sum(c['skipped'] for c in clicks)} auto-skipped; "
          f"review {D / 'auto_contacts_sheet.jpg'}")


if __name__ == "__main__":
    main()
