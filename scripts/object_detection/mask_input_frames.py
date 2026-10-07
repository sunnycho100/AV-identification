"""Mask the detector's input to the highway: everything off the road becomes neutral grey.

road_mask.py filters boxes after BEVHeight has run on the full picture. This tests
the other order: the network never sees the lots, the road below the bridge or the
buildings, which sit on other planes than the one it assumes. A pixel is kept
when its viewing ray passes through the road band (road_mask.ROAD_X by ROAD_Y in
road coordinates) anywhere between the road surface and HEIGHT_M above it, so a
car's roof stays visible even where it rises above the road behind it. The band
runs to the horizon (the track-built road_mask.png stops near 100 m). Masked
pixels get the ImageNet mean colour, which the detector's normalisation turns into
zeros, so the mask adds no edges of its own beyond the road boundary.

Writes data/camera-data/<clip>/frames_masked/ and, for review,
outputs/calibration/camera-data/<clip>/input_mask.png and input_mask_preview.jpg.

--feature instead writes feature_mask.npy for the other option: the image stays
whole and lss_fpn.py zeroes the off-deck cells of the 54 x 96 feature grid (one
cell = 16 px of the 1536 x 864 network input) just before they are placed on the
BEV map. A cell is kept when at least half of it is on the band.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/object_detection/mask_input_frames.py --clip AV_T_EW_3 --preview-only
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/object_detection"))
from road_mask import ROAD_X, ROAD_Y   # noqa: E402

HEIGHT_M = 2.5                        # tallest vehicle part kept visible (input mask, as run)
FEATURE_HEIGHT_M = 2.5                # feature mask; 4.5 m kept a 4 m truck's roof in the outer lane but let
                                      # 40 parking-lot boxes through (5 clips) against 2 at 2.5 m
FEATURE_GRID = (54, 96)               # final_dim (864, 1536) / downsample_factor 16
MEAN_BGR = (104, 116, 124)            # ImageNet mean, the detector's zero after normalisation
EXTRINSIC = "metric_extrinsic_h151_dpm031.json"


def input_mask(K, R, t, w=1920, h=1080, height=HEIGHT_M, n=12):
    """bool HxW: the pixel's ray crosses the road band between z = 0 and z = height."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = R.T @ np.linalg.solve(K, np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))
    c = -R.T @ t                                         # camera centre, road frame
    keep = np.zeros(u.size, bool)
    down = rays[2] < -1e-9
    for z in np.linspace(0.0, height, n):                # sample the slab between road and roof
        s = (z - c[2]) / np.where(down, rays[2], -1.0)
        x, y = c[0] + s * rays[0], c[1] + s * rays[1]
        keep |= down & (x >= ROAD_X[0]) & (x <= ROAD_X[1]) & (y >= ROAD_Y[0]) & (y <= ROAD_Y[1])
    return keep.reshape(h, w)


def feature_mask(K, R, t, grid=FEATURE_GRID, height=FEATURE_HEIGHT_M):
    """float32 grid (fH, fW): 1 where at least half the cell's pixels are on the band."""
    m = input_mask(K, R, t, height=height).astype(np.float32)
    return (cv2.resize(m, grid[::-1], interpolation=cv2.INTER_AREA) >= 0.5).astype(np.float32)


def load(clip):
    cal = ROOT / "outputs/calibration/camera-data" / clip
    fx, fy, cx, cy = json.loads(next(cal.glob("*_anycalib_pinhole_pinhole.json")).read_text())["prediction"]["intrinsics"]
    ext = json.loads((cal / EXTRINSIC).read_text())
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array(ext["rotation"]), np.array(ext["translation"])


def main():
    ap = argparse.ArgumentParser("Mask detector input to the highway")
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--feature", action="store_true", help="write feature_mask.npy only")
    a = ap.parse_args()
    for clip in a.clip:
        K, R, t = load(clip)
        if a.feature:
            fm = feature_mask(K, R, t)
            np.save(ROOT / "outputs/calibration/camera-data" / clip / "feature_mask.npy", fm)
            print(f"{clip}: feature mask keeps {fm.mean():.0%} of {fm.size} cells")
            continue
        m = input_mask(K, R, t)
        cal = ROOT / "outputs/calibration/camera-data" / clip
        cv2.imwrite(str(cal / "input_mask.png"), m.astype(np.uint8) * 255)
        frames = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))
        im = cv2.imread(str(frames[len(frames) // 2]))
        masked = np.where(m[..., None], im, np.array(MEAN_BGR, np.uint8))
        edge = cv2.dilate(m.astype(np.uint8), np.ones((5, 5))) - m.astype(np.uint8)
        over = im.copy()
        over[edge > 0] = (0, 255, 255)
        cv2.imwrite(str(cal / "input_mask_preview.jpg"),
                    np.vstack([cv2.resize(over, (960, 540)), cv2.resize(masked, (960, 540))]))
        print(f"{clip}: keeps {m.mean():.0%} of the image -> {cal / 'input_mask_preview.jpg'}")
        if a.preview_only:
            continue
        out = ROOT / f"data/camera-data/{clip}/frames_masked"
        out.mkdir(exist_ok=True)
        for f in frames:
            cv2.imwrite(str(out / f.name), np.where(m[..., None], cv2.imread(str(f)), np.array(MEAN_BGR, np.uint8)),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   {len(frames)} masked frames -> {out}")


def _selfcheck():
    """A point on the road and a car roof above it are kept; a parked car 10 m off the band is not."""
    K, R, t = load("AV_T_EW_3")
    m = input_mask(K, R, t)
    px = lambda p: tuple(int(round(q)) for q in ((K @ (R @ np.array(p) + t))[:2] / (K @ (R @ np.array(p) + t))[2]))
    for p in ([60.0, -10.0, 0.0], [60.0, -10.0, 2.0], [40.0, -30.0, 1.5]):
        u, v = px(p)
        assert m[v, u], p
    u, v = px([45.0, -49.0, 0.0])
    assert not m[v, u]
    fm = feature_mask(K, R, t)
    assert fm.shape == FEATURE_GRID
    cell = lambda p: (px(p)[1] * FEATURE_GRID[0] // 1080, px(p)[0] * FEATURE_GRID[1] // 1920)
    assert fm[cell([60.0, -10.0, 1.5])] == 1, "car on the road dropped"
    assert fm[cell([45.0, -49.0, 0.0])] == 0, "parked car kept"
    print(f"selfcheck ok (road and roof kept, parked car off the band masked; keeps {m.mean():.0%}; "
          f"feature grid keeps {fm.mean():.0%})")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
