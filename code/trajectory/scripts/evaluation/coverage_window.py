"""Fixed coverage denominator for the GPS car (decision D35): the frames it is visible and in range.

A frame counts when the GPS car's 3D box (gps_to_image's 4.6 x 1.9 x 1.5 m, at the GPS
position and heading), projected through the clip's calibration, is
  1. fully inside the image        KITTI tracking evaluation drops ground truth with truncation > 0
  2. at least 25 px tall           KITTI moderate minimum box height
  3. centred inside the BEV grid   x 0 to 102.4 m, |y| < 51.2 m: a range cut, as nuScenes evaluates cars within 50 m
Occlusion is not modelled (KITTI's occlusion levels are hand labels); the by-eye review covers it.

GPS timing: the 2026 GPS clock is off by seconds (17 to 137 m along the road), so each pass gets
one rigid shift, the median of the compared methods' mean offsets to the GPS. The shift is the
same for every method, so the window does not move with the method being scored.
Window = the longest run of counted frames (frozen copies neither count nor break it).

    python3 scripts/evaluation/coverage_window.py            # windows and coverage
    python3 scripts/evaluation/coverage_window.py render     # review sheet for the by-eye check
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import heldout_eval as he           # noqa: E402
import round2_same_frames as sf     # noqa: E402
import round2_study as r2           # noqa: E402
from gps_to_image import H, L, ROAD_Z, W, gps_road, load_clip, load_site   # noqa: E402

IMG_W, IMG_H = 1920, 1080
MIN_H_PX, GRID_X, GRID_Y = 25, (0.0, 102.4), 51.2
OUT = ROOT / "outputs/evaluation/coverage_window"
SHEET = ROOT / "outputs/reports/2026-11-04_new_todd/annotated/coverage_window_review.jpg"
SCORED = {"r1 S4": ("r1", "s4"), "r1 hb30": ("r1", "hb"), "r2 full S4": ("r2full", "s4"), "r2 full hb30": ("r2full", "hb"),
          "r2 no-2D hb30": ("r2no2d", "hb")}


def methods(clip):
    """Every compared run: the round-2 study's eight plus the 2D baseline and the pretrained head."""
    m = {f"{h} {t}": r2.tag(h, t) for h in r2.HEADS for t in r2.TRACK}
    m["2D"], m["S0"] = he.tag(clip, "2D + ground plane"), he.tag(clip, "S0 pretrained")
    return m


def box_rect(x, y, yaw, K, l2c):
    """Image rectangle (u0, v0, u1, v1) of the car box, or None if any corner is behind the camera."""
    d = np.array([[a * L / 2, b * W / 2, z * H] for a in (-1, 1) for b in (-1, 1) for z in (0, 1)])
    cs, sn = np.cos(yaw), np.sin(yaw)
    pts = np.c_[x + d[:, 0] * cs - d[:, 1] * sn, y + d[:, 0] * sn + d[:, 1] * cs, ROAD_Z + d[:, 2]]
    c = (l2c @ np.c_[pts, np.ones(8)].T)[:3]
    if (c[2] <= 1e-6).any():
        return None
    uv = (K @ c)[:2] / c[2]
    return (*uv.min(1), *uv.max(1))


def rule(x, y, r):
    """The three D35 conditions for one frame: 'in', or the first one that fails."""
    if not (GRID_X[0] < x < GRID_X[1] and abs(y) < GRID_Y):
        return "range"
    if r is None or r[0] < 0 or r[1] < 0 or r[2] > IMG_W - 1 or r[3] > IMG_H - 1:
        return "edge"
    return "in" if r[3] - r[1] >= MIN_H_PX else "small"


def longest_run(ok, skip):
    best, cur = [], []
    for i, (o, s) in enumerate(zip(ok, skip)):
        if s:
            continue
        cur = cur + [i] if o else []
        if len(cur) > len(best):
            best = cur
    return best


def window(clip, T):
    c = load_clip(clip, he.tag(clip, "S4"))
    gps, t = c["gps"], c["t"]
    obs, offs = {}, {}
    for name, tag in methods(clip).items():
        d = sf.load(clip, tag, T, gps)
        if d is None or d["observed"].sum() < 10:
            continue
        k = d["observed"]
        obs[name] = set(d["frame"][k].tolist())
        offs[name] = (d["q"][k] - d["g"][k]).mean(0)
    off = np.median(np.array(list(offs.values())), axis=0)
    has = (t >= gps["t"][0]) & (t <= gps["t"][-1])
    xy, psi, v = (np.full((len(t), 2), np.nan), np.full(len(t), np.nan), np.full(len(t), np.nan))
    xy[has], psi[has], v[has] = gps_road(gps, T, t[has])
    xy = xy + off
    why = ["no_gps" if not h else rule(p[0], p[1], box_rect(p[0], p[1], a, c["K"], c["l2c"]))
           for h, p, a in zip(has, xy, psi)]
    win = longest_run([w == "in" for w in why], c["copy"])
    speed = float(np.nanmedian(v))
    spread = max(abs(o[0] - off[0]) for o in offs.values()) / speed * 30      # frames
    before = next((why[i] for i in range(win[0] - 1, -1, -1) if not c["copy"][i]), "clip_start")
    after = next((why[i] for i in range(win[-1] + 1, len(t)) if not c["copy"][i]), "clip_end")
    W_ = set(win)
    return {"clip": clip, "entry": win[0], "exit": win[-1], "frames": len(win), "seconds": round(len(win) / 30, 2),
            "ends": [before, after], "offset_m": [round(float(o), 2) for o in off],
            "offset_spread_frames": round(float(spread), 1), "methods_aligned": sorted(offs),
            "coverage": {m: {"in": len(obs.get(" ".join(ht), set()) & W_), "outside": len(obs.get(" ".join(ht), set()) - W_)}
                         for m, ht in SCORED.items()} | {"2D": {"in": len(obs.get("2D", set()) & W_),
                                                                "outside": len(obs.get("2D", set()) - W_)}},
            "_why": why, "_xy": xy.tolist(), "_psi": psi.tolist()}


def score():
    T = load_site()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {c: window(c, T) for c in he.HELD}
    (OUT / "windows.json").write_text(json.dumps(res))
    names = list(next(iter(res.values()))["coverage"])
    print("| Pass | Window (frames) | Seconds | Ends (before, after) | Shift spread (frames) | " + " | ".join(names) + " |")
    print("|---|---|---|---|---|" + "---|" * len(names))
    for c, r in res.items():
        cov = " | ".join(f"{r['coverage'][m]['in'] / r['frames']:.0%}" for m in names)
        print(f"| {c} | {r['entry']} to {r['exit']} | {r['seconds']} | {', '.join(r['ends'])} | {r['offset_spread_frames']} | {cov} |")
    tot = sum(r["frames"] for r in res.values())
    print(f"| all 9 | {tot} | {tot / 30:.1f} | | | " + " | ".join(
        f"{sum(r['coverage'][m]['in'] for r in res.values()) / tot:.0%}" for m in names) + " |")
    print("tracked outside the window (frames):", {m: sum(r["coverage"][m]["outside"] for r in res.values()) for m in names})


def render():
    res = json.loads((OUT / "windows.json").read_text())
    rows = []
    for clip, r in res.items():
        c = load_clip(clip, he.tag(clip, "S4"))
        tiles = []
        for name, f in (("before entry", r["entry"] - 10), ("entry", r["entry"]), ("exit", r["exit"]), ("after exit", r["exit"] + 10)):
            f = min(max(f, 0), len(c["t"]) - 1)
            img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{f:03d}.jpg"))
            x, y = r["_xy"][f]
            ctr = (IMG_W // 2, IMG_H // 2)
            rc = box_rect(x, y, r["_psi"][f], c["K"], c["l2c"]) if np.isfinite(x) else None
            if rc is not None:                      # the rectangle the rule judges (orange = counted)
                u, v, u1, v1 = map(int, rc)
                cv2.rectangle(img, (u, v), (u1, v1), (0, 140, 255) if r["_why"][f] == "in" else (180, 180, 180), 2)
                ctr = ((u + u1) // 2, (v + v1) // 2)
            u0 = min(max(ctr[0] - 320, 0), IMG_W - 640)
            v0 = min(max(ctr[1] - 180, 0), IMG_H - 360)
            tile = img[v0:v0 + 360, u0:u0 + 640].copy()
            cv2.rectangle(tile, (0, 0), (640, 30), (0, 0, 0), -1)
            txt = f"{clip} {name}: frame {f}, {r['_why'][f]}" + (f", x {x:.0f} m" if np.isfinite(x) else "")
            cv2.putText(tile, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(SHEET), np.vstack(rows))
    print(SHEET)


def _selfcheck():
    assert longest_run([1, 1, 0, 1, 1, 1, 0], [0] * 7) == [3, 4, 5]
    assert longest_run([1, 0, 1, 1], [0, 1, 0, 0]) == [0, 2, 3]          # a frozen copy does not break the run
    assert rule(150.0, -10.0, (900, 500, 920, 540)) == "range"
    assert rule(50.0, -10.0, (-3, 500, 100, 600)) == "edge"
    assert rule(50.0, -10.0, (900, 500, 920, 520)) == "small"
    assert rule(50.0, -10.0, (900, 500, 960, 540)) == "in"


if __name__ == "__main__":
    _selfcheck()
    {"render": render}.get(sys.argv[1] if len(sys.argv) > 1 else "", score)()
