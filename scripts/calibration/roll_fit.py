"""Camera roll and height from lane-dash cycles on both carriageways, GPS-free (D26).

A roll error (rotation about the optical axis) tilts the ground plane sideways, so one
carriageway's dashes read short and the other's long; a height error scales both. The
dashes found by dash_cycle_check.py (image points stored per lane line) are pooled over
the clips of one camera pose and back-projected with R = R_roll(phi) @ R_vp, where R_vp
comes from the lane vanishing point with no GPS pitch correction. phi and the height are
chosen so the dash cycle reads 15.24 m (WisDOT 50 ft) on every line, weighted by the
dashes per line.

    python scripts/calibration/roll_fit.py OUT_DIR CLIP [CLIP ...]   # reads OUT_DIR/<CLIP>.json
"""
import glob
import json
import sys

import numpy as np

CYCLE, NAME = 15.24, "vp_dp0"
PHI = np.radians(np.arange(-3.0, 3.0001, 0.05))
H = np.arange(13.5, 17.0001, 0.05)


def rot_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def period(x):
    """Lattice period of along-road dash positions, the dash_cycle_check.py fit (None if not a dashed line)."""
    x = np.sort(x)
    kk = np.round((x - x[0]) / 15.0)
    for _ in range(3):
        c = np.linalg.lstsq(np.c_[np.ones_like(kk), kk], x, rcond=None)[0]
        kk = np.round((x - c[0]) / c[1])
    for _ in range(3):
        good = np.abs(x - (c[0] + c[1] * kk)) < 1.2
        if good.sum() < 3:
            break
        x, kk = x[good], kk[good]
        c = np.linalg.lstsq(np.c_[np.ones_like(kk), kk], x, rcond=None)[0]
    return float(c[1]) if len(x) >= 4 and 13 < c[1] < 17.5 else None


def load(out_dir, clips):
    lines = []
    for clip in clips:
        cal = f"outputs/calibration/camera-data/{clip}/"
        k = json.load(open(glob.glob(cal + "*_anycalib_pinhole_pinhole.json")[0]))["prediction"]["intrinsics"][:4]
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
        R = np.array(json.load(open(cal + "metric_extrinsic_site.json"))["rotation"])
        for ln in json.load(open(f"{out_dir}/{clip}.json"))["lines"][NAME]:
            if ln["n"] >= 4 and ln["resid_rms"] <= 1.0:
                lines.append({"clip": clip, "lateral": ln["lateral"], "w": ln["n"] - 1, "K": K, "R": R, "uv": np.array(ln["uv"])})
    return lines


def periods(lines, phi, h):
    out = []
    for ln in lines:
        Rp = rot_z(phi) @ ln["R"]
        d = Rp.T @ (np.linalg.inv(ln["K"]) @ np.c_[ln["uv"], np.ones(len(ln["uv"]))].T)
        out.append(period((-h / d[2]) * d[0]))
    return out


def summary(lines, phi, h):
    p = periods(lines, phi, h)
    side = lambda near: [(q, ln["w"]) for q, ln in zip(p, lines) if q and (ln["lateral"] > -20) == near]
    wm = lambda r: round(float(np.average([q for q, _ in r], weights=[w for _, w in r])), 3) if r else None
    return {"roll_deg": round(float(np.degrees(phi)), 2), "height_m": round(float(h), 2), "all": wm(side(True) + side(False)),
            "near": wm(side(True)), "far": wm(side(False)), "lines": sum(q is not None for q in p)}


def main():
    out_dir, clips = sys.argv[1], sys.argv[2:]
    lines = load(out_dir, clips)
    best = None
    for phi in PHI:
        for h in H:
            p = periods(lines, phi, h)
            r = [(q - CYCLE, ln["w"]) for q, ln in zip(p, lines) if q]
            if len(r) < len(lines) * 0.8:
                continue
            cost = sum(w * e * e for e, w in r) / sum(w for _, w in r)
            if best is None or cost < best[0]:
                best = (cost, phi, h)
    rows = {"roll 0, height 15.1 (VP, no GPS pitch)": summary(lines, 0.0, 15.1),
            "roll 0, height fitted": None, "roll and height fitted": summary(lines, best[1], best[2])}
    hs = [(sum(w * (q - CYCLE) ** 2 for q, w in ((q, ln["w"]) for q, ln in zip(periods(lines, 0.0, h), lines)) if q), h) for h in H]
    rows["roll 0, height fitted"] = summary(lines, 0.0, min(hs)[1])
    res = {"clips": clips, "lines_used": len(lines), "rows": rows}
    json.dump(res, open(f"{out_dir}/roll_fit.json", "w"), indent=1)
    for k, v in rows.items():
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main()
