"""Join track fragments that are the same vehicle (I-24 MOTION stitching).

Why: far and partly hidden cars lose detections for a few frames, AB3DMOT drops
the track after max_age, and the car comes back under a new ID. One vehicle
becomes several short tracks, which breaks trajectories and keeps those cars out
of the motion-corrected pseudo-labels (they need 15 frames).

Method, ported from I-24 MOTION (Wang, Gloudemans et al., Transportation
Research Part C 2024; github.com/I24-MOTION/I24-postprocessing-lite,
utils/utils_stitcher_cost.py, BSD-3):
  - two fragments are candidates only if they do not overlap in time and the
    gap is at most TIME_WIN_S;
  - straight-line motion is fitted (weighted least squares, weights rising
    towards the gap) to the last ~1 s of the longer fragment and projected
    across the gap onto the first ~1 s of the other (or backwards when the
    later one is longer);
  - cost = mean Bhattacharyya distance between the projection and the other
    fragment, with a cone sigma = c + m * dt * |v| that widens with the time
    projected, plus 0.1 * gap in seconds; x velocity is not allowed to reverse;
  - pairs under STITCH_THRESH are joined, cheapest first, each fragment with at
    most one predecessor and one successor (their online min-cost flow, done
    greedily: our clips have at most a few hundred fragments).
Kept from the paper: the cost, cone weights mx, my and the threshold. Changed:
metres instead of feet, the noise floors cx, cy set to our measured box noise,
a 1 s window instead of 15 s, and a limit on the sideways jump across the
gap of SIDE_NOISE_M + SIDE_SPEED_MPS * gap, i.e. box noise plus a lane change in
progress (the fastest sideways motion in our smoothed tracks is 1.1 m/s, p95
0.7 m/s; a lane change inside one track is never affected). Without these,
checked by eye on AV_T_EW_3, the cost alone joined different cars across
1.6-2.8 s gaps and a 3 m lane offset, and a 2.6 m jump in 0.33 s (a car in the
next lane).

Coasted tails are trimmed first (postprocess_tracks.clean does that and the
smoothing; run it after this).

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/stitch_tracks.py \
        --tracks outputs/tracking/camera-data/AV_T_EW_3_v2_road/tracks.json \
        --out-dir outputs/tracking/camera-data/AV_T_EW_3_v2_road_stitch
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from score_heading import coasted   # noqa: E402

TIME_WIN_S = 1.0
SIDE_NOISE_M, SIDE_SPEED_MPS = 0.5, 1.5
STITCH_THRESH = 3.0
CX, MX, CY, MY = 0.5, 0.1, 0.6, 0.1    # metres; I-24: 0.2 ft, 0.1, 2 ft, 0.1


def trim(track):
    c = coasted(track)
    return track[:len(track) - int(np.argmax(~c[::-1]))]


def wls(t, v, w):
    A = np.c_[t, np.ones_like(t)] * np.sqrt(w)[:, None]
    slope, icpt = np.linalg.lstsq(A, v * np.sqrt(w), rcond=None)[0]
    return slope, icpt


def stitch_cost(a, b, fps):
    """I-24 stitch_cost for fragment a ending before fragment b starts."""
    ta = np.array([s["frame"] for s in a], float) / fps
    tb = np.array([s["frame"] for s in b], float) / fps
    gap = tb[0] - ta[-1]
    if gap <= 0 or gap > TIME_WIN_S:
        return 1e6
    xa, ya = np.array([[s["x"], s["y"]] for s in a]).T
    xb, yb = np.array([[s["x"], s["y"]] for s in b]).T
    if abs(yb[0] - ya[-1]) > SIDE_NOISE_M + SIDE_SPEED_MPS * gap:
        return 1e6
    n_a, n_b = min(len(a), int(fps)), min(len(b), int(fps))
    if len(a) >= len(b):        # project a forward
        t, x, y, w = ta[-n_a:], xa[-n_a:], ya[-n_a:], np.linspace(1e-6, 1, n_a)
        mt, mx_, my_ = tb[:n_b], xb[:n_b], yb[:n_b]
        pt, sign = ta[-1], 1
    else:                       # project b backwards
        t, x, y, w = tb[:n_b], xb[:n_b], yb[:n_b], np.linspace(1, 1e-6, n_b)
        mt, mx_, my_ = ta[-n_a:], xa[-n_a:], ya[-n_a:]
        pt, sign = tb[0], -1
    if len(t) < 2:
        return 1e6
    vx, bx = wls(t, x, w)
    vy, by = wls(t, y, w)
    direction = np.sign(xa[-1] - xa[0] + xb[-1] - xb[0]) or 1.0
    if vx * direction < 0:      # x velocity may not reverse
        vx, bx = 0.0, float(np.sum(w * x) / np.sum(w))
    tdiff = (mt - pt) * sign
    tx, ty = vx * mt + bx, vy * mt + by
    var_x = (CX + MX * tdiff * abs(vx)) ** 2
    var_y_pred = (CY + MY * tdiff * abs(vy)) ** 2
    var_y_meas = max(np.var(my_), CY ** 2)
    # Bhattacharyya distance between two diagonal Gaussians, as in I-24
    mu = np.r_[tx - mx_, ty - my_]
    v1 = np.r_[var_x, var_y_pred]
    v2 = np.r_[np.full(len(mt), var_x[0]), np.full(len(mt), var_y_meas)]
    v = (v1 + v2) / 2
    bd = 0.125 * np.sum(mu ** 2 / v) + 0.5 * np.sum(np.log(v / np.sqrt(v1 * v2)))
    return bd / len(mt) + 0.1 * gap


def stitch(tracks, fps):
    """{tid: states} -> ({new tid: states}, list of (tid_a, tid_b, cost) joined)."""
    frags = {k: trim(v) for k, v in tracks.items()}
    frags = {k: v for k, v in frags.items() if len(v) >= 2}
    pairs = sorted((stitch_cost(frags[a], frags[b], fps), a, b)
                   for a in frags for b in frags if a != b)
    nxt, prv, joined = {}, {}, []
    for c, a, b in pairs:
        if c >= STITCH_THRESH:
            break
        if a in nxt or b in prv:
            continue
        nxt[a], prv[b] = b, a
        joined.append((a, b, round(float(c), 3)))
    out = {}
    for head in (k for k in frags if k not in prv):
        chain, k = [], head
        while k is not None:
            chain += frags[k]
            k = nxt.get(k)
        out[head] = chain
    return out, joined


def fragmentation(tracks, fps, min_s=1.0):
    """Counts that need no ground truth."""
    lens = np.array([len(v) for v in tracks.values()])
    return {"tracks": int(len(lens)), "short_under_15": int((lens < 15).sum()),
            "at_least_1s": int((lens >= fps * min_s).sum()),
            "median_states": float(np.median(lens)) if len(lens) else 0.0,
            "states_in_tracks_1s_plus": float(lens[lens >= fps * min_s].sum() / max(lens.sum(), 1))}


def main():
    ap = argparse.ArgumentParser("Join track fragments of the same vehicle")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.tracks).read_text())
    fps = d["meta"]["frame_rate_hz"]
    before = fragmentation({k: trim(v) for k, v in d["tracks"].items()}, fps)
    d["tracks"], joined = stitch(d["tracks"], fps)
    after = fragmentation(d["tracks"], fps)
    d["meta"]["stitch"] = {"method": "I-24 MOTION stitch_cost (stitch_tracks.py)", "joined": joined,
                           "time_win_s": TIME_WIN_S, "side_limit": f"{SIDE_NOISE_M} m + {SIDE_SPEED_MPS} m/s x gap",
                           "thresh": STITCH_THRESH}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps(d))
    print(json.dumps({"joins": len(joined), "before": before, "after": after}, indent=2))


def _selfcheck():
    """One car split by a 10-frame gap is joined; a car in the next lane, and a
    car going the other way, are not."""
    fps = 30.0
    car = lambda f0, f1, y, v=28.0, x0=20.0: [
        {"frame": f, "x": x0 + v * f / fps, "y": y, "score": 0.5 + f * 1e-4, "vx": float(f)} for f in range(f0, f1)]
    tracks = {"a": car(0, 40, -8.0), "b": car(50, 90, -8.0),
              "c": car(55, 95, -11.6), "d": car(50, 90, -8.0, v=-28.0, x0=120.0)}
    out, joined = stitch(tracks, fps)
    assert [(j[0], j[1]) for j in joined] == [("a", "b")], joined
    assert len(out) == 3 and len(out["a"]) == 80, {k: len(v) for k, v in out.items()}
    # a lane change at 1 m/s through a 20-frame gap is still one car
    lc = lambda f0, f1: [dict(s, y=-8.0 - 1.0 * max(0, s["frame"] - 30) / fps) for s in car(f0, f1, -8.0)]
    _, joined = stitch({"e": lc(0, 40), "f": lc(60, 100)}, fps)
    assert [(j[0], j[1]) for j in joined] == [("e", "f")], joined
    print("selfcheck ok (gap joined, lane change through a gap joined, next lane and oncoming left apart)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
