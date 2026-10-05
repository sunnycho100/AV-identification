"""Clean AB3DMOT tracks: drop coasted tails, then RTS-smooth position and speed.

Why, three problems with one cause each:
- AB3DMOT outputs up to max_age constant-velocity states after a track's last
  detection. They are guesses, 10 to 20% of all states. Trailing ones are cut.
- Its velocity starts at zero, so speed_mps reads about half of GPS for a
  track's first 15 frames. A backward (RTS) pass gives the early frames the
  velocity the later ones measured.
- Raw positions jump sideways frame to frame. The smoother removes the jitter.

Interior coasted states stay in the track (so frame numbering has no holes),
but the smoother treats them as missing, not as measurements. Yaw is untouched.
Writes a tracks.json of the same shape to --out-dir, so every grader runs on it.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/tracking/postprocess_tracks.py \
        --tracks outputs/tracking/camera-data/AV_T_EW_3_h151p/tracks.json \
        --out-dir outputs/tracking/camera-data/AV_T_EW_3_h151p_post
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/tracking"), str(ROOT / "scripts/evaluation")]
from rts_smooth_track import build_kf   # noqa: E402
from score_heading import coasted       # noqa: E402


V_MAX = 45.0     # m/s (100 mph); no car in these clips goes faster, so a faster jump is a wrong join


def footprint_cov(cam, f, px):
    """Extra measurement covariance (x, y) -> 2x2 for a detection whose ground point
    is off by `px` pixels: one pixel covers (r^2 + h^2) / (f h) m along the line of
    sight and sqrt(r^2 + h^2) / f across it, r the ground distance from the camera
    foot, h the camera height. Far away a pixel is metres, so a far box that wobbles
    a pixel moves metres; the smoother has to know that or it reads it as speed.
    cam = (x, y, height above the road) in the road frame, f in pixels."""
    cx, cy, h = cam

    def cov(x, y):
        dx, dy = x - cx, y - cy
        r2 = dx * dx + dy * dy
        rad, tan = (r2 + h * h) / (f * h) * px, np.sqrt(r2 + h * h) / f * px
        c, s = (dx, dy) / max(np.sqrt(r2), 1e-9)
        U = np.array([[c, -s], [s, c]])
        return U @ np.diag([rad ** 2, tan ** 2]) @ U.T
    return cov


def split_jumps(track, fps, times=None, cov=None):
    """Cut a track where it moves faster than V_MAX between consecutive states, after
    allowing three sigma of measurement noise (0.5 m, plus the pixel footprint with cov).
    Such a jump is two cars joined by the tracker or the stitcher, not one car's motion.
    Returns the pieces, in order."""
    r0 = 0.25
    sig2 = lambda s: r0 + (np.linalg.eigvalsh(cov(s["x"], s["y"]))[-1] if cov else 0.0)
    t = lambda s: times[s["frame"]][0] if times and s["frame"] in times else s["frame"] / fps
    parts = [[track[0]]]
    for a, b in zip(track, track[1:]):
        dt = max(t(b) - t(a), (b["frame"] - a["frame"]) / fps, 1.0 / fps)
        if np.hypot(b["x"] - a["x"], b["y"] - a["y"]) > V_MAX * dt + 3 * np.sqrt(sig2(a) + sig2(b)):
            parts.append([])
        parts[-1].append(b)
    return parts


def clean(track, fps, times=None, q_vel=None, cov=None):
    """One track (list of states) to its trimmed, smoothed copy. None if nothing is left.

    Missing frames (a stitched gap) are filled with copies of the previous state
    before smoothing, so the filter steps one frame at a time and treats them as
    missing; without this a gap of n frames is integrated as one frame.

    times (frame_times.load): frame -> (t_s, copy). The filter then steps by the
    real time between frames, and frames that repeat an earlier picture are
    treated as missing. Without it, frame / fps. Each state gets its t_s.

    cov (footprint_cov): per-detection measurement noise added to the base 0.5 m,
    so positions far from the camera, where a pixel is metres, count for less."""
    c = coasted(track)
    last = len(track) - 1 - int(np.argmax(~c[::-1]))      # last detected state
    track, c = [dict(s) for s in track[:last + 1]], list(c[:last + 1])
    full, miss = [], []
    for s, cs in zip(track, c):
        while full and s["frame"] > full[-1]["frame"] + 1:
            full.append({**full[-1], "frame": full[-1]["frame"] + 1})
            miss.append(True)
        full.append(s)
        miss.append(bool(cs))
    track, c = full, miss
    t = [times[s["frame"]][0] if times and s["frame"] in times else s["frame"] / fps for s in track]
    if times:
        c = [m or times.get(s["frame"], (0, False))[1] for s, m in zip(track, c)]
    for s, ts, m in zip(track, t, c):
        s["t_s"] = round(ts, 4)
        if not m:                                         # the measurement, kept to check the smoother against
            s["x_det"], s["y_det"] = s["x"], s["y"]
    if len(track) < 3:
        return track
    kf = build_kf(1.0 / fps)
    Q1 = kf.Q * fps                                       # process noise per second
    if q_vel is not None:                                 # velocity random-walk strength, m^2/s^3
        Q1[2, 2] = Q1[3, 3] = q_vel
    R0 = kf.R.copy()
    kf.x = np.array([track[0]["x"], track[0]["y"], 0, 0.])
    means, covs, Fs, Qs = [], [], [], []
    for i, (s, miss) in enumerate(zip(track, c)):
        dt = t[i] - t[i - 1] if i else 1.0 / fps
        kf.F[0, 2] = kf.F[1, 3] = dt
        kf.Q = Q1 * dt
        kf.predict()
        if not miss:
            kf.R = R0 + cov(s["x"], s["y"]) if cov else R0
            kf.update(np.array([s["x"], s["y"]]))
        means.append(kf.x.copy())
        covs.append(kf.P.copy())
        Fs.append(kf.F.copy())
        Qs.append(kf.Q.copy())
    sm, _, _, _ = kf.rts_smoother(np.array(means), np.array(covs), Fs, Qs)
    for s, m in zip(track, sm):
        s["x"], s["y"], s["vx"], s["vy"] = (float(v) for v in m)
        s["speed_mps"] = float(np.hypot(m[2], m[3]))
    if times:                                             # a frozen copy is not a new moment: drop its state
        track = [s for s in track if not times.get(s["frame"], (0, False))[1]]
    return track


def main():
    ap = argparse.ArgumentParser("Trim coasted tails and RTS-smooth tracks")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.tracks).read_text())
    fps = d["meta"]["frame_rate_hz"]
    before = sum(len(t) for t in d["tracks"].values())
    d["tracks"] = {k: clean(t, fps) for k, t in d["tracks"].items()}
    after = sum(len(t) for t in d["tracks"].values())
    d["meta"]["postprocess"] = "coasted tails trimmed, RTS-smoothed x, y, v (postprocess_tracks.py)"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps(d))
    print(f"states {before} -> {after} ({1 - after / before:.1%} tails cut), wrote {out / 'tracks.json'}")


def _selfcheck():
    """Straight car at 20 m/s with 0.3 m noise and a 5-frame coasted tail: the tail
    goes, interior misses stay, and speed is right from the first frame."""
    rng = np.random.default_rng(0)
    track = []
    for i in range(60):
        miss = i in (20, 21) or i >= 55
        s = {"frame": i, "x": 20 * i / 30 + rng.normal(0, 0.3), "y": 0.1 * i / 30 + rng.normal(0, 0.3),
             "score": 0.5 if miss else 0.5 + i * 1e-3, "vx": 0.0 if miss else float(i), "speed_mps": 0.0}
        if miss:
            s["score"], s["vx"] = track[-1]["score"], track[-1]["vx"]
        track.append(s)
    out = clean(track, 30)
    assert len(out) == 55, len(out)
    early = np.mean([s["speed_mps"] for s in out[:15]])
    assert abs(early - 20) < 1.0, early
    lat = np.std([s["y"] - 0.1 * s["frame"] / 30 for s in out])
    assert lat < 0.15, lat
    gap = [s for s in track[:55] if not 30 <= s["frame"] < 38]                 # an 8-frame stitched gap
    out2 = clean(gap, 30)
    sp = np.array([s["speed_mps"] for s in out2])
    assert len(out2) == 55 and [s["frame"] for s in out2] == list(range(55)), len(out2)
    assert np.abs(sp - 20).max() < 2.0, sp.round(1)
    # frames really 1/31 s apart, plus a frozen stretch (frames 20-24 repeat frame 19's picture)
    times = {f: (f / 31.0, False) for f in range(60)}
    times.update({f: (19 / 31.0, True) for f in range(20, 25)})
    car = [{"frame": f, "x": 20 * times[f][0] + rng.normal(0, 0.05), "y": 0.0, "score": 0.6 + f * 1e-3,
            "vx": 1.0, "speed_mps": 0.0} for f in range(60)]
    out3 = clean(car, 30, times)
    sp3 = np.array([s["speed_mps"] for s in out3 if s["frame"] > 30])
    assert abs(np.median(sp3) - 20) < 0.3, np.median(sp3)
    assert out3[-1]["t_s"] == round(59 / 31.0, 4) and not any(20 <= s["frame"] <= 24 for s in out3)
    # two cars joined into one track: car B is 15 m ahead of car A when the track switches
    joined = [{"frame": f, "x": 20 * f / 30 + (15.0 if f >= 30 else 0.0), "y": 0.0, "score": 0.6 + f * 1e-3,
               "vx": 1.0, "speed_mps": 0.0} for f in range(60)]
    parts = split_jumps(joined, 30)
    assert [len(p) for p in parts] == [30, 30], [len(p) for p in parts]
    assert all(abs(np.median([s["speed_mps"] for s in clean(p, 30)]) - 20) < 0.5 for p in parts)
    # a car 300-380 m out whose ground point wobbles 3 m along the view (1.5 px at 2 m/px):
    # with that noise in the smoother it is not split and its speed stays sane
    cam = np.array([0.0, 0.0, 21.0])
    cov = footprint_cov(cam, 1950.0, 1.5)
    far = [{"frame": f, "x": 300 + 25 * f / 30 + rng.normal(0, 3.0), "y": 40.0, "score": 0.6 + f * 1e-3,
            "vx": 1.0, "speed_mps": 0.0} for f in range(90)]
    assert len(split_jumps(far, 30, cov=cov)) == 1
    spf = np.array([s["speed_mps"] for s in clean(far, 30, cov=cov)])
    spn = np.array([s["speed_mps"] for s in clean(far, 30)])
    assert spf.max() < 32 and abs(np.median(spf) - 25) < 2, (spf.max(), np.median(spf))
    print(f"selfcheck ok (55 of 60 states kept, early speed {early:.2f} m/s, lateral std {lat:.3f} m, "
          f"8-frame gap filled, speed {sp.min():.1f}-{sp.max():.1f}; 31 fps clip with a freeze: "
          f"{np.median(sp3):.2f} m/s, frame/30 would read {np.median(sp3) * 30 / 31:.2f}; joined cars split "
          f"in two; far car 25 m/s read {np.median(spf):.1f}, max {spf.max():.1f} (without range noise max "
          f"{spn.max():.1f}))")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
