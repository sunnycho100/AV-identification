"""Evaluation only: compare the SHAPE of the lab vehicle's BEV trajectory with its GPS.

The GPS-to-video timing and the GPS-to-road mapping both carry a systematic
error (lab timing is the old matching method; the site transform is fitted on
two clips). Following Hang (2026-10-07), we only care about vehicle behaviour,
so the constant offset between the two trajectories is removed before
comparing: the mean BEV-minus-GPS difference along and across the road. What
is left is how well the camera reproduces the car's motion.

Per clip and detector tag, on the image-identified lab vehicle, at the lab's
timing (video_time_sec, no fitted time shift), frames that are frozen copies
excluded:
  offset_m        the constant (along, across) offset that was removed
  along_rms_m     along-road residual after removing it (contains any timing
                  error that is not constant, e.g. tau times a speed change)
  across_rms_m    across-road residual after removing it
  across_slope_deg  the straight-line trend in the across residual, as an angle:
                  a constant direction difference between BEV and GPS shows up
                  as a ramp, so across_rms_m double-counts it
  across_detrended_rms_m  across residual with that trend removed (lane keeping)
  heading_mae_deg travel direction of the smoothed track vs GPS heading
  speed_mae_mph   track speed vs GPS speed
  speed_ratio     mean track speed / mean GPS speed (scale check)
and a plot outputs/reports/relative_trajectory/<clip>.png comparing tags.

All numbers are lab-timing agreement, not pure BEV error: a 0.4-2 s timing
error times a speed change leaks into along_rms_m and speed_mae_mph. Tags are
scored on the frames every tag's track covers (common), so they share that
leak; each tag's own full-track numbers are kept under "own".

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/relative_trajectory.py \
        --tags ft102 ft102mask ft102feat25
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]

CLIPS = ["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "AV_T_EW_3"]   # HV_T_EW_2: car not in view
MPH = 2.23694


def compare(q, v, g, psi, vg):
    """q, v: BEV positions and velocities (N,2); g, psi, vg: GPS positions (N,2),
    heading (N,) and speed (N,) at the same times, all in the road frame."""
    from site_error_model import wrap
    d = q - g
    off = d.mean(0)
    r = d - off
    along = g[:, 0] - g[0, 0]                    # GPS distance along the road
    slope = np.polyfit(along, r[:, 1], 1)[0] if np.ptp(along) > 1 else 0.0
    det = r[:, 1] - np.polyval(np.polyfit(along, r[:, 1], 1), along) if np.ptp(along) > 1 else r[:, 1]
    sp = np.hypot(v[:, 0], v[:, 1])
    hd = np.arctan2(v[:, 1], v[:, 0])
    return {"frames": len(q), "offset_m": [round(float(o), 3) for o in off],
            "along_rms_m": round(float(np.sqrt(np.mean(r[:, 0] ** 2))), 3),
            "across_rms_m": round(float(np.sqrt(np.mean(r[:, 1] ** 2))), 3),
            "across_slope_deg": round(float(np.degrees(np.arctan(slope))), 3),
            "across_detrended_rms_m": round(float(np.sqrt(np.mean(det ** 2))), 3),
            "heading_mae_deg": round(float(np.degrees(np.mean(np.abs(wrap(hd - psi))))), 3),
            "speed_mae_mph": round(float(np.mean(np.abs(sp - vg))) * MPH, 3),
            "speed_ratio": round(float(sp.mean() / vg.mean()), 4)}, r


def load(clip, tag, T):
    """The picked car's states inside the GPS window, with GPS at the same times."""
    from gps_to_image import gps_road, load_clip
    c = load_clip(clip, tag)
    car = c["car"]
    if car is None:
        return None
    t0, t1 = c["gps"]["t"][0], c["gps"]["t"][-1]
    k = (car["t"] >= t0) & (car["t"] <= t1)          # np.interp would clamp outside
    g, psi, vg = gps_road(c["gps"], T, car["t"][k])
    return {"id": car["id"], "frame": car["frame"][k], "t": car["t"][k], "q": car["q"][k],
            "v": car["v"][k], "g": g, "psi": psi, "vg": vg}


def score(d, frames=None):
    k = np.ones(len(d["frame"]), bool) if frames is None else np.isin(d["frame"], frames)
    res, r = compare(d["q"][k], d["v"][k], d["g"][k], d["psi"][k], d["vg"][k])
    return res, {"t": d["t"][k], "r": r, "sp": np.hypot(*d["v"][k].T), "vg": d["vg"][k]}


def plot(clip, series, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    first = next(iter(series.values()))
    ax[0].plot(first["t"], first["r"][:, 1] * 0, color="0.6", lw=1)
    for tag, s in series.items():
        ax[0].plot(s["t"], s["r"][:, 1], label=tag)
        ax[1].plot(s["t"], s["r"][:, 0], label=tag)
        ax[2].plot(s["t"], s["sp"] * MPH, label=tag)
    ax[2].plot(first["t"], first["vg"] * MPH, "k--", label="GPS")
    ax[0].set(title="Across-road difference after removing the offset", xlabel="video time (s)", ylabel="m")
    ax[1].set(title="Along-road difference after removing the offset", xlabel="video time (s)", ylabel="m")
    ax[2].set(title="Speed", xlabel="video time (s)", ylabel="mph")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{clip}: lab vehicle, BEV trajectory vs GPS (constant offset removed, lab timing)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _selfcheck():
    """A track equal to the GPS plus a constant offset and known noise returns the
    offset and the noise; a constant offset alone leaves zero residual."""
    rng = np.random.default_rng(0)
    t = np.arange(90) / 30
    g = np.c_[100 - 27 * t, np.full_like(t, -5.2)]
    psi = np.full_like(t, math.pi)
    vg = np.full_like(t, 27.0)
    v = np.c_[np.full_like(t, -27.0), np.zeros_like(t)]
    res, _ = compare(g + [3.0, -0.4], v, g, psi, vg)
    assert res["offset_m"] == [3.0, -0.4] and res["along_rms_m"] == 0 and res["across_rms_m"] == 0, res
    assert res["heading_mae_deg"] < 1e-9 and res["speed_mae_mph"] == 0 and res["speed_ratio"] == 1, res
    noise = rng.normal(0, 0.2, (len(t), 2))
    res, _ = compare(g + [3.0, -0.4] + noise, v, g, psi, vg)
    assert abs(res["across_rms_m"] - 0.2) < 0.04 and abs(res["along_rms_m"] - 0.2) < 0.04, res
    # a 1 deg direction difference is a ramp across: reported as slope, detrended to ~0
    a = math.radians(1.0)
    gr = g @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    res, _ = compare(gr, v, g, psi, vg)
    assert abs(abs(res["across_slope_deg"]) - 1.0) < 0.02 and res["across_detrended_rms_m"] < 0.01, res
    # heading on both sides of +-pi (EW car) is 0.5 deg apart, not 359.5; 1 m/s slower is 2.237 mph
    v2 = np.c_[np.full_like(t, -26.0), np.full_like(t, 26.0 * math.tan(math.radians(0.5)))]
    res, _ = compare(g, v2, g, np.full_like(t, -math.pi + 1e-6), vg)
    assert abs(res["heading_mae_deg"] - 0.5) < 0.01 and abs(res["speed_mae_mph"] - 2.237) < 0.01, res
    # same for a WE car heading near 0
    res, _ = compare(g, -v2 * [1, -1], g, np.zeros_like(t), vg)
    assert abs(res["heading_mae_deg"] - 0.5) < 0.01, res
    print("selfcheck ok (offset removed, noise recovered, direction ramp split out, "
          "heading wraps for both directions, mph factor)")


def main():
    from gps_to_image import load_site
    ap = argparse.ArgumentParser("Relative trajectory comparison against GPS")
    ap.add_argument("--tags", nargs="+", default=["ft102"])
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    a = ap.parse_args()
    T = load_site()
    rep = ROOT / "outputs/reports/relative_trajectory"
    rep.mkdir(parents=True, exist_ok=True)
    allres = {}
    for clip in a.clips:
        data = {tag: load(clip, tag, T) for tag in a.tags}
        data = {k: v for k, v in data.items() if v is not None}
        if not data:
            continue
        common = sorted(set.intersection(*(set(d["frame"].tolist()) for d in data.values())))
        series = {}
        for tag, d in data.items():
            res, s = score(d, common)
            own, _ = score(d)
            allres[f"{clip}|{tag}"] = {"track_id": d["id"], "common": res, "own": own}
            series[tag] = s
            print(f"{clip:10s} {tag:12s} {res['frames']:3d} common frames (own {own['frames']})  "
                  f"along {res['along_rms_m']:.2f} m  across {res['across_rms_m']:.2f} m "
                  f"(slope {res['across_slope_deg']:+.2f} deg, detrended {res['across_detrended_rms_m']:.2f} m)  "
                  f"heading {res['heading_mae_deg']:.2f} deg  speed {res['speed_mae_mph']:.2f} mph  "
                  f"ratio {res['speed_ratio']:.4f}")
        if series:
            plot(clip, series, rep / f"{clip}.png")
    out = ROOT / "outputs/evaluation/relative_trajectory.json"
    out.write_text(json.dumps(allres, indent=2))
    print("->", out, "and", rep)


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
