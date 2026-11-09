"""Evaluation-only guard: candidate yaw against GPS heading on the instrumented car.

Why. The frozen score (scripts/evaluation/score_heading.py) grades every yaw
against the track's own direction of motion, because there is no orientation
ground truth on these clips and the score has to stay usable where there is no
GPS at all. That reference is a proxy. This file is the one place the proxy is
checked: on the clips where the instrumented vehicle is in frame, its own GPS
heading is read and the same candidates are graded against it.

This is a guard, not a second score. It never feeds a candidate, a calibration
or a keep decision; per the design spec's operating rules GPS is re-checked only
for candidates the frozen score already kept. Nothing here is tuned.

Which frames. Only the target track, the instrumented vehicle, identified from
the image alone (hood marker template match) in target_track.json. The frame set
is exactly the frozen score's: not coasted, motion heading defined, a raw
detection within 1 m, and here additionally inside the GPS time window. All
candidates are graded on that one frame set, so the columns compare.

Frames. GPS heading is read at frame/30 s via site_error_model.load_gps and
gps_at; load_gps already negates processed_heading_y (the SOUTH component) so
psi is ENU, counter-clockwise from east.

Bearing. psi lives in site ENU, the yaws live in the camera ground frame, so one
rotation per clip connects them:

    psi_camera = psi + beta

beta is read from outputs/orientation/gps_bearing.json when it is there,
otherwise estimated as the circular mean over the track of (motion heading in
the camera frame minus psi) and written to that file. It is a single constant
offset per clip, not a fit of orientation: it cannot absorb any per-frame error,
and the motion-vs-GPS column below is what shows whether it landed. Note the
sign convention is the inverse of site_error_model's beta, which rotates the
camera frame into ENU.

    .venv/bin/python scripts/evaluation/gps_heading_guard.py
    .venv/bin/python scripts/evaluation/gps_heading_guard.py --run 'temporal_median:window=5'
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import score_heading as sh
from site_error_model import gps_at, load_gps

FPS = 30.0
BEARING_JSON = ROOT / "outputs/orientation/gps_bearing.json"
LEDGER = ROOT / "outputs/orientation/ledger.csv"
# the candidates the task asks for: the raw detection yaw, plus the two kept runs
DEFAULT_RUNS = ("baseline", "temporal_median:window=15",
                "yaw_track_axis_consensus:window=31,score_pow=2")


def circmean(a):
    a = np.asarray(a, float)
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def run_dir(spec):
    """'name:k=v,k=v' to outputs/orientation/runs/<name>_<cfg_hash>, via the ledger.

    The cfg hash is over the candidate's whole config, so the run is found by
    matching the ledger's cfg column instead of re-hashing a partial config.
    Several runs can match a partial config (window=31,score_pow=2 matches the
    range_pow variants too), so the fewest-extra-keys row wins and the chosen
    directory is printed.
    """
    name, _, cfg = spec.partition(":")
    want = dict(p.split("=", 1) for p in cfg.split(",") if p)
    hits = []
    for row in csv.DictReader(open(LEDGER)):
        if row["candidate"] != name:
            continue
        have = {k: str(v) for k, v in json.loads(row["cfg"]).items()}
        if all(have.get(k) == v for k, v in want.items()):
            d = ROOT / "outputs/orientation/runs" / (f"{name}_{row['cfg_hash']}"
                                                     if row["cfg_hash"] else name)
            if (d / "yaws.json").exists():
                hits.append((len(have), d))
    if not hits:
        raise SystemExit(f"no ledger row with a yaws.json for {spec!r}")
    return min(hits)[1]


def clip_frames(clip):
    """One clip's target track: states, the graded frame mask, motion heading, psi."""
    trk_dir = ROOT / "outputs/tracking/camera-data" / f"{clip}_phase1"
    d = json.loads((trk_dir / "target_track.json").read_text())
    track = sorted(d["states"], key=lambda s: s["frame"])
    det_dir = ROOT / "outputs/object_detection/camera-data" / f"{clip}_phase1"

    motion = sh.motion_heading(track)
    ydet = sh.detection_yaw(track, det_dir)
    gps = load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv", 0.0, 0.0)
    t = np.array([s["frame"] for s in track], float) / FPS
    _, psi, _ = gps_at(gps, t)
    # np.interp clamps outside the CSV window, which would invent a heading
    inside = (t >= gps["t"].min()) & (t <= gps["t"].max())
    keep = ~sh.coasted(track) & ~np.isnan(motion) & ~np.isnan(ydet) & inside
    return {"clip": clip, "track_id": str(d["target_track_id"]), "track": track,
            "keep": keep, "motion": motion, "psi": psi, "ydet": ydet,
            "rng": np.array([s["x"] for s in track], float)}


def bearing(c, store):
    """psi_camera - psi for this clip, from the store or estimated and stored."""
    if c["clip"] in store:
        return float(store[c["clip"]]["beta_rad"]), "file"
    k = c["keep"]
    beta = circmean(sh.wrap(c["motion"][k] - c["psi"][k]))
    store[c["clip"]] = {"beta_rad": beta, "beta_deg": float(np.degrees(beta)),
                        "n_frames": int(k.sum()),
                        "from": "circular mean of motion heading minus GPS psi "
                                "over the target track; psi_camera = psi + beta"}
    return beta, "estimated"


def grade(yaw, reference, rng):
    """score_heading's statistics, overall and per range bin, for one yaw series."""
    raw = sh.wrap(yaw - reference)
    out = sh.stats(raw)
    out["per_bin"] = {name: sh.stats(raw[(rng >= lo) & (rng < hi)])
                      for name, lo, hi in sh.BINS}
    return out


def candidate_yaw(c, spec, source):
    """The candidate's yaw on the target track: detection yaw, overridden by the run."""
    yaw = c["ydet"].copy()
    if source is None:
        return yaw
    yaws = json.loads((source / "yaws.json").read_text())
    per_frame = yaws.get(c["clip"], {}).get(c["track_id"], {})
    if not per_frame:
        raise SystemExit(f"{spec}: no yaws for track {c['track_id']} of {c['clip']}")
    return np.array([float(per_frame.get(str(s["frame"]), y))
                     for s, y in zip(c["track"], yaw)], float)


def report(clips, specs):
    store = json.loads(BEARING_JSON.read_text()) if BEARING_JSON.exists() else {}
    sources = {s: None if s == "baseline" else run_dir(s) for s in specs}
    out = {"sources": {s: "raw detection yaw" if d is None else str(d.relative_to(ROOT))
                       for s, d in sources.items()}, "clips": {}}
    for clip in clips:
        c = clip_frames(clip)
        beta, src = bearing(c, store)
        k = c["keep"]
        psi_cam, rng = c["psi"][k] + beta, c["rng"][k]
        r = {"track_id": c["track_id"], "beta_deg": float(np.degrees(beta)),
             "beta_source": src, "n": int(k.sum()),
             "motion_vs_gps": grade(c["motion"][k], psi_cam, rng),
             "candidates": {}}
        for spec in specs:
            yaw = candidate_yaw(c, spec, sources[spec])[k]
            r["candidates"][spec] = {"vs_gps": grade(yaw, psi_cam, rng),
                                     "vs_motion": grade(yaw, c["motion"][k], rng)}
        out["clips"][clip] = r
    BEARING_JSON.parent.mkdir(parents=True, exist_ok=True)
    BEARING_JSON.write_text(json.dumps(store, indent=2))
    return out


def fmt(s):
    return "  n/a  " if s["median_folded_deg"] is None else f"{s['median_folded_deg']:6.2f}"


def print_report(rep):
    for spec, src in rep["sources"].items():
        print(f"{spec:<44} {src}")
    for clip, r in rep["clips"].items():
        m = r["motion_vs_gps"]
        print(f"\n{clip}  target track {r['track_id']}  n={r['n']}  "
              f"beta {r['beta_deg']:+.2f} deg ({r['beta_source']})")
        print(f"  motion reference vs GPS: median folded {m['median_folded_deg']:.2f} deg, "
              f"flip frac {m['frac_raw_gt90']:.3f}   per bin "
              + "  ".join(f"{b}={fmt(m['per_bin'][b])}" for b, _, _ in sh.BINS))
        print(f"  {'candidate':<44}{'n':>5}{'medfold_gps':>13}{'flip_gps':>10}"
              f"{'medfold_motion':>16}   " + "".join(f"{b:>9}" for b, _, _ in sh.BINS))
        for spec, s in r["candidates"].items():
            g, v = s["vs_gps"], s["vs_motion"]
            print(f"  {spec:<44}{g['n']:>5}{g['median_folded_deg']:>13.2f}"
                  f"{g['frac_raw_gt90']:>10.3f}{v['median_folded_deg']:>16.2f}   "
                  + "".join(f"{fmt(g['per_bin'][b]):>9}" for b, _, _ in sh.BINS))


def _self_check():
    """A known constant yaw offset over a synthetic clip must come back unchanged."""
    assert abs(circmean([np.pi - 0.1, -np.pi + 0.1])) > 3.0      # wraps, not averages to 0
    psi = np.full(40, 2.9)                                        # ENU heading, constant
    motion = sh.wrap(psi + 1.5)                                   # camera frame, beta = 1.5
    beta = circmean(sh.wrap(motion - psi))
    assert abs(sh.wrap(beta - 1.5)) < 1e-9, beta
    rng = np.linspace(10.0, 70.0, 40)
    g = grade(sh.wrap(motion + np.radians(8.0)), sh.wrap(psi + beta), rng)
    assert abs(g["median_folded_deg"] - 8.0) < 1e-6, g
    assert g["frac_raw_gt90"] == 0.0 and g["n"] == 40
    # a nose-to-tail flip is zero folded error and every frame flipped
    f = grade(sh.wrap(motion + np.pi), sh.wrap(psi + beta), rng)
    assert f["median_folded_deg"] < 1e-6 and f["frac_raw_gt90"] == 1.0
    assert sum(b["n"] for b in g["per_bin"].values()) == 40


def main():
    ap = argparse.ArgumentParser("Candidate yaw vs GPS heading on the instrumented car")
    ap.add_argument("--clips", nargs="*", help="default: every clip with a target_track.json "
                                               "and a GPS csv")
    ap.add_argument("--run", nargs="*", default=list(DEFAULT_RUNS),
                    help="'baseline' or 'candidate:key=value,...' matched against the ledger")
    ap.add_argument("--out")
    args = ap.parse_args()
    _self_check()
    clips = args.clips or sorted(
        p.parent.name[: -len("_phase1")]
        for p in (ROOT / "outputs/tracking/camera-data").glob("*_phase1/target_track.json")
        if (ROOT / "Camera data" / f"{p.parent.name[:-len('_phase1')]}_trajectory.csv").exists())
    if not clips:
        raise SystemExit("no clip has both a target_track.json and a GPS csv")
    rep = report(clips, args.run)
    print_report(rep)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
