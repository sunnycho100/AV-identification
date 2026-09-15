"""Run one orientation candidate on the held-out clips and grade it.

Why. Every candidate in the orientation research loop has to be scored the same
way against the same clips, or the ledger stops comparing like with like. This
runner owns that: it finds the held-out clips that are on disk, computes the
baseline fresh on exactly that clip set, runs the candidate, scores it with
scripts/evaluation/score_heading.py, applies the frozen keep rule, and appends
one row to outputs/orientation/ledger.csv.

A candidate is a module in candidates/ exposing

    run(clip, det_dir, tracks, cfg) -> {track_id: {frame: yaw}}

Frame keys may come back as ints or strings; score_heading takes both.

GPS is held out from candidates, not only from the score. Two guards: the
runner grep-refuses a candidate file that mentions "Camera data" or
"trajectory.csv", and it exports ORIENTATION_NO_GPS=1 so a candidate that does
its own file access can see the rule.

    .venv/bin/python scripts/orientation/run_candidate.py identity
    .venv/bin/python scripts/orientation/run_candidate.py temporal_median --cfg window=15
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import score_heading as sh

CLIPS = ("AV_T_EW_3", "HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3", "HV_T_EW_2",
         "AV_V_WE_3", "AV_W_WE_1", "AV_W_WE_3")
CANDIDATE_DIR = Path(__file__).resolve().parent / "candidates"
FORBIDDEN = ("Camera data", "trajectory.csv")
LEDGER_COLUMNS = ["date", "candidate", "cfg", "cfg_hash", "clips",
                  "mean_median_folded_deg", "mean_frac_raw_gt90",
                  "per_clip_median_folded", "baseline_mean_median_folded_deg",
                  "kept", "reason", "git_commit"]


def collect_clips(root=ROOT, clips=CLIPS):
    """[(clip, tracks_path, det_dir)] for the clips whose tracks.json exists."""
    found = []
    for clip in clips:
        tracks = Path(root) / "outputs/tracking/camera-data" / f"{clip}_phase1" / "tracks.json"
        det = Path(root) / "outputs/object_detection/camera-data" / f"{clip}_phase1"
        if tracks.exists():
            found.append((clip, tracks, det))
    return found


def check_no_gps(path):
    """Refuse a candidate whose source mentions the held-out GPS folder or files."""
    text = Path(path).read_text()
    hit = [w for w in FORBIDDEN if w in text]
    if hit:
        raise SystemExit(f"{path} mentions {hit}: candidates may not read GPS")


def load_candidate(name):
    """Import candidates/<name>.py after the GPS grep guard passes."""
    path = CANDIDATE_DIR / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"no candidate at {path}")
    check_no_gps(path)
    os.environ["ORIENTATION_NO_GPS"] = "1"
    spec = importlib.util.spec_from_file_location(f"candidate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise SystemExit(f"{path} has no run(clip, det_dir, tracks, cfg)")
    return module


def score(found, yaws=None):
    """Score the clip set, optionally with {clip: {track_id: {frame: yaw}}}."""
    yaws = yaws or {}
    return sh.score_clips([(clip, sh.load_tracks(t), yaws.get(clip), d)
                           for clip, t, d in found])


def run_all(module, found, cfg):
    """The candidate's yaws for every clip, as {clip: {track_id: {frame: yaw}}}."""
    return {clip: module.run(clip, det, sh.load_tracks(t), cfg)
            for clip, t, det in found}


def parse_cfg(pairs):
    """key=value pairs into a dict, numbers parsed as int then float."""
    cfg = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        cfg[key] = value
    return cfg


def cfg_hash(cfg):
    """Short stable hash of a config, empty string for the empty config."""
    if not cfg:
        return ""
    blob = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def append_ledger(path, row):
    """One row per run, header written the first time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_table(base, cand, kept, reason):
    print(f"{'clip':<12}{'baseline':>10}{'candidate':>11}{'delta':>9}")
    for clip, b in base["per_clip"].items():
        c = cand["per_clip"][clip]
        if b["median_folded_deg"] is None or c["median_folded_deg"] is None:
            print(f"{clip:<12}{'no frames':>10}")
            continue
        print(f"{clip:<12}{b['median_folded_deg']:>10.3f}{c['median_folded_deg']:>11.3f}"
              f"{c['median_folded_deg'] - b['median_folded_deg']:>+9.3f}")
    print(f"{'mean':<12}{base['mean_median_folded_deg']:>10.3f}"
          f"{cand['mean_median_folded_deg']:>11.3f}"
          f"{cand['mean_median_folded_deg'] - base['mean_median_folded_deg']:>+9.3f}")
    print(f"{'flip frac':<12}{base['mean_frac_raw_gt90']:>10.4f}"
          f"{cand['mean_frac_raw_gt90']:>11.4f}"
          f"{cand['mean_frac_raw_gt90'] - base['mean_frac_raw_gt90']:>+9.4f}")
    print(f"{'KEPT' if kept else 'REJECTED'}: {reason}")


def main():
    ap = argparse.ArgumentParser("Run and grade one orientation candidate")
    ap.add_argument("name", help="module name under scripts/orientation/candidates/")
    ap.add_argument("--cfg", nargs="*", default=[], help="key=value pairs for the candidate")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    args = ap.parse_args()

    cfg = parse_cfg(args.cfg)
    found = collect_clips(clips=args.clips)
    if not found:
        raise SystemExit("no held-out clip has a tracks.json on disk")
    missing = [c for c in args.clips if c not in {f[0] for f in found}]
    if missing:
        print(f"skipped (no tracks.json): {', '.join(missing)}")
    print(f"clips: {', '.join(c for c, _, _ in found)}")

    module = load_candidate(args.name)
    base = score(found)
    yaws = run_all(module, found, cfg)
    cand = score(found, yaws)
    kept, reason = sh.keep_decision(base, cand)

    h = cfg_hash(cfg)
    out_dir = ROOT / "outputs/orientation/runs" / (f"{args.name}_{h}" if h else args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "yaws.json").write_text(json.dumps(yaws))
    (out_dir / "score.json").write_text(json.dumps(cand, indent=2))

    per_clip = ";".join(f"{c}={cand['per_clip'][c]['median_folded_deg']:.3f}"
                        for c, _, _ in found)
    append_ledger(ROOT / "outputs/orientation/ledger.csv", {
        "date": datetime.date.today().isoformat(),
        "candidate": args.name,
        "cfg": json.dumps(cfg, sort_keys=True),
        "cfg_hash": h,
        "clips": ";".join(c for c, _, _ in found),
        "mean_median_folded_deg": f"{cand['mean_median_folded_deg']:.4f}",
        "mean_frac_raw_gt90": f"{cand['mean_frac_raw_gt90']:.4f}",
        "per_clip_median_folded": per_clip,
        "baseline_mean_median_folded_deg": f"{base['mean_median_folded_deg']:.4f}",
        "kept": "yes" if kept else "no",
        "reason": reason,
        "git_commit": git_commit()})

    print_table(base, cand, kept, reason)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
