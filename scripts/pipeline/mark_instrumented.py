"""Mark which extracted vehicle is the instrumented car, from the image only.

Two sources, never GPS:
  - hood marker (EW clips, car facing the camera): identify_target_by_marker.py
    votes for the track whose projected box contains the template-matched
    marker most often (outputs/target_id/<clip>/marker_matches.json);
  - reference track (WE clips, marker not visible): the track picked by hand
    in an earlier run (compare_height_runs.REFERENCE) is projected into the
    image, and the new vehicle that sits within MATCH_PX of it most often wins.
Writes outputs/trajectories/<clip>_<tag>/instrumented.json.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/pipeline/mark_instrumented.py --clip AV_T_EW_3
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import compare_height_runs as ch   # noqa: E402


def project(K, M, states):
    q = (M[:3, :3] @ np.array([[s["x"], s["y"], s.get("z", -1.73)] for s in states]).T).T + M[:3, 3]
    uv = (K @ q.T).T
    return {s["frame"]: p for s, p in zip(states, uv[:, :2] / uv[:, 2:])}


def by_reference(clip, out_dir, det_dir):
    ref_run, ref_tid, _ = ch.REFERENCE[clip]
    ref = ch.to_px(clip, ref_run, ch.load_track(clip, ref_run, ref_tid))
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    tracks = json.loads((out_dir / "tracks.json").read_text())["tracks"]
    votes = {tid: sum(1 for f, p in project(K, M, t).items()
                      if f in ref and np.linalg.norm(p - ref[f]) < ch.MATCH_PX)
             for tid, t in tracks.items()}
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    return {"vehicle_id": ranked[0][0], "method": f"reference track {ref_run}/{ref_tid}, pixel match",
            "votes": ranked[0][1], "reference_frames": len(ref),
            "runner_up_votes": ranked[1][1] if len(ranked) > 1 else 0}


def by_marker(clip, out_dir, det_dir):
    tmp = out_dir / "instrumented_marker.json"
    r = subprocess.run([sys.executable, "scripts/tracking/identify_target_by_marker.py", "--clip", clip,
                        "--tracks", out_dir / "tracks.json", "--det-dir", det_dir, "--out", tmp],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    d = json.loads(tmp.read_text())
    tmp.unlink()
    warn = [l for l in r.stdout.splitlines() if l.startswith("WARN")]
    return {"vehicle_id": str(d["target_track_id"]), "method": "hood marker template match",
            "votes": d["votes"], "marker_frames": d["marker_frames"],
            "runner_up_votes": d["runner_up_votes"], "warnings": warn}


def main():
    ap = argparse.ArgumentParser("Mark the instrumented vehicle")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tag", default="ft102")
    a = ap.parse_args()
    out_dir = ROOT / "outputs/trajectories" / f"{a.clip}_{a.tag}"
    det_dir = ROOT / "outputs/object_detection/camera-data" / f"{a.clip}_{a.tag}_road"
    if (ROOT / "outputs/target_id" / a.clip / "marker_matches.json").exists():
        res = by_marker(a.clip, out_dir, det_dir)
    elif a.clip in ch.REFERENCE:
        res = by_reference(a.clip, out_dir, det_dir)
    else:
        raise SystemExit(f"{a.clip}: no marker matches and no reference track")
    res["clip"], res["identified_from"] = a.clip, "image only, GPS not used"
    (out_dir / "instrumented.json").write_text(json.dumps(res, indent=2))
    print(f"{a.clip}: instrumented vehicle = {res['vehicle_id']} ({res['method']}, votes {res['votes']}, "
          f"runner-up {res['runner_up_votes']})")


if __name__ == "__main__":
    main()
