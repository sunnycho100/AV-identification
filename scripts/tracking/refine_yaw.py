"""Post-tracking heading refinement: write the kept orientation fix onto tracks.json.

The orientation research loop kept two steps, in this order: a score-weighted
31-frame circular consensus of a track's own detection yaws, which fixes the
axis (`scripts/orientation/candidates/yaw_track_axis_consensus.py`), then one
travel-direction sign bit per track, which fixes front from back
(`candidates/sign_from_motion.py`). Both are imported from there rather than
reimplemented, so the pipeline and the ledger can never drift apart.

Three fields per state, and the tracker's own yaw is not one of them:

    yaw           untouched, the AB3DMOT Kalman heading
    yaw_det       the raw detection yaw matched within 1.0 m, or null
    yaw_refined   the consensus axis with the track's sign bit applied, or null

`yaw_refined` is null wherever no detection matched, since there was nothing for
the consensus to vote on. Keeping all three means a downstream classifier can
carry the appearance sign and the motion sign together and learn where they
disagree.

GPS is never read here, exactly as in the candidates: nothing under
`Camera data/` is opened, and ORIENTATION_NO_GPS is exported so anything further
down the import chain sees the same rule.

    .venv/bin/python scripts/tracking/refine_yaw.py --clip AV_T_EW_3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ["ORIENTATION_NO_GPS"] = "1"
sys.path.insert(0, str(ROOT / "scripts" / "orientation" / "candidates"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import score_heading as sh
import sign_from_motion
import yaw_track_axis_consensus as consensus


def refine(tracks_path, det_dir, clip=None):
    """Add yaw_det and yaw_refined to every state of tracks.json, in place."""
    tracks_path = Path(tracks_path)
    tracks = sh.load_tracks(tracks_path)        # also refuses a path under Camera data/
    doc = json.loads(tracks_path.read_text())
    refined = sign_from_motion.run(clip, str(det_dir), tracks, {})

    for tid, states in doc["tracks"].items():
        per_frame = refined.get(str(tid), {})
        ydet = sh.detection_yaw(tracks[str(tid)], str(det_dir))
        by_frame = {s["frame"]: y for s, y in zip(tracks[str(tid)], ydet)}
        for s in states:
            det = by_frame[s["frame"]]
            s["yaw_det"] = None if np.isnan(det) else float(det)
            new = per_frame.get(s["frame"], per_frame.get(str(s["frame"])))
            s["yaw_refined"] = None if new is None else float(new)

    doc.setdefault("meta", {})["yaw_refined"] = {
        "method": f"{consensus.__name__} then {sign_from_motion.__name__}",
        "window": sign_from_motion.WINDOW,
        "score_pow": sign_from_motion.SCORE_POW,
        "sign": "one bit per track from travel direction"}
    tracks_path.write_text(json.dumps(doc, indent=2))
    return doc


def main():
    ap = argparse.ArgumentParser("Refine tracked headings in place")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--run", default="phase1", help="run suffix, e.g. phase1")
    args = ap.parse_args()

    tracks_path = (ROOT / "outputs/tracking/camera-data" /
                   f"{args.clip}_{args.run}" / "tracks.json")
    if not tracks_path.exists():
        sys.exit(f"no tracks at {tracks_path}")
    det_dir = sh.default_det_dir(tracks_path)
    doc = refine(tracks_path, det_dir, args.clip)

    n = sum(len(v) for v in doc["tracks"].values())
    got = sum(1 for v in doc["tracks"].values() for s in v if s["yaw_refined"] is not None)
    print(f"{args.clip}_{args.run}: {len(doc['tracks'])} tracks, {n} states, "
          f"{got} refined ({n - got} with no matched detection)")
    print(f"wrote {tracks_path}")


if __name__ == "__main__":
    main()
