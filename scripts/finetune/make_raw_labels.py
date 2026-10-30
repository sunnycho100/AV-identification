"""Baseline labels for the self-training ablation: raw detections, no track cleanup.

Same source and filters as pseudo_labels_v2 (the <clip>_phase1 detections, cars at
score >= 0.3, off-road boxes dropped with the same road mask), but every box keeps
the model's own position, heading, height and size. Fine-tuning on these vs on
pseudo_labels_v2 isolates what the track-based cleanup adds.

    python scripts/finetune/make_raw_labels.py --clip HV_T_EW_1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/object_detection"))
import road_mask as rm  # noqa: E402


def main():
    ap = argparse.ArgumentParser("Raw-detection labels (baseline)")
    ap.add_argument("--clip", required=True)
    # standard self-training baseline: --min-score 0.6 --name std_labels (confident pseudo-labels only)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--name", default="raw_labels")
    a = ap.parse_args()
    det_dir = ROOT / f"outputs/object_detection/camera-data/{a.clip}_phase1"
    cal = json.loads((det_dir / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    z = cal.get("road_plane_z_in_output", -1.73)
    road = rm.load_built(a.clip)
    dst = ROOT / "outputs/finetune" / a.name / a.clip
    dst.mkdir(parents=True, exist_ok=True)
    n_in = n_off = n_out = 0
    for pf in sorted(det_dir.glob("*_pred.json")):
        out = []
        for o in json.loads(pf.read_text()):
            if o["class_name"] != "car" or o["score"] < a.min_score:
                continue
            n_in += 1
            if not rm.keep(o, None, K, M, z, road):
                n_off += 1
                continue
            out.append({**o, "track_id": None, "yaw_source": "model", "z_source": "model"})
        n_out += len(out)
        if out:
            (dst / f"{pf.stem.split('_')[0]}_label.json").write_text(json.dumps(out, indent=2))
    manifest = {"clip": a.clip, "labels_in": n_in, "dropped_off_road": n_off, "labels_out": n_out,
                "min_score": a.min_score,
                "provenance": f"phase1 detections as detected (score >= {a.min_score}, cars, on road). No tracking, "
                              "no heading, height, size or position correction. GPS not used."}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
