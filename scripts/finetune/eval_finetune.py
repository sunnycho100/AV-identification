"""Did fine-tuning actually help? Measured on the held-out clip only.

Loss falling is not evidence: the earlier MPS collapse showed loss dropping to
0.0001 while the model lost every detection. So this compares the two checkpoints
on the clip that was never trained on, using the quantities we care about:

  heading   offset from the road axis, and the >45 deg tail that shows up as the
            visibly crooked boxes
  recall    car detections per frame, to catch a model that "improved" its
            heading by simply detecting fewer, easier cars
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from scripts.object_detection.run_bevheight_single import (
    CKPT_PATH, build_model, filter_and_pack, load_checkpoint)
from scripts.finetune.train_finetune import clip_paths, load_sample, to_device, frames_for


def yaw_off(y):
    """Offset from the road axis, folded to 0-90 deg (x runs along the road)."""
    return np.degrees(abs(np.arctan2(np.sin(2 * y), np.cos(2 * y)))) / 2


def run(model, clip, frames, K, l2c):
    offs, per_frame = [], []
    for f in frames:
        img, mats, _, _ = load_sample(clip, f, K, l2c)
        i2, m2 = to_device(img, mats, "cpu")
        with torch.no_grad():
            r = model.get_bboxes(model(i2, m2), [{"box_type_3d": LiDARInstance3DBoxes}])
        dets = filter_and_pack(r[0][0].tensor.numpy(), r[0][1].numpy(), r[0][2].numpy())
        cars = [d for d in dets if d["class_name"] == "car"]
        per_frame.append(len(cars))
        offs += [yaw_off(d["yaw"]) for d in cars]
    o = np.array(offs)
    return {"cars_total": int(sum(per_frame)),
            "cars_per_frame": float(np.mean(per_frame)),
            "yaw_median_deg": float(np.median(o)) if len(o) else float("nan"),
            "yaw_within_15_pct": float((o < 15).mean()) if len(o) else 0.0,
            "yaw_over_45_pct": float((o > 45).mean()) if len(o) else 0.0}


def main():
    ap = argparse.ArgumentParser("Evaluate a fine-tuned head on the held-out clip")
    ap.add_argument("--clip", default="AV_T_EW_3")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    K, l2c = clip_paths(args.clip)
    frames = frames_for(args.clip)[::args.every]
    print(f"held-out clip {args.clip}: {len(frames)} frames")

    rows = {}
    for tag, ck in [("pretrained", None), ("fine-tuned", args.ckpt)]:
        m = build_model(); load_checkpoint(m, CKPT_PATH)
        if ck:
            m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
        m.eval()
        rows[tag] = run(m, args.clip, frames, K, l2c)

    print(f"{'':<12} {'cars/frame':>11} {'yaw median':>11} {'within 15d':>11} {'>45d tail':>10}")
    for tag, r in rows.items():
        print(f"{tag:<12} {r['cars_per_frame']:>11.2f} {r['yaw_median_deg']:>10.1f}d "
              f"{r['yaw_within_15_pct']:>10.1%} {r['yaw_over_45_pct']:>9.1%}")
    a, b = rows["pretrained"], rows["fine-tuned"]
    print(f"change: cars/frame {b['cars_per_frame']-a['cars_per_frame']:+.2f}, "
          f"yaw median {b['yaw_median_deg']-a['yaw_median_deg']:+.1f} deg, "
          f">45deg tail {b['yaw_over_45_pct']-a['yaw_over_45_pct']:+.1%}")
    out = Path(args.ckpt).parent / f"eval_{args.clip}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
