"""Recompute BatchNorm running statistics on our own footage. No gradient steps.

Candidate 3 `bn_stats_recalib` in docs/papers/orientation-research-2026-09.md.
The free half of test-time adaptation (MonoTTA arXiv:2405.19682, STMono3D
arXiv:2204.11590): every learnable weight stays exactly as the DAIR checkpoint
shipped it, and only the BatchNorm running_mean / running_var buffers are
re-estimated by forward passes over our frames.

The layer this is aimed at is `HeightNet.bn`, a BatchNorm1d(27) on the camera
parameter vector. It was fitted on DAIR focal lengths 2183 and 2758 and is being
fed 1493 to 1619, so its modulation of the height and context branches runs
several standard deviations outside its trained range. The backbone and neck BNs
are re-estimated at the same time because they are the same free operation.

Two mechanics matter:

  * momentum=None makes PyTorch accumulate a cumulative average over all batches
    rather than an exponential one, so the result does not depend on frame order.
  * the 27-dim BatchNorm1d refuses a batch of 1 in train mode ("Expected more
    than 1 value per channel"), so frames are pushed through in pairs. The
    camera vector is constant within a clip, so a calibration set must span more
    than one clip or that layer's variance collapses to zero.

    /Users/sunghwan_cho/miniforge/bin/python scripts/orientation/bn_recalib.py \
        --frames data/camera-data/AV_T_EW_3/frames_all \
                 data/camera-data/AV_T_WE_1/frames_all \
        --n 60 --out checkpoints/BEVHeight_R50_128_102.4_bnrecal.ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.adapter.calib_to_bevheight_input import build_mats_dict
from scripts.object_detection.run_bevheight_single import (
    CKPT_PATH, build_model, final_dim, img_conf, load_checkpoint)

BATCH = 2  # BatchNorm1d(27) rejects a batch of 1 in train mode
DEFAULT_OUT = ROOT / "checkpoints/BEVHeight_R50_128_102.4_bnrecal.ckpt"


def clip_calibration(clip, root=ROOT):
    """The K and lidar2cam the phase1 detection run used, ground shift included."""
    p = Path(root) / "outputs/object_detection/camera-data" / f"{clip}_phase1" / "calibration_used.json"
    d = json.loads(p.read_text())
    return np.array(d["K"], float), np.array(d["lidar2cam"], float)


def pick_frames(dirs, n):
    """[(path, clip)] for n frames spread evenly over the given frame directories."""
    dirs = [Path(d) for d in dirs]
    per_dir = max(1, n // len(dirs))
    out = []
    for d in dirs:
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        if not files:
            raise SystemExit(f"no frames in {d}")
        stride = max(1, len(files) // per_dir)
        out += [(f, d.parent.name) for f in files[::stride][:per_dir]]
    return out[:n]


def prepare_bn(model):
    """Every BatchNorm into train mode with reset stats and a cumulative average."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    return bns


def batch_inputs(frames, calib):
    """Stack per-frame (img, mats) into batches of BATCH. calib is clip -> (K, ext)."""
    built = []
    for path, clip in frames:
        K, ext = calib[clip]
        img, mats, _ = build_mats_dict(str(path), K, ext, final_dim, img_conf)
        built.append((img, mats))
    for i in range(0, len(built) - len(built) % BATCH, BATCH):
        chunk = built[i:i + BATCH]
        yield (torch.cat([c[0] for c in chunk], 0),
               {k: torch.cat([c[1][k] for c in chunk], 0) for k in chunk[0][1]})


def bn_keys(model):
    """State-dict keys of the BatchNorm buffers, the only tensors allowed to move."""
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            for buf in ("running_mean", "running_var", "num_batches_tracked"):
                keys.add(f"{name}.{buf}")
    return keys


def diff_keys(before, after):
    """State-dict keys whose tensor is not bit-identical between two snapshots."""
    return {k for k in before if not torch.equal(before[k], after[k])}


def main():
    ap = argparse.ArgumentParser("Recompute BatchNorm running stats on our frames")
    ap.add_argument("--frames", nargs="+", required=True,
                    help="frame directories, e.g. data/camera-data/<CLIP>/frames_all")
    ap.add_argument("--n", type=int, default=60, help="total frames to push through")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    frames = pick_frames(args.frames, args.n)
    clips = sorted({c for _, c in frames})
    calib = {c: clip_calibration(c) for c in clips}
    print(f"{len(frames)} frames from {', '.join(clips)}; "
          f"fx {', '.join(f'{calib[c][0][0, 0]:.1f}' for c in clips)}")

    model = build_model()
    info = load_checkpoint(model, CKPT_PATH)
    print(f"checkpoint {CKPT_PATH.name}: {info['matched']} keys matched, "
          f"{len(info['missing'])} missing")

    before = {k: v.clone() for k, v in model.state_dict().items()}
    bns = prepare_bn(model)
    print(f"{len(bns)} BatchNorm layers reset to a cumulative average")

    with torch.no_grad():
        for i, (img, mats) in enumerate(batch_inputs(frames, calib)):
            model(img, mats)
            print(f"batch {i + 1}: {(i + 1) * BATCH}/{len(frames)} frames", flush=True)
    model.eval()

    after = model.state_dict()
    moved = diff_keys(before, after)
    allowed = bn_keys(model)
    stray = moved - allowed
    if stray:
        raise SystemExit(f"non-BatchNorm tensors moved: {sorted(stray)[:5]}")
    print(f"{len(moved)} BatchNorm buffers updated, every parameter bit-identical")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": after}, out)
    (out.with_suffix(".json")).write_text(json.dumps({
        "source_checkpoint": CKPT_PATH.name,
        "frames": [str(f) for f, _ in frames],
        "clips": clips,
        "n_frames": len(frames),
        "batch": BATCH,
        "bn_layers": len(bns),
        "buffers_changed": sorted(moved),
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
