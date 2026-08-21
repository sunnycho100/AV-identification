"""Fine-tune BEVHeight's detection head on our own pseudo-labelled footage.

Only the head trains. The image branch is frozen, not as a research choice but
because mmcv 1.7's deformable conv has no working backward against torch 2.10
(`DeformConv2dFunctionBackward has no attribute bufs_`), and that DCN sits inside
the height branch. Inference never hit it because inference runs under no_grad.
Freezing everything upstream means autograd never invokes it, since it prunes
branches with no trainable parameter behind them. Consequence worth stating in
any writeup: yaw IS trainable here (the head regresses rotation), the depth warp
is NOT (that lives in the frozen height net) and needs a CUDA box.

The box column order is the trap in this file. The head's targets and the model's
outputs must agree, and the two are easy to swap. `--selfcheck` settles it
empirically instead of by reading: feed the model its OWN predictions back as
labels, and the loss must come out near zero. Wrong ordering makes it large.

    .venv/bin/python scripts/finetune/train_finetune.py --selfcheck
    .venv/bin/python scripts/finetune/train_finetune.py --epochs 6
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.adapter.calib_to_bevheight_input import (
    build_mats_dict, load_K_from_anycalib, load_extrinsic_json)
from scripts.object_detection.run_bevheight_generic import to_dair_ground
from scripts.object_detection.run_bevheight_single import (
    CKPT_PATH, build_model, final_dim, img_conf, load_checkpoint)

CAR_TASK = 0              # the head has six task groups; ours labels only cars
TRAIN_CLIPS = ["HV_T_EW_1", "AV_T_WE_1"]
HELD_OUT = ["AV_T_EW_3"]          # never trained on, so the result means something
CAR_CLASS_INDEX = 0


def clip_paths(clip):
    cal_dir = ROOT / "outputs/calibration/camera-data" / clip
    anycal = sorted(cal_dir.glob("*_anycalib_pinhole_pinhole.json"))[0]
    return (load_K_from_anycalib(anycal),
            to_dair_ground(load_extrinsic_json(cal_dir / "metric_extrinsic_site.json")))


def load_sample(clip, frame, K, l2c):
    img_p = ROOT / f"data/camera-data/{clip}/frames_all/{frame:03d}.jpg"
    img, mats, meta = build_mats_dict(str(img_p), K, l2c, final_dim, img_conf)
    lab_p = ROOT / f"outputs/finetune/pseudo_labels/{clip}/{frame:03d}_label.json"
    objs = json.loads(lab_p.read_text())
    # column order must match the model's own output space, see _selfcheck
    boxes = torch.tensor([[o["x"], o["y"], o["z"], o["w"], o["l"], o["h"],
                           o["yaw"], o.get("vx", 0.0), o.get("vy", 0.0)]
                          for o in objs], dtype=torch.float32)
    labels = torch.full((len(objs),), CAR_CLASS_INDEX, dtype=torch.long)
    return img, mats, boxes, labels


def frames_for(clip):
    d = ROOT / f"outputs/finetune/pseudo_labels/{clip}"
    return sorted(int(p.stem.split("_")[0]) for p in d.glob("*_label.json"))


def car_only_loss(model, targets, preds):
    """Loss on the car task alone.

    The head carries six task groups (car; truck/construction; bus/trailer;
    barrier; motorcycle/bicycle; pedestrian/traffic_cone). Our labels contain
    only cars, so the other five targets are entirely zero, which reads as
    "there are no pedestrians, bicycles or trucks anywhere in this scene". The
    model does predict those classes, and because the task heads share a trunk
    and neck, driving five of six outputs to zero collapses the shared features
    and takes car detection down with them: a smoke run lost every detection
    (max score 0.624 -> 0.171) in 24 steps while the loss fell to 0.003.
    Training the car task alone leaves the other classes untouched.
    """
    heatmaps, anno_boxes, inds, masks = targets
    sliced = ([heatmaps[CAR_TASK]], [anno_boxes[CAR_TASK]],
              [inds[CAR_TASK]], [masks[CAR_TASK]])
    return model.loss(sliced, [preds[CAR_TASK]])


def to_device(img, mats, dev):
    return img.to(dev), {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in mats.items()}


def freeze_image_branch(model):
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.eval()                      # BatchNorm stays in inference mode at bs=1
    return [p for p in model.parameters() if p.requires_grad]


def _selfcheck():
    """Feeding the model its own predictions back as labels must give ~0 loss.

    This pins the box column order, which no amount of reading the code settles:
    the dataset writes one order and the inference decoder reads another.
    """
    dev = "cpu"
    model = build_model(); load_checkpoint(model, CKPT_PATH); model.to(dev).eval()
    K, l2c = clip_paths("HV_T_EW_1")
    img, mats, _ = build_mats_dict(
        str(ROOT / "data/camera-data/HV_T_EW_1/frames_all/238.jpg"),
        K, l2c, final_dim, img_conf)
    img, mats = to_device(img, mats, dev)
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    with torch.no_grad():
        preds = model(img, mats)
        res = model.get_bboxes(preds, [{"box_type_3d": LiDARInstance3DBoxes}])
    raw, scores, labels = res[0][0].tensor, res[0][1], res[0][2]
    keep = (scores >= 0.3) & (labels == CAR_CLASS_INDEX)
    boxes = raw[keep][:, :9].clone().float()
    assert len(boxes) > 3, f"only {len(boxes)} car detections to test with"

    with torch.no_grad():
        targets = model.get_targets([boxes], [labels[keep]])
        preds2 = model(img, mats)
        matched = float(model.loss(targets, preds2))
        shuffled = boxes.clone()
        shuffled[:, [3, 4]] = shuffled[:, [4, 3]]        # swap the two dims
        loss_swapped = float(model.loss(model.get_targets([shuffled], [labels[keep]]),
                                        model(img, mats)))
    print(f"selfcheck: {len(boxes)} own detections as labels -> loss {matched:.3f}; "
          f"with length/width swapped -> {loss_swapped:.3f}")
    assert matched < loss_swapped, (
        "swapping dims did not increase the loss: the column order assumption "
        "in load_sample is not verified by this test")
    print("selfcheck ok (self-consistent ordering confirmed)")


def main():
    ap = argparse.ArgumentParser("Fine-tune the BEVHeight head on pseudo-labels")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    # CPU, deliberately. MPS computes this model WRONG: on identical data and
    # weights a single step gives loss 6.409 / grad-norm 291.3 on MPS against
    # 0.658 / 6.9 on CPU. Those inflated gradients destroyed the model in 24
    # steps (car detections 20 -> 0) while the loss appeared to fall. MPS is
    # roughly 2x faster and completely unusable here; do not switch it back
    # without re-running that comparison.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="frames per clip, for a quick run")
    ap.add_argument("--out", default="outputs/finetune/run1")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = args.device
    model = build_model(); load_checkpoint(model, CKPT_PATH); model.to(dev)
    trainable = freeze_image_branch(model)
    print(f"trainable: {len(trainable)} tensors, "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M params (head only)")

    cal = {c: clip_paths(c) for c in TRAIN_CLIPS + HELD_OUT}
    samples = [(c, f) for c in TRAIN_CLIPS
               for f in (frames_for(c)[:args.limit] if args.limit else frames_for(c))]
    val = [(c, f) for c in HELD_OUT
           for f in (frames_for(c)[:args.limit] if args.limit else frames_for(c))[::5]]
    print(f"train {len(samples)} frames from {TRAIN_CLIPS}; "
          f"val {len(val)} frames from {HELD_OUT} (held out)")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    history = []

    def evaluate():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for c, f in val:
                K, l2c = cal[c]
                img, mats, boxes, labels = load_sample(c, f, K, l2c)
                if len(boxes) == 0:
                    continue
                img, mats = to_device(img, mats, dev)
                t = model.get_targets([boxes.to(dev)], [labels.to(dev)])
                tot += float(car_only_loss(model, t, model(img, mats)))
        return tot / max(len(val), 1)

    v0 = evaluate()
    print(f"epoch 0 (before training): held-out loss {v0:.4f}")
    history.append({"epoch": 0, "train_loss": None, "val_loss": v0})

    for ep in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.eval(); freeze_image_branch(model)
        run, n, t0 = 0.0, 0, time.time()
        for i, (c, f) in enumerate(samples):
            K, l2c = cal[c]
            img, mats, boxes, labels = load_sample(c, f, K, l2c)
            if len(boxes) == 0:
                continue
            img, mats = to_device(img, mats, dev)
            targets = model.get_targets([boxes.to(dev)], [labels.to(dev)])
            loss = car_only_loss(model, targets, model(img, mats))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            opt.step()
            run += float(loss); n += 1
            if (i + 1) % 100 == 0:
                print(f"   ep{ep} {i+1}/{len(samples)} loss {run/max(n,1):.4f} "
                      f"({(time.time()-t0)/(i+1):.2f}s/step)")
        tr = run / max(n, 1)
        vl = evaluate()
        history.append({"epoch": ep, "train_loss": tr, "val_loss": vl})
        print(f"epoch {ep}: train {tr:.4f} | held-out {vl:.4f} "
              f"({(time.time()-t0)/60:.1f} min)")
        torch.save({"state_dict": model.state_dict(), "epoch": ep,
                    "val_loss": vl, "train_clips": TRAIN_CLIPS,
                    "held_out": HELD_OUT}, out / f"head_ft_ep{ep}.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    best = min(history[1:], key=lambda h: h["val_loss"])
    print(f"\nbest epoch {best['epoch']}: held-out {best['val_loss']:.4f} "
          f"(started {v0:.4f})")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
