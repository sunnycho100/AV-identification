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

Masking variants (2026-10, Hang and Bofeng), all from the same base checkpoint:
  --frames-root DIR   train on other frames, e.g. input-masked images
  --feat-mask         drop off-road feature cells before the BEV projection,
                      using outputs/calibration/camera-data/<clip>/feature_mask.npy
  --mode gps_only     labels from make_gps_labels.py. Only the GPS car supervises,
                      and only inside a window around it; every other vehicle
                      (found or missed by the detector) is outside the loss.
                      The GPS car teaches presence, heading, velocity and the road
                      height; its in-cell offset and its size are not trained,
                      because GPS position carries a systematic error and the
                      size is a guess.
  --mode hybrid       gps_only plus the cleaned tracks of the other cars as
                      labels at HYBRID_WEIGHT; untracked detections stay out.
  --no-val            skip the held-out loss (GPS modes never look at AV_T_EW_3)
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
LABELS = "pseudo_labels"          # label set under outputs/finetune/, set by --labels
EXTRINSIC = "metric_extrinsic_site.json"   # the frame the labels are in, set by --extrinsic
FRAMES = "data/camera-data/{clip}/frames_all"
FEAT_MASK = False
GPS_HALF_WIN_M = (6.0, 3.0)       # along, across HALF-extents: a 12 m x 6 m loss window round the GPS car
CAR_HALF_M = (3.0, 1.5)           # half-extents of a known other car's cells, kept out of the GPS window
GPS_CODE = [0, 0, 1, 0, 0, 0, 1, 1, 0.5, 0.5]   # dx dy z | w l h | sin cos | vx vy (vel as in code_weights)
HYBRID_WEIGHT = 0.5


def clip_paths(clip):
    cal_dir = ROOT / "outputs/calibration/camera-data" / clip
    anycal = sorted(cal_dir.glob("*_anycalib_pinhole_pinhole.json"))[0]
    return (load_K_from_anycalib(anycal),
            to_dair_ground(load_extrinsic_json(cal_dir / EXTRINSIC)))


def load_sample(clip, frame, K, l2c, roles=False):
    img_p = ROOT / FRAMES.format(clip=clip) / f"{frame:03d}.jpg"
    img, mats, meta = build_mats_dict(str(img_p), K, l2c, final_dim, img_conf)
    if FEAT_MASK:
        fm = np.load(ROOT / "outputs/calibration/camera-data" / clip / "feature_mask.npy")
        mats["feat_mask"] = torch.as_tensor(fm, dtype=torch.float32)[None, None]
    lab_p = ROOT / f"outputs/finetune/{LABELS}/{clip}/{frame:03d}_label.json"
    objs = json.loads(lab_p.read_text())
    if roles:
        return img, mats, {r: to_boxes([o for o in objs if o.get("role") == r])
                           for r in ("gps", "other", "ignore")}
    # column order must match the model's own output space, see _selfcheck.
    # Labels store the box bottom (what get_bboxes outputs), but the head regresses
    # the box centre and get_bboxes subtracts h/2 on decode, so add it back here.
    boxes = to_boxes(objs)
    labels = torch.full((len(objs),), CAR_CLASS_INDEX, dtype=torch.long)
    return img, mats, boxes, labels


def to_boxes(objs):
    return torch.tensor([[o["x"], o["y"], o["z"] + o["h"] / 2, o["w"], o["l"], o["h"],
                          o["yaw"], o.get("vx", 0.0), o.get("vy", 0.0)]
                         for o in objs], dtype=torch.float32).reshape(-1, 9)


def cell_window(head, boxes, half_m):
    """bool (fH, fW): BEV cells within half_m = (along, across) of any box centre."""
    cfg = head.train_cfg
    cell = cfg["voxel_size"][0] * cfg["out_size_factor"]
    fW = cfg["grid_size"][0] // cfg["out_size_factor"]
    fH = cfg["grid_size"][1] // cfg["out_size_factor"]
    xs = cfg["point_cloud_range"][0] + (torch.arange(fW, device=boxes.device) + 0.5) * cell
    ys = cfg["point_cloud_range"][1] + (torch.arange(fH, device=boxes.device) + 0.5) * cell
    win = torch.zeros(fH, fW, dtype=torch.bool, device=boxes.device)
    for b in boxes:
        win |= ((ys[:, None] - b[1]).abs() <= half_m[1]) & ((xs[None, :] - b[0]).abs() <= half_m[0])
    return win


def gps_loss(model, preds, boxes, mode):
    """Car-task loss with the GPS car as the only full-weight label.

    Heatmap: focal loss only where we know the truth. gps_only: the window round
    the GPS car (its peak and the empty road next to it), minus the cells of any
    other known car or detection. hybrid: also every cell inside a tracked other
    car's Gaussian, at HYBRID_WEIGHT. Every other cell has weight 0, so a car the
    detector misses is not taught as road (unless it is unlabelled AND inside the
    window; the label builder found no labelled car there in any frame).
    Regression: GPS car with GPS_CODE (no in-cell offset, no size); hybrid other
    cars with the normal code weights times HYBRID_WEIGHT, all over one shared
    count of objects, so each other car weighs HYBRID_WEIGHT of the GPS car.
    """
    from mmdet3d.models.utils import clip_sigmoid
    head = model.head
    dev = boxes["gps"].device
    lab = lambda b: torch.full((len(b),), CAR_CLASS_INDEX, dtype=torch.long, device=dev)
    if len(boxes["gps"]) != 1:
        raise ValueError(f"expected exactly one GPS label, got {len(boxes['gps'])}")
    tg = head.get_targets([boxes["gps"]], [lab(boxes["gps"])])
    if int(tg[3][CAR_TASK].sum()) != 1:
        raise ValueError("GPS label fell outside the BEV grid")
    gh = tg[0][CAR_TASK]                                   # (1, 1, fH, fW)
    hm = gh
    weight = cell_window(head, boxes["gps"], GPS_HALF_WIN_M).float()[None, None].expand_as(gh).clone()
    known = torch.cat([boxes["other"], boxes["ignore"]])
    if len(known):
        weight[cell_window(head, known, CAR_HALF_M)[None, None].expand_as(gh)] = 0
    parts = [(tg[1][CAR_TASK], tg[2][CAR_TASK], tg[3][CAR_TASK], torch.tensor(GPS_CODE, dtype=torch.float32, device=dev))]
    if mode == "hybrid" and len(boxes["other"]):
        to = head.get_targets([boxes["other"]], [lab(boxes["other"])])
        oh = to[0][CAR_TASK]
        weight[oh > 0] = HYBRID_WEIGHT
        hm = torch.maximum(gh, oh)
        parts.append((to[1][CAR_TASK], to[2][CAR_TASK], to[3][CAR_TASK],
                      HYBRID_WEIGHT * torch.tensor(head.train_cfg["code_weights"], dtype=torch.float32, device=dev)))
    weight[gh > 0] = 1.0                                   # the GPS car's own Gaussian always counts fully
    p = preds[CAR_TASK][0]
    pred_hm = clip_sigmoid(p["heatmap"])
    num_pos = max(float((hm.eq(1).float() * weight).sum()), 1.0)
    loss = head.loss_cls(pred_hm, hm, weight=weight, avg_factor=num_pos)
    anno = torch.cat((p["reg"], p["height"], p["dim"], p["rot"], p["vel"]), dim=1)
    anno = anno.permute(0, 2, 3, 1).reshape(anno.size(0), -1, anno.size(1))
    n_obj = sum(float(m.sum()) for _, _, m, _ in parts)   # one shared count across GPS and others
    for target, ind, mask, code in parts:
        if float(mask.sum()) == 0:
            continue
        pred = head._gather_feat(anno, ind)
        w = mask.unsqueeze(2).expand_as(target).float() * code
        loss = loss + head.loss_bbox(pred, target, w, avg_factor=n_obj)
    return loss


def frames_for(clip):
    d = ROOT / f"outputs/finetune/{LABELS}/{clip}"
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
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # the compiled voxel pooling is CUDA only
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
    boxes = raw[keep][:, :9].clone().float().to(dev)
    assert len(boxes) > 3, f"only {len(boxes)} car detections to test with"

    with torch.no_grad():
        targets = model.get_targets([boxes], [labels[keep]])
        preds2 = model(img, mats)
        matched = float(model.loss(targets, preds2))
        shuffled = boxes.clone()
        shuffled[:, [3, 4]] = shuffled[:, [4, 3]]        # swap the two dims
        loss_swapped = float(model.loss(model.get_targets([shuffled], [labels[keep]]),
                                        model(img, mats)))
        # get_bboxes outputs the box bottom; the targets must be the centre (load_sample adds h/2)
        centred = boxes.clone()
        centred[:, 2] += centred[:, 5] / 2
        loss_centred = float(model.loss(model.get_targets([centred], [labels[keep]]),
                                        model(img, mats)))
    print(f"selfcheck: {len(boxes)} own detections as labels -> loss {matched:.3f}; "
          f"with length/width swapped -> {loss_swapped:.3f}; z as centre -> {loss_centred:.3f}")
    assert matched < loss_swapped, (
        "swapping dims did not increase the loss: the column order assumption "
        "in load_sample is not verified by this test")
    assert loss_centred < matched, (
        "z as box centre did not lower the loss: the h/2 shift in load_sample is wrong")
    print("selfcheck ok (self-consistent ordering confirmed)")


def main():
    global LABELS, EXTRINSIC, FRAMES, FEAT_MASK
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
    ap.add_argument("--labels", default=None, help="label set (default: pseudo_labels, or gps_labels in GPS modes)")
    ap.add_argument("--mode", choices=["pseudo", "gps_only", "hybrid"], default="pseudo")
    ap.add_argument("--extrinsic", default=None,
                    help="default metric_extrinsic_site.json (pseudo recipe) or, in GPS modes, "
                         "metric_extrinsic_h151_dpm031.json, the frame the GPS labels were built in")
    ap.add_argument("--frames-root", default=FRAMES, help="frame folder, {clip} is filled in")
    ap.add_argument("--feat-mask", action="store_true")
    ap.add_argument("--train-clips", nargs="+", default=None,
                    help=f"default {TRAIN_CLIPS}, or in GPS modes HV_T_EW_1 AV_T_WE_1 AV_T_WE_3")
    ap.add_argument("--no-val", action="store_true")
    args = ap.parse_args()
    gps = args.mode != "pseudo"
    args.labels = args.labels or ("gps_labels" if gps else "pseudo_labels")
    args.extrinsic = args.extrinsic or ("metric_extrinsic_h151_dpm031.json" if gps else "metric_extrinsic_site.json")
    args.train_clips = args.train_clips or (["HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3"] if gps else TRAIN_CLIPS)
    if gps:
        args.no_val = True                    # GPS modes never look at the test clip, not even for validation
        if args.labels != "gps_labels" or set(args.train_clips) & set(HELD_OUT):
            raise ValueError("GPS modes train on gps_labels and never on the test clip")
    LABELS, EXTRINSIC, FRAMES, FEAT_MASK = args.labels, args.extrinsic, args.frames_root, args.feat_mask

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = args.device
    model = build_model(); load_checkpoint(model, CKPT_PATH); model.to(dev)
    trainable = freeze_image_branch(model)
    print(f"trainable: {len(trainable)} tensors, "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M params (head only)")

    train_clips = args.train_clips
    cal = {c: clip_paths(c) for c in train_clips + ([] if args.no_val else HELD_OUT)}
    samples = [(c, f) for c in train_clips
               for f in (frames_for(c)[:args.limit] if args.limit else frames_for(c))]
    val = [] if args.no_val else [(c, f) for c in HELD_OUT
           for f in (frames_for(c)[:args.limit] if args.limit else frames_for(c))[::5]]
    print(f"mode {args.mode}, labels {LABELS}, extrinsic {EXTRINSIC}, frames {FRAMES}, "
          f"feature mask {FEAT_MASK}; train {len(samples)} frames from {train_clips}; "
          f"val {len(val)} frames from {[] if args.no_val else HELD_OUT}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    history = []

    def evaluate():
        if not val:
            return None
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
    print(f"epoch 0 (before training): held-out loss {v0}")
    history.append({"epoch": 0, "train_loss": None, "val_loss": v0})

    for ep in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.eval(); freeze_image_branch(model)
        run, n, t0 = 0.0, 0, time.time()
        for i, (c, f) in enumerate(samples):
            K, l2c = cal[c]
            if gps:
                img, mats, boxes = load_sample(c, f, K, l2c, roles=True)
                img, mats = to_device(img, mats, dev)
                loss = gps_loss(model, model(img, mats), {k: v.to(dev) for k, v in boxes.items()}, args.mode)
            else:
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
        print(f"epoch {ep}: train {tr:.4f} | held-out {vl} "
              f"({(time.time()-t0)/60:.1f} min)")
        torch.save({"state_dict": model.state_dict(), "epoch": ep,
                    "val_loss": vl, "train_clips": train_clips, "held_out": HELD_OUT,
                    "args": vars(args)}, out / f"head_ft_ep{ep}.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    if val:
        best = min(history[1:], key=lambda h: h["val_loss"])
        print(f"\nbest epoch {best['epoch']}: held-out {best['val_loss']:.4f} (started {v0:.4f})")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
