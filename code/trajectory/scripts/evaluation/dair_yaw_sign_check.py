"""Does BEVHeight's predicted yaw agree in sign with DAIR-V2X-I ground truth?

On our roadside clips the predicted yaw sits near 0 rad for vehicles travelling
in both directions, so about half the boxes point backwards. This runs the same
checkpoint on its own training domain, where ground truth exists, to separate a
decode/convention bug from a model property from a domain effect.

Frame and convention (no conversion is applied, and here is why)
---------------------------------------------------------------
Prediction: `model.get_bboxes` returns LiDARInstance3DBoxes whose column 6 is
yaw in the virtuallidar (ego) frame. `evaluators/det_evaluators.py` copies that
straight into `box_yaw`, and `evaluators/result2kitti.py` only converts it to a
KITTI camera-frame rotation_y at export time, with ry = pi/2 - yaw_lidar.

Ground truth: `dair_12hz_infos_{train,val}.pkl` stores each annotation's
rotation as a quaternion in that same virtuallidar frame (the infrastructure
ego_pose is identity), and `NuscMVDetDataset.get_gt` turns it into the yaw the
model was trained to regress. Both sides are therefore already virtuallidar-frame
yaw, and comparing them directly is the apples-to-apples test. `--selfcheck`
proves that by pulling the KITTI camera-frame labels for the same frames and
checking they agree through the relation the label converter actually used,
ry = -pi/2 - yaw_lidar (scripts/data_converter/gen_kitti/label_json2kitti.py:23).

Side finding the selfcheck also pins down: `evaluators/result2kitti.py:226`
exports predictions with ry = +pi/2 - yaw_lidar, which is 180 degrees away from
the GT convention above. KITTI 3D IoU is flip-symmetric for a box, so 3D AP is
unaffected, but anything that reads exported orientation is off by pi.

Run (miniforge python, CPU):
    /Users/sunghwan_cho/miniforge/bin/python \
        scripts/evaluation/dair_yaw_sign_check.py --split both
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn  # noqa: E402
from evaluators.result2kitti import get_lidar2cam  # noqa: E402
from models.bev_height import BEVHeight  # noqa: E402
from scripts.object_detection.run_bevheight_single import load_checkpoint  # noqa: E402

_EXP = ROOT / "experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py"
_spec = importlib.util.spec_from_file_location("bev_exp", _EXP)
bev_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bev_exp)

DATA_ROOT = ROOT / "data/dair-v2x-i"
KITTI_LABELS = ROOT / "data/dair-v2x-i-kitti/training/label_2"
CKPT = ROOT / "checkpoints/BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt"
CAR = bev_exp.CLASSES.index("car")


def wrap_pi(a):
    """Wrap to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def fold_pi(a):
    """Absolute error modulo pi, i.e. ignoring a 180 degree flip. In [0, pi/2]."""
    e = np.abs(wrap_pi(a))
    return np.minimum(e, np.pi - e)


def build_dataset(split):
    return NuscMVDetDataset(
        ida_aug_conf=bev_exp.ida_aug_conf,
        classes=bev_exp.CLASSES,
        data_root=str(DATA_ROOT) + "/",
        info_path=str(DATA_ROOT / f"dair_12hz_infos_{split}.pkl"),
        is_train=False,
        img_conf=bev_exp.img_conf,
        num_sweeps=1,
        return_depth=False,
    )


def predict(model, dataset, idx):
    """One frame through the repo's DAIR eval path. Returns (xy, yaw, score) for cars."""
    sweep_imgs, mats, _, img_metas, _, _ = collate_fn([dataset[idx]])
    with torch.no_grad():
        results = model.get_bboxes(model(sweep_imgs, mats), img_metas)
    boxes = results[0][0].tensor.cpu().numpy()
    scores = results[0][1].cpu().numpy()
    labels = results[0][2].cpu().numpy()
    keep = labels == CAR
    return boxes[keep][:, [0, 1]], boxes[keep][:, 6], scores[keep]


def gt_cars(dataset, idx):
    """GT car centres and virtuallidar yaw, via the dataloader's own get_gt."""
    boxes, labels = dataset.get_gt(dataset.infos[idx], ["CAM_FRONT"])
    if len(boxes) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    boxes, labels = boxes.numpy(), labels.numpy()
    keep = labels == CAR
    return boxes[keep][:, [0, 1]], boxes[keep][:, 6]


def match(pred_xy, gt_xy, max_dist):
    """Greedy nearest-neighbour one-to-one match in BEV. Returns (pred_i, gt_j) pairs."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return []
    d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    pairs, used_p, used_g = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > max_dist:
            break
        if i in used_p or j in used_g:
            continue
        pairs.append((int(i), int(j)))
        used_p.add(i)
        used_g.add(j)
    return pairs


def summarize(raw, gt_yaw):
    """Error table for one set of matched pairs."""
    if len(raw) == 0:
        return {"n": 0}
    raw = np.asarray(raw)
    return {
        "n": int(len(raw)),
        "median_folded_err_rad": float(np.median(fold_pi(raw))),
        "median_folded_err_deg": float(np.degrees(np.median(fold_pi(raw)))),
        "median_abs_raw_err_rad": float(np.median(np.abs(wrap_pi(raw)))),
        "frac_raw_gt_half_pi": float(np.mean(np.abs(wrap_pi(raw)) > np.pi / 2)),
        "sign_agreement": float(np.mean(np.abs(wrap_pi(raw)) < np.pi / 2)),
        "mean_gt_yaw_rad": float(np.mean(gt_yaw)) if len(gt_yaw) else None,
    }


def hist12(values):
    counts, edges = np.histogram(wrap_pi(values), bins=12, range=(-np.pi, np.pi))
    return {"edges_rad": [round(e, 4) for e in edges.tolist()],
            "counts": counts.tolist()}


def run_split(model, split, limit, score_thresh, max_dist):
    dataset = build_dataset(split)
    n = len(dataset) if limit is None else min(limit, len(dataset))
    raw, gt_all, pred_all, tokens = [], [], [], []
    for idx in range(n):
        pred_xy, pred_yaw, scores = predict(model, dataset, idx)
        keep = scores > score_thresh
        pred_xy, pred_yaw = pred_xy[keep], pred_yaw[keep]
        g_xy, g_yaw = gt_cars(dataset, idx)
        pairs = match(pred_xy, g_xy, max_dist)
        for i, j in pairs:
            raw.append(wrap_pi(pred_yaw[i] - g_yaw[j]))
            gt_all.append(g_yaw[j])
            pred_all.append(pred_yaw[i])
        tokens.append(dataset.infos[idx]["sample_token"])
        print(f"  [{split} {idx + 1}/{n}] {dataset.infos[idx]['sample_token']}: "
              f"{len(pred_xy)} pred cars, {len(g_xy)} gt cars, {len(pairs)} matched")

    raw = np.array(raw)
    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)
    # GT heading quadrant: pointing broadly +x versus broadly -x
    fwd = (wrap_pi(gt_all) > -np.pi / 2) & (wrap_pi(gt_all) <= np.pi / 2)
    return {
        "split": split,
        "frames": n,
        "frame_tokens": tokens,
        "score_thresh": score_thresh,
        "match_dist_m": max_dist,
        "all": summarize(raw, gt_all),
        "gt_forward_quadrant": summarize(raw[fwd], gt_all[fwd]),
        "gt_backward_quadrant": summarize(raw[~fwd], gt_all[~fwd]),
        "hist_pred_yaw": hist12(pred_all),
        "hist_gt_yaw": hist12(gt_all),
    }


def selfcheck(splits=("val", "train")):
    """The GT yaw we compare against is virtuallidar-frame, so it must reproduce
    the KITTI camera-frame labels through the converter's own relation."""
    assert abs(fold_pi(np.pi - 0.1) - 0.1) < 1e-9, "fold_pi must ignore a 180 flip"
    assert abs(fold_pi(0.1) - 0.1) < 1e-9
    assert abs(wrap_pi(3 * np.pi / 2) + np.pi / 2) < 1e-9

    checked, worst, worst_r2k = 0, 0.0, np.pi
    for split in splits:
        dataset = build_dataset(split)
        for info in dataset.infos:
            sid = int(info["sample_token"].split("/")[1].split(".")[0])
            label = KITTI_LABELS / f"{sid:06d}.txt"
            calib = DATA_ROOT / "calib/virtuallidar_to_camera" / f"{sid:06d}.json"
            if not label.exists():
                continue
            cam2lidar = np.linalg.inv(get_lidar2cam(str(calib))[0])
            lbl = []
            for line in label.read_text().splitlines():
                f = line.split()
                if not f:
                    continue
                loc_cam = np.array([float(f[11]), float(f[12]), float(f[13]), 1.0])
                lbl.append(((cam2lidar @ loc_cam)[:2], float(f[14])))
            for ann in info["ann_infos"]:
                c = np.array(ann["translation"])[:2]
                best = min(lbl, key=lambda e: np.linalg.norm(e[0] - c), default=None)
                if best is None or np.linalg.norm(best[0] - c) > 0.5:
                    continue
                ry = best[1]
                # GT labels: label_json2kitti.py writes ry = -pi/2 - yaw_lidar
                err = abs(wrap_pi(ann["yaw_lidar"] - (-np.pi / 2 - ry)))
                # result2kitti exports predictions as ry = +pi/2 - yaw_lidar,
                # which is the same angle turned through pi
                err_r2k = abs(wrap_pi(ann["yaw_lidar"] - (np.pi / 2 - ry)))
                worst = max(worst, float(err))
                worst_r2k = min(worst_r2k, float(err_r2k))
                assert err < 0.05, (sid, ann["yaw_lidar"], ry, err)
                checked += 1
    assert checked > 50, f"only {checked} objects cross-checked"
    assert worst_r2k > np.pi - 0.05, worst_r2k
    print(f"selfcheck ok: {checked} objects. pkl yaw_lidar == -pi/2 - ry to "
          f"{worst:.2e} rad, so the GT we compare against is the KITTI labels' own "
          f"angle. The result2kitti export relation (+pi/2 - ry) is {worst_r2k:.4f} "
          f"rad away from it for every object, i.e. flipped by pi.")


def main():
    ap = argparse.ArgumentParser("BEVHeight yaw sign check on DAIR-V2X-I")
    ap.add_argument("--split", default="val", choices=["val", "train", "both"])
    ap.add_argument("--limit", type=int, default=None, help="max frames per split")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--match-dist", type=float, default=2.0, help="BEV metres")
    ap.add_argument("--out", default=str(ROOT / "outputs/orientation/dair_yaw_sign_check.json"))
    args = ap.parse_args()

    model = BEVHeight(bev_exp.backbone_conf, bev_exp.head_conf)
    model.eval()
    info = load_checkpoint(model, CKPT)
    print(f"checkpoint {CKPT.name}: {info['matched']} keys matched, "
          f"{len(info['missing'])} missing")

    splits = ["val", "train"] if args.split == "both" else [args.split]
    out = {
        "checkpoint": CKPT.name,
        "convention": "both sides are virtuallidar (ego) frame yaw; no conversion "
                      "applied. GT from the infos pkl quaternion via "
                      "NuscMVDetDataset.get_gt, predictions from box[:, 6] which "
                      "result2kitti later exports as ry = pi/2 - yaw_lidar.",
        "splits": [run_split(model, s, args.limit, args.score_thresh, args.match_dist)
                   for s in splits],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
