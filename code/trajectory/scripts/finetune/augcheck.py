"""Self-check for train_finetune --augment (server, CUDA).

Pretrained model on one frame, plain vs augmented input. If ida/bda/feature-mask
bookkeeping is right, the augmented detections land on the same ground points
(after undoing the BEV flip) and the own-detection loss stays as low as plain.
A deliberately broken variant (augmented image, ida left as is) must disagree.
"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.finetune.train_finetune as T
from mmdet3d.core.bbox import LiDARInstance3DBoxes

T.FEAT_MASK = True; T.LABELS = "std_labels"
dev = "cuda"
model = T.build_model(); T.load_checkpoint(model, T.CKPT_PATH); model.to(dev).eval()
K, l2c = T.clip_paths("HV_T_EW_1")
img, mats, _, _ = T.load_sample("HV_T_EW_1", 238, K, l2c)

def dets(img, mats):
    i, m = T.to_device(img, mats, dev)
    with torch.no_grad():
        p = model(i, m)
        r = model.get_bboxes(p, [{"box_type_3d": LiDARInstance3DBoxes}])[0]
    c = r[2] == 0
    print(f"   car scores max {float(r[1][c].max()) if c.any() else 0:.2f}, n>=0.2 {int((r[1][c] >= 0.2).sum())}")
    k = (r[1] >= 0.4) & c
    return r[0].tensor[k][:, :9].float().cpu()

def own_loss(img, mats, boxes):
    i, m = T.to_device(img, mats, dev)
    lab = torch.zeros(len(boxes), dtype=torch.long, device=dev)
    with torch.no_grad():
        t = model.get_targets([boxes.to(dev)], [lab])
        return float(T.car_only_loss(model, t, model(i, m)))

def agree(a, b):
    """median xy distance from each box in a to its nearest in b, and yaw diff."""
    if len(b) == 0:
        return float("inf"), float("inf")
    d = torch.cdist(a[:, :2], b[:, :2]); j = d.argmin(1)
    dy = torch.atan2(torch.sin(a[:, 6] - b[j, 6]), torch.cos(a[:, 6] - b[j, 6])).abs()
    return float(d.min(1).values.median()), float(dy.median().rad2deg())

base = dets(img, mats)
lab0 = base.clone(); lab0[:, 2] += lab0[:, 5] / 2
print(f"plain: {len(base)} cars, own-det loss {own_loss(img, mats, lab0):.3f}")
ok = True
for flip, scale, bev in [(False, 1.0, False), (False, 1.08, False), (False, 0.92, False), (False, 1.0, True), (True, 1.0, False), (True, 0.92, True)]:
    i2, m2, b2 = T.augment(img, mats, lab0, flip=flip, scale=scale, bev_flip=bev)
    d = dets(i2, m2)
    if bev:
        d[:, [1, 6, 8]] *= -1
    xy, yaw = agree(base, d)
    l = own_loss(i2, m2, b2)
    print(f"flip {flip} scale {scale} bev {bev}: {len(d)} cars, median shift {xy:.2f} m, yaw {yaw:.1f} deg, own-det loss {l:.3f}")
    # flip alone or BEV flip alone: the pretrained model never saw a mirrored car or road and
    # detects ~nothing, so only cases where the view is unmirrored or doubly mirrored test geometry
    if flip == bev:
        ok &= xy < 1.0
# broken: augmented image but original ida (geometry not updated)
i2, m2, _ = T.augment(img, mats, lab0, flip=True, scale=1.0, bev_flip=False)
m2["ida_mats"] = mats["ida_mats"]
xy, yaw = agree(base, dets(i2, m2))
print(f"BROKEN (flip, ida not updated): median shift {xy:.2f} m, yaw {yaw:.1f} deg")
print("augcheck", "ok" if ok and xy > 2 else "FAILED")
