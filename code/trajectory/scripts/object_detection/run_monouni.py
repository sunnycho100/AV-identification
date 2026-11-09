"""MonoUNI inference on a prepared root (monouni_prep.py), no labels, no distributed init.

Their test entry (lib/train_val.py -e) always starts an NCCL process group, builds
the train set and runs the Rope3D eval, which crashes without labels. This drives
their own dataset, model and decoder directly and only writes predictions:
<out>/<id>.txt, KITTI 16 columns as their tester writes them
(class 0.0 0 alpha x1 y1 x2 y2 h w l x y z ry score; x y z = bottom centre in
camera frame, m). --random-weights skips the checkpoint to test the plumbing.

Runs on the lab with the MonoUNI clone on the path:
    CUDA_VISIBLE_DEVICES=1 python scripts/object_detection/run_monouni.py \
        --monouni ~/roadside-camera/MonoUNI --root /home/data/scho242/monouni/AV_T_EW_3_A \
        --ckpt /home/data/scho242/monouni/rope3d.pth --out /home/data/scho242/monouni/AV_T_EW_3_A/pred
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    ap = argparse.ArgumentParser("MonoUNI label-free inference")
    ap.add_argument("--monouni", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--random-weights", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser(a.monouni))
    from torch.utils.data import DataLoader
    from lib.datasets.rope3d import Rope3D
    from lib.helpers.model_helper import build_model
    from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections

    cfg = yaml.load(open(os.path.join(os.path.expanduser(a.monouni), "lib/config.yaml")), Loader=yaml.Loader)
    ds = Rope3D(root_dir=a.root, split="val", cfg=cfg["dataset"])
    dl = DataLoader(ds, batch_size=a.batch, num_workers=2, shuffle=False)
    dev = torch.device("cuda:0")
    model = build_model(cfg["model"], ds.cls_mean_size)
    if not a.random_weights:
        sd = torch.load(a.ckpt, map_location="cpu")
        sd = sd.get("model_state", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.to(dev).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_box = 0
    with torch.no_grad():
        for inputs, calibs, coord_ranges, _, info, pcos, psin in dl:
            outputs = model(inputs.to(dev), coord_ranges.to(dev), calibs.to(dev), K=50, mode="val",
                            calib_pitch_sin=psin.to(dev), calib_pitch_cos=pcos.to(dev))
            dets = extract_dets_from_outputs(outputs, calibs.to(dev), K=50).cpu().numpy()
            ids = list(info["img_id"])
            info = {"img_id": ids, "img_size": info["img_size"].numpy(),
                    "bbox_downsample_ratio": info["bbox_downsample_ratio"].numpy()}
            res = decode_detections(dets=dets, info=info, calibs=[ds.get_calib(i) for i in ids],
                                    denorms=[ds.get_denorm(i) for i in ids], cls_mean_size=ds.cls_mean_size,
                                    threshold=a.threshold)
            for i, preds in res.items():
                lines = [" ".join([ds.class_name[int(p[0])], "0.0", "0"] + [f"{v:.4f}" for v in p[1:]]) for p in preds]
                (out / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_box += len(lines)
    n = len(list(out.glob("*.txt")))
    assert n == len(ds), f"wrote {n} of {len(ds)} frames"
    print(f"wrote {n} frames, {n_box} boxes ({n_box / max(n, 1):.1f} per frame) -> {out}")


if __name__ == "__main__":
    main()
