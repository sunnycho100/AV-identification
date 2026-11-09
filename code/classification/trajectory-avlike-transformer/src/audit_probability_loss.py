"""Compare untrained and saved AUROC-selected models using identical losses."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from cross_validate import predict, make_source_holdout_splits
from dataset import make_dataloaders, split_trajectory_ids
from train import build_model
from utils import load_config, set_seed


def main():
    root = Path("artifacts/experiments/20260920_probability_audit")
    root.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [("original_random_split", Path("artifacts/checkpoints/best_model.pt"),
             Path("artifacts/logs/train_log.jsonl"), None, 42)]
    runs = Path("artifacts/experiments/20260920_diagnostics_v2/baseline")
    for seed in (42, 52, 62):
        for i, fold in enumerate(("part2", "part3", "highway")):
            base = runs / f"seed_{seed}"
            jobs.append((f"cv_{seed}_{fold}", base/"checkpoints"/fold/"best_model.pt",
                         base/"logs"/fold/"train_log.jsonl", base/"logs"/fold/"splits.json", seed+i))
    rows, histories = [], []
    for name, checkpoint_path, log_path, splits_path, seed in jobs:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        cfg = ckpt["config"]
        df = pd.read_csv(cfg["data"]["processed_csv"])
        if splits_path:
            splits = json.loads(splits_path.read_text())
        else:
            s = cfg["split"]
            splits = split_trajectory_ids(df, "Trajectory_ID", "label", s["train_ratio"],
                                          s["val_ratio"], s["test_ratio"], seed)
        set_seed(seed)
        loaders, datasets, diagnostics = make_dataloaders(cfg, df, splits,
            stats_path=str(root/name/"stats.json"))
        model = build_model(cfg["model"], device)
        train = diagnostics["train"]
        prevalence = train["num_av_windows"] / train["num_windows"]
        weight = train["num_hv_windows"] / train["num_av_windows"] if cfg["training"]["use_pos_weight"] else 1.0
        for state in ("random_init", "saved_best_auroc"):
            if state == "saved_best_auroc":
                model.load_state_dict(ckpt["model_state_dict"])
            for split in ("val", "test"):
                pred = predict(model, loaders[split], datasets[split], device)
                models = {state: (pred.prob_AV.to_numpy(), pred.logit.to_numpy())}
                if state == "random_init":
                    for label, p in (("constant_0.5", 0.5), ("train_prevalence", prevalence)):
                        models[label] = (np.full(len(pred), p), np.full(len(pred), np.log(p/(1-p))))
                y = pred.label.to_numpy()
                for label, (prob, logit) in models.items():
                    weighted = np.mean(weight*y*np.logaddexp(0, -logit)+(1-y)*np.logaddexp(0, logit))
                    row = dict(run=name, split=split, predictor=label, pos_weight=weight,
                               weighted_bce=weighted, log_loss=log_loss(y, prob),
                               brier=brier_score_loss(y, prob), auroc=roc_auc_score(y, prob),
                               selected_epoch=ckpt["epoch"])
                    tp = pred.assign(prob_AV=prob).groupby("Trajectory_ID").agg(label=("label","first"),p=("prob_AV","mean"))
                    if label in ("constant_0.5", "train_prevalence"):
                        # Avoid float-roundoff creating spurious ranks after aggregation.
                        tp["p"] = prob[0]
                    row.update(trajectory_log_loss=log_loss(tp.label, tp.p),
                               trajectory_auroc=roc_auc_score(tp.label, tp.p))
                    rows.append(row)
        history = [json.loads(line) for line in log_path.read_text().splitlines()]
        minimum = min(history, key=lambda h: h["val"]["loss"])
        for tag, rec in (("first_trained_epoch", history[0]), ("min_val_loss", minimum),
                         ("selected_best_auroc", next(h for h in history if h["epoch"]==ckpt["epoch"])),
                         ("last_epoch", history[-1])):
            histories.append(dict(run=name, stage=tag, epoch=rec["epoch"],
                                  val_loss=rec["val"]["loss"], val_auroc=rec["val"]["auroc"]))
        print("Audited",name,flush=True)
    pd.DataFrame(rows).to_csv(root/"probability_metrics.csv",index=False)
    pd.DataFrame(histories).to_csv(root/"history_summary.csv",index=False)
    print(pd.DataFrame(rows).query("run == 'original_random_split'").to_string(index=False))
    print(pd.DataFrame(histories).to_string(index=False))


if __name__ == "__main__":
    main()
