import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import make_dataloaders
from train import build_model, run_epoch
from utils import compute_metrics, load_config, set_seed


def source_from_trajectory_id(trajectory_id):
    return str(trajectory_id).rsplit("_", 1)[0]


def make_source_holdout_splits(config, df, test_sources, seed):
    data_cfg = config["data"]
    id_col = data_cfg["trajectory_id_col"]
    label_col = data_cfg["label_col"]
    val_ratio = config["cross_validation"]["validation_ratio"]

    trajectories = df.groupby(id_col, as_index=False)[label_col].first()
    trajectories["source"] = trajectories[id_col].map(source_from_trajectory_id)

    available_sources = set(trajectories["source"])
    missing_sources = sorted(set(test_sources) - available_sources)
    if missing_sources:
        raise ValueError(f"Unknown test sources: {missing_sources}")

    test_ids = trajectories.loc[trajectories["source"].isin(test_sources), id_col].tolist()
    remaining = trajectories.loc[~trajectories["source"].isin(test_sources)].copy()

    rng = np.random.RandomState(seed)
    train_ids = []
    val_ids = []
    for label in sorted(remaining[label_col].unique()):
        ids = sorted(remaining.loc[remaining[label_col] == label, id_col].tolist())
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@torch.no_grad()
def predict(model, loader, dataset, device):
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    rows = []
    for window, label, logit, probability in zip(
        dataset.windows, labels, logits, probabilities
    ):
        rows.append(
            {
                "Trajectory_ID": window["trajectory_id"],
                "source": source_from_trajectory_id(window["trajectory_id"]),
                "label": int(label),
                "logit": float(logit),
                "prob_AV": float(probability),
                "prediction": int(probability >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions):
    window_metrics = compute_metrics(predictions["label"], predictions["prob_AV"])

    trajectory_predictions = predictions.groupby("Trajectory_ID", as_index=False).agg(
        source=("source", "first"),
        label=("label", "first"),
        prob_AV=("prob_AV", "mean"),
        windows=("label", "size"),
    )
    trajectory_predictions["prediction"] = (
        trajectory_predictions["prob_AV"] >= 0.5
    ).astype(int)
    trajectory_metrics = compute_metrics(
        trajectory_predictions["label"], trajectory_predictions["prob_AV"]
    )

    source_metrics = {}
    for source, group in predictions.groupby("source"):
        source_metrics[source] = {
            "windows": len(group),
            "label": int(group["label"].iloc[0]),
            "accuracy": float((group["label"] == group["prediction"]).mean()),
            "mean_prob_AV": float(group["prob_AV"].mean()),
        }

    return window_metrics, trajectory_metrics, source_metrics, trajectory_predictions


def train_fold(config, df, fold, fold_index, device):
    fold_name = fold["name"]
    seed = config["training"]["seed"] + fold_index
    set_seed(seed)

    splits = make_source_holdout_splits(config, df, fold["test_sources"], seed)
    checkpoint_dir = os.path.join(config["output"]["checkpoint_dir"], fold_name)
    log_dir = os.path.join(config["output"]["log_dir"], fold_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    stats_path = os.path.join(log_dir, "feature_stats.json")
    loaders, datasets, diagnostics = make_dataloaders(
        config, df, splits, fit_stats=True, stats_path=stats_path
    )

    print(f"\n=== Fold {fold_name}: test={fold['test_sources']} ===")
    for split_name in ["train", "val", "test"]:
        print(f"{split_name}: {diagnostics[split_name]}")

    model = build_model(config["model"], device)
    train_cfg = config["training"]
    n_pos = diagnostics["train"]["num_av_windows"]
    n_neg = diagnostics["train"]["num_hv_windows"]
    pos_weight = None
    if train_cfg["use_pos_weight"]:
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    best_val_auroc = -np.inf
    epochs_without_improvement = 0
    history = []
    started = time.time()

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = run_epoch(model, loaders["val"], criterion, None, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_auroc = val_metrics["auroc"]
        print(
            f"[{fold_name} {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auroc={val_auroc:.4f}"
        )
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fold": fold_name,
                    "epoch": epoch,
                    "val_auroc": val_auroc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            break

    with open(os.path.join(log_dir, "train_log.jsonl"), "w") as handle:
        for record in history:
            handle.write(json.dumps(to_builtin(record)) + "\n")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict(model, loaders["test"], datasets["test"], device)
    window_metrics, trajectory_metrics, source_metrics, trajectory_predictions = (
        evaluate_predictions(predictions)
    )
    predictions.to_csv(os.path.join(log_dir, "test_predictions.csv"), index=False)
    trajectory_predictions.to_csv(
        os.path.join(log_dir, "trajectory_predictions.csv"), index=False
    )

    result = {
        "fold": fold_name,
        "test_sources": fold["test_sources"],
        "best_epoch": checkpoint["epoch"],
        "best_val_auroc": checkpoint["val_auroc"],
        "epochs_ran": len(history),
        "elapsed_seconds": time.time() - started,
        "diagnostics": diagnostics,
        "window_metrics": window_metrics,
        "trajectory_metrics": trajectory_metrics,
        "source_metrics": source_metrics,
    }
    result = to_builtin(result)
    with open(os.path.join(log_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    print(
        f"{fold_name} result: window_AUROC={window_metrics['auroc']:.4f}, "
        f"trajectory_AUROC={trajectory_metrics['auroc']:.4f}, "
        f"trajectory_bal_acc={trajectory_metrics['balanced_accuracy']:.4f}"
    )
    return result


def aggregate_results(results):
    summary = {"folds": results, "aggregate": {}}
    for level in ["window_metrics", "trajectory_metrics"]:
        summary["aggregate"][level] = {}
        for metric in ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc"]:
            values = np.array([result[level][metric] for result in results], dtype=float)
            summary["aggregate"][level][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "values": values.tolist(),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/transformer_improved.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    df = pd.read_csv(config["data"]["processed_csv"])

    results = []
    for fold_index, fold in enumerate(config["cross_validation"]["folds"]):
        results.append(train_fold(config, df, fold, fold_index, device))

    summary = aggregate_results(results)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    summary_path = os.path.join(config["output"]["log_dir"], "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Cross-validation aggregate ===")
    for level, metrics in summary["aggregate"].items():
        print(level)
        for metric, values in metrics.items():
            print(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
