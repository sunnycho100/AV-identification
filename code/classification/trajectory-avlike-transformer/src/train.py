import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import VanillaTrajectoryTransformer, TemporalConvClassifier, PooledTrajectoryTransformer
from utils import compute_metrics, load_config, set_seed


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            all_probs.append(torch.sigmoid(logit).detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, probs)
    metrics["loss"] = total_loss / n_samples
    return metrics


def build_model(model_cfg, device):
    architecture = model_cfg.get("architecture", "transformer")
    if architecture == "pooled_transformer":
        return PooledTrajectoryTransformer(
            **{key: model_cfg[key] for key in
               ("input_dim", "d_model", "num_heads", "ffn_dim", "seq_len", "dropout")
               if key in model_cfg}
        ).to(device)
    if architecture == "temporal_cnn":
        return TemporalConvClassifier(
            input_dim=model_cfg["input_dim"],
            channels=model_cfg.get("channels", 32),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)
    if architecture != "transformer":
        raise ValueError(f"Unknown architecture: {architecture}")
    model = VanillaTrajectoryTransformer(
        input_dim=model_cfg["input_dim"],
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        seq_len=model_cfg["seq_len"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer_baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders, _, diagnostics = get_dataloaders(config, fit_stats=True)

    print("=== Dataset statistics ===")
    for split in ["train", "val", "test"]:
        d = diagnostics[split]
        print(
            f"{split}: trajectories={d['num_trajectories']} windows={d['num_windows']} "
            f"AV={d['num_av_windows']} HV={d['num_hv_windows']}"
        )

    model = build_model(config["model"], device)

    train_cfg = config["training"]
    if train_cfg["use_pos_weight"]:
        n_pos = diagnostics["train"]["num_av_windows"]
        n_neg = diagnostics["train"]["num_hv_windows"]
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    out_cfg = config["output"]
    os.makedirs(out_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(out_cfg["log_dir"], exist_ok=True)
    checkpoint_path = os.path.join(out_cfg["checkpoint_dir"], "best_model.pt")
    log_path = os.path.join(out_cfg["log_dir"], "train_log.jsonl")

    best_val_auroc = -np.inf
    epochs_no_improve = 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = run_epoch(model, loaders["val"], criterion, None, device)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"({elapsed:.1f}s)"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_auroc = val_metrics["auroc"]
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_auroc": val_auroc,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
            break

    with open(log_path, "w") as f:
        for record in history:
            f.write(json.dumps(record) + "\n")

    print(f"Best val AUROC: {best_val_auroc:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
