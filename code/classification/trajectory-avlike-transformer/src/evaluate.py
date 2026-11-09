import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from train import build_model
from utils import compute_metrics, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer_baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, datasets, _ = get_dataloaders(config, fit_stats=False)
    test_dataset = datasets["test"]
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    out_cfg = config["output"]
    checkpoint_path = os.path.join(out_cfg["checkpoint_dir"], "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(config["model"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logit = model(x)
            all_logits.append(logit.cpu().numpy())
            all_labels.append(y.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(labels, probs)

    print("=== Test metrics ===")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        print(f"{k}: {v:.4f}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")

    os.makedirs(out_cfg["figure_dir"], exist_ok=True)
    os.makedirs(out_cfg["log_dir"], exist_ok=True)

    metrics_path = os.path.join(out_cfg["log_dir"], "test_metrics.json")
    save_json(metrics, metrics_path)

    cm = np.array(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["HV", "AV"])
    disp.plot(cmap="Blues", values_format="d")
    fig_path = os.path.join(out_cfg["figure_dir"], "confusion_matrix.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    pred_rows = [
        {
            "Trajectory_ID": w["trajectory_id"],
            "window_start_time": w["window_start_time"],
            "window_end_time": w["window_end_time"],
            "label": int(w["y"]),
            "logit": float(logit),
            "prob_AV": float(prob),
            "prediction": int(pred),
        }
        for w, logit, prob, pred in zip(test_dataset.windows, logits, probs, preds)
    ]
    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = os.path.join(out_cfg["log_dir"], "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix figure to {fig_path}")
    print(f"Saved predictions to {pred_csv_path}")


if __name__ == "__main__":
    main()
