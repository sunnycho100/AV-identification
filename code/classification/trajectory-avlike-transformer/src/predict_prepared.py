"""Predict strictly prepared 4-second windows with a saved experimental fold.

Keep feature order, training normalization, chosen iteration and validation
temperature together. A fold checkpoint is an experimental model, not a final
model fitted on all available data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from prepared_experiment import infer, normalize, window_statistics
from train import build_model


def predict(path, x, feature_names, selection="bce", calibrated=True):
    path = Path(path)
    stats = json.loads((path / "feature_stats.json").read_text())
    if list(feature_names) != stats["feature_names"]:
        raise ValueError("Feature order does not match the checkpoint")
    if x.ndim != 3 or x.shape[1:] != (40, len(feature_names)) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,40,F] strictly window-local features")
    normalized = normalize(x, stats)
    checkpoint = path / f"best_{selection}.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(state["model"], "cpu")
        model.load_state_dict(state["model_state_dict"])
        z = infer(model, torch.from_numpy(normalized))
    else:
        from joblib import load
        selected = json.loads((path / "selection.json").read_text())[selection]["epoch"]
        if selected == 0:
            counts = json.loads((path / "splits.json").read_text())["train"]["class_counts"]
            z = np.full(len(x), np.log(counts["1"] / counts["0"]))
        else:
            estimator = load(path / "all_iterations.joblib")
            for iteration, p in enumerate(estimator.staged_predict_proba(window_statistics(normalized)), 1):
                if iteration == selected:
                    z = np.log(np.clip(p[:, 1], 1e-12, 1-1e-12)) - np.log(np.clip(p[:, 0], 1e-12, 1-1e-12))
                    break
            else:
                raise ValueError("Selected iteration not found")
    temperature = (json.loads((path / "calibration.json").read_text())[selection]["temperature"]
                   if calibrated else 1.)
    return z.astype(np.float64) / temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fold", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", choices=["bce", "accuracy", "auroc"], default="bce")
    parser.add_argument("--uncalibrated", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    with np.load(args.npz, allow_pickle=False) as data:
        z = predict(args.run_fold, data["X"], data["feature_names"],
                    args.selection, not args.uncalibrated)
        frame = pd.DataFrame(dict(window_index=data["window_index"], logit=z,
                                  prob_ACC=expit(z), predicted_ACC=(z >= 0).astype(int)))
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
