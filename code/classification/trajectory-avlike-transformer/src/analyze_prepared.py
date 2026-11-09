"""Audit paired prepared-window experiments and produce window-only figures."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from predict_prepared import predict
from prepared_experiment import fit_stats, load_prepared, probability_metrics, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.output)
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["args"]
    torch.set_num_threads(config["threads"])
    x, y, manifest, names = load_prepared(config["prepared"])
    frame = summarize(root)
    paired = {}
    checks, train_rows, source_rows = [], [], []
    for path in sorted(root.glob("*/seed_*/*/metrics.json")):
        fold_dir = path.parent
        variant, seed_str, fold = fold_dir.parts[-3:]
        seed = int(seed_str.split("_")[1])
        splits = json.loads((fold_dir / "splits.json").read_text())
        preds = pd.read_csv(fold_dir / "test_predictions.csv")
        identity = preds[["window_index", "label", "source_id"]]
        key = (seed, fold)
        if key in paired:
            pd.testing.assert_frame_equal(paired[key][0], identity)
            assert paired[key][1] == {k: splits[k] for k in ("train", "val", "test")}
        else:
            paired[key] = (identity, {k: splits[k] for k in ("train", "val", "test")})
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not set(splits[a]["sources"]) & set(splits[b]["sources"])
        assert len(preds) == len(splits["test"]["window_indices"])
        assert np.isfinite(preds.filter(like="_logit").to_numpy()).all()
        history = json.loads((fold_dir / "history.json").read_text())
        selected = json.loads((fold_dir / "selection.json").read_text())
        for metric, info in selected.items():
            values = [r["val"][metric] for r in history]
            optimum = min(values) if metric == "bce" else max(values)
            assert abs(info["value"] - optimum) <= 1e-8
            assert abs(history[info["epoch"]]["val"][metric] - optimum) <= 1e-8
        saved_stats = json.loads((fold_dir / "feature_stats.json").read_text())
        computed_stats = fit_stats(x[np.array(splits["train"]["window_indices"])])
        for name, value in computed_stats.items():
            np.testing.assert_allclose(saved_stats[name], value, atol=1e-12, rtol=1e-12)
        calibration = json.loads((fold_dir / "calibration.json").read_text())
        for selection, info in calibration.items():
            assert info["val_bce_after"] <= info["val_bce_before"] + 1e-10
            np.testing.assert_array_equal(preds[selection+"_logit"] >= 0,
                                          preds[selection+"_temperature_logit"] >= 0)
            np.testing.assert_allclose(preds[selection+"_logit"] / info["temperature"],
                                       preds[selection+"_temperature_logit"], rtol=1e-6, atol=1e-6)
        # Exercise the standalone prediction path, including tree iteration and saved stats.
        indices = preds.window_index.to_numpy()[:64]
        np.testing.assert_array_equal(manifest.iloc[indices].window_index, indices)
        actual = predict(fold_dir, x[indices], names)
        expected = preds.bce_temperature_logit.to_numpy()[:64]
        error = float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        train_indices = np.array(splits["train"]["window_indices"])
        for calibrated in (False, True):
            logits = predict(fold_dir, x[train_indices], names, calibrated=calibrated)
            metrics = probability_metrics(y[train_indices], logits)
            train_rows.append(dict(variant=variant, seed=seed, fold=fold, calibrated=calibrated,
                                   **{k: v for k, v in metrics.items() if k != "confusion_matrix"}))
        for source, group in preds.groupby("source_id"):
            for selection in ("bce", "bce_temperature", "auroc_temperature"):
                label = int(group.label.iloc[0])
                assert group.label.nunique() == 1
                z = group[selection+"_logit"].to_numpy()
                source_rows.append(dict(variant=variant, seed=seed, fold=fold, source=source,
                                        label=label, selection=selection, windows=len(group),
                                        accuracy=float(np.mean((z >= 0) == label)),
                                        bce=float(np.mean(np.logaddexp(0, z)-label*z))))
        checks.append(dict(variant=variant, seed=seed, fold=fold, prediction_max_error=error))
    expected = len(config["seeds"]) * len(config["folds"]) * len(config["variants"])
    assert len(checks) == expected, (len(checks), expected)
    write_json(root / "audit.json", dict(runs=len(checks), paired_splits=len(paired),
                                         source_disjoint=True, calibration_validation_only=True,
                                         inference_roundtrip=checks))
    pd.DataFrame(train_rows).to_csv(root / "train_diagnostics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(root / "source_metrics.csv", index=False)
    frame.groupby(["variant", "selection", "fold"])[["accuracy", "bce", "human_recall", "acc_recall"]].mean().to_csv(root / "by_fold.csv")
    reference = frame[(frame.variant == "transformer") & (frame.selection == "bce_temperature")]
    primary = frame[frame.selection == "bce_temperature"].merge(
        reference[["seed", "fold", "accuracy", "bce"]], on=["seed", "fold"], suffixes=("", "_transformer"))
    primary["accuracy_delta"] = primary.accuracy - primary.accuracy_transformer
    primary["bce_delta"] = primary.bce - primary.bce_transformer
    primary.to_csv(root / "paired_deltas.csv", index=False)

    # Exploratory variance reduction; fixed equal weights, no test-based tuning.
    ensemble_rows = []
    for (variant, fold), group in frame.groupby(["variant", "fold"]):
        predictions = [pd.read_csv(root / variant / f"seed_{seed}" / fold / "test_predictions.csv")
                       for seed in sorted(group.seed.unique())]
        first = predictions[0]
        for item in predictions[1:]:
            pd.testing.assert_frame_equal(first[["window_index", "label"]], item[["window_index", "label"]])
        for selection in ("bce_temperature", "auroc_temperature"):
            p = np.mean([expit(item[selection+"_logit"].to_numpy()) for item in predictions], axis=0)
            p = np.clip(p, 1e-12, 1-1e-12)
            z = np.log(p) - np.log1p(-p)
            metrics = probability_metrics(first.label.to_numpy(), z)
            ensemble_rows.append(dict(variant=variant, fold=fold, selection=selection,
                                      members=len(predictions),
                                      **{k: v for k, v in metrics.items() if k != "confusion_matrix"}))
    pd.DataFrame(ensemble_rows).to_csv(root / "seed_ensembles.csv", index=False)

    variants = config["variants"]
    titles = {"transformer": "Transformer 420k", "pooled_transformer": "Pooled Transformer 10.4k",
              "temporal_cnn": "CNN 7.3k", "cnn_aux": "CNN + auxiliary ACC", "histogram_tree": "Window statistics + trees"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ["accuracy", "bce"], ["4-second accuracy", "Unweighted test BCE"]):
        for offset, selection, color in [(-.18, "bce", "#4183b4"), (.18, "bce_temperature", "#df9341")]:
            means = frame[frame.selection == selection].groupby(["variant", "seed"])[metric].mean()
            values = [means.loc[v].mean() for v in variants]
            errors = [float(means.loc[v].std()) if len(means.loc[v]) > 1 else 0.
                      for v in variants]
            ax.bar(np.arange(len(variants))+offset, values, width=.36, yerr=errors,
                   label="Raw" if selection == "bce" else "Validation temperature", color=color, capsize=3)
        ax.set_xticks(np.arange(len(variants)), [titles[v] for v in variants], rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 1)
    axes[1].axhline(np.log(2), color="black", linestyle="--", label="Constant 0.5")
    axes[1].legend(fontsize=8)
    fig.suptitle("Source-disjoint 4-second windows; validation BCE selection\nError bars: seed SD, not independent-domain confidence intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(root / "model_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(config["seeds"]), len(config["folds"]), figsize=(12, 8), squeeze=False)
    for row, seed in enumerate(config["seeds"]):
        for col, fold_index in enumerate(config["folds"]):
            from prepared_experiment import PARTS
            ax = axes[row, col]
            for variant in variants:
                if variant == "histogram_tree":
                    continue
                history = json.loads((root / variant / f"seed_{seed}" / PARTS[fold_index] / "history.json").read_text())
                ax.plot([r["epoch"] for r in history], [r["val"]["bce"] for r in history], label=titles[variant], linewidth=1.)
            ax.axhline(np.log(2), color="black", linestyle="--", linewidth=.7)
            ax.set_title(f"seed {seed}, test {PARTS[fold_index]}", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation BCE")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "validation_curves.png", dpi=170)
    plt.close(fig)
    snapshot = root / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("analyze_prepared.py", "predict_prepared.py"):
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    print(json.dumps(dict(runs=len(checks), paired_splits=len(paired), status="passed")))


if __name__ == "__main__":
    main()
