"""Paired source-held-out ablations; preserve existing experiment artifacts.

Run from the project root. Each seed controls both initialization and the
inner trajectory split; all variants use identical splits for a given seed.
Outer test results are exploratory, not a fresh final generalization estimate.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_validate import (
    aggregate_results, evaluate_predictions, make_source_holdout_splits,
    source_from_trajectory_id, to_builtin, train_fold,
)
from dataset import build_windows
from features import FEATURE_COLS
from utils import load_config


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), indent=2), encoding="utf-8")


def configs():
    baseline = load_config("configs/transformer_baseline_cv.yaml")
    small = load_config("configs/transformer_small_cv.yaml")
    features = load_config("configs/transformer_features_cv.yaml")
    improved = load_config("configs/transformer_improved.yaml")
    variants = dict(baseline=baseline, small_bundle=small,
                    features_bundle=features, improved_bundle=improved)
    size = copy.deepcopy(baseline)
    for key in ("d_model", "num_layers", "ffn_dim"):
        size["model"][key] = small["model"][key]
    variants["size_only"] = size
    reg = copy.deepcopy(baseline)
    reg["model"]["dropout"] = 0.3
    reg["training"]["weight_decay"] = 0.001
    variants["regularization_only"] = reg
    stride = copy.deepcopy(baseline)
    stride["features"]["stride"] = 20
    variants["stride_only"] = stride
    for name, columns in {
        "leader_only": ["v_LV_long", "a_LV"],
        "no_jerk": [c for c in FEATURE_COLS if c != "jerk_FV"],
        "no_lateral": [c for c in FEATURE_COLS if c != "lateral_offset"],
    }.items():
        cfg = copy.deepcopy(baseline)
        cfg["features"]["feature_cols"] = columns
        cfg["model"]["input_dim"] = len(columns)
        variants[name] = cfg
    for cfg in variants.values():
        cfg["features"]["eval_stride"] = 10
    return variants


def logistic_fold(cfg, df, fold, index, columns, folder):
    seed = cfg["training"]["seed"] + index
    splits = make_source_holdout_splits(cfg, df, fold["test_sources"], seed)
    write_json(folder / "splits.json", splits)
    matrices, labels, metadata = {}, {}, {}
    for split, ids in splits.items():
        windows = build_windows(df, ids, columns, "label", "Trajectory_ID",
                                "Time_Index", 40, 10)
        x = np.stack([w["x"] for w in windows])
        matrices[split] = np.concatenate(
            [x.mean(1), x.std(1)] + [np.quantile(x, q, axis=1) for q in (0.1, 0.5, 0.9)], axis=1)
        labels[split] = np.array([w["y"] for w in windows])
        metadata[split] = pd.DataFrame({
            "Trajectory_ID": [w["trajectory_id"] for w in windows],
            "label": labels[split],
        })
        metadata[split]["source"] = metadata[split]["Trajectory_ID"].map(source_from_trajectory_id)
    # Match the transformer's window-weighted, class-balanced training objective.
    best_score, best_model, best_c = -np.inf, None, None
    candidates = []
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed))
        model.fit(matrices["train"], labels["train"])
        val = metadata["val"].copy()
        val["prob_AV"] = model.predict_proba(matrices["val"])[:, 1]
        val["prediction"] = (val.prob_AV >= 0.5).astype(int)
        wm, tm, _, _ = evaluate_predictions(val)
        score = wm["auroc"]  # Same selection level as existing Transformer.
        candidates.append({"C": c, "window_auroc": score, "trajectory_auroc": tm["auroc"]})
        if score > best_score:
            best_score, best_model, best_c = score, model, c
    pred = metadata["test"].copy()
    pred["prob_AV"] = best_model.predict_proba(matrices["test"])[:, 1]
    pred["prediction"] = (pred.prob_AV >= 0.5).astype(int)
    wm, tm, sm, tp = evaluate_predictions(pred)
    pred.to_csv(folder / "test_predictions.csv", index=False)
    tp.to_csv(folder / "trajectory_predictions.csv", index=False)
    result = dict(fold=fold["name"], window_metrics=wm, trajectory_metrics=tm,
                  source_metrics=sm, selected_C=best_c, validation_candidates=candidates,
                  feature_cols=columns, statistics=["mean", "std", "q10", "q50", "q90"])
    write_json(folder / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Expected CUDA for this experiment suite; stopping before expensive CPU runs")
    # Excessive CPU thread counts can dominate these small GPU jobs.
    torch.set_num_threads(4)
    variants = configs()
    frames = {p: pd.read_csv(p) for p in {c["data"]["processed_csv"] for c in variants.values()}}
    write_json(root / "protocol.json", {
        "seeds": args.seeds, "device": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__), "eval_stride": 10,
        "selection": "validation window AUROC; threshold=0.5",
        "scope": "four main variants on every seed; diagnostic variants on first seed",
        "seed_semantics": "seed + fold index controls both inner split and initialization",
        "interpretation": "exploratory reuse of outer folds; dates remain confounded with labels",
        "configs": variants,
    })
    rows = []
    start = time.time()
    for seed in args.seeds:
        names = list(variants) if seed == args.seeds[0] else list(variants)[:4]
        for name in names + ["logistic_all", "logistic_leader"]:
            folder = root / name / f"seed_{seed}"
            cfg = copy.deepcopy(variants.get(name, variants["baseline"]))
            cfg["training"]["seed"] = seed
            cfg["output"].update(checkpoint_dir=str(folder / "checkpoints"),
                                 log_dir=str(folder / "logs"),
                                 feature_stats_path=str(folder / "feature_stats.json"))
            write_json(folder / "config.json", cfg)
            print(f"\nEXPERIMENT {name} seed={seed}", flush=True)
            df = frames[cfg["data"]["processed_csv"]]
            results = []
            for index, fold in enumerate(cfg["cross_validation"]["folds"]):
                splits = make_source_holdout_splits(cfg, df, fold["test_sources"], seed + index)
                assert not (set(splits["train"]) & set(splits["val"]))
                assert not (set(splits["test"]) & (set(splits["train"]) | set(splits["val"])))
                fold_dir = folder / "logs" / fold["name"]
                write_json(fold_dir / "splits.json", splits)
                if name.startswith("logistic"):
                    columns = FEATURE_COLS if name == "logistic_all" else ["v_LV_long", "a_LV"]
                    result = logistic_fold(cfg, df, fold, index, columns, fold_dir)
                else:
                    result = train_fold(cfg, df, fold, index, device)
                results.append(result)
                row = dict(variant=name, seed=seed, fold=fold["name"],
                           best_epoch=result.get("best_epoch"), epochs_ran=result.get("epochs_ran"))
                for level in ("window_metrics", "trajectory_metrics"):
                    for metric in ("auroc", "balanced_accuracy", "f1", "auprc"):
                        row[f"{level}_{metric}"] = result[level][metric]
                rows.append(row)
                pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
            write_json(folder / "summary.json", aggregate_results(results))
            print(f"COMPLETE {name} seed={seed}; elapsed={time.time()-start:.1f}s", flush=True)
    write_json(root / "completion.json", {"elapsed_seconds": time.time()-start, "fold_runs": len(rows)})
    print(pd.DataFrame(rows).groupby("variant")[["window_metrics_auroc", "trajectory_metrics_auroc",
                                              "trajectory_metrics_balanced_accuracy"]].mean().to_string())


if __name__ == "__main__":
    main()
