"""Source-disjoint experiments on strictly window-local, prepared 4-second data.

This is a within-date source holdout, NOT a date-independent evaluation: only
one Human date exists. Test sets never participate in fitting or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn
import torch
import torch.nn.functional as F

from train import build_model
from utils import compute_metrics, set_seed


VARIANTS = ("transformer", "pooled_transformer", "temporal_cnn", "cnn_aux", "histogram_tree")
PARTS = ("part2", "part3", "part4_highway")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_prepared(root):
    root = Path(root)
    manifest = pd.read_csv(root / "windows.csv")
    with np.load(root / "windows_4s.npz", allow_pickle=False) as data:
        x, y = data["X"], data["y"].astype(np.float32)
        names = data["feature_names"].tolist()
        if not (np.array_equal(data["window_index"], manifest.window_index)
                and np.array_equal(y, manifest.label)
                and np.array_equal(data["split_group"], manifest.split_group)):
            raise ValueError("NPZ and manifest identities/labels do not match")
    if x.shape != (len(manifest), 40, len(names)) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,40,F] input")
    if manifest.window_index.duplicated().any():
        raise ValueError("Duplicate window identities")
    return x, y, manifest, names


def make_split(manifest, fold_index, train_stride=40):
    if train_stride < 10 or train_stride % 10:
        raise ValueError("train_stride must be a positive multiple of prepared stride 10")
    split = {}
    # Three fixed cyclic assignments, shared across all variants and seeds.
    for key, offset in (("test", 0), ("val", 1), ("train", 2)):
        part = PARTS[(fold_index + offset) % 3]
        mask = ((manifest.campaign == "Vicolungo")
                & manifest.source_id.str.endswith("_" + part))
        if key == "train":
            mask &= manifest.start_in_segment.mod(train_stride).eq(0)
        split[key] = np.flatnonzero(mask)
        if set(manifest.iloc[split[key]].label) != {0, 1}:
            raise ValueError(f"{key} requires both labels")
    sources = {k: set(manifest.iloc[v].source_id) for k, v in split.items()}
    if any(sources[a] & sources[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("A source crosses data splits")
    split["aux"] = np.flatnonzero(
        manifest.campaign.isin(["ASta", "Casale"])
        & manifest.start_in_segment.mod(train_stride).eq(0))
    return split


def fit_stats(x):
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    lower, upper = np.quantile(flat, [0.005, 0.995], axis=0)
    clipped = np.clip(flat, lower, upper)
    return dict(lower=lower.tolist(), upper=upper.tolist(),
                mean=clipped.mean(0).tolist(), std=np.maximum(clipped.std(0), 1e-6).tolist())


def normalize(x, stats):
    return ((np.clip(x, stats["lower"], stats["upper"]) - stats["mean"])
            / np.array(stats["std"])).astype(np.float32)


def probability_metrics(y, z):
    z = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite predictions")
    p = expit(z)
    result = compute_metrics(y, p)
    result.update(bce=float(np.mean(np.logaddexp(0, z) - y * z)),
                  brier=float(np.mean((y-p)**2)),
                  human_recall=float(np.mean(z[y == 0] < 0)),
                  acc_recall=float(np.mean(z[y == 1] >= 0)))
    return result


def fit_temperature(y, z):
    z = np.asarray(z, dtype=np.float64)
    def loss(log_t):
        scaled = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, scaled) - y * scaled))
    fit = minimize_scalar(loss, bounds=(np.log(.1), np.log(20)), method="bounded")
    best = min((0., np.log(.1), np.log(20.), float(fit.x)), key=loss)
    return dict(temperature=float(np.exp(best)), fit_split="val",
                val_bce_before=loss(0.), val_bce_after=loss(best))


def auxiliary_sampling_weights(manifest, train_ids, aux_ids):
    """Keep core class prior and update count; allocate half of ACC mass to aux."""
    core = manifest.iloc[train_ids]
    aux = manifest.iloc[aux_ids]
    if len(aux) == 0 or not aux.label.eq(1).all():
        raise ValueError("Auxiliary data must be present and ACC-only")
    labels = core.label.to_numpy()
    prior = labels.mean()
    w = np.empty(len(core) + len(aux), dtype=np.float64)
    w[:len(core)] = np.where(labels == 0, 1 / len(core), .5 / len(core))
    for campaign in ("ASta", "Casale"):
        mask = aux.campaign.to_numpy() == campaign
        if not mask.any():
            raise ValueError(f"Missing auxiliary campaign {campaign}")
        w[len(core) + np.flatnonzero(mask)] = .25 * prior / mask.sum()
    if not np.isclose(w.sum(), 1.):
        raise ValueError("Invalid sampling mass")
    return w


@torch.no_grad()
def infer(model, x, batch_size=256):
    model.eval()
    return torch.cat([model(batch) for batch in x.split(batch_size)]).cpu().numpy()


def model_config(variant):
    if variant == "transformer":
        return dict(architecture="transformer", input_dim=12, d_model=128,
                    num_layers=2, num_heads=4, ffn_dim=512, seq_len=40, dropout=.1)
    if variant == "pooled_transformer":
        return dict(architecture="pooled_transformer", input_dim=12, d_model=32,
                    num_heads=4, ffn_dim=64, seq_len=40, dropout=.3)
    return dict(architecture="temporal_cnn", input_dim=12, channels=32, dropout=.3)


def improved(metric, current, best):
    # Strict improvement keeps the earliest epoch on exact ties.
    return current < best - 1e-8 if metric == "bce" else current > best + 1e-8


def window_statistics(x):
    return np.concatenate([x.mean(1), x.std(1),
                           *np.quantile(x, [.1, .5, .9], axis=1)], axis=1)


def run_fold(x, y, manifest, names, split, seed, fold_index, variant, args, root):
    path = root / variant / f"seed_{seed}" / PARTS[fold_index]
    path.mkdir(parents=True, exist_ok=False)
    set_seed(seed + fold_index)
    start = time.time()
    stats = fit_stats(x[split["train"]])
    stats["feature_names"] = names
    stats["fit_split"] = "core_train_only"
    write_json(path / "feature_stats.json", stats)
    write_json(path / "splits.json", {
        k: dict(window_indices=manifest.iloc[v].window_index.tolist(),
                sources=sorted(manifest.iloc[v].source_id.unique().tolist()),
                class_counts={str(int(a)): int(b) for a, b in
                              manifest.iloc[v].label.value_counts().items()})
        for k, v in split.items() if k != "aux" or variant == "cnn_aux"})
    best = {k: dict(value=float("inf") if k == "bce" else -1., epoch=0)
            for k in ("bce", "accuracy", "auroc")}
    history = []
    candidates = {}
    val_y, test_y = y[split["val"]], y[split["test"]]
    if variant == "histogram_tree":
        from joblib import dump
        sx = {k: window_statistics(normalize(x[split[k]], stats))
              for k in ("train", "val", "test")}
        cfg = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
                   l2_regularization=10., learning_rate=.05,
                   early_stopping=False, random_state=seed+fold_index)
        estimator = HistGradientBoostingClassifier(**cfg).fit(sx["train"], y[split["train"]])
        prior = float(y[split["train"]].mean())
        val_stages = [np.full(len(val_y), np.log(prior/(1-prior)))]
        val_stages.extend(np.log(np.clip(p[:, 1], 1e-12, 1-1e-12))
                          - np.log(np.clip(p[:, 0], 1e-12, 1-1e-12))
                          for p in estimator.staged_predict_proba(sx["val"]))
        for epoch, z in enumerate(val_stages):
            metrics = probability_metrics(val_y, z)
            history.append(dict(epoch=epoch, train_loss=None, val=metrics))
            for key in best:
                if improved(key, metrics[key], best[key]["value"]):
                    best[key] = dict(value=metrics[key], epoch=epoch)
        # Access test predictions only after validation selection is complete.
        test_stages = [np.full(len(test_y), np.log(prior/(1-prior)))]
        test_stages.extend(np.log(np.clip(p[:, 1], 1e-12, 1-1e-12))
                           - np.log(np.clip(p[:, 0], 1e-12, 1-1e-12))
                           for p in estimator.staged_predict_proba(sx["test"]))
        for key, info in best.items():
            candidates[key] = (val_stages[info["epoch"]], test_stages[info["epoch"]])
        initial = test_stages[0]
        dump(estimator, path / "all_iterations.joblib")
        parameters = None
    else:
        cfg = model_config(variant)
        train_ids = (np.concatenate([split["train"], split["aux"]])
                     if variant == "cnn_aux" else split["train"])
        tx = torch.from_numpy(normalize(x[train_ids], stats)).to(args.device)
        ty = torch.from_numpy(y[train_ids]).to(args.device)
        vx = torch.from_numpy(normalize(x[split["val"]], stats)).to(args.device)
        model = build_model(cfg, args.device)
        parameters = sum(p.numel() for p in model.parameters())
        initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        baseline = variant == "transformer"
        smoothing, wd = (0., .0001) if baseline else (.05, .01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=wd)
        core_y = y[split["train"]]
        pw = torch.tensor(float((core_y == 0).sum() / (core_y == 1).sum())
                          if baseline else 1., device=args.device)
        weights = (torch.tensor(auxiliary_sampling_weights(manifest, split["train"], split["aux"]),
                                device=args.device)
                   if variant == "cnn_aux" else None)
        for epoch in range(args.epochs + 1):
            loss_value = None
            if epoch:
                model.train()
                order = (torch.multinomial(weights, len(core_y), replacement=True)
                         if weights is not None else torch.randperm(len(core_y), device=args.device))
                total = 0.
                for ids in order.split(args.batch_size):
                    logits = model(tx[ids])
                    targets = ty[ids] * (1 - smoothing) + .5 * smoothing
                    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    total += float(loss.detach()) * len(ids)
                loss_value = total / len(order)
            metrics = probability_metrics(val_y, infer(model, vx))
            history.append(dict(epoch=epoch, train_loss=loss_value, val=metrics))
            for key in best:
                if improved(key, metrics[key], best[key]["value"]):
                    best[key] = dict(value=metrics[key], epoch=epoch)
                    torch.save(dict(model_state_dict=model.state_dict(), model=cfg,
                                    epoch=epoch, selection=key), path / f"best_{key}.pt")
            if epoch % 10 == 0:
                print(f"{variant} seed={seed} {PARTS[fold_index]} epoch={epoch} "
                      f"val_bce={metrics['bce']:.4f} val_acc={metrics['accuracy']:.4f}", flush=True)
            if all(epoch - v["epoch"] >= args.patience for v in best.values()):
                break
        # No test inference until checkpoint selection has finished.
        test_x = torch.from_numpy(normalize(x[split["test"]], stats)).to(args.device)
        model.load_state_dict(initial_state)
        initial = infer(model, test_x)
        for key in best:
            ckpt = torch.load(path / f"best_{key}.pt", map_location=args.device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            candidates[key] = (infer(model, vx), infer(model, test_x))
        cfg = dict(model=cfg, smoothing=smoothing, weight_decay=wd, lr=.0003,
                   pos_weight=float(pw), gradient_clip=1., updates_per_epoch=int(np.ceil(len(core_y)/args.batch_size)))
    write_json(path / "config.json", cfg)
    write_json(path / "history.json", history)
    write_json(path / "selection.json", best)
    rows = []
    predictions = manifest.iloc[split["test"]].copy()
    def record(selection, logits, epoch=0, temperature=1.):
        metrics = probability_metrics(test_y, logits)
        rows.append(dict(variant=variant, seed=seed, fold=PARTS[fold_index], selection=selection,
                         epoch=epoch, temperature=temperature, parameters=parameters,
                         **{k: v for k, v in metrics.items() if k != "confusion_matrix"}))
        predictions[selection + "_logit"] = logits
    record("initial", initial)
    record("constant_half", np.zeros(len(test_y)))
    prior = float(y[split["train"]].mean())
    record("train_prior", np.full(len(test_y), np.log(prior/(1-prior))))
    calibration = {}
    for key, (vz, tz) in candidates.items():
        record(key, tz, best[key]["epoch"])
        calibration[key] = fit_temperature(val_y, vz)
        temperature = calibration[key]["temperature"]
        record(key + "_temperature", tz / temperature, best[key]["epoch"], temperature)
    write_json(path / "calibration.json", calibration)
    write_json(path / "metrics.json", rows)
    write_json(path / "runtime.json", dict(seconds=time.time()-start))
    predictions.to_csv(path / "test_predictions.csv", index=False)
    print(f"DONE {variant} seed={seed} {PARTS[fold_index]} "
          f"seconds={time.time()-start:.1f} best={best}", flush=True)
    return rows


def summarize(root):
    rows = []
    for path in root.glob("*/seed_*/*/metrics.json"):
        rows.extend(json.loads(path.read_text()))
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "comparison.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "bce", "brier", "auroc", "human_recall", "acc_recall"]
    means = frame.groupby(["variant", "selection", "seed"])[metrics].mean()
    means.to_csv(root / "seed_means.csv")
    means.groupby(["variant", "selection"]).agg(["mean", "std"]).to_csv(root / "overview.csv")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default="data/prepared_20260921")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--folds", type=int, nargs="+", choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    x, y, manifest, names = load_prepared(args.prepared)
    protocol = dict(args=vars(args), features=names, python=platform.python_version(),
                    torch=str(torch.__version__), numpy=np.__version__, sklearn=sklearn.__version__,
                    primary="4-second accuracy, fixed threshold 0.5; checkpoint selected by validation BCE",
                    diagnostics="same-run accuracy/AUROC checkpoints; validation-only temperature calibration",
                    limitations="within-date source holdout; Human has only one date; windows are correlated",
                    normalization="0.5/99.5% clipping and mean/std fitted to core training windows only",
                    auxiliary="ACC only, training only, core class prior and update count preserved in expectation",
                    split="cyclic: test=fold, val=fold+1, train=fold+2; all pairs of each source stay together",
                    data_sha256={p: hashlib.sha256((Path(args.prepared)/p).read_bytes()).hexdigest()
                                 for p in ["windows_4s.npz", "windows.csv"]})
    write_json(root / "protocol.json", protocol)
    shutil.copytree(Path(__file__).parent, root / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    for seed in args.seeds:
        for fold in args.folds:
            split = make_split(manifest, fold, args.train_stride)
            for variant in args.variants:
                run_fold(x, y, manifest, names, split, seed, fold, variant, args, root)
                summarize(root)
    results = summarize(root)
    print(results.groupby(["variant", "selection"])[["accuracy", "bce"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
