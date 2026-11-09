import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from features import FEATURE_COLS as DEFAULT_FEATURE_COLS
from utils import load_json, save_json

FEATURE_COLS = DEFAULT_FEATURE_COLS


def split_trajectory_ids(df, trajectory_id_col, label_col, train_ratio, val_ratio, test_ratio, seed):
    traj_labels = df.groupby(trajectory_id_col)[label_col].first()
    rng = np.random.RandomState(seed)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(traj_labels.unique()):
        ids = sorted(traj_labels[traj_labels == label].index.tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        splits["train"] += ids[:n_train]
        splits["val"] += ids[n_train:n_train + n_val]
        splits["test"] += ids[n_train + n_val:]
    return splits


def compute_norm_stats(df, feature_cols, train_ids, trajectory_id_col, clip_quantiles=None):
    train_df = df[df[trajectory_id_col].isin(train_ids)]
    train_features = train_df[feature_cols]

    clip_lower = None
    clip_upper = None
    if clip_quantiles is not None:
        if len(clip_quantiles) != 2 or not 0 <= clip_quantiles[0] < clip_quantiles[1] <= 1:
            raise ValueError("clip_quantiles must contain two increasing values in [0, 1]")
        clip_lower = train_features.quantile(clip_quantiles[0]).to_numpy(dtype=np.float32)
        clip_upper = train_features.quantile(clip_quantiles[1]).to_numpy(dtype=np.float32)
        train_features = train_features.clip(
            lower=dict(zip(feature_cols, clip_lower)),
            upper=dict(zip(feature_cols, clip_upper)),
            axis="columns",
        )

    mean = train_features.mean().to_numpy(dtype=np.float32)
    std = train_features.std().to_numpy(dtype=np.float32)
    return mean, std, clip_lower, clip_upper


def build_windows(df, trajectory_ids, feature_cols, label_col, trajectory_id_col, time_col, seq_len, stride):
    windows = []
    subset = df[df[trajectory_id_col].isin(trajectory_ids)]
    for traj_id, g in subset.groupby(trajectory_id_col):
        g = g.sort_values(time_col).reset_index(drop=True)
        n = len(g)
        label = g[label_col].iloc[0]
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        times = g[time_col].to_numpy()
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            windows.append(
                {
                    "x": feats[start:end],
                    "y": float(label),
                    "trajectory_id": traj_id,
                    "window_start_time": float(times[start]),
                    "window_end_time": float(times[end - 1]),
                }
            )
    return windows


class TrajectoryWindowDataset(Dataset):
    def __init__(self, windows, mean, std, eps=1e-6, clip_lower=None, clip_upper=None):
        self.windows = windows
        self.mean = mean
        self.std = std
        self.eps = eps
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = w["x"]
        if self.clip_lower is not None:
            x = np.clip(x, self.clip_lower, self.clip_upper)
        x = (x - self.mean) / (self.std + self.eps)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.tensor(w["y"], dtype=torch.float32)
        return x, y


def dataset_diagnostics(windows, split_ids):
    n_windows = len(windows)
    n_av_windows = sum(1 for w in windows if w["y"] == 1)
    n_hv_windows = n_windows - n_av_windows
    return {
        "num_trajectories": len(split_ids),
        "num_windows": n_windows,
        "num_av_windows": n_av_windows,
        "num_hv_windows": n_hv_windows,
    }


def make_dataloaders(config, df, splits, fit_stats=True, stats_path=None):
    data_cfg = config["data"]
    feat_cfg = config["features"]
    train_cfg = config["training"]
    out_cfg = config["output"]
    feature_cols = feat_cfg.get("feature_cols", FEATURE_COLS)

    stats_path = stats_path or out_cfg["feature_stats_path"]
    if fit_stats:
        mean, std, clip_lower, clip_upper = compute_norm_stats(
            df,
            feature_cols,
            splits["train"],
            data_cfg["trajectory_id_col"],
            feat_cfg.get("clip_quantiles"),
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats = {"feature_cols": feature_cols, "mean": mean.tolist(), "std": std.tolist()}
        if clip_lower is not None:
            stats["clip_lower"] = clip_lower.tolist()
            stats["clip_upper"] = clip_upper.tolist()
        save_json(stats, stats_path)
    else:
        stats = load_json(stats_path)
        if stats["feature_cols"] != feature_cols:
            raise ValueError("Feature columns do not match the saved normalization statistics")
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        clip_lower = (
            np.array(stats["clip_lower"], dtype=np.float32) if "clip_lower" in stats else None
        )
        clip_upper = (
            np.array(stats["clip_upper"], dtype=np.float32) if "clip_upper" in stats else None
        )

    datasets = {}
    diagnostics = {}
    for split_name in ["train", "val", "test"]:
        windows = build_windows(
            df,
            splits[split_name],
            feature_cols,
            data_cfg["label_col"],
            data_cfg["trajectory_id_col"],
            data_cfg["time_col"],
            feat_cfg["seq_len"],
            feat_cfg["stride"] if split_name == "train" else feat_cfg.get("eval_stride", feat_cfg["stride"]),
        )
        datasets[split_name] = TrajectoryWindowDataset(
            windows, mean, std, clip_lower=clip_lower, clip_upper=clip_upper
        )
        diagnostics[split_name] = dataset_diagnostics(windows, splits[split_name])

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=train_cfg["batch_size"], shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=train_cfg["batch_size"], shuffle=False),
    }

    return loaders, datasets, diagnostics


def get_dataloaders(config, fit_stats=True):
    data_cfg = config["data"]
    split_cfg = config["split"]
    train_cfg = config["training"]

    df = pd.read_csv(data_cfg["processed_csv"])
    splits = split_trajectory_ids(
        df,
        trajectory_id_col=data_cfg["trajectory_id_col"],
        label_col=data_cfg["label_col"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=train_cfg["seed"],
    )
    return make_dataloaders(config, df, splits, fit_stats=fit_stats)
