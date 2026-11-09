"""Merge raw AV/HV trajectory CSVs and compute the 12-dim behavior features.

Raw files live in a directory (default "converted data/"). Each file's name
prefix ("AV_" or "HV_") is the label source -- there is no label column in
the raw data. Each raw file also numbers Trajectory_ID starting at 0, so IDs
are NOT globally unique across files; this script rewrites them into a
globally unique Trajectory_ID of the form "<source_stem>_<local_id>".
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

RAW_DROP_COLS = ["ID_LV", "Speed_LV", "ID_FV", "Speed_FV"]

FEATURE_COLS = [
    "v_FV_long",
    "a_FV",
    "jerk_FV",
    "v_LV_long",
    "a_LV",
    "gap",
    "lateral_offset",
    "delta_v",
    "closing_speed",
    "delta_acc",
    "THW",
    "TTC",
]


def label_from_filename(path):
    name = os.path.basename(path)
    if name.startswith("AV_"):
        return 1
    if name.startswith("HV_"):
        return 0
    raise ValueError(f"Cannot infer AV/HV label from filename: {name}")


def load_raw(raw_dir):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        label = label_from_filename(path)
        source = os.path.splitext(os.path.basename(path))[0]

        df["label"] = label
        df["source_file"] = source
        df["Trajectory_ID"] = source + "_" + df["Trajectory_ID"].astype(str)

        df = df.drop(columns=RAW_DROP_COLS)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["Trajectory_ID", "Time_Index"]).reset_index(drop=True)
    return merged


def _smooth_velocity_components(df, window):
    columns = ["Speed_x_FV", "Speed_y_FV", "Speed_x_LV", "Speed_y_LV"]
    for col in columns:
        df[col] = df.groupby("Trajectory_ID", sort=False)[col].transform(
            lambda values: values.rolling(window=window, center=True, min_periods=1).mean()
        )
    return df


def _differentiate(df, value_col, dt, eps, use_actual_dt):
    groups = df.groupby("Trajectory_ID", sort=False)
    delta = groups[value_col].diff()
    if use_actual_dt:
        elapsed = groups["Time_Index"].diff()
        elapsed = elapsed.where(elapsed > eps)
        derivative = delta / elapsed
    else:
        derivative = delta / dt
    return derivative.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_features(
    df,
    eps=1e-6,
    dt=0.1,
    ttc_cap=30.0,
    thw_cap=10.0,
    smooth=False,
    smooth_window=5,
    use_actual_dt=False,
):
    df = df.copy()

    if smooth:
        if smooth_window < 1 or smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer")
        df = _smooth_velocity_components(df, smooth_window)

    v_fv_mag = np.sqrt(df["Speed_x_FV"] ** 2 + df["Speed_y_FV"] ** 2)
    e_x = df["Speed_x_FV"] / (v_fv_mag + eps)
    e_y = df["Speed_y_FV"] / (v_fv_mag + eps)
    n_x = -e_y
    n_y = e_x

    dx = df["processed_x_LV"] - df["processed_x_FV"]
    dy = df["processed_y_LV"] - df["processed_y_FV"]

    gap = dx * e_x + dy * e_y
    lateral_offset = dx * n_x + dy * n_y

    v_fv_long = df["Speed_x_FV"] * e_x + df["Speed_y_FV"] * e_y
    v_lv_long = df["Speed_x_LV"] * e_x + df["Speed_y_LV"] * e_y

    delta_v = v_lv_long - v_fv_long
    closing_speed = v_fv_long - v_lv_long

    thw = gap / (v_fv_long + eps)
    thw = thw.clip(lower=0.0, upper=thw_cap)

    ttc = pd.Series(ttc_cap, index=df.index, dtype=float)
    valid = (closing_speed > eps) & (gap > 0)
    ttc[valid] = gap[valid] / closing_speed[valid]
    ttc = ttc.clip(lower=0.0, upper=ttc_cap)

    df["v_FV_long"] = v_fv_long
    df["v_LV_long"] = v_lv_long
    df["gap"] = gap
    df["lateral_offset"] = lateral_offset
    df["delta_v"] = delta_v
    df["closing_speed"] = closing_speed
    df["THW"] = thw
    df["TTC"] = ttc

    a_fv = _differentiate(df, "v_FV_long", dt, eps, use_actual_dt)
    a_lv = _differentiate(df, "v_LV_long", dt, eps, use_actual_dt)
    df["a_FV"] = a_fv
    df["a_LV"] = a_lv
    df["delta_acc"] = df["a_LV"] - df["a_FV"]

    jerk_fv = _differentiate(df, "a_FV", dt, eps, use_actual_dt)
    df["jerk_FV"] = jerk_fv

    return df


def print_diagnostics(df, feature_cols=FEATURE_COLS):
    n_traj = df["Trajectory_ID"].nunique()
    n_rows = len(df)
    pct_gap_invalid = (df["gap"] <= 0).mean() * 100

    traj_labels = df.groupby("Trajectory_ID")["label"].first()
    n_av_traj = (traj_labels == 1).sum()
    n_hv_traj = (traj_labels == 0).sum()
    n_av_rows = (df["label"] == 1).sum()
    n_hv_rows = (df["label"] == 0).sum()

    print("=== Preprocessing diagnostics ===")
    print(f"Rows: {n_rows}, Trajectories: {n_traj}")
    print(f"AV trajectories: {n_av_traj}, HV trajectories: {n_hv_traj}")
    print(f"AV rows: {n_av_rows}, HV rows: {n_hv_rows}")
    print(f"gap <= 0: {pct_gap_invalid:.2f}%")
    print()
    print(df[feature_cols].agg(["min", "max", "mean", "std"]).T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer_baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    feat_cfg = config["features"]

    raw = load_raw(data_cfg["raw_dir"])
    feats = compute_features(
        raw,
        eps=feat_cfg["eps"],
        dt=feat_cfg["dt"],
        ttc_cap=feat_cfg["ttc_cap"],
        thw_cap=feat_cfg["thw_cap"],
        smooth=feat_cfg.get("smooth", False),
        smooth_window=feat_cfg.get("smooth_window", 5),
        use_actual_dt=feat_cfg.get("use_actual_dt", False),
    )

    feature_cols = feat_cfg.get("feature_cols", FEATURE_COLS)
    unknown = sorted(set(feature_cols) - set(FEATURE_COLS))
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")

    out_cols = ["Trajectory_ID", "Time_Index", "label"] + feature_cols
    processed = feats[out_cols]

    out_path = data_cfg["processed_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processed.to_csv(out_path, index=False)

    print_diagnostics(feats, feature_cols)
    print(f"\nSaved processed features to {out_path}")


if __name__ == "__main__":
    main()
