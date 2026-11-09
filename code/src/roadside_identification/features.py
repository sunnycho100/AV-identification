"""Eleven motion features at 10 Hz, with explicit leader availability.

All filtering and differentiation are local to the supplied window. A missing leader
never removes an ego observation, and derivatives never cross a leader identity change.
"""

import numpy as np
from scipy.signal import savgol_filter

FEATURE_NAMES = (
    "ego_speed",
    "ego_acceleration",
    "ego_jerk",
    "leader_speed",
    "leader_acceleration",
    "net_gap",
    "relative_speed",
    "closing_speed",
    "relative_acceleration",
    "time_headway",
    "time_to_collision",
)
SAMPLE_INTERVAL_S = 0.1


def smooth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    size = min(7, len(values) if len(values) % 2 else len(values) - 1)
    if size < 3:
        return values.copy()
    return savgol_filter(values, size, 2, axis=0, mode="interp")


def motion_features(
    ego_speed: np.ndarray,
    leader_speed: np.ndarray,
    gap: np.ndarray,
    leader_available: np.ndarray,
    pair_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [time, 11] float32 values and a same-shaped boolean feature mask.

    Inputs use m/s and metres. Missing leader numeric values can be NaN because only
    observations with leader_available=True are read. Supply pair_id whenever a
    leader can change within the window. Crop raw measurements before calling.
    """
    ego = np.asarray(ego_speed, dtype=float)
    leader = np.asarray(leader_speed, dtype=float)
    clearance = np.asarray(gap, dtype=float)
    available = np.asarray(leader_available, dtype=bool)
    size = len(ego)
    if size < 3 or any(a.shape != (size,) for a in (ego, leader, clearance, available)):
        raise ValueError(
            "Motion inputs must be equally sized 1D arrays with at least three samples"
        )
    identities = np.zeros(size, dtype=int) if pair_id is None else np.asarray(pair_id)
    if identities.shape != (size,):
        raise ValueError("pair_id must contain one identity per time sample")
    if not np.isfinite(ego).all() or not np.isfinite(leader[available]).all():
        raise ValueError("Observed speeds must be finite")
    if not np.isfinite(clearance[available]).all() or (clearance[available] <= 0).any():
        raise ValueError("Observed net gaps must be finite and positive")

    values = np.zeros((size, 11))
    mask = np.zeros((size, 11), dtype=bool)
    ego = smooth(ego)
    acceleration = np.gradient(ego, SAMPLE_INTERVAL_S, edge_order=2)
    jerk = np.gradient(acceleration, SAMPLE_INTERVAL_S, edge_order=2)
    values[:, :3] = np.column_stack([ego, acceleration, jerk])
    mask[:, :3] = True
    boundaries = np.r_[
        True, (available[1:] != available[:-1]) | (identities[1:] != identities[:-1])
    ]
    groups = np.cumsum(boundaries)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if not available[indices[0]]:
            continue
        observed_speed = smooth(leader[indices])
        observed_gap = np.maximum(smooth(clearance[indices]), 1e-6)
        relative = observed_speed - ego[indices]
        closing = -relative
        headway = np.clip(observed_gap / np.maximum(ego[indices], 1e-6), 0, 10)
        ttc = np.full(len(indices), 30.0)
        np.divide(observed_gap, closing, out=ttc, where=closing > 1e-6)
        columns = [3, 5, 6, 7, 9, 10]
        values[np.ix_(indices, columns)] = np.column_stack(
            [observed_speed, observed_gap, relative, closing, headway, np.clip(ttc, 0, 30)]
        )
        mask[np.ix_(indices, columns)] = True
        if len(indices) >= 3:
            leader_acceleration = np.gradient(observed_speed, SAMPLE_INTERVAL_S, edge_order=2)
            values[np.ix_(indices, [4, 8])] = np.column_stack(
                [leader_acceleration, leader_acceleration - acceleration[indices]]
            )
            mask[np.ix_(indices, [4, 8])] = True
    return values.astype(np.float32), mask
