"""Video-level binary classification summaries; no parameter fitting."""

import numpy as np


def classification_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty matching 1D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (HV) or 1 (ACC)")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    predicted = probabilities >= threshold
    recalls = [
        float((predicted[labels == k] == k).mean()) if (labels == k).any() else None for k in (0, 1)
    ]
    return {
        "n": len(labels),
        "correct": int((predicted == labels).sum()),
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(np.mean([r for r in recalls if r is not None])),
        "hv_recall": recalls[0],
        "av_recall": recalls[1],
    }
