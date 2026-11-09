"""Immutable source-model inference with explicit data contracts and provenance."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .io import read_json, save_json, sha256
from .metrics import classification_metrics
from .model import SourceTransformer
from .workspace import Workspace


def validate_arrays(
    features, feature_mask, valid_time
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features)
    mask = np.asarray(feature_mask, dtype=bool)
    valid = np.asarray(valid_time, dtype=bool)
    if (
        values.ndim != 3
        or not len(values)
        or values.shape[2] != 11
        or not 5 <= values.shape[1] <= 40
    ):
        raise ValueError("Expected a nonempty [windows, 5..40 samples, 11 features] array")
    if mask.shape != values.shape or valid.shape != values.shape[:2]:
        raise ValueError("Feature mask and time mask must match input dimensions")
    if (valid.sum(1) < 5).any() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Real time samples must form a prefix of at least five samples")
    if not np.array_equal(mask[:, :, :3], np.repeat(valid[:, :, None], 3, axis=2)):
        raise ValueError("All three ego features must be observed at every real time sample")
    mask = mask & valid[:, :, None]
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed feature values must be finite")
    return values, mask, valid


class FrozenClassifier:
    """Load a trusted project checkpoint once; never optimize or calibrate on target data."""

    def __init__(self, workspace: Workspace, config: dict):
        self.workspace = workspace
        self.config = config
        self.path = workspace.path(config["checkpoint"])
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Classifier checkpoint is not bundled: {self.path}. "
                "See code/docs/GETTING_STARTED.md and configure a trusted local checkpoint."
            )
        expected = config.get("checkpoint_sha256")
        if expected and sha256(self.path) != expected:
            raise ValueError("Checkpoint hash differs from the configured source model")
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        architecture = checkpoint["config"]
        if architecture.get("architecture", "transformer") != "transformer":
            raise ValueError("The maintained default supports Transformer checkpoints only")
        self.model = SourceTransformer(
            architecture["width"], architecture["dropout"], architecture["layers"]
        ).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.requires_grad_(False)
        stats_path = config.get("feature_stats")
        self.stats = read_json(
            workspace.path(stats_path) if stats_path else self.path.parent / "feature_stats.json"
        )
        for name in ("mean", "std"):
            if (
                np.asarray(self.stats[name]).shape != (11,)
                or not np.isfinite(self.stats[name]).all()
            ):
                raise ValueError(f"Invalid source {name}: expected eleven finite values")
        if (np.asarray(self.stats["std"]) <= 0).any():
            raise ValueError("Source standard deviations must be positive")

    @torch.no_grad()
    def predict(self, features, mask, valid, clipped: bool = False) -> np.ndarray:
        """Return one probability per window; labels are not accepted by this API."""
        values, mask, valid = validate_arrays(features, mask, valid)
        if clipped:
            values = np.clip(values, self.stats["lower"], self.stats["upper"])
        normalized = (values - np.asarray(self.stats["mean"])) / np.asarray(self.stats["std"])
        normalized = np.where(mask, normalized, 0).astype(np.float32)
        probabilities = []
        for start in range(0, len(values), 256):
            logits = self.model(
                torch.from_numpy(normalized[start : start + 256]),
                torch.from_numpy(valid[start : start + 256]),
            )
            probabilities.append(torch.sigmoid(logits).numpy())
        return np.concatenate(probabilities)


def classify_dataset(
    workspace: Workspace,
    config: dict,
    dataset: str | Path,
    metadata: str | Path,
    output: str | Path,
) -> dict:
    """Classify prepared windows, average within video, and optionally evaluate labels."""
    dataset_path = workspace.path(dataset)
    metadata_path = workspace.path(metadata)
    if not dataset_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Prepared NPZ/CSV inputs are missing. See code/docs/DATA_CONTRACTS.md."
        )
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("features", "feature_mask", "valid_tokens")}
    rows = pd.read_csv(metadata_path)
    if "clip" not in rows or rows["clip"].isna().any() or len(rows) != len(arrays["features"]):
        raise ValueError("Metadata needs one nonempty clip ID for every NPZ window")
    threshold = float(config.get("threshold", 0.5))
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between zero and one")
    model = FrozenClassifier(workspace, config)
    rows["probability"] = model.predict(
        arrays["features"], arrays["feature_mask"], arrays["valid_tokens"]
    )
    video = rows.groupby("clip", sort=False).agg(
        probability=("probability", "mean"), windows=("probability", "size")
    )
    video["predicted_label"] = (video.probability >= threshold).astype(int)
    report = {
        "protocol": config,
        "trained_parameters": 0,
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(metadata_path),
        "checkpoint_sha256": sha256(model.path),
    }
    if "label" in rows:
        if not rows.groupby("clip").label.nunique().eq(1).all():
            raise ValueError("All windows from a video must share one label")
        video["label"] = rows.groupby("clip", sort=False).label.first()
        video["correct"] = video.predicted_label == video.label
        report["metrics"] = classification_metrics(video.label, video.probability, threshold)
        if "HV_T_EW_2" in video.index and len(video) > 1:
            subset = video.drop(index="HV_T_EW_2")
            report["excluding_HV_T_EW_2_descriptive"] = classification_metrics(
                subset.label, subset.probability, threshold
            )
    destination = workspace.path(output)
    destination.mkdir(parents=True, exist_ok=True)
    rows.to_csv(destination / "window_predictions.csv", index=False)
    video.to_csv(destination / "video_predictions.csv")
    save_json(destination / "report.json", report)
    return report
