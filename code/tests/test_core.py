"""Behavioral contracts for public interfaces; no field data or pretrained weights."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from roadside_identification.features import motion_features
from roadside_identification.metrics import classification_metrics
from roadside_identification.workspace import Workspace


class MotionFeatureTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(26) * 0.1
        self.ego = 12 + 0.5 * self.t
        self.leader = self.ego + 1
        self.gap = 20 + self.t

    def test_constant_acceleration_units(self):
        values, mask = motion_features(self.ego, self.leader, self.gap, np.ones(26, bool))
        np.testing.assert_allclose(values[:, 1], 0.5, atol=1e-5)
        np.testing.assert_allclose(values[:, 2], 0, atol=1e-5)
        self.assertTrue(mask.all())

    def test_hidden_leader_values_cannot_leak_into_features(self):
        available = np.ones(26, bool)
        available[7:15] = False
        original = motion_features(self.ego, self.leader, self.gap, available)
        corrupted_speed, corrupted_gap = self.leader.copy(), self.gap.copy()
        corrupted_speed[~available] = np.nan
        corrupted_gap[~available] = 1e20
        changed = motion_features(self.ego, corrupted_speed, corrupted_gap, available)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])

    def test_leader_identity_changes_break_derivatives(self):
        leader = np.r_[np.full(13, 12.0), np.full(13, 25.0)]
        values, mask = motion_features(
            self.ego, leader, self.gap, np.ones(26, bool), np.r_[np.zeros(13), np.ones(13)]
        )
        np.testing.assert_allclose(values[:, 4], 0, atol=1e-5)
        self.assertTrue(mask[:, 4].all())

    def test_absent_leader_keeps_ego_features(self):
        values, mask = motion_features(
            self.ego, np.full(26, np.nan), np.full(26, np.nan), np.zeros(26, bool)
        )
        self.assertTrue(mask[:, :3].all())
        self.assertFalse(mask[:, 3:].any())
        self.assertTrue((values[:, 3:] == 0).all())

    def test_invalid_observed_gap_rejected(self):
        with self.assertRaises(ValueError):
            motion_features(self.ego, self.leader, -self.gap, np.ones(26, bool))


class WorkspaceTests(unittest.TestCase):
    def test_explicit_workspace_outside_current_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project with spaces" / "code"
            (root / "configs").mkdir(parents=True)
            (root / "configs/workspace.json").write_text("{}")
            workspace = Workspace.discover(root.parent)
            self.assertEqual(workspace.root, root.resolve())
            self.assertEqual(workspace.path("data\\sample.json"), root / "data/sample.json")

    def test_metrics_reject_inconsistent_labels(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 2], [0.1, 0.9])
        result = classification_metrics([0, 0, 1], [0.1, 0.8, 0.7])
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["balanced_accuracy"], 0.75)


try:
    import torch
    from roadside_identification.classifier import FrozenClassifier, classify_dataset
    from roadside_identification.model import SourceTransformer
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the classifier extra for inference tests")
class ClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        model = SourceTransformer(width=16, dropout=0, layers=1)
        torch.save(
            {
                "config": {"architecture": "transformer", "width": 16, "dropout": 0, "layers": 1},
                "model_state_dict": model.state_dict(),
            },
            self.root / "model.pt",
        )
        (self.root / "feature_stats.json").write_text(
            json.dumps(
                {
                    "mean": [0] * 11,
                    "std": [1] * 11,
                    "lower": [-100] * 11,
                    "upper": [100] * 11,
                }
            )
        )
        self.config = {"checkpoint": "model.pt", "threshold": 0.5}
        self.model = FrozenClassifier(Workspace(self.root), self.config)
        self.values = np.random.default_rng(4).normal(size=(1, 26, 11)).astype(np.float32)
        self.mask = np.ones_like(self.values, bool)
        self.valid = np.ones((1, 26), bool)

    def test_padding_is_not_observed_time(self):
        self.mask[:, 11:] = False
        self.valid[:, 11:] = False
        padded = self.model.predict(self.values, self.mask, self.valid)
        short = self.model.predict(self.values[:, :11], self.mask[:, :11], self.valid[:, :11])
        np.testing.assert_allclose(padded, short, atol=1e-6)

    def test_masked_values_and_frozen_parameters(self):
        self.mask[:, 7:18, 3:] = False
        original = self.model.predict(self.values, self.mask, self.valid)
        self.values[~self.mask] = np.nan
        changed = self.model.predict(self.values, self.mask, self.valid)
        np.testing.assert_array_equal(original, changed)
        self.assertFalse(any(p.requires_grad for p in self.model.model.parameters()))

    def test_internal_time_holes_are_rejected(self):
        self.valid[:, 10] = False
        self.mask[:, 10] = False
        with self.assertRaises(ValueError):
            self.model.predict(self.values, self.mask, self.valid)

    def test_checkpoint_integrity_is_enforced(self):
        with self.assertRaises(ValueError):
            FrozenClassifier(Workspace(self.root), {**self.config, "checkpoint_sha256": "invalid"})

    def test_labels_change_metrics_not_predictions(self):
        import pandas as pd

        np.savez(
            self.root / "input.npz",
            features=self.values,
            feature_mask=self.mask,
            valid_tokens=self.valid,
        )
        predictions = []
        for label in (0, 1):
            pd.DataFrame({"clip": ["example"], "label": [label]}).to_csv(
                self.root / "input.csv", index=False
            )
            classify_dataset(
                Workspace(self.root), self.config, "input.npz", "input.csv", f"out{label}"
            )
            predictions.append(
                pd.read_csv(self.root / f"out{label}/video_predictions.csv").probability.iloc[0]
            )
        self.assertEqual(*predictions)


if __name__ == "__main__":
    unittest.main()
