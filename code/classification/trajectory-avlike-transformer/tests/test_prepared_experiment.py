import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepared_experiment import (PARTS, auxiliary_sampling_weights, fit_stats,
                                 fit_temperature, make_split, normalize,
                                 probability_metrics)
from train import build_model


class PreparedExperimentTests(unittest.TestCase):
    def manifest(self):
        rows = []
        for part in PARTS:
            for label in [0, 1]:
                for pair in [1, 2]:
                    for start in [0, 10, 40]:
                        rows.append(dict(campaign="Vicolungo", label=label,
                                         source_id=f"Vicolungo_{label}_{part}",
                                         pair_id=pair, start_in_segment=start))
        for campaign in ["ASta", "Casale"]:
            rows.append(dict(campaign=campaign, label=1, source_id=campaign,
                             pair_id=1, start_in_segment=0))
        return pd.DataFrame(rows)

    def test_source_isolation_and_nonoverlapping_training_windows(self):
        manifest = self.manifest()
        test_ids = []
        for fold in range(3):
            split = make_split(manifest, fold)
            sources = {k: set(manifest.iloc[split[k]].source_id)
                       for k in ["train", "val", "test"]}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                self.assertFalse(sources[a] & sources[b])
            self.assertTrue(manifest.iloc[split["train"]].start_in_segment.mod(40).eq(0).all())
            test_ids.extend(split["test"])
        self.assertEqual(len(test_ids), len(set(test_ids)))
        self.assertEqual(len(test_ids), int(manifest.campaign.eq("Vicolungo").sum()))

    def test_auxiliary_mass_preserves_class_prior_and_campaign_budget(self):
        manifest = self.manifest()
        split = make_split(manifest, 0)
        weights = auxiliary_sampling_weights(manifest, split["train"], split["aux"])
        joined = manifest.iloc[np.concatenate([split["train"], split["aux"]])]
        prior = manifest.iloc[split["train"]].label.mean()
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights[joined.label.eq(1)].sum(), prior)
        for campaign in ["ASta", "Casale"]:
            self.assertAlmostEqual(weights[joined.campaign.eq(campaign)].sum(), prior/4)

    def test_normalization_uses_frozen_train_statistics(self):
        rng = np.random.RandomState(42)
        train = rng.normal(size=(8, 40, 12)).astype(np.float32)
        train[:, :, 0] = 2.
        stats = fit_stats(train)
        before = {k: list(v) for k, v in stats.items()}
        heldout = normalize(np.full((2, 40, 12), 1e10), stats)
        self.assertEqual(stats, before)
        self.assertTrue(np.isfinite(heldout).all())
        self.assertTrue(np.all(normalize(train, stats)[:, :, 0] == 0))

    def test_calibration_never_worsens_validation_bce_or_changes_threshold(self):
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        z = np.array([-20., 30., -10., 20.])
        result = fit_temperature(y, z)
        self.assertLessEqual(result["val_bce_after"], result["val_bce_before"] + 1e-12)
        np.testing.assert_array_equal(z >= 0, z / result["temperature"] >= 0)
        self.assertAlmostEqual(probability_metrics(y, np.zeros(4))["bce"], np.log(2))
        self.assertTrue(np.isfinite(probability_metrics(y, z * 1000)["bce"]))

    def test_compact_transformer_backward_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        cfg = dict(architecture="pooled_transformer")
        model = build_model(cfg, "cpu")
        x = torch.randn(3, 40, 12)
        z = model(x)
        self.assertEqual(tuple(z.shape), (3,))
        z.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        restored = build_model(cfg, "cpu")
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(x), restored(x))


if __name__ == "__main__":
    unittest.main()
