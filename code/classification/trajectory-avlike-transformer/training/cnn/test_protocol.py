import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_protocol import features, fit_stats, forecast_examples, splits, joint_weights
from networks import TrajectoryNet


class ProtocolTests(unittest.TestCase):
    def test_prefix_inputs_do_not_see_future(self):
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(4, 40, 3))+[30, 29, 40]
        for mode in ['legacy11', 'primitives3', 'stable5', 'kinematic7', 'ego3', 'centered6']:
            cfg = dict(features=mode, smoothing=11)
            stats = fit_stats(features(raw, mode, 11))
            before, target = forecast_examples(raw, cfg, stats)
            changed = raw.copy(); changed[:, 30:] += 5
            after, target_after = forecast_examples(changed, cfg, stats)
            np.testing.assert_array_equal(before, after)
            self.assertFalse(np.allclose(target, target_after))

    def test_polynomial_derivatives_and_window_isolation(self):
        t = np.arange(40)*.1
        raw = np.stack([25+.5*t, 24+.5*t, 40-t], -1)[None]
        x = features(raw, 'kinematic7', 11)
        np.testing.assert_allclose(x[0, :, 4:6], .5, atol=1e-5)
        batch = np.concatenate([raw, raw+1000])
        np.testing.assert_array_equal(features(batch, 'kinematic7', 11)[0], x[0])

    def test_all_encoders_support_prefix_and_backprop(self):
        torch.set_num_threads(1)
        for arch in ['transformer', 'pooled_transformer', 'conv_transformer', 'multiscale_cnn', 'gru', 'tcn', 'statistics']:
            model = TrajectoryNet(3, arch, 32, .1, 2)
            for length in [30, 40]:
                x = torch.randn(2, length, 3, requires_grad=True)
                z = model(x)
                self.assertEqual(tuple(z.shape), (2,))
                z.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_actual_source_splits_and_expansion(self):
        m = pd.read_csv(Path(json.loads(Path('experiment.json').read_text())['prepared']) / 'windows.csv')
        for fold in range(3):
            s = splits(m, fold)
            self.assertEqual(len(s['external']), 887)
            extra = np.r_[s['external'], s['aux']]
            w = joint_weights(m, s['train'], extra)
            yy = m.iloc[np.r_[s['train'], extra]].label.to_numpy()
            self.assertAlmostEqual(float(w[yy == 1].sum()), float(m.iloc[s['train']].label.mean()), places=6)


if __name__ == '__main__':
    unittest.main()
