"""Literature classifier families adapted to the project's native 11-feature protocol."""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import torch
from torch import nn

from data_protocol import PARTS, splits, features, fit_stats, normalize, joint_weights
from run import save, metrics, temperature


class NativeLSTM(nn.Module):
    """Ordinary stacked LSTM, last-step readout; all eleven input channels retained."""
    def __init__(self, input_dim, width, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([nn.LSTM(input_dim if i == 0 else width,
                                           width, batch_first=True) for i in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(layer(x)[0])
        return self.head(x[:, -1]).squeeze(-1)


def training_pool(m, split, cfg):
    ids = split['train']
    mass = np.full(len(ids), 1 / len(ids))
    if cfg['data_mode'] == 'expanded':
        extra = np.r_[split['external'], split['aux']]
        mass = joint_weights(m, ids, extra, cfg['external_mass'])
        ids = np.r_[ids, extra]
    else:
        assert cfg['data_mode'] == 'core'
    prior = float(m.iloc[split['train']].label.mean())
    label = m.iloc[ids].label.to_numpy().astype(int)
    class_weights = np.array([.5 / (1 - prior), .5 / prior])
    weights = mass * class_weights[label]
    # Same expected class/source mass as the existing NN sampler and weighted BCE.
    return ids, weights * len(weights) / weights.sum(), mass


class WeightedKNN:
    """k-neighbor voting with native class/source observation weights."""
    def __init__(self, n_neighbors=15, distance=False):
        self.n_neighbors = n_neighbors
        self.distance = distance

    def fit(self, x, y, sample_weight):
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean',
                                   algorithm='brute', n_jobs=4).fit(x)
        self.y = np.asarray(y, dtype=float)
        self.sample_weight = np.asarray(sample_weight, dtype=float)
        return self

    def predict_proba(self, x):
        result = []
        for start in range(0, len(x), 512):
            distance, ids = self.nn.kneighbors(x[start:start + 512])
            kernel = np.ones_like(distance)
            if self.distance:
                kernel = 1 / np.maximum(distance, 1e-12)
                zero = distance == 0
                kernel[zero.any(1)] = zero[zero.any(1)]
            w = kernel * self.sample_weight[ids]
            p = (w * self.y[ids]).sum(1) / w.sum(1)
            result.append(np.stack([1 - p, p], axis=1))
        return np.concatenate(result)


def candidates(model):
    if model == 'rf':
        return [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
                for n, d, l in [(15, 8, 1), (300, 8, 1), (300, None, 5)]]
    if model == 'knn':
        return [dict(n_neighbors=k, distance=distance)
                for k in [5, 15, 31] for distance in [False, True]]
    if model == 'svm':
        return [dict(C=c, gamma=g) for c in [.1, 1., 10.] for g in ['scale', .001, .01]]
    raise ValueError(model)


def make_classifier(model, params, seed):
    if model == 'rf':
        return RandomForestClassifier(**params, max_features='sqrt', bootstrap=True,
                                      random_state=seed, n_jobs=4)
    if model == 'svm':
        return SVC(**params, kernel='rbf', probability=True, cache_size=512,
                   random_state=seed)
    return WeightedKNN(**params)


def predict(model, x):
    return logit(np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6))


def run_fit(raw, legacy, y, m, seed, fold, cfg, root):
    start = time.time()
    path = root / f'seed_{seed}' / PARTS[fold]
    path.mkdir(parents=True, exist_ok=False)
    split = splits(m, fold, cfg['train_stride'])
    x = features(raw, cfg['features'], cfg['smoothing'])
    np.testing.assert_allclose(x, legacy, atol=4e-3, rtol=1e-4)
    stats = fit_stats(x[split['train']])
    x = normalize(x, stats).reshape(len(x), -1)
    train_ids, weights, mass = training_pool(m, split, cfg)
    save(path / 'feature_stats.json', dict(**stats, fit_split='core_train', input_dim=11))
    save(path / 'splits.json', {k: dict(window_indices=v.tolist(),
         sources=sorted(m.iloc[v].source_id.unique()),
         class_counts={str(k): int(n) for k,n in m.iloc[v].label.value_counts().items()})
         for k,v in split.items()})
    save(path / 'effective_config.json', cfg)
    save(path / 'supervised_pool.json', dict(window_indices=train_ids.tolist(),
         campaign_counts=m.iloc[train_ids].campaign.value_counts().to_dict(),
         sample_weight=weights.tolist(), sampling_mass=mass.tolist()))
    trials, best, winner, selected = [], float('inf'), None, None
    for params in candidates(cfg['model']):
        model = make_classifier(cfg['model'], params, seed + fold)
        model.fit(x[train_ids], y[train_ids], sample_weight=weights)
        val = metrics(y[split['val']], predict(model, x[split['val']]))
        trial = dict(params=params, val=val)
        trials.append(trial)
        print('CANDIDATE ' + json.dumps(dict(seed=seed, fold=fold, **trial)), flush=True)
        if val['bce'] < best:
            best, winner, selected = val['bce'], model, params
    save(path / 'validation_search.json', trials)
    joblib.dump(winner, path / 'model.joblib')
    val_z = predict(winner, x[split['val']])
    t = temperature(y[split['val']], val_z)
    save(path / 'calibration.json', dict(temperature=t, fit_split='val', threshold=.5))
    result = dict(seed=seed, fold=fold, fold_name=PARTS[fold], epoch=None,
                  parameters=None, seconds=time.time()-start, selected_params=selected,
                  candidate_fits=len(trials), val=metrics(y[split['val']], val_z),
                  val_temperature=metrics(y[split['val']], val_z/t), evaluations={})
    for role in ['val', 'test', 'external_test']:
        ids = split[role]
        z = val_z if role == 'val' else predict(winner, x[ids])
        pred = m.iloc[ids].copy()
        pred['logit'], pred['calibrated_logit'] = z, z/t
        pred.to_csv(path / f'{role}_predictions.csv', index=False)
        if role != 'val':
            result['evaluations'][role] = dict(raw=metrics(y[ids], z),
                                               calibrated=metrics(y[ids], z/t))
    save(path / 'result.json', result)
    print('RESULT ' + json.dumps(result), flush=True)
    return result


def main():
    cfg = json.loads(Path('experiment.json').read_text())
    assert cfg['include_ngsim'] is False and cfg['features'] == 'legacy11'
    torch.set_num_threads(cfg['threads'])
    root = Path('artifacts/run'); root.mkdir(parents=True, exist_ok=False)
    prepared = Path(cfg['prepared'])
    m = pd.read_csv(prepared / 'windows.csv')
    with np.load(prepared / 'windows_4s.npz', allow_pickle=False) as d:
        raw, legacy, y = d['primitive_windows'], d['X'], d['y'].astype(np.float32)
        np.testing.assert_array_equal(d['window_index'], m.window_index)
        np.testing.assert_array_equal(y, m.label)
    protocol = dict(config=cfg, dataset_sha256={name: hashlib.sha256((prepared/name).read_bytes()).hexdigest()
                    for name in ['windows.csv', 'windows_4s.npz']},
                    scaling='unchanged core-train quantile clipping plus standardization',
                    weight_policy='expected native class/source mass; mean sample weight one',
                    encoding='time-major 40 x 11 flatten',
                    selection_scope='validation BCE only; test never used for search',
                    caveats=['classical methods do not use NN label smoothing or optimizer',
                             'previously inspected sources; exploratory, not independent new vehicles'])
    save(root / 'protocol.json', protocol)
    print('PROTOCOL ' + json.dumps(protocol), flush=True)
    results = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            results.append(run_fit(raw, legacy, y, m, seed, fold, cfg, root))
            save(root / 'results.json', results)
    summary = dict(validation={k: float(np.mean([r['val'][k] for r in results]))
                               for k in results[0]['val']}, fits=len(results), test_evaluated=True)
    save(root / 'summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)
