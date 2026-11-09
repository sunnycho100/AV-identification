"""Check restored neural checkpoints on synthetic inputs; never train or fit stats.

Use an environment with torch, numpy and scipy. Random forests require the
separate pinned environment and the original smoke_models.py.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

workspace = Path(__file__).resolve().parent.parent
project = workspace / 'classification' / 'trajectory-avlike-transformer'
python_files = sorted(project.rglob('*.py'))
for file in python_files:
    ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))

import numpy as np
import torch
sys.path.insert(0, str(project / 'training' / 'transformer'))
from networks import TrajectoryNet
from data_protocol import features, normalize

torch.set_num_threads(1)
raw = np.empty((8, 40, 3), np.float32)
t = np.arange(40, dtype=np.float32) * .1
for i in range(8):
    raw[i, :, 0] = 27 + .02*i + .1*np.sin(t)
    raw[i, :, 1] = 28 + .02*i + .1*np.sin(t)
    raw[i, :, 2] = 30 + t
x = features(raw, 'legacy11', 7)
records = []
for path in sorted((project / 'artifacts').rglob('best_bce.pt')):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    model = TrajectoryNet(11, cfg['architecture'], cfg['width'], cfg['dropout'], cfg['layers']).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    stats = json.loads((path.parent / 'feature_stats.json').read_text(encoding='utf-8'))
    tx = torch.from_numpy(normalize(x, stats))
    with torch.no_grad():
        logits = (model.classifier(model.encode(tx).mean(0, keepdim=True)).reshape(-1)
                  if cfg.get('study') == 'multisegment' else model(tx))
        probabilities = torch.sigmoid(logits).numpy()
    expected_shape = (1,) if cfg.get('study') == 'multisegment' else (8,)
    assert probabilities.shape == expected_shape
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    parameters = sum(p.numel() for p in model.parameters())
    if cfg['architecture'] == 'transformer':
        assert parameters == 420353
    records.append(dict(path=path.relative_to(project).as_posix(), architecture=cfg['architecture'],
                        mode=cfg.get('mode', 'single_segment'), parameters=parameters, status='PASS'))
assert len(records) == 60
assert sum(r['architecture'] == 'transformer' for r in records) == 15
versions = {}
for package in ('torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib'):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
reference_versions = {'torch': '2.6.0', 'numpy': '1.26.4', 'scipy': '1.15.3',
                      'pandas': '2.2.3', 'scikit-learn': '1.7.2', 'joblib': '1.5.2'}
environment_matches = (sys.version_info[:2] == (3, 10) and all(
    (versions.get(k) or '').split('+')[0] == v for k, v in reference_versions.items()))
report = dict(
    status='PASS_FOR_NEURAL_SUBSET', verified_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='CPU strict weight loading and finite outputs on synthetic primitives; no training or accuracy evaluation',
    python=sys.version, versions=versions, environment_matches_supplied_pins=environment_matches,
    python_files_parsed=len(python_files), neural_models_checked=len(records),
    architectures=dict(Counter(r['architecture'] for r in records)),
    rf_models_present=len(list((project / 'artifacts').rglob('model.joblib'))),
    rf_loading_tested=False,
    rf_not_tested_reason='Available environments lack scikit-learn 1.7.2 and joblib; source weight hashes checked separately',
    prepared_training_npz_present=(project / 'data/prepared_highspeed_20261010/windows_4s.npz').is_file(),
    records=records,
)
out = workspace / 'catalogs' / 'classification_validation.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'PASS: {len(python_files)} Python files parsed; 60 neural models restored with finite CPU outputs, including 15 Transformers (420,353 parameters each).')
print('RF deserialization, full training, and real-data accuracy were not tested.')