"""Paired source-held-out probability experiments; never overwrite an existing run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
import torch.nn.functional as F

from cross_validate import make_source_holdout_splits
from dataset import make_dataloaders
from train import build_model
from utils import compute_metrics, load_config, set_seed


def probability_metrics(labels, logits, pos_weight=1.0):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    probs = expit(logits)
    result = compute_metrics(labels, probs)
    losses = np.logaddexp(0, logits) - labels * logits
    result.update(bce=float(losses.mean()), brier=float(np.mean((labels-probs)**2)),
                  weighted_bce=float(np.mean(losses * np.where(labels == 1, pos_weight, 1))))
    return result


def score_frame(frame, pos_weight):
    win = probability_metrics(frame.label, frame.logit, pos_weight)
    trajectory = frame.groupby('Trajectory_ID', as_index=False).agg(label=('label','first'), prob_AV=('prob_AV','mean'))
    # Preserve exact ties for constant baselines despite floating-point group means.
    if frame.logit.nunique() == 1:
        trajectory['prob_AV'] = float(expit(frame.logit.iloc[0]))
    p = np.clip(trajectory.prob_AV.to_numpy(), 1e-15, 1-1e-15)
    traj = probability_metrics(trajectory.label, np.log(p)-np.log1p(-p), pos_weight)
    return {**{'window_'+k:v for k,v in win.items() if not isinstance(v,list)},
            **{'trajectory_'+k:v for k,v in traj.items() if not isinstance(v,list)}}


@torch.no_grad()
def infer(model, x, batch_size):
    model.eval()
    return torch.cat([model(batch) for batch in x.split(batch_size)]).cpu().numpy()


def run_fold(config, df, fold, fold_index, seed, variant, output, device):
    actual_seed = seed + fold_index
    set_seed(actual_seed)
    cfg = copy.deepcopy(config)
    cfg['training']['seed'] = actual_seed
    cfg['features']['eval_stride'] = 10
    if variant != 'weighted_transformer':
        cfg['training']['use_pos_weight'] = False
    if variant == 'temporal_cnn':
        cfg['model'].update(architecture='temporal_cnn', channels=32, dropout=0.3)
        cfg['training']['weight_decay'] = 0.01
    path = output / variant / f'seed_{seed}' / fold['name']
    path.mkdir(parents=True)
    splits = make_source_holdout_splits(cfg, df, fold['test_sources'], actual_seed)
    assert not (set(splits['train']) & set(splits['val']) or set(splits['train']) & set(splits['test']) or set(splits['val']) & set(splits['test']))
    (path/'splits.json').write_text(json.dumps(splits, indent=2))
    (path/'config.json').write_text(json.dumps(cfg, indent=2))
    _, datasets, diagnostics = make_dataloaders(cfg, df, splits, stats_path=str(path/'feature_stats.json'))
    # Cache exactly the Dataset-normalized windows; no resampling or preprocessing changes.
    tensors = {}
    for split, dataset in datasets.items():
        items = [dataset[i] for i in range(len(dataset))]
        tensors[split] = (torch.stack([x for x,y in items]).to(device), torch.stack([y for x,y in items]).to(device))
    train_x, train_y = tensors['train']
    model = build_model(cfg['model'], device)
    n_pos = diagnostics['train']['num_av_windows']
    audit_weight = diagnostics['train']['num_hv_windows'] / n_pos
    pw = torch.tensor(audit_weight if cfg['training']['use_pos_weight'] else 1., device=device)
    weights = torch.ones_like(train_y)
    if variant == 'equal_trajectory':
        ids = pd.Series([w['trajectory_id'] for w in datasets['train'].windows])
        counts = ids.value_counts()
        weights = torch.tensor((len(ids)/len(counts)/ids.map(counts)).to_numpy(), dtype=torch.float32, device=device)
        masses = pd.DataFrame({'id':ids, 'weight':weights.cpu().numpy()}).groupby('id').weight.sum()
        assert np.allclose(masses, masses.iloc[0], rtol=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    batch_size = cfg['training']['batch_size']
    best = {key: {'value': -np.inf if key=='auroc' else np.inf, 'epoch':0} for key in ['auroc','bce','weighted_bce']}
    history = []
    start = time.time()
    initial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for epoch in range(cfg['training']['epochs']+1):
        train_loss = None
        if epoch:
            model.train()
            permutation = torch.randperm(len(train_y), device=device)
            total = 0.
            for indices in permutation.split(batch_size):
                logits = model(train_x[indices])
                loss = (F.binary_cross_entropy_with_logits(logits, train_y[indices], pos_weight=pw, reduction='none') * weights[indices]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()*len(indices)
            train_loss = total/len(train_y)
        val_logits = infer(model, tensors['val'][0], batch_size)
        val = probability_metrics(tensors['val'][1].cpu().numpy(), val_logits, audit_weight)
        history.append({'epoch':epoch, 'train_loss':train_loss, 'val':val})
        for key in best:
            improved = val[key] > best[key]['value'] if key=='auroc' else val[key] < best[key]['value']
            if improved:
                best[key] = {'value':val[key], 'epoch':epoch}
                torch.save({'model_state_dict':model.state_dict(), 'config':cfg, 'epoch':epoch,
                            'selection_metric':key, 'validation':val}, path/f'best_{key}.pt')
        if epoch and all(epoch-item['epoch'] >= cfg['training']['early_stopping_patience'] for item in best.values()):
            break
    (path/'history.json').write_text(json.dumps(history, indent=2))
    rows = []
    test_y = tensors['test'][1].cpu().numpy()
    test_windows = datasets['test'].windows
    def record(name, logits, selected_epoch, temperature=1.):
        frame = pd.DataFrame({'Trajectory_ID':[w['trajectory_id'] for w in test_windows],
                              'window_start_time':[w['window_start_time'] for w in test_windows],
                              'label':test_y, 'logit':logits, 'prob_AV':expit(logits)})
        frame.to_csv(path/f'{name}_predictions.csv', index=False)
        row = dict(variant=variant, seed=seed, fold=fold['name'], selection=name,
                   epoch=selected_epoch, temperature=float(temperature), parameters=sum(p.numel() for p in model.parameters()),
                   **score_frame(frame, audit_weight))
        rows.append(row)
    # Test is accessed only after all training and checkpoint selection have finished.
    model.load_state_dict(initial_state)
    record('initial', infer(model, tensors['test'][0], batch_size), 0)
    record('constant_half', np.zeros(len(test_y)), 0)
    prior = float(train_y.mean().cpu())
    record('train_prior', np.full(len(test_y), np.log(prior/(1-prior))), 0)
    for key in best:
        ckpt = torch.load(path/f'best_{key}.pt', map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        test_logits = infer(model, tensors['test'][0], batch_size).astype(np.float64)
        record(key, test_logits, ckpt['epoch'])
        if key == 'bce':
            val_logits = infer(model, tensors['val'][0], batch_size).astype(np.float64)
            val_y = tensors['val'][1].cpu().numpy()
            def objective(log_temperature):
                z = val_logits / np.exp(log_temperature)
                return np.mean(np.logaddexp(0,z)-val_y*z)
            fit = minimize_scalar(objective, bounds=(np.log(0.1),np.log(20.)), method='bounded')
            candidates = [float(fit.x), 0., np.log(0.1), np.log(20.)]
            log_t = min(candidates, key=objective)
            temperature = float(np.exp(log_t))
            (path/'calibration.json').write_text(json.dumps({'temperature':temperature, 'fit_split':'val', 'val_bce_before':objective(0.), 'val_bce_after':objective(log_t)}, indent=2))
            record('bce_temperature', test_logits/temperature, ckpt['epoch'], temperature)
    (path/'metrics.json').write_text(json.dumps(rows, indent=2))
    print(f'{variant} seed={seed} {fold["name"]}: epochs={epoch}, best={best}, seconds={time.time()-start:.1f}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/transformer_baseline_cv.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,52,62])
    parser.add_argument('--variants', nargs='+', choices=['weighted_transformer','plain_transformer','equal_trajectory','temporal_cnn'], default=['weighted_transformer','plain_transformer','equal_trajectory','temporal_cnn'])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = load_config(args.config)
    df = pd.read_csv(config['data']['processed_csv'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shutil.copytree(Path(__file__).parent, output/'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
    protocol = dict(config=config, seeds=args.seeds, variants=args.variants, device=device, torch=torch.__version__,
                    data_sha256=hashlib.sha256(Path(config['data']['processed_csv']).read_bytes()).hexdigest(),
                    stopping='all three validation criteria stalled for patience epochs; epoch 0 eligible',
                    sampling='same shuffled windows and optimizer budget; equal_trajectory uses inverse count loss without class weights')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows=[]
    for seed in args.seeds:
        for index, fold in enumerate(config['cross_validation']['folds']):
            for variant in args.variants:
                rows.extend(run_fold(config, df, fold, index, seed, variant, output, device))
                pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    results = pd.DataFrame(rows)
    metrics = [c for c in results if c.startswith(('window_','trajectory_'))]
    seed_means = results.groupby(['variant','selection','seed'])[metrics].mean()
    seed_means.to_csv(output/'seed_means.csv')
    seed_means.groupby(['variant','selection']).agg(['mean','std']).to_csv(output/'overview.csv')
    print(results.groupby(['variant','selection'])[['window_bce','window_brier','window_auroc','trajectory_bce','trajectory_auroc']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
