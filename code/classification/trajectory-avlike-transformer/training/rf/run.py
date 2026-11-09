"""Fixed-command GPU study; development jobs emit validation results only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
import torch.nn.functional as F

from data_protocol import PARTS, splits, features, fit_stats, normalize, joint_weights, source_weights, forecast_examples
from networks import TrajectoryNet


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y, z):
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    assert np.isfinite(z).all() and set(y) == {0, 1}
    losses = np.logaddexp(0, z)-y*z
    return dict(accuracy=float(np.mean((z >= 0) == y)), bce=float(losses.mean()),
        balanced_bce=float(.5*(losses[y == 0].mean()+losses[y == 1].mean())),
        balanced_accuracy=float(balanced_accuracy_score(y, z >= 0)),
        auroc=float(roc_auc_score(y, z)), human_recall=float(np.mean(z[y == 0] < 0)),
        acc_recall=float(np.mean(z[y == 1] >= 0)), brier=float(np.mean((expit(z)-y)**2)))


@torch.no_grad()
def infer(model, x):
    model.eval()
    return torch.cat([model(a) for a in x.split(512)]).cpu().numpy()


def temperature(y, z):
    def objective(log_t):
        q = z/np.exp(log_t)
        return np.mean(np.logaddexp(0, q)-y*q)
    result = minimize_scalar(objective, bounds=(np.log(.1), np.log(20)), method='bounded')
    best = min([0., np.log(.1), np.log(20), float(result.x)], key=objective)
    return float(np.exp(best))


def pretrain(model, raw, m, ids, stats, cfg, path, n_draws):
    if cfg['pretrain'] == 'none':
        return
    device = next(model.parameters()).device
    px, target = forecast_examples(raw[ids], cfg, stats)
    tx = torch.from_numpy(px).to(device)
    ty = torch.from_numpy(target).to(device)
    head = torch.nn.Linear(model.representation_dim, 30).to(device)
    params = list(model.parameters())+list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['pretrain_lr'], weight_decay=cfg['weight_decay'])
    weights = torch.tensor(source_weights(m, ids), device=device)
    history = []
    # The zero-residual predictor is the constant-speed / constant-relative-speed baseline.
    baseline = float(F.smooth_l1_loss(torch.zeros_like(ty), ty).item())
    for epoch in range(cfg['pretrain_epochs']):
        model.train(); head.train(); total = 0.
        order = torch.multinomial(weights, n_draws, replacement=True)
        for batch in order.split(cfg['batch_size']):
            pred = head(model.encode(tx[batch])).reshape(-1, 10, 3)
            loss = F.smooth_l1_loss(pred, ty[batch])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step(); total += float(loss.detach())*len(batch)
        history.append(dict(epoch=epoch+1, loss=total/n_draws, persistence_loss=baseline))
    save(path/'pretraining_history.json', history)
    save(path/'pretraining_sources.json', dict(window_indices=ids.tolist(),
        campaigns=m.iloc[ids].campaign.value_counts().to_dict(), input_frames=30, target_frames=10,
        actual_prefix_seconds=3., maximum_observation_seconds=4.))
    torch.save({k:v.cpu() for k,v in model.state_dict().items()}, path/'pretrained.pt')


def run_fit(raw, legacy, y, m, seed, fold, cfg, root):
    start = time.time()
    path = root/f'seed_{seed}'/PARTS[fold]
    path.mkdir(parents=True, exist_ok=False)
    effective = copy.deepcopy(cfg)
    effective.update(cfg.get('fold_overrides', {}).get(str(fold), {}))
    cfg = effective
    split = splits(m, fold, cfg['train_stride'])
    x = features(raw, cfg['features'], cfg['smoothing'])
    if cfg['features'] == 'legacy11' and cfg['smoothing'] == 7:
        np.testing.assert_allclose(x, legacy, atol=4e-3, rtol=1e-4)
    stats = fit_stats(x[split['train']])
    save(path/'feature_stats.json', dict(**stats, fit_split='core_train', input_dim=x.shape[-1]))
    save(path/'splits.json', {k:dict(window_indices=ids.tolist(),
        sources=sorted(m.iloc[ids].source_id.unique()), class_counts={str(k):int(v)
        for k,v in m.iloc[ids].label.value_counts().items()}) for k,ids in split.items()})
    save(path/'effective_config.json', cfg)
    normalized = normalize(x, stats)
    device = torch.device(cfg['device'])
    tx, ty = torch.from_numpy(normalized).to(device), torch.from_numpy(y).to(device)
    seed_all(seed+fold)
    if cfg['architecture'] == 'plain_lstm':
        from native_benchmark import NativeLSTM
        model = NativeLSTM(x.shape[-1], cfg['width'], cfg['layers'], cfg['dropout']).to(device)
    else:
        model = TrajectoryNet(x.shape[-1], cfg['architecture'], cfg['width'], cfg['dropout'], cfg['layers']).to(device)
    init_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if cfg['pretrain'] != 'none':
        ids = split['train'] if cfg['pretrain'] == 'core' else np.r_[split['train'], split['external']]
        if cfg['pretrain'] == 'expanded':
            ids = np.r_[ids, split['aux']]
        pretrain(model, raw, m, ids, stats, cfg, path, len(split['train']))
    seed_all(seed+fold+10000)
    train_ids = split['train']
    sample_weights = None
    if cfg['data_mode'] != 'core':
        extra = split['external']
        if cfg['data_mode'] == 'expanded':
            extra = np.r_[extra, split['aux']]
        sample_weights = torch.tensor(joint_weights(m, train_ids, extra, cfg['external_mass']), device=device)
        train_ids = np.r_[train_ids, extra]
    save(path/'supervised_pool.json', dict(window_indices=train_ids.tolist(),
        campaign_counts=m.iloc[train_ids].campaign.value_counts().to_dict()))
    train_ids = torch.tensor(train_ids, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    prior = float(y[split['train']].mean())
    class_weights = torch.tensor([.5/(1-prior), .5/prior], device=device)
    best = {k:dict(value=-1. if k == 'accuracy' else float('inf'), epoch=0)
            for k in ('bce', 'accuracy', 'balanced_bce')}
    history = []
    fixed_noise = None
    if cfg.get('augment', False):
        # Small offsets consistent across frames; shared speed bias preserves relative speed.
        rng = np.random.default_rng(seed+fold+555)
        augmented = raw.copy()
        speed_offset = rng.normal(0, .15, (len(raw), 1, 1))
        augmented[:, :, :2] += speed_offset
        augmented[:, :, 2] += rng.normal(0, .5, (len(raw), 1))
        fixed_noise = torch.from_numpy(normalize(features(augmented, cfg['features'], cfg['smoothing']), stats)).to(device)
    primary = cfg['selection']
    for epoch in range(cfg['epochs']+1):
        objective = None
        if epoch:
            model.train(); total = 0.
            order = (torch.multinomial(sample_weights, len(split['train']), replacement=True)
                     if sample_weights is not None else torch.randperm(len(train_ids), device=device))
            for batch in order.split(cfg['batch_size']):
                ids = train_ids[batch]
                targets = ty[ids]*(1-cfg['label_smoothing'])+.5*cfg['label_smoothing']
                xx = tx[ids]
                if fixed_noise is not None:
                    use_aug = torch.rand(len(ids), 1, 1, device=device) < .5
                    xx = torch.where(use_aug, fixed_noise[ids], xx)
                per_sample = F.binary_cross_entropy_with_logits(model(xx), targets, reduction='none')
                if cfg['class_balanced_loss']:
                    per_sample = per_sample*class_weights[ty[ids].long()]
                loss = per_sample.mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step(); total += float(loss.detach())*len(ids)
            objective = total/len(order)
        val_z = infer(model, tx[split['val']])
        met = metrics(y[split['val']], val_z)
        history.append(dict(epoch=epoch, train_loss=objective, val=met))
        for key in best:
            improved = met[key] > best[key]['value']+1e-8 if key == 'accuracy' else met[key] < best[key]['value']-1e-8
            if improved:
                best[key] = dict(value=met[key], epoch=epoch)
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(dict(model_state_dict=state, epoch=epoch, input_dim=x.shape[-1], config=cfg), path/f'best_{key}.pt')
        if epoch % 10 == 0:
            print(json.dumps(dict(event='epoch', seed=seed, fold=fold, epoch=epoch,
                                  val_bce=met['bce'], val_accuracy=met['accuracy'])), flush=True)
        if epoch-best[primary]['epoch'] >= cfg['patience']:
            break
    save(path/'history.json', history); save(path/'selection.json', best)
    checkpoint = torch.load(path/f'best_{primary}.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_z = infer(model, tx[split['val']])
    calibration = temperature(y[split['val']], val_z)
    save(path/'calibration.json', dict(temperature=calibration, fit_split='val', threshold=.5))
    pred = m.iloc[split['val']].copy(); pred['logit'] = val_z
    pred.to_csv(path/'val_predictions.csv', index=False)
    result = dict(seed=seed, fold=fold, fold_name=PARTS[fold], epoch=checkpoint['epoch'],
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.time()-start,
        val=metrics(y[split['val']], val_z), val_temperature=metrics(y[split['val']], val_z/calibration))
    if cfg['evaluate_test']:
        result['evaluations'] = {}
        for name in ('test', 'external_test'):
            ids = split[name]; z = infer(model, tx[ids])
            result['evaluations'][name] = dict(raw=metrics(y[ids], z), calibrated=metrics(y[ids], z/calibration))
            pred = m.iloc[ids].copy(); pred['logit'] = z; pred['calibrated_logit'] = z/calibration
            pred.to_csv(path/f'{name}_predictions.csv', index=False)
    save(path/'result.json', result)
    print('RESULT '+json.dumps(result), flush=True)
    return result


def main():
    cfg = json.loads(Path('experiment.json').read_text())
    assert cfg['include_ngsim'] is False
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU protocol requires CUDA; no silent CPU fallback')
    root = Path('artifacts/run'); root.mkdir(parents=True, exist_ok=False)
    prepared = Path(cfg['prepared'])
    m = pd.read_csv(prepared/'windows.csv')
    with np.load(prepared/'windows_4s.npz', allow_pickle=False) as d:
        raw, legacy, y = d['primitive_windows'], d['X'], d['y'].astype(np.float32)
        np.testing.assert_array_equal(d['window_index'], m.window_index)
        np.testing.assert_array_equal(y, m.label)
    protocol = dict(config=cfg, torch=str(torch.__version__), numpy=np.__version__,
        device=torch.cuda.get_device_name(0) if cfg['device']=='cuda' else 'cpu',
        dataset_sha256={name:hashlib.sha256((prepared/name).read_bytes()).hexdigest()
                        for name in ('windows.csv', 'windows_4s.npz')},
        selection_scope='validation only; test prediction disabled in development',
        caveats=['Vicolungo dates confounded with labels', 'CATS held recordings reuse vehicles',
                 'Upstream OpenACC interpolation cannot be undone', 'Previously inspected benchmarks are exploratory'])
    save(root/'protocol.json', protocol)
    print('PROTOCOL '+json.dumps(protocol), flush=True)
    results = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            results.append(run_fit(raw, legacy, y, m, seed, fold, cfg, root))
            save(root/'results.json', results)
    summary = {k:float(np.mean([r['val'][k] for r in results])) for k in results[0]['val']}
    save(root/'summary.json', dict(validation=summary, fits=len(results), test_evaluated=cfg['evaluate_test']))
    print('SUMMARY '+json.dumps(dict(validation=summary, fits=len(results), test_evaluated=cfg['evaluate_test'])), flush=True)


if __name__ == '__main__':
    if json.loads(Path('experiment.json').read_text()).get('study') == 'native11_classical':
        from native_benchmark import main as classical_main
        classical_main()
    else:
        main()
