"""Paired high-speed data-transfer experiments; JSON config is the run contract."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

from prepared_experiment import (PARTS, load_prepared, fit_stats, normalize,
    probability_metrics, fit_temperature, infer, window_statistics, write_json)
from train import build_model
from utils import set_seed


def make_splits(manifest, fold, train_stride=40):
    result = {}
    for role, offset in [('test',0),('val',1),('train',2)]:
        mask = manifest.campaign.eq('Vicolungo') & manifest.source_id.str.endswith('_'+PARTS[(fold+offset)%3])
        if role == 'train':
            mask &= manifest.start_in_segment.mod(train_stride).eq(0)
        result[role] = np.flatnonzero(mask)
    held = manifest.campaign.eq('CATS') & manifest.source_id.isin(['CATS__test1124_test9','CATS__test1124_test10'])
    result['external_test'] = np.flatnonzero(held)
    candidates = manifest.campaign.isin(['CATS','Vanderbilt']) & ~held
    result['external_train'] = np.flatnonzero(candidates & manifest.start_in_segment.mod(train_stride).eq(0))
    # NGSIM remains in archived prepared data, but is excluded from all new fits.
    ssl = candidates & manifest.start_in_segment.mod(train_stride).eq(0)
    result['external_ssl'] = np.flatnonzero(ssl)
    for key in ['train','val','test','external_test','external_train']:
        if set(manifest.iloc[result[key]].label) != {0,1}:
            raise ValueError(f'{key} must contain both classes')
    sources={k:set(manifest.iloc[v].source_id) for k,v in result.items()}
    for train in ['train','external_train','external_ssl']:
        for test in ['val','test','external_test']:
            if sources[train] & sources[test]:
                raise ValueError(f'Source leakage: {train} / {test}')
    return result


def enforce_training_data_policy(cfg):
    if cfg.get('include_ngsim', False):
        raise ValueError('NGSIM is excluded from new training; use frozen source snapshots for historical reproduction')
    cfg['include_ngsim'] = False


def joint_weights(manifest, core, external, external_mass=.5):
    """Preserve core class prior, fixed update count; source-balance extra data."""
    joined=np.r_[core,external]
    y=manifest.iloc[core].label.to_numpy()
    weights=np.zeros(len(joined),dtype=np.float64)
    weights[:len(core)]=(1-external_mass)/len(core)
    extra=manifest.iloc[external]
    for label in [0,1]:
        class_mass=float(np.mean(y==label))*external_mass
        rows=np.flatnonzero(extra.label.to_numpy()==label)
        # First balance campaigns, then recording files within each campaign.
        campaigns=extra.iloc[rows].campaign.unique()
        for campaign in campaigns:
            groups=extra[(extra.label==label)&(extra.campaign==campaign)].source_id.unique()
            for source in groups:
                ids=np.flatnonzero((extra.label.to_numpy()==label)&(extra.source_id.to_numpy()==source))
                weights[len(core)+ids]=class_mass/len(campaigns)/len(groups)/len(ids)
    assert np.isclose(weights.sum(),1)
    assert np.isclose(weights[manifest.iloc[joined].label.eq(1)].sum(),y.mean())
    return weights


def source_weights(manifest, ids):
    rows=manifest.iloc[ids]
    counts=rows.groupby(['campaign','source_id']).size()
    campaigns=rows.campaign.unique()
    out=np.zeros(len(rows))
    for campaign in campaigns:
        sources=rows[rows.campaign==campaign].source_id.unique()
        for source in sources:
            mask=rows.source_id.eq(source).to_numpy()
            out[mask]=1/len(campaigns)/len(sources)/mask.sum()
    assert np.isclose(out.sum(),1)
    return torch.tensor(out,dtype=torch.float32)


def masked_pretrain(model, inputs, weights, n_draws, cfg, path):
    """Reconstruct three primitives in an entirely hidden 1-second span."""
    head=torch.nn.Linear(cfg['model']['d_model'],3)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),
        lr=cfg['pretrain_lr'],weight_decay=cfg['weight_decay'])
    history=[]
    for epoch in range(cfg['pretrain_epochs']):
        model.train(); head.train()
        order=torch.multinomial(weights,n_draws,replacement=True)
        total=0.
        for ids in order.split(cfg['batch_size']):
            target=inputs[ids]
            starts=torch.randint(0,31,(len(ids),))
            mask=(torch.arange(40)[None,:]>=starts[:,None]) & (torch.arange(40)[None,:]<starts[:,None]+10)
            corrupted=target.clone()+torch.randn_like(target)*.03
            corrupted[mask]=0
            h=model.input_proj(corrupted)
            h=torch.cat([model.cls_token.expand(len(ids),-1,-1),h],dim=1)+model.pos_embed
            prediction=head(model.encoder(h)[:,1:])
            loss=F.smooth_l1_loss(prediction[mask],target[:,:,[0,3,5]][mask])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.)
            optimizer.step(); total+=float(loss.detach())*len(ids)
        history.append(dict(epoch=epoch+1,loss=total/n_draws))
    torch.save(model.state_dict(),path/'pretrained_encoder_and_initial_head.pt')
    write_json(path/'pretraining_history.json',history)
    return history[-1]['loss']


def snapshot_splits(path,manifest,split):
    write_json(path/'splits.json',{k:dict(window_indices=manifest.iloc[v].window_index.tolist(),
        sources=sorted(manifest.iloc[v].source_id.unique()),
        class_counts={str(int(a)):int(b) for a,b in manifest.iloc[v].label.value_counts().items()})
        for k,v in split.items()})


def run_one(x,y,m,names,split,seed,fold,variant,cfg,root):
    start=time.time()
    path=root/variant/f'seed_{seed}'/PARTS[fold]
    path.mkdir(parents=True,exist_ok=False)
    set_seed(seed+fold)
    stats=fit_stats(x[split['train']]); stats.update(feature_names=names,fit_split='core_train_only')
    write_json(path/'feature_stats.json',stats)
    snapshot_splits(path,m,split)
    normalized=normalize(x,stats)
    history=[]; best={key:dict(value=float('inf') if key=='bce' else -1.,epoch=0) for key in ['bce','accuracy']}
    candidates={}; initial={}; pretrain_loss=None
    if variant=='tree':
        sx={k:window_statistics(normalized[v]) for k,v in split.items() if k in ['train','val','test','external_test']}
        estimator=HistGradientBoostingClassifier(**cfg['tree'],random_state=seed+fold)
        estimator.fit(sx['train'],y[split['train']])
        prior=float(y[split['train']].mean()); prior_logit=float(np.log(prior/(1-prior)))
        val_stages=[np.full(len(split['val']),prior_logit)]
        for p in estimator.staged_predict_proba(sx['val']):
            val_stages.append(np.log(np.clip(p[:,1],1e-12,1))-np.log(np.clip(p[:,0],1e-12,1)))
        for epoch,z in enumerate(val_stages):
            metrics=probability_metrics(y[split['val']],z)
            history.append(dict(epoch=epoch,train_loss=None,val=metrics))
            for key in best:
                if (metrics[key]<best[key]['value']-1e-8 if key=='bce' else metrics[key]>best[key]['value']+1e-8):
                    best[key]=dict(value=metrics[key],epoch=epoch)
        for evaluation in ['test','external_test']:
            stages=[np.full(len(split[evaluation]),prior_logit)]
            for p in estimator.staged_predict_proba(sx[evaluation]):
                stages.append(np.log(np.clip(p[:,1],1e-12,1))-np.log(np.clip(p[:,0],1e-12,1)))
            initial[evaluation]=stages[0]
            candidates[evaluation]={key:stages[info['epoch']] for key,info in best.items()}
        val_logits={key:val_stages[info['epoch']] for key,info in best.items()}
        dump(estimator,path/'tree.joblib')
    else:
        model=build_model(cfg['model'],'cpu')
        initial_state=copy.deepcopy(model.state_dict())
        tx=torch.from_numpy(normalized)
        if variant.startswith('ssl_'):
            ids=split['external_ssl'] if variant=='ssl_external' else split['train']
            pretrain_loss=masked_pretrain(model,tx[ids],source_weights(m,ids),len(split['train']),cfg,path)
        # Identical post-initialization RNG contract for core and SSL comparisons.
        set_seed(seed+fold+10000)
        train_ids=np.r_[split['train'],split['external_train']] if variant=='joint' else split['train']
        weights=torch.tensor(joint_weights(m,split['train'],split['external_train']),dtype=torch.float32) if variant=='joint' else None
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        vy=y[split['val']]
        for epoch in range(cfg['epochs']+1):
            loss_value=None
            if epoch:
                model.train(); total=0.
                order=(torch.multinomial(weights,len(split['train']),replacement=True).numpy()
                       if weights is not None else torch.randperm(len(train_ids)).numpy())
                for start_idx in range(0,len(order),cfg['batch_size']):
                    ids=train_ids[order[start_idx:start_idx+cfg['batch_size']]]
                    target=torch.from_numpy(y[ids])*(1-cfg['label_smoothing'])+.5*cfg['label_smoothing']
                    loss=F.binary_cross_entropy_with_logits(model(tx[ids]),target)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step(); total+=float(loss.detach())*len(ids)
                loss_value=total/len(order)
            metrics=probability_metrics(vy,infer(model,tx[split['val']]))
            history.append(dict(epoch=epoch,train_loss=loss_value,val=metrics))
            for key in best:
                if (metrics[key]<best[key]['value']-1e-8 if key=='bce' else metrics[key]>best[key]['value']+1e-8):
                    best[key]=dict(value=metrics[key],epoch=epoch)
                    torch.save(dict(model_state_dict=model.state_dict(),model=cfg['model'],epoch=epoch,
                                    selection=key),path/f'best_{key}.pt')
            if epoch%10==0:
                print(f'{variant} seed={seed} {PARTS[fold]} epoch={epoch} val_bce={metrics["bce"]:.4f} val_acc={metrics["accuracy"]:.4f}',flush=True)
            if all(epoch-b['epoch']>=cfg['patience'] for b in best.values()):
                break
        # Held-out predictions happen only after all checkpoint selection.
        val_logits={}
        model.load_state_dict(initial_state)
        for evaluation in ['test','external_test']:
            initial[evaluation]=infer(model,tx[split[evaluation]])
            candidates[evaluation]={}
        for key in best:
            model.load_state_dict(torch.load(path/f'best_{key}.pt',weights_only=True)['model_state_dict'])
            val_logits[key]=infer(model,tx[split['val']])
            for evaluation in candidates:
                candidates[evaluation][key]=infer(model,tx[split[evaluation]])
    calibration={key:fit_temperature(y[split['val']],z) for key,z in val_logits.items()}
    write_json(path/'calibration.json',calibration); write_json(path/'selection.json',best)
    write_json(path/'history.json',history)
    rows=[]
    for evaluation,predictions in candidates.items():
        ids=split[evaluation]; output=m.iloc[ids].copy()
        logits=dict(initial=initial[evaluation],constant_half=np.zeros(len(ids)),**predictions)
        for key,z in predictions.items():
            logits[key+'_temperature']=z/calibration[key]['temperature']
        for selection,z in logits.items():
            metrics=probability_metrics(y[ids],z)
            output[selection+'_logit']=z
            rows.append(dict(variant=variant,seed=seed,fold=PARTS[fold],evaluation=evaluation,
                selection=selection,n=len(ids),**{k:v for k,v in metrics.items() if k!='confusion_matrix'}))
        output.to_csv(path/f'{evaluation}_predictions.csv',index=False)
    write_json(path/'metrics.json',rows)
    write_json(path/'runtime.json',dict(seconds=time.time()-start,pretrain_final_loss=pretrain_loss))
    headline=[r for r in rows if r['selection']=='bce_temperature']
    print('RESULT '+json.dumps(headline),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/highspeed_no_ngsim_20261010.json')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    enforce_training_data_policy(cfg)
    root=Path(cfg['output']); root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg['threads'])
    x,y,m,names=load_prepared(cfg['prepared'])
    assert len(names)==cfg['model']['input_dim']
    protocol=dict(config=cfg,selection='validation BCE primary; accuracy secondary; threshold0.5',
        test_policy='Vicolungo cyclic source holdout and fixed CATS test9/10 holdout; no external-test data in SSL',
        joint_policy='50% core/50% trusted external mass; preserve core class prior and supervised updates',
        ssl_policy='fixed pretrain epochs; equal update counts in core and external controls; mask all channels of 10 frames; reconstruct vFV,vLV,gap',
        limitations='within-date Vicolungo holdout; CATS vehicles fixed roles; inherited upstream resampling; no paired roadside-video validation',
        versions=dict(torch=str(torch.__version__),numpy=np.__version__),
        data_sha256={f:hashlib.sha256((Path(cfg['prepared'])/f).read_bytes()).hexdigest() for f in ['windows.csv','windows_4s.npz']})
    write_json(root/'protocol.json',protocol)
    shutil.copytree(Path(__file__).parent,root/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(root/'config.json',cfg)
    print('CONFIG '+json.dumps(protocol),flush=True)
    all_rows=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            split=make_splits(m,fold,cfg['train_stride'])
            for variant in cfg['variants']:
                if variant=='tree' and seed!=cfg['seeds'][0]:
                    continue
                all_rows.extend(run_one(x,y,m,names,split,seed,fold,variant,cfg,root))
                pd.DataFrame(all_rows).to_csv(root/'comparison.csv',index=False)
    results=pd.DataFrame(all_rows)
    summary=results.groupby(['variant','evaluation','selection'])[['accuracy','balanced_accuracy','bce','auroc','human_recall','acc_recall']].mean()
    summary.to_csv(root/'overview.csv')
    print('FINAL SUMMARY\n'+summary.round(4).to_string(),flush=True)


if __name__=='__main__':
    main()
