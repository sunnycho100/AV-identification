"""Audit frozen high-speed runs and aggregate folds within each seed first."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load

from highspeed_experiment import make_splits, PARTS
from prepared_experiment import fit_stats, normalize, probability_metrics, infer, window_statistics
from train import build_model


def read_json(path):
    return json.loads(path.read_text())


def expected_splits_for_run(manifest, fold, cfg):
    """Audit the recorded policy, including legacy runs made before exclusion."""
    stride = cfg['train_stride']
    split = make_splits(manifest, fold, stride)
    if cfg.get('include_ngsim', True):
        legacy = np.flatnonzero(manifest.campaign.eq('NGSIM') & manifest.start_in_segment.mod(stride).eq(0))
        split['external_ssl'] = np.sort(np.r_[split['external_ssl'], legacy])
    if cfg.get('include_microsim', False):
        extra = np.flatnonzero(manifest.campaign.eq('MicroSim') & manifest.start_in_segment.mod(stride).eq(0))
        split['external_ssl'] = np.r_[split['external_ssl'], extra]
    return split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/experiments/20261010_highspeed_v1')
    args=parser.parse_args(); root=Path(args.run)
    cfg=read_json(root/'config.json'); torch.set_num_threads(4)
    m=pd.read_csv(Path(cfg['prepared'])/'windows.csv')
    with np.load(Path(cfg['prepared'])/'windows_4s.npz') as data:
        x=data['X']; y=data['y']; names=data['feature_names'].tolist()
    expected=sum(1 for seed in cfg['seeds'] for fold in cfg['folds'] for variant in cfg['variants']
                 if variant!='tree' or seed==cfg['seeds'][0])
    metrics_files=sorted(root.glob('*/seed_*/*/metrics.json'))
    assert len(metrics_files)==expected,(len(metrics_files),expected)
    rows=[]; diagnostics=[]; source_rows=[]; max_difference=0.; checked_predictions=0
    for metrics_path in metrics_files:
        path=metrics_path.parent
        variant=path.parent.parent.name; seed=int(path.parent.name.split('_')[1]); fold=PARTS.index(path.name)
        splits=read_json(path/'splits.json'); expected_split=expected_splits_for_run(m,fold,cfg)
        for key,indices in expected_split.items():
            np.testing.assert_array_equal(splits[key]['window_indices'],indices)
        stats=read_json(path/'feature_stats.json'); assert stats['feature_names']==names
        fitted=fit_stats(x[expected_split['train']])
        for key in fitted: np.testing.assert_allclose(stats[key],fitted[key],atol=1e-12,rtol=0)
        history=read_json(path/'history.json'); selection=read_json(path/'selection.json')
        calibration=read_json(path/'calibration.json')
        for key in ['bce','accuracy']:
            scores=np.array([h['val'][key] for h in history])
            winner=int(np.argmin(scores) if key=='bce' else np.argmax(scores))
            assert history[winner]['epoch']==selection[key]['epoch']
            assert calibration[key]['fit_split']=='val'
        normalized=normalize(x,stats)
        if variant=='tree':
            model=load(path/'tree.joblib')
            prior=float(y[expected_split['train']].mean())
        else:
            model=build_model(cfg['model'],'cpu')
        metrics=read_json(metrics_path); rows.extend(metrics)
        for evaluation in ['test','external_test']:
            pred=pd.read_csv(path/f'{evaluation}_predictions.csv')
            ids=expected_split[evaluation]
            np.testing.assert_array_equal(pred.window_index,ids)
            np.testing.assert_array_equal(pred.label,y[ids])
            for record in [r for r in metrics if r['evaluation']==evaluation]:
                actual=probability_metrics(y[ids],pred[record['selection']+'_logit'])
                for key in ['accuracy','bce','auroc','human_recall','acc_recall']:
                    # CSV stores float32 logits with finite decimal precision.
                    assert np.isclose(actual[key],record[key],atol=1e-7,rtol=0),(path,key)
            for key in ['bce','accuracy']:
                if variant=='tree':
                    epoch=selection[key]['epoch']
                    z=np.full(len(ids),np.log(prior/(1-prior)))
                    for iteration,p in enumerate(model.staged_predict_proba(window_statistics(normalized[ids])),1):
                        if iteration==epoch:
                            z=np.log(np.clip(p[:,1],1e-12,1))-np.log(np.clip(p[:,0],1e-12,1)); break
                else:
                    checkpoint=torch.load(path/f'best_{key}.pt',weights_only=True)
                    assert checkpoint['epoch']==selection[key]['epoch']
                    model.load_state_dict(checkpoint['model_state_dict'])
                    z=infer(model,torch.from_numpy(normalized[ids]))
                difference=float(np.max(np.abs(z-pred[key+'_logit'].to_numpy())))
                assert difference<2e-5,(path,key,difference)
                max_difference=max(max_difference,difference)
                np.testing.assert_allclose(pred[key+'_temperature_logit'],pred[key+'_logit']/calibration[key]['temperature'],atol=1e-6)
                np.testing.assert_array_equal(pred[key+'_temperature_logit']>=0,pred[key+'_logit']>=0)
                checked_predictions+=len(ids)
            for source,group in pred.groupby('source_id'):
                if group.label.nunique()==2:
                    values=probability_metrics(group.label,group.bce_temperature_logit)
                else:
                    gy=group.label.to_numpy(); gz=group.bce_temperature_logit.to_numpy()
                    values=dict(accuracy=float(np.mean((gz>=0)==gy)),bce=float(np.mean(np.logaddexp(0,gz)-gy*gz)),
                        human_recall=float(np.mean(gz[gy==0]<0)) if (gy==0).any() else float('nan'),
                        acc_recall=float(np.mean(gz[gy==1]>=0)) if (gy==1).any() else float('nan'),auroc=float('nan'))
                source_rows.append(dict(variant=variant,seed=seed,fold=path.name,evaluation=evaluation,
                    source=source,n=len(group),**{k:values[k] for k in ['accuracy','bce','human_recall','acc_recall','auroc']}))
        diagnostics.append(dict(variant=variant,seed=seed,fold=path.name,
            initial_val_bce=history[0]['val']['bce'],best_val_bce=selection['bce']['value'],
            best_bce_epoch=selection['bce']['epoch'],best_accuracy_epoch=selection['accuracy']['epoch'],
            last_epoch=history[-1]['epoch'],last_val_bce=history[-1]['val']['bce'],
            last_train_objective=history[-1]['train_loss']))
    frame=pd.DataFrame(rows)
    saved=pd.read_csv(root/'comparison.csv')
    keys=['variant','seed','fold','evaluation','selection']
    left=frame.sort_values(keys).reset_index(drop=True); right=saved.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys],right[keys])
    np.testing.assert_allclose(left[['accuracy','bce']],right[['accuracy','bce']],atol=1e-10)
    measures=['accuracy','balanced_accuracy','bce','auroc','human_recall','acc_recall']
    by_seed=frame.groupby(['variant','evaluation','selection','seed'])[measures].mean().reset_index()
    by_seed.to_csv(root/'seed_means.csv',index=False)
    summary=[]
    for (variant,evaluation,selection),group in by_seed.groupby(['variant','evaluation','selection']):
        summary.append(dict(variant=variant,evaluation=evaluation,selection=selection,n_seeds=len(group),
            **{f'{measure}_{stat}':float(group[measure].mean() if stat=='mean' else group[measure].std(ddof=1))
               for measure in measures for stat in ['mean','sd']}))
    pd.DataFrame(summary).to_csv(root/'seed_summary.csv',index=False)
    frame.groupby(['variant','evaluation','selection','fold'])[measures].mean().to_csv(root/'fold_means.csv')
    pd.DataFrame(diagnostics).to_csv(root/'training_diagnostics.csv',index=False)
    pd.DataFrame(source_rows).to_csv(root/'source_metrics.csv',index=False)
    report=dict(status='passed',completed_fits=expected,checkpoint_predictions_checked=checked_predictions,
        checkpoint_max_abs_diff=max_difference,checks=['full source split equality','train-only feature statistics',
        'validation checkpoint selection','saved-prediction metrics','checkpoint reload and inference',
        'temperature sign invariance','comparison table equality'],
        aggregation='Equal mean over three source folds for each seed, then mean and sample SD over seeds. CATS is the same holdout across model fits; not nine independent datasets.')
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(pd.DataFrame(summary).query('selection=="bce_temperature"')[['variant','evaluation','accuracy_mean','accuracy_sd','bce_mean']].to_string(index=False))


if __name__=='__main__':main()
