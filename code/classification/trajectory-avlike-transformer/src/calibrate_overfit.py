"""Fit validation-only temperature to existing AUROC checkpoints, without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from dataset import make_dataloaders
from overfit_experiment import infer, score_frame
from train import build_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    results=pd.read_csv(root/'comparison.csv')
    if (results.selection=='auroc_temperature').any():
        raise ValueError('AUROC calibration already exists; refusing duplicate results')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    rows=[]
    for _, row in results[results.selection=='auroc'].iterrows():
        path=root/row.variant/f'seed_{row.seed}'/row.fold
        ckpt=torch.load(path/'best_auroc.pt',map_location=device,weights_only=True)
        config=ckpt['config']
        splits=json.loads((path/'splits.json').read_text())
        df=pd.read_csv(config['data']['processed_csv'])
        _,datasets,diagnostics=make_dataloaders(config,df,splits,fit_stats=False,stats_path=str(path/'feature_stats.json'))
        model=build_model(config['model'],device)
        model.load_state_dict(ckpt['model_state_dict'])
        x=torch.stack([datasets['val'][i][0] for i in range(len(datasets['val']))]).to(device)
        y=np.array([w['y'] for w in datasets['val'].windows])
        logits=infer(model,x,config['training']['batch_size']).astype(np.float64)
        def objective(log_t):
            z=logits/np.exp(log_t)
            return float(np.mean(np.logaddexp(0,z)-y*z))
        fit=minimize_scalar(objective,bounds=(np.log(.1),np.log(20.)),method='bounded')
        log_t=min([float(fit.x),0.,np.log(.1),np.log(20.)],key=objective)
        temperature=float(np.exp(log_t))
        (path/'auroc_calibration.json').write_text(json.dumps({'temperature':temperature,'fit_split':'val','val_bce_before':objective(0.),'val_bce_after':objective(log_t)},indent=2))
        # Apply the already fitted temperature to held-out predictions.
        frame=pd.read_csv(path/'auroc_predictions.csv')
        frame.logit/=temperature
        frame.prob_AV=expit(frame.logit)
        frame.to_csv(path/'auroc_temperature_predictions.csv',index=False)
        metrics=score_frame(frame,diagnostics['train']['num_hv_windows']/diagnostics['train']['num_av_windows'])
        assert abs(metrics['window_auroc']-row.window_auroc)<1e-10
        updated=row.to_dict()
        updated.update(selection='auroc_temperature',temperature=temperature,**metrics)
        rows.append(updated)
    combined=pd.concat([results,pd.DataFrame(rows)],ignore_index=True)
    combined.to_csv(root/'comparison.csv',index=False)
    metrics=[c for c in combined if c.startswith(('window_','trajectory_'))]
    means=combined.groupby(['variant','selection','seed'])[metrics].mean()
    means.to_csv(root/'seed_means.csv')
    means.groupby(['variant','selection']).agg(['mean','std']).to_csv(root/'overview.csv')
    print(pd.DataFrame(rows).groupby('variant')[['window_bce','window_auroc','trajectory_auroc']].mean())


if __name__=='__main__':
    main()
