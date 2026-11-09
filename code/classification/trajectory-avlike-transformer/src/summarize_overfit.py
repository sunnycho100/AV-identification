"""Summarize saved experiments without training or selecting on test scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args=parser.parse_args()
    root=args.directory
    results=pd.read_csv(root/'comparison.csv')
    # Correct ranking ties in constant baselines (group mean rounding can break ties).
    for index, row in results[results.selection.isin(['constant_half','train_prior'])].iterrows():
        frame=pd.read_csv(root/row.variant/f'seed_{row.seed}'/row.fold/f'{row.selection}_predictions.csv')
        assert frame.logit.nunique()==1
        results.loc[index,'trajectory_auroc']=0.5
        results.loc[index,'trajectory_auprc']=frame.drop_duplicates('Trajectory_ID').label.mean()
    results.to_csv(root/'comparison.csv',index=False)
    metric_columns=[c for c in results if c.startswith(('window_','trajectory_'))]
    seed_means=results.groupby(['variant','selection','seed'])[metric_columns].mean()
    seed_means.to_csv(root/'seed_means.csv')
    seed_means.groupby(['variant','selection']).agg(['mean','std']).to_csv(root/'overview.csv')
    for (variant,seed,fold), rows in results.groupby(['variant','seed','fold']):
        (root/variant/f'seed_{seed}'/fold/'metrics.json').write_text(rows.to_json(orient='records',indent=2))
    # Each variant must evaluate the very same test windows, labels and split IDs.
    audits=[]
    for (seed,fold), group in results.groupby(['seed','fold']):
        reference=None
        ref_splits=None
        for variant in group.variant.unique():
            path=root/variant/f'seed_{seed}'/fold
            pred=pd.read_csv(path/'bce_predictions.csv')
            identities=pred[['Trajectory_ID','window_start_time','label']]
            splits=json.loads((path/'splits.json').read_text())
            if reference is not None:
                pd.testing.assert_frame_equal(reference,identities)
                assert ref_splits==splits
            reference=identities
            ref_splits=splits
            for pred_path in path.glob('*_predictions.csv'):
                frame=pd.read_csv(pred_path)
                assert np.isfinite(frame[['prob_AV','logit']]).all().all()
                assert frame.prob_AV.between(0,1).all()
        audits.append({'seed':int(seed),'fold':fold,'matched_variants':int(group.variant.nunique()),'windows':len(reference)})
    (root/'audit.json').write_text(json.dumps(audits,indent=2))
    metrics=['window_bce','window_brier','window_auroc','trajectory_bce','trajectory_auroc','trajectory_balanced_accuracy']
    means=results.groupby(['variant','selection'])[metrics].mean()
    print(means.round(4).to_string())
    counts=[]
    for (variant,selection), group in results.groupby(['variant','selection']):
        initial=results[(results.variant==variant)&(results.selection=='initial')]
        paired=group.merge(initial,on=['seed','fold'],suffixes=('','_initial'))
        counts.append(dict(variant=variant,selection=selection,folds=len(group),
                           bce_better_than_half=int((group.window_bce<np.log(2)-1e-10).sum()),
                           bce_better_than_initial=int((paired.window_bce<paired.window_bce_initial-1e-10).sum()),
                           epoch_zero_selected=int((group.epoch==0).sum())))
    pd.DataFrame(counts).to_csv(root/'baseline_comparisons.csv',index=False)
    # One figure: matched seed 42 weighted model illustrates conflicting selectors.
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,fold in zip(axes,['part2','part3','highway']):
        history=json.loads((root/'weighted_transformer'/'seed_42'/fold/'history.json').read_text())
        ax.plot([r['epoch'] for r in history], [r['val']['bce'] for r in history],label='Validation ordinary BCE')
        ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history],label='Training weighted BCE',alpha=.7)
        ax.axhline(np.log(2),color='gray',ls=':',label='Constant 0.5 ordinary BCE')
        choices=results[(results.variant=='weighted_transformer')&(results.seed==42)&(results.fold==fold)]
        for selector,color in [('bce','green'),('auroc','red')]:
            ax.axvline(choices[choices.selection==selector].epoch.iloc[0],color=color,ls='--',label=f'Best val {selector}')
        ax.set(title=fold,xlabel='Epoch',ylabel='Loss')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.16,1,1))
    fig.savefig(root/'selection_curves.png',dpi=160)
    plt.close(fig)
    chosen=means.loc[[(v,s) for v,s in means.index if s in ['bce','auroc','bce_temperature','auroc_temperature']]]
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(np.arange(len(chosen)),chosen.window_bce)
    ax.axhline(np.log(2),color='red',ls='--',label='Constant 0.5')
    ax.set_xticks(np.arange(len(chosen)),[f'{v}\n{s}' for v,s in chosen.index],rotation=35,ha='right')
    ax.set_ylabel('Held-out ordinary window BCE (9-fold mean)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/'probability_comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
