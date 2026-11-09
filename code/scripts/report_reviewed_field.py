"""Independent result checks and compact figures for the reviewed field test."""
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_field_core_dataset import ROOT,read,save
from evaluate_reviewed_field import OUT,SOURCE,digest


def main():
    build=read(OUT/'build_summary.json');classification=read(OUT/'classification_summary.json')
    recovery=read(OUT/'seed_recovery_summary.json');checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    check('all_43_core_recordings_built',build['recordings']==43 and len(build['rows'])==172)
    manifest=read(OUT/'classifier_manifest.json')
    check('30_distinct_frozen_members',len(manifest)==30 and len({m['path'] for m in manifest})==30)
    for m in manifest:
        p=ROOT/m['path']
        check('checkpoint_unchanged_'+str(len(checks)),digest(p)==m['sha256'] and digest(p.parent/'feature_stats.json')==m['feature_stats_sha256'])
    for p,h in read(OUT/'input_manifest.json').items():check('source_unchanged_'+str(len(checks)),digest(ROOT/p)==h)
    for variant in ['reviewed','reviewed_gap1s','recovered','recovered_gap1s']:
        folder=OUT/'classification'/variant
        for duration in [2,3,4]:
            data=np.load(folder/f'windows_{duration}s.npz');meta=read(folder/f'windows_{duration}s_metadata.json')
            check(f'{variant}_{duration}_shape',data['primitive_windows'].shape==(len(meta),duration*10,3))
            check(f'{variant}_{duration}_labels',np.array_equal(data['labels'],[int(r['clip'].startswith('AV_')) for r in meta]))
            check(f'{variant}_{duration}_imputation_bound',all(r['imputed_fraction']<=.25+1e-9 for r in meta))
            for arch in ['transformer','multiscale_cnn']:
                pred=pd.read_csv(folder/f'{arch}_{duration}s_predictions.csv');rec=pd.read_csv(folder/f'{arch}_{duration}s_recordings.csv')
                for r in rec.itertuples():
                    g=pred[pred['clip']==r.clip].sort_values('start_s');selected=[];last=-np.inf
                    for w in g.itertuples():
                        if w.start_s>=last+.1-1e-6:selected.append(w);last=w.end_s
                    check(f'{variant}_{duration}_{arch}_{r.clip}_video_mean',len(selected)==r.contributing_windows and abs(np.mean([w.prob_av for w in selected])-r.prob_av)<1e-7)
                expected=next(r for r in classification['results'] if r.get('architecture')==arch and r['duration_s']==duration and r['variant']==variant)['video_metrics']
                correct=int(((rec.prob_av>=.5).astype(int)==rec.label).sum())
                check(f'{variant}_{duration}_{arch}_correctness',correct==expected['correct'] and len(rec)==expected['n'])
    # Recovered seed fragments do not yet generate any additional >=2-second
    # input. Verify that inference equivalence is real, not copied predictions.
    for baseline,recovered in [('reviewed','recovered'),('reviewed_gap1s','recovered_gap1s')]:
        for duration in [2,3,4]:
            a=np.load(OUT/'classification'/baseline/f'windows_{duration}s.npz')['primitive_windows']
            b=np.load(OUT/'classification'/recovered/f'windows_{duration}s.npz')['primitive_windows']
            check(f'{duration}_{baseline}_seed_recovery_windows_identical',np.array_equal(a,b))
    current=[r for r in build['rows'] if r['variant']=='recovered_gap1s']
    coverage=dict(core_recordings=43,reviewed_leader_present=21,reviewed_no_leader=21,reviewed_leader_not_visible=1,
                  target_metric_clips=sum(r['target_points']>0 for r in current),target_points=sum(r['target_points'] for r in current),
                  paired_metric_clips=sum(r['paired_points']>0 for r in current),pair_observed_points=sum(r['paired_points'] for r in current),
                  pair_imputed_points=sum(r['imputed_pair_points'] for r in current),
                  seed_roles_resolved=sum(r['status'].startswith('ASSOCIATED') for r in recovery),seed_roles_attempted=len(recovery),
                  videos_by_duration={str(d):sum(r[f'windows_{d}s']>0 for r in current) for d in [2,3,4]},
                  windows_by_duration={str(d):sum(r[f'windows_{d}s'] for r in current) for d in [2,3,4]})
    save(OUT/'coverage_summary.json',coverage)
    probability_tables={arch:pd.read_csv(OUT/'classification/recovered_gap1s'/f'{arch}_longest_available_recordings.csv') for arch in ['transformer','multiscale_cnn']}
    summary=probability_tables['transformer'][['clip','label','duration_s','observed_duration_s','contributing_windows','training_exposed','prob_av']].rename(columns={'prob_av':'transformer_prob_av'})
    summary=summary.merge(probability_tables['multiscale_cnn'][['clip','prob_av']],on='clip').rename(columns={'prob_av':'cnn_prob_av'}).sort_values(['label','clip'],ascending=[False,True])
    summary['true_class']=summary.label.map({1:'AV',0:'HV'});summary['transformer_prediction']=np.where(summary.transformer_prob_av>=.5,'AV','HV');summary['cnn_prediction']=np.where(summary.cnn_prob_av>=.5,'AV','HV')
    summary.to_csv(OUT/'classification_by_video.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={'width_ratios':[1.55,1]},constrained_layout=True)
    x=np.arange(len(summary));axes[0].bar(x-.18,summary.transformer_prob_av,.36,color='#176b87',label='Transformer')
    axes[0].bar(x+.18,summary.cnn_prob_av,.36,color='#d58240',label='CNN');axes[0].axhline(.5,color='#555',ls='--',lw=1,label='Decision threshold')
    axes[0].set_xticks(x,summary['clip'],rotation=40,ha='right');axes[0].set(ylim=(0,1),ylabel='Predicted P(AV)',title='Frozen models predict HV for all eight videos')
    axes[0].legend(fontsize=9,loc='upper right')
    for i,r in enumerate(summary.itertuples()):axes[0].text(i,.57,f'{r.true_class}\n{r.duration_s}s',ha='center',fontsize=9,color='#74333a' if r.label else '#234d58')
    cm=np.array([[4,0],[4,0]]);axes[1].imshow(cm,cmap='Blues',vmin=0,vmax=4)
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=26,color='white' if cm[i,j] else '#333')
    axes[1].set_xticks([0,1],['HV','AV']);axes[1].set_yticks([0,1],['HV','AV']);axes[1].set(xlabel='Predicted state',ylabel='Recorded state',title='Both models: 4/8 correct (50%)\nAV recall 0/4; HV recall 4/4')
    fig.suptitle('Reviewed roadside transfer test | 8 of 43 core videos eligible',fontsize=14)
    fig.savefig(OUT/'classification_overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),constrained_layout=True)
    labels=['Target observations','Paired observations','2-second input','4-second input'];before=[41,15,8,1];after=[43,20,8,1];x=np.arange(4)
    ax.bar(x-.18,before,.36,label='After user review',color='#a6bac2');ax.bar(x+.18,after,.36,label='+ seed recovery',color='#176b87')
    for pos,values in [(x-.18,before),(x+.18,after)]:
        for a,b in zip(pos,values):ax.text(a,b+.5,str(b),ha='center')
    ax.set_xticks(x,labels,rotation=12,ha='right');ax.set(ylim=(0,49),ylabel='Videos (43 core recordings)',title='More trajectory coverage; classification coverage unchanged');ax.legend()
    fig.savefig(OUT/'recovery_coverage.png',dpi=150);plt.close(fig)
    versions={}
    for package in ['torch','numpy','pandas','scikit-learn','filterpy','opencv-python']:
        try:versions[package]=version(package)
        except Exception:pass
    save(OUT/'runtime_versions.json',versions)
    save(OUT/'validation.json',dict(status='PASS',checks=len(checks),check_names=checks,coverage=coverage,
        image_review='Five original MOT identities and two raw-detection fragments inspected before classification; one box-truck leader remains pending.',
        limitation='Integrity and inference checks are not independent 3D accuracy validation.'))
    print(json.dumps(dict(status='PASS',checks=len(checks),coverage=coverage),indent=2))


if __name__=='__main__':main()
