"""Independent invariance and split checks for missing-leader adaptation."""
import json
import numpy as np
import pandas as pd
import torch
import todd_experiment_data as D
import todd_experiment_train as T
from data_protocol import features
from workspace_paths import verify_recorded_input


def main():
    torch.set_num_threads(2);torch.manual_seed(81)
    ds=np.load(D.OUT/'dataset.npz');ci=ds['clip_index'];checks=[]
    cp=torch.load(D.ROOT/D.CFG['initial_checkpoint'],map_location='cpu',weights_only=True)
    source=D.Q.read((D.ROOT/D.CFG['initial_checkpoint']).parent/'feature_stats.json')
    old={k:np.array(source[k]) for k in ['mean','std']};ss={k:v.tolist() for k,v in old.items()}
    rng=np.random.default_rng(81);raw=rng.normal([27,28,35],[.5,.5,2],size=(21,3))
    f,m,s=D.featurize(raw[:,0],raw[:,1],raw[:,2],np.ones(21,bool),np.zeros(21,int),np.zeros(21,int))
    np.testing.assert_allclose(f,features(raw[None],smoothing=7)[0],rtol=1e-5,atol=1e-5)
    checks.append('all-present features reproduce existing legacy11 source features')
    available=np.ones(21,bool);available[7:14]=False
    a=D.featurize(raw[:,0],raw[:,1],raw[:,2],available,np.zeros(21,int),np.where(available,0,-1))
    altered=raw.copy();altered[7:14,1:]=1e6
    b=D.featurize(altered[:,0],altered[:,1],altered[:,2],available,np.zeros(21,int),np.where(available,0,-1))
    np.testing.assert_array_equal(a[0],b[0]);assert not a[1][7:14,3:].any()
    checks.append('unobserved leader values do not influence features or derivatives')
    tx=T.tensors(ds,ss,'masked');base=T.MaskedTransformer(cp,ss,old).eval()
    with torch.no_grad():
        reference=base(*tx)
        mutated=[v.clone() for v in tx];mutated[0][~ds['valid_tokens']]=1e4;mutated[2][~ds['valid_tokens']]=1
        np.testing.assert_allclose(base(*mutated).numpy(),reference.numpy(),atol=2e-5,rtol=1e-5)
    checks.append('arbitrary padded values cannot affect CLS prediction')
    for fold in D.Q.read(D.OUT/'folds.json'):
        tr=[D.CLIPS.index(c) for c in fold['train']];te=D.CLIPS.index(fold['test'][0]);va=fold['validation']
        assert len(tr)==10 and len(va)==2 and fold['test'][0] not in fold['train']+va
        st=T.stats_for(ds['features'],ds['feature_mask'],ci,tr)
        xx=ds['features'].copy();xx[ci==te]=1e5
        st2=T.stats_for(xx,ds['feature_mask'],ci,tr);assert st==st2
        adapted=T.MaskedTransformer(cp,st,old).eval();at=T.tensors(ds,st,'masked')
        with torch.no_grad():np.testing.assert_allclose(adapted(*at).numpy(),reference.numpy(),rtol=2e-4,atol=2e-4)
    checks+=['all 13 folds video-disjoint','test perturbation leaves fitted normalization unchanged',
             'train-normalized initialization equals source normalization even with missing values']
    meta=pd.read_csv(D.OUT/'windows.csv')
    for _,g in meta.groupby('clip'):
        g=g.sort_values('start_s');assert (g.start_s.iloc[1:].to_numpy()>g.end_s.iloc[:-1].to_numpy()).all()
    checks.append('no two windows within a video share time samples')
    if (D.OUT/'classification_summary.json').exists():
        pred=pd.read_csv(D.OUT/'predictions_by_seed.csv');video=pd.read_csv(D.OUT/'predictions_by_video.csv')
        assert len(video)==65 and len(pred)==117
        assert ((pred.prob_av>=0)&(pred.prob_av<=1)).all()
        for row in video.itertuples():
            p=pred[(pred.model==row.model)&(pred['clip']==row.clip)].prob_av.mean();assert abs(p-row.prob_av)<1e-8
        for fold in D.Q.read(D.OUT/'folds.json'):
            for mode in ['ego','masked']:
                for seed in D.CFG['training_seeds']:
                    path=D.OUT/'checkpoints'/f'{mode}_fold{fold["fold"]:02d}_seed{seed}.pt'
                    c=torch.load(path,map_location='cpu',weights_only=True)
                    assert c['train']==fold['train'] and c['validation']==fold['validation'] and c['test']==fold['test']
                    assert c['normalization']['training_clip_indices']==[D.CLIPS.index(v) for v in fold['train']]
                    for k,v in cp['model_state_dict'].items():
                        if k.startswith('encoder.') or k in ['pos_embed','cls_token']:
                            assert torch.equal(c['model_state_dict'][k],v)
        checks+=['all 78 checkpoint splits and normalization owners verified','source encoder unchanged in every fold',
                 'video ensemble predictions match per-seed records']
        # Reload a nontrivial fold for independent reproduction of saved probability.
        c=torch.load(D.OUT/'checkpoints'/'masked_fold00_seed42.pt',map_location='cpu',weights_only=True)
        model=T.MaskedTransformer(cp,c['normalization'],old).eval();model.load_state_dict(c['model_state_dict'])
        test=np.flatnonzero(ci==0);p=T.predict(model,T.tensors(ds,c['normalization'],'masked'),test,ci)[0]
        row=pred[(pred.model=='finetuned_transformer_masked')&(pred.seed==42)&(pred.fold==0)].iloc[0]
        assert abs(p-row.prob_av)<1e-7;checks.append('saved fold checkpoint reproduces held-video probability')
    for p,h in D.Q.read(D.OUT/'input_manifest.json').items():verify_recorded_input(p,h)
    checks.append('original video, trajectory, GPS, annotation and checkpoint hashes unchanged')
    D.Q.save(D.OUT/'independent_validation.json',dict(status='PASS',checks=checks))
    print('PASS',len(checks),'independent invariance and protocol checks')


if __name__=='__main__':main()
