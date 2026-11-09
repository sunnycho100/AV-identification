"""Meaningful padding, missingness, data-contract and held-out reproducibility checks."""
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from variable_length_classifier import ROOT,OUT,OLD,MAX,MIN,VariableTransformer,features,variants,normalize,save


def main(final=False):
    torch.set_num_threads(2);torch.manual_seed(501);checks=[]
    model=VariableTransformer().eval();n=17
    x=torch.randn(1,MAX,11);v=torch.arange(MAX)[None,:]<n;m=v[:,:,None].expand(-1,-1,11).clone();m[:,5:10,3:]=False
    with torch.no_grad():
        expected=model(x,m,v);bad=x.clone();bad[~m]=float('nan');actual=model(bad,m,v)
        assert torch.equal(expected,actual) and torch.isfinite(actual).all()
        truncated=model(x[:,:n],m[:,:n],v[:,:n]);assert torch.allclose(expected,truncated,atol=1e-6,rtol=1e-6)
    checks.extend(['Masked padding and missing-leader NaNs cannot change prediction','A short input and its right-padded form have equal predictions'])
    raw=np.column_stack([np.linspace(20,21,26),np.linspace(21,22,26),np.linspace(25,28,26)])
    a=np.ones(26,bool);a[7:16]=False;f,m=features(raw,a);bad=raw.copy();bad[~a,1:]=1e8;g,k=features(bad,a)
    assert np.array_equal(f,g) and np.array_equal(m,k);assert not m[7:16,3:].any()
    checks.append('Hidden leader values do not influence features or derivatives across missing intervals')
    rng=np.random.default_rng(9)
    for length in [5,6,17,22,26]:
        xx,mm,vv,ll=variants(raw[:length],a[:length],rng)
        assert np.array_equal(mm[:,:,:3],np.repeat(vv[:,:,None],3,axis=2))
        assert (ll>=MIN).all() and (ll<=length).all() and not mm[~vv].any()
        with torch.no_grad():assert torch.isfinite(model(torch.from_numpy(xx),torch.from_numpy(mm),torch.from_numpy(vv))).all()
    checks.append('All supported real lengths work, including 0.4s minimum and 2.5s maximum')
    z=np.load(OUT/'dataset.npz');meta=pd.read_csv(OUT/'windows.csv');reviews={r['clip']:r for r in json.loads((OUT/'human_review.json').read_text())['records']}
    assert z['features'].shape==(len(meta),26,11)
    assert np.array_equal(z['feature_mask'][:,:,:3],np.repeat(z['valid_tokens'][:,:,None],3,axis=2))
    assert not z['feature_mask'][~z['valid_tokens']].any() and np.isfinite(z['features']).all()
    for i,row in enumerate(meta.to_dict('records')):
        s=pd.read_csv(OUT/'clips'/row['clip']/'selected.csv');n=len(s)
        assert n==row['samples']==int(z['valid_tokens'][i].sum())
        assert np.allclose(np.diff(s.t_s),.1) and np.isclose(s.t_s.iloc[-1]-s.t_s.iloc[0],row['duration_s'])
        assert row['duration_s']<=2.5+1e-8
        if row['clip'] in reviews:
            r=reviews[row['clip']];assert s.t_s.min()>=r['start_s']-1e-8 and s.t_s.max()<=r['end_s']+1e-8
        tracks=pd.read_csv(OUT/'clips'/row['clip']/'target_trajectory.csv')
        assert s.t_s.min()>=tracks.t_s.min()-1e-8 and s.t_s.max()<=tracks.t_s.max()+1e-8
    assert meta.loc[meta['clip']=='HV_T_EW_2','end_s'].iloc[0]<5.973
    checks.extend(['Field inputs retain true 0.1s sampling, reviewed bounds and explicit valid lengths','No extrapolated time or HV_T_EW_2 frozen-tail timestamp jump enters the classifier'])
    manifest=json.loads((OUT/'trajectory_input_manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks.append('Frozen trajectory inputs and protocol match provenance hashes')
    if final:
        sources=json.loads((OUT/'source_provenance.json').read_text())
        for name,digest in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        checks.append('Source calibration, native timestamps, original review and initial checkpoints remain unchanged')
        split=json.loads((OUT/'public/split.json').read_text());groups=[set(split['source_ids'][k]) for k in ['train','validation','test']]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
        checks.append('Public source groups remain disjoint, including initialization checkpoint training sources')
        pred=pd.read_csv(OUT/'field/seed_predictions.csv');folds=json.loads((OUT/'field/splits.json').read_text());reloaded=0
        for held,fold in enumerate(folds):
            assert fold['test'] not in fold['train']+fold['validation'] and set(fold['train']).isdisjoint(fold['validation'])
            for seed in [42,52,62]:
                ck=torch.load(OUT/f'field/held{held}_seed{seed}.pt',map_location='cpu');model.load_state_dict(ck['model_state_dict']);model.eval()
                x=torch.from_numpy(normalize(z['features'][held:held+1],z['feature_mask'][held:held+1],ck['normalization']))
                with torch.no_grad():p=float(torch.sigmoid(model(x,torch.from_numpy(z['feature_mask'][held:held+1]),torch.from_numpy(z['valid_tokens'][held:held+1])))[0])
                stored=pred[(pred['clip']==fold['test'])&(pred.seed==seed)].probability.iloc[0];assert abs(p-stored)<1e-5
                changed=json.loads((OUT/f'field/held{held}_seed{seed}_log.json').read_text())['parameter_change_l2']
                assert any(v>0 for k,v in changed.items() if k.startswith('encoder.'));reloaded+=1
        checks.extend(['All variants of each held-out video are excluded from fitting and inner validation',f'All {reloaded} field checkpoints reproduce saved predictions','Encoder weights actually updated during adaptation'])
    save(OUT/('validation.json' if final else 'pretraining_validation.json'),dict(status='PASS',checks=checks,
        dataset_sha256=hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest()))
    print(json.dumps(dict(status='PASS',checks=checks)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();main(args.final)
