"""Contract, leakage and reproducibility checks for the three-second experiment."""
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from three_second_classifier import ROOT,OUT,DATA,ThreeSecondTransformer,features,normalize,save


def main():
    torch.set_num_threads(2);torch.manual_seed(10);checks=[]
    model=ThreeSecondTransformer().eval();x=torch.randn(2,31,11);mask=torch.ones_like(x,dtype=torch.bool);mask[:,:,3:]=False
    with torch.no_grad():
        a=model(x,mask);changed=x.clone();changed[:,:,3:]=float('nan');b=model(changed,mask)
        assert torch.equal(a,b) and torch.isfinite(b).all()
    checks.append('Masked leader values, including NaN, cannot alter predictions')
    for length in [21,30,32,40]:
        try:model(torch.zeros(1,length,11),torch.ones(1,length,11,dtype=torch.bool))
        except ValueError:pass
        else:raise AssertionError('Accepted wrong duration')
    checks.append('Reject 21,30,32,40 samples instead of silently padding or changing time scale')
    raw=np.column_stack([np.linspace(20,21,31),np.linspace(21,22,31),np.linspace(25,28,31)])
    avail=np.ones(31,bool);avail[9:19]=False
    f,m=features(raw,avail);bad=raw.copy();bad[~avail,1:]=1e8;g,n=features(bad,avail)
    assert np.array_equal(f,g) and np.array_equal(m,n);assert not m[9:19,3:].any()
    checks.append('Leader smoothing/derivatives are independent of hidden measurements across the missing interval')
    for fold in range(3):
        s=json.loads((OUT/f'public/fold{fold}_split.json').read_text());a,b,c=[set(s['source_ids'][k]) for k in ['train','validation','test']]
        assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    checks.append('All public training/validation/test source groups are disjoint')
    windows=pd.read_csv(OUT/'windows.csv');data=np.load(OUT/'dataset.npz');assert data['features'].shape==(len(windows),31,11)
    assert data['feature_mask'][:,:,:3].all() and np.isfinite(data['features']).all()
    for row in windows.to_dict('records'):
        s=pd.read_csv(OUT/'clips'/row['clip']/'selected_3s.csv');assert len(s)==31 and np.allclose(np.diff(s.t_s),.1) and np.isclose(s.t_s.iloc[-1]-s.t_s.iloc[0],3)
    checks.append('Every eligible video contains 31 genuine time samples spanning 3.0 seconds, with complete ego input')
    inv=pd.read_csv(OUT/'inventory.csv').set_index('clip');assert not inv.loc['HV_T_EW_2','eligible_3s'] and inv.loc['HV_T_EW_2','source_time_jump']>3
    checks.append('Source-video timestamp jump is not interpolated into a valid window')
    pred=pd.read_csv(OUT/'field/seed_predictions.csv')
    for held,fold in enumerate(json.loads((OUT/'field/splits.json').read_text())):
        assert fold['test'] not in fold['train']+fold['validation'] and set(fold['train']).isdisjoint(fold['validation'])
        ck=torch.load(OUT/f'field/held{held}_seed42.pt',map_location='cpu');model.load_state_dict(ck['model_state_dict']);model.eval()
        xx=torch.from_numpy(normalize(data['features'][held:held+1],data['feature_mask'][held:held+1],ck['normalization']))
        with torch.no_grad():p=float(torch.sigmoid(model(xx,torch.from_numpy(data['feature_mask'][held:held+1])))[0])
        stored=pred[(pred['clip']==fold['test'])&(pred.seed==42)].probability.iloc[0];assert abs(p-stored)<1e-5,(p,stored)
        changes=json.loads((OUT/f'field/held{held}_seed42_log.json').read_text())['parameter_change_l2']
        assert any(v>0 for k,v in changes.items() if k.startswith('encoder.'))
    checks.extend(['All field test videos are disjoint from fitting and inner validation','Reloaded saved models reproduce probabilities','Transformer encoder parameters actually changed during full-network adaptation'])
    public=np.load(DATA/'windows_3s.npz');meta=pd.read_csv(DATA/'windows.csv');assert public['primitive_windows'].shape==(len(meta),31,3)
    assert np.allclose(meta.observation_span,3.)
    checks.append('Rebuilt public windows have the same three-second input contract')
    save(OUT/'validation.json',dict(status='PASS',checks=checks,eligible_videos=len(windows),public_windows=len(meta),
        dataset_sha256=hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='PASS',checks=checks),indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
