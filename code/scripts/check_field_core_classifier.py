"""Exercise restored Transformer weights on actual short field windows.

This is an interface check. Source normalization and weights are not calibrated
for the lower-speed field domain, and no field accuracy or fine-tuning is claimed.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent.parent
PROJECT=ROOT/'classification/trajectory-avlike-transformer'
sys.path.insert(0,str(PROJECT/'inference_cnn'))
from networks import TrajectoryNet
from data_protocol import features,normalize
from build_field_core_dataset import interval_groups,windows_for_group,save

def main():
    torch.set_num_threads(4)
    out=ROOT/'experiments/field_core_20261101';checks=[]
    paths=[p for p in sorted((PROJECT/'artifacts').rglob('best_bce.pt')) if 'transformer' in str(p.relative_to(PROJECT/'artifacts')).lower()]
    members=[]
    for p in paths:
        checkpoint=torch.load(p,map_location='cpu',weights_only=True);c=checkpoint['config']
        if c['architecture']!='transformer' or c.get('study')=='multisegment':continue
        model=TrajectoryNet(checkpoint['input_dim'],c['architecture'],c['width'],c['dropout'],c['layers']).eval()
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        members.append((p,model,c,json.loads((p.parent/'feature_stats.json').read_text())))
    assert len(members)==15
    for duration in [2,3,4]:
        p=out/f'windows_{duration}s.npz';d=np.load(p,allow_pickle=False);raw=d['primitive_windows'];meta=pd.read_csv(out/f'windows_{duration}s.csv')
        assert raw.shape==(len(meta),10*duration,3) and np.isfinite(raw).all()
        assert float(d['dt'])==.1
        assert np.array_equal(d['labels'],meta.label.to_numpy(dtype=np.int64))
        if not len(raw):checks.append(dict(duration_s=duration,windows=0,status='NO_WINDOWS'));continue
        probs=[];clipped=[]
        for path,model,c,stats in members:
            assert raw.shape[1]<=model.seq_len
            x=features(raw,c.get('features','legacy11'),c.get('smoothing',7))
            outside=(x<np.array(stats['lower'])) | (x>np.array(stats['upper']))
            clipped.append(float(outside.mean()))
            with torch.no_grad():p=torch.sigmoid(model(torch.from_numpy(normalize(x,stats)))).numpy()
            assert p.shape==(len(raw),) and np.isfinite(p).all();probs.append(p)
        probabilities=np.stack(probs)
        pd.concat([meta,pd.DataFrame(dict(source_model_acc_probability=probabilities.mean(0),model_sd=probabilities.std(0)))],axis=1).to_csv(out/f'transformer_{duration}s_interface_predictions.csv',index=False)
        checks.append(dict(duration_s=duration,windows=len(raw),models=len(members),status='FINITE_OUTPUTS',
                           input_speed_range_mps=[float(raw[:,:,:2].min()),float(raw[:,:,:2].max())],
                           source_feature_clipping_fraction_range=[min(clipped),max(clipped)]))
    # Behavioral safeguards: stopped traffic remains valid; a leader switch and
    # a long missing-data gap must break segments even when total time is enough.
    rows=[dict(t_s=i*.1,token=1,follower_speed_mps=0.,leader_speed_mps=0.,net_gap_m=5.) for i in range(40)]
    assert len(windows_for_group(rows,2,10,1))>0
    switched=[dict(r,token=1 if i<20 else 2) for i,r in enumerate(rows)]
    assert len(interval_groups(switched,.2))==2
    assert not any(windows_for_group(g,3,10,1) for g in interval_groups(switched,.2))
    hole=rows[:12]+rows[18:];assert len(interval_groups(hole,.2))==2
    result=dict(status='PASS_INTERFACE_ONLY',checks=checks,behavioral_checks=['zero-speed samples permitted','leader changes split windows','long missing-data gaps split windows'],
                classifier_finetuning='NOT_RUN: current exported samples are HV-only and belong to the existing test recording.',
                field_accuracy=None,note='Actual field windows at native 10 Hz; no padding or time warping. Shorter positional-embedding prefixes execute successfully, but source normalization clips low-speed inputs and the model needs field-domain training/validation. Overlapping durations are not independent examples.')
    save(out/'classifier_check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
