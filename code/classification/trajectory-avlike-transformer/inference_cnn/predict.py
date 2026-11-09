"""Inference for new 4-second 10 Hz windows; no gradients or extra context."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from networks import TrajectoryNet
from data_protocol import features,normalize

BUNDLE=Path(__file__).resolve().parent

def check_windows(raw):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=3 or raw.shape[1:]!=(40,3) or len(raw)==0:
        raise ValueError('Expected nonempty [N,40,3] windows: follower speed, leader speed (m/s), net gap (m).')
    if not np.isfinite(raw).all():
        raise ValueError('Input contains nonfinite values.')
    x=features(raw,'legacy11',7)
    residual=np.abs(raw[:,-1,2]-raw[:,0,2]-np.trapz(raw[:,:,1]-raw[:,:,0],dx=.1,axis=1))
    valid=((raw[:,:,0]>22.352).all(1)&(raw[:,:,:2]<55).all((1,2))
           &(raw[:,:,1]>=0).all(1)&(raw[:,:,2]>1).all(1)&(raw[:,:,2]<200).all(1)
           &(x[:,:,0]>22.352).all(1)&(x[:,:,5]>0).all(1)
           &(np.abs(x[:,:,[1,4]]).max((1,2))<=8.001)&(residual<=8.001))
    if not valid.all():
        raise ValueError(f'{int((~valid).sum())} windows fail the training speed/physical-quality protocol; '
                         f'first invalid row positions: {np.flatnonzero(~valid)[:10].tolist()}. No rows were silently removed.')
    return raw,x

@torch.no_grad()
def predict(raw,fold=None,calibration='none',source_ids=None,device='cpu'):
    if calibration not in ('none','validation'):
        raise ValueError('calibration must be none or validation')
    _,x=check_windows(raw)
    info=json.loads((BUNDLE/'bundle.json').read_text())
    members=[member for member in info['members'] if fold is None or member['fold_name']==fold]
    if not members:
        raise ValueError('Unknown fold; use part2, part3, or part4_highway.')
    if source_ids is not None:
        if len(source_ids)!=len(x):
            raise ValueError('source_ids length does not match windows')
        incoming=set(np.asarray(source_ids).astype(str))
        for member in members:
            overlap=incoming&set(member['training_sources'])
            if overlap:
                raise ValueError(f'Input overlaps training sources {sorted(overlap)}. '
                                 'For Vicolungo evaluation select only the matching outer fold.')
    probabilities=[]
    for member in members:
        file=BUNDLE/member['checkpoint']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=member['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {file.name}')
        checkpoint=torch.load(file,map_location=device,weights_only=True)
        cfg=checkpoint['config']
        model=TrajectoryNet(checkpoint['input_dim'],cfg['architecture'],cfg['width'],cfg['dropout'],cfg['layers']).to(device)
        model.load_state_dict(checkpoint['model_state_dict']);model.eval()
        xx=torch.from_numpy(normalize(x,member['stats'])).to(device)
        z=torch.cat([model(batch) for batch in xx.split(512)]).cpu().numpy()
        divisor=member['temperature'] if calibration=='validation' else 1.
        probabilities.append(expit(z/divisor))
    probabilities=np.stack(probabilities)
    p=probabilities.mean(0)
    return pd.DataFrame(dict(acc_probability=p,predicted_label=(p>=.5).astype(int),
        model_probability_sd=probabilities.std(0),models=len(members))),info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path,help='NPZ with primitive_windows [N,40,3] at 10 Hz')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fold',choices=['part2','part3','part4_highway'])
    parser.add_argument('--calibration',choices=['none','validation'],default='none')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.input,allow_pickle=False) as data:
        raw=data['primitive_windows']
        ids=data['window_index'] if 'window_index' in data else np.arange(len(raw))
        sources=data['split_group'] if 'split_group' in data else None
        if 'dt' in data and not np.allclose(data['dt'],.1):
            raise ValueError('Input dt must be 0.1 seconds.')
    pred,info=predict(raw,args.fold,args.calibration,sources,args.device)
    pred.insert(0,'window_index',ids)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pred.to_csv(args.output,index=False)
    print(json.dumps(dict(rows=len(pred),models=int(pred.models.iloc[0]),fold=args.fold,
                         calibration=args.calibration,threshold=.5,selection=info['selection'])))

if __name__=='__main__':
    main()
