"""A separate 5Hz protocol adapter; the completed 10Hz engine stays unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import highspeed_experiment as engine


def load_native5(root):
    root=Path(root); m=pd.read_csv(root/'windows.csv')
    with np.load(root/'windows_4s.npz',allow_pickle=False) as d:
        x,y,names=d['X'],d['y'].astype(np.float32),d['feature_names'].tolist()
        assert x.shape==(len(m),20,11) and np.isfinite(x).all()
        assert np.array_equal(d['window_index'],m.window_index)
        assert np.array_equal(d['split_group'],m.split_group)
        assert np.array_equal(y,m.label)
    return x,y,m,names


def pretrain_native5(model,inputs,weights,n_draws,cfg,path):
    head=torch.nn.Linear(cfg['model']['d_model'],3)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=cfg['pretrain_lr'],weight_decay=cfg['weight_decay'])
    history=[]
    for epoch in range(cfg['pretrain_epochs']):
        model.train();head.train();total=0.
        order=torch.multinomial(weights,n_draws,replacement=True)
        for ids in order.split(cfg['batch_size']):
            target=inputs[ids];starts=torch.randint(0,16,(len(ids),))
            mask=(torch.arange(20)[None,:]>=starts[:,None]) & (torch.arange(20)[None,:]<starts[:,None]+5)
            corrupted=target.clone()+torch.randn_like(target)*.03;corrupted[mask]=0
            h=model.input_proj(corrupted)
            h=torch.cat([model.cls_token.expand(len(ids),-1,-1),h],dim=1)+model.pos_embed
            prediction=head(model.encoder(h)[:,1:])
            loss=F.smooth_l1_loss(prediction[mask],target[:,:,[0,3,5]][mask])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.)
            optimizer.step();total+=float(loss.detach())*len(ids)
        history.append(dict(epoch=epoch+1,loss=total/n_draws))
    torch.save(model.state_dict(),path/'pretrained_encoder_and_initial_head.pt')
    engine.write_json(path/'pretraining_history.json',history)
    return history[-1]['loss']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    original_split=engine.make_splits
    def native_split(m,fold,train_stride=40):
        split=original_split(m,fold,train_stride)
        if cfg['include_microsim']:
            extra=np.flatnonzero(m.campaign.eq('MicroSim') & m.start_in_segment.mod(train_stride).eq(0))
            split['external_ssl']=np.r_[split['external_ssl'],extra]
        return split
    # Explicitly replace only protocol-specific I/O, split extension and mask rate.
    engine.load_prepared=load_native5
    engine.make_splits=native_split
    engine.masked_pretrain=pretrain_native5
    print('NATIVE5: 20 frames, dt=0.2s, 5-frame mask; include_microsim='+str(cfg['include_microsim']),flush=True)
    engine.main()
    protocol_path=Path(cfg['output'])/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    protocol['ssl_policy']='5Hz: mask all channels in 5 frames (1 second), reconstruct 3 primitives; fixed matched updates'
    protocol['native5']=dict(include_microsim=cfg['include_microsim'],seq_len=20,grid_span_seconds=3.8,
                            maximum_observation_seconds=4.,seeds='single-seed exploratory paired follow-up')
    engine.write_json(protocol_path,protocol)


if __name__=='__main__':main()
