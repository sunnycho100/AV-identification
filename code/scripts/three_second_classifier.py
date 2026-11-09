"""Strict three-second Transformer with explicit missing-interaction training."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from todd_experiment_data import featurize

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/field_todd_3s_20261102'
DATA=ROOT/'classification/trajectory-avlike-transformer/data/prepared_3s_20261102'
CFG=json.loads((ROOT/'configs/field_todd_3s.json').read_text(encoding='utf-8'))
PATTERNS=['full','none','prefix','suffix','internal']


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def features(raw,available):
    available=np.asarray(available,bool)
    return featurize(raw[:,0],raw[:,1],raw[:,2],available,np.zeros(31,int),np.where(available,0,-1))[:2]


def augment(raw,observed,rng):
    result=[];masks=[]
    left=int(rng.integers(7,14));right=int(rng.integers(18,25))
    for k in range(5):
        a=observed.copy()
        if k==1:a[:]=False
        if k==2:a[right:]=False
        if k==3:a[:left]=False
        if k==4:a[left:right]=False
        f,m=features(raw,a);result.append(f);masks.append(m)
    return np.stack(result),np.stack(masks)


class ThreeSecondTransformer(nn.Module):
    def __init__(self,width=64,dropout=.2):
        super().__init__();self.ego=nn.Linear(3,width)
        self.interaction=nn.Sequential(nn.Linear(16,width),nn.GELU(),nn.Linear(width,width))
        self.missing=nn.Parameter(torch.zeros(1,1,width))
        self.position=nn.Parameter(torch.zeros(1,32,width));self.cls=nn.Parameter(torch.zeros(1,1,width))
        layer=nn.TransformerEncoderLayer(width,4,width*4,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
        for p in [self.missing,self.position,self.cls]:nn.init.trunc_normal_(p,std=.02)

    def forward(self,x,mask):
        if x.shape[1:]!=(31,11):raise ValueError('Requires 31 real samples at 10Hz spanning exactly 3.0s')
        if not mask[:,:,:3].all():raise ValueError('Missing ego samples are not supported; no padding-as-time')
        x=torch.where(mask,x,torch.zeros_like(x))
        leader=mask[:,:,5:6].float()
        inter=self.interaction(torch.cat([x[:,:,3:],mask[:,:,3:].float()],-1))
        h=self.ego(x[:,:,:3])+leader*inter+(1-leader)*self.missing
        h=torch.cat([self.cls.expand(len(x),-1,-1),h],1)+self.position
        return self.head(self.encoder(h)[:,0]).squeeze(-1)


def prepare():
    z=np.load(DATA/'windows_3s.npz');raw=z['primitive_windows'];N=len(raw)
    xx=np.empty((N,5,31,11),np.float32);mm=np.empty_like(xx,dtype=bool);rng=np.random.default_rng(142)
    for i,r in enumerate(raw):
        xx[i],mm[i]=augment(r,np.ones(31,bool),rng)
        if i%3000==0:print('Features',i,'/',N,flush=True)
    dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'features.npy',xx);np.save(dest/'masks.npy',mm)
    save(dest/'feature_protocol.json',dict(patterns=PATTERNS,random_seed=142,source_npz_sha256=hashlib.sha256((DATA/'windows_3s.npz').read_bytes()).hexdigest(),
        shape=list(xx.shape),derivatives='Recomputed only within each available leader segment; no dependence on masked leader values',
        none_pattern='Leader unobserved, not a synthetic assertion that no physical leader exists'))


def stats(x,m,ids):
    a=x[ids,0];v=m[ids,0]
    mean=[];std=[]
    for j in range(11):
        vals=a[:,:,j][v[:,:,j]];mean.append(float(vals.mean()));std.append(float(max(.1,vals.std())))
    return dict(mean=mean,std=std)


def normalize(x,m,st):return np.where(m,(x-np.array(st['mean'],np.float32))/np.array(st['std'],np.float32),0).astype(np.float32)


def metrics(y,p):
    pred=p>=.5;cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(n=len(y),correct=int((pred==y).sum()),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        av_recall=float(cm[1,1]/cm[1].sum()),hv_recall=float(cm[0,0]/cm[0].sum()),confusion_hv_av=cm.tolist(),
        bce=float(np.mean(-y*np.log(np.clip(p,1e-7,1))-(1-y)*np.log(np.clip(1-p,1e-7,1)))))


def weights(meta,ids):
    """Half mass core/half external, then class, campaign, source-balanced."""
    g=meta.iloc[ids];w=np.zeros(len(g))
    pools=g.campaign.eq('Vicolungo').to_numpy()
    for core in [True,False]:
        group=g.iloc[np.flatnonzero(pools==core)]
        for label in [0,1]:
            sub=group[group.label==label]
            for cam in sub.campaign.unique():
                sources=sub[sub.campaign==cam].source_id.unique()
                for src in sources:
                    hit=(g.label.to_numpy()==label)&(g.source_id.to_numpy()==src)
                    w[hit]=.25/sub.campaign.nunique()/len(sources)/hit.sum()
    assert np.isclose(w.sum(),1);return w


@torch.no_grad()
def predict(model,x,m,indices,device,patterns=range(5)):
    model.eval();out=[]
    for pattern in patterns:
        pp=[]
        for start in range(0,len(indices),512):
            ix=indices[start:start+512]
            pp.append(torch.sigmoid(model(x[ix,pattern].to(device),m[ix,pattern].to(device))).cpu().numpy())
        out.append(np.concatenate(pp))
    return np.stack(out,1)


def train_public():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dest=OUT/'public';meta=pd.read_csv(DATA/'windows.csv');rawx=np.load(dest/'features.npy');masks=np.load(dest/'masks.npy')
    y=meta.label.to_numpy(np.int64);parts=['part2','part3','part4_highway'];reports=[];start=time.time();cfg=CFG['public_training']
    for fold in range(3):
        test=np.flatnonzero(meta.campaign.eq('Vicolungo')&meta.source_id.str.endswith('_'+parts[fold]))
        val=np.flatnonzero(meta.campaign.eq('Vicolungo')&meta.source_id.str.endswith('_'+parts[(fold+1)%3]))
        core=meta.campaign.eq('Vicolungo')&meta.source_id.str.endswith('_'+parts[(fold+2)%3])
        ext=meta.campaign.isin(['ASta','Casale','Vanderbilt','CATS'])&~meta.source_id.isin(['CATS__test1124_test9','CATS__test1124_test10'])
        train=np.flatnonzero(core|ext);st=stats(rawx,masks,train)
        for a,b in [(train,val),(train,test),(val,test)]:
            assert set(meta.iloc[a].source_id).isdisjoint(set(meta.iloc[b].source_id))
        x=torch.from_numpy(normalize(rawx,masks,st));m=torch.from_numpy(masks)
        w=weights(meta,train);vw=np.array([1/meta.iloc[val].source_id.nunique()/sum(meta.iloc[val].source_id==s) for s in meta.iloc[val].source_id])
        split={k:list(map(int,v)) for k,v in [('train',train),('validation',val),('test',test)]}
        save(dest/f'fold{fold}_split.json',dict(indices=split,normalization=st,source_ids={k:sorted(meta.iloc[v].source_id.unique()) for k,v in split.items()}))
        for seed in CFG['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            rng=np.random.default_rng(seed);model=ThreeSecondTransformer().to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            best=float('inf');epoch_best=0;stale=0;log=[];sd=None
            for epoch in range(1,cfg['epochs']+1):
                model.train();order=rng.permutation(len(train));losses=[]
                for off in range(0,len(order),cfg['batch_size']):
                    j=order[off:off+cfg['batch_size']];ix=train[j];pat=rng.choice(5,len(ix),p=[1/3,1/6,1/6,1/6,1/6])
                    xx=x[ix,pat].to(device);mm=m[ix,pat].to(device);yy=torch.tensor(y[ix],dtype=torch.float32,device=device)
                    ww=torch.tensor(w[j]*len(train),dtype=torch.float32,device=device)
                    opt.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(model(xx,mm),yy,reduction='none')*ww).mean()
                    assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                p=predict(model,x,m,val,device);yy=y[val,None]
                bce=(-yy*np.log(np.clip(p,1e-7,1))-(1-yy)*np.log(np.clip(1-p,1e-7,1)))
                value=float((bce.mean(1)*vw).sum());log.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_bce=value))
                if value<best-1e-6:best=value;epoch_best=epoch;sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if epoch%5==0:print('Public',fold,seed,'epoch',epoch,'val BCE',round(value,4),'elapsed',round(time.time()-start),flush=True)
                if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
            model.load_state_dict(sd)
            checkpoint=dict(model_state_dict=sd,normalization=st,seed=seed,fold=fold,best_epoch=epoch_best,
                config=CFG,training_sources=sorted(meta.iloc[train].source_id.unique()),validation_sources=sorted(meta.iloc[val].source_id.unique()))
            torch.save(checkpoint,dest/f'fold{fold}_seed{seed}.pt');save(dest/f'fold{fold}_seed{seed}_log.json',log)
            p=predict(model,x,m,test,device)
            pd.DataFrame(dict(window_index=test,label=y[test],**{name:p[:,j] for j,name in enumerate(PATTERNS)})).to_csv(dest/f'fold{fold}_seed{seed}_predictions.csv',index=False)
            for j,name in enumerate(PATTERNS):reports.append(dict(fold=fold,seed=seed,pattern=name,**metrics(y[test],p[:,j])))
            save(dest/'results.json',dict(status='in_progress',results=reports));print('FINISHED PUBLIC',fold,seed,'best epoch',epoch_best,flush=True)
            del model,opt;torch.cuda.empty_cache()
    save(dest/'results.json',dict(status='complete',device=device,models=9,results=reports,elapsed_s=time.time()-start,
        initialization='random, all weights trained',new_protocol_not_directly_comparable_to_old_4s=True))


def train_field():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    dest=OUT/'field';dest.mkdir(exist_ok=True);meta=pd.read_csv(OUT/'windows.csv');z=np.load(OUT/'dataset.npz')
    raw=z['primitive_windows'];observed=z['leader_available'];y=z['labels'];n=len(y)
    # Fixed before field outcomes; use public training statistics unchanged.
    path=OUT/'public/fold0_seed42.pt';ck=torch.load(path,map_location='cpu');st=ck['normalization']
    x=torch.from_numpy(normalize(z['features'],z['feature_mask'],st)).to(device);m=torch.from_numpy(z['feature_mask']).to(device)
    original=ThreeSecondTransformer().to(device);original.load_state_dict(ck['model_state_dict']);original.eval()
    with torch.no_grad():source=torch.sigmoid(original(x,m)).cpu().numpy()
    preds=[];folds=[];cfg=CFG['field_training']
    for held in range(n):
        remaining=[i for i in range(n) if i!=held];val=[]
        for label in [0,1]:
            choices=[i for i in remaining if y[i]==label]
            assert len(choices)>=2,'Need another training video per class after inner validation'
            val.append(choices[held%len(choices)])
        train=[i for i in remaining if i not in val]
        assert set(train).isdisjoint(val) and held not in train+val
        fold=dict(test=meta['clip'].iloc[held],train=meta['clip'].iloc[train].tolist(),validation=meta['clip'].iloc[val].tolist())
        folds.append(fold);save(dest/'splits.json',folds)
        for seed in CFG['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            model=ThreeSecondTransformer().to(device);model.load_state_dict(ck['model_state_dict'])
            assert all(p.requires_grad for p in model.parameters())
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
            best=float('inf');sd=None;epoch_best=0;stale=0;logs=[]
            for epoch in range(1,cfg['epochs']+1):
                xx=[];mm=[];yy=[];ww=[]
                for i in train:
                    a,b=augment(raw[i],observed[i],rng)
                    # Duplicate actual observation pattern to assign it probability 1/3.
                    patterns=[0,0,1,2,3,4]
                    xx.extend(a[patterns]);mm.extend(b[patterns]);yy.extend([y[i]]*6)
                    ww.extend([len(train)/(2*sum(y[j]==y[i] for j in train))]*6)
                xx=np.stack(xx);mm=np.stack(mm)
                tx=torch.from_numpy(normalize(xx,mm,st)).to(device);tm=torch.from_numpy(mm).to(device)
                ty=torch.tensor(yy,dtype=torch.float32,device=device);tw=torch.tensor(ww,dtype=torch.float32,device=device)
                model.train();opt.zero_grad()
                loss=(nn.functional.binary_cross_entropy_with_logits(model(tx,tm),ty,reduction='none')*tw).mean()
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.eval()
                with torch.no_grad():value=float(nn.functional.binary_cross_entropy_with_logits(model(x[val],m[val]),torch.tensor(y[val],dtype=torch.float32,device=device)))
                logs.append(dict(epoch=epoch,loss=float(loss.detach()),validation_bce=value))
                if value<best-1e-6:best=value;epoch_best=epoch;sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
            model.load_state_dict(sd);model.eval()
            with torch.no_grad():p=float(torch.sigmoid(model(x[held:held+1],m[held:held+1])).cpu()[0])
            changed={k:float(torch.linalg.vector_norm(sd[k]-ck['model_state_dict'][k])) for k in sd}
            assert any(v>0 for k,v in changed.items() if k.startswith('encoder.'))
            name=f'held{held}_seed{seed}';torch.save(dict(model_state_dict=sd,normalization=st,split=fold,seed=seed,best_epoch=epoch_best),dest/f'{name}.pt')
            save(dest/f'{name}_log.json',dict(epochs=logs,parameter_change_l2=changed))
            preds.append(dict(clip=meta['clip'].iloc[held],label=int(y[held]),seed=seed,probability=p,best_epoch=epoch_best,source_only_probability=float(source[held])))
            pd.DataFrame(preds).to_csv(dest/'seed_predictions.csv',index=False)
        print('Field finished',held+1,'/',n,meta['clip'].iloc[held],flush=True)
    pp=pd.DataFrame(preds).groupby(['clip','label'],sort=False).agg(probability=('probability','mean'),seed_probability_min=('probability','min'),seed_probability_max=('probability','max'),source_only_probability=('source_only_probability','first')).reset_index()
    pp['predicted_label']=(pp.probability>=.5).astype(int);pp['correct']=pp.predicted_label==pp.label;pp.to_csv(dest/'video_predictions.csv',index=False)
    save(dest/'results.json',dict(status='complete',eligible_videos=n,unavailable_videos=13-n,source_only=metrics(pp.label.to_numpy(),pp.source_only_probability.to_numpy()),
        adapted=metrics(pp.label.to_numpy(),pp.probability.to_numpy()),models=n*3,device=device,
        initial_checkpoint=str(path.relative_to(ROOT)),initial_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        normalization='Fixed public-training statistics; no Todd test fitting',all_model_weights_trainable=True,
        cohort_limit='Video-disjoint development estimate; same instrumented vehicle/camera and potentially sessions recur across folds'))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','field']);a=ap.parse_args()
    {'prepare':prepare,'train':train_public,'field':train_field}[a.stage]()
