"""Variable-duration Transformer: true 0.1s time, separate padding/leader masks."""
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
import three_second_classifier as S
from todd_experiment_data import featurize

ROOT=S.ROOT;CFG=json.loads((ROOT/'configs/field_todd_variable.json').read_text(encoding='utf-8'))
OUT=ROOT/'experiments'/CFG['experiment'];OLD=S.OUT;DATA=S.DATA;MAX=CFG['maximum_samples'];MIN=CFG['minimum_samples']
save=S.save;normalize=S.normalize;metrics=S.metrics
PATTERNS=['full','none','prefix','suffix','internal']
VARIANT_WEIGHTS=np.tile([1/3,1/6,1/6,1/6,1/6],2)/2


class VariableTransformer(S.ThreeSecondTransformer):
    def forward(self,x,mask,valid):
        if x.ndim!=3 or x.shape[2]!=11 or not MIN<=x.shape[1]<=MAX:raise ValueError('Invalid variable input dimensions')
        if valid.shape!=x.shape[:2] or mask.shape!=x.shape:raise ValueError('Mask dimensions differ')
        if (valid.sum(1)<MIN).any() or (valid[:,1:]&~valid[:,:-1]).any():raise ValueError('Valid ego samples must be a nonempty prefix')
        if not torch.equal(mask[:,:,:3],valid[:,:,None].expand(-1,-1,3)):raise ValueError('Ego mask and valid time disagree')
        mask=mask&valid[:,:,None]
        x=torch.where(mask,x,torch.zeros_like(x));leader=mask[:,:,5:6].float()
        interaction=self.interaction(torch.cat([x[:,:,3:],mask[:,:,3:].float()],-1))
        h=self.ego(x[:,:,:3])+leader*interaction+(1-leader)*self.missing
        h=torch.cat([self.cls.expand(len(x),-1,-1),h],1)+self.position[:,:x.shape[1]+1]
        padding=torch.cat([torch.zeros(len(x),1,dtype=torch.bool,device=x.device),~valid],1)
        return self.head(self.encoder(h,src_key_padding_mask=padding)[:,0]).squeeze(-1)


def features(raw,a):
    n=len(raw)
    return featurize(raw[:,0],raw[:,1],raw[:,2],a,np.zeros(n,int),np.where(a,0,-1))[:2]


def variants(raw,observed,rng):
    """Crop raw measurements BEFORE smoothing/derivatives; never infer time from zeros."""
    n=len(raw);long=min(MAX,n);short=int(rng.integers(MIN,long)) if long>MIN else long
    xx=[];mm=[];vv=[];lengths=[]
    for length in [long,short]:
        start=int(rng.integers(0,n-length+1));r=raw[start:start+length];base=observed[start:start+length]
        left=max(1,int(rng.integers(max(1,length//5),max(2,length//2))))
        right=min(length-1,max(left+1,int(rng.integers(max(2,length//2),length))))
        for pattern in range(5):
            a=base.copy()
            if pattern==1:a[:]=False
            if pattern==2:a[right:]=False
            if pattern==3:a[:left]=False
            if pattern==4:a[left:right]=False
            f,m=features(r,a);pad=MAX-length
            xx.append(np.pad(f,((0,pad),(0,0))));mm.append(np.pad(m,((0,pad),(0,0))))
            vv.append(np.arange(MAX)<length);lengths.append(length)
    return np.stack(xx),np.stack(mm),np.stack(vv),np.array(lengths)


def prepare():
    dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    raw=np.load(DATA/'windows_3s.npz')['primitive_windows'];n=len(raw);rng=np.random.default_rng(3142)
    x=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float32,shape=(n,10,MAX,11))
    m=np.lib.format.open_memmap(dest/'feature_masks.npy',mode='w+',dtype=bool,shape=x.shape)
    v=np.lib.format.open_memmap(dest/'valid_tokens.npy',mode='w+',dtype=bool,shape=(n,10,MAX))
    lengths=np.zeros((n,10),np.int16);started=time.time()
    for i,r in enumerate(raw):
        x[i],m[i],v[i],lengths[i]=variants(r,np.ones(len(r),bool),rng)
        if (i+1)%2000==0:print('Prepared',i+1,'/',n,'seconds',round(time.time()-started),flush=True)
    x.flush();m.flush();v.flush();np.save(dest/'lengths.npy',lengths)
    split=json.loads((OLD/'public/fold0_split.json').read_text())
    save(dest/'split.json',split)
    save(dest/'augmentation.json',dict(windows=n,variants=10,seed=3142,shape=list(x.shape),
        real_durations_s=[float((lengths.min()-1)*.1),float((lengths.max()-1)*.1)],
        source_npz_sha256=hashlib.sha256((DATA/'windows_3s.npz').read_bytes()).hexdigest(),
        no_time_stretch=True,crop_before_derivatives=True,label_independent_lengths=True))


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


@torch.no_grad()
def public_predict(model,x,m,v,ids,device):
    model.eval();out=[]
    for k in range(10):
        pp=[]
        for start in range(0,len(ids),512):
            ii=ids[start:start+512];pp.append(torch.sigmoid(model(x[ii,k].to(device),m[ii,k].to(device),v[ii,k].to(device))).cpu().numpy())
        out.append(np.concatenate(pp))
    return np.stack(out,1)


def train():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';dest=OUT/'public'
    raw=np.load(dest/'features.npy',mmap_mode='r');mask=np.load(dest/'feature_masks.npy');valid=np.load(dest/'valid_tokens.npy')
    meta=pd.read_csv(DATA/'windows.csv');y=meta.label.to_numpy(np.int64);split=json.loads((dest/'split.json').read_text())
    train=np.array(split['indices']['train']);val=np.array(split['indices']['validation']);test=np.array(split['indices']['test'])
    for a,b in [(train,val),(train,test),(val,test)]:assert set(meta.iloc[a].source_id).isdisjoint(meta.iloc[b].source_id)
    w=S.weights(meta,train);vw=np.array([1/meta.iloc[val].source_id.nunique()/sum(meta.iloc[val].source_id==s) for s in meta.iloc[val].source_id])
    cfg=CFG['public_training'];results=[];started=time.time()
    for seed in CFG['seeds']:
        ckpath=OLD/f'public/fold0_seed{seed}.pt';ck=torch.load(ckpath,map_location='cpu');st=ck['normalization']
        assert ck['training_sources']==split['source_ids']['train']
        seed_everything(seed);rng=np.random.default_rng(seed);model=VariableTransformer().to(device);model.load_state_dict(ck['model_state_dict'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        x=torch.from_numpy(normalize(raw,mask,st));m=torch.from_numpy(mask);v=torch.from_numpy(valid)
        best=float('inf');stale=0;best_epoch=0;log=[];sd=None
        for epoch in range(1,cfg['epochs']+1):
            model.train();order=rng.permutation(len(train));losses=[]
            for off in range(0,len(order),cfg['batch_size']):
                j=order[off:off+cfg['batch_size']];ix=train[j];pat=rng.choice(10,len(ix),p=VARIANT_WEIGHTS)
                tx=x[ix,pat].to(device);tm=m[ix,pat].to(device);tv=v[ix,pat].to(device)
                ty=torch.tensor(y[ix],dtype=torch.float32,device=device);ww=torch.tensor(w[j]*len(train),dtype=torch.float32,device=device)
                opt.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(model(tx,tm,tv),ty,reduction='none')*ww).mean()
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            p=public_predict(model,x,m,v,val,device);yy=y[val,None]
            bce=-yy*np.log(np.clip(p,1e-7,1))-(1-yy)*np.log(np.clip(1-p,1e-7,1));value=float(((bce*VARIANT_WEIGHTS).sum(1)*vw).sum())
            log.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_bce=value))
            if value<best-1e-6:best=value;stale=0;best_epoch=epoch;sd={k:a.detach().cpu().clone() for k,a in model.state_dict().items()}
            else:stale+=1
            print('Variable public',seed,'epoch',epoch,'val',round(value,4),'seconds',round(time.time()-started),flush=True)
            if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
        model.load_state_dict(sd)
        changes={k:float(torch.linalg.vector_norm(sd[k]-ck['model_state_dict'][k])) for k in sd}
        assert any(a>0 for k,a in changes.items() if k.startswith('encoder.'))
        torch.save(dict(model_state_dict=sd,normalization=st,seed=seed,best_epoch=best_epoch,config=CFG,
            training_sources=split['source_ids']['train'],validation_sources=split['source_ids']['validation'],
            initial_checkpoint=str(ckpath.relative_to(ROOT))),dest/f'seed{seed}.pt')
        save(dest/f'seed{seed}_log.json',dict(epochs=log,parameter_change_l2=changes))
        pp=public_predict(model,x,m,v,test,device)
        pd.DataFrame(dict(window_index=test,label=y[test],**{f'variant{k}':pp[:,k] for k in range(10)})).to_csv(dest/f'seed{seed}_predictions.csv',index=False)
        for k in range(10):results.append(dict(seed=seed,length_mode='2.5s' if k<5 else 'variable_short',leader_pattern=PATTERNS[k%5],**metrics(y[test],pp[:,k])))
        save(dest/'results.json',dict(status='in_progress',results=results));del x,m,v,model,opt;torch.cuda.empty_cache()
    save(dest/'results.json',dict(status='complete',models=len(CFG['seeds']),device=device,results=results,elapsed_s=time.time()-started,
        evaluation_scope='One prespecified public source partition, 3 random seeds; not all three cyclic folds'))


def field():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';dest=OUT/'field';dest.mkdir(exist_ok=True)
    meta=pd.read_csv(OUT/'windows.csv');z=np.load(OUT/'dataset.npz');y=z['labels'];n=len(y);cfg=CFG['field_training'];folds=[];preds=[]
    for held in range(n):
        remaining=[i for i in range(n) if i!=held];val=[]
        for label in [0,1]:
            choices=[i for i in remaining if y[i]==label];assert len(choices)>=2;val.append(choices[held%len(choices)])
        train=[i for i in remaining if i not in val]
        fold=dict(test=meta['clip'].iloc[held],train=meta['clip'].iloc[train].tolist(),validation=meta['clip'].iloc[val].tolist())
        assert fold['test'] not in fold['train']+fold['validation'];folds.append(fold);save(dest/'splits.json',folds)
        for seed in CFG['seeds']:
            ck=torch.load(OUT/f'public/seed{seed}.pt',map_location='cpu');st=ck['normalization'];seed_everything(seed);rng=np.random.default_rng(seed)
            x=torch.from_numpy(normalize(z['features'],z['feature_mask'],st)).to(device)
            m=torch.from_numpy(z['feature_mask']).to(device);v=torch.from_numpy(z['valid_tokens']).to(device)
            model=VariableTransformer().to(device);model.load_state_dict(ck['model_state_dict']);model.eval()
            with torch.no_grad():source=float(torch.sigmoid(model(x[held:held+1],m[held:held+1],v[held:held+1]))[0])
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=float('inf');stale=0;best_epoch=0;log=[];sd=None
            for epoch in range(1,cfg['epochs']+1):
                xx=[];mm=[];vv=[];yy=[];ww=[]
                for i in train:
                    length=int(z['valid_tokens'][i].sum());a,b,c,_=variants(z['primitive_windows'][i,:length],z['leader_available'][i,:length],rng)
                    xx.extend(a);mm.extend(b);vv.extend(c);yy.extend([y[i]]*10)
                    ww.extend(VARIANT_WEIGHTS*10*len(train)/(2*sum(y[j]==y[i] for j in train)))
                xx=np.stack(xx);mm=np.stack(mm);tx=torch.from_numpy(normalize(xx,mm,st)).to(device)
                tm=torch.from_numpy(mm).to(device);tv=torch.from_numpy(np.stack(vv)).to(device)
                ty=torch.tensor(yy,dtype=torch.float32,device=device);tw=torch.tensor(ww,dtype=torch.float32,device=device)
                model.train();opt.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(model(tx,tm,tv),ty,reduction='none')*tw).mean()
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.eval()
                with torch.no_grad():value=float(nn.functional.binary_cross_entropy_with_logits(model(x[val],m[val],v[val]),torch.tensor(y[val],dtype=torch.float32,device=device)))
                log.append(dict(epoch=epoch,training_bce=float(loss.detach()),validation_bce=value))
                if value<best-1e-6:best=value;stale=0;best_epoch=epoch;sd={k:a.detach().cpu().clone() for k,a in model.state_dict().items()}
                else:stale+=1
                if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
            model.load_state_dict(sd);model.eval()
            with torch.no_grad():prob=float(torch.sigmoid(model(x[held:held+1],m[held:held+1],v[held:held+1]))[0])
            changes={k:float(torch.linalg.vector_norm(sd[k]-ck['model_state_dict'][k])) for k in sd}
            assert any(a>0 for k,a in changes.items() if k.startswith('encoder.'))
            name=f'held{held}_seed{seed}';torch.save(dict(model_state_dict=sd,normalization=st,split=fold,seed=seed,best_epoch=best_epoch),dest/f'{name}.pt')
            save(dest/f'{name}_log.json',dict(epochs=log,parameter_change_l2=changes))
            preds.append(dict(clip=fold['test'],label=int(y[held]),seed=seed,probability=prob,source_only_probability=source,best_epoch=best_epoch))
            pd.DataFrame(preds).to_csv(dest/'seed_predictions.csv',index=False)
        print('Variable field finished',held+1,'/',n,fold['test'],flush=True)
    pp=pd.DataFrame(preds).groupby(['clip','label'],sort=False).agg(probability=('probability','mean'),source_only_probability=('source_only_probability','mean'),
        seed_probability_min=('probability','min'),seed_probability_max=('probability','max')).reset_index()
    pp=pp.merge(meta[['clip','duration_s','samples','less_than_one_second']],on='clip',validate='one_to_one')
    pp['predicted_label']=(pp.probability>=.5).astype(int);pp['correct']=pp.predicted_label==pp.label;pp.to_csv(dest/'video_predictions.csv',index=False)
    save(dest/'results.json',dict(status='complete',videos=n,source_only=metrics(pp.label.to_numpy(),pp.source_only_probability.to_numpy()),
        adapted=metrics(pp.label.to_numpy(),pp.probability.to_numpy()),models=n*len(CFG['seeds']),device=device,
        cohort_limit='Video-disjoint development result; same camera, vehicle and potentially sessions recur; not independent generalization.'))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','field']);a=p.parse_args()
    {'prepare':prepare,'train':train,'field':field}[a.stage]()
