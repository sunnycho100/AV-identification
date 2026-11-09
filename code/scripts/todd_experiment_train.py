"""Video-held-out Todd adaptation of the existing source Transformer."""
import copy
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score
import todd_experiment_data as D

ROOT,OUT,CFG,CLIPS=D.ROOT,D.OUT,D.CFG,D.CLIPS
sys.path.insert(0,str(ROOT/'classification/trajectory-avlike-transformer/inference_cnn'))
from networks import TrajectoryNet


class MaskedTransformer(TrajectoryNet):
    def __init__(self,cp,stats,source_stats):
        c=cp['config'];super().__init__(11,'transformer',c['width'],c['dropout'],c['layers'])
        self.load_state_dict(cp['model_state_dict'],strict=True)
        self.missing_proj=nn.Linear(14,c['width'],bias=False)
        with torch.no_grad():
            w=self.input_proj.weight.detach().clone()
            ratio=torch.tensor(np.array(stats['std'])/source_stats['std'],dtype=w.dtype)
            shift=torch.tensor((np.array(stats['mean'])-source_stats['mean'])/source_stats['std'],dtype=w.dtype)
            self.input_proj.weight.mul_(ratio)
            self.input_proj.bias.add_(w@shift)
            self.missing_proj.weight.zero_()
            # Make missing values source-mean neutral at initialization even
            # when the training-fold mean differs from the source mean.
            self.missing_proj.weight[:,:11]=-w*shift
        for p in self.encoder.parameters():p.requires_grad=False
        self.pos_embed.requires_grad=False;self.cls_token.requires_grad=False

    def forward(self,x,feature_mask,relation,padding):
        mask=torch.cat([(~feature_mask).float(),relation],dim=-1)
        h=self.input_proj(x)+self.missing_proj(mask)
        h=torch.cat([self.cls_token.expand(len(x),-1,-1),h],dim=1)
        pad=torch.cat([torch.zeros(len(x),1,dtype=torch.bool,device=x.device),~padding],dim=1)
        h=self.encoder(h+self.pos_embed[:,:h.shape[1]],src_key_padding_mask=pad)
        return self.classifier(h[:,0]).squeeze(-1)


def stats_for(x,m,ci,training):
    mu=[];std=[]
    for j in range(11):
        means=[];squares=[]
        for v in training:
            values=x[ci==v,:,j][m[ci==v,:,j]]
            if len(values):means.append(float(values.mean()));squares.append(float(np.square(values).mean()))
        mean=float(np.mean(means)) if means else 0.
        variance=max(0,float(np.mean(squares))-mean*mean) if squares else 1.
        mu.append(mean);std.append(float(max(.1,np.sqrt(variance))))
    return dict(mean=mu,std=std,training_clip_indices=training)


def tensors(ds,stats,mode):
    m=ds['feature_mask'].copy();s=ds['relation_state'].copy()
    if mode=='ego':m[:,:,3:]=False;s[:]=0
    x=(ds['features']-np.array(stats['mean']))/np.array(stats['std'])
    x=np.where(m,x,0).astype(np.float32)
    assert np.isfinite(x).all()
    return [torch.from_numpy(v) for v in [x,m,s,ds['valid_tokens']]]


def predict(model,tx,indices,ci):
    model.eval()
    with torch.no_grad():p=torch.sigmoid(model(*[a[indices] for a in tx])).numpy()
    return {int(c):float(p[ci[indices]==c].mean()) for c in np.unique(ci[indices])}


def video_bce(probs,labels):
    return float(np.mean([-np.log(np.clip(p if labels[c] else 1-p,1e-7,1)) for c,p in probs.items()]))


def scores(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=(p>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(videos=len(y),correct=int((pred==y).sum()),accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        hv_recall=float(cm[0,0]/cm[0].sum()) if cm[0].sum() else None,
        av_recall=float(cm[1,1]/cm[1].sum()) if cm[1].sum() else None,
        confusion_hv_av=cm.tolist(),bce=float(np.mean(-y*np.log(np.clip(p,1e-7,1))-(1-y)*np.log(np.clip(1-p,1e-7,1)))))


def train():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    ds=np.load(OUT/'dataset.npz');ci=ds['clip_index'];labels=np.array([int(c.startswith('AV_')) for c in CLIPS])
    inv=pd.read_csv(OUT/'inventory.csv').set_index('clip').loc[CLIPS]
    folds=D.Q.read(OUT/'folds.json');cp_path=ROOT/CFG['initial_checkpoint']
    cp=torch.load(cp_path,map_location='cpu',weights_only=True);source=D.Q.read(cp_path.parent/'feature_stats.json')
    old={k:np.array(source[k]) for k in ['mean','std']};source_stats={k:v.tolist() for k,v in old.items()}
    (OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    predictions=[];training=[];start=time.time()
    # Source weights, no source speed clipping, explicit neutral missingness.
    frozen=MaskedTransformer(cp,source_stats,old).eval();tx=tensors(ds,source_stats,'masked')
    pr=predict(frozen,tx,np.arange(len(ci)),ci)
    for i,p in pr.items():predictions.append(dict(model='source_transformer_masked',seed=-1,clip=CLIPS[i],label=int(labels[i]),prob_av=p,fold=i))
    for fold in folds:
        fi=fold['fold'];train_ids=[CLIPS.index(c) for c in fold['train']];val_ids=[CLIPS.index(c) for c in fold['validation']];test_id=CLIPS.index(fold['test'][0])
        tr=np.flatnonzero(np.isin(ci,train_ids));va=np.flatnonzero(np.isin(ci,val_ids));te=np.flatnonzero(ci==test_id)
        assert not set(train_ids)&set(val_ids) and test_id not in train_ids+val_ids
        # Fixed classical controls use exactly the same training videos.
        controls={'mean_speed_logistic':['mean_ego_speed_mps'],
                  'availability_duration_logistic':['retained_duration_s','windows','shortest_window_s','leader_available_fraction','absent_fraction','not_visible_fraction']}
        for name,cols in controls.items():
            a=inv[cols].to_numpy(float);sc=StandardScaler().fit(a[train_ids]);lr=LogisticRegression(C=1,class_weight='balanced',random_state=42,max_iter=1000)
            lr.fit(sc.transform(a[train_ids]),labels[train_ids]);p=float(lr.predict_proba(sc.transform(a[[test_id]]))[0,1])
            predictions.append(dict(model=name,seed=-1,clip=CLIPS[test_id],label=int(labels[test_id]),prob_av=p,fold=fi))
            D.Q.save(OUT/'logs'/f'{name}_fold{fi:02d}.json',dict(train=fold['train'],validation=fold['validation'],test=fold['test'],
                features=cols,mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coefficient=lr.coef_.tolist(),intercept=lr.intercept_.tolist()))
        for mode in ['ego','masked']:
            fm=ds['feature_mask'].copy()
            if mode=='ego':fm[:,:,3:]=False
            stats=stats_for(ds['features'],fm,ci,train_ids);tx=tensors(ds,stats,mode)
            for seed in CFG['training_seeds']:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                model=MaskedTransformer(cp,stats,old)
                opt=torch.optim.AdamW([
                    {'params':model.classifier.parameters(),'lr':CFG['head_lr']},
                    {'params':model.missing_proj.parameters(),'lr':CFG['head_lr']},
                    {'params':model.input_proj.parameters(),'lr':CFG['input_lr']}],weight_decay=CFG['weight_decay'])
                y=torch.tensor(labels[ci[tr]],dtype=torch.float32)
                weights=np.zeros(len(tr))
                for label in [0,1]:
                    videos=[c for c in train_ids if labels[c]==label]
                    for c in videos:
                        ix=ci[tr]==c;weights[ix]=.5/len(videos)/ix.sum()
                w=torch.tensor(weights,dtype=torch.float32);assert abs(w.sum().item()-1)<1e-6
                v0=video_bce(predict(model,tx,va,ci),labels)
                best=v0;best_epoch=0;best_sd=copy.deepcopy(model.state_dict());stale=0
                log=[dict(epoch=0,validation_bce=v0,training_bce=None)]
                for epoch in range(1,CFG['epochs']+1):
                    model.train();model.encoder.eval();opt.zero_grad()
                    out=model(*[a[tr] for a in tx]);loss=(nn.functional.binary_cross_entropy_with_logits(out,y,reduction='none')*w).sum()
                    assert torch.isfinite(loss)
                    loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
                    value=video_bce(predict(model,tx,va,ci),labels)
                    log.append(dict(epoch=epoch,validation_bce=value,training_bce=float(loss.detach())))
                    if value<best-1e-7:
                        best=value;best_epoch=epoch;best_sd=copy.deepcopy(model.state_dict());stale=0
                    else:stale+=1
                    if epoch>=CFG['minimum_epochs'] and stale>=CFG['patience']:break
                model.load_state_dict(best_sd);prob=predict(model,tx,te,ci)[test_id]
                name=f'finetuned_transformer_{mode}';run=f'{mode}_fold{fi:02d}_seed{seed}'
                torch.save(dict(model_state_dict=best_sd,normalization=stats,mode=mode,best_epoch=best_epoch,
                    train=fold['train'],validation=fold['validation'],test=fold['test'],source_checkpoint=CFG['initial_checkpoint']),OUT/'checkpoints'/f'{run}.pt')
                D.Q.save(OUT/'logs'/f'{run}.json',dict(train=fold['train'],validation=fold['validation'],test=fold['test'],normalization=stats,epochs=log,best_epoch=best_epoch))
                predictions.append(dict(model=name,seed=seed,clip=CLIPS[test_id],label=int(labels[test_id]),prob_av=prob,fold=fi))
                training.append(dict(model=name,seed=seed,fold=fi,train_videos=len(train_ids),validation_videos=len(val_ids),test_videos=1,
                    epochs_run=epoch,best_epoch=best_epoch,best_validation_bce=best))
        pd.DataFrame(predictions).to_csv(OUT/'predictions_by_seed.csv',index=False)
        pd.DataFrame(training).to_csv(OUT/'training_runs.csv',index=False)
        print(f'Completed fold {fi+1}/13 ({CLIPS[test_id]}), elapsed {time.time()-start:.1f}s',flush=True)
    pred=pd.DataFrame(predictions);video=pred.groupby(['model','clip','label'],as_index=False).agg(prob_av=('prob_av','mean'),seed_sd=('prob_av','std'),members=('seed','size'))
    video['prediction']=(video.prob_av>=.5).astype(int);video['correct']=video.prediction==video.label
    video.to_csv(OUT/'predictions_by_video.csv',index=False)
    summary=[]
    for name,g in video.groupby('model'):
        assert set(g['clip'])==set(CLIPS)
        row=dict(model=name,**scores(g.label,g.prob_av));ss=[]
        for seed,gg in pred[pred.model==name].groupby('seed'):
            ss.append(dict(seed=int(seed),**scores(gg.label,gg.prob_av)))
        row['per_seed']=ss
        common=g[g['clip'].isin(['AV_T_WE_1','HV_T_EW_6'])]
        row['legacy_two_video_cohort']=scores(common.label,common.prob_av)
        summary.append(row)
    D.Q.save(OUT/'classification_summary.json',dict(results=summary,threshold=.5,outer_unit='video',videos=13,
        models_finetuned=len(training),source_backbone_frozen=True,training_device='CPU',
        prevalence_only_baseline=dict(always_AV=7/13,always_HV=6/13),elapsed_s=time.time()-start,
        independent_camera_or_vehicle_test=False))
    manifest=D.Q.read(OUT/'input_manifest.json')
    for p,h in manifest.items():assert D.Q.sha(ROOT/p)==h
    D.Q.save(OUT/'training_validation.json',dict(status='PASS',folds=13,training_runs=len(training),
        source_hashes_unchanged=True,outer_test_never_used_for_normalization_or_checkpoint_selection=True))
    print(pd.DataFrame(summary)[['model','correct','videos','accuracy','balanced_accuracy','av_recall','hv_recall']].to_string(index=False),flush=True)


def portable_checkpoints():
    """Repair numpy scalar metadata in this experiment's own generated files."""
    paths=list((OUT/'checkpoints').glob('*.pt'));assert len(paths)==78
    for path in paths:
        # These are files generated by this module in this run, not downloads.
        c=torch.load(path,map_location='cpu',weights_only=False)
        log=D.Q.read(OUT/'logs'/f'{path.stem}.json')
        assert c['train']==log['train'] and c['test']==log['test']
        for k in ['mean','std']:c['normalization'][k]=[float(v) for v in c['normalization'][k]]
        torch.save(c,path)
        restored=torch.load(path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,restored['model_state_dict'][k]) for k,v in c['model_state_dict'].items())
    print('Portable metadata verified in',len(paths),'self-generated checkpoints')


if __name__=='__main__':
    if '--portable-checkpoints' in sys.argv:portable_checkpoints()
    else:train()
