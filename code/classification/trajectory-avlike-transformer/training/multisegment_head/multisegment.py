"""Prespecified ideal-matched trajectory bag experiment; see MULTISEGMENT_PLAN.txt."""
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
import torch.nn.functional as F
from data_protocol import PARTS, splits, features, normalize
from networks import TrajectoryNet
from run import save, seed_all

KS = (1, 2, 4, 8)
SUBSETS = {k: np.array(list(itertools.combinations(range(8), k))) for k in KS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bags(m):
    """Eight disjoint windows within an unchanged-state, record-local interval."""
    rows = []
    for tid, part in m[m.start_in_segment.mod(40).eq(0) & ~m.campaign.eq('NGSIM')].groupby('Trajectory_ID', sort=True):
        part = part.sort_values('raw_start_time')
        assert part.pair_id.nunique() == part.label.nunique() == part.source_id.nunique() == 1
        times = part.raw_start_time.to_numpy()
        ids = part.window_index.to_numpy()
        def greedy(spacing, start=0):
            selected = []
            for j in range(start, len(times)):
                if not selected or times[j]-times[selected[-1]] >= spacing-1e-6:
                    selected.append(j)
            return selected
        for regime, spacing in [('gap4', 8), ('gap12', 16)]:
            selected = greedy(spacing)
            for b in range(len(selected)//8):
                j = selected[b*8:b*8+8]
                shared = dict(Trajectory_ID=tid, pair_id=part.iloc[0].pair_id,
                    source_id=part.iloc[0].source_id, campaign=part.iloc[0].campaign,
                    label=int(part.iloc[0].label), anchor=int(ids[j[0]]))
                for name, jj in [(regime, j)] + ([('dense', greedy(4, j[0])[:8])] if regime == 'gap4' else []):
                    assert len(jj) == 8
                    rows.append(dict(**shared, regime=name, bag_id=f'{tid}__{regime}__{b}',
                        indices=ids[jj].tolist(), starts=times[jj].tolist(),
                        elapsed_seconds=float(times[jj[-1]]-times[jj[0]]+4)))
    return rows


def audit_bags(m, bags):
    for regime in ['dense', 'gap4', 'gap12']:
        selected = [b for b in bags if b['regime'] == regime]
        ids = [i for b in selected for i in b['indices']]
        assert len(ids) == len(set(ids)), 'views reused across bags'
        for b in selected:
            q = m.iloc[b['indices']]
            assert (q.Trajectory_ID == b['Trajectory_ID']).all()
            assert (q.label == b['label']).all() and (q.pair_id == b['pair_id']).all()
            assert (q.min_raw_speed > 22.352).all()
            assert q.start_in_segment.mod(40).eq(0).all()
            assert np.all(np.diff(b['starts']) >= {'dense':4,'gap4':8,'gap12':16}[regime]-1e-6)
            assert (q.raw_end_time.to_numpy()[:-1] < q.raw_start_time.to_numpy()[1:]).all()
    a = {(b['Trajectory_ID'],b['anchor'],b['label']) for b in bags if b['regime']=='gap4'}
    b = {(b['Trajectory_ID'],b['anchor'],b['label']) for b in bags if b['regime']=='dense'}
    assert a == b


def role_bags(bags, allowed, regime='gap4'):
    allowed = set(map(int, allowed))
    return [b for b in bags if b['regime']==regime and set(b['indices']) <= allowed]


def bag_frame(bags, probs):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1-1e-7)
    y = np.array([b['label'] for b in bags])[:, None]
    return pd.DataFrame([dict(bag_id=b['bag_id'], pair_id=b['pair_id'], label=b['label'],
                             source_id=b['source_id'], elapsed_seconds=b['elapsed_seconds']) for b in bags]).assign(
        accuracy=((p>=.5)==y).mean(1), bce=(-(y*np.log(p)+(1-y)*np.log1p(-p))).mean(1),
        positive_rate=(p>=.5).mean(1), brier=((p-y)**2).mean(1))


def summarize(frame):
    # Each record-local follower has equal weight, irrespective of length.
    u = frame.groupby(['pair_id','label'])[['accuracy','bce','positive_rate','brier']].mean().reset_index()
    assert set(u.label)=={0,1}
    return dict(accuracy=float(u.accuracy.mean()), bce=float(u.bce.mean()), brier=float(u.brier.mean()),
        balanced_accuracy=float(u.groupby('label').accuracy.mean().mean()),
        human_recall=float(u[u.label==0].accuracy.mean()), acc_recall=float(u[u.label==1].accuracy.mean()),
        bag_accuracy=float(frame.accuracy.mean()), bags=len(frame), pairs=len(u))


def fused(z, k, method):
    q = z[:, SUBSETS[k]]
    if method == 'mean_probability': return expit(q).mean(-1)
    if method == 'mean_logit': return expit(q.mean(-1))
    if method == 'majority': return (q>=0).mean(-1)
    raise ValueError(method)


def model_path(cfg, name, seed, fold):
    return Path(cfg['original_models'][name])/f'seed_{seed}'/PARTS[fold]


def checked_original(cfg, name, path, checks):
    base = Path(cfg['original_models'][name])
    manifest = json.loads((base.parent.parent/'manifest.json').read_text())
    item = next(r for r in manifest if r['name']==base.name)
    rel = str(path.relative_to(base))
    digest = sha(path)
    assert digest == item['files'][rel], str(path)
    checks[str(path)] = digest
    return path


def read_logits(cfg, name, seed, fold, role, m, checks):
    p = checked_original(cfg, name, model_path(cfg,name,seed,fold)/f'{role}_predictions.csv', checks)
    df = pd.read_csv(p, float_precision='round_trip')
    ids = df.window_index.to_numpy()
    for c in ['Trajectory_ID','pair_id','source_id','label','raw_start_time']:
        np.testing.assert_array_equal(df[c], m.iloc[ids][c])
    z = np.full(len(m), np.nan)
    z[ids] = df.logit.to_numpy(dtype=np.float64 if name=='rf' else np.float32)
    return z


def record_eval(bags, probabilities, info, path, rows, metrics):
    arrays = {}
    for k, p in probabilities.items():
        frame = bag_frame(bags,p)
        for key,value in dict(**info,k=k).items(): frame[key]=value
        rows.append(frame)
        met = dict(**info,k=k,**summarize(frame)); metrics.append(met)
        arrays[f'p_{k}']=p
    arrays['bag_ids']=np.array([b['bag_id'] for b in bags])
    np.savez_compressed(path,**arrays)


def frozen(cfg, m, bags, root, checks):
    rows, metrics, mismatch = [], [], []
    for name in ['cnn','transformer','rf']:
        for seed in cfg['seeds']:
            for fold in cfg['folds']:
                split=splits(m,fold)
                for role in ['val','test']:
                    z=read_logits(cfg,name,seed,fold,role,m,checks)
                    for regime in ['dense','gap4','gap12']:
                        bb=role_bags(bags,split[role],regime)
                        zz=z[np.array([b['indices'] for b in bb])]; assert np.isfinite(zz).all()
                        for method in ['mean_probability','mean_logit','majority']:
                            info=dict(model=name,seed=seed,fold=fold,role=role,regime=regime,fusion=method)
                            filename=f'{name}_{seed}_{fold}_{role}_{regime}_{method}.npz'
                            record_eval(bb,{k:fused(zz,k,method) for k in KS},info,root/'predictions'/filename,rows,metrics)
                    if role=='test':
                        bb=role_bags(bags,split[role]); ids=np.array([b['indices'] for b in bb])
                        pool=split[role][m.iloc[split[role]].start_in_segment.mod(40).eq(0)]
                        donors=[pool[m.iloc[pool].pair_id.to_numpy()!=b['pair_id']] for b in bb]
                        for rate in [0,.1,.25,.5]:
                            for draw in range(20):
                                rng=np.random.default_rng(20261017+100*fold+draw)
                                mixed=ids.copy()
                                for j in range(len(bb)):
                                    mask=rng.random(7)<rate
                                    replacements=rng.choice(donors[j],7,replace=True)
                                    mixed[j,1:][mask]=replacements[mask]
                                frame=bag_frame(bb,expit(z[mixed]).mean(1,keepdims=True))
                                mismatch.append(dict(model=name,seed=seed,fold=fold,rate=rate,draw=draw,**summarize(frame)))
                print('RESULT '+json.dumps(dict(model=name,seed=seed,fold=fold,completed=True)),flush=True)
    pd.concat(rows,ignore_index=True).to_csv(root/'group_metrics.csv',index=False)
    pd.DataFrame(mismatch).to_csv(root/'misassociation.csv',index=False)
    return metrics


def train_weights(bags):
    q=pd.DataFrame(bags); q['side']=np.where(q.campaign.eq('Vicolungo'),'core','external')
    w=np.zeros(len(q))
    for label in [0,1]:
        for side in ['core','external']:
            a=q[(q.label==label)&(q.side==side)]
            assert len(a)>0
            for _,c in a.groupby('campaign'):
                for _,s in c.groupby('source_id'):
                    for _,p in s.groupby('pair_id'):
                        w[p.index]=.25/a.campaign.nunique()/c.source_id.nunique()/s.pair_id.nunique()/len(p)
    assert np.isclose(w.sum(),1)
    return w


@torch.no_grad()
def encode_all(model, tx):
    model.eval()
    return torch.cat([model.encode(x) for x in tx.split(512)])


@torch.no_grad()
def pooled_predictions(model, h, bags):
    model.eval(); ids=torch.tensor([b['indices'] for b in bags],device=h.device)
    bh=h[ids]; out={}
    for k in KS:
        subsets=torch.tensor(SUBSETS[k],device=h.device)
        v=bh[:,subsets].mean(2).flatten(0,1)
        z=torch.cat([model.classifier(x).squeeze(-1) for x in v.split(1024)])
        out[k]=expit(z.cpu().numpy().astype(np.float64)).reshape(len(bags),len(subsets))
    return out


def learned(cfg,m,bags,root,checks):
    raw=np.load(Path(cfg['prepared'])/'windows_4s.npz')['primitive_windows']
    x=features(raw,'legacy11',7); assert x.shape[-1]==11
    rows, metrics=[],[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            begin=time.time(); seed_all(seed+fold+20261017)
            path=root/f'seed_{seed}'/PARTS[fold]; path.mkdir(parents=True)
            original=model_path(cfg,'cnn',seed,fold)
            cp=checked_original(cfg,'cnn',original/'best_bce.pt',checks)
            statsfile=checked_original(cfg,'cnn',original/'feature_stats.json',checks)
            stats=json.loads(statsfile.read_text()); save(path/'feature_stats.json',stats)
            tx=torch.from_numpy(normalize(x,stats)).to(cfg['device'])
            model=TrajectoryNet(11,'multiscale_cnn',32,.2,1).to(cfg['device'])
            model.load_state_dict(torch.load(cp,map_location=cfg['device'],weights_only=False)['model_state_dict'])
            if cfg['mode']=='head':
                for p in model.parameters(): p.requires_grad=False
                for p in model.classifier.parameters(): p.requires_grad=True
            sp=splits(m,fold); pool=np.r_[sp['train'],sp['external'],sp['aux']]
            tr=role_bags(bags,pool); va=role_bags(bags,sp['val'])
            assert not {b['source_id'] for b in tr}&set(m.iloc[np.r_[sp['val'],sp['test'],sp['external_test']]].source_id)
            weights=torch.tensor(train_weights(tr),device=tx.device)
            save(path/'training_bags.json',dict(bag_ids=[b['bag_id'] for b in tr],weights=weights.cpu().tolist(),
                train_window_indices=pool.tolist(),validation_bag_ids=[b['bag_id'] for b in va]))
            indices=torch.tensor([b['indices'] for b in tr],device=tx.device)
            labels=torch.tensor([b['label'] for b in tr],device=tx.device,dtype=torch.float32)
            params=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            fixed_h=encode_all(model,tx) if cfg['mode']=='head' else None
            best=float('inf'); best_epoch=0; history=[]
            # One epoch draws len(training bags), with replacement under frozen hierarchical weights.
            for epoch in range(cfg['epochs']+1):
                loss_value=None
                if epoch:
                    model.train(); total=0.
                    order=torch.multinomial(weights,len(tr),replacement=True)
                    for batch in order.split(cfg['batch_size']):
                        k=KS[int(torch.randint(4,()).item())]
                        choose=torch.rand(len(batch),8,device=tx.device).argsort(1)[:,:k]
                        idx=indices[batch].gather(1,choose)
                        h=(fixed_h[idx] if fixed_h is not None else model.encode(tx[idx.flatten()]).reshape(len(batch),k,-1))
                        z=model.classifier(h.mean(1)).squeeze(-1)
                        target=labels[batch]*(1-cfg['label_smoothing'])+.5*cfg['label_smoothing']
                        loss=F.binary_cross_entropy_with_logits(z,target)
                        opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
                        total+=float(loss.detach())*len(batch)
                    loss_value=total/len(tr)
                h=fixed_h if fixed_h is not None else encode_all(model,tx)
                pred=pooled_predictions(model,h,va)
                met={str(k):summarize(bag_frame(va,pred[k])) for k in KS}
                score=float(np.mean([v['bce'] for v in met.values()]))
                history.append(dict(epoch=epoch,train_loss=loss_value,val_bce=score,val=met))
                if score<best-1e-8:
                    best,best_epoch=score,epoch
                    torch.save(dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        epoch=epoch,val_bce=score,config=cfg),path/'best_bce.pt')
                if epoch%10==0: print('EPOCH '+json.dumps(dict(mode=cfg['mode'],seed=seed,fold=fold,epoch=epoch,val_bce=score)),flush=True)
                if epoch-best_epoch>=cfg['patience']: break
            save(path/'history.json',history)
            model.load_state_dict(torch.load(path/'best_bce.pt',map_location=tx.device,weights_only=False)['model_state_dict'])
            h=encode_all(model,tx)
            for role in ['val','test']:
                for regime in ['dense','gap4','gap12']:
                    bb=role_bags(bags,sp[role],regime)
                    info=dict(model='cnn_'+cfg['mode'],seed=seed,fold=fold,role=role,regime=regime,fusion='embedding_mean')
                    record_eval(bb,pooled_predictions(model,h,bb),info,
                        root/'predictions'/f"cnn_{cfg['mode']}_{seed}_{fold}_{role}_{regime}.npz",rows,metrics)
            save(path/'result.json',dict(seed=seed,fold=fold,epoch=best_epoch,val_bce=best,train_bags=len(tr),
                val_bags=len(va),seconds=time.time()-begin))
            print('RESULT '+json.dumps(dict(mode=cfg['mode'],seed=seed,fold=fold,epoch=best_epoch,val_bce=best)),flush=True)
    pd.concat(rows,ignore_index=True).to_csv(root/'group_metrics.csv',index=False)
    return metrics


def main():
    cfg=json.loads(Path('experiment.json').read_text()); assert cfg['include_ngsim'] is False
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    root=Path('artifacts/run'); root.mkdir(parents=True,exist_ok=False); (root/'predictions').mkdir()
    m=pd.read_csv(Path(cfg['prepared'])/'windows.csv',float_precision='round_trip')
    np.testing.assert_array_equal(m.window_index,np.arange(len(m)))
    bags=make_bags(m); audit_bags(m,bags); save(root/'bags.json',bags)
    checks={}
    protocol=dict(config=cfg,dataset_sha256={n:sha(Path(cfg['prepared'])/n) for n in ['windows.csv','windows_4s.npz']},
        plan_sha256=sha('MULTISEGMENT_PLAN.txt'),numpy=np.__version__,torch=str(torch.__version__),
        selection='validation BCE averaged over K=1,2,4,8, equal record-local follower weights; includes epoch0',
        caveats=['ideal association, GPS same-interval surrogate', 'ACC/HV state not permanent autonomy',
                 'source CV not unseen vehicle CV', 'already inspected sources: exploratory',
                 'K subsets are repeated views, not independent samples'])
    save(root/'protocol.json',protocol)
    print('PROTOCOL '+json.dumps(protocol),flush=True)
    result=frozen(cfg,m,bags,root,checks) if cfg['mode']=='frozen' else learned(cfg,m,bags,root,checks)
    save(root/'source_hashes.json',checks); save(root/'results.json',result)
    df=pd.DataFrame(result); summary=[]
    for keys,part in df.groupby(['model','role','regime','fusion','k']):
        seed=part.groupby('seed')[['accuracy','balanced_accuracy','bce']].mean()
        summary.append(dict(zip(['model','role','regime','fusion','k'],[v.item() if isinstance(v,np.generic) else v for v in keys]),**{
            f'{c}_{stat}':float(getattr(seed[c],stat)()) for c in seed for stat in ['mean','std']}))
    save(root/'summary.json',summary)
    print('SUMMARY '+json.dumps(dict(mode=cfg['mode'],completed=True,evaluations=len(result))),flush=True)
