"""Build masked Todd SE classifier inputs from frozen reviewed trajectories."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import evaluate_trajectory_quality as Q

ROOT=Q.ROOT
CFG=Q.read(ROOT/'configs/field_todd_complete.json')
OUT=ROOT/'experiments'/CFG['experiment']
SOURCE=ROOT/CFG['source']
FOCUS=Q.read(ROOT/CFG['focus'])
CLIPS=FOCUS['primary_clips']
STATES=['present','absent','not_visible']


def smooth(x):
    n=min(7,len(x) if len(x)%2 else len(x)-1)
    return savgol_filter(x,n,2,axis=0,mode='interp') if n>=3 else np.asarray(x).copy()


def featurize(vf,vl,gap,available,states,pair_id):
    """Never smooth/derive through unavailable leader observations or ID changes."""
    n=len(vf);f=np.zeros((n,11));m=np.zeros((n,11),bool)
    vf=smooth(vf);af=np.gradient(vf,.1,edge_order=2);jerk=np.gradient(af,.1,edge_order=2)
    f[:,:3]=np.column_stack([vf,af,jerk]);m[:,:3]=True
    changes=np.r_[True,(available[1:]!=available[:-1])|(pair_id[1:]!=pair_id[:-1])]
    for k in np.unique(np.cumsum(changes)):
        ix=np.flatnonzero(np.cumsum(changes)==k)
        if not available[ix[0]]:continue
        lv=smooth(vl[ix]);gg=np.maximum(smooth(gap[ix]),1e-6);dv=lv-vf[ix]
        closing=-dv;thw=np.clip(gg/np.maximum(vf[ix],1e-6),0,10)
        ttc=np.full(len(ix),30.);np.divide(gg,closing,out=ttc,where=closing>1e-6)
        f[np.ix_(ix,[3,5,6,7,9,10])]=np.column_stack([lv,gg,dv,closing,thw,np.clip(ttc,0,30)])
        m[np.ix_(ix,[3,5,6,7,9,10])]=True
        if len(ix)>=3:
            al=np.gradient(lv,.1,edge_order=2)
            f[np.ix_(ix,[4,8])]=np.column_stack([al,al-af[ix]])
            m[np.ix_(ix,[4,8])]=True
    return f.astype(np.float32),m,np.eye(3,dtype=np.float32)[states]


def pair_values(pair,q):
    values=np.zeros((len(q),2));available=np.zeros(len(q),bool);ids=np.full(len(q),-1,int)
    if not len(pair):return values,available,ids
    p=pair.sort_values('t_s').copy()
    new=p.t_s.diff().gt(.2)
    for col in ['target_id','leader_id','token']:
        new|=p[col].astype(str).ne(p[col].astype(str).shift())
    for k,(_,g) in enumerate(p.groupby(new.cumsum(),sort=False)):
        if len(g)<2:continue
        hit=(q>=g.t_s.iloc[0]-1e-8)&(q<=g.t_s.iloc[-1]+1e-8)
        values[hit]=np.column_stack([np.interp(q[hit],g.t_s,g[c]) for c in ['leader_speed_mps','net_gap_m']])
        available[hit]=True;ids[hit]=k
    assert (values[available,1]>0).all()
    return values,available,ids


def build():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'series').mkdir(exist_ok=True)
    Q.save(OUT/'protocol.json',CFG)
    manifest={}
    def use(p):manifest[str(p.relative_to(ROOT))]=Q.sha(p);return p
    use(Path(__file__).resolve());use(ROOT/'scripts/evaluate_trajectory_quality.py')
    use(ROOT/'configs/field_todd_complete.json');use(ROOT/CFG['focus']);use(ROOT/'data/annotations/leader_review_decisions.json')
    cat=pd.read_csv(use(ROOT/'catalogs/field_clips.csv')).set_index('clip_id')
    quality=pd.read_csv(use(ROOT/'experiments/field_trajectory_quality_20261102/clip_metrics.csv')).set_index('clip')
    timing=pd.read_csv(use(ROOT/'experiments/field_time_alignment_20261102/clip_results.csv')).set_index('clip')
    cp=use(ROOT/CFG['initial_checkpoint']);use(cp.parent/'feature_stats.json')
    data=[];masks=[];states=[];valids=[];metadata=[];inventory=[]
    for ci,clip in enumerate(CLIPS):
        cr=cat.loc[clip];vp=use(ROOT/cr.video_path);assert Q.sha(vp)==cr.video_sha256
        folder=SOURCE/'clips'/clip
        track=Q.csv(use(folder/'target_trajectory.csv')).sort_values('t_s').reset_index(drop=True)
        pair=Q.csv(use(folder/'following_pairs.csv'))
        audit=Q.csv(use(folder/'frame_audit.csv')).sort_values('t_s')
        assert not track.is_imputed.any() and track.frame.is_unique
        allrows=[];start_index=len(data)
        for k,g in enumerate(Q.pieces(track,.2)):
            n=int(np.floor((g.t_s.iloc[-1]-g.t_s.iloc[0])/.1+1e-7))+1
            if n<CFG['minimum_window_samples']:continue
            q=g.t_s.iloc[0]+np.arange(n)*.1
            vf=np.interp(q,g.t_s,abs(g.vx))
            vals,available,pids=pair_values(pair,q)
            at=audit.t_s.to_numpy();idx=np.searchsorted(at,q).clip(0,len(at)-1)
            prev=np.maximum(idx-1,0);idx=np.where(abs(at[prev]-q)<abs(at[idx]-q),prev,idx)
            state=np.array([STATES.index(s) for s in audit.iloc[idx].leader_state])
            available&=(state==0);pids[~available]=-1;vals[~available]=0
            for j in range(0,n,CFG['maximum_window_samples']):
                end=min(j+CFG['maximum_window_samples'],n)
                if end-j<CFG['minimum_window_samples']:continue
                f,m,s=featurize(vf[j:end],vals[j:end,0],vals[j:end,1],available[j:end],state[j:end],pids[j:end])
                pad=CFG['maximum_window_samples']-len(f)
                data.append(np.pad(f,((0,pad),(0,0))));masks.append(np.pad(m,((0,pad),(0,0))))
                states.append(np.pad(s,((0,pad),(0,0))));valids.append(np.r_[np.ones(len(f),bool),np.zeros(pad,bool)])
                meta=dict(window=len(data)-1,clip=clip,clip_index=ci,label=int(clip.startswith('AV_')),piece=k,
                          start_s=float(q[j]),end_s=float(q[end-1]),samples=len(f),duration_s=float((len(f)-1)*.1),
                          leader_available_fraction=float(available[j:end].mean()),
                          absent_fraction=float((state[j:end]==1).mean()),not_visible_fraction=float((state[j:end]==2).mean()))
                metadata.append(meta)
                for z in range(len(f)):
                    allrows.append(dict(window=meta['window'],t_s=float(q[j+z]),ego_speed_mps=float(f[z,0]),
                        ego_acc_mps2=float(f[z,1]),ego_jerk_mps3=float(f[z,2]),leader_available=bool(m[z,5]),
                        net_gap_m=float(f[z,5]) if m[z,5] else None,leader_state=STATES[state[j+z]],piece=k))
        assert len(data)>start_index,clip
        ss=pd.DataFrame(allrows);ss.to_csv(OUT/'series'/f'{clip}.csv',index=False)
        qr=quality.loc[clip];tr=timing.loc[clip]
        row=dict(clip=clip,label=int(clip.startswith('AV_')),view=cr['view'],windows=len(data)-start_index,
            target_observations=len(track),reviewed_coverage=float(qr.observed_frame_fraction),
            retained_samples=len(ss),retained_duration_s=float(sum(x['duration_s'] for x in metadata[start_index:])),
            shortest_window_s=float(min(x['duration_s'] for x in metadata[start_index:])),
            leader_available_fraction=float(ss.leader_available.mean()),
            absent_fraction=float((ss.leader_state=='absent').mean()),not_visible_fraction=float((ss.leader_state=='not_visible').mean()),
            mean_ego_speed_mps=float(ss.ego_speed_mps.mean()),median_ego_speed_mps=float(ss.ego_speed_mps.median()),
            speed_p10_mps=float(ss.ego_speed_mps.quantile(.1)),speed_p90_mps=float(ss.ego_speed_mps.quantile(.9)),
            median_acceleration_mps2=float(ss.ego_acc_mps2.median()),median_abs_jerk_mps3=float(abs(ss.ego_jerk_mps3).median()),
            median_observed_net_gap_m=float(ss.net_gap_m.median()) if ss.net_gap_m.notna().any() else None,
            gps_status=tr.status,gps_test_speed_mae_mps=None,delay_s=None,delay_uncertainty=None,
            detector_training_exposed=bool(qr.training_exposed),leader_state=str(qr.leader_state))
        if bool(qr.gps_available):
            _,gps=Q.gps_frame(use(ROOT/cr.gps_path))
            if tr.status=='ESTIMATED_DELAY':
                old=pd.read_csv(use(ROOT/'experiments/field_time_alignment_20261102/series'/f'{clip}.csv'))
                ref,ok=Q.gps_interp(gps,old.t_s.to_numpy()+tr.delay_s);assert ok.all()
                err=old.visual_speed_mps-np.linalg.norm(ref[:,2:],axis=1)
                score=float(abs(err[old.is_test]).mean());assert abs(score-tr.test_speed_after_mae_mps)<1e-8
                row.update(gps_test_speed_mae_mps=score,delay_s=float(tr.delay_s),delay_uncertainty=tr.identifiability_flags)
        row['quality_note']='GPS speed MAE >1 m/s' if row['gps_test_speed_mae_mps'] is not None and row['gps_test_speed_mae_mps']>1 else 'GPS speed agreement <=1 m/s' if row['gps_test_speed_mae_mps'] is not None else 'Not GPS-verified for this temporal split'
        if row['shortest_window_s']<1:row['quality_note']+='; includes <1 s input'
        inventory.append(row)
    x=np.stack(data);m=np.stack(masks);st=np.stack(states);valid=np.stack(valids)
    assert np.isfinite(x).all() and (x[~m]==0).all()
    assert (m[:,:,:3]==valid[:,:,None]).all()
    assert (st.sum(-1)==valid).all()
    meta=pd.DataFrame(metadata);inv=pd.DataFrame(inventory)
    np.savez_compressed(OUT/'dataset.npz',features=x,feature_mask=m,relation_state=st,valid_tokens=valid,
        labels=meta.label.to_numpy(np.int64),clip_index=meta.clip_index.to_numpy(np.int64))
    meta.to_csv(OUT/'windows.csv',index=False);inv.to_csv(OUT/'inventory.csv',index=False)
    # Video-disjoint outer and inner folds, chosen before any outcome is computed.
    folds=[]
    for i,test in enumerate(CLIPS):
        remaining=[c for c in CLIPS if c!=test]
        val=[]
        for label in ['AV_','HV_']:
            choices=[c for c in remaining if c.startswith(label)];val.append(choices[i%len(choices)])
        train=[c for c in remaining if c not in val]
        assert len(train)==10 and not set(train)&set(val) and test not in train+val
        folds.append(dict(fold=i,test=[test],validation=val,train=train))
    Q.save(OUT/'folds.json',folds)
    Q.save(OUT/'input_manifest.json',manifest)
    for p,h in manifest.items():assert Q.sha(ROOT/p)==h
    Q.save(OUT/'build_validation.json',dict(status='PASS',videos=len(inv),windows=len(meta),raw_video_hashes_verified=13,
        all_13_covered=True,gps_not_classifier_input=True,window_duration_range_s=[float(meta.duration_s.min()),float(meta.duration_s.max())],
        explicit_missing_features=True,padding_has_attention_mask=True,source_inputs_unchanged=True))
    print(inv[['clip','windows','retained_samples','shortest_window_s','leader_available_fraction','gps_test_speed_mae_mps']].to_string(index=False),flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');build()
