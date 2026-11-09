"""Human-anchor recovery and native-time <=2.5s Todd inputs; preserve older runs."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from todd_three_second_tracks import ROOT, OUT as PREVIOUS, OLD, CLIPS, context, read, save, inside_image
from track_review_seeds import adaptive_flow_direction, iou, contact
from visualize_field_baseline import project
from evaluate_reviewed_field import smooth_raw
from build_todd_three_second import segments, LIMITS, VISUAL_IN_FRAME
from todd_experiment_data import pair_values, featurize

CFG=read(ROOT/'configs/field_todd_variable.json')
OUT=ROOT/'experiments'/CFG['experiment']
MAX=CFG['maximum_samples']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'configs/field_todd_variable.json'
    frozen=OUT/'protocol.json'
    if frozen.exists():assert read(frozen)==CFG,'Protocol changed after initialization'
    else:save(frozen,CFG)
    target=OUT/'human_review.json'
    if not target.exists():
        sys.path.insert(0,str(ROOT/'review_web'))
        from trajectory_review import discover
        p=discover(ROOT,ROOT/'review_web')['todd_se_extension']
        reviews=p.reviews()['records'];assert len(reviews)==5
        assert all(r['identity_continuous'] for r in reviews)
        doc=dict(schema='trajectory-review/v1',project_id=p.id,records=reviews)
        p.validate(doc);save(target,doc)
    return {r['clip']:r for r in read(target)['records']}


def image_candidates(clip,r):
    _,folder,_,_,_,_,times=context(clip)
    low=max(r['start_frame'],r['trim']['keep_start_frame']);high=min(r['end_frame'],r['trim']['keep_end_frame'])
    anchors=sorted(r['anchors'],key=lambda a:a['frame']);votes={};stops=[]
    for k,seed in enumerate(anchors):
        a=anchors[k-1]['frame'] if k else low;b=anchors[k+1]['frame'] if k+1<len(anchors) else high
        for f in range(seed['frame'],a,-1):
            if times[f]-times[f-1]>.2:a=f;break
        for f in range(seed['frame'],b):
            if times[f+1]-times[f]>.2:b=f;break
        back,bs=adaptive_flow_direction(folder,seed,a) if a<seed['frame'] else ([],None)
        forward,fs=adaptive_flow_direction(folder,seed,b) if b>seed['frame'] else ([],None)
        for row in back+[dict(frame=seed['frame'],bbox=seed['bbox'],source='human_anchor',image_correlation=1.)]+forward:
            if any(e['start_frame']<=row['frame']<=e['end_frame'] for e in r['excluded_ranges']):continue
            row['anchor_frame']=seed['frame'];votes.setdefault(row['frame'],[]).append(row)
        stops.append(dict(anchor=seed['frame'],backward=bs,forward=fs))
    rows=[]
    for f,choices in sorted(votes.items()):
        explicit=[v for v in choices if v['source']=='human_anchor']
        best=copy.deepcopy((explicit or sorted(choices,key=lambda v:abs(f-v['anchor_frame'])))[0])
        best.update(t_s=times[f],image_consensus=bool(explicit) or all(iou(best['bbox'],v['bbox'])>=.2 for v in choices),
                    anchor_votes=[dict(anchor=v['anchor_frame'],bbox=v['bbox']) for v in choices])
        rows.append(best)
    return dict(clip=clip,review_record_id=r['record_id'],stops=stops,rows=rows)


def match_row(row,ds,K,M,path):
    matches=[]
    for j,d in enumerate(ds):
        if d['class_name'] not in ('car','truck','bus') or d['score']<CFG['association']['minimum_score']:continue
        uv=project(d,K,M)
        if uv is None:continue
        bb=np.r_[uv.min(0),uv.max(0)];value=iou(row['bbox'],bb)
        if value>0:matches.append(dict(index=j,iou=value,bbox=bb.tolist(),detection=d,source_file=str(path.relative_to(ROOT))))
    matches.sort(key=lambda v:v['iou'],reverse=True)
    valid=bool(matches and matches[0]['iou']>=CFG['association']['minimum_iou'] and
               (len(matches)==1 or matches[0]['iou']-matches[1]['iou']>=CFG['association']['minimum_iou_margin']))
    return matches[0] if valid else None,matches[:2]


def recover():
    cv2.setNumThreads(2);reviews=initialize();model=None;summary=[]
    for clip,r in reviews.items():
        dest=OUT/'recovery'/clip;dest.mkdir(parents=True,exist_ok=True)
        _,folder,dd,cal,K,M,times=context(clip)
        flowpath=dest/'image_flow.json'
        if not flowpath.exists():save(flowpath,image_candidates(clip,r))
        doc=read(flowpath);fresh=0
        for row in doc['rows']:
            row['matched_detection']=None
            if not row['image_consensus']:continue
            frame=row['frame'];path=dd/f'{frame:03d}_pred.json'
            match,top=match_row(row,read(path),K,M,path)
            if match is None:
                cached=PREVIOUS/'low_score_detections'/clip/f'{frame:03d}_pred.json'
                path=cached if cached.exists() else dest/f'{frame:03d}_pred.json'
                if not path.exists():
                    if model is None:
                        import torch
                        torch.set_num_threads(4)
                        import verify_field_branches
                        from scripts.finetune import train_finetune as T
                        from scripts.object_detection import run_bevheight_generic as G
                        model=T.build_model();info=T.load_checkpoint(model,ROOT/'trajectory/checkpoints/ft102_run3_v2labels_ep5.ckpt')
                        assert not info['missing'] and not info['unexpected']
                        model.head.bbox_coder.score_threshold=.05;model.head.test_cfg['score_threshold']=.05
                        model.to(G.DEVICE).eval();exp=G.load_exp(G.DEFAULT_CONFIG)
                    save(path,G.run_frame(model,folder/f'{frame:03d}.jpg',K,M,exp,score_thresh=.05));fresh+=1
                match,top=match_row(row,read(path),K,M,path)
            row.update(matched_detection=match,top_candidates=top)
        save(dest/'image_candidates.json',doc)
        contact(doc['rows'],folder,dest/'candidate_review.jpg',clip)
        summary.append(dict(clip=clip,reviewed_span_s=r['end_s']-r['start_s'],image_frames=len(doc['rows']),
                            matched=sum(bool(v['matched_detection']) for v in doc['rows']),new_inference_frames=fresh,
                            conflicting_image_frames=sum(not v['image_consensus'] for v in doc['rows'])))
        save(OUT/'recovery_summary.json',summary);print(summary[-1],flush=True)


def select(groups,camera):
    candidates=[]
    for k,g in enumerate(groups):
        t=g.t_s.to_numpy();n=min(MAX,int(np.floor((t[-1]-t[0])*10+1e-7))+1)
        while n>=CFG['minimum_samples']:
            span=(n-1)*.1;last=t[-1]-span;found=False
            for start in np.unique(np.r_[t[t<=last+1e-8],last]):
                q=start+np.arange(n)*.1;dt=np.diff(t)
                extension=float(np.maximum(0,np.minimum(t[1:],q[-1])-np.maximum(t[:-1],q[0]))[dt>.2+1e-8].sum())
                if extension>span*CFG['maximum_extended_gap_fraction']+1e-8:continue
                xyz=np.column_stack([np.interp(q,t,g[c]) for c in ['x','y','z']])
                distance=float(np.linalg.norm(xyz-camera,axis=1).mean())
                candidates.append((-n,distance,float(start),k,extension,q,g));found=True
            if found:break
            n-=1
    return min(candidates,key=lambda v:v[:3]) if candidates else None


def build():
    reviews=initialize();inventory=[];windows=[];xs=[];ms=[];valids=[];raws=[];available=[];manifest={}
    for clip in CLIPS:
        record,folder,dd,cal,K,M,times=context(clip);dest=OUT/'clips'/clip;dest.mkdir(parents=True,exist_ok=True)
        source=(OUT/'recovery' if clip in reviews else PREVIOUS/'low_score_detections')/clip/'image_candidates.json'
        manifest[str(source.relative_to(ROOT))]=digest(source);doc=read(source)
        lo,hi=(reviews[clip]['start_frame'],reviews[clip]['end_frame']) if clip in reviews else LIMITS.get(clip,(0,100000))
        rows=[r for r in doc['rows'] if lo<=r['frame']<=hi and r.get('image_consensus',True)]
        matches=[r for r in rows if r.get('matched_detection')]
        states,dims,audit=smooth_raw(matches,cal,times,dict(maximum_internal_gap_s=.5,minimum_observations=8,minimum_span_s=.4))
        states=[dict(s,track_id='reviewed_ego',record_id=record['record_id'],is_imputed=False) for s in states
                if inside_image(s,dims,K,M,1920,1080) or s['frame'] in VISUAL_IN_FRAME.get(clip,[])]
        df=pd.DataFrame(states);groups=segments(df,rows,times);df.to_csv(dest/'target_trajectory.csv',index=False)
        save(dest/'reconstruction_audit.json',dict(pieces=audit,dimensions=dims,reviewed_frame_limits=[lo,min(hi,max(times))],
            human_anchor_review=clip in reviews,raw_associated_points=len(matches),new_3d_geometry=False))
        span=max((g.t_s.iloc[-1]-g.t_s.iloc[0] for g in groups),default=0.)
        inv=dict(clip=clip,label=int(clip.startswith('AV_')),recovered_points=len(df),recovered_longest_s=float(span),
                 human_anchor_review=clip in reviews,detector_training_exposed=clip in ['AV_T_WE_1','HV_T_EW_1'])
        camera=-M[:3,:3].T@M[:3,3];chosen=select(groups,camera)
        if chosen is None:
            inv.update(eligible=False,reason='No supported metric segment with at least five 10Hz points');inventory.append(inv);continue
        negn,distance,start,piece,extension,q,g=chosen;n=len(q)
        vf=np.interp(q,g.t_s,abs(g.vx));sign=1 if np.median(g.vx)>0 else -1
        from evaluate_trajectory_quality import csv
        lead=csv(OLD/'clips'/clip/'leader_trajectory.csv');pairs=[]
        if len(lead):
            olddims=read(OLD/'clips'/clip/'dimensions.json')
            for s in lead.to_dict('records'):
                if not g.t_s.iloc[0]<=s['t_s']<=g.t_s.iloc[-1]:continue
                gap=(s['x']-np.interp(s['t_s'],g.t_s,g.x))*sign-(dims['length_m']+olddims[str(s['track_id'])]['length_m'])/2
                if gap<=0 or s['vx']*sign<0:continue
                pairs.append(dict(t_s=s['t_s'],leader_speed_mps=abs(s['vx']),net_gap_m=gap,target_id='ego',leader_id=s['track_id'],token=str(s['record_id'])))
        vals,a,pids=pair_values(pd.DataFrame(pairs),q);f,m,_=featurize(vf,vals[:,0],vals[:,1],a,np.zeros(n,int),pids)
        raw=np.column_stack([vf,vals]);pad=MAX-n
        xs.append(np.pad(f,((0,pad),(0,0))));ms.append(np.pad(m,((0,pad),(0,0))))
        raws.append(np.pad(raw,((0,pad),(0,0))));available.append(np.pad(a,(0,pad)));valids.append(np.arange(MAX)<n)
        series=pd.DataFrame(dict(t_s=q,ego_speed_mps=f[:,0],ego_acc_mps2=f[:,1],ego_jerk_mps3=f[:,2],
            x_m=np.interp(q,g.t_s,g.x),y_m=np.interp(q,g.t_s,g.y),leader_available=a,
            leader_speed_mps=np.where(a,f[:,3],np.nan),net_gap_m=np.where(a,f[:,5],np.nan)))
        series.to_csv(dest/'selected.csv',index=False)
        meta=dict(clip=clip,label=inv['label'],samples=n,start_s=start,end_s=float(q[-1]),duration_s=float((n-1)*.1),
                  mean_camera_distance_m=distance,extended_gap_s=extension,leader_available_fraction=float(a.mean()),
                  less_than_one_second=n<11,source_piece=piece)
        save(dest/'selection.json',meta);windows.append(meta);inv.update(meta,eligible=True);inventory.append(inv)
        print(clip,'samples',n,'duration',round((n-1)*.1,1),'leader',int(a.sum()),flush=True)
    for p in [OUT/'human_review.json',ROOT/'configs/field_todd_variable.json',Path(__file__)]:manifest[str(p.relative_to(ROOT))]=digest(p)
    save(OUT/'trajectory_input_manifest.json',manifest)
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'windows.csv',index=False)
    np.savez_compressed(OUT/'dataset.npz',features=np.stack(xs),feature_mask=np.stack(ms),valid_tokens=np.stack(valids),
        primitive_windows=np.stack(raws),leader_available=np.stack(available),labels=np.array([w['label'] for w in windows]))
    save(OUT/'trajectory_summary.json',dict(total_videos=len(CLIPS),eligible=len(windows),
        durations={w['clip']:w['duration_s'] for w in windows},gps_or_classification_consulted=False,
        timestamps_stretched=False,unobserved_time_used=False,metric_interpolation_is_bounded=True))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('stage',choices=['recover','build']);a=p.parse_args()
    {'recover':recover,'build':build}[a.stage]()
