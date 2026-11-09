"""Freeze image-reviewed metric recovery and select nearest supported 3.0 seconds."""
import sys
import hashlib
import numpy as np
import pandas as pd
from todd_three_second_tracks import ROOT,OUT,OLD,CLIPS,context,read,save,inside_image
from evaluate_reviewed_field import smooth_raw
import evaluate_trajectory_quality as Q
from todd_experiment_data import pair_values,featurize

# Contact-sheet visual review by assistant, before new field predictions.
# Earlier backward flow locked on another car/logo in these two recordings.
LIMITS={'AV_T_WE_1':(56,100000),'AV_T_WE_3':(45,100000),'HV_T_WE_6':(85,100000)}
VISUAL_IN_FRAME={'AV_T_EW_3':[293]}


def segments(df,image_rows,times):
    if not len(df):return []
    result=[];g=[]
    support={r['frame'] for r in image_rows}
    for r in df.sort_values('t_s').to_dict('records'):
        split=False
        if g:
            p=g[-1];gap=r['t_s']-p['t_s']
            split=gap>.5+1e-8 or r['metric_piece']!=p['metric_piece']
            if gap>.2:
                split |= not all(f in support for f in range(p['frame'],r['frame']+1))
                split |= any(times[f]-times[f-1]>.2 for f in range(p['frame']+1,r['frame']+1))
        if split:result.append(pd.DataFrame(g));g=[]
        g.append(r)
    if g:result.append(pd.DataFrame(g))
    return result


def select(groups,camera):
    windows=[]
    for piece,g in enumerate(groups):
        t=g.t_s.to_numpy();last=t[-1]-3
        if last<t[0]-1e-8:continue
        # Starts at observed timestamps or ends at the last observation.
        starts=np.unique(np.r_[t[t<=last+1e-8],last])
        for start in starts:
            q=start+np.arange(31)*.1
            dt=np.diff(t);length=np.maximum(0,np.minimum(t[1:],q[-1])-np.maximum(t[:-1],q[0]))
            extended=float(length[dt>.2+1e-8].sum())
            if extended>3*.2+1e-8:continue
            xyz=np.column_stack([np.interp(q,t,g[k]) for k in ['x','y','z']])
            distance=float(np.linalg.norm(xyz-camera,axis=1).mean())
            windows.append((distance,start,piece,extended,q,g))
    if not windows:return None
    return min(windows,key=lambda x:(x[0],x[1]))


def build():
    inventory=[];xs=[];ms=[];raws=[];availabilities=[];windows=[];audits=[];manifest={}
    cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    for clip in CLIPS:
        record,folder,dd,cal,K,M,times=context(clip)
        p=OUT/'low_score_detections'/clip/'image_candidates.json';manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        doc=read(p);lo,hi=LIMITS.get(clip,(0,100000));rows=[v for v in doc['rows'] if lo<=v['frame']<=hi]
        matches=[r for r in rows if r.get('matched_detection')]
        states,dims,audit=smooth_raw(matches,cal,times,dict(maximum_internal_gap_s=.5,minimum_observations=8,minimum_span_s=.4))
        states=[dict(s,track_id='image_verified_ego',record_id=record['record_id'],is_imputed=False) for s in states if inside_image(s,dims,K,M,1920,1080) or s['frame'] in VISUAL_IN_FRAME.get(clip,[])]
        df=pd.DataFrame(states);groups=segments(df,rows,times)
        dest=OUT/'clips'/clip;dest.mkdir(parents=True,exist_ok=True);df.to_csv(dest/'target_trajectory.csv',index=False)
        save(dest/'reconstruction_audit.json',dict(visual_identity_review='Assistant reviewed nine evenly sampled image panels plus recovery panels; not independent manual metric ground truth',
            classification_predictions_consulted=False,accepted_frame_limits=[lo,min(hi,max(times))],raw_associated_points=len(matches),pieces=audit,
            visually_confirmed_in_frame_overrides=VISUAL_IN_FRAME.get(clip,[]),override_reason='Full native image inspected: actual car front is inside frame; projected 3D length extends beyond image',
            dimensions=dims,source_calibration=cal,low_score_observations=sum(bool(v.get('low_threshold_recovery')) for v in matches)))
        before=Q.csv(OLD/'clips'/clip/'target_trajectory.csv');camera=-M[:3,:3].T@M[:3,3];chosen=select(groups,camera)
        inv=dict(clip=clip,label=int(clip.startswith('AV_')),old_longest_s=Q.span(Q.pieces(before)),old_points=len(before),
                 recovered_points=len(df),recovered_longest_s=Q.span(groups),eligible_3s=chosen is not None,
                 image_span_s=rows[-1]['t_s']-rows[0]['t_s'],source_time_jump=max(np.diff(list(times.values()))),
                 detector_training_exposed=clip in ['HV_T_EW_1','AV_T_WE_1'])
        if chosen is None:
            inv['reason']='No identity-verified, in-image, 3.0s metric interval under gap limits'
            inventory.append(inv);print(clip,'NO 3s; longest',round(inv['recovered_longest_s'],3),flush=True);continue
        distance,start,piece,extended,q,g=chosen
        vf=np.interp(q,g.t_s,abs(g.vx));sign=1 if np.median(g.vx)>0 else -1
        # Preserve the reviewed leader observations in the SAME original coordinate frame.
        lead=Q.csv(OLD/'clips'/clip/'leader_trajectory.csv');pairs=[]
        if len(lead):
            olddims=read(OLD/'clips'/clip/'dimensions.json')
            for s in lead.to_dict('records'):
                if s['t_s']<g.t_s.iloc[0] or s['t_s']>g.t_s.iloc[-1]:continue
                ex=np.interp(s['t_s'],g.t_s,g.x);gap=(s['x']-ex)*sign-(dims['length_m']+olddims[str(s['track_id'])]['length_m'])/2
                if gap<=0 or s['vx']*sign<0:continue
                pairs.append(dict(t_s=s['t_s'],leader_speed_mps=abs(s['vx']),net_gap_m=gap,target_id='ego',leader_id=s['track_id'],token=str(s['record_id'])))
        vals,a,pids=pair_values(pd.DataFrame(pairs),q)
        f,m,_=featurize(vf,vals[:,0],vals[:,1],a,np.zeros(31,int),pids)
        raw=np.column_stack([vf,vals]);xs.append(f);ms.append(m);raws.append(raw);availabilities.append(a)
        relation={'confirm_pair':'present','no_leader':'absent','leader_not_visible':'not_visible'}[record['decision']]
        audit_times=(q>=times[record['start_frame']])&(q<=times[record['end_frame']])
        series=pd.DataFrame(dict(t_s=q,ego_speed_mps=f[:,0],ego_acc_mps2=f[:,1],ego_jerk_mps3=f[:,2],
            x_m=np.interp(q,g.t_s,g.x),y_m=np.interp(q,g.t_s,g.y),leader_available=a,
            leader_speed_mps=np.where(a,f[:,3],np.nan),net_gap_m=np.where(a,f[:,5],np.nan),
            review_relation=np.where(audit_times,relation,'outside_review_unknown')))
        series.to_csv(dest/'selected_3s.csv',index=False)
        meta=dict(clip=clip,label=inv['label'],start_s=float(q[0]),end_s=float(q[-1]),samples=31,duration_s=3.,mean_camera_distance_m=distance,
                  extended_gap_s=extended,leader_available_fraction=float(a.mean()),review_relation=relation,
                  source_piece=piece,selection_candidates_do_not_use_gps_or_label=True)
        windows.append(meta);inv.update(meta);inventory.append(inv)
        save(dest/'selection.json',meta);print(clip,'3s',round(q[0],3),'..',round(q[-1],3),'leader',int(a.sum()),flush=True)
    save(OUT/'trajectory_input_manifest.json',manifest)
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'windows.csv',index=False)
    np.savez_compressed(OUT/'dataset.npz',features=np.stack(xs),feature_mask=np.stack(ms),primitive_windows=np.stack(raws),leader_available=np.stack(availabilities),labels=np.array([v['label'] for v in windows]))
    save(OUT/'trajectory_summary.json',dict(videos=13,eligible=len(windows),av=sum(v['label'] for v in windows),hv=sum(1-v['label'] for v in windows),
        all_windows_duration_s=3.,samples=31,mean_selection_rule='Nearest-camera mean Euclidean distance among valid windows',
        no_ego_padding=True,no_extrapolation=True,no_gps_selection=True,visual_exclusions=LIMITS))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');build()
