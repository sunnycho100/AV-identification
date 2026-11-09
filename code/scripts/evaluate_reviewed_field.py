"""Recover reviewed seeds, build bounded windows, and test frozen classifiers.

Use .venv-bev for build and .venv-classifier for classify. Original review,
detector outputs, automatic trajectories and model checkpoints remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from build_field_core_dataset import ROOT,read,save,inside_image
import trial_field_gap_repair as G

CFG=ROOT/'configs/field_review_test.json'
OUT=ROOT/'experiments/field_review_test_20261102'
SOURCE=ROOT/'experiments/field_review_applied_20261102'
PAIR_COLS=['frame','t_s','target_id','leader_id','target_x_m','target_y_m','leader_x_m','leader_y_m',
           'follower_speed_mps','leader_speed_mps','net_gap_m','direction','token','record_id','relation_source','is_imputed','lane']
VARIANTS=[('reviewed',False,None),('reviewed_gap1s',False,1.),('recovered',True,None),('recovered_gap1s',True,1.)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def table(path):
    return pd.read_csv(path,dtype={'target_id':str,'leader_id':str,'track_id':str,'token':str})


def smooth_raw(rows,cal,times,params):
    """Existing CV/RTS noise model with explicit detection support and native dt."""
    sys.path.insert(0,str(ROOT/'trajectory/scripts/tracking'))
    from postprocess_tracks import remove_jumps,footprint_cov
    from rts_smooth_track import build_kf
    M=np.array(cal['lidar2cam']);K=np.array(cal['K']);camera=-M[:3,:3].T@M[:3,3]
    cov=footprint_cov((camera[0],camera[1],15.),float((K[0,0]+K[1,1])/2),1.)
    points=[]
    for r in rows:
        d=r['matched_detection']['detection']
        points.append(dict(d,frame=r['frame'],t_s=r['t_s'],vx=0.,vy=0.,vz=0.,x_det=d['x'],y_det=d['y'],
                           raw_detection_index=r['matched_detection']['index'],image_iou=r['matched_detection']['iou']))
    if not points:return [],None,[]
    pieces=remove_jumps(points,30,times={f:(t,False) for f,t in times.items()},cov=cov)
    separated=[]
    for p in pieces:
        current=[]
        for r in p:
            if current and r['t_s']-current[-1]['t_s']>params['maximum_internal_gap_s']:
                separated.append(current);current=[]
            current.append(r)
        if current:separated.append(current)
    accepted=[];audit=[]
    for i,p in enumerate(separated):
        span=p[-1]['t_s']-p[0]['t_s']
        keep=len(p)>=params['minimum_observations'] and span>=params['minimum_span_s']
        audit.append(dict(piece=i,observations=len(p),span_s=span,accepted=keep,start_frame=p[0]['frame'],end_frame=p[-1]['frame']))
        if not keep:continue
        kf=build_kf(1/30);kf.x=np.array([p[0]['x'],p[0]['y'],0.,0.]);means=[];covs=[];Fs=[];Qs=[]
        for j,r in enumerate(p):
            dt=r['t_s']-p[j-1]['t_s'] if j else 1/30
            assert dt>0
            kf.F[0,2]=kf.F[1,3]=dt;kf.Q=np.diag([.01,.01,.1,.1])*dt
            kf.predict();kf.R=np.eye(2)*.25+cov(r['x'],r['y']);kf.update(np.array([r['x'],r['y']]))
            means.append(kf.x.copy());covs.append(kf.P.copy());Fs.append(kf.F.copy());Qs.append(kf.Q.copy())
        sm,*_=kf.rts_smoother(np.array(means),np.array(covs),Fs,Qs)
        for r,m in zip(p,sm):
            accepted.append(dict(r,x=float(m[0]),y=float(m[1]),vx=float(m[2]),vy=float(m[3]),
                                 speed_mps=float(np.hypot(m[2],m[3])),metric_piece=i,geometry_source='seed_associated_raw_detector_RTS'))
    dims=None
    if accepted:
        dims={name:float(np.median([r[key] for r in accepted])) for name,key in [('length_m','l'),('width_m','w'),('height_m','h')]}
        dims.update(matches=len(accepted),coverage=1.)
    return accepted,dims,audit


def export_pair(record,ids,tracks,dims,K,M,times,direction,overrides):
    own=[];lead=[];pairs=[];audit=[]
    for f in range(record['start_frame'],record['end_frame']+1):
        relation={'confirm_pair':'present','no_leader':'absent','leader_not_visible':'not_visible'}[record['decision']]
        if ('leader' in overrides and f>overrides['leader']):relation='not_visible'
        selected={};why=[]
        for role in ('target','leader'):
            k=ids[role];s=tracks.get(k,{}).get(f)
            if role=='leader' and relation!='present':why.append('leader_'+relation);continue
            if role in overrides and f>overrides[role]:why.append(role+'_outside_image_by_review');continue
            if k is None:why.append(role+'_unresolved_seed');continue
            if s is None:why.append(role+'_state_missing');continue
            if k not in dims:why.append(role+'_dimensions_missing');continue
            if 'x_det' not in s:why.append(role+'_detection_gap');continue
            if not inside_image(s,dims[k],K,M,1920,1080):why.append(role+'_outside_image');continue
            if not all(np.isfinite(s[key]) for key in ['x','y','vx','vy']):why.append(role+'_nonfinite');continue
            selected[role]=s
            row=dict(s,t_s=times[f],track_id=k,record_id=record['record_id'],leader_state=relation,is_imputed=False)
            (own if role=='target' else lead).append(row)
        sign=direction.get(f,0)
        if len(selected)==2:
            a,b=selected['target'],selected['leader']
            gap=(b['x']-a['x'])*sign-(dims[ids['target']]['length_m']+dims[ids['leader']]['length_m'])/2
            if sign not in (-1,1):why.append('unknown_direction')
            if b['vx']*sign<0:why.append('opposite_direction')
            if not np.isfinite(gap) or gap<=0:why.append('nonpositive_net_gap')
            if not why:
                pairs.append(dict(frame=f,t_s=times[f],target_id=ids['target'],leader_id=ids['leader'],
                    target_x_m=a['x'],target_y_m=a['y'],leader_x_m=b['x'],leader_y_m=b['y'],
                    follower_speed_mps=abs(a['vx']),leader_speed_mps=abs(b['vx']),net_gap_m=float(gap),
                    direction=sign,token=f'{record["record_id"]}:{sign}',record_id=record['record_id'],
                    relation_source='user_relation_with_reviewed_seed_association',is_imputed=False,lane=0))
        audit.append(dict(frame=f,t_s=times[f],leader_state=relation,target_id=ids['target'],
                          leader_id=ids['leader'] if relation=='present' else None,
                          target_usable='target' in selected,leader_usable='leader' in selected,pair_usable=not why,reason=';'.join(why)))
    return own,lead,pairs,pd.DataFrame(audit)


def build():
    cfg=read(CFG);resolution_path=ROOT/'configs/field_seed_resolution.json';resolution=read(resolution_path)
    assert resolution.get('visual_review_completed_before_classification') is True
    assert resolution['classification_predictions_consulted'] is False
    OUT.mkdir(exist_ok=True);save(OUT/'protocol.json',cfg);save(OUT/'seed_resolution.json',resolution)
    source_doc=read(SOURCE/'decisions_source.json');source_summary=read(SOURCE/'summary.json')
    assert digest(ROOT/'data/annotations/leader_review_decisions.json')==source_summary['decisions_sha256']
    choices={(r['clip'],r['role']):r for r in resolution['records']}
    survey=read(ROOT/'configs/field_full_survey.json');clip_specs={r['clip']:r for r in survey['clips']}
    manifest={};reports=[];recoveries=[];arrays={(v,d):[] for v,_,_ in VARIANTS for d in cfg['classification']['durations_s']};metadata={k:[] for k in arrays}
    def used(path):
        manifest[str(path.relative_to(ROOT))]=digest(path);return path
    for p in [CFG,resolution_path,SOURCE/'decisions_source.json',ROOT/'data/annotations/leader_review_decisions.json',ROOT/'configs/field_full_survey.json']:used(p)
    for r in source_doc['records']:
        clip,tag=r['clip'],r['tag'];td=ROOT/'trajectory/outputs/trajectories'/f'{clip}_{tag}'
        cached=read(used(td/'tracks.json'))['tracks'];tracks={k:{s['frame']:dict(s) for s in ss} for k,ss in cached.items()}
        dims=read(used(ROOT/'experiments/field_leader_review_20261102/data'/f'{clip}_dimensions.json'))
        cal=read(used(ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}'/'calibration_used.json'))
        K,M=np.array(cal['K']),np.array(cal['lidar2cam'])
        tdoc=read(used(ROOT/'trajectory/data/camera-data'/clip/'frame_times.json'));times={s['frame']:s['t_s'] for s in tdoc['frames']}
        for ss in tracks.values():
            for f,s in ss.items():s['tracker_t_s']=s['t_s'];s['t_s']=times[f]
        ids={role:r[role].get('id') if r.get(role) else None for role in ('target','leader')}
        for role in ('target','leader'):
            choice=choices.get((clip,role))
            if not choice:continue
            result=dict(clip=clip,role=role,method=choice['method'],note=choice['note'])
            if choice['method']=='existing_mot':
                ids[role]=choice['track_id'];assert ids[role] in tracks and ids[role] in dims
                result.update(status='ASSOCIATED_EXISTING_MOT',track_id=ids[role])
            elif choice['method']=='raw_detections':
                p=OUT/'seeds'/clip/role/f'{choice["image_source"]}_track.json'
                rows=read(used(p))['rows'];rows=[v for v in rows if v.get('matched_detection') and
                     choice.get('image_frame_min',r['start_frame'])<=v['frame']<=choice.get('image_frame_max',r['end_frame'])]
                for row in rows:
                    used(ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}'/f'{row["frame"]:03d}_pred.json')
                ss,dim,audit=smooth_raw(rows,cal,times,cfg['association'])
                if ss:
                    k='seed_'+role;ids[role]=k;tracks[k]={s['frame']:s for s in ss};dims[k]=dim
                    result.update(status='ASSOCIATED_RAW_DETECTION_FRAGMENT',track_id=k,observations=len(ss),
                                  start_frame=ss[0]['frame'],end_frame=ss[-1]['frame'],pieces=audit)
                    save(OUT/'seeds'/clip/role/'metric_track.json',dict(states=ss,dimensions=dim,audit=audit,source=choice))
                else:result.update(status='PENDING_INSUFFICIENT_METRIC_SUPPORT',pieces=audit)
            else:result.update(status='PENDING_INSUFFICIENT_METRIC_SUPPORT')
            recoveries.append(result);print('SEED',clip,role,result['status'],flush=True)
        direction={}
        if ids['target'] and (td/f'vehicle_{ids["target"]}.csv').exists():
            direction={int(s.frame):int(s.direction) for s in table(used(td/f'vehicle_{ids["target"]}.csv')).itertuples()}
        elif ids['target']:
            direction={f:int(np.sign(s['vx'])) for f,s in tracks[ids['target']].items()}
        overrides={v['role']:v['last_visible_frame'] for v in read(SOURCE/'application_config.json')['visibility_overrides'] if v['clip']==clip}
        own,lead,pairs,audit=export_pair(r,ids,tracks,dims,K,M,times,direction,overrides)
        dest=OUT/'clips'/clip;dest.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(own,columns=list(own[0]) if own else ['frame','t_s','x','y','vx','vy']).to_csv(dest/'target_trajectory.csv',index=False)
        pd.DataFrame(lead,columns=list(lead[0]) if lead else ['frame','t_s','x','y','vx','vy']).to_csv(dest/'leader_trajectory.csv',index=False)
        pd.DataFrame(pairs,columns=PAIR_COLS).to_csv(dest/'following_pairs.csv',index=False);audit.to_csv(dest/'frame_audit.csv',index=False)
        save(dest/'dimensions.json',{k:dims[k] for k in ids.values() if k in dims})
        # Existing reviewed observations must remain unchanged after seed recovery.
        old_pairs=table(used(SOURCE/'clips'/clip/'following_pairs.csv'))
        new_by={p['frame']:p for p in pairs}
        for p in old_pairs.to_dict('records'):
            assert p['frame'] in new_by
            for key in ['target_x_m','target_y_m','leader_x_m','leader_y_m','follower_speed_mps','leader_speed_mps','net_gap_m']:
                assert abs(p[key]-new_by[p['frame']][key])<1e-8,(clip,key)
        old_own=table(used(SOURCE/'clips'/clip/'target_trajectory.csv'))
        new_own={p['frame']:p for p in own}
        for p in old_own.to_dict('records'):
            for key in ['x','y','vx','vy']:assert abs(p[key]-new_own[p['frame']][key])<1e-8
        if r['decision']!='confirm_pair':assert not pairs
        if clip=='HV_V_EW_3':assert all(p['frame']<=274 for p in pairs)
        if clip=='HV_V_EW_6':assert all(p['frame']<=308 for p in own)
        specs=clip_specs[clip];model=next((x for x in specs['models'] if x!='original'),'original')
        base=dict(clip=clip,label=specs['label'],core=True,site=specs['site'],view=specs['view'],
                  training_exposed=clip in survey['models'][model]['training_clips'],has_recovered_seed=(clip,'leader') in choices)
        for variant,recovered,maximum in VARIANTS:
            src=dest if recovered else SOURCE/'clips'/clip
            rr=table(src/'following_pairs.csv').to_dict('records');au=table(src/'frame_audit.csv')
            au['reason']=au.reason.fillna('').str.replace('nonpositive_estimated_gap','nonpositive_net_gap',regex=False).str.replace('left_image_by_user_review','outside_image_by_review',regex=False)
            for p in rr:p['lane']=0
            if rr:
                tid,lid=rr[0]['target_id'],rr[0]['leader_id']
                directions={p['frame']:p['direction'] for p in rr};default=int(np.sign(np.median(list(directions.values()))))
                frame_table=pd.DataFrame([dict(frame=f,lead_id=lid if au.loc[au.frame==f,'leader_state'].iloc[0]=='present' else None,
                    lane=0,direction=directions.get(f,direction.get(f,default))) for f in range(r['start_frame'],r['end_frame']+1)])
            else:frame_table=pd.DataFrame(columns=['frame','lead_id','lane','direction'])
            frame_times=[(f,times[f]) for f in range(r['start_frame'],r['end_frame']+1)]
            originals=G.rebuild(rr,frame_table,au,frame_times,dims,None,tracks)[0]
            groups,gaps,holes=G.rebuild(rr,frame_table,au,frame_times,dims,maximum,tracks)
            G.DURATIONS=cfg['classification']['durations_s']
            windows=G.window_metadata(groups,holes,False,[g[0]['t_s'] for g in originals] if maximum else [])
            run=OUT/'runs'/variant/clip;run.mkdir(parents=True,exist_ok=True)
            flat=[dict(p,repaired_group=i) for i,g in enumerate(groups) for p in g]
            pd.DataFrame(flat,columns=PAIR_COLS+['repaired_group','target_position_imputed','leader_position_imputed']).to_csv(run/'following_pairs.csv',index=False)
            save(run/'gaps.json',gaps);save(run/'windows.json',windows)
            original_rows=[p for p in flat if not p['is_imputed']]
            assert len(original_rows)==len(rr)
            for a,b in zip(original_rows,rr):
                for key in ['target_x_m','target_y_m','leader_x_m','leader_y_m','follower_speed_mps','leader_speed_mps','net_gap_m']:
                    assert abs(a[key]-b[key])<1e-9
            counts=Counter()
            for w in windows:
                if not w['accepted']:continue
                g=groups[w['group']];ts=np.array([x['t_s'] for x in g]);values=np.array([[x['follower_speed_mps'],x['leader_speed_mps'],x['net_gap_m']] for x in g])
                q=w['start_s']+np.arange(w['duration_s']*10)/10
                assert q[0]>=ts[0]-1e-7 and q[-1]<=ts[-1]+1e-7
                x=np.stack([np.interp(q,ts,values[:,i]) for i in range(3)],1).astype(np.float32)
                assert np.isfinite(x).all() and (x[:,:2]>=0).all() and (x[:,2]>0).all()
                arrays[variant,w['duration_s']].append(x);metadata[variant,w['duration_s']].append(dict(w,**base,variant=variant));counts[w['duration_s']]+=1
            reports.append(dict(**base,variant=variant,target_points=len(own) if recovered else len(old_own),paired_points=len(rr),
                                imputed_pair_points=sum(p['is_imputed'] for p in flat),accepted_gaps=sum(g['accepted'] for g in gaps),
                                longest_pair_span_s=max((g[-1]['t_s']-g[0]['t_s'] for g in groups),default=0.),
                                **{f'windows_{d}s':counts[d] for d in cfg['classification']['durations_s']}))
    for (v,d),xs in arrays.items():
        folder=OUT/'classification'/v;folder.mkdir(parents=True,exist_ok=True)
        raw=np.stack(xs) if xs else np.empty((0,d*10,3),np.float32);meta=metadata[v,d]
        np.savez_compressed(folder/f'windows_{d}s.npz',primitive_windows=raw,labels=np.array([m['label'] for m in meta],np.int64),dt=np.array(.1))
        save(folder/f'windows_{d}s_metadata.json',meta)
    for p,h in manifest.items():assert digest(ROOT/p)==h
    save(OUT/'input_manifest.json',manifest);save(OUT/'seed_recovery_summary.json',recoveries)
    pd.DataFrame(reports).to_csv(OUT/'recording_results.csv',index=False)
    save(OUT/'build_summary.json',dict(complete=True,recordings=43,variants=VARIANTS,rows=reports,source_annotations_sha256=source_summary['decisions_sha256']))
    save(OUT/'build_validation.json',dict(status='PASS',original_review_observations_preserved=True,source_hashes_unchanged=True,
        native_timestamps=True,missing_leader_not_zero_filled=True,user_exit_boundaries_preserved=True,positive_net_gap=True,
        no_independent_vehicle_transform=True,seed_selection_before_classifier_predictions=True))
    print('BUILD COMPLETE',dict(Counter(r['status'] for r in recoveries)))


def classify():
    import torch
    from evaluate_field_baseline import TrajectoryNet,features,normalize,metrics,nonoverlap
    cfg=read(CFG);assert read(OUT/'build_summary.json')['complete'];torch.set_num_threads(4)
    members=[];manifest=read(ROOT/'experiments/field_baseline_20261101/model_manifest.json')
    for item in manifest:
        p=ROOT/item['path'];cp=torch.load(p,map_location='cpu',weights_only=True);c=cp['config']
        m=TrajectoryNet(cp['input_dim'],c['architecture'],c['width'],c['dropout'],c['layers']).eval();m.load_state_dict(cp['model_state_dict'],strict=True)
        stats=read(p.parent/'feature_stats.json');members.append((m,c,stats));item['sha256']=digest(p);item['feature_stats_sha256']=digest(p.parent/'feature_stats.json')
    assert len(members)==30
    save(OUT/'classifier_manifest.json',manifest);all_recordings=[];results=[]
    for variant,_,_ in VARIANTS:
        folder=OUT/'classification'/variant
        for d in cfg['classification']['durations_s']:
            raw=np.load(folder/f'windows_{d}s.npz')['primitive_windows'];meta=read(folder/f'windows_{d}s_metadata.json')
            if not len(raw):results.append(dict(variant=variant,duration_s=d,status='NO_WINDOWS'));continue
            for arch in ['transformer','multiscale_cnn']:
                probabilities=[];clipping=[]
                for m,c,stats in members:
                    if c['architecture']!=arch:continue
                    x=features(raw,c.get('features','legacy11'),c.get('smoothing',7))
                    clipping.append(float(((x<np.array(stats['lower']))|(x>np.array(stats['upper']))).mean()))
                    with torch.no_grad():p=torch.sigmoid(m(torch.from_numpy(normalize(x,stats)))).numpy()
                    assert p.shape==(len(raw),) and np.isfinite(p).all();probabilities.append(p)
                assert len(probabilities)==15
                pred=pd.DataFrame(meta);pred['prob_av']=np.mean(probabilities,0);pred['model_sd']=np.std(probabilities,0);pred['predicted_label']=(pred.prob_av>=.5).astype(int)
                pred.to_csv(folder/f'{arch}_{d}s_predictions.csv',index=False);aggregated=[]
                for clip,g in pred.groupby('clip',sort=False):
                    selected=nonoverlap(g);a=selected.iloc[0].to_dict();a.update(prob_av=float(selected.prob_av.mean()),
                        contributing_windows=len(selected),observed_duration_s=float((selected.end_s-selected.start_s+.1).sum()),
                        window_starts=selected.start_s.tolist(),architecture=arch,imputed_fraction=float(selected.imputed_fraction.mean()),
                        maximum_imputed_fraction=float(selected.imputed_fraction.max()),leader_ids=selected.leader_id.unique().tolist())
                    a['predicted_label']=int(a['prob_av']>=.5);aggregated.append(a)
                recordings=pd.DataFrame(aggregated);recordings.to_csv(folder/f'{arch}_{d}s_recordings.csv',index=False);all_recordings+=aggregated
                result=dict(variant=variant,architecture=arch,duration_s=d,windows=len(raw),video_metrics=metrics(recordings),
                    first_window_metrics=metrics(pred.sort_values('start_s').groupby('clip',sort=False).head(1)),
                    not_detector_adaptation_training=metrics(recordings[~recordings.training_exposed]),
                    source_feature_clipping_range=[min(clipping),max(clipping)])
                results.append(result);print('RESULT',variant,arch,d,result['video_metrics'],flush=True)
    allr=pd.DataFrame(all_recordings);longest=[]
    for (variant,arch),g in allr.groupby(['variant','architecture']):
        selected=g.sort_values('duration_s',ascending=False).groupby('clip',sort=False).head(1)
        selected.to_csv(OUT/'classification'/variant/f'{arch}_longest_available_recordings.csv',index=False)
        mm=metrics(selected);cm=mm['confusion_HV_AV'];mm['HV_recall']=cm[0][0]/sum(cm[0]) if sum(cm[0]) else None;mm['AV_recall']=cm[1][1]/sum(cm[1]) if sum(cm[1]) else None
        mm['majority_class_baseline']=max(mm['class_counts'].values())/mm['n']
        longest.append(dict(variant=variant,architecture=arch,metrics=mm,duration_counts=dict(Counter(str(int(v)) for v in selected.duration_s)),
                            coverage_of_43=mm['n']/43,not_detector_adaptation_training=metrics(selected[~selected.training_exposed])))
    for path,h in read(OUT/'input_manifest.json').items():assert digest(ROOT/path)==h
    save(OUT/'classification_summary.json',dict(status='EXPLORATORY_REVIEWED_FIELD_TRANSFER',results=results,longest_available=longest,
        threshold=.5,all_models_frozen=True,field_classifier_training=False,primary_duration_s=4,
        note='Video accuracy is conditional on paired-input availability. Manual-review development data and candidate metric geometry; not held-out field generalization. No absent leader was padded into the classifier.'))
    print('CLASSIFICATION COMPLETE')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','classify']);args=p.parse_args()
    (build if args.stage=='build' else classify)()
