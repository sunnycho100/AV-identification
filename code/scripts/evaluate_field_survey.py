"""Frozen all-video diagnostics; automatic candidate windows are not ground truth."""
import argparse,time,traceback
from collections import Counter
import cv2,numpy as np,pandas as pd
from field_full_survey import ROOT,TR,OUT,CONFIG,read,save
from field_full_review import ANNOT

def longest_span(states,gap=.2):
    longest=0.;start=prev=None
    for s in sorted(states,key=lambda s:s['t_s']):
        t=s['t_s']
        if prev is None or t-prev>gap+1e-7:start=t
        longest=max(longest,t-start);prev=t
    return float(longest)

def export_one(row,key):
    from identify_baseline_targets import identify,marker_centres
    from build_field_core_dataset import convert
    import diagnose_view_refinement as diag
    cfg=read(CONFIG);mc=cfg['models'][key];c=row['clip'];tag=mc['tag'];dest=OUT/'runs'/f'{c}_{key}'
    base=dict(clip=c,model=key,site=row['site'],view=row['view'],core=row['core'],label=row['label'],training_exposed=c in mc['training_clips'],previously_inspected=row['previously_inspected'])
    mp=OUT/'markers'/f'{c}_markers.json'
    if not mp.exists():return dict(base,status='SOURCE_IDENTITY_UNRESOLVED')
    centers=marker_centres(mp)[0]
    if not centers:return dict(base,status='MARKER_RUN_UNRESOLVED',marker_frames=0)
    ident=identify(c,centers,dict(tag=tag,marker_seeds={c:True}),dest)
    result=dict(base,identity=ident,marker_frames=len(centers),identity_accepted=ident['accepted'],marker_votes=ident['votes'])
    early=diag.identity_before_mot(c,tag,OUT/'markers');save(dest/'detection_identity.json',early)
    result.update(marker_detection_frames=early['observed'],marker_ambiguous_frames=early['ambiguous'],marker_below_birth_frames=early['below_birth'])
    report,windows=convert(dict(clip=c,tag=tag,role='training_exposed' if base['training_exposed'] else 'descriptive_survey'),cfg,OUT)
    result.update(report)
    for duration in cfg['classification']['window_durations_s']:
        picked=[(x,m) for x,m in windows if m['duration_s']==duration]
        raw=np.stack([x for x,m in picked]) if picked else np.empty((0,round(duration*cfg['classification']['hz']),3),np.float32)
        assert np.isfinite(raw).all() and (raw[:,:,:2]>=0).all() and (raw[:,:,2]>0).all()
        np.savez_compressed(dest/f'windows_{duration:g}s.npz',primitive_windows=raw)
        save(dest/f'windows_{duration:g}s_metadata.json',[dict(m,**{k:base[k] for k in ['model','site','view','core','training_exposed','previously_inspected']}) for x,m in picked])
    if ident['accepted']:
        td=TR/'outputs/trajectories'/f'{c}_{tag}';st=read(td/'tracks.json')['tracks'][str(ident['vehicle_id'])];observed=[s for s in st if 'x_det' in s]
        pairdir=OUT/'pairs'/f'{c}_{tag}';valid=pd.read_csv(pairdir/'target_trajectory.csv') if result.get('target_observations',0) else pd.DataFrame()
        result.update(track_states=len(st),detection_supported_states=len(observed),prediction_only_states=len(st)-len(observed),track_span_s=float(st[-1]['t_s']-st[0]['t_s']),longest_observed_span_s=longest_span(observed),longest_in_image_observed_span_s=longest_span(valid.to_dict('records')))
        diag.OUT=OUT;(OUT/'reviews').mkdir(exist_ok=True);diag.review(c,tag)
        if row['gps_present'] and len(valid):
            try:
                g=diag.gps(c);v,ok=diag.interpolate(g,valid.t_s);speed=np.hypot(v[:,2],v[:,3]);comparison=valid[['frame','t_s','speed_mps']].copy();comparison['gps_speed_mps']=speed;comparison=comparison.loc[ok];comparison['absolute_error_mps']=abs(comparison.speed_mps-comparison.gps_speed_mps);comparison.to_csv(dest/'speed_check.csv',index=False)
                result['gps_points']=len(comparison);result['gps_speed_mae_mps']=float(comparison.absolute_error_mps.mean()) if len(comparison) else None
                result['gps_note']='Supplied time alignment, no fitted shift or scale; candidate trajectory only, timing and geometry unvalidated.'
            except Exception as e:result['gps_error']=str(e)
    return result

def export_all(watch=False):
    cfg=read(CONFIG);cv2.setNumThreads(1)
    while True:
        pending=0;ann=read(ANNOT)
        for row in cfg['clips']:
            for key in row['models']:
                d=OUT/'runs'/f'{row["clip"]}_{key}';p=d/'evaluation.json'
                if p.exists():
                    prior=read(p)
                    if prior['status']=='SOURCE_IDENTITY_UNRESOLVED' and (OUT/'markers'/f'{row["clip"]}_markers.json').exists():save(d/'evaluation_before_annotation.json',prior)
                    else:continue
                if not (d/'tracking.json').exists():pending+=1;continue
                if read(d/'tracking.json')['status']!='COMPLETE':continue
                a=ann['clips'].get(row['clip'],{})
                if (a.get('seed') or a.get('marker_source')) and not (OUT/'markers'/f'{row["clip"]}_markers.json').exists():pending+=1;continue
                try:r=export_one(row,key)
                except Exception as e:r=dict(clip=row['clip'],model=key,status='EVALUATION_FAILED',error=str(e),traceback=traceback.format_exc())
                save(p,r);print('EVALUATED',row['clip'],key,r['status'],r.get('windows_by_duration',{}),flush=True)
        results=[read(p) for p in sorted((OUT/'runs').glob('*/evaluation.json'))];save(OUT/'dataset_summary.json',dict(rows=results,expected_runs=90,complete=len(results)==90))
        flat=[]
        for r in results:
            z={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
            for dur in [1,2,3,4]:z[f'windows_{dur}s']=r.get('windows_by_duration',{}).get(str(float(dur)),0)
            flat.append(z)
        pd.DataFrame(flat).to_csv(OUT/'recording_results.csv',index=False,encoding='utf-8-sig')
        if not watch or not pending:break
        time.sleep(4)

def classify():
    import torch
    from evaluate_field_baseline import TrajectoryNet,features,normalize,metrics,nonoverlap
    torch.set_num_threads(4);cfg=read(CONFIG);members=[];manifest=read(ROOT/'experiments/field_baseline_20261101/model_manifest.json')
    for row in cfg['clips']:
        for key in row['models']:
            r=read(OUT/'runs'/f'{row["clip"]}_{key}'/'evaluation.json')
            assert r['status'] not in ['EVALUATION_FAILED','SOURCE_IDENTITY_UNRESOLVED'], (row['clip'],key,r['status'])
    for r in manifest:
        path=ROOT/r['path'];cp=torch.load(path,map_location='cpu',weights_only=True);c=cp['config'];m=TrajectoryNet(cp['input_dim'],c['architecture'],c['width'],c['dropout'],c['layers']).eval();m.load_state_dict(cp['model_state_dict'],strict=True);members.append((m,c,read(path.parent/'feature_stats.json')))
    save(OUT/'classifier_manifest.json',manifest);results=[]
    for key in cfg['models']:
        folder=OUT/'classification'/key;folder.mkdir(parents=True,exist_ok=True);cohort=[r for r in cfg['clips'] if key in r['models']]
        for dur in [1,2,3,4]:
            arrays=[];meta=[]
            for row in cohort:
                d=OUT/'runs'/f'{row["clip"]}_{key}';p=d/f'windows_{dur}s.npz'
                if not p.exists():continue
                arrays.append(np.load(p)['primitive_windows']);meta+=read(d/f'windows_{dur}s_metadata.json')
            raw=np.concatenate(arrays) if arrays else np.empty((0,dur*10,3),np.float32)
            assert len(raw)==len(meta) and np.isfinite(raw).all();np.savez_compressed(folder/f'windows_{dur}s.npz',primitive_windows=raw,labels=np.array([m['label'] for m in meta]),split_group=np.array([m['clip'] for m in meta],dtype=str),dt=np.array(.1));save(folder/f'windows_{dur}s_metadata.json',meta)
            if not len(raw):results.append(dict(model=key,duration_s=dur,status='NO_WINDOWS'));continue
            for arch in ['transformer','multiscale_cnn']:
                ps=[];clipping=[]
                for m,c,stats in members:
                    if c['architecture']!=arch:continue
                    x=features(raw,c.get('features','legacy11'),c.get('smoothing',7));clipping.append(float(((x<np.array(stats['lower']))|(x>np.array(stats['upper']))).mean()))
                    with torch.no_grad():pr=torch.sigmoid(m(torch.from_numpy(normalize(x,stats)))).numpy()
                    assert np.isfinite(pr).all();ps.append(pr)
                assert len(ps)==15;pred=pd.DataFrame(meta);pred['prob_av']=np.mean(ps,0);pred['model_sd']=np.std(ps,0);pred['predicted_label']=(pred.prob_av>=.5).astype(int);pred.to_csv(folder/f'{arch}_{dur}s_predictions.csv',index=False)
                first=pred.sort_values('start_s').groupby('clip',sort=False).head(1);first.to_csv(folder/f'{arch}_{dur}s_first_per_clip.csv',index=False);fusion=[];matched=[]
                for clip,g in pred.groupby('clip',sort=False):
                    pair=nonoverlap(g).head(2)
                    if len(pair)<2:continue
                    fusion.append(dict(clip=clip,label=int(pair.iloc[0].label),prob_av=float(pair.prob_av.mean())));matched.append(pair.iloc[0].to_dict())
                save(folder/f'{arch}_{dur}s_fusion.json',fusion)
                subsets={}
                for name,mask in [('core',first.core),('supplementary',~first.core),('training_exposed',first.training_exposed),('not_checkpoint_training',~first.training_exposed),('not_previously_in_pipeline',~first.previously_inspected)]:subsets[name]=metrics(first.loc[mask])
                results.append(dict(model=key,architecture=arch,duration_s=dur,windows=len(raw),recording_denominator=len(cohort),recordings_with_window=len(first),first_per_recording=metrics(first),subsets=subsets,fusion_two_nonoverlap=metrics(pd.DataFrame(fusion)),single_on_fusion_cohort=metrics(pd.DataFrame(matched)),all_windows_correlated=metrics(pred),feature_clipping_range=[min(clipping),max(clipping)]))
                print('CLASSIFIED',key,arch,dur,metrics(first),flush=True)
    # Fixed view-based composition, not a per-clip choice based on results:
    # all Verona -> adapted_V, all Seminole -> adapted_S, other views -> original.
    folder=OUT/'classification/view_adapted_mix';folder.mkdir(parents=True,exist_ok=True)
    for dur in [1,2,3,4]:
        for arch in ['transformer','multiscale_cnn']:
            selected=[]
            for row in cfg['clips']:
                key=next((k for k in row['models'] if k!='original'),'original');p=OUT/'classification'/key/f'{arch}_{dur}s_predictions.csv'
                if p.exists():
                    f=pd.read_csv(p);selected.append(f[f['clip']==row['clip']])
            if not selected or not sum(len(s) for s in selected):continue
            pred=pd.concat(selected,ignore_index=True);pred.to_csv(folder/f'{arch}_{dur}s_predictions.csv',index=False);first=pred.sort_values('start_s').groupby('clip',sort=False).head(1);first.to_csv(folder/f'{arch}_{dur}s_first_per_clip.csv',index=False);fusion=[];matched=[]
            for clip,g in pred.groupby('clip',sort=False):
                pair=nonoverlap(g).head(2)
                if len(pair)<2:continue
                fusion.append(dict(clip=clip,label=int(pair.iloc[0].label),prob_av=float(pair.prob_av.mean())));matched.append(pair.iloc[0].to_dict())
            save(folder/f'{arch}_{dur}s_fusion.json',fusion);subsets={}
            for name,mask in [('core',first.core),('supplementary',~first.core),('training_exposed',first.training_exposed),('not_checkpoint_training',~first.training_exposed),('not_previously_in_pipeline',~first.previously_inspected)]:subsets[name]=metrics(first.loc[mask])
            results.append(dict(model='view_adapted_mix',architecture=arch,duration_s=dur,windows=len(pred),recording_denominator=60,recordings_with_window=len(first),first_per_recording=metrics(first),subsets=subsets,fusion_two_nonoverlap=metrics(pd.DataFrame(fusion)),single_on_fusion_cohort=metrics(pd.DataFrame(matched)),all_windows_correlated=metrics(pred)))
    save(OUT/'classification_summary.json',dict(status='DESCRIPTIVE_CANDIDATE_WINDOWS_ONLY',primary_duration_s=2,threshold=.5,results=results,notes=['All 30 source classifier members frozen, no new training or label-dependent exclusions','Original covers all 60 videos; view_adapted_mix always uses adapted_V for Verona, adapted_S for Seminole, original elsewhere, never result-dependent checkpoint selection','First eligible window per recording is the single-segment unit; overlapping windows are correlated','1/2/3 s windows differ from 4 s source training','Current classifier needs a valid leader; missing-leader target tracks retained but not padded','Candidate automatic lanes, gaps, metric geometry and track tails are not independently validated','Previously used training/development recordings remain visible and separately flagged; same physical cars recur','Fusion is mean probability of earliest two non-overlapping windows, not retraining of historical multisegment network']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','classify']);p.add_argument('--watch',action='store_true');a=p.parse_args();export_all(a.watch) if a.stage=='export' else classify()
