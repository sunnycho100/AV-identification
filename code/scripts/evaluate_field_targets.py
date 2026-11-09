"""Image-only target selection followed by timestamp-aligned GPS diagnostics."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;TR=ROOT/'trajectory'
sys.path.insert(0,str(TR/'scripts/tracking'))
from grade_target_vs_gps import rigid_align,similarity_align
cfg=json.loads((ROOT/'configs/field_pilot.json').read_text());exp=ROOT/'experiments'/cfg['experiment']
catalog=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
records=[]
for tag in ['pilot_base','pilot_ft']:
    for role in ['train','validation','test']:
        for clip in cfg[role]:
            td=TR/'outputs/trajectories'/f'{clip}_{tag}'
            if not (td/'tracks.json').exists():continue
            marker=TR/'outputs/target_id'/clip/'marker_matches.json'
            if not marker.exists():
                records.append(dict(clip=clip,tag=tag,split=role,status='NO_IMAGE_IDENTITY'));continue
            md=json.loads(marker.read_text())
            if not md['target_runs']:
                records.append(dict(clip=clip,tag=tag,split=role,status='NO_ACCEPTED_MARKER_RUN'));continue
            result=subprocess.run([sys.executable,'scripts/pipeline/mark_instrumented.py','--clip',clip,'--tag',tag],cwd=TR,capture_output=True,text=True,env=dict(os.environ,PYTHONUTF8='1'))
            (exp/f'{clip}_{tag}_identity.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                records.append(dict(clip=clip,tag=tag,split=role,status='IDENTITY_MATCH_FAILED'));continue
            identity=json.loads((td/'instrumented.json').read_text())
            if identity.get('warnings'):
                records.append(dict(clip=clip,tag=tag,split=role,status='WEAK_IDENTITY',identity=identity));continue
            tid=str(identity['vehicle_id']);states=json.loads((td/'tracks.json').read_text())['tracks'][tid]
            df=pd.read_csv(ROOT/catalog.loc[clip,'gps_path'])
            cols=['video_time_sec','processed_x','processed_y','processed_vx','processed_vy']
            df=df[cols].dropna().groupby('video_time_sec',as_index=False).mean().sort_values('video_time_sec')
            st=[s for s in states if df.video_time_sec.iloc[0]<=s['t_s']<=df.video_time_sec.iloc[-1]]
            if len(st)<10:continue
            ts=np.array([s['t_s'] for s in st]);gt=df.video_time_sec.to_numpy()
            ix=np.searchsorted(gt,ts).clip(1,len(gt)-1);valid=(gt[ix]-gt[ix-1])<=.1
            st=[s for s,ok in zip(st,valid) if ok];ts=ts[valid]
            if len(st)<10:continue
            A=np.stack([np.interp(ts,gt,df[c]) for c in ['processed_x','processed_y']],axis=1)
            B=np.array([[s['x'],s['y']] for s in st]);v=np.array([s['speed_mps'] for s in st])
            vg=np.hypot(np.interp(ts,gt,df.processed_vx),np.interp(ts,gt,df.processed_vy))
            R,t=rigid_align(A,B);_,_,scale=similarity_align(A,B)
            error=np.linalg.norm(A@R.T+t-B,axis=1)
            row=dict(clip=clip,tag=tag,split=role,status='SCORED',vehicle_id=tid,frames=len(st),span_s=float(ts[-1]-ts[0]),speed_mae_mps=float(abs(v-vg).mean()),speed_bias_mps=float((v-vg).mean()),mean_camera_speed_mps=float(v.mean()),mean_gps_speed_mps=float(vg.mean()),rigid_aligned_rmse_m=float(np.sqrt((error**2).mean())),camera_to_gps_displacement_scale=float(scale),identity=identity)
            pd.DataFrame(dict(t_s=ts,camera_speed_mps=v,gps_speed_mps=vg,rigid_aligned_position_error_m=error)).to_csv(exp/f'{clip}_{tag}_gps_comparison.csv',index=False)
            records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='identity'}))
report=dict(method='Image identity fixed before reading GPS; GPS interpolated to decoded video presentation times, max reference gap 0.1 s; no time-offset search. Rigid alignment and scale fitted on each evaluated trajectory are diagnostics, not absolute georeferencing accuracy.',scale_definition='camera displacement / GPS displacement; 1 is ideal; fitted scale is never applied to the reported camera speed',records=records)
matched=[]
for clip in cfg['validation']+cfg['test']:
    paths=[exp/f'{clip}_{tag}_gps_comparison.csv' for tag in ['pilot_base','pilot_ft']]
    if not all(p.exists() for p in paths):continue
    a,b=map(pd.read_csv,paths);both=a.merge(b,on='t_s',suffixes=('_base','_ft'))
    if len(both)<10:continue
    np.testing.assert_allclose(both.gps_speed_mps_base,both.gps_speed_mps_ft)
    matched.append(dict(clip=clip,frames=len(both),span_s=float(both.t_s.max()-both.t_s.min()),baseline_speed_mae_mps=float(abs(both.camera_speed_mps_base-both.gps_speed_mps_base).mean()),finetuned_speed_mae_mps=float(abs(both.camera_speed_mps_ft-both.gps_speed_mps_base).mean())))
    both.to_csv(exp/f'{clip}_matched_gps_comparison.csv',index=False)
report['matched_speed_comparisons']=matched
(exp/'gps_evaluation.json').write_text(json.dumps(report,indent=2))
