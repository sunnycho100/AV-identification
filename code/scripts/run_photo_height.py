"""Photo-assumed 15 m workflow and prespecified 17 m sensitivity, fresh outputs."""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
from verify_field_branches import ROOT, TR, CFG, read, save, calibration, gps, interpolate
from scripts.object_detection import run_bevheight_generic as G
from scripts.finetune import train_finetune as T

def main():
    torch.set_num_threads(4)
    cfg=read(ROOT/'configs/field_photo_height.json');out=ROOT/'experiments'/cfg['experiment'];out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_DISABLE_JIT='1')
    # Produce explicitly named calibration files for every pilot recording.
    for c in sum([CFG[r] for r in ['train','validation','test']],[]):
        d=TR/'outputs/calibration/camera-data'/c;old=read(d/'metric_extrinsic_site.json');R=np.array(old['rotation'])
        for h in [cfg['height_m'],cfg['sensitivity_height_m']]:
            new=dict(old,translation=(-R@np.array([0.,0.,h])).tolist(),note=cfg['height_source'])
            new['diagnostics']=dict(old['diagnostics'],height_m=h,height_source=cfg['height_source'])
            save(d/f'metric_extrinsic_photo_h{h:g}.json',new)
    model=T.build_model();info=T.load_checkpoint(model,TR/cfg['checkpoint']);assert not info['missing'] and not info['unexpected']
    model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG);runs=[];evals=[]
    for h in [cfg['height_m'],cfg['sensitivity_height_m']]:
        tag=f'photo_h{h:g}_excerpt'
        for row in cfg['excerpts']:
            c=row['clip'];K,_=calibration(c);ext=read(TR/'outputs/calibration/camera-data'/c/f'metric_extrinsic_photo_h{h:g}.json')
            M=np.eye(4);M[:3,:3]=ext['rotation'];M[:3,3]=ext['translation'];M=G.to_dair_ground(M)
            d=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}';d.mkdir(parents=True,exist_ok=True)
            start=time.time()
            for f in range(row['start'],row['end']+1):
                p=d/f'{f:03d}_pred.json'
                if p.exists():continue
                pred=G.run_frame(model,TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg',K,M,exp,score_thresh=cfg['score_thresh'])
                save(p,pred)
                if (f-row['start'])%40==0:print(c,h,f,flush=True)
            assert len(list(d.glob('*_pred.json')))==row['end']-row['start']+1
            save(d/'calibration_used.json',dict(K=K.tolist(),lidar2cam=M.tolist(),road_plane_z_in_output=-1.73,checkpoint=cfg['checkpoint'],height_m=h,source=cfg['height_source'],frame_start=row['start'],frame_end=row['end'],scope='contiguous excerpt'))
            cmd=[sys.executable,'scripts/pipeline/extract_trajectories.py','--clip',c,'--tag',tag,'--extrinsic',f'metric_extrinsic_photo_h{h:g}.json','--score-thresh','0.2','--track-low-thresh','0.2']
            with (out/f'{c}_{tag}.log').open('w',encoding='utf-8') as log:
                subprocess.run(cmd,cwd=TR,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'scripts/pipeline/mark_instrumented.py','--clip',c,'--tag',tag],cwd=TR,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            td=TR/'outputs/trajectories'/f'{c}_{tag}';run=read(td/'run.json');identity=read(td/'instrumented.json')
            rec=dict(row,height_m=h,tag=tag,frames=row['end']-row['start']+1,tracks=run['vehicles'],runtime_s=time.time()-start,identity=identity);runs.append(rec)
            if not identity.get('warnings'):
                states=read(td/'tracks.json')['tracks'][str(identity['vehicle_id'])]
                gv,ok=interpolate(gps(c),[s['t_s'] for s in states]);ss=[s for s,k in zip(states,ok) if k];gv=gv[ok]
                df=pd.DataFrame(dict(t_s=[s['t_s'] for s in ss],camera_speed_mps=[s['speed_mps'] for s in ss],gps_speed_mps=np.hypot(gv[:,2],gv[:,3])))
                df.to_csv(out/f'{c}_{tag}_speed.csv',index=False)
                evals.append(dict(clip=c,height_m=h,frames=len(df),mae_mps=float(abs(df.camera_speed_mps-df.gps_speed_mps).mean())))
            save(out/'runs.json',dict(config=cfg,runs=runs,own_span_gps=evals));print('DONE',c,h,run['vehicles'],flush=True)
    common=[]
    for row in cfg['excerpts']:
        c=row['clip'];p=[out/f'{c}_photo_h{h:g}_excerpt_speed.csv' for h in [15.,17.]]
        if not all(x.exists() for x in p):continue
        a,b=map(pd.read_csv,p);d=a.merge(b,on='t_s',suffixes=('_15','_17'))
        np.testing.assert_allclose(d.gps_speed_mps_15,d.gps_speed_mps_17)
        common.append(dict(clip=c,split=row['split'],frames=len(d),span_s=float(d.t_s.max()-d.t_s.min()),mae_15_mps=float(abs(d.camera_speed_mps_15-d.gps_speed_mps_15).mean()),mae_17_mps=float(abs(d.camera_speed_mps_17-d.gps_speed_mps_15).mean())))
        d.to_csv(out/f'{c}_common_speed.csv',index=False)
    save(out/'summary.json',dict(config=cfg,runs=runs,own_span_gps=evals,common_timestamps=common,note='15 m fixed by photo/user assumption before comparison; 17 m sensitivity only. No GPS-selected height, classifier retraining, or new detection/MOT ground truth.'))
    print(json.dumps(common),flush=True)

if __name__=='__main__':main()
