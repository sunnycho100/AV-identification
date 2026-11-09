"""All-recording field survey. Preparation and inference can run concurrently.

No training or classifier-driven sample selection. Checkpoints and settings are
frozen before the survey. Source videos and prior experiment artifacts stay intact.
"""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;TR=ROOT/'trajectory'
OUT=ROOT/'experiments/field_full_survey_20261101';CONFIG=ROOT/'configs/field_full_survey.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').fillna('');core=read(ROOT/'configs/field_core.json')['core_views'];allowed={(v['site'],v['view']) for v in core}
    old=read(ROOT/'configs/field_pilot.json');v=read(ROOT/'configs/view_refinement.json');s=read(ROOT/'configs/view_adaptation.json')
    seen=set(old['train']+old['validation']+old['test']+old['excluded_legacy_training'])
    for f in ['field_baseline.json','view_adaptation.json','view_refinement.json','seminole_following.json']:
        c=read(ROOT/'configs'/f);seen.update(c.get('clips',[]));seen.update(sum(c.get('train',{}).values(),[]) if isinstance(c.get('train'),dict) else []);seen.update(sum(c.get('development',{}).values(),[]))
    models={
      'original':dict(checkpoint='trajectory/checkpoints/ft102_run3_v2labels_ep5.ckpt',tag='survey_original_h15',training_clips=old['excluded_legacy_training'],cache_tags=['field_baseline_h15','follow_base_h15']),
      'adapted_V':dict(checkpoint='experiments/view_refinement_20261101/V_shared_class_ep5.ckpt',tag='survey_verona_h15',training_clips=old['excluded_legacy_training']+v['train']['V'],cache_tags=['view_refine_h15']),
      'adapted_S':dict(checkpoint='experiments/view_adaptation_20261101/S_car_presence_ep3.ckpt',tag='survey_seminole_h15',training_clips=old['excluded_legacy_training']+s['train']['S'],cache_tags=['view_adapt_h15','follow_adapt_h15'])}
    for m in models.values():m['sha256']=sha(ROOT/m['checkpoint'])
    clips=[]
    for r in cat.to_dict('records'):
        site=r['clip_id'].split('_')[1];keys=['original']+(['adapted_'+site] if site in ['V','S'] else [])
        clips.append(dict(clip=r['clip_id'],site=r['site'],view=r['view'],core=(r['site'],r['view']) in allowed,video_path=r['video_path'],video_sha256=r['video_sha256'],label=int(r['clip_id'].startswith('AV_')),models=keys,previously_inspected=r['clip_id'] in seen,quality_note=r['quality_note'],gps_present=r['gps_status']=='present'))
    cfg=dict(experiment=OUT.name,height_m=15,models=models,clips=clips,score_thresh=.2,tracker=dict(birth_thresh=.45,low_thresh=.2,max_age=6,fps=30,q_vel=.1,px_noise=1,max_m_per_px=1),classification=dict(hz=10.,window_durations_s=[1.,2.,3.,4.],primary_duration_s=2.,stride_s=1.,max_interpolation_gap_s=.2,speed_floor_mps=None),protocol='All 60 supplied short videos; full native frame scan. Main 43 core views and 17 supplementary clips reported separately. Frozen original model on all, existing adaptation on all V/S. No new training. Candidate trajectory/classification diagnostics, not independent 3D or held-out generalization accuracy. Only image-confirmed target and immediate leader exported. All failures retained in denominator.')
    if CONFIG.exists():assert read(CONFIG)==cfg,'Configuration frozen; use a new experiment for changes'
    else:save(CONFIG,cfg)
    save(OUT/'scope.json',dict(recordings=len(clips),core=sum(c['core'] for c in clips),model_recording_runs=sum(len(c['models']) for c in clips),models=models))
    print('INITIALIZED',len(clips),sum(len(c['models']) for c in clips),flush=True)

def prepare_all():
    from prepare_baseline_videos import prepare,calibrate
    cfg=read(CONFIG);cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id');results=[]
    for r in cfg['clips']:
        c=r['clip'];dest=OUT/'preparation'/c;dest.mkdir(parents=True,exist_ok=True)
        try:
            prepare(dict(clips=[c]),cat,dest);calibrate(dict(clips=[c],height_m=cfg['height_m']),cat,dest)
            ft=read(TR/'data/camera-data'/c/'frame_times.json');ff=list((TR/'data/camera-data'/c/'frames_all').glob('*.jpg'));assert len(ff)==len(ft['frames'])
            result=dict(clip=c,status='READY',frames=len(ff),time_method=ft.get('method','existing extracted frames with per-frame times'),time_end_s=ft['frames'][-1]['t_s'])
        except Exception as e:result=dict(clip=c,status='FAILED',error=str(e),traceback=traceback.format_exc())
        save(dest/'status.json',result);results.append(result);save(OUT/'preparation_summary.json',dict(rows=results,complete=len(results)==len(cfg['clips'])));print('PREPARE_RESULT',result,flush=True)

def inference(only_model=None):
    import cv2,torch
    from verify_field_branches import ROOT as _root
    from scripts.finetune import train_finetune as T
    from scripts.object_detection import run_bevheight_generic as G
    cfg=read(CONFIG);torch.set_num_threads(4);cv2.setNumThreads(1);os.environ.setdefault('TORCH_HOME',str(ROOT/'.runtime/torch-cache'));runs=[]
    # Independent model worker has its own progress/summary files. Per-run outputs
    # are shared; launch only for models not concurrently handled by another worker.
    def logpath(stem):return OUT/(stem+('_'+only_model if only_model else '')+'.json')
    if only_model:cfg['models']={only_model:cfg['models'][only_model]}
    for key,mc in cfg['models'].items():
        cp=ROOT/mc['checkpoint'];assert sha(cp)==mc['sha256'];model=T.build_model();info=T.load_checkpoint(model,cp);assert not info['missing'] and not info['unexpected'];model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG)
        pending=[r for r in cfg['clips'] if key in r['models']]
        while pending:
            ready=[r for r in pending if (OUT/'preparation'/r['clip']/'status.json').exists()]
            if not ready:time.sleep(2);continue
            r=ready[0];pending.remove(r);c=r['clip'];prepared=read(OUT/'preparation'/c/'status.json');dest=OUT/'runs'/f'{c}_{key}';dest.mkdir(parents=True,exist_ok=True)
            if prepared['status']!='READY':
                result=dict(clip=c,model=key,status='PREPARATION_FAILED',error=prepared['error']);save(dest/'inference.json',result);runs.append(result);continue
            previous=dest/'inference.json'
            if previous.exists() and read(previous)['status']=='COMPLETE':runs.append(read(previous));continue
            start=time.time();reused=0;computed=0
            try:
                caldir=TR/'outputs/calibration/camera-data'/c;K=T.load_K_from_anycalib(caldir/'150_anycalib_pinhole_pinhole.json');M=G.to_dair_ground(T.load_extrinsic_json(caldir/'metric_extrinsic_baseline_h15.json'))
                dd=TR/'outputs/object_detection/camera-data'/f'{c}_{mc["tag"]}';dd.mkdir(parents=True,exist_ok=True)
                if (dd/'calibration_used.json').exists():
                    old=read(dd/'calibration_used.json');assert old['checkpoint_sha256']==mc['sha256'] and np.array_equal(old['K'],K) and np.array_equal(old['lidar2cam'],M)
                save(dd/'calibration_used.json',dict(K=K.tolist(),lidar2cam=M.tolist(),road_plane_z_in_output=-1.73,checkpoint=mc['checkpoint'],checkpoint_sha256=mc['sha256'],height_m=15,scope='All supplied native clip frames; later identity/visibility filtering',survey_config=str(CONFIG.relative_to(ROOT))))
                caches=[]
                for tag in mc['cache_tags']:
                    p=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}'
                    if not (p/'calibration_used.json').exists():continue
                    old=read(p/'calibration_used.json');oldcp=Path(old.get('checkpoint',''))
                    oldcp=oldcp if oldcp.is_absolute() else ROOT/oldcp
                    if not oldcp.exists():oldcp=TR/old.get('checkpoint','')
                    if oldcp.exists() and sha(oldcp)==mc['sha256'] and np.array_equal(old['K'],K) and np.array_equal(old['lidar2cam'],M):caches.append(p)
                frames=sorted((TR/'data/camera-data'/c/'frames_all').glob('*.jpg'),key=lambda p:int(p.stem))
                for fp in frames:
                    f=int(fp.stem);p=dd/f'{f:03d}_pred.json'
                    if not p.exists():
                        cached=next((d/p.name for d in caches if (d/p.name).exists()),None)
                        if cached is not None:save(p,read(cached));reused+=1
                        else:save(p,G.run_frame(model,fp,K,M,exp,score_thresh=cfg['score_thresh']));computed+=1
                    if f%100==0:
                        progress=dict(model=key,clip=c,frame=f,total_frames=len(frames),completed_runs=len(runs),pending_this_model=len(pending),elapsed_clip_s=round(time.time()-start,1));save(logpath('progress'),progress);print('PROGRESS',progress,flush=True)
                assert len(list(dd.glob('*_pred.json')))==len(frames)
                result=dict(clip=c,model=key,tag=mc['tag'],status='COMPLETE',frames=len(frames),computed= computed,reused=reused,elapsed_s=time.time()-start,checkpoint_sha256=mc['sha256'])
            except Exception as e:result=dict(clip=c,model=key,status='FAILED',error=str(e),traceback=traceback.format_exc())
            save(previous,result);runs.append(result);save(logpath('inference_summary'),dict(runs=runs,complete=False));print('INFERENCE_RESULT',result,flush=True)
        del model;torch.cuda.empty_cache()
    save(logpath('inference_summary'),dict(runs=runs,complete=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['initialize','prepare','infer']);ap.add_argument('--model',choices=['original','adapted_V','adapted_S']);args=ap.parse_args();{'initialize':initialize,'prepare':prepare_all,'infer':lambda:inference(args.model)}[args.stage]()
