"""Compare frozen original/v1 models on additional image-selected following clips."""
import argparse,hashlib,os,subprocess,sys
import cv2,numpy as np,torch
from verify_field_branches import ROOT,TR,read,save
from scripts.finetune import train_finetune as T
from scripts.object_detection import run_bevheight_generic as G
from identify_baseline_targets import marker_centres,identify,markers
from build_field_core_dataset import convert

CFG=read(ROOT/'configs/seminole_following.json');OUT=ROOT/'experiments'/CFG['experiment'];OUT.mkdir(exist_ok=True)
os.environ.setdefault('TORCH_HOME',str(ROOT/'.runtime/torch-cache'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(CFG['models']));ap.add_argument('--markers-only',action='store_true');args=ap.parse_args();runs=[];torch.set_num_threads(4);cv2.setNumThreads(1)
    for row in CFG['excerpts']:
        c=row['clip']
        if not (OUT.parent/f'{c}_markers.json').exists():markers(c,CFG['marker_seeds'][c],CFG,OUT.parent)
    if args.markers_only:return
    for name,mc in CFG['models'].items():
        if args.model and name!=args.model:continue
        dst=OUT/name;dst.mkdir(exist_ok=True);model=T.build_model();cp=ROOT/mc['checkpoint'];info=T.load_checkpoint(model,cp);assert not info['missing'] and not info['unexpected'];model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG);sha=hashlib.sha256(cp.read_bytes()).hexdigest();reports=[];windows=[]
        for row in CFG['excerpts']:
            c=row['clip'];caldir=TR/'outputs/calibration/camera-data'/c;K=T.load_K_from_anycalib(caldir/'150_anycalib_pinhole_pinhole.json');M=G.to_dair_ground(T.load_extrinsic_json(caldir/'metric_extrinsic_baseline_h15.json'))
            mask=np.zeros((1080,1920),np.uint8);cv2.fillPoly(mask,[np.array(CFG['road_polygon_px'],np.int32)],255);mp=caldir/'road_mask.png'
            if mp.exists():assert np.array_equal(cv2.imread(str(mp),0),mask)
            else:cv2.imwrite(str(mp),mask)
            dd=TR/'outputs/object_detection/camera-data'/f'{c}_{mc["tag"]}';dd.mkdir(exist_ok=True)
            if any(dd.glob('*_pred.json')):assert read(dd/'calibration_used.json')['checkpoint_sha256']==sha
            save(dd/'calibration_used.json',dict(K=K.tolist(),lidar2cam=M.tolist(),road_plane_z_in_output=-1.73,checkpoint=mc['checkpoint'],checkpoint_sha256=sha,height_m=15,frame_start=row['start'],frame_end=row['end'],extrinsic_json=str(caldir/'metric_extrinsic_baseline_h15.json')))
            for f in range(row['start'],row['end']+1):
                p=dd/f'{f:03d}_pred.json'
                if not p.exists():save(p,G.run_frame(model,TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg',K,M,exp,score_thresh=.2))
                if (f-row['start'])%40==0:print('DETECT',name,c,f,flush=True)
            cmd=[sys.executable,'scripts/pipeline/extract_trajectories.py','--clip',c,'--tag',mc['tag'],'--extrinsic','metric_extrinsic_baseline_h15.json','--score-thresh','0.2','--track-low-thresh','0.2']
            with (dst/f'{c}_tracking.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=TR,env=dict(os.environ,NUMBA_DISABLE_JIT='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            centers=marker_centres(OUT.parent/f'{c}_markers.json')[0];identity=identify(c,centers,dict(tag=mc['tag'],marker_seeds=CFG['marker_seeds']),dst)
            report,ww=convert(dict(clip=c,tag=mc['tag'],role='additional_development'),CFG,dst);reports.append(report);windows+=ww
            runs.append(dict(row,model=name,tag=mc['tag'],identity=identity));print('EXPORT',name,c,identity,report,flush=True)
        save(dst/'dataset_summary.json',dict(processed_runs=reports))
        for dur in CFG['classification']['window_durations_s']:
            selected=[(x,m) for x,m in windows if m['duration_s']==dur];raw=np.stack([x for x,_ in selected]) if selected else np.empty((0,int(round(dur*10)),3),np.float32)
            np.savez_compressed(dst/f'windows_{dur:g}s.npz',primitive_windows=raw,labels=np.array([m['label'] for _,m in selected],np.int64),dt=np.array(.1),split_group=np.array([m['clip'] for _,m in selected],dtype=str));save(dst/f'windows_{dur:g}s_metadata.json',[dict(m,window_index=i) for i,(_,m) in enumerate(selected)])
        del model;torch.cuda.empty_cache()
    save(OUT/('runs_'+args.model+'.json' if args.model else 'runs.json'),dict(runs=runs))

if __name__=='__main__':main()
