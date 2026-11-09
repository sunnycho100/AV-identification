"""Frozen BEVHeight baseline on six image-selected field excerpts."""
import argparse,json,os,subprocess,sys,time
import cv2
import numpy as np
import torch
from verify_field_branches import ROOT,TR,read,save
from scripts.object_detection import run_bevheight_generic as G
from scripts.finetune import train_finetune as T

def main():
 torch.set_num_threads(4)
 os.environ.setdefault('TORCH_HOME',str(ROOT/'.runtime/torch-cache'))
 cfg=read(ROOT/'configs/field_baseline.json');out=ROOT/'experiments'/cfg['experiment'];tag=cfg['tag']
 ap=argparse.ArgumentParser();ap.add_argument('--clips',nargs='*');args=ap.parse_args()
 env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_DISABLE_JIT='1')
 model=T.build_model();info=T.load_checkpoint(model,TR/cfg['checkpoint']);assert not info['missing'] and not info['unexpected']
 model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG);runs=read(out/'runs.json')['runs'] if (out/'runs.json').exists() else []
 for row in cfg['excerpts']:
  clip=row['clip'];cal=TR/'outputs/calibration/camera-data'/clip
  if args.clips and clip not in args.clips:continue
  intr=read(cal/'150_anycalib_pinhole_pinhole.json')['prediction']['intrinsics'];fx,fy,cx,cy=intr[:4];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  ext=read(cal/'metric_extrinsic_baseline_h15.json');M=np.eye(4);M[:3,:3]=ext['rotation'];M[:3,3]=ext['translation'];M=G.to_dair_ground(M)
  site=clip.split('_')[1]
  if site in cfg['road_polygons_px']:
   mask=np.zeros((1080,1920),np.uint8);cv2.fillPoly(mask,[np.array(cfg['road_polygons_px'][site],np.int32)],255)
   path=cal/'road_mask.png'
   if path.exists():assert np.array_equal(cv2.imread(str(path),0),mask),'Existing different mask: '+clip
   else:cv2.imwrite(str(path),mask)
   save(cal/'road_mask_provenance.json',dict(method='Manual roadway polygon reviewed from source images before classification',config='configs/field_baseline.json',polygon_px=cfg['road_polygons_px'][site]))
   im=cv2.imread(str(TR/'data/camera-data'/clip/'frames_all'/f"{row['start']:03d}.jpg"));im[mask==0]=(im[mask==0]*.3).astype('uint8');cv2.imwrite(str(out/f'{clip}_road_mask.jpg'),im)
  dst=TR/'outputs/object_detection/camera-data'/f'{clip}_{tag}';dst.mkdir(parents=True,exist_ok=True);start=time.time()
  for f in range(row['start'],row['end']+1):
   p=dst/f'{f:03d}_pred.json'
   if not p.exists():save(p,G.run_frame(model,TR/'data/camera-data'/clip/'frames_all'/f'{f:03d}.jpg',K,M,exp,score_thresh=cfg['score_thresh']))
   if (f-row['start'])%40==0:print('DETECT',clip,f,flush=True)
  assert len(list(dst.glob('*_pred.json')))==row['end']-row['start']+1
  save(dst/'calibration_used.json',dict(K=K.tolist(),lidar2cam=M.tolist(),road_plane_z_in_output=-1.73,checkpoint=cfg['checkpoint'],height_m=cfg['height_m'],frame_start=row['start'],frame_end=row['end'],scope='Source-image selected contiguous excerpt',extrinsic_json=str(cal/'metric_extrinsic_baseline_h15.json')))
  cmd=[sys.executable,'scripts/pipeline/extract_trajectories.py','--clip',clip,'--tag',tag,'--extrinsic','metric_extrinsic_baseline_h15.json','--score-thresh','0.2','--track-low-thresh','0.2']
  with (out/f'{clip}_tracking.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=TR,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  run=read(TR/'outputs/trajectories'/f'{clip}_{tag}'/'run.json');runs=[r for r in runs if r['clip']!=clip];runs.append(dict(row,tag=tag,frames=row['end']-row['start']+1,runtime_s=time.time()-start,tracks=run['vehicles']))
  save(out/'runs.json',dict(config=cfg,runs=runs));print('DONE',clip,run['vehicles'],flush=True)

if __name__=='__main__':main()
