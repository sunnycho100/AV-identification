"""Prepare native frames and per-view calibration for a frozen baseline run."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;TR=ROOT/'trajectory'
sys.path[:0]=[str(ROOT/'.runtime/modern-deps'),str(TR/'third_party/AnyCalib'),str(TR)]
import cv2
from PIL import Image,ImageDraw

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')

def prepare(cfg,cat,exp):
 import imageio_ffmpeg
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 for clip in cfg['clips']:
  row=cat.loc[clip];data=TR/'data/camera-data'/clip;frames=data/'frames_all'
  if not (data/'frame_times.json').exists():
   frames.mkdir(parents=True,exist_ok=True)
   if any(frames.glob('*.jpg')):raise RuntimeError('Partial existing decode: '+clip)
   cmd=[ffmpeg,'-hide_banner','-i',str(ROOT/row.video_path),'-vf','showinfo','-vsync','0','-start_number','0','-q:v','2',str(frames/'%03d.jpg')]
   run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace');(exp/f'{clip}_decode.log').write_text(run.stderr,encoding='utf-8');run.check_returncode()
   times=[]
   for line in run.stderr.splitlines():
    m=re.search(r'\bn:\s*(\d+)\s+pts:.*?pts_time:([^\s]+)',line)
    if m:times.append((int(m.group(1)),float(m.group(2))))
   assert len(times)==len(list(frames.glob('*.jpg'))) and len(times)>1
   cap=cv2.VideoCapture(str(ROOT/row.video_path));fps=cap.get(cv2.CAP_PROP_FPS);cap.release();t0=times[0][1]
   save(data/'frame_times.json',dict(clip=clip,source=row.video_path,fps_nominal=fps,first_pts_s=t0,method='Native decoded frame indices and ffmpeg PTS; no duplicate insertion',frames=[dict(frame=i,src=i,t_s=round(t-t0,6),copy=False) for i,t in times]))
  ff=sorted(frames.glob('*.jpg'),key=lambda p:int(p.stem));tiles=[]
  # A compact 3x3 view for image-only target and road review.
  selected=np.unique(np.linspace(0,len(ff)-1,9).astype(int))
  for ix in selected:
   im=Image.open(ff[ix]).convert('RGB');im.thumbnail((640,360));tile=Image.new('RGB',(640,390),'white');tile.paste(im,(0,30));ImageDraw.Draw(tile).text((8,8),f'{clip} frame {ff[ix].stem}',fill='black');tiles.append(tile)
  canvas=Image.new('RGB',(1920,390*((len(tiles)+2)//3)),'white')
  for i,im in enumerate(tiles):canvas.paste(im,((i%3)*640,(i//3)*390))
  canvas.save(exp/f'{clip}_review.jpg');print('PREPARED',clip,len(ff),flush=True)

def calibrate(cfg,cat,exp):
 import torch
 from anycalib import AnyCalib
 from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
 from scripts.calibration.vp_extrinsic_from_frame import solve_pose
 torch.set_num_threads(4);os.environ['TORCH_HOME']=str(ROOT/'.runtime/torch-cache')
 model=None;report=[]
 for clip in cfg['clips']:
  cal=TR/'outputs/calibration/camera-data'/clip;cal.mkdir(parents=True,exist_ok=True)
  path=cal/'metric_extrinsic_baseline_h15.json'
  if path.exists():report.append(dict(clip=clip,**read(path)['diagnostics']));continue
  ff=sorted((TR/'data/camera-data'/clip/'frames_all').glob('*.jpg'),key=lambda p:int(p.stem));intrinsic=[]
  old=cal/'150_anycalib_pinhole_pinhole.json'
  if old.exists():
   j=read(old);med=np.array(j['prediction']['intrinsics']);intrinsic=j.get('per_frame',[])
  else:
   if model is None:model=AnyCalib(model_id='anycalib_pinhole').cpu().eval()
   for frac in [.2,.5,.8]:
    p=ff[min(int(frac*len(ff)),len(ff)-1)];rgb=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB);tensor=torch.from_numpy(rgb).float().permute(2,0,1)/255
    with torch.no_grad():res=model.predict(tensor,cam_id='pinhole')
    v=res['intrinsics'];v=v[0] if isinstance(v,list) else v
    intrinsic.append(dict(frame=int(p.stem),intrinsics=v.detach().cpu().numpy().reshape(-1).tolist()))
   med=np.median([r['intrinsics'] for r in intrinsic],axis=0);save(old,dict(prediction=dict(intrinsics=med.tolist()),per_frame=intrinsic,method='Median of three image-only AnyCalib estimates'))
  fx,fy,cx,cy=med[:4];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  vps=[]
  for ix in np.linspace(0,len(ff)-1,15).astype(int):
   vp,lines,diag=lsd_vanishing_point(cv2.imread(str(ff[ix])))
   if vp is not None:vps.append(dict(frame=int(ff[ix].stem),vp=np.asarray(vp).tolist(),diagnostics=diag))
  if len(vps)<3:raise RuntimeError('Not enough vanishing point estimates: '+clip)
  vp=np.median([r['vp'] for r in vps],axis=0);R,t,pitch,yaw=solve_pose(K,vp,cfg['height_m'])
  diag=dict(height_m=cfg['height_m'],vp_px=vp.tolist(),pitch_deg=pitch,yaw_deg=yaw,roll_deg=0,frames_used=len(vps),intrinsics=med.tolist())
  save(path,dict(rotation=R.tolist(),translation=t.tolist(),diagnostics=diag,note='Image-derived intrinsics and lane direction; fixed height setting; no GPS fit.'))
  save(cal/'baseline_vp_samples.json',vps);report.append(dict(clip=clip,**diag));print('CALIBRATED',clip,json.dumps(diag),flush=True)
 save(exp/'calibration.json',report)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','calibrate','both'],default='both');a=ap.parse_args()
 cfg=read(ROOT/'configs/field_baseline.json');exp=ROOT/'experiments'/cfg['experiment'];exp.mkdir(parents=True,exist_ok=True);cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
 if a.stage in ['prepare','both']:prepare(cfg,cat,exp)
 if a.stage in ['calibrate','both']:calibrate(cfg,cat,exp)
