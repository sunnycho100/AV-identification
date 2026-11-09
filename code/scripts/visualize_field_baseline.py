"""Actual raw detections versus current MOT/RTS; diagnostics, not MOT ground truth."""
import argparse,shutil,sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_field_core_dataset import ROOT,read,save,recover_dimensions,corners8
sys.path[:0]=[str(ROOT/'trajectory/scripts/tracking'),str(ROOT/'trajectory/scripts/evaluation')]
import postprocess_tracks as pp
from verify_field_branches import gps,interpolate
TR=ROOT/'trajectory'
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def project(b,K,M):
 xyz=corners8([b['l'],b['w'],b['h']],b['yaw'],[b['x'],b['y'],b['z']]);cam=xyz@M[:3,:3].T+M[:3,3]
 if (cam[:,2]<=0).any():return None
 q=cam@K.T;return q[:,:2]/q[:,2:]

def draw(im,b,K,M,color,label):
 uv=project(b,K,M)
 if uv is None:return
 pts=uv.round().astype(int)
 for i,j in EDGES:cv2.line(im,tuple(pts[i]),tuple(pts[j]),color,2,cv2.LINE_AA)
 u,v=pts.min(0);cv2.putText(im,label,(max(0,u),max(20,v-8)),0,.65,color,2)

def midpoint_residual(t,xy):
 if len(t)<3:return None
 dt1=t[1:-1]-t[:-2];dt2=t[2:]-t[1:-1];ok=(dt1>0)&(dt2>0)&(dt1<=.2)&(dt2<=.2)
 expected=(xy[:-2]*dt2[:,None]+xy[2:]*dt1[:,None])/(dt1+dt2)[:,None]
 return float(np.median(np.linalg.norm((xy[1:-1]-expected)[ok],axis=1))) if ok.any() else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--refine-yaw',action='store_true');ap.add_argument('--config',default='configs/field_baseline.json');args=ap.parse_args()
 cfg=read(ROOT/args.config);out=ROOT/'experiments'/cfg['experiment'];figs=out/('figures_refined_yaw' if args.refine_yaw else 'figures');figs.mkdir(exist_ok=True);results=[]
 if 'clips' not in cfg:cfg['clips']=sum(cfg['development'].values(),[])
 for clip in cfg['clips']:
  td=TR/'outputs/trajectories'/f"{clip}_{cfg['tag']}";ident=read(td/'instrumented.json')
  if not ident['accepted']:continue
  tid=ident['vehicle_id'];tracks=read(td/'tracks.json')['tracks'];target=tracks[tid];dd=TR/'outputs/object_detection/camera-data'/f"{clip}_{cfg['tag']}_road";cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);dims=recover_dimensions(tracks,dd)
  yaw_summary=None
  if args.refine_yaw:
   import refine_yaw
   dst=out/'refined_yaw'/clip/'tracks.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(td/'tracks.json',dst);tracks=refine_yaw.refine(dst,dd,clip)['tracks'];target=tracks[tid]
   changes=[abs(np.arctan2(np.sin(s['yaw_refined']-s['yaw']),np.cos(s['yaw_refined']-s['yaw']))) for s in target if s['yaw_refined'] is not None]
   yaw_summary=dict(states=len(target),refined=len(changes),filled=int(sum(s['yaw_filled'] for s in target)),changed_over_30deg=int(sum(v>np.pi/6 for v in changes)),note='Change magnitude only, not error against independent yaw truth; no position, speed or gap changed')
  dets={int(p.stem.split('_')[0]):read(p) for p in dd.glob('*_pred.json')};assigned=pp.assign_detections(tracks,{k:[b for b in v if b['class_name']=='car'] for k,v in dets.items()});byid={k:{s['frame']:s for s in v} for k,v in tracks.items()};table=pd.read_csv(td/f'vehicle_{tid}.csv',dtype={'lead_id':str}).set_index('frame')
  rows=[]
  for s in target:
   b=assigned[tid].get(s['frame'])
   if b is not None:rows.append(dict(frame=s['frame'],t_s=s['t_s'],raw_x=b['x'],raw_y=b['y'],smooth_x=s['x'],smooth_y=s['y'],smooth_speed=s['speed_mps']))
  df=pd.DataFrame(rows);t=df.t_s.to_numpy();rawxy=df[['raw_x','raw_y']].to_numpy();smoothxy=df[['smooth_x','smooth_y']].to_numpy();vraw=np.gradient(rawxy,t,axis=0);df['raw_finite_difference_speed']=np.linalg.norm(vraw,axis=1)
  # Do not differentiate across long unmatched gaps.
  gap=np.diff(t)>.2;invalid=np.r_[gap,False]|np.r_[False,gap];df.loc[invalid,'raw_finite_difference_speed']=np.nan
  gps_mae=None;gps_note='GPS unavailable'
  try:
   gv,ok=interpolate(gps(clip),t);gspd=np.hypot(gv[:,2],gv[:,3]);df['gps_speed_supplied_time']=np.where(ok,gspd,np.nan)
   if ok.any():gps_mae=float(np.abs(df.smooth_speed.to_numpy()[ok]-gspd[ok]).mean())
   gps_note='Diagnostic using supplied video_time_sec; synchronization and absolute coordinate registration are not independently validated'
  except (KeyError,FileNotFoundError,TypeError):pass
  df.to_csv(out/f'{clip}_raw_vs_stabilized.csv',index=False)
  fig,ax=plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
  for j,key in enumerate(['x','y']):
   origin=df[f'raw_{key}'].iloc[0];ax[j].plot(t,df[f'raw_{key}']-origin,'.-',lw=.7,ms=2,color='#d8750d',label='Raw detector (one-to-one association)');ax[j].plot(t,df[f'smooth_{key}']-origin,lw=2,color='#008966',label='MOT + RTS');ax[j].set_ylabel(('Along-road' if j==0 else 'Lateral')+' change (m)');ax[j].grid(alpha=.2)
  ax[2].plot(t,df.raw_finite_difference_speed,color='#d8750d',lw=.8,label='Raw finite difference');ax[2].plot(t,df.smooth_speed,color='#008966',lw=2,label='MOT + RTS speed')
  if 'gps_speed_supplied_time' in df:ax[2].plot(t,df.gps_speed_supplied_time,'--',color='#535bac',label='GPS, supplied time alignment')
  ax[2].set_ylabel('Speed (m/s)');ax[2].set_xlabel('Source video time (s)');ax[2].grid(alpha=.2);ax[0].legend(fontsize=8);ax[2].legend(fontsize=8);fig.suptitle(f'{clip}: actual detections and stabilized target trajectory\nSmoothness is not an independent position-accuracy measure',fontsize=11);fig.savefig(figs/f'{clip}_trajectory.png',dpi=160);plt.close(fig)
  selected=set(df.frame.iloc[np.unique(np.linspace(0,len(df)-1,3).astype(int))].astype(int));tiles_raw=[];tiles_final=[];video=cv2.VideoWriter(str(figs/f'{clip}_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,580));assert video.isOpened()
  for s in target:
   f=s['frame'];im=cv2.imread(str(TR/'data/camera-data'/clip/'frames_all'/f'{f:03d}.jpg'));a=im.copy();b=im.copy();lead=str(table.loc[f,'lead_id']);roles=[(tid,'target')]+([(lead,'leader')] if lead in byid and f in byid[lead] else [])
   for oid,role in roles:
    det=assigned.get(oid,{}).get(f)
    if det is not None:draw(a,det,K,M,(0,160,255),role+' raw')
    if oid in dims:
     st=byid[oid][f];di=dims[oid];box=dict(st,l=di['length_m'],w=di['width_m'],h=di['height_m'])
     if args.refine_yaw and st.get('yaw_refined') is not None:box['yaw']=st['yaw_refined']
     draw(b,box,K,M,(60,220,40) if role=='target' else (230,220,0),f'{role} ID {oid}')
   caption='MOT + RTS / track median size; '+('refined yaw' if args.refine_yaw else 'original yaw')
   canvas=np.full((580,1920,3),245,np.uint8);canvas[40:,:960]=cv2.resize(a,(960,540));canvas[40:,960:]=cv2.resize(b,(960,540));cv2.putText(canvas,f'{clip} f{f} t={s["t_s"]:.3f}s | RAW DETECTION',(15,27),0,.65,(40,40,40),2);cv2.putText(canvas,caption,(975,27),0,.65,(40,40,40),2);video.write(canvas)
   if f in selected:
    dim=dims[tid];uv=project(dict(s,l=dim['length_m'],w=dim['width_m'],h=dim['height_m']),K,M);u,v=uv.mean(0);left=max(0,min(1920-640,int(u-320)));top=max(0,min(1080-480,int(v-240)))
    for source,collection,title in [(a,tiles_raw,'Raw detector'),(b,tiles_final,'MOT + RTS'+(' + yaw fix' if args.refine_yaw else ''))]:
     tile=np.full((515,640,3),245,np.uint8);tile[35:]=source[top:top+480,left:left+640];cv2.putText(tile,f'{title} | frame {f} | {s["t_s"]:.2f}s',(10,24),0,.65,(40,40,40),2);collection.append(tile)
  video.release();cv2.imwrite(str(figs/f'{clip}_boxes.jpg'),np.vstack([np.hstack(tiles_raw),np.hstack(tiles_final)]))
  result=dict(clip=clip,target_id=tid,track_states=len(target),track_span_s=target[-1]['t_s']-target[0]['t_s'],raw_associations=len(df),raw_midpoint_jitter_m=midpoint_residual(t,rawxy),stabilized_midpoint_jitter_m=midpoint_residual(t,smoothxy),raw_to_stabilized_position_shift_median_m=float(np.median(np.linalg.norm(rawxy-smoothxy,axis=1))),gps_speed_mae_supplied_alignment_mps=gps_mae,gps_note=gps_note,yaw_refinement=yaw_summary,interpretation='Within-track diagnostic, no independent 3D/MOT annotations. Median box size is an explicit visualization stabilization. Does not isolate BEVHeight architectural changes.');results.append(result);print('FIGURES',clip,result,flush=True)
 save(out/('trajectory_diagnostics_refined_yaw.json' if args.refine_yaw else 'trajectory_diagnostics.json'),results)

if __name__=='__main__':main()
