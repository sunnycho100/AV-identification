"""Image-reviewed seed, local hood template, then projected-box identity votes.

Semi-automatic identity annotation only; no GPS or AV predictions are read.
The original vehicle reference photos define the roof apparatus and hood mark.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from build_field_core_dataset import ROOT,read,save,recover_dimensions,corners8
sys.path.insert(0,str(ROOT/'trajectory/scripts/tracking'))
from match_marker_template import coherent_runs
from identify_target_by_marker import marker_centres
TR=ROOT/'trajectory'

def markers(clip,seed,cfg,out):
 frames=TR/'data/camera-data'/clip/'frames_all';f0=seed['frame'];x0,y0,x1,y1=seed['box']
 template=cv2.imread(str(frames/f'{f0:03d}.jpg'))[y0:y1,x0:x1];cv2.imwrite(str(out/f'{clip}_hood_template.png'),template)
 row=next(r for r in cfg['excerpts'] if r['clip']==clip);pf={};cv2.setNumThreads(1)
 rx0=max(0,x0-550);ry0=max(180,y0-400);rx1=min(1920,x1+550);ry1=min(1080,y1+400)
 for f in range(max(row['start'],f0-25),min(row['end'],f0+15)+1):
  im=cv2.imread(str(frames/f'{f:03d}.jpg'))[ry0:ry1,rx0:rx1];best=None
  for scale in [.55,.65,.75,.85,.95,1.,1.1,1.2,1.35,1.5,1.7]:
   t=cv2.resize(template,None,fx=scale,fy=scale);res=cv2.matchTemplate(im,t,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(res)
   if best is None or score>best['score']:best=dict(score=float(score),x=loc[0]+rx0,y=loc[1]+ry0,w=t.shape[1],h=t.shape[0],scale=scale)
  pf[f]=best
 runs=[r for r in coherent_runs(pf) if r[0][0]<=f0<=r[-1][0]]
 doc=dict(score_min=.8,seed=seed,method='Image-reviewed local template; same-clip identity annotation, not an independent detection benchmark',target_runs=[dict(start=r[0][0],end=r[-1][0],frames=len(r),peak_score=max(v['score'] for _,v in r)) for r in runs],per_frame={str(k):v for k,v in pf.items()})
 path=out/f'{clip}_markers.json';save(path,doc)
 centers,_=marker_centres(path);tiles=[]
 for f in np.linspace(min(centers),max(centers),5).astype(int) if centers else [f0]:
  f=min(centers,key=lambda k:abs(k-f)) if centers else f0;rec=pf[f];im=cv2.imread(str(frames/f'{f:03d}.jpg'));u,v=int(rec['x']+rec['w']/2),int(rec['y']+rec['h']/2)
  cv2.rectangle(im,(rec['x'],rec['y']),(rec['x']+rec['w'],rec['y']+rec['h']),(0,255,255),2)
  left=max(0,min(1920-360,u-180));top=max(0,min(1080-280,v-180));tile=im[top:top+280,left:left+360].copy();cv2.putText(tile,f'{clip} f{f} {rec["score"]:.2f}',(5,20),0,.5,(0,255,255),1);tiles.append(tile)
 cv2.imwrite(str(out/f'{clip}_identity_review.jpg'),np.hstack(tiles));return centers

def identify(clip,centers,cfg,out):
 tag=cfg['tag'];td=TR/'outputs/trajectories'/f'{clip}_{tag}';dd=TR/'outputs/object_detection/camera-data'/f'{clip}_{tag}_road'
 if not (td/'tracks.json').exists():return dict(clip=clip,status='TRACKS_PENDING')
 tracks=read(td/'tracks.json')['tracks'];dims=recover_dimensions(tracks,dd);cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);votes=Counter();evidence=[]
 for tid,states in tracks.items():
  if tid not in dims:continue
  dim=dims[tid]
  for s in states:
   if s['frame'] not in centers:continue
   xyz=corners8([dim['length_m'],dim['width_m'],dim['height_m']],s['yaw'],[s['x'],s['y'],s['z']]);cam=xyz@M[:3,:3].T+M[:3,3]
   if (cam[:,2]<=0).any():continue
   q=cam@K.T;uv=q[:,:2]/q[:,2:];dist=cv2.pointPolygonTest(cv2.convexHull(uv.astype(np.float32)),tuple(map(float,centers[s['frame']])),True)
   if dist>=-25:votes[tid]+=1;evidence.append(dict(frame=s['frame'],track=tid,margin_px=float(dist)))
 ranked=votes.most_common();best,n=ranked[0] if ranked else (None,0);runner=ranked[1][1] if len(ranked)>1 else 0
 warnings=[]
 if n<max(3,len(centers)*.5):warnings.append('Too few marker containment votes')
 if runner and n<runner*2:warnings.append('Ambiguous runner-up')
 doc=dict(clip=clip,vehicle_id=best,method='Source-image marker review and template votes' if clip in cfg['marker_seeds'] else 'Existing reference-hood template votes',identified_from='image only; GPS and classifier predictions not used',votes=n,marker_frames=len(centers),runner_up_votes=runner,accepted=not warnings,warnings=warnings,ranked_votes=ranked)
 save(td/'instrumented.json',doc);save(out/f'{clip}_identity_evidence.json',dict(identity=doc,evidence=evidence));return doc

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--markers-only',action='store_true');a=ap.parse_args();cfg=read(ROOT/'configs/field_baseline.json');out=ROOT/'experiments'/cfg['experiment'];results=[]
 for clip in cfg['clips']:
  path=out/f'{clip}_markers.json'
  if clip in cfg['marker_seeds']:
   centers=marker_centres(path)[0] if path.exists() else markers(clip,cfg['marker_seeds'][clip],cfg,out)
  else:centers=marker_centres(TR/'outputs/target_id'/clip/'marker_matches.json')[0]
  if not a.markers_only:results.append(identify(clip,centers,cfg,out))
  print(clip,'markers',len(centers),results[-1] if results and not a.markers_only else '',flush=True)
 if not a.markers_only:save(out/'identities.json',results)

if __name__=='__main__':main()
