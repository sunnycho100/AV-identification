"""Propagate all eight user seeds with a bounded, reversible image search."""
import sys
from pathlib import Path
import cv2
import numpy as np
from build_field_core_dataset import ROOT, read, save
from trial_manual_leader import iou
from visualize_field_baseline import project
from track_selected_leader import normalized_patch

CONFIG = ROOT/'configs/field_review_test.json'


def patch(gray, box):
    a,b,c,d=np.round(box).astype(int)
    return gray[b:d,a:c]


def search(image, template, center, size, scales):
    radius=max(24.,max(size)*.70)
    a,b=np.maximum(0,np.floor(center-size/2-radius)).astype(int)
    c,d=np.minimum([image.shape[1],image.shape[0]],np.ceil(center+size/2+radius)).astype(int)
    roi=image[b:d,a:c]; candidates=[]
    for scale in scales:
        w,h=np.maximum(10,np.round(size*scale)).astype(int)
        if w>=roi.shape[1] or h>=roi.shape[0]:continue
        small=cv2.resize(template,(w,h))
        response=cv2.matchTemplate(roi,small,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(response)
        other=response.copy();ex=max(5,int(w*.55));ey=max(5,int(h*.55))
        other[max(0,y-ey):y+ey+1,max(0,x-ex):x+ex+1]=-1
        second=float(other.max())
        candidates.append((float(score),float(score-second),np.array([a+x,b+y,a+x+w,b+y+h],float)))
    return max(candidates,key=lambda r:r[0]) if candidates else None


def direction(folder,seed,stop,cfg):
    frame=seed['frame']; box=np.array(seed['bbox'],float)
    prev=cv2.imread(str(folder/f'{frame:03d}.jpg'),0); rows=[]; velocity=np.zeros(2)
    step=1 if stop>frame else -1
    for f in range(frame+step,stop+step,step):
        now=cv2.imread(str(folder/f'{f:03d}.jpg'),0)
        template=patch(prev,box);size=box[2:]-box[:2];center=(box[2:]+box[:2])/2
        best=search(now,template,center+velocity,size,cfg['scales'])
        reason=None
        if best is None:reason='empty_search'
        else:
            score,margin,new=best; nc=(new[2:]+new[:2])/2
            if score<cfg['minimum_correlation']:reason='low_template_correlation'
            elif margin<cfg['minimum_peak_margin']:reason='ambiguous_template_peak'
            elif np.linalg.norm(nc-center)>.65*max(size):reason='large_image_step'
            elif min(new[2:]-new[:2])<14:reason='too_small'
            else:
                back=search(prev,patch(now,new),center,new[2:]-new[:2],cfg['scales'])
                error=float(np.linalg.norm((back[2][:2]+back[2][2:])/2-center)) if back else 1e9
                if not back or back[0]<cfg['minimum_backward_correlation'] or error>cfg['maximum_backward_error_fraction']*min(size)+2:
                    reason='forward_backward_failure'
                else:
                    rows.append(dict(frame=f,bbox=new.tolist(),image_correlation=score,peak_margin=margin,
                                     backward_error_px=error,source='automatic_template_candidate'))
                    velocity=.5*velocity+.5*(nc-center);box=new
        if reason:return rows,dict(frame=f,reason=reason)
        prev=now
    return rows,None


def adaptive_flow_direction(folder,seed,stop):
    frame=seed['frame'];bbox=np.array(seed['bbox'],float)
    prev=cv2.imread(str(folder/f'{frame:03d}.jpg'),0);result=[]
    reference=normalized_patch(prev,bbox).astype(np.float32)
    step=1 if stop>frame else -1
    for f in range(frame+step,stop+step,step):
        now=cv2.imread(str(folder/f'{f:03d}.jpg'),0);mask=np.zeros_like(prev)
        a,b,c,d=np.round(bbox).astype(int);inset=max(2,int(min(c-a,d-b)*.10))
        mask[b+inset:d-inset,a+inset:c-inset]=255
        pts=cv2.goodFeaturesToTrack(prev,100,.01,3,mask=mask);reason=None
        if pts is None or len(pts)<6:reason='too_few_points'
        else:
            nxt,sf,_=cv2.calcOpticalFlowPyrLK(prev,now,pts,None,winSize=(25,25),maxLevel=4)
            back,sb,_=cv2.calcOpticalFlowPyrLK(now,prev,nxt,None,winSize=(25,25),maxLevel=4)
            good=(sf.ravel()==1)&(sb.ravel()==1)&(np.linalg.norm(pts-back,axis=2).ravel()<1.5)
            if good.sum()<6:reason='flow_inconsistency'
            else:
                affine,inliers=cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2)
                if affine is None or inliers.sum()<6 or inliers.mean()<.55:reason='motion_inconsistency'
                else:
                    scale=float(np.hypot(affine[0,0],affine[0,1]));angle=float(np.arctan2(affine[1,0],affine[0,0]))
                    center=(bbox[:2]+bbox[2:])/2;nc=affine[:,:2]@center+affine[:,2]
                    size=(bbox[2:]-bbox[:2])*scale;new=np.r_[nc-size/2,nc+size/2]
                    if not .9<=scale<=1.1 or abs(angle)>.10 or np.linalg.norm(nc-center)>.6*max(size):reason='motion_gate'
                    elif (new[:2]<0).any() or new[2]>=now.shape[1] or new[3]>=now.shape[0] or min(size)<14:reason='outside_or_small'
                    else:
                        appearance=normalized_patch(now,new).astype(np.float32)
                        rolling=float(cv2.matchTemplate(appearance,reference,cv2.TM_CCOEFF_NORMED)[0,0])
                        previous=float(cv2.matchTemplate(appearance,normalized_patch(prev,bbox).astype(np.float32),cv2.TM_CCOEFF_NORMED)[0,0])
                        if rolling<.30 or previous<.45:reason='appearance_changed'
                        else:
                            result.append(dict(frame=f,bbox=new.tolist(),image_correlation=rolling,previous_correlation=previous,
                                               flow_points=int(good.sum()),flow_inliers=int(inliers.sum()),source='adaptive_flow_candidate'))
                            bbox=new;reference=.9*reference+.1*appearance
        if reason:return result,dict(frame=f,reason=reason)
        prev=now
    return result,None


def annotate(rows,clip,tag,root):
    dd=ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}'
    cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
    time=read(ROOT/'trajectory/data/camera-data'/clip/'frame_times.json')
    times={t['frame']:t for t in time['frames']}
    cfg=read(CONFIG)['association']
    for r in rows:
        r['t_s']=times[r['frame']]['t_s'];r['copy']=times[r['frame']].get('copy',False)
        matches=[]
        for j,d in enumerate(read(dd/f'{r["frame"]:03d}_pred.json')):
            if d['class_name'] not in ('car','truck','bus') or d['score']<cfg['minimum_detection_score']:continue
            uv=project(d,K,M)
            if uv is None:continue
            bb=np.r_[uv.min(0),uv.max(0)];value=iou(r['bbox'],bb)
            if value>0:matches.append(dict(index=j,iou=value,bbox=bb.tolist(),detection=d))
        matches.sort(key=lambda m:m['iou'],reverse=True)
        accepted=bool(matches and matches[0]['iou']>=cfg['minimum_iou'] and
                      (len(matches)==1 or matches[0]['iou']-matches[1]['iou']>=cfg['minimum_iou_margin']))
        r['matched_detection']=matches[0] if accepted and not r['copy'] else None
        r['top_candidates']=matches[:2]
    return rows


def contact(rows,folder,path,title):
    indices=sorted(set(np.linspace(0,len(rows)-1,min(9,len(rows))).astype(int)))
    tiles=[]
    for i in indices:
        r=rows[i];im=cv2.imread(str(folder/f'{r["frame"]:03d}.jpg'))
        bb=np.array(r['bbox']);cx,cy=(bb[:2]+bb[2:])/2
        # A 480x270 crop retains enough adjacent roadway to judge identity.
        left=int(np.clip(cx-240,0,im.shape[1]-480));top=int(np.clip(cy-135,0,im.shape[0]-270))
        tile=im[top:top+270,left:left+480].copy();a,b,c,d=np.round(bb-[left,top,left,top]).astype(int)
        cv2.rectangle(tile,(a,b),(c,d),(255,230,0),2)
        m=r.get('matched_detection')
        if m:
            a,b,c,d=np.round(np.array(m['bbox'])-[left,top,left,top]).astype(int)
            cv2.rectangle(tile,(a,b),(c,d),(0,200,0),2)
        tile=cv2.copyMakeBorder(tile,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        status=r.get('association_label',f'3D:{bool(m)}')
        cv2.putText(tile,f'{title} f{r["frame"]} {status}',(5,23),0,.50,(0,0,0),1);tiles.append(tile)
    while len(tiles)%3:tiles.append(np.full_like(tiles[0],255))
    cv2.imwrite(str(path),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))


def main():
    cv2.setNumThreads(2);cfg=read(CONFIG);source=ROOT/cfg['source_review'];out=ROOT/'experiments'/cfg['experiment'];out.mkdir(exist_ok=True)
    save(out/'protocol.json',cfg)
    records={r['clip']:r for r in read(source/'decisions_source.json')['records']};reports=[]
    for s in read(source/'pending_3d_seeds.json'):
        clip,role=s['clip'],s['role'];r=records[clip];seed=r[role]
        folder=ROOT/'trajectory/data/camera-data'/clip/'frames_all'
        backward,bs=direction(folder,seed,r['start_frame'],cfg['image_tracking'])
        forward,fs=direction(folder,seed,r['end_frame'],cfg['image_tracking'])
        rows=sorted(backward+[dict(frame=seed['frame'],bbox=seed['bbox'],source='user_manual_seed',image_correlation=1.)]+forward,key=lambda v:v['frame'])
        dest=out/'seeds'/clip/role;dest.mkdir(parents=True,exist_ok=True)
        rows=annotate(rows,clip,r['tag'],out)
        save(dest/'template_track.json',dict(clip=clip,role=role,seed=seed,backward_stop=bs,forward_stop=fs,rows=rows))
        contact(rows,folder,dest/'template_review.jpg',clip)
        previous=read(source/'image_candidates'/clip/role/'image_track.json')['rows']
        previous=annotate(previous,clip,r['tag'],out)
        save(dest/'flow_track.json',dict(clip=clip,role=role,seed=seed,rows=previous))
        contact(previous,folder,dest/'flow_review.jpg',clip)
        back,ab=adaptive_flow_direction(folder,seed,r['start_frame'])
        forward,af=adaptive_flow_direction(folder,seed,r['end_frame'])
        adaptive=sorted(back+[dict(frame=seed['frame'],bbox=seed['bbox'],source='user_manual_seed',image_correlation=1.)]+forward,key=lambda v:v['frame'])
        adaptive=annotate(adaptive,clip,r['tag'],out)
        save(dest/'adaptive_flow_track.json',dict(clip=clip,role=role,seed=seed,backward_stop=ab,forward_stop=af,rows=adaptive))
        contact(adaptive,folder,dest/'adaptive_flow_review.jpg',clip)
        item=dict(clip=clip,role=role,template_frames=len(rows),template_range=[rows[0]['frame'],rows[-1]['frame']],
                  template_matches=sum(bool(x['matched_detection']) for x in rows),flow_frames=len(previous),flow_matches=sum(bool(x['matched_detection']) for x in previous),backward_stop=bs,forward_stop=fs,
                  adaptive_frames=len(adaptive),adaptive_range=[adaptive[0]['frame'],adaptive[-1]['frame']],adaptive_matches=sum(bool(x['matched_detection']) for x in adaptive),adaptive_backward_stop=ab,adaptive_forward_stop=af)
        reports.append(item);print(item,flush=True)
    save(out/'seed_tracking_summary.json',reports)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
