"""Re-exercise migrated research branches on the frozen field-pilot split.

Writes separate artifacts; never replaces pilot detections, calibration, or labels.
CPU stages call the original methods, with explicit new-data paths and bounded
frame sampling. These are integration checks, not independent accuracy labels.
"""
import argparse, json, shutil, sys, traceback
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / 'trajectory'
OUT = ROOT / 'experiments/field_branches_20261101'
CFG = json.loads((ROOT / 'configs/field_pilot.json').read_text())
sys.path[:0] = [str(TR), str(TR/'scripts/tracking'), str(TR/'scripts/evaluation'),
               str(TR/'scripts/object_detection')]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf-8')
def detdir(c): return TR/'outputs/object_detection/camera-data'/f'{c}_pilot_base'
def trackpath(c): return TR/'outputs/trajectories'/f'{c}_pilot_base'/'tracks.json'
def frames(c): return sorted((TR/'data/camera-data'/c/'frames_all').glob('*.jpg'))
def sample(a, n): return [a[i] for i in np.unique(np.linspace(0, len(a)-1, min(n,len(a))).astype(int))]
def calibration(c):
    d = read(detdir(c)/'calibration_used.json')
    return np.array(d['K']), np.array(d['lidar2cam'])
def target(c):
    tid = str(read(trackpath(c).parent/'instrumented.json')['vehicle_id'])
    return tid, read(trackpath(c))['tracks'][tid]
def gps(c):
    catalog = pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    cols = ['video_time_sec','processed_x','processed_y','processed_vx','processed_vy']
    return pd.read_csv(ROOT/catalog.loc[c,'gps_path'])[cols].dropna().groupby('video_time_sec',as_index=False).mean().sort_values('video_time_sec')
def interpolate(g, t):
    ts = g.video_time_sec.to_numpy(); t = np.asarray(t)
    ix = np.searchsorted(ts,t).clip(1,len(ts)-1)
    ok = (t >= ts[0]) & (t <= ts[-1]) & ((ts[ix]-ts[ix-1]) <= .1)
    return np.stack([np.interp(t,ts,g[k]) for k in ['processed_x','processed_y','processed_vx','processed_vy']],1),ok

def geometry():
    from scripts.object_detection import mask_input_frames as mi
    from scripts.calibration import build_road_mask as br, site_extrinsic as se
    reference = calibration(CFG['train'][0])[1][:3,:3]
    rows=[]
    for c in sum([CFG[r] for r in ['train','validation','test']],[]):
        K,M=calibration(c); R=M[:3,:3]
        # Undo DAIR's road z=-1.73 before the road-z=0 slab calculation.
        t=M[:3,3]-R@np.array([0.,0.,1.73])
        mask=mi.input_mask(K,R,t); fm=mi.feature_mask(K,R,t)
        d=OUT/'geometry'/c; d.mkdir(parents=True,exist_ok=True)
        np.save(d/'feature_mask.npy',fm); cv2.imwrite(str(d/'input_mask.png'),mask.astype('uint8')*255)
        im=cv2.imread(str(frames(c)[len(frames(c))//2]))
        masked=np.where(mask[...,None],im,np.array(mi.MEAN_BGR,np.uint8))
        cv2.imwrite(str(d/'input_mask_preview.jpg'),np.vstack([cv2.resize(im,(960,540)),cv2.resize(masked,(960,540))]))
        tracks=read(trackpath(c))['tracks'].values()
        raw=br.paint(tracks,K,M,-1.73,im.shape[:2])
        raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
        regs=br.regions(raw); road=np.zeros_like(raw)
        for reg in regs: road[reg]=255
        cv2.imwrite(str(d/'road_mask.png'),road)
        over=im.copy(); over[road==0]=(over[road==0]*.3).astype('uint8')
        cv2.imwrite(str(d/'road_mask_preview.jpg'),over)
        angle=se.angle_between(reference,R)
        rows.append(dict(clip=c,input_fraction=float(mask.mean()),feature_fraction=float(fm.mean()),road_fraction=float((road>0).mean()),road_regions=len(regs),reference=CFG['train'][0],pose_difference_deg=angle,reference_flag=angle>se.MAX_POSE_DIFF_DEG))
    return dict(status='PASS',rows=rows,note='Pose comparison reuses previously estimated rotations; it does not independently validate height. All masks remain in this experiment directory.')

def yaw():
    import refine_yaw, score_heading as sh
    rows=[]
    for c in sum([CFG[r] for r in ['train','validation','test']],[]):
        p=OUT/'yaw'/c/'tracks.json'; p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(trackpath(c),p); doc=refine_yaw.refine(p,detdir(c),c)
        tracks=sh.load_tracks(p)
        override={tid:{s['frame']:s['yaw_refined'] for s in states if s['yaw_refined'] is not None} for tid,states in doc['tracks'].items()}
        base=sh.score_clip(tracks,det_dir=detdir(c)); refined=sh.score_clip(tracks,override,detdir(c))
        assert base['n']==refined['n']
        states=[s for v in doc['tracks'].values() for s in v]
        rows.append(dict(clip=c,tracks=len(tracks),states=len(states),with_heading=sum(s['yaw_refined'] is not None for s in states),filled=sum(s['yaw_filled'] for s in states),baseline=base,refined=refined))
    return dict(status='PASS',rows=rows,note='Motion consistency proxy only: the reference and sign correction share track motion; no independent yaw ground truth.')

def corrected_labels():
    from scripts.finetune import correct_labels as cl
    import road_mask as rm
    rows=[]
    for c in CFG['train']:
        K,M=calibration(c); cam=-M[:3,:3].T@M[:3,3]
        ff=frames(c); selected=sample(ff,24)
        bg=np.median(np.stack([cv2.imread(str(p)) for p in sample(ff,12)]),0).astype('uint8')
        src=TR/'outputs/finetune/pilot_labels'/c
        labels={p.stem:read(src/f'{p.stem}_label.json') for p in ff if (src/f'{p.stem}_label.json').exists()}
        fits=[]; per_track={}; fit_rows=[]
        for p in selected:
            for i,blob in cl.one_to_one(labels.get(p.stem,[]),cv2.imread(str(p)),bg,K,M):
                b=labels[p.stem][i]; shift,iou=cl.best_shift(K,M,b,blob,cam)
                fit_rows.append(dict(frame=int(p.stem),index=i,track_id=b.get('track_id'),shift_m=shift,iou=iou))
                if iou>=cl.MIN_IOU:
                    fits.append(shift)
                    if b.get('track_id') is not None: per_track.setdefault(b['track_id'],[]).append(shift)
        # Never propagate NaN if no foreground fit is accepted.
        clip_shift=float(np.median(fits)) if fits else 0.
        shifts={tid:float(np.median(v)) for tid,v in per_track.items() if len(v)>=cl.MIN_FITS}
        moving=np.array([[b['x'],b['y'],b['yaw']] for labs in labels.values() for b in labs if b.get('yaw_source')=='track_motion']).reshape(-1,3)
        road=cv2.imread(str(OUT/'geometry'/c/'road_mask.png'),0)
        n_in=n_out=n_off=n_lane=n_dup=0
        for stem,labs in labels.items():
            out=[]
            for b in labs:
                n_in+=1
                if not rm.keep(b,None,K,M,-1.73,road): n_off+=1; continue
                s=shifts.get(b.get('track_id'),clip_shift); direction=cl.ray_dir(b,cam)
                nb=dict(b,x=b['x']+s*direction[0],y=b['y']+s*direction[1],shift_m=s,pos_source='image_track' if b.get('track_id') in shifts else ('image_clip_median' if fits else 'unchanged_no_fit'))
                if b.get('yaw_source')!='track_motion' and len(moving):
                    heading=cl.lane_heading(b,moving)
                    if heading is not None: nb.update(yaw=heading,yaw_source='lane_flow'); n_lane+=1
                out.append(nb)
            out,dropped=cl.dedupe(out); n_dup+=dropped; n_out+=len(out)
            save(OUT/'labels_v2'/c/f'{stem}_label.json',out)
        row=dict(clip=c,fit_frames=[int(p.stem) for p in selected],background_frames=12,accepted_fits=len(fits),clip_shift_m=clip_shift,tracks_with_own_shift=len(shifts),labels_in=n_in,labels_out=n_out,off_road_dropped=n_off,duplicates_dropped=n_dup,lane_headings=n_lane,fit_rows=fit_rows)
        save(OUT/'labels_v2'/c/'manifest.json',row);rows.append(row)
        print('v2 labels',c,n_in,n_out,'fits',len(fits),flush=True)
    return dict(status='PASS',rows=rows,note='Original silhouette search and filtering; bounded to 24 fit frames and 12 background frames per training clip. No GPS, validation or test labels used.')

def contacts():
    from scripts.calibration import fit_height_from_gps_distances as fh
    result=[];sets=[]
    for c in CFG['train']:
        K,M=calibration(c); tid,states=target(c); ff=frames(c)
        bg=np.median(np.stack([cv2.imread(str(p)) for p in sample(ff,12)]),0).astype('uint8')
        selected=sample([s for s in states if 15<s['x']<90],16)
        rows=[];tiles=[]
        for s in selected:
            im=cv2.imread(str(ff[s['frame']]))
            body=K@(M[:3,:3]@np.array([s['x'],s['y'],-.73])+M[:3,3]); body=body[:2]/body[2]
            mask=(cv2.absdiff(im,bg).max(2)>35).astype('uint8')
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
            _,lab,stats,_=cv2.connectedComponentsWithStats(mask); ys,xs=np.nonzero(lab>0)
            if not len(xs): continue
            j=np.argmin((xs-body[0])**2+(ys-body[1])**2); k=lab[ys[j],xs[j]]; by,bx=np.nonzero(lab==k)
            v=int(by.max());u=float(np.median(bx[by>=v-2])); edge=bool(v>=im.shape[0]-2 or bx.min()<=1 or bx.max()>=im.shape[1]-2)
            # A distant unrelated blob is not a contact observation of this target.
            distance=float(np.hypot(xs[j]-body[0],ys[j]-body[1]))
            rows.append(dict(frame=s['frame'],t_s=s['t_s'],u=u,v=v,blob_px=len(by),skipped=edge or distance>40,edge=edge,nearest_blob_distance_px=distance))
            x0=max(0,min(im.shape[1]-240,int(u-120))); y0=max(0,min(im.shape[0]-200,v-150))
            tile=im[y0:y0+200,x0:x0+240].copy();ov=np.zeros_like(tile);ov[(lab==k)[y0:y0+200,x0:x0+240]]=(255,120,0)
            tile=cv2.addWeighted(tile,1,ov,.35,0); cv2.circle(tile,(int(u-x0),v-y0),4,(0,0,255),-1)
            cv2.putText(tile,f"{s['frame']} {'SKIP' if rows[-1]['skipped'] else ''}",(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
        d=OUT/'contacts'/c;d.mkdir(parents=True,exist_ok=True)
        if tiles:
            tiles+= [np.zeros_like(tiles[0])]*((-len(tiles))%4)
            cv2.imwrite(str(d/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
        valid=[r for r in rows if not r['skipped']]; gv,ok=interpolate(gps(c),[r['t_s'] for r in valid]); valid=[r for r,k in zip(valid,ok) if k];gv=gv[ok]
        output=dict(clip=c,target_id=tid,clicks=rows,n_valid=len(valid),review='pending; automatic foreground points are not manual ground truth')
        if len(valid)>=5:
            uv=np.array([[r['u'],r['v']] for r in valid]); P=gv[:,:2];D=np.linalg.norm(P[:,None]-P[None],axis=2)
            output['height_pitch']=fh.fit(uv,D,K,M[:3,:3],h0=16.26)
            loo=[fh.fit(np.delete(uv,i,0),np.delete(np.delete(D,i,0),i,1),K,M[:3,:3],h0=16.26)['h_m'] for i in range(len(uv))]
            output['leave_one_frame_out_height_m']=[min(loo),max(loo)]; sets.append((uv,D,K,M[:3,:3]))
        save(d/'auto_contacts.json',output);result.append(output)
    return dict(status='PASS' if len(sets)==2 else 'PARTIAL',rows=result,joint=fh.fit_joint(sets,h0=16.26) if len(sets)==2 else None,note='Training clips only; 12-frame background approximation. Raw automatic contact estimates need visual review. Height not adopted by baseline.')

def gps_labels():
    from grade_target_vs_gps import rigid_align
    from scripts.finetune.make_gps_labels import in_image
    from site_error_model import enu_from_latlon
    catalog=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    reference=pd.read_csv(ROOT/catalog.loc[CFG['train'][0],'gps_path']).iloc[0]
    origin=[float(reference.latitude),float(reference.longitude)]
    As=[];Bs=[];datasets={}
    for c in CFG['train']:
        tid,st=target(c)
        marker=read(TR/'outputs/target_id'/c/'marker_matches.json')
        st=[s for s in st if 'x_det' in s and marker['per_frame'][str(s['frame'])]['score']>=marker['score_min'] and any(r['start']<=s['frame']<=r['end'] for r in marker['target_runs'])]
        raw=pd.read_csv(ROOT/catalog.loc[c,'gps_path'])
        e,n=enu_from_latlon(raw.latitude,raw.longitude,*origin)
        # processed_x/y have a separate zero origin per recording and their
        # y axis is southward. Shared registration must instead use common ENU.
        g=pd.DataFrame(dict(video_time_sec=raw.video_time_sec,processed_x=e,processed_y=n,processed_vx=raw.east_velocity,processed_vy=raw.north_velocity)).dropna().groupby('video_time_sec',as_index=False).mean().sort_values('video_time_sec')
        v,ok=interpolate(g,[s['t_s'] for s in st]);st=[s for s,k in zip(st,ok) if k];v=v[ok]
        A=v[:,:2];B=np.array([[s['x'],s['y']] for s in st]);As.append(A);Bs.append(B);datasets[c]=(tid,st,v)
    R,t=rigid_align(np.concatenate(As),np.concatenate(Bs))
    save(OUT/'gps_labels_enu/registration.json',dict(rotation=R.tolist(),translation=t.tolist(),enu_origin_latlon=origin,train_clips=CFG['train'],frames=len(np.concatenate(As)),residual_rmse_m=float(np.sqrt(np.mean(np.sum((np.concatenate(As)@R.T+t-np.concatenate(Bs))**2,axis=1)))),method='shared unit-scale ENU GPS-to-road fit; only detected states with accepted image-marker matches; zero time-offset search',limitation='Same-direction tracks cannot independently identify antenna lever arm and camera origin. This is an integration check, not surveyed calibration. Older gps_labels folder is a superseded coordinate-adapter trial and must not be trained on.'))
    rows=[]
    for c,(tid,states,v) in datasets.items():
        K,M=calibration(c);count=0;selected=[]
        for s,g in zip(states,v):
            xy=R@g[:2]+t; vel=R@g[2:];b=dict(x=float(xy[0]),y=float(xy[1]),z=-1.73,l=4.6,w=1.9,h=1.5,yaw=float(np.arctan2(vel[1],vel[0])),vx=float(vel[0]),vy=float(vel[1]),role='gps',track_id=tid,score=1.)
            # BEV extent, complete visibility, and moving target are required.
            if not (5<b['x']<100 and -50<b['y']<50 and np.linalg.norm(vel)>.5):continue
            if not in_image(b['x'],b['y'],b['yaw'],K,M):continue
            labs=read(TR/'outputs/finetune/pilot_labels'/c/f"{s['frame']:03d}_label.json")
            other=[]
            for o in labs:
                if str(o.get('track_id'))==tid or np.hypot(o['x']-b['x'],o['y']-b['y'])<4:continue
                other.append(dict(o,role='other' if o.get('yaw_source')=='track_motion' else 'ignore'))
            save(OUT/'gps_labels_enu'/c/f"{s['frame']:03d}_label.json",[b]+other);count+=1;selected.append(s['frame'])
        rows.append(dict(clip=c,frames=count,frame_ids=selected))
    return dict(status='PASS' if all(r['frames']>0 for r in rows) else 'PARTIAL',rows=rows,label_directory='gps_labels_enu',note='GPS supervision is confined to training clips and accepted image-marker frames. Common ENU positions and east/north velocities replace per-recording processed coordinates; historical heading columns and offset files are absent.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['geometry','yaw','corrected_labels','contacts','gps_labels']);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name in a.stages:
        print('START',name,flush=True)
        try: result=globals()[name]()
        except Exception as e:
            traceback.print_exc();result=dict(status='FAILED',error=str(e),traceback=traceback.format_exc())
        save(OUT/f'{name}.json',result);print('DONE',name,result['status'],flush=True)
