"""Image annotation and tracking integration for the all-recording survey."""
import argparse,os,shutil,subprocess,sys,time,traceback
import cv2,numpy as np,pandas as pd
from field_full_survey import ROOT,TR,OUT,CONFIG,read,save
ANNOT=ROOT/'configs/field_full_annotations.json'

def bootstrap():
    known={}
    for config,folder in [('field_baseline.json','field_baseline_20261101'),('view_adaptation.json','view_adaptation_20261101'),('view_refinement.json','view_refinement_20261101'),('seminole_following.json','view_refinement_20261101')]:
        cfg=read(ROOT/'configs'/config)
        for c,s in cfg.get('marker_seeds',{}).items():
            p=ROOT/'experiments'/folder/f'{c}_markers.json'
            if p.exists():known[c]=dict(seed=s,marker_source=str(p.relative_to(ROOT)),image_review='prior source-marker review retained')
    for c in ['AV_T_EW_6','HV_T_EW_5','HV_T_EW_6']:
        p=TR/'outputs/target_id'/c/'marker_matches.json';known[c]=dict(marker_source=str(p.relative_to(ROOT)),image_review='prior source-marker review retained')
    base=read(ROOT/'configs/field_baseline.json');road={'V_Verona_Rd/View_SE':base['road_polygons_px']['V'],'S_Seminole_Hwy/View_NW':base['road_polygons_px']['S'],'F_Fish_Hatchery_Rd/View_SW':[[0,680],[1660,115],[1790,115],[1720,260],[1540,430],[700,1079],[0,1079]]}
    doc=dict(method='Source images and supplied vehicle reference photos only; state labels do not determine target appearance. No GPS position or classifier prediction used for identity. Separate from detector training annotations.',road_polygons_px=road,clips=known)
    if not ANNOT.exists():save(ANNOT,doc)
    (OUT/'markers').mkdir(exist_ok=True);(OUT/'source_review').mkdir(exist_ok=True)
    for r in read(CONFIG)['clips']:
        c=r['clip'];p=OUT/'source_review'/f'{c}.jpg'
        if p.exists():continue
        cap=cv2.VideoCapture(str(ROOT/r['video_path']));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[]
        for f in np.unique(np.linspace(0,count-1,9).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,im=cap.read()
            if not ok:continue
            tile=np.full((390,640,3),255,np.uint8);tile[30:]=cv2.resize(im,(640,360));cv2.putText(tile,f'{c} source f{f}',(8,22),0,.6,(0,0,0),1);tiles.append(tile)
        cap.release()
        if len(tiles)%3:tiles.extend([np.zeros_like(tiles[0])]*((-len(tiles))%3))
        cv2.imwrite(str(p),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))

def markers_all(only=None):
    from identify_baseline_targets import markers
    cfg=read(CONFIG);ann=read(ANNOT);results=[];(OUT/'markers').mkdir(exist_ok=True)
    extra=ROOT/'configs/field_full_manual_seeds.json'
    if extra.exists():
        for c,a in read(extra).items():ann['clips'][c]=dict(seed=dict(frame=a[0],box=a[1:]),image_review='Source frame checked against roof apparatus and white vehicle marker; state label not used')
        save(ANNOT,ann)
    for r in cfg['clips']:
        if only and r['clip'] not in only:continue
        c=r['clip'];a=ann['clips'].get(c);p=OUT/'markers'/f'{c}_markers.json'
        if not a:continue
        if p.exists():continue
        if a.get('marker_source'):shutil.copy2(ROOT/a['marker_source'],p)
        elif a.get('seed'):
            if not (TR/'data/camera-data'/c/'frames_all'/f'{a["seed"]["frame"]:03d}.jpg').exists():continue
            ft=read(TR/'data/camera-data'/c/'frame_times.json');sel=dict(excerpts=[dict(clip=c,start=0,end=len(ft['frames'])-1)])
            markers(c,a['seed'],sel,OUT/'markers')
        print('MARKER',c,flush=True)

def tracking(c,mc,row):
    from verify_field_branches import ROOT as _root
    from scripts.pipeline import extract_trajectories as P
    import road_mask as rm
    dd=TR/'outputs/object_detection/camera-data'/f'{c}_{mc["tag"]}';rd=dd.with_name(dd.name+'_road');rd.mkdir(exist_ok=True);cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);times=P.ft.load(c);ann=read(ANNOT)
    first=cv2.imread(str(TR/'data/camera-data'/c/'frames_all'/'000.jpg'));h,w=first.shape[:2];group=row['site']+'/'+row['view'];poly=ann['clips'].get(c,{}).get('road_polygon_px',ann['road_polygons_px'].get(group))
    if poly is not None:
        mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.array(poly,np.int32)],255);source='image-review polygon'
    elif group=='T_Todd_Dr/View_SE':
        prior=TR/'outputs/calibration/camera-data'/c/'road_mask.png'
        if prior.exists():mask=cv2.imread(str(prior),0);source='existing Todd road mask'
        else:mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[rm.road_polygon(K,M,-1.73)],255);source='existing Todd-only projected road band'
    else:raise ValueError('ROAD_REVIEW_PENDING: '+group)
    assert mask.shape==(h,w);save(rd/'calibration_used.json',dict(cal,road_mask=dict(source=source,polygon_px=poly,survey_annotations=str(ANNOT.relative_to(ROOT)))));cv2.imwrite(str(rd/'road_mask.png'),mask)
    center=-M[:3,:3].T@M[:3,3];cam=(center[0],center[1],center[2]+1.73);fpx=K[1,1];rmax=np.sqrt(max(fpx*cam[2]-cam[2]**2,0));n_in=n_out=0
    for fp in sorted(dd.glob('*_pred.json')):
        f=int(fp.stem.split('_')[0]);boxes=read(fp);keep=[] if times[f][1] else [b for b in boxes if rm.keep(b,None,K,M,-1.73,mask) and np.hypot(b['x']-cam[0],b['y']-cam[1])<=rmax];save(rd/fp.name,keep);n_in+=len(boxes);n_out+=len(keep)
    td=TR/'outputs/tracking/camera-data'/f'{c}_{mc["tag"]}_road';dest=OUT/'runs'/f'{c}_{row["model"]}'
    cmd=[sys.executable,'scripts/tracking/run_ab3dmot.py','--det-dir',str(rd),'--out-dir',str(td),'--low-thresh','0.2','--max-age','6','--fps','30']
    with (dest/'tracking.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=TR,env=dict(os.environ,NUMBA_DISABLE_JIT='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    doc=read(td/'tracks.json');tracks,joined=P.st.stitch(doc['tracks'],30);cov=P.pp.footprint_cov(cam,fpx,1);nid=max((int(k) for k in tracks),default=0)+1;pieces={}
    for k,st in tracks.items():
        parts=P.pp.split_jumps(st,30,times,cov);pieces[k]=parts[0]
        for part in parts[1:]:pieces[str(nid)]=part;nid+=1
    smooth={k:P.pp.clean(st,30,times,.1,cov) for k,st in pieces.items()};smooth={k:s for k,s in smooth.items() if len(s)>=3};smooth=P.drop_duplicates(smooth);centres=P.lane_centres(smooth);out=TR/'outputs/trajectories'/f'{c}_{mc["tag"]}';out.mkdir(exist_ok=True);P.export(smooth,centres,out,30)
    save(out/'tracks.json',dict(meta=doc['meta'],tracks=smooth));save(out/'lanes.json',dict(centres_by_direction={str(k):v for k,v in centres.items()}));save(out/'run.json',dict(clip=c,settings=read(CONFIG)['tracker'],detections_from=cal,road_mask_source=source,boxes_in=n_in,boxes_kept=n_out,vehicles=len(smooth),stitch_joins=len(joined),scope='Survey; original tracking/stitching/smoothing/lane functions; full supplied clip. Other traffic are intermediate candidates only.'))
    im=first.copy();im[mask==0]=(im[mask==0]*.3).astype('uint8');cv2.imwrite(str(dest/'road_review.jpg'),cv2.resize(im,(960,540)))
    return dict(clip=c,model=row['model'],status='COMPLETE',tracks=len(smooth),boxes_in=n_in,boxes_kept=n_out,tag=mc['tag'])

def track_all(watch):
    cfg=read(CONFIG);done=[]
    while True:
        pending=0
        for r in cfg['clips']:
            c=r['clip']
            for key in r['models']:
                d=OUT/'runs'/f'{c}_{key}';inp=d/'inference.json';op=d/'tracking.json'
                if op.exists() and read(op)['status']=='COMPLETE':continue
                if not inp.exists():pending+=1;continue
                if read(inp)['status']!='COMPLETE':continue
                try:result=tracking(c,cfg['models'][key],dict(r,model=key))
                except Exception as e:
                    result=dict(clip=c,model=key,status='FAILED',error=str(e),traceback=traceback.format_exc())
                    if 'ROAD_REVIEW_PENDING' in str(e):pending+=1;continue
                save(op,result);done.append(result);print('TRACK_RESULT',result,flush=True)
        if not watch or not pending:break
        time.sleep(3)
    save(OUT/'tracking_summary.json',dict(rows=[read(p) for p in sorted((OUT/'runs').glob('*/tracking.json'))]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['bootstrap','markers','track']);ap.add_argument('--watch',action='store_true');ap.add_argument('--clips',nargs='+');a=ap.parse_args();cv2.setNumThreads(1)
    {'bootstrap':bootstrap,'markers':lambda:markers_all(a.clips),'track':lambda:track_all(a.watch)}[a.stage]()
