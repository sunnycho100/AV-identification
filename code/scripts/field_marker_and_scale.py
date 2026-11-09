"""Run existing image-only target-marker and lane-width scale methods on pilot footage."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'trajectory'))
import cv2,numpy as np
from scripts.tracking import match_marker_template as marker
from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
from scripts.calibration.solve_height_lane_width import solve_height,MEASURE_ROW,LANE_WIDTH_M
cfg=json.loads((ROOT/'configs/field_pilot.json').read_text());exp=ROOT/'experiments'/cfg['experiment']
ap=argparse.ArgumentParser();ap.add_argument('--clips',nargs='+',default=['AV_T_EW_6','HV_T_EW_5','HV_T_EW_6']);args=ap.parse_args()
# Supplied reference is a training recording, HV_T_EW_5, around 20 seconds.
# Exact crop excludes its roof equipment and keeps the hood emblem.
ref=ROOT/'data/field/figures/vehicles/red_lincoln_context.png'
im=cv2.imread(str(ref));tpl=im[795:822,617:659]
cv2.imwrite(str(exp/'hood_marker_template.png'),tpl)
provenance={'source':str(ref.relative_to(ROOT)), 'crop_xyxy':[617,795,659,822], 'source_recording':'HV_T_EW_5', 'selection':'supplied annotated vehicle reference; no GPS used', 'method':'existing multiscale color normalized-correlation and temporal-coherence gates'}
(exp/'hood_marker_template.json').write_text(json.dumps(provenance,indent=2))
def parallel_matches(folder, template):
    """Same OpenCV method and thresholds as match_frames, with frame parallelism."""
    cv2.setNumThreads(1)
    templates=[(s,cv2.resize(template,None,fx=s,fy=s)) for s in marker.SCALES]
    def one(fp):
        img=cv2.imread(str(fp));roi=img[marker.ROI_TOP:,:];best=(-1.,0,0,0,0,1.)
        for s,t in templates:
            if t.shape[0]>=roi.shape[0] or t.shape[1]>=roi.shape[1]:continue
            _,mx,_,loc=cv2.minMaxLoc(cv2.matchTemplate(roi,t,cv2.TM_CCOEFF_NORMED))
            if mx>best[0]:best=(mx,loc[0],loc[1]+marker.ROI_TOP,t.shape[1],t.shape[0],s)
        score,x,y,w,h,s=best
        return int(fp.stem),dict(score=round(float(score),3),x=x,y=y,w=w,h=h,scale=s)
    with ThreadPoolExecutor(max_workers=4) as pool:return dict(pool.map(one,sorted(folder.glob('*.jpg'))))
for clip in args.clips:
    existing=ROOT/'trajectory/outputs/target_id'/clip/'marker_matches.json'
    if existing.exists():
        print('Reusing completed marker matches '+clip,flush=True);continue
    fd=ROOT/'trajectory/data/camera-data'/clip/'frames_all'
    print('Marker matching '+clip,flush=True)
    per=parallel_matches(fd,tpl);runs=marker.coherent_runs(per)
    good=[r for r in runs if max(x[1]['score'] for x in r)>=marker.PEAK_MIN]
    describe=lambda r:dict(start=r[0][0],end=r[-1][0],frames=len(r),peak_score=max(x[1]['score'] for x in r))
    result=dict(frames=len(per),score_min=marker.SCORE_MIN,peak_min=marker.PEAK_MIN,target_runs=list(map(describe,good)),rejected_runs=[describe(r) for r in runs if r not in good],per_frame=per,template=provenance)
    out=ROOT/'trajectory/outputs/target_id'/clip;out.mkdir(parents=True,exist_ok=True)
    (out/'marker_matches.json').write_text(json.dumps(result,indent=1))
    if good:
        f,r=max(max(good,key=len),key=lambda x:x[1]['score'])
        img=cv2.imread(str(fd/f'{f:03d}.jpg'));cv2.rectangle(img,(r['x'],r['y']),(r['x']+r['w'],r['y']+r['h']),(0,255,255),3)
        cv2.imwrite(str(out/f'marker_peak_{f:03d}.jpg'),img)
    print(json.dumps(dict(clip=clip,target_runs=result['target_runs'])),flush=True)
records=[]
for clip in cfg['train']:
    cal=ROOT/'trajectory/outputs/calibration/camera-data'/clip
    fx,fy,cx,cy=json.loads((cal/'150_anycalib_pinhole_pinhole.json').read_text())['prediction']['intrinsics']
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    files=sorted((ROOT/'trajectory/data/camera-data'/clip/'frames_all').glob('*.jpg'))
    samples=[]
    for i in np.linspace(0,len(files)-1,5).astype(int):
        vp,lines,diag=lsd_vanishing_point(cv2.imread(str(files[i])))
        if vp is None:continue
        lx=np.array(sorted((d-n[1]*MEASURE_ROW)/n[0] for n,d,_ in lines if abs(n[0])>1e-9))
        if len(lx)<4:continue
        h,misfit,_=solve_height(K,vp,lx,LANE_WIDTH_M)
        if h is not None:samples.append(dict(frame=int(files[i].stem),height_m=float(h),misfit_px=float(misfit)))
    row=dict(clip=clip,samples=samples,nominal_lane_width_m=LANE_WIDTH_M,used_in_pilot_calibration=False)
    if samples:
        hs=[s['height_m'] for s in samples];row.update(median_height_m=float(np.median(hs)),height_iqr_m=np.percentile(hs,[25,75]).tolist())
    records.append(row);print(json.dumps(row),flush=True)
(exp/'lane_width_scale_check.json').write_text(json.dumps(records,indent=2))
