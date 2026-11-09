"""Estimate pilot intrinsics/rotation from images, keeping metric-scale assumptions explicit."""
import argparse,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(root/'.runtime/modern-deps'),str(root/'trajectory/third_party/AnyCalib'),str(root/'trajectory')]
os.environ['TORCH_HOME']=str(root/'.runtime/torch-cache')
import cv2,numpy as np,torch
from anycalib import AnyCalib
from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
from scripts.calibration.vp_extrinsic_from_frame import solve_pose
clips=['AV_T_EW_6','HV_T_EW_5','AV_T_WE_4','HV_T_WE_5','AV_T_WE_5','HV_T_EW_6']
ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=ap.parse_args()
model=AnyCalib(model_id='anycalib_pinhole').to(args.device).eval()
report=[]
for clip in clips:
    frames=sorted((root/'trajectory/data/camera-data'/clip/'frames_all').glob('*.jpg'),key=lambda p:int(p.stem))
    cal=root/'trajectory/outputs/calibration/camera-data'/clip;cal.mkdir(parents=True,exist_ok=True)
    intrinsic_records=[]
    for frac in (.2,.5,.8):
        p=frames[min(int(frac*len(frames)),len(frames)-1)]
        bgr=cv2.imread(str(p));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(rgb).float().permute(2,0,1).to(args.device)/255
        with torch.no_grad():out=model.predict(tensor,cam_id='pinhole')
        intr=out['intrinsics'];intr=intr[0] if isinstance(intr,list) else intr
        intrinsic_records.append({'frame':int(p.stem),'intrinsics':intr.detach().cpu().numpy().reshape(-1).tolist()})
    values=np.array([r['intrinsics'] for r in intrinsic_records]);med=np.median(values,axis=0)
    fx,fy,cx,cy=med[:4];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    intr_path=cal/'150_anycalib_pinhole_pinhole.json'
    intr_path.write_text(json.dumps({'image':clip,'prediction':{'intrinsics':med.tolist()},'method':'componentwise median of three AnyCalib image estimates','per_frame':intrinsic_records,'intrinsics_mad':np.median(abs(values-med),axis=0).tolist()},indent=2))
    vps=[]
    for n in np.linspace(0,len(frames)-1,15).astype(int):
        vp,lines,diag=lsd_vanishing_point(cv2.imread(str(frames[n])))
        if vp is not None:vps.append({'frame':int(frames[n].stem),'vp':np.asarray(vp).tolist(),'diagnostics':diag})
    if len(vps)<3:raise RuntimeError(f'{clip}: insufficient vanishing points')
    v=np.array([r['vp'] for r in vps]);vp=np.median(v,axis=0)
    height=16.26
    R,t,pitch,yaw=solve_pose(K,vp,height)
    extr={'rotation':R.tolist(),'translation':t.tolist(),'diagnostics':{'height_m':height,'height_source':'historical Todd site value; not a new metric validation','vp_px':vp.tolist(),'vp_std_px':np.std(v,axis=0).tolist(),'pitch_deg':pitch,'yaw_deg':yaw,'roll_deg':0,'frames_used':len(vps)},'note':'Exploratory pilot calibration: image-derived intrinsics/vanishing point, assumed historical 16.26m metric scale; no GPS used in this initialization.'}
    (cal/'metric_extrinsic_site.json').write_text(json.dumps(extr,indent=2))
    (cal/'vanishing_point_samples.json').write_text(json.dumps(vps,indent=2))
    record={'clip':clip,'intrinsics':med.tolist(),'vp':vp.tolist(),'pitch_deg':pitch,'yaw_deg':yaw,'height_m':height,'metric_scale_validated':False}
    report.append(record);print(json.dumps(record),flush=True)
(root/'experiments/field_pilot_20261101/calibration_initial.json').write_text(json.dumps(report,indent=2))
