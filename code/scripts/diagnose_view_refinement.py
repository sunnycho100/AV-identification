"""Image identity before MOT, fixed-frame reviews and supplied-time speed checks.

Writes diagnostic copies only. No target relabelling or independent car transforms.
"""
import cv2,numpy as np,pandas as pd
from verify_field_branches import ROOT,TR,read,save,gps,interpolate
from identify_baseline_targets import marker_centres
from visualize_field_baseline import project,draw,recover_dimensions

OUT=ROOT/'experiments/view_refinement_20261101'
BASE=ROOT/'experiments/field_baseline_20261101'

def inputs():
    rows=[(c,'view_refine_h15',BASE) for c in ['AV_V_EW_5','HV_V_EW_6']]
    rows += [(c,t,BASE) for c in ['AV_S_WE_6','HV_S_WE_6'] for t in ['field_baseline_h15','view_adapt_h15']]
    rows += [(c,t,OUT) for c in ['AV_S_EW_6','HV_S_EW_4'] for t in ['follow_base_h15','follow_adapt_h15']]
    return rows

def identity_before_mot(c,tag,mp):
    """One image-guided detection per time, with ambiguity and motion gates.

    Uses only actual car boxes >= .2; never changes class/score/metric geometry.
    Marker annotation is semi-automatic and is NOT an independent recall test.
    """
    centers=marker_centres(mp/f'{c}_markers.json')[0]
    dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}_road'
    cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
    ft={r['frame']:r for r in read(TR/'data/camera-data'/c/'frame_times.json')['frames']}
    rows=[];last=None
    for f,point in sorted(centers.items()):
        if not (dd/f'{f:03d}_pred.json').exists():continue
        candidates=[]
        for i,b in enumerate(read(dd/f'{f:03d}_pred.json')):
            if b['class_id']!=0 or b['score']<.2 or not(2<b['l']<8 and .8<b['w']<3):continue
            uv=project(b,K,M)
            if uv is None:continue
            margin=cv2.pointPolygonTest(cv2.convexHull(uv.astype(np.float32)),tuple(map(float,point)),True)
            if margin<0:continue
            dt=ft[f]['t_s']-last['t_s'] if last else None
            if last and 0<dt<=.2 and np.hypot(b['x']-last['box']['x'],b['y']-last['box']['y'])>45*dt+2:continue
            candidates.append((b['score'],i,b,margin))
        candidates.sort(key=lambda x:-x[0]);status='no_candidate';chosen=None
        if candidates:
            best=candidates[0]
            ambiguous=any(best[0]-v[0]<.1 and np.hypot(best[2]['x']-v[2]['x'],best[2]['y']-v[2]['y'])>2 for v in candidates[1:])
            if ambiguous:status='ambiguous'
            else:
                chosen=best[2];status='observed';last=dict(t_s=ft[f]['t_s'],box=chosen)
        rows.append(dict(frame=f,t_s=ft[f]['t_s'],status=status,candidates=len(candidates),box=chosen))
    return dict(clip=c,tag=tag,marker_frames=len(centers),observed=sum(r['status']=='observed' for r in rows),ambiguous=sum(r['status']=='ambiguous' for r in rows),below_birth=sum(r['box'] is not None and r['box']['score']<.45 for r in rows),rows=rows)

def review(c,tag):
    td=TR/'outputs/trajectories'/f'{c}_{tag}';identity=read(td/'instrumented.json');tid=str(identity['vehicle_id'])
    if tid not in read(td/'tracks.json')['tracks']:return
    tracks=read(td/'tracks.json')['tracks'];dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}_road';cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);dims=recover_dimensions(tracks,dd)
    table=pd.read_csv(td/f'vehicle_{tid}.csv',dtype={'lead_id':str}).set_index('frame');byid={k:{s['frame']:s for s in st} for k,st in tracks.items()};tiles=[];states=tracks[tid]
    for ix in np.unique(np.linspace(0,len(states)-1,6).astype(int)):
        s=states[ix];f=s['frame'];lead=str(table.loc[f,'lead_id']);im=cv2.imread(str(TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'))
        for oid,color,role in [(tid,(0,255,255),'SELF'),(lead,(255,255,0),'LEADER?')]:
            if oid not in dims or f not in byid.get(oid,{}):continue
            di=dims[oid];b=dict(byid[oid][f],l=di['length_m'],w=di['width_m'],h=di['height_m']);draw(im,b,K,M,color,f'{role} {oid}')
        tile=cv2.resize(im,(960,540));tile[:30]=0;cv2.putText(tile,f'{c} {tag} f{f} accepted={identity["accepted"]}',(8,21),0,.48,(255,255,255),1);tiles.append(tile)
    if len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT/'reviews'/f'{c}_{tag}.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))

def main():
    cv2.setNumThreads(1);(OUT/'reviews').mkdir(exist_ok=True)
    speed=[];early=[];series={}
    for c,tag,mp in inputs():
        td=TR/'outputs/trajectories'/f'{c}_{tag}'
        if not (td/'instrumented.json').exists():continue
        ident=read(td/'instrumented.json');early.append(identity_before_mot(c,tag,mp));review(c,tag)
        if not ident['accepted']:continue
        st=[s for s in read(td/'tracks.json')['tracks'][str(ident['vehicle_id'])] if s.get('x_det') is not None]
        gv,ok=interpolate(gps(c),[s['t_s'] for s in st]);gspd=np.hypot(gv[:,2],gv[:,3]);rows=[]
        for s,g,k in zip(st,gspd,ok):
            if k:rows.append(dict(frame=s['frame'],t_s=s['t_s'],video_speed=s['speed_mps'],gps_speed=float(g),error=abs(s['speed_mps']-g)))
        series[c,tag]={r['frame']:r for r in rows};pd.DataFrame(rows).to_csv(OUT/f'{c}_{tag}_speed_check.csv',index=False)
        speed.append(dict(clip=c,tag=tag,n=len(rows),mae_mps=float(np.mean([r['error'] for r in rows])) if rows else None,mean_video_mps=float(np.mean([r['video_speed'] for r in rows])) if rows else None,mean_gps_mps=float(np.mean([r['gps_speed'] for r in rows])) if rows else None))
    common=[]
    for c,old,new in [(c,'field_baseline_h15','view_adapt_h15') for c in ['AV_S_WE_6','HV_S_WE_6']]+[(c,'follow_base_h15','follow_adapt_h15') for c in ['AV_S_EW_6','HV_S_EW_4']]:
        a=series.get((c,old),{});b=series.get((c,new),{});ff=sorted(a.keys()&b.keys())
        if ff:common.append(dict(clip=c,n=len(ff),original_mae_mps=float(np.mean([a[f]['error'] for f in ff])),adapted_mae_mps=float(np.mean([b[f]['error'] for f in ff]))))
    save(OUT/'motion_check.json',dict(rows=speed,common_observed_frames=common,note='Uses supplied video_time_sec; timing not independently validated. No GPS fit, height fit, time shift, scale correction or separate vehicle transform. More samples / smoother positions are not proof of accuracy.'))
    save(OUT/'image_guided_candidates.json',dict(method='Strict projected-car hull contains image marker, dimensions plausible, ambiguity and short-time motion gates; actual detections only. Separate diagnostic before generic MOT, not substituted into official output.',rows=early))
    pairs=[]
    for c in ['AV_V_EW_5','HV_V_EW_6']:
        folder=OUT/'pairs'/f'{c}_view_refine_h15';df=pd.read_csv(folder/'following_pairs.csv',dtype={'target_id':str,'leader_id':str});dims=read(folder/'dimensions.json')
        expected=(df.leader_x_m-df.target_x_m)*df.direction-np.array([(dims[r.target_id]['length_m']+dims[r.leader_id]['length_m'])/2 for r in df.itertuples()])
        assert len(df) and np.isfinite(expected).all() and (expected>0).all()
        err=float(np.max(np.abs(expected-df.net_gap_m)));assert err<1e-6
        pairs.append(dict(clip=c,rows=len(df),net_gap_range_m=[float(df.net_gap_m.min()),float(df.net_gap_m.max())],maximum_gap_formula_difference_m=err))
    save(OUT/'pair_geometry_check.json',dict(rows=pairs,note='Checks common-coordinate gap calculation and positivity only. Vehicle length, metric scale, lane membership and actual gap accuracy are not independently validated; no car-specific transform applied.'))
    print('SPEED',speed);print('COMMON',common);print('EARLY', [{k:v for k,v in r.items() if k!='rows'} for r in early])

if __name__=='__main__':main()
