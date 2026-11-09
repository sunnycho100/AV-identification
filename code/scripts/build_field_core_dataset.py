"""Build target/leader-only field samples without a speed floor or fixed length.

Reuses already computed tracks; it does not claim to redo MOT on 43 recordings.
Outputs contain the instrumented vehicle and its immediate leader only. Other
road users remain intermediate candidates in the cached legacy tracker.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from trajectories_to_windows import recover_dimensions

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'trajectory/scripts/tracking'))
from identify_target_by_marker import corners8

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def inside_image(s,dim,K,M,width,height):
    xyz=corners8([dim['length_m'],dim['width_m'],dim['height_m']],s['yaw'],[s['x'],s['y'],s['z']])
    camera=xyz@M[:3,:3].T+M[:3,3]
    if (camera[:,2]<=0).any():return False
    p=camera@K.T;uv=p[:,:2]/p[:,2:]
    return bool((uv[:,0]>=0).all() and (uv[:,0]<width).all() and (uv[:,1]>=0).all() and (uv[:,1]<height).all())

def interval_groups(records,max_gap):
    groups=[]
    for r in records:
        if not groups or r['token']!=groups[-1][-1]['token'] or r['t_s']-groups[-1][-1]['t_s']>max_gap+1e-7:
            groups.append([])
        groups[-1].append(r)
    return groups

def windows_for_group(group,duration,hz,stride):
    n=int(round(duration*hz));span=(n-1)/hz
    t=np.array([r['t_s'] for r in group]);values=np.array([[r['follower_speed_mps'],r['leader_speed_mps'],r['net_gap_m']] for r in group])
    if len(t)<2 or t[-1]-t[0]<span-1e-7:return []
    first=np.ceil((t[0]-1e-8)*hz)/hz;out=[]
    for start in np.arange(first,t[-1]-span+1e-7,stride):
        query=start+np.arange(n)/hz
        raw=np.stack([np.interp(query,t,values[:,i]) for i in range(3)],1).astype(np.float32)
        if not np.isfinite(raw).all() or (raw[:,:2]<0).any() or (raw[:,2]<=0).any():continue
        flags=[]
        if np.abs(np.gradient(raw[:,:2],1/hz,axis=0)).max()>8:flags.append('acceleration_review')
        if abs(raw[-1,2]-raw[0,2]-np.trapz(raw[:,1]-raw[:,0],dx=1/hz))>8:flags.append('gap_speed_review')
        out.append((raw,float(start),float(query[-1]),';'.join(flags)))
    return out

def convert(run,cfg,out):
    clip,tag=run['clip'],run['tag'];td=ROOT/'trajectory/outputs/trajectories'/f'{clip}_{tag}'
    dst=out/'pairs'/f'{clip}_{tag}';dst.mkdir(parents=True,exist_ok=True)
    if not (td/'instrumented.json').exists():return dict(clip=clip,tag=tag,status='identity_pending'),[]
    ident=read(td/'instrumented.json')
    if ident.get('warnings') or not ident.get('accepted',True):return dict(clip=clip,tag=tag,status='identity_review'),[]
    tid=str(ident['vehicle_id']);tracks=read(td/'tracks.json')['tracks']
    detdir=ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}'
    dims=recover_dimensions(tracks,detdir.with_name(detdir.name+'_road'))
    if tid not in dims:return dict(clip=clip,tag=tag,status='target_dimensions_pending'),[]
    cal=read(detdir/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
    target=tracks[tid];f0=target[0]['frame']
    image=cv2.imread(str(ROOT/'trajectory/data/camera-data'/clip/'frames_all'/f'{f0:03d}.jpg'))
    height,width=image.shape[:2]
    states={k:{s['frame']:s for s in v} for k,v in tracks.items()}
    table=pd.read_csv(td/f'vehicle_{tid}.csv',dtype={'lead_id':str})
    audit=[];pairs=[];token=0;previous=None;relevant={tid:[]};selected_target=[]
    for row in table.itertuples():
        fs=states[tid][row.frame];lead=str(row.lead_id);key=(lead,str(row.lane),row.direction)
        if key!=previous:token+=1
        previous=key
        if 'x_det' in fs and inside_image(fs,dims[tid],K,M,width,height):selected_target.append(fs)
        reason=[];ls=states.get(lead,{}).get(row.frame)
        if lead not in dims or ls is None:reason.append('leader_not_observed')
        if 'x_det' not in fs:reason.append('target_detection_gap')
        if not inside_image(fs,dims[tid],K,M,width,height):reason.append('target_outside_image')
        gap=None
        if ls is not None and lead in dims:
            if 'x_det' not in ls:reason.append('leader_detection_gap')
            if not inside_image(ls,dims[lead],K,M,width,height):reason.append('leader_outside_image')
            gap=(ls['x']-fs['x'])*row.direction-(dims[tid]['length_m']+dims[lead]['length_m'])/2
            if not np.isfinite(gap) or gap<=0:reason.append('nonpositive_net_gap')
        audit.append(dict(frame=row.frame,t_s=fs['t_s'],target_id=tid,leader_id=lead,usable=not reason,reason=';'.join(reason)))
        if reason:continue
        relevant[tid].append(fs);relevant.setdefault(lead,[]).append(ls)
        pairs.append(dict(frame=row.frame,t_s=fs['t_s'],target_id=tid,leader_id=lead,lane=row.lane,direction=row.direction,
                          target_x_m=fs['x'],target_y_m=fs['y'],leader_x_m=ls['x'],leader_y_m=ls['y'],
                          follower_speed_mps=abs(fs['vx']),leader_speed_mps=abs(ls['vx']),net_gap_m=float(gap),token=token))
    # Retain usable single-vehicle observations even where the leader is absent.
    pd.DataFrame(selected_target).to_csv(dst/'target_trajectory.csv',index=False)
    cols=['frame','t_s','target_id','leader_id','lane','direction','target_x_m','target_y_m','leader_x_m','leader_y_m','follower_speed_mps','leader_speed_mps','net_gap_m','token']
    pd.DataFrame(pairs,columns=cols).to_csv(dst/'following_pairs.csv',index=False)
    pd.DataFrame(audit).to_csv(dst/'pair_audit.csv',index=False)
    save(dst/'tracks.json',dict(identity=ident,tracks={k:v for k,v in relevant.items() if v},scope='Only the target and its immediate leader during usable following observations.'))
    save(dst/'dimensions.json',{k:dims[k] for k in relevant if k in dims})
    groups=interval_groups(pairs,cfg['classification']['max_interpolation_gap_s']);windows=[];intervals=[]
    for gi,group in enumerate(groups):
        intervals.append(dict(group=gi,leader_id=group[0]['leader_id'],start_s=group[0]['t_s'],end_s=group[-1]['t_s'],observations=len(group)))
        for duration in cfg['classification']['window_durations_s']:
            for raw,start,end,flags in windows_for_group(group,duration,cfg['classification']['hz'],cfg['classification']['stride_s']):
                meta=dict(clip=clip,tag=tag,source_role=run['role'],target_id=tid,leader_id=group[0]['leader_id'],
                          group=gi,duration_s=duration,start_s=start,end_s=end,label=int(clip.startswith('AV_')),quality_flags=flags,
                          identity_source='image_only',split_group=clip)
                windows.append((raw,meta))
    save(dst/'intervals.json',intervals)
    return dict(clip=clip,tag=tag,status='EXPORTED',identified_target=tid,target_observations=len(selected_target),
                following_observations=len(pairs),following_intervals=intervals,
                windows_by_duration=dict(Counter(str(m['duration_s']) for _,m in windows)),
                audit_reasons=dict(Counter(r for a in audit for r in a['reason'].split(';') if r)),
                limits='Existing cached car-only MOT and leader association; not independent 3D truth. Heavy-vehicle leaders require detection/tracking support.'),windows

def preview_catalog(catalog,out,cfg):
    folder=out/'screening';folder.mkdir(parents=True,exist_ok=True)
    for site,group in catalog.groupby('site',sort=False):
        for row in group.itertuples():
            p=folder/f'{row.clip_id}.jpg'
            if p.exists():continue
            cap=cv2.VideoCapture(str(ROOT/row.video_path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            indices=np.unique(np.linspace(0,max(0,count-1),7).astype(int));tiles=[]
            for f in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read()
                if not ok:continue
                tile=Image.new('RGB',(480,295),'white');im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((480,270));tile.paste(im,(0,25))
                ImageDraw.Draw(tile).text((5,5),f'{row.clip_id}  source frame {f}  ~{f/fps:.2f}s',fill='black');tiles.append(tile)
            cap.release()
            canvas=Image.new('RGB',(480*len(tiles),295),'white')
            for i,im in enumerate(tiles):canvas.paste(im,(480*i,0))
            canvas.save(p)
        print('Previewed',site,len(group),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/field_core.json');ap.add_argument('--previews',action='store_true');a=ap.parse_args()
    cfg=read(ROOT/a.config);out=ROOT/'experiments'/cfg['experiment'];out.mkdir(parents=True,exist_ok=True)
    cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').fillna('');allowed={(r['site'],r['view']) for r in cfg['core_views']}
    core=cat[[ (r.site,r.view) in allowed for r in cat.itertuples()]].copy()
    core['field_label']=core.clip_id.str.startswith('AV_').astype(int)
    core['identity_status']='pending_image_review';core['processing_status']='not_processed_by_core_pipeline'
    all_windows=[];reports=[]
    for run in cfg['pilot_runs']:
        assert run['clip'] in set(core.clip_id)
        report,windows=convert(run,cfg,out);reports.append(report);all_windows+=windows
        core.loc[core.clip_id==run['clip'],'processing_status']=report['status']
        if 'identified_target' in report:core.loc[core.clip_id==run['clip'],'identity_status']='image_marker_identified'
    core.to_csv(out/'core_clips.csv',index=False)
    paths=[]
    for duration in cfg['classification']['window_durations_s']:
        selected=[(x,m) for x,m in all_windows if m['duration_s']==duration];n=int(round(duration*cfg['classification']['hz']))
        raw=np.stack([x for x,m in selected]) if selected else np.empty((0,n,3),np.float32)
        stem=f'windows_{duration:g}s';meta=[dict(m,window_index=i) for i,(x,m) in enumerate(selected)]
        np.savez_compressed(out/f'{stem}.npz',primitive_windows=raw,labels=np.array([m['label'] for m in meta],np.int64),
                            split_group=np.array([m['split_group'] for m in meta],dtype=str),dt=np.array(1/cfg['classification']['hz']))
        pd.DataFrame(meta,columns=['window_index','clip','tag','source_role','target_id','leader_id','group','duration_s','start_s','end_s','label','quality_flags','identity_source','split_group']).to_csv(out/f'{stem}.csv',index=False)
        paths.append(dict(path=f'{stem}.npz',shape=list(raw.shape),labels=dict(Counter(str(m['label']) for m in meta))))
    summary=dict(config=a.config,core_clips=len(core),core_gps_files=int((core.gps_status=='present').sum()),
                 core_labels=dict(Counter(core.mode_from_source_filename)),camera_groups=[dict(site=s,view=v,clips=len(g)) for (s,v),g in core.groupby(['site','view'])],
                 processed_runs=reports,arrays=paths,speed_floor_mps=None,
                 note='Field protocol pilot, not classifier accuracy. Durations are alternatives, not independent samples; do not combine overlapping windows across splits. Missing leader data do not erase the target trajectory.')
    save(out/'dataset_summary.json',summary)
    if a.previews:preview_catalog(core,out,cfg)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
