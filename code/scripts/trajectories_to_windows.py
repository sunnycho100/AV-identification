"""Convert visual trajectories to the classifiers' strict 10 Hz primitive contract.

Unknown road users keep label=-1. Filename ACC/HV labels apply exclusively to an
image-identified instrumented track. Rejected windows remain in the audit CSV.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'classification/trajectory-avlike-transformer/inference_cnn'))
from data_protocol import features

def protocol_reasons(raw):
    if raw.shape!=(40,3) or not np.isfinite(raw).all():return ['shape_or_nonfinite']
    x=features(raw[None],'legacy11',7)[0]
    checks={
        'speed_below_source_domain':bool((raw[:,0]>22.352).all() and (x[:,0]>22.352).all()),
        'invalid_speed':bool((raw[:,:2]<55).all() and (raw[:,:2]>=0).all()),
        'invalid_net_gap':bool((raw[:,2]>1).all() and (raw[:,2]<200).all() and (x[:,5]>0).all()),
        'excess_acceleration':bool(abs(x[:,[1,4]]).max()<=8.001),
        'gap_speed_inconsistent':bool(abs(raw[-1,2]-raw[0,2]-np.trapz(raw[:,1]-raw[:,0],dx=.1))<=8.001)}
    return [k for k,v in checks.items() if not v]

def recover_dimensions(tracks, det_dir):
    """One-to-one frame assignments, within 2 m, followed by robust track medians."""
    frames={};sizes={k:[] for k in tracks}
    for tid,states in tracks.items():
        for s in states:frames.setdefault(s['frame'],[]).append((tid,s))
    for f,states in frames.items():
        p=det_dir/f'{f:03d}_pred.json'
        if not p.exists():continue
        boxes=[b for b in json.loads(p.read_text()) if b['class_name']=='car' and b['score']>=.2]
        if not boxes:continue
        a=np.array([[s['x'],s['y']] for _,s in states]);b=np.array([[b['x'],b['y']] for b in boxes])
        dist=np.linalg.norm(a[:,None]-b[None],axis=2)
        ii,jj=linear_sum_assignment(np.minimum(dist,2.001))
        for i,j in zip(ii,jj):
            if dist[i,j]<=2 and 2<boxes[j]['l']<15 and .8<boxes[j]['w']<4:
                sizes[states[i][0]].append([boxes[j]['l'],boxes[j]['w'],boxes[j]['h']])
    return {k:dict(length_m=float(np.median(np.array(v)[:,0])),width_m=float(np.median(np.array(v)[:,1])),height_m=float(np.median(np.array(v)[:,2])),matches=len(v),coverage=len(v)/len(tracks[k]))
            for k,v in sizes.items() if len(v)>=5 and len(v)/len(tracks[k])>=.5}

def convert_clip(clip,tag,split,max_gap=.2):
    folder=ROOT/'trajectory/outputs/trajectories'/f'{clip}_{tag}'
    tracks=json.loads((folder/'tracks.json').read_text())['tracks']
    dims=recover_dimensions(tracks,ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}_road')
    (folder/'track_dimensions.json').write_text(json.dumps({'method':'one-to-one detection matches within 2 m; median dimensions, >=5 matches and >=50% coverage','tracks':dims},indent=2))
    tables={p.stem.removeprefix('vehicle_'):pd.read_csv(p,dtype={'lead_id':str}) for p in folder.glob('vehicle_*.csv')}
    states={tid:{s['frame']:s for s in st} for tid,st in tracks.items()}
    target=None;ip=folder/'instrumented.json'
    if ip.exists():
        identity=json.loads(ip.read_text())
        if identity.get('accepted',True) and not identity.get('warnings'):target=str(identity['vehicle_id'])
    accepted=[];metadata=[];audit=[];segments=0;short=0;pair_rows=0
    for tid,tab in tables.items():
        if tid not in dims:continue
        records=[]; token=0; previous=None
        for row in tab.itertuples():
            lead=str(row.lead_id)
            key=(lead,str(row.lane),row.direction)
            if key!=previous: token+=1
            previous=key
            if lead not in dims or lead not in tables or not row.detected or pd.isna(row.lane):continue
            ltab=tables[lead];match=ltab[ltab.frame==row.frame]
            if len(match)!=1:continue
            lr=match.iloc[0]
            if not lr.detected or lr.direction!=row.direction or lr.lane!=row.lane:continue
            fs,ls=states[tid][row.frame],states[lead][row.frame]
            # Center spacing is distinct from bumper-to-bumper clearance.
            gap=(ls['x']-fs['x'])*row.direction-(dims[tid]['length_m']+dims[lead]['length_m'])/2
            records.append((float(row.t_s),lead,int(row.lane),int(row.direction),abs(fs['vx']),abs(ls['vx']),gap,token))
        groups=[]
        for r in records:
            if not groups or r[-1]!=groups[-1][-1][-1] or r[0]-groups[-1][-1][0]>max_gap:
                groups.append([])
            groups[-1].append(r)
        pair_rows+=len(records)
        for group in groups:
            segments+=1
            t=np.array([r[0] for r in group]);values=np.array([r[4:7] for r in group])
            if len(t)<2 or t[-1]-t[0]<3.9-1e-6:short+=1;continue
            first=np.ceil((t[0]-1e-8)*10)/10
            starts=np.arange(first,t[-1]-3.9+1e-7,1.)
            for start in starts:
                query=start+np.arange(40)*.1
                raw=np.stack([np.interp(query,t,values[:,i]) for i in range(3)],axis=-1).astype(np.float32)
                reason=protocol_reasons(raw)
                meta=dict(clip=clip,split=split,vehicle_id=tid,leader_id=group[0][1],lane=group[0][2],direction=group[0][3],start_s=round(float(start),4),end_s=round(float(start+3.9),4),label=int(clip.startswith('AV_')) if tid==target else -1,identity_source='image_only' if tid==target else 'unidentified_traffic',follower_length_m=dims[tid]['length_m'],leader_length_m=dims[group[0][1]]['length_m'])
                audit.append(dict(**meta,accepted=not reason,rejection=';'.join(reason)))
                if not reason:accepted.append(raw);metadata.append(meta)
    report=dict(clip=clip,split=split,tracks=len(tracks),tracks_with_dimensions=len(dims),pair_rows=pair_rows,pair_intervals=segments,short_pair_intervals=short,candidate_windows=len(audit),accepted_windows=len(accepted),accepted_labeled_windows=sum(r['label']>=0 for r in metadata),identified_target=target,rejections=dict(Counter(x for a in audit for x in a['rejection'].split(';') if x)))
    return accepted,metadata,audit,report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='pilot_base');ap.add_argument('--clips',nargs='+');a=ap.parse_args()
    cfg=json.loads((ROOT/'configs/field_pilot.json').read_text());exp=ROOT/'experiments'/cfg['experiment']/a.tag;exp.mkdir(parents=True,exist_ok=True)
    raw=[];meta=[];audit=[];reports=[]
    for split in ['train','validation','test']:
        for clip in cfg[split]:
            if a.clips and clip not in a.clips:continue
            if not (ROOT/'trajectory/outputs/trajectories'/f'{clip}_{a.tag}'/'tracks.json').exists():continue
            rr,mm,aa,report=convert_clip(clip,a.tag,split,cfg['classification']['max_interpolation_gap_s'])
            raw+=rr;meta+=mm;audit+=aa;reports.append(report);print(json.dumps(report))
    for i,m in enumerate(meta):m['window_index']=i
    arr=np.stack(raw) if raw else np.empty((0,40,3),dtype=np.float32)
    np.savez_compressed(exp/'windows_4s.npz',primitive_windows=arr,labels=np.array([m['label'] for m in meta],dtype=np.int64),window_index=np.arange(len(raw)),split_group=np.array([m['clip'] for m in meta],dtype=str),dt=np.array(.1))
    cols=['window_index','clip','split','vehicle_id','leader_id','lane','direction','start_s','end_s','label','identity_source','follower_length_m','leader_length_m']
    pd.DataFrame(meta,columns=cols).to_csv(exp/'windows.csv',index=False)
    pd.DataFrame(audit,columns=cols[1:]+['accepted','rejection']).to_csv(exp/'window_audit.csv',index=False)
    (exp/'window_report.json').write_text(json.dumps(dict(clips=reports,total_windows=len(raw),labeled_windows=sum(m['label']>=0 for m in meta),protocol='unchanged source speed and physical gates; estimated dimensions; no cross-leader, cross-lane or long-gap interpolation'),indent=2))

if __name__=='__main__':main()
