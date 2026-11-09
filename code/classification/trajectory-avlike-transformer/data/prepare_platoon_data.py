"""Convert OpenACC platoons using follower Driver labels; export strict 4 s windows.

Run from any directory. Never overwrites an output directory. AV means ACC here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

FEATURES = ['v_FV_long', 'a_FV', 'jerk_FV', 'v_LV_long', 'a_LV', 'gap',
            'lateral_offset', 'delta_v', 'closing_speed', 'delta_acc', 'THW', 'TTC']
LABELS = {'ACC': 1, 'Human': 0}
DT = 0.1
SEQ = 40


def read_platoon(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        meta = {}
        for _ in range(5):
            row = next(reader)
            meta[row[0].strip()] = [v.strip() for v in row[1:] if v.strip()]
    count = int(meta['Number_of_vehicles'][0])
    vehicles = meta['Vehicle_order']
    if len(vehicles) != count:
        raise ValueError(f'{path}: vehicle count/order mismatch')
    day, month, year = map(int, meta['Date'][:3])
    meta['date_iso'] = f'{year:04d}-{month:02d}-{day:02d}'
    frame = pd.read_csv(path, skiprows=5)
    if not np.isfinite(frame.Time).all() or not (frame.Time.diff().dropna() > 0).all():
        raise ValueError(f'{path}: timestamps must be finite and strictly increasing')
    for i in range(1, count+1):
        for name in ['Speed', 'E', 'N']:
            if f'{name}{i}' not in frame:
                raise ValueError(f'{path}: missing {name}{i}')
    return meta, frame


def consecutive_segments(time, labels, valid):
    """Split before filtering, so unknown labels/invalid rows cannot be bridged."""
    breaks = np.r_[True, (~np.isclose(np.diff(time), DT, atol=1e-5, rtol=0)) |
                   (labels[1:] != labels[:-1]) | (~valid[:-1]) | (~valid[1:])]
    ids = np.cumsum(breaks)
    return [np.flatnonzero((ids == sid) & valid) for sid in np.unique(ids[valid])]


def pair_segments(frame, leader, follower, mismatch=2.):
    n = len(frame)
    time = frame.Time.to_numpy(float)
    driver_col = f'Driver{follower}'
    drivers = frame[driver_col].astype('string').str.strip() if driver_col in frame else pd.Series(pd.NA,index=frame.index,dtype='string')
    labels = drivers.map(LABELS).fillna(-1).to_numpy(int)
    # The supplied campaign notes specify a common ENU origin for all vehicles.
    # Keep the existing project convention: x east, y south.
    xy = np.stack([frame[f'E{leader}'], -frame[f'N{leader}'], frame[f'E{follower}'], -frame[f'N{follower}']],axis=-1).astype(float)
    speeds = frame[[f'Speed{leader}', f'Speed{follower}']].to_numpy(float)
    dt = np.diff(time)
    velocity = np.full_like(xy, np.nan)
    velocity[1:] = np.diff(xy,axis=0) / dt[:,None]
    magnitudes = np.stack([np.linalg.norm(velocity[:,:2],axis=1), np.linalg.norm(velocity[:,2:],axis=1)],axis=1)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(speeds).all(axis=1) & np.isfinite(velocity).all(axis=1)
    speed_ok = (np.abs(magnitudes-speeds)<=mismatch).all(axis=1) & (speeds>=0).all(axis=1)
    fv_mag = magnitudes[:,1]
    projection = np.sum((xy[:,:2]-xy[:,2:])*velocity[:,2:],axis=1)/np.maximum(fv_mag,1e-6)
    front = (fv_mag>1e-3) & (projection>0)
    contiguous = np.r_[False,np.isclose(dt,DT,atol=1e-5,rtol=0)]
    valid = (labels>=0) & finite & speed_ok & front & contiguous
    groups = consecutive_segments(time,labels,valid)
    retained = [g for g in groups if len(g)>=SEQ]
    audit = dict(raw_pair_rows=n, known_acc_rows=int((labels==1).sum()), known_human_rows=int((labels==0).sum()),
                 unknown_label_rows=int((labels<0).sum()), label_transitions=int(np.sum(labels[1:]!=labels[:-1])),
                 invalid_numeric_rows=int((~finite).sum()), speed_mismatch_rows=int((finite & ~speed_ok).sum()),
                 not_in_front_or_stationary_rows=int((finite & ~front).sum()),
                 invalid_dt_rows=int((~contiguous).sum()), quality_valid_rows=int(valid.sum()),
                 short_segment_rows=int(sum(len(g) for g in groups if len(g)<SEQ)),
                 retained_rows=int(sum(map(len,retained))), retained_segments=len(retained),
                 driver_counts={str(k):int(v) for k,v in drivers.fillna('<missing>').value_counts().items()},
                 missing_driver_column=driver_col not in frame)
    return retained, xy, velocity, labels, audit


def window_features(positions, times):
    """All differences computed INSIDE each [40,4] observation, no external frames."""
    dt = np.diff(times,axis=1)
    velocity = np.empty_like(positions)
    velocity[:,1:] = np.diff(positions,axis=1)/dt[:,:,None]
    velocity[:,0] = velocity[:,1]  # forward edge uses frame 1 within this window
    lv = velocity[:,:,:2]
    fv = velocity[:,:,2:]
    speed = np.linalg.norm(fv,axis=-1)
    direction = fv/(speed[:,:,None]+1e-6)
    relative = positions[:,:,:2]-positions[:,:,2:]
    gap = np.sum(relative*direction,axis=-1)
    lateral = relative[:,:,0]*(-direction[:,:,1])+relative[:,:,1]*direction[:,:,0]
    vfv = np.sum(fv*direction,axis=-1)
    vlv = np.sum(lv*direction,axis=-1)
    def diff(v):
        out=np.zeros_like(v)
        out[:,1:]=np.diff(v,axis=1)/dt
        return out
    afv, alv = diff(vfv),diff(vlv)
    jerk = diff(afv)
    delta = vlv-vfv
    closing = -delta
    thw=np.clip(gap/(vfv+1e-6),0,10)
    ttc=np.full_like(gap,30.)
    np.divide(gap,closing,out=ttc,where=(closing>1e-6)&(gap>0))
    ttc=np.clip(ttc,0,30)
    values=np.stack([vfv,afv,jerk,vlv,alv,gap,lateral,delta,closing,alv-afv,thw,ttc],axis=-1).astype(np.float32)
    return values


def convert(raw, output, stride=10):
    if stride < 1:
        raise ValueError('stride must be positive')
    paths=sorted(raw.rglob('*.csv'))
    if not paths:
        raise ValueError(f'No raw CSV files under {raw}')
    output.mkdir(parents=True,exist_ok=False)
    (output/'pairs').mkdir()
    shutil.copy2(__file__,output/'converter_snapshot.py')
    source_records=[]
    pair_records=[]
    segment_records=[]
    window_records=[]
    all_x=[]
    all_y=[]
    for path in paths:
        meta,frame=read_platoon(path)
        relative=path.relative_to(raw)
        campaign=relative.parts[0] if len(relative.parts)>1 else 'Vicolungo'
        source=campaign+'__'+path.stem
        split_group=campaign+'__'+meta['date_iso']
        source_records.append(dict(source_id=source,raw_file=relative.as_posix(),campaign=campaign,date=meta['date_iso'],
                                   split_group=split_group,raw_rows=len(frame),vehicles=meta['Vehicle_order'],
                                   metadata_acc=meta.get('ACC'),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        output_frames=[]
        for leader in range(1,len(meta['Vehicle_order'])):
            follower=leader+1
            pair_id=f'{source}__p{leader}-{follower}'
            groups,xy,velocity,labels,audit=pair_segments(frame,leader,follower)
            pair_records.append(dict(source_id=source,pair_id=pair_id,leader_index=leader,follower_index=follower,
                                     leader_vehicle=meta['Vehicle_order'][leader-1],follower_vehicle=meta['Vehicle_order'][follower-1],
                                     label_source=f'Driver{follower}',**audit))
            for number,indices in enumerate(groups):
                trajectory=f'{pair_id}__s{number:04d}'
                label=int(labels[indices[0]])
                assert np.all(labels[indices]==label)
                times=frame.Time.to_numpy(float)[indices]
                assert np.allclose(np.diff(times),DT,atol=1e-5,rtol=0)
                origin=xy[indices[0],2:]
                local_xy=xy[indices]-np.tile(origin,2)
                data=pd.DataFrame(dict(Trajectory_ID=trajectory,Time_Index=times-times[0],raw_time=times,
                                       raw_row=indices,source_id=source,split_group=split_group,pair_id=pair_id,label=label,
                                       Driver_FV='ACC' if label else 'Human',ID_LV=meta['Vehicle_order'][leader-1],ID_FV=meta['Vehicle_order'][follower-1]))
                for k,role in enumerate(['LV','FV']):
                    data[f'Speed_{role}']=frame[f'Speed{leader if k==0 else follower}'].to_numpy()[indices]
                    for axis_index,axis in enumerate(['x','y']):
                        data[f'processed_{axis}_{role}']=local_xy[:,2*k+axis_index]
                        data[f'Speed_{axis}_{role}']=velocity[indices,2*k+axis_index]
                # No invented altitude: horizontal features do not need it (Casale LV lacks Alt).
                output_frames.append(data)
                starts=np.arange(0,len(indices)-SEQ+1,stride)
                offsets=starts[:,None]+np.arange(SEQ)
                features=window_features(local_xy[offsets],times[offsets])
                good=np.isfinite(features).all(axis=(1,2)) & (features[:,:,5]>0).all(axis=1)
                segment_records.append(dict(Trajectory_ID=trajectory,source_id=source,split_group=split_group,campaign=campaign,
                                            pair_id=pair_id,label=label,raw_start_row=int(indices[0]),raw_end_row=int(indices[-1]),
                                            raw_start_time=float(times[0]),raw_end_time=float(times[-1]),frames=len(indices),
                                            candidate_windows=len(starts),valid_windows=int(good.sum())))
                for start,valid_window in zip(starts,good):
                    if not valid_window:
                        continue
                    window_records.append(dict(window_index=len(window_records),Trajectory_ID=trajectory,source_id=source,
                                               split_group=split_group,campaign=campaign,pair_id=pair_id,label=label,
                                               start_in_segment=int(start),raw_start_row=int(indices[start]),
                                               raw_end_row=int(indices[start+SEQ-1]),raw_start_time=float(times[start]),
                                               raw_end_time=float(times[start+SEQ-1])))
                all_x.append(features[good])
                all_y.append(np.full(int(good.sum()),label,dtype=np.int8))
        if output_frames:
            pd.concat(output_frames,ignore_index=True).to_csv(output/'pairs'/f'{source}.csv',index=False)
        print(f'{source}: {len(output_frames)} segments',flush=True)
    segments=pd.DataFrame(segment_records)
    windows=pd.DataFrame(window_records)
    pairs=pd.DataFrame(pair_records)
    segments.to_csv(output/'segments.csv',index=False)
    windows.to_csv(output/'windows.csv',index=False)
    pairs.to_csv(output/'pair_audit.csv',index=False)
    (output/'sources.json').write_text(json.dumps(source_records,indent=2))
    if not all_x:
        raise ValueError('No valid labeled segments; inspect pair_audit.csv')
    x=np.concatenate(all_x)
    y=np.concatenate(all_y)
    assert len(x)==len(windows)==len(y) and np.array_equal(y,windows.label)
    assert x.shape[1:]==(SEQ,len(FEATURES)) and np.isfinite(x).all()
    np.savez_compressed(output/'windows_4s.npz',X=x,y=y,feature_names=np.array(FEATURES),
                        window_index=np.arange(len(y)),split_group=windows.split_group.to_numpy(dtype=str))
    counts=windows.groupby(['campaign','label']).size().rename('windows').reset_index()
    counts.to_csv(output/'counts_by_campaign.csv',index=False)
    summary=dict(source_files=len(paths),pairs=len(pairs),segments=len(segments),rows=int(segments.frames.sum()),
                 windows=len(windows),av_acc_windows=int(y.sum()),hv_human_windows=int((y==0).sum()),
                 window_frames=SEQ,sampling_hz=10,window_timestamp_span_seconds=3.9,stride_frames=stride,
                 label_rule='follower Driver: ACC=1 (project AV shorthand), Human=0; unknown excluded; never filename/date',
                 split_rule='group by split_group (campaign/date); all same-source pairs and parts stay together',
                 coordinates='provided common ENU; x=E, y=-N; per-segment follower origin',
                 velocity_rule='quality filter backward differences; NPZ recomputes velocities and derivatives inside each window',
                 quality_rules={'max_coordinate_scalar_speed_mismatch_mps':2.,'positive_front_projection':True,
                                'min_segment_frames':40,'unknown_labels_excluded':True,'label_switch_splits':True},
                 normalization='none; fit normalization only on training windows after grouped split')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=Path(__file__).parent/'raw data')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'prepared_20260921')
    parser.add_argument('--stride',type=int,default=10)
    args=parser.parse_args()
    convert(args.raw_dir.resolve(),args.output.resolve(),args.stride)


if __name__=='__main__':
    main()
