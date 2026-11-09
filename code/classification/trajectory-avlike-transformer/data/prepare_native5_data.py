"""Common 5 Hz inputs plus corrected MicroSim, with <=4 s donor support.

This separate protocol never claims upsampling creates genuine 10 Hz data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from prepare_highspeed_data import compute_features, FEATURES, THRESHOLD


def local_interpolate(times, values, grid):
    left=int(np.searchsorted(times,grid[0],side='right'))-1
    right=int(np.searchsorted(times,grid[-1],side='left'))
    if left<0 or right>=len(times):
        return None
    t=times[left:right+1]
    if len(t)<2 or np.any(np.diff(t)<=0) or np.max(np.diff(t))>.3:
        return None
    result=np.stack([np.interp(grid,t,values[left:right+1,k]) for k in range(values.shape[1])],axis=-1)
    return result,t[0],t[-1],left,right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',default='data/prepared_highspeed_20261010')
    parser.add_argument('--external',default='data/external_raw_20261010')
    parser.add_argument('--output',default='data/prepared_highspeed_native5_20261010')
    args=parser.parse_args(); output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    parent=Path(args.prepared)
    m=pd.read_csv(parent/'windows.csv')
    with np.load(parent/'windows_4s.npz',allow_pickle=False) as data:
        raw=data['primitive_windows'][:,::2].copy()
    x=compute_features(raw,dt=.2)
    good=(x[:,:,0]>THRESHOLD).all(1)&(x[:,:,5]>0).all(1)&(np.abs(x[:,:,[1,4]]).max((1,2))<=8)
    # Preserve provenance and observed input budget of original 10Hz windows.
    m=m[good].copy(); raws=[raw[good]]; xs=[x[good]]
    m['native_dt']=.2; m['grid_span']=3.8
    m['original_window_index']=m.window_index
    rows=m.to_dict('records'); audit=[]; seen=set()
    archive_path=Path(args.external)/'microsim_original.zip'
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.csv') or not Path(name).name.startswith('60_'):
                continue
            df=pd.read_csv(archive.open(name))
            required=['timestamps','timestamps2','speed1','speed2',
                      'Cumulative Distance Travelled (m)1','Cumulative Distance Travelled (m)2']
            if not set(required).issubset(df):
                audit.append(dict(file=name,status='missing_columns',windows=0)); continue
            df=df.dropna(subset=required).reset_index(drop=True)
            if len(df)<20:
                audit.append(dict(file=name,status='too_short',windows=0)); continue
            t1=pd.to_datetime(df.timestamps,utc=True,format='mixed').astype('int64').to_numpy()/1e9
            t2=pd.to_datetime(df.timestamps2,utc=True,format='mixed').astype('int64').to_numpy()/1e9
            # Do not reorder nonmonotonic streams and accidentally stitch them.
            if (np.diff(t1)<=0).any() or (np.diff(t2)<=0).any():
                audit.append(dict(file=name,status='nonmonotonic_time',windows=0)); continue
            day=str(pd.to_datetime(df.timestamps2.iloc[0],utc=True).date())
            v1=np.column_stack([df.speed1.to_numpy()/3.6,df['Cumulative Distance Travelled (m)1']])
            v2=np.column_stack([df.speed2.to_numpy()/3.6,df['Cumulative Distance Travelled (m)2']])
            accepted=0; duplicates=0
            for start in range(0,len(df)-19,5):
                grid=t2[start]+np.arange(20)*.2
                a,b=local_interpolate(t1,v1,grid),local_interpolate(t2,v2,grid)
                if a is None or b is None: continue
                observed_start=min(a[1],b[1]); observed_end=max(a[2],b[2])
                if observed_end-observed_start>4.000001: continue
                # Source points used by interpolation must also satisfy threshold.
                if not (v2[b[3]:b[4]+1,0]>THRESHOLD).all(): continue
                primitive=np.column_stack([b[0][:,0],a[0][:,0],a[0][:,1]-b[0][:,1]-4.5])
                if not np.isfinite(primitive).all() or not ((primitive[:,2]>1)&(primitive[:,2]<200)).all(): continue
                if not ((primitive[:,0]>THRESHOLD)&(primitive[:,0]<55)&(primitive[:,1]>=0)&(primitive[:,1]<55)).all(): continue
                features=compute_features(primitive[None],dt=.2)
                residual=float(abs(primitive[-1,2]-primitive[0,2]-np.trapz(primitive[:,1]-primitive[:,0],dx=.2)))
                if (features[0,:,0]<=THRESHOLD).any() or (features[0,:,5]<=0).any() or np.abs(features[0,:,[1,4]]).max()>8 or residual>8: continue
                key=(round(float(grid[0]),3),hashlib.sha256(np.round(primitive,5).tobytes()).hexdigest())
                if key in seen: duplicates+=1; continue
                seen.add(key)
                source='MicroSim__'+day
                rows.append(dict(window_index=len(rows),Trajectory_ID=source+'__'+Path(name).stem,
                    source_id=source,split_group=source,campaign='MicroSim',pair_id=Path(name).stem,
                    label=1,label_quality='experiment_control_assignment',date=day,
                    # Keep stride units comparable to 10Hz parent: every fourth 1s window.
                    start_in_segment=start*2,raw_start_row=b[3],raw_end_row=b[4],
                    raw_start_time=float(observed_start),raw_end_time=float(observed_end),
                    min_raw_speed=float(v2[b[3]:b[4]+1,0].min()),kinematic_residual=residual,
                    gap_origin='time_aligned_integrated_distance_minus_4.5m_assumption',
                    native_dt=.2,observation_span=float(observed_end-observed_start),grid_span=3.8,
                    raw_file=name,leader_start_row=a[3],leader_end_row=a[4],grid_start_time=float(grid[0]),
                    original_window_index=-1))
                xs.append(features); raws.append(primitive[None].astype(np.float32)); accepted+=1
            audit.append(dict(file=name,status='processed',windows=accepted,duplicates=duplicates))
    m=pd.DataFrame(rows); m.window_index=np.arange(len(m))
    x=np.concatenate(xs); raw=np.concatenate(raws)
    assert x.shape==(len(m),20,11) and np.isfinite(x).all()
    assert (m.observation_span<=4.000001).all()
    m.to_csv(output/'windows.csv',index=False)
    np.savez_compressed(output/'windows_4s.npz',X=x,y=m.label.to_numpy(np.int8),primitive_windows=raw,
        feature_names=np.array(FEATURES),window_index=m.window_index.to_numpy(),split_group=m.split_group.to_numpy(str))
    pd.DataFrame(audit).to_csv(output/'microsim_file_audit.csv',index=False)
    counts=m.groupby(['campaign','label']).size().rename('windows').reset_index()
    counts.to_csv(output/'counts.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(dt=.2,seq_len=20,grid_span_seconds=3.8,
        maximum_observation_seconds=4.,threshold_mps=THRESHOLD,
        inherited_data=str(parent),microsim_velocity='raw OBD speed /3.6; no upstream Smoothed columns',
        microsim_gap='interpolate both cumulative distances to same time; subtract assumed 4.5m length',
        caution='Integrated distance gap is source-derived; vehicle length/initial offset assumed. MicroSim only enters unlabeled pretraining in this round.',
        smoothing='Same seven-point quadratic Savitzky-Golay for all 5Hz sources; derivatives computed after resampling within each window',
        sha256={f:hashlib.sha256((output/f).read_bytes()).hexdigest() for f in ['windows.csv','windows_4s.npz']}),indent=2))
    shutil.copy2(__file__,output/'converter_snapshot.py')
    print(counts.to_string(index=False))


if __name__=='__main__': main()
