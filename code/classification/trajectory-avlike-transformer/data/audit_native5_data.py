"""Re-read MicroSim ZIP donors and audit the complete released 5Hz matrix."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from prepare_native5_data import local_interpolate
from prepare_highspeed_data import compute_features, THRESHOLD


def main():
    root=Path('data/prepared_highspeed_native5_20261010')
    m=pd.read_csv(root/'windows.csv')
    with np.load(root/'windows_4s.npz') as data:
        raw=data['primitive_windows']; x=data['X']
        np.testing.assert_array_equal(data['window_index'],m.window_index)
    feature_difference=float(np.max(np.abs(compute_features(raw,dt=.2)-x)))
    assert feature_difference<.003
    assert (raw[:,:,0]>THRESHOLD).all() and (m.observation_span<=4.000001).all()
    micro=m[m.campaign.eq('MicroSim')]
    difference=0.; donor_difference=0.
    with zipfile.ZipFile('data/external_raw_20261010/microsim_original.zip') as archive:
        for name,rows in micro.groupby('raw_file'):
            df=pd.read_csv(archive.open(name)).dropna(subset=['timestamps','timestamps2','speed1','speed2',
                'Cumulative Distance Travelled (m)1','Cumulative Distance Travelled (m)2']).reset_index(drop=True)
            ts=[pd.to_datetime(df[c],utc=True,format='mixed').astype('int64').to_numpy()/1e9 for c in ['timestamps','timestamps2']]
            values=[np.column_stack([df[f'speed{i}']/3.6,df[f'Cumulative Distance Travelled (m){i}']]) for i in [1,2]]
            for row in rows.itertuples():
                # Avoid serializing/reparsing the epoch float as the reconstruction origin.
                start=int(row.start_in_segment)//2
                grid=ts[1][start]+np.arange(20)*.2
                a,b=[local_interpolate(t,v,grid) for t,v in zip(ts,values)]
                assert a is not None and b is not None
                assert b[3]==row.raw_start_row and b[4]==row.raw_end_row
                assert a[3]==row.leader_start_row and a[4]==row.leader_end_row
                begin,end=min(a[1],b[1]),max(a[2],b[2])
                assert end-begin<=4.000001
                assert (values[1][b[3]:b[4]+1,0]>THRESHOLD).all()
                expected=np.column_stack([b[0][:,0],a[0][:,0],a[0][:,1]-b[0][:,1]-4.5])
                difference=max(difference,float(np.max(np.abs(expected-raw[row.window_index]))))
                donor_difference=max(donor_difference,abs(begin-row.raw_start_time),abs(end-row.raw_end_time))
    assert difference<.0001 and donor_difference<.000001
    result=dict(status='passed',windows_checked=len(m),microsim_windows_checked=len(micro),
        feature_max_abs_diff=feature_difference,raw_microsim_max_abs_diff=difference,
        donor_time_max_abs_diff=donor_difference,microsim_max_observation_seconds=float(micro.observation_span.max()))
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
