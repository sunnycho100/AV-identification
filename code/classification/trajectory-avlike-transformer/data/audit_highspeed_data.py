"""Independently recheck exported primitives, windows, controls, and source split."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from prepare_highspeed_data import compute_features, FEATURES, THRESHOLD
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from highspeed_experiment import make_splits


def audit(root):
    m=pd.read_csv(root/'windows.csv')
    with np.load(root/'windows_4s.npz',allow_pickle=False) as data:
        x,raw=data['X'],data['primitive_windows']
        assert data['feature_names'].tolist()==FEATURES
        assert np.array_equal(data['window_index'],m.window_index)
        assert np.array_equal(data['y'],m.label)
        assert np.array_equal(data['split_group'],m.split_group)
        assert x.shape==(len(m),40,11) and np.isfinite(x).all()
        assert (raw[:,:,0]>THRESHOLD).all() and (x[:,:,0]>THRESHOLD).all()
        difference=float(np.max(np.abs(compute_features(raw)-x)))
        assert difference<.003, difference
    checked=0
    for pair, rows in m.groupby('pair_id'):
        original=pd.read_csv(root/'pairs'/f'{pair}.csv')
        for trajectory, windows in rows.groupby('Trajectory_ID'):
            p=original[original.Trajectory_ID.eq(trajectory)].reset_index(drop=True)
            for row in windows.itertuples():
                w=p.iloc[row.start_in_segment:row.start_in_segment+40]
                assert len(w)==40 and np.allclose(np.diff(w.time),.1,atol=1e-5)
                assert np.isclose(w.time.iloc[0],row.raw_start_time)
                assert np.isclose(w.time.iloc[-1],row.raw_end_time)
                assert w.label.eq(row.label).all()
                np.testing.assert_allclose(w[['v_fv','v_lv','gap']].to_numpy(),raw[row.window_index],atol=1e-5,rtol=1e-6)
                checked+=1
    # The labels for the original CATS trailing vehicles are explicitly Human.
    cats=m[m.campaign.eq('CATS')]
    assert cats[cats.pair_id.str.contains('__p3-4|__p4-5')].label.eq(0).all()
    assert cats[cats.pair_id.str.contains('__p1-2|__p2-3')].label.eq(1).all()
    splits=[]
    for fold in range(3):
        split=make_splits(m,fold)
        splits.append({k:len(v) for k,v in split.items()})
    report=dict(windows_checked=checked,feature_max_abs_diff=difference,
                split_counts=splits,status='passed',
                limits='Audit starts at released source/pair primitives; does not undo upstream resampling or establish roadside-video accuracy.')
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',default='data/prepared_highspeed_20261010')
    audit(Path(parser.parse_args().prepared))
