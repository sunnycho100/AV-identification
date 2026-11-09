"""Verify every exported window against original follower labels and timestamps."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_platoon_data import read_platoon, window_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=Path(__file__).parent/'raw data')
    parser.add_argument('--prepared',type=Path,default=Path(__file__).parent/'prepared_20260921')
    args=parser.parse_args()
    root=args.prepared
    windows=pd.read_csv(root/'windows.csv')
    segments=pd.read_csv(root/'segments.csv')
    pairs=pd.read_csv(root/'pair_audit.csv')
    sources=json.loads((root/'sources.json').read_text())
    with np.load(root/'windows_4s.npz',allow_pickle=False) as archive:
        x,y,groups=archive['X'],archive['y'],archive['split_group']
    assert len(x)==len(windows) and x.shape[1:]==(40,12)
    assert np.isfinite(x).all() and np.array_equal(y,windows.label)
    assert np.array_equal(groups,windows.split_group)
    assert windows.window_index.tolist()==list(range(len(windows)))
    assert not windows.duplicated(['source_id','pair_id','raw_start_row']).any()
    checked=0
    largest_error=0.
    for source in sources:
        meta,frame=read_platoon(args.raw_dir/source['raw_file'])
        group=windows[windows.source_id==source['source_id']]
        source_pairs=pairs[pairs.source_id==source['source_id']]
        assert len(source_pairs)==len(meta['Vehicle_order'])-1
        for _,pair in source_pairs.iterrows():
            lv,fv=int(pair.leader_index),int(pair.follower_index)
            assert fv==lv+1
            assert pair.leader_vehicle==meta['Vehicle_order'][lv-1]
            assert pair.follower_vehicle==meta['Vehicle_order'][fv-1]
            w=group[group.pair_id==pair.pair_id]
            if w.empty:
                continue
            rows=w.raw_start_row.to_numpy()[:,None]+np.arange(40)
            assert np.array_equal(rows[:,-1],w.raw_end_row)
            drivers=frame[f'Driver{fv}'].astype('string').str.strip().to_numpy()[rows]
            expected=np.where(w.label.to_numpy()==1,'ACC','Human')
            assert (drivers==expected[:,None]).all()
            time=frame.Time.to_numpy()[rows]
            assert np.allclose(np.diff(time,axis=1),.1,atol=1e-5,rtol=0)
            np.testing.assert_allclose(time[:,0],w.raw_start_time,atol=1e-7,rtol=0)
            xy=np.stack([frame[f'E{lv}'],-frame[f'N{lv}'],frame[f'E{fv}'],-frame[f'N{fv}']],axis=-1)
            expected_x=window_features(xy[rows],time)
            actual=x[w.window_index]
            error=float(np.max(np.abs(expected_x-actual)))
            largest_error=max(largest_error,error)
            np.testing.assert_allclose(expected_x,actual,atol=1e-3,rtol=1e-4)
            checked+=len(w)
        assert group.split_group.nunique()<=1
    assert checked==len(x)
    result=dict(passed=True,windows_checked=checked,pairs_checked=len(pairs),
                source_files_checked=len(sources),raw_label_matches=True,
                four_second_contiguity=True,adjacent_pair_order_matches=True,
                raw_window_feature_recomputation_max_abs_difference=largest_error,
                source_group_consistency=True,finite_features=True)
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print(segments.groupby(['campaign','label']).agg(segments=('frames','size'),frames=('frames','sum')).to_string())


if __name__=='__main__':
    main()
