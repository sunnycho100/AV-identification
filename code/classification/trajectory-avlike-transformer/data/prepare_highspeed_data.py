"""Build common longitudinal 4 s inputs, with source/label provenance and audits.

All newly applied smoothing and differentiation is window-local. Upstream
resampling already present in released datasets cannot be undone by this code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.signal import savgol_filter

from prepare_platoon_data import read_platoon

FEATURES = ['v_FV_long', 'a_FV', 'jerk_FV', 'v_LV_long', 'a_LV', 'gap',
            'delta_v', 'closing_speed', 'delta_acc', 'THW', 'TTC']
THRESHOLD = 50 * .44704
DT, SEQ, STRIDE = .1, 40, 10
GEOD = Geod(ellps='WGS84')


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def compute_features(raw, dt=DT):
    """raw [..., T, 3] is follower speed, leader speed, bumper gap in SI units."""
    raw = np.asarray(raw, dtype=np.float64)
    smoothed = savgol_filter(raw, 7, 2, axis=-2, mode='interp')
    vf, vl, gap = np.moveaxis(smoothed, -1, 0)
    af = np.gradient(vf, dt, axis=-1, edge_order=2)
    al = np.gradient(vl, dt, axis=-1, edge_order=2)
    jerk = np.gradient(af, dt, axis=-1, edge_order=2)
    dv = vl-vf
    closing = -dv
    thw = np.clip(gap / np.maximum(vf, 1e-6), 0, 10)
    ttc = np.full_like(gap, 30.)
    np.divide(gap, closing, out=ttc, where=closing > 1e-6)
    ttc = np.clip(ttc, 0, 30)
    return np.stack([vf, af, jerk, vl, al, gap, dv, closing, al-af, thw, ttc], axis=-1).astype(np.float32)


def continuous_runs(time, labels, valid, dt=DT):
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return []
    boundary = ((np.diff(ids) != 1) | (np.abs(np.diff(time[ids])-dt) > 1e-5)
                | (np.diff(labels[ids]) != 0))
    return np.split(ids, np.flatnonzero(boundary)+1)


class Builder:
    def __init__(self, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        (output/'pairs').mkdir()
        self.rows, self.x, self.raw, self.audit, self.sources = [], [], [], [], []

    def source(self, path, campaign, **extra):
        self.sources.append(dict(file=str(path), campaign=campaign,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra))

    def pair(self, frame, source, campaign, pair, label_quality, gap_origin, date):
        frame = frame.copy().sort_values('time').reset_index(drop=True)
        if frame.time.duplicated().any():
            raise ValueError(f'{pair}: duplicate times must be resolved before conversion')
        t = frame.time.to_numpy(float)
        y = frame.label.to_numpy(int)
        raw = frame[['v_fv','v_lv','gap']].to_numpy(float)
        valid = (np.isfinite(raw).all(1) & np.isin(y, [0,1])
                 & (raw[:,0] > THRESHOLD) & (raw[:,0] < 55)
                 & (raw[:,1] >= 0) & (raw[:,1] < 55)
                 & (raw[:,2] > 1) & (raw[:,2] < 200))
        record = dict(source_id=source, campaign=campaign, pair_id=pair, raw_rows=len(frame),
                      valid_rows=int(valid.sum()), candidate_windows=0, windows=0,
                      quality_rejected=0, label_quality=label_quality)
        exported = []
        for number, ids in enumerate(continuous_runs(t, y, valid)):
            if len(ids) < SEQ:
                continue
            trajectory = f'{pair}__s{number:04d}'
            starts = np.arange(0, len(ids)-SEQ+1, STRIDE)
            idx = ids[starts[:,None]+np.arange(SEQ)]
            windows = raw[idx]
            features = compute_features(windows)
            # Shared physical checks; no thresholds are learned from val/test.
            delta = windows[:,:,1]-windows[:,:,0]
            residual = np.abs(windows[:,-1,2]-windows[:,0,2]
                              - np.trapz(delta, dx=DT, axis=1))
            max_acc = np.max(np.abs(features[:,:,[1,4]]), axis=(1,2))
            good = (np.isfinite(features).all(axis=(1,2))
                    & (features[:,:,0] > THRESHOLD).all(1)
                    & (features[:,:,5] > 0).all(1)
                    & (max_acc <= 8) & (residual <= 8))
            record['candidate_windows'] += len(starts)
            record['windows'] += int(good.sum())
            record['quality_rejected'] += int((~good).sum())
            for j in np.flatnonzero(good):
                a, b = int(idx[j,0]), int(idx[j,-1])
                self.rows.append(dict(window_index=len(self.rows), Trajectory_ID=trajectory,
                    source_id=source, split_group=source, campaign=campaign, pair_id=pair,
                    label=int(y[a]), label_quality=label_quality, date=date,
                    start_in_segment=int(starts[j]), raw_start_row=int(frame.raw_row.iloc[a]),
                    raw_end_row=int(frame.raw_row.iloc[b]), raw_start_time=float(t[a]),
                    raw_end_time=float(t[b]), min_raw_speed=float(windows[j,:,0].min()),
                    kinematic_residual=float(residual[j]), gap_origin=gap_origin,
                    native_dt=DT, observation_span=3.9))
            self.x.append(features[good])
            self.raw.append(windows[good].astype(np.float32))
            export = frame.iloc[ids].copy()
            export['Trajectory_ID'] = trajectory
            exported.append(export)
        if exported:
            pd.concat(exported).to_csv(self.output/'pairs'/f'{pair}.csv', index=False)
        self.audit.append(record)

    def finish(self):
        manifest = pd.DataFrame(self.rows)
        x, raw = np.concatenate(self.x), np.concatenate(self.raw)
        assert len(x) == len(manifest) and np.isfinite(x).all()
        assert (raw[:,:,0] > THRESHOLD).all()
        manifest.to_csv(self.output/'windows.csv', index=False)
        np.savez_compressed(self.output/'windows_4s.npz', X=x, y=manifest.label.to_numpy(np.int8),
            primitive_windows=raw, feature_names=np.array(FEATURES),
            window_index=manifest.window_index.to_numpy(), split_group=manifest.split_group.to_numpy(str))
        pd.DataFrame(self.audit).to_csv(self.output/'pair_audit.csv', index=False)
        counts = manifest.groupby(['campaign','label','label_quality']).size().rename('windows').reset_index()
        counts.to_csv(self.output/'counts.csv', index=False)
        json_write(self.output/'sources.json', self.sources)
        json_write(self.output/'protocol.json', dict(threshold_mps=THRESHOLD, seq_len=SEQ, dt=DT,
            stride=STRIDE, features=FEATURES, smoothing='Savitzky-Golay 7 points order 2 inside each window',
            derivatives='np.gradient edge_order=2, within-window only',
            physical_checks='raw gap in (1,200)m; speeds below55m/s; smooth |a|<=8m/s2; gap/dv residual<=8m',
            gap='bumper-to-bumper; CATS assumes 4.92m for GPS reference-point separation correction',
            upstream_limit='Released OpenACC already interpolated from 3-5Hz; Vanderbilt step2 inherited cleaning. New processing never accesses outside-window frames.',
            excluded_sources=['Arizona original: official download 404',
                              'MicroSim: preserved raw; separate native 5Hz conversion required'],
            sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [self.output/'windows_4s.npz', self.output/'windows.csv']}))
        shutil.copy2(__file__, self.output/'converter_snapshot.py')
        print(counts.to_string(index=False), flush=True)


def openacc(builder, root):
    for path in sorted(root.rglob('*.csv')):
        meta, df = read_platoon(path)
        campaign = path.parent.name if path.parent != root else 'Vicolungo'
        source = campaign+'__'+path.stem
        builder.source(path, campaign, label_basis='per-frame follower Driver', metadata=meta)
        for leader in range(1, len(meta['Vehicle_order'])):
            follower = leader+1
            labels = df[f'Driver{follower}'].astype(str).str.strip().map({'ACC':1,'Human':0}).fillna(-1)
            f = pd.DataFrame(dict(time=df.Time, v_fv=df[f'Speed{follower}'],
                v_lv=df[f'Speed{leader}'], gap=df[f'IVS{leader}'], label=labels.astype(int),
                raw_row=np.arange(len(df))))
            builder.pair(f, source, campaign, f'{source}__p{leader}-{follower}',
                         'per_frame_control', 'published_IVS_bumper_gap', meta['date_iso'])
        print(f'CONVERT {source}', flush=True)


def cats(builder, root):
    # Original experiment description explicitly defines all five vehicle roles.
    roles = {1:0, 2:1, 3:1, 4:0, 5:0}
    for folder in ['test1118','test1124']:
        for path in sorted((root/folder).glob('test*.xlsx')):
            sheets = pd.read_excel(path, sheet_name=None, header=None, engine='openpyxl')
            vehicles = {}
            for name, df in sheets.items():
                vid = int(re.search(r'(\d+)', name).group(1))
                if df.empty:
                    builder.audit.append(dict(source_id=f'CATS__{folder}_{path.stem}',
                        campaign='CATS', pair_id=f'missing_vehicle_{vid}', raw_rows=0,
                        windows=0, reason='empty original worksheet; adjacent pairs omitted'))
                    continue
                df = df.iloc[:,:5].copy()
                df.columns = ['raw_row','gps_time','lon','lat','speed']
                df['tick'] = df.gps_time.map(lambda t: int(round(float(str(t).split(':')[1])*10)))
                if df.tick.duplicated().any():
                    raise ValueError(f'{path} {name}: duplicated GPS time')
                vehicles[vid] = df
            if not set(vehicles).issubset(roles):
                raise ValueError(f'{path}: unexpected vehicle roles')
            source = f'CATS__{folder}_{path.stem}'
            builder.source(path, 'CATS', label_basis=f'{folder}/test description veh2,3 ACC; veh4,5 Human')
            for lead in range(1,5):
                follow=lead+1
                if lead not in vehicles or follow not in vehicles:
                    continue
                df = vehicles[follow].merge(vehicles[lead], on='tick', suffixes=('_fv','_lv'), validate='one_to_one')
                _, _, distance = GEOD.inv(df.lon_fv.to_numpy(), df.lat_fv.to_numpy(),
                                         df.lon_lv.to_numpy(), df.lat_lv.to_numpy())
                f = pd.DataFrame(dict(time=df.tick.to_numpy()/10, v_fv=df.speed_fv,
                    v_lv=df.speed_lv, gap=distance-4.92, label=roles[follow], raw_row=df.raw_row_fv))
                builder.pair(f, source, 'CATS', f'{source}__p{lead}-{follow}',
                             'experiment_control_assignment', 'WGS84_GPS_distance_minus_4.92m',folder)
            print(f'CONVERT {source}', flush=True)


def vanderbilt(builder, path):
    df = pd.read_csv(path)
    builder.source(path, 'Vanderbilt', label_basis='following ACC vehicle per published experiment; never Type_LV')
    for tid, d in df.groupby('Trajectory_ID', sort=False):
        f = pd.DataFrame(dict(time=d.Time_Index, v_fv=d.Speed_FAV, v_lv=d.Speed_LV,
                             gap=d.Spatial_Gap, label=1, raw_row=d.index))
        builder.pair(f, 'Vanderbilt__Tennessee2019_CAN', 'Vanderbilt',
                     f'Vanderbilt__Tennessee2019_CAN__s{tid}', 'experiment_control_assignment',
                     'published_radar_bumper_gap', '2019')


def ngsim(builder, root):
    target = pd.read_csv(root/'ngsim_fast.csv').drop_duplicates()
    leaders = pd.read_csv(root/'ngsim_leaders.csv').drop_duplicates()
    keys = ['location','global_time','vehicle_id']
    if leaders.duplicated(keys).any():
        raise ValueError('Conflicting NGSIM leader states')
    joined = target.merge(leaders, left_on=['location','global_time','preceding'],
                          right_on=keys, suffixes=('_fv','_lv'), validate='many_to_one')
    joined = joined[(joined.lane_id_fv == joined.lane_id_lv) & (joined.v_class_lv == 2)].copy()
    joined['recording_key'] = joined.global_time - joined.frame_id_fv*100
    builder.source(root/'ngsim_fast.csv', 'NGSIM', label_basis='weak: conventional traffic, no controller-state annotations')
    builder.source(root/'ngsim_leaders.csv', 'NGSIM', label_basis='leader states only')
    for (location, recording), data in joined.groupby(['location','recording_key']):
        source = f'NGSIM__{location}_{recording}'
        for (fv, lv, lane), d in data.groupby(['vehicle_id_fv','vehicle_id_lv','lane_id_fv']):
            gap = (d.local_y_lv-d.local_y_fv-d.v_length_lv)*.3048
            f = pd.DataFrame(dict(time=(d.global_time-recording)/1000,
                v_fv=d.v_vel_fv*.3048, v_lv=d.v_vel_lv*.3048, gap=gap, label=0, raw_row=d.frame_id_fv))
            builder.pair(f,source,'NGSIM',f'{source}__p{lv}-{fv}_lane{lane}',
                         'weak_conventional_traffic','front_centres_minus_leader_length',str(recording))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw data')
    parser.add_argument('--external', default='data/external_raw_20261010')
    parser.add_argument('--output', default='data/prepared_highspeed_20261010')
    args=parser.parse_args()
    builder=Builder(Path(args.output))
    openacc(builder,Path(args.raw))
    external=Path(args.external)
    cats_root=next(external.glob('Filed-Experiment-Data-ACC_Data-*'))
    cats(builder,cats_root)
    vanderbilt(builder,external/'vanderbilt_step2.csv')
    ngsim(builder,external)
    builder.finish()


if __name__ == '__main__':
    main()
