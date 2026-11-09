"""Evaluate frozen reviewed tracks; all GPS scores retain supplied synchronization.

No model, time-offset or spatial-scale fit. 'select' precedes GPS error evaluation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT/'configs/field_trajectory_quality.json'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
CFG = read(CONFIG)
OUT = ROOT/'experiments'/CFG['experiment']
SOURCE = ROOT/CFG['source']
TR = ROOT/'trajectory'
sys.path.insert(0, str(TR/'scripts/evaluation'))
from site_error_model import enu_from_latlon


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv(p):
    return pd.read_csv(p, dtype={'track_id': str, 'target_id': str, 'leader_id': str, 'record_id': str})


def pieces(df, gap=.2, strict=False):
    if not len(df): return []
    new = df.t_s.diff().gt(gap).to_numpy()
    if strict: new |= df.frame.diff().ne(1).to_numpy()
    for col in ['track_id', 'record_id', 'metric_piece']:
        if col in df:
            values = df[col].fillna('').astype(str)
            new |= values.ne(values.shift()).to_numpy()
    new[0] = True
    return [g.reset_index(drop=True) for _, g in df.groupby(np.cumsum(new))]


def span(groups):
    return max([float(g.t_s.iloc[-1]-g.t_s.iloc[0]) for g in groups], default=0.)


def supported(groups, query, cols):
    query = np.asarray(query)
    values = np.full((len(query), len(cols)), np.nan)
    ids = np.full(len(query), -1, dtype=int)
    for i, g in enumerate(groups):
        use = (query >= g.t_s.iloc[0]-1e-8) & (query <= g.t_s.iloc[-1]+1e-8)
        if len(g) < 2: continue
        for j, col in enumerate(cols): values[use, j] = np.interp(query[use], g.t_s, g[col])
        ids[use] = i
    return values, ids


def gps_interp(g, query):
    ts=g.video_time_sec.to_numpy(); query=np.asarray(query)
    ix=np.searchsorted(ts,query).clip(1,len(ts)-1)
    ok=(query>=ts[0]) & (query<=ts[-1]) & ((ts[ix]-ts[ix-1])<=CFG['maximum_gps_interpolation_gap_s'])
    vals=np.column_stack([np.interp(query,ts,g[c]) for c in ['east_m','north_m','east_velocity','north_velocity']])
    vals[~ok]=np.nan
    return vals,ok


def gps_frame(path):
    raw=pd.read_csv(path)
    cols=['video_time_sec','latitude','longitude','east_velocity','north_velocity']
    valid=raw.dropna(subset=cols)
    g=valid[cols].groupby('video_time_sec',as_index=False).mean().sort_values('video_time_sec')
    g['east_m'],g['north_m']=enu_from_latlon(g.latitude,g.longitude,float(g.latitude.iloc[0]),float(g.longitude.iloc[0]))
    return raw,g


def mae(x): return float(np.mean(np.abs(x))) if len(x) else None
def rmse(x): return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
def med(x): return float(np.median(x)) if len(x) else None


def jitter(groups, cols):
    residuals=[]
    for g in groups:
        if len(g)<3: continue
        t=g.t_s.to_numpy(); xy=g[cols].to_numpy()
        w=((t[1:-1]-t[:-2])/(t[2:]-t[:-2]))[:,None]
        residuals.extend(np.linalg.norm(xy[1:-1]-((1-w)*xy[:-2]+w*xy[2:]),axis=1))
    return med(residuals)


def records():
    return read(ROOT/CFG['review'])['records']


def load_track(clip):
    df=csv(SOURCE/'clips'/clip/'target_trajectory.csv').sort_values('t_s').reset_index(drop=True)
    assert df[['x','y','vx','vy','x_det','y_det','t_s']].notna().all().all()
    assert not df.is_imputed.any() and df.frame.is_unique and df.t_s.is_monotonic_increasing
    return df


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    meta=csv(SOURCE/'recording_results.csv');meta=meta[meta.variant=='recovered'].set_index('clip')
    rows=[]
    for r in records():
        clip=r['clip'];df=load_track(clip)
        rows.append(dict(clip=clip,site=str(meta.loc[clip,'site']),label=int(meta.loc[clip,'label']),
                         training_exposed=bool(meta.loc[clip,'training_exposed']),gps_present=pd.notna(cat.loc[clip,'gps_path']),
                         longest_supported_s=span(pieces(df)),points=len(df),
                         reviewed_frame_fraction=len(df)/(r['end_frame']-r['start_frame']+1)))
    table=pd.DataFrame(rows); chosen=[]
    for (site,label),g in table.groupby(['site','label']):
        g=g[g.gps_present].sort_values(['training_exposed','longest_supported_s','reviewed_frame_fraction','clip'],ascending=[True,False,False,True])
        chosen.append(g.iloc[0].to_dict())
    save(OUT/'case_selection.json',dict(policy=CFG['illustration_selection'],cases=chosen,
                                      selection_before_gps_errors=True,annotation_sha256=sha(ROOT/'data/annotations/leader_review_decisions.json')))
    save(OUT/'protocol.json',CFG)
    table.to_csv(OUT/'coverage_selection_inputs.csv',index=False)
    print('SELECTED', [r['clip'] for r in chosen],flush=True)


def evaluate():
    selection=read(OUT/'case_selection.json')
    assert selection['annotation_sha256']==sha(ROOT/'data/annotations/leader_review_decisions.json')
    assert read(OUT/'protocol.json')==CFG
    cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    meta=csv(SOURCE/'recording_results.csv');meta=meta[meta.variant=='recovered'].set_index('clip')
    overrides=read(ROOT/'experiments/field_review_applied_20261102/application_config.json')['visibility_overrides']
    inputs={};rows=[];sync_rows=[];offset_rows=[];all_samples=[];checks=[]
    def used(path): inputs[str(path.relative_to(ROOT))]=sha(path);return path
    used(CONFIG);used(ROOT/CFG['review']);used(ROOT/'data/annotations/leader_review_decisions.json');used(OUT/'case_selection.json')
    used(ROOT/'catalogs/field_clips.csv');used(SOURCE/'recording_results.csv')
    for r in records():
        c=r['clip'];folder=SOURCE/'clips'/c
        df=load_track(c);used(folder/'target_trajectory.csv');used(folder/'following_pairs.csv');used(folder/'frame_audit.csv')
        times=read(used(TR/'data/camera-data'/c/'frame_times.json'))['frames']
        byframe={s['frame']:s for s in times}
        last=r['end_frame']
        for o in overrides:
            if o['clip']==c and o['role']=='target':last=min(last,o['last_visible_frame'])
        eligible=[s for s in times if r['start_frame']<=s['frame']<=last and not s.get('copy',False)]
        assert all(not byframe[f].get('copy',False) and abs(byframe[f]['t_s']-t)<1e-6 for f,t in zip(df.frame,df.t_s))
        assert df.frame.min()>=r['start_frame'] and df.frame.max()<=last
        checks.extend([f'{c}:native_time_and_no_copy',f'{c}:user_visibility_bounds'])
        gr=pieces(df,CFG['maximum_visual_interpolation_gap_s']); strict=pieces(df,strict=True)
        pairs=csv(folder/'following_pairs.csv')
        dt=np.diff(df.t_s);row=dict(clip=c,site=str(meta.loc[c,'site']),view=str(meta.loc[c,'view']),
            label='AV' if int(meta.loc[c,'label']) else 'HV',training_exposed=bool(meta.loc[c,'training_exposed']),
            leader_state=r['leader_state'],review_start_s=float(eligible[0]['t_s']),review_end_s=float(eligible[-1]['t_s']),
            review_frames=len(eligible),target_points=len(df),observed_frame_fraction=len(df)/len(eligible),
            target_span_s=float(df.t_s.iloc[-1]-df.t_s.iloc[0]),longest_observed_run_s=span(strict),
            longest_supported_run_s=span(gr),supported_segments=len(gr),largest_observation_gap_s=float(dt.max()) if len(dt) else 0.,
            paired_observed_points=len(pairs),raw_jitter_m=jitter(gr,['x_det','y_det']),smooth_jitter_m=jitter(gr,['x','y']),
            smoothing_shift_median_m=med(np.linalg.norm(df[['x','y']].to_numpy()-df[['x_det','y_det']].to_numpy(),axis=1)),
            gps_available=pd.notna(cat.loc[c,'gps_path']),speed_n=0,disp_n=0,acc_n=0,sync_status='GPS_UNAVAILABLE')
        q=np.arange(np.ceil(df.t_s.min()*10)/10,df.t_s.max()+1e-8,.1)
        cols=['x','y','x_det','y_det','speed_mps']
        v,vid=supported(gr,q,cols)
        sample=pd.DataFrame(dict(t_s=q,piece=vid,x=v[:,0],y=v[:,1],raw_x=v[:,2],raw_y=v[:,3],visual_speed_mps=v[:,4]))
        if row['gps_available']:
            gp=used(ROOT/str(cat.loc[c,'gps_path']));raw,g=gps_frame(gp)
            assert sha(gp)==cat.loc[c,'gps_sha256']
            gv,gok=gps_interp(g,q);gspd=np.linalg.norm(gv[:,2:],axis=1)
            ok=(vid>=0)&gok;err=v[ok,4]-gspd[ok]
            sample['gps_speed_mps']=gspd;sample['speed_error_mps']=v[:,4]-gspd;sample['speed_valid']=ok
            row.update(speed_n=int(ok.sum()),speed_mae_mps=mae(err),speed_rmse_mps=rmse(err),
                       speed_bias_mps=float(err.mean()) if len(err) else None,
                       mean_visual_speed_mps=float(v[ok,4].mean()) if ok.any() else None,
                       mean_gps_speed_mps=float(gspd[ok].mean()) if ok.any() else None,
                       gps_speed_range_p95_p05_mps=float(np.percentile(gspd[ok],95)-np.percentile(gspd[ok],5)) if ok.any() else None,
                       sync_status='SUPPLIED_TIMELINE_NOT_INDEPENDENTLY_VERIFIED')
            if ok.any():
                row['timing_evidence_weak']=row['target_span_s']<3 or row['gps_speed_range_p95_p05_mps']<1
                gps_all=np.hypot(g.east_velocity,g.north_velocity).to_numpy()
                row['gps_full_record_speed_min_mps']=float(gps_all.min());row['gps_full_record_speed_max_mps']=float(gps_all.max())
                vm=row['mean_visual_speed_mps']
                row['mean_visual_speed_outside_full_gps_range_mps']=float(max(gps_all.min()-vm,vm-gps_all.max(),0))
            # Same endpoints for raw and stabilized one-second displacement length.
            lag=CFG['displacement_lag_s'];v2,id2=supported(gr,q+lag,cols);gv2,gok2=gps_interp(g,q+lag)
            valid=ok & gok2 & (id2==vid)
            ds=np.linalg.norm(v2[:,:2]-v[:,:2],axis=1);dr=np.linalg.norm(v2[:,2:4]-v[:,2:4],axis=1)
            dg=np.linalg.norm(gv2[:,:2]-gv[:,:2],axis=1)
            sample['disp_valid']=valid;sample['raw_disp_m']=dr;sample['smooth_disp_m']=ds;sample['gps_disp_m']=dg
            row.update(disp_n=int(valid.sum()),raw_disp_mae_m=mae(dr[valid]-dg[valid]),smooth_disp_mae_m=mae(ds[valid]-dg[valid]),
                       raw_disp_rmse_m=rmse(dr[valid]-dg[valid]),smooth_disp_rmse_m=rmse(ds[valid]-dg[valid]))
            # Compare position-derived speed using identical 0.5-second differences.
            h=CFG['position_velocity_difference_s']/2
            va,ia=supported(gr,q-h,cols);vb,ib=supported(gr,q+h,cols)
            ga,ka=gps_interp(g,q-h);gb,kb=gps_interp(g,q+h)
            valid=(ia>=0)&(ia==ib)&ka&kb
            raw_speed=np.linalg.norm(vb[:,2:4]-va[:,2:4],axis=1)/(2*h)
            smooth_speed=np.linalg.norm(vb[:,:2]-va[:,:2],axis=1)/(2*h)
            gps_chord=np.linalg.norm(gb[:,:2]-ga[:,:2],axis=1)/(2*h)
            sample['position_speed_valid']=valid;sample['raw_position_speed_mps']=raw_speed
            sample['smooth_position_speed_mps']=smooth_speed;sample['gps_position_speed_mps']=gps_chord
            row.update(position_speed_n=int(valid.sum()),raw_position_speed_mae_mps=mae(raw_speed[valid]-gps_chord[valid]),
                       smooth_position_speed_mae_mps=mae(smooth_speed[valid]-gps_chord[valid]))
            h=CFG['acceleration_difference_s']/2
            va,ia=supported(gr,q-h,cols);vb,ib=supported(gr,q+h,cols);ga,ka=gps_interp(g,q-h);gb,kb=gps_interp(g,q+h)
            valid=(ia>=0)&(ia==ib)&ka&kb
            acc=(vb[:,4]-va[:,4])/(2*h);gacc=(np.linalg.norm(gb[:,2:],axis=1)-np.linalg.norm(ga[:,2:],axis=1))/(2*h)
            sample['acc_valid']=valid;sample['visual_acc_mps2']=acc;sample['gps_acc_mps2']=gacc
            row.update(acc_n=int(valid.sum()),acc_mae_mps2=mae(acc[valid]-gacc[valid]),
                       visual_acc_rms_mps2=rmse(acc[valid]),gps_acc_rms_mps2=rmse(gacc[valid]))
            # Timing sensitivity is diagnostic, evaluated on one common sample set.
            offsets=CFG['timing_sensitivity_offsets_s'];shifted=[gps_interp(g,q+d) for d in offsets]
            common=(vid>=0)&np.logical_and.reduce([b for a,b in shifted])
            for offset,(vals,valid) in zip(offsets,shifted):
                e=v[common,4]-np.linalg.norm(vals[common,2:],axis=1)
                offset_rows.append(dict(clip=c,offset_s=offset,n=int(common.sum()),speed_mae_mps=mae(e),applied_to_tracks=False))
            native_errors=[]
            for s in raw.dropna(subset=['nearest_frame_index','nearest_frame_time_sec']).itertuples():
                if int(s.nearest_frame_index) in byframe:
                    native_errors.append(float(s.nearest_frame_time_sec)-byframe[int(s.nearest_frame_index)]['t_s'])
            supplied=np.hypot(raw.east_velocity,raw.north_velocity);processed=np.hypot(raw.processed_vx,raw.processed_vy)
            velocity_delta=float(np.nanmax(abs(supplied-processed)))
            integrated=np.trapz(g[['east_velocity','north_velocity']].to_numpy(),g.video_time_sec,axis=0)
            displacement=g[['east_m','north_m']].iloc[-1].to_numpy()-g[['east_m','north_m']].iloc[0].to_numpy()
            sync_rows.append(dict(clip=c,gps_rows=len(raw),gps_unique_times=len(g),
                gps_video_time_start_s=float(g.video_time_sec.iloc[0]),gps_video_time_end_s=float(g.video_time_sec.iloc[-1]),
                video_gps_epoch_offset_range_s=float((raw.epoch_time_sec-raw.video_time_sec).max()-(raw.epoch_time_sec-raw.video_time_sec).min()),
                gps_time_backsteps=int((raw.video_time_sec.diff()<0).sum()),gps_max_gap_s=float(g.video_time_sec.diff().max()),
                supplied_frame_vs_native_pts_max_abs_s=float(np.max(np.abs(native_errors))) if native_errors else None,
                processed_vs_original_speed_max_difference_mps=velocity_delta,
                gps_position_velocity_endpoint_difference_m=float(np.linalg.norm(integrated-displacement)),
                independently_verified_time_anchor=False))
            assert velocity_delta<1e-8
            checks.extend([f'{c}:gps_hash_unchanged',f'{c}:original_processed_speed_norm_agrees',f'{c}:same_endpoints_raw_smooth'])
        else:
            sample['speed_valid']=False
        sample['clip']=c;sample['site']=row['site'];all_samples.append(sample)
        (OUT/'series').mkdir(exist_ok=True);sample.to_csv(OUT/'series'/f'{c}.csv',index=False)
        rows.append(row)
    result=pd.DataFrame(rows).sort_values(['site','clip']);result.to_csv(OUT/'clip_metrics.csv',index=False)
    pd.DataFrame(sync_rows).to_csv(OUT/'gps_timing_audit.csv',index=False)
    pd.DataFrame(offset_rows).to_csv(OUT/'timing_sensitivity.csv',index=False)
    samples=pd.concat(all_samples,ignore_index=True);samples.to_csv(OUT/'all_samples.csv',index=False)
    site_rows=[]
    for site,g in list(result.groupby('site'))+[('All',result)]:
        ss=samples if site=='All' else samples[samples.site==site]
        valid=ss.speed_valid.fillna(False).astype(bool)
        errors=ss.loc[valid,'speed_error_mps'].to_numpy()
        site_rows.append(dict(site=site,videos=len(g),gps_videos=int(g.gps_available.sum()),speed_videos=int((g.speed_n>0).sum()),
            paired_videos=int((g.paired_observed_points>0).sum()),observed_frame_fraction_median=float(g.observed_frame_fraction.median()),
            supported_span_median_s=float(g.longest_supported_run_s.median()),observed_run_median_s=float(g.longest_observed_run_s.median()),
            speed_mae_clip_mean_mps=float(g.speed_mae_mps.mean()),speed_mae_clip_median_mps=float(g.speed_mae_mps.median()),
            speed_rmse_clip_mean_mps=float(g.speed_rmse_mps.mean()),speed_mae_pooled_mps=mae(errors),
            raw_disp_mae_clip_mean_m=float(g.raw_disp_mae_m.mean()),smooth_disp_mae_clip_mean_m=float(g.smooth_disp_mae_m.mean()),
            disp_videos=int((g.disp_n>0).sum()),raw_jitter_clip_median_m=float(g.raw_jitter_m.median()),
            smooth_jitter_clip_median_m=float(g.smooth_jitter_m.median()),
            raw_position_speed_mae_clip_mean_mps=float(g.raw_position_speed_mae_mps.mean()),
            smooth_position_speed_mae_clip_mean_mps=float(g.smooth_position_speed_mae_mps.mean()),
            acc_mae_clip_mean_mps2=float(g.acc_mae_mps2.mean())))
    site=pd.DataFrame(site_rows);site.to_csv(OUT/'site_summary.csv',index=False)
    selected=result[result['clip'].isin([x['clip'] for x in selection['cases']])]
    selected.to_csv(OUT/'six_cases.csv',index=False)
    for p,h in inputs.items(): assert sha(ROOT/p)==h
    assert len(result)==43 and result.gps_available.sum()==40
    checks+=['all_43_clips_evaluated','all_40_gps_files_included','selection_precedes_errors','all_inputs_unchanged']
    save(OUT/'input_manifest.json',inputs)
    save(OUT/'validation.json',dict(status='PASS',checks=len(checks),check_names=checks,
         absolute_location_accuracy_available=False,leader_gap_accuracy_available=False,
         note='Consistency checks do not establish independent GPS synchronization or metric ground truth.'))
    save(OUT/'summary.json',dict(sites=site_rows,cases=selection['cases'],
         observations=int(result.target_points.sum()),speed_videos=int((result.speed_n>0).sum()),
         independently_synchronized_videos=0,absolute_position_rmse_m=None,
         note='All GPS-referenced errors are conditional on supplied timestamps. Displacement is rotation/translation-invariant length error, not absolute position accuracy.'))
    print(site.to_string(index=False));print(selected[['clip','longest_supported_run_s','speed_mae_mps','smooth_disp_mae_m','acc_mae_mps2']].to_string(index=False))


def figures():
    results=csv(OUT/'clip_metrics.csv').set_index('clip');rec={r['clip']:r for r in records()}
    selected=read(OUT/'case_selection.json')['cases'];(OUT/'figures').mkdir(exist_ok=True);(OUT/'reviews').mkdir(exist_ok=True)
    from visualize_field_baseline import draw,project
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for case,ax in zip(selected,axes.flat):
        c=case['clip'];df=csv(OUT/'series'/f'{c}.csv');st=load_track(c);r=rec[c];row=results.loc[c]
        ax.plot(df.t_s,df.gps_speed_mps,color='black',label='GPS (supplied timing)',lw=1.8)
        ax.plot(df.t_s,df.visual_speed_mps,color='#007b9b',label='Stabilized track speed',lw=1.8)
        ax.set(title=f'{c} | speed MAE {row.speed_mae_mps:.2f} m/s',xlabel='Video time (s)',ylabel='Speed (m/s)');ax.grid(alpha=.25);ax.legend(fontsize=8)
        f,aa=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        aa[0,0].plot(st.x_det,st.y_det,'.-',color='#bb8045',alpha=.65,label='Associated detections')
        aa[0,0].plot(st.x,st.y,'-',color='#007b9b',label='Stabilized');aa[0,0].axis('equal');aa[0,0].set(xlabel='Road x (m)',ylabel='Road y (m)',title='Common camera coordinates')
        aa[0,1].plot(df.t_s,df.gps_speed_mps,'k-',label='GPS speed')
        aa[0,1].plot(df.t_s,df.visual_speed_mps,color='#007b9b',label='Tracker speed')
        aa[0,1].plot(df.t_s,df.raw_position_speed_mps,color='#bb8045',alpha=.6,label='Detection chord speed (0.5 s)')
        aa[0,1].set(xlabel='Video time (s)',ylabel='Speed (m/s)',title='Supplied timeline; no offset fitting')
        valid=df.disp_valid.fillna(False).astype(bool)
        aa[1,0].plot(df.loc[valid,'t_s'],df.loc[valid,'raw_disp_m']-df.loc[valid,'gps_disp_m'],color='#bb8045',label='Detection displacement error')
        aa[1,0].plot(df.loc[valid,'t_s'],df.loc[valid,'smooth_disp_m']-df.loc[valid,'gps_disp_m'],color='#007b9b',label='Stabilized displacement error')
        aa[1,0].axhline(0,color='gray',lw=.6);aa[1,0].set(xlabel='Interval start (s)',ylabel='Signed distance difference (m)',title='1-second displacement length')
        aa[1,1].plot(df.t_s,df.gps_acc_mps2,'k-',label='GPS-derived acceleration')
        aa[1,1].plot(df.t_s,df.visual_acc_mps2,color='#007b9b',label='Track-derived acceleration')
        aa[1,1].set(xlabel='Video time (s)',ylabel='Acceleration (m/s²)',title='Same 0.5-second speed difference')
        for a in aa.flat:a.legend(fontsize=8);a.grid(alpha=.25)
        f.suptitle(f'{c}: frozen reviewed trajectory diagnostics');f.savefig(OUT/'figures'/f'{c}.png',dpi=150);plt.close(f)
        cal=read(TR/'outputs/object_detection/camera-data'/f'{c}_{r["tag"]}'/'calibration_used.json')
        K=np.array(cal['K']);M=np.array(cal['lidar2cam']);dims=read(SOURCE/'clips'/c/'dimensions.json');tiles=[]
        for ix in np.unique(np.linspace(0,len(st)-1,4).astype(int)):
            s=st.iloc[ix].to_dict();fr=int(s['frame']);im=cv2.imread(str(TR/'data/camera-data'/c/'frames_all'/f'{fr:03d}.jpg'))
            d=dims[str(s['track_id'])];b=dict(s,l=d['length_m'],w=d['width_m'],h=d['height_m'])
            draw(im,b,K,M,(0,220,255),'SELF '+str(s['track_id']))
            uv=project(b,K,M);center=np.mean(uv,axis=0) if uv is not None else np.array([im.shape[1]/2,im.shape[0]/2])
            xx=int(np.clip(center[0]-320,0,max(0,im.shape[1]-640)));yy=int(np.clip(center[1]-180,0,max(0,im.shape[0]-360)))
            tile=im[yy:yy+360,xx:xx+640].copy();tile[:28]=0
            cv2.putText(tile,f'{c} f{fr} t={s["t_s"]:.2f}s',(8,20),0,.52,(255,255,255),1);tiles.append(tile)
        cv2.imwrite(str(OUT/'reviews'/f'{c}.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    fig.suptitle('Three-view trajectory speed check: supplied GPS timing, no fitted corrections',fontsize=14)
    fig.savefig(OUT/'figures'/'six_case_speeds.png',dpi=170);fig.savefig(OUT/'figures'/'six_case_speeds.pdf');plt.close(fig)
    print('Six image reviews and trajectory figures saved')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','evaluate','figures']);args=parser.parse_args()
    {'select':select,'evaluate':evaluate,'figures':figures}[args.stage]()
