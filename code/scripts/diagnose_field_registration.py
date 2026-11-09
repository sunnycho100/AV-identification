"""Reproduce the shared-fit failure without modifying GPS or calibration.

Per-clip fits and time-shift scans are diagnostics, never accuracy ground truth
or automatically accepted synchronization corrections.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_field_branches import ROOT, TR, CFG, read, save, target, interpolate
from grade_target_vs_gps import rigid_align
from site_error_model import enu_from_latlon

OUT = ROOT/'experiments/field_core_20261101/registration'

def rms(a):
    return float(np.sqrt(np.mean(np.sum(a*a, axis=1))))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    old = read(ROOT/'experiments/field_branches_20261101/gps_labels_enu/registration.json')
    origin = old['enu_origin_latlon']
    datasets = []
    for c in CFG['train']:
        tid, st = target(c)
        m = read(TR/'outputs/target_id'/c/'marker_matches.json')
        st = [s for s in st if 'x_det' in s and
              m['per_frame'][str(s['frame'])]['score'] >= m['score_min'] and
              any(r['start'] <= s['frame'] <= r['end'] for r in m['target_runs'])]
        raw = pd.read_csv(ROOT/cat.loc[c,'gps_path'])
        e, n = enu_from_latlon(raw.latitude, raw.longitude, *origin)
        g = pd.DataFrame(dict(video_time_sec=raw.video_time_sec, processed_x=e,
                             processed_y=n, processed_vx=raw.east_velocity,
                             processed_vy=raw.north_velocity)).dropna().groupby('video_time_sec',as_index=False).mean().sort_values('video_time_sec')
        v, ok = interpolate(g, [s['t_s'] for s in st])
        st = [s for s,k in zip(st,ok) if k]; v = v[ok]
        A = v[:,:2]; B = np.array([[s['x'],s['y']] for s in st])
        datasets.append(dict(clip=c, target=tid, states=st, gps=g, A=A, B=B, raw=raw))
    A = np.concatenate([d['A'] for d in datasets]); B = np.concatenate([d['B'] for d in datasets])
    R, t = rigid_align(A,B)
    assert abs(rms(A@R.T+t-B)-old['residual_rmse_m']) < 1e-7
    rows=[]; fig, axes=plt.subplots(1,3,figsize=(15,4.8))
    for d, color in zip(datasets, ['tab:blue','tab:orange']):
        a,b=d['A'],d['B']; rr,tt=rigid_align(a,b); st=d['states']; raw=d['raw']
        # This is rounding to a nearby video frame, not an independent sync check.
        quant = np.abs(raw.nearest_frame_time_error_sec.dropna())
        row=dict(clip=d['clip'],target_id=d['target'],frames=len(st),
                 time_interval_s=[st[0]['t_s'],st[-1]['t_s']],
                 frame_interval=[st[0]['frame'],st[-1]['frame']],
                 gps_centroid_enu_m=a.mean(0).tolist(),road_centroid_m=b.mean(0).tolist(),
                 shared_rmse_m=rms(a@R.T+t-b),separate_rigid_fit_rmse_m=rms(a@rr.T+tt-b),
                 separate_rotation_deg=float(np.degrees(np.arctan2(rr[1,0],rr[0,0]))),
                 separate_translation_m=tt.tolist(),
                 max_frame_quantization_error_s=float(quant.max()),
                 gps_velocity_displacement_consistency_m=float(np.linalg.norm(
                     np.array([d['gps'].processed_x.iloc[-1]-d['gps'].processed_x.iloc[0],d['gps'].processed_y.iloc[-1]-d['gps'].processed_y.iloc[0]])-
                     np.trapz(d['gps'][['processed_vx','processed_vy']].to_numpy(),d['gps'].video_time_sec,axis=0))))
        rows.append(row)
        pd.DataFrame(dict(frame=[s['frame'] for s in st],video_time_s=[s['t_s'] for s in st],
                          gps_e_m=a[:,0],gps_n_m=a[:,1],road_x_m=b[:,0],road_y_m=b[:,1],
                          shared_x_m=(a@R.T+t)[:,0],shared_y_m=(a@R.T+t)[:,1],
                          separate_x_m=(a@rr.T+tt)[:,0],separate_y_m=(a@rr.T+tt)[:,1])).to_csv(OUT/f"{d['clip']}_matched_points.csv",index=False)
        axes[0].plot(a[:,0],a[:,1],'.-',color=color,label=d['clip'])
        axes[1].plot(b[:,0],b[:,1],'-',color=color,label=d['clip']+' visual')
        axes[1].plot((a@R.T+t)[:,0],(a@R.T+t)[:,1],'--',color=color,label='shared fit')
        axes[2].plot([s['t_s'] for s in st],np.linalg.norm(a@rr.T+tt-b,axis=1),'.-',color=color,label=d['clip'])
    da=datasets[0]['A'].mean(0)-datasets[1]['A'].mean(0)
    db=datasets[0]['B'].mean(0)-datasets[1]['B'].mean(0)
    w=len(datasets[0]['A'])/len(A)
    lower=np.sqrt(w*(1-w))*abs(np.linalg.norm(da)-np.linalg.norm(db))
    # Fix AV timing to several diagnostic anchors. Optimize HV timing within
    # available rows, with all original matched frames retained (no extrapolation).
    scans=[]; av,hv=datasets
    hv_t=np.array([s['t_s'] for s in hv['states']])
    lo=float(hv['gps'].video_time_sec.iloc[0]-hv_t.min())
    hi=float(hv['gps'].video_time_sec.iloc[-1]-hv_t.max())
    for av_dt in [-1.,0.,1.]:
        av_t=np.array([s['t_s'] for s in av['states']])+av_dt
        av_v,valid=interpolate(av['gps'],av_t)
        if not valid.all(): continue
        def evaluate(dt):
            hv_v,ok=interpolate(hv['gps'],hv_t+dt)
            if not ok.all(): return (float('inf'),None)
            aa=np.concatenate([av_v[:,:2],hv_v[:,:2]])
            rr,tt=rigid_align(aa,B)
            return rms(aa@rr.T+tt-B),(rr,tt)
        grid=np.linspace(lo+1e-7,hi-1e-7,600);cost=[evaluate(x)[0] for x in grid]
        ix=int(np.argmin(cost));bounds=(grid[max(ix-1,0)],grid[min(ix+1,len(grid)-1)])
        opt=minimize_scalar(lambda dt:evaluate(dt)[0],bounds=bounds,method='bounded')
        scans.append(dict(av_offset_fixed_s=av_dt,hv_offset_fitted_s=float(opt.x),rmse_m=float(opt.fun),hv_offset_bounds_s=[lo,hi],accepted_for_sync=False))
    result=dict(status='REPRODUCED',source_height_m=16.26,source_tag='pilot_base',
                shared_rmse_m=rms(A@R.T+t-B),matched_frames=len(A),
                rotation=R.tolist(),translation=t.tolist(),enu_origin_latlon=origin,
                gps_centroid_separation_m=float(np.linalg.norm(da)),road_centroid_separation_m=float(np.linalg.norm(db)),
                rigid_invariant_rmse_lower_bound_m=float(lower),clips=rows,
                diagnostic_time_shift_scans=scans,
                conclusion='The shared fit is dominated by inconsistent between-clip longitudinal registration at the supplied timing. The data do not independently identify whether synchronization, source pairing, GPS absolute bias, or other registration errors caused it. Per-clip fits are in-sample shape diagnostics only.',
                proposed_action='Keep visual target/leader extraction independent of GPS registration. Verify image identity and source pairing, locate common physical crossing events and/or distinctive speed changes, then freeze per-clip timing and camera pose registration on training recordings. Never silently adopt fitted test offsets as accuracy.')
    save(OUT/'diagnosis.json',result)
    titles=['Matched GPS positions in shared ENU','Shared transformation vs. visual tracks','After a separate rigid fit per clip']
    for ax,title in zip(axes,titles): ax.set_title(title,fontsize=10);ax.grid(alpha=.25);ax.legend(fontsize=7)
    axes[0].set(xlabel='East (m)',ylabel='North (m)');axes[1].set(xlabel='Road x (m)',ylabel='Road y (m)')
    axes[2].set(xlabel='Video time (s)',ylabel='In-sample position residual (m)')
    fig.tight_layout();fig.savefig(OUT/'registration_diagnosis.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
