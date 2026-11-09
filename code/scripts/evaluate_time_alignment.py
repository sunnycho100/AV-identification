"""One constant delay per recording; fit early, evaluate later on fixed support."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_trajectory_quality as Q

ROOT=Q.ROOT
CONFIG=ROOT/'configs/field_time_alignment.json'
CFG=Q.read(CONFIG)
OUT=ROOT/'experiments'/CFG['experiment']
OLD=ROOT/CFG['source_quality']
SOURCE=ROOT/CFG['source_tracks']
NAMES={'S_Seminole_Hwy':'Seminole NW','T_Todd_Dr':'Todd SE','V_Verona_Rd':'Verona SE','All':'All'}


def interpolate_speed(g,t):
    vals,ok=Q.gps_interp(g,t)
    return np.linalg.norm(vals[:,2:],axis=1),ok


def fit_delay(g,t,v,lo,hi):
    step=CFG['coarse_delay_step_s']
    grid=np.unique(np.r_[np.arange(np.ceil(lo/step)*step,hi+1e-9,step),lo,hi,0.])
    grid=grid[(grid>=lo-1e-9)&(grid<=hi+1e-9)]
    def mse(delay):
        ref,ok=interpolate_speed(g,t+delay)
        if not ok.all():return float('inf')
        return float(np.mean((v-ref)**2))
    costs=np.array([mse(d) for d in grid]);minimum=costs.min()
    if not np.isfinite(minimum):raise ValueError('No fully supported candidate')
    tied=np.where(costs<=minimum+1e-10)[0];ix=int(tied[np.argmin(abs(grid[tied]))])
    candidates=[(float(grid[ix]),float(costs[ix]))]
    a,b=grid[max(ix-1,0)],grid[min(ix+1,len(grid)-1)]
    if b-a>1e-8:
        opt=minimize_scalar(mse,bounds=(a,b),method='bounded',options={'xatol':CFG['refinement_tolerance_s']})
        candidates.append((float(opt.x),float(opt.fun)))
    best=min(x[1] for x in candidates);delay,cost=min([x for x in candidates if x[1]<=best+1e-10],key=lambda x:abs(x[0]))
    ref,ok=interpolate_speed(g,t+delay);assert ok.all()
    near=grid[np.sqrt(costs)<=np.sqrt(cost)+CFG['reliability_flags']['near_optimal_rmse_tolerance_mps']]
    return delay,cost,grid,costs,dict(
        near_optimal_delay_min_s=float(min(near.min(),delay)),near_optimal_delay_max_s=float(max(near.max(),delay)),
        near_optimal_delay_span_s=float(max(near.max(),delay)-min(near.min(),delay)),
        fit_visual_speed_range_mps=float(np.percentile(v,95)-np.percentile(v,5)),
        fit_gps_speed_range_mps=float(np.percentile(ref,95)-np.percentile(ref,5)))


def flags(row):
    r=CFG['reliability_flags'];result=[]
    if row['fit_span_s']<r['fit_span_below_s']:result.append('short_fit_interval')
    if row['fit_visual_speed_range_mps']<r['fit_visual_p95_p05_below_mps']:result.append('weak_visual_speed_variation')
    if row['fit_gps_speed_range_mps']<r['fit_gps_p95_p05_below_mps']:result.append('weak_gps_speed_variation')
    if row['near_optimal_delay_span_s']>r['near_optimal_delay_span_above_s']:result.append('broad_delay_minimum')
    if min(row['delay_s']-row['delay_lower_s'],row['delay_upper_s']-row['delay_s'])<r['boundary_distance_below_s']:result.append('delay_at_search_boundary')
    return result


def metrics(a,b):
    error=np.asarray(a)-np.asarray(b)
    return dict(mae=Q.mae(error),rmse=Q.rmse(error),bias=float(error.mean()) if len(error) else None)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ['series','profiles','figures','tables']: (OUT/d).mkdir(exist_ok=True)
    Q.save(OUT/'protocol.json',CFG)
    inputs={}
    def use(path):inputs[str(path.relative_to(ROOT))]=Q.sha(path);return path
    use(CONFIG);use(ROOT/'scripts/evaluate_trajectory_quality.py');use(ROOT/'scripts/evaluate_time_alignment.py')
    use(ROOT/'data/annotations/leader_review_decisions.json');use(ROOT/'catalogs/field_clips.csv');use(OLD/'case_selection.json')
    old=Q.csv(use(OLD/'clip_metrics.csv')).set_index('clip');cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    rows=[];distances=[];checks=[]
    for clip,r in old.iterrows():
        row=dict(clip=clip,site=r.site,label=r.label,training_exposed=bool(r.training_exposed),
                 gps_available=bool(r.gps_available),status='GPS_UNAVAILABLE',old_full_speed_mae_mps=None if pd.isna(r.speed_mae_mps) else float(r.speed_mae_mps))
        if not r.gps_available:rows.append(row);continue
        df=Q.load_track(clip);use(SOURCE/'clips'/clip/'target_trajectory.csv')
        use(SOURCE/'clips'/clip/'leader_trajectory.csv');use(SOURCE/'clips'/clip/'following_pairs.csv')
        _,g=Q.gps_frame(use(ROOT/cat.loc[clip,'gps_path']))
        groups=Q.pieces(df,CFG['maximum_visual_gap_s'])
        q=np.arange(np.ceil(df.t_s.min()*10)/10,df.t_s.max()+1e-8,.1)
        visual,ids=Q.supported(groups,q,['x','y','speed_mps']);gps0,ok=Q.gps_interp(g,q)
        keep=(ids>=0)&ok;q=q[keep];visual=visual[keep];ids=ids[keep];gps0=gps0[keep]
        cutoff=float(df.t_s.min()+CFG['fit_fraction_of_observed_time_span']*(df.t_s.max()-df.t_s.min()))
        fit=q<=cutoff+1e-8;test=q>=cutoff+CFG['guard_after_fit_s']-1e-8
        lo=max(-CFG['maximum_abs_delay_s'],float(g.video_time_sec.iloc[0]-q.min())+1e-8)
        hi=min(CFG['maximum_abs_delay_s'],float(g.video_time_sec.iloc[-1]-q.max())-1e-8)
        # Endpoints on GPS support may require accepting zero within rounding tolerance.
        lo=min(lo,0.);hi=max(hi,0.)
        row.update(fit_n=int(fit.sum()),test_n=int(test.sum()),fit_cutoff_s=cutoff,
                   test_start_s=float(q[test].min()) if test.any() else None,
                   fit_span_s=float(np.ptp(q[fit])) if fit.any() else 0.,delay_lower_s=lo,delay_upper_s=hi)
        series=pd.DataFrame(dict(t_s=q,piece=ids,x=visual[:,0],y=visual[:,1],visual_speed_mps=visual[:,2],
                                 gps_before_speed_mps=np.linalg.norm(gps0[:,2:],axis=1),is_fit=fit,is_test=test))
        if fit.sum()<CFG['minimum_fit_samples'] or test.sum()<CFG['minimum_test_samples'] or hi-lo<.04:
            row['status']='INSUFFICIENT_TEMPORAL_SUPPORT';rows.append(row)
            series.to_csv(OUT/'series'/f'{clip}.csv',index=False);continue
        tau,cost,grid,profile,diagnostics=fit_delay(g,q[fit],visual[fit,2],lo,hi)
        row.update(status='ESTIMATED_DELAY',delay_s=tau,**diagnostics)
        issues=flags(row);row['identifiability_flags']=';'.join(issues);row['delay_informative']=not issues
        shifted,valid=Q.gps_interp(g,q+tau);assert valid.all()
        gps1=np.linalg.norm(shifted[:,2:],axis=1);series['gps_after_speed_mps']=gps1;series['gps_query_after_s']=q+tau
        for label,mask in [('fit',fit),('test',test),('whole',np.ones(len(q),bool))]:
            for side,ref in [('before',series.gps_before_speed_mps.to_numpy()),('after',gps1)]:
                m=metrics(visual[mask,2],ref[mask])
                row.update({f'{label}_speed_{side}_{k}_mps':v for k,v in m.items()})
        assert row['fit_speed_after_rmse_mps']<=row['fit_speed_before_rmse_mps']+1e-6
        # The whole-interval fit is explicitly an optimistic diagnostic, not the test result.
        oracle,ocost,_,_,_=fit_delay(g,q,visual[:,2],lo,hi)
        og,valid=interpolate_speed(g,q+oracle);assert valid.all()
        row['in_sample_delay_s']=oracle;row['in_sample_speed_mae_mps']=Q.mae(visual[:,2]-og)
        row['in_sample_speed_rmse_mps']=float(np.sqrt(ocost))
        row['test_speed_mae_improvement_mps']=row['test_speed_before_mae_mps']-row['test_speed_after_mae_mps']
        for horizon in CFG['displacement_horizons_s']:
            end,eid=Q.supported(groups,q+horizon,['x','y','speed_mps'])
            before_end,bok=Q.gps_interp(g,q+horizon);after_end,aok=Q.gps_interp(g,q+tau+horizon)
            eligible=(eid==ids)&bok&aok&np.isfinite(end[:,0])
            # The primary mask starts after fitting; both endpoints remain held out.
            for partition,base_mask in [('test',test),('whole',np.ones(len(q),bool))]:
                mask=eligible&base_mask
                vs=np.linalg.norm(end[:,:2]-visual[:,:2],axis=1)
                d0=np.linalg.norm(before_end[:,:2]-gps0[:,:2],axis=1)
                d1=np.linalg.norm(after_end[:,:2]-shifted[:,:2],axis=1)
                m0=metrics(vs[mask],d0[mask]);m1=metrics(vs[mask],d1[mask])
                distances.append(dict(clip=clip,site=r.site,partition=partition,horizon_s=horizon,n=int(mask.sum()),
                    before_mae_m=m0['mae'],after_mae_m=m1['mae'],before_rmse_m=m0['rmse'],after_rmse_m=m1['rmse'],
                    informative_delay=not issues,delay_s=tau))
                if partition=='test':
                    series[f'disp{horizon}_test']=mask;series[f'disp{horizon}_visual_m']=vs
                    series[f'disp{horizon}_before_m']=d0;series[f'disp{horizon}_after_m']=d1
        pd.DataFrame(dict(delay_s=grid,fit_rmse_mps=np.sqrt(profile))).to_csv(OUT/'profiles'/f'{clip}.csv',index=False)
        series.to_csv(OUT/'series'/f'{clip}.csv',index=False)
        checks += [f'{clip}:no_gps_extrapolation',f'{clip}:disjoint_fit_test',f'{clip}:fit_objective_not_worse',f'{clip}:same_test_support']
        rows.append(row)
    clips=pd.DataFrame(rows);clips.to_csv(OUT/'clip_results.csv',index=False)
    distance=pd.DataFrame(distances);distance.to_csv(OUT/'displacement_results.csv',index=False)
    summaries=[]
    for name,cg in list(clips.groupby('site'))+[('All',clips)]:
        fitted=cg[cg.status=='ESTIMATED_DELAY'];tested=distance[(distance.partition=='test')&distance['clip'].isin(fitted['clip'])]
        sr=dict(site=name,all_recordings=len(cg),gps_recordings=int(cg.gps_available.sum()),aligned_recordings=len(fitted),
                informative_delays=int(fitted.delay_informative.sum()),
                speed_before_mae_mps=float(fitted.test_speed_before_mae_mps.mean()),speed_after_mae_mps=float(fitted.test_speed_after_mae_mps.mean()),
                speed_before_rmse_mps=float(fitted.test_speed_before_rmse_mps.mean()),speed_after_rmse_mps=float(fitted.test_speed_after_rmse_mps.mean()),
                speed_improved_recordings=int((fitted.test_speed_mae_improvement_mps>1e-6).sum()),
                speed_worsened_recordings=int((fitted.test_speed_mae_improvement_mps < -1e-6).sum()),
                whole_before_mae_mps=float(fitted.whole_speed_before_mae_mps.mean()),whole_after_fixed_delay_mae_mps=float(fitted.whole_speed_after_mae_mps.mean()),
                optimistic_in_sample_mae_mps=float(fitted.in_sample_speed_mae_mps.mean()))
        for h in CFG['displacement_horizons_s']:
            z=tested[(tested.horizon_s==h)&(tested.n>0)]
            sr[f'disp{h}_recordings']=len(z);sr[f'disp{h}_before_mae_m']=float(z.before_mae_m.mean()) if len(z) else None
            sr[f'disp{h}_after_mae_m']=float(z.after_mae_m.mean()) if len(z) else None
        summaries.append(sr)
    site=pd.DataFrame(summaries);site.to_csv(OUT/'site_summary.csv',index=False)
    for path,h in inputs.items():assert Q.sha(ROOT/path)==h
    Q.save(OUT/'input_manifest.json',inputs)
    Q.save(OUT/'validation.json',dict(status='PASS',checks=len(checks),check_names=checks,
          source_tracks_and_gps_unchanged=True,independent_hardware_sync_verified=False))
    Q.save(OUT/'summary.json',dict(sites=summaries,all_recordings=43,
        note='Estimated delays fit on early interval only; main metrics on later interval with same before/after support. Weak estimates retained and flagged; whole-interval optimized alignment is diagnostic only.'))
    print(site.to_string(index=False),flush=True)
    print(clips[['clip','status','delay_s','test_speed_before_mae_mps','test_speed_after_mae_mps','identifiability_flags']].to_string(index=False),flush=True)
    figures(clips)


def figures(clips):
    selected=Q.read(OLD/'case_selection.json')['cases'];byclip=clips.set_index('clip')
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for case,ax in zip(selected,axes.flat):
        c=case['clip'];d=pd.read_csv(OUT/'series'/f'{c}.csv');r=byclip.loc[c]
        for i, (_, segment) in enumerate(d.groupby('piece', sort=False)):
            ax.plot(segment.t_s,segment.visual_speed_mps,color='#007b9b',lw=2,
                    label='Video trajectory' if i == 0 else None)
        ax.plot(d.t_s,d.gps_before_speed_mps,'--',color='#888888',label='GPS before')
        if r.status=='ESTIMATED_DELAY':
            ax.plot(d.t_s,d.gps_after_speed_mps,color='#c16c26',label='GPS after fixed delay')
            ax.axvspan(d.t_s.min(),r.fit_cutoff_s,color='#dce6f0',alpha=.5)
            ax.axvline(r.test_start_s,color='#555555',lw=.8,linestyle=':')
            title=f'{c}: delay {r.delay_s:+.2f} s | test MAE {r.test_speed_before_mae_mps:.2f} -> {r.test_speed_after_mae_mps:.2f}'
        else:title=f'{c}: insufficient support for split alignment'
        ax.set(title=title,xlabel='Native video time (s)',ylabel='Speed (m/s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed-delay alignment: shaded interval fits delay; later interval evaluates error',fontsize=13)
    fig.savefig(OUT/'figures'/'six_case_alignment.png',dpi=160);fig.savefig(OUT/'figures'/'six_case_alignment.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for case,ax in zip(selected,axes.flat):
        c=case['clip'];p=OUT/'profiles'/f'{c}.csv';r=byclip.loc[c]
        if p.exists():
            d=pd.read_csv(p);ax.plot(d.delay_s,d.fit_rmse_mps,color='#007b9b');ax.axvline(r.delay_s,color='#c16c26',label=f'Fitted {r.delay_s:+.2f} s');ax.legend(fontsize=8)
        ax.set(title=c,xlabel='GPS query delay (s)',ylabel='Fit-interval speed RMSE (m/s)');ax.grid(alpha=.25)
    fig.suptitle('Delay profiles from calibration interval only; broad/boundary minima remain uncertain')
    fig.savefig(OUT/'figures'/'six_delay_profiles.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');run()
