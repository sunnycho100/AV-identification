"""Recompute temporal-holdout errors and verify a known synthetic clock delay."""
import json
import numpy as np
import pandas as pd
import evaluate_time_alignment as A
import evaluate_trajectory_quality as Q
from workspace_paths import verify_recorded_input

checks=[]
def check(condition,name):
    if not condition:raise AssertionError(name)
    checks.append(name)


def main():
    for path,h in Q.read(A.OUT/'input_manifest.json').items():check(True,verify_recorded_input(path,h)+':'+path)
    rows=pd.read_csv(A.OUT/'clip_results.csv');dist=pd.read_csv(A.OUT/'displacement_results.csv')
    cat=pd.read_csv(A.ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    for row in rows.itertuples():
        if row.status!='ESTIMATED_DELAY':continue
        d=pd.read_csv(A.OUT/'series'/f'{row.clip}.csv');fit=d.is_fit.astype(bool);test=d.is_test.astype(bool)
        check(not (fit&test).any(),row.clip+':fit_test_disjoint')
        check(d.loc[test,'t_s'].min()-row.fit_cutoff_s>=A.CFG['guard_after_fit_s']-1e-7,row.clip+':guard')
        check(row.delay_lower_s-1e-7<=row.delay_s<=row.delay_upper_s+1e-7,row.clip+':delay_bounds')
        raw,g=Q.gps_frame(A.ROOT/cat.loc[row.clip,'gps_path'])
        for side,shift in [('before',0.),('after',row.delay_s)]:
            t=d.t_s.to_numpy()+shift
            check(t.min()>=g.video_time_sec.min()-1e-7 and t.max()<=g.video_time_sec.max()+1e-7,row.clip+':'+side+'_gps_support')
            ve=np.interp(t,g.video_time_sec,g.east_velocity);vn=np.interp(t,g.video_time_sec,g.north_velocity)
            speed=np.sqrt(ve*ve+vn*vn)
            check(np.allclose(speed,d[f'gps_{side}_speed_mps'],atol=1e-9),row.clip+':'+side+'_reference')
            error=d.loc[test,'visual_speed_mps'].to_numpy()-speed[test]
            check(np.isclose(abs(error).mean(),getattr(row,f'test_speed_{side}_mae_mps')),row.clip+':'+side+'_mae')
            check(np.isclose(np.sqrt(np.mean(error**2)),getattr(row,f'test_speed_{side}_rmse_mps')),row.clip+':'+side+'_rmse')
        tau,*_=A.fit_delay(g,d.loc[fit,'t_s'].to_numpy(),d.loc[fit,'visual_speed_mps'].to_numpy(),row.delay_lower_s,row.delay_upper_s)
        check(abs(tau-row.delay_s)<1e-7,row.clip+':delay_reproduced_without_test_values')
        for h in A.CFG['displacement_horizons_s']:
            mask=d[f'disp{h}_test'].astype(bool);r=dist[(dist['clip']==row.clip)&(dist.partition=='test')&(dist.horizon_s==h)].iloc[0]
            check(mask.sum()==r.n,row.clip+f':{h}s_n')
            check((d.loc[mask,'t_s']>=row.test_start_s-1e-7).all(),row.clip+f':{h}s_test_start')
            for side in ['before','after']:
                if mask.any():
                    err=d.loc[mask,f'disp{h}_visual_m']-d.loc[mask,f'disp{h}_{side}_m']
                    check(np.isclose(abs(err).mean(),r[side+'_mae_m']),row.clip+f':{h}s_'+side+'_mae')
                else:check(pd.isna(r[side+'_mae_m']),row.clip+f':{h}s_'+side+'_missing_not_zero')
    sites=pd.read_csv(A.OUT/'site_summary.csv')
    for r in sites.itertuples():
        g=rows[rows.status=='ESTIMATED_DELAY'];g=g if r.site=='All' else g[g.site==r.site]
        check(len(g)==r.aligned_recordings,r.site+':same_cohort')
        for side in ['before','after']:
            check(np.isclose(g[f'test_speed_{side}_mae_mps'].mean(),getattr(r,f'speed_{side}_mae_mps')),r.site+':equal_clip_'+side)
        for h in A.CFG['displacement_horizons_s']:
            dd=dist[(dist.partition=='test')&(dist.horizon_s==h)&(dist.n>0)&dist['clip'].isin(g['clip'])]
            check(len(dd)==getattr(r,f'disp{h}_recordings'),r.site+f':{h}s_cohort')
    # Known positive sensor delay, rich non-periodic speed variation.
    true_delay=1.27;gt=np.arange(0,20,.008)
    motion=lambda t:12+1.2*np.sin(.7*t)+.7*np.sin(2.2*t)+.05*t
    gps=pd.DataFrame(dict(video_time_sec=gt,east_m=gt*0,north_m=gt*0,east_velocity=motion(gt-true_delay),north_velocity=gt*0))
    t=np.arange(4,8,.1);v=motion(t)
    tau,*_=A.fit_delay(gps,t,v,-3,3)
    check(abs(tau-true_delay)<.002,'known_positive_delay_sign_and_recovery')
    later=np.arange(9,12,.1);values,ok=A.interpolate_speed(gps,later+tau)
    check(ok.all() and np.mean(abs(values-motion(later)))<.002,'known_delay_later_interval_accuracy')
    gps.east_velocity=14;flat=np.full(len(t),12.)
    tau,cost,grid,profile,diag=A.fit_delay(gps,t,flat,-3,3)
    check(abs(tau)<1e-8,'constant_speed_delay_unidentifiable_tie_zero')
    check(np.isclose(cost,4),'speed_bias_not_removed')
    check(diag['near_optimal_delay_span_s']>=5.9,'flat_profile_marked_broad')
    check(len(rows)==43 and rows.gps_available.sum()==40,'all43_included40_gps')
    Q.save(A.OUT/'independent_validation.json',dict(status='PASS',checks=len(checks),check_names=checks,
        note='Algorithm and data consistency checks; uncertain fitted delays do not become independently verified clock offsets.'))
    print('PASS',len(checks),'independent checks')


if __name__=='__main__':main()
