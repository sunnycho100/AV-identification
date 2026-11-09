"""Independently recheck saved metrics and measurement semantics."""
import hashlib,json,sys
from pathlib import Path
from workspace_paths import verify_recorded_input
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/field_trajectory_quality_20261102'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
checks=[]
def check(ok,name):
    if not ok:raise AssertionError(name)
    checks.append(name)


def main():
    for path,h in read(OUT/'input_manifest.json').items():
        check(True,verify_recorded_input(path,h)+':'+path)
    clips=pd.read_csv(OUT/'clip_metrics.csv').set_index('clip');site=pd.read_csv(OUT/'site_summary.csv').set_index('site')
    for c,r in clips.iterrows():
        df=pd.read_csv(OUT/'series'/f'{c}.csv')
        check(r.target_points<=r.review_frames,c+':coverage_denominator')
        check(r.longest_observed_run_s<=r.longest_supported_run_s+1e-6,c+':continuity_order')
        if not r.gps_available:check(r.speed_n==0 and pd.isna(r.speed_mae_mps),c+':missing_gps_not_zero');continue
        valid=df.speed_valid.fillna(False).astype(bool);err=df.loc[valid,'visual_speed_mps']-df.loc[valid,'gps_speed_mps']
        check(len(err)==r.speed_n,c+':speed_n')
        check(np.isclose(abs(err).mean(),r.speed_mae_mps,atol=1e-10),c+':mae_recomputed')
        check(np.isclose(np.sqrt((err**2).mean()),r.speed_rmse_mps,atol=1e-10),c+':rmse_recomputed')
        valid=df.disp_valid.fillna(False).astype(bool)
        check(valid.sum()==r.disp_n,c+':displacement_n')
        if valid.any():
            for method in ['raw','smooth']:
                error=df.loc[valid,method+'_disp_m']-df.loc[valid,'gps_disp_m']
                check(np.isclose(abs(error).mean(),r[method+'_disp_mae_m'],atol=1e-10),c+':'+method+'_displacement')
        valid=df.acc_valid.fillna(False).astype(bool)
        check(valid.sum()==r.acc_n,c+':acceleration_n')
        if valid.any():
            error=df.loc[valid,'visual_acc_mps2']-df.loc[valid,'gps_acc_mps2']
            check(np.isclose(abs(error).mean(),r.acc_mae_mps2,atol=1e-10),c+':acceleration')
    for name,r in site.iterrows():
        g=clips if name=='All' else clips[clips.site==name]
        check(np.isclose(g.speed_mae_mps.mean(),r.speed_mae_clip_mean_mps),name+':equal_clip_weight')
        check(np.isclose(g.smooth_disp_mae_m.mean(),r.smooth_disp_mae_clip_mean_m),name+':displacement_weight')
        check(int((g.disp_n>0).sum())==r.disp_videos,name+':displacement_denominator')
    sensitivity=pd.read_csv(OUT/'timing_sensitivity.csv')
    check((sensitivity.groupby('clip').n.nunique()==1).all(),'same_samples_all_time_offsets')
    check(not sensitivity.applied_to_tracks.any(),'no_fitted_time_correction')
    selection=read(OUT/'case_selection.json')
    check(len(selection['cases'])==6 and selection['selection_before_gps_errors'],'six_preselected_cases')
    for item in selection['cases']:
        check(not item['training_exposed'],item['clip']+':not_listed_in_detector_adaptation_training')
    # Known geometry example: displacement length is translation/rotation invariant,
    # but not scale invariant, so it cannot be reported as absolute location error.
    a=np.array([100.,-20.]);b=a+np.array([3.,4.]);rot=np.array([[0.,-1.],[1.,0.]])
    check(np.isclose(np.linalg.norm(b-a),5),'known_displacement_5m')
    check(np.isclose(np.linalg.norm((b@rot+1000)-(a@rot+1000)),5),'rigid_invariance')
    check(np.isclose(np.linalg.norm(2*b-2*a),10),'scale_error_not_removed')
    import evaluate_trajectory_quality as e
    track=pd.DataFrame({'frame':[0,1,30,31],'t_s':[0.,.1,3.,3.1],'x':[0.,1.,30.,31.],'record_id':['a']*4,'track_id':['1']*4})
    values,ids=e.supported(e.pieces(track),[.05,1.,3.05],['x'])
    check(np.isnan(values[1,0]) and ids[1]==-1,'no_interpolation_over_long_gap')
    check(np.allclose(values[[0,2],0],[.5,30.5]),'known_linear_interpolation')
    report=dict(status='PASS',checks=len(checks),check_names=checks,
                scope='Independent recomputation and geometry/gap boundary checks, not physical ground-truth certification')
    (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS',len(checks),'independent checks')


if __name__=='__main__':main()
