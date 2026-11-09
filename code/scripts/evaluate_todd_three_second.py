"""Post-selection GPS checks with previously frozen per-record delay; no refitting."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import evaluate_trajectory_quality as Q
ROOT=Q.ROOT
OUT=ROOT/'experiments/field_todd_3s_20261102'


def main():
    meta=pd.read_csv(OUT/'windows.csv');cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    delays=pd.read_csv(ROOT/'experiments/field_time_alignment_20261102/clip_results.csv').set_index('clip')
    rows=[];fig,axes=plt.subplots(len(meta),2,figsize=(11,2.6*len(meta)),squeeze=False)
    dest=OUT/'quality';dest.mkdir(exist_ok=True)
    for k,w in enumerate(meta.to_dict('records')):
        clip=w['clip'];s=pd.read_csv(OUT/'clips'/clip/'selected_3s.csv');t=s.t_s.to_numpy();v=s.ego_speed_mps.to_numpy();tr=delays.loc[clip]
        row=dict(clip=clip,points=31,gps_status=tr.status)
        a,b=axes[k];a.plot(t-t[0],v,label='Extracted',color='#bb4b25');a.set_title(clip+' | speed');a.set_ylabel('m/s')
        dist=cumulative_trapezoid(v,t,initial=0);b.plot(t-t[0],dist,color='#bb4b25',label='Extracted');b.set_title('Travelled distance from window start');b.set_ylabel('m')
        if tr.status=='ESTIMATED_DELAY' and pd.notna(cat.loc[clip,'gps_path']):
            _,gps=Q.gps_frame(ROOT/cat.loc[clip,'gps_path']);ref,ok=Q.gps_interp(gps,t+float(tr.delay_s));gv=np.linalg.norm(ref[:,2:],axis=1)
            err=v-gv;test=ok&(t>=tr.test_start_s)
            s['gps_speed_mps']=gv;s['previous_temporal_test_region']=test
            row.update(delay_s=float(tr.delay_s),delay_identifiability=str(tr.identifiability_flags),gps_points=int(ok.sum()),
                selected_speed_mae_mps=Q.mae(err[ok]),selected_speed_rmse_mps=Q.rmse(err[ok]),
                previous_test_region_points=int(test.sum()),previous_test_region_speed_mae_mps=Q.mae(err[test]))
            if ok.all():
                gd=cumulative_trapezoid(gv,t,initial=0);s['gps_travelled_m']=gd;s['extracted_travelled_m']=dist
                row.update(three_second_travel_error_m=float(dist[-1]-gd[-1]),three_second_abs_travel_error_m=float(abs(dist[-1]-gd[-1])),
                           displacement_curve_mae_m=Q.mae(dist-gd))
                b.plot(t-t[0],gd,color='#247ba0',label='GPS, fixed delay')
            a.plot(t-t[0],gv,color='#247ba0',label='GPS, fixed delay')
            a.text(.02,.94,f'MAE {row["selected_speed_mae_mps"]:.2f} m/s; delay {row["delay_s"]:.2f}s',transform=a.transAxes,va='top',fontsize=9)
        else:a.text(.03,.9,'No eligible frozen-delay GPS comparison',transform=a.transAxes,fontsize=9)
        for ax in [a,b]:ax.grid(alpha=.2);ax.set_xlabel('Elapsed time (s)');ax.legend(fontsize=8,loc='lower right')
        rows.append(row);s.to_csv(dest/f'{clip}.csv',index=False)
    fig.suptitle('Selected 3-second Todd SE trajectories — descriptive GPS checks\nPreviously frozen delay; zero distance at window start; no speed/scale fitting',fontsize=13,y=1.001)
    fig.tight_layout();fig.savefig(dest/'selected_3s_gps.png',dpi=150,bbox_inches='tight');fig.savefig(dest/'selected_3s_gps.pdf',bbox_inches='tight');plt.close(fig)
    table=pd.DataFrame(rows);table.to_csv(dest/'clip_metrics.csv',index=False)
    Q.save(dest/'summary.json',dict(selected_videos=len(meta),gps_comparable_videos=int(table.selected_speed_mae_mps.notna().sum()),
        video_macro_speed_mae_mps=float(table.selected_speed_mae_mps.mean()),
        video_macro_3s_absolute_travel_error_m=float(table.three_second_abs_travel_error_m.mean()),
        temporal_test_region_video_macro_speed_mae_mps=float(table.previous_test_region_speed_mae_mps.mean()),
        scope='Descriptive check of windows selected without GPS/classification. Delays frozen from prior temporal calibration; full selected windows may overlap its fitting region. Weakly identified delays remain flagged. No position translation, rotation, scale or speed bias fit.',
        no_new_delay_fitting=True,no_gps_window_selection=True))
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
