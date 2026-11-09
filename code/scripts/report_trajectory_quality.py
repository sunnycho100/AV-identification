"""Produce paper-ready tables and transparent workbook inputs from frozen metrics."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/field_trajectory_quality_20261102'
NAMES={'S_Seminole_Hwy':'Seminole NW','T_Todd_Dr':'Todd SE','V_Verona_Rd':'Verona SE','All':'All'}


def main():
    site=pd.read_csv(OUT/'site_summary.csv');clips=pd.read_csv(OUT/'clip_metrics.csv')
    cases=pd.read_csv(OUT/'six_cases.csv');timing=pd.read_csv(OUT/'gps_timing_audit.csv')
    offsets=pd.read_csv(OUT/'timing_sensitivity.csv')
    catalog=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    for df in [site,clips,cases]:df['site']=df.site.map(NAMES)
    clips['gps_source']=[str(catalog.loc[c,'gps_path']) if pd.notna(catalog.loc[c,'gps_path']) else 'GPS unavailable' for c in clips['clip']]
    clips['trajectory_source']=['experiments/field_review_test_20261102/clips/'+c+'/target_trajectory.csv' for c in clips['clip']]
    data=dict(sites=json.loads(site.to_json(orient='records')),clips=json.loads(clips.to_json(orient='records')),
              cases=json.loads(cases.to_json(orient='records')),timing=json.loads(timing.to_json(orient='records')),
              offsets=json.loads(offsets.to_json(orient='records')))
    (OUT/'workbook_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tables=OUT/'tables';tables.mkdir(exist_ok=True)
    table1=site[['site','videos','speed_videos','observed_frame_fraction_median','supported_span_median_s','speed_mae_clip_mean_mps','disp_videos','smooth_disp_mae_clip_mean_m']]
    table1.to_csv(tables/'trajectory_quality_by_view.csv',index=False)
    table2=site[['site','raw_position_speed_mae_clip_mean_mps','smooth_position_speed_mae_clip_mean_mps','raw_disp_mae_clip_mean_m','smooth_disp_mae_clip_mean_m','raw_jitter_clip_median_m','smooth_jitter_clip_median_m']]
    table2.to_csv(tables/'stabilization_comparison.csv',index=False)
    lines=[r'\begin{table}[pos=tbp]',r'\centering',r'\caption{Reviewed trajectory coverage and GPS-referenced motion errors. GPS timing is supplied, not independently verified. Errors are means of per-recording MAEs; coverage and span are medians. The displacement metric measures the length of a one-second displacement, not absolute position error.}',r'\label{tab:reviewed-motion-quality}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'View & Videos & $N_v$ & Obs. (\%) & Span (s) & $E_v$ (m/s) & $N_d$ & $E_d$ (m) \\',r'\midrule']
    for r in table1.itertuples(index=False):
        lines.append(f'{r.site} & {r.videos} & {r.speed_videos} & {100*r.observed_frame_fraction_median:.1f} & {r.supported_span_median_s:.2f} & {r.speed_mae_clip_mean_mps:.2f} & {r.disp_videos} & {r.smooth_disp_mae_clip_mean_m:.2f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.97\linewidth}\footnotesize $N_v$/$N_d$: recordings contributing speed/displacement errors. Obs.: detection-supported frames divided by reviewed frames after the explicit ego exit cutoff. Span: longest interval with consecutive observation gaps at most 0.2~s; this permits bounded interpolation and is not a fully observed run. $E_v$: stabilized tracker speed MAE; $E_d$: stabilized one-second displacement-length MAE.\end{minipage}',r'\end{table}']
    (tables/'reviewed_motion_quality.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table}[pos=tbp]',r'\centering',r'\caption{Associated detections (D) versus stabilized positions (S), evaluated on the same supported timestamps. $E_{v,0.5}$ is the error of position-derived speed over 0.5~s, distinct from tracker speed in Table~\ref{tab:reviewed-motion-quality}. Jitter is a time-weighted three-point position residual, a smoothness proxy without independent ground truth.}',r'\label{tab:reviewed-stabilization}',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r' & \multicolumn{2}{c}{$E_{v,0.5}$ (m/s)} & \multicolumn{2}{c}{$E_d$ (m)} & \multicolumn{2}{c}{Median jitter (mm)} \\',r'View & D & S & D & S & D & S \\',r'\midrule']
    for r in table2.itertuples(index=False):
        vals=[r.raw_position_speed_mae_clip_mean_mps,r.smooth_position_speed_mae_clip_mean_mps,r.raw_disp_mae_clip_mean_m,r.smooth_disp_mae_clip_mean_m,r.raw_jitter_clip_median_m*1000,r.smooth_jitter_clip_median_m*1000]
        lines.append(r.site+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.97\linewidth}\footnotesize Error entries average recording-level MAEs over available comparisons. Jitter entries are medians of recording-level medians; 43 recordings contribute jitter, 39 contribute 0.5-s speed, and 37 contribute 1-s displacement. This is a paired postprocessing comparison, not an isolated BEVHeight architecture ablation.\end{minipage}',r'\end{table}']
    (tables/'reviewed_stabilization.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Workbook inputs and two paper tables saved')


if __name__=='__main__':main()
