"""Save one paired paper table and workbook inputs; preserve baseline artifacts."""
import json
import pandas as pd
import evaluate_time_alignment as A


def main():
    site=pd.read_csv(A.OUT/'site_summary.csv');clips=pd.read_csv(A.OUT/'clip_results.csv');dist=pd.read_csv(A.OUT/'displacement_results.csv')
    for d in [site,clips,dist]:d['site']=d.site.map(A.NAMES)
    case_ids=[x['clip'] for x in A.Q.read(A.OLD/'case_selection.json')['cases']]
    cases=clips[clips['clip'].isin(case_ids)].copy()
    dt=dist[(dist.partition=='test')&(dist.horizon_s==1)][['clip','n','before_mae_m','after_mae_m']]
    cases=cases.merge(dt,on='clip',how='left')
    data=dict(sites=json.loads(site.to_json(orient='records')),clips=json.loads(clips.to_json(orient='records')),
              displacement=json.loads(dist.to_json(orient='records')),cases=json.loads(cases.to_json(orient='records')))
    A.Q.save(A.OUT/'workbook_data.json',data);cases.to_csv(A.OUT/'six_cases.csv',index=False)
    columns=['site','aligned_recordings','speed_before_mae_mps','speed_after_mae_mps','disp1_recordings','disp1_before_mae_m','disp1_after_mae_m']
    site[columns].to_csv(A.OUT/'tables'/'time_alignment_comparison.csv',index=False)
    lines=[r'\begin{table}[pos=tbp]',r'\centering',r'\caption{Temporal-holdout comparison before (B) and after (A) a fitted constant video--GPS delay. Each delay minimizes speed MSE on the first 40\% of the observed time span; evaluation uses the later interval after a 0.2-s guard. Before/after samples and recordings are identical. Entries are means of recording-level MAEs.}',r'\label{tab:time-alignment}',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r' & & \multicolumn{2}{c}{Speed MAE (m/s)} & & \multicolumn{2}{c}{1-s distance MAE (m)} \\',r'View & $N_v$ & B & A & $N_d$ & B & A \\',r'\midrule']
    for r in site.itertuples():
        lines.append(f'{r.site} & {r.aligned_recordings} & {r.speed_before_mae_mps:.2f} & {r.speed_after_mae_mps:.2f} & {r.disp1_recordings} & {r.disp1_before_mae_m:.2f} & {r.disp1_after_mae_m:.2f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\medskip',r'\begin{tabular}{rrrr}',r'\toprule',r'Displacement horizon (s) & Recordings & B (m) & A (m) \\',r'\midrule']
    total=site[site.site=='All'].iloc[0]
    for h in [1,2,3,5]:lines.append(f'{h} & {int(total[f"disp{h}_recordings"])} & {total[f"disp{h}_before_mae_m"]:.2f} & {total[f"disp{h}_after_mae_m"]:.2f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.97\linewidth}\footnotesize $N_v$/$N_d$ denote contributing recordings. Of 43 reviewed recordings, three lack GPS and five have insufficient support for the temporal split. All 35 fitted delays retain at least one uncertainty flag: weak variation, short fit span, a broad minimum, or a boundary solution. They are exploratory alignments, not independently verified sensor delays. Three- and five-second results each come only from \texttt{HV\_T\_EW\_6}; overlapping starts are not independent recordings.\end{minipage}',r'\end{table}']
    (A.OUT/'tables'/'time_alignment.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Paired tables and workbook inputs saved')


if __name__=='__main__':main()
