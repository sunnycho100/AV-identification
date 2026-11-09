"""GPS diagnostics after selection and compact research tables for the variable run."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import evaluate_trajectory_quality as Q
ROOT=Q.ROOT;OUT=ROOT/'experiments/field_todd_variable_20261103';OLD=ROOT/'experiments/field_todd_3s_20261102'


def quality():
    meta=pd.read_csv(OUT/'windows.csv');cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    delays=pd.read_csv(ROOT/'experiments/field_time_alignment_20261102/clip_results.csv').set_index('clip');dest=OUT/'quality';dest.mkdir(exist_ok=True)
    rows=[];overview,axes=plt.subplots(4,4,figsize=(16,11));axes=axes.ravel()
    for k,w in enumerate(meta.to_dict('records')):
        clip=w['clip'];s=pd.read_csv(OUT/'clips'/clip/'selected.csv');t=s.t_s.to_numpy();v=s.ego_speed_mps.to_numpy();elapsed=t-t[0];tr=delays.loc[clip]
        row=dict(clip=clip,points=len(s),duration_s=w['duration_s'],gps_status=tr.status)
        fig,axs=plt.subplots(1,2,figsize=(10,3.4));a,b=axs;dist=cumulative_trapezoid(v,t,initial=0)
        for ax in [a,axes[k]]:ax.plot(elapsed,v,color='#bc5938',label='Extracted');ax.set_title(clip+f' | {w["duration_s"]:.1f}s');ax.set_ylabel('Speed (m/s)')
        b.plot(elapsed,dist,color='#bc5938',label='Extracted');b.set_title('Travel from window start');b.set_ylabel('Distance (m)')
        if tr.status=='ESTIMATED_DELAY' and pd.notna(cat.loc[clip,'gps_path']):
            _,gps=Q.gps_frame(ROOT/cat.loc[clip,'gps_path']);ref,ok=Q.gps_interp(gps,t+float(tr.delay_s));gv=np.linalg.norm(ref[:,2:],axis=1)
            err=v-gv;test=ok&(t>=tr.test_start_s);s['gps_speed_mps']=np.where(ok,gv,np.nan)
            row.update(delay_s=float(tr.delay_s),delay_identifiability=str(tr.identifiability_flags),gps_points=int(ok.sum()),
                selected_speed_mae_mps=Q.mae(err[ok]),selected_speed_rmse_mps=Q.rmse(err[ok]),
                previous_test_region_points=int(test.sum()),previous_test_region_speed_mae_mps=Q.mae(err[test]))
            for ax in [a,axes[k]]:ax.plot(elapsed,np.where(ok,gv,np.nan),color='#267b8b',label='GPS, fixed delay')
            if ok.all():
                gd=cumulative_trapezoid(gv,t,initial=0);s['gps_travelled_m']=gd;s['extracted_travelled_m']=dist
                row.update(window_abs_travel_error_m=float(abs(dist[-1]-gd[-1])),displacement_curve_mae_m=Q.mae(dist-gd))
                b.plot(elapsed,gd,color='#267b8b',label='GPS, fixed delay')
            a.text(.03,.96,f'Speed MAE: {row["selected_speed_mae_mps"]:.2f} m/s',transform=a.transAxes,va='top',fontsize=9)
        else:a.text(.03,.96,'No eligible frozen-delay GPS',transform=a.transAxes,va='top',fontsize=9)
        for ax in [a,b,axes[k]]:ax.grid(alpha=.2);ax.set_xlabel('Elapsed time (s)');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(dest/f'{clip}.png',dpi=150);plt.close(fig)
        s.to_csv(dest/f'{clip}.csv',index=False);rows.append(row)
    for ax in axes[len(rows):]:ax.axis('off')
    overview.suptitle('Todd SE: selected variable-duration inputs | frozen GPS delays, no time/scale refitting',fontsize=14)
    overview.tight_layout();overview.savefig(dest/'speed_overview.png',dpi=150);overview.savefig(dest/'speed_overview.pdf');plt.close(overview)
    table=pd.DataFrame(rows);table.to_csv(dest/'clip_metrics.csv',index=False)
    old=pd.read_csv(OLD/'quality/clip_metrics.csv');paired=table.merge(old[['clip','selected_speed_mae_mps']],on='clip',suffixes=('_new','_old3s'))
    paired.to_csv(dest/'common_video_speed_comparison.csv',index=False)
    Q.save(dest/'summary.json',dict(videos=len(table),gps_comparable_videos=int(table.selected_speed_mae_mps.notna().sum()),
        video_macro_speed_mae_mps=float(table.selected_speed_mae_mps.mean()),complete_gps_windows=int(table.window_abs_travel_error_m.notna().sum()),
        video_macro_window_abs_travel_error_m=float(table.window_abs_travel_error_m.mean()),
        common_video_old3s_speed_mae_mps=float(paired.selected_speed_mae_mps_old3s.mean()),common_video_new_speed_mae_mps=float(paired.selected_speed_mae_mps_new.mean()),
        no_new_delay_fitting=True,no_gps_window_selection=True,
        scope='Descriptive diagnostics; prior frozen delay may overlap fit regions or be weakly identified. Window displacement errors cover different durations, not a fixed 3s prediction task.'))
    print(table[['clip','duration_s','selected_speed_mae_mps','window_abs_travel_error_m']].to_string(index=False),flush=True)


def report():
    dest=OUT/'tables';dest.mkdir(exist_ok=True)
    inventory=pd.read_csv(OUT/'inventory.csv');quality=pd.read_csv(OUT/'quality/clip_metrics.csv');pred=pd.read_csv(OUT/'field/video_predictions.csv')
    merged=inventory.merge(quality[['clip','gps_points','selected_speed_mae_mps','window_abs_travel_error_m']],on='clip',how='left').merge(
        pred[['clip','probability','predicted_label','correct','source_only_probability','seed_probability_min','seed_probability_max']],on='clip',how='left')
    merged.to_csv(dest/'video_results.csv',index=False)
    cls=json.loads((OUT/'field/results.json').read_text());gps=json.loads((OUT/'quality/summary.json').read_text())
    from variable_length_classifier import metrics
    short=pred[~pred.less_than_one_second];cls['at_least_one_second']=metrics(short.label.to_numpy(),short.probability.to_numpy())
    old=pd.read_csv(OLD/'field/video_predictions.csv');paired=pred.merge(old[['clip','label','probability','correct']],on=['clip','label'],suffixes=('_new','_old3s'))
    paired.to_csv(dest/'common_video_classification.csv',index=False)
    cls['common_videos_new']=metrics(paired.label.to_numpy(),paired.probability_new.to_numpy())
    cls['common_videos_old3s']=metrics(paired.label.to_numpy(),paired.probability_old3s.to_numpy())
    summary=dict(trajectory=json.loads((OUT/'trajectory_summary.json').read_text()),classification=cls,gps=gps)
    Q.save(OUT/'summary.json',summary)
    rows=[]
    for name,result in [('Old strict 3s, original cohort',cls['common_videos_old3s']),
                        ('Variable <=2.5s, public transfer',cls['source_only']),
                        ('Variable <=2.5s, video-held-out adaptation',cls['adapted']),
                        ('Variable >=1s subset, same fitted models',cls['at_least_one_second'])]:
        rows.append(dict(method=name,videos=result['n'],correct=result['correct'],accuracy=result['accuracy'],
                         balanced_accuracy=result['balanced_accuracy'],av_recall=result['av_recall'],hv_recall=result['hv_recall']))
    pd.DataFrame(rows).to_csv(dest/'classification_summary.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,5.5));colors=['#ba5942' if v else '#267b8b' for v in pred.label]
    axs[0].barh(pred['clip'],pred.duration_s,color=colors);axs[0].axvline(2.5,color='#777',ls='--');axs[0].set_xlabel('Actual input duration (s)');axs[0].invert_yaxis()
    axs[1].barh(pred['clip'],pred.probability,color=colors);axs[1].axvline(.5,color='#777',ls='--');axs[1].set_xlim(0,1);axs[1].set_xlabel('Out-of-fold P(AV), three-seed mean');axs[1].invert_yaxis()
    for i,r in enumerate(pred.to_dict('records')):axs[1].text(min(.93,r['probability']+.02),i,'OK' if r['correct'] else 'X',va='center',fontsize=8)
    fig.suptitle('Todd SE variable-duration experiment | orange: AV, blue: HV');fig.tight_layout();fig.savefig(dest/'duration_and_predictions.png',dpi=150);plt.close(fig)
    text=['Todd SE 变长轨迹实验（2026-11-03）','',f'输入：13段；11段2.5秒，HV_T_WE_6为2.1秒，HV_T_EW_2为0.5秒。',
          '已应用五段人工审核和后续身份确认。2D锚点只帮助关联车辆，不作为三维深度。',
          'HV_T_EW_2完整人工区间可跟踪，但只有后段匹配到三维检测；未通过复制、外推或跨冻结画面补足时长。','',
          f'公开数据变长训练后直接迁移：{cls["source_only"]["correct"]}/{len(pred)} = {cls["source_only"]["accuracy"]:.1%}。',
          f'Todd留一视频适配：{cls["adapted"]["correct"]}/{len(pred)} = {cls["adapted"]["accuracy"]:.1%}；平衡准确率 {cls["adapted"]["balanced_accuracy"]:.1%}。',
          f'AV召回率 {cls["adapted"]["av_recall"]:.1%}；HV召回率 {cls["adapted"]["hv_recall"]:.1%}。',
          f'至少1秒的12段子集：{cls["at_least_one_second"]["correct"]}/{len(short)} = {cls["at_least_one_second"]["accuracy"]:.1%}；使用相同模型，无重新选模。',
          f'与旧三秒实验共同的8段：旧 {cls["common_videos_old3s"]["correct"]}/8，新 {cls["common_videos_new"]["correct"]}/8。',
          '输入区间、审核支持、训练协议和可用样本数同时变化；以上差别不能归因于时长单一因素。','',
          f'GPS诊断：{gps["gps_comparable_videos"]}段可比较，逐视频平均速度MAE {gps["video_macro_speed_mae_mps"]:.3f} m/s。',
          f'完整GPS覆盖的{gps["complete_gps_windows"]}段，实际窗口行驶距离绝对误差均值 {gps["video_macro_window_abs_travel_error_m"]:.3f} m。',
          f'共同视频速度MAE：旧3秒 {gps["common_video_old3s_speed_mae_mps"]:.3f}，新短窗 {gps["common_video_new_speed_mae_mps"]:.3f} m/s。',
          '沿用以前冻结的GPS延迟，窗口起点位移归零，不新拟合时间偏移、尺度或速度偏差。不是未来轨迹预测指标。','',
          '训练：同一公开来源划分，3个种子，各自从旧三秒公开模型继续训练全部参数；39个Todd留出适配模型。',
          '最长26点，10Hz；时间补齐掩码与前车特征掩码分离。先裁剪实际轨迹，再重新计算速度导数。',
          '没有重新训练BEV检测器。原视频、原审核、旧模型和旧实验均保留。',
          '本结果是同摄像头、同车辆的小样本开发测试，不代表独立车辆或新路口的泛化性能。','',
          'AV_T_WE_1、HV_T_EW_1曾参与检测器训练；分类按视频留出不等于全pipeline的完全独立测试。','',
          '核心文件：tables/video_results.csv（逐视频），tables/classification_summary.csv（分类汇总），',
          'quality/clip_metrics.csv（GPS诊断），quality/speed_overview.png（速度对比），validation.json（验证），',
          'tables/duration_and_predictions.png（长度与分类），recovery/<clip>/candidate_review.jpg（自车关联）。','',
          '复现：在code目录使用.venv-bev运行todd_variable_data.py recover/build；variable_length_classifier.py prepare/train/field；',
          '使用.venv-classifier运行report_todd_variable.py quality/report；用.venv-bev运行validate_variable_length.py --final。']
    (OUT/'RESULTS.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    latex=['% Generated from the frozen variable-duration experiment; development results.',r'\begin{tabular}{lrrr}',r'\hline',r'Clip & Seconds & $P(AV)$ & Correct \\',r'\hline']
    for r in pred.to_dict('records'):latex.append(r['clip'].replace('_',r'\_')+f' & {r["duration_s"]:.1f} & {r["probability"]:.3f} & '+('Yes' if r['correct'] else 'No')+r' \\')
    latex.extend([r'\hline',r'\end{tabular}']);(dest/'video_results.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['quality','report']);a=p.parse_args();{'quality':quality,'report':report}[a.stage]()
