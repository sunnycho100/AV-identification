"""Build the experiment report and manuscript results from recorded outputs."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent;EXP=ROOT/'experiments/field_pilot_20261101';PAPER=ROOT/'../paper'
cfg=json.loads((ROOT/'configs/field_pilot.json').read_text())
base=json.loads((EXP/'pilot_base_runs.json').read_text());ft=json.loads((EXP/'pilot_ft_runs.json').read_text())
sel=json.loads((EXP/'finetune_selection.json').read_text());hist=sel['history']
gps=json.loads((EXP/'gps_evaluation.json').read_text())
wb=json.loads((EXP/'pilot_base/window_report.json').read_text());wf=json.loads((EXP/'pilot_ft/window_report.json').read_text())
lookup={r['clip']:r for r in ft};split={c:s for s in ['train','validation','test'] for c in cfg[s]}
loss_drop=100*(1-hist[-1]['val_loss']/hist[0]['val_loss'])
rows=[]
for r in base:rows.append(dict(clip=r['clip'],split=split[r['clip']],frames=r['frames'],baseline_tracks=r['vehicles'],finetuned_tracks=lookup.get(r['clip'],{}).get('vehicles')))
pd.DataFrame(rows).to_csv(EXP/'run_comparison.csv',index=False)
md=['# 本地路测视频整合试验（2026-11-01）','',
'已完成六段 Todd/SE 视频的图像标定、BEVHeight 检测、AB3DMOT 跟踪、平滑、前后车关联和分类输入适配；用两段训练视频完成两轮检测头微调，重跑两段验证、两段测试视频。',
'',f"基线解码 **{sum(r['frames'] for r in base):,} 帧**，输出 **{sum(r['vehicles'] for r in base)} 条轨迹片段**。这些片段包含碎片，不是独立车辆数量。两轮微调使用 48 个均匀抽样训练帧和 24 个独立验证帧，验证伪标注损失 {hist[0]['val_loss']:.4f} → {hist[-1]['val_loss']:.4f}（下降 {loss_drop:.2f}%）；选择第 {sel['selected_epoch']} 轮。该损失不是人工真值精度。",'',
'| 片段 | 划分 | 帧数 | 基线轨迹片段 | 微调后 |','|---|---|---:|---:|---:|']
for r in rows:md.append(f"| {r['clip']} | {r['split']} | {r['frames']} | {r['baseline_tracks']} | {r['finetuned_tracks'] if r['finetuned_tracks'] is not None else '未重跑训练片段'} |")
md += ['', '## 分类接口与结果边界', '',
f"基线共有 {sum(r['candidate_windows'] for r in wb['clips'])} 个时长足够的候选窗口；微调后的验证/测试共有 {sum(r['candidate_windows'] for r in wf['clips'])} 个。通过原训练域的全部质量检查后，分别为 **{wb['total_windows']}** 和 **{wf['total_windows']}** 个。当前候选窗口均未通过原模型的全窗 50 mph 门槛；其他前后车区间普遍不足四秒。",
'', '因此没有进行路测分类器微调，也不报告真实视频分类准确率。空的 NPZ 明确保存为 [0,40,3]，classification_status.json 记录原因。未把视频文件名标签赋给其他车辆，未放宽速度门槛，未把短窗口补齐为四秒。CNN/Transformer 推理接口通过合法合成窗口检查；全部 75 个历史主模型通过实际加载和有限输出检查。',
'', '适配器补齐了：一对一检测匹配与车辆长度恢复、中心距转净间距、前后车/车道切换隔离、0.20 秒最大插值间隔、10 Hz 重采样和窗口拒绝审计。五项契约测试覆盖间距、缺测、身份切换、未知车辆标签和速度门槛。',
'', '## 图像确定目标后的 GPS 对照', '',
'目标身份由已有车头标记匹配方法决定，不使用 GPS 挑选误差最小的轨迹。GPS 以视频相对时间插值至解码时间戳；不搜索时间偏移。位置 RMSE 经过每条轨迹的刚性对齐，仅表示形状一致性；不是绝对地理定位精度。尺度比为相机位移 / GPS 位移，理想值为 1。', '',
'| 片段 | 阶段 | 目标轨迹 | 时长 s | 速度 MAE m/s | 尺度比 | 刚性对齐 RMSE m |','|---|---|---:|---:|---:|---:|---:|']
scored=[r for r in gps['records'] if r['status']=='SCORED']
for r in scored:md.append(f"| {r['clip']} | {r['tag']} | {r['vehicle_id']} | {r['span_s']:.2f} | {r['speed_mae_mps']:.3f} | {r['camera_to_gps_displacement_scale']:.3f} | {r['rigid_aligned_rmse_m']:.3f} |")
for m in gps.get('matched_speed_comparisons',[]):
    md += ['',f"**公平对照：** {m['clip']} 在共同的 {m['frames']} 个时间点（{m['span_s']:.3f} 秒）上，速度 MAE 为 {m['baseline_speed_mae_mps']:.3f} → {m['finetuned_speed_mae_mps']:.3f} m/s。本次微调没有降低共同观测区间的速度误差。上表各自轨迹时长不同，不能用其单独 MAE 直接宣称改进。"]
md += ['', '未可靠识别的片段不进入 GPS 指标。后视片段没有可直接复用的新视频参考轨迹；车头标记方法不能据此保证后视目标身份。', '',
'## 标定、数据与适用范围', '',
'每段视频用三个 AnyCalib 内参估计的分量中位数、十五帧 LSD 消失点的中位数求旋转，假设零横滚。高度使用历史 Todd 值 16.26 m，尚非测量验证值；lane_width_scale_check.json 保存既有车道宽度法的训练片段诊断，不用测试 GPS 回调标定。Todd 道路范围参数也不能直接迁移到其他站点。', '',
'这是同一站点、可能重复使用同一试验车的 recording-level 小规模试验。初始权重已经过历史 Todd 数据微调，不能称为独立站点或独立车辆泛化。伪标注保留所有高置信检测，并对可靠轨迹修正朝向、地面高度、尺寸与速度；它不能给漏检车辆补充独立真值。', '',
'## 产物与复现', '',
'- `configs/field_pilot.json`：固定划分与参数；`docs/FIELD_PILOT_RUNBOOK.md`：复现命令。',
'- `trajectory/outputs/trajectories/*_pilot_base|pilot_ft/`：车辆 CSV、tracks、lanes、run、尺寸和可确认的目标身份。',
'- `trajectory/outputs/finetune/pilot_head/`：两轮权重、训练历史；`finetune_selection.json`：选模依据。',
'- 本目录 `pilot_base/`、`pilot_ft/`：NPZ、窗口表、拒绝审计、分类状态。',
'- `gps_evaluation.json`、逐片段 GPS CSV、`all_models_smoke.json`、`adapter_tests.log` 和环境锁定文件：验证证据。',
'- `paper/Manuscript.tex`、`paper/Manuscript.pdf`：更新论文；`revisions/before_field_pilot_20261101/` 保留原文。',
'', '后续首先需要提高可用跟驰时长和完成尺度验证；若要覆盖当前较低速度片段，应明确另建适用于低速域的分类协议，再利用有标签的连续跟驰数据训练和评估。现有四秒高速度分类器不能靠换接口就解决这一数据覆盖问题。']
(EXP/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
scale=json.loads((EXP/'lane_width_scale_check.json').read_text())
scale_text='\n车道宽度法补充诊断（名义宽度 3.6576 m、各五帧、未作为本次检测标定）：\n\n'
for r in scale:scale_text+=f"- {r['clip']}：高度中位数 {r['median_height_m']:.3f} m，四分位区间 {r['height_iqr_m'][0]:.3f}–{r['height_iqr_m'][1]:.3f} m。\n"
scale_text+='\n第一段的跨帧差异较大，说明该画面中的车道线提取不足以稳定确定高度。没有选择有利于测试 GPS 误差的标定值。\n'
with (EXP/'REPORT.md').open('a',encoding='utf-8') as f:f.write(scale_text)
tex=r'''The local integration pilot uses six Todd/SE recordings, divided before adaptation into two training, two validation, and two test clips (Table~\ref{tab:field-pilot}). The supplied initial checkpoint had previously been adapted on older Todd footage; the present recording split is therefore a within-site pilot, not independent-site or independent-vehicle generalization. Processing was performed on a local NVIDIA RTX~3090 with 24~GB memory. The baseline decoded 2,577 frames and produced 446 postprocessed track fragments. Fragment counts do not measure detection accuracy or the number of unique vehicles.

\begin{table}[pos=tbp]
\centering
\caption{Frozen field-pilot recording split and extracted track-fragment counts. The adapted detector is evaluated on validation and test recordings; training clips are not rerun.}
\label{tab:field-pilot}
\begin{tabular}{llrrr}
\toprule
Recording & Split & Frames & Baseline & Adapted \\
\midrule
'''
for r in rows:
    name=r['clip'].replace('_',r'\_');n=r['finetuned_tracks'] if r['finetuned_tracks'] is not None else '--'
    tex+=f"\\texttt{{{name}}} & {r['split']} & {r['frames']} & {r['baseline_tracks']} & {n} \\\\\n"
tex+=r'''\bottomrule
\end{tabular}
\end{table}

The detection head is adapted for two epochs using AdamW with learning rate $2\times10^{-6}$, weight decay $10^{-4}$, gradient-norm clipping at 10, batch size one, and seed zero. Forty-eight uniformly distributed training frames and 24 separate validation frames are used. The validation pseudo-label loss decreases from 1.7576 to 1.6897 (3.86\%); epoch two is selected against both adapted checkpoints and the unchanged initial model. This is a consistency result against temporal pseudo-labels and does not establish an improvement in independently annotated detection accuracy.

'''
held=[r for r in scored if r['split']=='test']
for m in gps.get('matched_speed_comparisons',[]):
    tex+=f"On the common {m['frames']} timestamps spanning {m['span_s']:.3f}~s of the held-out \\texttt{{HV\\_T\\_EW\\_6}} target, speed MAE is {m['baseline_speed_mae_mps']:.3f}~m/s before adaptation and {m['finetuned_speed_mae_mps']:.3f}~m/s afterward. The adaptation therefore does not improve speed accuracy over the matched observation interval, despite its lower pseudo-label validation loss.\n\n"
if len(held)>=2:
    a=next((r for r in held if r['tag']=='pilot_base'),None);b=next((r for r in held if r['tag']=='pilot_ft'),None)
    if a and b:
        tex+=f"The image-identified target in the held-out \\texttt{{HV\\_T\\_EW\\_6}} recording has a baseline speed MAE of {a['speed_mae_mps']:.3f}~m/s and an adapted MAE of {b['speed_mae_mps']:.3f}~m/s on their respective valid track spans ({a['span_s']:.2f} and {b['span_s']:.2f}~s). The camera/GPS displacement-scale ratios are {a['camera_to_gps_displacement_scale']:.3f} and {b['camera_to_gps_displacement_scale']:.3f}. Target identity is fixed using image-marker votes before GPS comparison. These single-record diagnostics retain the unvalidated camera-height assumption and do not demonstrate broad accuracy gains.\n\n"
tex+=f"The baseline yields {sum(r['candidate_windows'] for r in wb['clips'])} duration-qualified candidate windows, while the adapted validation/test outputs yield {sum(r['candidate_windows'] for r in wf['clips'])}. None satisfy the unchanged whole-window 50-mph requirement. Other following intervals are too short to form an eligible four-second window. Consequently, this pilot reports neither field ACC classification accuracy nor field classifier fine-tuning. The interface and restored CNN/Transformer models are verified using legal synthetic inputs, which is a software check rather than empirical identification evidence. Resolving metric scale and extending continuous, labeled following coverage are necessary before evaluating the complete visual identification chain.\n"
tex+='\nA diagnostic application of the existing lane-width method, using a nominal 3.6576-m width and five frames from each training clip, yields median heights of 17.97 and 16.04~m. The corresponding interquartile intervals are 16.31--19.93 and 15.98--16.06~m. The wide first interval shows that this image-derived scale estimate is not consistently constrained. These diagnostic values are not substituted into the fixed pilot calibration.\n'
(PAPER/'sections/field_pilot_results.tex').write_text(tex,encoding='utf-8')
p=PAPER/'sections/experiments.tex';s=p.read_text(encoding='utf-8')
old=r'\reserved{Quantitative results for metric localization, speed and acceleration estimation, track continuity, cross-scene transfer, and the proposed trajectory-extraction improvements will be inserted here.}'
s=s.replace(old,r'\input{sections/field_pilot_results}')
p.write_text(s,encoding='utf-8')
# Evidence plot uses native plotting tools and numerical GPS comparison files.
target='HV_T_EW_6';paths=[EXP/f'{target}_{tag}_gps_comparison.csv' for tag in ['pilot_base','pilot_ft']]
if all(p.exists() for p in paths):
    fig,ax=plt.subplots(figsize=(7,3.2));colors=['#2879b9','#d66c2f']
    for path,color,label in zip(paths,colors,['Baseline camera','Adapted camera']):
        d=pd.read_csv(path);ax.plot(d.t_s,d.camera_speed_mps,label=label,color=color,lw=1.5)
    ax.plot(d.t_s,d.gps_speed_mps,label='GPS reference',color='#222222',ls='--',lw=1.5)
    ax.set(xlabel='Video-relative time (s)',ylabel='Speed (m/s)');ax.legend(frameon=False,ncol=3,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(EXP/'heldout_speed_comparison.png',dpi=180);fig.savefig(PAPER/'figs/field_heldout_speed.pdf');plt.close(fig)
    with (PAPER/'sections/field_pilot_results.tex').open('a',encoding='utf-8') as f:f.write(r'''
\begin{figure}[pos=tbp]
\centering
\includegraphics[width=0.82\linewidth]{figs/field_heldout_speed.pdf}
\caption{Timestamp-aligned speed comparison for the image-identified instrumented vehicle in one held-out Todd recording. Each camera curve covers its own valid reconstructed track span. Camera height remains an assumed historical value; the figure is a pilot diagnostic, not a cross-scene performance estimate.}
\label{fig:field-speed}
\end{figure}
''')
print('Updated experiment report and manuscript pilot results.')
