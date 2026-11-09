"""Report frozen Todd results; reload held-out models for a post-hoc fusion diagnostic.

No fitting or checkpoint selection occurs here. The chronological-first comparison
was added after the main results, and is explicitly descriptive.
"""
import json
import platform
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import todd_experiment_data as D
import todd_experiment_train as T

OUT=D.OUT
ORDER=['source_transformer_masked','finetuned_transformer_ego','finetuned_transformer_masked',
       'mean_speed_logistic','availability_duration_logistic']
NAMES=['Frozen Transformer\n(mask-compatible)','Fine-tuned Transformer\n(ego only)',
       'Fine-tuned Transformer\n(ego + available leader)','Mean speed only','Availability + duration only']
CN=['冻结 Transformer（兼容缺失输入）','微调 Transformer（仅自车）','微调 Transformer（自车＋可用前车）',
    '仅平均速度 Logistic','仅可见性／时长 Logistic']


def window_predictions():
    torch.set_num_threads(2)
    ds=np.load(OUT/'dataset.npz');meta=pd.read_csv(OUT/'windows.csv');ci=ds['clip_index']
    cp=torch.load(D.ROOT/D.CFG['initial_checkpoint'],map_location='cpu',weights_only=True)
    src=D.Q.read((D.ROOT/D.CFG['initial_checkpoint']).parent/'feature_stats.json')
    old={k:np.array(src[k]) for k in ['mean','std']};stats={k:v.tolist() for k,v in old.items()}
    records=[]
    def collect(model,tx,ix,name,seed):
        model.eval()
        with torch.no_grad():p=torch.sigmoid(model(*[a[ix] for a in tx])).numpy()
        for j,prob in zip(ix,p):
            records.append(dict(model=name,seed=seed,**meta.iloc[j].to_dict(),prob_av=float(prob)))
    collect(T.MaskedTransformer(cp,stats,old),T.tensors(ds,stats,'masked'),np.arange(len(ci)),ORDER[0],-1)
    for fold in D.Q.read(OUT/'folds.json'):
        ix=np.flatnonzero(ci==D.CLIPS.index(fold['test'][0]))
        for mode in ['ego','masked']:
            for seed in D.CFG['training_seeds']:
                c=torch.load(OUT/'checkpoints'/f'{mode}_fold{fold["fold"]:02d}_seed{seed}.pt',map_location='cpu',weights_only=True)
                assert c['test']==fold['test']
                model=T.MaskedTransformer(cp,c['normalization'],old);model.load_state_dict(c['model_state_dict'])
                collect(model,T.tensors(ds,c['normalization'],mode),ix,f'finetuned_transformer_{mode}',seed)
    raw=pd.DataFrame(records);raw.to_csv(OUT/'predictions_by_window_seed.csv',index=False)
    recorded=pd.read_csv(OUT/'predictions_by_seed.csv')
    grouped=raw.groupby(['model','seed','clip']).prob_av.mean()
    for key,p in grouped.items():
        g=recorded[(recorded.model==key[0])&(recorded.seed==key[1])&(recorded['clip']==key[2])]
        assert len(g)==1 and abs(p-g.prob_av.iloc[0])<1e-6
    avg=raw.groupby(['model','clip','label','window','start_s','end_s','duration_s'],as_index=False).prob_av.mean()
    avg.to_csv(OUT/'predictions_by_window.csv',index=False)
    rows=[];summaries=[]
    for (model,clip),g in avg.groupby(['model','clip']):
        g=g.sort_values('start_s');first=g.iloc[0]
        rows.append(dict(model=model,clip=clip,label=int(first.label),windows=len(g),
                         first_probability=first.prob_av,fused_probability=g.prob_av.mean(),
                         first_duration_s=first.duration_s,total_observed_duration_s=g.duration_s.sum(),
                         elapsed_span_s=g.end_s.max()-g.start_s.min()))
    comparisons=pd.DataFrame(rows);comparisons.to_csv(OUT/'fusion_by_video.csv',index=False)
    for model,g in comparisons.groupby('model'):
        for cohort,gg in [('all13',g),('at_least_two_windows',g[g.windows>=2])]:
            summaries.append(dict(model=model,cohort=cohort,
                first=T.scores(gg.label,gg.first_probability),fused=T.scores(gg.label,gg.fused_probability),
                first_duration_mean_s=float(gg.first_duration_s.mean()),
                fused_duration_mean_s=float(gg.total_observed_duration_s.mean())))
    D.Q.save(OUT/'fusion_summary.json',dict(status='descriptive_post_hoc',
        protocol='Chronological first window versus probability average of all disjoint windows in the same held-out video. Same fold checkpoints and seed averaging. No refitting. Adjacent windows are correlated; not cross-camera or independent passages.',
        all_saved_neural_predictions_reproduced=True,results=summaries))
    return comparisons,summaries


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    inv=pd.read_csv(OUT/'inventory.csv');win=pd.read_csv(OUT/'windows.csv')
    predictions=pd.read_csv(OUT/'predictions_by_video.csv')
    summary=D.Q.read(OUT/'classification_summary.json');metrics=pd.DataFrame(summary['results']).set_index('model').loc[ORDER]
    comparisons,fusion=window_predictions()
    for col in ['per_seed','legacy_two_video_cohort','confusion_hv_av']:metrics=metrics.drop(columns=col)
    metrics.to_csv(OUT/'classification_table.csv')
    p=predictions.pivot(index='clip',columns='model',values='prob_av')
    detailed=inv.merge(p,on='clip');detailed.to_csv(OUT/'video_results.csv',index=False)
    speed=inv.groupby('label').agg(videos=('clip','size'),mean_of_video_mean_speed_mps=('mean_ego_speed_mps','mean'),
        median_of_video_mean_speed_mps=('mean_ego_speed_mps','median'),
        min_video_mean_speed_mps=('mean_ego_speed_mps','min'),max_video_mean_speed_mps=('mean_ego_speed_mps','max'))
    speed.to_csv(OUT/'descriptive_motion.csv')
    inv['direction']=inv['clip'].str.split('_').str[2]
    quality=inv.groupby('direction').agg(videos=('clip','size'),gps_temporal_holdout_videos=('gps_test_speed_mae_mps','count'),
                                        mean_speed_mae_mps=('gps_test_speed_mae_mps','mean'))
    quality.to_csv(OUT/'quality_by_direction.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
    y=np.arange(5);colors=['#7e8fa6','#7e8fa6','#276f98','#cc8442','#7e8fa6']
    axes[0].barh(y,metrics.accuracy*100,color=colors)
    axes[0].set_yticks(y,NAMES);axes[0].invert_yaxis();axes[0].set_xlim(0,100)
    axes[0].set_xlabel('Video accuracy (%)');axes[0].set_title('(a) All 13 videos; fixed threshold 0.5')
    axes[0].axvline(100*7/13,color='#555555',ls=':',label='Always AV: 7/13')
    for i,row in enumerate(metrics.itertuples()):axes[0].text(row.accuracy*100+1,i,f'{row.correct}/13 ({row.accuracy:.1%})',va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1))
    axes[0].legend(loc='lower right',fontsize=9)
    g=predictions[predictions.model=='finetuned_transformer_masked'].set_index('clip').loc[D.CLIPS]
    for i,(clip,row) in enumerate(g.iterrows()):
        color='#276f98' if row.label else '#bd6735'
        axes[1].plot(row.prob_av,i,'o' if row.correct else 'x',color=color,ms=8,mew=1.7)
    axes[1].set_yticks(np.arange(13),D.CLIPS,fontsize=9);axes[1].invert_yaxis();axes[1].set_xlim(-.03,1.03)
    axes[1].axvline(.5,color='#555555',ls=':');axes[1].set_xlabel('Three-seed mean P(AV)')
    axes[1].set_title('(b) Held-out prediction by video\nCircle: correct; cross: incorrect')
    fig.savefig(OUT/'classification_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.2,5.8),sharey=True,layout='constrained')
    yy=np.arange(len(inv));axes[0].barh(yy,inv.retained_duration_s,color='#276f98');axes[0].invert_yaxis()
    axes[0].set_yticks(yy,inv['clip'],fontsize=9);axes[0].set_xlabel('Sum of window durations (s)');axes[0].set_title('(a) Actual retained observation')
    for i,r in inv.iterrows():axes[0].text(r.retained_duration_s+.07,i,str(r.windows),va='center',fontsize=8)
    axes[0].set_xlim(0,9);axes[0].text(.98,.02,'Numbers: disjoint windows',transform=axes[0].transAxes,ha='right',fontsize=8)
    for i,r in inv.iterrows():
        if pd.notna(r.gps_test_speed_mae_mps):axes[1].plot(r.gps_test_speed_mae_mps,i,'o',color='#bd6735' if r.gps_test_speed_mae_mps>1 else '#276f98')
        else:axes[1].text(.07,i,'No GPS test score',va='center',fontsize=8,color='#777777')
    axes[1].set_xlim(0,3.65);axes[1].axvline(1,color='#777777',ls=':');axes[1].set_xlabel('GPS speed MAE (m/s)');axes[1].set_title('(b) Later-interval quality\nFixed per-video GPS delay')
    axes[2].barh(yy,inv.leader_available_fraction*100,color='#276f98');axes[2].set_xlim(0,105)
    axes[2].set_xlabel('Leader-observed samples (%)');axes[2].set_title('(c) Leader availability\nMissing observations kept with masks')
    fig.savefig(OUT/'coverage_and_quality.png',dpi=180);plt.close(fig)
    # The image review is an assistant inspection, not new user ground truth.
    review=D.Q.read(OUT/'image_review/review.json')
    review.update(image_review='completed_with_quality_flags',reviewer='assistant visual inspection of 26 sampled frames',
        findings=['The displayed ego targets are visually consistent with the marked test vehicle at sampled mid/end frames; this does not rule out between-frame ID errors.',
                  'Several WE/receding views show projected-box offsets; AV_T_WE_3 has conspicuous compression artifacts in the endpoint frame.',
                  'HV_T_EW_6 endpoint has a substantial projected yaw/box mismatch despite favorable speed agreement.',
                  'Distant targets and short support remain limitations; no new accuracy labels or inclusion changes were made after classifier evaluation.'])
    D.Q.save(OUT/'image_review/review.json',review)
    mf=next(x for x in fusion if x['model']=='finetuned_transformer_masked' and x['cohort']=='all13')
    mm=next(x for x in fusion if x['model']=='finetuned_transformer_masked' and x['cohort']=='at_least_two_windows')
    text=['Todd SE 完整开发实验（2026-11-02）','',
        '范围与执行：全部 13 段 SE 视频（7 AV、6 HV），W 视角不在本轮。核验 13 个原视频哈希，使用已人工审计、冻结的 BEVHeight/MOT 轨迹导出；本轮未重跑检测器或重新训练检测网络。实际新执行了输入构建、GPS 误差核验、78 次分类适配、全部留出预测和独立检查。',
        'AV/HV 是试验车文件名中的 ACC 开启／人工驾驶状态；不是所有车辆的自动驾驶等级。','',
        '1. 轨迹重建与观测质量',
        f'13 段全部保留，共 {len(win)} 个互不重叠窗口、{int(inv.retained_samples.sum())} 个 10 Hz 样本。窗口 {win.duration_s.min():.1f}–{win.duration_s.max():.1f} 秒，累计真实窗口时长 {inv.retained_duration_s.sum():.1f} 秒；不是把整段视频只取首个两秒，也没有将短轨迹拉长。仅跨 <=0.2 秒的观测间隔插值，不使用此前 1 秒缺口补全输出。',
        f'12 段有 GPS，10 段支持既定前段拟合时差／后段评价，平均逐视频速度 MAE {inv.gps_test_speed_mae_mps.mean():.4f} m/s；4 段 >1 m/s，3 段没有该时间划分下的分数。1 m/s 仅为描述标记，没有按误差筛掉分类测试视频。',
        '沿用之前固定时差，不重新为这些分类窗口拟合时差、不调整距离尺度，也不分别平移自车和前车。时差均有不确定性标记；GPS 只用于质量检查，不进入分类器。质量分数的时间范围与分类窗口范围不同。',
        '逐段误差、观测时长、前车可用率见 inventory.csv / video_results.csv / coverage_and_quality.png。按文件名方向分组的质量差异见 quality_by_direction.csv；这是事后描述，不是独立视角验证。',
        '每段中间和末尾各复核一帧，共 26 帧；存在远车投影偏移、压缩残影、HV_T_EW_6 末帧朝向偏差。不能据此宣称框或三维位置全部准确，未测密集 MOT 身份精度或前车间距真值。','',
        '2. AV/HV 识别（视频级留出）',
        '13 个外层折，每次留出完整 1 段测试，余下取 1 AV＋1 HV 验证、10 段训练。一个视频的全部窗口只能在同一集合。仅训练集拟合标准化，验证 BCE 选 checkpoint；测试结果不用于选择参数。',
        '从固定的原 Transformer 权重出发，冻结 encoder、CLS 和位置嵌入，只训练输入投影、缺失投影和分类头。两个微调版本 ×13 折 ×3 个固定种子（42/52/62）=78 个模型。CPU 执行。',
        '支持无前车、前车不可见、部分缺失和短序列：缺失值标准化后填零并附有效性标记，padding 使用 attention mask，不把零解释为前车静止或零间距。',
        '主指标先平均同一视频窗口概率，再平均三种子概率；固定阈值 0.5，以 13 段视频为分母：']
    for name,cn in zip(ORDER,CN):
        r=metrics.loc[name];text.append(f'  {cn}: {int(r.correct)}/13 = {r.accuracy:.1%}; balanced accuracy {r.balanced_accuracy:.1%}; AV recall {r.av_recall:.1%}; HV recall {r.hv_recall:.1%}')
    text += ['始终预测 AV 的常量基线为 7/13（53.8%）。微调没有超过兼容缺失输入的冻结 Transformer；仅速度控制更高，且可见性／时长控制也达到 8/13，尚不能证明网络学到可泛化的 ACC 行为。',
        '三个 masked 单种子各为 9/13（69.2%），但答错的具体视频与概率不同，按事先固定的概率平均得到 8/13（61.5%）；不以事后挑种子的结果替代主指标。',
        '旧前后车协议只覆盖 AV_T_WE_1、HV_T_EW_6，旧 Transformer/CNN 均 1/2。此次覆盖 13/13；不能将旧 50% 与新 61.5% 当成同样本准确率提升。新冻结参照沿用源权重，但改变了缺失／padding／截断处理，亦不同于旧 15 模型集成。','',
        '3. 同一视频内多窗口融合（事后补充诊断）',
        f'不再训练，重新载入全部 78 个 checkpoint，导出逐窗口留出概率，核对与已保存的视频概率一致。masked Transformer：全 13 段首个时间窗口 {mf["first"]["correct"]}/13，全部窗口平均 {mf["fused"]["correct"]}/13。',
        f'公平限制到具有至少两个窗口的 {mm["first"]["videos"]} 段：首窗 {mm["first"]["correct"]}/{mm["first"]["videos"]}，融合 {mm["fused"]["correct"]}/{mm["fused"]["videos"]}；平均观测时长从 {mm["first_duration_mean_s"]:.2f} 到 {mm["fused_duration_mean_s"]:.2f} 秒。',
        '该对照在主结果之后增加，明确属于事后描述；既不是独立穿越路口的多次观测，也不是跨摄像头匹配。相邻窗口相关，增加观测时长与融合本身的作用未分离。不虚构固定 K 个足长窗口。','',
        '行为统计与论文解释',
        f'每视频等权的视觉平均速度：AV {speed.loc[1,"mean_of_video_mean_speed_mps"]:.2f} m/s，HV {speed.loc[0,"mean_of_video_mean_speed_mps"]:.2f} m/s。HV 含两段明显低速记录；这只是所采集片段的描述，不能解释为 ACC 造成更快通行，也不能推断道路流量、密度或容量。',
        '轨迹提取／稳定化仍为论文核心；分类只是下游验证。本轮在单一摄像头、重复试验车辆的开发数据上进行，且此前有两段被用于检测器适配；不是独立车辆／拍摄批次／摄像头泛化测试。',
        '全部记录保留，未根据分类正确与否筛样本。未生成或部署按全部数据训练的最终分类器。下一轮优先处理方向相关的几何误差、短轨迹和速度／状态混杂，然后使用独立批次确认。','',
        '复现（在 identification 工作区执行，注意训练命令会覆盖本实验自己的生成物）：',
        '  .venv-classifier/Scripts/python.exe scripts/todd_experiment_data.py',
        '  .venv-classifier/Scripts/python.exe scripts/todd_experiment_train.py',
        '  .venv-classifier/Scripts/python.exe scripts/check_todd_experiment.py',
        '  .venv-bev/Scripts/python.exe scripts/review_todd_images.py',
        '  .venv-classifier/Scripts/python.exe scripts/report_todd_experiment.py',
        '其中图像 review.json 的观察记录来源于本轮实际复核，自动重建图片不替代人工／视觉复核。',
        'protocol.json 保存训练前固定协议；input_manifest.json 保存原始输入哈希；folds.json、logs/、checkpoints/、predictions_by_* 保存每一步；independent_validation.json 记录独立验证。']
    (OUT/'RESULTS.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    D.Q.save(OUT/'report_validation.json',dict(status='PASS',all_78_saved_neural_models_reloaded=True,
        all_heldout_video_probabilities_reproduced=True,visual_review='completed_with_quality_flags',
        classifier_refitting_in_report=False,aggregation_comparison='post_hoc',
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)))
    print(metrics[['correct','accuracy','balanced_accuracy']].to_string())
    print('Fusion masked:',json.dumps([x for x in fusion if x['model']=='finetuned_transformer_masked']))
    print('Quality by direction:',quality.to_string())


if __name__=='__main__':main()
