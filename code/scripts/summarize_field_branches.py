"""Build the workflow figure, evidence tables, and reproducible branch report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
EXP=ROOT/'experiments/field_branches_20261101'
PAPER=ROOT/'../paper'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def esc(t):return t.replace('_',r'\_')

def workflow():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,6.5));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,10),ylim=(0,6.5));ax.axis('off')
    blue='#dcebf5';green='#e5f0e8';orange='#f7ebd6';edge='#35506a'
    def box(x,y,w,h,title,body,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.08',linewidth=1,edgecolor=edge,facecolor=color))
        ax.text(x+w/2,y+h-.16,title,ha='center',va='top',fontsize=10,fontweight='bold',color='#1a3448')
        ax.text(x+w/2,y+h-.40,body,ha='center',va='top',fontsize=8.5,linespacing=1.35,color='#243746')
    def arrow(a,b,dash=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color=edge,linestyle='--' if dash else '-'))
    # A single, unambiguous top-to-bottom inference path.
    x,w=0.35,5.25
    nodes=[(5.58,.65,'1  Video and recording split','Three core views; target identity; select visible intervals'),
           (4.64,.65,'2  Camera geometry','AnyCalib intrinsics + lane VP pose; h = 15 m (17 m sensitivity)'),
           (3.70,.65,'3  BEVHeight and roadway filtering','Ground convention + 3D boxes; export score >= 0.20'),
           (2.76,.65,'4  Multi-object tracking','AB3DMOT: high-score birth + low-score continuation'),
           (1.82,.65,'5  Target and immediate leader','Stitch / split / RTS; target trajectory and net gap'),
           (.88,.65,'6  Window eligibility and identification','10 Hz; variable windows; no field speed floor')]
    for y,h,t,b in nodes:box(x,y,w,h,t,b)
    for a,b in zip(nodes,nodes[1:]):arrow((x+w/2,a[0]-.05),(x+w/2,b[0]+b[1]+.05))
    box(6.25,4.69,3.35,1.40,'Optional adaptation','Input / lifted-feature masks\nTemporal labels + silhouette correction\nFrozen-backbone head training\nGPS/hybrid loss requires registration',orange)
    arrow((6.20,5.30),(5.65,4.05),True)
    box(6.25,2.85,3.35,1.30,'Trajectory refinements','31-state appearance-axis consensus\nTrack motion resolves front/back\nPreserve raw and filled headings',green)
    arrow((6.20,3.25),(5.65,2.14),True)
    box(6.25,.88,3.35,1.55,'Reference and evaluation','Image marker establishes target identity\nGPS checks require verified timing\nTrain / validation / test kept separate\nNo valid window -> no field classifier score',green)
    arrow((5.65,1.20),(6.20,1.20),True)
    ax.text(.35,.30,'Solid: primary processing     Dashed: optional branch or reference check',fontsize=9,color=edge)
    fig.savefig(PAPER/'figs/roadside_workflow.pdf');fig.savefig(EXP/'workflow_preview.png',dpi=160);plt.close(fig)

def manuscript_results():
    yaw=read(EXP/'yaw.json')['rows'];adapt=read(EXP/'adaptation_feature_aug_v2.json')
    inf=read(EXP/'gpu_inference.json')['rows'];photo=read(ROOT/'experiments/field_photo_height_20261101/summary.json')
    n=sum(r['baseline']['n'] for r in yaw);b=np.mean([r['baseline']['median_folded_deg'] for r in yaw]);a=np.mean([r['refined']['median_folded_deg'] for r in yaw])
    txt=r'''\subsubsection{Additional implementation and adaptation checks}
The additional branch checks retain the initial pilot's 16.26-m geometry and frozen recording split. Table~\ref{tab:branch-status} distinguishes execution evidence from accuracy evaluation. On twelve identical sampled training/validation frames, the initial model, input masking, feature masking, the supplied feature-mask checkpoint, and virtual-camera cropping all produce finite detections. Their car-detection totals at score 0.20 are '''+', '.join(str(r['car_detections']) for r in inf)+r''', respectively. These counts are not recall or precision; virtual cropping also changes the field of view. Batch-normalization recalibration uses eight training frames in four mixed-recording batches. All 381 changed state buffers belong to 127 batch-normalization layers, and all learned parameters remain bit-identical. These variants are not selected as replacements on the basis of execution alone.

The second pseudo-label pass evaluates 24 distributed fit frames and a 12-frame median background per training recording. It accepts 87 silhouette fits and reduces the two training sets from 10,572 to 9,415 labels through road filtering and duplicate removal. Feature masking, these labels, and geometric augmentation are then combined for sixteen head-update steps, eight per training recording. Adam uses learning rate $2\times10^{-6}$, batch size one, seed zero, and gradient clipping at 5. The backbone remains bit-identical. On eight fixed validation frames with the original temporal labels, loss changes from '''+f"{adapt['validation_before']:.4f} to {adapt['validation_after']:.4f}"+r'''. This small combined-variant trial establishes trainability; it does not isolate the contribution of any component or establish independent detection accuracy.

The independent yaw-refinement pass processes all six baseline trajectory files. On '''+f'{n:,}'+r''' matched eligible states, the unweighted mean of recording-level median folded errors against track motion changes from '''+f'{b:.3f} to {a:.3f}'+r''' degrees. The same states are used before and after refinement. This is a motion-consistency diagnostic: the correction itself uses track motion to choose the heading sign, and no independent yaw annotations are available.

GPS-only and hybrid field adaptation do not reach training. A new-data adapter converts positions to a common east--north frame and restricts registration observations to image-marker-confirmed training states. The resulting shared registration is inconsistent with the visible vehicle locations and yields no eligible complete-box labels. Training is therefore stopped before either reference-supervised loss is optimized. The previously processed local coordinates, which reset their origin per recording, are not accepted as a common geographic frame. The full shared-site leave-one-recording-out evaluation also remains unvalidated with the available registered identities and directional coverage.

\begin{table}[pos=tbp]
\centering
\caption{Scope of the local checks. A successful execution is not a detection, tracking, or classification accuracy benchmark.}
\label{tab:branch-status}
\small
\begin{tabular}{p{0.32\linewidth}p{0.60\linewidth}}
\toprule
Component & Evidence and remaining limit \\
\midrule
Intrinsics, pose, BEVHeight, MOT, RTS & Six complete recordings, 2,577 frames; no independent 3D or MOT labels. \\
Road and feature masks & Six mask sets; input/feature inference on twelve frames each. \\
Second-pass labels and augmentation & Two training recordings; sixteen combined-variant gradient updates and eight validation frames. \\
Independent yaw refinement & Six recordings; matched-state motion-consistency evaluation only. \\
BN and virtual-camera variants & Eight-frame BN update; twelve-frame virtual-camera inference; no held-out accuracy selection. \\
GPS-only / hybrid supervision & Attempted; no usable labels after shared registration. No completed field training. \\
Trajectory classifier interface & Restored models and synthetic inputs pass; original pilot has zero windows under the source protocol. \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Camera-height sensitivity}
The current setting fixes camera height at 15~m and compares it with 17~m. Intrinsics, orientation, detector checkpoint, score thresholds, and postprocessing are unchanged. Two contiguous excerpts are fixed before comparison: frames 140--299 of the training recording \texttt{AV\_T\_EW\_6} and frames 600--779 of the test recording \texttt{HV\_T\_EW\_6}. Each height therefore requires 340 frames of detection followed by roadway filtering, MOT, fragment processing, RTS smoothing, and lane/leader export. This is an excerpt-based sensitivity check, not a rerun of all six complete recordings.

Table~\ref{tab:height-sensitivity} compares image-identified target speeds on the common timestamps of the two height settings. The nominal 15-m setting is fixed before these reference comparisons and is not selected by minimizing test GPS error. Height changes both geometric scale and association, so the comparison reruns the visual pipeline. The 15-m and 17-m excerpts yield two and three duration-qualified following windows, respectively; all fail the unchanged speed requirement. This is a result under the source protocol, not the revised field eligibility rule in Section~\ref{sec:field-core-results}. The speed comparisons use the supplied video/GPS timing, whose absolute synchronization has not been independently verified. The short, same-site excerpts cannot establish broad scale accuracy or classifier performance.

\begin{table}[pos=tbp]
\centering
\caption{Height sensitivity on common target timestamps. Speed MAE is in m/s; each row compares the same observations at both heights.}
\label{tab:height-sensitivity}
\begin{tabular}{llrrrr}
\toprule
Recording & Split & Samples & Span (s) & $h=15$ m & $h=17$ m \\
\midrule
'''
    for r in photo['common_timestamps']:
        txt+=rf"\texttt{{{esc(r['clip'])}}} & {r['split']} & {r['frames']} & {r['span_s']:.3f} & {r['mae_15_mps']:.3f} & {r['mae_17_mps']:.3f} \\"+'\n'
    txt+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    (PAPER/'sections/field_branch_results.tex').write_text(txt,encoding='utf-8')

def report():
    y=read(EXP/'yaw.json')['rows'];geo=read(EXP/'geometry.json')['rows'];photo=read(ROOT/'experiments/field_photo_height_20261101/summary.json')
    lines=['# 完整流程与研究分支验证记录','', '日期：2026-11-01。原六段 pilot 使用 16.26 m；新增主设置为 15 m，17 m 为敏感性对照。所有输出分别保存，旧结果未覆盖。论文按用户要求仅给出高度参数，不描述其来源。','',
    '## 实际运行范围','',
    '| 内容 | 运行证据 | 状态与限制 |','|---|---|---|',
    '| 原主流程 | 6 段、2,577 帧；AnyCalib/LSD、BEVHeight、AB3DMOT、拼接/RTS、车道/前车、窗口适配 | 通过；不是全 60 段测试 |',
    '| 新高度流程 | 2 段连续片段，每个高度 340 帧，15/17 m 共 680 帧；完整检测至轨迹导出 | 通过；使用初始 checkpoint，未沿用旧尺度微调权重 |',
    '| 道路/输入/特征遮罩 | 六段分别生成；姿态参考检查最大偏差 %.3f°；输入和 feature 分支各 12 帧 | 通过；旧 pilot 输出不变 |'%max(r['pose_difference_deg'] for r in geo),
    '| 二次标签修正 | 两段训练视频，24 拟合帧/片段、12 背景帧；87 个接受拟合 | 标签 10,572 → 9,415；不等于人工真值 |',
    '| feature + v2 + 增强训练 | 16 步，8 帧独立验证；2.1752 → 2.0990；backbone 未变 | 通过；组合试跑，不是各分支消融 |',
    '| 独立 yaw 修正 | 六段所有轨迹；原 yaw 保留；新增 refined/filled | 通过；仅运动一致性指标 |',
    '| BN 重估 | 8 帧/4 批；127 个 BN、381 个 buffer 变化；所有参数逐位一致 | 通过；验证推理输出未证明精度改善 |',
    '| 虚拟相机 | 12 帧真实图像裁剪、K 变换、推理 | 通过；裁剪视野不同 |',
    '| GPS-only/hybrid | 尝试新数据 ENU 注册和标签可见性筛选 | 未跑通：共享注册 RMSE 46.68 m，0 合格标签，未训练 |',
    '| 自动接地点/GPS 距离拟合 | 两段训练片段运行完成并查看接地点图 | 计算通过但结果拒绝采用；HV 片段粘连阴影/车辆/文字，不能作可靠标定 |',
    '| 场地共享参数留一评估 | 原实现存在，当前图像确认目标缺少双向覆盖且共享注册失败 | 未完成有效实测评估 |',
    '| 140.8 m / Rope3D 候选 | 配置/方法代码在，指定权重未包含 | 未运行推理 |',
    '| 路测 ACC 分类 | 原 pilot 的候选窗口全部未达原 >50 mph 门槛 | 无路测准确率或分类器微调；模型接口合成检查另有记录 |','',
    '## 高度对照：共同时间戳','', '| 片段 | 划分 | 样本数 | 时间跨度/s | 15 m MAE/(m/s) | 17 m MAE/(m/s) |','|---|---|---:|---:|---:|---:|']
    for r in photo['common_timestamps']:lines.append(f"| {r['clip']} | {r['split']} | {r['frames']} | {r['span_s']:.3f} | {r['mae_15_mps']:.3f} | {r['mae_17_mps']:.3f} |")
    lines+=['','15 m 在对照前固定；没有按 GPS 测试误差择高。两段短片段不足以证明跨场景精度。','', '## 朝向运动一致性','', '| 片段 | 同一批状态数 | 原始轴线误差中位数/° | 修正后/° |','|---|---:|---:|---:|']
    for r in y:lines.append(f"| {r['clip']} | {r['baseline']['n']} | {r['baseline']['median_folded_deg']:.3f} | {r['refined']['median_folded_deg']:.3f} |")
    lines+=['','参照和符号修正共享轨迹运动信息，因此不能表述为独立 yaw 精度。','', '新高度窗口检查：15 m 有 2 个、17 m 有 3 个时长合格候选，全部未达原速度门槛，接受数仍为 0。输出位于原 pilot 实验目录下单独的 photo_h15_excerpt / photo_h17_excerpt 文件夹。','', '已有六个组件检查程序全部通过，见 component_checks.json。相机对齐与边缘候选仅做既有几何/合成自检，未做完整实景候选评测；不能混同为全量运行。','', '## 未完成部分与原因','',
    '缺的不是 GPS loss 或 Transformer 训练模块，而是有效的场地/GPS 注册，以及满足分类协议的连续、已确认标签跟驰窗口。没有独立 3D/MOT 标注时不能报告 mAP、MOTA、IDF1 等实测精度。',
    '`gps_labels/` 是已废弃的新接口试验：processed_x/y 每段以零为原点，不能直接拼成场地坐标。训练脚本只读取 `gps_labels_enu/`，该目录无合格标签，因此 GPS 训练已停止。其调用失败日志保留。',
    '自动接地点图已人工查看（本次助手视觉核查）：AV 片段点多位于车头下缘/阴影边界；HV 片段多处取到无关车辆或阴影/文字粘连。未把拟合高度写回运行配置，也未继续调参。',
    '', '## 复现入口','',
    '在 identification 目录，以 `.venv-bev/Scripts/python.exe` 执行：','',
    '```powershell', '$env:PYTHONUTF8="1"', '$env:PYTHONDONTWRITEBYTECODE="1"', '$env:NUMBA_DISABLE_JIT="1"', '$env:TORCH_HOME="$PWD/.runtime/torch-cache"', '.venv-bev/Scripts/python.exe scripts/verify_field_branches.py', '.venv-bev/Scripts/python.exe scripts/verify_field_gpu_branches.py', '.venv-bev/Scripts/python.exe scripts/run_photo_height.py', '```','',
    '执行状态见本目录 JSON/分支 checkpoint；新高度结果见 `../field_photo_height_20261101/summary.json`。GPS 分支应维持显式无有效标签状态，不得把它视为完整成功。旧 pilot 复现脚本和摘要生成脚本不应覆盖本次扩展的论文方法章节。']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':workflow();manuscript_results();report()
