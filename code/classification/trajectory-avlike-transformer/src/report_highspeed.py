"""Render the completed, audited data-transfer study without picking test winners."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

BASE=Path('artifacts/experiments')
MAIN=BASE/'20261010_highspeed_v1'
BALANCED=BASE/'20261010_highspeed_balanced'
LABELS=dict(core='原数据 Transformer',joint='混合 CATS / Vanderbilt',ssl_core='原数据自监督预训练',ssl_external='外部自监督预训练',tree='统计特征树模型')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def headline(root,variant,evaluation,metric='accuracy',selection='bce_temperature'):
    d=pd.read_csv(root/'seed_summary.csv')
    r=d[d.variant.eq(variant)&d.evaluation.eq(evaluation)&d.selection.eq(selection)].iloc[0]
    scale=100 if metric=='accuracy' else 1
    mean=r[metric+'_mean']*scale; sd=r[metric+'_sd']*scale
    value=f'{mean:.2f}' if metric=='accuracy' else f'{mean:.4f}'
    return value+(f' ± {sd:.2f}' if metric=='accuracy' and np.isfinite(sd) else '')


def comparison_rows(root,variants):
    d=pd.read_csv(root/'comparison.csv').query('selection=="bce_temperature"')
    rows=[]
    for variant in variants:
        highway=d[d.variant.eq(variant)&d.evaluation.eq('test')&d.fold.eq('part4_highway')].accuracy.mean()*100
        rows.append([LABELS[variant],headline(root,variant,'test'),f'{highway:.2f}',headline(root,variant,'external_test'),headline(root,variant,'test','bce')])
    return rows


def main():
    runs=[MAIN,BALANCED,BASE/'20261010_native5_without_micro',BASE/'20261010_native5_with_micro']
    audits=[json.loads((r/'verification.json').read_text()) for r in runs]
    assert all(a['status']=='passed' for a in audits)
    nfits=sum(a['completed_fits'] for a in audits)
    counts=pd.read_csv('data/prepared_highspeed_20261010/counts.csv')
    main_table=table(['方案','Vicolungo 准确率 %','其中 highway %','CATS 留出准确率 %','Vicolungo 校准 BCE'],comparison_rows(MAIN,list(LABELS)))
    paired=[]
    for root,description,left,right in [(MAIN,'普通混合 − 原数据','joint','core'),
        (MAIN,'外部 SSL − 核心 SSL','ssl_external','ssl_core'),(BALANCED,'加权混合 − 加权原数据','joint','core')]:
        frame=pd.read_csv(root/'seed_means.csv').query('selection=="bce_temperature"')
        cells=[description]
        for evaluation in ['test','external_test']:
            pivot=frame[frame.evaluation.eq(evaluation)].pivot(index='seed',columns='variant',values='accuracy')
            delta=(pivot[left]-pivot[right])*100
            cells.append(f'{delta.mean():+.2f} ± {delta.std(ddof=1):.2f}')
        paired.append(cells)
    paired_table=table(['配对差值（百分点）','Vicolungo 均值 ± seed SD','CATS 均值 ± seed SD'],paired)
    balanced_table=table(['类别加权方案','Vicolungo 准确率 %','其中 highway %','CATS 留出准确率 %','Vicolungo 校准 BCE'],comparison_rows(BALANCED,['core','joint']))
    native_rows=[]
    for include in ['without','with']:
        p=BASE/f'20261010_native5_{include}_micro'
        native_rows.append(['不加入 MicroSim' if include=='without' else '加入 MicroSim',headline(p,'ssl_external','test'),headline(p,'ssl_external','external_test'),headline(p,'ssl_external','test','bce')])
    native_table=table(['5 Hz 外部预训练方案','Vicolungo 准确率 %','CATS 准确率 %','Vicolungo 校准 BCE'],native_rows)
    data_rows=[]
    for campaign in ['Vicolungo','ASta','Casale','CATS','Vanderbilt','NGSIM']:
        c=counts[counts.campaign.eq(campaign)]
        value=lambda y:int(c.loc[c.label.eq(y),'windows'].sum())
        usage={'Vicolungo':'核心训练 / 验证 / 测试','ASta':'已转换，本轮不加入训练','Casale':'已转换，本轮不加入训练',
               'CATS':'可信外部监督 / 自监督；test9、10 整段留出','Vanderbilt':'外部监督 / 自监督','NGSIM':'仅自监督；Human 标签为交通背景假设'}[campaign]
        data_rows.append([campaign,value(0),value(1),usage])
    data_table=table(['来源','Human 窗口','ACC 窗口','本轮用途'],data_rows)
    diagnostics=[]
    for root,prefix in [(MAIN,''),(BALANCED,'类别加权：')]:
        h=pd.read_csv(root/'training_diagnostics.csv')
        d=pd.read_csv(root/'comparison.csv')
        for variant,g in h.groupby('variant'):
            if variant=='tree':continue
            selected=d[d.variant.eq(variant)&d.evaluation.eq('test')&d.selection.eq('bce_temperature')]
            diagnostics.append([prefix+LABELS[variant],f'{int(g.best_bce_epoch.eq(0).sum())}/{len(g)}',
                f'{g.best_val_bce.mean():.4f}',f'{g.last_val_bce.mean():.4f}',f'{int(selected.bce.lt(np.log(2)).sum())}/{len(selected)}'])
    diagnostic_table=table(['方案','选中 epoch 0','最佳验证 BCE','末轮验证 BCE','校准测试 BCE < ln2'],diagnostics)
    secondary=table(['方案','按验证准确率选模：Vicolungo %','按验证准确率选模：CATS %'],
        [[LABELS[v],headline(MAIN,v,'test',selection='accuracy_temperature'),headline(MAIN,v,'external_test',selection='accuracy_temperature')] for v in LABELS])
    report=f'''# 高速外部数据接入与实验报告（2026-10-10）

已下载、转换并完成 **{nfits} 次正式训练 / 拟合**：10 Hz 主实验 39 次，类别加权补测 18 次，MicroSim 的 5 Hz 配对实验 6 次。另有运行冒烟检查，不计入正式结果。数据、模型、逐窗口预测和审计均保存在本地；历史数据和 checkpoint 未覆盖。本轮完成的是轨迹模型实验，没有运行视频端到端微调。

**结论：新数据可用，但本轮没有解决原项目的跨来源泛化问题。** 类别加权的原数据基线为 61.23%，混合外部数据为 61.05%；同一批 CATS 车辆的留出记录从 59.07% 提高到 89.97%。高速路留出组仍约 50–52%。外部数据能帮助识别已覆盖的实验条件，尚未形成向原项目测试来源的稳定迁移。

## 1. 结果如何解释

以下主表预先使用**验证 BCE 选 checkpoint、验证集温度校准、固定 0.5 阈值**。先对三个来源折等权平均，再统计三个随机种子的均值与标准差。树模型每折拟合一次。CATS 是同一批留出窗口在不同模型上的结果，不是九个独立测试数据集。

{main_table}

新增外部数据的作用应按配对差值判断；下面同时列出预训练的等步数对照，以及后文的类别加权补测：

{paired_table}

![四秒准确率对比](artifacts/experiments/20261010_highspeed_v1/figures/accuracy_comparison.svg)

误差线只反映三个 seed 的波动，不是对独立驾驶场景泛化能力的置信区间。滑动窗口有重叠，不能将数千窗口当成数千独立试验。

原来的 **63.30%** 来自不限速的旧 12 特征协议；本轮改为 >50 mph、原始标量速度、统一净距和 11 个纵向特征。只能与本表的“原数据 Transformer”比较，不能把协议变化算成外部数据收益。

## 2. 下载和数据处理

{data_table}

10 Hz 共 **23,430** 个窗口，每个 `[40,11]`；4 秒内共 40 个采样点，首末间隔 3.9 秒。训练按 4 秒步长抽样，验证和测试按 1 秒步长，表内为全部可用重叠窗口，不能理解为实际训练样本数量。

- **CATS 原始数据恢复了 Human**：实验说明中车 2/3 为 ACC、车 4/5 为 Human；从 3→4 和 4→5 相邻车对恢复人工驾驶数据。这比直接使用 Ultra-AV 的 ACC 整理表更适合本项目。标签来自固定实验角色，不是逐帧 CAN 控制开关；缺帧 / 空工作表不插值补齐。CATS 采用 GPS 参考点距离减假设车长 4.92 m，仍有净距偏差风险。
- Vanderbilt 使用明确标识的 Ultra-AV v4 step2 文件，只读取速度和净距，导数重新按窗口计算；两个片段作为同一采集来源。官方 Tennessee 和 Arizona 原始 Box 链接此次返回 404，**Arizona 未下载、未进入实验**。
- NGSIM 下载 I-80 / US-101 全部高速小汽车候选，并补取前车状态。按采集时段、车辆、连续时间、固定前车与车道配对；ft / ft/s 转为 SI，净距为车头位置差减前车长度。配对和质量检查后仅剩 94 个重叠窗口，其中 28 个进入非重叠预训练池。因此它不能在当前速度门槛下提供大量 Human 监督样本；没有 ACC 状态真值，本轮仅作无标签数据。
- MicroSim 从原始 ZIP 读取未平滑速度，**km/h ÷ 3.6**，分别对齐两车时间戳；不沿用 Ultra-AV 转换中的错误 mph 系数或 0.1 秒求导间隔。它以约 5 Hz 单独处理，得到 **1,064** 个合格窗口。原始积分距离减假设车长 4.5 m，初始偏置 / 车长有不确定性，因此只作自监督数据。

旧 0.63 模型没有使用 MicroSim；此次发现并修正的 MicroSim 问题不能解释旧模型的低准确率。

统一字段：`v_FV_long, a_FV, jerk_FV, v_LV_long, a_LV, gap, delta_v, closing_speed, delta_acc, THW, TTC`。无共同横向字段，所有来源统一删去横向输入，而不是给某个来源补零。仍采用 NPZ 的 `X/y/feature_names/window_index/split_group` 与逐窗口 CSV；**新数据不能直接加载旧 12 输入维度权重**。

筛选要求：后车整个原始窗口及处理后的速度均 >22.352 m/s（50 mph）；时间连续、控制标签恒定、净距 1–200 m、速度 <55 m/s、窗口平滑加速度绝对值 ≤8 m/s²、净距变化与相对速度积分残差 ≤8 m。七点二阶 Savitzky–Golay、加速度和 jerk 只用当前窗口计算。前车可以低于 50 mph；任务门槛施加于被识别的后车。

上游限制：OpenACC 已从 3–5 Hz 插值成发布的 10 Hz，Vanderbilt step2 继承发布方清洗；我们不能逆转或证明上游处理没有利用更长时间。本次**新增**处理严格局限于四秒输入。

原始文件及 SHA-256：[下载清单](data/external_raw_20261010/downloads.json)。来源：[CATS 固定版本](https://github.com/CATS-Lab/Filed-Experiment-Data-ACC_Data/tree/c680d560976a5b78afe485e54416730e97d3a5cb)、[Ultra-AV v4](https://doi.org/10.6084/m9.figshare.26339512.v4)、[MicroSim 原始仓库](https://github.com/microSIM-ACC/ICE)、[NGSIM 官方表](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajectories/8ect-6jqj)、[Vanderbilt 下载入口](https://acc-dataset.github.io/datasets/)。

## 3. 实验隔离与公平对照

Vicolungo 三个来源组 cyclic holdout：test=part2 / part3 / part4_highway，val=下一组，train=再下一组；不随机切分相邻窗口。每折核心训练分别为 **918 / 883 / 446** 个非重叠窗口，测试分别为 **3,506 / 1,763 / 3,624**。

CATS test1124 的 **test9 / test10 全部车辆整段留出**，共 962 个窗口（395 Human / 567 ACC），不进入监督、自监督或归一化。训练池的 CATS 其他记录与 Vanderbilt 共 887 个非重叠窗口；加 28 个 NGSIM 后外部自监督池为 915。ASta / Casale 已转换，但本轮不加入，以便辨识新增来源的作用。

- 所有方法的截尾界限 / 均值 / 标准差只拟合核心训练集；checkpoint 和温度只用对应验证集。
- 混合训练以核心 / 外部各 50% 的采样概率，保持核心类别比例的期望值，外部按类别、campaign、记录平衡。每 epoch 的更新数和最大监督更新预算与核心基线相同；因早停时刻不同，实际总更新数可能不同。混合训练每 epoch 对核心数据的重复暴露次数会减少。
- 自监督掩蔽连续 1 秒的全部特征，重建速度 / 前车速度 / 净距；20 个预训练 epoch，每 epoch 抽样数等于核心训练窗口数。原数据 SSL 与外部 SSL 更新数相同，避免将额外计算误认为外部数据收益。之后训练全部轨迹编码器与分类头；不是冻结编码器，也不是视频端到端。
- Transformer：d_model=128，2 层，4 heads，FFN=512，dropout=0.2，lr=3e-4，weight decay=0.01，label smoothing=0.05，最多 40 epoch。初始化 checkpoint 也参与验证选模。训练在 CPU 上完成。
- 种子 42 / 52 / 62；树模型 100 轮、7 叶、L2=10。完整配置见 [主实验配置](configs/highspeed_20261010.json)，逐折数据见 [comparison.csv](artifacts/experiments/20261010_highspeed_v1/comparison.csv)。

## 4. 为什么仍可能失效

在此次高速窗口里，类别和车距的关系随来源反转：

| 来源 | Human 平均净距 m | ACC 平均净距 m | Human 平均 THW s | ACC 平均 THW s |
| --- | --- | --- | --- | --- |
| Vicolungo part2 | 38.64 | 49.09 | 1.238 | 1.576 |
| Vicolungo part3 | 47.15 | 41.01 | 1.473 | 1.342 |
| Vicolungo highway | 43.96 | 48.22 | 1.426 | 1.537 |
| CATS | 28.64 | 42.71 | 1.168 | 1.743 |

这是数据中的描述性证据，**不能单凭均值证明某个网络只学了车距**。但它说明“车距大就是 ACC”不是稳定规则；采集段、车辆、驾驶者、设定时距可能混杂。CATS 的 ACC / Human 平均速度相近（约 24.5 m/s），Vicolungo 则约 31 m/s，仅同为 >50 mph 仍有明显场景差别。NGSIM 窗口加速度 RMS 均值约 1.97 m/s²，明显高于 CATS / Vicolungo；可能同时包含驾驶工况与测量误差差异，不能全解释为驾驶风格。

highway 留出折的核心训练来自 part3，ACC 占 29.6%，而验证来自 part2，ACC 约占 60.9%。源间类别比例变化和条件分布变化同时存在。四秒窗口在匀速跟驰时可能缺少控制器响应线索；这一可辨识性问题不能靠增加网络参数保证解决。

### 类别加权补测

观察主实验早期验证诊断后新增这组探索性实验；此前部分测试结果已经看过，**不是从未查看测试集的全新确认试验**。保持三个 seed / 三折 / 模型 / 数据 / 每 epoch 更新数 / 最大 epoch 数 / 早停规则不变，只令每个训练样本 BCE 权重为 ACC `0.5/p_train`、Human `0.5/(1-p_train)`，分别对核心及混合训练补测。实际早停 epoch 可以不同。类别比例来自核心训练，绝不从验证 / 测试估计训练权重。

{balanced_table}

这些结果应与主表对应的 core / joint 配对比较，不能把类别加权的收益全归因于新增数据。配置：[类别加权补测](configs/highspeed_balanced_20261010.json)。

### 过拟合和选模

{diagnostic_table}

SSL 的 epoch 0 表示完成自监督但尚未监督微调；core / joint 的 epoch 0 是随机初始化。末轮并不是交付 checkpoint。温度校准能缓解概率过度自信，**不能改变 0.5 阈值分类结果**；仍然有来源折的监督训练无法超过初始验证 BCE。不能宣称过拟合或跨域泛化已解决。

为方便对照，保留事先定义的第二种选模规则，而不是用测试结果挑选规则：

{secondary}

## 5. MicroSim 单独 5 Hz 对照

全部来源都以统一 5 Hz / 20 点处理，窗口内部重新算同样 11 个特征。MicroSim 两车插值的全部原始 donor 时间跨度必须 ≤4 秒；逐窗口重读 ZIP 审计后最大 3.999950 秒。不把 5 Hz 插值成 10 Hz 后声称获得额外信息。

{native_table}

这组只用 seed 42、三个来源折，每组 3 次训练；两组只有自监督池是否含 MicroSim 不同，预训练总步数固定。加入一个来源也会改变各来源的采样份额。它与 10 Hz 主表的采样率及平滑时间尺度不同，不能直接相减归因于 MicroSim。此结果未显示在 Vicolungo 上的收益，不应直接采用，也不能推广为所有预训练方法均无效。

## 6. 当前建议与剩余工作

1. 以本轮 matched baseline 和来源留出结果为依据；外部数据是否有用，应同时看原项目来源和 CATS 留出表现。CATS 的同一车型固定角色与同一天实验仍可能形成捷径，较高结果不等于未知车辆 / 未知道路准确率。
2. 保留新数据接入和自监督入口，但不要仅凭增加预训练数据就更换当前主模型。优先扩充**同一车辆 ACC 开 / 关、相同速度 / 时距、跨日跨道路**的四秒片段，并按完整车辆 / 日期划分；高速 Human 和明确控制状态比更多单一 ACC 片段更关键。
3. 下一步可用训练侧的车辆 / 来源信息约束表示，并做同车模式切换对照；比较去除绝对车距 / 时距后的性能，检查是否依赖来源捷径。这些是建议，尚未运行，不能承诺提升。
4. 接上游视频时先采集配对视频与真实控制状态，测检测 / 跟踪误差下的四秒性能，再做固定关联条件下图像编码器与轨迹头的联合微调。当前公开轨迹不能为上游相机模型提供端到端像素梯度监督；详见 [上游接入分析](UPSTREAM_INTEGRATION_20261010.md)。

## 7. 产物、审计和复现

- [10 Hz NPZ 与清单](data/prepared_highspeed_20261010/)；[5 Hz 含 MicroSim 版本](data/prepared_highspeed_native5_20261010/)。原始下载位于 `data/external_raw_20261010/`，所有产物另存，未覆盖旧数据。
- 17 项测试通过；10 Hz **23,430** 个窗口重算与来源隔离审计通过；5 Hz **24,475** 个窗口审计通过，其中所有 1,064 个 MicroSim 窗口重读原始 donor 验证。
- **{nfits} 次拟合**的训练统计、选模、预测指标和 checkpoint 重载审计通过；逐条重载预测共 **{sum(a['checkpoint_predictions_checked'] for a in audits):,}** 条（包含两个选模规则、重复模型预测，并非独立窗口数），最大 logit 差 **{max(a['checkpoint_max_abs_diff'] for a in audits):.3g}**。另核对 45 对神经网络实验的初始预测 / 统计 / 划分一致，以及三个 5 Hz 配对实验仅增加 269 个 MicroSim 训练窗口。
- 每个实验目录保存配置、源码快照、split 索引、归一化参数、history、温度、两种 checkpoint、逐窗口预测、逐 seed 汇总和 `verification.json`。汇总：[主实验](artifacts/experiments/20261010_highspeed_v1/seed_summary.csv)、[类别加权](artifacts/experiments/20261010_highspeed_balanced/seed_summary.csv)。
- [PDF 图](artifacts/experiments/20261010_highspeed_v1/figures/accuracy_comparison.pdf) / [SVG 图](artifacts/experiments/20261010_highspeed_v1/figures/accuracy_comparison.svg)。分析脚本 `src/analyze_highspeed.py`、报告脚本 `src/report_highspeed.py`，图脚本与样式保存在 figures 目录。

环境为当前机器的 PyTorch 2.6.0+cu124（CUDA 不可用）、NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.7.2、SciPy 1.15.3；新增 openpyxl / pyproj 等装于 `/tmp/identification-deps`，该目录不是永久环境。[记录的依赖版本](configs/highspeed_requirements_20261010.txt) 供在持久项目环境复现。

```bash
# 从仓库根运行；现有输出禁止覆盖，复跑时更换 --output 或配置里的 output。
export PYTHONPATH=/tmp/identification-deps
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/identification-mpl
python3 data/fetch_highspeed_sources.py
python3 data/prepare_highspeed_data.py --output data/prepared_highspeed_new
# 配置 prepared 指向新路径，output 使用一个新的实验目录。
python3 src/highspeed_experiment.py --config configs/highspeed_20261010.json
python3 src/highspeed_balanced_experiment.py --config configs/highspeed_balanced_20261010.json
python3 data/prepare_native5_data.py --output data/prepared_native5_new
python3 src/highspeed_native5_experiment.py --config configs/highspeed_native5_without_micro_20261010.json
python3 src/highspeed_native5_experiment.py --config configs/highspeed_native5_with_micro_20261010.json
python3 src/analyze_highspeed.py --run artifacts/experiments/20261010_highspeed_v1
python3 -m unittest discover -s tests -v
```

新目录名需同步到配置；每个 run 的 analyzer 应指向对应目录。源码和数据快照固定后，分析不会重新训练或挑选测试最优模型。
'''
    Path('HIGH_SPEED_EXPERIMENT_20261010.md').write_text(report)
    shutil.copy2(__file__,MAIN/'report_generator_snapshot.py')
    print('Wrote HIGH_SPEED_EXPERIMENT_20261010.md')


if __name__=='__main__':main()
