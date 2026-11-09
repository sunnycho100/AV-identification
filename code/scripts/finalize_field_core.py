"""Reconcile the field-first protocol with workspace and manuscript narratives."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
EXP=ROOT/'experiments/field_core_20261101'

def replace(rel,old,new):
    p=ROOT/rel;s=p.read_text(encoding='utf-8')
    if new in s:return
    if old in s:s=s.replace(old,new)
    elif new not in s:raise ValueError('Expected text not found: '+rel+' '+old[:70])
    p.write_text(s,encoding='utf-8')

def main():
    replace('paper/sections/data_collection.tex',
        r'\subsection{Time alignment, geometric references, and quality control}',
        r'''The primary field study is restricted to Verona Road/SE, Todd Drive/SE, and Seminole Highway/NW: 43 recordings comprising 22 ACC and 21 HV recordings, with 40 paired positioning files. Whitney Way is retained as backup; the remaining views are supplementary. The unit of analysis is the instrumented vehicle and its immediate leader during a sufficiently visible interval. Other road users are not classification subjects. Full-recording, all-vehicle trajectory extraction is not required, and the end of a leader's visibility ends the usable following interval without discarding the target's remaining trajectory.

\subsection{Time alignment, geometric references, and quality control}''')
    p=ROOT/'../paper/sections/trajectory_reconstruction.tex';s=p.read_text(encoding='utf-8')
    start=s.index('The adapter requires a constant follower--leader identity') if 'The adapter requires a constant follower--leader identity' in s else s.index('The field adapter exports only')
    s=s[:start]+r'''The field adapter exports only the image-identified instrumented follower and its immediate leader. Filename ACC/HV labels belong to the follower alone. A leader change, lane/direction change, or observation gap longer than 0.20~s breaks a following interval. Both boxes must remain inside the image, and only detection-supported observations enter the pair series. Target-only observations are retained when no leader is available. The current adapter reuses cached MOT candidates; its exported dataset does not include unrelated road users. Extending the legacy car-only tracker to heavy-vehicle leaders remains necessary where the immediate leader is a truck or bus.

Longitudinal speeds and positive net gap are resampled to 10~Hz without extrapolation or time warping. The field protocol permits variable observation lengths; the present implementation compares 20, 30, and 40 samples, corresponding to nominal two-, three-, and four-second windows. There is no minimum follower-speed requirement. Acceleration and gap--relative-speed consistency are retained as review flags rather than automatic domain exclusions. These rules are distinct from the fixed four-second, above-50-mph public-trajectory benchmark in Section~\ref{sec:identification}. Primitive arrays have shape $N\times T\times3$, with separate provenance and quality records. Window length and normalization must ultimately be selected using field training/validation recordings; overlapping windows or observations from the same drive must not cross evaluation splits.

The restored Transformer can consume shorter prefixes of its learned positional embedding at the original sampling interval. This supports interface testing but does not establish transfer accuracy: normalization and model weights learned from high-speed sources require field-domain assessment. Repeated-observation fusion additionally requires supported identity continuity, which is not inferred merely from similar motion.
'''
    p.write_text(s,encoding='utf-8')
    replace('paper/sections/identification.tex',
        'with $T=40$ samples at 10 Hz and $d=11$ features.',
        'with $d=11$ features. The public-trajectory benchmark below fixes $T=40$ samples at 10 Hz; the separate field protocol permits shorter observations.')
    replace('paper/sections/experiments.tex',r'\input{sections/field_branch_results}',
        '\\input{sections/field_branch_results}\n\\input{sections/field_core_results}')
    replace('paper/sections/framework.tex',
        'Its output retains identities, timestamps, positions, velocities, and observation-quality flags.',
        'For the primary three-view field study, image references identify the instrumented vehicle, reviewed intervals delimit useful visibility, and the output retains only its trajectory and immediate-leader interactions, with identities, timestamps, positions, velocities, and quality flags.')
    replace('paper/Manuscript.tex',
        'with 56 paired positioning files and recording-level ACC/HV assignments.',
        'with 56 paired positioning files and recording-level ACC/HV assignments; the primary field scope comprises 43 recordings from three camera views and focuses on the instrumented vehicle and its immediate leader.')
    replace('paper/sections/conclusion.tex',
        'GPS-supervised training remains blocked by registration.',
        'GPS-supervised training still requires verified registration. The field-first adapter removes the public benchmark speed floor and supports shorter target--leader windows; real two- and three-second windows pass the Transformer interface check. The primary three-view cohort contains 43 recordings, most of which still require target-specific extraction. Absolute GPS registration is not a prerequisite for camera-relative following features.')
    # Preserve historical results, but explicitly name the old protocol.
    for rel in ['paper/sections/field_branch_results.tex','scripts/summarize_field_branches.py']:
        replace(rel,'original pilot has zero eligible field windows.',
                'original pilot has zero windows under the source protocol.')
        replace(rel,'Thus these runs also yield no eligible field classification windows.',
                'This is a result under the source protocol, not the revised field eligibility rule in Section~\\ref{sec:field-core-results}. The speed comparisons use the supplied video/GPS timing, whose absolute synchronization has not been independently verified.')
    replace('paper/sections/field_pilot_results.tex',
        'None satisfy the unchanged whole-window 50-mph requirement.',
        "None satisfy the source protocol's whole-window 50-mph requirement; this gate is not imposed by the revised field protocol.")
    replace('paper/sections/field_pilot_results.tex',
        'The adaptation therefore does not improve speed accuracy over the matched observation interval, despite its lower pseudo-label validation loss.',
        'The adaptation therefore does not improve this speed-agreement diagnostic over the matched interval, despite its lower pseudo-label validation loss. These comparisons use the supplied video/GPS time mapping rather than independently verified synchronization.')
    # Keep the workflow generator consistent with the new figure.
    replace('scripts/summarize_field_branches.py',
        "'Native frame times; retain source identities'",
        "'Three core views; target identity; select visible intervals'")
    replace('scripts/summarize_field_branches.py',
        "'5  Metric trajectories and interactions','Stitch / split / RTS; lanes, leader IDs, net gap'",
        "'5  Target and immediate leader','Stitch / split / RTS; target trajectory and net gap'")
    replace('scripts/summarize_field_branches.py',
        "'10 Hz, 40 samples; quality gates -> CNN / Transformer / RF'",
        "'10 Hz; variable windows; no field speed floor'")
    replace('scripts/summarize_field_branches.py',
        "'Image marker establishes target identity\\nGPS speed at video timestamps",
        "'Image marker establishes target identity\\nGPS checks require verified timing")
    from summarize_field_branches import workflow
    workflow()
    field=r'''\subsubsection{Field-first scope and registration diagnosis}
\label{sec:field-core-results}
The revised field protocol is applied to the existing 15-m excerpts without rerunning detection or changing their original recording roles. After excluding detection gaps and boxes crossing the image boundary, the instrumented HV and its immediate leader supply 93 paired observations over 20.153--23.423~s, a 3.270-s interval. At a one-second stride this yields two 20-sample windows and one 30-sample window; no 40-sample window fits. Speeds in these windows range from 5.374 to 8.229~m/s. The two observation lengths reuse the same passage and are alternative inputs, not three independent experiments. The ACC excerpt retains 53 detection-supported, fully visible target states but no usable detected leader in the cached car-only output. This does not establish that the recording lacks a physical leader.

All fifteen restored single-segment Transformer members produce finite outputs on the actual two- and three-second inputs, using the original 0.1-s sampling interval and shorter positional-embedding prefixes. This is an interface check only. Between 18.2\% and 27.0\% of feature entries are outside individual members' source-normalization clipping bounds, illustrating domain mismatch. The present exported windows are HV-only and belong to the existing test recording, so they are not used to fine-tune the classifier or report field accuracy. Seven-frame screening sheets and a processing catalog cover all 43 core recordings; the new pair-export check itself covers only the two Todd excerpts.

To diagnose the optional GPS-supervision failure, the original 16.26-m registration is reproduced on 89 detected, image-marker-confirmed training observations: 16 from \texttt{AV\_T\_EW\_6} and 73 from \texttt{HV\_T\_EW\_5}. A shared unit-scale rotation and translation from common east--north coordinates to road coordinates gives 46.677~m pointwise position RMSE. The two GPS point-set centroids are 122.900~m apart, whereas the corresponding visual centroids are only 1.427~m apart. Since rigid transforms preserve separation, the centroid discrepancy alone implies an RMSE lower bound of 46.646~m at these sample weights. It therefore accounts for almost all of the failed shared fit.

Separate per-recording rigid fits yield 0.480 and 0.544~m residuals on their respective short fitted intervals; these are in-sample path-shape residuals, not independently validated positioning errors. With ACC timing fixed, a diagnostic shift of the HV GPS query time by $-14.608$~s reduces the shared residual to 3.427~m. Other fixed ACC shifts produce different HV shifts and residuals. No such shift is adopted as corrected synchronization: the short same-direction observations do not independently resolve timing, camera origin, or absolute reference bias. The nearest-frame errors of at most 0.032~s describe rounding after the supplied time mapping, not validation of that mapping.

The resolution path is to verify source pairing and image identity, use informative speed changes and independently timed or georeferenced crossing events, and then estimate and freeze per-recording timing with a camera-pose-specific geographic transform using designated calibration recordings. Held-out position error must be reported separately from any alignment fitted on that same trajectory. This registration problem affects absolute GPS validation and reference-supervised losses; it does not prevent visual follower/leader speed and gap extraction, because a common planar translation cancels in relative position.
'''
    (ROOT/'../paper/sections/field_core_results.tex').write_text(field,encoding='utf-8')
    report='''# 自采数据优先：范围、坐标配准诊断与实际运行结果

日期：2026-11-01。新入口为 configs/field_core.json；旧 pilot 及其输出保留。

## 46.7 米是什么

这不是 GPS 精度，也不是整套轨迹算法的定位精度。旧 pilot 在 16.26 m 高度下，取两段训练视频的图像标记确认帧，拟合共同 ENU → 道路坐标的二维旋转和平移，不拟合尺度、不搜索时间偏移。对全部 89 个匹配点计算欧氏距离平方均值的平方根，得到 46.676924 m。

| 片段 | 匹配帧 | 视频区间 | 共用变换 RMSE | 各自拟合 RMSE |
|---|---:|---|---:|---:|
| AV_T_EW_6 | 16 | 7.841–8.475 s | 99.672 m | 0.480 m |
| HV_T_EW_5 | 73 | 18.385–20.854 s | 21.882 m | 0.544 m |

46.7 m 按点数加权，并非上述两行误差的算术平均。两组视觉位置的均值分别约 (34.67, -5.75) m、(33.42, -5.06) m，相距仅 1.427 m；按现有时间映射取到的 GPS 均值却相距 122.900 m。任何刚体变换都保持两点距离，因此无法同时对齐。仅这个矛盾就给出 46.646 m 的整体误差下界。

分别拟合之所以得到约半米，是每段都允许一个自己的平移，吸收了片段间偏移，而且拟合区间很短。不能把这个数作为“误差已修到半米”的结论。旧代码里已有 trial_time_offsets.py、gps_benchmark.py 等时间偏移诊断，并不是完全没有相关模块。

现有数据首先提示逐段视频/GPS 同步或裁剪对应有问题，但仅凭这两段还不能排除错误配对、绝对 GPS 偏差或其他注册原因。两份 GPS 的全段位置位移与速度积分相差约 0.010 m、0.040 m，内部很一致；这不能独立证明其绝对坐标或视频时间正确。CSV 的 nearest_frame_time_error_sec 最大约 0.03 s，只是按已有时间映射取最近帧的误差，不是相机与 GPS 时钟同步精度。

为诊断而固定 AV 时间、拟合 HV 的 GPS 查询偏移，得到 -14.608 s、残差 3.427 m；改变 AV 的固定偏移，HV 解也变化。这些解没有被写回数据或用作训练标注，不能当作已确认的同步修正。相机高度 15/17 m 的选择也不能直接消除片段间约 123 m 的对应矛盾。

## 怎么解决

1. 先按车辆图片及车顶设备/车身标记确认自车，核对文件配对、原视频版本、解码 PTS 和裁剪零点。
2. 在同一摄像头姿态中选择共同的道路横断面/标线过点事件；结合速度变化明显的片段核对时间对应。要解除绝对位置与时间的混淆，需要独立可信的时间锚点、已知地理位置的过点事件，或信息充分且经验证的速度变化；只有同一画面中的过线事件还不够。匀速短段中，沿路平移与时间偏移相互混淆，不能只看拟合误差。
3. 用指定标定/训练片段确定逐段时间映射、相机姿态和共享地理变换，并检查留出片段。如果所需时间超出当前 GPS 裁剪范围，应回到较长原始记录，不能外插补造。
4. 绝对 GPS 定位验证与对齐后的形状误差分开报告。在此之前继续做视觉自车/前车速度、间距与文件名标签分类；相对位置会抵消共同平移，并不依赖绝对 GPS 注册完成。

复现：.venv-classifier/Scripts/python.exe scripts/diagnose_field_registration.py
详见 registration/diagnosis.json、逐帧 matched_points.csv 和 registration_diagnosis.png。

## 新实验范围

| 核心摄像头 | 视频 | AV/ACC | HV | GPS |
|---|---:|---:|---:|---:|
| Verona SE | 16 | 8 | 8 | 15 |
| Todd SE | 13 | 7 | 6 | 12 |
| Seminole NW | 14 | 7 | 7 | 13 |
| 合计 | 43 | 22 | 21 | 40 |

Whitney 为备用，其他视角为补充。core_clips.csv 列出所有核心视频及实际处理状态；screening 下每段有七帧抽样图。抽样不等于逐帧完成身份审核，不因自动匹配暂未成功就说没有自车。缺 GPS 的三段仍可用于视觉分类。

后续按“低成本浏览 → 图片确认自车 → 确定自车和前车同时清晰的时间/图像范围 → 只在所选范围做高帧率检测与跟踪 → 导出两车及质量记录”组织。对远处、严重遮挡和出画面区间按各视角实际可见性筛选，不要求全视频、全交通流导出。前车出画后仍保存自车轨迹。

新适配器没有速度下限，分别导出二、三、四秒窗口，保持原始 10 Hz，不把短片段拉伸为四秒。负间距、无效数值、长观测空缺或前车切换仍需正确处理；加速度/间距一致性作为审核标记。窗口和归一化在训练/验证集上决定，同一驾驶通过三个相机形成的记录需核对分组，不能让重叠窗口分散到训练和测试中。

## 本次实跑

复用两段已完成 15 m 检测/MOT 的缓存，只导出自车与其前车，不重新运行无关交通流。

- HV_T_EW_6：去除出画边界后，保留 20.153–23.423 s 的 93 个两车检测支持点，连续跨度 3.270 s。生成两个 [20,3] 窗口和一个 [30,3] 窗口；它们来自同一次 HV 通过，不能当成三个独立试验。
- AV_T_EW_6：保留 53 个可见、自车检测支持点；当前缓存没有合格前车轨迹，所以保存自车轨迹而不虚构跟驰输入。这不等于视频里没有前车。
- 15 个 Transformer 主模型都已实际接收二/三秒自采输入并输出有限数值，无填充、无时间拉伸。新样本速度约 5.37–8.23 m/s。旧高速数据归一化会裁剪约 18%–27% 的特征项，所以这些输出只用于接口验证，不能当成已适配的分类结果。
- 当前样本只有 HV 且属于旧测试片段，没有拿它训练，也没有计算路测分类准确率。零速允许、前车切换及长缺帧必须断窗的检查通过。

复现：

    .venv-classifier/Scripts/python.exe scripts/build_field_core_dataset.py --previews
    .venv-classifier/Scripts/python.exe scripts/check_field_core_classifier.py

## 还需要补的具体工作

首先在三个核心视角逐段确认车辆和前车，并处理选定的近/中距离区间。旧 MOT 的 car-only 过滤以及当前权重对货车/巴士的漏检，会让真实前车缺失；必须支持实际前车类别。逆向/背向视角不能只依赖当前正面机盖模板，需要结合车顶设备、车辆外观与片段内连续运动。

新导出器已经限制学习数据为自车/前车，但其两个实跑例子仍复用旧全场景候选轨迹；面向全部 43 段的自动选段、跨视角身份确认与仅目标对跟踪尚未全部完成。不能把“建好 43 段清单和抽样图”说成“43 段已跑通”。可先形成两类、按驾驶记录隔离的视觉样本集，进行短窗 Transformer 适配；GPS 同步及绝对定位验证作为并行工作。
'''
    (EXP/'REPORT.md').write_text(report,encoding='utf-8')
    replace('docs/PROJECT_HISTORY.md','## 先看这些文件',
        '## 当前自采实验入口\n\n按数据 README 指定的三个视角组织核心实验：Verona SE、Todd SE、Seminole NW，共 43 段（22 AV/ACC、21 HV），40 份 GPS。仅导出自车和前车；窗口长度可变、无最低速度门槛。详细的 46.7 m 诊断、解决步骤和实跑范围见 [自采优先报告](experiments/field_core_20261101/REPORT.md)，配置见 [field_core.json](configs/field_core.json)。\n\n新适配器已从两段 15 m 缓存中生成二/三秒 HV 样本，15 个 Transformer 已通过真实短窗接口检查；其余核心片段目前完成目录和七帧抽样，尚未全部提取。旧报告中的“零合格窗口”仅指旧高速四秒协议，不能解释为自采数据不可用。尚未进行路测分类器微调或得到路测分类准确率。\n\n## 先看这些文件')
    replace('docs/PROJECT_HISTORY.md','## 已完成与尚未获得的结果','## 旧 pilot 及研究分支记录（保留原协议）')
    replace('docs/PROJECT_HISTORY.md','当前仍无合格分类窗口。','在旧高速四秒协议下无合格分类窗口；新短窗结果见上方当前入口。')
    # Existing Markdown has inline code around the array; target the full sentence.
    replace('docs/PROJECT_HISTORY.md','当前形状为 '+chr(96)+'[0,40,3]'+chr(96),
            '旧协议形状为 '+chr(96)+'[0,40,3]'+chr(96))
    print('Updated field-first report, manuscript scope/methods/results and workflow figure.')

if __name__=='__main__':main()
