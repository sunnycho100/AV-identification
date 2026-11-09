"""Create the experiment report from saved outputs; never choose models or windows."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/field_todd_3s_20261102'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def main():
    inv=pd.read_csv(OUT/'inventory.csv');pred=pd.read_csv(OUT/'field/video_predictions.csv');q=pd.read_csv(OUT/'quality/clip_metrics.csv')
    public=pd.DataFrame(read(OUT/'public/results.json')['results']);field=read(OUT/'field/results.json')
    reasons={
        'AV_T_WE_4':'原始候选跨3.17秒，但第168/169帧三维位置跳约4米，过滤后最长2.64秒；特征掩码分支2.87秒。未强行跨跳点。',
        'AV_T_WE_5':'接近102.4米检测范围后无可靠三维检测；更早的近端帧车辆尚未完整进入画面。',
        'HV_T_EW_2':'有效检测只有约0.57秒，随后原始帧时间戳从5.973跳到8.976秒；更早目标位于检测范围之外。',
        'HV_T_WE_5':'目标身份在图像中连续，但中间仍有约0.63秒无可靠三维检测；特征掩码分支最长2.67秒。',
        'HV_T_WE_6':'近端目标进入画面较晚，随后视频结束；可确认身份与三维支持的片段不足3秒。'}
    inv['limitation']=inv['clip'].map(reasons).fillna('3秒输入可用；精度另看GPS诊断，非独立3D标注验证')
    inv.to_csv(OUT/'inventory.csv',index=False)
    full=inv.merge(pred,on=['clip','label'],how='left').merge(q,on='clip',how='left');full.to_csv(OUT/'video_results.csv',index=False)
    yy=np.arange(len(inv));fig,ax=plt.subplots(figsize=(11,6.7));ax.barh(yy-.17,inv.old_longest_s,.32,label='Previous export, gaps <=0.2 s',color='#a7b4c2')
    ax.barh(yy+.17,inv.recovered_longest_s,.32,label='Image-supported recovery, bounded gaps',color=['#237d87' if v else '#c48042' for v in inv.eligible_3s])
    ax.axvline(3,color='#a83e35',ls='--',label='Required 3.0 seconds');ax.set_yticks(yy,inv['clip']);ax.invert_yaxis();ax.set_xlabel('Longest supported interval (seconds)')
    ax.set_title('Todd SE: 8 of 13 recordings support an exact 3-second window');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=9);fig.tight_layout();fig.savefig(OUT/'duration_recheck.png',dpi=170);plt.close(fig)
    downloads=read(ROOT/'classification/trajectory-avlike-transformer/data/restored_20261102/downloads.json')
    assert len(downloads)==26 and all(d['matches_original'] for d in downloads)
    for d in downloads:assert hashlib.sha256((ROOT/d['path']).read_bytes()).hexdigest()==d['expected_sha256']
    lines=['Todd SE 固定三秒轨迹与缺失前车 Transformer 重训','2026-11-02','',
        '本轮结论：已复查全部13段，8段（5 AV / 3 HV）形成严格3.0秒输入。其余5段保留失败记录。',
        '已经重新下载原项目的26个公开来源文件，全部SHA256与原清单一致；重建22,371个三秒窗口。',
        '新 Transformer 从随机初始化开始训练全部参数，已完成9个公开数据模型和24个Todd留出适配模型。',
        'Todd直接迁移2/8=25.0%；留一视频适配并平均三个种子后4/8=50.0%，平衡准确率46.7%。',
        '旧结果8/13=61.5%使用0.4–2秒、不同轨迹/网络/样本范围，不能直接将两者解读为提升或下降。','',
        '1. 逐段长度（秒）','视频             旧最长     本轮最长   固定3秒输入']
    for r in inv.to_dict('records'):lines.append(f'{r["clip"]:14s} {r["old_longest_s"]:8.3f} {r["recovered_longest_s"]:10.3f}   '+('可用' if r['eligible_3s'] else '不可用'))
    lines+=['','2. 未达到3秒的原因']+[c+'：'+t for c,t in reasons.items()]
    lines+=['','3. 提取与选窗规则',
        '固定31点、10Hz，首尾实际相隔3.0秒。所有合格候选中选择相机距离均值最小的窗口；不查GPS误差或分类对错。',
        '双向光流仅帮助保持车辆身份；三维位置仍来自同标定、同原检测权重的输出。阈值降到0.05，IoU>=0.35、候选差>=0.15后再审图、去跳点和平滑。',
        '原始三维几何保持共同坐标，不对前后车分别平移。普通插值间隔上限0.2秒；有连续图像支持可到0.5秒，但这类间隔在三秒内累计不超过0.6秒。',
        '不把光流框变成三维深度，不外推，不复制短序列凑长度，不跨原视频时间跳跃。',
        'AV_T_WE_1的向前回溯曾漂到另一辆车，已剔除早于56帧的候选；HV_T_WE_6早期回溯锁到水印，已剔除85帧以前的候选。',
        'AV_T_EW_3第293帧经原图复核：真实车辆仍在画面内，投影三维框超出下边界；仅该帧采用明确记录的图像边界例外。',
        '查看 candidates/ 与 low_score_detections/ 下逐视频 candidate_review.jpg。青框是图像跟踪，绿框是关联三维检测。',
        '这是助手抽样图像复核，不是独立逐帧标注或三维精度真值。','',
        '4. 分类器如何处理缺失前车',
        '新结构：64维、2层、4头，独立自车/交互投影，前车可用性门控和可学习缺失向量。输入11特征及其有效性掩码。',
        '自车31点必须全有。前车缺失处数字归零，同时有效性为false；导数只在连续可用前车片段内计算，不把零当成测得速度或间距。',
        '训练使用完整、全部缺失、仅前段、仅后段和中间缺口五种输入；完整概率1/3，其余各1/6。公开数据的缺失是模拟观测不可用，不是标注车辆物理上没有前车。',
        '无前车/不可见/超出原审核范围保留为元数据；这些类别、文件名、GPS、绝对时间和方向不进入网络。',
        '本轮8个输入里5个没有观测到前车，3个仅局部前车。leader_available_fraction见windows.csv。',
        '恢复公开数据后取消旧50mph门槛，其余物理有效性检查保留。新三秒数据位于 classification/trajectory-avlike-transformer/data/prepared_3s_20261102。',
        '公开数据：3个Vicolungo源分组循环划分，每组3个种子；全部权重训练；仅验证集BCE选择checkpoint。各fold与seed的均值：']
    for name,g in public.groupby('pattern',sort=False):lines.append(f'  {name:8s}: 准确率 {g.accuracy.mean()*100:.2f}%，平衡准确率 {g.balanced_accuracy.mean()*100:.2f}%')
    lines+=['Todd：预先固定公开fold0/seed42作为初始模型。每折留出1视频测试，1 AV+1 HV验证，其余5个训练；公开训练标准化保持固定。',
        '全网络适配lr=3e-5，最多50轮，验证选模，种子42/52/62平均。没有用测试结果选seed或阈值。',
        '视频                    P(AV)      预测   真值']
    for r in pred.to_dict('records'):lines.append(f'{r["clip"]:20s} {r["probability"]:.4f}    '+('AV' if r['predicted_label'] else 'HV')+'     '+('AV' if r['label'] else 'HV'))
    lines+=['小样本开发评价：同车辆、同摄像头和可能相同采集批次重复出现；2段曾参与检测器训练。不是独立车辆/场景泛化准确率。','',
        '5. 轨迹精度检查（取窗固定后才查看GPS）',
        '沿用此前每段冻结的时间偏移，不重新拟合时间/尺度/速度偏差。7段有部分或完整GPS覆盖，逐视频均值速度MAE=0.786 m/s。',
        '其中AV_T_EW_6仅21/31点、AV_T_WE_3仅30/31点GPS有效。AV_T_EW_7没有GPS；仅5段能计算完整三秒行驶距离，平均绝对误差1.939米。',
        '整体新窗口可能与此前时间偏移拟合段重叠，属于描述性诊断。previous_test_region指标另外保留，但仍是同开发视频。',
        'EW的5个可比视频MAE约0.12–0.56 m/s；WE的两段约1.35和2.50 m/s。离相机近并非充分的质量保证。',
        '详见 quality/selected_3s_gps.png 和 quality/clip_metrics.csv；GPS延迟弱可辨识标记保留。','',
        '6. 文件与复现（工作区根目录运行）',
        'identification/code/.venv-classifier/Scripts/python.exe identification/code/scripts/restore_classifier_sources.py',
        'identification/code/.venv-classifier/Scripts/python.exe identification/code/scripts/prepare_three_second_public.py',
        'identification/code/.venv-classifier/Scripts/python.exe identification/code/scripts/three_second_classifier.py prepare',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/three_second_classifier.py train',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/todd_three_second_tracks.py',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/todd_low_score_recovery.py',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/todd_low_score_recovery.py --feature',
        '图像复核记录固定后：',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/build_todd_three_second.py',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/three_second_classifier.py field',
        'identification/code/.venv-classifier/Scripts/python.exe identification/code/scripts/evaluate_todd_three_second.py',
        'identification/code/.venv-bev/Scripts/python.exe identification/code/scripts/validate_three_second.py',
        'identification/code/.venv-classifier/Scripts/python.exe identification/code/scripts/report_todd_three_second.py',
        '验证已通过：固定时间契约、缺失值/NaN不影响预测、缺口两侧导数隔离、源/视频划分不泄漏、模型重载复现、编码器权重实际更新。',
        '原始视频、用户leader_review_decisions.json、旧checkpoint、旧实验和README.txt保留。']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    prefix='**2026-11-02：固定三秒复查和全网络重训已完成。** 13段Todd SE中 **8段（5 AV / 3 HV）**可形成31点、真实跨度3.0秒的近相机窗口；另外5段未凑长。原公开数据26个下载文件均与原哈希一致，已重建22,371个三秒窗口，并完成9个公开数据Transformer和24个视频留出适配模型。公开数据完整前车输入平均准确率60.5%；Todd直接迁移2/8，适配后 **4/8=50.0%**。新旧协议和测试样本不同，不能直接与此前8/13比较。新网络显式处理全部/部分前车缺失，所有参数参与重训；无前车状态不再是网络的额外标签提示。见 [本轮完整结果](experiments/field_todd_3s_20261102/RESULTS.txt)、[长度复查图](experiments/field_todd_3s_20261102/duration_recheck.png)、[三秒GPS图](experiments/field_todd_3s_20261102/quality/selected_3s_gps.png)、[逐段结果](experiments/field_todd_3s_20261102/video_results.csv) 和 [协议](configs/field_todd_3s.json)。'
    rp=ROOT/'docs/PROJECT_HISTORY.md';text=rp.read_text(encoding='utf-8')
    if prefix not in text:text=text.replace('## 当前自采实验入口\n','## 当前自采实验入口\n\n'+prefix+'\n',1)
    text=text.replace('历史公共训练 NPZ、原始训练数据仍未迁入。','历史四秒训练 NPZ 文件仍未迁入；原始公开训练来源现已按原哈希重新下载，三秒 NPZ 已重建，见上方新实验。')
    rp.write_text(text,encoding='utf-8')
    pointer=ROOT/'configs/field_active_dataset.json';v=read(pointer);v['todd_fixed_three_second_experiment']='experiments/field_todd_3s_20261102';v['todd_fixed_three_second_config']='configs/field_todd_3s.json';v['todd_fixed_three_second_status']='8/13 videos eligible; full-network mask-aware retraining; 4/8 held-video adaptation correct; separate from legacy general 43-video exports'
    pointer.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Report, plot, per-video results, README and experiment pointer updated')


if __name__=='__main__':main()
