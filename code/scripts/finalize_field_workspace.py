"""Refresh workspace entry points and reproducibility evidence for this pilot."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;EXP=ROOT/'experiments/field_pilot_20261101'
readme='''# 路侧摄像头轨迹提取与 AV/ACC 识别工作区

三份压缩包已整合；路测视频的小规模本地 GPU 试验和论文修订已完成。原始数据、原始权重和 ZIP 保留。日期：2026-11-01。

## 先看这些文件

| 内容 | 入口 |
|---|---|
| 本次整合、微调、测试的实际结果与限制 | [实验报告](experiments/field_pilot_20261101/REPORT.md) |
| 复现命令和环境说明 | [运行说明](docs/FIELD_PILOT_RUNBOOK.md) |
| 固定训练、验证、测试划分 | [field_pilot.json](configs/field_pilot.json) |
| 更新后的论文 | [PDF](paper/Manuscript.pdf) / [LaTeX](paper/Manuscript.tex) |
| 轨迹模型与标定、跟踪代码 | [trajectory](trajectory) |
| CNN、Transformer、RF 训练与已有权重 | [classification](classification) |
| 自采数据：60 段视频、56 份 GPS、32 张素材图 | [data/field](data/field) / [逐段目录](catalogs/field_clips.csv) |
| 原始迁移记录 | [轨迹项目](docs/migration) / [分类项目](docs/classification_migration) |
| 原始压缩包与修订备份 | [archives](archives) / [revisions](revisions/before_field_pilot_20261101) |
| 文件来源检查和本次软件检查 | [verification.json](catalogs/verification.json) / [pilot_validation.json](experiments/field_pilot_20261101/pilot_validation.json) |

## 已完成与尚未获得的结果

六段 Todd/SE 视频共 2,577 帧已经跑通标定、BEVHeight、AB3DMOT、平滑、车道/前车关联。基线产生 446 条轨迹片段。检测头在两段训练视频上完成两轮微调，独立验证伪标注损失从 1.7576 降到 1.6897；两段验证、两段测试视频完成前后对照。轨迹片段数和伪标注损失不是独立标注精度。

新增适配器将车辆中心距转换成净间距，生成 10 Hz、40 点的分类窗口并记录拒绝原因。此次基线和微调结果没有通过原高速度协议的合格窗口，因此**没有进行路测分类器微调，也不报告路测 ACC 分类准确率**。已有全部 75 个主模型通过实际加载与合成输入检查，CNN/Transformer 推理接口也已接通。

ACC/HV 含义按用户确认的文件名处理，仅用于图像确认的试验车。其他车辆保持未知标签。相机标定沿用现有方法；初始高度 16.26 m 仍是历史 Todd 假设，详细尺度诊断和 GPS 对照见实验报告。

历史公共训练 NPZ、原始训练数据仍未迁入。原 [README.txt](README.txt) 是历史训练数据交接清单。新生成的路测 NPZ 位于实验目录；当前形状为 `[0,40,3]`，与未迁入的历史训练 NPZ 是两回事。

## 管理与完整性

在本目录运行 `scripts/Test-Workspace.ps1` 校验原包、原始输入、备份和有意修订的文件。`catalogs/archive_files.csv` 保留原始来源哈希，`catalogs/workspace_changes.json` 记录授权修改。数据按五站点、七视角组织；四段缺 GPS，已有时间匹配质量问题保留在目录中。

可用 VS Code 打开 [identification.code-workspace](identification.code-workspace)。检测与分类依赖分别在 `.venv-bev` 和 `.venv-classifier` 中，详细包版本及执行证据保存在实验目录。
'''
(ROOT/'docs/PROJECT_HISTORY.md').write_text(readme,encoding='utf-8')
for name in ['docs/INTEGRATION.md','docs/DATA.md','classification/README.md']:
    p=ROOT/name;s=p.read_text(encoding='utf-8')
    note='> 本文件保留初始整理阶段的说明。本地视频整合、微调、测试、用户确认的文件名标签含义及当前限制，以工作区 `experiments/field_pilot_20261101/REPORT.md` 和 `docs/FIELD_PILOT_RUNBOOK.md` 为准。\n\n'
    if not s.startswith('> 本文件保留'):p.write_text(note+s,encoding='utf-8')
cfg=json.loads((ROOT/'configs/field_pilot.json').read_text())
assert not set(cfg['train'])&set(cfg['validation']) and not set(cfg['train'])&set(cfg['test']) and not set(cfg['validation'])&set(cfg['test'])
checks=[]
for tag,clips in [('pilot_base',cfg['train']+cfg['validation']+cfg['test']),('pilot_ft',cfg['validation']+cfg['test'])]:
    for c in clips:
        fd=ROOT/'trajectory/data/camera-data'/c/'frames_all';dd=ROOT/'trajectory/outputs/object_detection/camera-data'/f'{c}_{tag}'
        frames=len(list(fd.glob('*.jpg')));dets=len(list(dd.glob('*_pred.json')))
        assert frames==dets and frames>0
        rd=ROOT/'trajectory/outputs/trajectories'/f'{c}_{tag}';run=json.loads((rd/'run.json').read_text())
        assert run['vehicles']==len(list(rd.glob('vehicle_*.csv')))
        checks.append(dict(clip=c,tag=tag,frames=frames,prediction_files=dets,trajectory_files=run['vehicles']))
result=subprocess.run([sys.executable,str(ROOT/'scripts/test_field_adapter.py')],capture_output=True,text=True)
(EXP/'adapter_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
if result.returncode:raise RuntimeError('Adapter tests failed: '+result.stderr)
from predict_field_windows import predict,transformer
import numpy as np,torch
torch.set_num_threads(4)
a=np.tile(np.array([27,28,30],dtype=np.float32),(2,40,1));a[:,:,2]+=np.arange(40)*.1
cnn,_=predict(a);tf=transformer(a)
assert np.isfinite(cnn.acc_probability).all() and np.isfinite(tf.acc_probability).all()
smoke=json.loads((EXP/'all_models_smoke.json').read_text());assert smoke['models_checked']==75
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance=[]
for rel in ['configs/field_pilot.json','trajectory/checkpoints/ft102_run3_v2labels_ep5.ckpt','trajectory/outputs/finetune/pilot_head/head_ft_ep2.ckpt']:
    provenance.append(dict(path=rel,sha256=sha(ROOT/rel)))
report=dict(status='PASS',scope='six baseline and four adapted clip runs complete; adapter contracts; restored model inference. This is not field classification accuracy.',python=sys.version,platform=platform.platform(),recording_splits_disjoint=True,adapter_contract_tests=5,principal_models_checked=75,cnn_ensemble_members=15,transformer_ensemble_members=15,synthetic_acc_probabilities=dict(cnn=cnn.acc_probability.tolist(),transformer=tf.acc_probability.tolist()),runs=checks,provenance=provenance)
(EXP/'pilot_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS: complete outputs, disjoint recording splits, five adapter tests, 75 restored models, CNN/Transformer interfaces.')
