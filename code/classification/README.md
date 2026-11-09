> 本文件保留初始整理阶段的说明。本地视频整合、微调、测试、用户确认的文件名标签含义及当前限制，以工作区 `experiments/field_pilot_20261101/REPORT.md` 和 `docs/FIELD_PILOT_RUNBOOK.md` 为准。

# AV / ACC 识别模块

**独立识别工程已经补齐。** 新上传的 `av_identification_module_20261101.zip` 正是先前缺少的 `trajectory-avlike-transformer` 项目，现完整保存在 [trajectory-avlike-transformer/](trajectory-avlike-transformer)。包含原训练代码、分类器权重、预处理/归一化统计、切分、预测和数据交接记录；没有重写算法。

原项目与 BEVHeight/MOT 视频前端是两个独立工程。补充包记录的原电脑位置为 `/home/cats/Documents/HangZhou/identification/trajectory-avlike-transformer/`。本工作区的训练代码和模型已恢复，**原始训练数据与处理后的 NPZ/窗口索引仍未迁入**。详见原工程 [README_CURRENT.txt](trajectory-avlike-transformer/README_CURRENT.txt)。

## 从这些入口查看代码

| 内容 | 入口 |
| --- | --- |
| 当前 Transformer 模型定义 | [training/transformer/networks.py](trajectory-avlike-transformer/training/transformer/networks.py) 中的 `TrajectoryNet` |
| 当前 Transformer 训练、验证选模和预测导出 | [training/transformer/run.py](trajectory-avlike-transformer/training/transformer/run.py)；配置 [experiment.json](trajectory-avlike-transformer/training/transformer/experiment.json) |
| 当前 CNN | [training/cnn/](trajectory-avlike-transformer/training/cnn) |
| 完整 11 特征 RF 训练与选参 | [training/rf/native_benchmark.py](trajectory-avlike-transformer/training/rf/native_benchmark.py) |
| 同车多段融合 | [training/multisegment_frozen/](trajectory-avlike-transformer/training/multisegment_frozen)、[multisegment_head/](trajectory-avlike-transformer/training/multisegment_head)、[multisegment_joint/](trajectory-avlike-transformer/training/multisegment_joint) |
| 窗口内特征及归一化 | [training/transformer/data_protocol.py](trajectory-avlike-transformer/training/transformer/data_protocol.py) |
| 已导出的 CNN 推理接口 | [inference_cnn/predict.py](trajectory-avlike-transformer/inference_cnn/predict.py)、[推理说明](trajectory-avlike-transformer/inference_cnn/README.txt) |
| 早期 Transformer 工程 | [src/model.py](trajectory-avlike-transformer/src/model.py)、[src/train.py](trajectory-avlike-transformer/src/train.py)；当前论文优先使用 `training/` |
| 原始冻结实验及模型 | [artifacts/experiments/](trajectory-avlike-transformer/artifacts/experiments) |
| 数据来源、哈希、真实切分 | [data_records/](trajectory-avlike-transformer/data_records) |
| 当前论文识别结果 | 完整研究工作区中的 `trajectory-avlike-transformer/paper_results/`；论文材料单独维护，不包含在代码仓库 |

Transformer 配置为 128 维、2 层、4 注意力头，使用分类 token；本次核对每个模型为 420,353 个参数。五个 seed × 三个来源折均保留。

## 已恢复的模型

| 模型组 | 主权重数 |
| --- | ---: |
| 单段 CNN | 15 |
| 单段 Transformer | 15 |
| RF | 15 |
| CNN 多段 head-only | 15 |
| CNN 多段 joint | 15 |
| **合计** | **75** |

`inference_cnn/models/` 另有 15 个 CNN 部署副本，不是新增独立训练模型。`multisegment_frozen` 使用原模型预测做融合，不另算一组新权重。scaler、验证温度、训练来源等随各模型记录保留。

## 已做的本地检查

- 三个原压缩包及 1,226 个源文件均通过哈希检查，新增包还通过原 774 条 SHA256 校验。报告：[verification.json](../catalogs/verification.json)。
- 本工程 96 个 Python 文件通过语法解析；60 个神经网络主权重均严格加载成功，并在合成物理量输入上得到有限概率，其中包括全部 15 个 Transformer。报告：[classification_validation.json](../catalogs/classification_validation.json)。
- 检查临时使用已有 OpenPAV-Infra 环境的 Python 3.12 / torch 2.7.1；它与补包推荐版本不同，没有修改该环境，也不能由此声称完整训练环境已复现。
- 15 个 RF 权重通过文件校验，未在本机反序列化。已有环境缺少 scikit-learn 1.7.2、joblib；原包 [VALIDATION.json](../docs/classification_migration/VALIDATION.json) 中全部 75 模型通过的结论属于原电脑历史验证。
- 本次没有重训、没有用真实测试数据重新计算准确率，也没有运行完整视频到分类标签的流水线。

可在装有 torch、NumPy、SciPy 的环境中，从工作区根目录运行 `python scripts/Check-ClassifierModels.py` 重复神经网络检查。按 [requirements.identification.txt](trajectory-avlike-transformer/requirements.identification.txt) 准备原工程推荐的独立环境后，可运行其 `smoke_models.py` 检查全部 75 个主模型。

## 数据接入与继续训练

尚需从原电脑迁回 `data/prepared_highspeed_20261010/`，特别是 `windows_4s.npz`、`windows.csv` 以及配套连续轨迹和协议。当前 `data/` 包含准备/审计脚本，不能将“有 data 目录”理解为已带入训练数据。

`data_records/records/prepared_protocol.json` 已保存 NPZ 与 CSV 的准确 SHA256。数据到位后，在识别工程根目录运行：

```text
python configure_paths.py --data /实际路径/prepared_highspeed_20261010
```

该命令验证两个数据文件的哈希，并更新 `training/*/experiment.json` 的可运行路径；不更改冻结实验。然后在相应 `training/transformer/` 或其他训练目录运行 `python run.py`。默认配置要求 CUDA，保留五 seed、三折及原训练预算；当前尚未启动训练。路径配置变化会被本工作区的原文件哈希基线报告为变化，属于有意配置修改。

需从原始输入重做处理时，还要迁回 OpenACC、CATS/Vanderbilt 的原始数据。PI-Koopman 辅助实验属于另一分支，本补包没有恢复该完整工程/数据。

## 与轨迹提取的接口

分类入口的基础输入是 `primitive_windows: [N,40,3]`：每 0.1 秒一个样本，依次为**后车速度、前车速度、车头车尾之间的净间距**，单位 m/s、m/s、m。窗口内处理代码将它转换为 `[N,40,11]`，再使用原训练统计量归一化。标签 `1=ACC`、`0=Human`；全窗口后车速度须大于 22.352 m/s。

`trajectory/outputs/trajectories/<clip>_<tag>/vehicle_<id>.csv` 已有时间、速度、车道、前车 ID 和中心距，但不能直接传给分类器：还需可靠身份/状态确认、时间对齐、前车轨迹联接、车辆尺寸/参考点一致的净距转换、10 Hz 重采样及质量筛选。当前导出没有完整车辆尺寸；不能直接把 `spacing_m` 中心距当成净距，或套用 CATS 的固定 4.92 m 假设。单车 GPS 同样不能提供全部三通道。

`trajectory_features.py` 是行为统计汇总工具，不替代此输入接口。已有 CNN 推理程序可接受合格 NPZ，但本次没有凭空补造前车或间距数据。Transformer/RF 的权重加载方式可参考原 `smoke_models.py`。

多段联合微调仅涉及轨迹层 CNN encoder 与分类头；真实跨摄像头 ReID，以及视觉检测、跟踪到识别的整体可微反向传播仍未实现。回测 Vicolungo 时必须使用对应留出折，不能把见过该来源的模型加入集成。详细规则见原推理说明。