# Python 环境与运行入口

现有机器日常使用 `code/run.ps1`；公开仓库的跨平台入口为 `python code/manage.py`，安装见 [GETTING_STARTED.md](GETTING_STARTED.md)。两套研究环境都位于 `code/`，通过文件交换连接；不要把虚拟环境复制到 GitHub。

`code/src/roadside_identification/` 是当前维护的 CPU 接口，默认冻结旧分类器。`run.ps1 identify/doctor/demo/test/release/inventory` 自动选 classifier 环境。审核产品只需标准库；历史 BEV/CUDA 入口继续使用 bev。锁文件是机器快照，不是对所有系统通用的一键安装承诺。发布检查使用的临时工具环境位于忽略的 `.runtime/` 内，不参与研究推理或训练。

| 环境 | 核心版本 | 使用范围 |
|---|---|---|
| `.venv-bev` | Python 3.10.20，Torch 1.13.0+cu117，torchvision 0.14.0+cu117，mmcv-full 1.7.0，mmdet 2.28.2 | BEVHeight/CUDA 扩展、MOT、轨迹恢复、新三秒 Transformer 训练及核验 |
| `.venv-classifier` | Python 3.10.20，Torch 2.6.0+cpu，NumPy 1.26.4，pandas 2.2.3 | 历史分类 checkpoint、历史分类适配、公开数据处理、GPS/表格分析 |

这不是要求两个模块永远分开。2026-11-03 已实测 AnyCalib 可以在 BEV 环境用现有权重完成一帧标定，新三秒分类器也能在同一 GPU 环境运行。但是 1.13 的 `torch.load(weights_only=True)` 在读取历史模型时出现 `Unsupported operand 71`；原 2.6 环境保留用于复现它们。此次整理没有为合并环境而改变权重加载策略、训练算法或推理参数。

完整包版本在 `../environments/requirements-bev.lock.txt` 和 `requirements-classifier.lock.txt`；环境快照在该目录 JSON 中。锁文件是当前安装版本记录，不代表仅靠 pip 就能在任意电脑完成安装：BEV 的已编译算子还依赖匹配的 Torch/CUDA，mmdetection3d 源码来自 `.runtime/vendor/`，AnyCalib 的补充依赖来自 `.runtime/modern-deps/`。

两套环境共用 `.runtime/python/` 中的 Python 安装。目录搬迁后已修正 `pyvenv.cfg`、vendor 搜索路径及激活脚本。不要把其中某个环境单独搬到其他电脑当作可移植安装包。

Windows 的 69 个包命令入口已依据安装元数据重新生成，包本身未重装。当前主线使用单 GPU 脚本；旧 Torch 1.13 的分布式 `torchrun` 仍有 Windows `SIGKILL` 兼容问题，不属于当前主线启动方式。

常用脚本选择：

| 脚本或操作 | 环境 |
|---|---|
| `review_web/server.py` | 任一环境；统一入口默认 bev |
| `field_full_survey.py prepare` | classifier |
| `field_full_survey.py infer` | bev |
| `todd_three_second_tracks.py`、`todd_low_score_recovery.py`、`build_todd_three_second.py` | bev |
| `restore_classifier_sources.py`、`prepare_three_second_public.py` | classifier |
| `three_second_classifier.py prepare` | classifier |
| `three_second_classifier.py train/field`、`validate_three_second.py` | bev |
| `evaluate_todd_three_second.py`、`report_todd_three_second.py` | classifier |
| `todd_experiment_train.py`、历史模型 `smoke_models.py` | classifier |

VS Code 默认解释器设为 BEV；需要运行历史分类脚本时使用任务或 `run.ps1 python -Environment classifier ...`，不必反复修改编辑器解释器。
