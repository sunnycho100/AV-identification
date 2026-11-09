> 当前运行状态见 [本地试验报告](../experiments/field_pilot_20261101/REPORT.md)；原有 BEVHeight 调整、注释位置、历史权重证据及未启用分支见 [调整核查](../docs/BEVHEIGHT_ADJUSTMENTS.md)。下文保留初始迁移阶段的说明。

# 轨迹提取模块入口

本目录保留迁移包中的全部源码和权重。原交接文档：[README.txt](README.txt)；其“包根目录”现对应 [../docs/migration](../docs/migration)。环境说明为 [SERVER_SETUP.txt](../docs/migration/SERVER_SETUP.txt)。

主入口是 [scripts/pipeline/extract_trajectories.py](scripts/pipeline/extract_trajectories.py)：

```text
图像帧 + 相机标定
  → BEVHeight 检测 → 道路 mask → AB3DMOT
  → 轨迹拼接 → 异常跳变处理/RTS 平滑 → 车道及前车分配 → 逐车 CSV
```

## 数据已连接的位置

- 规范数据目录：`../data/field/<site>/<view>/`。
- 本工程输入：`Camera data/<clip>.mp4` 和 `<clip>_trajectory.csv`；116 个文件均为独立副本，来源映射见 [runtime_inputs.csv](../catalogs/runtime_inputs.csv)。GPS 供参考/评估，不作为普通视觉推理输入。
- 尚需生成：`data/camera-data/<clip>/frames_all/` 和 `frame_times.json`。
- 尚需准备：`outputs/calibration/camera-data/<clip>/` 下的内参、`metric_extrinsic_site.json`、`road_mask.png`。
- 结果目录：`outputs/trajectories/<clip>_<tag>/`；目前未生成真实视频推理结果。

输入副本约占额外 0.68 GB，避免硬链接导致修改运行输入时同时改动规范数据。不要直接在副本里修订数据；如确需修正，应单独保存新版本并更新来源记录。

## 权重

默认 `checkpoints/ft102_run3_v2labels_ep5.ckpt`；备选 `ft102feat_run5_feat_ep5.ckpt`。二者是 BEVHeight 检测器权重，**不是 AV 分类器权重**。feature-mask 模型还需相应特征 mask 和显式选项，不能只换文件名。源文件与权重未作算法修改。

运行示例、配置限制、GPU 尚未验证的范围见 [整合说明](../docs/INTEGRATION.md)。旧说明里的本机绝对路径及场景专用参数保留作历史背景，不应直接用于其他站点。