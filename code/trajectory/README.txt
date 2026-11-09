轨迹提取代码

主入口：scripts/pipeline/extract_trajectories.py
视频图像 → BEVHeight → 道路过滤 → AB3DMOT → 拼接 → RTS 平滑 → 逐车 CSV。
先阅读包根目录 SERVER_SETUP.txt。

默认模型已改为本包 checkpoints/ft102_run3_v2labels_ep5.ckpt，使用 102.4 m 配置。
默认外参文件名为 metric_extrinsic_site.json；应提供当前摄像机真实标定。
train_finetune.py 新增 --ckpt 参数，原有冻结策略和 batch size 1 保持不变。
算法和权重未改动，全部源文件差异见包根目录 migration_changes.diff。

third_party/AB3DMOT：上游 tracker core 与许可证，不包括其示例数据/评测输出。
third_party/AnyCalib：标定库源码及许可证，不包括预训练权重缓存。
上游原始说明保存在 UPSTREAM_README.md 和 BEVHEIGHT_UPSTREAM_README.md。
它们中的历史机器路径、默认 checkpoint 和性能记录不是本迁移包的测试结论。
requirements.upstream-legacy.txt 是原始旧依赖清单，不能在新服务器上直接照装。
