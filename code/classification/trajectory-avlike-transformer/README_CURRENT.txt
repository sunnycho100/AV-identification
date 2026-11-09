当前识别工程入口

优先阅读 training/ 下的源码。src/model.py 和 src/train.py 是最早的
Transformer 工程，不能只阅读它们就代表当前论文的全部实现。

training/cnn/networks.py 和 training/transformer/networks.py 中的
TrajectoryNet 定义当前 CNN/Transformer；run.py 包含训练、验证选模、
温度校准、切分与预测导出。training/rf/native_benchmark.py 包含 RF。
三个 multisegment 目录的 multisegment.py 包含同车分段、融合与联合微调。
这些均从对应完成实验的源码快照导出；算法没有重写。

数据与标签
  输入基础物理量 primitive_windows: [N,40,3]，依次为后车速度、前车速度、
  保险杠之间净间距；单位 m/s、m/s、m，采样间隔 0.1 s。
  data_protocol.py 在窗口内生成完整 11 特征并使用训练统计量归一化。
  后车整个窗口速度 >22.352 m/s。0=Human，1=ACC，AV 是 ACC 标签简称。
  当前 expanded 主线有 Vicolungo/ASta/Casale/CATS/Vanderbilt；排除 NGSIM。
  源数据与 prepared NPZ 不在本包；data_records/ 记录准确来源和 SHA256。

接入现有数据
将原电脑的 trajectory-avlike-transformer/data/prepared_highspeed_20261010/
复制到本工程 data/ 下，或者保存在服务器的其他目录，然后运行：
  python configure_paths.py --data /服务器路径/prepared_highspeed_20261010
这会核对 windows.csv 和 windows_4s.npz 的 SHA256，并仅更新 training/*/
experiment.json 的数据路径与原模型路径，不修改归档权重或切分。
未传输数据前，也能阅读代码、核对文件完整性，并进行合成输入的模型加载检查。

环境
  独立 Python 3.10 环境，PyTorch 2.6.0（按服务器驱动安装相应 CUDA/CPU 版本）。
  再安装 requirements.identification.txt。RF 模型按 scikit-learn 1.7.2 保存，
  应保持该版本，避免 joblib 反序列化的版本差异。
  本工程不需要 MMCV，也不需要安装 BEVHeight；它们属于独立的视频前端。

模型加载检查（不训练，也不需要数据）：
  python smoke_models.py
该检查在 CPU 上加载全部 75 个主线 checkpoint，验证有限输出；不测识别精度。

继续训练
  各 training/<方案>/ 都有独立 experiment.json；数据接好后，在对应目录
  使用固定入口 python run.py。每次输出 artifacts/run，已有输出会拒绝覆盖。
  默认五个 seed、三折、原训练预算全部保留；不会因迁移悄悄减少训练量。
  延续 OpenResearch 时，把相应目录导入为新实验工程，将 python run.py
  配为固定运行命令，在新节点修改/提交配置后通过 orx exp run 启动。
  不要修改已经回答实验的冻结节点；本补包没有在本机启动新训练。
  如运行多段可学习方案，原 CNN 模型和其 feature_stats 已随包保留。
  多段 frozen 方案读取随包提供的原始 val/test 预测，并核对历史 manifest。

已有 CNN 推理接口
  进入 inference_cnn/，使用 predict.py --input new_windows.npz --output predictions.csv。
  具体输入检查、跨折回测和温度校准规则见该目录 README.txt。
  它的说明中的 /tmp/identification-deps 是原电脑路径，新服务器使用自己的环境。
  Transformer 与 RF 的模型定义/加载方式可参考 smoke_models.py 和各训练代码。

论文口径
  1/2/4/8 段结果来自同一匹配队列；八段包含 32 秒观察，通常跨 60 秒。
  五个 seed × 三个来源折是评估配置，不是 15 份独立数据。
  回测某个 Vicolungo 来源时只能使用对应该来源的留出折，不能把其他折
  见过该来源的模型混入后声称测试准确率提升。
  原 manifest 中还记录一些为精简而未携带的历史文件；多段实际读取的预测、
  checkpoint 和统计量都已保留，并保持原 hash。

与视频提取模块衔接
  当前 BEVHeight/MOT 输出 CSV 还不是分类器的直接输入。应完成时间对齐、
  速度单位、前车匹配、中心距转净间距、采样频率和质量筛选的接口适配。
  轨迹层的 CNN 联合微调已实现；视频检测/跟踪/识别的整体反向传播未实现。
  附录 PI 残差辅助试验属于另一研究分支，本包聚焦三种核心分类器和多段主线。
