四秒高速 ACC/HV 分类推理包

模型：开发验证集事先选定的 multiscale CNN + expanded 标签池。
单个模型 22,689 参数；5 个种子 × 3 个外层来源折，共 15 个成员。
NGSIM 未用于任何成员的训练。见上级 refinement_report.md 中完整结果与适用范围。

输入：NPZ 中 primitive_windows，shape [N,40,3]，每 0.1 秒一帧。
三个通道依次是：后车速度 (m/s)、前车速度 (m/s)、保险杠之间净间距 (m)。
整窗后车速度必须 >22.352 m/s，即 50 mph；物理质量检查与训练处理一致。
可以附带 window_index、split_group（原始来源字符串）、dt=0.1。
不能把 5 Hz 的 20 帧直接当成这个模型的 40 帧，也不能把经纬度当作米。

从本目录运行（当前会话依赖在 /tmp/identification-deps）：
  PYTHONPATH=/tmp/identification-deps python3 predict.py --input new_windows.npz --output predictions.csv

独立环境可以安装 requirements.txt；CPU 即可推理，--device cuda 可选。
默认对所有成员的原始概率求平均。--calibration validation 使用各模型的
Vicolungo 验证温度；跨来源上它不保证改善校准，本轮 CATS BCE 反而变差。
输出 acc_probability、predicted_label（0=Human，1=ACC）、model_probability_sd、models。
model_probability_sd 是模型间分歧，不是统计置信区间。

如果回测 Vicolungo，输入必须只含指定的外层测试来源，并指定
  --fold part2
或 --fold part3 / --fold part4_highway，限制到该折五个模型。
严禁用全部 15 个模型回测 Vicolungo：其他折的模型见过对应数据。
提供 split_group 时，程序会主动拒绝与所选模型训练来源重叠的输入；
缺少来源元数据时，程序不能自动识别这种误用。

模型没有经过独立新车辆、同步 CAN/GPS 或真实路侧视频端到端验证。
它识别的是训练标签定义下的纵向 ACC/Human 状态，不识别车型是否配备
自动驾驶功能，也不区分 ACC、FSD、其他横向辅助功能。

bundle.json 包含精确特征统计量、验证温度、训练来源和权重 SHA256。
predict.py 每次加载验证权重摘要。无任何窗口外平滑或整轨迹概率聚合。
没有静默删除不合格输入；遇到超出训练数据质量协议的窗口会报告位置。
