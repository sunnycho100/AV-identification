"""Audit paired evaluation populations and summarize the diagnostic suite."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts/experiments/plotting_dependencies"))
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    provenance = {"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "sklearn": sklearn.__version__,
                  "torch": str(torch.__version__), "files": {}}
    source_files = [Path("src") / (name+".py") for name in
                    ("features", "dataset", "model", "train", "utils", "cross_validate",
                     "experiment_analysis", "summarize_experiments")]
    for path in source_files + [Path("data/processed/trajectory_features.csv"),
                                Path("data/processed/trajectory_features_improved.csv")]:
        provenance["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "source_snapshot").mkdir(exist_ok=True)
    for path in source_files:
        shutil.copy2(path, root / "source_snapshot" / path.name)
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    table = pd.read_csv(root / "comparison.csv")
    metrics = ["window_metrics_auroc", "trajectory_metrics_auroc",
               "trajectory_metrics_balanced_accuracy", "trajectory_metrics_f1"]
    per_seed = table.groupby(["variant", "seed"])[metrics].mean()
    per_seed.to_csv(root / "seed_means.csv")
    overview = per_seed.groupby("variant").agg(["mean", "std", "count"])
    overview.to_csv(root / "overview.csv")
    paired = per_seed.subtract(per_seed.loc["baseline"], level="seed")
    paired.to_csv(root / "paired_delta_vs_baseline.csv")
    # Check identical inner splits and exact test window populations across variants.
    checked = []
    for row in table.itertuples(index=False):
        folder = root / row.variant / f"seed_{row.seed}" / "logs" / row.fold
        base = root / "baseline" / f"seed_{row.seed}" / "logs" / row.fold
        a = json.loads((folder / "splits.json").read_text())
        b = json.loads((base / "splits.json").read_text())
        assert a == b, (row.variant, row.seed, row.fold, "split mismatch")
        pred = pd.read_csv(folder / "test_predictions.csv")
        reference = pd.read_csv(base / "test_predictions.csv")
        # Logistic labels are serialized as 0.0/1.0, Transformer labels as 0/1.
        for column in ("Trajectory_ID", "label"):
            assert np.array_equal(pred[column].to_numpy(), reference[column].to_numpy()), (
                row.variant, row.seed, row.fold, column, "test population mismatch")
        assert np.isfinite(pred.prob_AV).all()
        assert pred.prob_AV.between(0, 1).all()
        checked.append(dict(variant=row.variant, seed=row.seed, fold=row.fold, windows=len(pred)))
    (root / "audit.json").write_text(json.dumps({"passed": True, "checks": checked}, indent=2))

    # Historical seed-42 baseline reproducibility (same test population).
    historical = []
    for fold in ("part2", "part3", "highway"):
        old = pd.read_csv(Path("artifacts/logs/baseline_cv") / fold / "test_predictions.csv")
        new = pd.read_csv(root / "baseline/seed_42/logs" / fold / "test_predictions.csv")
        assert old[["Trajectory_ID", "label"]].equals(new[["Trajectory_ID", "label"]])
        historical.append({"fold": fold, "max_probability_difference": float(np.max(np.abs(old.prob_AV-new.prob_AV)))})
    (root / "historical_reproduction.json").write_text(json.dumps(historical, indent=2))

    # Loss and validation AUROC are shown separately: increasing loss can reflect
    # worse confidence even while the ordering measured by AUROC improves.
    loss_rows = []
    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        f"{fold}: {metric}" for fold in ("part2", "part3", "highway")
        for metric in ("loss", "validation window AUROC")])
    for i, fold in enumerate(("part2", "part3", "highway")):
        for variant, color in (("baseline", "#2563eb"), ("small_bundle", "#ea580c")):
            folder = root / variant / "seed_42/logs" / fold
            history = [json.loads(line) for line in (folder / "train_log.jsonl").read_text().splitlines()]
            result = json.loads((folder / "metrics.json").read_text())
            epochs = [r["epoch"] for r in history]
            for split, dash in (("train", "dash"), ("val", "solid")):
                fig.add_trace(go.Scatter(x=epochs, y=[r[split]["loss"] for r in history],
                    name=variant+" "+split, legendgroup=variant+split, showlegend=i==0,
                    line=dict(color=color, dash=dash)), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=[r["val"]["auroc"] for r in history],
                name=variant+" AUROC", legendgroup=variant+"auc", showlegend=i==0,
                line=dict(color=color)), row=i+1, col=2)
            loss_rows.append(dict(variant=variant, fold=fold, best_epoch=result["best_epoch"],
                                  min_val_loss=min(r["val"]["loss"] for r in history),
                                  last_val_loss=history[-1]["val"]["loss"],
                                  last_train_loss=history[-1]["train"]["loss"]))
    fig.update_layout(height=1000, width=1200, title="Seed 42 learning curves", template="plotly_white")
    fig.update_xaxes(title_text="Epoch")
    fig.write_html(root / "learning_curves.html", include_plotlyjs=True)
    pd.DataFrame(loss_rows).to_csv(root / "loss_diagnostics.csv", index=False)

    main_variants = ["baseline", "small_bundle", "features_bundle", "improved_bundle", "logistic_all", "logistic_leader"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=metrics[:2])
    for column, metric in enumerate(metrics[:2], 1):
        for i, name in enumerate(main_variants):
            values = per_seed.loc[name][metric].values
            fig.add_trace(go.Scatter(x=[name]*len(values), y=values, mode="markers",
                text=[f"seed={s}" for s in per_seed.loc[name].index], name=name,
                showlegend=False, marker=dict(color="#2563eb", size=9)), row=1, col=column)
            fig.add_trace(go.Scatter(x=[name], y=[values.mean()], mode="markers",
                name=name+" mean", showlegend=False,
                marker=dict(color="black", size=13, symbol="line-ew")), row=1, col=column)
    fig.update_layout(height=600, width=1250, template="plotly_white",
        title="Each dot: one seed's mean over 3 source-held-out folds; black: mean across seeds")
    fig.update_yaxes(range=[0, 1.01])
    fig.update_xaxes(tickangle=35)
    fig.write_html(root / "comparison.html", include_plotlyjs=True)
    names = {
        "baseline": "原始 Transformer", "small_bundle": "小模型＋正则化＋训练 stride 20",
        "features_bundle": "只改特征处理", "improved_bundle": "小模型训练方案＋特征处理",
        "logistic_all": "全特征统计量＋逻辑回归", "logistic_leader": "前车相关统计量＋逻辑回归",
        "size_only": "只缩小模型", "regularization_only": "只增强 dropout 和 weight decay",
        "stride_only": "只把训练 stride 改为 20", "leader_only": "只用前车相关特征的 Transformer",
        "no_jerk": "移除 jerk", "no_lateral": "移除 lateral_offset",
    }
    lines = ["# AVIdentification 实验分析（2026-09-20）", "",
             "## 实验范围与可复现性", "",
             "- 四组主方案以及两个逻辑回归基线：seeds 42、52、62，每个 seed 做 source-held-out 三折。",
             "- 六组诊断消融：仅 seed 42，每组做三折，不能据此声称跨 seed 稳定。",
             f"- 共 {len(table)} 个训练／评估折，其中 Transformer 54 折、逻辑回归 18 折。",
             "- 每个 seed 同时改变初始化与内部 train/validation 轨迹划分；同一 seed 下各方案划分一致。",
             "- 全部验证和测试窗口固定 40 帧、stride 10；小模型组合和 stride 消融仅改变训练 stride。",
             "- 保持原有窗口级验证 AUROC 选 checkpoint、patience 10、最多 50 epochs、分类阈值 0.5。",
             "- 逻辑回归使用每个窗口的 mean/std/q10/q50/q90；StandardScaler 仅拟合训练集；C 从 0.01/0.1/1/10 按验证窗口 AUROC 选择。",
             "- 原始与改进特征 CSV 的轨迹、时间和标签逐行相同，均为 97,499 行、82 条轨迹。",
             "- 已核验每个对应实验的 split 和测试窗口身份／标签完全一致，预测均为有限且位于 [0,1] 的值。",
             "- 现有 artifacts/logs 与 artifacts/checkpoints 中的历史结果没有被覆盖。",
             "", "## 多种子主实验", "",
             "下表先对每个 seed 的三个来源折取等权均值，再报告三个 seed 的均值 ± 样本标准差。",
             "此标准差反映初始化及内部划分变动，不是新日期／新车辆的泛化置信区间。", "",
             "| 方案 | 窗口 AUROC | 轨迹 AUROC | 轨迹 Balanced Accuracy | 轨迹 F1 |",
             "|---|---:|---:|---:|---:|"]
    for name in main_variants:
        values = per_seed.loc[name]
        cells = [f"{values[m].mean():.4f} ± {values[m].std():.4f}" for m in metrics]
        lines.append("| " + names[name] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 单种子诊断（seed 42；三折均值）", "",
              "| 方案 | 窗口 AUROC | 轨迹 AUROC | 轨迹 Balanced Accuracy |",
              "|---|---:|---:|---:|"]
    for name in ["baseline", "size_only", "regularization_only", "stride_only", "leader_only", "no_jerk", "no_lateral"]:
        values = per_seed.loc[(name, 42)]
        lines.append("| " + names[name] + " | " + " | ".join(f"{values[m]:.4f}" for m in metrics[:3]) + " |")
    lines += ["", "## 解读边界", "",
              "- 这些外层折此前已经被查看，本轮也是探索性比较，不能把选出的最高分作为新的无偏测试结果。",
              "- AV/HV 与两个采集日期仍完全绑定，source-held-out 不能消除日期混淆。",
              "- 前车相关探针使用 v_LV_long 和 a_LV；其中速度投影依赖后车朝向，并非严格独立于后车信息的原始前车信号。",
              "- 简单统计量逻辑回归失败不能证明模型一定学到了时序规律，也可能存在非线性统计关系。",
              "- 删除特征同时改变输入投影层形状及其初始化；单 seed 差异只能用于筛选后续假设。",
              "- stride 20 同时减少训练窗口和每个 epoch 的更新步数，不能将差异完全归因为重复度变化。",
              "- 两种训练方案都使用窗口级训练和选模，尚未测试轨迹等权损失、轨迹级选模、多实例学习或固定观察时长。",
              "- 仅查看概率阈值 0.5；当前未做校准或验证集阈值选择。",
              "", "## 结果文件与复现", "",
              f"结果目录：`{root.as_posix()}`", "",
              "- `comparison.csv`：全部折的指标。",
              "- `seed_means.csv` / `overview.csv`：种子均值及汇总。",
              "- `paired_delta_vs_baseline.csv`：同种子配对比较。",
              "- `audit.json`：划分与测试窗口一致性检查。",
              "- `historical_reproduction.json`：新旧 seed 42 baseline 预测差异。",
              "- `protocol.json` / `provenance.json` / `source_snapshot/`：配置、软件版本、代码与数据 SHA256、代码快照。",
              "- `learning_curves.html` / `comparison.html`：可离线打开的交互式训练曲线与种子比较图。",
              "- 每个方案／种子下保存实际配置、划分、预测、训练日志和 checkpoint。",
              "", "```powershell", "conda activate trajectory-avlike",
              "python src/experiment_analysis.py --output artifacts/experiments/NEW_UNIQUE_RUN --seeds 42 52 62",
              "python -m pip install --target artifacts/experiments/plotting_dependencies plotly==7.1.0",
              "python src/summarize_experiments.py artifacts/experiments/NEW_UNIQUE_RUN", "```", "",
              "运行器拒绝覆盖已存在的输出目录。matplotlib 的渲染 DLL 被 Windows 策略阻止，因此图表改用 Plotly 生成，依赖只安装在 artifacts/experiments/plotting_dependencies，不修改原 Conda 环境包。", ""]
    (root / "results_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(overview.to_string())
    print("\nPaired mean changes:\n", paired.groupby("variant").mean().to_string())
    print("\nHistorical reproduction:", historical)
    print("\nAudit passed for", len(checked), "fold runs")


if __name__ == "__main__":
    main()
