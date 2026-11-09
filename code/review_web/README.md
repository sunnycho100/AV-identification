# Roadside Trajectory Review / 路侧轨迹审阅

Version **3.0.1** · Python **3.10+** · Standard library only · Windows / Linux / macOS

A local research tool for reviewing roadside video before trajectory reconstruction. Mark a visible ego interval, identify the same vehicle across frames, and record unusable or frozen sections. Every submitted review creates a new JSON revision. The original leader relation review remains available as a separate project.

## Start

Within the identification workspace, run `./code/run.ps1 review` from `identification/`. Or run from this folder:

```sh
python server.py --workspace ../ --port 8765
```

The standalone source package works immediately with synthetic demonstration frames:

```sh
python server.py --port 8765
```

Open `http://127.0.0.1:8765/`. For VS Code Remote SSH, forward remote port **8765** in the **Ports** panel, then open the displayed local address. The local port may differ from the remote port. No GPU, Node, build step or Python package installation is needed. The service binds to loopback; use SSH forwarding for remote access.

On Windows, `./START.ps1` selects the existing project Python environment when present, otherwise the `python` on PATH. On Linux/macOS, use `python3` if that is your Python 3.10+ executable.

## 当前五段视频怎么审

首页选择 **Todd SE · 五段轨迹补充审核**。本轮包含 `AV_T_WE_4`、`AV_T_WE_5`、`HV_T_EW_2`、`HV_T_WE_5`、`HV_T_WE_6`，开放整段原始帧，不受旧检测边界限制。

1. 浏览视频或拖动时间轴，选自车清楚、连续可见的至少三秒。在相应帧点击“设为区间起点 / 终点”。原始时间戳有跳跃的位置会显示黄色标记。
2. 在区间的起点、中间、终点分别拖框标记自车；也可点击已有候选框。至少需要两个不同帧的自车框，建议三个或更多。绿色框才是你确认的自车，黄色框仅供选择。
3. 对 `HV_T_EW_2`，跳到冻结开始的第一帧，点击“从当前帧起标为冻结尾段”。系统记录排除区间，并把“保留末帧”设为前一帧。请确保选择的三秒位于保留范围内。这里只记录裁剪范围，原视频不变。
4. 勾选自车身份连续、选择结论、填写必要说明，点击“保存本段审核”。然后切换下一段。草稿暂存在当前浏览器；只有提交后才写入远程磁盘。
5. 五段完成后可“导出已保存审核”。之后修改同一段会生成新版本；左侧修订历史可下载旧版本。

已保存标注默认位于工作区的 `data/annotations/trajectory_reviews/todd_se_extension/`。`ready_for_retracking` 表示人工确认可以尝试重提取，**不代表三维轨迹已重建成功**。

原来的前车关系页仍通过首页和原 URL 访问，继续区分有前车、无前车、前车暂时不可见。原有标注 JSON、浏览器存储键和旧保存目录保持兼容。旧开发包说明保存在 [docs/legacy_v2.txt](docs/legacy_v2.txt)。

遇到范围冲突时，页面会同时显示审核帧号和保留帧号。普通视频默认保留全部；保留范围填错就改正它，保留范围正确则点击“将审核区间改为 … 帧”。调整保留所有自车框，区间外的框会标红；不足三秒仍需重新选择区间。所有数字输入框单位都是帧号。

## Reuse with another dataset

Place a project under `<workspace>/data/review_projects/<project_id>/project.json`, or launch with `--project /path/to/project.json`. See [the data contract](docs/DATA_FORMAT.md), [JSON schemas](schemas/), and [the runnable demo](examples/demo/project.json). The Todd input preparation script is `../scripts/prepare_trajectory_review_project.py`; it creates inputs only and never writes reviewer decisions.

The application is a static HTML/JS interface plus `server.py` and `trajectory_review.py`. Source videos and extracted frames stay in the dataset workspace. Local saves are versioned; importing JSON validates it and creates browser drafts for inspection before submission.

## Verification and release

```sh
python -S -m unittest discover -s tests -v
python -S tools/check_browser.py --chrome /path/to/chrome --output /path/to/qa-output
python -S tools/package_release.py --output /path/to/releases
```

The unit suite uses temporary synthetic data. The optional browser suite exercises real canvas interactions, save/reload, import/export, trim annotations and revision history. Pass `--workspace /path/to/code` to also verify field project frames; its saves still go to a temporary QA directory.

The release archive contains source, documentation, schemas and synthetic demo assets. It excludes field videos, field annotations, model weights, environments, QA output and browser profiles. See [release notes](CHANGELOG.md) and [paper method note](docs/PAPER_METHOD_NOTE.md). A repository URL, author information and an explicit software license can be added when preparing the public paper release; none is invented here.
