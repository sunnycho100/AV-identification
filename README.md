# Roadside trajectory reconstruction and AV/HV identification

路侧轨迹重建与 AV/HV 识别。本仓库仅包含代码、配置、测试与使用说明，论文单独维护；当前默认分类器固定源模型权重，不使用 Todd 标签微调。自有新增代码采用 MIT，第三方组件保留各自条款。

This research workspace connects camera calibration, BEVHeight/MOT reconstruction, human trajectory review, and source-only ACC/HV classification. Trajectory reconstruction is the core contribution; classification is a downstream application. A source export includes runnable synthetic review and feature demos, but intentionally omits field recordings and model weights.

## Start here

- [Installation and commands](code/docs/GETTING_STARTED.md)
- [Architecture and maintenance boundaries](code/docs/ARCHITECTURE.md)
- [Prepared-data contract](code/docs/DATA_CONTRACTS.md)
- [All research entry points](code/docs/CODE_INDEX.md)
- [Current results and limitations](code/docs/RESULTS_OVERVIEW.md)
- [Stabilization experiment: collaborator handoff](code/docs/STABILIZATION_HANDOFF.md)
- [Todd 2026 passes: evaluation, 2D-confirmed tracking and round-2 self-training](code/docs/TODD_2026_TRAJECTORY.md)
- [Preparing a GitHub source release](code/docs/RELEASE.md)

## Layout

```text
identification/
  code/
    src/roadside_identification/  maintained reusable Python package
    manage.py             cross-platform command line
    trajectory/           BEVHeight, camera geometry, MOT and postprocessing
    classification/       original classifier training and research protocols
    review_web/           standalone, versioned review product and synthetic demo
    scripts/              recorded experiments and publication-generation code; see CODE_INDEX
    configs/              explicit model and experiment protocols
    tests/                synthetic behavioral and publication-boundary tests
    tools/                code indexing and audited source export
    environments/         tested machine dependency snapshots
    docs/                 setup, architecture, contracts and collaboration
    data/, experiments/, outputs/   local assets and generated results (not Git)
  archive/                immutable original packages and revisions (not Git)
```

Open `identification.code-workspace` in VS Code to work on the code folder. New reusable logic goes in `code/src/`. Historical research scripts retain their recorded paths and may require the original research workspace; see `code/docs/GETTING_STARTED.md`.

## Quick start

The review demo uses only Python 3.10+:

```sh
python code/manage.py review --port 8765
```

Open `http://127.0.0.1:8765/`. For Remote SSH, forward port 8765 through VS Code. To use feature processing and tests, create/activate a Python environment and follow the installation guide:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -e ".[analysis,dev]"
python code/manage.py doctor
python code/manage.py demo
python code/manage.py test
```

With a compatible checkpoint, `python code/manage.py classify --config configs/classifier.local.json` runs frozen inference. Start from `code/configs/classifier.example.json`; the synthetic demo has no labels and is not an accuracy test. From another directory, use `--workspace /path/to/code` before the command.

On the existing Windows research machine, `code/run.ps1 identify` uses the preserved classifier environment and real Todd inputs. BEV/CUDA and the historical classifier retain separate compatible environments; see [environment details](code/docs/ENVIRONMENTS.md). They exchange files, so a single Python environment is not required.

## Research scope and reporting

The current default gives 9/13 correct (69.2%) on Todd SE development videos. The model was chosen after inspecting development results; this is not an independent deployment benchmark. Prior supervised adaptation results and the pending matched-window stabilization protocol remain documented.

The manuscript, bibliography, LaTeX template, compiled paper and publication figures are maintained separately and are excluded from this repository. Source code for regenerating the original classification figures and tables is retained in `code/scripts/publication/`; it requires archived experiment evidence and writes to `code/outputs/publication/` by default. No manuscript is required to use the maintained code or synthetic review demo.

## Publish and contribute

```sh
python code/manage.py inventory
python code/manage.py release
```

The exporter creates a fresh, audited source tree and ZIP under `code/outputs/github_release/`; `latest.json` points to the newest build. It does not upload anything. Raw data, reviews, weights, environments, generated experiments, manuscript files, package installation metadata and the redistribution-restricted AB3DMOT dependency are excluded. See [release instructions](code/docs/RELEASE.md), [contributing](CONTRIBUTING.md), and [third-party notices](THIRD_PARTY.md).
