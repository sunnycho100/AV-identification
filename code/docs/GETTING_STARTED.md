# Getting started

Run commands from the repository root, `identification/`. Python 3.10 is the tested version. The review product needs only the Python standard library. Prepared trajectory inference uses CPU PyTorch 2.6; BEVHeight requires its separate legacy CUDA stack.

## Review product: no installation or private data needed

```sh
python code/manage.py review --port 8765
```

Open `http://127.0.0.1:8765/`. A clean source export contains a synthetic review project. In VS Code Remote SSH, forward port 8765 in the Ports panel and open the forwarded local address. Existing field reviews and their revision history remain available only in the full research workspace. See [review documentation](../review_web/README.md).

## CPU feature processing and classification

Create an environment with `python -m venv .venv`; activate it using the normal command for your platform. Then:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -e ".[analysis,dev]"
python code/manage.py doctor
python code/manage.py demo
python code/manage.py test
```

The demo generates three synthetic NPZ/CSV windows with complete, missing, and partially observed leaders. It deliberately contains no AV/HV labels and no trained checkpoint. It is an interface demo, not a recognition benchmark.

For inference, obtain the project checkpoint and accompanying `feature_stats.json` through the authors' chosen distribution channel. No weight download URL is invented in this release. Copy `code/configs/classifier.example.json` to `code/configs/classifier.local.json`, set the two model paths and checksum, then run:

```sh
python code/manage.py classify --config configs/classifier.local.json
```

Paths inside configs are relative to `code/`, unless absolute. No model weight or statistical parameter is fitted by this command. Labels in an optional CSV column are read only to calculate metrics after prediction. Without labels the command writes probabilities only. The public data contract is in [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

## Existing research machine

Keep the two validated research environments:

```powershell
.\code\run.ps1 identify
.\code\run.ps1 python -Environment classifier manage.py doctor
.\code\run.ps1 python -Environment classifier manage.py test
```

The old `scripts/frozen_classifier.py` command forwards to the maintained package, preserving historical imports and default predictions. Running `identify` needs the local checkpoint and prepared Todd inputs. A fresh GitHub checkout intentionally lacks both.

## GPU reconstruction

The detector stack remains separate: Python 3.10, Torch 1.13/CUDA 11.7, matching MMCV/MMDetection/MMDetection3D, compiled BEV operators, AnyCalib, and the separately obtained AB3DMOT dependency. Install these according to the upstream projects and the tested environment inventory; do not install the CPU classifier extra over the detector environment. [ENVIRONMENTS.md](ENVIRONMENTS.md) distinguishes a tested machine snapshot from a portable installation recipe.

The detector entry is `code/trajectory/scripts/pipeline/extract_trajectories.py`. It requires decoded frames, camera calibration and checkpoints. [ARCHITECTURE.md](ARCHITECTURE.md) identifies each stage. `code/scripts/` contains fixed research protocols; these often also need historical configs, reviews and results, so they are not advertised as clean-checkout one-command demos.

## Optional figure and table generation

The repository does not include the manuscript, its template, bibliography or generated figures. The three original Python helpers are preserved under `code/scripts/publication/`.

With the analysis dependencies and the archived experiment evidence available:

```sh
python code/scripts/publication/build_displays.py --evidence-root /path/to/evidence
python code/scripts/publication/build_table.py --evidence-root /path/to/evidence
```

The first command reads `multisegment/` and `highspeed/`; the second reads `residual/all_metrics.csv`. Both write generated files and provenance to `code/outputs/publication/` by default. Use `--output` to choose another destination. These commands regenerate recorded displays; they do not train models or fill pending experimental results.

Other historical paper-editing/build scripts retain their original research-workspace paths for provenance. They are not clean-checkout commands and are not compatible with the separate single-file manuscript without adaptation. The VS Code workspace no longer exposes a paper folder or build-paper task. The stabilization experiment protocol is documented in [STABILIZATION_HANDOFF.md](STABILIZATION_HANDOFF.md).
