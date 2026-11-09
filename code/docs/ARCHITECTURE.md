# Architecture and maintenance boundaries

```text
video + calibration
  -> trajectory/ : BEVHeight detection, association/MOT, metric reconstruction
  -> review_web/ : versioned human identity, interval and leader review
  -> scripts/   : recorded reconstruction and experimental protocols
  -> prepared trajectory values + masks (NPZ), video metadata (CSV)
  -> src/roadside_identification/ : validated frozen classifier inference
  -> experiment results -> selected paper tables and figures
```

## Maintained library

| File | Responsibility |
|---|---|
| `workspace.py` | Explicit workspace resolution; no host-specific root |
| `io.py` | JSON and streamed SHA256 provenance |
| `features.py` | Window-local SI-unit motion features; leader boundaries and missingness |
| `model.py` | Historical source Transformer parameter layout |
| `classifier.py` | Input validation, fixed normalization, immutable inference and video aggregation |
| `metrics.py` | Classification scores; no training or model selection |
| `cli.py` | User commands and lazy optional-dependency loading |

Importing the package does not load field data, GPU libraries or a model. `FrozenClassifier.predict` accepts no labels. Time-padding and missing interaction features are distinct. No absolute GPS position, clip filename or review state becomes a neural input.

## Research and upstream code

`trajectory/` keeps the working BEVHeight implementation and its established subdirectories: calibration, object detection, tracking, orientation, adaptation, reporting and pipeline. Existing source headers and upstream license terms are retained. The package does not pretend to replace CUDA extension installation or make the legacy detector compatible with Torch 2.6.

`classification/trajectory-avlike-transformer/` preserves original training recipes, source feature protocols and model layouts. This is research provenance, not another installation root for the maintained CLI.

`scripts/` preserves recorded experimental entry paths. Moving these scripts indiscriminately would break imports, relative paths and stored input hashes. Their responsibilities are now indexed in [CODE_INDEX.md](CODE_INDEX.md); the complete publication-source list is [source_index.json](source_index.json). New reusable implementation belongs in `src/`; new experiments should call it rather than duplicating model or feature code. A historical protocol change requires a new experiment/version, not silently regenerated old scores.

`review_web/` remains an independent standard-library product with its own schemas, versioned annotation storage, synthetic demo and tests. Existing URLs and review revisions are unchanged by this reorganization.

## State and dependencies

Private recordings, review decisions, models and computed outputs are local assets, not Python package resources. They are excluded from Git and the source exporter. The classifier environment and BEV environment exchange files across a documented boundary; they do not need to share a Python process.

Tests target behavioral contracts: no masked-value leakage, correct units, identity-boundary handling, padding equivalence, weight immutability, no label influence on predictions, and no private payloads in a source release. GPU and full-data checks remain separate from portable CI.
