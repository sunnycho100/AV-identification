# Current research status

The current working manuscript centers on roadside trajectory reconstruction. AV/HV classification is a downstream application; the available labels mean ACC-active versus human driving, not all SAE automation levels.

| Development result | Scope |
|---|---|
| Frozen source Transformer: 9/13 (69.2%) | Todd SE, one reviewed window per video; no Todd classifier weight fitting |
| Excluding the 0.5-s HV_T_EW_2 window: 8/12 (66.7%) | Descriptive subset; the default retains all thirteen |
| Updated 64-dimensional model after Todd adaptation: 6/13 | Separate supervised development comparison, retained in history |
| Speed MAE 0.794 m/s | Ten GPS-comparable final-window recordings; previously fixed delay |
| Travel-distance absolute error 1.852 m | Eight final windows with complete GPS support |
| Frozen model duration check: 70.51%, 70.91%, 72.16% | 2.0, 2.5, 3.9 s on the same 886 public held-out windows; pair/source weighting |
| Matched Todd duration check: 8/11 at both 2.0 and 2.5 s | Same eleven eligible recordings, centered crops |

The current default was selected after inspecting development results. These thirteen recordings do not provide independent confirmation of that selection. Upstream camera/detector adaptation and human review use target-scene information; only classifier fitting is source-only. The public and field cohorts differ and their percentages are not directly comparable.

In the full local workspace, detailed reports are under `code/experiments/frozen_transfer_20261103/` and `code/experiments/field_todd_variable_20261103/`. Those generated/private-input artifacts are not bundled in the public source ZIP. The source code and exact protocols are included, and required asset locations are documented.

The additional matched-interval stabilization experiment is **pending for a collaborator**. Its table cells remain empty in the manuscript. Follow [STABILIZATION_HANDOFF.md](STABILIZATION_HANDOFF.md); do not substitute earlier three-view results or fill missing cells with zero.
