# Trajectory stabilization: collaborator handoff

**Status: pending. No experiment results have been entered.**

This document specifies the experiment protocol. The manuscript is maintained outside this code repository; its matched-interval stabilization subsection remains pending until measurements are available.

## Frozen comparison scope

- Use the current reviewed Todd SE windows described by `code/configs/classifier_default.json` and `code/experiments/field_todd_variable_20261103/windows.csv` in the full local workspace.
- There are 13 windows: eleven 2.5 s, one 2.1 s, one 0.5 s. Do not replace or extend them based on GPS errors or classification outcomes.
- Compare the associated detector/MOT observations before the evaluated stabilization with the final reconstruction. Freeze calibration, checkpoint, identities, timestamps, and review intervals when measuring postprocessing alone.
- If comparing detector adaptation or changed association, declare that as another condition; do not call the entire change a smoothing ablation.
- Reuse the existing per-record GPS delay. Do not optimize new delay, scale, or velocity bias for each condition. Keep ego and leader in their shared geometry.

## Outputs to supply

1. A protocol/config identifying all conditions and input hashes.
2. Per-video metrics: supported coverage, longest uninterrupted span, three-point position residual, speed MAE, and window travel-distance absolute error. Specify units, sample counts, interpolation support and GPS availability.
3. Paired differences on the same supported times. Give each video equal weight. Report coverage separately on the full interval to expose discarded difficult observations.
4. Two or three before/after visual examples with identical frames, time axes and scales. Include GPS speed where available; mark gaps rather than drawing unsupported motion.
5. A short interpretation, including accuracy degradation if any. Reduced jitter alone does not establish improved measurement accuracy.

Use a new `code/experiments/<descriptive_name>/` directory for computation. Transfer selected tables and figures to the separately maintained manuscript only after the experiment and its provenance are reviewed. Existing historical stabilization tables remain historical results, not substitutes for this matched final-window comparison.
