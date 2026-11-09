# Prepared trajectory input contract

The maintained classifier consumes an NPZ and a CSV with the same number and ordering of rows/windows. Use `numpy.load(..., allow_pickle=False)`. File formats represent real observation time; padding never adds duration.

| NPZ key | Shape and type | Meaning |
|---|---|---|
| `features` | `[N,T,11]`, numeric | Window-local motion features, before source normalization |
| `feature_mask` | `[N,T,11]`, boolean | True only for available feature measurements |
| `valid_tokens` | `[N,T]`, boolean | True for real ego time samples, false for right-padding |

`5 <= T <= 40`; each window has at least five real samples forming a contiguous prefix. Sampling interval is 0.1 s, so `n` points span `(n-1)/10` seconds. All three ego features must exist at each real time. The first three feature-mask channels equal the time mask. An internal ego time hole must split a window; do not disguise it as padding. Invalid observed values raise an error. Unobserved leader numeric values are ignored even if NaN.

Feature order:

1. ego speed (m/s), acceleration (m/s²), jerk (m/s³);
2. leader speed (m/s), acceleration (m/s²), net bumper gap (m);
3. relative speed `leader - ego` (m/s), closing speed `ego - leader` (m/s), relative acceleration (m/s²);
4. time headway (s), time to collision (s).

Use `motion_features()` on each raw crop to construct these features. Its seven-point quadratic smoothing and derivatives never read omitted observations. Leader derivatives restart when availability or pair identity changes. Features dependent on an unavailable leader remain masked; this is not a measured zero gap or zero leader speed.

The CSV requires `clip`. Optional `label` is 0 for HV or 1 for ACC-active operation; every window of a clip must have one consistent label. Other columns such as `samples` and `duration_s` are retained in the window output. Labels are evaluation metadata, not model input. Same-clip window probabilities are averaged equally, then thresholded once for the video decision.

Outputs are `window_predictions.csv`, `video_predictions.csv`, and `report.json`, including hashes and the exact protocol. The frozen default standardizes with fixed source statistics and omits quantile clipping, matching the reported field reference. `clipped=True` is reserved for explicit archival source comparisons. It is not a target-data fitting step.

Human annotations have a separate contract in [review_web/docs/DATA_FORMAT.md](../review_web/docs/DATA_FORMAT.md). A reviewed 2D visible interval is not proof of valid metric depth; preserve that distinction when preparing classifier inputs.
