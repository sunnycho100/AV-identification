# Todd 2026 passes: evaluation, 2D-confirmed tracking and round-2 self-training

Merged from the `research-loop` branch in November 2026. These are research scripts in the historical style: run them from `code/trajectory`, where they resolve `data/`, `outputs/`, `Camera data/`, `checkpoints/` and `third_party/AB3DMOT/` (all local, listed in `.gitignore`). Paths below are relative to `code/trajectory/scripts/`.

| Stage | Entry points |
|---|---|
| Calibration of the re-aimed 2026 pose | `calibration/focal_from_rotation.py` (shared focal length from the rotation homography to the 2025 view), `calibration/roll_fit.py`, `calibration/otc3d_2026_pose_compare.py` (runs from the OpenTrafficCam3D root on the lab server) |
| Instrumented-car identity | `tracking/match_marker_template.py`, `pipeline/mark_instrumented_manual.py` |
| Low-score tracking | `tracking/run_ab3dmot.py` (`--low-thresh`, `--low-gate`, `--confirm-birth`), `pipeline/extract_trajectories.py`; the adopted setting is `--track-low-thresh 0.30 --low-noise 4 --low-filter confirm --confirm-floor 0.30`. Studies: `evaluation/lowscore_sweep.py`, `lowscore_precision.py`, `lowfilter_study.py`, `hybrid_study.py` |
| Round-2 self-training | `finetune/make_pseudo_labels.py` (`--run`, round-2 gates), `finetune/mark_label_roles.py` (label and ignore roles from 2D agreement), `finetune/train_finetune.py` (ignore-zone loss, `--selfcheck-ignore`), `evaluation/round2_study.py`, `evaluation/round2_same_frames.py` |
| Held-out evaluation | `evaluation/new_todd_eval.py`, `heldout_eval.py` (nine held-out passes, intervals from resampling passes), `coverage_window.py` (fixed visibility window), `accel_study.py`, `accel_zero_reference.py`, `bias_check.py`, `range_dropoff.py`, `trajectory_quality.py`, `smoother_retune.py`, `roll_test.py`, `road_mask_2d.py` |
| Review renders | `reporting/render_*.py`, `reporting/match2d.py` |

GPS rule: the per-clip GPS trajectory CSVs are held-out ground truth and never calibration input. Only HV_T_EW_1, AV_T_WE_1 and AV_T_WE_3 may serve as fine-tuning labels for the instrumented car; AV_T_EW_3 stays test only.
