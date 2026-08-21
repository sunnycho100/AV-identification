# Getting started

What to download, where each file goes, and which script and document to use
for each task. The repository ships code only: datasets, the pretrained
checkpoint, extracted frames, and run outputs are all gitignored, so a fresh
clone runs nothing until the assets below are in place.

## 1. Assets to obtain

| Asset | Where to get it | Where it goes |
|---|---|---|
| Pretrained checkpoint (917 MB) | [BEVHeight R50 102.4](https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1) (from the upstream model zoo) | `checkpoints/BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt` |
| WI DOT roadside clips + GPS trajectories | Lab shared storage (not public; ask within the group) | `Camera data/<CLIP>.mp4` and `Camera data/<CLIP>_trajectory.csv` |
| DAIR-V2X-I (optional, benchmark only) | [Official site](https://thudair.baai.ac.cn/index), then `docs/prepare_dataset.md` | `data/dair-v2x-i/` |

The GPS trajectory CSVs are held-out evaluation ground truth. They are never
used as input to calibration, detection, or tracking; the only code that reads
them is the grading step.

Environment: `docs/install.md` describes the upstream GPU setup;
`docs/server-finetune-setup.md` describes the CUDA server environment we use
for training. On macOS, inference runs on CPU with a current PyTorch plus
mmcv-full 1.7.x, mmdet 2.28.x, mmdet3d 1.0.0rc6.

## 2. From a clip to results

Extract frames first (frame numbers can be irregular; that is encoder frame
drops and is expected):

```bash
mkdir -p "data/camera-data/<CLIP>/frames" "data/camera-data/<CLIP>/frames_all"
# sparse set used by calibration
ffmpeg -y -i "Camera data/<CLIP>.mp4" -vf "select=not(mod(n\,30))" -vsync 0 \
       -frame_pts 1 "data/camera-data/<CLIP>/frames/%03d.jpg"
# full set used by detection and tracking
ffmpeg -y -i "Camera data/<CLIP>.mp4" -frame_pts 1 \
       "data/camera-data/<CLIP>/frames_all/%03d.jpg"
```

Then run the five pipeline steps in the README section "Pipeline on roadside
camera data": intrinsics, extrinsics, detection, tracking, grading.

## 3. Which file runs what

| Task | Script | Notes |
|---|---|---|
| BEVHeight on our camera frames | `scripts/object_detection/run_bevheight_generic.py` | The entry point for our data. Applies the DAIR ground-plane convention; writes `calibration_used.json` provenance beside the predictions |
| BEVHeight on the DAIR benchmark | `experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py` | Upstream evaluation path, CPU-patched |
| Intrinsics (no ground truth) | `scripts/calibration/run_anycalib_single.py` | One frame in, `[fx, fy, cx, cy]` out |
| Extrinsics for a clip | `scripts/calibration/site_extrinsic.py` | Rotation from the lane vanishing point; height passed in as a site constant; warns if the camera pose moved |
| Site height from lane geometry | `scripts/calibration/solve_height_lane_width.py` | Per-frame solve with spread reporting; used to establish the site constant, not run per clip |
| Tracking | `scripts/tracking/run_ab3dmot.py` | `--fps 30 --max-age 6` for these clips; needs `NUMBA_DISABLE_JIT=1` |
| Identify the instrumented vehicle | `scripts/tracking/identify_target_by_marker.py` | Image-only (hood-marker template votes); GPS is never read here |
| Grade a track against GPS | `scripts/tracking/grade_target_vs_gps.py` | Similarity scale, speed, rigid RMSE, depth error by range |
| Pseudo-labels for fine-tuning | `scripts/finetune/make_pseudo_labels.py` | Then `review_pseudo_labels.py` to inspect, `train_finetune.py` to train, `eval_finetune.py` to evaluate on the held-out clip |
| Shareable visualization package | `scripts/reporting/build_share_package.py` | Per-clip annotated video, contact sheet, stats |

## 4. Which document covers what

| Question | Document |
|---|---|
| Project overview, status, pipeline commands, known issues | `README.md` |
| Environment installation (upstream GPU) | `docs/install.md` |
| DAIR and Rope3D download and conversion | `docs/prepare_dataset.md` |
| Benchmark training and evaluation commands | `docs/run_and_eval.md` |
| Codebase orientation (what each module is) | `docs/summary.md` |
| GPU-server environment for fine-tuning | `docs/server-finetune-setup.md` |
| Original upstream README | `docs/prev-readme.md` |

## 5. Verifying what a detection run actually used

Every `run_bevheight_generic.py` run writes `calibration_used.json` next to its
predictions, recording the intrinsics file, the extrinsics file, the applied
ground-plane shift, and the exact matrices consumed. To confirm a set of
results really ran through our calibration, compare that sidecar against the
named source files; the matrices must match exactly (the extrinsic after adding
the documented 1.73 m shift). This check has been run on all five processed
clips and holds for each.
