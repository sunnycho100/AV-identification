# Goal: MonoUNI vs BEVHeight on the Todd clips (2026-10-04)

Advisor request (Hang, 2026-10-04): run MonoUNI and compare it with BEVHeight,
since highway elevation changes may hurt BEVHeight; also try our method on the
data of arXiv 2511.14977. Parent goal and rules:
2026-10-03-pipeline-refinement-goal.md (read its rules section first; all still apply).

## /goal statement

Run MonoUNI zero-shot on the five Todd clips through our existing road mask,
tracker, smoother and GPS benchmark, and deliver a side-by-side comparison with
BEVHeight ft102: one table (GPS benchmark plus GPS-free scorecard) and one
comparison video per clip, with a short verdict on whether MonoUNI should
replace, join or lose to BEVHeight. Then run a 2D-box plus ground-plane baseline
(the SVBRD-LLM style) the same way, and prepare our pipeline on a sample of the
SVBRD-LLM roadside dataset. Done when the table, videos and verdict are in
outputs/reports/monouni_compare/ and logged in this doc, every number reproducible
from a committed script.

## What the two papers say (extracted 2026-10-04)

MonoUNI (NeurIPS 2023, github.com/Traffic-X/MonoUNI)
- CenterNet style, DLA34. Predicts depth with focal length and pitch divided out;
  pitch comes from the ground normal in denorm/<id>.txt (a b c d, unit normal
  pointing up in camera frame, so b < 0; for pitch t down: 0, -cos t, -sin t, h).
  Camera height d is read but never used. Roll is not modeled.
- Input 1080p resized to 960x512 (hardcoded 1080/512). Range to about 184 m.
- Trained focal range 2100 to 2800 px, pitch 5 to 20 deg. Their own test: trained
  on one focal, 5.8 AP on another. **Our focal is about 1490 px (AnyCalib), out of
  range; our pitch is about 19 deg, in range.** This is the main risk.
- Env: py3.8, torch 1.5 (1.8+cu111 works on Ampere), numba 0.53. Always calls
  nccl init, Linux plus NVIDIA only. Config lib/config.yaml, run lib/train_val.py -e.
- No labels: needs empty label files for every id; Rope3D eval crashes after
  predictions are written (output/rope3d_eval/data/<id>.txt, KITTI 16 columns,
  bottom center in camera frame, ry about camera Y).
- Weights: Rope3D only, Baidu Pan (link in the README, code g86j). No mirror
  anywhere (HF, forks, issues checked). Baidu blocks new international accounts.
  Author says zero-shot on new scenes is not guaranteed, recommends fine-tuning.
- Fork github.com/potentialming/MonoUNI-Training adds DAIR-V2X training (no weights).

SVBRD-LLM (arXiv 2511.14977, UT Austin)
- Not Waymo Open data: their own fixed roadside camera at one signalized Austin
  intersection inside Waymo's service area. Goal is AV vs human-driven
  identification from trajectories.
- Pipeline: YOLO 2D boxes, ByteTrack, Kalman on pixel tracks, ground-plane
  homography from hand-picked points for metres. Features speed, accel, jerk,
  headway, lane changes. GPT-5 discovers rules; 90.0 acc, 93.3 F1. Baselines:
  LR 82.3, DT 84.7, RF 88.2, LSTM 91.5 acc. AV vs HV gaps: accel std 0.31 vs
  0.53 m/s^2, jerk std 0.65 vs 1.22, headway 3.1 vs 1.8 s. They admit speed bias
  from the oblique view.
- Data: huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av,
  974 mp4 clips, about 27 GB, CC-BY-4.0, no account. No trajectories, labels,
  calibration or code. AV vs not only by folder (960 AV clips, 14 non-AV).
- For us: a fixed roadside camera, so our whole pipeline applies in principle,
  but we must calibrate it ourselves (no K, PTZ zoom may vary), and the road
  frame must be planar, not lane-aligned (intersection with turns). No GPS there.

## Phases

Each phase ends with a log entry in this doc and one commit. Agents may run
phases in parallel where marked.

### P0. Blockers (user)
- MonoUNI Rope3D checkpoint from Baidu (the user, Hang or Bofeng), copied to the lab
  at ~/roadside-camera/MonoUNI/checkpoints/rope3d.pth. Claude cannot log in to
  Baidu. Until it arrives, run P1, P2, P4 and the P3 fallback.
- `ssh cee` logged in. Check GPU vacancy before every job (nvidia-smi; shared
  with jliu2487, hzhou364); use one free GPU, never kill others' jobs.

### P1. Inputs and adapter (local, no checkpoint needed)
New files only: scripts/object_detection/monouni_prep.py, monouni_to_road.py.
- prep: per clip, writes a MonoUNI root: image_2/<id>.jpg, calib/<id>.txt (P2
  from K), denorm/<id>.txt (normal from metric_extrinsic_h151_dpm031.json
  rotation: up vector of the road in camera frame, then a b c d with b < 0),
  ImageSets/val.txt and train.txt (same ids), empty label dirs.
- **Focal fix:** our f about 1490 px. Variant A as is. Variant B "virtual zoom":
  scale the image by s so s*f is about 2200 (s about 1.48), crop the 1920x1080
  window covering the far road (record the crop offset; K' = s*K shifted by the
  crop). Variant C two crops (near and far) merged by NMS in the road frame. Run
  A and B first; C only if B loses near cars that A finds.
- to_road: reads output/rope3d_eval/data/<id>.txt, maps bottom centers and ry to
  road frame with the extrinsic (undo the crop and scale first), writes
  <frame>_pred.json in the run_bevheight_generic.py schema to
  outputs/object_detection/camera-data/<clip>_monouni_<variant>/ so road mask,
  tracker and smoother run unchanged.
- Check: a selfcheck that a synthetic box placed in the road frame survives
  road -> camera KITTI line -> road within 1 cm and 0.1 deg; and a denorm sign
  check that pitch read back by MonoUNI's own Denorm parser equals ours.

### P2. Lab environment (parallel with P1)
- Clone MonoUNI to ~/roadside-camera/MonoUNI. Try the existing bevheight env
  (torch 1.9) first; else a new conda env py3.8, torch 1.8.0+cu111,
  torchvision 0.9, numba 0.53. Single GPU: patch the nccl init to a
  single-process group (world size 1) in our wrapper, not in their files.
- Wrapper scripts/object_detection/run_monouni.py: writes the config (root_dir,
  resume_model, threshold 0.2), runs the test, tolerates the eval crash after
  predictions exist, fails loudly if no prediction files were written.
- Check: run on 20 frames of AV_T_EW_3 with random weights to prove the plumbing
  (files written, schema valid), then with the real checkpoint.

### P3. Zero-shot run and comparison (needs P0 checkpoint)
- Detect all frames of the 5 Todd clips, variants A and B. Copy detections back,
  run extract_trajectories with det_tag monouni_<variant>, same q_vel 0.1, same
  frame_times.
- Metrics, both detectors on equal terms (the 102.4 m ft102 run and the 140.8 m
  run if present):
  - GPS benchmark (gps_benchmark.py): share of GPS-car frames with a box within
    2 m, split 15-100 m and 100-140 m and beyond 140 m; held-out AV_T_EW_3 speed
    ratio, RMSE, speed MAE, heading, accel vs GPS.
  - scorecard.py on all 5 clips.
  - Detection recall proxy without GPS: tracks per minute and states in tracks of
    at least 1 s.
  - Elevation question (Hang's reason): per-range bias of the GPS-car position
    along the road (does BEVHeight drift with range where MonoUNI does not?).
- Fallback if no checkpoint by the time P1, P2, P4 are done: train MonoUNI on
  DAIR-V2X-I with the potentialming fork on the lab (check the dataset is on the
  server first; BEVHeight used it), focal range of DAIR set into the depth bins.
  Ask the user before starting a multi-day training job.

### P4. Simple baseline: 2D boxes plus ground plane (parallel, local plus lab)
The SVBRD-LLM approach on our clips: a 2D detector (YOLO, whatever is already
installed on the lab or in third_party; trafcam_3d's YOLOv3 path counts), box
bottom center through our road-plane homography from K and the extrinsic,
ByteTrack-like association is replaced by our tracker so only the detector
changes. Same metrics as P3. This answers what 3D detection buys us.

### P5. Video assets
- scripts/reporting/render_compare_detectors.py (extend render_two_runs.py if
  it fits; it already does two raw-box runs side by side): per clip, left
  BEVHeight ft102, right MonoUNI (best variant), tracked and smoothed, id and
  mph labels, GPS car orange, range rings at 50, 100, 140 m.
- Output outputs/reports/monouni_compare/<clip>.mp4 plus a contact sheet jpg at
  5 timestamps. Look at the frames before claiming anything about them.

### P6. SVBRD-LLM data (after P3 or in parallel if lab is busy)
- Download a small sample only: 10 day clips from clear/ (about 0.3 GB) to
  data/svbrd/ (git-ignored). Needs the user's yes before the first download.
- Calibrate one clip GPS-free (VP for pitch, lane width or stop-line dimensions
  for scale), planar road frame, run BEVHeight and MonoUNI if ready, tracker,
  smoother. Report: tracks look sane (speeds 0 to 45 mph in an intersection),
  Waymo cars trackable, our accel std and jerk std for Waymo vs others against
  their 0.31 vs 0.53 and 0.65 vs 1.22.
- Not a full reproduction of their classifier; just whether our trajectories
  separate AV and HV on their data with their 6 features and a random forest.

### P7. Report
- outputs/reports/monouni_compare/summary.md: one table (rows: BEVHeight ft102,
  MonoUNI A, MonoUNI B, 2D plus ground plane; columns: the metrics above), the
  verdict in three sentences, links to the videos. Update the vault note
  "Roadside monocular 3D detector shortlist (2026-10)". Then ping the user.

## Rules for autonomous work
- GPS is evaluation only (exceptions in the parent goal). Never tune MonoUNI
  variants on AV_T_EW_3: pick the variant on HV_T_EW_1 and AV_T_WE_1, report
  AV_T_EW_3 held out.
- HV_T_EW_2 and AV_T_WE_3 stay excluded from GPS validation; scorecard only.
- New files over edits. Do not touch the user's uncommitted
  run_bevheight_generic.py, run_ab3dmot.py, test_run_ab3dmot.py.
- Never commit data/roadside_box/, data/svbrd/, personal-documents/, weights,
  Box links or server GPS paths. Repo is public.
- Every number in the report comes from a committed script with its command in
  the log. Stop and ask before: multi-day training, downloads over 1 GB,
  anything needing a login.

## Log
(empty)

### 2026-10-04 run 1 (autonomous)
- P1 done: monouni_prep.py (A as is, B zoom 1.48, focal 2210-2396 px per clip,
  pitch 18.0-18.9 deg, denorm d 15.10 m), monouni_to_road.py (ry lifted back onto
  the road plane; round-trip check 1 cm, 0.1 deg). Commit 35b239a.
- P2 done: no new env needed. The bevheight env (py3.8, torch 1.9 cu111) runs
  MonoUNI through run_monouni.py (no NCCL, no labels); random-weight run wrote
  300 of 300 frames. Lab root disk is full (1.3 GB free): MonoUNI clone and roots
  live in /home/data/scho242/ (MonoUNI, monouni/<clip>_<A|B>, all 5 clips prepped).
- P3 blocked: no checkpoint. Fallback (train on DAIR-V2X-I) not possible as is:
  the lab holds only a 71-frame DAIR sample; the full set is a large download.
- P4 done: run_2d_ground.py (torchvision Faster R-CNN COCO + ground plane),
  tag det2d, all 5 clips; gps_benchmark.py --tag; compare_detectors.py. Commit 7b969b9.
  Coverage within 2 m: ft102 78/83/1% at 15-60/60-100/100-140 m, det2d 93/95/57%.
  Held-out AV_T_EW_3: ft102 speed ratio 1.00, 0.16 mph MAE, accel rms 0.14 (GPS
  0.12); det2d 0.91, 6.0 mph, 1.37. Scorecard accel_steady 0.25 vs 0.88.
  BEVHeight along error stays within +0.1 to +0.4 m up to 140 m where it detects,
  so no sign of a range-growing (elevation) bias in its matched boxes.
- P5 done: render_compare_detectors.py, videos for all 5 clips (ft102 vs det2d).
  Commit 2fe76ac.
- P6 waiting for the user's yes to download 10 clips (~0.3 GB).
- P7 interim: outputs/reports/monouni_compare/summary.md.
