# Site calibration loop (Todd Drive): goal, context, ledger

Working file for an iterative loop. Read this first after a context reset; update
the ledger at the bottom after every test.

## Goal

Find the Todd Drive camera calibration (height h, pitch correction dp, optionally
roll) that makes trajectories from BOTH detectors (BEVHeight 140.8 m checkpoint
and trafcam_3d) agree with GPS on the held-out clips, and report the best one.

Success, on held-out clips AV_T_EW_3 (main) and AV_T_WE_3 (secondary, see caveat):
1. Position-derived speed ratio (camera / GPS displacement over the same 0.5 s
   windows) within 1.00 +- 0.02 for both detectors. NOT the Kalman `speed_mps`
   ratio: AB3DMOT's velocity reads 2-7% low (see ledger), so it mixes a tracker
   bias into the calibration test.
2. Along-ray error medians within +-0.5 m in every range band (0-40, 40-60,
   60-80, 80-120 m), no sign pattern with range.
3. Rigid RMSE no worse than the best seen (about 0.6 m).
4. Heading axis error not worse than the current run (BEVHeight ~5 deg,
   trafcam ~2.5 deg); calibration should barely move it.

Fast inner check (no detector run): image contact points on a held-out clip,
back-projected with the candidate calibration, compared with GPS distances
between frames (scripts/calibration/fit_height_from_gps_distances.py,
load_inputs + ground). Confirm a promising candidate with full detector runs.

## Rules

- GPS is evaluation-only by repo rule (BEVHeights-mac/CLAUDE.md). Exception
  agreed by Sunghwan on 2026-09-30: GPS distances may set height/pitch, but only
  from the declared calibration clips HV_T_EW_1 and AV_T_WE_1. Never grade on
  those two. Never fit anything on AV_T_EW_3 or AV_T_WE_3.
- Do not overwrite `_phase1` outputs or `metric_extrinsic_site.json`. Each
  candidate gets its own suffix and extrinsic file.
- Detector runs: `--score-thresh 0.45`; tracker `--low-thresh 0.45 --max-age 6
  --fps 30` (the uncommitted defaults are 0.20 and must not leak in).
- Use bash -c for loops with flag variables (zsh does not word-split).

## Established facts (2026-09-30)

- Five T clips share one camera (CCTV-13-0025, US 12/18 @ Todd Dr, facing SE),
  PTZ zoom differs slightly per clip; all pass the 2 deg same-pose check.
- VP rotation (site_extrinsic.py) is uncertain by ~0.25 deg pitch (lines from
  the two carriageways disagree by ~12 px; ramp_corr 0.87). Roll assumed 0.
- GPS-distance fit on the calibration clips: h 15.06 (EW), 15.05 (WE), joint
  15.10 m with dp -0.31 deg; with VP pitch kept, 15.64 m. OTC3D + lane width:
  15.64 m. In use: 16.26 m (partly GPS-derived, July).
- Geometry at 16.26 m measures every step ~4.5% too long on both calibration clips.
- GPS-video timing: per-clip start offset uncertain by 0.7-2 s (lab's
  video_time_sec is "estimated"). Speed ratio is insensitive to it; chord and
  rigid RMSE are not fully.
- AV_T_WE_3: local mp4 is a shorter cut than the one its GPS was aligned to;
  treat its GPS numbers as unreliable.
- Target IDs: AV_T_EW_3 and HV_T_EW_1 from target_track.json in _phase1 (hood
  marker); AV_T_WE_1 track 28 and AV_T_WE_3 track 17 in _phase1_102 (manual).
  compare_height_runs.py maps them into any run in image pixels.

## Tools

- Extrinsic variant: scripts/calibration/make_extrinsic_variant.py
- BEVHeight: scripts/object_detection/run_bevheight_generic.py (miniforge python)
- trafcam: scripts/object_detection/run_trafcam3d.py --extrinsic <file>
  (third_party/trafcam_3d/.venv/bin/python)
- Tracking: scripts/tracking/run_ab3dmot.py
- Grading: scripts/evaluation/compare_height_runs.py (functions pick_target,
  load_track, to_px) + grade_target_vs_gps.score; heading via score_heading.score_clip
- Ledger: outputs/evaluation/calibration_loop_ledger.csv

## Ledger (AV_T_EW_3, held out)

| Candidate | Detector | Speed ratio | Chord | Rigid RMSE | Along-ray 0-40/40-60/60-80/80-120 | Heading |
|---|---|---|---|---|---|---|
| h16.26 dp0 | BEVHeight | 0.948 | 0.989 | 1.13 | -1.40/-0.38/+0.24/+0.68 | 5.0 |
| h15.6 dp0 | BEVHeight | 0.925 | 0.960 | 0.63 | +0.05/+0.41/+0.20/-0.33 | 4.6 |
| h16.26 dp0 | trafcam | 1.007 | 1.026 | 1.11 | -1.52/-0.23/+0.76/+0.80 | 2.5 |
| h15.6 dp0 | trafcam | 0.958 | 0.978 | 0.61 | -0.07/+0.33/+0.63/-0.62 | 2.7 |
| h15.1 dp-0.31 | BEVHeight | 0.915 (Kalman) / 0.988 (position) | 0.953 | 0.79 | +0.03/+0.15/-0.04/+0.05 | 4.5 |
| h15.1 dp-0.31 | trafcam | 0.973 (Kalman) / 1.000 (position) | 1.000 | 0.51 | -0.28/+0.23/+0.46/-0.34 | 2.7 |

Position-derived speed ratios for the earlier rows: BEVHeight 16.26 1.018,
15.6 0.992; trafcam 16.26 1.038, 15.6 0.992.

## Findings by iteration

1. Height 15.6 (dp 0): fixes the near-range error, scale ~1.00 on geometry.
2. trafcam at 15.6 behaves like BEVHeight: the remaining "too slow" was shared,
   so not learned depth.
3. The "too slow" was AB3DMOT's Kalman velocity (speed_mps), 2-7% low. From
   positions both detectors are within ~1% of GPS at 15.1-15.6 m.
4. Lens distortion: k1 = -0.045 (OpenCV radial) makes the two carriageways'
   lane lines agree on the VP (pooled gap 3.2 px vs ~12 px), GPS-free. With it
   the calibration-clip fit gives h 15.07, dp -0.16; held-out geometry ratio
   0.986, under/over 50 m +0.7%/-4.4% (no better than without). Not adopted;
   real but minor. Detectors assume a pinhole, so adopting it means feeding
   undistorted frames.
5. Held-out geometry (contact points, AV_T_EW_3): h16.26 ratio 1.044 rms 1.36;
   h15.6 1.001 rms 0.56; h15.1 dp-0.31 1.003 rms 0.46. A small near +2-4% /
   far -3% pattern remains in all.

## Current best (2026-09-30)

h 15.1 m with pitch correction -0.31 deg on the VP rotation, roll 0, no
distortion: fitted only on HV_T_EW_1 + AV_T_WE_1, chosen over 15.6/0 on the
held-out clip (best trafcam RMSE 0.51 m, chord 1.000, position speed 1.000;
BEVHeight along-ray within +-0.15 m in every band). Speed features must come
from positions, not AB3DMOT speed_mps.

## Open

- Kalman velocity bias is a warm-up: AB3DMOT starts each track at zero
  velocity. On the AV_T_EW_3 target (h151p) speed_mps / GPS is 0.49 for the
  first 15 frames, 0.87 for 15-30, then 0.99-1.02 after 1 s. Fix: ignore
  speed_mps for the first 30 frames of a track, or use position-derived speed.
- The remaining near/far geometry pattern: candidates are roll, road not
  flat, or the contact point (shadow edge) drifting with range.
- AV_T_WE_3 needs the correct video cut before it can be a second held-out clip.
- Timing offsets per clip still unknown (lab question or landmark check).

## Night run (2026-09-30, unattended)

### Head fine-tune pilot (outputs/finetune/run1/head_ft_ep5.ckpt, 102.4 m base)

Raw detector heading, no motion sign step, held-out AV_T_EW_3 (score 0.45, site extrinsic):

| Run | Front right, away / oncoming | Axis vs motion | Cars per frame | GPS car: axis, backwards |
|---|---|---|---|---|
| pretrained | 95% / 3% | 5.7 | 5.1 | 3.75, 100% |
| fine-tuned | 100% / 98% | 2.7 | 7.4 | 0.94, 2.5% |
| fine-tuned, frame rotated 20 deg | 100% / 98% | 18.8 | 3.9 | |
| pretrained, frame rotated 20 deg | 97% / 1% | 14.9 | 2.1 | |
| fine-tuned, image mirrored | 0% / 83% | | | |
| pretrained, image mirrored | 1% / 75% | | | |
| Verona (never trained on), pretrained | +x 82% / -x 18% (axis 17.8 / 11.4) | | 4.1 | |
| Verona, fine-tuned | +x 100% / -x 55% (axis 5.8 / 10.3) | | 7.9 | |

Reading: the fine-tune fixed the front decision, and it partly carries to a new
site. The axis is still locked to the ego frame (rotation test), same as the
pretrained model. Mirroring breaks both models the same way, so that test
mostly shows an out-of-distribution layout. Verona's extrinsic (v2) is rough
(boxes sit beside cars), so only its heading-sign numbers are usable.
Script: scripts/orientation/camera_aligned_detect.py now takes --ckpt --config
--rot-deg --mirror --extrinsic (selfcheck covers the mirror).

### Track post-process (scripts/tracking/postprocess_tracks.py)

Trims trailing coasted states, RTS-smooths x, y, v; interior coasted states
treated as missing. Held-out GPS car, AV_T_EW_3:

| Run | RMSE | Chord | Speed ratio | First 15 frames speed | Sideways jitter per frame |
|---|---|---|---|---|---|
| h151p raw | 0.79 | 0.953 | 0.915 | 0.49 | 0.039 m |
| h151p post | 0.44 | 0.983 | 0.983 | 0.91 | 0.002 m |
| trafcam_h151p raw | 0.51 | 1.000 | 0.973 | 0.88 | 0.034 m |
| trafcam_h151p post | 0.38 | 0.998 | 0.997 | 0.95 | 0.006 m |

Cuts 20% of BEVHeight states (tails), 6% for trafcam. This closes the Kalman
warm-up open item. Caveat: the smoother assumes near-constant velocity; a real
lane change would be softened (none in these clips to test on).

lowthr (continuation threshold 0.20) on the GPS car: RMSE 1.08 vs 1.13, same
along-ray pattern. Neutral on accuracy; its effect is fewer broken tracks
(139 to 69).

### Corrections after watching the videos

- Fine-tuned z is 0.68 m too low (median -2.35 vs road -1.73; pretrained -1.67).
  x and y are unchanged (median shift 0.01 m), so tracks are unaffected, but the
  drawn footprints sit below the cars. Likely a z convention mismatch in the
  pseudo-labels or train loss (labels are -1.73 in output space). Fix before
  the next fine-tune.
- Some of the extra fine-tuned detections are false: the "US 12/18" overlay
  text, parked cars in the lot, an off-road box at Verona's left edge.
- Verona's doubled cars per frame is partly parked and off-road boxes, and both
  models miss the nearest cars there. Treat Verona as weak evidence.

### Height-fixed fine-tune on the lab GPU and automatic road mask (2026-09-30, late)

- Bug: labels stored the box bottom, the head regresses the centre and
  get_bboxes subtracts h/2. Fix in train_finetune.load_sample (z + h/2); the
  selfcheck now checks it (own detections: bottom 1.580, centre 1.430).
- Trained on cee-r030232 GPU 0 (GPU 1 was in use by another user), env
  ~/BEVverify/.envs/bevheight (py3.8, torch 1.9 cu111, DCN backward passes,
  voxel_pooling_ext copied from ~/BEVverify/BEVHeight). Same recipe as the
  pilot: head only, 5 epochs, ~4 min. outputs/finetune/run2_zfix. Held-out loss
  1.04 -> 0.73 (ep 2) -> 0.77 (ep 5); ep 5 used to match the pilot.
- AV_T_EW_3 (run zfix): z median -1.71 (pilot -2.35, road -1.73). Front right
  99% / 98% (pilot 98 / 97), GPS car axis 1.07 deg, backwards 2.5% (pilot
  0.94, 2.5%). Off-road boxes 206 (pilot 361). Cars per frame 6.8 (pilot 7.4):
  a few real misses visible, e.g. frame 150 far red car.
- Positions here are on the 16.26 m extrinsic (same as the pilot), so RMSE is
  ~1.0 m; rerun on the 15.1 m extrinsic for the calibrated numbers.
- Road mask: scripts/calibration/build_road_mask.py paints where moving tracks
  drove (travel >= 15 m), pooled over the 5 T clips; road_mask.py uses
  road_mask.png when present, else the hand band. Agrees with the hand band on
  all but 1 to 3 boxes per run, never keeps a box the band drops.

### Fine-tune on corrected labels (2026-10-01)

Labels: pseudo_labels_v2 (correct_labels.py: ~1 m line-of-sight shift from
silhouette fit, road mask, carriageway heading for untracked labels, overlap
dedupe). Same recipe as before (head only, 5 epochs, 102.4 m base, site
extrinsic), lab GPU 1. outputs/finetune/run3_v2labels; held-out loss on v2
labels 1.78 -> 0.56.

AV_T_EW_3, pretrained / previous fine-tune (zfix) / new (v2):
- Image alignment, box contact minus car contact: about -0.3 to -0.8 / -0.7 to
  -1.2 / -0.2 to +0.2 m; overlap with car 0.62-0.70 / 0.66-0.71 / 0.69-0.79.
- Front right, away and oncoming: 95/3 / 99/98 / 100/100 %; axis 8.1/4.3 /
  3.3/2.1 / 3.0/1.5 deg. GPS car backwards 100 / 2.5 / 0 %.
- On-road cars per frame: 4.80 / 6.13 / 5.66 (beyond 90 m 0.58 / 0.99 / 0.89).
  The new model misses some large vehicles the previous one found (frame 60
  pickup, frame 150 van) and finds others it missed.
- At 15.1 m with mask and post-process, GPS car rigid RMSE 0.10 / 0.16 / 0.15 m
  (rigid fit hides constant offsets, so the image alignment is the position
  check that sees the ~1 m bias).

### 140.8 m checkpoint fine-tuned on the same labels (2026-10-01)

BEVH_RANGE=140, same recipe and v2 labels, lab GPU 0, outputs/finetune/run4_v2labels_140
(frozen weights verified equal to the 140.8 base). Held-out loss 2.70 -> 0.83
(best 0.80 at epoch 2). AV_T_EW_3, 140.8 pretrained / 102.4 fine-tune (v2) /
140.8 fine-tune (v2140):
- Image alignment, contact offset: -0.25 to -0.75 / -0.23 to +0.15 / -0.40 to
  +0.05 m; overlap 0.66-0.72 / 0.69-0.79 / 0.71-0.79.
- Front right away/oncoming: 82/10 / 100/100 / 100/100 %; axis 10.4/3.5 /
  3.0/1.5 / 2.6/1.7 deg. GPS car backwards 100 / 0 / 0 %.
- GPS car at 15.1 m, masked + post-processed: v2 0.15 m over 82 frames, v2140
  0.18 m over 93 frames.
- On-road cars beyond 110 m per frame: 0.52 / 0.00 / 0.01. The fine-tune lost
  the 140.8 model's far range: only 73 of 3,978 labels lie beyond 102 m
  (labels come from the 102.4 detector), so the head learned the far grid is
  empty. Fix: labels from the 140.8 detector, or no loss beyond ~102 m.

### Trajectory extraction on all five Todd Drive clips (2026-10-01)

scripts/pipeline: extract_trajectories.py (detect, road mask, track, stitch,
smooth, lanes, per-vehicle CSV), mark_instrumented.py, trajectory_features.py,
validate_against_gps.py. Detection on the lab GPU (102.4 m fine-tune, 15.1 m
calibration for every clip). Outputs in outputs/trajectories/<clip>_ft102.

Vehicles per clip: AV_T_EW_3 35, AV_T_WE_1 33, AV_T_WE_3 21 (188-frame cut),
HV_T_EW_1 33, HV_T_EW_2 29; 80 with at least 2 s of motion in features.csv.
Lanes: 4 outgoing (a quiet outer lane), 3 oncoming, the same centres within
~0.5 m on every clip.

Instrumented vehicle (image only): AV_T_EW_3 40 and HV_T_EW_1 25 by hood
marker; AV_T_WE_1 13 and AV_T_WE_3 7 by the hand-picked reference track; all
with no runner-up. HV_T_EW_2: the marker never reaches the 0.87 acceptance
score (peaks 0.82-0.83) and the weak matches land on plates of different cars;
needs a person to point out the car. Update: the only car with the marker is
vehicle 33, which stays 96-140 m away (tracked 7 frames by the 102.4 m model,
151 by the 140.8 m model) and moves ~9 m/s, while the clip's GPS reads 25.5
m/s (rigid RMSE 26 m). Either the GPS is from another, unmarked car or the
GPS-video alignment is off. HV_T_EW_2's GPS is excluded from validation until
the lab confirms; its other vehicles stay usable as traffic.

GPS (instrumented vehicle, final output):
- AV_T_EW_3 held out: RMSE 0.15 m, speed 1.004 (MAE 0.17), heading 0.19 deg,
  accel noise ~0.3 m/s^2 (rms camera 0.35, GPS 0.13).
- AV_T_WE_3 held out, cut mismatch: RMSE 0.44, speed 0.959, accel rms 1.53 vs
  0.12 (speed wobbles +-1 m/s as the car drives away).
- HV_T_EW_1 / AV_T_WE_1 (calibration clips): RMSE 0.35 / 0.10, speed 0.985 /
  1.000.
Bug found and fixed on the way: stitched gaps were smoothed as a single frame,
creating speed bursts at every join (AV_T_WE_3 accel rms 3.58 -> 1.53).
Speed beyond 95 m reads 2-3% low against 40-80 m on every clip.

### Height review: 15.6 m rerun, timing sensitivity, Codex review (2026-10-02)

Five clips rerun at 15.6 m with the VP pitch (tag ft102_h156, lab GPU detection,
same pipeline). Against 15.1 m with dp -0.31:
- Metric scale is the same within ~1%: GPS speed ratio 1.000 vs 1.004 (AV_T_EW_3
  held out), median vehicle speed 26.76 vs 26.98 m/s. The two settings sit on the
  same height-pitch ridge; they differ in how scale changes with range.
- 15.6 m is noisier everywhere: held-out RMSE 0.19 vs 0.15 m, speed MAE 0.40 vs
  0.17 m/s, camera accel rms 0.67 vs 0.35 (GPS 0.13); over all 80 vehicles median
  accel rms 0.75 vs 0.64, speed std 0.49 vs 0.38, lateral std 0.15 vs 0.08 m.
  The camera-only numbers need no GPS timing.
Timing sensitivity of the GPS-distance fit (GPS interpolated at frame/fps + tau):
HV_T_EW_1 (car slowing ~1 m/s in the window) h 14.6-15.1 free pitch, 15.1-15.9 with
dp 0 for tau -2..+1 s; AV_T_WE_1 (steady) 15.0-15.2 and 15.5-15.7. Fit RMS barely
changes with tau, so the offset is not identifiable from this data. 29.97 fps: +0.02 m.
Contact points are automatic (lowest blob pixel); the manual click files are empty.
OTC3D "15.64 m" is the local-plane height 14.52 m times the 12 ft lane-width
correction; the dash correction gives ~11.9 m and the audit rejects a single scale,
so it is not independent support for 15.6 m.
Codex review (gpt-5.6-sol): GPS fit constrains a ridge, its standard errors are
optimistic (correlated pairs); OTC3D is corroboration, not a primary source;
report about +-4% scale uncertainty plus the +-3% range-dependent part.
Decision for now: keep 15.1 m dp -0.31 as the main calibration (same scale, less
range-dependent distortion), report scale uncertainty about +-4%. Decisive check
still open: real lane width or dash spacing from WisDOT plans or aerial imagery.
