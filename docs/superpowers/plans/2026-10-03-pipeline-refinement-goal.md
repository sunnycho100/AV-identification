# Goal: trajectories we can defend, from every clip we have (2026-10-03)

## Goal statement

Turn every roadside clip we hold into per-vehicle trajectories whose speed,
acceleration, lane and headway are accurate enough to compare automated and
human drivers, and prove that accuracy against GPS on clips that never touched
the calibration. Work on the tracker, the smoother, frame timing, data
generation and the GPS benchmark. The detector stays BEVHeight (102.4 m
fine-tune) until the model search below says otherwise.

Done when:
1. Every clip runs on its real frame timestamps; frozen and dropped frames are
   missing, not copies (no track ever "stops" because the video froze).
2. A GPS benchmark exists that is computed the same way for every GPS clip,
   separates what timing can and cannot affect, and is never used to tune.
3. At least 4 held-out GPS clips (today: 1) across Todd, Verona and the W site,
   with sites calibrated GPS-free (vanishing point + dash ruler).
4. On held-out clips: speed within 1.5% (now 0.4% on one clip), position RMSE
   under 0.3 m, acceleration noise under 0.2 m/s^2 (now ~0.3), with the
   GPS-free scorecard no worse on every clip.
5. Each change is kept only if the scorecard says so; every iteration logged
   below with its numbers.

## What we have (inventory)

| Data | Frames | GPS | Status |
|---|---|---|---|
| Todd 10 s clips: AV_T_EW_3, AV_T_WE_1, HV_T_EW_1 | ~300 each | yes | clean video |
| AV_T_WE_3 | 180 (6 s, 7 drops) | yes | cut shorter than the GPS alignment |
| HV_T_EW_2 | 211 (3.0 s hole at 5.97-8.98 s) | yes | instrumented pass likely inside the hole |
| Verona: AV_V_EW_3 (29.17 fps), AV_V_WE_3 | ~295 | yes | needs calibration |
| W site: AV_W_EW_3, AV_W_WE_1, AV_W_WE_3, HV_W_WE_2 | ~297 | yes | needs calibration; 1-2 drops each |
| HV_V_EW_2, HV_W_EW_2 | none | yes | GPS without video |
| Box raw recordings (data/roadside_box): Todd 10 min (Sep 1 2026, 17,842 frames), Verona 10 + 5 min, Seminole, E Washington x2, two corrupted 5 min | ~150k | no | new: label generation, tracker statistics |
| OpenACC Vicolungo AV/HV car following | n/a | yes | reference behaviour for the AV vs HV comparison |

GPS clips are from Oct and Dec 2025; the raw recordings are from Jul and Sep
2026, so they are not the uncut source of the GPS clips. The lab logs each
recording's start unix time (recording_log.csv); the original Oct and Dec 2025
recordings plus their log would give exact GPS-video timing.

The lab's own topic (Idea and task.docx) is trajectory reconstruction from
corrupted video. HV_T_EW_2 is a real example of it; frame-health handling here
is the first piece of that work.

## Rules (unchanged)

- GPS is evaluation only. Calibration exception stays HV_T_EW_1 + AV_T_WE_1.
  New sites are calibrated without GPS. Never pick a car or tune a parameter
  with held-out GPS.
- Never push personal-documents/ or data/roadside_box/; the repo is public, so
  no Box links or server paths to GPS in committed files.
- Commit per working change, push research-loop, fast-forward main.

## Scorecard (run after every change)

GPS-free, all clips:
- fragmentation: share of states in tracks of at least 1 s, joins, median track length
- steady-car acceleration rms (cars without a lane change), lateral std
- leader-follower speed agreement (spacing under 40 m)
- dash-ruler scale per carriageway
- frames with video freeze or drop inside a track (must be 0 after iteration 1)

GPS, held-out clips only (benchmark below): speed ratio and MAE, acceleration
difference, heading, lateral offset, along-road error after timing alignment.

## GPS benchmark (subagent A)

Map each GPS clip into the image: GPS position -> camera road frame -> pixels,
for the instrumented car on every frame. Use it to (a) confirm which tracked
car is the GPS car, by eye, (b) measure what the unknown GPS-video timing can
and cannot affect (a time offset and an along-road shift are the same thing at
steady speed), (c) produce one benchmark file per clip with timing-free metrics
separated from timing-dependent ones.

## Iterations (in order)

1. Frame timing: real timestamps from the video packets; frozen or duplicated
   frames marked missing; tracker, smoother and features use seconds, not
   frame/30. Check: HV_T_EW_2 tracks break at the hole instead of freezing;
   AV_V_EW_3 speeds change by ~2.8%.
2. Smoother: compare the constant-velocity RTS with a constant-acceleration
   model and with tuned process noise; pick by steady-car acceleration and
   leader-follower agreement, confirm on held-out GPS acceleration.
3. Tracker: two-stage association vs current (audit item 4); recover the ~8% of
   detections lost to association; appearance check for stitching.
4. New sites: Verona and W calibration GPS-free (VP + dash ruler), then run the
   pipeline; their GPS clips become held-out tests.
5. Data generation: pseudo-labels from the 10-minute Todd recording (same pose
   check first), corrected with correct_labels.py; fine-tune v3 on many more
   cars and traffic states; compare with the 102.4 m fine-tune on all clips.
6. Corruption: simulate drops, freezes and block corruption on clean clips
   (lab task doc) and measure how much the pipeline degrades; this reuses
   iteration 1.

## Model search (subagent B, read-only)

Literature and code survey for a detector to replace or back up BEVHeight:
roadside monocular 3D, long range, robust to camera height and pitch error,
open code and weights. Builds on the vault note "Roadside 3D pipeline - papers
to borrow from (2026-10)". Output: a ranked shortlist, not a switch.

## Log

(each iteration: what changed, scorecard before and after, kept or reverted)

### Iteration 1: real frame times, frozen copies dropped (2026-10-03) - kept

frame_times.py matches every extracted frame to its decoded source frame and
takes that frame's timestamp; a frame repeating the previous source frame is a
copy. The tracker sees no detections on copies, the RTS smoother steps by real
time and drops copy states, features differentiate by real time.
- AV_T_WE_3 frames were cut on a ~31 fps grid (stream 31/1): frame/30 stretched
  6.00 s to 6.23 s. GPS speed ratio 0.959 -> 0.992, speed MAE 1.18 -> 0.49 m/s.
  The lab's GPS frame index for this clip also uses 31 fps; its video ran to
  frame 280 (9 s), ours ends at 187, so the cut start is still unknown.
- HV_T_EW_2: 89 copies (3.0 s hole); no track runs through it now. The lab's
  frame index for this clip runs to 299 on a 29.97 grid, so their copy of the
  video had no hole: ours (and the Box copy, identical) is corrupted.
- Clean clips unchanged (their frames are on the 1/30 s grid).
Scorecard after (baseline for iteration 2): frag_1s 0.941, accel_steady 0.703
m/s^2, lat_std 0.085 m, follow 1.72 mph, frozen 0 (was 36 on AV_T_WE_3).
Open: AV_T_WE_3 GPS car speed swings 27.3-29.4 m/s with range while GPS is
steady; points at this clip's pitch (VP from 6 frames). A per-clip pitch from the
dash ruler (spacing vs range) would test that without GPS.

### Iteration 2: smoother process noise (2026-10-03) - kept, q_vel 0.5 -> 0.1

Swept the RTS velocity random-walk strength q_vel (m^2/s^3), same detections
and tracks (tags sm_q*). GPS-free scorecard (mean of 5 clips):

| q_vel | accel_steady | lat_std | follow mph | resid m | resid ac1 |
|---|---|---|---|---|---|
| 0.5 (old) | 0.703 | 0.085 | 1.72 | 0.333 | 0.50 |
| 0.1 | 0.245 | 0.074 | 1.56 | 0.361 | 0.56 |
| 0.03 | 0.088 | 0.069 | 1.47 | 0.377 | 0.57 |
| 0.01 | 0.032 | 0.068 | 1.46 | 0.385 | 0.58 |

GPS-free metrics alone keep improving as q falls, and the residual guard is
weak (residuals are already autocorrelated at 0.5: detection error drifts slowly),
so they cannot say where smoothing starts to erase real acceleration. GPS check
(accel rms camera / GPS, and rms of their difference):
- AV_T_EW_3 (held out, steady): diff 0.30 / 0.12 / 0.10 / 0.12 for q 0.5 / 0.1 / 0.03 / 0.01.
- HV_T_EW_1 (decelerating, GPS accel 0.51): camera 1.00 / 0.59 / 0.29 / 0.12,
  diff 0.59 / 0.28 / 0.30 / 0.41. Below 0.1 the smoother flattens a real slow-down.
Choice: 0.1, the strongest smoothing that keeps HV_T_EW_1's deceleration.
HV_T_EW_1 is declared the smoother-tuning clip (it already set the height);
AV_T_EW_3 stays held out: accel diff 0.30 -> 0.12 m/s^2, speed MAE 0.17 -> 0.13,
RMSE 0.15 -> 0.14, speed ratio 1.005. Acceleration is now usable on cruising
cars; jerk still is not.
Scorecard after: frag_1s 0.941, accel_steady 0.245, lat_std 0.074, follow 1.56 mph.
