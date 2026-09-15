# Orientation research loop: design

Date: 2026-09-15. Approved approach: research-driven candidate ladder (A), with the existing-knob sweep (B) as its first batch of candidates.

## Goal

Reduce per-frame box orientation error from BEVHeight on our roadside footage with changes that are generic across clips, found by reading published methods and testing each one against a frozen score. Later, the same score grades fine-tuned checkpoints.

## Score (frozen before any candidate runs)

`scripts/evaluation/score_heading.py`

- Input: a tracks file (`tracks.json`, AB3DMOT output) or a per-frame yaw file produced by a candidate (`{clip: {track_id: {frame: yaw}}}`), plus the tracks file for motion.
- Motion heading per frame: direction of the track's displacement over a centred window of 5 frames (0.17 s). Frames where the track moved under 1.0 m in the window are skipped (stationary or coasted).
- Error per frame: `wrap(yaw - motion)`; reported raw and folded mod 180. Raw catches the front/back flip, folded catches axis error.
- Reported per clip and averaged over clips: median folded error (deg), mean folded error, fraction raw error > 45 deg, fraction raw error > 90 deg (flips), and the same per range bin (0 to 40 m, 40 to 60 m, over 60 m). Coasted frames (repeated score and vx) are excluded.
- Held-out clips: the 8 with frames on disk. AV_T_EW_3, HV_T_EW_1, AV_T_WE_1, AV_T_WE_3, HV_T_EW_2 (also have GPS) and AV_V_WE_3, AV_W_WE_1, AV_W_WE_3 (heading only). The loop never tunes on them; pseudo-labels for fine-tuning come from other clips or from frames outside the graded windows.
- Primary number: mean over the 8 clips of median folded error. Secondary: flip fraction. A candidate is kept if the primary improves and no single clip's primary worsens by more than 1 degree, and the flip fraction does not rise.
- Self-check: a synthetic track with known yaw offsets must reproduce the offsets; a mirrored track must not change the folded error.

The score is written once, self-checked, committed, and not edited while candidates run. Any change to it re-scores every logged candidate.

## Candidate interface

`scripts/orientation/candidates/<name>.py`, each exposing

```
def run(clip: str, det_dir: Path, tracks: dict, cfg: dict) -> dict[track_id, dict[frame, yaw]]
```

A candidate may read per-frame detections (`NNN_pred.json`), tracks, calibration and frames. It may not read GPS. Candidates that change detection itself (checkpoint, roll, height, NMS) rerun `run_bevheight_generic.py` and `run_ab3dmot.py` into `outputs/orientation/<candidate>/<clip>/` and then return the new track yaws. Detection reruns go to the lab server GPU 1 when it is free; post-processing candidates run on the Mac.

`scripts/orientation/run_candidate.py <name> [--clips ...]` runs a candidate on every held-out clip, scores it, appends one row to `outputs/orientation/ledger.csv` (candidate, config hash, per-clip primary, mean primary, flip fraction, kept yes/no, git commit, date), and writes the per-frame yaws next to it so a result can be re-scored without rerunning.

## Batch 1: existing knobs (approach B)

Baseline first: current phase1 outputs scored as they are. Then, one at a time:

1. Box convention check: score the yaw as stored versus yaw + 90 deg. Settles whether the "perpendicular boxes look steadier" observation is a real signal or a rendering convention.
2. Checkpoint 102.4 m versus 140.8 m.
3. Roll: values from `sweep_roll.py` (`ROLLS`).
4. Camera height: 15.5 m (OTC3D) versus 16.26 m.
5. NMS radius (also fixes the documented squared-distance bug, scored separately).
6. Tracker max-age 6, 15, 30 frames.

These need detection reruns for 2 to 5 (server) and only tracker reruns for 6 (Mac).

## Batch 2: research candidates

Deep research runs in parallel with batch 1 (Opus 5 subagents). Questions:

- Temporal consistency of 3D box orientation across frames in monocular and roadside detection (tracking-aware yaw, yaw voting along a track, motion-prior fusion).
- Roadside 3D detection trained on DAIR-V2X or Rope3D and deployed on other camera heights and pitches (domain gap methods: test-time augmentation, camera-parameter conditioning, height and pitch normalisation, pseudo-label self-training).
- Orientation estimation in traffic surveillance from 2D cues (vehicle 2D box plus ground homography, keypoints), as an independent yaw source to fuse with BEVHeight.

Output: PDFs in `docs/papers/`, one note `docs/papers/orientation-research-2026-09.md` with, per paper, the mechanism, what it would take to try here, expected gain, and cost. Ranked list of at most 8 candidates. A vault copy goes to `03 Resources/Articles/`.

Each research candidate becomes one `candidates/<name>.py` and goes through the same ledger. Expected first ones: per-track yaw voting with a motion prior (cheap, generic), test-time horizontal flip averaging, yaw from 2D box bottom edge through the ground plane.

## Batch 3: fine-tuned checkpoints

Once batch 1 and 2 stall, fine-tuned checkpoints from `scripts/finetune/train_finetune.py` on the server (masked loss, other vehicles excluded, per Hang) are scored as candidates named `ckpt_<tag>`. Training runs at night when Hang's jobs are done, under 20 GB in the home directory, with intermediate checkpoints deleted.

## Operating rules

- Generic first: a candidate must improve the mean over all 8 clips, never be tuned per clip.
- GPS is never read by any candidate. Position accuracy against GPS is re-checked with the existing grader only for kept candidates, as a guard that heading gains do not cost position.
- Server: check `nvidia-smi` before every launch; GPU 1 only while Hang's jobs run; nothing over one hour during the day.
- Git: branch `research-loop`. Kept candidates are committed with the ledger row in the commit body. Rejected candidates stay in the ledger and in `candidates/` but are not committed as changes to the pipeline.
- Orchestration by Fable; research and implementation subagents run on Opus 5.

## Not in scope

- Changing the AV-versus-human classifier.
- Fixing the range-dependent depth bias except where a candidate happens to move it (recorded, not optimised).
- The V and W site calibration. Their clips are scored for heading only.
