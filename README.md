# AV Identification from Roadside 3D Detection

Detecting whether a vehicle is driven autonomously or by a human, using only
video from fixed roadside infrastructure cameras. The system extracts 3D
vehicle trajectories from monocular roadside footage and will classify each
trajectory from its motion characteristics. The closest prior work (Maresca et
al., arXiv:2403.09571) addresses this task in simulation only; this project
targets the real-world setting.

This repository extends **BEVHeight** (Yang et al., CVPR 2023). The upstream
README is preserved at [`docs/prev-readme.md`](docs/prev-readme.md).

---

## Architecture

![AV identification architecture](docs/assets/AV-identification-plan.png)

```
roadside video
  -> camera calibration            (intrinsics + extrinsics, no ground truth required)
  -> per-frame 3D bounding boxes   (BEVHeight, CPU-ported)
  -> identity-linked trajectories  (AB3DMOT multi-object tracking)
  -> AV vs human classification    (motion-behavior analysis; not started)
```

---

## Status at a glance

| Component | Status |
|---|---|
| BEVHeight detector, Mac CPU port | **Done.** Matches the published DAIR-V2X-I benchmark (Car 3D AP@0.5 moderate 69.4 on a 200-frame subset vs 65.46 published full-set) |
| Self-calibration (intrinsics + extrinsics, no ground truth) | **Done for the primary site.** Camera height 16.2 to 16.3 m confirmed by four independent estimates within 0.1 m; per-clip rotation from a vanishing-point solver with a pose-consistency gate |
| Ground-plane convention fix | **Done.** The training data places the road at z = -1.73 m; feeding it at z = 0 put every box 1.6 m underground. Box bottoms now sit within 0.08 to 0.11 m of the road across all five processed clips |
| 3D detection on WI DOT camera data | **Done for five clips** (1,386 frames, both view directions, two seasons) |
| Trajectory extraction (30 Hz MOT) | **Working.** Tracks graded against held-out GPS using an image-only target identification (see below) |
| Trajectory accuracy vs GPS | Position RMSE **0.90 to 0.99 m** on the instrumented vehicle, identified independently of GPS |
| Heading (yaw) refinement via self-training | **Pilot complete.** Median heading error 7.9 to 1.7 degrees on a held-out clip, detection rate unchanged |
| AV vs human classification | Not started (the research question) |

---

## Pipeline on roadside camera data

The full chain on a new clip, using the environment notes in
[`docs/install.md`](docs/install.md). Calibration and analysis run under
`.venv`; the detector and tracker run under the miniforge Python (see
[`docs/run_and_eval.md`](docs/run_and_eval.md)).

```bash
# 1. intrinsics from a single frame, no ground truth
python scripts/calibration/run_anycalib_single.py \
    --image data/camera-data/<CLIP>/frames/150.jpg \
    --out-dir outputs/calibration/camera-data/<CLIP>

# 2. extrinsics: per-clip rotation from the lane vanishing point, site height
#    as a constant established independently (guards against PTZ re-pointing)
python scripts/calibration/site_extrinsic.py \
    --frames-dir data/camera-data/<CLIP>/frames \
    --anycalib-json outputs/calibration/camera-data/<CLIP>/150_anycalib_pinhole_pinhole.json \
    --height 16.26 \
    --reference outputs/calibration/camera-data/AV_T_WE_1/metric_extrinsic_site.json \
    --out outputs/calibration/camera-data/<CLIP>/metric_extrinsic_site.json

# 3. 3D detection (applies the DAIR ground-plane convention automatically)
python scripts/object_detection/run_bevheight_generic.py \
    --frames-dir data/camera-data/<CLIP>/frames_all \
    --anycalib-json outputs/calibration/camera-data/<CLIP>/150_anycalib_pinhole_pinhole.json \
    --extrinsic-json outputs/calibration/camera-data/<CLIP>/metric_extrinsic_site.json \
    --out-dir outputs/object_detection/camera-data/<CLIP>_phase1

# 4. tracking at the true frame rate
NUMBA_DISABLE_JIT=1 python scripts/tracking/run_ab3dmot.py --fps 30 --max-age 6 \
    --det-dir outputs/object_detection/camera-data/<CLIP>_phase1 \
    --out-dir  outputs/tracking/camera-data/<CLIP>_phase1

# 5. identify the instrumented vehicle from its hood marker (image only,
#    GPS is never read), then grade that pre-selected track against GPS
python scripts/tracking/identify_target_by_marker.py --clip <CLIP>
python scripts/tracking/grade_target_vs_gps.py --clip <CLIP>
```

Step 5 exists because selecting the track that best matches GPS and then
reporting that track's error is selection on the metric being reported. The
marker-based identification chooses the vehicle before GPS is consulted; on
both front-view clips the choice was unanimous across every marker frame, and
in both cases the best-GPS-match track was a different vehicle.

The DAIR-V2X-I benchmark evaluation is unchanged from upstream:

```bash
python experiments/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py \
    --ckpt_path ./checkpoints -e -b 1 --gpus 0
```

## Fine-tuning on self-generated labels

`scripts/finetune/` implements a self-training loop: high-confidence tracks
generate 3D box labels (position from the track, heading from the track's
direction of motion, height from the road plane, dimensions from the per-track
median), a review tool renders labels beside raw predictions for human
inspection, and a trainer updates the detection head with one clip held out.
`docs/server-finetune-setup.md` documents the GPU-server environment for
training the full model.

Pilot result (detection head only, held-out clip): median heading error
7.9 to 1.7 degrees, boxes more than 45 degrees off reduced from 8.7% to 3.3%,
detections per frame unchanged (7.30 to 7.57). Measured caveat: heading
concentration on intersection scenes rises slightly (0.783 to 0.838), so
preservation of lane-change behavior is tracked explicitly before classifier
use.

---

## Known issues under investigation

**Per-frame heading instability.** Box orientation is predicted per frame from
appearance alone; no temporal information links frames. On this footage roughly
10% of car boxes deviate more than 45 degrees from the direction of travel
(measured against track motion over 12,330 detections), concentrated at 40 to
60 m range. The cameras sit ~16 m above the road with a 59 to 66 degree field
of view, against ~6 m and 42 degrees in the training data, so orientation is
read from viewpoints absent from training. Downstream consumers therefore take
heading from track motion, not from per-frame boxes, and the self-training
pilot above reduces the per-frame error directly.

**Range-dependent depth bias.** Graded against GPS, detections read about
1.2 m too near below 40 m and up to 1 m too far beyond 60 m, with the same
signature on both graded clips. This is model depth behavior, not calibration:
swapping the entire extrinsic leaves it unchanged. Addressing it requires
training the height branch (GPU server; see `docs/server-finetune-setup.md`).

**Duplicate detections.** The circle-NMS distance test compares squared
distance against the radius parameter, so the effective suppression radius is
the square root of the configured value (2.0 m for cars). Same-vehicle
duplicates at ~2.5 m along the viewing ray survive. Transient duplicates do not
hold tracker IDs, so the trajectory stage is largely unaffected.

---

## Repository layout

```
models/, layers/, ops/        BEVHeight detector (backbone, BEV head, voxel pooling)
experiments/                  training and evaluation configs (dair-v2x R50_102 is CPU-patched)
evaluators/                   KITTI-format evaluation
dataset/                      DAIR-format dataloader
scripts/calibration/          self-calibration (AnyCalib intrinsics, VP extrinsics, site height)
scripts/object_detection/     BEVHeight inference drivers (DAIR and generic camera data)
scripts/tracking/             AB3DMOT driver, target identification, GPS grading, visualizers
scripts/finetune/             pseudo-label generation, label review, head fine-tuning
scripts/reporting/            shareable visualization packages
scripts/data_converter/       DAIR and Rope3D to KITTI conversion
docs/                         install, dataset preparation, evaluation, server setup
```

Datasets, model checkpoints, run outputs, and vendored third-party code are not
committed; see `.gitignore`.

## Datasets

- **DAIR-V2X-I** (infrastructure side): detection benchmark and the training
  domain of the pretrained checkpoint (7,058 frames from five intersections).
  Frames are sampled, not continuous video.
- **WI DOT roadside clips**: 10-second 1920x1080 clips at 30 Hz from Beltline
  highway cameras, each paired with a GPS trajectory of one instrumented
  vehicle used exclusively for evaluation. Not distributed with this
  repository.

## Provenance and credit

This repository extends **BEVHeight** (Yang et al., CVPR 2023); the detector
architecture, configs, and pretrained weights originate there. Additions here:
the Mac CPU port, the self-calibration pipeline, the ground-plane convention
correction, the generic-camera inference drivers, the MOT integration with
image-only target identification and GPS grading, the self-training pipeline,
and the reporting tools. Tracking uses the **AB3DMOT** core (Weng et al., IROS
2020).

Built on: [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight),
[BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth),
[DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X),
[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT),
[AnyCalib](https://github.com/javrtg/AnyCalib).

```bibtex
@inproceedings{yang2023bevheight,
    title={BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection},
    author={Yang, Lei and Yu, Kaicheng and Tang, Tao and Li, Jun and Yuan, Kun and Wang, Li and Zhang, Xinyu and Chen, Peng},
    booktitle={IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```
