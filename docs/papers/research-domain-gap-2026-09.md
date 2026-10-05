# Domain gap: DAIR-V2X-I / Rope3D training vs a 16 m Wisconsin DOT highway camera

Literature review, September 2026. Scope: why a BEVHeight checkpoint trained on
DAIR-V2X-I degrades on our footage, what the published work offers, and which of
it is worth spending a week on. Companion note to the orientation research loop
(batch 2).

No code was modified while writing this.

---

## 1. How far out of distribution we actually are

Measured from this repository, not from memory.

| Parameter | DAIR-V2X-I (our checkpoint's training set) | Rope3D | Our WI DOT cameras |
|---|---|---|---|
| Camera height | ~6 m, 5 intersections | 6.1 to 8.1 m (Rope3D Fig. 8) | **16.2 to 16.3 m** |
| Pitch | shallow, urban intersection | 9 to 15 deg (Fig. 8); MonoUNI quotes 5 to 20 deg | **18 to 19 deg** |
| fx (1920 px wide) | 2183 and 2758 (read from `data/dair-v2x-i-s2-kitti/training/calib`) | 2100 to 3000 px (Fig. 8) | **1493 to 1556** (AnyCalib, per clip) |
| Horizontal FOV | 38 to 47 deg | ~35 to 49 deg | **59 to 66 deg** |
| Frames | sampled, not continuous | sampled | continuous 30 Hz |
| Ground planes in view | one | one | several (overpass, ramps) |

Three separate shifts, and only two of them are correctable by any image
transform:

- **Focal length / apparent scale.** Our fx is 0.54 to 0.68 of DAIR's. A car at
  50 m fills 1.5x to 1.9x fewer pixels than anything the model was trained on.
  This is exactly STMono3D's "depth-shift" and DG-BEV's focal-length pseudo-domain
  problem. **Correctable at inference by crop and resize.**
- **Pitch.** 18 to 19 deg vs ~10. A pure rotation, so **correctable by a
  homography.**
- **Camera height.** 16 m vs 6 m. This is a change of viewpoint, not of
  orientation or zoom, so **no 2D warp can fix it.** At 50 m range the depression
  angle onto a vehicle roof is atan(16/50) = 17.7 deg for us against atan(6/50) =
  6.8 deg in training. We see roof where the model learned to read yaw off the
  vehicle side. Combined with the 1.5x smaller apparent size, that is a complete
  and sufficient explanation for "yaw wobble and flips, worst at 40 to 60 m": that
  band is where the target is simultaneously roof-dominated and too small for the
  side panels to disambiguate front from back.

**The single most useful number in this whole review.** SGV3D (Yang et al. 2024,
Sec. 1 and Tab. 3) reports what happens to these detectors when only the camera
changes, inside the same dataset and city:

| Detector | DAIR-V2X-I homologous Veh. AP@0.5 | heterologous (unseen camera) |
|---|---|---|
| BEVDepth | 63.58 | **2.48** |
| BEVHeight | 65.77 | **7.86** |
| BEVHeight++ | 68.62 | **8.18** |
| SGV3D | 69.78 | **50.75** |

An unseen camera at the *same* height, pitch and focal family costs BEVHeight 88%
of its accuracy. Our camera is 2.7x higher with 1.5x the FOV. Any framing of our
problem as "the model mostly works, it just needs a nudge" is wrong, and the fact
that we get usable tracks at all is largely the calibration work plus the tracker,
not the detector's transfer.

Note also that BEVHeight++ does not help here (8.18 vs 7.86). Upgrading the
architecture is not the axis.

### Where our depth bias comes from, mechanically

`layers/backbones/lss_fpn.py`:

- `HeightNet.forward` builds a 27-dim `mlp_input` of `fx, fy, cx, cy`, the IDA
  affine, the BDA affine and the flattened `sensor2ego`, pushes it through a
  `BatchNorm1d` whose running statistics were fitted on DAIR, and uses it to
  SE-modulate both the context and the height features. Feeding fx = 1493 to a BN
  calibrated on 2183/2758 puts the modulation several standard deviations outside
  anything seen in training. The height head is being conditioned by extrapolation.
- `height2localtion` takes a **single scalar** `reference_heights` per camera and
  intersects every frustum ray with **one** ground plane. There is no per-pixel
  ground.
- `d_bound` is `[-2.0, 0.0, 90]` in `experiments/dair-v2x/...102.py` and
  `[-1.5, 3.0, 180]` in `experiments/rope3d/...102.py`. The two checkpoints are
  not interchangeable under one config: the height head channel count differs.

The measured signature (1.2 m too near below 40 m, 1 m too far beyond 60 m) is a
*compression toward a mid-range pivot*, not a uniform shift. That is what
regression toward a training-set mean looks like when the test range distribution
is wider than the training one, which is what a 16 m camera with a 65 deg FOV
gives you. A pure focal mismatch would be multiplicative and too-far everywhere,
so focal alone does not explain it. Treat this as a hypothesis, but it is a
directly testable one and it is the reason candidate 1 below is ranked first.

---

## 2. Papers

PDFs are in `docs/papers/`. BEVHeight itself is already there as
`bev-height-paper.pdf`.

### 2.1 Yang et al., BEVHeight, CVPR 2023
`bev-height-paper.pdf` · https://arxiv.org/abs/2303.08498

Regresses height above the ground plane instead of depth from the camera centre,
because for a roadside camera the depth difference between a car roof and the
road behind it collapses as range grows while the height difference does not. The
lifted frustum point is placed by intersecting the ray with a plane at
`reference_height - predicted_height`, which is where the single-plane assumption
enters. The paper's headline robustness claim (+26.88% over BEVDepth under
extrinsic perturbation) is about *noise on* the extrinsic, a few degrees of roll
and pitch jitter, not about a different camera installation. Our deployment is
not a perturbation of DAIR, it is a different distribution.

- Retraining: n/a, this is the baseline.
- Here: already deployed.
- Yaw 40 to 60 m / depth: this is the problem statement.
- Cost: none.

### 2.2 Yang et al., BEVHeight++, TPAMI 2023/2024
`yang_2023_bevheight-plus-plus.pdf` · https://arxiv.org/abs/2309.16179

Adds a depth branch alongside the height branch and fuses the two BEV
representations, arguing height is robust to extrinsic change while depth carries
better near-field geometry. Gains on homologous roadside benchmarks and on
nuScenes are real but modest (+1.9 NDS over BEVDepth on nuScenes). Code and
checkpoints exist (`yanglei18/BEVHeight_Plus`) but **only for KITTI, KITTI-360 and
Waymo**; no Rope3D or DAIR weights are published there.

- Retraining: yes, full, and no usable roadside checkpoint is released.
- Here: a full retrain on the server plus a data-pipeline port. Days.
- Yaw / depth: SGV3D's Tab. 3 measures BEVHeight++ at 8.18 heterologous against
  BEVHeight's 7.86. Within noise. **Expect nothing on our gap.**
- Cost: high. Verdict: skip.

### 2.3 Ye et al., Rope3D, CVPR 2022
`ye_2022_rope3d.pdf` · https://arxiv.org/abs/2203.13608

The roadside dataset with real camera diversity: 50k images, 1.5M boxes, mounting
heights 6.1 to 8.1 m, pitch 9 to 15 deg, focal 2100 to 3000 px (Fig. 8). Defines
the homologous split (same scenes in train and val) and the heterologous split
(80% of cameras for training, the remaining 20% of cameras held out), which is the
right way to think about our situation. Introduces Average Ground Center
Similarity, Average Orientation Similarity and Average Area Similarity. Their own
baselines lose a lot going from homologous to heterologous, and they partly
recover it by feeding the ground plane equation to the network ("-(G)" and
"-(GG)" variants).

- Retraining: the dataset itself. Public BEVHeight Rope3D checkpoints exist, see
  section 3.
- Here: no data download needed to use the checkpoint.
- Yaw / depth: indirect. Its value is the AOS metric definition (which folds
  head/tail, `cos(2*delta_theta)`, matching our folded-error score) and the
  quantified camera-diversity envelope that tells us 16 m is outside it.
- Cost: reading time.

### 2.4 Yang et al., SGV3D, arXiv 2024 / IEEE T-ITS 2025
`yang_2024_sgv3d.pdf` · https://arxiv.org/abs/2401.16110 · code
https://github.com/yanglei18/SGV3D

**The most directly relevant paper in this list.** Built on BEVHeight, aimed
squarely at deploying a roadside detector on a camera it was not trained on. Two
parts. (1) Background-Suppressed Module: a semantic segmentation branch predicts
foreground masks, the masks gate the image features, and only foreground features
are lifted to BEV, so the network stops memorising one intersection's background
and one camera pose. (2) Semi-supervised data generation: run the current model on
unlabeled target-scene images, keep high-confidence pseudo-labels, **rectify the
source-domain instances to the target camera pose and installation height**, cut
them out with SAM, and paste them onto empty target-scene backgrounds to build a
synthetic labeled target set. Results: DAIR-V2X-I heterologous vehicle AP 50.75
against BEVHeight's 7.86 (+42.57), Rope3D heterologous car +14.48. Ablation: BSM
alone contributes +8.53 vehicle, so **most of the gain is the target-domain data
generation, not the masking.**

- Retraining: yes, full, on the server.
- Here: this is the closest published thing to what we already do. Our
  self-training pilot is a weaker SSDG: we generate pseudo-labels from tracks but
  we do not rectify source instances into our geometry, we do not composite onto
  empty backgrounds, and we do not suppress background features. Adding the
  rectify-and-paste step is the highest-ceiling change available. Needs SAM (we
  already have `ravi_2024_sam2.pdf` in this folder) plus a segmentation head and
  labels for it.
- Yaw at 40 to 60 m: high ceiling. Their heterologous gains are on the exact
  failure mode (large localisation and orientation deviation at long range,
  Fig. 7 and 8). Realistically a multi-week build.
- Depth bias: should improve, since the pasted instances carry correct metric
  geometry under our camera.
- Cost: high, 1 to 3 weeks. Highest ceiling of anything here.

### 2.5 Jia et al., MonoUNI, NeurIPS 2023
`jia_2023_monouni.pdf` · https://proceedings.neurips.cc/paper_files/paper/2023/hash/2703a0e3c2b33506295a77762338cf24-Abstract-Conference.html
· code https://github.com/Traffic-X/MonoUNI

Observes that vehicle-side and infrastructure-side monocular detection differ only
by pitch angle and focal length, then removes both analytically. Instead of
regressing metric depth `z`, the network regresses

```
normalized_depth = z / [ (cos(theta) - sin(theta) * tan(delta)) * f ]
```

where `theta` is camera pitch from the ground equation, `f` the focal length, and
`delta = arctan((v_p - c_y)/f)` the elevation of the pixel row. At inference the
prediction is multiplied back. Also adds a "3D normalized cube depth" supervising
the depth of the box's eight corners rather than the centre alone, which is what
gives the accuracy on top of the invariance. SOTA on Rope3D, DAIR-V2X-I, KITTI and
Waymo with one model.

- Retraining: yes. It is a different architecture (not BEV), not a wrapper.
- Here: we would be swapping detectors entirely, and re-deriving our
  ground-plane convention fix for it. Their ablation says the focal-length term
  carries most of the benefit and the pitch term matters little when pitch is near
  zero, which is not our case.
- **The skeptical point: normalized depth is invariant to pitch and focal, not to
  camera height.** Our largest shift is the one it does not normalise. Expect it
  to remove the focal-induced part of our depth bias and little of the rest.
- Yaw: no direct mechanism; yaw is still regressed from appearance.
- Cost: high. The *idea* is cheap to steal (candidate 2 below); the model is not.

### 2.6 Simonelli et al., MoVi-3D, ECCV 2020
`simonelli_2020_movi3d.pdf` · https://arxiv.org/abs/1912.08035

The origin of the "virtual camera" trick. Generates virtual views at both train
and test time by cropping a region around a hypothesised object depth and
rescaling so that the object's apparent size is normalised with respect to
distance, then runs a small detector on each virtual view. The network never has
to learn a depth-conditional appearance model, so it generalises across depths it
never saw and the architecture gets smaller.

- Retraining: the full method needs it. The *inference-time* half (crop and
  rescale to a canonical focal, run, map boxes back) does not.
- Here: this is the mechanism behind candidate 2. Our fx is 0.54 to 0.68 of
  DAIR's, so a centre crop of 1920 * (1493/2183) ~ 1310 px upscaled to the network
  input reproduces DAIR's apparent scale exactly. Covering our full 65 deg FOV
  needs 2 or 3 overlapping crops per frame, so 2 to 3x inference cost, which is
  affordable and Mac-runnable.
- Yaw at 40 to 60 m: plausible real gain. Objects in that band go from 1.5x
  smaller than trained to exactly trained scale, and the detector's yaw head is
  the part most sensitive to apparent size.
- Depth bias: should flatten the multiplicative component. The mid-range pivot
  probably survives.
- Cost: **low**, 1 to 2 days, no GPU.

### 2.7 Li et al., STMono3D, ECCV 2022
`li_2022_stmono3d.pdf` · https://arxiv.org/abs/2204.11590 · code
https://github.com/zhyever/STMono3D

The first unsupervised-domain-adaptation study for monocular 3D detection. Names
and measures the "depth-shift": a model trained at one pixel size localises 2D
boxes correctly on a new camera but puts them at systematically wrong depth,
because it learned "far things are small" under one focal length. Fixes it with
geometry-aligned multi-scale training, rescaling camera parameters to a constant
and resizing images to match, then runs a teacher-student loop on the target
domain with a quality-aware supervision weight that scales each pseudo-label's
loss by its confidence.

- Retraining: yes for the full loop, but it is a fine-tuning loop, not a new
  architecture.
- Here: the geometry-alignment half is the same move as MoVi-3D and is free. The
  teacher-student half is a direct upgrade to our existing self-training pilot:
  replace hard pseudo-label filtering with an EMA teacher plus confidence-weighted
  loss. That addresses the recall-collapse failure we already hit once.
- Yaw: moderate. Our pilot already got 7.9 to 1.7 deg median with naive
  self-training; QAS mainly protects recall while doing it.
- Depth bias: this is the paper that says our depth symptom is the expected one
  and that rescaling is the first thing to try.
- Cost: low for the rescaling, medium (server, a few days) for the mean-teacher.

### 2.8 Wang et al., DG-BEV, CVPR 2023
`wang_2023_dgbev.pdf` · https://arxiv.org/abs/2303.01686

Domain generalisation for multi-view BEV detection, with three moves: predict
scale-invariant depth (divide out the focal so the target is intrinsics-decoupled),
dynamic perspective augmentation (apply a homography to the image *and* the
corresponding camera matrix during training, so the model sees many synthetic
camera poses), and adversarial training over pseudo-domains created by perturbing
the focal length, to push the features toward domain-agnostic. Evaluated
Waymo/nuScenes/Lyft.

- Retraining: yes for the adversarial part. The perspective augmentation is a
  training-time data augmentation we could bolt onto our fine-tuning.
- Here: the cheapest useful piece is the augmentation: during head or height
  fine-tuning on the server, randomly homography-warp our frames and update the
  intrinsics and extrinsics consistently. That teaches the model that the camera
  moves, rather than memorising ours.
- Yaw: indirect, mainly a regulariser against re-overfitting when we fine-tune.
- Depth bias: scale-invariant depth is the principled version of what candidate 2
  approximates at inference.
- Cost: low to medium once we are already fine-tuning.

### 2.9 Zhou et al., WARM-3D, 2024 (TUMTraf)
`zhou_2024_warm3d.pdf` · https://arxiv.org/abs/2407.20818

Sim2real weak supervision for **roadside** monocular 3D detection, on TUMTraf. A
detector is pretrained on a synthetic roadside dataset, then adapted to real
footage using only **2D** labels from an off-the-shelf 2D detector as weak
supervision on the target domain, plus consistency between the 3D prediction's
projection and the 2D box. +12.40 mAP 3D over the baseline with pseudo-2D
supervision alone; with ground-truth 2D labels it approaches an oracle.

- Retraining: yes, on the server.
- Here: **this is the honest answer to "we only have GPS for one vehicle".** We do
  not need 3D ground truth for the other cars, we need a cheap 2D signal for them,
  and a COCO-class 2D detector gives that for free on every vehicle in every
  frame. A projection-consistency loss against those 2D boxes supervises all
  traffic without inventing 3D labels, and it constrains exactly the two things we
  get wrong: the projected box footprint (depth) and the projected box shape
  (yaw). Applicable without any simulation work; the sim2real pretraining is
  separable from the weak-supervision loss.
- Yaw at 40 to 60 m: moderate to good. A wrong yaw changes the projected box's
  aspect ratio, so 2D-consistency does penalise it, though a head/tail flip is
  nearly invisible to a 2D box and will not be caught.
- Depth bias: good. Projected footprint size is a direct depth constraint, and it
  covers every vehicle at every range rather than one vehicle in one lane.
- Cost: medium. A 2D detector run over our frames (cheap, Mac or server) plus a
  projection-consistency loss term in `train_finetune.py`.

### 2.10 Yang et al., MonoGAE, T-ITS 2024
`yang_2023_monogae.pdf` · https://arxiv.org/abs/2310.00400

Roadside monocular detection that injects the ground plane as a learned prior:
ground-aware embeddings are supervised during training and fused with image
features by cross-attention. The part that matters for us is the replacement of a
single global ground plane with a **pixel-level refined ground plane equation
map**, dividing the image into regions and fitting a plane equation per region,
explicitly to be robust to camera pose divergence and non-flat ground.

- Retraining: yes, full.
- Here: we cannot adopt the architecture cheaply, but the idea transfers directly
  to `height2localtion`, which currently takes one scalar `reference_heights`. A
  per-region reference height is a small, local change with no retraining, and it
  is the *correct* answer to our overpass problem (see verdict (a)).
- Yaw: little direct effect.
- Depth bias: for vehicles on the main road, none. For anything on a different
  plane, it is the difference between right and badly wrong.
- Cost: low for the borrowed idea, high for the paper's method.

### 2.11 Shi et al., CoBEV, T-IP 2024
`shi_2023_cobev.pdf` · https://arxiv.org/abs/2310.02815

Argues depth and height are complementary rather than alternatives: depth carries
precise geometry, height carries semantic context (height intervals separate
pedestrian from car from truck). Estimates per-pixel distributions of both, lifts
both into 3D, fuses laterally with a two-stage complementary feature selection
module, and distils BEV features. SOTA on DAIR-V2X-I and Rope3D, with the claimed
benefit concentrated in long-range scenarios.

- Retraining: yes, full, and no roadside checkpoint we could load.
- Here: same category as BEVHeight++. There is no published evidence it survives
  a camera change; all its numbers are homologous.
- Yaw / depth: unknown on our gap, and the SGV3D table's verdict on BEVHeight++
  (no heterologous benefit from a better homologous architecture) is the prior I
  would apply.
- Cost: high. Skip.

### 2.12 Wang et al., BEVSpread, CVPR 2024
`wang_2024_bevspread.pdf` · https://arxiv.org/abs/2406.08785

A drop-in replacement for voxel pooling. Standard frustum-to-BEV pooling assigns
each frustum point's feature to exactly one BEV grid cell, which quantises away
sub-cell position; BEVSpread spreads each point's feature over neighbouring cells
with distance- and depth-dependent weights. As a plug-in it adds (1.12, 5.26,
3.01) AP for vehicle, pedestrian, cyclist, with a CUDA kernel that keeps inference
time comparable.

- Retraining: yes. It changes the lifting operator, so existing weights do not
  transfer cleanly.
- Here: our repo already has a voxel-pooling extension (`ops/`, built with
  `setup.py develop` on the server). Swapping in BEVSpread means a new CUDA kernel
  and a retrain.
- Yaw at 40 to 60 m: the mechanism (position quantisation error) grows with range,
  so there is a plausible story. But this fixes a sub-grid-cell error of order the
  BEV cell size (0.8 m at 128x128 over 102.4 m), not a 45 deg yaw flip.
- Depth bias: sub-cell only. Will not move a 1.2 m bias.
- Cost: high, benefit small relative to our error sizes. Skip.

### 2.13 Liu et al., HeightFormer, 2024
`liu_2024_heightformer.pdf` · https://arxiv.org/abs/2410.07758

Roadside monocular detection that keeps the height-based 2D-to-3D projection but
adds a Spatial Former for height alignment and a Voxel Pooling Former for
efficient BEV feature extraction, arguing prior height-based lifting ignores
height alignment. Evaluated on Rope3D and DAIR-V2X-I, better on vehicles and
cyclists.

- Retraining: yes, full, no released roadside checkpoint found.
- Here: same objection as CoBEV and BEVHeight++. Homologous-only evidence.
- Yaw / depth: unknown on our gap.
- Cost: high. Skip, but keep on the shelf as a height-branch design reference if
  we ever rewrite the lifting.

### 2.14 Fan et al., CBR, 2023
`fan_2023_cbr.pdf` · https://arxiv.org/abs/2303.03583

Calibration-free BEV representation. Drops camera parameters from the pipeline
entirely: two MLPs decouple perspective-view features into a front view and a
bird's-eye view under box-induced foreground supervision, then a cross-view
fusion module matches features across the two orthogonal views by similarity. On
DAIR-V2X it reaches acceptable (not SOTA) accuracy with no camera parameters at
all, and is by construction immune to calibration noise.

- Retraining: yes, and it is a different model.
- Here: relevant as a diagnostic argument rather than a method. CBR's existence
  proves that a roadside detector *can* work without intrinsics, which means our
  problem is not "we need better calibration". We already showed that empirically
  (swapping the entire extrinsic left the depth bias unchanged). CBR is the
  published confirmation.
- Yaw / depth: n/a for adoption.
- Cost: high. Skip as a method; cite as evidence.

### 2.15 Chen et al., MOSE, CVPR 2024
`chen_2024_mose.pdf` · https://arxiv.org/abs/2404.05280

Exploits the one thing roadside cameras give you for free: the camera never
moves, so the ground geometry per pixel is time-invariant. Aggregates
frame-invariant, object-invariant, scene-specific features ("scene cues", which
are essentially the height of the real ground surface at each image location)
into a scene cue bank across many frames of the same scene, with an extrinsic
augmentation strategy, then a transformer decoder lifts 2D proposals using the
cached scene cues plus a 3D position embedding. Reports strong generalisation to
heterologous scenes.

- Retraining: yes, full.
- Here: the *concept* is directly actionable and cheap. We have 30 Hz continuous
  video and thousands of tracked vehicles per clip. Every confidently tracked
  vehicle's wheel contact point gives one sample of "the ground height at this
  image location". Accumulating those over a clip yields an empirical per-pixel
  ground surface, which is both the multi-plane fix and an independent check on
  our single-plane assumption. No network involved.
- Yaw: none directly.
- Depth bias: potentially large for anything off the main plane, and it gives a
  measurable answer to whether the overpass is even causing errors.
- Cost: low for the borrowed idea (a script over existing tracks), high for the
  model.

### 2.16 Lin et al., MonoTTA, ECCV 2024
`lin_2024_monotta.pdf` · https://arxiv.org/abs/2405.19682

Fully test-time adaptation for monocular 3D detection: adapt a trained model to
unlabeled test data with no access to training data and no labels. The paper's
key observation is the failure mode we should worry about: out-of-distribution
input depresses detection scores below the preset threshold, so naive TTA sees
few positives and many noisy ones and collapses. Fixes: reliability-driven
adaptation (self-adaptively pick high-score objects, since those stay reliable and
optimising them lifts confidence globally) and noise-guard negative regularisation
on the many low-score detections to avoid overfitting to noise.

- Retraining: no training data needed, but it updates model parameters at test
  time (typically BN affine parameters), so it needs a backward pass. On the Mac
  the DCN backward is unavailable, so this is server-only for the height branch;
  BN-only adaptation of the image branch might run on CPU.
- Here: interesting because BN running statistics are exactly what we identified
  as mis-calibrated (the 27-dim camera MLP's BN, and every BN in the backbone).
  The cheapest version is not even TTA: **recompute BN running statistics on our
  footage with a few hundred forward passes, no gradients at all.** That is a
  half-day experiment with a clean hypothesis.
- Yaw / depth: uncertain. Worth one run precisely because it is so cheap.
- Cost: very low for BN-stat recomputation, medium for full MonoTTA.

### 2.17 Yang, Liang, Carin, Object Detection as a Positive-Unlabeled Problem, BMVC 2020
`yang_2020_pu-detection.pdf` · https://arxiv.org/abs/2002.04672

The reference for verdict (b). Standard detection training is positive-negative:
every region not covered by a label is assumed to be background and pushed down.
When annotations are incomplete, those pushes are false negatives and the
detector's recall degrades in proportion to the missingness. The paper reformulates
the classification loss as positive-unlabeled, removing the assumption that
unlabeled means negative, and shows it beats the PN loss on VOC and COCO across a
range of label-missingness rates.

- Retraining: it is a loss change, so it applies to any fine-tuning we do.
- Here: directly decides how to implement Hang's masked-loss proposal. See
  verdict (b).
- Yaw / depth: n/a; it is about not destroying recall.
- Cost: low. It is a change of a few lines in the loss, and there are simpler
  equivalents (ignore regions) that get most of the benefit.

### 2.18 Zimmer et al., TUMTraf / A9 Intersection Dataset, ITSC 2023
`zimmer_2023_tumtraf-intersection.pdf` · https://arxiv.org/abs/2306.09266

Roadside camera and LiDAR dataset from a gantry near Munich: 4.8k frames, 57.4k
3D boxes, 10 classes, with turns, overtaking and U-turns explicitly included. The
gantry is ~7 m. Useful to us as the target domain of WARM-3D, as a second
roadside domain with published cross-domain numbers, and as a source of turning
manoeuvres if we ever need heading diversity that our straight highway clips lack
(relevant to the 0.783 to 0.838 heading-concentration regression noted in
`docs/server-finetune-setup.md`).

- Retraining: n/a, it is data.
- Here: optional. Public download, no GPS, no 16 m cameras.
- Cost: low, value low unless we need turning data.

---

## 3. Is there a public checkpoint trained on Rope3D or a wider height range?

**Yes, from the original BEVHeight repo, and it is the cheapest real candidate we
have.** From `https://github.com/ADLab-AutoDrive/BEVHeight` README:

| Checkpoint | Config | Verified |
|---|---|---|
| Rope3D R50, 102.4 m | `experiments/rope3d/bev_height_lss_r50_864_1536_128x128_102.py` | `https://cloud.tsinghua.edu.cn/f/fa3e2d07d62a44b7a337/?dl=1` returns HTTP 200, 925 MB, filename `BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt` |
| Rope3D R50, 140.8 m | `experiments/rope3d/bev_height_lss_r50_864_1536_128x128_140.py` | `https://cloud.tsinghua.edu.cn/f/343be049d5e74d14a5af/?dl=1` |

Both are the **homologous** split, so they are not trained to generalise across
cameras, but they were trained across Rope3D's whole camera set: heights 6.1 to
8.1 m, pitch 9 to 15 deg, focal 2100 to 3000 px, many intersections. That is far
more camera diversity than our DAIR checkpoint's five intersections at one height.

Two things to know before running it:

1. **The config is not interchangeable.** `d_bound` is `[-2.0, 0.0, 90]` for DAIR
   and `[-1.5, 3.0, 180]` for Rope3D, so the height head has 180 channels instead
   of 90. Load the Rope3D checkpoint with the Rope3D config. The wider and
   finer-binned height range is itself a reason to expect it to behave better at
   16 m.
2. **Classes are `["Car", "Bus"]`**, not `["Car", "Pedestrian", "Cyclist"]`. Fine
   for the AV-identification pipeline, which only consumes cars.

No public checkpoint exists for a camera above ~8 m. BEVHeight++ publishes weights
only for KITTI, KITTI-360 and Waymo. MonoUNI, MonoGAE, CoBEV, HeightFormer and
BEVSpread either publish nothing roadside or publish homologous-only weights we
would still have to retrain. **Nobody has released a roadside 3D detector trained
anywhere near 16 m.** If we want one, we build it.

---

## 4. Verdict on the lab lead's two proposals

### (a) Masking image regions belonging to other ground planes (overpasses)

**Verdict: the underlying concern is real and well-founded, but pixel masking is
the wrong instrument. Do the geometry fix instead, and measure the problem first.**

What the literature says.

- The concern is correct in mechanism. `height2localtion` intersects every ray
  with **one** plane defined by a single scalar `reference_heights`. A vehicle on
  a 5 m overpass seen by a 16 m camera is placed at roughly
  `range * 16/(16-5)` = 1.45x its true range. Nothing in BEVHeight can express two
  ground heights. Every paper that deals with non-flat roadside geometry agrees:
  MonoGAE explicitly replaces the global plane with a **pixel-level refined ground
  plane equation map**, Rope3D's own baseline gains come from its "-(G)" and
  "-(GG)" variants feeding the ground plane and a gridded ground, and MOSE learns a
  per-pixel scene cue that is precisely "the height of the real ground surface
  here". The field's answer to multiple planes is a **per-pixel or per-region
  ground height**, not a mask.
- Masking is a published technique, but not this kind of masking. SGV3D's
  Background-Suppressed Module is the closest thing, and it differs on three axes
  that matter: it masks **features, not pixels**; the mask comes from a **learned
  semantic segmentation branch trained jointly**, not a hand-drawn polygon; and it
  suppresses **background** (road surface, buildings, sky) to keep foreground
  vehicle features only, which is the opposite of what we would be doing by
  deleting a region that contains vehicles. Its standalone contribution is +8.53
  vehicle AP on the DAIR heterologous benchmark, real but the smaller half of
  SGV3D's gain.

Risks of doing it as proposed.

- **Deleting pixels shifts feature statistics.** BEVHeight is a fully
  convolutional backbone with BatchNorm everywhere, plus a camera-conditioned SE
  modulation. A large black rectangle is itself out of distribution and will
  perturb detections *outside* the mask, including on the main road. This is the
  same class of failure MonoTTA documents: OOD input depresses scores below the
  threshold and you silently lose recall. We have already been bitten once by a
  change that "improved heading by detecting nothing at all".
- **It fixes nothing for the boxes we care about.** Our 40 to 60 m yaw failures
  and our 1.2 m depth bias are measured on main-road vehicles on the main plane.
  Masking an overpass cannot move either number. It only removes some spurious
  detections we could remove for free downstream.
- **It costs recall on ramps.** Ramp traffic merging onto the highway is exactly
  the behaviour an AV-versus-human classifier would want.

How to do it properly, cheapest first.

1. **Measure before building.** Count what fraction of detections and of tracked
   trajectories fall outside the main-road BEV polygon. If it is 1%, this is not a
   problem worth a method. Free, one script over existing outputs.
2. **Filter in BEV, not in the image.** Drop boxes whose BEV footprint lies
   outside the drivable polygon for this site. Post-hoc, reversible, zero effect on
   the network, no recall loss on the main road. This is the simplest option and it
   captures most of the practical value.
3. **If off-plane vehicles must be detected correctly, go per-region.** Give
   `height2localtion` a per-region `reference_height` (MonoGAE's idea, minus the
   learning): a coarse image-space map of ground height, built empirically from
   tracked vehicles' wheel-contact points accumulated over a clip (MOSE's idea,
   minus the transformer). No retraining, and it can be validated against our
   existing box-bottom-to-road residuals (currently 0.08 to 0.11 m on the main
   plane).
4. **Only if we retrain anyway**, adopt SGV3D's BSM properly: a learned
   foreground segmentation branch gating the features before the BEV lift. Do not
   hand-mask.

### (b) Fine-tuning with a loss masked to the single instrumented vehicle

**Verdict: as stated, this will work statistically and fail in practice. The
instinct (only trust the box we actually know) is right; "mask the loss to one
vehicle" is the wrong way to express it, and the literature names the exact
failure.**

What the literature says.

- Yang, Liang and Carin (2020) is the direct hit. Standard detection loss is
  positive-negative: every region not labeled positive is supervised as
  background. If we label only the instrumented vehicle, **every other car in
  every frame becomes an explicit negative**. That is not 1 missing label, it is
  roughly 6 to 7 false negatives per frame at 30 Hz. Their result is that recall
  degrades proportionally to missingness, and our missingness would be ~90%. The
  model's cheapest way to satisfy that loss is to stop detecting cars.
- We have already observed this failure in this repository. `docs/server-finetune-setup.md`
  records an earlier run that "improved heading by detecting nothing at all", which
  is why `cars_per_frame` is in the eval output. A single-vehicle masked loss is
  the most efficient possible way to reproduce it.
- **Sample size and diversity.** One vehicle at 30 Hz over a 10 s clip is ~300
  frames, but they are ~300 near-identical, highly autocorrelated views of one car
  driving straight down one lane. The effective sample size for a 3D detection head
  is closer to a handful of independent observations. Worse, that vehicle's yaw is
  nearly constant, so the model learns "point along the road". We already measured
  that leak with much weaker supervision: heading concentration on DAIR
  intersection frames rose 0.783 to 0.838 in the head-only pilot. Single-vehicle
  supervision would amplify it, and it would erase the lane-change signal the
  AV-versus-human classifier depends on. Do not trade the research question for a
  metric.
- **Range and lane bias.** The instrumented vehicle occupies a specific lane and
  a specific range distribution. Our depth bias is range-dependent with a pivot
  near 50 m. Fitting it from one vehicle's trajectory risks learning that
  vehicle's range profile, not the model's depth warp.
- WARM-3D is the published alternative: supervise all target-domain objects
  weakly with cheap 2D labels rather than supervising one object strongly.
  STMono3D's quality-aware supervision and SGV3D's SSDG are the same instinct,
  confidence-weighted or synthesised labels for *everything*, not exact labels for
  *one thing*.

How to do it properly.

1. **Never let unlabeled vehicles be negatives.** Use ignore regions, the KITTI
   `DontCare` mechanism the DAIR format already carries: any region covered by a
   confident detection but not by a label contributes **neither** positive nor
   negative gradient. Only genuinely empty road stays negative. This is one masking
   decision and it removes the recall-collapse risk almost entirely. The
   positive-unlabeled loss is the more principled version if ignore regions prove
   insufficient.
2. **Do not use the GPS vehicle as a detection target at all if you can avoid
   it.** It is far more valuable as a *metric anchor*. The depth bias is a smooth
   1-D function of range; with GPS on one vehicle we can fit that curve offline and
   apply the correction to every box, no training, no recall risk, no yaw prior.
   See candidate 1.
3. **If we do fine-tune with it**, combine three supervision tiers rather than
   masking to one: GPS vehicle at high loss weight (exact, few), track-derived
   pseudo-labels at confidence-weighted low weight (STMono3D QAS), and 2D
   projection consistency from an off-the-shelf 2D detector for every other
   vehicle (WARM-3D). Keep `cars_per_frame` and the DAIR heading-concentration
   check as hard gates on every run, as `docs/server-finetune-setup.md` already
   specifies.
4. **Hold the GPS clip out of training entirely** if it is also the evaluation
   anchor. Training on GPS and then grading against GPS is the same selection
   problem the marker-based target identification was built to avoid.

### One more caution the parent asked for

Be skeptical of any result reported for the instrumented vehicle alone. Our 0.90
to 0.99 m position RMSE is measured on one car, identified by a hood marker,
usually in a middle lane at moderate range. It is not evidence that the detector
is accurate at 16 m in general, and a fine-tune that improves that one number can
easily be worse everywhere else. Every candidate below should be graded by
`scripts/evaluation/score_heading.py` across all 8 held-out clips first, with GPS
grading used only as a secondary guard on kept candidates, exactly as the design
spec says.

---

## 5. Ranked candidates for this repo

Cheapest first among those with a real expected gain.

**1. Offline range-dependent depth recalibration from the GPS vehicle.
Cost: half a day, Mac only, no training.**
Fit a one-dimensional correction `d_corrected = d + f(d)` from the GPS-graded
residual on the two graded clips, as a monotone spline or a two-parameter affine
in range, then apply it to every detection at output time. The measured signature
(1.2 m near below 40 m, 1 m far beyond 60 m, same on both graded clips) is a
smooth compression toward a mid-range pivot, which is what STMono3D's depth-shift
plus regression-toward-the-training-mean predicts, and it is a property of the
model and the geometry rather than of a particular vehicle. Fit on one clip, apply
to the other, and check the residual flattens on the clip that was not fitted; if
it does not transfer between clips it is not a model property and this candidate
dies immediately, cheaply, which is itself worth knowing. Expected effect: removes
most of the depth bias at a fraction of the cost of unfreezing the height branch;
no effect on yaw. This is not a paper method, it is what the papers imply once you
accept that the bias is systematic, and it is strictly safer than teaching a
network the same correction from 300 correlated frames of one car.

**2. Virtual-camera crop-and-rescale to the DAIR focal length (MoVi-3D / STMono3D
GAMS / DG-BEV scale-invariant depth). Cost: 1 to 2 days, Mac CPU, no training.**
Our fx is 1493 to 1556 against DAIR's 2183 and 2758, so objects appear 1.5x to
1.9x smaller than anything the model saw. Centre-crop each 1920x1080 frame to
about 1310 px wide, upscale to the network input, and adjust the intrinsic and the
IDA matrix consistently so the geometry stays exact; tile 2 or 3 overlapping crops
to cover the full 65 deg field of view and merge with the existing NMS. Optionally
also apply the pitch-correcting homography in the same warp, which is exact since
pitch is a pure rotation, bringing 18 to 19 deg to DAIR's ~10. This lands the
27-dim camera MLP's BatchNorm input back inside its trained range and restores the
apparent object scale for the 40 to 60 m band where our yaw is worst. Expected
effect: the most plausible cheap win on yaw, plus the multiplicative part of the
depth bias; it cannot touch the 16 m versus 6 m viewpoint elevation, so do not
expect it to close the gap alone. Runs as a `scripts/orientation/candidates/`
entry with a detection rerun, gradeable in one afternoon.

**3. Rope3D checkpoint swap. Cost: 1 hour of download plus one inference rerun per
clip, server or Mac.**
`BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt` (925 MB, link verified above) with
`experiments/rope3d/bev_height_lss_r50_864_1536_128x128_102.py`. It was trained
across Rope3D's whole camera set (6.1 to 8.1 m heights, 9 to 15 deg pitch, 2100 to
3000 px focal, many intersections) rather than five intersections at one height,
and its height bins are wider and finer (`d_bound` [-1.5, 3.0, 180] against
[-2.0, 0.0, 90]). Mind the config mismatch (the head has 180 channels, not 90) and
the class list (`Car`, `Bus`). Be skeptical: Rope3D's own heterologous split shows
BEVHeight losing most of its accuracy on cameras it did not train on, and 16 m is
outside Rope3D's envelope entirely, so this is a lottery ticket. It is just an
extremely cheap lottery ticket, and it is already on the batch 1 list as a
checkpoint variant.

**4. Recompute BatchNorm statistics on our footage (the free half of MonoTTA).
Cost: half a day, forward passes only, Mac CPU.**
Put the model in train mode for BN only, freeze all weights, run a few hundred of
our frames to re-estimate every BN running mean and variance including the 27-dim
camera-parameter BN, then switch back to eval. No gradients, so no deformable-conv
backward problem, so it runs on the Mac. MonoTTA's premise is that OOD input
depresses scores and collapses detection; our mis-calibrated BN statistics are a
concrete instance of that, and this is the cheapest possible test of whether they
matter. Expected effect: honestly uncertain, plausibly a small gain on both yaw
and detection rate, possibly nothing. Worth running because it is a day and it is
a clean hypothesis. Watch `cars_per_frame`: BN recalibration can shift the score
distribution and silently change the operating point.

**5. Weak 2D supervision for all traffic, plus ignore regions, in the server
fine-tune (WARM-3D + STMono3D QAS + PU/ignore-region handling).
Cost: 1 to 2 weeks, server.**
This is the replacement for the single-vehicle masked loss, and the only candidate
here that can plausibly fix the 16 m viewpoint gap rather than work around it. Run
an off-the-shelf 2D detector over our frames to get a 2D box for every vehicle,
then fine-tune with three supervision tiers: the GPS vehicle at high weight,
track-derived 3D pseudo-labels at confidence-proportional weight, and a projection
consistency loss between each predicted 3D box and its matched 2D box for
everything else. Mark every confidently detected but unlabeled region as ignore so
no unlabeled vehicle is ever supervised as background. Unfreeze the height branch,
which is the whole reason for being on the server, and add DG-BEV's homography
perspective augmentation so we do not simply re-overfit to one camera pose. Gate
every run on `cars_per_frame` not falling and on DAIR heading concentration not
rising, both already in `eval_finetune.py`. SGV3D's rectify-and-paste pipeline is
the natural next step above this one if it works, and the natural fallback if the
weak supervision proves too loose.

**Not recommended:** BEVHeight++, CoBEV, HeightFormer and BEVSpread as
architecture swaps. All four report homologous-only gains, all four require a full
retrain, none publishes a roadside checkpoint we could load, and SGV3D's table
shows BEVHeight++ gaining nothing on the heterologous split where our whole
problem lives (8.18 against BEVHeight's 7.86). Architecture is not our axis;
camera domain is.

---

## Files

- `docs/papers/research-domain-gap-2026-09.md` (this note)
- PDFs added to `docs/papers/`: `yang_2023_bevheight-plus-plus.pdf`,
  `jia_2023_monouni.pdf`, `fan_2023_cbr.pdf`, `yang_2023_monogae.pdf`,
  `wang_2024_bevspread.pdf`, `shi_2023_cobev.pdf`, `ye_2022_rope3d.pdf`,
  `yang_2024_sgv3d.pdf`, `simonelli_2020_movi3d.pdf`, `li_2022_stmono3d.pdf`,
  `wang_2023_dgbev.pdf`, `zhou_2024_warm3d.pdf`, `yang_2020_pu-detection.pdf`,
  `liu_2024_heightformer.pdf`, `zimmer_2023_tumtraf-intersection.pdf`,
  `chen_2024_mose.pdf`
- Already present: `bev-height-paper.pdf`, `lin_2024_monotta.pdf`,
  `ravi_2024_sam2.pdf`
