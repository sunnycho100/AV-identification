# Temporal consistency of 3D box orientation: literature survey

Date: 2026-09-15. Scope: methods that make per-frame 3D box yaw consistent across
video frames, for monocular and roadside 3D detection. Written for the orientation
research loop.

## Why this survey exists

BEVHeight predicts yaw from single-frame appearance. On WI DOT footage about 10% of
car boxes are more than 45 degrees off the direction of travel, concentrated at 40 to
60 m, and yaw wobbles frame to frame. The lab constraint is that the improved yaw must
stay an independent signal from position: motion heading is the evaluation reference,
not an allowed output. So for every paper below there is an explicit line on whether
the method derives orientation from position.

### How to read the "uses position" line

Three levels are used consistently.

- **No.** The output yaw is a function only of image appearance, or only of the
  detector's own per-frame yaw outputs. Track membership may come from position based
  association, which is allowed, because association selects *which* yaws to combine,
  not *what value* they take.
- **Partly.** The method never writes a motion heading into the output, but position or
  displacement enters the feature that produces the yaw, so the learned yaw is
  correlated with motion by construction.
- **Yes.** The output yaw is computed from displacement. Not usable as an output here.

### Two facts that constrain everything below

**No sequential roadside training data.** DAIR-V2X-I is 7,058 sampled frames from five
intersections, not continuous video, and Rope3D is likewise frame-sampled. Every
temporal detector in section B needs consecutive frames with labels to train. The only
continuous video available here is the WI DOT clips, which have no 3D ground truth. So
any temporal detector has to be trained on our own pseudo-labels, and the existing
pseudo-label heading comes from track motion, which is exactly what the constraint
forbids. This kills most of section B as a near-term option, regardless of its merit
on nuScenes.

**Roadside viewing geometry is not the ego-vehicle geometry these methods were tuned
on.** At 16 m height, a car at 40 m is seen at about 22 degrees of elevation and one at
60 m at about 15 degrees. A 6 m DAIR camera sees the same ranges at about 8.5 and 5.7
degrees, and a 1.5 m ego camera at about 2 and 1.4 degrees. A car at 60 m in our frames
is roughly 110 px long and 45 px wide, so it is not small; what is weak is the
front-versus-back cue, because at a steep elevation the roof dominates and the
windshield-versus-rear-window and taillight evidence shrinks. Any method that only
reweights or regularizes appearance features cannot manufacture a cue that the pixels
do not contain. Methods that propagate a confident yaw from one part of a track to
another can.

A caution about the 40 to 60 m band itself: elevation angle falls monotonically with
range, and pixel size falls monotonically with range, so neither explains a *band*. The
error concentration at 40 to 60 m is unexplained by anything in this literature. Treat
every "expected gain at 40 to 60 m" number below as a hypothesis to be measured by
`scripts/evaluation/score_heading.py`, not a prediction.

---

## A. Roadside and infrastructure

### A1. Pro3D: Roadside Monocular 3D Detection Prompted by 2D Detection

Ma, Hua, Li, Kong, 2024 (v5 Dec 2025). arXiv:2404.01064.
<https://arxiv.org/abs/2404.01064>
Local: `docs/papers/ma_2024_pro3d.pdf` (v5), `docs/papers/ma_2024_pro3d_v2_yaw-tuning.pdf` (v2, contains the yaw tuning section).

**Mechanism.** Pro3D feeds off-the-shelf 2D detections into a roadside 3D detector as
prompts, on the reasoning that 2D detection is far more reliable than 3D on the same
pixels. Because the roadside camera is fixed, it also averages object-masked training
frames into an empty background image and concatenates that "scene prior" to the input.
Separately, and this is the part that matters here, version 2 describes a post-hoc
**yaw tuning** step: take each 3D cuboid, rotate it about the yaw axis, project it to
the image, and pick the yaw that maximizes IoU between the projected 2D extent and the
2D detector's box. On DAIR-V2X-I with BEVHeight as one of the baselines, yaw tuning
raised vehicle AOS from 92.7 to 95.6 easy and AP by about 0.7, on both DAIR-V2X-I and
Rope3D.

**Uses position to infer orientation.** No. The objective is IoU between two image-space
boxes in a single frame. Nothing temporal, nothing positional.

**What it would take here.** Run a 2D detector on the frames (an off-the-shelf COCO
model is enough for cars), match each `NNN_pred.json` box to a 2D box by projecting the
cuboid with the existing intrinsics and extrinsics and taking best IoU, then do a 1D
search over yaw in [-pi, pi) at say 1 degree steps, re-projecting the eight corners each
time. All the projection code needed already exists in the inference driver. Output is a
new per-frame yaw, which is exactly the candidate interface in the design doc. No
retraining, no GPU beyond the 2D detector pass.

**Expected gain at 40 to 60 m.** Moderate on the folded-mod-180 primary, near zero on the
flip fraction. The projected-box-IoU objective is symmetric under a 180 degree yaw flip
for a box-shaped vehicle, so it cannot fix front-versus-back errors at all. It also gets
flat as the vehicle approaches axis-aligned with the image, and at a 15 degree elevation
the projected extent changes slowly with yaw, so the optimum is poorly conditioned
exactly in the band we care about. Their own gain was 2 to 3 AOS points on DAIR, where
baseline yaw is already good; expect less lift per frame here but on a much worse
baseline, so the absolute movement could still be real. Worth running, with low
confidence.

**Cost.** 1 to 2 days. 2D detector inference on 1,386 frames is minutes on CPU, seconds
on GPU. No training.

### A2. MOSE: Boosting Vision-based Roadside 3D Object Detection with Scene Cues

Chen, Chen, Tang, Niu, Zhu, 2024 (CVPR 2024). arXiv:2404.05280.
<https://arxiv.org/abs/2404.05280>
Local: `docs/papers/chen_2024_mose.pdf`

**Mechanism.** MOSE exploits the fact that a roadside camera is static, so the scene
itself is a frame-invariant feature. It defines a scene cue as the height between the
real road surface and a virtual ground plane, and maintains a **scene cue bank** that
aggregates these cues over many frames of the same scene, with an extrinsic augmentation
strategy so the bank does not overfit one camera pose. A transformer decoder consumes
the aggregated cues with 3D position embeddings. The frames aggregated need not be
consecutive, which is the point: it is temporal aggregation of *scene* statistics, not
of *object* state.

**Uses position to infer orientation.** No, but also: it is not an orientation method.
The scene cue improves localization and cross-scene generalization. Nothing in the
method targets yaw.

**What it would take here.** A full retrain with a new head and a bank module. Our
pipeline already does the closest thing that matters, by fixing the ground-plane
convention and solving per-clip extrinsics from a vanishing point, so the marginal
value of the scene cue for us is mostly about depth, not yaw.

**Expected gain at 40 to 60 m.** Essentially none on heading. Listed here because it is
the best current example of "use the fact that the camera is fixed", and because its
extrinsic augmentation strategy is a directly reusable idea for the domain gap work.

**Cost.** Weeks, GPU, full training. Not recommended for the heading loop.

### A3. BEVHeight++: Toward Robust Visual Centric 3D Object Detection

Yang, Tang, Li, Chen, Yuan, Wang, Huang, Zhang, Yu, 2023. arXiv:2309.16179.
<https://arxiv.org/abs/2309.16179>
Local: `docs/papers/yang_2023_bevheight-plus-plus.pdf`

**Mechanism.** The journal extension of the detector this repo runs. BEVHeight regresses
height-to-ground instead of depth-to-camera, because depth differences between a car and
the road collapse with range while height does not. BEVHeight++ fuses both height and
depth encodings so the model works on ego-vehicle as well as roadside data. The relevant
experiments for us are the extrinsic disturbance ones: rotating the camera by 1 degree in
roll, or in roll and pitch together, and showing BEVHeight++ still finds seventeen
vehicles where BEVDepth finds far fewer. It is single-frame throughout; the related work
surveys temporal fusion but the method does not use it.

**Uses position to infer orientation.** No. Single frame, appearance only. This is the
current failure mode, documented.

**What it would take here.** Swapping to the BEVHeight++ checkpoint is a checkpoint
change, so it fits batch 1 of the design doc rather than this survey. Worth scoring since
the pretrained weights exist.

**Expected gain at 40 to 60 m.** Unknown and probably small for heading specifically. The
height-plus-depth fusion targets localization. Its extrinsic robustness result is
relevant to us in a different way: it suggests our yaw errors are unlikely to be a
1-degree calibration artifact, which matches the finding that swapping the entire
extrinsic did not move the depth bias.

**Cost.** Hours if the checkpoint drops in. GPU for inference only, or CPU on the Mac at
the usual speed.

### A4. Towards Stable 3D Object Detection (Stability Index and Prediction Consistency Learning)

Jiabao Wang, Meng, Liu, Yan, Ke Wang, Cheng, Hou, ECCV 2024. arXiv:2407.04305.
<https://arxiv.org/abs/2407.04305>
Local: `docs/papers/wang_2024_stability-index.pdf`

**Mechanism.** The paper argues that mAP and MOTA cannot see temporal stability, and
defines a Stability Index over four attributes, one of which is heading (SI_h). It then
proposes **Prediction Consistency Learning**: run the detector on two samples of the same
object, differing in timestamp and in augmentation (flips, rotation, scaling),
de-augment each prediction back to a common frame, match predictions across the pair by
object ID, and add a loss that the two predictions' *errors* agree. The heading error is
encoded as (sin(theta - theta_gt), cos(theta - theta_gt)) so the loss is well behaved
across the wrap. On CenterPoint it lifts vehicle SI from 80.52 to 86.00. Two findings are
directly on point for us: heading is the attribute PCL improves least, and in their
offline-refinement study heading stability improves only for objects beyond 50 m, which
they read as near-range heading already being good enough.

**Uses position to infer orientation.** No, in the consistency term itself: agreement
between two augmented views says nothing about where the object went. As published it is
partly position-dependent, because the error is defined against a ground-truth box, and
the cross-frame pairing uses object IDs from labels.

**What it would take here.** The published form needs 3D ground truth, which we do not
have on WI DOT footage. A consistency-only variant is the adaptable part: take one frame,
produce a horizontally flipped and a slightly rotated copy, require the detector's yaw
predictions to agree after de-augmentation, and train that term alone with no yaw target.
This is attractive under the lab constraint precisely because it never tells the network
what the yaw should be, so no motion heading enters training at any point. It plugs into
`scripts/finetune/train_finetune.py` as an extra loss on the detection head. De-augmenting
BEVHeight's yaw under a horizontal image flip is not free: the flip changes the camera
geometry, so the intrinsic principal point and the extrinsic lateral axis both have to be
mirrored, and the output yaw negated about the mirror plane.

**Expected gain at 40 to 60 m.** Modest but real on the wobble; the paper's own ablation
says the heading component of the loss contributes least of the four, and their offline
study found heading the hardest attribute to stabilize. It reduces variance, not bias, so
if our 40 to 60 m errors are systematic rather than noisy it will not help.

**Cost.** 3 to 5 days to implement, plus GPU training runs on the lab server. Needs
training. High risk relative to the section C options.

---

## B. Video and streaming 3D detectors

All three below were developed and measured on nuScenes, a 1.5 m ego-vehicle rig with a
moving camera. Two shared caveats apply to all of them and are not repeated per paper.
First, none can be trained here, because DAIR-V2X-I and Rope3D are frame-sampled, not
video. Second, their reported orientation gains are explained by their own authors as a
by-product of velocity estimation, which is a position-derived signal entering the
feature that produces yaw. Under a strict reading of the constraint these are still
allowed, because the paper text calls them video 3D detection; under an honest reading
they are the closest thing in this survey to smuggling motion into the yaw.

### B1. BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection

Junjie Huang, Guan Huang, 2022. arXiv:2203.17054.
<https://arxiv.org/abs/2203.17054>
Local: `docs/papers/huang_2022_bevdet4d.pdf`

**Mechanism.** BEVDet4D is the minimal temporal upgrade to an LSS-style BEV detector,
which is the same family BEVHeight belongs to. It keeps the previous frame's BEV feature
map, aligns it to the current frame's BEV grid using ego motion, and concatenates the two
before the BEV encoder, so the head can compare a cell's feature now against the same
cell a moment ago. Velocity error drops by up to 62.9%, and orientation error drops 12.0%,
from 0.523 to 0.460 mAOE on the Tiny model. The paper states the reason outright: "the
orientation and velocity of the targets are strong-coupled." Compute overhead is
negligible.

**Uses position to infer orientation.** Partly, and the authors say so. The yaw gain comes
from the network seeing the object's BEV displacement between two frames. No motion
heading is written into the output, but the learned yaw is downstream of displacement.

**What it would take here.** Architecturally this is the easiest temporal upgrade for
BEVHeight, because the BEV grid and the voxel pooling already exist in `layers/` and
`ops/`, and our camera is static so the BEV alignment between frames is the identity, no
ego-motion warp required. That is genuinely simpler than the published version. The
blocker is training data: two consecutive labeled frames, which DAIR-V2X-I does not have.
Training on WI DOT pseudo-labels means training on motion headings, which the constraint
forbids as an output and which would also make the "independent signal" claim
indefensible.

**Expected gain at 40 to 60 m.** On paper the largest of any method here, roughly a 12%
relative orientation error reduction, and probably more on our data because our baseline
is far worse. In practice unreachable without sequential labels, and contaminated by
motion coupling if reached via pseudo-labels.

**Cost.** Weeks. GPU training required. Blocked on data.

### B2. StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection

Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, Xiangyu Zhang, ICCV 2023. arXiv:2303.11926.
<https://arxiv.org/abs/2303.11926>
Local: `docs/papers/wang_2023_streampetr.pdf`

**Mechanism.** Instead of carrying a dense BEV feature forward, StreamPETR carries a small
memory queue of **object queries** frame to frame, so long-term history propagates at the
object level at almost no cost. A motion-aware layer normalization conditions each
propagated query on the object's motion between frames, so a query arrives at the next
frame already shifted to where the object should be. It reaches 67.6 NDS on nuScenes test,
the first online multi-view method comparable with LiDAR, and the lightweight version runs
at 31.7 FPS. Orientation error is one of the metrics it improves over the single-frame
Focal-PETR baseline.

**Uses position to infer orientation.** Partly, and more explicitly than BEVDet4D:
motion-aware layer normalization takes the motion vector as its conditioning input, so
displacement is an input to the block that produces the query that produces the yaw.

**What it would take here.** A different detector architecture entirely. BEVHeight is
dense LSS, PETR is sparse query-based. This is a replacement, not an upgrade, plus the
same sequential-label blocker as B1.

**Expected gain at 40 to 60 m.** Not measurable from here. Its per-object memory is the
right idea for our problem, which is essentially "remember this car's yaw from when it was
closer", but that idea is available far more cheaply in section C without any training.

**Cost.** Weeks to months. GPU training. Not recommended.

### B3. Time Will Tell (SOLOFusion): New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection

Jinhyung Park, Chenfeng Xu, Shijia Yang, Kurt Keutzer, Kris Kitani, Masayoshi Tomizuka,
Wei Zhan, ICLR 2023. arXiv:2210.02443.
<https://arxiv.org/abs/2210.02443>
Local: `docs/papers/park_2022_solofusion.pdf`

**Mechanism.** SOLOFusion reframes temporal multi-view detection as temporal stereo
matching and introduces "localization potential", a measure of how easy depth estimation
is for a given baseline and resolution. It then combines short-term high-resolution stereo
with long-term low-resolution fusion over a long history of aligned BEV features, gaining
5.2 mAP and 3.7 NDS over prior work on nuScenes. Inference latency and memory are its known
weaknesses.

**Uses position to infer orientation.** Partly, and only incidentally. The method is about
depth from temporal parallax. Orientation is not a target.

**What it would take here.** Nothing usable. Temporal stereo requires camera motion to
create a baseline. Our camera is bolted to a pole and does not move, so the temporal
baseline is exactly zero and the entire mechanism degenerates. This is the clearest example
in this survey of a strong ego-vehicle method that does not transfer to a fixed roadside
camera at all, for a reason that has nothing to do with the 16 m height.

**Expected gain at 40 to 60 m.** Zero. Include in the note as a documented dead end so it
is not re-proposed.

**Cost.** Not applicable.

---

## C. Tracking-by-detection yaw refinement

This is where the practical options are. Everything in this section runs on
`NNN_pred.json` and `tracks.json` on the Mac, needs no training, and produces a yaw that is
a pure function of the detector's own per-frame yaw outputs.

### C1. AB3DMOT: 3D Multi-Object Tracking, A Baseline and New Evaluation Metrics

Weng, Wang, Held, Kitani, IROS 2020. arXiv:1907.03961 (extended), also arXiv:2008.08063 (ECCVW short).
<https://arxiv.org/abs/1907.03961>
Local: `docs/papers/weng_2020_ab3dmot.pdf` (extended version, contains the orientation correction section)

**Mechanism.** A 3D Kalman filter with a constant-velocity model plus Hungarian association
on 3D IoU. The state is (x, y, z, theta, l, w, h) plus linear velocity; **angular velocity is
deliberately excluded**, and their ablation shows adding it does not clearly help. The part
that matters here is the **orientation correction** step: when a detection is associated to a
track and the two orientations differ by more than pi/2, pi is added to the track's
orientation before the update, so the difference is always under pi/2. Their ablation
(variant d, "remove orientation correction") shows it improves every metric. This is the
mechanism that stops a 180 degree detector flip from destroying a track's filtered yaw.

**Uses position to infer orientation.** No, for the yaw value. Association uses 3D IoU,
which is positional, and the filtered yaw is smoothed by the constant-velocity dynamics,
but the yaw value itself is only ever a blend of detector yaws. This is the pattern the lab
constraint explicitly permits.

**What it would take here.** Nothing; this is already the tracker in
`scripts/tracking/run_ab3dmot.py`, and `tracks.json` already carries a filtered `yaw` field
alongside `vx`, `vy`. The immediate free experiment is to score the raw `NNN_pred.json` yaw
against the `tracks.json` yaw with the frozen scorer. That measures how much the existing
Kalman smoothing already buys and sets the bar every candidate must beat. It also verifies
that the orientation correction is actually enabled in our vendored copy, which has not
been checked.

**Expected gain at 40 to 60 m.** Whatever it is, we are already getting it. But the
measurement is the single cheapest thing in this document and it is a prerequisite for
interpreting every other result.

**Cost.** 1 to 2 hours, Mac, no training.

### C2. SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking

Ziqi Pang, Zhichao Li, Naiyan Wang, ECCVW 2022. arXiv:2111.09621.
<https://arxiv.org/abs/2111.09621>
Local: `docs/papers/pang_2021_simpletrack.pdf`

**Mechanism.** Decomposes 3D MOT into pre-processing, association, motion model and life
cycle, then fixes each. Key results: GIoU beats IoU for association because it degrades
gracefully when boxes do not overlap; a **two-stage association** that admits low-score
detections recovers real objects that a single score threshold drops; and on motion models,
the **Kalman filter fits better at high frame rates** while constant velocity is more robust
at low rates, because the filter needs dense observations. When a matched detection is of
poor quality, they prefer the motion prediction as the tracklet state rather than the box.

**Uses position to infer orientation.** No for the yaw value, same reasoning as C1.

**What it would take here.** We run at 30 Hz, which is the regime where the paper says the
Kalman filter is the right choice, so the current setup is already on the correct side of
that trade. The transferable pieces are the two-stage association (recovering low-score
detections lengthens tracks, which gives any yaw-consensus scheme more votes) and GIoU
association. Both are edits inside the vendored AB3DMOT driver, scored through the normal
ledger.

**Expected gain at 40 to 60 m.** Indirect. Longer, less fragmented tracks at long range are
worth more to C4 below than to the yaw directly. Treat as an enabler, not a candidate.

**Cost.** 1 day, Mac, no training.

### C3. Immortal Tracker: Tracklet Never Dies

Qitai Wang, Yuntao Chen, Ziqi Pang, Naiyan Wang, Zhaoxiang Zhang, 2021. arXiv:2111.13672.
<https://arxiv.org/abs/2111.13672>
Local: `docs/papers/wang_2021_immortaltracker.pdf`

**Mechanism.** Standard trackers kill a tracklet after a few unmatched frames, so an object
that is briefly occluded comes back as a new ID. ImmortalTracker never terminates: it keeps
predicting an unmatched tracklet's state indefinitely and re-associates when the object
reappears. This removes 96% of vehicle identity switches caused by premature termination,
with no learned parameters, at a mismatch ratio of order 1e-4 on Waymo.

**Uses position to infer orientation.** No. It changes life-cycle management only.

**What it would take here.** Our driver already exposes `--max-age`, and batch 1 of the
design doc sweeps it at 6, 15 and 30 frames. The immortal variant is the limit of that
sweep. The relevance is that a car crossing the 40 to 60 m band for 3 to 5 seconds at
highway speed should be one track, so that a good yaw observed at 25 m can be propagated
out to 55 m. If tracks break in that band, C4 has nothing to propagate.

**Expected gain at 40 to 60 m.** None alone. Potentially decisive as an enabler for C4, and
it is already half-scheduled in batch 1.

**Cost.** Hours, Mac, no training. Comes free with the existing max-age sweep.

### C4. DetZero: Rethinking Offboard 3D Object Detection with Long-term Sequential Point Clouds

Tao Ma, Xuemeng Yang, Hongbin Zhou, Xin Li, Botian Shi, Junjie Yan, Yikang Li, Liang He,
Yu Qiao, ICCV 2023. arXiv:2306.06023.
<https://arxiv.org/abs/2306.06023>
Local: `docs/papers/ma_2023_detzero.pdf`

**Mechanism.** DetZero is offboard, so it can see the whole sequence at once. It runs
detection and tracking first, then refines each object at the **tracklet level** with
attention: a geometry refining model reconstructs the object's shape from all its frames,
and a position refining model transforms every box in the tracklet into a shared local
coordinate system, encodes each frame as a query, and attends from those per-frame queries
to the whole track's features to predict per-frame corrections, including a **bin-based
heading angle** over 12 bins of 30 degrees. The architecture is exactly "let every frame of
a track see every other frame before committing to an answer", and the heading is
predicted as a classification over bins plus a residual rather than a raw regression.

**Uses position to infer orientation.** Partly. Nothing computes a heading from
displacement, but the refinement attends over the whole tracklet's positions and its local
coordinate system is anchored to a sampled box pose, so the track's positional history is
inside the receptive field of the heading prediction.

**What it would take here.** The published model is LiDAR only and cannot be ported; we have
no point clouds. What ports is the **idea**: a tracklet is the right unit for deciding yaw,
and bin-plus-residual is the right output form for an angle. Concretely that becomes a
non-learned version, a per-track circular consensus over the detector's yaws, which is
candidate 1 in the ranked list below. Also worth carrying over is the related finding from
A4's offline-refinement study that heading is the attribute offline refinement improves
least; do not expect tracklet refinement to be a free win.

**Expected gain at 40 to 60 m.** The learned version, not applicable. The distilled
non-learned version, potentially large, see the ranked list.

**Cost.** As published, not portable. As an idea, hours.

---

## D. Orientation representation and test-time methods

### D1. 3D Bounding Box Estimation Using Deep Learning and Geometry (MultiBin)

Mousavian, Anguelov, Flynn, Kosecka, CVPR 2017. arXiv:1612.00496.
<https://arxiv.org/abs/1612.00496>
Local: `docs/papers/mousavian_2017_multibin.pdf`

**Mechanism.** The origin of the discrete-continuous orientation head. Rather than
regressing a single angle with an L2 loss, which is broken at the wrap and biased toward
the mean of a bimodal distribution, MultiBin divides the angle range into overlapping bins,
classifies which bin the object falls in, and regresses a residual within the bin as a
(cos, sin) pair. The paper shows this "significantly outperforms the L2 loss" on KITTI
orientation. The same paper recovers translation from 2D-3D box constraints, which is the
ancestor of A1's yaw tuning.

**Uses position to infer orientation.** No. Single frame, appearance only.

**What it would take here.** BEVHeight's head already uses a (sin, cos) style rotation
output rather than raw angle regression, so the basic fix is present. What is not present
is a *bin classification*, which is the part that lets a model express "I am confident this
is one of two opposite headings but not which". Adding a coarse heading classification
branch beside the existing rotation regression, and exposing its per-bin scores in
`NNN_pred.json`, would give the temporal consensus in C4 a confidence weight instead of
treating every frame's yaw as equally trustworthy. That requires touching the head and
retraining.

**Expected gain at 40 to 60 m.** Indirect but potentially the highest-leverage change in
the whole document, because our core problem is that we cannot tell a good yaw from a bad
one within a track. A calibrated per-frame yaw confidence would turn the consensus from an
unweighted vote into a weighted one. But it is a training change, so it belongs in batch 3.

**Cost.** 1 week plus training. GPU.

### D2. MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection

Hongbin Lin, Yifan Zhang, Shuaicheng Niu, Shuguang Cui, Zhen Li, ECCV 2024. arXiv:2405.19682.
<https://arxiv.org/abs/2405.19682>
Local: `docs/papers/lin_2024_monotta.pdf`

**Mechanism.** Adapts a trained monocular 3D detector to a shifted test distribution using
only unlabeled test images. Two terms: a reliability-driven term that optimizes
high-confidence detections on the assumption they are correct, and a noise-guard negative
regularization that exploits the many low-score detections through negative learning so the
model does not collapse onto its own noise. Reported gains are very large in relative terms,
around 190% on KITTI and 198% on nuScenes corrupted or shifted settings, which reflects how
badly the unadapted baselines do rather than how good the adapted ones are.

**Uses position to infer orientation.** No. Entirely score-driven, and never touches
geometry targets.

**What it would take here.** It is a natural fit on paper, because our situation is exactly
"trained at 6 m, deployed at 16 m, no labels". It fine-tunes at test time on the WI DOT
frames using only confidence. But note what the objective actually optimizes: detection
confidence. There is no term that says anything about yaw. A model can become much more
confident about boxes whose yaw is still wrong.

**Expected gain at 40 to 60 m.** Low for heading specifically, and the headline percentages
should not be read as transferable. It may improve the detection rate at long range, which
helps the tracker, which helps C4. Recommend it to the domain-gap track, not the heading
track.

**Cost.** 2 to 3 days plus GPU. Needs gradient updates, so it is a training method even
though it is called test-time.

### D3. Test-time augmentation by flipping and yaw rotation (practice, not a single paper)

No single citation; the practice is documented across Waymo and nuScenes challenge reports.
Representative: the 1st place Waymo 3D detection solution, arXiv:2006.15505,
<https://arxiv.org/abs/2006.15505>, which uses double flip and yaw rotations at
0, +/-6.25, +/-12.5, +/-25 degrees, and weighted box fusion to merge the results. On
CenterPoint-Voxel/nuScenes val this kind of TTA moves NDS from 0.694 to 0.715 and mAP from
0.641 to 0.667.

**Mechanism.** Run the detector several times on geometrically transformed copies of the
same input, map every prediction back to the canonical frame, then fuse. Errors that depend
on the specific viewpoint or on a left-right asymmetry in the learned features partially
cancel.

**Uses position to infer orientation.** No. Every pass is a single frame.

**What it would take here.** A horizontal image flip is the cheap version, and it is the one
most likely to help, because a detector trained on DAIR's camera set may well have a
left-right bias that our sites do not share. Implementing it correctly means flipping the
image, mirroring the principal point in the intrinsics, mirroring the lateral axis of the
extrinsic, running inference, and then mirroring the output boxes back, which negates yaw
about the mirror plane. Point-cloud style yaw rotations do not apply directly to a fixed
monocular camera; the equivalent would be rotating the BEV grid, which changes what the
voxel pooling sees, and is a larger change. Fusion across the two passes should be a
circular mean of yaw, not an arithmetic one.

**Expected gain at 40 to 60 m.** Uncertain. TTA typically buys a couple of points where
baselines are already strong. Two passes is a small ensemble and the flip is the only
transform that is nearly free here, so this is worth one experiment but is not where the
10% gross-error population is going to be fixed.

**Cost.** 1 to 2 days including the geometry bookkeeping, doubles inference time, no
training.

---

## Ranked candidates for this repo

Cheapest first among those with a real expected gain. Every one of these produces a yaw
that is a function only of BEVHeight's own per-frame yaw outputs or of single-frame
appearance, so none of them violates the independence constraint. Each becomes one
`scripts/orientation/candidates/<name>.py` behind the frozen scorer.

**1. Baseline separation: score raw detection yaw against tracked yaw.**
Before any candidate, run the frozen scorer twice, once on the yaw in `NNN_pred.json` and
once on the yaw in `tracks.json`, on all eight held-out clips. AB3DMOT already Kalman
filters yaw and, if the orientation correction described in the extended AB3DMOT paper is
enabled in our vendored copy, already handles 180 degree detector flips at association time.
So part of the smoothing gain may already be banked, and every later candidate needs to beat
the better of these two numbers rather than the raw detections. While you are in there,
confirm by reading the vendored tracker that orientation correction is on; if it is off,
turning it on is a one-line change with a documented ablation behind it. Roughly 2 hours,
Mac, no training, and it is a prerequisite rather than an option.

**2. Per-track circular yaw consensus, weighted by score and by range.**
This is DetZero's tracklet-level refinement (C4) reduced to its non-learned core, using
ImmortalTracker's and SimpleTrack's insight (C2, C3) that the value of a long unbroken track
is that it lets a confident observation travel. For each track in `tracks.json`, collect the
per-frame detector yaws, fold them modulo 180 degrees so front-back flips do not corrupt the
axis estimate, take a weighted circular mean or median with weights rising for high `score`
and for short range (a car at 25 m is where the appearance cue is strongest), and assign that
single axis to every frame of the track. Then resolve the 180 degree ambiguity once per track
by majority vote over the raw per-frame yaws relative to that axis, not by motion. Optionally
allow the axis to drift with a long smoothing window so genuine turns and lane changes
survive, which matters because the self-training pilot already flagged rising heading
concentration on intersection scenes as a risk. Expect the largest movement of anything in
this list on both the folded primary and the flip fraction, because the 10% gross errors are
per-frame events scattered along tracks whose other frames are good, and expect the gain to
concentrate at 40 to 60 m for exactly that reason. Roughly 3 to 4 hours, Mac, no training.

**3. Offline forward-backward smoothing of the track yaw.**
AB3DMOT's Kalman filter is causal, so a frame's yaw is informed only by its past. Offline we
have the whole clip, so run a Rauch-Tung-Striebel smoother over the yaw state, or simply run
the existing filter forward and backward and blend, applying the AB3DMOT orientation
correction (add pi when the difference exceeds pi/2) before every update in both directions.
This is strictly more information than candidate 1 and complements candidate 2: consensus
gives a track one stable axis, smoothing lets the axis change smoothly when the vehicle
actually turns. Note AB3DMOT's own ablation found angular velocity in the state did not
clearly help, so do not add it; smooth the angle, not its rate. Roughly 1 day, Mac, no
training. Keep it as a separate ledger row from candidate 2 so their contributions are
separable.

**4. Test-time horizontal-flip averaging of the detector yaw.**
Run `run_bevheight_generic.py` a second time on horizontally flipped frames with a mirrored
principal point and mirrored extrinsic lateral axis, map the output boxes and yaws back, match
them to the unflipped detections by 3D IoU, and fuse the two yaws with a circular mean. This
is standard challenge practice (D3) and it tests a specific hypothesis worth testing on its
own: that part of our yaw error is a left-right asymmetry learned from DAIR's camera set,
which our two view directions would then express differently. If that hypothesis is right, the
score will improve more on one view direction than the other, which is diagnostic information
even if the candidate is rejected. Roughly 1 to 2 days, doubles detection time, needs the
server GPU for a comfortable turnaround but will run on the Mac, no training.

**5. Yaw tuning against 2D detections (Pro3D v2).**
Run an off-the-shelf 2D car detector on the frames, associate each 3D box to a 2D box by
projecting the cuboid and taking best IoU, then search yaw over a 1 degree grid for the value
that maximizes IoU between the reprojected cuboid extent and the 2D box, keeping position and
dimensions fixed. This is the only candidate here that introduces a genuinely independent
appearance-based orientation signal rather than reorganizing BEVHeight's own outputs, which
makes it the most interesting one scientifically and the right thing to fuse with candidate 2
if both work. Be realistic about its ceiling: the objective is symmetric under a 180 degree
flip so it cannot touch the flip fraction, and at 15 to 22 degrees of elevation the projected
extent changes slowly with yaw so the optimum is poorly conditioned in exactly the 40 to 60 m
band. Its published gain was 2 to 3 AOS points on DAIR-V2X-I with BEVHeight as the baseline,
on a baseline that was already good. Roughly 1 to 2 days, GPU convenient for the 2D pass but
not required, no training.

### Explicitly not recommended now

- **BEVDet4D, StreamPETR, any trained temporal detector.** Blocked on sequential labeled
  roadside data, and their orientation gains are attributed by their own authors to coupling
  with velocity, which is the signal we are required to keep out.
- **SOLOFusion and temporal stereo.** The camera does not move, so the temporal baseline is
  zero and the mechanism does not exist for us.
- **MonoTTA.** Optimizes confidence, not orientation. Belongs to the domain-gap track.
- **MOSE.** Not an orientation method; its extrinsic augmentation idea belongs to the
  domain-gap track.

### Open question this literature does not answer

Nothing found explains an error band at 40 to 60 m. Elevation angle and pixel size both fall
monotonically with range, so neither produces a band. The nearest relevant observation is in
A4, where offline refinement improved heading stability only beyond 50 m and the authors
concluded that near-range heading was already good enough. If candidate 2 works and the gain
is concentrated in the band, that is evidence the band is a per-frame noise phenomenon. If
candidate 2 does not help there, the band is systematic and the answer is in batch 3, not in
post-processing.
