# Orientation research: merged candidate list

Date: 2026-09-15. Merges the three arm notes (`research-temporal-yaw-2026-09.md`,
`research-2d-cue-yaw-2026-09.md`, `research-domain-gap-2026-09.md`) into one ranked
list behind the frozen scorer in
`docs/superpowers/specs/2026-09-15-orientation-research-loop-design.md`.

Measured state that overrides the three source notes where they conflict: the scorer
grades **raw detection yaw**, baseline mean median folded error 8.88 deg over 5 Todd
Drive clips, worst in the 40 to 60 m band (up to 16.9 deg); BEVHeight reports a fixed
absolute heading near 0 rad for **both** traffic streams, so about 51% of boxes point
backwards, 5 to 15% flipped for one stream and 81 to 92% for the other, stable within
a track. GPS is evaluation only and may not feed any pipeline step.

## (a) What the literature says about our two symptoms

1. Folded axis error is the symptom every roadside paper reports under a camera change, and none of them reports it as a band.
2. The 40 to 60 m concentration is unexplained: elevation angle and pixel size both fall monotonically with range, so neither produces a band.
3. The nearest observation is Wang 2024 (Stability Index), where offline refinement improved heading only beyond 50 m and near range was already good enough.
4. The mechanical story the domain-gap literature supports is roof dominance: at 16 m a car at 50 m is seen at 17.7 deg depression against 6.8 deg in DAIR, so the side panels that carry yaw are foreshortened while the roof fills the box.
5. SGV3D quantifies the cost of a camera change on the *same* dataset: BEVHeight 65.77 homologous against 7.86 heterologous, so a 2.7x height change is outside anything the architecture was shown to survive.
6. Sign collapse is the sharper symptom and the literature explains its shape, not its presence: a single-value rotation regression has no way to express "this axis, unknown direction", which is what MultiBin's bin classification was invented for.
7. A near-constant absolute heading across both traffic streams is a collapse onto the training prior, not per-frame noise, so nothing that averages, votes or smooths the detector's own yaws can recover the missing bit.
8. Every published fix for direction is therefore either a training change (MultiBin bins, prediction consistency, weakly supervised fine-tuning) or a second model whose rotation head has no discrete front-back branch to collapse (Cube R-CNN's 6D rotation).
9. Projection-based objectives cannot help: Pro3D yaw tuning, 2.5D footprint parallelograms and 2D box consistency are all symmetric under a 180 deg flip.
10. Nothing in any of the three arms recovers direction from appearance alone with released roadside weights, which is why the one-motion-bit request and the DAIR-V2X-I collapse check are both on the critical path.

## (b) Ranked candidates

Cheapest with a real expected gain first. Each becomes
`scripts/orientation/candidates/<name>.py`. "Position independence" follows the lab
constraint: yes = yaw is a function only of appearance or of the detector's own yaws;
one bit = a single binary direction taken from motion, reference only, pending lab
approval.

### 1. `yaw_track_axis_consensus`

- Changes: post-processing of `NNN_pred.json` yaw. No rerun.
- Inputs: per-frame detector yaw, `score`, range, `tracks.json` for membership only.
- Position independence: yes. Association selects which yaws to combine, not their value.
- Folded: the largest expected movement in this list. The gross per-frame errors are scattered along tracks whose other frames are good, so fold mod 180, take a weighted circular median with weights rising for high score and short range, and assign one axis per track. Optionally let the axis drift over a long window so genuine turns survive.
- Sign: none. The source note proposed resolving the 180 deg ambiguity by majority vote over raw yaws; the measured collapse makes that vote unanimous and wrong, so the candidate emits an axis only and inherits the detector's sign.
- Cost: 3 to 4 hours, Mac.
- Source: DetZero (arXiv:2306.06023) tracklet refinement reduced to its non-learned core; SimpleTrack (arXiv:2111.09621) and ImmortalTracker (arXiv:2111.13672) for why longer tracks give it more votes.

### 2. `yaw_edge_align`

- Changes: post-processing of `NNN_pred.json` yaw, reading frames. No rerun.
- Inputs: frames, detections, `calibration_used.json` (K, lidar2cam, road plane z).
- Position independence: yes.
- Folded: good inside 40 m where the box projects to 60 to 125 px and the contact edges are sharp. Keep x, y, z, l, w, h fixed, sweep yaw at 2 deg steps, score each by mean Sobel magnitude along the projected wireframe with the four ground-contact edges weighted about 3x, parabola-interpolate the argmax, record best-to-second-best ratio as a confidence. Flattens past 70 m, so gate by range and fall back to the detector yaw.
- Sign: none. The gradient landscape is near-symmetric under a flip; score raw and folded separately so a folded win cannot hide a worse flip rate.
- Cost: half a day, Mac, no new dependencies.
- Source: Pro3D v2 yaw tuning (arXiv:2404.01064) with image gradients replacing the 2D-box IoU objective.

### 3. `bn_stats_recalib`

- Changes: detection rerun, forward passes only, no gradients.
- Inputs: a few hundred of our frames; the checkpoint unchanged.
- Position independence: yes.
- Folded: uncertain, plausibly small. Put BN layers in train mode with all weights frozen, re-estimate every running mean and variance including the 27-dim camera-parameter BN in `HeightNet.forward`, switch back to eval. That BN was fitted on fx 2183 and 2758 and is being fed 1493 to 1556, so its modulation is running several standard deviations outside its trained range.
- Sign: the cheapest test of whether the collapse is an out-of-distribution statistics artifact rather than a learned prior. If the fixed near-0 heading moves at all, that is informative out of proportion to the cost.
- Cost: half a day. Mac CPU is viable because there is no backward pass.
- Source: MonoTTA (arXiv:2405.19682), the free half; STMono3D (arXiv:2204.11590) for the depth-shift framing.
- Guard: watch `cars_per_frame`, BN recalibration shifts the score distribution and silently moves the operating point.

### 4. `ckpt_rope3d`

- Changes: detection rerun with a different checkpoint and config.
- Inputs: `BEVHeight_R50_128_102.4_72.45_39_epochs.ckpt` (925 MB) with `experiments/rope3d/bev_height_lss_r50_864_1536_128x128_102.py`. Head is 180 channels not 90, classes are Car and Bus.
- Position independence: yes.
- Folded: a lottery ticket. Rope3D spans 6.1 to 8.1 m heights, 9 to 15 deg pitch and 2100 to 3000 px focal across many intersections, against five intersections at one height, and its height bins are wider and finer. 16 m is still outside the envelope.
- Sign: the only cheap way to learn whether the collapse is specific to the DAIR checkpoint or general to the architecture. Pairs directly with the DAIR-V2X-I collapse check already in progress.
- Cost: an hour of download plus one rerun per clip. Lab GPU at night, or Mac CPU at minutes per clip.
- Source: Rope3D (arXiv:2203.13608); checkpoint from the upstream BEVHeight repo.

### 5. `virtual_camera_rescale`

- Changes: detection rerun on rescaled crops.
- Inputs: frames, K, extrinsic. Centre-crop about 1310 px wide, upscale to the network input, adjust the intrinsic and the IDA matrix consistently, tile 2 to 3 overlapping crops for the 65 deg FOV, merge with the existing NMS. Optionally fold the pitch-correcting homography into the same warp, which is exact because pitch is a pure rotation.
- Position independence: yes.
- Folded: the most plausible cheap win from the domain-gap arm. Objects in the 40 to 60 m band go from 1.5x to 1.9x smaller than trained back to trained scale, and the yaw head is the part most sensitive to apparent size. It cannot touch the 16 m versus 6 m viewpoint elevation, so it will not close the gap alone.
- Sign: possible but unproven. If the collapse is driven by apparent scale rather than depression angle, this is where it shows.
- Cost: 1 to 2 days, Mac CPU feasible, GPU comfortable.
- Source: MoVi-3D (arXiv:1912.08035), STMono3D geometry-aligned multi-scale, DG-BEV scale-invariant depth (arXiv:2303.01686).

### 6. `yaw_roadside25d`

- Changes: post-processing. An external model runs on frames; our detections keep their position and dimensions.
- Inputs: frames, released Polley weights, K and lidar2cam to lift the predicted footprint parallelogram to the road plane at z = -1.73, BEV footprint IoU to associate.
- Position independence: yes, and the authors reject trajectory-derived orientation explicitly, for the same reason the lab does.
- Folded: the single published method whose output representation is exactly the quantity we want. Plausibly 5 to 10 deg median inside 50 m if the transfer holds. Their cameras sit around 7 m, so we inherit a gap of the same kind as BEVHeight's from a different model, and their scenes are intersection-scale rather than 90 m highway. Report per-range-bin numbers before anyone gets attached to it.
- Sign: none. A footprint parallelogram carries an axis, not a direction.
- Cost: 1 to 2 days, weights only, no training.
- Source: Polley et al., 2.5D object detection for intelligent roadside infrastructure (arXiv:2507.03564).

### 7. `yaw_cubercnn`

- Changes: post-processing. External model on frames, association to existing detections by BEV IoU.
- Inputs: frames, K, released Omni3D outdoor weights, lidar2cam for the road plane normal.
- Position independence: yes. Only the viewing-ray azimuth enters, which is the correct and necessary use.
- Folded: probably no better than BEVHeight's median, guess 15 to 25 deg. Its value is decorrelation and a per-prediction uncertainty that gives a principled fusion weight with candidate 1.
- Sign: the only appearance-based candidate here with a real chance at it. The 6D continuous rotation parameterization has no discrete front-back branch to collapse onto a training prior, so its direction errors should be independent of ours. Only fuse after both are scored alone on all 8 held-out clips and only if the ledger shows the flips land on different frames.
- Cost: 2 to 3 days, most of it the allocentric-to-global bookkeeping. Write the synthetic round-trip check first: place a box of known yaw at several image positions spanning the full 64 deg FOV, forward-project, invert, assert the yaw returns.
- Source: Omni3D / Cube R-CNN (arXiv:2207.10660); MultiBin (arXiv:1612.00496) for why the conversion is where this breaks.

### 8. `finetune_weak2d_ignore`

- Changes: training, lab GPU at night.
- Inputs: our frames, an off-the-shelf 2D detector's boxes for every vehicle, track-derived 3D pseudo-labels with confidence weights, ignore regions. No GPS at any tier: the repo rule forbids it as a training input, so the source note's "GPS vehicle at high weight" tier is dropped.
- Position independence: yes for the axis. One bit for direction, since the only available sign source is motion heading and that stays pending lab approval. Without that bit this candidate improves the axis and leaves the collapse intact.
- Folded: the only candidate that addresses the 16 m viewpoint gap rather than working around it. Unfreeze the height branch, add DG-BEV homography perspective augmentation so we do not re-overfit to one camera pose.
- Sign: the only structural fix available. Add a coarse heading bin classification beside the rotation regression and expose per-bin scores in `NNN_pred.json`, so the model can say "this axis, direction uncertain" instead of defaulting to 0 rad, and so candidate 1 gets a weighted vote instead of an unweighted one. Note that the 2D projection-consistency term itself is blind to a flip; it constrains the axis and the footprint, nothing else.
- Cost: 1 to 2 weeks, server. Gate every run on `cars_per_frame` not falling and DAIR heading concentration not rising.
- Source: WARM-3D (arXiv:2407.20818), STMono3D quality-aware supervision, positive-unlabeled detection (arXiv:2002.04672), MultiBin (arXiv:1612.00496), SGV3D (arXiv:2401.16110) as the next step above this one.

## (c) Verdicts on the two lab proposals

**Plane masking (overpasses and other ground planes).** The concern is real and the
instrument is wrong. `height2localtion` intersects every ray with one plane from a
single scalar `reference_heights`, so a vehicle on a 5 m overpass seen from 16 m lands
at about 1.45x its true range, and nothing in BEVHeight can express two ground heights.
But deleting pixels puts a large out-of-distribution rectangle into a fully
convolutional BatchNorm backbone with camera-conditioned SE modulation, which perturbs
detections outside the mask as well as inside, and we have already been bitten once by
a change that improved heading by detecting nothing. It also cannot move either number
we care about, since the 40 to 60 m yaw failures and the depth bias are both measured
on main-road vehicles on the main plane, and it costs recall on exactly the ramp merges
the AV-versus-human classifier wants. Cheapest first: count what fraction of detections
and tracks fall outside the main-road BEV polygon before building anything; if it
matters, filter in BEV rather than in the image, which is post-hoc, reversible and
costs no recall; if off-plane vehicles must be detected correctly, give
`height2localtion` a per-region reference height built empirically from tracked
vehicles' wheel-contact points, which is MonoGAE's idea minus the learning and MOSE's
minus the transformer, needs no retraining, and can be validated against the existing
0.08 to 0.11 m box-bottom residuals. Only adopt SGV3D's learned foreground gating if we
are retraining anyway, and never hand-mask.

**Single-vehicle masked loss.** The instinct is right and the formulation reproduces a
known failure. Standard detection loss is positive-negative, so labelling only the
instrumented vehicle makes every other car in every frame an explicit negative: about 6
to 7 false negatives per frame at 30 Hz, roughly 90% missingness, and Yang, Liang and
Carin show recall degrades in proportion. The model's cheapest way to satisfy that loss
is to stop detecting cars, which is the failure already recorded in
`docs/server-finetune-setup.md`. The supervision is also thin and biased: 300 frames of
one car is a handful of independent observations, its yaw is nearly constant so the
model learns "point along the road", and the head-only pilot already leaked that way
(DAIR heading concentration 0.783 to 0.838). It also sits in a specific lane and range
profile, which is the worst possible sample for a range-dependent bias. Separately, the
repo rule settles it: GPS is evaluation only, so the GPS vehicle cannot be a training
target at all. Do it properly instead: never let unlabeled vehicles be negatives, using
the DAIR format's existing `DontCare` ignore regions, and supervise all traffic weakly
with cheap 2D boxes rather than one object strongly. That is candidate 8.

## (d) Rejected and why

- **GPS depth recalibration** (domain-gap candidate 1). Repo rule: GPS never feeds calibration or any pipeline step. Permitted only as an evaluation-side analysis showing the bias transfers between clips.
- **Baseline separation, raw yaw against tracked yaw** (temporal candidate 1). Already done. The scorer grades raw detection yaw and the baseline is 8.88 deg mean median folded.
- **Flip decision by per-track majority vote over raw yaws.** Killed by the measured collapse: the detector's absolute heading is fixed near 0 rad and stable within a track, so the vote is unanimous and wrong. Only motion (one bit) or retraining can recover the sign.
- **Classic surveillance line** (Dubska, Sochor, Kocur). Vehicle orientation is the road's vanishing-point direction, aggregated from how traffic moves. Trajectory-derived, and structurally incapable of expressing a lane change, which is the signal the classifier needs.
- **BEVDet4D, StreamPETR, any trained temporal detector.** Blocked on sequential labeled roadside data, and their orientation gains are attributed by their own authors to coupling with velocity.
- **SOLOFusion and temporal stereo.** The camera does not move, so the temporal baseline is zero and the mechanism does not exist here.
- **BEVHeight++, CoBEV, HeightFormer, BEVSpread.** Homologous-only evidence, full retrain, no loadable roadside checkpoint, and SGV3D measures BEVHeight++ at 8.18 heterologous against BEVHeight's 7.86. Architecture is not our axis.
- **Full MonoTTA.** Optimizes detection confidence, not orientation. Only its BN half survives, as candidate 3.
- **MOSE as a model.** Not an orientation method; its scene-cue idea is borrowed in the plane-masking verdict instead.
- **Mask-to-OBB with any segmenter, YOLO11-OBB, Depth Anything V2, keypoint methods (ApolloCar3D, Occlusion-Net).** Resolution and smear. A warped silhouette stretches 5.5 m radially at 60 m, so its long axis is the viewing direction, not the vehicle; keypoint methods need 2 px accuracy on a 44 x 29 px footprint.
- **MonoFlex, GUPNet, MonoCon.** KITTI-only weights at 1.65 m, so the domain gap is larger than BEVHeight's and in the same direction, which makes the errors correlated rather than complementary.

Parked rather than rejected, revisit if candidates 1 to 8 stall: `yaw_contact_line`
(SAM2 lower-contour fit, 2 to 3 days, shadow lock-on is the hazard), `yaw_bev_rbox`
(Zhu et al. tailed r-box trained on renders at our exact camera pose, 2 to 3 weeks,
no human labels at any stage), and SGV3D's rectify-and-paste data generation, which has
the highest ceiling of anything read and costs weeks.
