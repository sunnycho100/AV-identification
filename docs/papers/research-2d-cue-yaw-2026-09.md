# Vehicle orientation from 2D image cues with a known ground plane

Date: 2026-09-15. Scope: a second, appearance-based yaw source for the roadside
pipeline, independent of the trajectory. Motion heading stays the evaluation
reference only (`scripts/evaluation/score_heading.py`), never the output.

Site: fixed WI DOT highway cameras, 1920x1080 at 30 Hz, height 16.26 m, pitch
about 18 to 19 degrees, horizontal FOV 64 degrees (fx 1525 to 1556 from
AnyCalib). Road plane known at z = -1.73 m in the detector's output frame.
Inputs available per clip: frames `data/camera-data/<CLIP>/frames_all/NNN.jpg`,
`outputs/object_detection/camera-data/<CLIP>_phase1/calibration_used.json`
(K, lidar2cam, `road_plane_z_in_output`), detections `NNN_pred.json`
(x, y, z, l, w, h, yaw, score, class_name).

---

## 0. The geometry that bounds every method here

Everything below is limited by the same thing: at long range this camera has
almost no depression angle, so the ground footprint of a car collapses along
the viewing direction while the vehicle body smears away from the camera.

With f = 1525 px, h = 16.26 m, ground range R:

| R (m) | depression (deg) | px per m of range | car length 4.5 m, along range (px) | car width 1.8 m, across (px) | roof smear of a 1.5 m tall car when projected to the road (m) | yaw error from a 2 px error on the contact line (deg) |
|---|---|---|---|---|---|---|
| 25 | 33.0 | 27.9 | 125 | 92 | 2.3 | 0.9 |
| 40 | 22.1 | 13.3 | 60 | 64 | 3.7 | 1.9 |
| 60 | 15.2 | 6.4 | 29 | 44 | 5.5 | 4.0 |
| 90 | 10.2 | 3.0 | 13 | 30 | 8.5 | 8.5 |

Three consequences, and they decide the ranking at the end.

**The silhouette mask is not the footprint.** Warp a whole vehicle mask through
the road homography and the body stretches radially by 1.5 / tan(depression):
2.3 m at 25 m, 5.5 m at 60 m, 8.5 m at 90 m. That smear is longer than the car
is wide and, past 50 m, comparable to the car's own length. A minimum-area
rectangle over the warped mask therefore returns a rectangle whose long axis
points at the camera, not along the vehicle. Naive mask-to-OBB via homography
is not a yaw estimator on this footage, it is a range-direction estimator. Zhu
et al. hit exactly this and invented the "tailed r-box" to work around it (see
below). Only the **lower** contour of the mask, the tire contact line and the
shadow boundary, carries yaw.

**Beyond about 70 m nothing appearance-based will be good.** A 4.5 m car spans
13 px along the range axis at 90 m. Even a perfect contact line localized to
2 px gives 8.5 degrees of yaw error, and real masks on a 30 px wide blob are
not good to 2 px. Realistic expectation: appearance yaw is useful out to
roughly 60 m and is noise past 75 m. The known BEVHeight failure band (40 to
60 m) is exactly the last band where a 2D-cue method can still help, which is
the good news.

**Egocentric versus allocentric matters more here than in KITTI.** Every
method that regresses orientation from a cropped patch learns the *allocentric*
(observation) angle, the vehicle's pose relative to the viewing ray, because
the crop cannot see where in the frame it came from. Global yaw is
`yaw = alpha + atan2(ray)`. With a 64 degree FOV the horizontal ray azimuth
swings plus or minus 32 degrees across the frame, so getting this conversion
wrong is itself a 30 degree error at the image edges. At 16 m there is a second
term KITTI does not have: the ray also has 10 to 33 degrees of *elevation*, so
the vehicle's own vertical axis is not parallel to the image's vertical axis
and the "rotate about the world z axis" assumption baked into most monocular
heads is violated. A detector trained at 6 m and 42 degrees FOV (DAIR-V2X-I)
saw 8 to 12 degrees of depression at these ranges; we present it with 15 to 33.
That is the most plausible mechanical cause of the observed flips, and it is
also why a second appearance model trained on the same ego-centric or low-mount
data is not guaranteed to be decorrelated from BEVHeight's errors.

**Negative finding, stated up front.** The entire classic surveillance line
(Dubska, Sochor, Kocur) builds the vehicle box on the *scene's* three
orthogonal vanishing points, where VP1 is by construction the direction the
traffic moves. Those methods therefore assign every vehicle the road's
direction. They are excellent for speed measurement and useless for us: they
cannot represent a vehicle yawed relative to its lane, which is precisely the
lane-change signal the AV-versus-human classifier needs, and their output is
not independent in any interesting sense. Read them for the calibration and
the box-fitting machinery, do not adopt their orientation model.

---

## Papers

### 1. Sochor, Juranek, Herout (2017), Traffic surveillance camera calibration by 3D model bounding box alignment for accurate vehicle speed measurement
`sochor_2017_calib-3dbox-align.pdf`, https://arxiv.org/abs/1702.06451 (CVIU 2017)

Extends Dubska et al., *Fully Automatic Roadside Camera Calibration for Traffic
Surveillance* (IEEE T-ITS 16(3), 2015, not on arXiv), which recovers two road
vanishing points from tracked feature points using a cascaded Hough transform
in parallel coordinates. This paper adds the missing scene scale by rendering
3D CAD models of common vehicles and matching their rendered bounding boxes to
the boxes detected in the image, replacing the earlier scale heuristics.
Vehicle boxes are then constructed as the axis-aligned cuboid in the VP frame
that circumscribes the vehicle's foreground blob.

- Inputs: long video for VP accumulation, foreground blobs, no calibration needed.
- Uses position to infer orientation: yes, structurally. Orientation is the road direction from VP1, derived from aggregated vehicle motion.
- What it would take here: nothing new. `scripts/calibration/site_extrinsic.py` already solves the lane VP, so the road direction per clip is in hand.
- Expected accuracy 25 to 90 m: the road direction itself is good to about 1 degree. As a *vehicle* yaw it is wrong by exactly the lane-relative yaw we want to measure.
- Cost: zero, already have it. Value: a prior and a sanity bound, not a candidate.

### 2. Sochor, Spanhel, Herout (2017), BoxCars: fine-grained recognition of vehicles using 3D bounding boxes in traffic surveillance
`sochor_2017_boxcars.pdf`, https://arxiv.org/abs/1703.00686 (IEEE T-ITS 2019)

Uses the VP-derived 3D box to "unpack" a vehicle image into a canonical
viewpoint before classification, plus a CNN branch that *estimates* the 3D box
when it cannot be constructed geometrically. The relevant artefact is the
dataset: BoxCars116k, 116k surveillance-camera vehicle images with 3D box
annotations from many camera poses including high overhead mounts, which is the
only large public label set for vehicles seen from a roadside vantage.

- Inputs: vehicle crop, 2D box. Orientation output is the box in the VP frame.
- Uses position to infer orientation: yes for the ground-truth construction, no for the estimation branch (which infers the box from the crop alone).
- What it would take here: BoxCars116k is the natural pretraining corpus for any appearance-yaw head we train ourselves, and its viewpoint range partly covers 16 m mounts. Labels are road-relative though, so a head trained on it inherits limitation (1).
- Expected accuracy: not directly applicable.
- Cost: low to obtain, moderate to exploit. Useful mainly as training data.

### 3. Kocur, Ftacnik (2020), Detection of 3D bounding boxes of vehicles using perspective transformation for accurate speed measurement
`kocur_2020_perspective-3dbox.pdf`, https://arxiv.org/abs/2003.13137 (MVA 2020), code at https://github.com/kocurvik/BCS_results

Warps the image with a perspective transform built from two of the three scene
vanishing points, so that in the warped frame two of the box's three axis
families become image-axis-parallel. The 3D box then reduces to an ordinary
2D box plus one scalar, and a standard 2D detector (CenterNet-style) regresses
it. Trained on BoxCars116k, evaluated on BrnoCompSpeed. It is the cleanest
demonstration that a ground homography plus a plain 2D detector recovers 3D
vehicle geometry cheaply.

- Inputs: frame, two vanishing points. No intrinsics, no metric extrinsics for the box itself.
- Uses position to infer orientation: yes, same structural issue. Section 3 states outright that vehicle orientation is taken to be tied to the VP positions.
- What it would take here: the VP solver exists, the rectification is 20 lines of OpenCV, but retraining on BoxCars116k plus accepting road-direction-only yaw. About a week for an answer we already know.
- Expected accuracy: box *extent* would be good to 50 m, yaw is the road direction.
- Cost: medium. Not recommended as a yaw source. Worth reading for the rectification trick, which is reusable for a *free* yaw head (candidate 5).

### 4. Zhu, Zhang, Zhong, Lu, Peng, Lenneman (2021), Monocular 3D vehicle detection using uncalibrated traffic cameras through homography
`zhu_2021_trafcam-homography-rbox.pdf`, https://arxiv.org/abs/2103.15293 (IROS 2021), code at https://github.com/minghanz/trafcam_3d (GPL-3.0)

The closest published match to what we want. The road-to-image homography is
the only calibration used, and it is recoverable from satellite imagery. The
image is inverse-perspective-mapped to a BEV raster and a detector regresses
**rotated** boxes there, so each vehicle gets an arbitrary yaw rather than the
road direction. It explicitly identifies the smear problem from section 0: an
r-box fitted to the visible BEV pixels can be disjoint from the vehicle's
footprint, so they add a "tail", the BEV vector from the footprint centre to
the roof centre, which both disambiguates the smear and gives the network a
reason to learn where the footprint actually is. Angle loss is `sin^2(residual)`
over a plus or minus 45 degree residual window, a detail worth copying. Trained
entirely on synthetic data, CARLA renders plus Blender vehicles composited onto
real empty-road backgrounds, and shown to transfer to unseen real cameras.

- Inputs: frame plus road homography. We have both, and a better homography than theirs (metric, from K and lidar2cam).
- Uses position to infer orientation: no. Yaw comes from the warped appearance.
- What it would take here: the released code is a training repo, no roadside-16 m weights. Realistically we render CARLA or Blender clips at our camera pose (h = 16.26 m, pitch 18.5 deg, f = 1525) and train the BEV r-box head on the lab GPU. No human labels needed at any point, which is the attraction. Two to three weeks.
- Expected accuracy 25 to 90 m: their IPM raster at our geometry gives a car about 125 x 92 px at 25 m and 29 x 44 px at 60 m before the tail correction. Expect single-digit degree median inside 50 m, degrading to 15 degrees plus past 75 m. Should beat BEVHeight in the 40 to 60 m band.
- Cost: high, but the only method here designed for exactly our camera class with an arbitrary per-vehicle yaw and no label requirement.

### 5. Polley, Koch, Zofka, Zoellner (2025), 2.5D object detection for intelligent roadside infrastructure
`polley_2025_roadside-2p5d.pdf`, https://arxiv.org/abs/2507.03564 (ITSC 2025), weights and inference code at https://gitlab.kit.edu/kit/aifb/ATKS/public/digit4taf/2.5d-object-detection

Drops height entirely and predicts each vehicle's **ground-plane footprint as a
parallelogram in image coordinates**, regressing three vertices and reflecting
two of them about the predicted centre to close the shape. That representation
is exactly the quantity we need: back-project the parallelogram's four image
points onto the known road plane and read the yaw off the long edge. The paper
argues that inferring orientation from past detections fails for stationary
objects and that geometric approaches need precise calibration, so they learn
the footprint directly. Trained on a mix of real infrastructure footage and
synthetic scenes, and reports cross-viewpoint generalization and real-time
speed. No calibration is required at inference.

- Inputs: RGB frame only. Calibration is needed by us afterwards, to lift the parallelogram to metric yaw, and we have it.
- Uses position to infer orientation: no, and the paper explicitly rejects doing so.
- What it would take here: clone, run inference on `frames_all/`, back-project through `K` and `lidar2cam` at `z = -1.73`, associate to the existing detections by footprint IoU, emit `{track_id: {frame: yaw}}`. One to two days, no training. The main risk is that their infrastructure cameras sit around 7 m, so our 16 m mount is again out of domain, and their vehicle scale at 60 to 90 m may be below what their heads were trained on.
- Expected accuracy 25 to 90 m: plausibly 5 to 10 degrees median inside 50 m if the domain transfer holds, unusable past 75 m. Unknown until run, which is the point of running it first.
- Cost: low. Highest value-per-hour of anything in this list that needs no training.

### 6. Mousavian, Anguelov, Flynn, Kosecka (2017), 3D bounding box estimation using deep learning and geometry
`mousavian_2017_multibin.pdf`, https://arxiv.org/abs/1612.00496 (CVPR 2017)

The reference for the allocentric question. The network regresses the
observation angle from the 2D crop using the MultiBin head, a coarse angular
bin classification plus a within-bin sine and cosine residual, and only then
converts to global yaw using the crop's position in the image and the
intrinsics. The paper is explicit that regressing global yaw from a crop is
ill-posed because two vehicles with identical appearance at different image
positions have different global yaw. Dimensions are regressed as residuals from
class means, and the translation is solved from the 2D box tangency
constraints.

- Inputs: 2D box, crop, K.
- Uses position to infer orientation: only through the ray azimuth, which is the correct and necessary use. Not trajectory position.
- What it would take here: MultiBin is a *head design*, not a deliverable. Its relevance is as the loss to use for any yaw head we train (candidates 4 and 5), and as the checklist for the conversion we must get right: at our 64 degree FOV the azimuth term alone spans plus or minus 32 degrees, and at 16 m the ray elevation (10 to 33 degrees) makes the usual planar-rotation shortcut wrong.
- Expected accuracy: the KITTI numbers do not transfer. Treat as design guidance.
- Cost: reading time. Should inform the loss in any trained candidate.

### 7. Zhang, Lu, Zhou (2021), Objects are different: flexible monocular 3D object detection (MonoFlex)
`zhang_2021_monoflex.pdf`, https://arxiv.org/abs/2104.02323 (CVPR 2021), code at https://github.com/zhangyp15/MonoFlex

Representative of the KITTI monocular family alongside GUPNet
(https://arxiv.org/abs/2107.13774) and MonoCon
(https://arxiv.org/abs/2112.04628). Predicts ten projected 3D box keypoints
(eight corners plus two face centres) from a centre heatmap, decouples
truncated objects at the feature-map edge so they do not corrupt the normal
objects, and ensembles a directly regressed depth with several keypoint-solved
depths under learned uncertainty. Orientation is a MultiBin observation angle
converted to global yaw. All three use the same conversion and all three are
trained exclusively on KITTI, camera height about 1.65 m, forward-facing.

- Inputs: frame plus K. Weights are KITTI-only.
- Uses position to infer orientation: no (ray azimuth only).
- What it would take here: running KITTI weights on 16 m roadside footage means a domain gap larger than BEVHeight's own, in the same direction (unseen depression angle) so the errors are likely correlated rather than complementary. Retraining needs KITTI-format 3D labels at our pose, which we do not have and would have to pseudo-label, at which point we are back in the self-training loop that already exists in `scripts/finetune/`.
- Expected accuracy 25 to 90 m: worse than BEVHeight. KITTI's evaluation range stops at about 60 m and its training distribution is dominated by objects under 40 m.
- Cost: low to try, low expected value. Listed to be dismissed with a reason.

### 8. Brazil, Kumar, Straub, Ravi, Johnson, Gkioxari (2022), Omni3D: a large benchmark and model for 3D object detection in the wild (Cube R-CNN)
`brazil_2022_omni3d-cubercnn.pdf`, https://arxiv.org/abs/2207.10660 (CVPR 2023), code and weights at https://github.com/facebookresearch/omni3d

234k images, 3M instances, 98 categories, assembled from KITTI, nuScenes,
Objectron, ARKitScenes, SUN RGB-D and Hypersim. Cube R-CNN extends Faster
R-CNN with a cube head predicting, per 2D proposal, a virtual depth, the three
dimensions, and a full **allocentric rotation** as a 6D continuous
parameterization rather than a single yaw, with an uncertainty weight. The
virtual-depth trick normalizes for differing focal lengths across datasets,
which is the one component that plausibly helps across camera changes. Released
weights cover the full Omni3D set and a KITTI-plus-nuScenes outdoor subset.

- Inputs: frame plus K only. Runs out of the box.
- Uses position to infer orientation: no.
- What it would take here: `pip install` the repo's detectron2 pin, run `demo.py` over `frames_all/` with our K, take `R_cam` per detection, convert allocentric to global via the ray, project to the road plane normal from `lidar2cam` to get yaw about the road z axis, associate to existing detections by BEV IoU. Two to three days including the rotation bookkeeping, which is where the bugs will be. Weights only, no training data needed.
- Expected accuracy 25 to 90 m: genuinely unknown. The training mix is all ego-centric or indoor, so the 16 m depression is unseen, but the 6D rotation output and the virtual-depth normalization mean its failure mode should differ from BEVHeight's. Even if its raw median is worse, a decorrelated second estimate is usable in a fusion, and its per-prediction uncertainty gives a principled weight. Guess: 15 to 25 degrees median, with a flip rate that may be lower than BEVHeight's because the 6D parameterization has no discrete front-back branch to collapse.
- Cost: low-medium, GPU preferred but CPU-feasible offline at a few seconds per frame.

### 9. Tang, Wang, Liu, Zhao (2022), CenterLoc3D: monocular 3D vehicle localization network for roadside surveillance cameras
`tang_2022_centerloc3d.pdf`, https://arxiv.org/abs/2203.14550

Roadside-specific. A centre-point network with a weighted-fusion module and a
loss combining 2D and 3D supervision, predicting the vehicle's 3D centre,
dimensions and orientation directly in the surveillance camera's frame,
evaluated on their own SVLD-3D dataset of roadside scenes.

- Inputs: frame plus per-scene calibration.
- Uses position to infer orientation: no.
- What it would take here: the dataset and weights are scene-specific and the release is thin. Training on our clips requires 3D boxes, which we do not have except as pseudo-labels.
- Expected accuracy: not assessable without weights.
- Cost: high, low confidence. Cite as prior art, do not implement.

### 10. Song, Wang, Zhao, Zhu, Yang et al. (2019), ApolloCar3D: a large 3D car instance understanding benchmark for autonomous driving
`song_2019_apollocar3d.pdf`, https://arxiv.org/abs/1811.12222 (CVPR 2019)

5,277 images, 60k car instances, each fitted with an industry CAD model at
absolute scale and 66 labelled semantic keypoints. The benchmark methods
predict 2D keypoints then solve pose either by PnP against the CAD model or by
direct regression, and the paper reports that keypoint-plus-PnP beats direct
pose regression when enough keypoints are visible.

- Inputs: crop, keypoints, K, CAD models.
- Uses position to infer orientation: no. PnP uses only the crop's keypoints and K.
- What it would take here: a keypoint detector that fires on 44 x 29 px vehicles. ApolloCar3D images are ego-centric at 1 to 2 m, resolution per car far higher than ours past 30 m. Would need retraining and we have no keypoint labels on roadside footage.
- Expected accuracy 25 to 90 m: potentially 2 to 3 degrees inside 30 m where a car spans 90 to 125 px, unusable past 50 m where the wheelbase projects to under 30 px and individual wheel contacts are 5 to 10 px apart.
- Cost: high, and it fails in exactly the range band we need. Not a candidate.

### 11. Reddy, Vo, Narasimhan (2019), Occlusion-Net: 2D/3D occluded keypoint localization using graph networks
`reddy_2019_occlusion-net.pdf`, https://openaccess.thecvf.com/content_CVPR_2019/papers/Reddy_Occlusion-Net_2D3D_Occluded_Keypoint_Localization_Using_Graph_Networks_CVPR_2019_paper.pdf (CVPR 2019), code at https://github.com/dineshreddy91/Occlusion_Net

Takes Mask R-CNN's visible vehicle keypoints, runs a graph encoder that
classifies which skeleton edges are occluded and a graph decoder that
hallucinates the occluded keypoint positions, then a 3D graph network that
recovers shape and camera pose under a self-supervised reprojection loss. The
occluded keypoints are supervised only indirectly, through a trifocal-tensor
loss across simultaneous views in the multi-camera CarFusion data, so no manual
occluded-keypoint labels are needed. It is the most relevant keypoint method
because dense highway traffic occludes wheels constantly and because CarFusion
is roadside-ish rather than ego-centric.

- Inputs: 2D box, crop. Outputs 12 vehicle keypoints including the four wheel contacts, plus a pose.
- Uses position to infer orientation: no.
- What it would take here: released weights on CarFusion, so a zero-training trial is possible. But CarFusion cameras are at a few metres and vehicles fill far more pixels. Applied to our crops the wheel keypoints would need to be accurate to about 2 px at 60 m to beat the alternatives, which the resolution table says is not going to happen.
- Expected accuracy 25 to 90 m: useful under 35 m, degrading fast, noise past 55 m.
- Cost: medium. Worth a two-hour smoke test on close-range crops only, not a pipeline candidate.

### 12. Ravi et al. (2024), SAM 2: segment anything in images and videos
`ravi_2024_sam2.pdf`, https://arxiv.org/abs/2408.00714, weights at https://github.com/facebookresearch/sam2 (Apache 2.0)

Promptable segmentation with a streaming memory across video frames, so a box
prompt on one frame propagates a temporally consistent mask along a track. Not
an orientation method. Its role here is to supply the one cue that survives the
smear analysis: a clean **lower** mask contour, the tire contact line, which
back-projects to the road plane without the height ambiguity. The video memory
matters because per-frame masks jitter and a jittering contact line is a
jittering yaw, which is the failure we are trying to fix.

- Inputs: frame plus a box prompt. We already have per-frame 2D boxes by projecting the BEVHeight 3D boxes.
- Uses position to infer orientation: no.
- Cost: sam2.1-hiera-tiny runs on Mac CPU or MPS at roughly 0.5 to 1.5 s per frame for 8 prompts, faster with the video predictor amortizing across a track. No training data needed.

### 13. Yang, Kang, Huang, Zhao, Xu, Feng, Zhao (2024), Depth Anything V2
`yang_2024_depth-anything-v2.pdf`, https://arxiv.org/abs/2406.09414, weights at https://github.com/DepthAnything/Depth-Anything-V2

Relative monocular depth from a DINOv2 backbone, trained on synthetic labelled
plus 62M pseudo-labelled real images. Sharper boundaries and better thin
structures than V1.

- Inputs: frame only. Output is relative, not metric, and has no fixed scale or shift per frame.
- Uses position to infer orientation: no.
- What it would take here: fitting a yaw would require lifting the vehicle surface to metric 3D, which needs a per-frame scale and shift solved against the known road plane. Feasible in principle (fit the affine alignment on road pixels, then lift vehicle pixels), but at 60 m the entire car occupies a few hundred pixels whose relative depth spread is under a metre against a 60 m baseline. The depth head's relative precision is nowhere near that.
- Expected accuracy: no useful yaw past 30 m. Recorded so we do not spend a week discovering it.
- Cost: low to try, near-zero expected value for orientation. Possibly useful later for the range-dependent depth bias, which is out of scope here.

### Off-the-shelf rotated-box detectors, for completeness

YOLO11-OBB (https://docs.ultralytics.com/tasks/obb/) ships weights trained on
DOTA v1, aerial imagery only: planes, ships, storage tanks, and small vehicles
seen from near-nadir. Nadir aerial views have no perspective smear, which is
the whole difficulty here, so the learned appearance-to-angle mapping does not
transfer to an 18 degree pitch. Using YOLO-OBB would mean training it on our
own rotated boxes, and the only rotated boxes we have are BEVHeight's, which
are the thing we are trying to replace. It is a useful *architecture* for
candidate 5, not a usable *model*. Same verdict for mask-to-OBB with any
segmenter, for the reason in section 0.

---

## Ranked candidates for this repo

Cheapest first among those with a real expected gain. Each becomes one
`scripts/orientation/candidates/<name>.py` exposing
`run(clip, det_dir, tracks, cfg) -> {track_id: {frame: yaw}}` and goes through
`run_candidate.py` and the frozen score. None of them reads GPS, and none of
them derives yaw from the trajectory.

**1. `yaw_edge_align`: one-parameter re-fit of the existing box against image gradients. Half a day, no new dependencies, no training data.**
Keep x, y, z, l, w, h from `NNN_pred.json` exactly as they are and re-solve only
yaw. For each detection, sweep yaw over 180 values at 2 degree steps; for each
candidate yaw, project the 3D box wireframe into the image with K and
`lidar2cam` and score it by the mean Sobel gradient magnitude sampled along the
projected edges, weighting the four ground-contact edges about three times the
vertical and roof edges since those are the edges the smear analysis says carry
the signal. Take the argmax, then parabola-interpolate between the two
neighbouring samples for sub-step resolution, and record the ratio of the best
to the second-best local maximum as a confidence so the ledger can report how
often the fit is ambiguous. Expect this to be decisive inside 40 m, where the
box projects to 60 to 125 px and the contact edges are sharp, and to tie with
BEVHeight past 70 m where the gradient landscape flattens. It is deliberately
the laziest thing that could work and it costs one afternoon, so it should be
run before anything with a download attached. Guard to include from day one:
the 180 degree sweep will happily report the front-back flipped solution, so
score raw and folded separately exactly as the frozen score does, and do not let
an apparent folded-error win hide a worse flip rate.

**2. `yaw_contact_line`: SAM2 masks, lower contour only, rectangle fit on the road plane. Two to three days, weights only, no training data.**
Project each detection's 3D box to a 2D box, prompt SAM2 (sam2.1-hiera-tiny,
video predictor so the mask is temporally propagated along the AB3DMOT track
rather than re-segmented per frame), and keep only the bottom 15 percent of the
mask's contour, which is the tire contact line and the near shadow edge.
Back-project those pixels to the road plane at `z = -1.73` using K and
`lidar2cam` and fit a rectangle of the detection's own l and w by minimizing
point-to-rectangle distance over yaw and centre, initialized at the BEVHeight
yaw and at that yaw plus 90 degrees so the l-w swap is tested explicitly. Do
not run a minimum-area rectangle over the full mask: section 0 shows the warped
silhouette stretches 5.5 m radially at 60 m and the resulting long axis is the
viewing direction, not the vehicle. Shadows are the main hazard, since a low sun
puts a hard contour on the road that a contact-line fit will happily lock onto,
so add a rejection on fitted footprint area more than 1.5 times l times w and
log the rejection rate per clip. Expected 2 to 4 degrees median inside 40 m and
5 to 10 degrees in the 40 to 60 m band, which is the band that matters, and
nothing usable past 75 m, so gate the candidate's output by range and fall back
to BEVHeight yaw beyond the gate.

**3. `yaw_roadside25d`: Polley's 2.5D parallelogram detector, off the shelf. One to two days, weights released, no training data.**
Clone the KIT GitLab repo, run their released inference on `frames_all/`, and
for each predicted parallelogram back-project its four image vertices to the
road plane and read yaw from the long edge. Associate to the existing
detections by BEV footprint IoU and emit yaw for the matched ones only. This is
the single published method whose output representation is exactly the quantity
we want, produced from RGB alone with no trajectory input, and the authors state
explicitly that they avoid trajectory-derived orientation because it fails on
stationary vehicles, which is the same independence constraint the lab lead
imposed. The risk to state plainly before running: their infrastructure cameras
sit around 7 m against our 16.26 m, so we inherit a domain gap of the same kind
as BEVHeight's even though it is a different model, and their real training
footage is intersection-scale rather than 90 m highway. Treat the run as a
measurement of whether the gap is smaller than BEVHeight's, not as a drop-in
replacement, and report the per-range-bin numbers before anyone gets attached to
it.

**4. `yaw_cubercnn`: Cube R-CNN as a deliberately decorrelated second opinion. Two to three days, weights only, no training data.**
Run the released Omni3D outdoor weights over `frames_all/` with our K, take the
predicted allocentric rotation per detection, convert to global yaw by composing
with the viewing-ray rotation and then projecting onto the road plane normal
recovered from `lidar2cam`, and associate to existing detections by BEV IoU.
The conversion is where this candidate will break, so write it against a
synthetic check first: place a box of known yaw at several image positions
spanning the full 64 degree FOV, forward-project, run the conversion backwards,
and assert the yaw comes back. Its value is not that it will beat BEVHeight on
median error, it probably will not, but that its 6D continuous rotation head has
no discrete front-back branch to collapse, so its flip behaviour should be
independent of BEVHeight's, and Cube R-CNN emits a per-prediction uncertainty
that gives a principled fusion weight. Only fuse after both have been scored
alone on all 8 held-out clips, and only if the ledger shows the flips actually
land on different frames.

**5. `yaw_bev_rbox`: tailed r-box in the IPM view, trained on renders at our camera pose. Two to three weeks, GPU, synthetic training data we generate ourselves.**
The Zhu et al. recipe, adapted. Warp each frame to a metric BEV raster through
the road homography we can compute exactly from K and `lidar2cam`, at a
resolution chosen so a car is at least 20 px across at 90 m, and train a small
rotated-box head on it with their `sin^2` angular loss over a plus or minus 45
degree residual window and their tailed r-box target, the tail being the BEV
vector from footprint centre to roof centre that tells the network which
stretched pixels belong to the body rather than the footprint. Training data
comes from CARLA and from Blender vehicles composited onto empty-road frames
from our own clips, rendered with our camera at 16.26 m, 18.5 degrees pitch and
f = 1525, so the network sees the exact depression angles that are absent from
DAIR-V2X-I. This needs no human labels at any stage, which is what makes it
worth the weeks. It is last because the first four answer the question of
whether an appearance yaw source can help at all, and if they all fail at 40 to
60 m for resolution reasons rather than model reasons, this one will fail there
too and the honest conclusion will be that 2D-cue yaw is a sub-50 m tool and the
40 to 60 m band needs the fine-tuned checkpoint instead.

---

## What to tell the lab lead

All five candidates derive yaw from pixels, not from the trajectory, so all five
satisfy the independence requirement. The classic surveillance line does not:
its vehicle orientation is the road's vanishing-point direction, aggregated from
how traffic moves, so it is both trajectory-derived and incapable of expressing
a lane change. The binding constraint is not method choice but resolution. A car
at 60 m presents a 44 by 29 px ground footprint from this mount and its body
smears 5.5 m radially when warped to the road plane, so appearance yaw is a tool
for 25 to 60 m, and the 60 to 90 m tail will have to come from the fine-tuned
detector or from temporal smoothing, not from a better 2D cue.
