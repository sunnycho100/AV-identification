# Running detection and tracking on the lab server

Server `cee` (cee-r030232), repo at `~/roadside-camera/BEVHeights`.
Target clips: `AV_V_WE_3`, `AV_W_WE_1`, `AV_W_WE_3`.

Verified 2026-09-15: code, calibration and both checkpoints are on the server,
and a 3 frame smoke test on GPU 1 produced `NNN_pred.json` in 17 s.

## Environment

Use the existing env. It is the torch 1.9 / mmcv 1.4 stack that
`run_bevheight_generic.py` was written against, including the cusolver CPU
inverse shim. Do not build the torch 2.0.1 env from
`server-finetune-setup.md` for inference: that one is for fine-tuning, and
root is 98% full.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/scho242/BEVverify/.envs/bevheight
cd ~/roadside-camera/BEVHeights
```

## GPU etiquette

Both GPUs are shared with hzhou364. Check before every launch and use GPU 1
only when it is well under load:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Detection peaks near 3 GB. Prefix every run with `CUDA_VISIBLE_DEVICES=1`.

## Step 0. Extract full rate frames

`data/camera-data/<CLIP>/frames` holds only 10 sampled frames (every 30th).
Tracking at 30 fps needs every frame. The source videos are already on the
server, so extract there:

```bash
for CLIP in AV_V_WE_3 AV_W_WE_1 AV_W_WE_3; do
  mkdir -p "data/camera-data/$CLIP/frames_all"
  ffmpeg -nostdin -i "Camera data/$CLIP.mp4" -start_number 0 -q:v 2 \
    "data/camera-data/$CLIP/frames_all/%03d.jpg"
done
```

About 298 frames per clip, roughly 60 MB total.

## Step 1. Site extrinsic

`metric_extrinsic_site.json` does not exist yet for these three clips, and it
is what the README pipeline feeds the detector. Generate it against the
`AV_T_WE_1` reference, which is already on the server:

```bash
for CLIP in AV_V_WE_3 AV_W_WE_1 AV_W_WE_3; do
  python scripts/calibration/site_extrinsic.py \
    --frames-dir "data/camera-data/$CLIP/frames" \
    --anycalib-json outputs/calibration/camera-data/$CLIP/*_anycalib_pinhole_pinhole.json \
    --height 16.26 \
    --reference outputs/calibration/camera-data/AV_T_WE_1/metric_extrinsic_site.json \
    --out "outputs/calibration/camera-data/$CLIP/metric_extrinsic_site.json"
done
```

Check each clip's reference gate before trusting it. A clip that fails the
2 degree check was likely re-pointed and needs its own pose label.

If you would rather not re-derive calibration, `metric_extrinsic_v2.json`
already exists for all three clips and is what the smoke test used. Swap the
filename in Step 2 and say so in the results.

## Step 2. Detection

Note the AnyCalib intrinsics filename differs per clip: 150 for `AV_V_WE_3`,
151 for `AV_W_WE_1`, 154 for `AV_W_WE_3`. The glob below handles that.

```bash
for CLIP in AV_V_WE_3 AV_W_WE_1 AV_W_WE_3; do
  CUDA_VISIBLE_DEVICES=1 python scripts/object_detection/run_bevheight_generic.py \
    --frames-dir "data/camera-data/$CLIP/frames_all" \
    --anycalib-json outputs/calibration/camera-data/$CLIP/*_anycalib_pinhole_pinhole.json \
    --extrinsic-json "outputs/calibration/camera-data/$CLIP/metric_extrinsic_site.json" \
    --out-dir "outputs/object_detection/camera-data/${CLIP}_phase1"
done
```

Expect roughly 5 minutes per clip. The voxel pooling CUDA extension is not
built, so the model uses the pure PyTorch scatter fallback. That is a speed
cost only, and at this clip length it is not worth building.

## Step 3. Tracking (runs on the Mac, not the server)

`run_ab3dmot.py` imports `AB3DMOT_libs` from `third_party/`, which is not synced
to the server. Rsync the detections back first (Step 4), then run this locally
under `/Users/sunghwan_cho/miniforge/bin/python3.12`:

```bash
for CLIP in AV_V_WE_3 AV_W_WE_1 AV_W_WE_3; do
  NUMBA_DISABLE_JIT=1 python scripts/tracking/run_ab3dmot.py --fps 30 --max-age 6 \
    --det-dir "outputs/object_detection/camera-data/${CLIP}_phase1" \
    --out-dir  "outputs/tracking/camera-data/${CLIP}_phase1"
done
```

Tracking is CPU only, so it does not need the GPU check.

## Step 4. Bring results back to the Mac

Run from the repo root on the Mac:

```bash
rsync -avz \
  --include='*_phase1/' --include='*_phase1/**' --exclude='*' \
  cee:roadside-camera/BEVHeights/outputs/object_detection/camera-data/ \
  outputs/object_detection/camera-data/

rsync -avz \
  --include='*_phase1/' --include='*_phase1/**' --exclude='*' \
  cee:roadside-camera/BEVHeights/outputs/tracking/camera-data/ \
  outputs/tracking/camera-data/
```

Grading against GPS stays on the Mac. Held out ground truth is never read on
the server.

## Step 5. Verify nothing private is on the server

```bash
ssh cee 'ls ~/roadside-camera/BEVHeights/personal-documents'   # must fail
```
