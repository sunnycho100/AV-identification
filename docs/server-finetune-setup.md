# Fine-tuning on the lab server (cee-r030232, RTX 6000 Ada)

Why move off the Mac: mmcv's deformable convolution has no working backward
against torch 2.10 on CPU/MPS, and that DCN sits inside BEVHeight's height
branch. Locally we had to freeze everything upstream and train the head alone,
which fixes heading but leaves the range-dependent depth error untouchable. A
CUDA build restores that backward, so **the height branch becomes trainable
here**. Speed is the secondary benefit.

Also: MPS produced ~40x inflated gradients while the loss appeared to fall, so
`check_server_env.py` compares CPU and GPU gradients before anything else runs.

## 0. Host key warning (do this first, it currently blocks ssh)

`ssh cee` fails with `REMOTE HOST IDENTIFICATION HAS CHANGED` for the jump host
`best-tux.cae.wisc.edu`. Usually this means the machine was rebuilt, but it is
also what a man-in-the-middle looks like, so **verify before clearing it** —
check with CAE/IT or from a known-good network that the fingerprint matches.
Only then:

```bash
ssh-keygen -R best-tux.cae.wisc.edu
```

Do not clear it blindly.

## 1. Environment

The RTX 6000 Ada is sm_89, which needs CUDA 11.8 or newer. torch 2.0.1+cu118 is
the sweet spot: new enough for the GPU, old enough for the mmcv 1.x line this
repo targets.

```bash
conda create -n bevh python=3.10 -y
conda activate bevh

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# mmcv-full 1.7.1 has prebuilt wheels for this exact torch/CUDA pair, which
# avoids a long and failure-prone source build
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

pip install mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6
pip install numba pandas scikit-image scipy opencv-python-headless \
            pytorch-lightning==1.5.10 tensorboardX nuscenes-devkit
```

Note `requirements.txt` pins numba 0.48 and numpy 1.19, which predate this stack;
install current versions instead, as we already do on the Mac.

## 2. Build the voxel-pooling CUDA extension

Optional but worth it: without it the model falls back to a slow pure-PyTorch
scatter path.

```bash
cd ~/roadside-camera/BEVHeights
python setup.py develop
```

## 3. Fetch the pretrained checkpoint on the server

Not synced (917 MB). Pull it directly:

```bash
mkdir -p checkpoints && cd checkpoints
wget -O BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt \
  "https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1"
```

## 4. Verify BEFORE syncing data or training

```bash
python scripts/finetune/check_server_env.py
```

Every check maps to a failure already hit locally. The one that matters most is
**deformable conv backward**: if it fails, the height branch stays frozen and
the server buys only speed.

## 5. Sync

From the Mac:

```bash
./scripts/sync_to_server.sh --dry-run   # inspect first
./scripts/sync_to_server.sh
```

It sends code, frames, calibration and pseudo-labels. It does **not** send
`personal-documents/`, the source videos, or the GPS trajectory CSVs, and it
verifies afterwards that none of them landed there. Home directories on this
server are world-readable by the lab, so held-out ground truth stays local and
evaluation is run on the Mac.

## 6. Train

Once DCN backward passes, unfreeze the height branch (that is the whole point of
being here). In `scripts/finetune/train_finetune.py`, `freeze_image_branch()`
currently freezes all of `model.backbone`; on the server, freeze only
`backbone.img_backbone` and `backbone.img_neck` and leave `backbone.height_net`
trainable, then use `--device cuda`.

```bash
python scripts/finetune/train_finetune.py --device cuda --epochs 20
```

Expect roughly 0.05 s/step against 1.7 s on the Mac, so an epoch runs in well
under a minute and sweeping learning rate and epochs becomes practical.

## 7. Evaluate honestly

Keep `AV_T_EW_3` held out, as on the Mac. The Mac pilot's numbers to beat, all on
that held-out clip:

| metric | pretrained | Mac pilot (head only) |
|---|---|---|
| heading median offset | 7.9 deg | 1.7 deg |
| within 15 deg of road axis | 68.3% | 93.2% |
| >45 deg tail | 8.7% | 3.3% |
| cars per frame | 7.30 | 7.57 |

```bash
python scripts/finetune/eval_finetune.py --ckpt outputs/finetune/<run>/head_ft_ep<N>.ckpt
```

Two things to check that a single accuracy number hides:

- **Recall must not fall.** An earlier run "improved" heading by detecting
  nothing at all. `cars_per_frame` is in the eval output for exactly this.
- **Heading diversity must survive.** The Mac pilot raised heading concentration
  on DAIR intersection frames from 0.783 to 0.838, meaning the road-direction
  prior leaks in statistically even though labels came from each vehicle's own
  motion. With the height branch also training, re-measure this. If cars in the
  DAIR turning frames start pointing all one way, the model has learned the lane
  prior, which would erase the lane-change signal the AV-vs-human classification
  depends on.

Unfreezing the height branch should additionally be judged on the depth warp:
re-run the GPS grading and check whether the ~1.2 m too-near / ~1 m too-far bend
flattens. That comparison must use the marker-identified target track, never the
best-matching one.
