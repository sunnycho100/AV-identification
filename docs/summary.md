# BEVHeights — Project Summary

## What This Project Does

**BEVHeights** is an infrastructure-based 3D object detection system for roadside cameras. It converts camera images from fixed roadside/infrastructure viewpoints into a Bird's Eye View (BEV) representation to detect 3D objects such as cars, trucks, buses, pedestrians, and cyclists.

This is research-level code — not an ego-vehicle (self-driving car) perception system, but rather a **roadside perception** system relevant to V2X (Vehicle-to-Everything) communication scenarios.

---

## Project Structure

```
BEVHeights-mac/
├── models/                  # Main model: BEVHeight (backbone + head)
├── layers/
│   ├── backbones/           # LSS FPN backbone (Lift, Splat, Shoot)
│   └── heads/               # BEV detection head (CenterHead-based)
├── ops/                     # Custom CUDA voxel pooling kernel
├── dataset/                 # Multi-view detection dataset loader
├── evaluators/              # KITTI-format evaluation pipeline
├── experiments/
│   ├── dair-v2x/            # Experiment configs for DAIR-V2X dataset
│   └── rope3d/              # Experiment configs for ROPE3D dataset
├── scripts/                 # Data conversion (DAIR-V2X/ROPE3D → KITTI format)
├── utils/                   # Distributed training & backup utilities
├── docs/                    # Install, dataset prep, and run guides
├── data/                    # Dataset directories (ROPE3D, split configs)
└── assets/                  # Documentation assets
```

---

## Technical Approach

1. **Input**: Single or multi-view RGB images from roadside infrastructure cameras
2. **Image Feature Extraction**: ResNet-50 or ResNet-101 backbone with FPN
3. **2D → BEV Lifting**: Lift, Splat, Shoot (LSS) approach — predicts depth per pixel, lifts features to 3D, then projects to a BEV grid via custom CUDA voxel pooling
4. **BEV Processing**: ResNet-18 + SECONDFPN refines the BEV feature map
5. **Detection Head**: Multi-task CenterHead predicts heatmaps, 3D box dimensions, height, rotation, and velocity for 6 task groups (10+ object classes)
6. **Height Normalization**: Ground plane estimation compensates for varying camera mount heights and angles

---

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| **DAIR-V2X** | Single infrastructure camera dataset (10Hz, range up to 102.4m or 140.8m) |
| **ROPE3D** | Roadside perception dataset with 3D bounding box annotations |

Both datasets are converted to KITTI format for training and evaluation.

---

## Model Configurations

| Variant | Backbone | BEV Grid | Detection Range |
|---------|----------|----------|----------------|
| R50, 128×128, 102.4m | ResNet-50 | 128×128 | 0–102.4m |
| R50, 128×128, 140.8m | ResNet-50 | 128×128 | 0–140.8m |
| R101, 256×256, 102.4m | ResNet-101 | 256×256 | 0–102.4m |
| R101, 256×256, 140.8m | ResNet-101 | 256×256 | 0–140.8m |

---

## Key Dependencies

- **PyTorch** + **PyTorch Lightning** (training framework)
- **mmdet / mmdet3d** (OpenMMLab detection toolkits — ResNet, FPN, CenterHead, 3D box ops)
- **nuscenes-devkit** (data utilities, quaternion operations)
- **numba** (JIT-compiled KITTI evaluation)
- **Custom CUDA extension** for voxel pooling (requires GPU build)

---

## Training & Evaluation Workflow

**Training:**
```bash
python experiments/dair-v2x/bev_height_lss_r101_864_1536_256x256_140.py \
    --amp_backend native -b 2 --gpus 8
```

**Pipeline:**
```
Images → ResNet → LSS Lift → Voxel Pooling (BEV) → BEV ResNet18 → SECONDFPN → Multi-task Heads → 3D Boxes
```

**Losses:** Gaussian Focal Loss (classification) + L1 Loss (regression, weight 0.25)

**Evaluation:** Results converted to KITTI format → official KITTI AP metrics (IoU 0.7, 0.5)

---

## Pre-trained Checkpoint

- `BEVHeight_R50_128_102.4_65.48_49_epochs.ckpt` (875 MB, gitignored)
- ResNet-50 backbone, 128×128 BEV, 102.4m range, trained for 49 epochs
- Reported metric: **65.48 AP**

---

## Paper & Publication

- **Title**: *BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection*
- **Venue**: **CVPR 2023** (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **arXiv**: https://arxiv.org/abs/2303.08498
- **Authors**: Lei Yang, Kaicheng Yu, Tao Tang, Jun Li, Kun Yuan, Li Wang, Xinyu Zhang, Peng Chen
- **Affiliations**: Alibaba DAMO Academy, Tsinghua University

---

## Key Results (from paper)

BEVHeight surpasses the BEVDepth baseline by:
- **+4.85%** on DAIR-V2X-I (clean setting)
- **+4.43%** on Rope3D (clean setting)
- **+26.88%** on robust settings (with external camera parameter changes)

### DAIR-V2X-I Benchmark

| Config | Range | Car 3D@0.5 (E/M/H) | Pedestrian 3D@0.25 (E/M/H) | Cyclist 3D@0.25 (E/M/H) |
|--------|-------|---------------------|----------------------------|--------------------------|
| R50_102 | 0–102.4m | 77.48 / 65.46 / 65.53 | 26.86 / 25.53 / 25.66 | 51.18 / 52.43 / 53.07 |
| R50_140 | 0–140.8m | 80.80 / 75.23 / 75.31 | 28.13 / 26.73 / 26.88 | 49.63 / 52.27 / 52.98 |
| R101_102 | 0–102.4m | 78.06 / 65.94 / 65.99 | 40.45 / 38.70 / 38.82 | 57.61 / 59.90 / 60.39 |
| R101_140 | 0–140.8m | 81.80 / 76.19 / 76.26 | 38.79 / 37.94 / 38.26 | 58.22 / 60.49 / 61.03 |

### Rope3D Benchmark

| Config | Range | Car 3D@0.5 (E/M/H) | Big Vehicle 3D@0.5 (E/M/H) | Car 3D@0.7 (E/M/H) | Big Vehicle 3D@0.7 (E/M/H) |
|--------|-------|---------------------|----------------------------|---------------------|----------------------------|
| R50_102 | 0–102.4m | 83.49 / 72.46 / 70.17 | 50.73 / 47.81 / 47.80 | 48.12 / 42.45 / 42.34 | 24.58 / 26.25 / 26.28 |
| R50_140 | 0–140.8m | 85.46 / 79.15 / 79.06 | 64.38 / 65.75 / 65.77 | 46.39 / 42.85 / 42.71 | 27.21 / 33.99 / 34.03 |

---

## Notable Milestones

- **2024/05** — Integrated into **NVIDIA DeepStream-3D** for sensor fusion
- **2023/03/15** — arXiv paper and codebase released
- **2023/02/27** — Accepted to CVPR 2023

---

## Acknowledgments

Built upon:
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) — baseline BEV detector
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) — dataset and V2X framework
- [pypcd](https://github.com/dimatura/pypcd) — point cloud data utilities

---

## Citation

```bibtex
@inproceedings{yang2023bevheight,
    title={BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection},
    author={Yang, Lei and Yu, Kaicheng and Tang, Tao and Li, Jun and Yuan, Kun and Wang, Li and Zhang, Xinyu and Chen, Peng},
    booktitle={IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
    month = mar,
    year={2023}
}
```
