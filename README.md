# BEACON: A Boundary-Enhanced and Content-Adaptive Query Framework for Transparent Object Instance Segmentation

Rotanakkosal Chhun, Vungsovanreach Kong, Anand Nayyar, and Tae-Kyung Kim

Department of Big Data, Chungbuk National University, Cheongju-si, South Korea

---

## Abstract

Transparent object instance segmentation remains difficult in robotic perception and scene understanding, as adjacent transparent instances often exhibit weak or ambiguous boundaries, while small or heavily occluded objects are easily missed. We propose **BEACON** (Boundary-Enhanced and Content-Adaptive Query Framework), a training-time framework for RGB-only transparent object instance segmentation built on Mask2Former. BEACON combines an auxiliary boundary head, a decoder boundary dice loss, and a hybrid content-adaptive query initialization strategy.

On the **ClearPose** dataset, BEACON achieves **44.88 AP**, improving over Mask2Former by 7.52 AP (+20.1%), with gains of 13.63 AP₅₀ (22.7%) and 3.12 APₛ (44.6%), and outperforming Mask R-CNN by 18.94 AP. On **Trans10K-v2**, it further improves AP by 1.14 and APₛ by 2.24 over Mask2Former.

---

## Main Results

### ClearPose Validation Set

| Method | Backbone | AP | AP₅₀ | AP₇₅ | APₛ | APₘ | APₗ |
|--------|----------|----|-------|-------|-----|-----|-----|
| Mask R-CNN | R-50 | 25.94 | 53.16 | 22.51 | 2.18 | 29.35 | 40.92 |
| Mask DINO | Swin-B | 27.91 | 43.96 | 25.56 | 3.20 | 32.39 | 41.25 |
| OneFormer | Swin-B | 31.69 | 58.97 | 29.72 | 4.38 | 36.66 | 47.14 |
| Mask2Former | Swin-B | 37.36 | 60.10 | 39.95 | 7.00 | 43.80 | 48.31 |
| **BEACON (ours)** | **Swin-B** | **44.88** | **73.73** | **46.34** | **10.12** | **50.22** | **63.83** |

### Trans10K-v2

| Method | AP | AP₅₀ | AP₇₅ | APₛ | APₘ | APₗ |
|--------|----|-------|-------|-----|-----|-----|
| Mask R-CNN | 36.48 | 53.45 | 40.10 | 0.47 | 5.12 | 47.75 |
| Mask DINO | 56.37 | 66.83 | 57.05 | 0.16 | 10.75 | 74.09 |
| Mask2Former | 67.37 | 68.72 | 68.72 | 9.81 | 36.65 | 81.61 |
| **BEACON (ours)** | **68.51** | **79.01** | **69.84** | **12.05** | 34.37 | **82.49** |

---

## Installation

### Requirements
- Ubuntu 24.04 LTS
- Python 3.10.19
- PyTorch 2.9.1+cu128
- CUDA 12.8, cuDNN 9.1.0
- NVIDIA RTX A6000 (48GB) or equivalent

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd Mask2Former

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
cd detectron2 && pip install -e . && cd ..
```

### Dataset Preparation

Download the **ClearPose** dataset from [https://github.com/opipari/ClearPose](https://github.com/opipari/ClearPose).
We use the heavy-occlusion subset: 2,878 training images and 773 validation images.

Download the **Trans10K-v2** dataset from [https://github.com/xieenze/SegmentTransparentObjects](https://github.com/xieenze/Trans2Seg).

Place datasets under `datasets/`:
```
datasets/
  clearpose_dataset/
    coco_clearpose_train.json       # COCO-format annotations
    coco_clearpose_val.json
    set1/ ... set9/                 # ClearPose image folders
  trans10k/
    coco_trans10k_train.json        # COCO-format annotations
    coco_trans10k_val.json
    images/                         # Trans10K-v2 images
```

To generate ClearPose annotations from raw label images, run:
```bash
python src/dataset-preparation/convert_clearpose_split.py
```

### Pretrained Backbone

Download the Swin-Base backbone pretrained on ImageNet-21K from the
[Mask2Former Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)
and place it under `weights/`:
```
weights/
  pkl/
    model_final_83d103.pkl          # Swin-B COCO Instance Segmentation checkpoint
```

See `weights/README.md` for download links.

---

## Training

### Train BEACON on ClearPose (main result — 44.88 AP)

```bash
python train-set/train_net_boundary_supervision.py \
  --config-file configs/clearpose/boundary_supervision/beacon_clearpose.yaml \
  --num-gpus 1
```

Training takes approximately **4.5 hours** on a single NVIDIA RTX A6000 (48GB).

### Train on Trans10K-v2 (68.51 AP)

```bash
python train-set/train_net_boundary_supervision.py \
  --config-file configs/trans10k/beacon_trans10k.yaml \
  --num-gpus 1
```

### Train baselines

```bash
# Mask R-CNN baseline (ClearPose)
python train-set/train_mask_rcnn_baseline.py \
  --config-file configs/clearpose/mask_rcnn_clearpose.yaml \
  --num-gpus 1

# Mask R-CNN baseline (Trans10K)
python train-set/train_mask_rcnn_baseline.py \
  --config-file configs/trans10k/mask_rcnn_trans10k.yaml \
  --num-gpus 1

# Mask2Former baseline (ClearPose)
python train-set/train_net.py \
  --config-file configs/clearpose/boundary_supervision/beacon_base_clearpose.yaml \
  --num-gpus 1

# Mask2Former baseline (Trans10K)
python train-set/train_net.py \
  --config-file configs/trans10k/mask2former_trans10k.yaml \
  --num-gpus 1
```

### Ablation experiments (Table 3)

```bash
# Row 2: + boundary supervision only (AP = 43.84)
# Note: This config uses beacon_base_clearpose.yaml without decoder boundary dice.
# For the exact boundary-only ablation, disable content queries manually.

# Row 3: + boundary supervision + content-adaptive queries (AP = 44.34)
python train-set/train_net_boundary_supervision.py \
  --config-file configs/clearpose/boundary_supervision/ablation_boundary_content_queries.yaml \
  --num-gpus 1

# Row 4: + decoder boundary dice loss = BEACON full model (AP = 44.88)
python train-set/train_net_boundary_supervision.py \
  --config-file configs/clearpose/boundary_supervision/beacon_clearpose.yaml \
  --num-gpus 1
```

---

## Evaluation

### Evaluate BEACON on ClearPose

```bash
python train-set/train_net_boundary_supervision.py \
  --config-file configs/clearpose/boundary_supervision/beacon_clearpose.yaml \
  --eval-only \
  MODEL.WEIGHTS output/beacon_clearpose/model_0011999.pth
```

### Evaluate on COCO transparent subset

```bash
python eval-set/eval_coco_transparent.py \
  --config-file configs/clearpose/boundary_supervision/beacon_clearpose.yaml \
  --checkpoint output/beacon_clearpose/model_0011999.pth
```

---

## Method Overview

BEACON addresses two failure modes in transformer-based transparent object segmentation:

1. **Weak boundary supervision** — adjacent transparent instances merge into a single prediction
2. **Static query initialization** — small or heavily occluded objects are missed at the earliest decoding stage

BEACON adds three components on top of Mask2Former:

- **Auxiliary boundary head**: supervises shared pixel-decoder features with boundary focal loss to learn edge-aware representations
- **Decoder boundary dice loss**: directly supervises per-query mask edge quality at every decoder stage with zero added inference parameters
- **Hybrid content-adaptive query initialization**: replaces 50 of 100 static queries with image-conditioned queries selected from encoder memory using a combined class+mask scoring function

The base inference architecture is **unchanged** — all additions are training-time only.

---

### Baseline Methods

Mask DINO and OneFormer results in the comparison tables were reproduced using their official implementations:
- **Mask DINO**: [https://github.com/IDEA-Research/MaskDINO](https://github.com/IDEA-Research/MaskDINO)
- **OneFormer**: [https://github.com/SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer)

Mask R-CNN and PointRend baselines are trained using the scripts and configs included in this repository.

---

## Repository Structure

```
configs/
  clearpose/
    boundary_supervision/
      beacon_base_clearpose.yaml              # shared base configuration
      beacon_clearpose.yaml                   # BEACON full model — 44.88 AP (Table 1)
      ablation_boundary_content_queries.yaml  # ablation: boundary + content queries (Table 3)
      ablation_content_queries_only.yaml      # ablation: content queries only (Table 4)
    mask_rcnn_clearpose.yaml                  # Mask R-CNN baseline (Table 1)
    pointrend_clearpose.yaml                  # PointRend baseline (Table 1)
  trans10k/
    beacon_trans10k.yaml                      # BEACON on Trans10K-v2 — 68.51 AP (Table 2)
    mask2former_trans10k.yaml                 # Mask2Former baseline (Trans10K)
    mask_rcnn_trans10k.yaml                   # Mask R-CNN baseline (Trans10K)
mask2former/                                  # core package
  modeling/
    boundary_supervision/                     # boundary head, criterion, query prior, overlap penalty
    transformer_decoder/                      # Mask2Former decoder + content-adaptive query init
    criterion.py                              # training losses + decoder boundary dice
  data/
    boundary_targets.py                       # GT boundary target generation
    dataset_mappers/                          # data loading with boundary augmentations
  models/
    boundary/                                 # MaskFormerBoundarySupervision (BEACON model)
train-set/
  train_net_boundary_supervision.py           # main BEACON training script
  train_net.py                                # standard Mask2Former training
  train_mask_rcnn_baseline.py                 # Mask R-CNN baseline training
eval-set/
  eval_coco_transparent.py                    # COCO transparent subset evaluation
src/
  dataset-preparation/                        # ClearPose annotation conversion
detectron2/                                   # Detectron2 (vendored dependency)
```

---

## License

The majority of this project is licensed under the [MIT License](LICENSE).

Portions are available under separate terms:
- Swin-Transformer: [MIT License](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE)
- Deformable-DETR: [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)
- Detectron2: [Apache-2.0 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)

---

## Citation

If you use this code, please cite:

```bibtex
@article{chhun2026beacon,
  title={Boundary-Enhanced and Content-Adaptive Query Framework for Transparent Object Instance Segmentation (BEACON)},
  author={Chhun Rotanakkosal and KongVungsovanreach and Nayyar Anand and Kim Tae-Kyung},
  journal={},
  year={2026}
}
```

This work builds on Mask2Former:

```bibtex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  booktitle={CVPR},
  year={2022}
}
```

---

## Acknowledgement

The authors acknowledge the AI Convergence Lab at Chungbuk National University for providing computing resources. This implementation is built on [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [Detectron2](https://github.com/facebookresearch/detectron2).
