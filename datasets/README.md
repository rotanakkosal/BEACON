# Dataset Preparation

## ClearPose

1. Download the ClearPose dataset from [https://github.com/opipari/ClearPose](https://github.com/opipari/ClearPose)
2. We use the **heavy-occlusion subset** (set1--set9): 2,878 training images and 773 validation images
3. Place the set folders under `datasets/clearpose_dataset/`
4. Generate COCO-format annotations:

```bash
python src/dataset-preparation/convert_clearpose_split.py
```

This produces `coco_clearpose_train.json` and `coco_clearpose_val.json`.

Expected structure:
```
clearpose_dataset/
  coco_clearpose_train.json
  coco_clearpose_val.json
  set1/scene1/000000-color.png, 000000-label.png, ...
  set1/scene2/...
  ...
  set9/...
```

Train sets: set1--set6 | Validation sets: set7--set9

## Trans10K-v2

1. Download the Trans10K-v2 dataset from [https://github.com/xieenze/SegmentTransparentObjects](https://github.com/xieenze/SegmentTransparentObjects)
2. Convert to COCO instance segmentation format
3. Place under `datasets/trans10k/`

Expected structure:
```
trans10k/
  coco_trans10k_train.json
  coco_trans10k_val.json
  images/
    train/...
    validation/...
```
