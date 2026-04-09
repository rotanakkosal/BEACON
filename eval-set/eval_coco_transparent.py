"""
Evaluate U1 model on COCO transparent object subset.

Filters COCO val2017 for transparent categories (bottle, wine_glass, cup),
remaps them to a single 'transparent_object' class (matching ClearPose),
and runs inference + COCO AP evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval-set/eval_coco_transparent.py \
        --config-file configs/clearpose/boundary_supervision/beacon_clearpose.yaml \
        --checkpoint output/boundary_supervision/setting_u1_decoder_boundary/model_0011999.pth
"""

import os
import sys
import json
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
except:
    pass

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former.configs.config import add_maskformer2_config
from mask2former.configs.config_boundary_supervision import add_boundary_supervision_config
from mask2former.models.boundary.maskformer_model_boundary_supervision import MaskFormerBoundarySupervision

# COCO transparent category IDs
TRANSPARENT_CATEGORIES = {
    44: "bottle",
    46: "wine glass",
    47: "cup",
}

COCO_ROOT = os.path.join(project_root, "datasets", "coco")
COCO_ANN_FILE = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
FILTERED_ANN_FILE = os.path.join(COCO_ROOT, "annotations", "instances_val2017_transparent.json")


def create_filtered_annotations():
    """
    Filter COCO val2017 annotations to keep only transparent object categories.
    Remap all to category_id=1 (transparent_object) to match ClearPose.
    """
    if os.path.exists(FILTERED_ANN_FILE):
        print(f"Filtered annotation already exists: {FILTERED_ANN_FILE}")
        with open(FILTERED_ANN_FILE, 'r') as f:
            data = json.load(f)
        n_images = len(data["images"])
        n_anns = len(data["annotations"])
        print(f"  {n_images} images, {n_anns} annotations")
        return FILTERED_ANN_FILE

    print(f"Loading COCO annotations from: {COCO_ANN_FILE}")
    with open(COCO_ANN_FILE, 'r') as f:
        coco_data = json.load(f)

    # Filter annotations for transparent categories
    transparent_cat_ids = set(TRANSPARENT_CATEGORIES.keys())
    filtered_anns = []
    image_ids_with_anns = set()

    for ann in coco_data["annotations"]:
        if ann["category_id"] in transparent_cat_ids:
            # Remap to single class (category_id=1, matching ClearPose)
            new_ann = ann.copy()
            new_ann["category_id"] = 1
            filtered_anns.append(new_ann)
            image_ids_with_anns.add(ann["image_id"])

    # Keep only images that have at least one transparent object
    filtered_images = [
        img for img in coco_data["images"]
        if img["id"] in image_ids_with_anns
    ]

    # Create new annotation file
    filtered_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": [{"id": 1, "name": "transparent_object", "supercategory": "object"}],
    }

    with open(FILTERED_ANN_FILE, 'w') as f:
        json.dump(filtered_data, f)

    print(f"Filtered annotations saved to: {FILTERED_ANN_FILE}")
    print(f"  Original: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"  Filtered: {len(filtered_images)} images, {len(filtered_anns)} annotations")
    print(f"  Categories: {', '.join(f'{v} (COCO ID {k})' for k, v in TRANSPARENT_CATEGORIES.items())}")
    print(f"  All remapped to category_id=1 (transparent_object)")

    return FILTERED_ANN_FILE


def register_dataset():
    """Register filtered COCO transparent dataset."""
    dataset_name = "coco_transparent_val"
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(
            dataset_name,
            {},
            FILTERED_ANN_FILE,
            os.path.join(COCO_ROOT, "val2017"),
        )
    return dataset_name


def setup_cfg(args):
    """Set up config for evaluation."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_boundary_supervision_config(cfg)

    cfg.merge_from_file(args.config_file)

    # Override for evaluation
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.DATASETS.TEST = ("coco_transparent_val",)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = os.path.join(
        os.path.dirname(args.checkpoint), "coco_transparent_eval"
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate on COCO transparent subset")
    parser.add_argument(
        "--config-file",
        default="configs/clearpose/boundary_supervision/beacon_clearpose.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/boundary_supervision/setting_u1_decoder_boundary/model_0011999.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    # Step 1: Create filtered annotations
    print("=" * 60)
    print("Step 1: Filter COCO annotations for transparent objects")
    print("=" * 60)
    create_filtered_annotations()

    # Step 2: Register dataset
    print("\n" + "=" * 60)
    print("Step 2: Register filtered dataset")
    print("=" * 60)
    dataset_name = register_dataset()
    metadata = MetadataCatalog.get(dataset_name)
    print(f"  Dataset: {dataset_name}")
    print(f"  thing_classes: {metadata.get('thing_classes', 'N/A')}")

    # Also register ClearPose (needed for config compatibility)
    _DATASET_ROOT = os.path.join(project_root, "datasets", "clearpose_dataset")
    if "clearpose_train" not in DatasetCatalog.list():
        register_coco_instances(
            "clearpose_train", {},
            f"{_DATASET_ROOT}/coco_clearpose_train.json", _DATASET_ROOT
        )
    if "clearpose_val" not in DatasetCatalog.list():
        register_coco_instances(
            "clearpose_val", {},
            f"{_DATASET_ROOT}/coco_clearpose_val.json", _DATASET_ROOT
        )

    # Step 3: Setup config and model
    print("\n" + "=" * 60)
    print("Step 3: Load model")
    print("=" * 60)
    cfg = setup_cfg(args)
    setup_logger(output=cfg.OUTPUT_DIR, name="mask2former")

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.eval()
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Step 4: Run evaluation
    print("\n" + "=" * 60)
    print("Step 4: Run evaluation on COCO transparent subset")
    print("=" * 60)
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=cfg.OUTPUT_DIR,
    )
    data_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, data_loader, evaluator)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: U1 on COCO Transparent Object Subset")
    print("=" * 60)
    print(f"Categories evaluated: {list(TRANSPARENT_CATEGORIES.values())}")
    print(f"(all mapped to single 'transparent_object' class)")
    print()
    if "segm" in results:
        segm = results["segm"]
        print(f"  AP    = {segm['AP']:.2f}")
        print(f"  AP50  = {segm['AP50']:.2f}")
        print(f"  AP75  = {segm['AP75']:.2f}")
        print(f"  APs   = {segm['APs']:.2f}")
        print(f"  APm   = {segm['APm']:.2f}")
        print(f"  APl   = {segm['APl']:.2f}")
    if "bbox" in results:
        bbox = results["bbox"]
        print(f"\n  Box AP    = {bbox['AP']:.2f}")
        print(f"  Box AP50  = {bbox['AP50']:.2f}")
        print(f"  Box AP75  = {bbox['AP75']:.2f}")

    print("\n" + "=" * 60)
    print(f"Full results saved to: {cfg.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
