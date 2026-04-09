#!/usr/bin/env python
"""
Train Mask R-CNN baseline on ClearPose dataset.

This script is for comparison with our Mask2Former boundary supervision approach.

Usage:
    python train-set/train_mask_rcnn_baseline.py --num-gpus 1

    # Evaluation only:
    python train-set/train_mask_rcnn_baseline.py --eval-only \
        MODEL.WEIGHTS ./output/mask_rcnn_baseline/model_final.pth
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

# Register ClearPose dataset
_DATASET_ROOT = os.path.join(project_root, "datasets", "clearpose_dataset")

if "clearpose_train" not in DatasetCatalog.list():
    register_coco_instances(
        "clearpose_train",
        {},
        os.path.join(_DATASET_ROOT, "coco_clearpose_train.json"),
        _DATASET_ROOT
    )
if "clearpose_val" not in DatasetCatalog.list():
    register_coco_instances(
        "clearpose_val",
        {},
        os.path.join(_DATASET_ROOT, "coco_clearpose_val.json"),
        _DATASET_ROOT
    )

# Register Trans10K dataset
_TRANS10K_ROOT = os.path.join(project_root, "datasets", "trans10k")

if "trans10k_train" not in DatasetCatalog.list():
    register_coco_instances(
        "trans10k_train",
        {},
        os.path.join(_TRANS10K_ROOT, "coco_trans10k_train.json"),
        os.path.join(_TRANS10K_ROOT, "images")
    )
if "trans10k_val" not in DatasetCatalog.list():
    register_coco_instances(
        "trans10k_val",
        {},
        os.path.join(_TRANS10K_ROOT, "coco_trans10k_val.json"),
        os.path.join(_TRANS10K_ROOT, "images")
    )


class MaskRCNNTrainer(DefaultTrainer):
    """Simple trainer for Mask R-CNN baseline."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = MaskRCNNTrainer.build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MaskRCNNTrainer.test(cfg, model)
        return res

    trainer = MaskRCNNTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.set_defaults(config_file="configs/clearpose/mask_rcnn_baseline.yaml")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
