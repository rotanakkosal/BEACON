# Copyright (c) 2025
# Boundary Instance Dataset Mapper for Boundary Supervision Approach
"""
Dataset mapper that adds boundary targets to the standard instance segmentation pipeline.

Extends MaskFormerInstanceDatasetMapper with:
- FG boundary ground truth
- Contact boundary ground truth
- Boundary band for overlap penalty
- Ignore mask for invalid regions
"""

import copy
import logging
import os
import numpy as np
import torch
import cv2
from torch.nn import functional as F

import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

from ..boundary_targets import BoundaryTargetGenerator

__all__ = ["BoundaryInstanceDatasetMapper"]


class BoundaryInstanceDatasetMapper:
    """
    Dataset mapper for instance segmentation with boundary supervision.

    Extends standard instance mapper with boundary target generation.
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations,
        image_format: str,
        size_divisibility: int,
        # Boundary-specific params
        boundary_dilation_radius: int = 2,
        contact_dilation: int = 1,
        handle_overlaps: bool = True,
        boundary_band_radius: int = 3,
        # Depth edges
        use_depth_edges: bool = False,
        depth_max_mm: float = 2000.0,
    ):
        """
        Args:
            is_train: training or inference mode
            augmentations: list of augmentations
            image_format: image format (RGB, BGR)
            size_divisibility: pad image size to be divisible by this
            boundary_dilation_radius: radius for boundary dilation
            contact_dilation: radius for contact boundary dilation
            handle_overlaps: whether to handle overlapping masks
            boundary_band_radius: radius for boundary band
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        # Depth edges
        self.use_depth_edges = use_depth_edges
        self.depth_max_mm = depth_max_mm

        # Initialize boundary target generator
        self.boundary_generator = BoundaryTargetGenerator(
            dilation_radius=boundary_dilation_radius,
            contact_dilation=contact_dilation,
            handle_overlaps=handle_overlaps,
            boundary_band_radius=boundary_band_radius,
        )

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Using boundary supervision in {mode}")
        logger.info(f"[{self.__class__.__name__}] Augmentations: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            # Boundary params from config
            "boundary_dilation_radius": cfg.MODEL.BOUNDARY.DILATION_RADIUS,
            "contact_dilation": cfg.MODEL.BOUNDARY.CONTACT_DILATION,
            "handle_overlaps": cfg.MODEL.BOUNDARY.HANDLE_OVERLAPS,
            "boundary_band_radius": cfg.MODEL.BOUNDARY.BAND_RADIUS,
            # Depth edges
            "use_depth_edges": cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES,
            "depth_max_mm": cfg.MODEL.BOUNDARY.DEPTH_MAX_MM,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Process one image and its annotations.

        Args:
            dataset_dict: Detectron2 dataset dict

        Returns:
            dict with image, instances, and boundary targets
        """
        assert self.is_train, "BoundaryInstanceDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Apply augmentations
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        # Transform instance annotations
        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        # Convert segmentations to masks
        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2
                masks.append(segm)
            else:
                raise ValueError(
                    f"Cannot convert segmentation of type '{type(segm)}' to BitMasks!"
                )

        # Convert to tensors
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks_np = np.array(masks) if len(masks) > 0 else np.zeros((0, image.shape[-2], image.shape[-1]))
        masks_tensor = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Generate boundary targets
        # NOTE: No padding here - Mask2Former handles padding via ImageList.from_tensors
        # Boundary targets will be resized to match model output in BoundaryCriterion
        boundary_targets = self.boundary_generator(
            masks_np,
            image_size=(image.shape[-2], image.shape[-1])
        )

        # Convert boundary targets to tensors
        for key in boundary_targets:
            if isinstance(boundary_targets[key], np.ndarray):
                boundary_targets[key] = torch.from_numpy(boundary_targets[key])

        image_shape = (image.shape[-2], image.shape[-1])

        # Store image
        dataset_dict["image"] = image

        # Prepare instances
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks_tensor) == 0:
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks_bitmask = BitMasks(torch.stack(masks_tensor))
            instances.gt_masks = masks_bitmask.tensor

        dataset_dict["instances"] = instances

        # Store boundary targets
        for key, value in boundary_targets.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            dataset_dict[key] = value.float()

        # Load sensor depth for depth edge computation
        if self.use_depth_edges:
            depth_path = dataset_dict["file_name"].replace("-color.png", "-depth.png")
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    depth = depth.astype(np.float32)
                    # Apply same augmentations as image
                    depth = transforms.apply_image(depth)
                    # Normalize to [0, 1]
                    depth = np.clip(depth / self.depth_max_mm, 0.0, 1.0)
                    dataset_dict["depth"] = torch.from_numpy(depth).float().unsqueeze(0)  # [1, H, W]
            if "depth" not in dataset_dict:
                # Fallback: zero depth if file missing
                dataset_dict["depth"] = torch.zeros(1, image_shape[0], image_shape[1], dtype=torch.float32)

        return dataset_dict
