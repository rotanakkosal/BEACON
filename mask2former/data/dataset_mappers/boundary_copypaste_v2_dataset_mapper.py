# Copyright (c) 2025
# Boundary Copy-Paste Dataset Mapper v2 - Simplified (No LSJ)
"""
Simplified Copy-Paste augmentation with Boundary Supervision.

FIXES from v1:
- BUG FIX: category_id now preserved from source (was hardcoded to 0, should be 1)
- Uses SAME augmentations as D2 (ResizeShortestEdge + RandomCrop + RandomFlip)
- NO LSJ (which was causing performance degradation)
- Copy-Paste applied BEFORE standard augmentations

This mapper extends the working D2 pipeline with Copy-Paste only.
"""

import copy
import logging
import os
import random
import numpy as np
import torch
import cv2

import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

from ..boundary_targets import BoundaryTargetGenerator

__all__ = ["BoundaryCopyPasteV2DatasetMapper"]


class BoundaryCopyPasteV2DatasetMapper:
    """
    Simplified Copy-Paste + Boundary Supervision mapper.

    Key differences from v1:
    - Uses SAME augmentations as D2 (no LSJ)
    - Fixes category_id bug (was 0, now preserved from source)
    - Simpler implementation
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations,
        image_format: str,
        size_divisibility: int,
        # Copy-Paste params
        copy_paste_prob: float = 0.5,
        max_paste_objects: int = 5,
        blend_sigma: float = 2.0,
        same_scene_only: bool = True,
        # Small object params
        small_object_threshold: int = 1024,
        small_object_priority: float = 0.8,
        min_object_size: int = 25,
        occlusion_threshold: float = 0.20,
        # Small boost mode params
        small_boost_prob: float = 0.25,
        small_boost_quantile: float = 0.30,
        small_boost_min: int = 2,
        small_boost_max: int = 4,
        small_boost_max_large: int = 1,
        # Boundary params
        boundary_dilation_radius: int = 2,
        contact_dilation: int = 1,
        handle_overlaps: bool = True,
        boundary_band_radius: int = 3,
        # Dataset
        dataset_name: str = None,
        # Depth edges
        use_depth_edges: bool = False,
        depth_max_mm: float = 2000.0,
        # Depth fusion
        use_depth_fusion: bool = False,
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations  # Same as D2!
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        # Copy-Paste params
        self.copy_paste_prob = copy_paste_prob
        self.max_paste_objects = max_paste_objects
        self.blend_sigma = blend_sigma
        self.same_scene_only = same_scene_only

        # Small object params
        self.small_object_threshold = small_object_threshold
        self.small_object_priority = small_object_priority
        self.min_object_size = min_object_size
        self.occlusion_threshold = occlusion_threshold

        # Small boost mode params
        self.small_boost_prob = small_boost_prob
        self.small_boost_quantile = small_boost_quantile
        self.small_boost_min = small_boost_min
        self.small_boost_max = small_boost_max
        self.small_boost_max_large = small_boost_max_large

        # Dataset (lazy loaded)
        self.dataset_name = dataset_name

        # Depth edges / depth fusion
        self.use_depth_edges = use_depth_edges
        self.use_depth_fusion = use_depth_fusion
        self.depth_max_mm = depth_max_mm
        self.dataset_dicts = None
        self.scene_to_images = None

        # Boundary target generator
        self.boundary_generator = BoundaryTargetGenerator(
            dilation_radius=boundary_dilation_radius,
            contact_dilation=contact_dilation,
            handle_overlaps=handle_overlaps,
            boundary_band_radius=boundary_band_radius,
        )

        logger = logging.getLogger(__name__)
        logger.info(f"[{self.__class__.__name__}] Copy-Paste + Boundary Supervision v2 (Simplified)")
        logger.info(f"  - Copy-Paste prob: {copy_paste_prob}")
        logger.info(f"  - Same scene only: {same_scene_only}")
        logger.info(f"  - Small object threshold: {small_object_threshold}px")
        logger.info(f"  - Small object priority: {small_object_priority}")
        logger.info(f"  - Small boost mode: {small_boost_prob:.1%} prob, paste {small_boost_min}-{small_boost_max} tiny objects")
        logger.info(f"  - Using D2 augmentations (no LSJ)")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # Build SAME augmentations as D2 (BoundaryInstanceDatasetMapper)
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
            # Copy-Paste params
            "copy_paste_prob": getattr(cfg.INPUT, 'COPY_PASTE_PROB', 0.5),
            "max_paste_objects": getattr(cfg.INPUT, 'MAX_PASTE_OBJECTS', 5),
            "blend_sigma": getattr(cfg.INPUT, 'BLEND_SIGMA', 2.0),
            "same_scene_only": getattr(cfg.INPUT, 'SAME_SCENE_ONLY', True),
            # Small object params
            "small_object_threshold": getattr(cfg.INPUT, 'SMALL_OBJECT_THRESHOLD', 1024),
            "small_object_priority": getattr(cfg.INPUT, 'SMALL_OBJECT_PRIORITY', 0.8),
            "min_object_size": getattr(cfg.INPUT, 'MIN_OBJECT_SIZE_EXTRACT', 25),
            "occlusion_threshold": getattr(cfg.INPUT, 'OCCLUSION_THRESHOLD', 0.20),
            # Small boost mode params
            "small_boost_prob": getattr(cfg.INPUT, 'SMALL_BOOST_PROB', 0.25),
            "small_boost_quantile": getattr(cfg.INPUT, 'SMALL_BOOST_QUANTILE', 0.30),
            "small_boost_min": getattr(cfg.INPUT, 'SMALL_BOOST_MIN', 2),
            "small_boost_max": getattr(cfg.INPUT, 'SMALL_BOOST_MAX', 4),
            "small_boost_max_large": getattr(cfg.INPUT, 'SMALL_BOOST_MAX_LARGE', 1),
            # Boundary params
            "boundary_dilation_radius": cfg.MODEL.BOUNDARY.DILATION_RADIUS,
            "contact_dilation": cfg.MODEL.BOUNDARY.CONTACT_DILATION,
            "handle_overlaps": cfg.MODEL.BOUNDARY.HANDLE_OVERLAPS,
            "boundary_band_radius": cfg.MODEL.BOUNDARY.BAND_RADIUS,
            # Dataset
            "dataset_name": cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None,
            # Depth edges
            "use_depth_edges": cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES,
            "depth_max_mm": cfg.MODEL.BOUNDARY.DEPTH_MAX_MM,
            # Depth fusion
            "use_depth_fusion": cfg.MODEL.BOUNDARY.USE_DEPTH_FUSION,
        }
        return ret

    def _load_dataset(self, dataset_name):
        """Lazy load dataset and build scene index."""
        if self.dataset_dicts is None:
            self.dataset_dicts = DatasetCatalog.get(dataset_name)

            # Build scene index
            self.scene_to_images = {}
            for d in self.dataset_dicts:
                file_name = d.get("file_name", "")
                parts = file_name.replace("\\", "/").split("/")
                scene = "default"
                for i, part in enumerate(parts):
                    if "clearpose" in part.lower() and i + 2 < len(parts):
                        scene = f"{parts[i+1]}/{parts[i+2]}"
                        break

                if scene not in self.scene_to_images:
                    self.scene_to_images[scene] = []
                self.scene_to_images[scene].append(d)

            logger = logging.getLogger(__name__)
            logger.info(f"  - Built scene index: {len(self.scene_to_images)} scenes")

    def _get_source_image(self, target_dict):
        """Get a source image for copy-paste (same scene if required)."""
        if self.same_scene_only:
            file_name = target_dict.get("file_name", "")
            parts = file_name.replace("\\", "/").split("/")
            scene = "default"
            for i, part in enumerate(parts):
                if "clearpose" in part.lower() and i + 2 < len(parts):
                    scene = f"{parts[i+1]}/{parts[i+2]}"
                    break

            scene_images = self.scene_to_images.get(scene, [])
            if len(scene_images) > 1:
                candidates = [d for d in scene_images if d["image_id"] != target_dict["image_id"]]
                if candidates:
                    return random.choice(candidates)

        return random.choice(self.dataset_dicts)

    def _decode_segmentation(self, segm, height, width):
        """Decode segmentation to binary mask."""
        if isinstance(segm, list):
            return polygons_to_bitmask(segm, height, width)
        elif isinstance(segm, dict):
            return mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            return segm
        else:
            raise ValueError(f"Unknown segmentation format: {type(segm)}")

    def _extract_objects(self, dataset_dict):
        """Extract objects from dataset dict."""
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        height, width = image.shape[:2]

        objects = []
        for anno in dataset_dict.get("annotations", []):
            if "segmentation" not in anno:
                continue
            if anno.get("iscrowd", 0) == 1:
                continue

            mask = self._decode_segmentation(anno["segmentation"], height, width)

            if mask.sum() < self.min_object_size:
                continue

            objects.append({
                "mask": mask.astype(np.uint8),
                "category_id": anno.get("category_id", 1),  # FIX: Default to 1, not 0
                "area": mask.sum(),
            })

        return image, objects

    def _alpha_blend(self, target_img, source_img, mask):
        """Alpha blend source onto target using mask."""
        alpha = mask.astype(np.float32)
        if self.blend_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), self.blend_sigma)
        alpha = np.clip(alpha, 0, 1)
        alpha_3d = alpha[:, :, np.newaxis]
        result = source_img.astype(np.float32) * alpha_3d + target_img.astype(np.float32) * (1 - alpha_3d)
        return result.astype(np.uint8)

    def _apply_copy_paste(self, target_img, target_annos, source_img, source_objects):
        """Apply copy-paste augmentation with small object prioritization."""
        if not source_objects:
            return target_img, target_annos, None

        h, w = target_img.shape[:2]
        src_h, src_w = source_img.shape[:2]

        # Resize source to match target
        if (src_h, src_w) != (h, w):
            source_img = cv2.resize(source_img, (w, h))
            for obj in source_objects:
                obj["mask"] = cv2.resize(obj["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
                obj["area"] = obj["mask"].sum()

        # ========================================
        # 1) SELECT CANDIDATE INSTANCES TO PASTE
        # ========================================
        small_objects = [obj for obj in source_objects if obj["area"] < self.small_object_threshold]
        large_objects = [obj for obj in source_objects if obj["area"] >= self.small_object_threshold]

        selected = []

        # ----------------------------------------
        # Small-object boost mode (targets recall)
        # ----------------------------------------
        # Defaults for small boost mode
        small_boost_prob = getattr(self, "small_boost_prob", 0.25)
        small_boost_quantile = getattr(self, "small_boost_quantile", 0.30)
        small_boost_min = getattr(self, "small_boost_min", 2)
        small_boost_max = getattr(self, "small_boost_max", 4)
        small_boost_max_large = getattr(self, "small_boost_max_large", 1)

        in_boost_mode = (small_objects and (random.random() < small_boost_prob))

        # ========================================
        # 2) DECIDE HOW MANY INSTANCES TO PASTE
        # 3) APPLY PASTE PROBABILITY / MODE
        # ========================================
        if in_boost_mode:
            # Focus on the tiniest small objects (bottom quantile by area)
            small_sorted = sorted(small_objects, key=lambda o: o["area"])
            k = max(1, int(len(small_sorted) * small_boost_quantile))
            tiny_pool = small_sorted[:k]

            num_small = random.randint(
                min(small_boost_min, len(tiny_pool)),
                min(small_boost_max, len(tiny_pool))
            )
            selected.extend(random.sample(tiny_pool, num_small))

            # Optionally add at most 0–1 large objects (keep boost focused)
            if large_objects and len(selected) < self.max_paste_objects:
                num_large = random.randint(0, min(small_boost_max_large, len(large_objects)))
                if num_large > 0:
                    selected.extend(random.sample(large_objects, num_large))

        else:
            # ---- Normal V2b behavior (keep as-is) ----
            if small_objects and random.random() < self.small_object_priority:
                num_small = random.randint(1, min(3, len(small_objects)))  # Paste 1-3 small objects
                selected.extend(random.sample(small_objects, num_small))

            remaining = self.max_paste_objects - len(selected)
            if remaining > 0 and large_objects:
                num_large = random.randint(0, min(remaining, len(large_objects)))
                if num_large > 0:
                    selected.extend(random.sample(large_objects, num_large))

        # Fallback: if still empty, randomly select from all
        if not selected and source_objects:
            num_to_paste = random.randint(1, min(self.max_paste_objects, len(source_objects)))
            selected = random.sample(source_objects, num_to_paste)

        if not selected:
            return target_img, target_annos, None

        # Create combined mask
        pasted_combined = np.zeros((h, w), dtype=np.uint8)
        for obj in selected:
            pasted_combined = np.maximum(pasted_combined, obj["mask"])

        # Alpha blend
        result_img = self._alpha_blend(target_img, source_img, pasted_combined)

        # Update target annotations (handle occlusion)
        updated_annos = []
        for anno in target_annos:
            if "segmentation" not in anno:
                continue

            mask = self._decode_segmentation(anno["segmentation"], h, w)
            original_area = mask.sum()

            if original_area == 0:
                continue

            # Subtract pasted region
            updated_mask = mask.copy().astype(np.uint8)
            updated_mask[pasted_combined > 0] = 0
            remaining_area = updated_mask.sum()

            # Keep if >20% visible
            if remaining_area > original_area * self.occlusion_threshold:
                new_anno = copy.deepcopy(anno)
                new_anno["segmentation"] = updated_mask
                updated_annos.append(new_anno)

        # Add pasted object annotations
        for obj in selected:
            if obj["mask"].sum() >= self.min_object_size:
                updated_annos.append({
                    "segmentation": obj["mask"],
                    "category_id": obj["category_id"],  # FIX: Use source category_id (1 for ClearPose)
                    "iscrowd": 0,
                })

        return result_img, updated_annos, pasted_combined

    def __call__(self, dataset_dict):
        """Process one training sample."""
        assert self.is_train, "BoundaryCopyPasteV2DatasetMapper is training-only!"

        dataset_dict = copy.deepcopy(dataset_dict)

        # Lazy load dataset
        if self.dataset_dicts is None and self.dataset_name:
            try:
                self._load_dataset(self.dataset_name)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load dataset: {e}")

        # Load target image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        annotations = dataset_dict.get("annotations", [])

        # Apply Copy-Paste BEFORE standard augmentations
        paste_mask = None  # Track pasted regions for depth zeroing
        if self.dataset_dicts is not None and random.random() < self.copy_paste_prob:
            try:
                source_dict = self._get_source_image(dataset_dict)
                source_img, source_objects = self._extract_objects(source_dict)

                if source_objects:
                    image, annotations, paste_mask = self._apply_copy_paste(
                        image, annotations, source_img, source_objects
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(f"Copy-paste failed: {e}")

        # Apply SAME augmentations as D2
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        # Transform paste_mask through same augmentations (for depth zeroing)
        if paste_mask is not None:
            paste_mask = transforms.apply_segmentation(paste_mask)

        # Transform annotations (same as D2)
        for anno in annotations:
            anno.pop("keypoints", None)

        # Process annotations
        height, width = image.shape[:2]
        processed_annos = []
        for anno in annotations:
            if "segmentation" not in anno:
                continue
            if anno.get("iscrowd", 0) == 1:
                continue

            segm = anno["segmentation"]
            if isinstance(segm, np.ndarray):
                # Already a mask (from copy-paste)
                mask = segm
                # Resize to match original image size for transform
                if mask.shape != (dataset_dict["height"], dataset_dict["width"]):
                    mask = cv2.resize(mask, (dataset_dict["width"], dataset_dict["height"]),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                mask = self._decode_segmentation(segm, dataset_dict["height"], dataset_dict["width"])

            # Apply transforms
            mask = transforms.apply_segmentation(mask.astype(np.float32))
            mask = (mask > 0.5).astype(np.uint8)

            if mask.sum() < self.min_object_size:
                continue

            processed_annos.append({
                "mask": mask,
                "category_id": anno.get("category_id", 1),  # FIX: Default to 1
            })

        # Convert to tensors
        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        masks = [a["mask"] for a in processed_annos]
        classes = [a["category_id"] for a in processed_annos]

        # Generate boundary targets
        if len(masks) > 0:
            masks_np = np.stack(masks, axis=0)
        else:
            masks_np = np.zeros((0, height, width), dtype=np.uint8)

        boundary_targets = self.boundary_generator(masks_np, image_size=(height, width))

        # Prepare output
        dataset_dict["image"] = image_tensor

        instances = Instances((height, width))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) > 0:
            masks_tensor = torch.stack([torch.from_numpy(m) for m in masks])
            instances.gt_masks = masks_tensor
        else:
            instances.gt_masks = torch.zeros((0, height, width), dtype=torch.uint8)

        dataset_dict["instances"] = instances

        # Boundary targets
        for key, value in boundary_targets.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            dataset_dict[key] = value.float()

        # Load sensor depth for depth edge computation or depth fusion
        if self.use_depth_edges or self.use_depth_fusion:
            depth_path = dataset_dict["file_name"].replace("-color.png", "-depth.png")
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    depth = depth.astype(np.float32)
                    # Apply same spatial augmentations as image
                    depth = transforms.apply_image(depth)
                    # Normalize to [0, 1]
                    depth = np.clip(depth / self.depth_max_mm, 0.0, 1.0)

                    # Zero depth at copy-paste regions with blurred edges.
                    # Rationale: All ClearPose objects are transparent, and depth
                    # sensors fail on transparent objects. So pasted transparent
                    # objects SHOULD have depth=0. Blur simulates the gradual
                    # depth dropout at sensor boundaries (~5px uncertainty zone).
                    if paste_mask is not None and self.use_depth_fusion:
                        blur_mask = cv2.GaussianBlur(
                            paste_mask.astype(np.float32), (11, 11), 3.0
                        )
                        blur_mask = np.clip(blur_mask, 0.0, 1.0)
                        depth = depth * (1.0 - blur_mask)
                    dataset_dict["depth"] = torch.from_numpy(depth).float().unsqueeze(0)  # [1, H, W]
            if "depth" not in dataset_dict:
                # Fallback: zero depth if file missing
                dataset_dict["depth"] = torch.zeros(1, height, width, dtype=torch.float32)

        return dataset_dict
