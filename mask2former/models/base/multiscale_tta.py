# Copyright (c) 2025
# Multi-Scale Test-Time Augmentation for Instance Segmentation
"""
Multi-scale TTA for Mask2Former instance segmentation.

Key insight for small object detection:
- Small objects are often missed at normal resolution
- By running inference at larger scales (1.5x, 2.0x), small objects become
  larger and more likely to be detected
- Predictions from all scales are merged using NMS

Usage:
    model = MultiScaleInstanceSegmentorTTA(cfg, model, scales=[1.0, 1.5, 2.0])
    results = model(batched_inputs)
"""

import copy
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.structures import Instances, Boxes

logger = logging.getLogger(__name__)

__all__ = ["MultiScaleInstanceSegmentorTTA", "MultiScaleInstanceSegmentorTTAFast", "build_multiscale_tta"]


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding boxes from binary masks.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        boxes: (N, 4) bounding boxes in (x1, y1, x2, y2) format
    """
    if masks.numel() == 0:
        return torch.zeros(0, 4, device=masks.device)

    n = masks.shape[0]
    boxes = torch.zeros(n, 4, device=masks.device)

    for i in range(n):
        mask = masks[i]
        if mask.sum() == 0:
            continue

        # Find non-zero indices
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)

        if rows.any() and cols.any():
            y_indices = torch.where(rows)[0]
            x_indices = torch.where(cols)[0]

            boxes[i, 0] = x_indices[0]  # x1
            boxes[i, 1] = y_indices[0]  # y1
            boxes[i, 2] = x_indices[-1] + 1  # x2
            boxes[i, 3] = y_indices[-1] + 1  # y2

    return boxes


def mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Perform NMS on instance masks using IoU.

    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        classes: (N,) class labels
        iou_threshold: IoU threshold for NMS

    Returns:
        keep: indices of masks to keep
    """
    if masks.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=masks.device)

    # Sort by score
    order = scores.argsort(descending=True)
    masks = masks[order]
    scores = scores[order]
    classes = classes[order]

    keep = []
    while len(order) > 0:
        # Keep the highest scoring mask
        keep.append(order[0].item())

        if len(order) == 1:
            break

        # Compute IoU with remaining masks
        mask_i = masks[0:1].float()  # (1, H, W)
        masks_rest = masks[1:].float()  # (N-1, H, W)

        intersection = (mask_i * masks_rest).sum(dim=(1, 2))
        union = mask_i.sum() + masks_rest.sum(dim=(1, 2)) - intersection
        iou = intersection / (union + 1e-6)

        # Also check class - only suppress same class
        same_class = classes[1:] == classes[0]
        suppress = (iou > iou_threshold) & same_class

        # Keep masks that are not suppressed
        remaining = ~suppress
        order = order[1:][remaining]
        masks = masks[1:][remaining]
        scores = scores[1:][remaining]
        classes = classes[1:][remaining]

    return torch.tensor(keep, dtype=torch.long, device=masks.device)


def box_nms_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Fast NMS using torchvision's batched_nms on bounding boxes.
    Much faster than mask-based NMS.

    Args:
        boxes: (N, 4) bounding boxes in (x1, y1, x2, y2) format
        scores: (N,) confidence scores
        classes: (N,) class labels
        iou_threshold: IoU threshold for NMS

    Returns:
        keep: indices of boxes to keep
    """
    from torchvision.ops import batched_nms
    return batched_nms(boxes, scores, classes, iou_threshold)


class MultiScaleInstanceSegmentorTTA(nn.Module):
    """
    Multi-scale Test-Time Augmentation for instance segmentation.

    Runs inference at multiple scales and merges predictions using NMS.
    Particularly useful for detecting small objects.
    """

    def __init__(
        self,
        cfg,
        model,
        scales: List[float] = None,
        flip: bool = True,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        """
        Args:
            cfg: Detectron2 config
            model: Instance segmentation model
            scales: List of scales to use (default: [1.0, 1.5, 2.0])
            flip: Whether to also use horizontal flip
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to return
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module

        self.cfg = cfg.clone()
        self.model = model
        self.scales = scales if scales is not None else [1.0, 1.5, 2.0]
        self.flip = flip
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        logger.info(
            f"[MultiScaleTTA] Initialized with scales={self.scales}, "
            f"flip={self.flip}, nms_threshold={self.nms_threshold}"
        )

    @property
    def device(self):
        return self.model.device

    def __call__(self, batched_inputs):
        """
        Run multi-scale TTA inference.

        Args:
            batched_inputs: List of input dicts with "image" key

        Returns:
            List of output dicts with "instances" key
        """
        results = []
        for x in batched_inputs:
            result = self._inference_one_image(x)
            results.append(result)
        return results

    def _inference_one_image(self, input_dict: dict) -> dict:
        """
        Run multi-scale inference on a single image.

        Args:
            input_dict: Dict with "image" (C, H, W), "height", "width" keys

        Returns:
            Dict with "instances" key
        """
        image = input_dict["image"]  # (C, H, W)
        orig_height = input_dict.get("height", image.shape[1])
        orig_width = input_dict.get("width", image.shape[2])

        # Collect predictions from all scales
        all_masks = []
        all_scores = []
        all_classes = []

        for scale in self.scales:
            # Get predictions at this scale
            masks, scores, classes = self._inference_at_scale(
                input_dict, scale, flip=False
            )
            if masks is not None:
                all_masks.append(masks)
                all_scores.append(scores)
                all_classes.append(classes)

            # Also try with horizontal flip
            if self.flip:
                masks, scores, classes = self._inference_at_scale(
                    input_dict, scale, flip=True
                )
                if masks is not None:
                    all_masks.append(masks)
                    all_scores.append(scores)
                    all_classes.append(classes)

        # Merge predictions from all scales
        if len(all_masks) == 0:
            # No detections
            result = Instances((orig_height, orig_width))
            result.pred_masks = torch.empty(0, orig_height, orig_width, device=self.device)
            result.pred_boxes = Boxes(torch.empty(0, 4, device=self.device))
            result.scores = torch.empty(0, device=self.device)
            result.pred_classes = torch.empty(0, dtype=torch.long, device=self.device)
            return {"instances": result}

        # Concatenate all predictions
        all_masks = torch.cat(all_masks, dim=0)  # (N_total, H, W)
        all_scores = torch.cat(all_scores, dim=0)  # (N_total,)
        all_classes = torch.cat(all_classes, dim=0)  # (N_total,)

        # Apply NMS to remove duplicates
        keep = mask_nms(all_masks, all_scores, all_classes, self.nms_threshold)

        # Limit to max detections
        if len(keep) > self.max_detections:
            # Sort by score and keep top-k
            kept_scores = all_scores[keep]
            topk = kept_scores.argsort(descending=True)[:self.max_detections]
            keep = keep[topk]

        # Build final result
        final_masks = all_masks[keep]
        result = Instances((orig_height, orig_width))
        result.pred_masks = final_masks
        result.pred_boxes = Boxes(masks_to_boxes(final_masks))
        result.scores = all_scores[keep]
        result.pred_classes = all_classes[keep]

        return {"instances": result}

    def _inference_at_scale(
        self,
        input_dict: dict,
        scale: float,
        flip: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run inference at a specific scale.

        Args:
            input_dict: Original input dict
            scale: Scale factor (1.0 = original size)
            flip: Whether to horizontally flip

        Returns:
            (masks, scores, classes) or (None, None, None) if no detections
        """
        image = input_dict["image"]  # (C, H, W)
        orig_height = input_dict.get("height", image.shape[1])
        orig_width = input_dict.get("width", image.shape[2])

        # Scale the image
        if scale != 1.0:
            new_h = int(image.shape[1] * scale)
            new_w = int(image.shape[2] * scale)
            scaled_image = F.interpolate(
                image.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            scaled_image = image
            new_h, new_w = image.shape[1], image.shape[2]

        # Flip if requested
        if flip:
            scaled_image = torch.flip(scaled_image, dims=[2])  # Flip width dimension

        # Create input for model
        scaled_input = {
            "image": scaled_image,
            "height": new_h,
            "width": new_w,
        }

        # Add other fields from original input (like file_name for debugging)
        for k, v in input_dict.items():
            if k not in scaled_input:
                scaled_input[k] = v

        # Run model inference
        with torch.no_grad():
            outputs = self.model([scaled_input])

        if "instances" not in outputs[0]:
            return None, None, None

        instances = outputs[0]["instances"]

        if len(instances) == 0:
            return None, None, None

        # Get predictions
        masks = instances.pred_masks  # (N, H_scaled, W_scaled) or (N, 1, H_scaled, W_scaled)
        scores = instances.scores
        classes = instances.pred_classes

        # Handle (N, 1, H, W) format - squeeze channel dimension
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Resize masks back to original resolution
        if scale != 1.0 or masks.shape[-2] != orig_height or masks.shape[-1] != orig_width:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            masks = (masks > 0.5).float()

        # Flip masks back if we flipped the input
        if flip:
            masks = torch.flip(masks, dims=[2])

        return masks, scores, classes


class MultiScaleInstanceSegmentorTTAFast(MultiScaleInstanceSegmentorTTA):
    """
    Optimized multi-scale TTA with faster NMS and optional FP16.

    Optimizations:
    1. Box-based NMS (faster than mask-based)
    2. Optional FP16 inference
    3. Single upscale option (skip 1.0× scale)
    """

    def __init__(
        self,
        cfg,
        model,
        scales: List[float] = None,
        flip: bool = False,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        use_fast_nms: bool = True,
        use_fp16: bool = False,
        upscale_only: bool = False,
    ):
        # If upscale_only, remove 1.0 scale
        if upscale_only and scales is not None:
            scales = [s for s in scales if s > 1.0]
            if len(scales) == 0:
                scales = [1.5]  # Default to 1.5× if no scales > 1.0

        super().__init__(cfg, model, scales, flip, nms_threshold, max_detections)
        self.use_fast_nms = use_fast_nms
        self.use_fp16 = use_fp16
        self.upscale_only = upscale_only

        logger.info(
            f"[MultiScaleTTAFast] use_fast_nms={use_fast_nms}, "
            f"use_fp16={use_fp16}, upscale_only={upscale_only}"
        )

    def _inference_one_image(self, input_dict: dict) -> dict:
        """Optimized inference with optional FP16 and fast NMS."""
        image = input_dict["image"]
        orig_height = input_dict.get("height", image.shape[1])
        orig_width = input_dict.get("width", image.shape[2])

        all_masks = []
        all_scores = []
        all_classes = []

        # Use autocast for FP16 if enabled
        autocast_ctx = torch.cuda.amp.autocast() if self.use_fp16 else torch.no_grad()

        with autocast_ctx:
            for scale in self.scales:
                masks, scores, classes = self._inference_at_scale(
                    input_dict, scale, flip=False
                )
                if masks is not None:
                    all_masks.append(masks)
                    all_scores.append(scores)
                    all_classes.append(classes)

                if self.flip:
                    masks, scores, classes = self._inference_at_scale(
                        input_dict, scale, flip=True
                    )
                    if masks is not None:
                        all_masks.append(masks)
                        all_scores.append(scores)
                        all_classes.append(classes)

        if len(all_masks) == 0:
            result = Instances((orig_height, orig_width))
            result.pred_masks = torch.empty(0, orig_height, orig_width, device=self.device)
            result.pred_boxes = Boxes(torch.empty(0, 4, device=self.device))
            result.scores = torch.empty(0, device=self.device)
            result.pred_classes = torch.empty(0, dtype=torch.long, device=self.device)
            return {"instances": result}

        all_masks = torch.cat(all_masks, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_classes = torch.cat(all_classes, dim=0)

        # Use fast box-based NMS instead of slow mask NMS
        if self.use_fast_nms:
            boxes = masks_to_boxes(all_masks)
            keep = box_nms_fast(boxes, all_scores, all_classes, self.nms_threshold)
        else:
            keep = mask_nms(all_masks, all_scores, all_classes, self.nms_threshold)

        if len(keep) > self.max_detections:
            kept_scores = all_scores[keep]
            topk = kept_scores.argsort(descending=True)[:self.max_detections]
            keep = keep[topk]

        final_masks = all_masks[keep]
        result = Instances((orig_height, orig_width))
        result.pred_masks = final_masks
        result.pred_boxes = Boxes(masks_to_boxes(final_masks))
        result.scores = all_scores[keep]
        result.pred_classes = all_classes[keep]

        return {"instances": result}


def build_multiscale_tta(cfg, model, fast=False, **kwargs):
    """
    Build multi-scale TTA wrapper from config.

    Config options (under TEST.AUG):
        SCALES: List of scales (default: [1.0, 1.5, 2.0])
        FLIP: Whether to use horizontal flip (default: True)
        NMS_THRESHOLD: IoU threshold for NMS (default: 0.5)
        MAX_DETECTIONS: Max detections per image (default: 100)

    Args:
        fast: If True, use optimized TTA with fast NMS
        **kwargs: Additional args for fast TTA (use_fp16, upscale_only)
    """
    scales = getattr(cfg.TEST.AUG, "SCALES", [1.0, 1.5, 2.0])
    flip = getattr(cfg.TEST.AUG, "FLIP", True)
    nms_threshold = getattr(cfg.TEST.AUG, "NMS_THRESHOLD", 0.5)
    max_detections = getattr(cfg.TEST.AUG, "MAX_DETECTIONS", 100)

    if fast:
        return MultiScaleInstanceSegmentorTTAFast(
            cfg,
            model,
            scales=scales,
            flip=flip,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            **kwargs,
        )
    else:
        return MultiScaleInstanceSegmentorTTA(
            cfg,
            model,
            scales=scales,
            flip=flip,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )
