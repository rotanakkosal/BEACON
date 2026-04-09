"""
mask_nms.py
Mask-based NMS for Mask2Former post-processing

This module provides Mask NMS functionality to filter duplicate
predictions that cover the same object (high mask IoU).
"""

import math
import torch


def mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.

    Args:
        mask1: (H, W) tensor
        mask2: (H, W) tensor

    Returns:
        IoU score (float)
    """
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    return intersection / (union + 1e-6)


def mask_iou_matrix(masks):
    """
    Compute pairwise IoU matrix for all masks.
    More efficient than computing one by one.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        iou_matrix: (N, N) pairwise IoU scores
    """
    n = masks.shape[0]
    masks_flat = masks.view(n, -1).float()  # (N, H*W)

    # Compute intersection: (N, N)
    intersection = torch.mm(masks_flat, masks_flat.t())

    # Compute areas: (N,)
    areas = masks_flat.sum(dim=1)

    # Compute union: (N, N)
    union = areas.unsqueeze(0) + areas.unsqueeze(1) - intersection

    # IoU
    iou_matrix = intersection / (union + 1e-6)

    return iou_matrix


def mask_nms(masks, scores, iou_threshold=0.5):
    """
    Apply NMS based on mask IoU.

    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_threshold: suppress if IoU > threshold

    Returns:
        keep: list of indices to keep (original indices)
    """
    n = masks.shape[0]
    if n == 0:
        return []

    # Binarize masks
    masks = masks > 0.5

    # Sort by score descending - order contains original indices
    order = scores.argsort(descending=True).tolist()

    keep = []
    suppressed = set()

    for idx_in_order, i in enumerate(order):
        if i in suppressed:
            continue

        keep.append(i)

        # Compare with remaining masks (those after current position in sorted order)
        for j in order[idx_in_order + 1:]:
            if j in suppressed:
                continue

            iou = mask_iou(masks[i], masks[j])

            if iou > iou_threshold:
                suppressed.add(j)

    return keep


def mask_nms_fast(masks, scores, iou_threshold=0.5):
    """
    Fast version of mask NMS using matrix operations.

    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_threshold: suppress if IoU > threshold

    Returns:
        keep: list of indices to keep (original indices)
    """
    n = masks.shape[0]
    if n == 0:
        return []

    # Binarize masks
    masks = (masks > 0.5).bool()

    # Compute full IoU matrix
    iou_matrix = mask_iou_matrix(masks)

    # Sort by score descending - order contains original indices
    order = scores.argsort(descending=True).tolist()

    keep = []
    suppressed = set()

    for idx_in_order, i in enumerate(order):
        if i in suppressed:
            continue

        keep.append(i)

        # Suppress all remaining masks with IoU > threshold
        for j in order[idx_in_order + 1:]:
            if j in suppressed:
                continue

            if iou_matrix[i, j] > iou_threshold:
                suppressed.add(j)

    return keep


def soft_mask_nms(masks, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.01):
    """
    Soft-NMS variant for masks.
    Instead of hard suppression, reduce scores based on IoU.

    Args:
        masks: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_threshold: threshold to start suppression
        sigma: Gaussian decay parameter
        score_threshold: minimum score to keep

    Returns:
        keep: indices sorted by updated scores
        new_scores: updated scores after soft suppression
    """
    n = masks.shape[0]
    if n == 0:
        return [], torch.tensor([])

    # Binarize masks
    masks = (masks > 0.5).bool()

    # Compute IoU matrix
    iou_matrix = mask_iou_matrix(masks)

    # Copy scores to avoid modifying original
    scores = scores.clone().float()

    # Sort by score descending - order contains original indices
    order = scores.argsort(descending=True).tolist()

    keep = []

    for idx_in_order, i in enumerate(order):
        if scores[i] < score_threshold:  # Skip very low scores
            continue

        keep.append(i)

        # Soft suppress remaining masks
        for j in order[idx_in_order + 1:]:
            iou = iou_matrix[i, j].item()

            if iou > iou_threshold:
                # Gaussian decay - use math.exp for efficiency
                weight = math.exp(-iou * iou / sigma)
                scores[j] = scores[j] * weight

    # Re-sort keep by final scores
    keep = sorted(keep, key=lambda x: scores[x], reverse=True)

    return keep, scores[keep]


def apply_mask_nms_to_instances(instances, iou_threshold=0.5, use_soft_nms=False, sigma=0.5):
    """
    Apply mask NMS to Detectron2 Instances object.

    Args:
        instances: Detectron2 Instances with pred_masks, scores
        iou_threshold: NMS threshold
        use_soft_nms: use Soft-NMS instead of hard NMS
        sigma: Gaussian decay parameter for Soft-NMS

    Returns:
        filtered Instances
    """
    if len(instances) == 0:
        return instances

    masks = instances.pred_masks   # (N, H, W)
    scores = instances.scores      # (N,)

    # Move to same device
    if masks.device != scores.device:
        masks = masks.to(scores.device)

    if use_soft_nms:
        keep, new_scores = soft_mask_nms(masks, scores, iou_threshold, sigma)
        filtered = instances[keep]
        # Update scores with soft-NMS adjusted scores
        filtered.scores = new_scores
        return filtered
    else:
        keep = mask_nms_fast(masks, scores, iou_threshold)
        return instances[keep]
