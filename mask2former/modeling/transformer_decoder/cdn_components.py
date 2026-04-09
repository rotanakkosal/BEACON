"""
Contrastive Denoising Training (CDN) components for Mask2Former.

Adapted from DINO (Zhang et al., ICLR 2023) and Mask DINO (Li et al., CVPR 2023).
Generates noised GT queries during training to teach the decoder progressive refinement.

Key functions:
  - prepare_for_cdn: Generate noised GT queries + attention mask
  - build_dn_loss_indices: Build direct GT assignment for DN loss (no Hungarian matching)
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Convert binary masks to cxcywh bounding boxes (normalized 0-1).

    Args:
        masks: [N, H, W] binary masks

    Returns:
        boxes: [N, 4] in cxcywh format, normalized to [0, 1]
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    n, h, w = masks.shape
    boxes = []
    for i in range(n):
        m = masks[i]
        ys, xs = torch.where(m > 0)
        if len(ys) == 0:
            boxes.append(torch.zeros(4, device=masks.device))
            continue
        x0 = xs.min().float()
        x1 = xs.max().float()
        y0 = ys.min().float()
        y1 = ys.max().float()
        cx = (x0 + x1) / 2.0 / w
        cy = (y0 + y1) / 2.0 / h
        bw = (x1 - x0 + 1) / w
        bh = (y1 - y0 + 1) / h
        boxes.append(torch.stack([cx, cy, bw, bh]))

    return torch.stack(boxes)  # [N, 4]


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse sigmoid function."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def prepare_for_cdn(
    targets: List[Dict[str, Tensor]],
    dn_number: int,
    label_noise_ratio: float,
    box_noise_scale: float,
    num_classes: int,
    hidden_dim: int,
    label_enc: nn.Embedding,
    num_queries: int,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]:
    """Generate CDN queries from GT targets.

    This is the core CDN function. It creates noised versions of GT objects as
    auxiliary decoder queries. Positive queries (small noise) should reconstruct GT.
    Negative queries (large noise) should predict "no object."

    Args:
        targets: list of dicts with 'labels' [n_gt] and 'masks' [n_gt, H, W]
        dn_number: target number of DN query groups (before pos/neg doubling)
        label_noise_ratio: probability scale for label flipping (effective rate = ratio * 0.5)
        box_noise_scale: scale of box perturbation (1.0 = box-sized noise)
        num_classes: number of object classes (1 for ClearPose)
        hidden_dim: transformer hidden dimension
        label_enc: nn.Embedding for encoding class labels → content queries
        num_queries: number of regular (matching) queries

    Returns:
        input_query_label: [B, pad_size, D] content embeddings for DN queries
        input_query_bbox:  [B, pad_size, 4] noised box coordinates (cxcywh, normalized)
        attn_mask:         [pad_size + num_queries, pad_size + num_queries] self-attn mask
        dn_meta:           dict with metadata for loss computation:
            - pad_size: total number of DN query slots
            - num_dn_group: number of CDN groups
            - single_pad: max_gt per group (= max number of GT objects in batch)
            - dn_positive_idx: indices of positive DN queries per image
            - dn_negative_idx: indices of negative DN queries per image
    """
    # Collect GT labels and boxes from all images
    known_labels_list = [t["labels"] for t in targets]
    known_num = [len(t["labels"]) for t in targets]

    # No GT objects in batch → no CDN
    if max(known_num) == 0:
        return None, None, None, None

    # Derive bounding boxes from masks
    known_boxes_list = []
    for t in targets:
        if len(t["labels"]) == 0:
            known_boxes_list.append(torch.zeros((0, 4), device=t["labels"].device))
        else:
            known_boxes_list.append(masks_to_boxes(t["masks"]))

    batch_size = len(targets)
    device = targets[0]["labels"].device

    # Compute number of CDN groups
    max_gt = max(known_num)
    single_pad = max_gt  # Each group has single_pad positive + single_pad negative

    if dn_number >= 1:
        num_dn_group = dn_number
    else:
        num_dn_group = 1

    # Cap groups so total DN queries don't massively exceed regular queries
    if num_dn_group * 2 * single_pad > num_queries * 3:
        num_dn_group = max(1, (num_queries * 3) // (2 * single_pad))

    pad_size = single_pad * 2 * num_dn_group  # Total DN query slots per image

    if pad_size == 0:
        return None, None, None, None

    # Initialize padded tensors
    input_query_label = torch.zeros(batch_size, pad_size, hidden_dim, device=device)
    input_query_bbox = torch.zeros(batch_size, pad_size, 4, device=device)

    # Track which slots are valid (have GT objects) for loss masking
    dn_positive_idx = []
    dn_negative_idx = []

    for b_idx in range(batch_size):
        n_gt = known_num[b_idx]
        if n_gt == 0:
            dn_positive_idx.append(torch.zeros(0, dtype=torch.long, device=device))
            dn_negative_idx.append(torch.zeros(0, dtype=torch.long, device=device))
            continue

        labels = known_labels_list[b_idx]  # [n_gt]
        boxes = known_boxes_list[b_idx]    # [n_gt, 4] cxcywh normalized

        # Repeat for all CDN groups (each group: positive + negative)
        # Layout: [group0_pos(n_gt), group0_neg(n_gt), group1_pos(n_gt), ...]
        # Padded to single_pad per half-group

        pos_indices = []
        neg_indices = []

        for g in range(num_dn_group):
            group_offset = g * 2 * single_pad

            # Positive queries: slots [group_offset, group_offset + n_gt)
            pos_start = group_offset
            pos_slots = torch.arange(pos_start, pos_start + n_gt, device=device)
            pos_indices.append(pos_slots)

            # Negative queries: slots [group_offset + single_pad, group_offset + single_pad + n_gt)
            neg_start = group_offset + single_pad
            neg_slots = torch.arange(neg_start, neg_start + n_gt, device=device)
            neg_indices.append(neg_slots)

            # === Apply label noise ===
            # Positive labels: flip with probability label_noise_ratio * 0.5
            pos_labels = labels.clone()
            neg_labels = labels.clone()

            if label_noise_ratio > 0:
                flip_mask_pos = torch.rand(n_gt, device=device) < (label_noise_ratio * 0.5)
                if flip_mask_pos.any():
                    random_labels = torch.randint(0, num_classes, (flip_mask_pos.sum(),), device=device)
                    pos_labels[flip_mask_pos] = random_labels

                flip_mask_neg = torch.rand(n_gt, device=device) < (label_noise_ratio * 0.5)
                if flip_mask_neg.any():
                    random_labels = torch.randint(0, num_classes, (flip_mask_neg.sum(),), device=device)
                    neg_labels[flip_mask_neg] = random_labels

            # Encode labels → content embeddings
            input_query_label[b_idx, pos_slots] = label_enc(pos_labels)
            input_query_label[b_idx, neg_slots] = label_enc(neg_labels)

            # === Apply box noise ===
            # Convert cxcywh to xyxy for noise application
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x0 = cx - w / 2
            y0 = cy - h / 2
            x1 = cx + w / 2
            y1 = cy + h / 2
            known_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)  # [n_gt, 4]

            # diff is half the box size (noise scale reference)
            diff = torch.stack([w / 2, h / 2, w / 2, h / 2], dim=-1)  # [n_gt, 4]
            diff = diff.clamp(min=1e-4)  # avoid zero-size boxes

            # Random noise direction
            rand_sign = torch.randint(0, 2, (n_gt, 4), device=device).float() * 2.0 - 1.0

            # POSITIVE: noise magnitude in [0, 1) → box still overlaps GT
            rand_part_pos = torch.rand(n_gt, 4, device=device)  # [0, 1)
            pos_noise = rand_sign * rand_part_pos * diff * box_noise_scale
            pos_xyxy = (known_xyxy + pos_noise).clamp(0.0, 1.0)

            # Convert back to cxcywh
            pos_cx = (pos_xyxy[:, 0] + pos_xyxy[:, 2]) / 2
            pos_cy = (pos_xyxy[:, 1] + pos_xyxy[:, 3]) / 2
            pos_w = (pos_xyxy[:, 2] - pos_xyxy[:, 0]).clamp(min=1e-4)
            pos_h = (pos_xyxy[:, 3] - pos_xyxy[:, 1]).clamp(min=1e-4)
            input_query_bbox[b_idx, pos_slots] = torch.stack([pos_cx, pos_cy, pos_w, pos_h], dim=-1)

            # NEGATIVE: noise magnitude in [1, 2) → box pushed outside GT
            rand_sign_neg = torch.randint(0, 2, (n_gt, 4), device=device).float() * 2.0 - 1.0
            rand_part_neg = torch.rand(n_gt, 4, device=device) + 1.0  # [1, 2)
            neg_noise = rand_sign_neg * rand_part_neg * diff * box_noise_scale
            neg_xyxy = (known_xyxy + neg_noise).clamp(0.0, 1.0)

            neg_cx = (neg_xyxy[:, 0] + neg_xyxy[:, 2]) / 2
            neg_cy = (neg_xyxy[:, 1] + neg_xyxy[:, 3]) / 2
            neg_w = (neg_xyxy[:, 2] - neg_xyxy[:, 0]).clamp(min=1e-4)
            neg_h = (neg_xyxy[:, 3] - neg_xyxy[:, 1]).clamp(min=1e-4)
            input_query_bbox[b_idx, neg_slots] = torch.stack([neg_cx, neg_cy, neg_w, neg_h], dim=-1)

        dn_positive_idx.append(torch.cat(pos_indices))
        dn_negative_idx.append(torch.cat(neg_indices))

    # Build self-attention mask
    attn_mask = build_cdn_attention_mask(pad_size, num_queries, num_dn_group, single_pad, device)

    dn_meta = {
        "pad_size": pad_size,
        "num_dn_group": num_dn_group,
        "single_pad": single_pad,
        "dn_positive_idx": dn_positive_idx,
        "dn_negative_idx": dn_negative_idx,
        "known_num": known_num,
    }

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def build_cdn_attention_mask(
    pad_size: int,
    num_queries: int,
    num_dn_group: int,
    single_pad: int,
    device: torch.device,
) -> Tensor:
    """Build block-diagonal attention mask for CDN self-attention.

    Rules:
    1. Regular queries CANNOT see DN queries (prevents cheating from GT info)
    2. Each DN group can only see itself (prevents leakage between groups)
    3. DN queries CAN see regular queries (provides context)

    Args:
        pad_size: total number of DN query slots
        num_queries: number of regular (matching) queries
        num_dn_group: number of CDN groups
        single_pad: max GT objects per group half (positive or negative)
        device: torch device

    Returns:
        attn_mask: [pad_size + num_queries, pad_size + num_queries] bool tensor
                   True = blocked, False = can attend
    """
    total = pad_size + num_queries
    attn_mask = torch.zeros(total, total, device=device, dtype=torch.bool)

    # Rule 1: Regular queries cannot see DN queries
    attn_mask[pad_size:, :pad_size] = True

    # Rule 2: Each DN group can only see itself
    for g in range(num_dn_group):
        group_start = g * 2 * single_pad
        group_end = (g + 1) * 2 * single_pad

        # Block all groups before this one
        if group_start > 0:
            attn_mask[group_start:group_end, :group_start] = True

        # Block all groups after this one
        if group_end < pad_size:
            attn_mask[group_start:group_end, group_end:pad_size] = True

    # Rule 3: DN queries CAN see regular queries (implicit — already False)

    return attn_mask


def build_dn_loss_indices(
    dn_meta: Dict,
    targets: List[Dict[str, Tensor]],
) -> Tuple[List[Tuple[Tensor, Tensor]], List[Tuple[Tensor, Tensor]]]:
    """Build direct GT assignment indices for DN loss (no Hungarian matching).

    Each DN positive query has a known GT target (the one it was created from).
    Each DN negative query should predict "no object."

    Args:
        dn_meta: metadata from prepare_for_cdn
        targets: list of target dicts

    Returns:
        positive_indices: list of (src_idx, tgt_idx) tuples per image
            src_idx: indices into DN output [0, pad_size) for positive queries
            tgt_idx: corresponding GT object indices
        negative_indices: list of src_idx tensors per image (negative queries)
    """
    num_dn_group = dn_meta["num_dn_group"]
    single_pad = dn_meta["single_pad"]
    known_num = dn_meta["known_num"]

    positive_indices = []
    negative_indices = []

    for b_idx, n_gt in enumerate(known_num):
        if n_gt == 0:
            device = targets[b_idx]["labels"].device
            positive_indices.append((
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            ))
            negative_indices.append(
                torch.zeros(0, dtype=torch.long, device=device)
            )
            continue

        device = targets[b_idx]["labels"].device
        gt_indices = torch.arange(n_gt, device=device)

        all_src_pos = []
        all_tgt_pos = []
        all_src_neg = []

        for g in range(num_dn_group):
            group_offset = g * 2 * single_pad

            # Positive: slots [group_offset, group_offset + n_gt) → GT [0, n_gt)
            src_pos = torch.arange(group_offset, group_offset + n_gt, device=device)
            all_src_pos.append(src_pos)
            all_tgt_pos.append(gt_indices)

            # Negative: slots [group_offset + single_pad, group_offset + single_pad + n_gt)
            src_neg = torch.arange(
                group_offset + single_pad,
                group_offset + single_pad + n_gt,
                device=device,
            )
            all_src_neg.append(src_neg)

        positive_indices.append((torch.cat(all_src_pos), torch.cat(all_tgt_pos)))
        negative_indices.append(torch.cat(all_src_neg))

    return positive_indices, negative_indices


def dn_boxes_to_mask(
    boxes: Tensor,
    mask_h: int,
    mask_w: int,
) -> Tensor:
    """Convert normalized cxcywh boxes to binary masks for attention initialization.

    Used to generate initial cross-attention masks for DN queries before they
    have predicted masks.

    Args:
        boxes: [B, N, 4] boxes in cxcywh format, normalized [0, 1]
        mask_h, mask_w: spatial dimensions of the mask

    Returns:
        masks: [B, N, mask_h, mask_w] float masks (1 inside box, 0 outside)
    """
    B, N, _ = boxes.shape
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    # Create coordinate grids
    y_grid = torch.linspace(0, 1, mask_h, device=boxes.device)
    x_grid = torch.linspace(0, 1, mask_w, device=boxes.device)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")  # [H, W]

    # Expand for broadcasting: [B, N, H, W]
    xx = xx[None, None, :, :]  # [1, 1, H, W]
    yy = yy[None, None, :, :]

    x0 = (cx - w / 2)[:, :, None, None]  # [B, N, 1, 1]
    y0 = (cy - h / 2)[:, :, None, None]
    x1 = (cx + w / 2)[:, :, None, None]
    y1 = (cy + h / 2)[:, :, None, None]

    # 1 inside box, 0 outside
    masks = ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).float()

    return masks
