# Copyright (c) 2025
# Boundary Criterion for Boundary Supervision Approach
"""
Combined boundary losses for training.

Includes:
1. FG boundary prediction loss (Focal Loss + optional Dice Loss)
2. Contact boundary prediction loss (Focal Loss + optional Dice Loss)
3. Overlap penalty loss (matched queries only)

Key features:
- DDP-safe: uses global_step from trainer (Fix Issue 8)
- Continuous teacher forcing mixing (Fix Issue 4)
- Warmup schedule for overlap penalty
- Loss magnitude monitoring and logging
- Optional dice loss for direct IoU optimization of boundary predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .overlap_penalty import OverlapPenaltyLoss

__all__ = ["BoundaryCriterion"]


class BoundaryCriterion(nn.Module):
    """
    Combined criterion for boundary supervision losses.

    Computes:
    - Focal loss for FG boundary prediction
    - Focal loss for contact boundary prediction
    - Optional dice loss for FG and contact boundary (direct IoU optimization)
    - Overlap penalty loss (with warmup)

    All schedules use global_step from trainer for DDP safety.
    """

    def __init__(
        self,
        fg_weight: float = 1.0,
        contact_weight: float = 2.0,
        overlap_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        overlap_warmup_iters: int = 2000,
        teacher_forcing_warmup: int = 5000,
        overlap_min_conf: float = 0.0,
        use_contact_band_only: bool = True,
        use_dice_loss: bool = False,
        dice_weight_fg: float = 1.0,
        dice_weight_contact: float = 2.0,
        gt_interpolation_mode: str = "nearest",
    ):
        """
        Args:
            fg_weight: weight for FG boundary loss
            contact_weight: weight for contact boundary loss
            overlap_weight: weight for overlap penalty
            focal_alpha: alpha for focal loss
            focal_gamma: gamma for focal loss
            overlap_warmup_iters: iterations to warm up overlap penalty
            teacher_forcing_warmup: iterations for teacher forcing warmup
            overlap_min_conf: minimum confidence for overlap penalty queries
            use_contact_band_only: use contact-only band for overlap (D1 vs D2)
            use_dice_loss: add dice loss alongside focal loss for boundary head
            dice_weight_fg: weight for FG boundary dice loss
            dice_weight_contact: weight for contact boundary dice loss
            gt_interpolation_mode: interpolation mode for GT downsampling
                "nearest" (default, preserves binary values)
                "bilinear" (smoother, preserves thin boundaries better)
        """
        super().__init__()

        self.fg_weight = fg_weight
        self.contact_weight = contact_weight
        self.overlap_weight = overlap_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.overlap_warmup_iters = overlap_warmup_iters
        self.teacher_forcing_warmup = teacher_forcing_warmup
        self.use_dice_loss = use_dice_loss
        self.dice_weight_fg = dice_weight_fg
        self.dice_weight_contact = dice_weight_contact
        self.gt_interpolation_mode = gt_interpolation_mode

        # Overlap penalty loss module
        self.overlap_loss = OverlapPenaltyLoss(
            weight=overlap_weight,
            min_conf=overlap_min_conf,
            use_contact_band_only=use_contact_band_only,
        )

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ignore_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Focal loss for sparse boundary prediction.

        Args:
            pred: [H, W] or [B, H, W] predictions (after sigmoid)
            target: [H, W] or [B, H, W] binary targets
            ignore_mask: [H, W] or [B, H, W] pixels to ignore (1 = ignore)

        Returns:
            loss: scalar focal loss
        """
        # Clamp predictions for numerical stability
        pred = pred.clamp(1e-6, 1 - 1e-6)

        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma

        # Alpha weighting (more weight on positive class)
        alpha_weight = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)

        # Combined loss
        loss = alpha_weight * focal_weight * bce

        # Apply ignore mask if provided
        if ignore_mask is not None:
            valid_mask = 1 - ignore_mask
            loss = loss * valid_mask
            # Normalize by valid pixels
            num_valid = valid_mask.sum() + 1e-6
            return loss.sum() / num_valid
        else:
            return loss.mean()

    def boundary_dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ignore_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Dice loss for boundary prediction — directly optimizes IoU-like metric.

        Args:
            pred: [H, W] predictions (after sigmoid, values in [0, 1])
            target: [H, W] binary targets
            ignore_mask: [H, W] pixels to ignore (1 = ignore)

        Returns:
            loss: scalar dice loss in [0, 1]
        """
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        if ignore_mask is not None:
            valid = (1 - ignore_mask).flatten().bool()
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]

        numerator = 2 * (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def _interpolate_gt(self, gt: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Interpolate GT to match prediction resolution.

        Args:
            gt: [H, W] ground truth tensor
            size: (H_pred, W_pred) target size

        Returns:
            resized gt: [H_pred, W_pred]
        """
        resized = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0),
            size=size,
            mode=self.gt_interpolation_mode,
            align_corners=False if self.gt_interpolation_mode == "bilinear" else None,
        ).squeeze()
        # For bilinear mode, re-binarize with a threshold to handle soft values
        if self.gt_interpolation_mode == "bilinear":
            resized = (resized > 0.3).float()
        return resized

    def get_teacher_forcing_ratio(self, global_step: int) -> float:
        """
        Get teacher forcing ratio for current step.

        Returns continuous ratio that decreases from 1.0 to 0.0.

        Args:
            global_step: current iteration from trainer

        Returns:
            ratio: float in [0, 1], 1.0 = use GT, 0.0 = use predicted
        """
        if global_step >= self.teacher_forcing_warmup:
            return 0.0
        return 1.0 - (global_step / self.teacher_forcing_warmup)

    def get_overlap_warmup_factor(self, global_step: int) -> float:
        """
        Get overlap penalty warmup factor.

        Returns factor that increases from 0.0 to 1.0.

        Args:
            global_step: current iteration from trainer

        Returns:
            factor: float in [0, 1]
        """
        if global_step >= self.overlap_warmup_iters:
            return 1.0
        return global_step / self.overlap_warmup_iters

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute boundary losses.

        Args:
            outputs: dict with
                - boundary_logits: [B, 2, H, W] boundary predictions
                - pred_masks: [B, Q, H, W] mask predictions (optional)
                - pred_logits: [B, Q, C+1] class logits (optional)
            targets: list of dicts (one per image) with
                - fg_boundary: [H, W] FG boundary GT
                - contact_boundary: [H, W] contact boundary GT
                - boundary_band: [H, W] boundary band for overlap penalty
                - ignore_mask: [H, W] pixels to ignore (optional)
            matched_indices: list of (pred_idx, gt_idx) from Hungarian matching
            global_step: current training iteration (from trainer, DDP-safe)

        Returns:
            losses: dict with loss values
        """
        losses = {}
        device = outputs['boundary_logits'].device

        boundary_logits = outputs.get('boundary_logits')
        pred_masks = outputs.get('pred_masks')
        pred_logits = outputs.get('pred_logits')

        B = boundary_logits.shape[0]

        # Apply sigmoid to get predictions
        boundary_preds = torch.sigmoid(boundary_logits)
        fg_preds = boundary_preds[:, 0]  # [B, H, W]
        contact_preds = boundary_preds[:, 1]  # [B, H, W]

        # Compute boundary prediction losses
        fg_focal_loss = torch.tensor(0.0, device=device)
        contact_focal_loss = torch.tensor(0.0, device=device)
        fg_dice_loss = torch.tensor(0.0, device=device)
        contact_dice_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            # Get targets for this image
            fg_gt = targets[b]['fg_boundary'].to(device)
            contact_gt = targets[b]['contact_boundary'].to(device)
            ignore_mask = targets[b].get('ignore_mask', None)
            if ignore_mask is not None:
                ignore_mask = ignore_mask.to(device)

            # Resize targets if needed
            H_pred, W_pred = fg_preds.shape[-2:]
            if fg_gt.shape != (H_pred, W_pred):
                fg_gt = self._interpolate_gt(fg_gt, (H_pred, W_pred))
                contact_gt = self._interpolate_gt(contact_gt, (H_pred, W_pred))
                if ignore_mask is not None:
                    ignore_mask = F.interpolate(
                        ignore_mask.unsqueeze(0).unsqueeze(0),
                        size=(H_pred, W_pred),
                        mode='nearest',
                    ).squeeze()

            # Focal losses
            fg_focal_loss = fg_focal_loss + self.focal_loss(fg_preds[b], fg_gt, ignore_mask)
            contact_focal_loss = contact_focal_loss + self.focal_loss(contact_preds[b], contact_gt, ignore_mask)

            # Dice losses (if enabled)
            if self.use_dice_loss:
                fg_dice_loss = fg_dice_loss + self.boundary_dice_loss(fg_preds[b], fg_gt, ignore_mask)
                contact_dice_loss = contact_dice_loss + self.boundary_dice_loss(contact_preds[b], contact_gt, ignore_mask)

        # NOTE: Return RAW losses without weight multiplication
        # Weights are applied via weight_dict in the main model (Detectron2 convention)
        losses['loss_boundary_fg'] = fg_focal_loss / B
        losses['loss_boundary_contact'] = contact_focal_loss / B

        # Dice losses for boundary head (if enabled)
        if self.use_dice_loss:
            losses['loss_boundary_dice_fg'] = fg_dice_loss / B
            losses['loss_boundary_dice_contact'] = contact_dice_loss / B

        # Overlap penalty (with warmup)
        # NOTE: warmup_factor is applied here since it's schedule-based, not a fixed weight
        if pred_masks is not None:
            warmup_factor = self.get_overlap_warmup_factor(global_step)

            if warmup_factor > 0:
                # Get boundary bands
                bands = []
                for b in range(B):
                    band = targets[b]['boundary_band'].to(device)
                    # Resize if needed
                    H_pred, W_pred = pred_masks.shape[-2:]
                    if band.shape != (H_pred, W_pred):
                        band = F.interpolate(
                            band.unsqueeze(0).unsqueeze(0),
                            size=(H_pred, W_pred),
                            mode='nearest',
                        ).squeeze()
                    bands.append(band)
                bands = torch.stack(bands).unsqueeze(1)  # [B, 1, H, W]

                # Sigmoid for overlap computation
                pred_masks_sigmoid = torch.sigmoid(pred_masks)

                # Compute overlap loss
                overlap_loss = self.overlap_loss(
                    pred_masks_sigmoid,
                    bands,
                    matched_indices=matched_indices,
                    pred_logits=pred_logits,
                )

                # Apply warmup factor (schedule-based, not fixed weight)
                # The fixed weight is applied via weight_dict in main model
                losses['loss_overlap'] = warmup_factor * overlap_loss
            else:
                losses['loss_overlap'] = torch.tensor(0.0, device=device)

        # Store teacher forcing ratio for logging
        tf_ratio = self.get_teacher_forcing_ratio(global_step)
        losses['tf_ratio'] = torch.tensor(tf_ratio, device=device)

        return losses


def build_boundary_criterion(cfg):
    """Build boundary criterion from config."""
    return BoundaryCriterion(
        fg_weight=cfg.MODEL.BOUNDARY.FG_WEIGHT,
        contact_weight=cfg.MODEL.BOUNDARY.CONTACT_WEIGHT,
        overlap_weight=cfg.MODEL.BOUNDARY.OVERLAP_WEIGHT,
        focal_alpha=cfg.MODEL.BOUNDARY.FOCAL_ALPHA,
        focal_gamma=cfg.MODEL.BOUNDARY.FOCAL_GAMMA,
        overlap_warmup_iters=cfg.MODEL.BOUNDARY.OVERLAP_WARMUP_ITERS,
        teacher_forcing_warmup=cfg.MODEL.BOUNDARY.TEACHER_FORCING_WARMUP,
        overlap_min_conf=cfg.MODEL.BOUNDARY.OVERLAP_MIN_CONF,
        use_contact_band_only=cfg.MODEL.BOUNDARY.OVERLAP_CONTACT_ONLY,
        use_dice_loss=cfg.MODEL.BOUNDARY.BOUNDARY_DICE_LOSS_ENABLED,
        dice_weight_fg=cfg.MODEL.BOUNDARY.BOUNDARY_DICE_FG_WEIGHT,
        dice_weight_contact=cfg.MODEL.BOUNDARY.BOUNDARY_DICE_CONTACT_WEIGHT,
        gt_interpolation_mode=cfg.MODEL.BOUNDARY.GT_INTERPOLATION_MODE,
    )
