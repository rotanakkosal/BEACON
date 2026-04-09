# Copyright (c) 2025
# Overlap Penalty Loss for Boundary Supervision Approach
"""
Overlap penalty loss to prevent mask overlap in contact regions.

Key features:
- Only applied to MATCHED positive queries from Hungarian matching (Fix Issue D)
- Vectorized computation using sum trick (Fix Issue 5)
- Optional min_conf threshold for class-aware filtering
- Warmup handled externally via weight scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OverlapPenaltyLoss"]


class OverlapPenaltyLoss(nn.Module):
    """
    Overlap penalty loss for reducing mask overlap in contact regions.

    Penalizes cases where multiple matched queries predict high probability
    at the same pixel location within the contact boundary band.

    Uses vectorized computation:
        overlap = 0.5 * ((sum(M))^2 - sum(M^2))
    which is equivalent to sum over all pairs (M_i * M_j) but O(K) instead of O(K^2).
    """

    def __init__(
        self,
        weight: float = 1.0,
        min_conf: float = 0.0,
        use_contact_band_only: bool = True,
    ):
        """
        Args:
            weight: loss weight (applied externally, stored for reference)
            min_conf: minimum foreground confidence to include query (0 = disabled)
            use_contact_band_only: if True, only penalize in contact band region
        """
        super().__init__()
        self.weight = weight
        self.min_conf = min_conf
        self.use_contact_band_only = use_contact_band_only

    def forward(
        self,
        pred_masks: torch.Tensor,
        contact_band: torch.Tensor,
        matched_indices: list = None,
        pred_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute overlap penalty loss.

        Args:
            pred_masks: [B, Q, H, W] sigmoid mask predictions
            contact_band: [B, 1, H, W] contact boundary band (binary)
            matched_indices: list of (pred_idx, gt_idx) tuples from Hungarian matching
                If None, uses all queries (not recommended)
            pred_logits: [B, Q, C+1] class logits (optional, for confidence filtering)

        Returns:
            loss: scalar overlap penalty loss
        """
        B, Q, H, W = pred_masks.shape
        device = pred_masks.device

        total_loss = torch.tensor(0.0, device=device)
        num_valid_images = 0

        for b in range(B):
            # Get valid queries for this image
            if matched_indices is not None:
                pred_idx, _ = matched_indices[b]
                valid_queries = pred_idx
            else:
                # Fallback: use all queries (not recommended)
                valid_queries = torch.arange(Q, device=device)

            if len(valid_queries) < 2:
                # Need at least 2 queries for overlap
                continue

            # Get masks for valid queries
            masks_b = pred_masks[b, valid_queries]  # [K, H, W]

            # Optional: filter by confidence
            if self.min_conf > 0 and pred_logits is not None:
                fg_conf = self._get_fg_confidence(pred_logits[b, valid_queries])
                conf_mask = fg_conf > self.min_conf
                if conf_mask.sum() < 2:
                    continue
                masks_b = masks_b[conf_mask]

            K = masks_b.shape[0]
            if K < 2:
                continue

            # Get contact band for this image
            band_b = contact_band[b, 0]  # [H, W]

            # Compute overlap using vectorized sum trick
            # overlap = 0.5 * (sum(M)^2 - sum(M^2))
            # This equals sum over all pairs (M_i * M_j) for i < j
            if self.use_contact_band_only:
                # Apply contact band mask
                masks_in_band = masks_b * band_b.unsqueeze(0)  # [K, H, W]
            else:
                masks_in_band = masks_b

            # Sum over queries at each pixel
            mask_sum = masks_in_band.sum(dim=0)  # [H, W]
            mask_sq_sum = (masks_in_band ** 2).sum(dim=0)  # [H, W]

            # Vectorized overlap computation
            overlap = 0.5 * (mask_sum ** 2 - mask_sq_sum)  # [H, W]

            # Sum over spatial dimensions
            loss_b = overlap.sum()

            # Normalize by number of pixels in band (avoid div by zero)
            if self.use_contact_band_only:
                num_band_pixels = band_b.sum() + 1e-6
            else:
                num_band_pixels = H * W

            loss_b = loss_b / num_band_pixels

            total_loss = total_loss + loss_b
            num_valid_images += 1

        # Average over images
        if num_valid_images > 0:
            total_loss = total_loss / num_valid_images

        return total_loss

    def _get_fg_confidence(self, pred_logits: torch.Tensor) -> torch.Tensor:
        """
        Get foreground confidence from class logits.

        Args:
            pred_logits: [K, C+1] class logits (last class is background/no-object)

        Returns:
            fg_conf: [K] foreground confidence scores
        """
        # Softmax over class dimension
        probs = F.softmax(pred_logits, dim=-1)
        # Foreground confidence = 1 - background probability
        # Background is the last class
        fg_conf = 1.0 - probs[:, -1]
        return fg_conf


class OverlapPenaltyLossD1(OverlapPenaltyLoss):
    """Overlap penalty in contact-only band (Setting D1)."""

    def __init__(self, weight: float = 1.0, min_conf: float = 0.0):
        super().__init__(
            weight=weight,
            min_conf=min_conf,
            use_contact_band_only=True,
        )


class OverlapPenaltyLossD2(OverlapPenaltyLoss):
    """Overlap penalty in fg+contact union band (Setting D2)."""

    def __init__(self, weight: float = 1.0, min_conf: float = 0.0):
        super().__init__(
            weight=weight,
            min_conf=min_conf,
            use_contact_band_only=False,  # Uses full boundary band
        )


def build_overlap_penalty_loss(cfg):
    """Build overlap penalty loss from config."""
    use_contact_only = cfg.MODEL.BOUNDARY.OVERLAP_CONTACT_ONLY
    return OverlapPenaltyLoss(
        weight=cfg.MODEL.BOUNDARY.OVERLAP_WEIGHT,
        min_conf=cfg.MODEL.BOUNDARY.OVERLAP_MIN_CONF,
        use_contact_band_only=use_contact_only,
    )
