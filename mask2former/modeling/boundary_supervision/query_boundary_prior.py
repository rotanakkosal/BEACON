# Copyright (c) 2025
# Query Boundary Prior for Boundary Supervision Approach
"""
Query-conditioned boundary prior for mask modulation.

Each query learns:
- alpha_fg: how much FG boundary affects its mask
- alpha_contact: how much contact boundary affects its mask

Key features:
- Separate projections for FG and contact boundaries (Fix Issue C)
- Smaller alpha range (0.3) to prevent killing recall (Fix Issue 3)
- Continuous teacher forcing mixing (Fix Issue 4)
- No internal iteration counter - uses global_step from trainer (Fix Issue 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QueryBoundaryPrior"]


class QueryBoundaryPrior(nn.Module):
    """
    Query-conditioned boundary prior for mask modulation.

    Modulation formula:
        mask_logits_q = mask_logits_q - alpha_fg(q) * g_fg(fg_map) - alpha_contact(q) * g_contact(contact_map)

    This suppresses mask predictions at boundary locations, with query-specific
    modulation strength.
    """

    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 32,
        alpha_scale: float = 0.3,
    ):
        """
        Args:
            query_dim: dimension of query embeddings
            hidden_dim: hidden dimension for boundary projections
            alpha_scale: scale for alpha values (controls modulation strength)
        """
        super().__init__()

        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.alpha_scale = alpha_scale

        # MLP to predict alpha_fg and alpha_contact per query
        # Output is 2D: [alpha_fg, alpha_contact]
        self.alpha_mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # Output in [-1, 1], then scaled by alpha_scale
        )

        # Separate projection networks for FG and contact boundaries
        # These learn to transform raw boundary maps into modulation signals
        self.fg_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1] for smooth modulation
        )

        self.contact_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for small but non-zero initial influence.

        Note: Zero initialization causes vanishing gradients - the alpha values
        never learn because the gradient signal through the mask loss is too weak.
        We use small random initialization to provide an initial gradient signal.
        """
        # Initialize alpha MLP with small random values (not zeros!)
        # This ensures gradients can flow from the start
        nn.init.xavier_uniform_(self.alpha_mlp[-2].weight, gain=0.1)
        nn.init.constant_(self.alpha_mlp[-2].bias, 0.1)  # Slight positive bias

        # Initialize projection convs
        for proj in [self.fg_proj, self.contact_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        query_embed: torch.Tensor,
        boundary_maps: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_boundaries: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
        return_stats: bool = False,
    ):
        """
        Apply query-conditioned boundary prior to mask logits.

        Args:
            query_embed: [B, Q, D] query embeddings from transformer decoder
            boundary_maps: [B, 2, H, W] PREDICTED boundary maps (after sigmoid)
                - channel 0: FG boundary
                - channel 1: Contact boundary
            mask_logits: [B, Q, H, W] mask logits (before sigmoid)
            gt_boundaries: [B, 2, H, W] GT boundary maps (for teacher forcing)
            teacher_forcing_ratio: float in [0, 1], mixing ratio
                - 1.0 = use GT boundaries
                - 0.0 = use predicted boundaries
                - Computed externally by trainer for DDP safety
            return_stats: if True, also return modulation statistics dict

        Returns:
            modulated_logits: [B, Q, H, W] modulated mask logits
            stats (optional): dict with modulation statistics
        """
        B, Q, D = query_embed.shape
        _, _, H, W = mask_logits.shape

        # First, resize predicted boundary maps to match mask_logits resolution
        if boundary_maps.shape[-2:] != (H, W):
            boundary_maps = F.interpolate(
                boundary_maps,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

        # Get boundary maps to use (continuous mixing for teacher forcing)
        if self.training and gt_boundaries is not None and teacher_forcing_ratio > 0:
            # Resize GT boundaries to match mask_logits resolution
            if gt_boundaries.shape[-2:] != (H, W):
                gt_boundaries_resized = F.interpolate(
                    gt_boundaries.float(),
                    size=(H, W),
                    mode='nearest',
                )
            else:
                gt_boundaries_resized = gt_boundaries.float()

            # Continuous mixing: blend GT and predicted
            boundary_to_use = (
                teacher_forcing_ratio * gt_boundaries_resized +
                (1 - teacher_forcing_ratio) * boundary_maps
            )
        else:
            boundary_to_use = boundary_maps

        # Split boundary maps
        fg_map = boundary_to_use[:, 0:1, :, :]  # [B, 1, H, W]
        contact_map = boundary_to_use[:, 1:2, :, :]  # [B, 1, H, W]

        # Project boundary maps through separate networks
        b_fg = self.fg_proj(fg_map)  # [B, 1, H, W]
        b_contact = self.contact_proj(contact_map)  # [B, 1, H, W]

        # Compute per-query alpha values
        alphas = self.alpha_mlp(query_embed) * self.alpha_scale  # [B, Q, 2]
        alpha_fg = alphas[:, :, 0:1]  # [B, Q, 1]
        alpha_contact = alphas[:, :, 1:2]  # [B, Q, 1]

        # Reshape for broadcasting
        # alpha: [B, Q, 1] -> [B, Q, 1, 1]
        # b_fg: [B, 1, H, W]
        alpha_fg_4d = alpha_fg.unsqueeze(-1)  # [B, Q, 1, 1]
        alpha_contact_4d = alpha_contact.unsqueeze(-1)  # [B, Q, 1, 1]

        # Compute modulation term
        modulation = alpha_fg_4d * b_fg + alpha_contact_4d * b_contact

        # Apply modulation: subtract boundary influence from mask logits
        # Positive alpha + positive boundary = suppress mask at boundary
        modulated_logits = mask_logits - modulation

        if return_stats:
            with torch.no_grad():
                stats = {
                    'alpha_fg_mean': alpha_fg.abs().mean().item(),
                    'alpha_contact_mean': alpha_contact.abs().mean().item(),
                    'b_fg_mean': b_fg.mean().item(),
                    'b_contact_mean': b_contact.mean().item(),
                    'modulation_mean': modulation.abs().mean().item(),
                    'mask_logits_std': mask_logits.std().item(),
                }
            return modulated_logits, stats

        return modulated_logits

    def get_alphas(self, query_embed: torch.Tensor) -> dict:
        """
        Get alpha values for debugging/visualization.

        Args:
            query_embed: [B, Q, D] query embeddings

        Returns:
            dict with:
                - alpha_fg: [B, Q] FG alpha values
                - alpha_contact: [B, Q] contact alpha values
        """
        alphas = self.alpha_mlp(query_embed) * self.alpha_scale  # [B, Q, 2]
        return {
            'alpha_fg': alphas[:, :, 0],
            'alpha_contact': alphas[:, :, 1],
        }


def build_query_boundary_prior(cfg):
    """Build query boundary prior from config."""
    return QueryBoundaryPrior(
        query_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        hidden_dim=cfg.MODEL.BOUNDARY.QUERY_PRIOR_DIM,
        alpha_scale=cfg.MODEL.BOUNDARY.ALPHA_SCALE,
    )
