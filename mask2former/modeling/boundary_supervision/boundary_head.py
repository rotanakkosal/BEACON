# Copyright (c) 2025
# Boundary Head for Boundary Supervision Approach
"""
Boundary prediction head for instance segmentation.

Predicts:
- FG boundary: foreground (object vs background) boundaries
- Contact boundary: instance-instance boundaries

Two variants:
- BoundaryHead: Original lightweight 2-layer head at 1/4 resolution
- BoundaryHeadV2: Enhanced head with 4 layers, residual connections,
  and optional 2x upsampling for 1/2 resolution predictions

Uses GroupNorm instead of BatchNorm for stability with small batch sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable

__all__ = ["BoundaryHead", "BoundaryHeadV2", "BoundaryToMaskProjection"]


class BoundaryHead(nn.Module):
    """
    Boundary prediction head (original).

    Takes pixel decoder features and predicts boundary maps.

    Architecture:
        Conv3x3 (in_channels -> hidden_dim) + GroupNorm + ReLU
        Conv3x3 (hidden_dim -> hidden_dim) + GroupNorm + ReLU
        Conv1x1 (hidden_dim -> 2)  # FG and Contact boundaries

    Output has negative bias initialization for sparse boundary prediction.
    """

    @configurable
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 64,
        num_groups: int = 8,
        output_bias: float = -2.0,
    ):
        """
        Args:
            in_channels: number of input channels from pixel decoder
            hidden_dim: hidden dimension for conv layers
            num_groups: number of groups for GroupNorm
            output_bias: negative bias for output layer (sparse boundaries)
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # First conv block
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, hidden_dim)

        # Second conv block
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, hidden_dim)

        # Output layer: 2 channels (FG, Contact)
        self.output_conv = nn.Conv2d(hidden_dim, 2, kernel_size=1)

        # Initialize weights
        self._init_weights(output_bias)

    def _init_weights(self, output_bias: float):
        """Initialize weights with proper initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Output conv: small weights, negative bias for sparse predictions
        nn.init.normal_(self.output_conv.weight, std=0.01)
        nn.init.constant_(self.output_conv.bias, output_bias)

    @classmethod
    def from_config(cls, cfg):
        feature_source = cfg.MODEL.BOUNDARY.FEATURE_SOURCE
        if feature_source == "concat":
            in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM * 2
        else:
            in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        # Add depth edge channel if enabled
        if cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES:
            in_channels += 1

        return {
            "in_channels": in_channels,
            "hidden_dim": cfg.MODEL.BOUNDARY.HEAD_HIDDEN_DIM,
            "num_groups": cfg.MODEL.BOUNDARY.NUM_GROUPS,
            "output_bias": cfg.MODEL.BOUNDARY.OUTPUT_BIAS,
        }

    def forward(self, features: torch.Tensor, return_intermediate: bool = False):
        """
        Forward pass.

        Args:
            features: [B, C, H, W] pixel decoder features
            return_intermediate: if True, also return intermediate features
                for B2M (Boundary-to-Mask) feature fusion

        Returns:
            boundary_logits: [B, 2, H, W] boundary predictions (logits)
            intermediate (optional): [B, hidden_dim, H, W] features after conv2
        """
        # First block
        x = self.conv1(features)
        x = self.norm1(x)
        x = F.relu(x)

        # Second block
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        intermediate = x  # [B, hidden_dim, H, W] at 1/4 resolution

        # Output
        boundary_logits = self.output_conv(x)

        if return_intermediate:
            return boundary_logits, intermediate
        return boundary_logits

    def get_boundary_maps(self, features: torch.Tensor) -> dict:
        """
        Get boundary maps with sigmoid activation.

        Args:
            features: [B, C, H, W] pixel decoder features

        Returns:
            dict with:
                - boundary_logits: [B, 2, H, W] raw logits
                - fg_boundary: [B, 1, H, W] sigmoid FG boundary
                - contact_boundary: [B, 1, H, W] sigmoid contact boundary
        """
        logits = self.forward(features)
        probs = torch.sigmoid(logits)

        return {
            'boundary_logits': logits,
            'fg_boundary': probs[:, 0:1, :, :],
            'contact_boundary': probs[:, 1:2, :, :],
        }


class BoundaryHeadV2(nn.Module):
    """
    Enhanced boundary prediction head (v2).

    Improvements over BoundaryHead:
    1. Deeper: 4 conv layers instead of 2 (more capacity for edge detection)
    2. Residual connections: skip connections for better gradient flow
    3. Optional 2x upsampling: predict at 1/2 resolution instead of 1/4
       (critical for thin boundaries that disappear at 1/4 resolution)
    4. Wider: configurable hidden dim (default 128 instead of 64)

    Architecture (with upsample=True):
        Conv3x3 (in_channels -> hidden_dim) + GN + ReLU           [1/4 res]
        Conv3x3 (hidden_dim -> hidden_dim) + GN + ReLU + residual [1/4 res]
        ConvTranspose 4x4 stride 2 (hidden_dim -> hidden_dim//2)  [1/2 res]
        Conv3x3 (hidden_dim//2 -> hidden_dim//2) + GN + ReLU      [1/2 res]
        Conv1x1 (hidden_dim//2 -> 2)                               [1/2 res]

    Architecture (without upsample):
        Conv3x3 (in_channels -> hidden_dim) + GN + ReLU           [1/4 res]
        Conv3x3 (hidden_dim -> hidden_dim) + GN + ReLU + residual [1/4 res]
        Conv3x3 (hidden_dim -> hidden_dim) + GN + ReLU            [1/4 res]
        Conv3x3 (hidden_dim -> hidden_dim) + GN + ReLU + residual [1/4 res]
        Conv1x1 (hidden_dim -> 2)                                  [1/4 res]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        num_groups: int = 8,
        output_bias: float = -2.0,
        upsample: bool = True,
    ):
        """
        Args:
            in_channels: input channels from pixel decoder
            hidden_dim: hidden dimension for conv layers
            num_groups: groups for GroupNorm
            output_bias: negative bias for output layer (sparse boundaries)
            upsample: if True, upsample 2x to predict at 1/2 resolution
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.upsample = upsample

        # Block 1: reduce channels
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, hidden_dim)

        # Block 2: with residual
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, hidden_dim)

        if upsample:
            # Transpose conv for 2x upsampling: 1/4 -> 1/2 resolution
            up_dim = hidden_dim // 2
            self.up_conv = nn.ConvTranspose2d(
                hidden_dim, up_dim, kernel_size=4, stride=2, padding=1
            )
            self.up_norm = nn.GroupNorm(num_groups, up_dim)

            # Block 3: refine at higher resolution
            self.conv3 = nn.Conv2d(up_dim, up_dim, kernel_size=3, padding=1)
            self.norm3 = nn.GroupNorm(num_groups, up_dim)

            # Output
            self.output_conv = nn.Conv2d(up_dim, 2, kernel_size=1)
        else:
            # Block 3: extra depth at same resolution
            self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.norm3 = nn.GroupNorm(num_groups, hidden_dim)

            # Block 4: with residual
            self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.norm4 = nn.GroupNorm(num_groups, hidden_dim)

            # Output
            self.output_conv = nn.Conv2d(hidden_dim, 2, kernel_size=1)

        self._init_weights(output_bias)

    def _init_weights(self, output_bias: float):
        """Initialize weights with proper initialization."""
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m is self.output_conv:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, output_bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, return_intermediate: bool = False):
        """
        Forward pass.

        Args:
            features: [B, C, H, W] pixel decoder features at 1/4 resolution
            return_intermediate: if True, also return intermediate features
                for B2M (Boundary-to-Mask) feature fusion

        Returns:
            boundary_logits: [B, 2, H', W'] boundary predictions (logits)
                H', W' = 2*H, 2*W if upsample=True, else H, W
            intermediate (optional): [B, hidden_dim, H, W] features at 1/4 res
        """
        # Block 1
        x = self.conv1(features)
        x = self.norm1(x)
        x = F.relu(x)

        # Block 2 with residual
        identity = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x + identity)

        # Capture intermediate features at 1/4 resolution (before upsample)
        # for B2M fusion — same spatial resolution as mask_features
        intermediate = x  # [B, hidden_dim, H/4, W/4]

        if self.upsample:
            # Upsample 2x
            x = self.up_conv(x)
            x = self.up_norm(x)
            x = F.relu(x)

            # Block 3: refine at higher resolution
            x = self.conv3(x)
            x = self.norm3(x)
            x = F.relu(x)
        else:
            # Block 3
            x = self.conv3(x)
            x = self.norm3(x)
            x = F.relu(x)

            # Block 4 with residual
            identity = x
            x = self.conv4(x)
            x = self.norm4(x)
            x = F.relu(x + identity)

        # Output
        boundary_logits = self.output_conv(x)

        if return_intermediate:
            return boundary_logits, intermediate
        return boundary_logits

    def get_boundary_maps(self, features: torch.Tensor) -> dict:
        """Get boundary maps with sigmoid activation."""
        logits = self.forward(features)
        probs = torch.sigmoid(logits)
        return {
            'boundary_logits': logits,
            'fg_boundary': probs[:, 0:1, :, :],
            'contact_boundary': probs[:, 1:2, :, :],
        }


class BoundaryToMaskProjection(nn.Module):
    """
    Projects boundary intermediate features to mask_features dimension.

    Used for B2M (Boundary-to-Mask) feature fusion:
        mask_features = mask_features + gamma * proj(boundary_intermediate)

    Following BMask R-CNN's finding that intermediate boundary features
    (rich high-dim representations) are more useful for mask prediction
    than final boundary predictions (2-channel maps with low precision).
    """

    def __init__(self, boundary_dim: int = 128, mask_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(boundary_dim, mask_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mask_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, boundary_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_features: [B, boundary_dim, H/4, W/4] intermediate features
        Returns:
            [B, mask_dim, H/4, W/4] projected features for fusion with mask_features
        """
        return self.proj(boundary_features)


def build_boundary_head(cfg):
    """Build boundary head from config."""
    feature_source = cfg.MODEL.BOUNDARY.FEATURE_SOURCE
    if feature_source == "concat":
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM * 2
    else:
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

    # Add depth edge channel if enabled
    if cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES:
        in_channels += 1

    # Choose head version
    use_v2 = getattr(cfg.MODEL.BOUNDARY, 'USE_BOUNDARY_HEAD_V2', False)
    if use_v2:
        return BoundaryHeadV2(
            in_channels=in_channels,
            hidden_dim=cfg.MODEL.BOUNDARY.HEAD_HIDDEN_DIM,
            num_groups=cfg.MODEL.BOUNDARY.NUM_GROUPS,
            output_bias=cfg.MODEL.BOUNDARY.OUTPUT_BIAS,
            upsample=getattr(cfg.MODEL.BOUNDARY, 'BOUNDARY_HEAD_UPSAMPLE', True),
        )

    return BoundaryHead(
        in_channels=in_channels,
        hidden_dim=cfg.MODEL.BOUNDARY.HEAD_HIDDEN_DIM,
        num_groups=cfg.MODEL.BOUNDARY.NUM_GROUPS,
        output_bias=cfg.MODEL.BOUNDARY.OUTPUT_BIAS,
    )
