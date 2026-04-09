"""
Depth Encoder for depth-guided feature modulation.

Lightweight CNN that encodes sensor depth into feature space matching
mask_features dimensions. Used for gated residual fusion:
    mask_features = mask_features + gamma * depth_encoder(depth)

Input is 2-channel:
  - Channel 0: Normalized sensor depth (0-1 range)
  - Channel 1: Depth invalid mask ((depth == 0).float())

The invalid mask is the key signal: IR-based depth sensors fail on
transparent objects, producing depth=0. This binary mask directly
indicates "something transparent is here."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """
    Lightweight depth encoder: [B, 2, H, W] -> [B, out_channels, H/4, W/4].

    Input channels:
        Ch 0: Normalized sensor depth (spatial context)
        Ch 1: Depth invalid mask (explicit transparency signal)

    Architecture:
        Conv 7x7 stride 2 (2 -> hidden_dim//2) + GN + ReLU
        Conv 3x3 stride 2 (hidden_dim//2 -> hidden_dim) + GN + ReLU
        Conv 3x3 stride 1 (hidden_dim -> hidden_dim) + GN + ReLU
        Conv 1x1 (hidden_dim -> out_channels)

    Total stride: 4x (matches mask_features at H/4, W/4)
    """

    def __init__(self, in_channels=2, hidden_dim=64, out_channels=256):
        super().__init__()
        half_hidden = hidden_dim // 2

        self.conv1 = nn.Conv2d(in_channels, half_hidden, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(8, half_hidden)

        self.conv2 = nn.Conv2d(half_hidden, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, hidden_dim)

        self.proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W] depth + invalid mask
        Returns:
            [B, out_channels, H/4, W/4] depth features
        """
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.proj(x)
        return x
