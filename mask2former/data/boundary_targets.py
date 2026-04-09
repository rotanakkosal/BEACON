
"""
Generates boundary ground truth targets for instance segmentation.

Key features:
- FG boundary: object vs background boundaries
- Contact boundary: instance vs instance boundaries (O(HW) algorithm)
- Ignore mask: pixels to exclude from loss (overlaps, border artifacts)
- Efficient label map approach instead of O(N²) pairwise comparisons
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple


class BoundaryTargetGenerator:

    def __init__(
        self,
        dilation_radius: int = 4,     
        contact_dilation: int = 3,    
        handle_overlaps: bool = True,
        boundary_band_radius: int = 5,
    ):
        self.dilation_radius = dilation_radius
        self.contact_dilation = contact_dilation
        self.handle_overlaps = handle_overlaps
        self.boundary_band_radius = boundary_band_radius

        self.boundary_kernel = np.ones(
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
            dtype=np.uint8
        )
        self.contact_kernel = np.ones(
            (2 * contact_dilation + 1, 2 * contact_dilation + 1),
            dtype=np.uint8
        )
        self.band_kernel = np.ones(
            (2 * boundary_band_radius + 1, 2 * boundary_band_radius + 1),
            dtype=np.uint8
        )

    def __call__(
        self,
        masks: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if len(masks) == 0:
            if image_size is None:
                raise ValueError("image_size required when masks is empty")
            H, W = image_size
            return {
                'fg_boundary': np.zeros((H, W), dtype=np.float32),
                'contact_boundary': np.zeros((H, W), dtype=np.float32),
                'boundary_band': np.zeros((H, W), dtype=np.float32),
                'ignore_mask': np.zeros((H, W), dtype=np.float32),
            }

        # Get image size from masks
        N, H, W = masks.shape
        if image_size is not None:
            assert (H, W) == image_size, f"Mask size {(H, W)} != image_size {image_size}"

        # compute ignore mask first (for overlapping regions)
        ignore_mask = self._compute_ignore_mask(masks)

        # compute FG boundary
        fg_boundary = self._compute_fg_boundary(masks)

        # compute contact boundary
        contact_boundary = self._compute_contact_boundary(masks, fg_boundary)

        # compute boundary band
        boundary_union = np.maximum(fg_boundary, contact_boundary)
        boundary_band = cv2.dilate(
            boundary_union.astype(np.uint8),
            self.band_kernel,
            iterations=1
        ).astype(np.float32)

        return {
            'fg_boundary': fg_boundary.astype(np.float32),
            'contact_boundary': contact_boundary.astype(np.float32),
            'boundary_band': boundary_band.astype(np.float32),
            'ignore_mask': ignore_mask.astype(np.float32),
        }

    def _compute_ignore_mask(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute ignore mask for overlapping pixels.

        Args:
            masks: [N, H, W] binary masks

        Returns:
            ignore_mask: [H, W] binary mask where True = ignore
        """
        N, H, W = masks.shape

        if not self.handle_overlaps or N < 2:
            return np.zeros((H, W), dtype=np.uint8)

        # Sum all masks - pixels with sum > 1 are overlaps
        mask_sum = masks.sum(axis=0)
        ignore_mask = (mask_sum > 1).astype(np.uint8)

        return ignore_mask

    def _compute_fg_boundary(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute foreground boundary (object vs background).

        Args:
            masks: [N, H, W] binary masks

        Returns:
            fg_boundary: [H, W] binary boundary map
        """
        N, H, W = masks.shape
        fg_boundary = np.zeros((H, W), dtype=np.uint8)

        for i in range(N):
            mask = (masks[i] > 0.5).astype(np.uint8)

            # Compute boundary as dilate XOR erode
            dilated = cv2.dilate(mask, self.boundary_kernel, iterations=1)
            eroded = cv2.erode(mask, self.boundary_kernel, iterations=1)
            boundary = dilated ^ eroded

            # Union with existing
            fg_boundary = np.maximum(fg_boundary, boundary)

        return fg_boundary

    def _compute_contact_boundary(
        self,
        masks: np.ndarray,
        fg_boundary: np.ndarray
    ) -> np.ndarray:
        """
        Compute contact boundary using label map approach (O(HW)).

        This is much faster than O(N²) pairwise comparison.

        Args:
            masks: [N, H, W] binary masks
            fg_boundary: [H, W] foreground boundary (for cleanup)

        Returns:
            contact_boundary: [H, W] binary contact map
        """
        N, H, W = masks.shape

        if N < 2:
            return np.zeros((H, W), dtype=np.uint8)

        # Create label map (0 = background, 1..N = instances)
        # Handle overlaps: last mask wins (or use ignore_mask)
        label_map = np.zeros((H, W), dtype=np.int32)
        for i, mask in enumerate(masks):
            label_map[mask > 0.5] = i + 1

        # Pad for neighbor access
        label_padded = np.pad(label_map, 1, mode='constant', constant_values=0)
        center = label_padded[1:H+1, 1:W+1]

        # Check 4-connected neighbors for different non-zero labels
        contact = np.zeros((H, W), dtype=np.uint8)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = label_padded[1+dy:H+1+dy, 1+dx:W+1+dx]
            # Contact: both pixels are foreground (>0) AND have different labels
            is_contact = (center > 0) & (neighbor > 0) & (center != neighbor)
            contact = np.maximum(contact, is_contact.astype(np.uint8))

        # Dilate slightly for cleaner result
        if self.contact_dilation > 0:
            contact = cv2.dilate(contact, self.contact_kernel, iterations=1)

        # Intersect with fg_boundary band for cleanup
        boundary_band = cv2.dilate(
            fg_boundary.astype(np.uint8),
            self.boundary_kernel,
            iterations=2
        )
        contact = contact & boundary_band

        return contact


def resize_boundary_targets(
    targets: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """
    Resize boundary targets to match model output resolution.

    Uses nearest neighbor interpolation for binary masks.

    Args:
        targets: dict with fg_boundary, contact_boundary, boundary_band, ignore_mask
        target_size: (H, W) target size

    Returns:
        resized targets dict
    """
    import torch.nn.functional as F

    resized = {}
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            # Add batch and channel dims if needed
            if value.dim() == 2:
                value = value.unsqueeze(0).unsqueeze(0)
            elif value.dim() == 3:
                value = value.unsqueeze(1)

            # Resize with nearest neighbor (for binary masks)
            resized_value = F.interpolate(
                value.float(),
                size=target_size,
                mode='nearest',
            )

            # Remove added dims
            resized[key] = resized_value.squeeze(1).squeeze(0)
        else:
            resized[key] = value

    return resized


def visualize_boundary_targets(
    image: np.ndarray,
    fg_boundary: np.ndarray,
    contact_boundary: np.ndarray,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize boundary targets overlaid on image.

    Args:
        image: [H, W, 3] RGB image
        fg_boundary: [H, W] binary foreground boundary
        contact_boundary: [H, W] binary contact boundary
        save_path: optional path to save visualization

    Returns:
        overlay: [H, W, 3] visualization image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()

    # Green for FG boundary
    fg_mask = fg_boundary > 0.5
    overlay[fg_mask] = [0, 255, 0]

    # Red for contact boundary (overlay on top)
    contact_mask = contact_boundary > 0.5
    overlay[contact_mask] = [255, 0, 0]

    # Blend with original
    alpha = 0.6
    result = (alpha * overlay + (1 - alpha) * image).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return result
