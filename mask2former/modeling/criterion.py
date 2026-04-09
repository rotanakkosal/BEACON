# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from .transformer_decoder.cdn_components import build_dn_loss_indices


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def _sobel_edges(masks: torch.Tensor) -> torch.Tensor:
    """Extract edges from masks using differentiable Sobel filters.

    Args:
        masks: [N, 1, H, W] float tensor (probabilities or soft masks)

    Returns:
        edges: [N, 1, H, W] edge magnitude
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=masks.dtype, device=masks.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=masks.dtype, device=masks.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(masks, sobel_x, padding=1)
    grad_y = F.conv2d(masks, sobel_y, padding=1)
    return (grad_x ** 2 + grad_y ** 2 + 1e-8).sqrt()


def boundary_dice_loss(
        pred_edges: torch.Tensor,
        gt_edges: torch.Tensor,
        num_masks: float,
    ):
    """
    Dice loss between predicted and GT edge maps.
    Both inputs should be non-negative, normalized to [0, 1] per mask.
    """
    pred_edges = pred_edges.flatten(1)
    gt_edges = gt_edges.flatten(1)
    numerator = 2 * (pred_edges * gt_edges).sum(-1)
    denominator = pred_edges.sum(-1) + gt_edges.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_boundary_dice(self, outputs, targets, indices, num_masks):
        """Boundary dice loss: Sobel edge extraction on matched masks.

        Extracts edges from predicted masks (after sigmoid) and GT masks
        using differentiable Sobel filters, then computes dice loss.
        Applied at every decoder layer via deep supervision.
        No new parameters, zero inference cost.
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx].float()  # [N, H, W]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]  # [N, H_full, W_full]

        # Handle empty matches
        if src_masks.shape[0] == 0:
            return {"loss_boundary_dice": src_masks.sum() * 0.0}

        # Downsample GT to prediction resolution (1/4 of input)
        h, w = src_masks.shape[-2:]
        tgt_small = F.interpolate(
            target_masks[:, None].float(), size=(h, w),
            mode="bilinear", align_corners=False,
        )  # [N, 1, H, W]

        # Predicted probabilities
        src_probs = src_masks[:, None].sigmoid()  # [N, 1, H, W]

        # Extract edges using Sobel
        src_edges = _sobel_edges(src_probs)[:, 0]  # [N, H, W]
        tgt_edges = _sobel_edges(tgt_small)[:, 0]  # [N, H, W]

        # Normalize per-mask to [0, 1] for scale-invariant dice
        src_max = src_edges.flatten(1).max(dim=1)[0].clamp(min=1e-6)
        src_edges = src_edges / src_max[:, None, None]

        tgt_max = tgt_edges.flatten(1).max(dim=1)[0].clamp(min=1e-6)
        tgt_edges = tgt_edges / tgt_max[:, None, None]

        loss = boundary_dice_loss(src_edges, tgt_edges, num_masks)
        return {"loss_boundary_dice": loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _compute_dn_loss(self, dn_outputs, targets, positive_indices, negative_indices, num_masks):
        """Compute denoising loss for CDN queries (direct GT assignment).

        Positive DN queries → reconstruct GT (CE + mask + dice).
        Negative DN queries → predict no-object (CE only).

        Args:
            dn_outputs: dict with 'pred_logits' [B, pad_size, C+1] and 'pred_masks' [B, pad_size, H, W]
            targets: list of target dicts
            positive_indices: list of (src_idx, tgt_idx) per image
            negative_indices: list of src_idx tensors per image
            num_masks: normalization factor (total GT objects across GPUs)
        """
        losses = {}
        device = dn_outputs["pred_logits"].device
        batch_size = dn_outputs["pred_logits"].shape[0]

        # === Classification loss (positive + negative) ===
        all_src_logits = []
        all_tgt_labels = []

        for b_idx in range(batch_size):
            src_pos, tgt_pos = positive_indices[b_idx]
            src_neg = negative_indices[b_idx]

            if len(src_pos) > 0:
                all_src_logits.append(dn_outputs["pred_logits"][b_idx, src_pos])
                all_tgt_labels.append(targets[b_idx]["labels"][tgt_pos])

            if len(src_neg) > 0:
                all_src_logits.append(dn_outputs["pred_logits"][b_idx, src_neg])
                all_tgt_labels.append(torch.full(
                    (len(src_neg),), self.num_classes, dtype=torch.int64, device=device
                ))

        if len(all_src_logits) > 0:
            cat_logits = torch.cat(all_src_logits, dim=0)  # [N_valid, C+1]
            cat_labels = torch.cat(all_tgt_labels, dim=0)  # [N_valid]
            # Uniform weighting (no eos_coef — DN has balanced pos/neg by construction)
            loss_ce = F.cross_entropy(cat_logits, cat_labels)
        else:
            loss_ce = torch.tensor(0.0, device=device)

        losses["loss_ce"] = loss_ce

        # === Mask losses (positive queries only) ===
        src_masks_list = []
        tgt_masks_list = []

        for b_idx in range(batch_size):
            src_pos, tgt_pos = positive_indices[b_idx]
            if len(src_pos) > 0:
                src_masks_list.append(dn_outputs["pred_masks"][b_idx, src_pos])
                tgt_masks_list.append(targets[b_idx]["masks"][tgt_pos])

        if len(src_masks_list) > 0:
            src_masks = torch.cat(src_masks_list, dim=0)[:, None]  # [N, 1, H, W]
            target_masks = torch.cat(tgt_masks_list, dim=0)[:, None].to(src_masks)

            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                point_labels = point_sample(
                    target_masks, point_coords, align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks, point_coords, align_corners=False,
            ).squeeze(1)

            losses["loss_mask"] = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
            losses["loss_dice"] = dice_loss_jit(point_logits, point_labels, num_masks)
        else:
            losses["loss_mask"] = torch.tensor(0.0, device=device)
            losses["loss_dice"] = torch.tensor(0.0, device=device)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boundary_dice': self.loss_boundary_dice,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Encoder loss for content-adaptive queries (Direction A)
        # The encoder predictions get their own Hungarian matching + losses
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            enc_indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, enc_indices, num_masks)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Denoising loss (CDN) — direct GT assignment, no Hungarian matching
        if "dn_outputs" in outputs and "dn_meta" in outputs:
            dn_meta = outputs["dn_meta"]
            positive_indices, negative_indices = build_dn_loss_indices(dn_meta, targets)

            # Final DN layer loss
            dn_outputs = outputs["dn_outputs"]
            l_dict = self._compute_dn_loss(
                dn_outputs, targets, positive_indices, negative_indices, num_masks
            )
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

            # Auxiliary DN layer losses
            if "dn_aux_outputs" in outputs:
                for i, dn_aux in enumerate(outputs["dn_aux_outputs"]):
                    l_dict = self._compute_dn_loss(
                        dn_aux, targets, positive_indices, negative_indices, num_masks
                    )
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
