# Copyright (c) 2025
# MaskFormer with Boundary Supervision for Transparent Objects
"""
MaskFormer model with boundary supervision approach.

Extends base MaskFormer with:
- Boundary prediction head
- Query-conditioned boundary prior
- Overlap penalty loss
- DDP-safe training with global_step passing

Ablation settings:
- A: Baseline (no boundary supervision)
- D2: + Boundary head, query prior, overlap penalty (union band)
- I-v2b: + Same-scene copy-paste augmentation
- L: + High-resolution training
- L+M: + Multi-scale TTA (inference only)
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.boundary_supervision import (
    BoundaryHead,
    QueryBoundaryPrior,
    BoundaryCriterion,
    BoundaryToMaskProjection,
    build_boundary_head,
    build_query_boundary_prior,
    build_boundary_criterion,
)
from mask2former.modeling.depth_encoder import DepthEncoder


@META_ARCH_REGISTRY.register()
class MaskFormerBoundarySupervision(nn.Module):
    """
    MaskFormer with boundary supervision for transparent object segmentation.

    Supports settings:
    - Setting A: Baseline (USE_BOUNDARY_SUPERVISION=False)
    - Setting D2: + Boundary head, query prior, overlap penalty
    - Setting I-v2b: + Same-scene copy-paste augmentation
    - Setting L: + High-resolution training
    - Setting L+M: + Multi-scale TTA
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # boundary supervision
        use_boundary_supervision: bool,
        boundary_head: nn.Module = None,
        use_query_prior: bool = False,
        query_boundary_prior: nn.Module = None,
        boundary_criterion: nn.Module = None,
        teacher_forcing_warmup: int = 5000,
        boundary_feature_source: str = "mask_features",
        detach_lateral: bool = False,
        use_depth_edges: bool = False,
        # depth-guided feature modulation
        use_depth_fusion: bool = False,
        depth_encoder: nn.Module = None,
        depth_gamma_init: float = 0.1,
        # boundary spatial prior for decoder queries
        use_boundary_spatial_prior: bool = False,
        # B2M (Boundary-to-Mask) feature fusion
        use_b2m_fusion: bool = False,
        b2m_proj: nn.Module = None,
        b2m_gamma_init: float = 0.1,
    ):
        """
        Args:
            backbone: backbone module
            sem_seg_head: semantic segmentation head
            criterion: loss criterion
            num_queries: number of object queries
            object_mask_threshold: threshold for mask classification
            overlap_threshold: overlap threshold for inference
            metadata: dataset metadata
            size_divisibility: size divisibility for padding
            sem_seg_postprocess_before_inference: whether to postprocess before inference
            pixel_mean: pixel mean for normalization
            pixel_std: pixel std for normalization
            semantic_on: enable semantic segmentation
            panoptic_on: enable panoptic segmentation
            instance_on: enable instance segmentation
            test_topk_per_image: top-k instances per image for testing
            use_boundary_supervision: enable boundary supervision
            boundary_head: boundary prediction head
            use_query_prior: enable query boundary prior
            query_boundary_prior: query boundary prior module
            boundary_criterion: boundary loss criterion
            teacher_forcing_warmup: warmup iterations for teacher forcing
            boundary_feature_source: feature source for boundary head
                "mask_features" (default), "lateral", or "concat"
            detach_lateral: detach lateral features from gradient computation
            use_depth_edges: use depth edge map as additional boundary head input
            use_depth_fusion: fuse sensor depth into mask_features via gated residual
            depth_encoder: lightweight CNN encoding depth to feature space
            depth_gamma_init: initial value for learnable gating scalar
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # Inference settings
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # Content-adaptive query initialization
        self.content_query_enabled = getattr(sem_seg_head.predictor, 'content_query_enabled', False)

        # Boundary supervision components
        self.use_boundary_supervision = use_boundary_supervision
        self.boundary_head = boundary_head
        self.use_query_prior = use_query_prior
        self.query_boundary_prior = query_boundary_prior
        self.boundary_criterion = boundary_criterion
        self.teacher_forcing_warmup = teacher_forcing_warmup
        self.boundary_feature_source = boundary_feature_source
        self.detach_lateral = detach_lateral
        self.use_depth_edges = use_depth_edges

        # Sobel kernels for depth edge computation (registered as buffers)
        if use_depth_edges:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x, persistent=False)
            self.register_buffer("sobel_y", sobel_y, persistent=False)

        # Depth-guided feature modulation
        # gamma = max_gamma * sigmoid(g_raw), bounded to (0, max_gamma)
        # Init: g_raw=0 → sigmoid(0)=0.5 → gamma=max_gamma/2
        self.use_depth_fusion = use_depth_fusion
        self.depth_encoder = depth_encoder
        if use_depth_fusion:
            self.depth_gamma_max = depth_gamma_init * 2.0  # max_gamma so init = max/2
            self.depth_gamma_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

        # Boundary spatial prior: use boundary head output to guide decoder queries.
        # When enabled, we run the boundary head BEFORE the decoder and pass
        # the detached boundary map as a spatial attention bias.
        self.use_boundary_spatial_prior = use_boundary_spatial_prior

        # B2M (Boundary-to-Mask) feature fusion
        # Fuse intermediate boundary features into mask_features via gated residual.
        # Follows the depth fusion gating pattern: gamma = max * sigmoid(raw)
        self.use_b2m_fusion = use_b2m_fusion
        self.b2m_proj = b2m_proj
        if use_b2m_fusion:
            self.b2m_gamma_max = b2m_gamma_init * 2.0
            self.b2m_gamma_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 -> gamma=gamma_init

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # Loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # Build matcher
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        # Add boundary loss weights if enabled
        if cfg.MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION:
            weight_dict["loss_boundary_fg"] = cfg.MODEL.BOUNDARY.FG_WEIGHT
            weight_dict["loss_boundary_contact"] = cfg.MODEL.BOUNDARY.CONTACT_WEIGHT
            if cfg.MODEL.BOUNDARY.USE_OVERLAP_PENALTY:
                weight_dict["loss_overlap"] = cfg.MODEL.BOUNDARY.OVERLAP_WEIGHT
            # Boundary head dice losses (direct IoU optimization)
            if cfg.MODEL.BOUNDARY.BOUNDARY_DICE_LOSS_ENABLED:
                weight_dict["loss_boundary_dice_fg"] = cfg.MODEL.BOUNDARY.BOUNDARY_DICE_FG_WEIGHT
                weight_dict["loss_boundary_dice_contact"] = cfg.MODEL.BOUNDARY.BOUNDARY_DICE_CONTACT_WEIGHT

        # Add decoder boundary dice loss weight
        if cfg.MODEL.BOUNDARY.DECODER_BOUNDARY_LOSS_ENABLED:
            weight_dict["loss_boundary_dice"] = cfg.MODEL.BOUNDARY.DECODER_BOUNDARY_LOSS_WEIGHT

        # Add encoder loss weights for content-adaptive queries (Direction A)
        if cfg.MODEL.MASK_FORMER.CONTENT_QUERY_ENABLED:
            weight_dict["loss_ce_enc"] = cfg.MODEL.MASK_FORMER.ENC_LOSS_CE_WEIGHT
            weight_dict["loss_mask_enc"] = cfg.MODEL.MASK_FORMER.ENC_LOSS_MASK_WEIGHT
            weight_dict["loss_dice_enc"] = cfg.MODEL.MASK_FORMER.ENC_LOSS_DICE_WEIGHT

        # Add CDN (denoising) loss weights
        if cfg.MODEL.MASK_FORMER.CDN_ENABLED:
            weight_dict["loss_ce_dn"] = cfg.MODEL.MASK_FORMER.CDN_LOSS_CE_WEIGHT
            weight_dict["loss_mask_dn"] = cfg.MODEL.MASK_FORMER.CDN_LOSS_MASK_WEIGHT
            weight_dict["loss_dice_dn"] = cfg.MODEL.MASK_FORMER.CDN_LOSS_DICE_WEIGHT

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        if cfg.MODEL.BOUNDARY.DECODER_BOUNDARY_LOSS_ENABLED:
            losses.append("boundary_dice")

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        # Build boundary components
        boundary_head = None
        query_boundary_prior = None
        boundary_criterion = None

        if cfg.MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION:
            boundary_head = build_boundary_head(cfg)
            boundary_criterion = build_boundary_criterion(cfg)

            if cfg.MODEL.BOUNDARY.USE_QUERY_PRIOR:
                query_boundary_prior = build_query_boundary_prior(cfg)

        # Build depth encoder for depth-guided feature modulation
        depth_encoder = None
        if cfg.MODEL.BOUNDARY.USE_DEPTH_FUSION:
            mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
            depth_encoder = DepthEncoder(
                in_channels=2,  # normalized depth + invalid mask
                hidden_dim=cfg.MODEL.BOUNDARY.DEPTH_ENCODER_HIDDEN_DIM,
                out_channels=mask_dim,
            )

        # Build B2M projection for boundary-to-mask feature fusion
        b2m_proj = None
        if cfg.MODEL.BOUNDARY.USE_B2M_FUSION:
            boundary_dim = cfg.MODEL.BOUNDARY.HEAD_HIDDEN_DIM  # 128 for V2
            mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM  # 256
            b2m_proj = BoundaryToMaskProjection(
                boundary_dim=boundary_dim,
                mask_dim=mask_dim,
            )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # boundary supervision
            "use_boundary_supervision": cfg.MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION,
            "boundary_head": boundary_head,
            "use_query_prior": cfg.MODEL.BOUNDARY.USE_QUERY_PRIOR,
            "query_boundary_prior": query_boundary_prior,
            "boundary_criterion": boundary_criterion,
            "teacher_forcing_warmup": cfg.MODEL.BOUNDARY.TEACHER_FORCING_WARMUP,
            "boundary_feature_source": cfg.MODEL.BOUNDARY.FEATURE_SOURCE,
            "detach_lateral": cfg.MODEL.BOUNDARY.DETACH_LATERAL,
            "use_depth_edges": cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES,
            # depth-guided feature modulation
            "use_depth_fusion": cfg.MODEL.BOUNDARY.USE_DEPTH_FUSION,
            "depth_encoder": depth_encoder,
            "depth_gamma_init": cfg.MODEL.BOUNDARY.DEPTH_FUSION_GAMMA_INIT,
            # boundary spatial prior
            "use_boundary_spatial_prior": cfg.MODEL.BOUNDARY.USE_BOUNDARY_SPATIAL_PRIOR,
            # B2M feature fusion
            "use_b2m_fusion": cfg.MODEL.BOUNDARY.USE_B2M_FUSION,
            "b2m_proj": b2m_proj,
            "b2m_gamma_init": cfg.MODEL.BOUNDARY.B2M_GAMMA_INIT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _compute_depth_edges(self, depth):
        """Compute depth edge magnitude using Sobel filters.

        Args:
            depth: [B, 1, H, W] normalized depth tensor

        Returns:
            edges: [B, 1, H, W] edge magnitude, normalized per-image to [0, 1]
        """
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        # Normalize per-image
        B = edges.shape[0]
        for i in range(B):
            max_val = edges[i].max()
            if max_val > 0:
                edges[i] = edges[i] / max_val
        return edges

    def forward(self, batched_inputs, global_step: int = 0):
        """
        Forward pass.

        Args:
            batched_inputs: list of dicts with image and annotations
            global_step: current training iteration (from trainer, DDP-safe)

        Returns:
            dict of losses (training) or list of predictions (inference)
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        # Compute depth modulation (if depth fusion enabled)
        depth_modulation = None
        if self.use_depth_fusion and self.depth_encoder is not None and "depth" in batched_inputs[0]:
            depth_maps = torch.stack([x["depth"].to(self.device) for x in batched_inputs])
            # Resize depth to match image tensor size (after padding)
            img_h, img_w = images.tensor.shape[-2:]
            if depth_maps.shape[-2:] != (img_h, img_w):
                depth_maps = F.interpolate(
                    depth_maps, size=(img_h, img_w), mode="bilinear", align_corners=False
                )
            # Build 2-channel input: [normalized_depth, invalid_mask]
            # Invalid mask: depth==0 means sensor failed (likely transparent object)
            depth_invalid = (depth_maps == 0).float()  # [B, 1, H, W]
            depth_input = torch.cat([depth_maps, depth_invalid], dim=1)  # [B, 2, H, W]
            depth_features = self.depth_encoder(depth_input)  # [B, mask_dim, H/4, W/4]
            gamma = self.depth_gamma_max * torch.sigmoid(self.depth_gamma_raw)
            depth_modulation = gamma * depth_features

        # Prepare targets early (needed for CDN which uses GT during decoder forward)
        targets = None
        if self.training and "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)

        # Get outputs (and optionally mask_features for boundary head)
        if self.use_boundary_supervision and self.boundary_head is not None:
            need_lateral = self.boundary_feature_source in ("lateral", "concat")

            # When boundary spatial prior is enabled, we need to:
            # 1. Run pixel decoder first to get mask_features
            # 2. Run boundary head on mask_features
            # 3. Pass detached boundary map as spatial prior to transformer decoder
            if self.use_boundary_spatial_prior or self.use_b2m_fusion:
                # Decomposed path: run pixel decoder separately from decoder
                # so we can insert B2M fusion and/or spatial prior between them.

                # Step 1: Pixel decoder only
                encoder_memory = None
                encoder_spatial_shapes = None
                if need_lateral:
                    result = self.sem_seg_head.pixel_decoder.forward_features(
                        features, return_lateral_features=True,
                        return_encoder_memory=self.content_query_enabled,
                    )
                    if self.content_query_enabled:
                        mask_features, transformer_encoder_features, multi_scale_features, lateral_features, encoder_memory, encoder_spatial_shapes = result
                    else:
                        mask_features, transformer_encoder_features, multi_scale_features, lateral_features = result
                else:
                    result = self.sem_seg_head.pixel_decoder.forward_features(
                        features,
                        return_encoder_memory=self.content_query_enabled,
                    )
                    if self.content_query_enabled:
                        mask_features, transformer_encoder_features, multi_scale_features, encoder_memory, encoder_spatial_shapes = result
                    else:
                        mask_features, transformer_encoder_features, multi_scale_features = result

                # Step 2: Apply depth modulation (before boundary head and decoder)
                if depth_modulation is not None:
                    mask_features = mask_features + depth_modulation

                # Step 3: Boundary head on detached features
                # CRITICAL: detach inputs so boundary head gradients do NOT
                # flow back to pixel decoder / backbone. The boundary head
                # trains independently as a "read-only sensor".
                boundary_input = self._get_boundary_input(
                    mask_features.detach(),
                    lateral_features.detach() if need_lateral and lateral_features is not None else None,
                    batched_inputs,
                )
                if self.use_b2m_fusion:
                    boundary_logits, boundary_intermediate = self.boundary_head(
                        boundary_input, return_intermediate=True
                    )
                else:
                    boundary_logits = self.boundary_head(boundary_input)

                # Step 4: B2M feature fusion — inject boundary features into mask_features
                # Intermediate features (128-dim) are richer than predictions (2-dim).
                # Detach to keep boundary head and B2M projection gradient-independent.
                if self.use_b2m_fusion and self.b2m_proj is not None:
                    b2m_gamma = self.b2m_gamma_max * torch.sigmoid(self.b2m_gamma_raw)
                    b2m_projected = self.b2m_proj(boundary_intermediate.detach())
                    mask_features = mask_features + b2m_gamma * b2m_projected

                # Step 5: Spatial prior (only if enabled)
                boundary_spatial_prior = None
                if self.use_boundary_spatial_prior:
                    boundary_spatial_prior = boundary_logits.detach().sigmoid().max(dim=1, keepdim=True)[0]
                    mf_h, mf_w = mask_features.shape[-2:]
                    if boundary_spatial_prior.shape[-2:] != (mf_h, mf_w):
                        boundary_spatial_prior = F.interpolate(
                            boundary_spatial_prior, size=(mf_h, mf_w),
                            mode='bilinear', align_corners=False,
                        )

                # Step 6: Transformer decoder
                outputs = self.sem_seg_head.predictor(
                    multi_scale_features, mask_features, None,
                    boundary_spatial_prior=boundary_spatial_prior,
                    encoder_memory=encoder_memory,
                    encoder_spatial_shapes=encoder_spatial_shapes,
                    global_step=global_step,
                    targets=targets,
                )
            else:
                # Original path: decoder runs first, then boundary head
                if need_lateral:
                    outputs, mask_features, lateral_features = self.sem_seg_head(
                        features, return_boundary_features=True, depth_modulation=depth_modulation,
                        return_encoder_memory=self.content_query_enabled, global_step=global_step,
                        targets=targets,
                    )
                else:
                    outputs, mask_features = self.sem_seg_head(
                        features, return_mask_features=True, depth_modulation=depth_modulation,
                        return_encoder_memory=self.content_query_enabled, global_step=global_step,
                        targets=targets,
                    )

                # Compute boundary head input
                boundary_input = self._get_boundary_input(
                    mask_features,
                    lateral_features if need_lateral else None,
                    batched_inputs,
                )
                boundary_logits = self.boundary_head(boundary_input)

            outputs["boundary_logits"] = boundary_logits

            # Apply query prior (if enabled) — at BOTH training and inference
            if self.use_query_prior and self.query_boundary_prior is not None and self.training:
                boundary_preds = torch.sigmoid(boundary_logits)

                # Get query embeddings (from transformer decoder outputs)
                query_embed = outputs.get("query_embed", None)

                if query_embed is not None:
                    # Training: use GT boundaries for teacher forcing
                    gt_boundaries = None
                    tf_ratio = 0.0
                    if self.training:
                        if "fg_boundary" in batched_inputs[0]:
                            fg_list = [x["fg_boundary"].to(self.device) for x in batched_inputs]
                            contact_list = [x["contact_boundary"].to(self.device) for x in batched_inputs]
                            max_h = max(t.shape[0] for t in fg_list)
                            max_w = max(t.shape[1] for t in fg_list)
                            fg_list = [torch.nn.functional.pad(t, (0, max_w - t.shape[1], 0, max_h - t.shape[0])) for t in fg_list]
                            contact_list = [torch.nn.functional.pad(t, (0, max_w - t.shape[1], 0, max_h - t.shape[0])) for t in contact_list]
                            fg_gts = torch.stack(fg_list)
                            contact_gts = torch.stack(contact_list)
                            gt_boundaries = torch.stack([fg_gts, contact_gts], dim=1)
                        tf_ratio = self._get_teacher_forcing_ratio(global_step)

                    # Apply query prior to mask logits
                    result = self.query_boundary_prior(
                        query_embed,
                        boundary_preds,
                        outputs["pred_masks"],
                        gt_boundaries=gt_boundaries,
                        teacher_forcing_ratio=tf_ratio,
                        return_stats=self.training,
                    )
                    if isinstance(result, tuple):
                        outputs["pred_masks"], outputs["modulation_stats"] = result
                    else:
                        outputs["pred_masks"] = result
        else:
            outputs = self.sem_seg_head(
                features, depth_modulation=depth_modulation,
                return_encoder_memory=self.content_query_enabled, global_step=global_step,
                targets=targets,
            )

        if self.training:
            # targets already prepared above (before decoder call, needed for CDN)

            # Add boundary targets if available
            if self.use_boundary_supervision:
                for i, x in enumerate(batched_inputs):
                    if "fg_boundary" in x:
                        targets[i]["fg_boundary"] = x["fg_boundary"].to(self.device)
                        targets[i]["contact_boundary"] = x["contact_boundary"].to(self.device)
                        targets[i]["boundary_band"] = x["boundary_band"].to(self.device)
                        if "ignore_mask" in x:
                            targets[i]["ignore_mask"] = x["ignore_mask"].to(self.device)

            # Compute main losses
            losses = self.criterion(outputs, targets)

            # Get matched indices (needed for boundary losses)
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            matched_indices = self.criterion.matcher(outputs_without_aux, targets)

            # Compute boundary losses
            if self.use_boundary_supervision and self.boundary_criterion is not None:
                boundary_losses = self.boundary_criterion(
                    outputs,
                    targets,
                    matched_indices=matched_indices,
                    global_step=global_step,
                )
                losses.update(boundary_losses)

            # Apply loss weights
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                elif k not in ["tf_ratio"]:  # Keep for logging
                    losses.pop(k)

            # Store modulation stats separately for diagnostics (don't add to losses dict)
            # This prevents the "Tensor + dict" error during loss summation
            # The stats can be accessed via self._last_modulation_stats if needed
            if "modulation_stats" in outputs:
                self._last_modulation_stats = outputs["modulation_stats"]

            return losses
        else:
            return self._inference(outputs, batched_inputs, images)

    def _get_boundary_input(self, mask_features, lateral_features, batched_inputs):
        """Select and prepare features for the boundary head.

        Args:
            mask_features: [B, C, H, W] from pixel decoder
            lateral_features: [B, C, H, W] or None
            batched_inputs: list of dicts (for depth edges)

        Returns:
            boundary_input: [B, C', H, W] features for boundary head
        """
        if self.boundary_feature_source == "lateral" and lateral_features is not None:
            boundary_input = lateral_features
            if self.detach_lateral:
                boundary_input = boundary_input.detach()
        elif self.boundary_feature_source == "concat" and lateral_features is not None:
            lat = lateral_features
            if self.detach_lateral:
                lat = lat.detach()
            boundary_input = torch.cat([lat, mask_features], dim=1)
        else:  # "mask_features" (default)
            boundary_input = mask_features

        # Append depth edges if enabled
        if self.use_depth_edges and "depth" in batched_inputs[0]:
            depth_maps = torch.stack([x["depth"].to(self.device) for x in batched_inputs])
            depth_edges = self._compute_depth_edges(depth_maps)
            feat_h, feat_w = boundary_input.shape[-2:]
            if depth_edges.shape[-2:] != (feat_h, feat_w):
                depth_edges = F.interpolate(
                    depth_edges, size=(feat_h, feat_w), mode="bilinear", align_corners=False
                )
            boundary_input = torch.cat([boundary_input, depth_edges], dim=1)

        return boundary_input

    def _get_teacher_forcing_ratio(self, global_step: int) -> float:
        """Get teacher forcing ratio for current step."""
        if global_step >= self.teacher_forcing_warmup:
            return 0.0
        return 1.0 - (global_step / self.teacher_forcing_warmup)

    def prepare_targets(self, targets, images):
        """Prepare targets for loss computation."""
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
        return new_targets

    def _guided_filter_refine(self, mask_pred, guide_image, r=4, eps=0.1):
        """
        Image-guided mask refinement using guided filter.

        Snaps blurry mask edges (from bilinear upsampling of 1/4 res)
        to sharp edges in the original image.

        Args:
            mask_pred: [Q, H, W] mask logits at full resolution
            guide_image: [3, H, W] image tensor (normalized)
            r: filter radius (pixels)
            eps: regularization (smaller = more edge-preserving)

        Returns:
            refined mask_pred: [Q, H, W] logits
        """
        Q, H, W = mask_pred.shape

        # Convert to probabilities for filtering
        mask_probs = mask_pred.sigmoid()  # [Q, H, W]

        def box_filter(x, radius):
            return F.avg_pool2d(x, kernel_size=2*radius+1, stride=1, padding=radius)

        # Grayscale guide
        guide = guide_image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
        src = mask_probs.unsqueeze(0)  # [1, Q, H, W]

        mean_I = box_filter(guide, r)      # [1, 1, H, W]
        mean_p = box_filter(src, r)        # [1, Q, H, W]
        corr_Ip = box_filter(guide * src, r)  # [1, Q, H, W] (broadcast)
        var_I = box_filter(guide * guide, r) - mean_I * mean_I  # [1, 1, H, W]

        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)

        filtered = (mean_a * guide + mean_b)[0]  # [Q, H, W]
        filtered = filtered.clamp(1e-6, 1 - 1e-6)

        # Convert back to logits
        refined_logits = torch.log(filtered / (1 - filtered))
        return refined_logits

    def _inference(self, outputs, batched_inputs, images):
        """Run inference."""
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # Upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for b_idx, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        )):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            # Image-guided mask refinement at full resolution
            # Uses the original image to snap blurry mask edges to sharp image edges
            guide_img = images.tensor[b_idx]  # [3, H_pad, W_pad]
            mask_pred_result = self._guided_filter_refine(
                mask_pred_result, guide_img, r=2, eps=0.1
            )

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # Semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r

            # Panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # Instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    })

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        image_size = mask_pred.shape[-2:]

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
