# Copyright (c) 2025
# Configuration for Boundary Supervision Approach
"""
Config additions for boundary supervision.

Usage:
    from mask2former.config_boundary_supervision import add_boundary_supervision_config
    add_boundary_supervision_config(cfg)
"""

from detectron2.config import CfgNode as CN


def add_boundary_supervision_config(cfg):
    """
    Add boundary supervision config options.

    Call this after add_maskformer2_config(cfg).
    """
    # Create BOUNDARY node if it doesn't exist
    if not hasattr(cfg.MODEL, "BOUNDARY"):
        cfg.MODEL.BOUNDARY = CN()

    # ============ Main switches ============
    # Enable boundary supervision (Setting B+)
    cfg.MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION = False

    # Enable query boundary prior (Setting C+)
    cfg.MODEL.BOUNDARY.USE_QUERY_PRIOR = False

    # Enable overlap penalty (Setting D)
    cfg.MODEL.BOUNDARY.USE_OVERLAP_PENALTY = False

    # ============ Boundary target generation ============
    # Dilation radius for boundary extraction
    cfg.MODEL.BOUNDARY.DILATION_RADIUS = 4

    # Dilation radius for contact boundary
    cfg.MODEL.BOUNDARY.CONTACT_DILATION = 3

    # Radius for boundary band (used in overlap penalty)
    cfg.MODEL.BOUNDARY.BAND_RADIUS = 5

    # Handle overlapping masks in GT generation
    cfg.MODEL.BOUNDARY.HANDLE_OVERLAPS = True

    # ============ Boundary head ============
    # Hidden dimension for boundary head
    cfg.MODEL.BOUNDARY.HEAD_HIDDEN_DIM = 64

    # Number of groups for GroupNorm in boundary head
    cfg.MODEL.BOUNDARY.NUM_GROUPS = 8

    # Output bias for boundary head (negative for sparse boundaries)
    cfg.MODEL.BOUNDARY.OUTPUT_BIAS = -2.0

    # Use enhanced BoundaryHeadV2 (deeper, wider, optional upsampling)
    cfg.MODEL.BOUNDARY.USE_BOUNDARY_HEAD_V2 = False

    # Enable 2x upsampling in BoundaryHeadV2 (predict at 1/2 instead of 1/4 res)
    cfg.MODEL.BOUNDARY.BOUNDARY_HEAD_UPSAMPLE = True

    # Feature source for boundary head
    # "mask_features" (default): semantic features from pixel decoder (existing behavior)
    # "lateral": edge-preserving features from FPN lateral conv (before semantic addition)
    # "concat": concatenation of lateral + mask_features (both streams)
    cfg.MODEL.BOUNDARY.FEATURE_SOURCE = "mask_features"

    # Detach lateral features from gradient computation
    # Prevents boundary loss from affecting backbone/pixel decoder via lateral path
    cfg.MODEL.BOUNDARY.DETACH_LATERAL = False

    # Use depth edge map as additional input channel to boundary head
    # Computes Sobel gradient magnitude of sensor depth → strong edges at transparent objects
    cfg.MODEL.BOUNDARY.USE_DEPTH_EDGES = False

    # Max depth value (mm) for normalization before edge computation
    cfg.MODEL.BOUNDARY.DEPTH_MAX_MM = 2000.0

    # ============ Depth-guided feature modulation ============
    # Fuse sensor depth into mask_features via gated residual:
    #   mask_features = mask_features + gamma * depth_encoder(depth)
    # Sensor depth fails on transparent objects → natural transparency signal
    cfg.MODEL.BOUNDARY.USE_DEPTH_FUSION = False

    # Initial value for learnable gamma (gating scalar)
    cfg.MODEL.BOUNDARY.DEPTH_FUSION_GAMMA_INIT = 0.1

    # Hidden dimension for DepthEncoder CNN
    cfg.MODEL.BOUNDARY.DEPTH_ENCODER_HIDDEN_DIM = 64

    # ============ Boundary spatial prior for decoder queries ============
    # When enabled, the boundary head runs BEFORE the transformer decoder.
    # Its output (detached) is used as a spatial attention bias in cross-attention,
    # guiding queries toward regions where objects/boundaries exist.
    # No gradient conflict: boundary head trains from its own loss independently.
    cfg.MODEL.BOUNDARY.USE_BOUNDARY_SPATIAL_PRIOR = False

    # Initial value for learnable alpha (attention bias strength).
    # Starts small so the model begins from the existing baseline
    # and gradually learns how much to trust the boundary signal.
    cfg.MODEL.BOUNDARY.BOUNDARY_ALPHA_INIT = 0.1

    # ============ B2M (Boundary-to-Mask) Feature Fusion ============
    # Fuse intermediate boundary features into mask_features via gated residual:
    #   mask_features = mask_features + gamma * proj(boundary_intermediate)
    # Following BMask R-CNN: intermediate features (128-dim) are richer than
    # final predictions (2-channel). Boundary head trains independently (detached);
    # B2M projection trains from mask losses via decoder backprop.
    cfg.MODEL.BOUNDARY.USE_B2M_FUSION = False

    # Initial value for learnable gamma (gating scalar).
    # gamma = gamma_max * sigmoid(gamma_raw), with gamma_raw=0 -> gamma=gamma_init.
    cfg.MODEL.BOUNDARY.B2M_GAMMA_INIT = 0.1

    # ============ Decoder Boundary Dice Loss ============
    # Per-layer boundary supervision using Sobel edge extraction.
    # Extracts edges from decoder mask predictions at each layer,
    # compares with GT edges via dice loss.
    # Zero inference cost, no new parameters.
    cfg.MODEL.BOUNDARY.DECODER_BOUNDARY_LOSS_ENABLED = False
    cfg.MODEL.BOUNDARY.DECODER_BOUNDARY_LOSS_WEIGHT = 2.0

    # ============ Query boundary prior ============
    # Hidden dimension for query prior projections
    cfg.MODEL.BOUNDARY.QUERY_PRIOR_DIM = 32

    # Alpha scale for query prior (controls modulation strength)
    cfg.MODEL.BOUNDARY.ALPHA_SCALE = 0.3

    # ============ Loss weights ============
    # FG boundary loss weight
    cfg.MODEL.BOUNDARY.FG_WEIGHT = 1.0

    # Contact boundary loss weight
    cfg.MODEL.BOUNDARY.CONTACT_WEIGHT = 2.0

    # Overlap penalty weight
    cfg.MODEL.BOUNDARY.OVERLAP_WEIGHT = 1.0

    # Focal loss alpha
    cfg.MODEL.BOUNDARY.FOCAL_ALPHA = 0.25

    # Focal loss gamma
    cfg.MODEL.BOUNDARY.FOCAL_GAMMA = 2.0

    # ============ Boundary Head Dice Loss ============
    # Add dice loss to boundary head (alongside focal loss) for direct IoU optimization.
    # Dice loss targets region overlap, complementing focal loss's per-pixel accuracy.
    cfg.MODEL.BOUNDARY.BOUNDARY_DICE_LOSS_ENABLED = False
    cfg.MODEL.BOUNDARY.BOUNDARY_DICE_FG_WEIGHT = 1.0
    cfg.MODEL.BOUNDARY.BOUNDARY_DICE_CONTACT_WEIGHT = 2.0

    # ============ GT Interpolation Mode ============
    # How to downsample boundary GT when prediction resolution differs from GT.
    # "nearest": preserves binary values but can lose thin boundaries at low resolution
    # "bilinear": smoother downsampling, preserves thin boundaries better (re-binarized at 0.3)
    cfg.MODEL.BOUNDARY.GT_INTERPOLATION_MODE = "nearest"

    # ============ Training schedules ============
    # Warmup iterations for overlap penalty
    cfg.MODEL.BOUNDARY.OVERLAP_WARMUP_ITERS = 2000

    # Warmup iterations for teacher forcing
    cfg.MODEL.BOUNDARY.TEACHER_FORCING_WARMUP = 5000

    # ============ Overlap penalty settings ============
    # Use contact band only for overlap penalty (D1 vs D2)
    cfg.MODEL.BOUNDARY.OVERLAP_CONTACT_ONLY = True

    # Minimum confidence for overlap penalty queries
    cfg.MODEL.BOUNDARY.OVERLAP_MIN_CONF = 0.0

    # ============ Dataset mapper ============
    # Use boundary-aware dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "boundary_instance"

    # ============ Copy-Paste Augmentation (Setting I) ============
    # LSJ (Large Scale Jittering) settings
    cfg.INPUT.IMAGE_SIZE = 384
    cfg.INPUT.MIN_SCALE = 0.5   # min >= 0.5 to preserve small objects
    cfg.INPUT.MAX_SCALE = 1.5

    # Copy-Paste settings
    cfg.INPUT.COPY_PASTE_PROB = 0.5           # 50% chance to apply copy-paste
    cfg.INPUT.MAX_PASTE_OBJECTS = 5           # Max objects to paste per image
    cfg.INPUT.BLEND_SIGMA = 2.0               # Gaussian blur for alpha blending
    cfg.INPUT.SAME_SCENE_ONLY = True          # Same scene for transparent objects

    # Small object prioritization
    cfg.INPUT.SMALL_OBJECT_THRESHOLD = 1024   # 32x32 = COCO small definition
    cfg.INPUT.SMALL_OBJECT_PRIORITY = 0.8     # 80% chance to prioritize small objects

    # Size filters (reduced for small objects)
    cfg.INPUT.MIN_OBJECT_SIZE_EXTRACT = 25    # Min pixels to extract (5x5)
    cfg.INPUT.MIN_OBJECT_SIZE_FINAL = 16      # Min pixels after augmentation (4x4)
    cfg.INPUT.OCCLUSION_THRESHOLD = 0.20      # Keep if >20% visible

    # Paste scale augmentation
    cfg.INPUT.PASTE_SCALE_MIN = 0.8
    cfg.INPUT.PASTE_SCALE_MAX = 1.5

    # V3: Region-aware pasting settings
    cfg.INPUT.REGION_AWARE_PASTE = True    # Paste within valid region (table area)
    cfg.INPUT.REGION_MARGIN = 20           # Margin around valid region (pixels)

    # ============ Guided Filter (inference-time mask refinement) ============
    cfg.MODEL.BOUNDARY.USE_GUIDED_FILTER = True  # enabled by default for backward compat

    # ============ Content-Adaptive Query Initialization (Direction A) ============
    # Replace static queries with encoder-selected content queries (Mask DINO style).
    # Uses hybrid approach: num_static + num_content = NUM_OBJECT_QUERIES.
    cfg.MODEL.MASK_FORMER.CONTENT_QUERY_ENABLED = False
    cfg.MODEL.MASK_FORMER.NUM_CONTENT_QUERIES = 50      # number of content queries
    cfg.MODEL.MASK_FORMER.CONTENT_QUERY_WARMUP = 2000   # iters before activating content queries
    cfg.MODEL.MASK_FORMER.CONTENT_MASK_SCORE_WEIGHT = 0.5  # weight for mask quality in scoring

    # Encoder loss weights (applied when content queries are enabled)
    cfg.MODEL.MASK_FORMER.ENC_LOSS_CE_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.ENC_LOSS_MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.ENC_LOSS_DICE_WEIGHT = 5.0

    # ============ Contrastive Denoising Training (CDN) ============
    # Adds noised GT queries during training to teach decoder progressive refinement.
    # Training-only mechanism — zero overhead at inference.
    cfg.MODEL.MASK_FORMER.CDN_ENABLED = False
    cfg.MODEL.MASK_FORMER.CDN_NUMBER = 5          # number of DN query groups
    cfg.MODEL.MASK_FORMER.CDN_BOX_NOISE_SCALE = 1.0  # box noise magnitude
    cfg.MODEL.MASK_FORMER.CDN_LABEL_NOISE_RATIO = 0.2  # label flip probability scale

    # CDN loss weights (separate from main loss for independent tuning)
    cfg.MODEL.MASK_FORMER.CDN_LOSS_CE_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.CDN_LOSS_MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.CDN_LOSS_DICE_WEIGHT = 5.0

    # ============ Multi-Scale Test-Time Augmentation (Setting M) ============
    # Runs inference at multiple scales to improve small object detection.
    #
    # Key insight: Small objects are often missed at normal resolution.
    # By running inference at larger scales (1.5x, 2.0x), small objects
    # become larger in feature space and more likely to be detected.
    #
    # This is inference-only - no retraining needed!
    # Use with best model (V2b) to test improvement.

    # Create TEST.AUG node if it doesn't exist
    if not hasattr(cfg.TEST, "AUG"):
        cfg.TEST.AUG = CN()

    # Enable multi-scale TTA (instance segmentation)
    cfg.TEST.AUG.MULTISCALE_ENABLED = False

    # Scales to use for multi-scale inference
    # Higher scales (1.5, 2.0) help detect small objects
    cfg.TEST.AUG.SCALES = [1.0, 1.5, 2.0]

    # Use horizontal flip augmentation
    cfg.TEST.AUG.FLIP = True

    # NMS threshold for merging predictions from multiple scales
    cfg.TEST.AUG.NMS_THRESHOLD = 0.5

    # Maximum detections per image (after NMS)
    cfg.TEST.AUG.MAX_DETECTIONS = 100


def get_setting_a_config():
    """
    Get config overrides for Setting A: Baseline.

    No boundary supervision - standard Mask2Former.
    """
    return {
        "MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION": False,
        "MODEL.BOUNDARY.USE_QUERY_PRIOR": False,
        "MODEL.BOUNDARY.USE_OVERLAP_PENALTY": False,
    }


def get_setting_d2_config():
    """
    Get config overrides for Setting D2: + Overlap Penalty (union band).

    Full model with overlap penalty in fg+contact union band.
    """
    return {
        "MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION": True,
        "MODEL.BOUNDARY.USE_QUERY_PRIOR": True,
        "MODEL.BOUNDARY.USE_OVERLAP_PENALTY": True,
        "MODEL.BOUNDARY.OVERLAP_CONTACT_ONLY": False,
    }
