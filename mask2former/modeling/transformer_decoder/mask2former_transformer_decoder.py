# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .cdn_components import prepare_for_cdn, build_dn_loss_indices


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        progressive_attn_enabled: bool = False,
        progressive_attn_min_threshold: float = 0.2,
        progressive_attn_max_threshold: float = 0.5,
        progressive_attn_warmup_layers: int = 3,
        use_boundary_spatial_prior: bool = False,
        boundary_alpha_init: float = 0.1,
        # Content-adaptive query initialization (Mask DINO style)
        content_query_enabled: bool = False,
        num_content_queries: int = 50,
        content_query_warmup: int = 2000,
        content_mask_score_weight: float = 0.5,
        # Contrastive Denoising Training (CDN)
        cdn_enabled: bool = False,
        cdn_number: int = 5,
        cdn_box_noise_scale: float = 1.0,
        cdn_label_noise_ratio: float = 0.2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            progressive_attn_enabled: use per-layer attention thresholds
            progressive_attn_min_threshold: threshold for earliest layer
            progressive_attn_max_threshold: threshold for later layers
            progressive_attn_warmup_layers: layers over which to ramp
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # Progressive masked attention: per-layer threshold schedule
        self.progressive_attn_enabled = progressive_attn_enabled
        self.attn_thresholds = []
        for i in range(dec_layers + 1):
            if not progressive_attn_enabled or i >= progressive_attn_warmup_layers:
                self.attn_thresholds.append(progressive_attn_max_threshold)
            else:
                t = progressive_attn_min_threshold + (
                    progressive_attn_max_threshold - progressive_attn_min_threshold
                ) * (i / progressive_attn_warmup_layers)
                self.attn_thresholds.append(t)

        # Boundary spatial prior: uses boundary head output as attention bias
        # to guide queries toward regions where objects (boundaries) exist.
        # The boundary map is .detach()'ed — no gradient conflict.
        self.use_boundary_spatial_prior = use_boundary_spatial_prior
        if use_boundary_spatial_prior:
            self.boundary_alpha = nn.Parameter(torch.tensor(boundary_alpha_init))

        # Content-adaptive query initialization (Direction A)
        # Selects top-K encoder tokens as content queries based on combined
        # class + mask quality scoring. Uses hybrid approach: n_static static
        # queries + n_content content queries = num_queries total.
        self.content_query_enabled = content_query_enabled
        self.num_content_queries = num_content_queries
        self.num_static_queries = num_queries - num_content_queries
        self.content_query_warmup = content_query_warmup
        self.content_mask_score_weight = content_mask_score_weight
        if content_query_enabled:
            # Project encoder memory for scoring (identity-initialized for stable start)
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            nn.init.eye_(self.enc_output.weight)
            nn.init.zeros_(self.enc_output.bias)
            # Positional embedding projection for content queries
            # Maps 2D normalized coordinates to hidden_dim positional embeddings
            self.content_pos_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        # Contrastive Denoising Training (CDN)
        # Adds noised GT queries during training to teach decoder refinement.
        # Training-only — removed at inference.
        self.cdn_enabled = cdn_enabled
        self.cdn_number = cdn_number
        self.cdn_box_noise_scale = cdn_box_noise_scale
        self.cdn_label_noise_ratio = cdn_label_noise_ratio
        if cdn_enabled:
            # Label embedding: encodes GT class labels → content queries for DN queries
            self.cdn_label_enc = nn.Embedding(num_classes + 1, hidden_dim)
            # Position projection: maps box-derived sinusoidal PE → positional queries
            self.cdn_pos_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["progressive_attn_enabled"] = cfg.MODEL.MASK_FORMER.PROGRESSIVE_ATTN_ENABLED
        ret["progressive_attn_min_threshold"] = cfg.MODEL.MASK_FORMER.PROGRESSIVE_ATTN_MIN_THRESHOLD
        ret["progressive_attn_max_threshold"] = cfg.MODEL.MASK_FORMER.PROGRESSIVE_ATTN_MAX_THRESHOLD
        ret["progressive_attn_warmup_layers"] = cfg.MODEL.MASK_FORMER.PROGRESSIVE_ATTN_WARMUP_LAYERS

        # Boundary spatial prior config (from BOUNDARY namespace)
        boundary_cfg = getattr(cfg.MODEL, "BOUNDARY", None)
        if boundary_cfg is not None:
            ret["use_boundary_spatial_prior"] = getattr(boundary_cfg, "USE_BOUNDARY_SPATIAL_PRIOR", False)
            ret["boundary_alpha_init"] = getattr(boundary_cfg, "BOUNDARY_ALPHA_INIT", 0.1)
        else:
            ret["use_boundary_spatial_prior"] = False
            ret["boundary_alpha_init"] = 0.1

        # Content-adaptive query initialization config
        ret["content_query_enabled"] = cfg.MODEL.MASK_FORMER.CONTENT_QUERY_ENABLED
        ret["num_content_queries"] = cfg.MODEL.MASK_FORMER.NUM_CONTENT_QUERIES
        ret["content_query_warmup"] = cfg.MODEL.MASK_FORMER.CONTENT_QUERY_WARMUP
        ret["content_mask_score_weight"] = cfg.MODEL.MASK_FORMER.CONTENT_MASK_SCORE_WEIGHT

        # Contrastive Denoising Training (CDN) config
        ret["cdn_enabled"] = cfg.MODEL.MASK_FORMER.CDN_ENABLED
        ret["cdn_number"] = cfg.MODEL.MASK_FORMER.CDN_NUMBER
        ret["cdn_box_noise_scale"] = cfg.MODEL.MASK_FORMER.CDN_BOX_NOISE_SCALE
        ret["cdn_label_noise_ratio"] = cfg.MODEL.MASK_FORMER.CDN_LABEL_NOISE_RATIO

        return ret

    def _apply_boundary_prior(self, attn_mask, boundary_spatial_prior, target_size, bs):
        """Combine boolean attention mask with boundary spatial prior.

        The boundary map acts as a soft spatial bias: regions with detected
        boundaries (= likely objects) get a positive attention boost,
        making queries more likely to attend there.

        Args:
            attn_mask: [B*nheads, Q, H*W] boolean mask (True = don't attend)
            boundary_spatial_prior: [B, 1, H_feat, W_feat] boundary probability [0,1]
            target_size: (H, W) spatial size for this feature level
            bs: batch size

        Returns:
            float_mask: [B*nheads, Q, H*W] float attention mask
        """
        # Resize boundary map to match this level's spatial resolution
        bp = F.interpolate(
            boundary_spatial_prior, size=target_size,
            mode="bilinear", align_corners=False
        )
        bp = bp.flatten(2)  # [B, 1, H*W]

        # Expand to [B*nheads, 1, H*W] — broadcasts across Q dimension
        bp = bp.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)

        # Convert boolean mask to float: True → -inf, False → 0
        float_mask = torch.zeros_like(attn_mask, dtype=bp.dtype)
        float_mask.masked_fill_(attn_mask, float('-inf'))

        # Add boundary bias: positive where boundaries exist → encourage attention
        float_mask = float_mask + self.boundary_alpha * bp

        return float_mask

    def _select_content_queries(self, encoder_memory, encoder_spatial_shapes, mask_features):
        """Select top-K encoder tokens as content queries.

        Uses combined classification + mask quality scoring for robust
        selection on single-class datasets with weak visual features.

        Args:
            encoder_memory: [B, N_tokens, D] raw encoder output
            encoder_spatial_shapes: [num_levels, 2] (H, W) per level
            mask_features: [B, D, H, W] for mask quality scoring

        Returns:
            content_feat: [B, n_content, D] selected content features (detached)
            content_pos: [B, n_content, D] positional embeddings for selected tokens
            enc_outputs: dict with encoder predictions for encoder loss
        """
        bs = encoder_memory.shape[0]
        n_content = self.num_content_queries

        # Project encoder tokens (identity-initialized for stable start)
        output_memory = self.enc_output_norm(self.enc_output(encoder_memory))  # [B, N, D]

        # Score 1: Classification confidence
        enc_cls = self.class_embed(output_memory)  # [B, N, num_classes+1]
        # For scoring, use max class score (excluding no-object class)
        class_scores = enc_cls[:, :, :-1].max(-1).values  # [B, N]

        # Score 2: Mask quality (how strong the mask activation is)
        enc_mask_embed = self.mask_embed(output_memory)  # [B, N, mask_dim]
        enc_masks = torch.einsum("bnc,bchw->bnhw", enc_mask_embed, mask_features)  # [B, N, H, W]
        mask_scores = enc_masks.flatten(2).max(-1).values  # [B, N] peak activation

        # Combined scoring
        combined_scores = class_scores + self.content_mask_score_weight * mask_scores

        # Select top-K
        topk_scores, topk_indices = combined_scores.topk(n_content, dim=1)  # [B, n_content]

        # Gather content queries (DETACHED — no gradient from decoder to encoder)
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
        content_feat = torch.gather(output_memory, 1, gather_idx).detach()  # [B, n_content, D]

        # Generate positional embeddings from token spatial positions
        # Compute reference points (normalized 2D coords) for selected tokens
        device = encoder_memory.device
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(encoder_spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_.item() - 0.5, H_.item(), dtype=torch.float32, device=device),
                torch.linspace(0.5, W_.item() - 0.5, W_.item(), dtype=torch.float32, device=device),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1) / H_.item()  # normalize to [0, 1]
            ref_x = ref_x.reshape(-1) / W_.item()
            ref = torch.stack((ref_x, ref_y), -1)  # [H*W, 2]
            reference_points_list.append(ref)
        all_ref_points = torch.cat(reference_points_list, 0)  # [N_tokens, 2]

        # Gather reference points for selected tokens
        ref_gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, 2)  # [B, n_content, 2]
        selected_ref = torch.gather(
            all_ref_points.unsqueeze(0).expand(bs, -1, -1), 1, ref_gather_idx
        )  # [B, n_content, 2]

        # Generate positional embeddings using sine encoding + projection
        # Use the same PositionEmbeddingSine pattern: encode x and y separately
        N_steps = output_memory.shape[-1] // 2  # hidden_dim // 2
        dim_t = torch.arange(N_steps, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / N_steps)

        pos_x = selected_ref[:, :, 0:1] / dim_t  # [B, n_content, N_steps]
        pos_y = selected_ref[:, :, 1:2] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        content_pos_raw = torch.cat((pos_y, pos_x), dim=2)  # [B, n_content, hidden_dim]
        content_pos = self.content_pos_proj(content_pos_raw)  # [B, n_content, hidden_dim]

        # Build encoder outputs for encoder loss (using UN-detached features)
        enc_cls_selected = torch.gather(
            enc_cls, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, enc_cls.shape[-1])
        )  # [B, n_content, num_classes+1]
        enc_masks_selected = torch.gather(
            enc_masks, 1,
            topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, enc_masks.shape[-2], enc_masks.shape[-1])
        )  # [B, n_content, H, W]

        enc_outputs = {
            'pred_logits': enc_cls_selected,  # [B, n_content, num_classes+1]
            'pred_masks': enc_masks_selected,  # [B, n_content, H, W]
        }

        return content_feat, content_pos, enc_outputs

    def _generate_cdn_pos_embed(self, boxes: Tensor) -> Tensor:
        """Generate positional embeddings from CDN box coordinates.

        Uses sinusoidal encoding of box center (cx, cy) mapped through a projection MLP.

        Args:
            boxes: [B, N, 4] cxcywh normalized boxes

        Returns:
            pos_embed: [B, N, hidden_dim] positional embeddings
        """
        # Use cx, cy as 2D reference points for sinusoidal PE
        ref_points = boxes[:, :, :2]  # [B, N, 2] (cx, cy)

        N_steps = self.pe_layer.num_pos_feats  # hidden_dim // 2
        dim_t = torch.arange(N_steps, dtype=torch.float32, device=boxes.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / N_steps)

        pos_x = ref_points[:, :, 0:1] / dim_t  # [B, N, N_steps]
        pos_y = ref_points[:, :, 1:2] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_raw = torch.cat((pos_y, pos_x), dim=2)  # [B, N, hidden_dim]

        return self.cdn_pos_proj(pos_raw)  # [B, N, hidden_dim]

    def forward(self, x, mask_features, mask=None, boundary_spatial_prior=None,
                encoder_memory=None, encoder_spatial_shapes=None, global_step=0,
                targets=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Whether to apply boundary spatial prior for this forward pass
        apply_boundary = (
            self.use_boundary_spatial_prior
            and boundary_spatial_prior is not None
        )

        # ========== Query initialization: static vs hybrid (content-adaptive) ==========
        enc_outputs = None

        use_content = (
            self.content_query_enabled
            and encoder_memory is not None
            and global_step >= self.content_query_warmup
        )

        if use_content:
            # Hybrid queries: n_static static + n_content from encoder
            n_static = self.num_static_queries

            # Static queries
            static_feat = self.query_feat.weight[:n_static].unsqueeze(1).repeat(1, bs, 1)  # [n_s, B, D]
            static_embed = self.query_embed.weight[:n_static].unsqueeze(1).repeat(1, bs, 1)

            # Content-adaptive queries from encoder
            content_feat, content_pos, enc_outputs = self._select_content_queries(
                encoder_memory, encoder_spatial_shapes, mask_features
            )
            # Transpose to [n_content, B, D] for consistency with static queries
            content_feat = content_feat.transpose(0, 1)   # [n_c, B, D]
            content_pos = content_pos.transpose(0, 1)      # [n_c, B, D]

            # Combine: static first, then content
            output = torch.cat([static_feat, content_feat], dim=0)       # [Q, B, D]
            query_embed = torch.cat([static_embed, content_pos], dim=0)  # [Q, B, D]
        else:
            # All static queries (original Mask2Former behavior or during warmup)
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, D]
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)        # [Q, B, D]

            # During warmup with content queries enabled, still compute encoder
            # predictions for the encoder loss (but don't use them as queries)
            if (self.content_query_enabled and encoder_memory is not None
                    and self.training and global_step < self.content_query_warmup):
                _, _, enc_outputs = self._select_content_queries(
                    encoder_memory, encoder_spatial_shapes, mask_features
                )

        # ========== CDN: Contrastive Denoising Training ==========
        # Prepend noised GT queries before regular queries (training only)
        dn_meta = None
        cdn_self_attn_mask = None

        if self.cdn_enabled and self.training and targets is not None:
            cdn_label, cdn_bbox, cdn_self_attn_mask, dn_meta = prepare_for_cdn(
                targets=targets,
                dn_number=self.cdn_number,
                label_noise_ratio=self.cdn_label_noise_ratio,
                box_noise_scale=self.cdn_box_noise_scale,
                num_classes=self.class_embed.out_features - 1,  # num_classes (excl. no-object)
                hidden_dim=output.shape[-1],
                label_enc=self.cdn_label_enc,
                num_queries=self.num_queries,
            )

            if dn_meta is not None:
                # Generate positional embeddings from noised boxes
                cdn_pos = self._generate_cdn_pos_embed(cdn_bbox)  # [B, pad_size, D]

                # Transpose to [pad_size, B, D] and prepend
                cdn_label_t = cdn_label.transpose(0, 1)   # [pad_size, B, D]
                cdn_pos_t = cdn_pos.transpose(0, 1)        # [pad_size, B, D]

                output = torch.cat([cdn_label_t, output], dim=0)        # [pad+Q, B, D]
                query_embed = torch.cat([cdn_pos_t, query_embed], dim=0)  # [pad+Q, B, D]

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        # layer_index=0: this attn_mask will be consumed by decoder layer 0
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], layer_index=0)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Build effective attention mask with optional boundary bias
            if apply_boundary:
                effective_mask = self._apply_boundary_prior(
                    attn_mask, boundary_spatial_prior,
                    size_list[level_index], bs
                )
            else:
                effective_mask = attn_mask

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=effective_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=cdn_self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # layer_index=i+1: this attn_mask will be consumed by decoder layer i+1
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_index=i + 1)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # ========== Split CDN outputs from regular outputs ==========
        if dn_meta is not None:
            pad_size = dn_meta["pad_size"]

            # Split predictions: DN part [B, :pad_size] and regular part [B, pad_size:]
            dn_predictions_class = [c[:, :pad_size] for c in predictions_class]
            dn_predictions_mask = [m[:, :pad_size] for m in predictions_mask]
            predictions_class = [c[:, pad_size:] for c in predictions_class]
            predictions_mask = [m[:, pad_size:] for m in predictions_mask]

            # Final query features: only regular queries for boundary supervision
            final_query_features = self.decoder_norm(output[pad_size:]).transpose(0, 1)
        else:
            final_query_features = self.decoder_norm(output).transpose(0, 1)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'query_embed': final_query_features,  # [B, Q, D] for boundary supervision
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        # Attach encoder outputs for encoder loss (content-adaptive queries)
        if enc_outputs is not None:
            out['enc_outputs'] = enc_outputs

        # Attach CDN outputs for denoising loss
        if dn_meta is not None:
            out['dn_outputs'] = {
                'pred_logits': dn_predictions_class[-1],
                'pred_masks': dn_predictions_mask[-1],
            }
            out['dn_aux_outputs'] = [
                {'pred_logits': c, 'pred_masks': m}
                for c, m in zip(dn_predictions_class[:-1], dn_predictions_mask[:-1])
            ]
            out['dn_meta'] = dn_meta

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_index=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        threshold = self.attn_thresholds[layer_index] if layer_index is not None else 0.5
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < threshold).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
