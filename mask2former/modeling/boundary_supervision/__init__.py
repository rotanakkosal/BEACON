# Copyright (c) 2025
# Boundary Supervision Approach for Transparent Object Instance Segmentation
"""
Boundary Supervision Module.

This module contains components for boundary-aware instance segmentation:
- BoundaryHead: predicts FG and contact boundaries
- QueryBoundaryPrior: query-conditioned boundary prior for mask modulation
- OverlapPenaltyLoss: penalizes mask overlap in contact regions
- BoundaryCriterion: combined boundary losses
"""

from .boundary_head import BoundaryHead, BoundaryHeadV2, BoundaryToMaskProjection, build_boundary_head
from .query_boundary_prior import QueryBoundaryPrior, build_query_boundary_prior
from .overlap_penalty import (
    OverlapPenaltyLoss,
    OverlapPenaltyLossD1,
    OverlapPenaltyLossD2,
    build_overlap_penalty_loss,
)
from .boundary_criterion import BoundaryCriterion, build_boundary_criterion

__all__ = [
    "BoundaryHead",
    "BoundaryHeadV2",
    "BoundaryToMaskProjection",
    "QueryBoundaryPrior",
    "OverlapPenaltyLoss",
    "OverlapPenaltyLossD1",
    "OverlapPenaltyLossD2",
    "BoundaryCriterion",
    "build_boundary_head",
    "build_query_boundary_prior",
    "build_overlap_penalty_loss",
    "build_boundary_criterion",
]
