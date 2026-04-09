# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_maskformer2_config
from .config_boundary_supervision import add_boundary_supervision_config

__all__ = [
    "add_maskformer2_config",
    "add_boundary_supervision_config",
]
