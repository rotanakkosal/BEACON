# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .configs.config import add_maskformer2_config
from .configs.config_boundary_supervision import add_boundary_supervision_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.boundary_instance_dataset_mapper import (
    BoundaryInstanceDatasetMapper,
)

# models
from .models.base.maskformer_model import MaskFormer
from .models.base.test_time_augmentation import SemanticSegmentorWithTTA
from .models.boundary.maskformer_model_boundary_supervision import MaskFormerBoundarySupervision

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
