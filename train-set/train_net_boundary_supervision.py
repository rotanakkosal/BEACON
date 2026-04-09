

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
except:
    pass

import copy
import itertools
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer imports
from mask2former import MaskFormerInstanceDatasetMapper
from mask2former.configs.config import add_maskformer2_config

# Boundary supervision imports
from mask2former.configs.config_boundary_supervision import add_boundary_supervision_config
from mask2former.data.dataset_mappers.boundary_instance_dataset_mapper import (
    BoundaryInstanceDatasetMapper,
)
from mask2former.data.dataset_mappers.boundary_copypaste_v2_dataset_mapper import (
    BoundaryCopyPasteV2DatasetMapper,
)
from mask2former.models.base.multiscale_tta import (
    MultiScaleInstanceSegmentorTTA,
    build_multiscale_tta,
)

from mask2former.models.boundary.maskformer_model_boundary_supervision import MaskFormerBoundarySupervision

from detectron2.data.datasets import register_coco_instances

# Register ClearPose dataset
_DATASET_ROOT = os.path.join(project_root, "datasets", "clearpose_dataset")

from detectron2.data import DatasetCatalog
if "clearpose_train" not in DatasetCatalog.list():
    register_coco_instances(
        "clearpose_train",
        {},
        os.path.join(_DATASET_ROOT, "coco_clearpose_train.json"),
        _DATASET_ROOT
    )
if "clearpose_val" not in DatasetCatalog.list():
    register_coco_instances(
        "clearpose_val",
        {},
        os.path.join(_DATASET_ROOT, "coco_clearpose_val.json"),
        _DATASET_ROOT
    )

# Register Trans10K dataset
_TRANS10K_ROOT = os.path.join(project_root, "datasets", "trans10k")
if "trans10k_train" not in DatasetCatalog.list():
    register_coco_instances(
        "trans10k_train",
        {},
        os.path.join(_TRANS10K_ROOT, "coco_trans10k_train.json"),
        os.path.join(_TRANS10K_ROOT, "images")
    )
if "trans10k_val" not in DatasetCatalog.list():
    register_coco_instances(
        "trans10k_val",
        {},
        os.path.join(_TRANS10K_ROOT, "coco_trans10k_val.json"),
        os.path.join(_TRANS10K_ROOT, "images")
    )


class BoundarySupervisionTrainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_logged_step = -1
        # Initialize GradScaler for AMP
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)

    def run_step(self):
        assert self.model.training, "[BoundarySupervisionTrainer] model was not in training mode"
        data = next(self._trainer._data_loader_iter)
        with torch.cuda.amp.autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data, global_step=self.iter)

        losses = sum(
            loss_dict[k] for k in loss_dict.keys()
            if k not in ["tf_ratio"] and isinstance(loss_dict[k], torch.Tensor)
        )

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        # Unscale before clipping
        self.grad_scaler.unscale_(self.optimizer)

        if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Log boundary-specific metrics
        self._log_boundary_metrics(loss_dict)

    def _log_boundary_metrics(self, loss_dict: Dict[str, torch.Tensor]):
        """Log boundary-specific metrics."""
        # Only log periodically
        if self.iter - self._last_logged_step < 20:
            return
        self._last_logged_step = self.iter

        storage = self.storage

        if "tf_ratio" in loss_dict:
            storage.put_scalar("boundary/tf_ratio", loss_dict["tf_ratio"].item())

        # Log boundary losses
        for key in [
            "loss_boundary_fg", "loss_boundary_contact",
            "loss_boundary_dice_fg", "loss_boundary_dice_contact",
            "loss_overlap", "loss_spatial_exclusion",
            "loss_boundary_dice",
        ]:
            if key in loss_dict:
                storage.put_scalar(f"boundary/{key}", loss_dict[key].item())

        # Log encoder losses (content-adaptive queries)
        for key in ["loss_ce_enc", "loss_mask_enc", "loss_dice_enc"]:
            if key in loss_dict:
                storage.put_scalar(f"encoder/{key}", loss_dict[key].item())

        # Log depth fusion gamma (learnable gating scalar)
        model = self.model
        if hasattr(model, "module"):
            model = model.module  # DDP unwrap
        if hasattr(model, "depth_gamma_raw"):
            import torch
            gamma = model.depth_gamma_max * torch.sigmoid(model.depth_gamma_raw)
            storage.put_scalar("depth/gamma", gamma.item())
            storage.put_scalar("depth/gamma_raw", model.depth_gamma_raw.item())

        # Log B2M fusion gamma (learnable gating scalar)
        if hasattr(model, "b2m_gamma_raw"):
            import torch as _torch
            b2m_gamma = model.b2m_gamma_max * _torch.sigmoid(model.b2m_gamma_raw)
            storage.put_scalar("boundary/b2m_gamma", b2m_gamma.item())
            storage.put_scalar("boundary/b2m_gamma_raw", model.b2m_gamma_raw.item())

        # Log boundary spatial prior alpha (learnable attention bias strength)
        if hasattr(model, "sem_seg_head") and hasattr(model.sem_seg_head, "predictor"):
            predictor = model.sem_seg_head.predictor
            if hasattr(predictor, "boundary_alpha"):
                storage.put_scalar(
                    "boundary/spatial_prior_alpha",
                    predictor.boundary_alpha.item()
                )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for COCO-style instance segmentation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if len(evaluator_list) == 0:
            # Fallback to COCO evaluator
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_MultiScaleTTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")

        # Get scales from config
        scales = getattr(cfg.TEST.AUG, "SCALES", [1.0, 1.5, 2.0])
        flip = getattr(cfg.TEST.AUG, "FLIP", True)

        logger.info(f"Running inference with multi-scale TTA (scales={scales}, flip={flip})")

        # Wrap model with multi-scale TTA
        model_with_tta = build_multiscale_tta(cfg, model)

        # Run evaluation
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_MultiScaleTTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model_with_tta, evaluators)

        # Rename results to indicate TTA was used
        res = OrderedDict({k + "_MultiScaleTTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with appropriate dataset mapper."""
        mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        logger = logging.getLogger(__name__)

        if mapper_name == "boundary_instance":
            mapper = BoundaryInstanceDatasetMapper(cfg, True)
            logger.info("[BoundarySupervisionTrainer] Using BoundaryInstanceDatasetMapper")
        elif mapper_name == "boundary_copypaste_v2":
            mapper = BoundaryCopyPasteV2DatasetMapper(cfg, True)
            logger.info("[BoundarySupervisionTrainer] Using BoundaryCopyPasteV2DatasetMapper (Simplified, No LSJ)")
        elif mapper_name == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        else:
            mapper = None

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """Build learning rate scheduler."""
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Build optimizer with proper weight decay settings."""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                # Backbone LR multiplier
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER

                # Position embedding weight decay
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0

                # Norm weight decay
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                # Embedding weight decay
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed

                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # Add configs in order
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_boundary_supervision_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Force bitmask format
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="mask2former"
    )
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="boundary_supervision"
    )

    return cfg


def main(args):
    # Dataset already registered at module level
    cfg = setup(args)

    # Log which setting is being used
    logger = logging.getLogger("boundary_supervision")
    logger.info("=" * 50)
    logger.info("Boundary Supervision Configuration:")
    logger.info(f"  USE_BOUNDARY_SUPERVISION: {cfg.MODEL.BOUNDARY.USE_BOUNDARY_SUPERVISION}")
    logger.info(f"  USE_QUERY_PRIOR: {cfg.MODEL.BOUNDARY.USE_QUERY_PRIOR}")
    logger.info(f"  USE_OVERLAP_PENALTY: {cfg.MODEL.BOUNDARY.USE_OVERLAP_PENALTY}")
    if cfg.MODEL.BOUNDARY.USE_OVERLAP_PENALTY:
        logger.info(f"  OVERLAP_CONTACT_ONLY: {cfg.MODEL.BOUNDARY.OVERLAP_CONTACT_ONLY}")
    logger.info(f"  CONTENT_QUERY_ENABLED: {cfg.MODEL.MASK_FORMER.CONTENT_QUERY_ENABLED}")
    if cfg.MODEL.MASK_FORMER.CONTENT_QUERY_ENABLED:
        logger.info(f"  NUM_CONTENT_QUERIES: {cfg.MODEL.MASK_FORMER.NUM_CONTENT_QUERIES}")
        logger.info(f"  CONTENT_QUERY_WARMUP: {cfg.MODEL.MASK_FORMER.CONTENT_QUERY_WARMUP}")
    logger.info("=" * 50)

    if args.eval_only:
        model = BoundarySupervisionTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        # Check if multi-scale TTA is enabled
        if getattr(cfg.TEST.AUG, "MULTISCALE_ENABLED", False):
            logger.info("=" * 50)
            logger.info("Running Multi-Scale TTA Evaluation")
            scales = getattr(cfg.TEST.AUG, "SCALES", [1.0, 1.5, 2.0])
            flip = getattr(cfg.TEST.AUG, "FLIP", True)
            logger.info(f"  SCALES: {scales}")
            logger.info(f"  FLIP: {flip}")
            logger.info("=" * 50)
            res = BoundarySupervisionTrainer.test_with_MultiScaleTTA(cfg, model)
        else:
            res = BoundarySupervisionTrainer.test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = BoundarySupervisionTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
