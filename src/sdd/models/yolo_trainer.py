"""Custom Ultralytics trainer: builds the variant (yaml -> backbone surgery -> weights) and swaps the loss.

Usage::

    trainer_cls = make_trainer(variant_cfg("A7"))
    YOLO(variant["model_cfg"]).train(trainer=trainer_cls, pretrained="yolo11n.pt", data=..., ...)

Checkpoints pickle :class:`SDDDetectionModel` and the modules from :mod:`sdd.models.modules`, so
``import sdd.models.yolo_trainer`` (done by ``sdd.engine``) must happen before loading them.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from .losses import patch_criterion
from .surgery import apply_backbone


class SDDDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True, sdd_cfg: dict | None = None):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.sdd_cfg = dict(sdd_cfg or {})

    def init_criterion(self):
        return patch_criterion(super().init_criterion(), getattr(self, "sdd_cfg", {}))


def build_yolo(cfg, nc: int, variant: dict, weights=None, verbose: bool = True) -> SDDDetectionModel:
    model = SDDDetectionModel(cfg, nc=nc, verbose=verbose, sdd_cfg=variant)
    if weights is not None:
        model.load(weights, verbose=verbose)  # plain source: fills every untouched layer
    changed = apply_backbone(model, variant)
    if weights is not None and changed:
        model.load(weights, verbose=False)  # source that already has the same surgery (e.g. phase-3 init from A7)
    return model


def make_trainer(variant: dict) -> type[DetectionTrainer]:
    class SDDTrainer(DetectionTrainer):
        VARIANT = dict(variant)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build_yolo(cfg, self.data["nc"], self.VARIANT, weights, verbose and RANK == -1)
            return self.set_model_names_for_load(model)

    SDDTrainer.__name__ = f"SDDTrainer_{variant.get('name', 'custom')}"
    return SDDTrainer
