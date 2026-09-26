"""Box-regression losses for small objects (plan §5.2), plugged into Ultralytics' ``v8DetectionLoss``.

Inside ``BboxLoss.forward`` boxes are xyxy in feature-grid units (divided by the anchor stride); ``stride``
is passed along, so distance-based terms (NWD) are computed back in input-image pixels and ``nwd_C`` is a
pixel constant.

* ``ciou``        Ultralytics default
* ``nwd``         (1 - r) * L_CIoU + r * (1 - NWD)                           Wang et al. 2021
* ``wiou``        Wise-IoU v3 (dynamic non-monotonic focusing)               Tong et al. 2023
* ``inner_ciou``  L_CIoU + IoU - IoU_inner (auxiliary boxes scaled by ratio)  Zhang et al. 2023

Classification (class imbalance, plan §2.5.3c):

* ``cls_loss: focal``  element-wise focal BCE (Lin et al. 2017) replacing the BCE of ``v8DetectionLoss``
* class-weighted BCE needs no patch: Ultralytics 8.4 applies inverse-frequency weights when the train argument
  ``cls_pw`` > 0 (1 = 1/freq, 0.5 = 1/sqrt(freq)); variants pass it through ``train_args``.
"""

from __future__ import annotations

import torch
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner


def wasserstein_similarity(b1: torch.Tensor, b2: torch.Tensor, C: float, eps: float = 1e-7) -> torch.Tensor:
    """Normalised Gaussian Wasserstein distance between xyxy boxes (element-wise, broadcastable). Returns (..., 1)."""
    c1, c2 = (b1[..., :2] + b1[..., 2:]) / 2, (b2[..., :2] + b2[..., 2:]) / 2
    wh1, wh2 = (b1[..., 2:] - b1[..., :2]).clamp(min=eps), (b2[..., 2:] - b2[..., :2]).clamp(min=eps)
    w2 = ((c1 - c2) ** 2).sum(-1, keepdim=True) + (((wh1 - wh2) / 2) ** 2).sum(-1, keepdim=True)
    return torch.exp(-torch.sqrt(w2 + eps) / C)


def _scale_boxes(b: torch.Tensor, ratio: float) -> torch.Tensor:
    c = (b[..., :2] + b[..., 2:]) / 2
    half = (b[..., 2:] - b[..., :2]) * ratio / 2
    return torch.cat([c - half, c + half], -1)


class SDDBboxLoss(BboxLoss):
    def __init__(self, reg_max: int = 16, loss: str = "ciou", nwd_ratio: float = 0.5, nwd_C: float = 12.8,
                 inner_ratio: float = 0.75, wiou_momentum: float = 0.01, wiou_alpha: float = 1.9,
                 wiou_delta: float = 3.0):
        super().__init__(reg_max)
        if loss not in {"ciou", "nwd", "wiou", "inner_ciou"}:
            raise ValueError(f"Unknown box loss '{loss}'")
        self.loss_type, self.nwd_ratio, self.nwd_C, self.inner_ratio = loss, nwd_ratio, nwd_C, inner_ratio
        self.wiou_m, self.wiou_alpha, self.wiou_delta = wiou_momentum, wiou_alpha, wiou_delta
        self.register_buffer("wiou_mean", torch.tensor(1.0))

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        loss_ciou, loss_dfl = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask, imgsz, stride)
        if self.loss_type == "ciou":
            return loss_ciou, loss_dfl
        idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[idx].sum(-1, keepdim=True)
        pb, tb = pred_bboxes[idx], target_bboxes[idx]

        if self.loss_type == "nwd":
            s = stride[idx[1]]  # (n_fg, 1) pixels per grid cell
            nwd = wasserstein_similarity(pb * s, tb * s, self.nwd_C)
            loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
            return (1 - self.nwd_ratio) * loss_ciou + self.nwd_ratio * loss_nwd, loss_dfl

        if self.loss_type == "inner_ciou":
            iou = bbox_iou(pb, tb, xywh=False)
            inner = bbox_iou(_scale_boxes(pb, self.inner_ratio), _scale_boxes(tb, self.inner_ratio), xywh=False)
            return loss_ciou + ((iou - inner) * weight).sum() / target_scores_sum, loss_dfl

        # Wise-IoU v3
        iou = bbox_iou(pb, tb, xywh=False)
        l_iou = 1.0 - iou
        cp, ct = (pb[:, :2] + pb[:, 2:]) / 2, (tb[:, :2] + tb[:, 2:]) / 2
        enc_wh = torch.max(pb[:, 2:], tb[:, 2:]) - torch.min(pb[:, :2], tb[:, :2])
        r_wiou = torch.exp(((cp - ct) ** 2).sum(-1, keepdim=True) / (enc_wh**2).sum(-1, keepdim=True).detach().clamp(min=1e-7))
        if torch.is_grad_enabled() and l_iou.numel():  # update the running mean in training only, not in val
            self.wiou_mean.mul_(1 - self.wiou_m).add_(self.wiou_m * l_iou.detach().mean())
        beta = l_iou.detach() / self.wiou_mean.clamp(min=1e-7)
        focus = beta / (self.wiou_delta * torch.pow(self.wiou_alpha, beta - self.wiou_delta))
        return ((focus * r_wiou * l_iou) * weight).sum() / target_scores_sum, loss_dfl


class FocalBCE(torch.nn.Module):
    """Element-wise focal BCE with logits; works with Ultralytics' soft (IoU-aware) targets."""

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label, reduction="none")
        p = pred.sigmoid()
        p_t = label * p + (1 - label) * (1 - p)
        loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha > 0:
            loss = loss * (label * self.alpha + (1 - label) * (1 - self.alpha))
        return loss


class NWDTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigner whose alignment metric uses (1 - r) * CIoU + r * NWD (inputs are already in pixels here)."""

    nwd_ratio: float = 0.5
    nwd_C: float = 12.8

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        ciou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        nwd = wasserstein_similarity(gt_bboxes, pd_bboxes, self.nwd_C).squeeze(-1)
        return (1 - self.nwd_ratio) * ciou + self.nwd_ratio * nwd


def patch_criterion(crit, cfg: dict):
    """Swap the box loss (and optionally the assigner metric) of a built ``v8DetectionLoss``."""
    loss = cfg.get("loss", "ciou")
    if loss != "ciou":
        crit.bbox_loss = SDDBboxLoss(
            crit.reg_max, loss, nwd_ratio=cfg.get("nwd_ratio", 0.5), nwd_C=cfg.get("nwd_C", 12.8),
            inner_ratio=cfg.get("inner_ratio", 0.75),
        ).to(crit.device)
    if cfg.get("cls_loss", "bce") == "focal":
        crit.bce = FocalBCE(cfg.get("focal_gamma", 1.5), cfg.get("focal_alpha", 0.25))
    if cfg.get("assigner_nwd"):
        crit.assigner.__class__ = NWDTaskAlignedAssigner
        crit.assigner.nwd_ratio = cfg.get("nwd_ratio", 0.5)
        crit.assigner.nwd_C = cfg.get("nwd_C", 12.8)
    return crit
