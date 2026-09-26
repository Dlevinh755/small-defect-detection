"""Backbone feature-map visualisation for the slides (idea taken from the team's DETR notebook).

Two views, both on the same letterboxed 640 input so runs are directly comparable:

* :func:`channel_grid`  - the ``n`` most active channels of one layer (the notebook's per-channel view)
* :func:`compare_runs`  - rows = runs (e.g. base vs SimAM vs SPD), columns = backbone layers; each cell is the
  channel-mean activation upsampled and overlaid on the image, with the GT boxes drawn. This is the view that
  shows whether a modification keeps small defects alive in the deeper (stride 16-32) maps.

For layers wrapped by :class:`sdd.models.modules.AttnWrap` the hook reads the post-attention output; pass
``pre_attention=True`` to read the block output before the attention module instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import style

# one-hue sequential ramp (low activation recedes, high activation is dark blue)
ACT_CMAP = LinearSegmentedColormap.from_list("act", ["#ffffff00", *style.SEQUENTIAL])


def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[torch.Tensor, np.ndarray, float, tuple[int, int]]:
    """Ultralytics-style letterbox (gray 114 padding). Returns (tensor 1x3xSxS in [0,1] RGB, RGB uint8, scale, pad)."""
    h, w = img_bgr.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return t, rgb, r, (left, top)


@torch.no_grad()
def capture(model: torch.nn.Module, x: torch.Tensor, layers: list[int], pre_attention: bool = False) -> dict:
    """Outputs of ``model.model[i]`` for an Ultralytics DetectionModel. Returns {layer: (C, H, W) tensor}."""
    feats, hooks = {}, []
    for i in layers:
        mod = model.model[i]
        if pre_attention and hasattr(mod, "block"):
            mod = mod.block
        hooks.append(mod.register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, out[0].detach().float().cpu())))
    was_training = model.training
    model.eval()
    try:
        p = next(model.parameters())
        model(x.to(p.device, p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return feats


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def mean_activation(feat: torch.Tensor, size: int, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Channel-mean |activation| upsampled to the input; ``crop`` (x1, y1, x2, y2) drops the letterbox padding
    before normalising, so the padding does not dominate the colour scale."""
    m = cv2.resize(feat.abs().mean(0).numpy(), (size, size), interpolation=cv2.INTER_LINEAR)
    if crop is not None:
        x1, y1, x2, y2 = crop
        m = m[y1:y2, x1:x2]
    return _norm(m)


def _draw_gt(ax, boxes, scale, pad):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(plt.Rectangle((x1 * scale + pad[0], y1 * scale + pad[1]), (x2 - x1) * scale, (y2 - y1) * scale,
                                   fill=False, ec="#eb6834", lw=1.2))


def channel_grid(feat: torch.Tensor, n: int = 16, cols: int = 8, title: str = ""):
    """The ``n`` channels with the highest mean activation of one layer."""
    idx = feat.abs().mean((1, 2)).argsort(descending=True)[:n]
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6 + 0.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, c in zip(np.atleast_1d(axes).ravel(), idx.tolist()):
        ax.imshow(_norm(feat[c].numpy()), cmap=ACT_CMAP, vmin=0, vmax=1)
        ax.set_title(f"ch {c}", fontsize=7, color=style.INK_2)
    fig.suptitle(title or f"top-{n} channels ({feat.shape[0]}x{feat.shape[1]}x{feat.shape[2]})", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


def compare_runs(models: dict[str, torch.nn.Module], img_bgr: np.ndarray, layers: list[int], gt_boxes=None,
                 size: int = 640, pre_attention: bool = False, alpha: float = 0.75):
    """Rows = models, columns = layers; channel-mean activation over the input, GT boxes in orange."""
    x, rgb, scale, pad = letterbox(img_bgr, size)
    h, w = img_bgr.shape[:2]
    crop = (pad[0], pad[1], pad[0] + round(w * scale), pad[1] + round(h * scale))
    rgb = rgb[crop[1] : crop[3], crop[0] : crop[2]]
    pad = (0, 0)  # boxes are drawn on the cropped view
    gt_boxes = np.zeros((0, 4)) if gt_boxes is None else np.asarray(gt_boxes)
    cell_h = 2.4
    cell_w = float(np.clip(cell_h * w / h, 1.3, 4.0))  # follow the image aspect ratio (no empty gutters)
    fig, axes = plt.subplots(len(models), len(layers) + 1, squeeze=False,
                             figsize=(cell_w * (len(layers) + 1) + 0.4, cell_h * len(models) + 0.4))
    for r, (name, model) in enumerate(models.items()):
        feats = capture(model, x, layers, pre_attention)
        ax = axes[r][0]
        ax.imshow(rgb)
        _draw_gt(ax, gt_boxes, scale, pad)
        ax.set_ylabel(name, fontsize=9, color=style.INK)
        for c, i in enumerate(layers, start=1):
            f = feats[i]
            ax = axes[r][c]
            ax.imshow(rgb, alpha=0.35)
            ax.imshow(mean_activation(f, size, crop), cmap=ACT_CMAP, vmin=0, vmax=1, alpha=alpha)
            _draw_gt(ax, gt_boxes, scale, pad)
            if r == 0:
                ax.set_title(f"layer {i}\nstride {size // f.shape[-1]}", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0][0].set_title("input\n+ GT", fontsize=9)
    fig.tight_layout()
    return fig


def load_run_model(run_dir: Path) -> torch.nn.Module:
    """The Ultralytics nn.Module of a finished run (best.pt)."""
    import sdd.models.yolo_trainer  # noqa: F401  (custom classes in the checkpoint)
    from ultralytics import YOLO

    return YOLO(str(run_dir / "train" / "weights" / "best.pt")).model.float()
