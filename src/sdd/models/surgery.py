"""Post-build edits of an Ultralytics DetectionModel (plan §5.2).

Editing the built ``nn.Sequential`` instead of the yaml keeps every layer index unchanged, so head
``from`` indices, ``freeze=<n>`` and weight transfer for untouched layers all keep working.
Replaced layers inherit the Ultralytics routing attributes (``f``, ``i``, ``type``, ``np``).
"""

from __future__ import annotations

from torch import nn
from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import initialize_weights

from .modules import ATTENTION, AttnWrap, SPDConv


def _adopt(new: nn.Module, old: nn.Module, suffix: str) -> nn.Module:
    new.f, new.i = old.f, old.i
    new.type = f"{old.type}+{suffix}"
    new.np = sum(p.numel() for p in new.parameters())
    return new


def _out_channels(block: nn.Module) -> int:
    if hasattr(block, "cv2") and hasattr(block.cv2, "conv"):  # C2f / C3k2: cv2 is the output projection
        return block.cv2.conv.out_channels
    convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1].out_channels


def n_backbone(model) -> int:
    return len(model.yaml["backbone"])


def add_attention(model, kind: str, layers: list[int] | None = None) -> list[int]:
    nb = n_backbone(model)
    if layers is None:
        layers = [i for i in range(nb) if type(model.model[i]).__name__ == "C3k2"]
    for i in layers:
        old = model.model[i]
        if isinstance(old, AttnWrap):
            continue
        attn = ATTENTION[kind](_out_channels(old))
        initialize_weights(attn)
        model.model[i] = _adopt(AttnWrap(old, attn), old, kind)
    return layers


def replace_strided_convs(model, layers: list[int] | None = None) -> list[int]:
    nb = n_backbone(model)
    if layers is None:
        layers = [i for i in range(1, nb) if isinstance(model.model[i], Conv) and model.model[i].conv.stride == (2, 2)]
    for i in layers:
        old = model.model[i]
        if not isinstance(old, Conv) or old.conv.stride != (2, 2):
            raise ValueError(f"layer {i} ({type(old).__name__}) is not a stride-2 Conv")
        new = SPDConv(old.conv.in_channels, old.conv.out_channels, old.conv.kernel_size[0])
        initialize_weights(new)
        model.model[i] = _adopt(new, old, "spd")
    return layers


def apply_backbone(model, cfg: dict) -> list[int]:
    """Apply the variant's backbone modification; returns the modified layer indices."""
    kind = cfg.get("backbone", "none") or "none"
    if kind == "none":
        return []
    if kind in ATTENTION:
        return add_attention(model, kind, cfg.get("attn_layers"))
    if kind == "spd":
        return replace_strided_convs(model, cfg.get("spd_layers"))
    raise ValueError(f"Unknown backbone modification '{kind}'")
