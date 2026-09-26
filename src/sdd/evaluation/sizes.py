"""Size groups (plan §1.2): absolute COCO areas and relative (box area / image area) bins."""

from __future__ import annotations

import numpy as np

from ..config import protocol

COCO_SMALL, COCO_MEDIUM = 32**2, 96**2
ABS_NAMES = ["small", "medium", "large"]


def abs_group(area) -> np.ndarray:
    area = np.asarray(area, float)
    return np.where(area < COCO_SMALL, 0, np.where(area < COCO_MEDIUM, 1, 2))


def rel_bins() -> tuple[list[float], list[str]]:
    e = protocol()["eval"]
    return list(e["rel_bins"]), list(e["rel_names"])


def rel_group(rel_area, bins: list[float] | None = None) -> np.ndarray:
    bins = bins or rel_bins()[0]
    k = np.searchsorted(np.asarray(bins), np.asarray(rel_area, float), side="right") - 1
    return np.clip(k, 0, len(bins) - 2)
