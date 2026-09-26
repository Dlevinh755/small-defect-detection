"""Dataset ids -> built dataset directory, building derived datasets on demand.

    neu              registry dataset (configs/datasets.yaml)
    gc10_bal_v1      balanced train split (sdd.data.balance), also _aug_v1 / _rfs_v1
    pcb_t640         640 tiles for train/val, full test images (sdd.data.tiling)
    pcb_bal_v1_t640  derived ids compose: tiles of the balanced dataset
    merged           must be built explicitly (scripts/prepare_data.py --merge ...)
"""

from __future__ import annotations

from pathlib import Path

from ..config import datasets_cfg
from ..env import paths
from .balance import balance_dataset, parse_balanced
from .build import build_dataset
from .tiling import parse_tiled, tile_dataset


def ensure_dataset(name: str) -> Path:
    if name in datasets_cfg()["datasets"]:
        return build_dataset(name)
    tiled = parse_tiled(name)
    if tiled:
        ensure_dataset(tiled[0])
        return tile_dataset(*tiled)
    bal = parse_balanced(name)
    if bal:
        return balance_dataset(*bal)
    d = paths().data_dir(name)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d} not built (merged datasets: scripts/prepare_data.py --merge ...)")
    return d


def base_dataset(name: str) -> str:
    """Registry dataset a derived id is evaluated against (its test split is that dataset's test split)."""
    while name not in datasets_cfg()["datasets"]:
        t, b = parse_tiled(name), parse_balanced(name)
        if t:
            name = t[0]
        elif b:
            name = b[0]
        else:
            return name
    return name


def data_version(name: str) -> str:
    """'orig', 'bal_v1', 'aug_v1', ... (tiling reported separately)."""
    t = parse_tiled(name)
    b = parse_balanced(t[0] if t else name)
    return f"{b[1]}_v{b[2]}" if b else "orig"
