"""Loading of the yaml configs in ``configs/`` and resolution of model variants."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("SDD_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
SPLITS = Path(os.environ.get("SDD_SPLITS", ROOT / "splits"))

YOLO_MODELS = {"yolo11n"}
RTDETR_MODELS = {"rtdetr-l"}
FRCNN_MODELS = {"frcnn"}


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


@lru_cache
def protocol() -> dict:
    return load_yaml(CONFIGS / "protocol.yaml")


@lru_cache
def datasets_cfg() -> dict:
    return load_yaml(CONFIGS / "datasets.yaml")


def dataset_cfg(name: str) -> dict:
    ds = datasets_cfg()["datasets"]
    if name not in ds:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(ds)}")
    return {"id": name, **ds[name]}


def model_family(model: str) -> str:
    if model in YOLO_MODELS:
        return "yolo"
    if model in RTDETR_MODELS:
        return "rtdetr"
    if model in FRCNN_MODELS:
        return "frcnn"
    raise KeyError(f"Unknown model '{model}'")


@lru_cache
def _variants_raw() -> dict:
    return load_yaml(CONFIGS / "variants.yaml")["variants"]


def _resolve(name: str, stack: tuple = ()) -> dict:
    raw = _variants_raw()
    if name not in raw:
        raise KeyError(f"Unknown variant '{name}'. Known: {list(raw)}")
    if name in stack:
        raise ValueError(f"Circular variant inheritance: {stack + (name,)}")
    entry = dict(raw[name])
    parents = entry.pop("inherit", [])
    if not parents:
        return entry
    base = _resolve("base", stack + (name,)) if name != "base" else {}
    out = copy.deepcopy(base)
    # Each parent contributes only what it changes w.r.t. base, so [p2, simam] keeps p2's model_cfg
    # even though simam (via base) would otherwise reset it.
    for p in parents:
        resolved = _resolve(p, stack + (name,))
        out.update({k: v for k, v in resolved.items() if base.get(k) != v})
    out.update(entry)
    return out


def variant_cfg(name: str) -> dict:
    """Fully resolved variant dict (inheritance applied), with ``name`` added."""
    cfg = _resolve(name)
    cfg["name"] = name
    mc = cfg.get("model_cfg", "")
    if mc and (ROOT / mc).exists():
        cfg["model_cfg"] = str(ROOT / mc)
    return cfg
