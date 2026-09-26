"""Target-data fractions for phase 3 (10/25/50/100% of train, stratified, fixed seed - plan §6.2).

A fraction does not copy data: it writes ``train_f<pct>_s<seed>.txt`` (list of image paths) next to the
dataset and a matching ``data_f<pct>_s<seed>.yaml`` whose ``train`` entry points at that list.
Val/test are always the full splits.
"""

from __future__ import annotations

from pathlib import Path

from .build import class_names, samples_from_coco, write_data_yaml
from .splits import stratified_split


def fraction_tag(fraction: float) -> str:
    return f"f{round(fraction * 100):03d}"


def subsample_train(data_dir: Path, fraction: float, seed: int) -> list[Path]:
    samples = samples_from_coco(data_dir, "train")
    if fraction >= 1:
        return [s.img_path for s in samples]
    keys = {s.uid: s.strat_key for s in samples}
    keep = set(stratified_split(keys, {"keep": fraction, "drop": 1 - fraction}, seed)["keep"])
    return [s.img_path for s in samples if s.uid in keep]


def fraction_data_yaml(data_dir: Path, fraction: float, seed: int) -> Path:
    if fraction >= 1:
        return data_dir / "data.yaml"
    tag = f"{fraction_tag(fraction)}_s{seed}"
    lst = data_dir / f"train_{tag}.txt"
    if not lst.exists():
        lst.write_text("\n".join(str(p) for p in subsample_train(data_dir, fraction, seed)) + "\n")
    return write_data_yaml(data_dir, class_names(data_dir), train=lst.name, filename=f"data_{tag}.yaml")
