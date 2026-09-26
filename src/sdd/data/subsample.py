"""Train subsets for phase 3 (plan §6.2): data fractions and the defect-free image ratio.

* fraction: 10/25/50/100% of train, stratified by rarest class / defect-free (so small fractions keep defects)
* bg_ratio: defect-free : defective images kept in train (1 -> 1:1, 3 -> 1:3 ...); None keeps all of them.
  Val/test always keep the original ratio (false alarms are measured there).

A subset does not copy data: it writes ``train_<tag>.txt`` (list of image paths) next to the dataset and a matching
``data_<tag>.yaml`` whose ``train`` entry points at that list.
"""

from __future__ import annotations

import random
from pathlib import Path

from .build import class_names, samples_from_coco, write_data_yaml
from .splits import rarest_class_keys, stratified_split


def fraction_tag(fraction: float) -> str:
    return f"f{round(fraction * 100):03d}"


def subsample_train(data_dir: Path, fraction: float, seed: int, bg_ratio: float | None = None) -> list[Path]:
    samples = samples_from_coco(data_dir, "train")
    if fraction < 1:
        keep = set(stratified_split(rarest_class_keys(samples), {"keep": fraction, "drop": 1 - fraction}, seed)["keep"])
        samples = [s for s in samples if s.uid in keep]
    if bg_ratio is not None:
        defect = [s for s in samples if len(s.boxes)]
        clean = [s for s in samples if not len(s.boxes)]
        n = min(len(clean), round(bg_ratio * len(defect)))
        keep = {s.uid for s in random.Random(seed).sample(clean, n)} | {s.uid for s in defect}
        samples = [s for s in samples if s.uid in keep]
    return [s.img_path for s in samples]


def fraction_data_yaml(data_dir: Path, fraction: float, seed: int, bg_ratio: float | None = None) -> Path:
    if fraction >= 1 and bg_ratio is None:
        return data_dir / "data.yaml"
    tag = f"{fraction_tag(fraction)}" + (f"_bg{bg_ratio:g}" if bg_ratio is not None else "") + f"_s{seed}"
    lst = data_dir / f"train_{tag}.txt"
    if not lst.exists():
        lst.write_text("\n".join(str(p) for p in subsample_train(data_dir, fraction, seed, bg_ratio)) + "\n")
    return write_data_yaml(data_dir, class_names(data_dir), train=lst.name, filename=f"data_{tag}.yaml")
