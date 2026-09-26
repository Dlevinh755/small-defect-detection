"""Merge built source datasets into one (phase 3 source model, plan §6.2).

Classes are kept apart by prefixing them with the dataset id (``neu/crazing``, ``gc10/crease``...), so no
semantic merging decision is baked in. Splits are the union of each dataset's fixed splits.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from ..env import paths
from .build import build_dataset, class_names, samples_from_coco, write_data_yaml, write_split
from .readers import Sample
from .splits import SPLIT_NAMES


def merge_datasets(sources: list[str], name: str = "merged", force: bool = False) -> Path:
    out = paths().data_dir(name)
    if (out / "meta.json").exists() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    names, offsets = [], {}
    for ds in sources:
        build_dataset(ds)
        offsets[ds] = len(names)
        names += [f"{ds}/{c}" for c in class_names(paths().data_dir(ds))]
    sizes = {}
    for split in SPLIT_NAMES:
        merged = []
        for ds in sources:
            for s in samples_from_coco(paths().data_dir(ds), split):
                merged.append(
                    Sample(f"{ds}__{s.uid}", s.img_path, s.width, s.height, s.boxes, s.classes + offsets[ds], {"source": ds})
                )
        write_split(out, split, merged, names)
        sizes[split] = len(merged)
    write_data_yaml(out, names)
    (out / "meta.json").write_text(
        json.dumps({"dataset": name, "sources": sources, "classes": names, "built": str(date.today()),
                    "split_sizes": sizes}, indent=2)
    )
    return out
