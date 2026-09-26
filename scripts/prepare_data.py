"""Build processed datasets (YOLO + COCO json) from the raw folders, and sanity-check the labels.

    python scripts/prepare_data.py --datasets neu gc10 pcb mt ksdd2
    python scripts/prepare_data.py --merge neu gc10 pcb mt --name merged
"""

import argparse
import json

import _bootstrap  # noqa: F401

from sdd.data.build import build_dataset
from sdd.data.merge import merge_datasets
from sdd.data.tiling import tile_dataset
from sdd.data.visualize import visualize_split
from sdd.env import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--merge", nargs="*", help="source datasets to merge (phase 3)")
    ap.add_argument("--name", default="merged", help="name of the merged dataset")
    ap.add_argument("--tile", nargs="*", default=[], help="datasets to cut into tiles (-> <ds>_t<size>)")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--force", action="store_true", help="rebuild processed data (splits are kept)")
    ap.add_argument("--n-vis", type=int, default=20, help="images with drawn GT per dataset (0 = none)")
    a = ap.parse_args()
    built = []
    for ds in a.datasets:
        built.append((ds, build_dataset(ds, force=a.force)))
    for ds in a.tile:
        built.append((f"{ds}_t{a.tile_size}", tile_dataset(ds, a.tile_size, force=a.force)))
    if a.merge:
        built.append((a.name, merge_datasets(a.merge, a.name, force=a.force)))
    for ds, d in built:
        meta = json.loads((d / "meta.json").read_text())
        print(f"{ds}: {meta['split_sizes']}  boxes={meta.get('boxes_per_split') or meta.get('tiling_stats')}"
              f"  report={meta.get('read_report')}")
        if a.n_vis:
            out = paths().results / "figures" / "labels" / ds
            print(f"  GT visualisations -> {out} ({len(visualize_split(d, out, 'train', a.n_vis))} images)")


if __name__ == "__main__":
    main()
