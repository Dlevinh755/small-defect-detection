"""Download and collect every raw dataset into one folder, then check them (sources: configs/sources.yaml).

    python scripts/download_data.py                      # all 5 datasets -> data/raw (local) or /kaggle/working/raw
    python scripts/download_data.py --only neu gc10      # a subset
    python scripts/download_data.py --check              # no download: only check what is there + manifest
    python scripts/download_data.py --bundle <user>/sdd-raw --copy
        # materialise real files (no symlinks) and write dataset-metadata.json, then upload once with
        #   kaggle datasets create -p <raw_root> --dir-mode zip
        # and attach it as /kaggle/input/sdd-raw in later sessions (first candidate in configs/paths.yaml)

Datasets already present (attached Kaggle input or earlier download) and passing the checks are skipped.
Kaggle sources need no API key inside a Kaggle notebook; locally put kaggle.json in ~/.kaggle
(or set KAGGLE_USERNAME / KAGGLE_KEY). Total download ~4 GB.
"""

import argparse

import _bootstrap  # noqa: F401

from sdd.data.download import check_dataset, collect, kaggle_bundle, load_sources, write_manifest
from sdd.env import paths

ALL = ["neu", "gc10", "pcb", "mt", "ksdd2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--check", action="store_true", help="do not download, only check + write the manifest")
    ap.add_argument("--force", action="store_true", help="download again even if a valid copy exists")
    ap.add_argument("--copy", action="store_true", help="copy Kaggle folders instead of symlinking them")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--bundle", metavar="USER/NAME", help="write dataset-metadata.json for a private Kaggle dataset")
    a = ap.parse_args()

    root = paths().raw_root
    print(f"raw root: {root}")
    if a.check:
        src = load_sources()
        rows = [{"dataset": d, "name": src[d]["name"], "license": src[d]["license"],
                 "source_page": src[d]["source_page"], "date": "-",
                 **check_dataset(d, paths().raw_dir(d), src[d])} for d in a.only]
        write_manifest(rows, root)
    else:
        rows = collect(a.only, force=a.force, copy=a.copy, keep_zip=a.keep_zip)

    print(f"\n{'dataset':8s} {'status':8s} {'images':>7s} {'boxes':>7s}  path")
    for r in rows:
        print(f"{r['dataset']:8s} {r['status']:8s} {r.get('images', '-')!s:>7s} {r.get('boxes', '-')!s:>7s}  {r['path'] if 'path' in r else ''}")
        for m in r.get("messages", []):
            print(f"{'':17s}! {m}")
    print(f"\nmanifest -> {root / 'MANIFEST.md'} (copy the table into data/README.md)")
    if a.bundle:
        print(f"Kaggle metadata -> {kaggle_bundle(root, a.bundle)}\n  upload: kaggle datasets create -p {root} --dir-mode zip")
    if any(r["status"] in ("fail", "missing") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
