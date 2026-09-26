"""Download and collect every raw dataset into one folder, then check them (sources: configs/sources.yaml).

    python scripts/download_data.py                      # all 5 datasets -> data/raw (local) or /kaggle/working/raw
    python scripts/download_data.py --only neu gc10      # a subset
    python scripts/download_data.py --check              # no download: only check what is there + manifest
    python scripts/download_data.py --bundle <user>/sdd-raw --upload
        # ONE-TIME: download everything as real files (no symlinks), write dataset-metadata.json and create the
        # PRIVATE Kaggle dataset <user>/sdd-raw (a new version if it exists). Later sessions: just "Add Data" ->
        # sdd-raw; configs/paths.yaml finds it under any /kaggle/input layout and nothing is downloaded again.
        # --upload needs Kaggle API credentials (KAGGLE_USERNAME / KAGGLE_KEY, e.g. from Kaggle Secrets). Without
        # them: Save Version the notebook, then Output -> "New Dataset" (see README "Data download").

Datasets already present (attached input or earlier download) and passing the checks are skipped.
Kaggle sources need no API key inside a Kaggle notebook; locally put kaggle.json in ~/.kaggle
(or set KAGGLE_USERNAME / KAGGLE_KEY). Total download ~4 GB.
"""

import argparse

import _bootstrap  # noqa: F401

from sdd.data.download import (check_dataset, collect, kaggle_bundle, kaggle_upload, load_sources, materialize,
                               write_manifest)
from sdd.env import paths

ALL = ["neu", "gc10", "pcb", "mt", "ksdd2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--check", action="store_true", help="do not download, only check + write the manifest")
    ap.add_argument("--force", action="store_true", help="download again even if a valid copy exists")
    ap.add_argument("--copy", action="store_true", help="copy Kaggle folders instead of symlinking them")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--bundle", metavar="USER/NAME", help="real files + dataset-metadata.json for a private Kaggle dataset")
    ap.add_argument("--upload", action="store_true", help="with --bundle: create / version the Kaggle dataset")
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
        rows = collect(a.only, force=a.force, copy=a.copy or bool(a.bundle), keep_zip=a.keep_zip)

    print(f"\n{'dataset':8s} {'status':8s} {'images':>7s} {'boxes':>7s}  path")
    for r in rows:
        print(f"{r['dataset']:8s} {r['status']:8s} {r.get('images', '-')!s:>7s} {r.get('boxes', '-')!s:>7s}  "
              f"{r.get('path', '')}")
        for m in r.get("messages", []):
            print(f"{'':17s}! {m}")
    print(f"\nmanifest -> {root / 'MANIFEST.md'} (copy the table into data/README.md)")

    failed = [r["dataset"] for r in rows if r["status"] in ("fail", "missing")]
    if a.bundle:
        if failed:
            raise SystemExit(f"not bundling: {failed} failed - fix them first (see messages above)")
        for line in materialize(a.only):
            print("  " + line)
        print(f"Kaggle metadata -> {kaggle_bundle(root, a.bundle)}")
        if a.upload:
            print(kaggle_upload(root, a.bundle))
        else:
            print(f"  upload: kaggle datasets create -p {root} --dir-mode zip   (or rerun with --upload)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
