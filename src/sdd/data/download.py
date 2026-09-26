"""Download, collect and check the raw datasets (plan §2.2). Sources: ``configs/sources.yaml``.

Every dataset ends up at ``<raw_root>/<folder>`` (``raw_root`` from ``configs/paths.yaml``). Kaggle datasets come
through ``kagglehub`` (inside a Kaggle notebook no API key is needed; the folder is linked, not copied), direct
URLs are streamed with resume support and unzipped. Each dataset is then checked with the same readers the
pipeline uses, plus the checks of the plan's source table (XML present, original resolution, boards, masks...),
and everything is written to ``<raw_root>/MANIFEST.json`` / ``MANIFEST.md`` (source, date, license, counts).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

from ..config import CONFIGS, dataset_cfg, load_yaml
from ..env import paths
from .readers import IMG_EXTS, read_dataset

log = logging.getLogger(__name__)


def load_sources() -> dict:
    return load_yaml(CONFIGS / "sources.yaml")


# ------------------------------------------------------------------------------------------ fetching
def _link_or_copy(src: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    if not copy:
        try:
            os.symlink(src.resolve(), dest, target_is_directory=True)
            return
        except OSError:
            pass  # e.g. Windows without symlink rights
    shutil.copytree(src, dest, symlinks=False)


def fetch_kaggle(slug: str, dest: Path, copy: bool = False) -> str:
    try:
        import kagglehub

        src = Path(kagglehub.dataset_download(slug))
        _link_or_copy(src, dest, copy)
        return f"kagglehub:{slug} -> {src}"
    except ImportError:
        pass
    dest.mkdir(parents=True, exist_ok=True)  # fallback: official CLI (needs ~/.kaggle/kaggle.json)
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
                   check=True)
    return f"kaggle-cli:{slug}"


def _download(url: str, out: Path, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``out`` with a progress line; resumes a partial ``.part`` file when the server allows."""
    part = out.with_suffix(out.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "sdd-downloader", **({"Range": f"bytes={done}-"} if done else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if done and r.status != 206:  # server ignored Range -> start over
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(part, "ab" if done else "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                done += len(b)
                size = f" / {total / 2**20:,.0f}" if total > done - len(b) else ""  # size unknown: GitHub zips
                print(f"\r  {out.name}: {done / 2**20:,.0f}{size} MB", end="", flush=True)
    print()
    part.rename(out)
    return out


def _extract(zip_path: Path, dest: Path) -> None:
    """Unzip into ``dest``; a single top-level folder in the archive is flattened away."""
    tmp = dest.with_name(dest.name + "._unzip")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    entries = [p for p in tmp.iterdir() if p.name != "__MACOSX"]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(root), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def fetch_url(url: str, dest: Path, keep_zip: bool = False) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / f"{dest.name}.zip"
    if not zip_path.exists():
        _download(url, zip_path)
    _extract(zip_path, dest)
    if not keep_zip:
        zip_path.unlink()
    return f"url:{url}"


# ------------------------------------------------------------------------------------------ checks
def _close(n: int, expected: int, tol: float = 0.05) -> bool:
    return abs(n - expected) <= tol * expected


def check_dataset(ds: str, root: Path, src: dict) -> dict:
    """Read the raw folder with the pipeline's reader and apply the source-table checks."""
    exp = src.get("expect", {})
    res = {"dataset": ds, "path": str(root), "status": "ok", "messages": []}

    def warn(msg):
        res["messages"].append(msg)
        res["status"] = "warn" if res["status"] == "ok" else res["status"]

    if not root.exists():
        return {**res, "status": "missing", "messages": [f"{root} does not exist"]}
    try:
        samples, rep = read_dataset(root, dataset_cfg(ds))
    except Exception as e:  # unknown class names, wrong layout, ...
        return {**res, "status": "fail", "messages": [f"reader failed: {e}"]}
    n_img = len(samples)
    cls = Counter(c for s in samples for c in s.classes.tolist())
    res.update(images=n_img, boxes=rep.n_boxes, classes_present=len(cls),
               defect_images=sum(len(s.boxes) > 0 for s in samples),
               clean_images=sum(len(s.boxes) == 0 for s in samples),
               dropped_small_components=rep.dropped_small, annotations_without_image=rep.missing_images)
    if n_img == 0:
        return {**res, "status": "fail", "messages": ["no annotated images found - wrong mirror (classification-only?)"]}
    if "images" in exp and not _close(n_img, exp["images"]):
        warn(f"{n_img} images, expected ~{exp['images']}")
    if "classes" in exp and len(cls) != exp["classes"]:
        warn(f"{len(cls)} classes with boxes, expected {exp['classes']}")
    all_imgs = sum(1 for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.stem.endswith("_GT"))
    if ds == "gc10":
        res["unlabeled_images_skipped"] = max(0, all_imgs - n_img)
    if "min_side" in exp:
        small = sum(min(s.width, s.height) < exp["min_side"] for s in samples)
        res["images_below_min_side"] = small
        if small:
            warn(f"{small} images smaller than {exp['min_side']} px - resized/augmented mirror? use the original")
    if src.get("board_regex"):
        rx = re.compile(src["board_regex"])
        boards = Counter(m.group(1) for s in samples if (m := rx.search(s.uid)))
        res["boards"] = dict(sorted(boards.items()))
        if not boards:
            warn("board id not found in file names - leakage check by board impossible")
    if "mt_folders" in exp:
        n = sum(1 for d in root.rglob("MT_*") if d.is_dir())
        res["mt_folders"] = n
        if n != exp["mt_folders"]:
            warn(f"{n} MT_* folders, expected {exp['mt_folders']}")
    if ds == "ksdd2":
        res["split_images"] = dict(Counter(s.meta.get("official_split") for s in samples))
    if rep.notes:
        warn(f"{len(rep.notes)} reader notes, e.g. {rep.notes[0]}")
    return res


# ------------------------------------------------------------------------------------------ orchestration
def collect(datasets: list[str], force: bool = False, copy: bool = False, keep_zip: bool = False) -> list[dict]:
    sources, root = load_sources(), paths().raw_root
    out = []
    for ds in datasets:
        src = sources[ds]
        existing = paths().raw_dir(ds)
        target = root / src["folder"]
        entry = {"dataset": ds, "name": src["name"], "license": src["license"], "source_page": src["source_page"],
                 "fallback": src.get("fallback", ""), "date": str(date.today())}
        if not force and existing.exists() and check_dataset(ds, existing, src)["status"] in ("ok", "warn"):
            entry["fetched"] = f"already present at {existing}"
        else:
            print(f"[{ds}] downloading {src['name']} ...")
            try:
                entry["fetched"] = (fetch_kaggle(src["slug"], target, copy) if src["kind"] == "kaggle"
                                    else fetch_url(src["url"], target, keep_zip))
            except Exception as e:
                entry.update(fetched="FAILED", error=repr(e))
                out.append({**entry, "status": "fail", "messages": [f"download failed: {e}. Fallback: {src.get('fallback')}"]})
                continue
            existing = paths().raw_dir(ds)
        out.append({**entry, **check_dataset(ds, existing, src)})
    write_manifest(out, root)
    return out


def write_manifest(rows: list[dict], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "MANIFEST.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = ["| id | dataset | status | images | boxes | defect / clean | source | license | date |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['name']} | {r.get('status')} | {r.get('images', '-')} | "
                     f"{r.get('boxes', '-')} | {r.get('defect_images', '-')} / {r.get('clean_images', '-')} | "
                     f"{r['source_page']} | {r['license']} | {r['date']} |")
    notes = [f"- **{r['dataset']}**: " + "; ".join(r["messages"]) for r in rows if r.get("messages")]
    (root / "MANIFEST.md").write_text("\n".join(lines + ([""] + notes if notes else [])) + "\n", encoding="utf-8")


def kaggle_bundle(root: Path, dataset_id: str) -> Path:
    """dataset-metadata.json so the collected raw folder can be uploaded once as a private Kaggle dataset
    (``kaggle datasets create -p <root> --dir-mode zip``) and attached in later sessions as /kaggle/input/sdd-raw."""
    meta = {"title": "SDD raw datasets", "id": dataset_id, "licenses": [{"name": "other"}]}
    p = root / "dataset-metadata.json"
    p.write_text(json.dumps(meta, indent=2))
    return p
