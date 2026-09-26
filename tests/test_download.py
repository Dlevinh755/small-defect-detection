"""Downloader: extraction, streaming, checks, path candidates, manifest (offline - no network)."""

import json
import os
import zipfile
from pathlib import Path

from sdd.data.download import _download, _extract, check_dataset, load_sources, write_manifest
from sdd.env import Paths


def test_extract_flattens_single_top_folder(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Repo-master/MT_Crack/Imgs/x.jpg", b"1")
        f.writestr("__MACOSX/junk", b"")
    _extract(z, tmp_path / "Magnetic-Tile")
    assert (tmp_path / "Magnetic-Tile" / "MT_Crack" / "Imgs" / "x.jpg").exists()


def test_download_streams_file_url(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(3 << 20))
    out = _download(src.resolve().as_uri(), tmp_path / "dl.bin", chunk=1 << 18)
    assert out.read_bytes() == src.read_bytes() and not (tmp_path / "dl.bin.part").exists()


def test_checks_on_synthetic_raw(raw_data):
    src = load_sources()
    neu = check_dataset("neu", Path(os.environ["SDD_RAW_NEU"]), src["neu"])
    assert neu["images"] == 30 and neu["status"] == "warn" and "expected ~1800" in neu["messages"][0]
    ks = check_dataset("ksdd2", Path(os.environ["SDD_RAW_KSDD2"]), src["ksdd2"])
    assert ks["split_images"] == {"train": 12, "test": 6} and ks["clean_images"] == 12
    gc = check_dataset("gc10", Path(os.environ["SDD_RAW_GC10"]), src["gc10"])
    assert gc["unlabeled_images_skipped"] == 0 and gc["classes_present"] == 4
    assert check_dataset("pcb", Path(raw_data) / "nope", src["pcb"])["status"] == "missing"
    empty = Path(raw_data) / "empty_pcb"
    empty.mkdir(exist_ok=True)
    assert check_dataset("pcb", empty, src["pcb"])["status"] == "fail"


def test_raw_path_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("SDD_RAW_NEU", raising=False)
    monkeypatch.delenv("SDD_RAW_GC10", raising=False)
    (tmp_path / "attached").mkdir()
    p = Paths("t", {"neu": [str(tmp_path / "attached"), str(tmp_path / "raw" / "NEU")],
                    "gc10": [str(tmp_path / "x"), str(tmp_path / "raw" / "GC10")]},
              tmp_path / "raw", tmp_path, tmp_path, tmp_path)
    assert p.raw_dir("neu") == tmp_path / "attached"         # first existing candidate (attached input)
    assert p.raw_dir("gc10") == tmp_path / "raw" / "GC10"    # nothing exists -> download target (last)


def test_manifest(tmp_path):
    rows = [{"dataset": "neu", "name": "NEU-DET", "status": "ok", "images": 1800, "boxes": 4189, "defect_images": 1800,
             "clean_images": 0, "source_page": "https://x", "license": "research", "date": "2026-09-27",
             "messages": []},
            {"dataset": "mt", "name": "Magnetic Tile", "status": "warn", "source_page": "https://y", "license": "r",
             "date": "2026-09-27", "messages": ["5 MT_* folders, expected 6"]}]
    write_manifest(rows, tmp_path)
    md = (tmp_path / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| neu | NEU-DET | ok | 1800 |" in md and "expected 6" in md
    assert json.loads((tmp_path / "MANIFEST.json").read_text())[0]["images"] == 1800


def test_raw_path_glob_finds_any_mount_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("SDD_RAW_NEU", raising=False)
    deep = tmp_path / "input" / "datasets" / "someone" / "sdd-raw" / "NEU-DET"
    (deep / "NEU-DET").mkdir(parents=True)                  # a nested folder with the same name
    p = Paths("t", {"neu": [str(tmp_path / "input" / "**" / "NEU-DET"), str(tmp_path / "raw" / "NEU-DET")]},
              tmp_path / "raw", tmp_path, tmp_path, tmp_path)
    assert p.raw_dir("neu") == deep                           # shallowest match wins


def test_materialize_replaces_symlink(tmp_path, monkeypatch):
    import sdd.data.download as dl

    real = tmp_path / "cache" / "mirror"
    (real / "x").mkdir(parents=True)
    (real / "x" / "a.txt").write_text("1")
    root = tmp_path / "raw"
    root.mkdir()
    link = root / "NEU-DET"
    try:
        os.symlink(real, link, target_is_directory=True)
    except OSError:
        import pytest
        pytest.skip("no symlink permission on this machine")
    fake = Paths("t", {"neu": [str(link)]}, root, tmp_path, tmp_path, tmp_path)
    monkeypatch.setattr(dl, "paths", lambda: fake)
    msg = dl.materialize(["neu"])
    assert not link.is_symlink() and (link / "x" / "a.txt").read_text() == "1" and "replaced" in msg[0]
