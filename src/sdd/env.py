"""Runtime environment: Kaggle/local profile, resolved directories, library versions."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import CONFIGS, ROOT, load_yaml

PINNED_ULTRALYTICS = "8.4.163"


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


@dataclass(frozen=True)
class Paths:
    profile: str
    raw: dict
    raw_root: Path
    work: Path
    results: Path
    weights: Path

    def raw_dir(self, dataset: str) -> Path:
        env = os.environ.get(f"SDD_RAW_{dataset.upper()}")
        if env:
            return Path(env)
        if dataset not in self.raw:
            raise KeyError(f"No raw path for '{dataset}' in configs/paths.yaml ({self.profile} profile)")
        cands = self.raw[dataset] if isinstance(self.raw[dataset], list) else [self.raw[dataset]]
        cands = [_abs(c) for c in cands]
        return next((c for c in cands if c.exists()), cands[-1])  # last = download target

    def data_dir(self, dataset: str) -> Path:
        """Processed dataset (YOLO layout + COCO json)."""
        return self.work / "data" / dataset

    def run_dir(self, run_name: str) -> Path:
        return self.results / "runs" / run_name


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@lru_cache
def paths() -> Paths:
    cfg = load_yaml(CONFIGS / "paths.yaml")
    profile = os.environ.get("SDD_PROFILE") or ("kaggle" if is_kaggle() else "local")
    c = cfg[profile]
    p = Paths(
        profile=profile,
        raw=c["raw"],
        raw_root=_abs(os.environ.get("SDD_RAW_ROOT", c.get("raw_root", "data/raw"))),
        work=_abs(os.environ.get("SDD_WORK", c["work"])),
        results=_abs(os.environ.get("SDD_RESULTS", c["results"])),
        weights=_abs(c["weights"]),
    )
    for d in (p.work, p.results, p.weights):
        d.mkdir(parents=True, exist_ok=True)
    return p


def versions() -> dict:
    """Library / hardware versions recorded with every run (plan §3.3)."""
    import torch
    import torchvision

    v = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import ultralytics

        v["ultralytics"] = ultralytics.__version__
    except ImportError:
        v["ultralytics"] = None
    return {k: None if x is None else str(x) for k, x in v.items()}  # torch versions are str subclasses


def check_ultralytics_version() -> None:
    import warnings

    import ultralytics

    if ultralytics.__version__ != PINNED_ULTRALYTICS:
        warnings.warn(
            f"ultralytics {ultralytics.__version__} installed, code tested with {PINNED_ULTRALYTICS}. "
            "Custom modules/losses patch Ultralytics internals - run `pytest tests/test_models.py` first.",
            stacklevel=2,
        )


def weight_path(name: str) -> str:
    """Return a local path for a pretrained checkpoint (Ultralytics downloads it there if missing)."""
    local = paths().weights / name
    if local.exists():
        return str(local)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset(name))
    if downloaded.resolve() != local.resolve():
        shutil.move(str(downloaded), local)
    return str(local)


def sync_results(src: str | Path, dst: str | Path | None = None) -> int:
    """Copy runs from a previous Kaggle session (attached as an input) into the working results dir.

    Existing runs are never overwritten. Returns the number of runs copied.
    """
    src, dst = Path(src), Path(dst) if dst else paths().results
    runs_src = src / "runs" if (src / "runs").exists() else src
    n = 0
    for run in sorted(p for p in runs_src.iterdir() if p.is_dir()):
        target = dst / "runs" / run.name
        if target.exists():
            continue
        shutil.copytree(run, target)
        n += 1
    return n
