"""One experiment run, end to end (plan §3.3)::

    train (resume if interrupted) -> best-on-val checkpoint -> predict test once -> metrics -> metrics.json

Run directory ``<results>/runs/<run_name>/``:
    config.yaml             spec + resolved variant + protocol values actually used + versions
    train/                  framework output (Ultralytics run dir / Faster R-CNN results.csv + weights)
    test_predictions.json   COCO detections on the test split (original pixels)
    metrics.json            flat dict -> one row of master_results.csv
    done.json               written last; its presence means "skip this run"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from ..config import datasets_cfg, model_family, protocol, save_yaml, variant_cfg
from ..data.build import build_dataset, load_coco_json
from ..data.subsample import fraction_data_yaml, fraction_tag
from ..data.tiling import parse_tiled, tile_dataset
from ..env import check_ultralytics_version, paths, versions, weight_path

log = logging.getLogger(__name__)


@dataclass
class RunSpec:
    phase: str
    dataset: str
    model: str
    variant: str = "base"
    seed: int = 0
    epochs: int | None = None
    batch: int | None = None
    # phase 3
    init: str | None = None          # label, e.g. T1 / T2 / T3
    init_from: str | None = None     # "coco" or the run name whose best.pt initialises this run
    fraction: float = 1.0            # of the train split
    finetune: str = "full"           # full | freeze
    freeze_layers: int = 11
    freeze_epochs: int = 10
    # large images (plan §5.1)
    infer: str = "full"              # full | sliced (SAHI-style tiles + full image, merged by NMS)
    reuse_phase: str | None = None   # evaluate the checkpoint of the same run from this phase (no training)
    smoke: bool = False
    extra: dict = field(default_factory=dict)  # extra Ultralytics train kwargs (recorded)

    @property
    def name(self) -> str:
        n = f"{self.phase}_{self.dataset}_{self.model}_{self.variant}"
        if self.init:
            n += f"_{self.init}_{fraction_tag(self.fraction)}_{self.finetune}"
        if self.infer == "sliced":
            n += "_sahi"
        n += f"_s{self.seed}"
        return f"smoke_{n}" if self.smoke else n

    def reuse_source(self) -> str:
        """Run whose checkpoint is re-evaluated when ``reuse_phase`` is set."""
        return replace(self, phase=self.reuse_phase, infer="full", reuse_phase=None).name


def _devices(device: str) -> tuple[str, str]:
    """User device ('0', 'cpu') -> (ultralytics device, torch device)."""
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu", "cpu"
    return str(device), f"cuda:{device}"


def _ensure_data(spec: RunSpec) -> Path:
    d = paths().data_dir(spec.dataset)
    if spec.dataset in datasets_cfg()["datasets"]:
        return build_dataset(spec.dataset)
    tiled = parse_tiled(spec.dataset)
    if tiled and tiled[0] in datasets_cfg()["datasets"]:
        return tile_dataset(*tiled)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d} not built (merged datasets: scripts/prepare_data.py --merge ...)")
    return d


def _train_params(spec: RunSpec) -> dict:
    fam = dict(protocol()["families"][model_family(spec.model)])
    if spec.epochs:
        fam["epochs"] = spec.epochs
    if spec.batch:
        fam["batch"] = spec.batch
    if spec.smoke:
        fam["epochs"] = protocol()["smoke"]["epochs"]
    return fam


def _init_weights(spec: RunSpec, variant: dict) -> str:
    if spec.init_from in (None, "coco"):
        name = variant["weights"] if model_family(spec.model) == "yolo" else f"{spec.model}.pt"
        return weight_path(name)
    src = paths().run_dir(spec.init_from) / "train" / "weights" / "best.pt"
    if not src.exists():
        raise FileNotFoundError(f"init checkpoint {src} missing - run the source grid first")
    return str(src)


def _ckpt_state(p: Path) -> str:
    """'missing' | 'resumable' | 'finished' for an Ultralytics last.pt."""
    if not p.exists():
        return "missing"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return "resumable" if ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0 else "finished"


def _train_ultralytics(spec: RunSpec, rd: Path, data_yaml: Path, fam: dict, device: str) -> Path:
    import sdd.models.yolo_trainer as yt  # noqa: F401
    from ultralytics import RTDETR, YOLO

    family = model_family(spec.model)
    variant = variant_cfg(spec.variant) if family == "yolo" else {"name": "base"}
    common = dict(data=str(data_yaml), imgsz=protocol()["imgsz"], batch=fam["batch"], seed=spec.seed,
                  deterministic=True, patience=fam.get("patience", 0), workers=protocol()["workers"],
                  amp=protocol()["amp"], device=device, project=str(rd), exist_ok=True, plots=True, verbose=False)
    if spec.smoke:
        common["fraction"] = protocol()["smoke"]["fraction"]
    common.update(spec.extra)
    trainer = yt.make_trainer(variant) if family == "yolo" else None
    cls = RTDETR if family == "rtdetr" else YOLO

    def stage(name: str, epochs: int, init: str, freeze: int | None) -> Path:
        wdir = rd / name / "weights"
        state = _ckpt_state(wdir / "last.pt")
        if state == "finished" and (wdir / "best.pt").exists():
            return wdir / "best.pt"
        kw = dict(common, name=name, epochs=epochs)
        if freeze:
            kw["freeze"] = freeze
        if state == "resumable":
            log.info("resuming %s/%s", spec.name, name)
            cls(str(wdir / "last.pt")).train(resume=True, trainer=trainer, device=device)
        elif family == "yolo":
            YOLO(variant["model_cfg"]).train(trainer=trainer, pretrained=init, **kw)
        else:
            cls(init).train(**kw)
        return wdir / "best.pt"

    init = _init_weights(spec, variant)
    total = fam["epochs"]
    if spec.finetune == "freeze" and not spec.smoke:
        k = min(spec.freeze_epochs, total - 1)
        init = str(stage("train_frozen", k, init, spec.freeze_layers))
        total -= k
    return stage("train", total, init, None)


def train_hours(rd: Path) -> float:
    """Total training time from the framework logs (survives resumes across sessions)."""
    import pandas as pd

    h = 0.0
    for stage in ("train_frozen", "train"):
        f = rd / stage / "results.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            h += df["time"].max() / 3600 if "time" in df else df.get("time_s", pd.Series([0])).sum() / 3600
    return round(float(h), 3)


def evaluate_run(spec: RunSpec, weights: Path, data_dir: Path, rd: Path, device: str) -> dict:
    from ..evaluation.coco_eval import coco_metrics
    from ..evaluation.efficiency import benchmark_module
    from ..evaluation.image_level import image_level_metrics
    from ..evaluation.matching import dets_by_image, gt_from_coco, operating_point_metrics
    from ..evaluation.tide_eval import tide_errors
    from .predict import predict_frcnn_split, predict_ultralytics, predict_ultralytics_sliced

    ev, bench = protocol()["eval"], protocol()["benchmark"]
    family = model_family(spec.model)
    udev, tdev = _devices(device)
    gt_path = data_dir / "coco" / "test.json"
    gt = load_coco_json(gt_path)
    if spec.infer == "sliced":
        if family == "frcnn":
            raise NotImplementedError("sliced inference is implemented for the Ultralytics models only")
        sh = protocol()["sahi"]
        dets, speed = predict_ultralytics_sliced(
            weights, family, data_dir, "test", imgsz=protocol()["imgsz"], conf=ev["pred_conf"], iou=ev["pred_iou"],
            max_det=ev["max_det"], device=udev, half=bench["half"] and udev != "cpu", tile=sh["tile"],
            overlap=sh["overlap"], full_image=sh["full_image"], merge_iou=sh["merge_iou"], batch=sh["batch"])
    elif family == "frcnn":
        dets, speed = predict_frcnn_split(weights, data_dir, "test", conf=ev["pred_conf"], device=tdev)
    else:
        dets, speed = predict_ultralytics(weights, family, data_dir, "test", imgsz=protocol()["imgsz"],
                                          conf=ev["pred_conf"], iou=ev["pred_iou"], max_det=ev["max_det"],
                                          device=udev, half=bench["half"] and udev != "cpu")
    (rd / "test_predictions.json").write_text(json.dumps(dets))

    m = coco_metrics(gt, dets)
    G, D = gt_from_coco(gt), dets_by_image(dets, [im["id"] for im in gt["images"]])
    m.update(operating_point_metrics(G, D, ev["match_iou"], ev["op_conf"]))
    m.update(image_level_metrics(G, D, ev["match_iou"], ev["op_conf"]))
    if ev.get("tide", True):
        try:
            m.update(tide_errors(gt_path, dets, spec.name))
        except Exception as e:  # TIDE is diagnostic only - never lose a run over it
            log.warning("TIDE failed for %s: %s", spec.name, e)
    m.update({f"e2e_{k}": v for k, v in speed.items()})

    if family == "frcnn":
        from ..models.frcnn import load_frcnn

        module, kind = load_frcnn(weights), "torchvision"
    else:
        from ultralytics import RTDETR, YOLO

        module, kind = (RTDETR if family == "rtdetr" else YOLO)(str(weights)).model, "ultralytics"
    m.update(benchmark_module(module, kind, protocol()["imgsz"], bench["warmup"], bench["iters"], tdev))
    return m


def run(spec: RunSpec, device: str = "0", force: bool = False) -> dict:
    rd = paths().run_dir(spec.name)
    if (rd / "done.json").exists() and not force:
        log.info("skip %s (done)", spec.name)
        return json.loads((rd / "metrics.json").read_text())
    rd.mkdir(parents=True, exist_ok=True)
    family = model_family(spec.model)
    if family != "yolo" and spec.variant != "base":
        raise ValueError(f"{spec.model} only supports variant 'base'")
    if family != "frcnn":
        check_ultralytics_version()
    data_dir = _ensure_data(spec)
    data_yaml = fraction_data_yaml(data_dir, spec.fraction, spec.seed)
    fam = _train_params(spec)
    save_yaml({"spec": asdict(spec), "run": spec.name, "train_params": fam,
               "variant": variant_cfg(spec.variant) if family == "yolo" else None,
               "data_yaml": str(data_yaml), "protocol": protocol(), "versions": versions()}, rd / "config.yaml")

    udev, tdev = _devices(device)
    t0 = time.time()
    if spec.reuse_phase:
        weights = paths().run_dir(spec.reuse_source()) / "train" / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"{spec.name}: source run {spec.reuse_source()} has no best.pt yet")
    elif family == "frcnn":
        from ..models.frcnn import train_frcnn

        smoke_frac = protocol()["smoke"]["fraction"] if spec.smoke else 1.0
        if spec.fraction < 1:
            raise NotImplementedError("data fractions are implemented for the Ultralytics models only")
        weights = train_frcnn(data_dir, rd / "train", epochs=fam["epochs"], batch=fam["batch"], lr=fam["lr"],
                              momentum=fam["momentum"], weight_decay=fam["weight_decay"], lr_steps=fam["lr_steps"],
                              warmup_iters=fam["warmup_iters"], eval_every=fam["eval_every"], seed=spec.seed,
                              workers=protocol()["workers"], amp=protocol()["amp"], device=tdev,
                              imgsz=protocol()["imgsz"], fraction=smoke_frac)
    else:
        weights = _train_ultralytics(spec, rd, data_yaml, fam, udev)
    train_h = (time.time() - t0) / 3600
    if not Path(weights).exists():
        raise FileNotFoundError(f"{spec.name}: no checkpoint at {weights}")

    trained_in = paths().run_dir(spec.reuse_source()) if spec.reuse_phase else rd
    tiled = parse_tiled(spec.dataset)
    metrics = {"run": spec.name, **{k: v for k, v in asdict(spec).items() if k != "extra"},
               "eval_dataset": tiled[0] if tiled else spec.dataset, "train_tile": tiled[1] if tiled else None,
               "epochs_run": fam["epochs"], "batch_run": fam["batch"], "train_hours": train_hours(trained_in),
               "train_hours_this_session": round(train_h, 3),
               "weights": str(weights), **versions()}
    metrics.update(evaluate_run(spec, Path(weights), data_dir, rd, device))
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (rd / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    log.info("%s: AP=%.4f AP_s=%s", spec.name, metrics.get("AP", float("nan")), metrics.get("AP_s"))
    return metrics
