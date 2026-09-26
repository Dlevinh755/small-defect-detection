"""Run a single experiment (train -> test eval -> metrics.json).

    python scripts/train.py --phase p1 --dataset neu --model yolo11n --variant p2 --seed 0
    python scripts/train.py --phase p1 --dataset neu --model frcnn --smoke
"""

import argparse
import json

import _bootstrap  # noqa: F401

from sdd.engine.run import RunSpec, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="dev")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="yolo11n", help="yolo11n | rtdetr-l | frcnn")
    ap.add_argument("--variant", default="base", help="see configs/variants.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--init", help="phase-3 label, e.g. T2")
    ap.add_argument("--init-from", default="coco", help="'coco' or a run name")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--finetune", default="full", choices=["full", "freeze"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of train: pipeline check")
    ap.add_argument("--force", action="store_true", help="re-run even if done.json exists")
    a = ap.parse_args()
    spec = RunSpec(phase=a.phase, dataset=a.dataset, model=a.model, variant=a.variant, seed=a.seed, epochs=a.epochs,
                   batch=a.batch, init=a.init, init_from=a.init_from, fraction=a.fraction, finetune=a.finetune,
                   smoke=a.smoke)
    m = run(spec, device=a.device, force=a.force)
    keys = ["run", "AP", "AP50", "AP75", "AP_s", "AP_rel_small", "R_rel_small", "params_M", "GFLOPs", "fps_fp16"]
    print(json.dumps({k: m.get(k) for k in keys}, indent=2, default=str))


if __name__ == "__main__":
    main()
