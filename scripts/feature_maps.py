"""Backbone feature maps of finished YOLO runs, for the slides.

    # same test image through three runs, backbone C3k2 layers 2/4/6/8 (strides 4/8/16/32)
    python scripts/feature_maps.py --runs p2s_neu_yolo11n_base_s0 p2s_neu_yolo11n_simam_s0 p2s_neu_yolo11n_spd_s0 \
        --dataset neu --n-images 3
    # also the per-channel grid of one layer (as in the DETR notebook)
    python scripts/feature_maps.py --runs p1_pcb_yolo11n_base_s0 --dataset pcb --channels-layer 4

Images: by default the test images containing the smallest GT boxes (that is what the figure is about);
``--uids`` picks specific ones. Output: <results>/figures/feature_maps/<dataset>/.
"""

import argparse

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch

from sdd.data.build import samples_from_coco
from sdd.env import paths
from sdd.reporting.feature_maps import capture, channel_grid, compare_runs, letterbox, load_run_model
from sdd.reporting.style import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", required=True, help="dataset whose test images are used")
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--uids", nargs="*")
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--pre-attention", action="store_true", help="read wrapped blocks before their attention")
    ap.add_argument("--channels-layer", type=int, help="also save the top-16 channel grid of this layer")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    dev = f"cuda:{a.device}" if torch.cuda.is_available() and a.device != "cpu" else "cpu"
    models = {r: load_run_model(paths().run_dir(r)).to(dev) for r in a.runs}
    samples = [s for s in samples_from_coco(paths().data_dir(a.dataset), "test") if len(s.boxes)]
    if a.uids:
        samples = [s for s in samples if s.uid in set(a.uids)]
    else:  # images whose smallest box is smallest relative to the image
        rel = [np.prod(s.boxes[:, 2:] - s.boxes[:, :2], 1).min() / (s.width * s.height) for s in samples]
        samples = [samples[i] for i in np.argsort(rel)[: a.n_images]]
    out = paths().results / "figures" / "feature_maps" / a.dataset
    for s in samples:
        img = cv2.imread(str(s.img_path))
        fig = compare_runs(models, img, a.layers, s.boxes, pre_attention=a.pre_attention)
        save(fig, out / f"compare_{s.uid}.png")
        if a.channels_layer is not None:
            x = letterbox(img)[0]
            for r, m in models.items():
                f = capture(m, x, [a.channels_layer], a.pre_attention)[a.channels_layer]
                save(channel_grid(f, title=f"{r}: layer {a.channels_layer}"), out / f"channels_{r}_{s.uid}.png")
        print("saved", s.uid)
    print("->", out)


if __name__ == "__main__":
    main()
