# Datasets

Record here, for every dataset actually used: **source URL / Kaggle slug, version, license, download date**
(plan §2.2). `scripts/download_data.py` fetches them (sources in `configs/sources.yaml`) and writes this information
to `<raw>/MANIFEST.md` - paste that table below. Raw data is never committed; paths are in `configs/paths.yaml`.

Known data issues found by the checks:
- Magnetic Tile: 6 images in `MT_Uneven` have an empty (4) or near-empty (26-45 px) mask -> skipped by the reader
  (otherwise they would become false "clean" backgrounds); 19 mask components < `min_area` dropped.

| id | Dataset | Source (fill in) | License | Downloaded | Notes |
|---|---|---|---|---|---|
| neu | NEU-DET (Song & Yan 2013) | Kaggle mirror: | research use | | 1800 imgs, 200x200, 6 classes, VOC xml |
| gc10 | GC10-DET (Lv et al. 2020) | Kaggle `alex000kim/gc10det` | | | 2048x1000, 10 classes; xml names mapped in datasets.yaml |
| pcb | PKU-Market-PCB | Kaggle mirror: | | | ~690 imgs, very high resolution, 6 classes |
| mt | Magnetic Tile (Huang et al. 2020) | github.com/abin24/Magnetic-tile-defect-datasets | | | masks -> boxes, `MT_Free` kept as background |
| ksdd2 | KolektorSDD2 (Božič et al. 2021) | vicos.si/resources/kolektorsdd2 | CC BY-NC-SA 4.0 | | target set (phase 3 only); masks -> boxes |

## Expected raw layouts

Readers search recursively below the configured root, so any mirror with these file patterns works:

- NEU-DET: `*.xml` + images with the same stem anywhere below the root
- GC10-DET: `lable/*.xml` (or `label/`) + `<class_no>/*.jpg`
- PKU-Market-PCB: `Annotations/<Class>/*.xml` + `images/<Class>/*.jpg`
- Magnetic Tile: `MT_<Class>/Imgs/<stem>.jpg` + `<stem>.png` (mask)
- KolektorSDD2: `train/<id>.png` + `train/<id>_GT.png`, same for `test/`

If a reader reports unknown class names, add them to `class_map` in `configs/datasets.yaml`.

## Processing decisions (keep in sync with the report)

- VOC boxes are treated as 1-based inclusive and converted to 0-based pixel extents.
- Mask datasets: connected components (8-connectivity); components smaller than `min_area` px are dropped and
  counted in `meta.json -> read_report.dropped_small`. Defect-free images get empty label files.
- Splits: stratified by dominant class (or defect-free), seed 42, stored in `splits/<id>/`; KolektorSDD2 keeps the
  official train/test and carves 10% of train as val.
