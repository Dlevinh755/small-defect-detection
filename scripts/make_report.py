"""Rebuild master_results.csv and produce every table / figure for the reports.

    python scripts/make_report.py                 # all phases present in the results
    python scripts/make_report.py --gallery p1_gc10_yolo11n_base_s0
"""

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from sdd.env import paths
from sdd.reporting import plots
from sdd.reporting.error_gallery import gallery
from sdd.reporting.master import load_master, write_master
from sdd.reporting.style import save
from sdd.reporting.tables import (main_data_version, save_table, table_ablation, table_balance, table_per_class,
                                  table_phase1, table_transfer, with_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="A0", help="ablation reference variant")
    ap.add_argument("--test-metric", default="AP_s")
    ap.add_argument("--include-smoke", action="store_true", help="also report --smoke runs (pipeline check)")
    ap.add_argument("--gallery", nargs="*", default=[], help="run names to build error galleries for")
    a = ap.parse_args()
    write_master(include_smoke=a.include_smoke)
    df = load_master()
    if df.empty:
        print("no finished runs yet")
        return
    tdir, fdir = paths().results / "tables", paths().results / "figures"
    phases = set(df.phase)
    if "p2t" in phases:  # large-image track next to the phase-1 rows it is compared with
        t = with_labels(df[df.phase == "p2t"])
        p1 = df[(df.phase == "p1") & (df.model == "yolo11n")]
        if "data_version" in p1:
            p1 = p1[p1.data_version == main_data_version(p1)]
        ref = p1[with_labels(p1).dataset.isin(set(t.dataset))]
        df = pd.concat([df, ref.assign(phase="p2t")], ignore_index=True)
    for ph in sorted(p for p in phases if p.startswith("p1") or p in ("p2s", "p2t")):
        d = df[df.phase == ph]
        save_table(table_phase1(df, ph, main_data_only=ph == "p1"), tdir, f"{ph}_results")  # table A
        for ds in sorted(set(with_labels(d).dataset)):
            pc = table_per_class(df, ph, ds)
            if not pc.empty:
                save_table(pc, tdir / "per_class", f"{ph}_{ds}")
        for metric in ("AP", "AP_s", "AP_rel_small", "R_rel_small"):
            if metric in d:
                save(plots.metric_by_model(d, metric, f"{ph.upper()}: {metric} by model"), fdir / ph / f"{metric}_by_model.png")
        for ds in sorted(set(with_labels(d).dataset)):
            fig = plots.tide_by_model(d, ds)
            if fig is not None:
                save(fig, fdir / ph / f"tide_{ds}.png")
            if "GFLOPs" in d:
                save(plots.cost_vs_accuracy(d, "AP_s", "GFLOPs", ds), fdir / ph / f"cost_{ds}.png")
    if "p1" in phases:  # table B: effect of the class-imbalance handling (plan §4.5)
        tb = table_balance(df, "p1")
        if not tb.empty:
            save_table(tb, tdir, "p1_balance_effect")
    if "p2" in phases:
        save_table(table_ablation(df, "p2", a.reference, a.test_metric), tdir, "p2_ablation")
        save(plots.metric_by_model(df[df.phase == "p2"], a.test_metric, f"Ablation: {a.test_metric}"),
             fdir / "p2" / f"{a.test_metric}_ablation.png")
    if "p3" in phases:
        save_table(table_transfer(df), tdir, "p3_transfer")
        for metric in ("AP_s", "AP", "img_detection_rate", "false_alarm_rate"):
            if metric in df:
                save(plots.transfer_curves(df, metric), fdir / "p3" / f"transfer_{metric}.png")
    for run in a.gallery:
        r = df[df.run == run].iloc[0]
        counts = gallery(paths().run_dir(run), paths().data_dir(r.dataset), fdir / "errors" / run)  # test = base test
        print(run, counts)
    print(f"tables -> {tdir}\nfigures -> {fdir}")


if __name__ == "__main__":
    main()
