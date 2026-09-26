"""Exploratory data analysis (plan §2.4): tables + figures to <results>/eda/<ds>/.

    python scripts/eda.py --datasets neu gc10 pcb mt
"""

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from sdd.data.build import build_dataset
from sdd.eda.analysis import run_eda
from sdd.env import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        d = build_dataset(ds)  # no-op if already built
        out = paths().results / "eda" / ds
        rows.append(run_eda(d, out, ds))
        print(json.dumps(rows[-1], indent=2))
        print(pd.read_csv(out / "rel_bin_candidates.csv").to_string(index=False))
    pd.DataFrame(rows).to_csv(paths().results / "eda" / "datasets_summary.csv", index=False)


if __name__ == "__main__":
    main()
