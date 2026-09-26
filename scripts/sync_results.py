"""Copy runs from a previous Kaggle session (its output attached as an input dataset) into the working results.

    python scripts/sync_results.py /kaggle/input/sdd-results-v3/results
"""

import argparse

import _bootstrap  # noqa: F401

from sdd.env import sync_results
from sdd.reporting.master import write_master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args()
    for s in a.sources:
        print(f"{s}: copied {sync_results(s)} runs")
    print("master table:", write_master())


if __name__ == "__main__":
    main()
