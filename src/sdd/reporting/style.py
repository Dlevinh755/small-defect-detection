"""Shared matplotlib style for every figure (slides / report): one palette, thin marks, recessive chrome.

Categorical colours are assigned in this fixed order and follow the entity (model / variant / init), never
its rank - use :func:`color_map` so a model keeps its colour across all figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal-safe blue ramp (step 250 -> 700)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "patch.linewidth": 0,
    })


def color_map(entities) -> dict:
    """Stable entity -> colour. Pass the full, ordered entity list (not a filtered one)."""
    ents = list(dict.fromkeys(entities))
    if len(ents) > len(CATEGORICAL):
        raise ValueError(f"{len(ents)} series > {len(CATEGORICAL)} colours: facet or fold into 'Other'")
    return {e: CATEGORICAL[i] for i, e in enumerate(ents)}


def save(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


apply()
