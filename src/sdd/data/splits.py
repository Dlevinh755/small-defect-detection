"""Fixed train/val/test splits (plan §2.3, §2.5.3a): created once with seed 42, saved to ``splits/<ds>/*.txt``,
never redrawn. Stratified by the rarest class present in each image, or by group (``group_regex``)."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..config import SPLITS
from .readers import Sample

log = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")


def _allocate(n: int, ratios: list[float]) -> list[int]:
    """Largest-remainder rounding of n * ratios so the counts sum to n."""
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    return counts


def stratified_split(keys: dict[str, str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Split ids by their stratification key. ``keys`` maps id -> key; ``ratios`` maps split -> fraction."""
    names = list(ratios)
    total = sum(ratios.values())
    fr = [ratios[s] / total for s in names]
    groups = defaultdict(list)
    for uid, k in sorted(keys.items()):
        groups[k].append(uid)
    rng = random.Random(seed)
    out = {s: [] for s in names}
    # Allocate on cumulative counts so rounding remainders carry over between groups: global totals are
    # exact and small groups do not all round the same way (which would e.g. leave the test split empty).
    assigned, seen = [0] * len(names), 0
    for k in sorted(groups):
        ids = groups[k]
        rng.shuffle(ids)
        seen += len(ids)
        counts = [max(t - a, 0) for t, a in zip(_allocate(seen, fr), assigned)]
        while sum(counts) > len(ids):
            counts[counts.index(max(counts))] -= 1
        while sum(counts) < len(ids):
            counts[0] += 1
        start = 0
        for j, (s, c) in enumerate(zip(names, counts)):
            out[s].extend(ids[start : start + c])
            start += c
            assigned[j] += c
    return {s: sorted(v) for s, v in out.items()}


def rarest_class_keys(samples: list[Sample]) -> dict[str, str]:
    """Stratification key per image = its rarest class (by number of images containing it), 'clean' if none.

    Keying on the rarest class (not the dominant one) guarantees rare classes appear in train, val and test
    (plan §2.5.3a)."""
    freq = Counter(c for s in samples for c in set(s.classes.tolist()))
    keys = {}
    for s in samples:
        cls = set(s.classes.tolist())
        keys[s.uid] = str(min(cls, key=lambda c: (freq[c], c))) if cls else "clean"
    return keys


def group_split(samples: list[Sample], groups: dict[str, str], ratios: dict[str, float],
                seed: int) -> dict[str, list[str]]:
    """Whole groups (e.g. physical boards) go to one split; groups are assigned largest-first to the split that is
    furthest below its target image count."""
    members = defaultdict(list)
    for s in samples:
        members[groups[s.uid]].append(s.uid)
    order = sorted(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(members[g]))  # stable: equal-size groups keep the seeded order
    total = sum(ratios.values())
    target = {k: len(samples) * v / total for k, v in ratios.items()}
    out = {k: [] for k in ratios}
    for g in order:
        k = max(out, key=lambda k: target[k] - len(out[k]))
        out[k].extend(members[g])
    return {k: sorted(v) for k, v in out.items()}


def make_splits(samples: list[Sample], cfg: dict, seed: int) -> dict[str, list[str]]:
    sp = cfg["split"]
    if sp.get("official"):
        tr = [s for s in samples if s.meta.get("official_split") == "train"]
        test = sorted(s.uid for s in samples if s.meta.get("official_split") == "test")
        tv = stratified_split(rarest_class_keys(tr), {"train": 1 - sp["val"], "val": sp["val"]}, seed)
        return {"train": tv["train"], "val": tv["val"], "test": test}
    ratios = {k: sp[k] for k in SPLIT_NAMES}
    if sp.get("group_regex"):
        rx = re.compile(sp["group_regex"])
        groups = {}
        for s in samples:
            m = rx.search(s.uid)
            if not m:
                raise ValueError(f"group_regex {sp['group_regex']!r} does not match '{s.uid}'")
            groups[s.uid] = m.group(1)
        log.info("group split: %d groups", len(set(groups.values())))
        return group_split(samples, groups, ratios, seed)
    return stratified_split(rarest_class_keys(samples), ratios, seed)


def split_dir(dataset: str) -> Path:
    return SPLITS / dataset


def load_splits(dataset: str) -> dict[str, list[str]] | None:
    d = split_dir(dataset)
    if not all((d / f"{s}.txt").exists() for s in SPLIT_NAMES):
        return None
    return {s: (d / f"{s}.txt").read_text().split() for s in SPLIT_NAMES}


def save_splits(dataset: str, splits: dict[str, list[str]]) -> None:
    d = split_dir(dataset)
    d.mkdir(parents=True, exist_ok=True)
    for s in SPLIT_NAMES:
        (d / f"{s}.txt").write_text("\n".join(splits[s]) + "\n")


def get_or_create_splits(dataset: str, samples: list[Sample], cfg: dict, seed: int) -> dict[str, list[str]]:
    splits = load_splits(dataset)
    uids = {s.uid for s in samples}
    if splits is None:
        splits = make_splits(samples, cfg, seed)
        save_splits(dataset, splits)
        log.info("Created splits for %s: %s", dataset, {k: len(v) for k, v in splits.items()})
    else:
        listed = set().union(*map(set, splits.values()))
        missing = listed - uids
        if missing:
            raise ValueError(
                f"{len(missing)} ids in splits/{dataset} not found in the raw data (e.g. {sorted(missing)[:3]}). "
                "Wrong raw folder? Splits are never regenerated automatically."
            )
        extra = uids - listed
        if extra:
            log.warning("%d raw images of %s are not in any split and will be ignored", len(extra), dataset)
    overlap = set(splits["train"]) & (set(splits["val"]) | set(splits["test"]))
    assert not overlap and not set(splits["val"]) & set(splits["test"]), "split leakage"
    return splits
