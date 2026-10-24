#!/usr/bin/env python3
"""Datasets for the supplementary RG / NSGD experiments (written to datasets/supp/).

    python tools/make_supp_datasets.py          # write the DOT files
    python tools/make_supp_datasets.py --stats  # print how separated each level is

Flat: fixed K x S graphs with the same 300 states, so community count and size trade off
at constant scale (the existing fixed5_60 / fixed60_5 are the two ends).
Hierarchy: chunked hierarchies with more top-level groups, more states, uneven sizes and
three levels, in the format of datasets/hierarchy/DCH1_fixed_chunk_6_3.dot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "supp"

FLAT = [(3, 100), (10, 30), (15, 20), (20, 15), (30, 10)]

# Nested lists, coarse to fine; the innermost numbers are chunk sizes.
HIERARCHY: dict[str, list[Any]] = {
    "H4x4x5": [[5, 5, 5, 5] for _ in range(4)],
    "HIMB3": [[8, 6], [5, 5, 4], [3, 3, 4, 4, 6]],
    "H3x3x3x4": [[[4, 4, 4] for _ in range(3)] for _ in range(3)],
}


def write_flat(k: int, s: int) -> Path:
    lines = ["digraph causal", "{"]
    for node in range(k * s):
        lines.append(f'\t{node} [label="{node // s + 1}"]\t')
    firsts = [c * s for c in range(k)]
    for c in range(k):
        last = c * s + s - 1
        lines += [f"\t{last} -> {f}" for f in firsts]
    for c in range(k):
        lines += [f"\t{c * s + i} -> {c * s + i + 1}" for i in range(s - 1)]
    lines.append("}")
    path = OUT / f"fixed{k}_{s}.dot"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _depth(spec: Any) -> int:
    return 1 + (_depth(spec[0]) if isinstance(spec, list) else 0)


def write_hierarchy(name: str, spec: list[Any]) -> Path:
    depth = _depth(spec) - 1  # label levels; the outermost list is the root
    counters = [0] * depth  # next group id per level, fine = index 0
    leaves: list[tuple[int, list[int]]] = []
    edges: list[tuple[str, str]] = []
    internal: list[str] = []
    next_leaf = 0

    def walk(node: Any, level: int, path: list[int]) -> str:
        # level counts label levels from coarse (depth-1) down to fine (0)
        nonlocal next_leaf
        counters[level] += 1
        gid = counters[level]
        name_ = f"c_{gid}_{level + 1}"
        internal.append(name_)
        labels = [gid] + path
        if isinstance(node, list):
            for child in node:
                child_name = walk(child, level - 1, labels)
                edges.append((child_name, name_))
        else:
            for _ in range(int(node)):
                leaves.append((next_leaf, labels))
                edges.append((str(next_leaf), name_))
                next_leaf += 1
        return name_

    for top in spec:
        edges.append((walk(top, depth - 1, []), "root"))

    lines = ["digraph causal", "{"]
    for leaf, labels in leaves:
        # each group prepends its id, so labels are already fine-first like the existing files
        lines.append(f'\t{leaf} [label="{",".join(str(v) for v in labels)}"]\t')
    lines += [f"\t{n}" for n in internal] + ["\troot"]
    lines += [f"\t{a} -> {b}" for a, b in edges]
    lines.append("}")
    path = OUT / f"{name}.dot"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stats(paths: list[Path], decays: list[float]) -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from test.HierarchySequenceTest import _build_exact_hierarchy_transition, _default_level_switch_probs
    from test.graph_dataset_loader import load_dot_observed_labels

    print("P(next state stays in its own group), stationary average, fine -> coarse")
    for path in paths:
        levels = np.asarray(load_dot_observed_labels(str(path)).label_levels, dtype=int)
        sizes = [len(np.unique(levels[:, i])) for i in range(levels.shape[1])]
        for decay in decays:
            a = _build_exact_hierarchy_transition(levels, _default_level_switch_probs(levels.shape[1], 0.45, decay))
            w, v = np.linalg.eig(a.T)
            pi = np.real(v[:, np.argmin(np.abs(w - 1))])
            pi = pi / pi.sum()
            stay = [float(sum(pi[i] * a[i, levels[:, l] == levels[i, l]].sum() for i in range(len(pi)))) for l in range(levels.shape[1])]
            print(f"  {path.stem:12s} n={len(pi):3d} groups={sizes} decay={decay:.2f} stay=" + " / ".join(f"{s:.2f}" for s in stay))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    flat = [write_flat(k, s) for k, s in FLAT]
    hier = [write_hierarchy(name, spec) for name, spec in HIERARCHY.items()]
    for p in flat + hier:
        print("wrote", p.relative_to(ROOT))
    if args.stats:
        stats([ROOT / "datasets/hierarchy/DCH1_fixed_chunk_6_3.dot", *hier], [0.25, 0.5, 0.75])


if __name__ == "__main__":
    main()
