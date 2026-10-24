#!/usr/bin/env python3
"""Tables and dark-theme figures for results/supp/run_results.csv (see tools/supp_run.py).

    python tools/supp_report.py [results/supp/run_results.csv]

Without the full run_results.csv (written by supp_run.py, or downloaded from the release) the
tracked run_comparison.csv rebuilds the same tables and figures.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "supp"
# Same theme and model colours as tools/benchmark.py; RG without recursion keeps RG's hue, dashed.
FIG_BG, SURFACE, INK, INK2, GRID = "#0d0f14", "#11131c", "#f2f2f2", "#b0b5c7", "#2f3a56"
COLORS = {"rg": "#3987e5", "rg_l0": "#3987e5", "nsgd": "#d95926", "syncmap": "#199e70",
          "symmetric": "#c98500", "tsf": "#d55181"}
NAMES = {"rg": "RG", "rg_l0": "RG w/o recursion", "nsgd": "NSGD", "syncmap": "SyncMap",
         "symmetric": "Symmetric", "tsf": "TSFMap"}
ORDER = ["syncmap", "symmetric", "tsf", "nsgd", "rg", "rg_l0"]
HIER_ORDER = ["DCH1_fixed_chunk_6_3", "HIMB3", "H4x4x5", "H3x3x3x4"]
HIER_TITLE = {"DCH1_fixed_chunk_6_3": "DCH1 · 18 states · 6/2", "HIMB3": "HIMB3 · 48 · 10/3 uneven",
              "H4x4x5": "H4x4x5 · 80 · 16/4", "H3x3x3x4": "H3x3x3x4 · 108 · 27/9/3"}


def level_mean(r: pd.Series) -> float:
    if r["dataset_family"] != "hierarchy":
        return float(r["ari_score"])
    n = int(r["hierarchy_scored_level_count"])
    vals = [r.get(f"cluster_level{i}_ari_score") for i in range(1, n + 1)]
    return float(np.mean([0.0 if pd.isna(v) else v for v in vals]))


def late_levels(curve: str, window: int = 5) -> dict[str, float]:
    """Per-level ARI averaged over the last `window` checkpoints of a saved learning curve."""
    points = ast.literal_eval(curve) if isinstance(curve, str) else []
    tail = points[-window:]
    keys = sorted({k for p in tail for k in p if re.fullmatch(r"cluster_level\d+_ari_score", k)})
    return {f"late_{k}": float(np.mean([p.get(k) or 0.0 for p in tail])) for k in keys}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["stem"] = df["dataset"].str.replace(".dot", "", regex=False)
    df["score"] = df.apply(level_mean, axis=1)
    # late window: mean of the last five sampled points; fall back to the endpoint when absent
    df["late"] = df["late_mean_ari"].where(df["late_mean_ari"].notna(), df["score"])
    if "learning_curve" in df.columns:
        df = pd.concat([df, pd.DataFrame([late_levels(c) for c in df["learning_curve"]], index=df.index)], axis=1)
    return df


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def fig_orientation(df: pd.DataFrame) -> Path:
    flat = df[df["dataset_family"] == "flat"].copy()
    flat["k"] = flat["stem"].str.extract(r"fixed(\d+)_")[0].astype(int)
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=150, facecolor=FIG_BG)
    style(ax)
    for v in ["syncmap", "symmetric", "tsf", "nsgd", "rg"]:
        g = flat[flat["variant"] == v].groupby("k")["late"].agg(["mean", "std"]).sort_index()
        if g.empty:
            continue
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[v], alpha=0.12, linewidth=0)
        ax.plot(g.index, g["mean"], color=COLORS[v], lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label=NAMES[v])
        ax.annotate(NAMES[v], (g.index[-1], g["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.set_xscale("log")
    ks = sorted(flat["k"].unique())
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}×{300 // k}" for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("communities × states per community (300 states, fixed family)", color=INK2, fontsize=10)
    ax.set_ylabel("ARI, late-window mean (5 seeds, ±1 sd)", color=INK2, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.45)
    leg = ax.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.12), fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    path = OUT / "figures" / "orientation_sweep.png"
    fig.savefig(path, facecolor=FIG_BG)
    plt.close(fig)
    return path


def fig_hierarchy(df: pd.DataFrame) -> Path:
    hier = df[df["dataset_family"] == "hierarchy"]
    stems = [s for s in HIER_ORDER if s in set(hier["stem"])]
    fig, axes = plt.subplots(1, len(stems), figsize=(3.4 * len(stems), 4.4), dpi=150, facecolor=FIG_BG, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, stem in zip(axes, stems):
        style(ax)
        sub = hier[hier["stem"] == stem]
        for v in ["nsgd", "rg", "rg_l0"]:
            g = sub[sub["variant"] == v].groupby("decay")["late"].agg(["mean", "std"]).sort_index()
            if g.empty:
                continue
            ls = "--" if v == "rg_l0" else "-"
            mfc = SURFACE if v == "rg_l0" else COLORS[v]
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COLORS[v], lw=2, ls=ls, marker="o", ms=6,
                        mfc=mfc, mec=COLORS[v], mew=1.6, capsize=3, elinewidth=1, label=NAMES[v])
        ax.set_title(HIER_TITLE.get(stem, stem), color=INK, fontsize=10)
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.set_xticklabels(["0.25\neasy", "0.5", "0.75\nhard"])
        ax.set_xlim(0.18, 0.82)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("level-mean ARI, late window (5 seeds, ±1 sd)", color=INK2, fontsize=10)
    fig.supxlabel("coarse-level mixing (level_switch_decay)", color=INK2, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "figures" / "hierarchy_difficulty.png"
    fig.savefig(path, facecolor=FIG_BG)
    plt.close(fig)
    return path


def fig_showcase() -> Path | None:
    """Seed-0 final maps on H4x4x5 in each model's own 3-D coordinates: colour = top group, spokes = fine chunk."""
    import json

    sys.path.insert(0, str(ROOT / "tools"))
    from benchmark import CLASS_COLORS, VIEW_AZIM, VIEW_ELEV, _style3d, _unit_cube

    runs = {(r["variant"], r["decay"]): r for r in (json.loads(p.read_text(encoding="utf-8"))
                                                   for p in sorted((OUT / "showcase").glob("*.json")))}
    decays = sorted({d for _, d in runs})
    variants = [v for v in ("nsgd", "rg_l0", "rg") if any(k[0] == v for k in runs)]
    if not runs:
        return None
    fig = plt.figure(figsize=(4.2 * len(variants), 4.1 * len(decays)), facecolor=FIG_BG)
    for row, decay in enumerate(decays):
        for col, v in enumerate(variants):
            run = runs.get((v, decay))
            ax = fig.add_subplot(len(decays), len(variants), row * len(variants) + col + 1, projection="3d")
            _style3d(ax)
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
            if run is None:
                continue
            xyz = _unit_cube(run["positions"])
            fine = np.asarray(run["true_label_levels"]["level_0"])
            top = np.asarray(run["true_label_levels"][f"level_{len(run['true_label_levels']) - 1}"])
            tops = sorted(set(top.tolist()))
            for f in sorted(set(fine.tolist())):
                idx = np.flatnonzero(fine == f)
                color = CLASS_COLORS[tops.index(top[idx[0]]) % 8]
                centre = xyz[idx].mean(axis=0)
                for i in idx:
                    ax.plot(*zip(xyz[i], centre), color=color, linewidth=0.8, alpha=0.5)
                ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], s=22, color=color, edgecolors=SURFACE, linewidths=0.5,
                           depthshade=False)
            lv = " / ".join(f"{run['level_ari'][k]:.2f}" for k in sorted(run["level_ari"]))
            ax.set_title(f"{NAMES[v]}   L1 / L2 = {lv}", color=INK, fontsize=10.5, pad=0)
        fig.text(0.005, 1 - (row + 0.5) / len(decays), f"decay {decay:g}", rotation=90, va="center", color=INK2, fontsize=11)
    fig.text(0.5, 0.01, "H4x4x5, seed 0, 100k steps · colour = true top-level group (4), spokes join each fine chunk (16) to its centre",
             ha="center", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.02, right=1, top=0.95, bottom=0.05, wspace=0, hspace=0.08)
    path = OUT / "figures" / "latent_H4x4x5.png"
    fig.savefig(path, dpi=150, facecolor=FIG_BG)
    plt.close(fig)
    return path


def fmt(m: float, s: float) -> str:
    return f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}"


def tables(df: pd.DataFrame) -> str:
    out: list[str] = []
    flat = df[df["dataset_family"] == "flat"]
    if not flat.empty:
        out.append("## Flat: community orientation (300 states)\n")
        vs = [v for v in ORDER if v in set(flat["variant"])]
        out.append("| dataset | " + " | ".join(NAMES[v] for v in vs) + " |")
        out.append("|---|" + "---:|" * len(vs))
        key = lambda s: int(s.split("fixed")[1].split("_")[0])
        for stem in sorted(flat["stem"].unique(), key=key):
            row = [fmt(*flat[(flat.stem == stem) & (flat.variant == v)]["late"].agg(["mean", "std"])) for v in vs]
            out.append(f"| {stem} | " + " | ".join(row) + " |")
        out.append("")
    hier = df[df["dataset_family"] == "hierarchy"]
    if not hier.empty:
        out.append("## Hierarchy: level-mean ARI (late window)\n")
        vs = [v for v in ORDER if v in set(hier["variant"])]
        out.append("| dataset | decay | " + " | ".join(NAMES[v] for v in vs) + " |")
        out.append("|---|---:|" + "---:|" * len(vs))
        for stem in [s for s in HIER_ORDER if s in set(hier["stem"])]:
            for d in sorted(hier["decay"].dropna().unique()):
                sub = hier[(hier.stem == stem) & (hier.decay == d)]
                row = [fmt(*sub[sub.variant == v]["late"].agg(["mean", "std"])) for v in vs]
                out.append(f"| {stem} | {d:g} | " + " | ".join(row) + " |")
        out.append("")
        out.append("## Hierarchy: per-level ARI, late window (fine → coarse), 5-seed mean\n")
        out.append("| dataset | decay | " + " | ".join(NAMES[v] for v in vs) + " |")
        out.append("|---|---:|" + "---|" * len(vs))
        for stem in [s for s in HIER_ORDER if s in set(hier["stem"])]:
            for d in sorted(hier["decay"].dropna().unique()):
                cells = []
                for v in vs:
                    sub = hier[(hier.stem == stem) & (hier.decay == d) & (hier.variant == v)]
                    if sub.empty:
                        cells.append("")
                        continue
                    n = int(sub["hierarchy_scored_level_count"].iloc[0])
                    col = "late_cluster_level{}_ari_score" if "late_cluster_level1_ari_score" in sub else "cluster_level{}_ari_score"
                    lv = [sub[col.format(i)].fillna(0).mean() for i in range(1, n + 1)]
                    cells.append(" / ".join(f"{x:.2f}" for x in lv))
                out.append(f"| {stem} | {d:g} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("## Mean seconds per run\n")
    rt = df.groupby(["stem", "variant"])["elapsed_time"].mean().unstack()
    vs = [v for v in ORDER if v in rt.columns]
    out.append("| dataset | " + " | ".join(NAMES[v] for v in vs) + " |")
    out.append("|---|" + "---:|" * len(vs))
    for stem, r in rt.iterrows():
        out.append(f"| {stem} | " + " | ".join(f"{r[v]:.0f}" for v in vs) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, default=None,
                        help="results CSV; default: run_results.csv from supp_run.py, else the tracked run_comparison.csv")
    args = parser.parse_args()
    path = args.csv or next((p for p in (OUT / "run_results.csv", OUT / "run_comparison.csv") if p.exists()), OUT / "run_results.csv")
    prefix = path.stem.split("_")[0]
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    df = load(path)
    if "learning_curve" in df.columns:
        # The repository keeps a compact table (late per-level columns included); the full CSV with
        # learning curves goes to the release.
        compact = df.drop(columns=[c for c in ("learning_curve", "stem", "score", "late") if c in df.columns])
        compact.to_csv(OUT / f"{prefix}_comparison.csv", index=False)
    md = tables(df)
    figs = []
    if (df["dataset_family"] == "flat").any():
        figs.append(fig_orientation(df))
    if (df["dataset_family"] == "hierarchy").any():
        figs.append(fig_hierarchy(df))
    if (OUT / "showcase").is_dir():
        figs += [p for p in [fig_showcase()] if p is not None]
    (OUT / f"{prefix}_tables.md").write_text(md, encoding="utf-8")
    print(md)
    for f in figs:
        print("figure", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
