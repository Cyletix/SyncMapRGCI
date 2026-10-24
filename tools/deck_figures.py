#!/usr/bin/env python3
"""Mechanism diagrams for the RG presentation (docs/presentation/), dark theme.

    python tools/deck_figures.py [--output docs/presentation/figures]
    python tools/deck_figures.py --bundle RG_presentation.html   # single-file copy for the release

Every diagram is drawn from the documented equations or from the public code paths it
illustrates (the relation-graph panel runs the real ``_NSTransitionTracker``; the
transition heatmaps read the real DOT files). No model is trained here; result figures
come from tools/benchmark.py and tools/supp_report.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from syncmap.SyncMapNS import _NSTransitionTracker
from test.graph_dataset_loader import load_dot_graph_dataset

DEFAULT_OUTPUT = ROOT / "docs" / "presentation" / "figures"

# Dark theme of tools/plot_theme.py / tools/benchmark.py; series hues are the benchmark's.
BG = "#0d0f14"
SURFACE = "#161a26"
INK = "#f2f2f2"
MUTED = "#b0b5c7"
GRID = "#2f3a56"
BLUE = "#3987e5"
ORANGE = "#d95926"
GREEN = "#199e70"
YELLOW = "#c98500"
PINK = "#d55181"
PURPLE = "#9085e9"
RED = "#e66767"
GRAY = "#6b7385"


def tint(color: str, amount: float = 0.22) -> str:
    """The hue mixed into the panel surface, for fills behind light text."""
    a = np.asarray(mcolors.to_rgb(color))
    b = np.asarray(mcolors.to_rgb(SURFACE))
    return mcolors.to_hex(b + (a - b) * amount)


BLUE_FILL, ORANGE_FILL, GREEN_FILL, RED_FILL, PURPLE_FILL = (tint(c) for c in (BLUE, ORANGE, GREEN, RED, PURPLE))

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.facecolor": BG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": BG,
    }
)


def canvas() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def save(fig: Figure, output: Path, filename: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / filename, dpi=150, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    print("wrote", (output / filename).relative_to(ROOT) if output.is_relative_to(ROOT) else output / filename)


def card(ax: Axes, xy: tuple[float, float], width: float, height: float, heading: str, body: str = "", *,
         color: str = BLUE, fill: str = SURFACE, fontsize: float = 13) -> None:
    x, y = xy
    ax.add_patch(patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.035,rounding_size=0.14",
                                        linewidth=1.6, edgecolor=color, facecolor=fill))
    ax.text(x + width / 2, y + height * 0.64, heading, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=INK)
    if body:
        ax.text(x + width / 2, y + height * 0.28, body, ha="center", va="center", fontsize=fontsize - 2, color=MUTED)


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = MUTED, width: float = 1.8,
          style: str = "-|>", connectionstyle: str = "arc3") -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": style, "color": color, "lw": width,
                                                      "connectionstyle": connectionstyle, "shrinkA": 2, "shrinkB": 2})


def dots(ax: Axes, xs, ys, color: str, size: float, z: int = 4, **kw) -> None:
    ax.scatter(xs, ys, s=size, color=color, edgecolor=BG, linewidth=1.4, zorder=z, **kw)


def cluster_points(ax: Axes, center: tuple[float, float], color: str, *, count: int = 7, radius: float = 0.55,
                   seed: int = 0, size: float = 58) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, radius * 0.37, size=(count, 2)) + np.asarray(center)
    dots(ax, pts[:, 0], pts[:, 1], color, size)
    return pts


def fig_cover(output: Path) -> None:
    fig, ax = canvas()
    ax.text(3.5, 8.38, "SyncMap", ha="center", va="center", fontsize=27, fontweight="bold", color=BLUE)
    ax.text(11.15, 8.38, "RG", ha="center", va="center", fontsize=27, fontweight="bold", color=ORANGE)
    arrow(ax, (6.95, 8.38), (7.95, 8.38), color=MUTED, width=2.2)
    ax.text(3.5, 7.82, "one geometry · flat DBSCAN readout", ha="center", fontsize=11.5, color=MUTED)
    ax.text(11.15, 7.82, "recursive geometry · distance and memory grow per level", ha="center", fontsize=11.5, color=MUTED)
    ax.add_patch(patches.FancyBboxPatch((0.7, 0.82), 5.6, 6.45, boxstyle="round,pad=0.035,rounding_size=0.14",
                                        linewidth=1.6, edgecolor=BLUE, facecolor=SURFACE))
    ax.text(3.5, 6.78, "one geometry space", ha="center", fontsize=13.5, fontweight="bold")
    for i, (cx, cy, color) in enumerate([(2.0, 3.45, BLUE), (4.8, 3.55, ORANGE), (3.4, 5.15, GREEN)]):
        cluster_points(ax, (cx, cy), color, seed=10 + i)
        ax.add_patch(patches.Ellipse((cx, cy), 1.65, 1.10, fill=False, edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.8))
    ax.text(3.5, 1.62, "symbols  →  geometry  →  DBSCAN", ha="center", fontsize=13, color=MUTED)

    ax.add_patch(patches.FancyBboxPatch((7.0, 0.82), 8.25, 6.45, boxstyle="round,pad=0.035,rounding_size=0.14",
                                        linewidth=1.6, edgecolor=ORANGE, facecolor=SURFACE))
    levels = [(2.05, "L0  symbols", BLUE, 12), (4.15, "L1  tokens", ORANGE, 4), (5.95, "L2  macro tokens", GREEN, 2)]
    positions: list[list[tuple[float, float]]] = []
    for y, label, color, n in levels:
        xs = np.linspace(9.55, 14.35, n)
        positions.append([(float(x), y) for x in xs])
        dots(ax, xs, [y] * n, color, 82 if n > 4 else 150)
        ax.text(7.42, y, label, ha="left", va="center", fontsize=11.8, fontweight="bold", color=color)
    for i, x in enumerate(np.linspace(9.55, 14.35, 12)):
        arrow(ax, (x, 2.18), (positions[1][min(i // 3, 3)][0], 3.93), color=GRID, width=1.1)
    for i, (x, _) in enumerate(positions[1]):
        arrow(ax, (x, 4.35), (positions[2][min(i // 2, 1)][0], 5.70), color=GRID, width=1.4)
    ax.text(11.15, 6.75, "tokenization  →  relearning", ha="center", fontsize=13.5, fontweight="bold")
    ax.text(14.55, 1.28, "same kernel · larger natural scale", ha="right", fontsize=11.5, color=MUTED)
    save(fig, output, "f01_cover_architecture.png")


def fig_syncmap(output: Path) -> None:
    fig, ax = canvas()
    ax.plot([7.95, 7.95], [0.9, 8.1], color=GRID, lw=1.4)
    ax.text(3.95, 7.80, "Paper dynamics", ha="center", fontsize=17, fontweight="bold")
    active = np.asarray([[1.25, 5.35], [2.65, 6.05], [4.25, 5.20]])
    inactive = np.asarray([[1.25, 2.35], [2.55, 1.70], [4.00, 2.55], [5.55, 1.85]])
    pc, nc = active.mean(axis=0), inactive.mean(axis=0)
    dots(ax, active[:, 0], active[:, 1], BLUE, 180)
    dots(ax, inactive[:, 0], inactive[:, 1], GRAY, 130)
    ax.scatter(*pc, marker="x", s=125, color=BLUE, linewidth=2.5, zorder=5)
    ax.scatter(*nc, marker="x", s=125, color=ORANGE, linewidth=2.5, zorder=5)
    for p in active:
        arrow(ax, tuple(p), tuple(p + 0.56 * (pc - p)), color=BLUE, width=3)
    for p in inactive:
        arrow(ax, tuple(p), tuple(p + 0.56 * (nc - p)), color=ORANGE, width=2.5)
    ax.annotate(r"positive centre  $c_p$", xy=pc, xytext=(2.95, 6.85), ha="center", fontsize=10.8, color=BLUE,
                fontweight="bold", arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1.2})
    ax.annotate(r"negative centre  $c_n$", xy=nc, xytext=(3.05, 3.40), ha="center", fontsize=10.8, color=ORANGE,
                fontweight="bold", arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.2})
    ax.text(6.95, 5.50, r"$PS_t \rightarrow c_p$", ha="right", fontsize=12.0, color=BLUE, fontweight="bold")
    ax.text(6.95, 2.30, r"$NS_t \rightarrow c_n$", ha="right", fontsize=12.0, color=ORANGE, fontweight="bold")
    ax.text(3.95, 0.96, "co-active symbols contract together; the rest contract elsewhere", ha="center", fontsize=10.5, color=MUTED)

    ax.text(11.95, 7.80, "Partition = DBSCAN readout", ha="center", fontsize=17, fontweight="bold")
    for i, (c, col) in enumerate(zip([(10.0, 5.2), (13.5, 5.2), (11.8, 3.2)], [BLUE, ORANGE, GREEN])):
        cluster_points(ax, c, col, count=6, seed=40 + i, radius=0.72, size=75)
        ax.add_patch(patches.Circle(c, radius=0.92, facecolor=col, edgecolor=col, alpha=0.10, lw=2))
    card(ax, (9.45, 0.90), 5.05, 1.05, "nearby dense points → same cluster ID", "isolated points may be marked as noise (−1)",
         color=GREEN, fill=GREEN_FILL, fontsize=11.5)
    save(fig, output, "f02_original_syncmap.png")


def fig_primitive(output: Path) -> None:
    fig, ax = canvas()
    ax.plot([8.0, 8.0], [1.0, 8.1], color=GRID, lw=1.4)
    ax.text(4.0, 7.75, "Active attraction", ha="center", fontsize=17, fontweight="bold", color=BLUE)
    active = np.asarray([[2.0, 4.9], [3.3, 5.9], [5.5, 4.5], [4.8, 3.1]])
    center = np.average(active, axis=0, weights=[1.0, 0.75, 0.9, 0.55])
    dots(ax, active[:, 0], active[:, 1], BLUE, 185)
    ax.scatter(*center, marker="x", s=150, color=INK, linewidth=2.6, zorder=5)
    for p in active:
        arrow(ax, tuple(p), tuple(p + 0.58 * (center - p)), color=BLUE, width=2.8)
    ax.text(center[0], center[1] + 0.38, r"weighted centre  $c_t$", ha="center", fontsize=12, fontweight="bold")
    ax.text(4.0, 1.55, r"active points contract toward $c_t$ (strength 0.5)", ha="center", fontsize=13, color=MUTED)

    ax.text(12.0, 7.75, "Inactive repulsion", ha="center", fontsize=17, fontweight="bold", color=ORANGE)
    inactive = np.asarray([[9.4, 5.1], [10.2, 2.8], [12.6, 5.9], [14.1, 3.2], [14.5, 5.2]])
    ic = np.asarray([12.0, 4.4])
    ax.scatter(*ic, marker="x", s=150, color=INK, linewidth=2.6, zorder=5)
    dots(ax, inactive[:, 0], inactive[:, 1], GRAY, 145)
    for p in inactive:
        d = (p - ic) / max(float(np.linalg.norm(p - ic)), 1e-9)
        arrow(ax, tuple(p), tuple(p + 0.72 * d), color=ORANGE, width=2.5)
    ax.text(ic[0], ic[1] + 0.38, r"same active centre  $c_t$", ha="center", fontsize=12, fontweight="bold")
    ax.text(12.0, 1.55, "push ∝ 1/distance (strength 2.0) · distance floor · step cap", ha="center", fontsize=12.3, color=MUTED)
    save(fig, output, "f08_primitive_force.png")


def fig_heatmaps(output: Path) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.95, top=0.90, bottom=0.22, wspace=0.24)
    specs = [
        ("datasets/chain_mixed5_6.dot", "chain_mixed5_6 · transitive chunks",
         "sparse one-step evidence; membership follows multi-step paths"),
        ("datasets/hierarchy/DCH1_fixed_chunk_6_3.dot", "DCH1 · two nested levels",
         "cyan = L1 chunks of 3 · white = L2 groups of 9"),
    ]
    norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
    images = []
    for col, (dataset, heading, note) in enumerate(specs):
        loaded = load_dot_graph_dataset(str(ROOT / dataset))
        transition = np.asarray(loaded.transition, dtype=float)
        levels = np.asarray(loaded.label_levels, dtype=int)
        axm = fig.add_subplot(gs[0, col])
        images.append(axm.imshow(transition, cmap="magma", norm=norm, interpolation="nearest", aspect="equal"))
        l1 = levels[:, 0]
        for b in np.flatnonzero(l1[1:] != l1[:-1]) + 1:
            axm.axhline(b - 0.5, color="#31D6D2", lw=0.9)
            axm.axvline(b - 0.5, color="#31D6D2", lw=0.9)
        if levels.shape[1] > 1:
            l2 = levels[:, 1]
            for b in np.flatnonzero(l2[1:] != l2[:-1]) + 1:
                axm.axhline(b - 0.5, color=INK, lw=2.2)
                axm.axvline(b - 0.5, color=INK, lw=2.2)
        centers = [float((np.flatnonzero(l1 == lab)[0] + np.flatnonzero(l1 == lab)[-1]) / 2) for lab in np.unique(l1)]
        axm.set_xticks(centers, [f"G{i + 1}" for i in range(len(centers))])
        axm.set_yticks(centers, [f"G{i + 1}" for i in range(len(centers))])
        axm.tick_params(length=0, labelsize=9.0)
        axm.set_ylabel("current symbol")
        axm.set_xlabel("next symbol")
        axm.set_title(heading, fontsize=14.2, fontweight="bold", pad=9, color=INK)
        axm.text(0.5, -0.17, note, transform=axm.transAxes, ha="center", fontsize=10.4, color=INK)
        for spine in axm.spines.values():
            spine.set_color(GRID)
    cbar = fig.colorbar(images[-1], ax=fig.axes[:2], orientation="horizontal", fraction=0.040, pad=0.20, aspect=42)
    cbar.set_label("P(next | current), one shared scale (power-law colour stretch only)", fontsize=9.5, color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(GRID)
    save(fig, output, "f03_transition_heatmaps.png")


def fig_lineage(output: Path) -> None:
    fig, ax = canvas()
    card(ax, (0.7, 3.95), 3.0, 1.55, "Original SyncMap", "single geometry", color=BLUE, fill=BLUE_FILL)
    arrow(ax, (3.75, 4.72), (5.0, 4.72), color=MUTED, width=2.6)
    card(ax, (5.05, 3.75), 3.0, 1.95, "Early NS (2025)", "second-order trials\nrelation-graph lessons", color=PURPLE, fill=SURFACE)
    arrow(ax, (8.10, 4.95), (9.55, 6.25), color=ORANGE, width=2.8, connectionstyle="arc3,rad=-0.08")
    card(ax, (9.65, 5.45), 5.45, 1.65, "NSGD · 2025-11-30", "first-order + plasticity + active pulse", color=ORANGE, fill=ORANGE_FILL)
    arrow(ax, (8.10, 4.45), (9.55, 3.05), color=GREEN, width=2.8, connectionstyle="arc3,rad=0.08")
    card(ax, (9.65, 2.10), 3.05, 1.65, "SyncMapNS", "rewritten reusable\nlevel kernel", color=GREEN, fill=GREEN_FILL)
    arrow(ax, (12.75, 2.92), (13.25, 2.92), color=GREEN, width=2.6)
    card(ax, (13.30, 2.10), 2.05, 1.65, "RG", "tokenize\n+ recurse", color=BLUE, fill=BLUE_FILL, fontsize=12.5)
    ax.text(8.0, 1.10, "Both branches learned from Early NS; RG does not inherit NSGD's class or parameter preset.",
            ha="center", fontsize=12.3, color=MUTED)
    save(fig, output, "f04_lineage.png")


def fig_kernel(output: Path) -> None:
    fig, ax = canvas()
    card(ax, (0.55, 2.65), 3.55, 4.15, "Shared Early NS lessons",
         "first-order position update\nno velocity / inertia\nonline relation dynamics", color=BLUE, fill=BLUE_FILL, fontsize=13.5)
    arrow(ax, (4.12, 4.70), (5.05, 4.70), color=MUTED, width=2.7)
    card(ax, (5.05, 2.30), 5.55, 4.85, "SyncMapNS · one reusable level",
         "transition statistics\n↓\nweighted graph + primitive contrast\n↓\ncanonical spring → coordinate update",
         color=PURPLE, fill=SURFACE, fontsize=13.2)
    ax.text(7.82, 1.82, "rewritten statistics, spring and force balance", ha="center", fontsize=11.5, color=PURPLE)
    arrow(ax, (10.62, 4.70), (11.45, 4.70), color=MUTED, width=2.7)
    card(ax, (11.45, 2.65), 3.95, 4.15, "SyncMapRG orchestration",
         "L0 kernel\n↓ stable tokens\nL1 same kernel\n↓ macro tokens\nL2 ...", color=GREEN, fill=GREEN_FILL, fontsize=13.2)
    ax.text(2.33, 1.82, "shared design experience", ha="center", fontsize=11.5, color=BLUE)
    ax.text(13.42, 1.82, "recursion added outside the kernel", ha="center", fontsize=11.5, color=GREEN)
    save(fig, output, "f05_ns_kernel_and_rg.png")


def fig_observation(output: Path) -> None:
    fig, ax0 = canvas()
    card(ax0, (0.65, 4.30), 3.15, 2.15, "GraphWalk state", "... A, A, A → B, B ...\ntime_delay = 10", color=BLUE, fill=BLUE_FILL)
    arrow(ax0, (3.85, 5.38), (5.00, 5.38), color=MUTED, width=2.6)
    card(ax0, (5.05, 4.30), 3.45, 2.15, "observation  x_t", "current state\n+ previous-state tail", color=PURPLE, fill=SURFACE)
    arrow(ax0, (8.55, 5.70), (10.00, 6.80), color=BLUE, width=2.5)
    arrow(ax0, (8.55, 5.06), (10.00, 3.90), color=ORANGE, width=2.5)
    card(ax0, (10.05, 6.20), 5.15, 1.35, "primitive force", "instantaneous co-activation drives geometry", color=BLUE, fill=BLUE_FILL, fontsize=12.5)
    card(ax0, (9.45, 3.05), 5.70, 1.45, "transition tracker", "adjacent thresholded active sets\n→ long-term relation memory",
         color=ORANGE, fill=ORANGE_FILL, fontsize=11.5)
    ax = fig.add_axes([0.30, 0.10, 0.30, 0.26], facecolor=BG)
    t = np.arange(0, 11)
    current, previous = np.exp(-0.1 * t), np.exp(-0.1 * (t + 10))
    ax.plot(t, current, color=BLUE, lw=2.5, label="current B")
    ax.plot(t, previous, color=ORANGE, lw=2.2, label="previous A tail")
    ax.fill_between(t, 0, np.minimum(current, previous), color=GREEN, alpha=0.18)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("steps after A→B", fontsize=8.5)
    ax.set_ylabel("amplitude", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(INK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax0.text(10.4, 1.30, "the switch frame holds both A and B:\nthat overlap is the transition evidence",
             ha="left", fontsize=11, color=MUTED)
    save(fig, output, "f06_graphwalk_observation.png")


def fig_relation_graph(output: Path) -> None:
    fig, ax = canvas()
    seq = [("A",), ("A", "B"), ("A", "B"), ("B",), ("B", "C"), ("C",), ("A", "C"), ("A",)]
    colors = {"A": BLUE, "B": ORANGE, "C": GREEN}
    index = {"A": 0, "B": 1, "C": 2}
    tracker = _NSTransitionTracker(3, memory_steps=300, power_gain=2.0)
    for active_set in seq:
        mask = np.zeros(3, dtype=bool)
        mask[[index[s] for s in active_set]] = True
        tracker.update(mask)
    sim = np.asarray(tracker.similarity(), dtype=float)
    edge = np.asarray(tracker.edge_strength(), dtype=float)
    ax.text(4.2, 7.55, "toy stream of thresholded active sets", ha="center", fontsize=12.5, color=MUTED)
    for i, active_set in enumerate(seq):
        x = 0.8 + i * 0.94
        fill = colors[active_set[0]] if len(active_set) == 1 else PURPLE
        ax.add_patch(patches.FancyBboxPatch((x - 0.34, 6.41), 0.68, 0.58, boxstyle="round,pad=0.025,rounding_size=0.14",
                                            facecolor=fill, edgecolor=BG, lw=1.2))
        ax.text(x, 6.70, "{" + ",".join(active_set) + "}", color=INK, ha="center", va="center", fontweight="bold", fontsize=9.5)
        if i < len(seq) - 1:
            arrow(ax, (x + 0.36, 6.70), (x + 0.56, 6.70), color=GRID, width=1.2)
    card(ax, (0.75, 2.45), 3.15, 1.35, "transition tracker", "adjacent-set Cartesian pairs\nsymmetric decayed counts C",
         color=PURPLE, fill=SURFACE, fontsize=11.2)
    arrow(ax, (2.6, 6.32), (2.6, 3.84), color=PURPLE, width=2.0)
    cell, x0, y0 = 0.82, 4.72, 2.55
    for row in range(3):
        for col in range(3):
            value = float(sim[row, col])
            ax.add_patch(patches.Rectangle((x0 + col * cell, y0 + (2 - row) * cell), cell, cell, facecolor=PURPLE,
                                           edgecolor=BG, lw=1.5, alpha=0.10 if value == 0 else 0.22 + 0.5 * value))
            ax.text(x0 + (col + 0.5) * cell, y0 + (2.5 - row) * cell, f"{value:.2f}", ha="center", va="center", fontsize=12,
                    fontweight="bold" if value > 0 else "normal")
    for i, s in enumerate(["A", "B", "C"]):
        ax.text(x0 + (i + 0.5) * cell, y0 + 3 * cell + 0.18, s, ha="center", fontsize=11.5, fontweight="bold")
        ax.text(x0 - 0.20, y0 + (2.5 - i) * cell, s, ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.text(x0 + 1.5 * cell, y0 - 0.40, r"similarity $C_{ij}/\sqrt{n_i n_j}$  (n = decayed activity)", ha="center", fontsize=10.2, color=MUTED)
    nodes = {"A": (10.4, 5.4), "B": (13.6, 5.6), "C": (12.15, 2.85)}
    for a, b in [("A", "B"), ("B", "C"), ("A", "C")]:
        w = max(edge[index[a], index[b]], edge[index[b], index[a]])
        if w <= 0:
            continue
        p, q = nodes[a], nodes[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.9 + w * 1.25, alpha=0.8, solid_capstyle="round", zorder=1)
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.18, f"w={w:.2f}", fontsize=10.5, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.12", "fc": BG, "ec": "none"})
    for s, p in nodes.items():
        ax.add_patch(patches.Circle(p, 0.42, facecolor=colors[s], edgecolor=BG, lw=2, zorder=3))
        ax.text(*p, s, color=INK, ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)
    arrow(ax, (3.92, 3.15), (4.62, 3.15), color=PURPLE, width=2.3)
    arrow(ax, (7.38, 3.62), (9.65, 4.22), color=PURPLE, width=2.3, connectionstyle="arc3,rad=-0.10")
    ax.text(12.0, 7.05, "power + top-k + threshold → sparse graph", ha="center", fontsize=14.0, fontweight="bold")
    ax.text(8.0, 1.05, "Undirected relation memory from the observation stream — not the DOT edges, not causal direction.",
            ha="center", fontsize=11.5, color=MUTED)
    save(fig, output, "f07_relation_graph.png")


def fig_spring(output: Path) -> None:
    fig, _ = canvas()
    ax = fig.add_axes([0.10, 0.20, 0.82, 0.70], facecolor=BG)
    x = np.linspace(0.18, 3.2, 500)
    f = np.where(x >= 1.0, x - 1.0, -(2.0 * (1.0 / x - 1.0) + 12.0 * (1.0 - x)))
    ax.plot(x, f, color=PURPLE, lw=4)
    ax.fill_between(x, f, 0, where=x < 1, color=ORANGE, alpha=0.22)
    ax.fill_between(x, 0, f, where=x >= 1, color=BLUE, alpha=0.22)
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.axvline(1, color=GREEN, lw=2.0, ls="--")
    ax.scatter([1], [0], s=90, color=GREEN, edgecolor=BG, zorder=5)
    ax.text(1.04, 1.0, r"equilibrium  $d=r_L$", color=GREEN, fontsize=12.5, fontweight="bold")
    ax.text(0.55, -13.8, "compression\n→ strong repulsion", ha="center", color=ORANGE, fontsize=12.5, fontweight="bold")
    ax.text(2.25, -2.6, "extension → attraction", ha="center", color=BLUE, fontsize=12.5, fontweight="bold")
    ax.set_xlim(0.18, 3.2)
    ax.set_ylim(-20.5, 2.5)
    ax.set_xlabel(r"normalised distance  $x=d/r_L$")
    ax.set_ylabel(r"radial factor  $F/(k_{attr}\,w_{eff}\,r_L)$")
    ax.grid(color=GRID, alpha=0.55)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.50, 0.07, "one ordinary edge; the real step also depends on edge weight, degree normalisation, "
             "cross-macro attenuation and clipping", ha="center", fontsize=10.8, color=MUTED)
    save(fig, output, "f09_spring_curve.png")


def fig_separations(output: Path) -> None:
    fig, ax = canvas()
    ax.plot([8.0, 8.0], [1.0, 8.1], color=GRID, lw=1.4)
    ax.text(4.0, 7.70, "Active pair + inactive point", ha="center", fontsize=16, fontweight="bold", color=BLUE)
    left, right = np.asarray([1.55, 4.65]), np.asarray([6.45, 4.65])
    pc = (left + right) / 2.0
    inactive = np.asarray([4.70, 5.55])
    away = (inactive - pc) / np.linalg.norm(inactive - pc)
    ax.plot([left[0], right[0]], [left[1], right[1]], color=BLUE, lw=5.5, alpha=0.25, solid_capstyle="round")
    dots(ax, [left[0], right[0]], [left[1], right[1]], BLUE, 220)
    dots(ax, [inactive[0]], [inactive[1]], GRAY, 155, z=5)
    ax.scatter(*pc, s=55, marker="x", color=PURPLE, linewidth=2.0, zorder=6)
    ax.plot([pc[0], inactive[0]], [pc[1], inactive[1]], color=MUTED, lw=1.4, ls="--")
    arrow(ax, tuple(left), (3.35, left[1]), color=BLUE, width=3.0)
    arrow(ax, tuple(right), (4.65, right[1]), color=BLUE, width=3.0)
    arrow(ax, tuple(inactive), tuple(inactive + 1.05 * away), color=ORANGE, width=3.0)
    ax.text(4.0, 3.75, "active points → positive centre", ha="center", fontsize=11.5, color=BLUE, fontweight="bold")
    ax.text(4.0, 3.15, "inactive point → away from the positive centre", ha="center", fontsize=11.2, color=ORANGE, fontweight="bold")
    ax.text(4.0, 2.30, "this primitive component does not push back on the active points", ha="center", fontsize=10.5)
    ax.text(4.0, 1.65, "arrows start at the point that moves · one component, not the full micro force", ha="center",
            fontsize=10.4, color=PURPLE, fontweight="bold")

    ax.text(12.0, 7.70, "Macro boundary separation", ha="center", fontsize=16, fontweight="bold", color=GREEN)
    ca, cb = np.asarray([10.55, 4.65]), np.asarray([13.45, 4.65])
    ax.add_patch(patches.Ellipse(tuple(ca), 2.20, 1.65, facecolor=BLUE_FILL, edgecolor=BLUE, lw=1.8, alpha=0.8))
    ax.add_patch(patches.Ellipse(tuple(cb), 2.20, 1.65, facecolor=ORANGE_FILL, edgecolor=ORANGE, lw=1.8, alpha=0.8))
    offsets = np.asarray([[-0.65, 0.0], [-0.34, 0.45], [-0.34, -0.45], [0.0, 0.0], [0.34, 0.45], [0.34, -0.45], [0.65, 0.0]])
    p1, p2 = offsets + ca, offsets + cb
    dots(ax, p1[:, 0], p1[:, 1], BLUE, 74)
    dots(ax, p2[:, 0], p2[:, 1], ORANGE, 74)
    ax.scatter([ca[0], cb[0]], [ca[1], cb[1]], s=80, marker="x", color=RED, linewidth=2.2, zorder=6)
    ax.plot([ca[0], cb[0]], [ca[1], cb[1]], color=RED, lw=1.8, ls="--", alpha=0.75, zorder=2)
    arrow(ax, tuple(ca), (9.20, ca[1]), color=BLUE, width=3.1)
    arrow(ax, tuple(cb), (14.80, cb[1]), color=ORANGE, width=3.1)
    q1, q2, my = p1[-1], p2[0], 3.52
    ax.plot([q1[0], q1[0]], [q1[1] - 0.10, my], color=MUTED, lw=1.0, ls=":")
    ax.plot([q2[0], q2[0]], [q2[1] - 0.10, my], color=MUTED, lw=1.0, ls=":")
    arrow(ax, (q1[0], my), (q2[0], my), color=RED, width=1.8, style="<->")
    ax.text(12.0, 5.80, "whole-group shift · arrows anchored at centroids", ha="center", fontsize=10.5, color=RED, fontweight="bold")
    ax.text(12.0, 3.10, "nearest-point gap activates the bounded shift", ha="center", fontsize=10.5, color=MUTED)
    ax.text(12.0, 2.30, "unsupervised macro labels · every member moves together\nwritten back to L0 only",
            ha="center", va="top", fontsize=10.3, color=GREEN, fontweight="bold", linespacing=1.4)
    save(fig, output, "f10_two_separations.png")


def fig_tokenization(output: Path) -> None:
    fig, ax = canvas()
    panel_centers = [2.7, 8.0, 13.3]
    headings = [("L0 · observed symbols", BLUE), ("L1 · coarse groups", ORANGE), ("L2 · groups of groups", GREEN)]
    local = np.asarray([[-1.05, 0.85], [0.0, 0.85], [1.05, 0.85], [-1.05, -0.85], [0.0, -0.85], [1.05, -0.85]])
    offsets = np.asarray([[-0.18, 0.0], [0.09, 0.16], [0.16, -0.13]])
    palette = [BLUE, ORANGE, GREEN, PURPLE, PINK, YELLOW]
    for pi, (px, (heading, hc)) in enumerate(zip(panel_centers, headings)):
        ax.text(px, 7.48, heading, ha="center", fontsize=15, fontweight="bold", color=hc)
        if pi >= 2:
            for ri, y in enumerate((5.35, 3.65)):
                ax.add_patch(patches.Ellipse((px, y), 3.55, 1.42, facecolor=SURFACE, edgecolor=INK if ri == 0 else MUTED,
                                             lw=2.0, ls="-" if ri == 0 else "--", zorder=0))
                ax.text(px - 2.02, y, f"L2-C{ri + 1}", ha="right", va="center", fontsize=9.2, fontweight="bold", color=MUTED)
        for gi, c in enumerate(local):
            cx, cy = px + c[0], 4.50 + c[1]
            if pi >= 1:
                ax.add_patch(patches.Circle((cx, cy), 0.48, facecolor=palette[gi], edgecolor=palette[gi], lw=1.5, alpha=0.18, zorder=1))
                ax.text(cx, cy + 0.40, f"C{gi + 1}", ha="center", fontsize=8.8, fontweight="bold", color=palette[gi])
            pts = offsets + np.asarray([cx, cy])
            dots(ax, pts[:, 0], pts[:, 1], palette[gi], 48, z=3)
        if pi < 2:
            arrow(ax, (px + 1.95, 4.50), (panel_centers[pi + 1] - 1.95, 4.50), color=MUTED, width=2.2)
    ax.text(8.0, 1.68, "Members never disappear: an L1 token stands for a set of points, an L2 token for a set of L1 sets.",
            ha="center", fontsize=12.6)
    ax.text(8.0, 1.12, "Points are repeated in place only to show membership; each upper level has its own coordinates.",
            ha="center", fontsize=11, color=MUTED)
    save(fig, output, "f11_recursive_tokenization.png")


def fig_scale_ladder(output: Path) -> None:
    fig, ax = canvas()
    levels = [("L0", 1.0, 300, 100), ("L1", 3.16, 3_000, 316), ("L2", 9.99, 30_000, 999), ("L3", 31.55, 300_000, 3_155),
              ("L4", 99.71, 3_000_000, 9_971)]
    colors = [BLUE, ORANGE, GREEN, PURPLE, PINK]
    xs = np.linspace(1.6, 14.4, len(levels))
    ax.plot([xs[0], xs[-1]], [5.2, 5.2], color=GRID, lw=3, zorder=1)
    for x, (name, r, mem, push), c in zip(xs, levels, colors):
        radius = 0.22 + 0.24 * np.log10(r) / np.log10(100)
        ax.add_patch(patches.Circle((x, 5.2), radius * 2.2, facecolor=tint(c, 0.35), edgecolor=c, lw=2, zorder=2))
        ax.text(x, 7.0, name, ha="center", fontsize=18, fontweight="bold", color=c)
        ax.text(x, 3.55, f"r = {r:g}", ha="center", fontsize=14, fontweight="bold")
        ax.text(x, 2.95, f"memory M = {mem:,}", ha="center", fontsize=11.5, color=MUTED)
        ax.text(x, 2.45, f"graph push every {push:,}", ha="center", fontsize=11.5, color=MUTED)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.text((a + b) / 2, 5.55, "×3.16", ha="center", fontsize=11, color=MUTED)
    ax.text(8.0, 1.25, r"natural length $r_L = 3.16^L$ (half a decade per level) · memory $M_L = 300\cdot 10^L$ observations of that level",
            ha="center", fontsize=12.5)
    save(fig, output, "f12_scale_ladder.png")


def fig_cadence(output: Path) -> None:
    fig, ax = canvas()
    ax.text(4.0, 7.75, "Design target: higher level, longer window", ha="center", fontsize=15, fontweight="bold", color=BLUE)
    for label, interval, color, y, count in [("L0 → L1", 50, BLUE, 6.70, 10), ("L1 → L2", 158, ORANGE, 5.60, 6),
                                             ("L2 → L3", 499, GREEN, 4.50, 4)]:
        ax.text(0.75, y, label, fontsize=12.8, fontweight="bold", color=color, va="center")
        ax.plot([2.30, 5.60], [y, y], color=GRID, lw=3, solid_capstyle="round")
        xs = np.linspace(2.30, 5.60, count)
        dots(ax, xs, [y] * count, color, 50, z=3)
        ax.text(5.85, y, f"every {interval} raw steps", fontsize=10.0, color=MUTED, va="center")
    ax.text(12.0, 7.75, "Default scheduler (nested_modulo)", ha="center", fontsize=15, fontweight="bold", color=RED)
    for label, interval, y in [("L0 → L1", "50", 6.70), ("L1 → L2", "3,950 = lcm(50, 158)", 5.60),
                               ("L2 → L3", "≈1,971,050", 4.50)]:
        ax.text(9.05, y, label, fontsize=12.8, fontweight="bold", color=MUTED, va="center")
        ax.text(10.65, y, f"first joint trigger at {interval}", fontsize=11.2, color=RED if label != "L0 → L1" else INK, va="center")
    card(ax, (1.2, 1.15), 13.6, 2.15, "Known limitation, kept as is in this release",
         "a deeper level only checks its clock when the level below fires, so its cadence aliases to the lcm;\n"
         "at 100k steps L2+ receive very few observations · the elapsed_geometric option exists but is not validated",
         color=RED, fill=RED_FILL, fontsize=13)
    save(fig, output, "f13_level_cadence.png")


def fig_macro_feedback(output: Path) -> None:
    fig, ax = canvas()
    card(ax, (1.00, 6.05), 3.20, 1.05, "stable L0 tokens", "geometry-derived", color=BLUE, fill=BLUE_FILL, fontsize=11.9)
    card(ax, (6.40, 6.05), 3.20, 1.05, "macro labels", "unsupervised grouping", color=ORANGE, fill=ORANGE_FILL, fontsize=11.9)
    card(ax, (11.80, 6.05), 3.20, 1.05, "bounded L0 feedback", "two effects below", color=GREEN, fill=GREEN_FILL, fontsize=11.9)
    arrow(ax, (4.23, 6.57), (6.35, 6.57), color=MUTED, width=2.4)
    arrow(ax, (9.63, 6.57), (11.75, 6.57), color=MUTED, width=2.4)
    ca, cb = np.asarray([5.45, 3.70]), np.asarray([10.55, 3.70])
    ax.add_patch(patches.Ellipse(tuple(ca), 2.55, 1.85, facecolor=BLUE_FILL, edgecolor=BLUE, lw=1.8))
    ax.add_patch(patches.Ellipse(tuple(cb), 2.55, 1.85, facecolor=ORANGE_FILL, edgecolor=ORANGE, lw=1.8))
    offsets = np.asarray([[-0.78, 0.0], [-0.42, 0.50], [-0.42, -0.50], [0.0, 0.0], [0.42, 0.50], [0.42, -0.50], [0.78, 0.0]])
    p1, p2 = offsets + ca, offsets + cb
    dots(ax, p1[:, 0], p1[:, 1], BLUE, 78)
    dots(ax, p2[:, 0], p2[:, 1], ORANGE, 78)
    ra, rb = p1[4], p2[1]
    ax.plot([ra[0], rb[0]], [ra[1], rb[1]], color=MUTED, lw=3.6, ls="--", alpha=0.45, zorder=2)
    arrow(ax, tuple(ra), (ra[0] + 0.72, ra[1]), color=MUTED, width=2.2)
    arrow(ax, tuple(rb), (rb[0] - 0.72, rb[1]), color=MUTED, width=2.2)
    ax.text(8.0, 5.05, "1  the cross-macro relation stays, but its attraction is scaled by 0.001", ha="center", fontsize=10.6,
            color=MUTED, fontweight="bold")
    ax.scatter([ca[0], cb[0]], [ca[1], cb[1]], s=82, marker="x", color=RED, linewidth=2.2, zorder=6)
    ax.plot([ca[0], cb[0]], [ca[1], cb[1]], color=RED, lw=1.8, ls="--", alpha=0.72, zorder=2)
    arrow(ax, tuple(ca), (3.85, ca[1]), color=BLUE, width=3.1)
    arrow(ax, tuple(cb), (12.15, cb[1]), color=ORANGE, width=3.1)
    ax.text(8.0, 2.94, "whole-group shift", ha="center", fontsize=10.1, color=RED, fontweight="bold")
    q1, q2, my = p1[-1], p2[0], 2.22
    ax.plot([q1[0], q1[0]], [q1[1] - 0.08, my], color=MUTED, lw=1.0, ls=":")
    ax.plot([q2[0], q2[0]], [q2[1] - 0.08, my], color=MUTED, lw=1.0, ls=":")
    arrow(ax, (q1[0], my), (q2[0], my), color=RED, width=1.8, style="<->")
    ax.text(8.0, 1.70, "2  a nearest-point gap below the target spacing triggers a bounded translation of both groups",
            ha="center", fontsize=10.6)
    ax.text(8.0, 1.05, "macro labels come from RG's own tokenization; upper coordinates are never copied down",
            ha="center", fontsize=10.6, color=MUTED)
    save(fig, output, "f14_macro_feedback.png")


def fig_readout(output: Path) -> None:
    fig, ax = canvas()
    ax.plot([8.0, 8.0], [1.05, 8.1], color=GRID, lw=1.5)
    ax.text(4.0, 7.70, "Formal readout (all five models)", ha="center", fontsize=16, fontweight="bold", color=BLUE)
    card(ax, (1.05, 6.35), 5.90, 0.84, "final L0 coordinates", "the same reader for every model", color=BLUE, fill=BLUE_FILL, fontsize=11.5)
    arrow(ax, (4.0, 6.30), (4.0, 5.90), color=MUTED, width=2.0)
    card(ax, (1.05, 5.00), 5.90, 0.84, "flat: DBSCAN · hierarchy: Euclidean MST", "sort the N − 1 edges, cut at large gaps",
         color=BLUE, fill=SURFACE, fontsize=11.2)
    arrow(ax, (4.0, 4.94), (4.0, 4.54), color=MUTED, width=2.0)
    lengths = np.asarray([0.34, 0.38, 0.41, 0.45, 0.47, 1.26, 1.33])
    for x, v in zip(np.linspace(1.45, 6.55, lengths.size), lengths):
        ax.add_patch(patches.Rectangle((x - 0.22, 2.90), 0.44, 0.96 * v, facecolor=RED if v > 1.0 else BLUE, edgecolor="none"))
    ax.plot([1.15, 6.85], [3.76, 3.76], color=RED, lw=1.4, ls="--")
    ax.text(4.0, 4.30, "adjacent edge ratio ≥ 1.05 → candidate cuts", ha="center", fontsize=11.0, color=RED, fontweight="bold")
    ax.text(4.0, 2.40, "one cut per required level · labels used only for scoring", ha="center", fontsize=10.5, color=MUTED)
    ax.text(4.0, 1.85, "score = mean ARI over required levels (missing level = 0)", ha="center", fontsize=10.5)
    ax.text(4.0, 1.30, "supplementary runs also average the last 5 checkpoints", ha="center", fontsize=10.5, color=GREEN)

    ax.text(12.0, 7.70, "Candidate protocol · not adopted", ha="center", fontsize=16, fontweight="bold", color=ORANGE)
    ax.text(12.0, 7.05, "persistent cuts over a late window", ha="center", fontsize=11.2, color=MUTED)
    tx = np.linspace(9.15, 14.85, 7)
    ax.plot([tx[0], tx[-1]], [6.62, 6.62], color=GRID, lw=3)
    dots(ax, tx, [6.62] * len(tx), ORANGE, 55, z=3)
    arrow(ax, (12.0, 6.30), (12.0, 5.90), color=MUTED, width=2.0)
    card(ax, (8.65, 5.00), 6.70, 0.90, "each checkpoint: split sorted MST edges at k",
         r"$L_k=\{e_1...e_k\}$  ·  $H_k=\{e_{k+1}...e_m\}$", color=ORANGE, fill=ORANGE_FILL, fontsize=10.8)
    ax.text(12.0, 4.52, r"$D_k=2(\mu_H-\mu_L)/(\mu_H+\mu_L)$", ha="center", fontsize=13.0, color=ORANGE, fontweight="bold")
    ax.text(12.0, 4.13, r"$D_k>0.4 \Leftrightarrow \mu_H/\mu_L>1.5$ (an untested slack, not derived from 3.16)", ha="center", fontsize=9.9, color=MUTED)
    card(ax, (8.65, 2.98), 3.10, 0.82, "geometry gate", "overlap / eps margin", color=GREEN, fill=GREEN_FILL, fontsize=10.5)
    card(ax, (12.25, 2.98), 3.10, 0.82, "time gate", "support / churn / mean ± sd", color=PURPLE, fill=PURPLE_FILL, fontsize=10.5)
    ax.text(12.0, 2.40, "keep several persistent nested cuts, not only the largest gap", ha="center", fontsize=10.3, fontweight="bold")
    save(fig, output, "f15_readout.png")


def fig_ari_nmi(output: Path) -> None:
    fig, ax = canvas()
    truth = np.repeat(np.arange(4), 6)
    over = np.repeat(np.arange(8), 3)
    merged = np.repeat([0, 0, 1, 1], 6)
    cases = [("exact", truth), ("every chunk split in two", over), ("pairs of chunks merged", merged)]
    xs = [3.0, 8.0, 13.0]
    for x, (name, pred) in zip(xs, cases):
        ari = adjusted_rand_score(truth, pred)
        nmi = normalized_mutual_info_score(truth, pred)
        ax.text(x, 7.65, name, ha="center", fontsize=14.5, fontweight="bold")
        for i, (lab, val, col) in enumerate([("ARI", ari, BLUE), ("NMI", nmi, ORANGE)]):
            bx = x - 1.0 + i * 1.3
            ax.add_patch(patches.Rectangle((bx - 0.38, 2.6), 0.76, 4.2 * val, facecolor=col, edgecolor="none"))
            ax.text(bx, 2.6 + 4.2 * val + 0.18, f"{val:.2f}", ha="center", fontsize=13, fontweight="bold")
            ax.text(bx, 2.2, lab, ha="center", fontsize=12, color=col, fontweight="bold")
        ax.plot([x - 1.7, x + 1.5], [2.6, 2.6], color=GRID, lw=1.5)
        for j, lab in enumerate(pred):
            ax.add_patch(patches.Rectangle((x - 1.7 + j * 0.133, 1.35), 0.12, 0.4,
                                           facecolor=[BLUE, ORANGE, GREEN, PURPLE, PINK, YELLOW, RED, GRAY][lab % 8], edgecolor="none"))
    ax.text(8.0, 0.75, "24 symbols in 4 true chunks; strips show predicted labels · values computed by sklearn",
            ha="center", fontsize=11, color=MUTED)
    save(fig, output, "f16_ari_vs_nmi.png")


FIGURES = [fig_cover, fig_syncmap, fig_heatmaps, fig_lineage, fig_kernel, fig_observation, fig_relation_graph,
           fig_primitive, fig_spring, fig_separations, fig_tokenization, fig_scale_ladder, fig_cadence,
           fig_macro_feedback, fig_readout, fig_ari_nmi]


def bundle(deck: Path, target: Path) -> None:
    """Write a single-file copy of the deck with every image inlined (the release asset)."""
    import base64
    import re

    html = deck.read_text(encoding="utf-8")

    def inline(match: re.Match[str]) -> str:
        path = (deck.parent / match.group(1)).resolve()
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    target.write_text(re.sub(r'src="([^"]+\.png)"', inline, html), encoding="utf-8")
    print("wrote", target, f"({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path, default=None,
                        help="instead of drawing, write docs/presentation/index.html as one self-contained HTML file here")
    args = parser.parse_args()
    if args.bundle is not None:
        bundle(ROOT / "docs" / "presentation" / "index.html", args.bundle.resolve())
        return
    for fig in FIGURES:
        fig(args.output.resolve())


if __name__ == "__main__":
    main()
