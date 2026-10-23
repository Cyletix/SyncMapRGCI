#!/usr/bin/env python3
"""Representative benchmark: eight scenarios x five models, with figures and a report.

    python tools/benchmark.py run      # experiments, then the report
    python tools/benchmark.py report   # figures + results/benchmark/REPORT.md from the saved runs

`run` drives main.py only, so every number can be reproduced with the public CLI:
- showcase: one seed-0 run per (dataset, model) with a score trace, final coordinates and a
  trajectory GIF;
- comparison: task 2 with five seeds per (dataset, model).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark"

# Scenarios used repeatedly as showcase/diagnostic cases in the research notes.
DATASETS: list[tuple[str, str]] = [
    ("fixed5_6", "datasets/fixed5_6.dot"),
    ("UB_mixed_chunk_20_5_5", "datasets/UB_mixed_chunk_20_5_5.dot"),
    ("imbalanced_fixed_300", "datasets/imbalanced_fixed_300.dot"),
    ("chain_mixed5_60", "datasets/chain_mixed5_60.dot"),
    ("DCH1", "datasets/hierarchy/DCH1_fixed_chunk_6_3.dot"),
    ("IH", "datasets/hierarchy/IH_fixed_chunk_unbalanced.dot"),
    ("EC", "datasets/hierarchy/EC_fixed_chunk_15_3.dot"),
    ("IEH", "datasets/hierarchy/IEH_fixed_chunk_deep_hierarchy.dot"),
]
PROFILES = ["syncmap", "symmetric", "tsf", "nsgd", "rg"]
GIF_PROFILES = set(PROFILES)
GIF_SIZE_IN, GIF_DPI, GIF_MAX_FRAMES = 4.0, 100, 48
DISPLAY = {"syncmap": "SyncMap", "symmetric": "Symmetric", "tsf": "TSFMap", "nsgd": "NSGD", "rg": "RG"}
# Fixed categorical slots, dark steps validated on the dark surface; colour follows the model, never its rank.
COLORS = {"rg": "#3987e5", "nsgd": "#d95926", "syncmap": "#199e70", "symmetric": "#c98500", "tsf": "#d55181"}
# Classes past five also differ by marker shape, which carries identity where hues get close.
CLASS_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
CLASS_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Dark theme of tools/plot_theme.py, shared with the repository's own PNG/GIF exports.
FIG_BG, SURFACE, INK, INK2, GRID = "#0d0f14", "#11131c", "#f2f2f2", "#b0b5c7", "#2f3a56"
_PLOT_LOCK = threading.Lock()


VIEW_ELEV, VIEW_AZIM = 22, -58


def _unit_cube(points: Any) -> Any:
    """Centre a 3-D layout and scale it into [-1, 1]^3 without changing its shape (the models renormalise space)."""
    import numpy as np

    p = np.asarray(points, dtype=float)
    p = p - p.mean(axis=0)
    return p / (np.abs(p).max() or 1.0)


def _style3d(ax: Any) -> None:
    import numpy as np
    from matplotlib.colors import to_rgba

    ax.set_facecolor(SURFACE)
    ticks = np.linspace(-1, 1, 5)
    for axis, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks), (ax.zaxis, ax.set_zticks)):
        setter(ticks)
        axis.set_ticklabels([])
        axis.set_pane_color(to_rgba(SURFACE))
        axis.line.set_color(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
    ax.tick_params(length=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def _render_gif(path: Path, title: str, trajectory: Any, steps: list[int], labels: list[int]) -> None:
    """Training trajectory in the model's own 3-D space, coloured by true chunk; the camera turns slowly to show depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    traj = np.asarray(trajectory, dtype=float)
    pick = np.unique(np.linspace(0, len(traj) - 1, min(len(traj), GIF_MAX_FRAMES)).round().astype(int))
    frames, steps = [_unit_cube(traj[i]) for i in pick], [int(steps[i]) for i in pick]

    classes = sorted(set(labels))
    members = [[i for i, c in enumerate(labels) if c == cls] for cls in classes]
    fig = plt.figure(figsize=(GIF_SIZE_IN, GIF_SIZE_IN), facecolor=FIG_BG)
    ax = fig.add_subplot(projection="3d")
    _style3d(ax)
    size = 14 if len(labels) > 60 else 40
    dots = [ax.scatter([], [], [], s=size, marker=_marker(r), color=CLASS_COLORS[r % 8], edgecolors=SURFACE,
                       linewidths=0.5, depthshade=False) for r in range(len(classes))]
    ax.set_title(title, color=INK, fontsize=11, pad=0)
    clock = fig.text(0.96, 0.04, "", ha="right", va="bottom", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    hold = 8  # pause on the final layout before looping
    total = len(frames) + hold

    def draw(k: int) -> list[Any]:
        i = min(k, len(frames) - 1)
        for dot, idx in zip(dots, members):
            dot._offsets3d = tuple(frames[i][idx].T)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM + 120 * k / total)
        clock.set_text(f"step {steps[i]:,}")
        return [*dots, clock]

    FuncAnimation(fig, draw, frames=total, blit=False).save(path, writer=PillowWriter(fps=10), dpi=GIF_DPI)
    plt.close(fig)


# ---------------------------------------------------------------- run

def _main_py(args: list[str], log: Path) -> str:
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"main.py {' '.join(args)} failed ({proc.returncode}); see {log}")
    return proc.stdout


def _showcase(name: str, dataset: str, profile: str, steps: int) -> None:
    run_dir = OUT / "runs"
    log = OUT / "logs" / f"{name}__{profile}.log"
    args = [
        "--task", "0", "--profile", profile, "--dataset", dataset, "--steps", str(steps), "--seed", "0",
        "--record-interval", str(steps // 100), "--trace", "true", "--trace-record-interval", str(steps // 50),
        "--save-json", "true", "--save-png", "false", "--save-plot", "false", "--save-gif", "false",
    ]
    stdout = _main_py(args, log)
    outputs = dict(re.findall(r"(saved_\w+)=(\S+)", stdout.split("outputs:")[-1]))
    snapshot_path = Path(outputs["saved_snapshot"])
    stem = snapshot_path.name.split("_snapshot_")[0]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    record = json.loads(Path(outputs["saved_json"]).read_text(encoding="utf-8"))
    trace = json.loads((snapshot_path.parent / f"{stem}_score_{steps}.json").read_text(encoding="utf-8"))

    levels = sorted((k for k in record if re.fullmatch(r"cluster_level\d+_ari_score", k)), key=lambda k: int(re.findall(r"\d+", k)[0]))
    compact = {
        "dataset": name, "dataset_path": dataset, "profile": profile, "steps": steps, "seed": 0,
        "ari": record.get("ari_score"), "nmi": record.get("nmi_score"),
        "level_ari": {k: record[k] for k in levels}, "elapsed_time": record.get("elapsed_time"),
        "positions": [node["position"] for node in snapshot["nodes"]],
        # Flat graphs have a single level, stored only as the display labels.
        "true_label_levels": snapshot["true_label_levels"] or {"level_0": snapshot["true_labels_display"]},
        "predicted_labels": snapshot["predicted_labels_display"],
        "curve": [{"step": int(p["iteration"]), "ari": p.get("ari"), "nmi": p.get("nmi")} for p in trace],
    }
    if profile in GIF_PROFILES:
        with _PLOT_LOCK:  # pyplot is not thread-safe and showcase runs share this process
                _render_gif(OUT / "gifs" / f"{name}__{profile}.gif", f"{DISPLAY[profile]} · {name}", snapshot["trajectory"],
                        snapshot["trajectory_steps"], compact["true_label_levels"]["level_0"])
    (run_dir / f"{name}__{profile}.json").write_text(json.dumps(compact), encoding="utf-8")
    print(f"showcase {name:18s} {profile:10s} ARI={compact['ari']:.3f}", flush=True)


def _comparison(name: str, dataset: str, steps: int, trials: int) -> None:
    log = OUT / "logs" / f"{name}__comparison.log"
    args = [
        "--task", "2", "--profiles", *PROFILES, "--datasets", dataset, "--comparison-sequence-lengths", str(steps),
        "--comparison-trials", str(trials), "--metric", "ari", "--seed", "0", "--resume-mode", "off",
    ]
    stdout = _main_py(args, log)
    archive = Path(re.findall(r"Results saved to: (\S+)", stdout)[-1])
    shutil.copyfile(archive / "comparison_full_results.csv", OUT / "comparison" / f"{name}.csv")
    print(f"comparison {name} done ({archive.name})", flush=True)


def run(steps: int, trials: int, workers: int) -> None:
    import pandas as pd

    for sub in ("runs", "gifs", "comparison", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    merged_path = OUT / "comparison_full_results.csv"
    merged = pd.read_csv(merged_path) if merged_path.exists() else None
    merged_files = set(merged["dataset"]) if merged is not None else set()
    jobs: list[tuple[Any, ...]] = []
    # Finished pieces are kept, so an interrupted run resumes where it stopped.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Task 2 archives by start second, so comparison launches are staggered.
        for name, dataset in DATASETS:
            if Path(dataset).name not in merged_files and not (OUT / "comparison" / f"{name}.csv").exists():
                jobs.append(pool.submit(_comparison, name, dataset, steps, trials))
                time.sleep(3)
        for name, dataset in DATASETS:
            for profile in PROFILES:
                gif = OUT / "gifs" / f"{name}__{profile}.gif"
                if (OUT / "runs" / f"{name}__{profile}.json").exists() and gif.exists() and gif.stat().st_size > 0:
                    continue
                jobs.append(pool.submit(_showcase, name, dataset, profile, steps))
        for job in jobs:
            job.result()

    frames = [pd.read_csv(p) for p in sorted((OUT / "comparison").glob("*.csv"))]
    if frames:
        new = pd.concat(frames, ignore_index=True)
        new = new.drop(columns=[c for c in ("learning_curve", "predicted_labels", "true_labels") if c in new.columns])
        if merged is not None:
            new = pd.concat([merged[~merged["dataset"].isin(set(new["dataset"]))], new], ignore_index=True)
        new.to_csv(merged_path, index=False)
    shutil.rmtree(OUT / "comparison")
    shutil.rmtree(OUT / "logs")


# ---------------------------------------------------------------- report

def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _marker(rank: int) -> str:
    # Past eight classes the colours repeat, so the marker shifts to keep every colour/marker pair unique.
    return CLASS_MARKERS[(rank + rank // 8) % 8]


def _pca2(points: Any) -> Any:
    import numpy as np

    x = np.asarray(points, dtype=float)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _load_runs() -> dict[tuple[str, str], dict[str, Any]]:
    runs = {}
    for path in (OUT / "runs").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[(data["dataset"], data["profile"])] = data
    return runs


def _summary() -> Any:
    import pandas as pd

    df = pd.read_csv(OUT / "comparison_full_results.csv")
    df["name"] = df["dataset"].map({Path(p).name: n for n, p in DATASETS})
    return df


def _fig_comparison(df: Any) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [n for n, _ in DATASETS]
    fig, ax = plt.subplots(figsize=(max(11.0, 1.75 * len(names)), 4.6), facecolor=FIG_BG)
    _style(ax)
    width = 0.16
    for i, profile in enumerate(PROFILES):
        sub = df[df["profile_name"] == profile].groupby("name")["ari_score"]
        mean, std = sub.mean().reindex(names), sub.std().reindex(names)
        xs = np.arange(len(names)) + (i - 2) * (width + 0.01)
        ax.bar(xs, mean.values, width, color=COLORS[profile], label=DISPLAY[profile], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, mean.values, yerr=std.values, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    def wrap(label: str) -> str:
        # Break long names at the underscore nearest the middle so neighbouring ticks do not collide.
        if len(label) <= 14:
            return label
        cuts = [i for i, ch in enumerate(label) if ch == "_"]
        cut = min(cuts, key=lambda i: abs(i - len(label) / 2))
        return label[:cut] + "\n" + label[cut:]

    ax.set_xticks(np.arange(len(names)), [wrap(n) for n in names], color=INK, fontsize=10)
    ax.set_ylabel("ARI (mean ± std, 5 seeds)", color=INK, fontsize=10)
    ax.set_ylim(min(0, float(df["ari_score"].min()) - 0.05), 1.05)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "comparison_ari.png", dpi=150, facecolor=FIG_BG)
    plt.close(fig)


def _fig_latent(name: str, runs: dict[tuple[str, str], dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(3.4 * len(PROFILES), 4.0), facecolor=FIG_BG)
    n_classes = 0
    for k, profile in enumerate(PROFILES):
        run = runs[(name, profile)]
        xyz = _unit_cube(run["positions"])  # the model's own 3-D space, not a projection
        labels = run["true_label_levels"]["level_0"]
        classes = sorted(set(labels))  # raw ids need not start at 0
        n_classes = len(classes)
        ax = fig.add_subplot(1, len(PROFILES), k + 1, projection="3d")
        _style3d(ax)
        for rank, cls in enumerate(classes):
            idx = [i for i, c in enumerate(labels) if c == cls]
            ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], s=14 if len(labels) > 60 else 36, marker=_marker(rank),
                       color=CLASS_COLORS[rank % 8], edgecolors=SURFACE, linewidths=0.5, depthshade=False)
        ax.set_title(f"{DISPLAY[profile]}   ARI {run['ari']:.2f}", color=INK, fontsize=11, pad=0)
    handles = [Line2D([], [], linestyle="none", marker=_marker(r), color=CLASS_COLORS[r % 8], markersize=7, label=f"chunk {r + 1}") for r in range(n_classes)]
    rows = -(-n_classes // 10)
    fig.legend(handles=handles, ncol=min(n_classes, 10), frameon=False, loc="lower center", fontsize=9, labelcolor=INK2)
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.06 * rows + 0.02, wspace=0)
    fig.savefig(OUT / "figures" / f"latent_{name}.png", dpi=150, facecolor=FIG_BG)
    plt.close(fig)


def _fig_curves(name: str, runs: dict[tuple[str, str], dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=FIG_BG)
    _style(ax)
    for profile in PROFILES:
        curve = runs[(name, profile)]["curve"]
        steps = [p["step"] for p in curve]
        ari = [p["ari"] if p["ari"] is not None else float("nan") for p in curve]
        ax.plot(steps, ari, color=COLORS[profile], linewidth=2, label=DISPLAY[profile])
    ax.set_xlabel("input steps", color=INK, fontsize=10)
    ax.set_ylabel("ARI (seed 0)", color=INK, fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(0, max(p["step"] for p in runs[(name, PROFILES[0])]["curve"]))
    ax.legend(ncol=5, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"curves_{name}.png", dpi=150, facecolor=FIG_BG)
    plt.close(fig)


def _hierarchy_tree(dot_path: Path) -> tuple[dict[str, tuple[float, int]], list[tuple[str, str]]]:
    """Lay out a hierarchy .dot (leaf symbols -> chunk nodes -> ... -> root) as a top-down tree."""
    edges = re.findall(r"^\s*(\S+)\s*->\s*(\S+)", dot_path.read_text(encoding="utf-8"), flags=re.M)
    children: dict[str, list[str]] = {}
    for child, parent in edges:
        children.setdefault(parent, []).append(child)
    leaves = sorted({c for c, _ in edges if c.isdigit()}, key=int)
    pos: dict[str, tuple[float, int]] = {leaf: (float(i), 0) for i, leaf in enumerate(leaves)}

    def place(node: str) -> tuple[float, int]:
        if node not in pos:
            kids = [place(k) for k in children[node]]
            pos[node] = (sum(x for x, _ in kids) / len(kids), max(d for _, d in kids) + 1)
        return pos[node]

    place("root")
    return pos, edges


def _padded_hull(ax: Any, xy: Any, pad: float, color: str, alpha: float, linestyle: str) -> None:
    import numpy as np
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    pts = np.asarray(xy, dtype=float)
    # Hull of a disc around every point gives a rounded, padded region.
    ring = np.concatenate([pts + pad * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    hull = ring[ConvexHull(ring).vertices]
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=alpha, edgecolor="none"))
    ax.add_patch(Polygon(hull, closed=True, facecolor="none", edgecolor=color, linewidth=1.2, linestyle=linestyle, alpha=0.9))


def _fig_hero(runs: dict[tuple[str, str], dict[str, Any]], name: str = "DCH1") -> None:
    """Generative hierarchy next to the map RG learns from the walk alone (seed 0)."""
    import matplotlib.pyplot as plt
    import numpy as np

    path = ROOT / dict(DATASETS)[name]
    run = runs[(name, "rg")]
    fine = run["true_label_levels"]["level_0"]
    coarse = run["true_label_levels"]["level_1"]
    fine_ids, coarse_ids = sorted(set(fine)), sorted(set(coarse))

    fig = plt.figure(figsize=(14, 6.2), facecolor=FIG_BG)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1.15], wspace=0.08, hspace=0.32)
    left = fig.add_subplot(grid[:, 0])
    right = fig.add_subplot(grid[0, 1:])
    zooms = [fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
    for ax in (left, right, *zooms):
        ax.set_facecolor(FIG_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    pos, edges = _hierarchy_tree(path)
    for child, parent in edges:
        (x0, y0), (x1, y1) = pos[child], pos[parent]
        left.plot([x0, x1], [y0, y1], color=GRID, linewidth=1.4, zorder=1)
    for node, (x, y) in pos.items():
        if node.isdigit():
            rank = fine_ids.index(fine[int(node)])
            left.scatter([x], [y], s=90, marker=CLASS_MARKERS[rank % 8], color=CLASS_COLORS[rank % 8], edgecolors=FIG_BG, linewidths=1, zorder=3)
        else:
            left.scatter([x], [y], s=60 if node != "root" else 90, color=INK2, zorder=3)
    top = max(y for _, y in pos.values())
    for depth, text in [(0, "symbols"), (1, "chunks"), (2, "groups"), (top, "root")]:
        if depth <= top:
            left.text(-1.6, depth, text, ha="right", va="center", fontsize=10, color=INK2)
    left.set_xlim(-4.2, len(fine) + 0.3)
    left.set_ylim(-0.6, top + 0.6)
    left.set_title(f"Hidden structure ({name}): {len(fine)} symbols, {len(fine_ids)} chunks, {len(coarse_ids)} groups",
                   color=INK, fontsize=12, pad=12)

    def draw_points(ax: Any, xy: Any, members: list[int]) -> None:
        for rank, cls in enumerate(fine_ids):
            idx = [i for i in members if fine[i] == cls]
            if idx:
                ax.scatter(xy[idx, 0], xy[idx, 1], s=80, marker=CLASS_MARKERS[rank % 8], color=CLASS_COLORS[rank % 8],
                           edgecolors=FIG_BG, linewidths=1, zorder=3)

    xy = _pca2(run["positions"])
    span = float(np.ptp(xy, axis=0).max())
    for g_rank, g in enumerate(coarse_ids):
        idx = [i for i, c in enumerate(coarse) if c == g]
        _padded_hull(right, xy[idx], 0.06 * span, INK2, 0.08, "--")
        right.text(xy[idx, 0].mean(), xy[idx, 1].min() - 0.1 * span, f"group {g_rank + 1}", ha="center", va="top",
                   fontsize=10, color=INK2)
    draw_points(right, xy, list(range(len(fine))))
    right.margins(0.1, 0.35)
    right.set_aspect("equal", adjustable="datalim")
    right.set_title(f"RG map after {run['steps']:,} steps, learned from the walk alone", color=INK, fontsize=12, pad=10)

    # Each group again at its own scale: the chunks inside it form the next, smaller clusters.
    for g_rank, (ax, g) in enumerate(zip(zooms, coarse_ids)):
        idx = [i for i, c in enumerate(coarse) if c == g]
        local = xy[idx]
        local_span = float(np.ptp(local, axis=0).max()) or 1.0
        for rank, cls in enumerate(fine_ids):
            members = [i for i in idx if fine[i] == cls]
            if members:
                _padded_hull(ax, xy[members], 0.07 * local_span, CLASS_COLORS[rank % 8], 0.12, "-")
        draw_points(ax, xy, idx)
        ax.margins(0.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"group {g_rank + 1}, zoomed in", color=INK2, fontsize=10, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)

    levels = " · ".join(f"level {int(k.split('level')[1].split('_')[0])} ARI {v:.2f}" for k, v in run["level_ari"].items())
    fig.text(0.69, 0.02, f"2-D PCA projection of the 3-D map; colour / marker = true chunk, dashed region = true group   (seed 0: {levels})",
             ha="center", fontsize=9, color=INK2)
    fig.savefig(OUT / "figures" / f"hero_rg_{name}.png", dpi=150, facecolor=FIG_BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    runs = _load_runs()
    df = _summary()
    _fig_comparison(df)
    if ("DCH1", "rg") in runs:
        _fig_hero(runs)
    for name, _ in DATASETS:
        _fig_latent(name, runs)
        _fig_curves(name, runs)

    steps = int(df["sequence_length"].iloc[0])
    trials = int(df.groupby(["name", "profile_name"]).size().max())
    lines = [
        "# Benchmark report", "",
        f"{len(DATASETS)} representative graphs, {len(PROFILES)} models, {steps:,} input steps, {trials} seeds (0-{trials - 1}). "
        "Generated by `tools/benchmark.py`; every run goes through `main.py`.", "",
        "## Setup", "",
        "| Dataset | File | Family | Nodes | Classes per level (fine to coarse) |", "|---|---|---|---:|---|",
    ]
    for name, path in DATASETS:
        levels = runs[(name, "rg")]["true_label_levels"]
        counts = " / ".join(str(len(set(levels[k]))) for k in sorted(levels, key=lambda k: int(k.split("_")[1])))
        family = "hierarchy" if "/hierarchy/" in path else "flat"
        lines.append(f"| {name} | `{path}` | {family} | {len(levels['level_0'])} | {counts} |")
    lines += [
        "", "- Models: " + ", ".join(f"{DISPLAY[p]} (`{p}`)" for p in PROFILES) + ", each with its maintained profile.",
        "- Score: ARI against the true chunks; on hierarchy graphs the mean ARI over every required level (a missing level scores 0). NMI is reported alongside.",
        "- Comparison: `python main.py --task 2 --profiles " + " ".join(PROFILES) + f" --datasets <file> --comparison-sequence-lengths {steps} --comparison-trials {trials} --metric ari --seed 0`.",
        f"- Showcase (figures, curves, GIFs): `python main.py --task 0 --profile <model> --dataset <file> --steps {steps} --seed 0 --trace true`.",
        "", "## Results", "", "![ARI by dataset and model](figures/comparison_ari.png)", "",
        "ARI, mean ± std over seeds:", "", "| Dataset | " + " | ".join(DISPLAY[p] for p in PROFILES) + " |",
        "|---|" + "---:|" * len(PROFILES),
    ]
    stats = df.groupby(["name", "profile_name"]).agg(ari_m=("ari_score", "mean"), ari_s=("ari_score", "std"),
                                                     nmi_m=("nmi_score", "mean"), nmi_s=("nmi_score", "std"),
                                                     time=("elapsed_time", "mean"))
    for name, _ in DATASETS:
        row = [stats.loc[(name, p)] for p in PROFILES]
        best = max(r.ari_m for r in row)
        cells = [(f"**{_fmt(r.ari_m, r.ari_s)}**" if r.ari_m == best else _fmt(r.ari_m, r.ari_s)) for r in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "NMI, mean ± std over seeds:", "", "| Dataset | " + " | ".join(DISPLAY[p] for p in PROFILES) + " |",
              "|---|" + "---:|" * len(PROFILES)]
    for name, _ in DATASETS:
        lines.append(f"| {name} | " + " | ".join(_fmt(stats.loc[(name, p)].nmi_m, stats.loc[(name, p)].nmi_s) for p in PROFILES) + " |")
    lines += ["", "Mean wall-clock seconds per run:", "", "| Dataset | " + " | ".join(DISPLAY[p] for p in PROFILES) + " |",
              "|---|" + "---:|" * len(PROFILES)]
    for name, _ in DATASETS:
        lines.append(f"| {name} | " + " | ".join(f"{stats.loc[(name, p)].time:.1f}" for p in PROFILES) + " |")

    level_cols = sorted((c for c in df.columns if re.fullmatch(r"cluster_level\d+_ari_score", c)), key=lambda c: int(re.findall(r"\d+", c)[0]))
    for name, path in DATASETS:
        lines += ["", f"## {name}", ""]
        if "/hierarchy/" in path:
            sub = df[df["name"] == name]
            used = [c for c in level_cols if sub[c].notna().any()]
            lines += ["Per-level ARI, mean over seeds (level 1 = finest):", "",
                      "| Model | " + " | ".join(f"level {re.findall(r'[0-9]+', c)[0]}" for c in used) + " |",
                      "|---|" + "---:|" * len(used)]
            for p in PROFILES:
                means = sub[sub["profile_name"] == p][used].mean()
                lines.append(f"| {DISPLAY[p]} | " + " | ".join(f"{v:.3f}" for v in means) + " |")
            lines.append("")
        lines += [
            f"Final layout at {steps:,} steps (seed 0, each model in its own 3-D space, coloured by true finest chunk):", "",
            f"![latent space, {name}](figures/latent_{name}.png)", "",
            "ARI during training (seed 0):", "", f"![ARI curves, {name}](figures/curves_{name}.png)", "",
            "Training trajectories in 3-D (seed 0, coloured by true chunk, slowly rotating camera): "
            + ", ".join(f"[{DISPLAY[p]}](gifs/{name}__{p}.gif)" for p in PROFILES if p in GIF_PROFILES), "",
            "| " + " | ".join(DISPLAY[p] for p in PROFILES if p in GIF_PROFILES) + " |",
            "|" + "---|" * len(GIF_PROFILES),
            "| " + " | ".join(f"![{DISPLAY[p]} {name}](gifs/{name}__{p}.gif)" for p in PROFILES if p in GIF_PROFILES) + " |",
        ]
    lines += ["", "## Files", "",
              "- `comparison_full_results.csv`: every comparison run with its scores and effective parameters.",
              "- `runs/<dataset>__<model>.json`: seed-0 showcase run: scores, final coordinates, true and predicted labels, ARI/NMI curve.",
              "- `figures/`, `gifs/`: the images above.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--trials", type=int, default=5)
    # Each task-2 process keeps every run of its dataset in memory (several GB on 300-node graphs).
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "run":
        run(args.steps, args.trials, args.workers)
    report()


if __name__ == "__main__":
    main()
