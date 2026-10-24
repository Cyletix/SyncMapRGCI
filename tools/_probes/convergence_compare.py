"""Multi-strategy convergence-curve comparison.

For each dataset, train every requested strategy on the SAME input sequence
(shared seed) and overlay their learning curves — the formal mean-over-levels
ARI sampled over training steps. One overlay PNG per dataset plus a combined
grid, so a few representative datasets can be compared at a glance.

Scores come from the canonical ``run_single_test`` path (same readout, same
``apply_hierarchy_total_scores`` level-mean aggregation as the unified
comparison), so the curves match reported numbers.

Usage:
    python tools/conv.py steps \
        --datasets datasets/hierarchy/IH_fixed_chunk_unbalanced.dot ... \
        --strategies syncmap symmetric rg tsf \
        --steps 100000 --interval 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.plot_theme import apply_basic_plot_theme, resolve_plot_palette  # noqa: E402
from unified_experiments import run_single_test  # noqa: E402
from main import _resolve_runtime_selection, _resolved_main_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

_MAIN_CONFIG: dict[str, Any] | None = None


def _config() -> dict[str, Any]:
    global _MAIN_CONFIG
    if _MAIN_CONFIG is None:
        _MAIN_CONFIG = _resolved_main_config()
    return _MAIN_CONFIG

# Stable strategy -> colour mapping so a strategy keeps its colour across every
# dataset panel.
_STRATEGY_COLORS = {
    "syncmap": "#4c78a8",
    "symmetric": "#f58518",
    "rg": "#54a24b",
    "tsf": "#e45756",
    "ns": "#72b7b2",
    "nsgd": "#b279a2",
    "ci": "#ff9da6",
}
_FALLBACK_CYCLE = plt.cm.tab10(np.linspace(0.0, 1.0, 10))


def _color_for(strategy: str, index: int) -> Any:
    return _STRATEGY_COLORS.get(strategy.lower(), _FALLBACK_CYCLE[index % 10])


def _run_curve(
    dataset: str,
    strategy: str,
    steps: int,
    interval: int,
    seed: int,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train one strategy and return (steps, scores, elapsed_seconds).

    Params are resolved exactly like the task-1/task-2 CLI (via
    ``_resolve_runtime_selection``), so tuned ``strategy_defaults`` /
    profile presets are applied — running ``rg`` with the bare class
    defaults collapses it to ~0.
    """
    _, runtime_name, _env_mode, params = _resolve_runtime_selection(
        _config(), dataset=dataset, strategy=strategy,
        profile=None, env_mode=None, override_params={},
    )
    np.random.seed(int(seed))  # same seed across strategies -> identical sequence
    result = run_single_test(
        dataset_name=dataset,
        strategy_name=runtime_name,
        strategy_params=dict(params),
        sequence_length=int(steps),
        sample_curves=True,
        sample_interval=int(interval),
        print_results=False,
        record_trajectory=False,
        metric_name="ari" if metric_key == "ari" else "nmi",
    )
    curve = result.get("learning_curve", []) or []
    xs = np.asarray([int(float(p.get("iteration", 0))) for p in curve], dtype=int)
    ys = np.asarray([float(p.get(metric_key, 0.0)) for p in curve], dtype=float)
    return xs, ys, float(result.get("elapsed_time", 0.0))


def _weighted_mean_std(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Return duration-weighted mean/std for an irregularly sampled curve."""
    x = np.asarray(xs, dtype=float).reshape(-1)
    y = np.asarray(ys, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if y.size == 0:
        return float("nan"), float("nan")
    if y.size == 1:
        return float(y[0]), 0.0
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    if float(np.sum(dx)) <= 0.0:
        return float(np.nanmean(y)), float(np.nanstd(y))
    weights = np.empty(y.size, dtype=float)
    weights[0] = dx[0] * 0.5
    weights[-1] = dx[-1] * 0.5
    if y.size > 2:
        weights[1:-1] = (dx[:-1] + dx[1:]) * 0.5
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.nanmean(y)), float(np.nanstd(y))
    mean = float(np.sum(weights * y) / total)
    var = float(np.sum(weights * (y - mean) ** 2) / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def _curve_stats(xs: np.ndarray, ys: np.ndarray) -> dict[str, Any]:
    """Temporal stability summary for one learning curve.

    Peak/final remain visible for debugging, but comparison claims should prefer
    ``mean`` / ``std`` and late-window ``late_mean`` / ``late_std``. A single
    checkpoint can be a transient lucky readout; a sustained method needs high
    average score and low variance over a time/step window.
    """
    if ys.size == 0:
        return {"peak": float("nan"), "peak_step": 0, "final": float("nan"),
                "mean": float("nan"), "std": float("nan"),
                "late_mean": float("nan"), "late_std": float("nan"),
                "stable_score": float("nan"),
                "drop": float("nan"), "verdict": "n/a"}
    peak_idx = int(np.nanargmax(ys))
    peak = float(ys[peak_idx])
    peak_step = int(xs[peak_idx])
    final = float(ys[-1])
    mean, std = _weighted_mean_std(xs, ys)
    x_arr = np.asarray(xs, dtype=float).reshape(-1)
    y_arr = np.asarray(ys, dtype=float).reshape(-1)
    if x_arr.size > 1 and np.isfinite(x_arr).any():
        span = float(np.nanmax(x_arr) - np.nanmin(x_arr))
        cutoff = float(np.nanmax(x_arr) - 0.2 * span)
        tail_mask = x_arr >= cutoff
        if int(np.count_nonzero(tail_mask)) < 2:
            tail = max(1, int(round(y_arr.size * 0.2)))
            tail_x = x_arr[-tail:]
            tail_y = y_arr[-tail:]
        else:
            tail_x = x_arr[tail_mask]
            tail_y = y_arr[tail_mask]
    else:
        tail = max(1, int(round(y_arr.size * 0.2)))
        tail_x = x_arr[-tail:]
        tail_y = y_arr[-tail:]
    late_mean, late_std = _weighted_mean_std(tail_x, tail_y)
    stable_score = late_mean - late_std
    drop = peak - final
    if peak >= 0.3 and late_mean < 0.1:
        verdict = "COLLAPSE"
    elif (drop >= 0.15 or late_std >= 0.15) and peak >= 0.2:
        verdict = "DEGRADES"
    else:
        verdict = "stable"
    return {"peak": peak, "peak_step": peak_step, "final": final,
            "mean": mean, "std": std, "late_mean": late_mean, "late_std": late_std,
            "stable_score": stable_score, "drop": drop, "verdict": verdict}


def _plot_dataset(
    ax: Any,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    metric_key: str,
    palette: dict[str, Any],
) -> None:
    for idx, (strategy, (xs, ys)) in enumerate(curves.items()):
        color = _color_for(strategy, idx)
        st = _curve_stats(xs, ys)
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0,
                color=color,
                label=f"{strategy}  late μ{st['late_mean']:.2f}/σ{st['late_std']:.2f}")
        # Mark the peak so degradation (peak -> right edge) is visible at a glance.
        if np.isfinite(st["peak"]):
            ax.plot([st["peak_step"]], [st["peak"]], marker="*", markersize=11,
                    color=color, markeredgecolor=palette["fig_bg"], markeredgewidth=0.6,
                    linestyle="none", zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel(metric_key.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0.0)
    ax.axhline(0.0, color=palette["inactive"], linewidth=0.8, alpha=0.6)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper left",
                       labelcolor=str(palette["fg"]))
    if legend is not None:
        legend.set_title("strategy (late mean/std, ★=peak)")
        legend.get_title().set_color(str(palette["fg"]))
        legend.get_title().set_fontsize(8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+",
                        default=["syncmap", "symmetric", "rg", "tsf"])
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--interval", type=int, default=2000,
                        help="Curve-sampling interval (steps between evaluations)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="ari", choices=["ari", "nmi"])
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    parser.add_argument("--accent-color", default="#6970ac")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default results/convergence/<timestamp>)")
    cli = parser.parse_args()

    metric_key = str(cli.metric).strip().lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cli.out_dir) if cli.out_dir else REPO_ROOT / "results" / "convergence" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = resolve_plot_palette(theme=cli.theme, accent=cli.accent_color)

    all_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    json_path = out_dir / "curves.json"

    def _dump_json() -> None:
        payload = {
            "config": {"steps": cli.steps, "interval": cli.interval, "seed": cli.seed,
                       "metric": metric_key, "strategies": list(cli.strategies),
                       "datasets": list(cli.datasets)},
            "datasets": {
                name: {
                    strat: {"steps": xs.tolist(), "scores": ys.tolist(),
                            "stats": stats.get(name, {}).get(strat, {})}
                    for strat, (xs, ys) in curves.items()
                }
                for name, curves in all_curves.items()
            },
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    saved: list[Path] = []
    started = time.time()

    for dataset in cli.datasets:
        name = os.path.basename(str(dataset))
        print(f"\n=== {name} ===")
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        stats[name] = {}
        for strategy in cli.strategies:
            xs, ys, dt = _run_curve(dataset, strategy, cli.steps, cli.interval,
                                    cli.seed, metric_key)
            curves[strategy] = (xs, ys)
            st = _curve_stats(xs, ys)
            stats[name][strategy] = st
            print(f"  {strategy:10s} mean={st['mean']:+.3f}±{st['std']:.3f}  "
                  f"late={st['late_mean']:+.3f}±{st['late_std']:.3f}  "
                  f"stable={st['stable_score']:+.3f}  peak={st['peak']:+.3f}@"
                  f"{st['peak_step']:>8d}  final={st['final']:+.3f}  "
                  f"[{st['verdict']}]  ({dt:.1f}s)")
        all_curves[name] = curves
        _dump_json()  # incremental: partial results survive an interrupted long run

        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        apply_basic_plot_theme(fig, ax, theme=cli.theme, accent=cli.accent_color,
                               grid_alpha=0.22)
        _plot_dataset(ax, curves, title=name, metric_key=metric_key, palette=palette)
        fig.tight_layout()
        target = out_dir / f"convergence_{Path(name).stem}.png"
        fig.savefig(target, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(target)

    # Combined grid: one panel per dataset.
    n = len(all_curves)
    if n > 1:
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.2, rows * 4.4),
                                 squeeze=False)
        fig.patch.set_facecolor(palette["fig_bg"])
        flat = [a for row in axes for a in row]
        for ax, (name, curves) in zip(flat, all_curves.items()):
            apply_basic_plot_theme(fig, ax, theme=cli.theme, accent=cli.accent_color,
                                   grid_alpha=0.22)
            _plot_dataset(ax, curves, title=name, metric_key=metric_key, palette=palette)
        for ax in flat[n:]:
            ax.axis("off")
        fig.suptitle(f"Convergence ({metric_key.upper()}, {cli.steps} steps, "
                     f"seed {cli.seed})", color=palette["fg"], fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        grid_target = out_dir / "convergence_grid.png"
        fig.savefig(grid_target, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(grid_target)

    _dump_json()

    print("\n=== temporal stability summary (mean/std; peak/final are diagnostic only) ===")
    print(f"{'dataset':32s} {'strategy':10s} {'mean':>7s} {'std':>7s} "
          f"{'late':>7s} {'lstd':>7s} {'stable':>7s} {'peak':>7s} {'final':>7s}  verdict")
    flagged: list[str] = []
    for name, per in stats.items():
        for strat, st in per.items():
            mark = "  <<<" if st["verdict"] in {"DEGRADES", "COLLAPSE"} else ""
            print(f"{name:32s} {strat:10s} {st['mean']:+7.3f} {st['std']:+7.3f} "
                  f"{st['late_mean']:+7.3f} {st['late_std']:+7.3f} "
                  f"{st['stable_score']:+7.3f} {st['peak']:+7.3f} {st['final']:+7.3f}  "
                  f"{st['verdict']}{mark}")
            if st["verdict"] in {"DEGRADES", "COLLAPSE"}:
                flagged.append(f"{name}/{strat}: {st['verdict']} "
                               f"(late {st['late_mean']:.2f}±{st['late_std']:.2f}, "
                               f"peak {st['peak']:.2f}@{st['peak_step']}, final {st['final']:.2f})")
    print(f"\nflagged ({len(flagged)}):")
    for line in flagged:
        print(f"  {line}")

    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"out_dir: {out_dir}")
    for p in saved:
        print(f"saved: {p}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
