"""Time-budgeted (wall-clock) multi-strategy convergence comparison.

A fairer, online-learning-flavoured variant of convergence_compare.py: instead
of giving every strategy the same number of STEPS, give each the same wall-clock
TRAINING time (default 60 s) and plot ARI vs elapsed training seconds. Steps are
not comparable across strategies — rg costs ~10x tsf per step, so equal-steps
quietly hands slow strategies far more compute. Equal-time answers "given N
seconds of compute, who learns the best embedding online?".

Only feeding counts toward the budget; the periodic readout (measurement, not
learning) is timed out separately, so the x-axis is pure learning compute. The
input sequence is drawn on the fly in step_chunk batches (genuinely online) with
the same hierarchy-refresh cadence as the unified trainer, and params are
resolved through the task-1/task-2 path so tuned profiles (e.g. rg) apply.

Usage:
    python tools/convergence_time_compare.py \
        --datasets datasets/chain_mixed5_60.dot ... \
        --strategies syncmap symmetric rg tsf --seconds 60
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.plot_theme import apply_basic_plot_theme, resolve_plot_palette  # noqa: E402
from unified_experiments import (  # noqa: E402
    _advance_model_chunk,
    _hierarchy_depth,
    _isolated_model_rng,
    _pop_evaluation_readout_config,
    _select_evaluation_prediction,
    apply_hierarchy_total_scores,
    compute_all_metric_scores,
    compute_hierarchy_metric_diagnostics,
    create_model,
    resolve_metric,
)
from main import _resolve_runtime_selection  # noqa: E402

# Shared helpers (colours, per-curve degradation stats, main-config cache).
from tools._probes.convergence_compare import _color_for, _config, _curve_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _aggregate_score(model, env, metric, readout_cfg, label_level, metric_key,
                     is_hier, depth) -> float:
    """Formal aggregate score (mean-over-levels ARI for hierarchy), matching the
    unified comparison's per-sample scoring."""
    res = _select_evaluation_prediction(model, env, metric,
                                        readout_config=readout_cfg, label_level=label_level)
    labels = np.asarray(res["labels"], dtype=int)
    truth = np.asarray(res["true_labels"], dtype=int)
    sc = compute_all_metric_scores(labels, truth)
    sc.update(compute_hierarchy_metric_diagnostics(model, env, readout_cfg,
                                                   outputs=res.get("outputs")))
    sc = apply_hierarchy_total_scores(
        {k: float(v) for k, v in sc.items() if k.endswith("_score")},
        is_hierarchical=is_hier, depth=depth,
    )
    key = "ari_score" if metric_key == "ari" else "nmi_score"
    return float(sc.get(key, res.get("score", 0.0)))


def _train_timed(dataset: str, strategy: str, seconds: float, seed: int,
                 metric_key: str, sample_dt: float
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """Train for ``seconds`` of wall-clock FEED time; sample score every
    ~``sample_dt`` s. Returns (times, scores, total_steps)."""
    _, runtime_name, _env_mode, params = _resolve_runtime_selection(
        _config(), dataset=dataset, strategy=strategy,
        profile=None, env_mode=None, override_params={},
    )
    np.random.seed(int(seed))  # identical env sequence across strategies
    model_params = dict(params)
    model_params.pop("seed", None)
    ll_raw = model_params.pop("label_level", None)
    label_level = int(ll_raw) if ll_raw is not None else None
    readout_cfg = _pop_evaluation_readout_config(model_params)

    model, env = create_model(
        dataset,
        runtime_name,
        model_params,
        record_trajectory=False,
        environment_seed=int(seed),
    )
    metric = resolve_metric("ari" if metric_key == "ari" else "nmi")
    is_hier = bool(getattr(env, "is_hierarchical", False))
    depth = _hierarchy_depth(env)
    feed_chunk = max(1, int(getattr(model, "step_chunk", 100)))
    refresh_on = is_hier or bool(getattr(model, "level1_enabled", False))
    hrefresh = (max(0, int(getattr(model, "hierarchy_refresh_interval", 0) or 0))
                if refresh_on else 0)
    refresher = getattr(model, "refresh_hierarchy_state", None)

    def score() -> float:
        return _aggregate_score(model, env, metric, readout_cfg, label_level,
                                metric_key, is_hier, depth)

    times: list[float] = [0.0]
    scores: list[float] = [score()]  # anchor at t=0 (pre-training)
    steps_total = 0
    train_elapsed = 0.0
    last_sample = 0.0

    while train_elapsed < seconds:
        t0 = time.perf_counter()
        seq, _ = env.getSequence(feed_chunk)
        _advance_model_chunk(model, seq, phase_offset=steps_total,
                             reset_trajectory=(steps_total == 0), append_final_frame=False)
        steps_total += feed_chunk
        if hrefresh > 0 and callable(refresher) and steps_total % hrefresh == 0:
            with _isolated_model_rng(model):
                refresher()
        train_elapsed += time.perf_counter() - t0  # feed time only

        if train_elapsed - last_sample >= sample_dt:
            scores.append(score())  # eval is NOT added to train_elapsed
            times.append(train_elapsed)
            last_sample = train_elapsed

    scores.append(score())
    times.append(train_elapsed)
    return np.asarray(times, dtype=float), np.asarray(scores, dtype=float), steps_total


def _plot_dataset(ax: Any, curves: dict[str, tuple[np.ndarray, np.ndarray]], *,
                  title: str, metric_key: str, seconds: float,
                  palette: dict[str, Any]) -> None:
    for idx, (strategy, (xs, ys)) in enumerate(curves.items()):
        color = _color_for(strategy, idx)
        st = _curve_stats(xs, ys)
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.0, color=color,
                label=f"{strategy}  late μ{st['late_mean']:.2f}/σ{st['late_std']:.2f}")
        if np.isfinite(st["peak"]):
            pk_idx = int(np.nanargmax(ys)) if ys.size else 0
            ax.plot([xs[pk_idx]], [ys[pk_idx]], marker="*", markersize=11, color=color,
                    markeredgecolor=palette["fig_bg"], markeredgewidth=0.6,
                    linestyle="none", zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("training time (s)")
    ax.set_ylabel(metric_key.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.0, seconds)
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
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Wall-clock feed-time budget per strategy per dataset")
    parser.add_argument("--points", type=int, default=60,
                        help="Target sample points per curve (sets the time cadence)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="ari", choices=["ari", "nmi"])
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    parser.add_argument("--accent-color", default="#6970ac")
    parser.add_argument("--out-dir", default=None)
    cli = parser.parse_args()

    metric_key = str(cli.metric).strip().lower()
    sample_dt = max(1e-3, float(cli.seconds) / max(1, int(cli.points)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(cli.out_dir) if cli.out_dir
               else REPO_ROOT / "results" / "convergence_time" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = resolve_plot_palette(theme=cli.theme, accent=cli.accent_color)

    all_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    steps_done: dict[str, dict[str, int]] = {}
    json_path = out_dir / "curves.json"

    def _dump_json() -> None:
        payload = {
            "config": {"seconds": cli.seconds, "points": cli.points, "seed": cli.seed,
                       "metric": metric_key, "x_axis": "training_time_seconds",
                       "strategies": list(cli.strategies), "datasets": list(cli.datasets)},
            "datasets": {
                name: {
                    strat: {"times": xs.tolist(), "scores": ys.tolist(),
                            "steps": steps_done.get(name, {}).get(strat, 0),
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
        steps_done[name] = {}
        for strategy in cli.strategies:
            xs, ys, nsteps = _train_timed(dataset, strategy, cli.seconds, cli.seed,
                                          metric_key, sample_dt)
            curves[strategy] = (xs, ys)
            st = _curve_stats(xs, ys)
            stats[name][strategy] = st
            steps_done[name][strategy] = nsteps
            print(f"  {strategy:10s} steps={nsteps:>9d}  "
                  f"mean={st['mean']:+.3f}±{st['std']:.3f}  "
                  f"late={st['late_mean']:+.3f}±{st['late_std']:.3f}  "
                  f"stable={st['stable_score']:+.3f}  peak={st['peak']:+.3f}@"
                  f"{st['peak_step']:>4d}s  final={st['final']:+.3f}  [{st['verdict']}]")
        all_curves[name] = curves
        _dump_json()

        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        apply_basic_plot_theme(fig, ax, theme=cli.theme, accent=cli.accent_color,
                               grid_alpha=0.22)
        _plot_dataset(ax, curves, title=name, metric_key=metric_key,
                      seconds=cli.seconds, palette=palette)
        fig.tight_layout()
        target = out_dir / f"timeconv_{Path(name).stem}.png"
        fig.savefig(target, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(target)

    n = len(all_curves)
    if n > 1:
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.2, rows * 4.4), squeeze=False)
        fig.patch.set_facecolor(palette["fig_bg"])
        flat = [a for row in axes for a in row]
        for ax, (name, curves) in zip(flat, all_curves.items()):
            apply_basic_plot_theme(fig, ax, theme=cli.theme, accent=cli.accent_color,
                                   grid_alpha=0.22)
            _plot_dataset(ax, curves, title=name, metric_key=metric_key,
                          seconds=cli.seconds, palette=palette)
        for ax in flat[n:]:
            ax.axis("off")
        fig.suptitle(f"Online learning ({metric_key.upper()} vs training time, "
                     f"{cli.seconds:g}s/strategy, seed {cli.seed})",
                     color=palette["fg"], fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        grid_target = out_dir / "timeconv_grid.png"
        fig.savefig(grid_target, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(grid_target)

    _dump_json()

    print("\n=== temporal stability summary over time budget ===")
    print(f"{'dataset':32s} {'strategy':10s} {'steps':>9s} {'mean':>7s} {'std':>7s} "
          f"{'late':>7s} {'lstd':>7s} {'stable':>7s} {'peak':>7s} {'final':>7s}  verdict")
    for name, per in stats.items():
        for strat, st in per.items():
            mark = "  <<<" if st["verdict"] in {"DEGRADES", "COLLAPSE"} else ""
            print(f"{name:32s} {strat:10s} {steps_done[name][strat]:>9d} "
                  f"{st['mean']:+7.3f} {st['std']:+7.3f} "
                  f"{st['late_mean']:+7.3f} {st['late_std']:+7.3f} "
                  f"{st['stable_score']:+7.3f} {st['peak']:+7.3f} "
                  f"{st['final']:+7.3f}  {st['verdict']}{mark}")

    print(f"\nelapsed: {time.time() - started:.1f}s")
    print(f"out_dir: {out_dir}")
    for p in saved:
        print(f"saved: {p}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
