#!/usr/bin/env python3
"""SyncMap scaling-law external evaluation.

Analogous to the LLM scaling law (loss falls as a power of scale, i.e. a straight
line on log-log axes), this tool plots ``loss = 1 - metric`` against the learning
step count and fits a power law in log-log space. The straight-line goodness-of-fit
(R^2) is reported as a rough "scaling adherence" score — how closely the algorithm's
convergence tracks a clean power law (a proxy for approaching some theoretical limit).

This is a *purely external* evaluation: it reuses the existing task-4 trace machinery
and never touches the core training/clustering flow. Invoke it directly:

    python tools/scaling_law.py --dataset <x.dot> --profile rg --steps 20000 \
        --record-interval 2000 --metric nmi

or feed an existing task-4 trace JSON (no re-run):

    python tools/scaling_law.py --trace-json results/.../score.json --metric ari
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.plot_theme import apply_basic_plot_theme
from tools.result_paths import timestamped_result_path
from unified_experiments import resolve_dataset_path

METRIC_ALIASES = {"nmi": "nmi", "ari": "ari", "nmiiso": "nmiiso", "nmi_iso": "nmiiso"}


def _normalize_metric(metric: str) -> str:
    key = str(metric).strip().lower()
    if key not in METRIC_ALIASES:
        raise ValueError(f"Unsupported metric {metric!r}; use one of nmi / ari / nmiiso.")
    return METRIC_ALIASES[key]


def _row_metric(scores: dict[str, Any], metric: str) -> float | None:
    """Aggregate a checkpoint's metric to a single scalar (mean across levels)."""
    values: list[float] = []
    for key, payload in scores.items():
        if not isinstance(payload, dict):
            continue
        if not (str(key).startswith("cluster_level") and str(key)[len("cluster_level"):].isdigit()):
            continue
        value = payload.get(metric)
        if value is not None:
            values.append(float(value))
    if values:
        return float(np.mean(values))
    fallback = scores.get("best_available")
    if isinstance(fallback, dict) and fallback.get(metric) is not None:
        return float(fallback[metric])
    return None


def extract_curve(record: dict[str, Any], metric: str, *, loss_floor: float) -> list[tuple[int, float]]:
    metric = _normalize_metric(metric)
    curve: list[tuple[int, float]] = []
    learning_curve = record.get("learning_curve")
    if isinstance(learning_curve, list):
        for row in learning_curve:
            if not isinstance(row, dict):
                continue
            step = int(float(row.get("iteration", row.get("step", 0))))
            value = row.get(metric)
            if value is None:
                value = row.get(f"{metric}_score")
            if value is None:
                continue
            loss = float(
                np.clip(1.0 - float(value), loss_floor, 1.0)
            )
            curve.append((step, loss))
        return curve
    for row in record.get("trace", []):
        step = int(row.get("step", 0))
        value = _row_metric(row.get("scores", {}) or {}, metric)
        if value is None:
            continue
        loss = float(np.clip(1.0 - value, loss_floor, 1.0))
        curve.append((step, loss))
    return curve


def fit_power_law(steps: np.ndarray, losses: np.ndarray) -> dict[str, Any]:
    """Linear fit of log10(loss) vs log10(step): loss ~= 10**intercept * step**exponent."""
    mask = (steps > 0) & (losses > 0)
    x = np.log10(steps[mask].astype(float))
    y = np.log10(losses[mask].astype(float))
    if x.size < 2 or np.allclose(x, x[0]):
        return {"exponent": None, "intercept": None, "r2": None, "rmse_log": None, "n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    rmse_log = float(np.sqrt(ss_res / x.size))
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r2": None if r2 is None else float(r2),
        "rmse_log": rmse_log,
        "n": int(x.size),
    }


def _load_record(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    """Return (trace record, dataset label, profile/strategy label)."""
    if args.trace_json:
        payload = json.loads(
            Path(args.trace_json).expanduser().read_text(encoding="utf-8")
        )
        if isinstance(payload, list):
            record = {"learning_curve": payload}
        elif isinstance(payload, dict):
            record = payload
        else:
            raise SystemExit(
                "Trace JSON must be either a task-0 learning-curve list or "
                "a task-4 trace record."
            )
        dataset = str(record.get("dataset") or args.dataset or "trace")
        strategy = str(record.get("strategy") or args.profile or "trace")
        return record, dataset, strategy

    if not args.dataset:
        raise SystemExit("Provide either --trace-json or --dataset.")
    from tools.trace_rg_hierarchy_scores import run_trace

    strategy = str(args.strategy or args.profile)
    dataset = resolve_dataset_path(args.dataset)
    seeds = [int(s) for s in (args.seeds_list or [args.seed])]
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(
            run_trace(
                dataset,
                strategy,
                int(args.steps),
                int(args.record_interval),
                args.env_mode,
                {},
                seed=int(seed),
                selection_metric=_normalize_metric(args.metric),
            )
        )
    if len(per_seed) == 1:
        return per_seed[0], dataset, strategy
    return {"trace": _average_traces(per_seed, args.metric, loss_floor=args.loss_floor), "strategy": strategy, "dataset": dataset}, dataset, strategy


def _average_traces(records: list[dict[str, Any]], metric: str, *, loss_floor: float) -> list[dict[str, Any]]:
    """Collapse multiple seeds into one trace by averaging the aggregated metric per step."""
    metric = _normalize_metric(metric)
    by_step: dict[int, list[float]] = {}
    for record in records:
        for step, loss in extract_curve(record, metric, loss_floor=loss_floor):
            by_step.setdefault(int(step), []).append(1.0 - loss)  # store metric, not loss
    merged: list[dict[str, Any]] = []
    for step in sorted(by_step):
        mean_metric = float(np.mean(by_step[step]))
        merged.append({"step": int(step), "scores": {"cluster_level1": {metric: mean_metric}}})
    return merged


def _plot(
    curve: list[tuple[int, float]],
    fit: dict[str, Any],
    *,
    metric: str,
    dataset: str,
    strategy: str,
    out_path: Path,
    theme: str,
    accent: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([s for s, _ in curve], dtype=float)
    losses = np.asarray([l for _, l in curve], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    palette = apply_basic_plot_theme(fig, ax, theme=theme, accent=accent, grid_alpha=0.25)

    ax.scatter(steps, losses, s=28, color=accent, zorder=3, label=f"loss = 1 - {metric.upper()}")
    if fit.get("exponent") is not None:
        xs = np.array([steps[steps > 0].min(), steps.max()], dtype=float)
        ys = 10.0 ** (fit["intercept"] + fit["exponent"] * np.log10(xs))
        r2_text = "n/a" if fit.get("r2") is None else f"{fit['r2']:.3f}"
        ax.plot(
            xs,
            ys,
            "--",
            color=palette.get("fg", "#333333"),
            linewidth=1.6,
            zorder=2,
            label=f"fit: b={fit['exponent']:.3f}, R²={r2_text}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning steps (log)")
    ax.set_ylabel(f"loss = 1 - {metric.upper()} (log)")
    ax.set_title(f"SyncMap scaling curve — {Path(dataset).stem} / {strategy}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_scaling_law(args: Any) -> dict[str, Any]:
    metric = _normalize_metric(args.metric)
    record, dataset, strategy = _load_record(args)
    curve = extract_curve(record, metric, loss_floor=float(args.loss_floor))
    if not curve:
        raise SystemExit("No metric checkpoints found in trace; nothing to fit.")

    fit_curve = [
        (s, l)
        for s, l in curve
        if (args.fit_min_step is None or s >= int(args.fit_min_step))
        and (args.fit_max_step is None or s <= int(args.fit_max_step))
    ] or curve
    steps = np.asarray([s for s, _ in fit_curve], dtype=float)
    losses = np.asarray([l for _, l in fit_curve], dtype=float)
    fit = fit_power_law(steps, losses)

    final_loss = float(curve[-1][1])
    summary = {
        "metric": metric,
        "dataset": Path(dataset).name,
        "strategy": strategy,
        "loss_floor": float(args.loss_floor),
        "points": [{"step": int(s), "loss": float(l)} for s, l in curve],
        "fit": fit,
        "fit_window": {"min_step": args.fit_min_step, "max_step": args.fit_max_step, "points_used": len(fit_curve)},
        "final_loss": final_loss,
        "adherence_r2": fit.get("r2"),
    }

    stem_prefix = f"SCALING_{Path(dataset).stem}_{strategy}_{metric}"
    png_path = Path(args.plot_path).expanduser() if args.plot_path else timestamped_result_path(stem_prefix, "png", kind="picture")
    json_path = Path(args.json_path).expanduser() if args.json_path else timestamped_result_path(stem_prefix, "json", kind="experiment")
    _plot(curve, fit, metric=metric, dataset=dataset, strategy=strategy, out_path=png_path, theme=args.theme, accent=args.accent)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    exp = fit.get("exponent")
    r2 = fit.get("r2")
    print(
        f"scaling_law[{metric}] {Path(dataset).stem}/{strategy}: "
        f"exponent={'n/a' if exp is None else f'{exp:.4f}'}  "
        f"adherence_R2={'n/a' if r2 is None else f'{r2:.4f}'}  "
        f"final_loss={final_loss:.4f}  (n={fit.get('n')})"
    )
    print(f"saved_plot: {png_path}")
    print(f"saved_json: {json_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SyncMap scaling-law (loss-vs-steps power-law) external evaluation.")
    src = parser.add_argument_group("data source (use --trace-json OR --dataset)")
    src.add_argument("--trace-json", default=None, help="Reuse an existing task-4 trace JSON (no re-run).")
    src.add_argument("--dataset", "-d", default=None, help="Dataset path to run a fresh trace on.")
    src.add_argument("--profile", default="rg", help="Profile/strategy to start from (default rg).")
    src.add_argument("--strategy", default=None, help="Explicit strategy name; overrides --profile when set.")
    src.add_argument("--steps", type=int, default=110000, help="Trace length for fresh runs.")
    src.add_argument("--record-interval", type=int, default=2000, help="Checkpoint interval for fresh runs.")
    src.add_argument("--seed", type=int, default=0)
    src.add_argument("--seeds-list", nargs="*", type=int, default=None, help="Multiple seeds; losses are averaged per step.")
    src.add_argument("--env-mode", default=None, help="Override env mode (default: auto-inferred).")

    parser.add_argument("--metric", default="nmi", choices=["nmi", "ari", "nmiiso", "nmi_iso"])
    parser.add_argument("--loss-floor", type=float, default=1e-3, help="Clamp floor for loss before log.")
    parser.add_argument("--fit-min-step", type=int, default=None, help="Only fit checkpoints with step >= this.")
    parser.add_argument("--fit-max-step", type=int, default=None, help="Only fit checkpoints with step <= this.")
    parser.add_argument("--plot-path", default=None, help="Explicit PNG output path.")
    parser.add_argument("--json-path", default=None, help="Explicit JSON output path.")
    parser.add_argument("--theme", default="light")
    parser.add_argument("--accent", default="#6970ac")
    return parser


def main() -> None:
    run_scaling_law(build_parser().parse_args())


if __name__ == "__main__":
    main()
