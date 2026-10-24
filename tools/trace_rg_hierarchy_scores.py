#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.metric_utils import nmi_iso_score
from main import _default_strategy_from_config, _resolved_main_config, _trace_output_qualifier
from tools.result_paths import classify_output_token, result_path, trace_result_paths
from tools.plot_theme import apply_basic_plot_theme
from test.hierarchy_levels import cluster_level_key_from_index
from tools.archive.rg_hierarchy_diagnostics import (
    build_params,
    level_position_stats,
    level_support_position_stats,
    representative_upper_level_state,
    true_leaf_level_geometry,
)
from tools.sequence_information_analysis import analyze_environment_sequence
from unified_experiments import (
    EvaluationReadoutConfig,
    _select_evaluation_prediction,
    create_model,
    generic_hierarchy_outputs,
    resolve_dataset_path,
    resolve_metric,
)


def clone_model_for_trace(model: Any) -> Any:
    """Collect diagnostics on a detached copy so checkpoints don't perturb the rollout."""
    return deepcopy(model)


def default_trace_result_paths(
    dataset: str,
    steps: int,
    *,
    strategy: str = "rg",
    params: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    qualifier = _trace_output_qualifier(params)
    if output_dir is not None:
        timestamped_json, timestamped_plot = trace_result_paths(
            strategy,
            dataset,
            int(steps),
            qualifier=qualifier,
        )
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        json_out = base_dir / timestamped_json.name
        if json_out.exists():
            counter = 1
            while True:
                candidate = base_dir / f"{Path(timestamped_json.name).stem}_{counter:02d}.json"
                if not candidate.exists():
                    json_out = candidate
                    break
                counter += 1
        plot_out = result_path(timestamped_plot.name, kind="picture")
    else:
        json_out, plot_out = trace_result_paths(strategy, dataset, int(steps), qualifier=qualifier)
    return json_out, plot_out


def _parse_bool_token(value: str | int | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_override_value(raw: str) -> Any:
    text = str(raw).strip()
    bool_value = _parse_bool_token(text)
    if bool_value is not None:
        return bool_value
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_set_override(raw: str) -> tuple[str, Any]:
    if "=" not in str(raw):
        raise argparse.ArgumentTypeError("--set expects key=value")
    key, value = str(raw).split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--set key cannot be empty")
    return key, _parse_override_value(value)


def _call_true_labels(env: Any, level: int | None = None) -> np.ndarray:
    true_label_fn = getattr(env, "trueLabel")
    if level is None:
        return np.asarray(true_label_fn(), dtype=int)
    try:
        return np.asarray(true_label_fn(level=level), dtype=int)
    except TypeError:
        return np.asarray(true_label_fn(), dtype=int)


def _score_outputs(
    model: Any,
    env: Any,
    *,
    selection_metric_name: str = "ari",
) -> dict[str, dict[str, float]]:
    outputs = generic_hierarchy_outputs(model, env)
    scores: dict[str, dict[str, float]] = {}
    depth = int(getattr(env, "hierarchy_depth", 1))
    score_depth = depth if bool(getattr(env, "is_hierarchical", False)) else 1
    for level_idx in range(max(1, int(score_depth))):
        key = cluster_level_key_from_index(level_idx)
        pred = np.asarray(outputs.get(key, np.full(model.input_size, -1, dtype=int)), dtype=int)
        true = _call_true_labels(env, level=level_idx if bool(getattr(env, "is_hierarchical", False)) else None)
        if pred.shape != true.shape:
            continue
        scores[key] = {
            "nmi": float(normalized_mutual_info_score(true, pred)),
            "nmiiso": float(nmi_iso_score(true, pred)),
            "ari": float(adjusted_rand_score(true, pred)),
            "cluster_count": int(len({int(x) for x in pred.tolist() if int(x) >= 0})),
            "noise_count": int(np.sum(pred < 0)),
        }
    if not bool(getattr(env, "is_hierarchical", False)):
        try:
            selection_metric = resolve_metric(selection_metric_name)
            selected = _select_evaluation_prediction(
                model,
                env,
                selection_metric,
                readout_config=EvaluationReadoutConfig(),
                label_level=None,
            )
            pred = np.asarray(selected["labels"], dtype=int)
            true = np.asarray(selected["true_labels"], dtype=int)
            scores["best_available"] = {
                "nmi": float(normalized_mutual_info_score(true, pred)),
                "nmiiso": float(nmi_iso_score(true, pred)),
                "ari": float(adjusted_rand_score(true, pred)),
                "cluster_count": int(len({int(x) for x in pred.tolist() if int(x) >= 0})),
                "noise_count": int(np.sum(pred < 0)),
                "evaluation_mode": str(selected.get("mode", "")),
                "evaluation_output_key": str(selected.get("output_key", "")),
                "evaluation_target_key": str(selected.get("truth_output_key", "")),
            }
        except Exception:
            pass
    return scores


def _level_snapshot(model: Any, level_index: int) -> dict[str, Any]:
    level = model.levels[level_index]
    stats = level_position_stats(np.asarray(level.syncmap, dtype=float))
    support = level_support_position_stats(model, level_index)
    return {
        "target_distance": float(model._level_equilibrium_distance(level_index)),
        "position_stats": stats,
        "support_position_stats": support,
    }


def _checkpoint_record(
    model: Any,
    env: Any,
    step: int,
    *,
    selection_metric_name: str = "ari",
) -> dict[str, Any]:
    """Inspect a checkpoint without advancing RNG streams used by the rollout."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    try:
        return _build_checkpoint_record(
            model,
            env,
            step,
            selection_metric_name=selection_metric_name,
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)


def _build_checkpoint_record(
    model: Any,
    env: Any,
    step: int,
    *,
    selection_metric_name: str = "ari",
) -> dict[str, Any]:
    inspect_model = clone_model_for_trace(model)
    record: dict[str, Any] = {
        "step": int(step),
        "inspection_mode": "passive_deepcopy",
        "scores": _score_outputs(inspect_model, env, selection_metric_name=selection_metric_name),
        "dynamic_levels": {},
        "upper_level_states": {},
        "leaf_true_geometry": true_leaf_level_geometry(inspect_model, env),
    }
    levels = getattr(inspect_model, "levels", None)
    if isinstance(levels, list):
        for level_index in range(len(levels)):
            record["dynamic_levels"][f"L{level_index}"] = _level_snapshot(inspect_model, level_index)
        for lower_level_index in range(len(levels) - 1):
            record["upper_level_states"][f"L{lower_level_index + 1}_to_L{lower_level_index + 2}"] = representative_upper_level_state(
                inspect_model,
                env,
                lower_level_index,
            )
    else:
        record["dynamic_levels"]["L0"] = {
            "target_distance": float(getattr(inspect_model, "graph_r0_fixed", 1.0)),
            "position_stats": level_position_stats(np.asarray(getattr(inspect_model, "syncmap"), dtype=float)),
        }
    return record


def print_summary(trace: list[dict[str, Any]]) -> None:
    print("=" * 120)
    print("RG hierarchy score trace")
    print("=" * 120)
    for row in trace:
        step = row["step"]
        scores = row.get("scores", {})
        upper = row.get("upper_level_states", {})
        fixed_parts: list[str] = []
        for key in sorted(
            (name for name in scores if name.startswith("cluster_level")),
            key=lambda name: int(name[len("cluster_level"):]),
        ):
            level = key[len("cluster_level"):]
            score = scores[key]
            fixed_parts.append(
                f"L{level} NMI/ARI={score.get('nmi', float('nan')):.3f}/{score.get('ari', float('nan')):.3f} "
                f"clusters={score.get('cluster_count', -1):>2} noise={score.get('noise_count', -1):>2}"
            )
        upper_parts: list[str] = []
        for name in sorted(upper):
            state = upper[name]
            upper_parts.append(
                f"{name} source={state.get('source_token_count', -1):>2} "
                f"pred={state.get('predicted_cluster_count', -1):>2}"
            )
        print(f"step={step:>7} | {' | '.join(fixed_parts + upper_parts)}")


# ── Integrated / cumulative NMI report ─────────────────────
# Measure overall clustering quality by the NMI area over the whole trace, not only peak/final.

def _extract_metric_series(
    trace: list[dict[str, Any]],
    level_key: str,
    metric: str = "nmi",
) -> list[float]:
    values: list[float] = []
    for row in trace:
        v = row.get("scores", {}).get(level_key, {}).get(metric)
        values.append(float(v) if v is not None else 0.0)
    return values


def compute_trace_integral(
    trace: list[dict[str, Any]],
    metric: str = "nmi",
) -> dict[str, Any]:
    """Compute integral NMI (or ARI) over a trace.

    Returns dict with per-level {area, mean, peak, peak_step, final}
    and joint_min / joint_product across all levels.
    """
    if not trace:
        return {}
    n = len(trace)
    steps = [int(row.get("step", 0)) for row in trace]

    all_keys: set[str] = set()
    for row in trace:
        for k in row.get("scores", {}):
            if k.startswith("cluster_level") or k == "best_available":
                all_keys.add(k)
    level_keys = sorted(all_keys, key=lambda key: (0 if key == "best_available" else 1, key))

    result: dict[str, Any] = {"checkpoints": n, "metric": metric, "levels": {}}
    all_series: list[list[float]] = []
    for key in level_keys:
        series = _extract_metric_series(trace, key, metric)
        all_series.append(series)
        area = float(np.sum(series))
        result["levels"][key] = {
            "area": round(area, 4),
            "mean": round(area / n, 4),
            "peak": round(float(np.max(series)), 4),
            "peak_step": steps[int(np.argmax(series))],
            "final": round(series[-1], 4),
        }

    cluster_series = [
        _extract_metric_series(trace, key, metric)
        for key in level_keys
        if key.startswith("cluster_level")
    ]
    if len(cluster_series) >= 2:
        joint_min = [float(min(s[i] for s in cluster_series)) for i in range(n)]
        joint_area = float(np.sum(joint_min))
        result["joint_min"] = {
            "area": round(joint_area, 4),
            "mean": round(joint_area / n, 4),
            "peak": round(float(np.max(joint_min)), 4),
            "peak_step": steps[int(np.argmax(joint_min))],
        }
        joint_prod = [float(np.prod([s[i] for s in cluster_series])) for i in range(n)]
        result["joint_product"] = {
            "area": round(float(np.sum(joint_prod)), 4),
            "mean": round(float(np.mean(joint_prod)), 4),
        }
    return result


def print_integral_summary(record: dict[str, Any], metric: str = "nmi") -> None:
    """Print a human-readable integral NMI/ARI report from a trace record."""
    trace = list(record.get("trace", []))
    if not trace:
        print("(empty trace)")
        return
    result = compute_trace_integral(trace, metric=metric)
    label = metric.upper()
    print("-" * 90)
    print(f"Integral {label} Summary  ({result.get('checkpoints', 0)} checkpoints)")
    print("-" * 90)
    for key, info in result.get("levels", {}).items():
        print(
            f"  {key:20s}  "
            f"mean={info['mean']:.4f}  "
            f"area={info['area']:.2f}  "
            f"peak={info['peak']:.4f} @{info['peak_step']}  "
            f"final={info['final']:.4f}"
        )
    jm = result.get("joint_min")
    if jm:
        print(f"  {'joint_min':20s}  mean={jm['mean']:.4f}  area={jm['area']:.2f}  peak={jm['peak']:.4f} @{jm['peak_step']}")
    jp = result.get("joint_product")
    if jp:
        print(f"  {'joint_product':20s}  mean={jp['mean']:.4f}  area={jp['area']:.2f}")
    print("-" * 90)


def save_score_plot(
    record: dict[str, Any],
    output_path: str | Path,
    *,
    metric: str = "nmi",
    theme: str = "light",
    accent_color: str = "#6970ac",
) -> Path:
    metric_key = str(metric).strip().lower()
    if metric_key not in {"nmi", "ari"}:
        raise ValueError(f"Unsupported plot metric: {metric}. Use 'nmi' or 'ari'.")
    trace = list(record.get("trace", []))
    steps = np.asarray([int(row.get("step", 0)) for row in trace], dtype=int)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    plotted = False
    keys = sorted(
        {
            str(key)
            for row in trace
            for key in row.get("scores", {})
            if str(key).startswith("cluster_level") and str(key)[len("cluster_level"):].isdigit()
        },
        key=lambda key: int(key[len("cluster_level"):]),
    )
    series_color_keys = ["series_l1", "series_l2", "series_l3"]
    palette = apply_basic_plot_theme(fig, ax, theme=theme, accent=accent_color, grid_alpha=0.22)
    extra_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(keys))))
    for index, key in enumerate(keys):
        level_number = key[len("cluster_level"):]
        label = f"L{level_number} {metric_key.upper()}"
        values = []
        has_any = False
        for row in trace:
            score = row.get("scores", {}).get(key, {})
            value = score.get(metric_key)
            if value is None:
                values.append(np.nan)
            else:
                has_any = True
                values.append(float(value))
        if has_any:
            ax.plot(
                steps,
                np.asarray(values, dtype=float),
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                label=label,
                color=(
                    str(palette[series_color_keys[index]])
                    if index < len(series_color_keys)
                    else extra_colors[index]
                ),
            )
            plotted = True

    ax.set_title(f"RG Hierarchy {metric_key.upper()} Over Time")
    ax.set_xlabel("step")
    ax.set_ylabel(metric_key.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0.0)
    if plotted:
        legend = ax.legend(frameon=False, labelcolor=str(palette["fg"]))
        if legend is not None and legend.get_title() is not None:
            legend.get_title().set_color(palette["fg"])
        apply_basic_plot_theme(fig, ax, theme=theme, accent=accent_color, grid_alpha=0.22)
    fig.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, facecolor=fig.get_facecolor(), edgecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def save_nmi_plot(
    record: dict[str, Any],
    output_path: str | Path,
    *,
    theme: str = "light",
    accent_color: str = "#6970ac",
) -> Path:
    return save_score_plot(record, output_path, metric="nmi", theme=theme, accent_color=accent_color)


def _git_reproducibility_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "git_head": None,
        "git_head_short": None,
        "git_is_dirty": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_head"] = head or None
        if head:
            metadata["git_head_short"] = head[:12]
    except Exception:
        pass
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_is_dirty"] = bool(dirty)
    except Exception:
        pass
    return metadata


def run_trace(
    dataset: str,
    strategy: str,
    total_steps: int,
    record_every: int,
    env_mode: str | None,
    override_params: dict[str, Any],
    seed: int = 0,
    selection_metric: str = "ari",
) -> dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    requested_params = build_params(dataset, strategy, env_mode, seed=int(seed))
    requested_params.update(override_params)
    resolved_dataset = resolve_dataset_path(dataset)
    model, env = create_model(resolved_dataset, strategy, deepcopy(requested_params), record_trajectory=False)
    try:
        sequence_information = analyze_environment_sequence(env)
    except Exception as exc:
        sequence_information = {"error": str(exc)}

    checkpoints: list[dict[str, Any]] = [
        _checkpoint_record(model, env, 0, selection_metric_name=selection_metric)
    ]
    for step in range(1, int(total_steps) + 1):
        x, _ = env.getSequence(1)
        model.input(x)
        if step % int(record_every) == 0 or step == int(total_steps):
            checkpoints.append(
                _checkpoint_record(model, env, step, selection_metric_name=selection_metric)
            )

    return {
        "dataset": resolved_dataset,
        "strategy": strategy,
        "env_mode": requested_params.get("env_mode"),
        "params": requested_params,
        "reproducibility": {
            "dataset_input": dataset,
            "dataset_resolved": resolved_dataset,
            "strategy": strategy,
            "seed": int(seed),
            "total_steps": int(total_steps),
            "record_every": int(record_every),
            "requested_params": requested_params,
            "effective_model_params": getattr(model, "effective_model_params", None),
            "effective_env_params": getattr(model, "effective_env_params", None),
            "model_class": type(model).__name__,
            "env_class": type(env).__name__,
            "python_executable": sys.executable,
            **_git_reproducibility_metadata(),
        },
        "sequence_information": sequence_information,
        "total_steps": int(total_steps),
        "record_every": int(record_every),
        "selection_metric": str(selection_metric),
        "trace": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RG hierarchy NMI/ARI and geometry over time.")
    config = _resolved_main_config()
    parser.add_argument("--dataset", default=str(config["dataset"]))
    parser.add_argument("--strategy", default=str(_default_strategy_from_config(config)))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--record-interval", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--env-mode", default=None)
    parser.add_argument("--t-push", type=int, default=None)
    parser.add_argument("--knn", type=int, default=None)
    parser.add_argument("--graph-theta", type=float, default=None)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--k-attr", type=float, default=None)
    parser.add_argument("--persistent-relation-force", default=None, type=_parse_bool_token)
    parser.add_argument("--persistent-relation-decay", type=float, default=None)
    parser.add_argument("--persistent-relation-threshold", type=float, default=None)
    parser.add_argument("--persistent-relation-signal-gain", type=float, default=None)
    parser.add_argument("--persistent-relation-same-force-strength", type=float, default=None)
    parser.add_argument("--persistent-relation-diff-force-strength", type=float, default=None)
    parser.add_argument("--cluster-distance-strength", type=float, default=None)
    parser.add_argument("--stats-mode", default=None)
    parser.add_argument("--stats-memory-steps", type=int, default=None)
    parser.add_argument("--stats-memory-scale", type=float, default=None)
    parser.add_argument("--stats-power-gain", type=float, default=None)
    parser.add_argument("--rg-knn-k", type=int, default=None)
    parser.add_argument("--rg-jump-ratio", type=float, default=None)
    parser.add_argument("--rg-fallback-scale", type=float, default=None)
    parser.add_argument("--rg-input-level", default=None)
    parser.add_argument("--graph-spring-extend-shoulder-gain", type=float, default=None)
    parser.add_argument("--graph-spring-extend-shoulder-rel-start", type=float, default=None)
    parser.add_argument("--support-unit-repulsion", default=None, type=_parse_bool_token)
    parser.add_argument("--support-unit-repulsion-strength", type=float, default=None)
    parser.add_argument("--inter-level-support-reanchor", default=None, type=_parse_bool_token)
    parser.add_argument("--inter-level-support-reanchor-strength", type=float, default=None)
    parser.add_argument("--source-unit-motion-coherence", default=None, type=_parse_bool_token)
    parser.add_argument("--source-unit-motion-coherence-strength", type=float, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-plot",
        "--plot",
        dest="save_plot",
        nargs="?",
        const="true",
        default=True,
        type=lambda x: bool(_parse_bool_token(x)),
        help="Save the NMI image; defaults to true and supports --save-plot=true/false",
    )
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--plot-metric", default="nmi", choices=["nmi", "ari"])
    parser.add_argument("--theme", default="dark", choices=["light", "dark"])
    parser.add_argument("--accent-color", default="#6970ac")
    parser.add_argument("--print-trace", action="store_true")
    parser.add_argument(
        "--set",
        dest="generic_overrides",
        action="append",
        default=[],
        type=_parse_set_override,
        metavar="KEY=VALUE",
        help="Generic strategy/runtime parameter override. May be repeated.",
    )
    args = parser.parse_args()

    dataset = resolve_dataset_path(str(args.dataset))
    overrides: dict[str, Any] = {}
    if args.t_push is not None:
        overrides["T_push"] = int(args.t_push)
    if args.knn is not None:
        overrides["knn"] = int(args.knn)
    if args.graph_theta is not None:
        overrides["graph_theta"] = float(args.graph_theta)
    if args.dbscan_eps is not None:
        overrides["dbscan_eps"] = float(args.dbscan_eps)
    if args.k_attr is not None:
        overrides["k_attr"] = float(args.k_attr)
    if args.persistent_relation_force is not None:
        overrides["persistent_relation_force_enabled"] = bool(args.persistent_relation_force)
    if args.persistent_relation_decay is not None:
        overrides["persistent_relation_decay"] = float(args.persistent_relation_decay)
    if args.persistent_relation_threshold is not None:
        overrides["persistent_relation_threshold"] = float(args.persistent_relation_threshold)
    if args.persistent_relation_signal_gain is not None:
        overrides["persistent_relation_signal_gain"] = float(args.persistent_relation_signal_gain)
    if args.persistent_relation_same_force_strength is not None:
        overrides["persistent_relation_same_force_strength_ratio"] = float(args.persistent_relation_same_force_strength)
    if args.persistent_relation_diff_force_strength is not None:
        overrides["persistent_relation_diff_force_strength_ratio"] = float(args.persistent_relation_diff_force_strength)
    if args.cluster_distance_strength is not None:
        overrides["cluster_distance_strength"] = float(args.cluster_distance_strength)
    if args.stats_mode is not None:
        overrides["stats_mode"] = str(args.stats_mode)
    if args.stats_memory_steps is not None:
        overrides["stats_memory_steps"] = int(args.stats_memory_steps)
    if args.stats_memory_scale is not None:
        overrides["stats_memory_scale"] = float(args.stats_memory_scale)
    if args.stats_power_gain is not None:
        overrides["stats_power_gain"] = float(args.stats_power_gain)
    if args.rg_knn_k is not None:
        overrides["rg_knn_k"] = int(args.rg_knn_k)
    if args.rg_jump_ratio is not None:
        overrides["rg_jump_ratio"] = float(args.rg_jump_ratio)
    if args.rg_fallback_scale is not None:
        overrides["rg_fallback_scale"] = float(args.rg_fallback_scale)
    if args.rg_input_level is not None:
        overrides["rg_input_level"] = str(args.rg_input_level)
    if args.graph_spring_extend_shoulder_gain is not None:
        overrides["graph_spring_extend_shoulder_gain"] = float(args.graph_spring_extend_shoulder_gain)
    if args.graph_spring_extend_shoulder_rel_start is not None:
        overrides["graph_spring_extend_shoulder_rel_start"] = float(args.graph_spring_extend_shoulder_rel_start)
    if args.support_unit_repulsion is not None:
        overrides["support_unit_repulsion_enabled"] = bool(args.support_unit_repulsion)
    if args.support_unit_repulsion_strength is not None:
        overrides["support_unit_repulsion_strength_ratio"] = float(args.support_unit_repulsion_strength)
    if args.inter_level_support_reanchor is not None:
        overrides["inter_level_support_reanchor_enabled"] = bool(args.inter_level_support_reanchor)
    if args.inter_level_support_reanchor_strength is not None:
        overrides["inter_level_support_reanchor_strength"] = float(args.inter_level_support_reanchor_strength)
    if args.source_unit_motion_coherence is not None:
        overrides["source_unit_motion_coherence_enabled"] = bool(args.source_unit_motion_coherence)
    if args.source_unit_motion_coherence_strength is not None:
        overrides["source_unit_motion_coherence_strength"] = float(args.source_unit_motion_coherence_strength)
    for key, value in args.generic_overrides:
        overrides[str(key)] = value

    record_interval = args.record_interval
    if args.record_every is not None:
        if record_interval is not None and int(record_interval) != int(args.record_every):
            raise SystemExit(
                f"Conflicting interval arguments: record_interval={int(record_interval)}, "
                f"record_every={int(args.record_every)}. Use only --record-interval."
            )
        record_interval = int(args.record_every)
    if record_interval is None:
        record_interval = 10000

    record = run_trace(
        dataset=dataset,
        strategy=str(args.strategy),
        total_steps=int(args.steps),
        record_every=int(record_interval),
        env_mode=args.env_mode,
        override_params=overrides,
        seed=int(args.seed),
    )
    if bool(args.print_trace):
        print_summary(record["trace"])

    json_mode = classify_output_token(args.json_out)
    if json_mode == "auto":
        json_out, default_plot_out = default_trace_result_paths(
            dataset,
            int(args.steps),
            strategy=str(args.strategy),
            params=record.get("params"),
        )
    elif json_mode == "path":
        json_out = Path(str(args.json_out))
        default_plot_out = result_path(f"{json_out.stem}.png", kind="picture")
    else:
        json_out = None
        _, default_plot_out = default_trace_result_paths(
            dataset,
            int(args.steps),
            strategy=str(args.strategy),
            params=record.get("params"),
        )
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json: {json_out}")
    else:
        print("saved_json: false")

    # Always print the integrated NMI summary.
    print_integral_summary(record, metric=str(args.plot_metric))

    if bool(args.save_plot):
        plot_mode = classify_output_token(args.plot_out)
        if plot_mode == "path":
            plot_out: str | Path | None = str(args.plot_out)
        elif plot_mode == "auto":
            plot_out = str(default_plot_out)
        else:
            plot_out = None
        if plot_out is not None:
            saved_plot = save_score_plot(
                record,
                plot_out,
                metric=str(args.plot_metric),
                theme=str(args.theme),
                accent_color=str(args.accent_color),
            )
            print(f"saved_plot: {saved_plot}")
        else:
            print("saved_plot: false")


if __name__ == "__main__":
    main()
