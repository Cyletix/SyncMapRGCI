"""Continual graph-switch probe for NSGD freezing and learning a new graph.

The probe keeps one model alive across ``source -> target`` and calls the
environment's real ``switch_dataset()`` path.  A fresh target-only model sees
the exact target sequence as the switched model.  This makes post-switch
learning, rather than transfer-file loading or sequence drift, the comparison.

Defaults are a cheap screen.  Use ``--stage guard`` only after the screen is
worth expanding; neither stage changes any public strategy/profile default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/syncmaprgci-matplotlib")

from tools.main_config import VARIANT_CONFIGS  # noqa: E402
from tools.result_paths import experiment_archive_dir, timestamp_now  # noqa: E402
from unified_experiments import (  # noqa: E402
    EvaluationReadoutConfig,
    _advance_model_chunk,
    _isolated_environment_rng,
    _select_evaluation_prediction,
    compute_all_metric_scores,
    create_model,
    resolve_dataset_path,
    resolve_metric,
    summarize_predicted_labels,
)


# Same-partition controls test structural/statistical adaptation while the
# desired clustering remains valid.  Partition-shift pairs are the primary
# test of whether the model can actually learn a new grouping.
DEFAULT_PAIRS = [
    ("datasets/fixed5_6.dot", "datasets/probabilistic5_6.dot"),
    ("datasets/UB_mixed_chunk_20_5_5.dot", "datasets/UB_prob_chunk_20_5_5.dot"),
    ("datasets/imbalanced_fixed_30.dot", "datasets/imbalanced_probablistic_30.dot"),
    ("datasets/fixed5_6.dot", "datasets/fixed6_5.dot"),
    ("datasets/fixed5_6.dot", "datasets/chain_mixed6_5.dot"),
    ("datasets/probabilistic5_6.dot", "datasets/probabilistic6_5.dot"),
    ("datasets/chain_mixed5_6.dot", "datasets/chain_mixed6_5.dot"),
    ("datasets/UB_mixed_chunk_20_5_5.dot", "datasets/fixed6_5.dot"),
    ("datasets/imbalanced_fixed_30.dot", "datasets/fixed5_6.dot"),
]

DEFAULT_STRATEGIES = ["syncmap", "rg", "nsgd", "nsgd_unfreeze_at_switch", "nsgd_no_freeze"]

STAGES: dict[str, dict[str, Any]] = {
    "quick": {"pre_steps": 20_000, "post_steps": 20_000, "interval": 1_000, "seeds": [0]},
    "guard": {"pre_steps": 100_000, "post_steps": 100_000, "interval": 5_000, "seeds": [0, 1, 2]},
    "long": {"pre_steps": 500_000, "post_steps": 500_000, "interval": 10_000, "seeds": [0, 1, 2]},
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    runtime: str
    params: dict[str, Any]
    switch_params: dict[str, Any] | None = None
    reset_plasticity_at_switch: bool = False


def _strategy_spec(name: str) -> StrategySpec:
    key = str(name).strip().lower()
    no_freeze = {
        "plasticity_decay": 1.0,
        "plasticity_min": 1.0,
        "plasticity_recovery": 1.0,
    }
    if key == "nsgd_unfreeze_at_switch":
        return StrategySpec(
            key,
            "nsgd",
            dict(VARIANT_CONFIGS["nsgd"]["params"]),
            switch_params=no_freeze,
            reset_plasticity_at_switch=True,
        )
    if key == "nsgd_no_freeze":
        params = dict(VARIANT_CONFIGS["nsgd"]["params"])
        params.update(no_freeze)
        return StrategySpec(key, "nsgd", params)
    if key == "nsgd_mem300":
        params = dict(VARIANT_CONFIGS["nsgd"]["params"])
        params["stats_memory_steps"] = 300
        return StrategySpec(key, "nsgd", params)
    if key not in {"syncmap", "rg", "nsgd"}:
        raise ValueError(f"Unknown strategy {name!r}; choose from {DEFAULT_STRATEGIES}")
    profile = VARIANT_CONFIGS[key]
    return StrategySpec(key, str(profile["public_runtime"]), dict(profile["params"]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sequence_digest(sequence: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(sequence)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _evaluate(model: Any, env: Any) -> dict[str, float]:
    metric = resolve_metric("ari")
    result = _select_evaluation_prediction(
        model,
        env,
        metric,
        readout_config=EvaluationReadoutConfig(),
    )
    labels = np.asarray(result["labels"], dtype=int)
    true_labels = np.asarray(result["true_labels"], dtype=int)
    scores = compute_all_metric_scores(labels, true_labels)
    diagnostics = summarize_predicted_labels(labels)
    return {
        "ari": float(scores["ari_score"]),
        "nmi": float(scores["nmi_score"]),
        "predicted_cluster_count": float(diagnostics["cluster_count"]),
        "noise_ratio": float(diagnostics["noise_ratio"]),
    }


def _plasticity_diagnostics(model: Any) -> dict[str, float]:
    values = getattr(model, "plasticity", None)
    if values is None:
        return {
            "plasticity_mean": float("nan"),
            "plasticity_min_observed": float("nan"),
            "plasticity_frozen_fraction": float("nan"),
            "plasticity_at_floor_fraction": float("nan"),
            "plasticity_fully_mobile_fraction": float("nan"),
            "plasticity_q10": float("nan"),
            "plasticity_q50": float("nan"),
            "plasticity_q90": float("nan"),
            "internal_core_fraction": float("nan"),
        }
    arr = np.asarray(values, dtype=float).reshape(-1)
    floor = float(getattr(model, "plasticity_min", np.nan))
    labels = getattr(model, "labels", None)
    internal_core_fraction = (
        float(np.mean(np.asarray(labels, dtype=int).reshape(-1) >= 0))
        if labels is not None
        else float("nan")
    )
    return {
        "plasticity_mean": float(np.mean(arr)),
        "plasticity_min_observed": float(np.min(arr)),
        "plasticity_frozen_fraction": float(np.mean(arr <= 0.05)),
        "plasticity_at_floor_fraction": (
            float(np.mean(arr <= floor + 1e-12))
            if np.isfinite(floor)
            else float("nan")
        ),
        "plasticity_fully_mobile_fraction": float(np.mean(arr >= 0.99)),
        "plasticity_q10": float(np.quantile(arr, 0.10)),
        "plasticity_q50": float(np.quantile(arr, 0.50)),
        "plasticity_q90": float(np.quantile(arr, 0.90)),
        "internal_core_fraction": internal_core_fraction,
    }


class _StepMotionMonitor:
    """Observe actual NSGD coordinate motion without changing its update."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.original_step_once = model.step_once
        self.reset()
        model.step_once = self._step_once

    def reset(self) -> None:
        self.physics_steps = 0
        self.node_updates = 0
        self.nonzero_node_updates = 0
        self.path_sum = 0.0
        self.path_max = 0.0
        self.plasticity_sum = 0.0

    def _step_once(self) -> None:
        before = np.asarray(self.model.syncmap, dtype=float).copy()
        plasticity = np.asarray(getattr(self.model, "plasticity", []), dtype=float)
        if plasticity.size:
            self.plasticity_sum += float(np.mean(plasticity))
        self.original_step_once()
        displacement = np.linalg.norm(np.asarray(self.model.syncmap, dtype=float) - before, axis=1)
        self.physics_steps += 1
        self.node_updates += int(displacement.size)
        self.nonzero_node_updates += int(np.count_nonzero(displacement > 1e-12))
        self.path_sum += float(np.sum(displacement))
        if displacement.size:
            self.path_max = max(self.path_max, float(np.max(displacement)))

    def diagnostics(self) -> dict[str, float]:
        return {
            "physics_steps_observed": float(self.physics_steps),
            "actual_motion_mean_per_node_step": (
                self.path_sum / float(self.node_updates)
                if self.node_updates
                else float("nan")
            ),
            "actual_motion_max_node_step": (
                self.path_max if self.node_updates else float("nan")
            ),
            "actual_motion_nonzero_fraction": (
                self.nonzero_node_updates / float(self.node_updates)
                if self.node_updates
                else float("nan")
            ),
            "plasticity_mean_during_updates": (
                self.plasticity_sum / float(self.physics_steps)
                if self.physics_steps
                else float("nan")
            ),
        }


def _install_step_motion_monitor(model: Any) -> _StepMotionMonitor | None:
    if not hasattr(model, "plasticity") or not callable(getattr(model, "step_once", None)):
        return None
    return _StepMotionMonitor(model)


def _coactivation_similarity(model: Any) -> np.ndarray | None:
    coact = getattr(model, "_coact", None)
    if coact is None or not callable(getattr(coact, "similarity", None)):
        return None
    return np.asarray(coact.similarity(), dtype=float).copy()


def _edge_keys(model: Any) -> set[tuple[int, int]]:
    rows = getattr(model, "_g_rows", None)
    cols = getattr(model, "_g_cols", None)
    if rows is None or cols is None:
        return set()
    return {
        (min(int(row), int(col)), max(int(row), int(col)))
        for row, col in zip(np.asarray(rows).reshape(-1), np.asarray(cols).reshape(-1))
    }


def _capture_reference_state(model: Any) -> dict[str, Any]:
    return {
        "syncmap": np.asarray(model.syncmap, dtype=float).copy(),
        "coactivation_similarity": _coactivation_similarity(model),
        "edge_keys": _edge_keys(model),
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).reshape(-1)
    right_flat = np.asarray(right, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left_flat, right_flat) / denominator)


def _model_diagnostics(
    model: Any,
    *,
    previous_syncmap: np.ndarray,
    reference_state: dict[str, Any],
    motion_monitor: _StepMotionMonitor | None,
) -> dict[str, float]:
    syncmap = np.asarray(model.syncmap, dtype=float)
    net_displacement = np.linalg.norm(syncmap - previous_syncmap, axis=1)
    reference_displacement = np.linalg.norm(syncmap - np.asarray(reference_state["syncmap"]), axis=1)
    scale = max(float(getattr(model, "space_size", 1.0)), 1e-12)
    coact = getattr(model, "_coact", None)
    similarity = _coactivation_similarity(model)
    reference_similarity = reference_state.get("coactivation_similarity")
    edges = _edge_keys(model)
    reference_edges = set(reference_state.get("edge_keys", set()))
    union = edges | reference_edges
    weights = getattr(model, "_g_w", None)
    result = {
        "model_step_count": float(getattr(model, "_step_count", np.nan)),
        "coordinate_net_mean": float(np.mean(net_displacement)),
        "coordinate_net_max": float(np.max(net_displacement)),
        "coordinate_net_mean_rel_space": float(np.mean(net_displacement) / scale),
        "coordinate_from_reference_mean": float(np.mean(reference_displacement)),
        "coordinate_from_reference_mean_rel_space": float(np.mean(reference_displacement) / scale),
        "relation_edge_count": float(len(edges)),
        "relation_edge_jaccard_to_reference": (
            float(len(edges & reference_edges) / len(union))
            if union
            else 1.0
        ),
        "relation_weight_mean": (
            float(np.mean(np.asarray(weights, dtype=float)))
            if weights is not None and np.asarray(weights).size
            else float("nan")
        ),
        "stats_alpha_effective": (
            float(getattr(coact, "alpha"))
            if coact is not None and hasattr(coact, "alpha")
            else float("nan")
        ),
        "stats_memory_steps_effective": (
            float(getattr(model, "stats_memory_steps"))
            if hasattr(model, "stats_memory_steps")
            else (
                float(1.0 / getattr(coact, "alpha"))
                if coact is not None and float(getattr(coact, "alpha", 0.0)) > 1e-12
                else float("nan")
            )
        ),
        "coactivation_similarity_mean": (
            float(np.mean(similarity)) if similarity is not None else float("nan")
        ),
        "coactivation_cosine_to_reference": (
            _cosine_similarity(similarity, reference_similarity)
            if similarity is not None and reference_similarity is not None
            else float("nan")
        ),
        "coactivation_relative_change_from_reference": (
            float(
                np.linalg.norm(similarity - reference_similarity)
                / max(float(np.linalg.norm(reference_similarity)), 1e-12)
            )
            if similarity is not None and reference_similarity is not None
            else float("nan")
        ),
    }
    result.update(_plasticity_diagnostics(model))
    if motion_monitor is not None:
        result.update(motion_monitor.diagnostics())
    else:
        result.update({
            "physics_steps_observed": float("nan"),
            "actual_motion_mean_per_node_step": float("nan"),
            "actual_motion_max_node_step": float("nan"),
            "actual_motion_nonzero_fraction": float("nan"),
            "plasticity_mean_during_updates": float("nan"),
        })
    return result


def _apply_switch_control(model: Any, spec: StrategySpec) -> None:
    if spec.switch_params:
        for key, value in spec.switch_params.items():
            setattr(model, key, float(value))
    if spec.reset_plasticity_at_switch and hasattr(model, "plasticity"):
        np.asarray(model.plasticity)[:] = 1.0


def _train_curve(
    model: Any,
    env: Any,
    sequence: np.ndarray,
    *,
    interval: int,
    phase_offset: int,
    phase: str,
    include_zero: bool,
    reference_state: dict[str, Any],
    motion_monitor: _StepMotionMonitor | None,
) -> list[dict[str, Any]]:
    total = int(sequence.shape[0])
    rows: list[dict[str, Any]] = []
    previous_syncmap = np.asarray(model.syncmap, dtype=float).copy()
    if motion_monitor is not None:
        motion_monitor.reset()
    if include_zero:
        rows.append({
            "iteration": 0,
            "phase": phase,
            **_evaluate(model, env),
            **_model_diagnostics(
                model,
                previous_syncmap=previous_syncmap,
                reference_state=reference_state,
                motion_monitor=motion_monitor,
            ),
        })
    feed_chunk = max(1, int(getattr(model, "step_chunk", 100)))
    for start in range(0, total, int(interval)):
        end = min(start + int(interval), total)
        if motion_monitor is not None:
            motion_monitor.reset()
        for chunk_start in range(start, end, feed_chunk):
            chunk_end = min(chunk_start + feed_chunk, end)
            _advance_model_chunk(
                model,
                sequence[chunk_start:chunk_end],
                phase_offset=int(phase_offset + chunk_start),
                reset_trajectory=bool(phase_offset == 0 and chunk_start == 0),
                append_final_frame=bool(chunk_end >= total),
            )
        rows.append({
            "iteration": end,
            "phase": phase,
            **_evaluate(model, env),
            **_model_diagnostics(
                model,
                previous_syncmap=previous_syncmap,
                reference_state=reference_state,
                motion_monitor=motion_monitor,
            ),
        })
        previous_syncmap = np.asarray(model.syncmap, dtype=float).copy()
    return rows


def _late_values(curve: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in curve], dtype=float)
    if values.size == 0:
        return values
    count = max(3, int(np.ceil(values.size * 0.25)))
    return values[-min(count, values.size):]


def _curve_summary(curve: list[dict[str, Any]], metric: str, horizon: int) -> dict[str, Any]:
    x = np.asarray([float(row["iteration"]) for row in curve], dtype=float)
    y = np.asarray([float(row[metric]) for row in curve], dtype=float)
    late = _late_values(curve, metric)
    auc_mean = float(np.trapezoid(y, x) / max(float(horizon), 1.0)) if y.size >= 2 else float(y[-1])
    return {
        "initial": float(y[0]),
        "final": float(y[-1]),
        "peak": float(np.max(y)),
        "late_mean": float(np.mean(late)),
        "late_std": float(np.std(late)),
        "stable_score": float(np.mean(late) - np.std(late)),
        "auc_mean": auc_mean,
    }


def _recovery_time(
    transfer_curve: list[dict[str, Any]],
    target_only_summary: dict[str, Any],
    metric: str,
) -> int | None:
    ceiling = float(target_only_summary["late_mean"])
    if ceiling < 0.10:
        return None
    threshold = 0.90 * ceiling
    for index in range(len(transfer_curve) - 1):
        if (
            float(transfer_curve[index][metric]) >= threshold
            and float(transfer_curve[index + 1][metric]) >= threshold
        ):
            return int(transfer_curve[index]["iteration"])
    return None


def _summarize_case(
    source_curve: list[dict[str, Any]],
    transfer_curve: list[dict[str, Any]],
    target_only_curve: list[dict[str, Any]],
    *,
    pre_steps: int,
    post_steps: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("ari", "nmi"):
        source = _curve_summary(source_curve, metric, pre_steps)
        transfer = _curve_summary(transfer_curve, metric, post_steps)
        target_only = _curve_summary(target_only_curve, metric, post_steps)
        denominator = float(target_only["auc_mean"])
        ratio = None if denominator < 0.10 else float(transfer["auc_mean"] / denominator)
        summary[metric] = {
            "source": source,
            "transfer": transfer,
            "target_only": target_only,
            "normalized_auc_ratio": ratio,
            "auc_advantage": float(transfer["auc_mean"] - target_only["auc_mean"]),
            "recovery_time": _recovery_time(transfer_curve, target_only, metric),
        }
    source_ari = summary["ari"]["source"]
    summary["source_ready"] = bool(
        float(source_ari["late_mean"]) >= max(0.10, 0.80 * float(source_ari["peak"]))
        and float(source_ari["late_std"]) <= 0.20
    )
    return summary


def _resolve_pairs(raw_pairs: Iterable[str] | None) -> list[tuple[str, str]]:
    pairs = DEFAULT_PAIRS if not raw_pairs else []
    if raw_pairs:
        for item in raw_pairs:
            if ":" not in item:
                raise ValueError(f"Invalid pair {item!r}; expected source.dot:target.dot")
            source, target = item.split(":", 1)
            pairs.append((source, target))
    resolved: list[tuple[str, str]] = []
    for source, target in pairs:
        source_path = str(resolve_dataset_path(source))
        target_path = str(resolve_dataset_path(target))
        resolved.append((source_path, target_path))
    return resolved


def _run_case(
    source: str,
    target: str,
    spec: StrategySpec,
    *,
    pre_steps: int,
    post_steps: int,
    interval: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    np.random.seed(int(seed))
    transfer_model, transfer_env = create_model(
        source,
        spec.runtime,
        dict(spec.params),
        record_trajectory=False,
        environment_seed=int(seed),
    )
    transfer_motion_monitor = _install_step_motion_monitor(transfer_model)
    source_reference = _capture_reference_state(transfer_model)
    with _isolated_environment_rng(transfer_env):
        source_sequence, _ = transfer_env.getSequence(int(pre_steps))
    source_true_labels = np.asarray(transfer_env.trueLabel(), dtype=int).copy()
    source_curve = _train_curve(
        transfer_model,
        transfer_env,
        source_sequence,
        interval=interval,
        phase_offset=0,
        phase="source",
        include_zero=False,
        reference_state=source_reference,
        motion_monitor=transfer_motion_monitor,
    )

    switch_reference = _capture_reference_state(transfer_model)
    plasticity_before_switch_control = _plasticity_diagnostics(transfer_model)
    with _isolated_environment_rng(transfer_env):
        transfer_env.switch_dataset(target)
        target_sequence, _ = transfer_env.getSequence(int(post_steps))
    target_true_labels = np.asarray(transfer_env.trueLabel(), dtype=int).copy()
    source_target_label_ari = float(adjusted_rand_score(source_true_labels, target_true_labels))
    _apply_switch_control(transfer_model, spec)
    plasticity_after_switch_control = _plasticity_diagnostics(transfer_model)
    target_digest = _sequence_digest(target_sequence)
    transfer_curve = _train_curve(
        transfer_model,
        transfer_env,
        target_sequence,
        interval=interval,
        phase_offset=pre_steps,
        phase="post_switch",
        include_zero=True,
        reference_state=switch_reference,
        motion_monitor=transfer_motion_monitor,
    )

    np.random.seed(int(seed))
    target_model, target_env = create_model(
        target,
        spec.runtime,
        dict(spec.params),
        record_trajectory=False,
        environment_seed=int(seed),
    )
    target_motion_monitor = _install_step_motion_monitor(target_model)
    _apply_switch_control(target_model, spec)
    target_reference = _capture_reference_state(target_model)
    target_only_curve = _train_curve(
        target_model,
        target_env,
        target_sequence,
        interval=interval,
        phase_offset=0,
        phase="target_only",
        include_zero=True,
        reference_state=target_reference,
        motion_monitor=target_motion_monitor,
    )

    summary = _summarize_case(
        source_curve,
        transfer_curve,
        target_only_curve,
        pre_steps=pre_steps,
        post_steps=post_steps,
    )
    final_plasticity = _plasticity_diagnostics(transfer_model)
    record = {
        "source": Path(source).name,
        "target": Path(target).name,
        "source_path": source,
        "target_path": target,
        "strategy": spec.name,
        "runtime_strategy": spec.runtime,
        "seed": int(seed),
        "pre_steps": int(pre_steps),
        "post_steps": int(post_steps),
        "interval": int(interval),
        "target_sequence_sha256": target_digest,
        "source_target_label_ari": source_target_label_ari,
        "partition_shift": bool(source_target_label_ari < 0.999999),
        "params": dict(spec.params),
        "switch_params": spec.switch_params,
        "reset_plasticity_at_switch": bool(spec.reset_plasticity_at_switch),
        "effective_model_params": getattr(transfer_model, "effective_model_params", None),
        "effective_env_params": getattr(transfer_model, "effective_env_params", None),
        "switch_control_effect": {
            "before": plasticity_before_switch_control,
            "after": plasticity_after_switch_control,
        },
        "summary": summary,
        "final_plasticity": final_plasticity,
        "elapsed_time": float(time.time() - started),
    }
    curve_rows: list[dict[str, Any]] = []
    for condition, curve in (
        ("source", source_curve),
        ("post_switch", transfer_curve),
        ("target_only", target_only_curve),
    ):
        for row in curve:
            curve_rows.append({
                "source": Path(source).name,
                "target": Path(target).name,
                "strategy": spec.name,
                "runtime_strategy": spec.runtime,
                "seed": int(seed),
                "condition": condition,
                "target_sequence_sha256": target_digest if condition != "source" else "",
                **row,
            })
    return record, curve_rows


def _flat_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in records:
        switch_effect = row.get("switch_control_effect", {})
        switch_before = switch_effect.get("before", {})
        switch_after = switch_effect.get("after", {})
        effective_model_params = row.get("effective_model_params") or {}
        effective_env_params = row.get("effective_env_params") or {}
        item: dict[str, Any] = {
            "source": row["source"],
            "target": row["target"],
            "strategy": row["strategy"],
            "seed": row["seed"],
            "source_ready": row["summary"]["source_ready"],
            "source_target_label_ari": row.get("source_target_label_ari", float("nan")),
            "partition_shift": row.get("partition_shift", False),
            "elapsed_time": row["elapsed_time"],
            "switch_plasticity_mean_before": switch_before.get("plasticity_mean", float("nan")),
            "switch_plasticity_mean_after": switch_after.get("plasticity_mean", float("nan")),
            "switch_frozen_fraction_before": switch_before.get("plasticity_frozen_fraction", float("nan")),
            "switch_frozen_fraction_after": switch_after.get("plasticity_frozen_fraction", float("nan")),
            "stats_memory_steps_effective": effective_model_params.get("stats_memory_steps", float("nan")),
            "env_time_delay": effective_env_params.get("env_time_delay", float("nan")),
            "env_history_length": effective_env_params.get("env_history_length", float("nan")),
            **row["final_plasticity"],
        }
        for metric in ("ari", "nmi"):
            data = row["summary"][metric]
            item[f"{metric}_post_auc_mean"] = data["transfer"]["auc_mean"]
            item[f"{metric}_target_only_auc_mean"] = data["target_only"]["auc_mean"]
            item[f"{metric}_normalized_auc_ratio"] = data["normalized_auc_ratio"]
            item[f"{metric}_auc_advantage"] = data["auc_advantage"]
            item[f"{metric}_recovery_time"] = data["recovery_time"]
            item[f"{metric}_post_late_mean"] = data["transfer"]["late_mean"]
            item[f"{metric}_post_late_std"] = data["transfer"]["late_std"]
        rows.append(item)
    return pd.DataFrame(rows)


def _aggregate_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _flat_summary(records)
    if frame.empty:
        return frame
    value_columns = [
        column for column in frame.columns
        if column.startswith(("ari_", "nmi_")) and column not in {"ari_recovery_time", "nmi_recovery_time"}
    ]
    aggregations: dict[str, list[str]] = {column: ["mean", "std"] for column in value_columns}
    aggregations["source_ready"] = ["mean", "sum", "count"]
    grouped = frame.groupby(["source", "target", "strategy"], dropna=False).agg(aggregations)
    grouped.columns = ["_".join(parts) for parts in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def _nsgd_freeze_effect(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _flat_summary(records)
    if frame.empty:
        return frame
    keys = ["source", "target", "seed"]
    original = frame[frame["strategy"] == "nsgd"].set_index(keys)
    rows: list[dict[str, Any]] = []
    for candidate_name in ("nsgd_unfreeze_at_switch", "nsgd_no_freeze"):
        candidate_frame = frame[frame["strategy"] == candidate_name].set_index(keys)
        shared = original.index.intersection(candidate_frame.index)
        for key in shared:
            base = original.loc[key]
            candidate = candidate_frame.loc[key]
            rows.append({
                "source": key[0],
                "target": key[1],
                "seed": int(key[2]),
                "candidate": candidate_name,
                "source_target_label_ari": float(base["source_target_label_ari"]),
                "partition_shift": bool(base["partition_shift"]),
                "source_ready_nsgd": bool(base["source_ready"]),
                "source_ready_candidate": bool(candidate["source_ready"]),
                "ari_post_auc_delta_candidate_minus_nsgd": float(candidate["ari_post_auc_mean"] - base["ari_post_auc_mean"]),
                "ari_post_late_mean_delta_candidate_minus_nsgd": float(candidate["ari_post_late_mean"] - base["ari_post_late_mean"]),
                "ari_post_late_std_delta_candidate_minus_nsgd": float(candidate["ari_post_late_std"] - base["ari_post_late_std"]),
                "nmi_post_auc_delta_candidate_minus_nsgd": float(candidate["nmi_post_auc_mean"] - base["nmi_post_auc_mean"]),
            })
    return pd.DataFrame(rows)


def _nsgd_memory_effect(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _flat_summary(records)
    if frame.empty:
        return frame
    keys = ["source", "target", "seed"]
    original = frame[frame["strategy"] == "nsgd"].set_index(keys)
    candidate = frame[frame["strategy"] == "nsgd_mem300"].set_index(keys)
    rows: list[dict[str, Any]] = []
    for key in original.index.intersection(candidate.index):
        base = original.loc[key]
        memory = candidate.loc[key]
        rows.append({
            "source": key[0],
            "target": key[1],
            "seed": int(key[2]),
            "source_ready_nsgd": bool(base["source_ready"]),
            "source_ready_nsgd_mem300": bool(memory["source_ready"]),
            "ari_post_auc_delta_mem300_minus_mem100": float(
                memory["ari_post_auc_mean"] - base["ari_post_auc_mean"]
            ),
            "ari_post_late_mean_delta_mem300_minus_mem100": float(
                memory["ari_post_late_mean"] - base["ari_post_late_mean"]
            ),
            "nmi_post_auc_delta_mem300_minus_mem100": float(
                memory["nmi_post_auc_mean"] - base["nmi_post_auc_mean"]
            ),
        })
    return pd.DataFrame(rows)


def _fairness_audit(records: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row["source"]), str(row["target"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (source, target, seed), group in grouped.items():
        digests = {str(row["target_sequence_sha256"]) for row in group}
        pre_steps = {int(row["pre_steps"]) for row in group}
        post_steps = {int(row["post_steps"]) for row in group}
        intervals = {int(row["interval"]) for row in group}
        label_ari = {round(float(row["source_target_label_ari"]), 12) for row in group}
        env_contracts = {
            json.dumps(_jsonable(row.get("effective_env_params")), sort_keys=True)
            for row in group
        }
        rows.append({
            "source": source,
            "target": target,
            "seed": seed,
            "strategy_count": len(group),
            "target_sequence_digest_count": len(digests),
            "pre_step_count": len(pre_steps),
            "post_step_count": len(post_steps),
            "interval_count": len(intervals),
            "source_target_label_ari_count": len(label_ari),
            "environment_contract_count": len(env_contracts),
            "fairness_pass": bool(
                len(digests) == 1
                and len(pre_steps) == 1
                and len(post_steps) == 1
                and len(intervals) == 1
                and len(label_ari) == 1
                and len(env_contracts) == 1
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default="quick")
    parser.add_argument("--pairs", nargs="+", default=None, metavar="SOURCE:TARGET")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--post-steps", type=int, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume an explicit --output-dir after exact matrix validation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage = STAGES[str(args.stage)]
    pre_steps = int(args.pre_steps if args.pre_steps is not None else stage["pre_steps"])
    post_steps = int(args.post_steps if args.post_steps is not None else stage["post_steps"])
    interval = int(args.interval if args.interval is not None else stage["interval"])
    seeds = list(args.seeds if args.seeds is not None else stage["seeds"])
    pairs = _resolve_pairs(args.pairs)
    specs = [_strategy_spec(name) for name in args.strategies]
    matrix = {
        "stage": args.stage,
        "pairs": pairs,
        "strategies": [spec.name for spec in specs],
        "pre_steps": pre_steps,
        "post_steps": post_steps,
        "interval": interval,
        "seeds": seeds,
        "cases": len(pairs) * len(specs) * len(seeds),
        "primary_metric": "post-switch ARI AUC normalized to exact-sequence target-only control",
        "supplemental_metric": "NMI",
    }
    print(json.dumps(matrix, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else experiment_archive_dir(timestamp_now()) / "graph_switch_plasticity_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    curves_path = output_dir / "curves.csv"
    summary_path = output_dir / "summary.csv"
    aggregate_path = output_dir / "aggregate.csv"
    freeze_effect_path = output_dir / "nsgd_freeze_effect.csv"
    memory_effect_path = output_dir / "nsgd_memory_effect.csv"
    fairness_path = output_dir / "fairness_audit.csv"
    if args.resume:
        if not json_path.exists():
            raise FileNotFoundError(f"Cannot resume without {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("matrix") != _jsonable(matrix):
            raise ValueError("Resume matrix does not match the saved graph-switch experiment")
        payload["status"] = "running"
        curve_rows = pd.read_csv(curves_path).to_dict("records") if curves_path.exists() else []
    else:
        payload = {"status": "running", "matrix": matrix, "records": []}
        curve_rows = []
    completed = {
        (row["source"], row["target"], row["strategy"], int(row["seed"]))
        for row in payload["records"]
    }
    total = int(matrix["cases"])
    count = 0
    for source, target in pairs:
        for spec in specs:
            for seed in seeds:
                count += 1
                key = (Path(source).name, Path(target).name, spec.name, int(seed))
                if key in completed:
                    print(f"[{count}/{total}] skip completed {key}", flush=True)
                    continue
                print(f"[{count}/{total}] {Path(source).name} -> {Path(target).name} {spec.name} seed={seed}", flush=True)
                record, new_curve_rows = _run_case(
                    source,
                    target,
                    spec,
                    pre_steps=pre_steps,
                    post_steps=post_steps,
                    interval=interval,
                    seed=int(seed),
                )
                payload["records"].append(record)
                completed.add(key)
                curve_rows.extend(new_curve_rows)
                json_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
                pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
                _flat_summary(payload["records"]).to_csv(summary_path, index=False)
                _aggregate_summary(payload["records"]).to_csv(aggregate_path, index=False)
                _nsgd_freeze_effect(payload["records"]).to_csv(freeze_effect_path, index=False)
                _nsgd_memory_effect(payload["records"]).to_csv(memory_effect_path, index=False)
                _fairness_audit(payload["records"]).to_csv(fairness_path, index=False)
                ari = record["summary"]["ari"]
                print(
                    f"  source_ready={record['summary']['source_ready']} "
                    f"post_AUC={ari['transfer']['auc_mean']:.3f} "
                    f"target_AUC={ari['target_only']['auc_mean']:.3f} "
                    f"ratio={ari['normalized_auc_ratio']} time={record['elapsed_time']:.1f}s",
                    flush=True,
                )
    payload["status"] = "complete"
    json_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
