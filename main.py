#!/usr/bin/env python
"""
Public CLI entrypoint and visible run-control panel for SyncMap experiments.

Design rules for this branch:
- `main.py` is the only public CLI entrypoint.
- `live.py` is viewer backend code, not a second public CLI surface.
- CLI exposes task-level controls plus a generic `--set key=value` override path.
- Algorithm-internal fields should not grow into dozens of dedicated top-level flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, cast

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from tools.result_paths import (
    artifact_result_path,
    classify_output_token,
    sequence_info_result_path,
    timestamp_now,
    trace_result_paths,
)
from unified_experiments import (
    COMPARISON_DISPLAY_NAME_KEY,
    COMPARISON_ENGINE_STRATEGY_KEY,
    COMPARISON_PROFILE_NAME_KEY,
    _save_learning_curve_json,
    _save_learning_curve_plot,
    EvaluationMetricName,
    create_strategy_config,
    generic_hierarchy_outputs,
    get_all_datasets,
    infer_default_env_mode,
    resolve_dataset_path,
    resolve_metric,
    run_all_experiments as run_all_experiments_lib,
    run_single_test as run_single_test_lib,
)


from tools.main_config import (
    DEFAULT_ACCENT_COLOR,
    HIERARCHY_DATASET_SUITE,
    DEFAULT_METRIC,
    DEFAULT_TASKNUM,
    DEFAULT_THEME,
    MAIN_CONFIG,
    MEMORY_PROFILE_CHOICES,
    TASK_ALIASES,
    VARIANT_CONFIGS,
)




class _RawDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


@dataclass(frozen=True)
class ProfileRunSpec:
    profile_name: str
    strategy_name: str
    strategy_params: dict[str, Any]
    summary: str


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(cast(dict[str, Any], target[key]), cast(dict[str, Any], value))
            continue
        target[key] = value
    return target


def _resolved_main_config() -> dict[str, Any]:
    config = deepcopy(MAIN_CONFIG)
    config["dataset"] = resolve_dataset_path(config["dataset"])
    return config


def _comparison_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("comparison")
    if isinstance(block, dict):
        return dict(block)
    return {}


def _strategy_defaults_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparison = _comparison_block(config)
    strategy_defaults = comparison.get("strategy_defaults", config.get("strategy_defaults"))
    if isinstance(strategy_defaults, dict) and strategy_defaults:
        return {
            str(name): dict(params) if isinstance(params, dict) else {}
            for name, params in strategy_defaults.items()
        }
    return {}


def _default_strategy_from_config(config: dict[str, Any]) -> str:
    explicit_default = config.get("default_strategy")
    if explicit_default not in {None, ""}:
        return str(explicit_default)
    comparison_block = _comparison_block(config)
    comparison = comparison_block.get("strategies", config.get("comparison_strategies"))
    if isinstance(comparison, (list, tuple)):
        for item in comparison:
            text = str(item).strip()
            if text:
                return text
    defaults = _strategy_defaults_from_config(config)
    if defaults:
        return next(iter(defaults))
    return "rg"


def _first_strategy_arg(raw: Any) -> Optional[str]:
    """Extract a single strategy name from CLI nargs+ output."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if len(raw) >= 1 else None
    return str(raw)


def _ordered_unique(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_memory_profile(value: Any) -> Optional[str]:
    if value in {None, "", "auto"}:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in MEMORY_PROFILE_CHOICES:
        choices = ", ".join(MEMORY_PROFILE_CHOICES)
        raise ValueError(f"Unsupported memory profile: {value}. Valid values: {choices}")
    return normalized


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


def _parse_cli_bool(value: str | int | bool | None) -> bool:
    parsed = _parse_bool_token(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"Expected true/false/on/off/1/0, got: {value!r}")
    return bool(parsed)


def _parse_scalar_token(raw: str) -> Any:
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_set_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"Invalid --set override {item!r}; expected key=value.")
        key, raw_value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --set override {item!r}; empty key.")
        overrides[key] = _parse_scalar_token(raw_value)
    return overrides


def _memory_profile_overrides(runtime_name: str, memory_profile: Optional[str]) -> dict[str, Any]:
    profile = _normalize_memory_profile(memory_profile)
    if profile is None or profile == "historical":
        return {}
    runtime = str(runtime_name).strip().lower()
    if profile == "transition":
        if runtime in {"ns", "rg"}:
            return {"stats_mode": "transition"}
        return {}
    if profile in {"coactivation", "ema"}:
        if runtime in {"ns", "rg"}:
            return {"stats_mode": "coactivation"}
        return {}
    return {}


def list_target_profiles() -> list[str]:
    return list(VARIANT_CONFIGS.keys())


def resolve_profile_run_spec(
    profile_name: str,
    *,
    env_mode: Optional[str] = None,
    override_params: Optional[dict[str, Any]] = None,
) -> ProfileRunSpec:
    normalized = str(profile_name).strip().lower()
    if normalized == "strict":
        raise KeyError(
            "Renamed profile 'strict': use --profile syncmap for the original SyncMap baseline."
        )
    if normalized not in VARIANT_CONFIGS:
        available = ", ".join(list_target_profiles())
        raise KeyError(f"Unknown profile '{profile_name}'. Available: {available}")
    profile = VARIANT_CONFIGS[normalized]
    params = dict(cast(dict[str, Any], profile.get("params", {})))
    if env_mode not in {None, "", "auto"}:
        params["env_mode"] = str(env_mode)
    memory_profile = _normalize_memory_profile((override_params or {}).get("memory_profile"))
    params.update(_memory_profile_overrides(str(profile["public_runtime"]), memory_profile))
    for key, value in (override_params or {}).items():
        if key == "memory_profile":
            continue
        params[str(key)] = value
    return ProfileRunSpec(
        profile_name=normalized,
        strategy_name=str(profile["public_runtime"]),
        strategy_params=params,
        summary=str(profile["summary"]),
    )


def _resolve_runtime_selection(
    config: dict[str, Any],
    *,
    dataset: str,
    strategy: Optional[str] = None,
    profile: Optional[str] = None,
    env_mode: Optional[str] = None,
    override_params: Optional[dict[str, Any]] = None,
) -> tuple[Optional[str], str, str, dict[str, Any]]:
    if profile not in {None, ""} and strategy not in {None, ""}:
        raise SystemExit("Use either --profile or --strategy, not both.")
    if profile not in {None, ""}:
        spec = resolve_profile_run_spec(str(profile), override_params=override_params)
        strategy_name = str(spec.strategy_name).lower()
        resolved_env_mode = infer_default_env_mode(dataset, strategy_name) if env_mode in {None, "", "auto"} else str(env_mode)
        spec = resolve_profile_run_spec(str(profile), env_mode=resolved_env_mode, override_params=override_params)
        return spec.profile_name, strategy_name, resolved_env_mode, dict(spec.strategy_params)

    strategy_name = str(strategy if strategy not in {None, ""} else _default_strategy_from_config(config)).lower()
    # If the name matches a VARIANT_CONFIGS profile entry, route through profile resolution
    # so that --strategy rg behaves identically to --profile rg.
    if strategy_name in VARIANT_CONFIGS:
        spec = resolve_profile_run_spec(strategy_name, override_params=override_params)
        real_strategy_name = str(spec.strategy_name).lower()
        resolved_env_mode = infer_default_env_mode(dataset, real_strategy_name) if env_mode in {None, "", "auto"} else str(env_mode)
        spec = resolve_profile_run_spec(strategy_name, env_mode=resolved_env_mode, override_params=override_params)
        return spec.profile_name, real_strategy_name, resolved_env_mode, dict(spec.strategy_params)
    resolved_env_mode = infer_default_env_mode(dataset, strategy_name) if env_mode in {None, "", "auto"} else str(env_mode)
    params = dict(_strategy_defaults_from_config(config).get(strategy_name, {}))
    params.update(_memory_profile_overrides(strategy_name, _normalize_memory_profile((override_params or {}).get("memory_profile"))))
    for key, value in (override_params or {}).items():
        if key == "memory_profile":
            continue
        params[str(key)] = value
    params["env_mode"] = resolved_env_mode
    return None, strategy_name, resolved_env_mode, params


def _comparison_profiles_from_config(config: dict[str, Any], selected: Optional[list[str]] = None) -> list[str]:
    if selected:
        return _ordered_unique(selected)
    comparison_block = _comparison_block(config)
    comparison = comparison_block.get("profiles", config.get("comparison_profiles"))
    if isinstance(comparison, (list, tuple)) and comparison:
        return _ordered_unique([str(item) for item in comparison])
    return []


def _comparison_strategies_from_config(config: dict[str, Any], selected: Optional[list[str]] = None) -> list[str]:
    if selected:
        return _ordered_unique(selected)
    comparison_block = _comparison_block(config)
    comparison = comparison_block.get("strategies", config.get("comparison_strategies"))
    if isinstance(comparison, (list, tuple)) and comparison:
        return _ordered_unique([str(item) for item in comparison])
    return [_default_strategy_from_config(config)]


def _comparison_sequence_lengths(config: dict[str, Any]) -> list[int]:
    comparison = _comparison_block(config)
    raw = comparison.get("sequence_lengths", config.get("all_sequence_lengths", [5000, 100000]))
    return [int(v) for v in raw]


def _comparison_trials(config: dict[str, Any]) -> int:
    comparison = _comparison_block(config)
    return int(comparison.get("trials", config.get("comparison_trials", 5)))


def _comparison_metric_name(config: dict[str, Any]) -> EvaluationMetricName:
    comparison = _comparison_block(config)
    return cast(EvaluationMetricName, str(comparison.get("metric_name", config.get("all_metric_name", DEFAULT_METRIC))))


def _normalize_label_source(label_source: str | None) -> str:
    token = str(label_source or "pred").strip().lower()
    if token in {"both", "all"}:
        return "both"
    if token in {"pred", "predicted", "learned", "learning"}:
        return "pred"
    if token in {"true", "gt", "groundtruth", "ground_truth"}:
        return "true"
    if token in {"none", "off", "hidden", "hide"}:
        return "none"
    return "pred"


def _resolved_output_label_modes(label_source: str | None) -> list[str]:
    normalized = _normalize_label_source(label_source)
    if normalized == "both":
        return ["pred", "true"]
    return [normalized]


def _label_mode_suffix(label_mode: str) -> str:
    normalized = _normalize_label_source(label_mode)
    if normalized == "true":
        return "true"
    if normalized == "none":
        return "nolabel"
    return "pred"


def _display_label_payload(result: dict[str, Any], label_source: str | None) -> tuple[np.ndarray | None, str]:
    resolved_source = _normalize_label_source(label_source)
    if resolved_source == "both":
        resolved_source = "pred"
    label_level = result.get("label_level")
    level_suffix = f" L{int(label_level)}" if label_level is not None else ""
    if resolved_source == "none":
        return None, f"labels=hidden{level_suffix}"
    if resolved_source == "true":
        true_labels = result.get("true_labels")
        if true_labels is not None:
            return np.asarray(true_labels, dtype=int), f"labels=true{level_suffix}"
        predicted_labels = result.get("predicted_labels")
        if predicted_labels is not None:
            return np.asarray(predicted_labels, dtype=int), f"labels=pred{level_suffix} fallback=true_missing"
        return None, f"labels=unavailable{level_suffix}"
    predicted_labels = result.get("predicted_labels")
    if predicted_labels is not None:
        return np.asarray(predicted_labels, dtype=int), f"labels=pred{level_suffix}"
    true_labels = result.get("true_labels")
    if true_labels is not None:
        return np.asarray(true_labels, dtype=int), f"labels=true{level_suffix} fallback=pred_missing"
    return None, f"labels=unavailable{level_suffix}"


def _trace_output_qualifier(params: dict[str, Any] | None) -> str | None:
    if not isinstance(params, dict):
        return None
    stats_mode = str(params.get("stats_mode", "")).strip().lower()
    if stats_mode in {"transition", "coactivation"}:
        return stats_mode
    return None


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _single_json_output_path(
    json_out: str | None,
    *,
    strategy: str,
    dataset: str,
    sequence_length: int,
    timestamp: str | None = None,
) -> Path | None:
    json_mode = classify_output_token(json_out)
    if json_mode == "path":
        target = Path(str(json_out)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    if json_mode == "auto":
        return artifact_result_path(
            strategy,
            dataset,
            "record",
            "json",
            step_count=int(sequence_length),
            kind="snapshots",
            timestamp=timestamp,
        )
    return None


def _single_snapshot_output_path(
    json_out: str | None = None,
    *,
    strategy: str,
    dataset: str,
    sequence_length: int,
    timestamp: str | None = None,
) -> Path:
    json_mode = classify_output_token(json_out)
    if json_mode == "path":
        target = Path(str(json_out)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.with_name(f"{target.stem}_snapshot{target.suffix}")
    return artifact_result_path(
        strategy,
        dataset,
        "snapshot",
        "json",
        step_count=int(sequence_length),
        kind="snapshots",
        timestamp=timestamp,
    )


def _resolve_output_token(*, save_enabled: bool, explicit_path: str | None, label: str) -> str | None:
    path_text = None if explicit_path is None else str(explicit_path).strip()
    if not save_enabled:
        if path_text not in {None, ""}:
            raise SystemExit(f"Cannot set --{label}-path while --save-{label}=false.")
        return "false"
    if path_text in {None, ""}:
        return "auto"
    return path_text


def _single_picture_output_path(
    png_out: str | None,
    *,
    strategy: str,
    dataset: str,
    sequence_length: int,
    artifact_name: str = "picture",
    timestamp: str | None = None,
) -> Path | None:
    png_mode = classify_output_token(png_out)
    if png_mode == "path":
        target = Path(str(png_out)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    if png_mode == "auto":
        return artifact_result_path(
            strategy,
            dataset,
            artifact_name,
            "png",
            step_count=int(sequence_length),
            kind="picture",
            timestamp=timestamp,
        )
    return None


def _single_gif_output_path(
    gif_out: str | None,
    *,
    strategy: str,
    dataset: str,
    sequence_length: int,
    artifact_name: str = "animation",
    timestamp: str | None = None,
) -> Path | None:
    gif_mode = classify_output_token(gif_out)
    if gif_mode == "path":
        target = Path(str(gif_out)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    if gif_mode == "auto":
        return artifact_result_path(
            strategy,
            dataset,
            artifact_name,
            "gif",
            step_count=int(sequence_length),
            kind="animations",
            timestamp=timestamp,
        )
    return None


def _derive_labeled_explicit_path(path: Path, label_mode: str) -> Path:
    suffix = _label_mode_suffix(label_mode)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _compact_param_summary(params: dict[str, Any]) -> str:
    filtered = {key: value for key, value in params.items() if key not in {"env_mode", "seed"}}
    if not filtered:
        return "none"
    parts = [f"{key}={filtered[key]}" for key in sorted(filtered.keys())]
    return ", ".join(parts)


def _print_single_test_summary(result: dict[str, Any], *, metric_name: EvaluationMetricName) -> None:
    metric_key = str(metric_name).strip().lower()
    engine_strategy = result.get("engine_strategy") or result.get("runtime_strategy") or result.get("strategy")
    profile_name = result.get("profile_name") or result.get("strategy")
    primary_score = result.get(f"{metric_key}_score")
    ari_score = result.get("ari_score")
    nmi_score = result.get("nmi_score")
    level_ari_items = sorted(
        (
            (key, value)
            for key, value in result.items()
            if re.fullmatch(r"cluster_level\d+_ari_score", str(key))
        ),
        key=lambda item: int(re.findall(r"\d+", item[0])[0]),
    )
    cluster_count = result.get("cluster_count")
    noise_ratio = result.get("noise_ratio")
    eval_key = result.get("evaluation_output_key")
    eval_mode = result.get("evaluation_mode")
    eval_method = result.get("eval_readout_method")
    elapsed_time = result.get("elapsed_time")
    primary_score_text = f"{float(primary_score):.4f}" if primary_score is not None else "n/a"
    ari_score_text = f"{float(ari_score):.4f}" if ari_score is not None else "n/a"
    nmi_score_text = f"{float(nmi_score):.4f}" if nmi_score is not None else "n/a"
    hierarchy_score_text = "".join(
        f"L{int(re.findall(r'[0-9]+', key)[0])}={float(value):.4f} "
        for key, value in level_ari_items
    )
    noise_ratio_text = f"{float(noise_ratio):.4f}" if noise_ratio is not None else "n/a"
    elapsed_text = f"{float(elapsed_time):.2f}s" if elapsed_time is not None else "n/a"
    print(
        "result: "
        f"engine={engine_strategy or 'n/a'} "
        f"profile={profile_name or 'none'} "
        f"metric={metric_key} score={primary_score_text} "
        f"ari={ari_score_text} "
        f"nmi={nmi_score_text} "
        f"{hierarchy_score_text}"
        f"clusters={int(cluster_count) if cluster_count is not None and not np.isnan(cluster_count) else 'n/a'} "
        f"noise={noise_ratio_text} "
        f"evaluation={eval_key}({eval_mode}) "
        f"readout=unified:{eval_method or 'n/a'} "
        f"elapsed={elapsed_text}"
    )


def _print_distance_diagnostics(model: Any) -> None:
    if model is None or not hasattr(model, "level0"):
        print("diag_distance: unavailable")
        return
    syncmap_l0 = np.asarray(model.level0.syncmap, dtype=float)
    distances_l0: list[float] = []
    for i in range(len(syncmap_l0)):
        for j in range(i + 1, len(syncmap_l0)):
            distances_l0.append(float(np.linalg.norm(syncmap_l0[i] - syncmap_l0[j])))
    if not distances_l0:
        print("diag_distance_l0: unavailable")
        return
    print(
        "diag_distance_l0: "
        f"mean={np.mean(distances_l0):.4f} "
        f"median={np.median(distances_l0):.4f} "
        f"min={np.min(distances_l0):.4f} "
        f"max={np.max(distances_l0):.4f}"
    )


def _distance_summary(points: np.ndarray) -> dict[str, float]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    distances: list[float] = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            distances.append(float(np.linalg.norm(arr[i] - arr[j])))
    if not distances:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    dist_arr = np.asarray(distances, dtype=float)
    return {
        "count": float(dist_arr.size),
        "mean": float(np.mean(dist_arr)),
        "median": float(np.median(dist_arr)),
        "min": float(np.min(dist_arr)),
        "max": float(np.max(dist_arr)),
    }


def _cluster_distance_payload(points: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pos = np.asarray(points, dtype=float)
    labs = np.asarray(labels, dtype=int).reshape(-1)
    valid = sorted({int(v) for v in labs.tolist() if int(v) >= 0})
    centroids: dict[str, list[float]] = {}
    cluster_sizes: dict[str, int] = {}
    for label in valid:
        mask = labs == label
        pts = pos[mask]
        if pts.size == 0:
            continue
        centroids[str(label)] = np.mean(pts, axis=0).tolist()
        cluster_sizes[str(label)] = int(np.sum(mask))
    return {
        "cluster_count": len(valid),
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
    }


def _task0_snapshot_payload(result: dict[str, Any], *, strategy: str, dataset: str) -> dict[str, Any] | None:
    model = result.get("syncmap")
    env = result.get("graphwalk")
    if model is None or env is None:
        return None

    try:
        hierarchy_outputs_raw = generic_hierarchy_outputs(model, env)
        hierarchy_outputs = {
            key: np.asarray(value, dtype=int).reshape(-1)
            for key, value in hierarchy_outputs_raw.items()
        }
    except Exception:
        hierarchy_outputs = {}

    level0_model = getattr(model, "level0", model)
    positions = np.asarray(
        getattr(level0_model, "syncmap", np.empty((0, max(1, int(getattr(level0_model, "dimensions", 3)))))),
        dtype=float,
    )
    pred_labels = result.get("predicted_labels")
    true_labels = result.get("true_labels")
    pred_arr = None if pred_labels is None else np.asarray(pred_labels, dtype=int).reshape(-1)
    true_arr = None if true_labels is None else np.asarray(true_labels, dtype=int).reshape(-1)

    true_label_levels: dict[str, list[int]] = {}
    get_levels = getattr(env, "get_true_label_levels", None)
    if callable(get_levels):
        try:
            arr = np.asarray(get_levels(), dtype=int)
            if arr.ndim == 2:
                for idx in range(arr.shape[1]):
                    true_label_levels[f"level_{idx}"] = arr[:, idx].tolist()
        except Exception:
            pass

    trajectory = np.asarray(getattr(model, "get_trajectory", lambda: np.array([]))(), dtype=float)
    trajectory_steps = np.asarray(getattr(model, "get_trajectory_steps", lambda: np.array([], dtype=int))(), dtype=int)

    node_records: list[dict[str, Any]] = []
    if positions.ndim == 2:
        for idx in range(positions.shape[0]):
            record: dict[str, Any] = {
                "node_index": int(idx),
                "position": positions[idx].tolist(),
                "projected_position": positions[idx].tolist(),
            }
            if true_arr is not None and idx < true_arr.shape[0]:
                record["true_label"] = int(true_arr[idx])
            if pred_arr is not None and idx < pred_arr.shape[0]:
                record["predicted_label"] = int(pred_arr[idx])
            for key, labels in hierarchy_outputs.items():
                if idx < labels.shape[0]:
                    record[key] = int(labels[idx])
            node_records.append(record)

    level_payloads: list[dict[str, Any]] = []
    levels = getattr(model, "levels", None)
    if isinstance(levels, list) and levels:
        for level_index, level_model in enumerate(levels):
            level_positions = np.asarray(
                getattr(level_model, "syncmap", np.empty((0, positions.shape[1] if positions.ndim == 2 else 0))),
                dtype=float,
            )
            level_payloads.append(
                {
                    "name": f"level_{level_index}",
                    "node_count": int(level_positions.shape[0]) if level_positions.ndim == 2 else 0,
                    "positions": level_positions.tolist() if level_positions.ndim == 2 else [],
                    "distance_summary": _distance_summary(level_positions),
                }
            )
    elif positions.ndim == 2:
        level_payloads.append(
            {
                "name": "level_0",
                "node_count": int(positions.shape[0]),
                "positions": positions.tolist(),
                "distance_summary": _distance_summary(positions),
            }
        )

    lifted_cluster_distances: dict[str, Any] = {}
    for key, labels in hierarchy_outputs.items():
        if positions.ndim == 2 and labels.shape[0] == positions.shape[0]:
            lifted_cluster_distances[key] = _cluster_distance_payload(positions, labels)

    payload: dict[str, Any] = {
        "metadata": {
            "dataset": resolve_dataset_path(dataset),
            "strategy": str(strategy),
            "runtime_strategy": result.get("runtime_strategy", strategy),
            "engine_strategy": result.get("engine_strategy", result.get("runtime_strategy", strategy)),
            "profile_name": result.get("profile_name", ""),
            "total_steps": int(result.get("sequence_length", 0)),
            "dimension": int(getattr(level0_model, "dimensions", positions.shape[1] if positions.ndim == 2 else 0)),
            "organize_level": getattr(model, "organize_level", None),
            "evaluation_output_key": result.get("evaluation_output_key"),
            "evaluation_target_key": result.get("evaluation_target_key"),
            "evaluation_mode": result.get("evaluation_mode"),
            "label_level": result.get("label_level"),
        },
        "effective_model_params": _json_safe_value(result.get("effective_model_params")),
        "effective_env_params": _json_safe_value(result.get("effective_env_params")),
        "true_labels_display": None if true_arr is None else true_arr.tolist(),
        "predicted_labels_display": None if pred_arr is None else pred_arr.tolist(),
        "true_label_levels": true_label_levels,
        "hierarchy_outputs": {key: labels.tolist() for key, labels in hierarchy_outputs.items()},
        "trajectory": trajectory.tolist() if trajectory.size else [],
        "trajectory_steps": trajectory_steps.tolist() if trajectory_steps.size else [],
        "nodes": node_records,
        "levels": level_payloads,
        "level0_distance_summary": _distance_summary(positions),
        "lifted_cluster_distances": lifted_cluster_distances,
    }
    return payload


def run_single_test_task(
    *,
    dataset: str,
    strategy: str,
    env_mode: str,
    strategy_params: dict[str, Any],
    sequence_length: int,
    json_out: Optional[str],
    diag_distance: Optional[bool],
    viewer: bool,
    viewer_playback: bool,
    viewer_start_paused: bool,
    viewer_dimension: int,
    trajectory_interval: int,
    save_gif: bool,
    gif_out: Optional[str],
    save_png: bool,
    png_out: Optional[str],
    label_source: str,
    metric_name: EvaluationMetricName,
    theme: str,
    accent_color: str,
    resolved_variant: Optional[str] = None,
    sample_curves: bool = False,
    sample_interval: int | None = None,
) -> dict[str, Any]:
    need_model = bool(diag_distance or viewer or save_gif or save_png or classify_output_token(json_out) != "disabled")
    need_trajectory = bool(viewer or save_gif or classify_output_token(json_out) != "disabled")
    seed = int(strategy_params.get("seed", 0))
    model_strategy_params = {key: value for key, value in strategy_params.items() if key != "seed"}
    run_timestamp = timestamp_now()
    np.random.seed(seed)
    output_label_modes = _resolved_output_label_modes(label_source)
    viewer_label_mode = output_label_modes[0]
    display_strategy = str(resolved_variant or strategy)
    print(
        "single_test: "
        f"dataset={Path(dataset).name} "
        f"strategy={display_strategy} "
        f"runtime_strategy={strategy} "
        f"env_mode={env_mode} "
        f"steps={sequence_length} "
        f"metric={metric_name} "
        f"seed={seed} "
        f"record_interval={trajectory_interval} "
        f"live={viewer} "
        f"label_source={_normalize_label_source(label_source)} "
        f"save_png={save_png} "
        f"save_gif={save_gif}"
    )
    if resolved_variant is not None:
        print(f"profile: {resolved_variant}")
    print(f"overrides: {_compact_param_summary(strategy_params)}")

    result = run_single_test_lib(
        dataset_name=dataset,
        strategy_name=strategy,
        strategy_params=model_strategy_params,
        sequence_length=sequence_length,
        print_results=False,
        include_model=need_model,
        trajectory_interval=int(trajectory_interval),
        record_trajectory=need_trajectory,
        metric_name=metric_name,
        sample_curves=bool(sample_curves),
        sample_interval=sample_interval,
        seed=seed,
    )
    result["runtime_strategy"] = display_strategy
    result["engine_strategy"] = str(strategy)
    result["profile_name"] = "" if resolved_variant is None else str(resolved_variant)
    result["strategy"] = display_strategy

    _print_single_test_summary(result, metric_name=metric_name)

    if diag_distance:
        _print_distance_diagnostics(result.get("syncmap"))

    display_labels, display_label_title = _display_label_payload(result, viewer_label_mode)

    saved_json_output = "false"
    saved_snapshot_output = "false"
    saved_png_outputs: list[str] = []
    saved_gif_outputs: list[str] = []

    target_json = _single_json_output_path(
        json_out,
        strategy=display_strategy,
        dataset=dataset,
        sequence_length=int(sequence_length),
        timestamp=run_timestamp,
    )
    if target_json is not None:
        serialized = {
            key: _json_safe_value(value)
            for key, value in result.items()
            if key not in {"syncmap", "graphwalk"}
        }
        with open(target_json, "w", encoding="utf-8") as fh:
            json.dump(serialized, fh, ensure_ascii=False, indent=2)
        saved_json_output = str(target_json)
        snapshot_payload = _task0_snapshot_payload(result, strategy=display_strategy, dataset=dataset)
        if snapshot_payload is not None:
            snapshot_json = _single_snapshot_output_path(
                json_out,
                strategy=display_strategy,
                dataset=dataset,
                sequence_length=int(sequence_length),
                timestamp=run_timestamp,
            )
            with open(snapshot_json, "w", encoding="utf-8") as fh:
                json.dump(snapshot_payload, fh, ensure_ascii=False, indent=2)
            saved_snapshot_output = str(snapshot_json)

    if save_gif:
        from tools.visual_tools import visualize_trajectory_animation

        if "syncmap" in result and "graphwalk" in result:
            multi_label_output = len(output_label_modes) > 1
            for label_mode in output_label_modes:
                labels_for_artifact, _ = _display_label_payload(result, label_mode)
                target_gif = _single_gif_output_path(
                    gif_out,
                    strategy=display_strategy,
                    dataset=dataset,
                    sequence_length=int(sequence_length),
                    artifact_name=f"animation_{_label_mode_suffix(label_mode)}",
                    timestamp=run_timestamp,
                )
                if target_gif is not None and classify_output_token(gif_out) == "path" and multi_label_output:
                    target_gif = _derive_labeled_explicit_path(target_gif, label_mode)
                    target_gif.parent.mkdir(parents=True, exist_ok=True)
                ok = visualize_trajectory_animation(
                    result["syncmap"],
                    graphwalk=result.get("graphwalk"),
                    labels=labels_for_artifact,
                    dataset=dataset,
                    strategy=display_strategy,
                    save_path=None if target_gif is None else str(target_gif),
                    total_steps=int(sequence_length),
                    show=False,
                    theme=theme,
                    accent_color=accent_color,
                )
                if ok and target_gif is not None:
                    saved_gif_outputs.append(str(target_gif))

    if save_png:
        from matplotlib import pyplot as plt
        from tools.visual_tools import plot_syncmap_over_time

        model = result.get("syncmap")
        positions_source = getattr(model, "level0", model)
        positions = getattr(positions_source, "syncmap", None)
        multi_label_output = len(output_label_modes) > 1
        if positions is not None:
            for label_mode in output_label_modes:
                labels_for_artifact, label_title = _display_label_payload(result, label_mode)
                target_png = _single_picture_output_path(
                    png_out,
                    strategy=display_strategy,
                    dataset=dataset,
                    sequence_length=int(sequence_length),
                    artifact_name=f"picture_{_label_mode_suffix(label_mode)}",
                    timestamp=run_timestamp,
                )
                if target_png is not None and classify_output_token(png_out) == "path" and multi_label_output:
                    target_png = _derive_labeled_explicit_path(target_png, label_mode)
                    target_png.parent.mkdir(parents=True, exist_ok=True)
                if target_png is None:
                    continue
                fig, _ = plot_syncmap_over_time(
                    np.asarray(positions, dtype=float),
                    labels_for_artifact,
                    steps=np.asarray([int(sequence_length)], dtype=int),
                    show=False,
                    method_name=str(display_strategy).upper(),
                    theme=theme,
                    accent_color=accent_color,
                    title_suffix=label_title,
                )
                fig.tight_layout()
                fig.savefig(target_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)
                saved_png_outputs.append(str(target_png))

    saved_png_output = ",".join(saved_png_outputs) if saved_png_outputs else "false"
    saved_gif_output = ",".join(saved_gif_outputs) if saved_gif_outputs else "false"
    print(
        "outputs: "
        f"saved_json={saved_json_output} "
        f"saved_snapshot={saved_snapshot_output} "
        f"saved_png={saved_png_output} "
        f"saved_gif={saved_gif_output}"
    )

    if viewer:
        from live import launch_single_test_viewer

        if "syncmap" not in result or "graphwalk" not in result:
            print("live: unavailable (missing syncmap / graphwalk)")
        else:
            launch_single_test_viewer(
                result,
                dimension=int(viewer_dimension),
                playback=bool(viewer_playback),
                start_paused=bool(viewer_start_paused),
                show=True,
                theme=theme,
                accent_color=accent_color,
                display_name=resolved_variant,
            )

    return {
        "run_timestamp": run_timestamp,
        "result": result,
    }


def run_single_trace_extension(
    *,
    dataset: str,
    strategy: str,
    env_mode: str,
    strategy_params: dict[str, Any],
    steps: int,
    record_every: int,
    save_json: bool,
    save_plot: bool,
    print_trace: bool,
    seed: int,
    json_out: str | None,
    plot_out: str | None,
    plot_metric: str,
    theme: str,
    accent_color: str,
    run_timestamp: str,
    resolved_variant: Optional[str] = None,
) -> None:
    print("trace_extension: enabled")
    trace_json_out = json_out
    if classify_output_token(json_out) == "path":
        trace_json_out = "auto"
        print("trace_extension: explicit --json-path applies to the single-test record; trace JSON uses the auto path.")
    trace_scores_task(
        dataset=dataset,
        strategy=strategy,
        env_mode=env_mode,
        strategy_params=dict(strategy_params),
        steps=int(steps),
        record_every=int(record_every),
        save_plot=bool(save_plot),
        print_trace=bool(print_trace),
        seed=int(seed),
        json_out=trace_json_out if save_json else "false",
        plot_out=plot_out if save_plot else "false",
        plot_metric=plot_metric,
        theme=theme,
        accent_color=accent_color,
        timestamp=run_timestamp,
        resolved_variant=resolved_variant,
    )


def run_live_task(
    *,
    dataset: str,
    strategy_name: str,
    env_mode: str,
    strategy_params: dict[str, Any],
    theme: str,
    accent_color: str,
    resolved_variant: Optional[str] = None,
    total_steps: int | None = None,
    record_interval: int | None = None,
) -> None:
    from live import LiveTuner
    import matplotlib.pyplot as plt

    seed = strategy_params.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    print(
        "run_live: "
        f"dataset={Path(dataset).name} "
        f"strategy={strategy_name} "
        f"env_mode={env_mode} "
        f"seed={strategy_params.get('seed', 0)} "
        f"theme={theme} "
        f"auto_pause={total_steps if total_steps is not None else 'off'} "
        f"record_interval={record_interval if record_interval is not None else 'auto'}"
    )
    if resolved_variant is not None:
        print(f"profile: {resolved_variant}")
    print(f"overrides: {_compact_param_summary(strategy_params)}")

    tuner = LiveTuner(
        dataset,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        dimension=3,
        display_name=resolved_variant,
        auto_pause_steps=total_steps,
        record_interval=record_interval,
    )
    tuner.apply_visual_theme(theme=theme, accent_color=accent_color)
    plt.show()


def trace_scores_task(
    *,
    dataset: str,
    strategy: str,
    env_mode: str,
    strategy_params: dict[str, Any],
    steps: int,
    record_every: int,
    save_plot: bool,
    print_trace: bool,
    seed: int,
    json_out: Optional[str],
    plot_out: Optional[str],
    plot_metric: str,
    theme: str,
    accent_color: str,
    timestamp: str | None = None,
    resolved_variant: Optional[str] = None,
) -> None:
    from tools.trace_rg_hierarchy_scores import print_integral_summary, print_summary, run_trace, save_score_plot

    started_at = time.time()
    print(
        "trace_scores: "
        f"dataset={Path(dataset).name} "
        f"strategy={strategy} "
        f"env_mode={env_mode} "
        f"steps={steps} "
        f"record_interval={record_every} "
        f"seed={seed}"
    )
    if resolved_variant is not None:
        print(f"profile: {resolved_variant}")
    print(f"overrides: {_compact_param_summary(strategy_params)}")

    record = run_trace(
        dataset=dataset,
        strategy=strategy,
        total_steps=steps,
        record_every=record_every,
        env_mode=env_mode,
        override_params=dict(strategy_params),
        seed=int(seed),
        selection_metric=str(plot_metric),
    )
    if bool(print_trace):
        print_summary(record["trace"])

    auto_qualifier = _trace_output_qualifier(record.get("params"))
    json_mode = classify_output_token(json_out)
    if json_mode == "path":
        target_json = Path(str(json_out)).expanduser()
        target_json.parent.mkdir(parents=True, exist_ok=True)
    elif json_mode == "auto":
        target_json, _ = trace_result_paths(strategy, dataset, steps, timestamp=timestamp, qualifier=auto_qualifier)
    else:
        target_json = None
    if target_json is not None:
        with open(target_json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        print(f"saved_json: {target_json}")
    else:
        print("saved_json: false")

    print_integral_summary(record, metric=plot_metric)

    if bool(save_plot):
        plot_mode = classify_output_token(plot_out)
        if plot_mode == "path":
            target_plot = Path(str(plot_out)).expanduser()
        elif plot_mode == "auto":
            _, target_plot = trace_result_paths(strategy, dataset, steps, timestamp=timestamp, qualifier=auto_qualifier)
        else:
            target_plot = None
        if target_plot is not None:
            saved_plot = save_score_plot(record, target_plot, metric=plot_metric, theme=theme, accent_color=accent_color)
            print(f"saved_plot: {saved_plot}")
        else:
            print("saved_plot: false")
    print(f"trace_elapsed: {time.time() - started_at:.2f}s")


def render_trace_plot_task(
    trace_json: str,
    plot_out: Optional[str] = None,
    *,
    plot_metric: str = "ari",
    theme: str = DEFAULT_THEME,
    accent_color: str = DEFAULT_ACCENT_COLOR,
) -> None:
    from tools.render_rg_trace_plot import render_one

    json_path = Path(str(trace_json)).expanduser()
    saved_plot = render_one(json_path, plot_out, metric=plot_metric, theme=theme, accent_color=accent_color)
    print(f"saved_plot: {saved_plot}")


def scaling_law_task(**kwargs: Any) -> None:
    """Thin wrapper around the external scaling-law evaluation tool (task 7)."""
    from types import SimpleNamespace

    from tools.scaling_law import run_scaling_law

    run_scaling_law(SimpleNamespace(**kwargs))


def analyze_sequence_information_task(
    *,
    dataset: str,
    strategy: str,
    env_mode: str,
    json_out: Optional[str],
    resolved_variant: Optional[str] = None,
) -> None:
    from tools.sequence_information_analysis import analyze_dataset_sequence, print_sequence_information_summary

    result = analyze_dataset_sequence(dataset, strategy=strategy, env_mode=env_mode)
    if resolved_variant is not None:
        print(f"profile: {resolved_variant}")
    print_sequence_information_summary(result)

    json_mode = classify_output_token(json_out)
    if json_mode == "path":
        target_json = Path(str(json_out)).expanduser()
        target_json.parent.mkdir(parents=True, exist_ok=True)
    elif json_mode == "auto":
        target_json = sequence_info_result_path(dataset)
    else:
        target_json = None
    if target_json is not None:
        with open(target_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"saved_json: {target_json}")
    else:
        print("saved_json: false")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SyncMap experiments from one CLI surface. "
            "Primary workflow: task 0 single_test, task 1 live, task 2 all_experiments. "
            "Advanced/occasional tasks remain available for special post-processing."
        ),
        formatter_class=_RawDefaultsFormatter,
        epilog=(
            "Primary tasks:\n"
            "  0 = single_test\n"
            "  1 = run_live\n"
            "  2 = all_experiments\n"
            "\n"
            "Advanced / occasional tasks:\n"
            "  4 = trace_scores (legacy dedicated trace path; prefer task 0 --trace)\n"
            "  5 = render_trace\n"
            "  6 = analyze_sequence_information\n"
            "  7 = scaling_law (external loss-vs-steps power-law evaluation)\n\n"
            "Policy:\n"
            "  - main.py is the only public CLI entrypoint\n"
            "  - use --profile for maintained presets\n"
            "  - use --set key=value for algorithm-specific overrides\n"
            "  - metric defaults to ARI\n"
            "  - prefer task 0 / 1 / 2 for normal work\n"
        ),
    )
    parser.add_argument("-t", "--task", default=None, help="Task id or canonical task name")
    parser.add_argument("-d", "--dataset", default=None, help="Dataset path")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset list for task 2")
    parser.add_argument("-s", "--strategy", nargs="+", default=None, help="Strategy/runtime name(s). Multiple strategies open a comparison grid view.")
    parser.add_argument("--strategies", nargs="+", default=None, help="Strategy list for task 2")
    parser.add_argument("--profile", default=None, help="Maintained parameter preset name")
    parser.add_argument("--profiles", nargs="+", default=None, help="Maintained parameter preset list for task 2")
    parser.add_argument("--list-profiles", action="store_true", help="Print maintained profile names and exit")
    parser.add_argument("--env-mode", default=None, help="Environment mode override")
    parser.add_argument("--metric", default=None, choices=["nmi", "ari"], help="Evaluation metric")
    parser.add_argument("--trace-json", default=None, help="Task 5/7 input: task-0 learning-curve JSON or task-4 trace JSON")
    parser.add_argument("--save-json", nargs="?", const="true", default=None, type=_parse_cli_bool, help="Save JSON outputs. Defaults to true for task 0/4/6.")
    parser.add_argument("--json-path", default=None, help="Optional explicit JSON output path")
    parser.add_argument("--steps", type=int, default=None, help="Step count for task 0/4/7; also the auto-pause target for task 1")
    parser.add_argument("--sequence-length", type=int, default=None, help="Explicit sequence length for task 0")
    parser.add_argument("--record-interval", type=int, default=None, help="Sampling interval for live metric curves / playback / GIF / trace exports")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. For task 2 comparison: trial i uses seed+i-1 (e.g. seed=0 -> trials use 0,1,2,3,4); omit for fully random trials.")
    parser.add_argument("--comparison-sequence-lengths", nargs="+", type=int, default=None, help="Sequence lengths for task 2 comparison runs")
    parser.add_argument("--comparison-trials", type=int, default=None, help="Trial count for task 2 comparison runs")
    parser.add_argument("--comparison-save-gif", nargs="?", const="true", default=False, type=_parse_cli_bool, help="During task 2, export a GIF for each finished comparison case")
    parser.add_argument("--comparison-trace-json", nargs="?", const="true", default=False, type=_parse_cli_bool, help="During task 2, save a learning-curve JSON for each finished comparison case")
    parser.add_argument("--comparison-trace-plot", nargs="?", const="true", default=False, type=_parse_cli_bool, help="During task 2, save a learning-curve plot for each finished comparison case")
    parser.add_argument("--comparison-plot-metric", default="auto", choices=["auto", "nmi", "ari"], help="Metric to plot for task 2 integrated learning-curve exports")
    parser.add_argument("--comparison-record-interval", type=int, default=None, help="Sampling interval for task 2 integrated artifact exports")
    parser.add_argument("--late-window-points", type=int, default=5, help="Number of trailing learning-curve points to average into late_mean/late_std columns (default: 5).")
    parser.add_argument("--comparison-include-hierarchy", action="store_true", help="For task 2 comparison, also include datasets under datasets/hierarchy")
    parser.add_argument("--test-subset", action="store_true", help="Use the representative fast-test subset for task 2")
    parser.add_argument("--resume-mode", default="auto", choices=["auto", "off"], help="Task 2 comparison partial-results policy")
    parser.add_argument("--resume-file", default=None, help="Explicit comparison partial CSV to resume for task 2")
    parser.add_argument("--memory-profile", default=None, choices=list(MEMORY_PROFILE_CHOICES), help="High-level memory family selector")
    parser.add_argument("--label-level", type=int, default=None, help="Hierarchy label level override for evaluation")
    parser.add_argument("--diag-distance", nargs="?", const="true", default=None, type=_parse_cli_bool, help="Show extra distance diagnostics for task 0")
    parser.add_argument("--trace", nargs="?", const="true", default=False, type=_parse_cli_bool, help="For task 0, also export a score trace and curve. This reruns the experiment in trace mode.")
    parser.add_argument("--trace-record-interval", type=int, default=None, help="Sampling interval for task 0 trace mode; defaults to --record-interval or an auto value.")
    parser.add_argument("--live", nargs="?", const="true", default=False, type=_parse_cli_bool, help="Open the read-only post-run GUI after task 0")
    parser.add_argument("--live-playback", default=None, type=_parse_bool_token)
    parser.add_argument("--live-start-paused", default=None, type=_parse_bool_token)
    parser.add_argument("--live-dimension", type=int, default=None)
    parser.add_argument("--save-gif", nargs="?", const="true", default=None, type=_parse_cli_bool, help="For task 0, export a trajectory GIF")
    parser.add_argument("--gif-path", default=None, help="Optional explicit GIF output path for task 0")
    parser.add_argument("--save-png", nargs="?", const="true", default=None, type=_parse_cli_bool, help="For task 0, export the final static PNG")
    parser.add_argument("--png-path", default=None, help="Optional explicit PNG output path for task 0")
    parser.add_argument("--label-source", default="pred", choices=["pred", "learned", "true", "both", "none"], help="For task 0 PNG/GIF/live labeling. Use both to export both true and predicted labeled assets.")
    parser.add_argument("--plot-path", default=None, help="Optional explicit plot output path for task 0 trace mode or task 4/5/7")
    parser.add_argument("--plot-metric", default="auto", choices=["auto", "nmi", "ari"], help="Trace-plot metric for task 0 trace mode / task 4 / task 5; auto follows --metric")
    parser.add_argument("--theme", default=DEFAULT_THEME, help="Visualization theme")
    parser.add_argument("--accent-color", default=DEFAULT_ACCENT_COLOR)
    parser.add_argument("--save-plot", nargs="?", const="true", default=True, type=_parse_cli_bool, help="For task 0 trace mode / task 4 / task 5, save the curve PNG")
    parser.add_argument("--print-trace", action="store_true")
    parser.add_argument("--set", dest="set_overrides", action="append", default=None, metavar="KEY=VALUE", help="Algorithm-specific override. Repeat as needed.")
    return parser


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv[1:])


def _resolve_task(cli: argparse.Namespace) -> int:
    if cli.task is None:
        return DEFAULT_TASKNUM
    token = str(cli.task).strip().lower()
    if not token:
        return DEFAULT_TASKNUM
    if token not in TASK_ALIASES:
        valid = ", ".join(str(index) for index in sorted(set(TASK_ALIASES.values())))
        raise SystemExit(
            f"Unknown task {cli.task!r}. Valid task ids are: {valid}."
        )
    return int(TASK_ALIASES[token])


def _selection_values(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    return [str(raw)]


def _validate_cli_contract(cli: argparse.Namespace, task: int) -> None:
    for option in ("strategy", "strategies"):
        for token in _selection_values(getattr(cli, option, None)):
            if "--" in token:
                raise SystemExit(
                    f"Malformed --{option} value {token!r}. It looks like a "
                    "line-continuation backslash joined two arguments. Put a "
                    "space before the backslash: end the strategy line with "
                    "rg, one space, then the backslash; put --dataset on "
                    "the next line."
                )

    strategy_values = _selection_values(getattr(cli, "strategy", None))
    batch_strategy_values = _selection_values(
        getattr(cli, "strategies", None)
    )
    batch_profile_values = _selection_values(
        getattr(cli, "profiles", None)
    )
    has_profile = getattr(cli, "profile", None) not in {None, ""}
    if task != 2 and (batch_strategy_values or batch_profile_values):
        raise SystemExit(
            f"task {task} uses --strategy/--profile; "
            "--strategies/--profiles are task-2 list options."
        )
    if task in {0, 4, 6, 7} and len(strategy_values) > 1:
        raise SystemExit(
            f"task {task} accepts one --strategy value; "
            "use task 1 for a live comparison or task 2 for a batch comparison."
        )
    if task == 1 and len(strategy_values) > 1 and has_profile:
        raise SystemExit(
            "Multi-strategy task 1 cannot combine --profile with --strategy."
        )
    if task == 2:
        if strategy_values and batch_strategy_values:
            raise SystemExit(
                "Use either --strategy or --strategies for task 2, not both."
            )
        if has_profile and batch_profile_values:
            raise SystemExit(
                "Use either --profile or --profiles for task 2, not both."
            )
    if task == 5 and (
        strategy_values
        or batch_strategy_values
        or has_profile
        or batch_profile_values
    ):
        raise SystemExit(
            f"task {task} is post-processing and does not consume a strategy."
        )


def _resolve_plot_metric(cli: argparse.Namespace) -> str:
    plot_metric = str(getattr(cli, "plot_metric", "auto") or "auto").strip().lower()
    if plot_metric not in {"", "auto"}:
        return plot_metric
    metric_name = str(getattr(cli, "metric", "") or "").strip().lower()
    if metric_name in {"nmi", "ari"}:
        return metric_name
    return str(DEFAULT_METRIC)


def _collect_override_params(cli: argparse.Namespace) -> dict[str, Any]:
    overrides = _parse_set_overrides(getattr(cli, "set_overrides", None))
    if getattr(cli, "memory_profile", None) not in {None, "", "auto"}:
        overrides["memory_profile"] = str(cli.memory_profile)
    if getattr(cli, "label_level", None) is not None:
        overrides["label_level"] = int(cli.label_level)
    if getattr(cli, "seed", None) is not None:
        overrides["seed"] = int(cli.seed)
    return overrides


def _run_single_test(cli: argparse.Namespace) -> None:
    started_at = time.time()
    config = _resolved_main_config()
    dataset = str(resolve_dataset_path(cli.dataset) if cli.dataset is not None else config["dataset"])
    overrides = _collect_override_params(cli)
    strategy_raw = cli.strategy
    if isinstance(strategy_raw, (list, tuple)):
        strategy_raw = strategy_raw[0] if len(strategy_raw) == 1 else strategy_raw[0]
    resolved_variant, strategy_name, env_mode, strategy_params = _resolve_runtime_selection(
        config,
        dataset=dataset,
        strategy=strategy_raw,
        profile=cli.profile,
        env_mode=cli.env_mode,
        override_params=overrides,
    )
    sequence_length = cli.sequence_length
    if sequence_length is None and cli.steps is not None:
        sequence_length = int(cli.steps)
    if sequence_length is None:
        sequence_length = int(config["sequence_length"])
    save_json = True if cli.save_json is None else bool(cli.save_json)
    save_png = True if cli.save_png is None else bool(cli.save_png)
    save_gif = True if cli.save_gif is None else bool(cli.save_gif)
    save_plot = True if cli.save_plot is None else bool(cli.save_plot)
    json_token = _resolve_output_token(save_enabled=save_json, explicit_path=cli.json_path, label="json")
    png_token = _resolve_output_token(save_enabled=save_png, explicit_path=cli.png_path, label="png")
    gif_token = _resolve_output_token(save_enabled=save_gif, explicit_path=cli.gif_path, label="gif")
    plot_token = _resolve_output_token(save_enabled=save_plot, explicit_path=cli.plot_path, label="plot")
    viewer_playback = True if cli.live_playback is None else bool(cli.live_playback)
    viewer_start_paused = True if cli.live_start_paused is None else bool(cli.live_start_paused)
    viewer_dimension = int(cli.live_dimension if cli.live_dimension is not None else 3)
    trace_record_every = None
    if bool(cli.trace):
        trace_record_every = cli.trace_record_interval
        if trace_record_every is None:
            trace_record_every = cli.record_interval
        if trace_record_every is None:
            trace_record_every = min(10000, max(10, int(sequence_length) // 10))

    task_payload = run_single_test_task(
        dataset=dataset,
        strategy=strategy_name,
        env_mode=env_mode,
        strategy_params=strategy_params,
        sequence_length=int(sequence_length),
        json_out=json_token,
        diag_distance=cli.diag_distance,
        viewer=bool(cli.live),
        viewer_playback=viewer_playback,
        viewer_start_paused=viewer_start_paused,
        viewer_dimension=viewer_dimension,
        trajectory_interval=int(cli.record_interval or 50),
        save_gif=save_gif,
        gif_out=gif_token,
        save_png=save_png,
        png_out=png_token,
        label_source=str(cli.label_source),
        metric_name=cast(EvaluationMetricName, str(cli.metric or DEFAULT_METRIC)),
        theme=str(cli.theme),
        accent_color=str(cli.accent_color),
        resolved_variant=resolved_variant,
        sample_curves=bool(cli.trace),
        sample_interval=None if trace_record_every is None else int(trace_record_every),
    )
    run_timestamp = str(task_payload["run_timestamp"])
    single_result = cast(dict[str, Any], task_payload["result"])
    if bool(cli.trace):
        curve = list(cast(list[dict[str, Any]], single_result.get("learning_curve", [])))
        if bool(cli.print_trace):
            run_single_trace_extension(
                dataset=dataset,
                strategy=strategy_name,
                env_mode=env_mode,
                strategy_params=strategy_params,
                steps=int(sequence_length),
                record_every=int(trace_record_every),
                save_json=save_json,
                save_plot=save_plot,
                print_trace=bool(cli.print_trace),
                seed=int(strategy_params.get("seed", 0)),
                json_out=json_token,
                plot_out=plot_token,
                plot_metric=_resolve_plot_metric(cli),
                theme=str(cli.theme),
                accent_color=str(cli.accent_color),
                run_timestamp=run_timestamp,
                resolved_variant=resolved_variant,
            )
        else:
            print("trace_extension: reused single_test learning_curve")
            auto_qualifier = _trace_output_qualifier(strategy_params)
            trace_json_target: Path | None
            if save_json:
                trace_json_target, _ = trace_result_paths(
                    strategy_name,
                    dataset,
                    int(sequence_length),
                    timestamp=run_timestamp,
                    qualifier=auto_qualifier,
                )
                saved_curve_json = _save_learning_curve_json(curve, trace_json_target)
                print(f"saved_json: {saved_curve_json}")
            else:
                print("saved_json: false")
            if save_plot:
                _, trace_plot_target = trace_result_paths(
                    strategy_name,
                    dataset,
                    int(sequence_length),
                    timestamp=run_timestamp,
                    qualifier=auto_qualifier,
                )
                saved_curve_plot = _save_learning_curve_plot(
                    curve,
                    trace_plot_target,
                    metric=_resolve_plot_metric(cli),
                    theme=str(cli.theme),
                    accent_color=str(cli.accent_color),
                    title=f"{str(strategy_name).upper()} {Path(dataset).name} {_resolve_plot_metric(cli).upper()} curve",
                )
                print(f"saved_plot: {saved_curve_plot}")
            else:
                print("saved_plot: false")
    print(f"task0_total_elapsed: {time.time() - started_at:.2f}s")


def _run_live(cli: argparse.Namespace) -> None:
    config = _resolved_main_config()
    dataset = str(resolve_dataset_path(cli.dataset) if cli.dataset is not None else config["dataset"])
    overrides = _collect_override_params(cli)

    strategies_raw = cli.strategy
    # Support both single string and list of strings
    if isinstance(strategies_raw, (list, tuple)) and len(strategies_raw) > 1:
        _run_live_multi(
            dataset=dataset,
            strategies=list(strategies_raw),
            config=config,
            overrides=overrides,
            theme=str(cli.theme),
            accent_color=str(cli.accent_color),
            total_steps=cli.steps,
            record_interval=cli.record_interval,
        )
        return

    # Single strategy path
    strategy_arg = strategies_raw[0] if isinstance(strategies_raw, (list, tuple)) else strategies_raw
    resolved_variant, strategy_name, env_mode, strategy_params = _resolve_runtime_selection(
        config,
        dataset=dataset,
        strategy=strategy_arg,
        profile=cli.profile,
        env_mode=cli.env_mode,
        override_params=overrides,
    )
    run_live_task(
        dataset=dataset,
        strategy_name=strategy_name,
        env_mode=env_mode,
        strategy_params=strategy_params,
        theme=str(cli.theme),
        accent_color=str(cli.accent_color),
        resolved_variant=resolved_variant,
        total_steps=cli.steps,
        record_interval=cli.record_interval,
    )


def _run_live_multi(
    *,
    dataset: str,
    strategies: list[str],
    config: dict[str, Any],
    overrides: dict[str, Any],
    theme: str,
    accent_color: str,
    total_steps: Optional[int] = None,
    record_interval: Optional[int] = None,
) -> None:
    """Multiple strategies in synchronized embedding and metric-curve grids."""
    from tools.multilive import launch_multi_strategy_viewer

    run_seed = int(overrides.get("seed", 0))
    strategy_names = _ordered_unique(strategies)
    specs: list[dict[str, Any]] = []
    for s_name in strategy_names:
        _, runtime_name, env_mode, params = _resolve_runtime_selection(
            config,
            dataset=dataset,
            strategy=s_name,
            profile=None,
            env_mode=None,
            override_params=dict(overrides),
        )
        params.pop("env_mode", None)
        params["env_mode"] = env_mode
        params.pop("seed", None)
        print(f"profile: {s_name.upper()}  overrides: {_compact_param_summary(params)}")
        specs.append({
            "strategy_name": runtime_name,
            "display_name": s_name.upper(),
            "params": dict(params),
        })

    print(f"[multilive] {len(specs)} strategies in synchronized grid windows "
          f"on {Path(dataset).name}")
    launch_multi_strategy_viewer(
        dataset, specs, total_steps=total_steps, record_interval=record_interval,
        theme=theme, accent_color=accent_color, show=True, seed=run_seed,
    )


def _expand_dataset_family_tokens(tokens: list[str]) -> list[str]:
    """Expand family aliases in --datasets: 'hierarchy' and 'flat'."""
    expanded: list[str] = []
    for token in tokens:
        key = str(token).strip().lower()
        if key == "hierarchy":
            expanded.extend(HIERARCHY_DATASET_SUITE)
        elif key == "flat":
            expanded.extend(get_all_datasets())
        else:
            expanded.append(str(token))
    return list(dict.fromkeys(expanded))


def _run_all_experiments(cli: argparse.Namespace) -> None:
    config = _resolved_main_config()
    overrides = _collect_override_params(cli)
    selected_strategy_args = (
        list(cli.strategies)
        if cli.strategies
        else _selection_values(cli.strategy)
    )
    selected_profile_args = (
        list(cli.profiles)
        if cli.profiles
        else _selection_values(cli.profile)
    )
    if selected_profile_args and selected_strategy_args:
        raise SystemExit(
            "Use either --profile/--profiles or "
            "--strategy/--strategies for task 2, not both."
        )
    selected_profiles = (
        _comparison_profiles_from_config(config, selected_profile_args)
        if selected_profile_args
        else []
    )
    profile_labels: dict[str, str] = {}
    if selected_profiles:
        # 'seed' is run-level (per-trial base seed), never a model parameter;
        # leaking it into profile params crashes engines with narrow kwargs.
        profile_overrides = {k: v for k, v in overrides.items() if k != "seed"}
        specs = [(name, resolve_profile_run_spec(name, override_params=profile_overrides)) for name in selected_profiles]
        strategies = {
            str(profile_name): {
                **dict(spec.strategy_params),
                COMPARISON_ENGINE_STRATEGY_KEY: str(spec.strategy_name),
                COMPARISON_PROFILE_NAME_KEY: str(profile_name),
                COMPARISON_DISPLAY_NAME_KEY: str(profile_name),
            }
            for profile_name, spec in specs
        }
        profile_labels = {}
    else:
        strategy_names = _comparison_strategies_from_config(
            config,
            selected_strategy_args or None,
        )
        strategy_defaults = _strategy_defaults_from_config(config)
        strategies = create_strategy_config(
            **{
                name: {
                    **dict(strategy_defaults.get(name, {})),
                    **_memory_profile_overrides(name, _normalize_memory_profile(overrides.get("memory_profile"))),
                    **{k: v for k, v in overrides.items() if k not in {"memory_profile", "seed"}},
                }
                for name in strategy_names
            }
        )
    resolved_datasets = None if not cli.datasets else [str(resolve_dataset_path(dataset)) for dataset in _expand_dataset_family_tokens(cli.datasets)]
    run_all_experiments_lib(
        use_test_subset=bool(cli.test_subset),
        comparison_include_hierarchy=bool(cli.comparison_include_hierarchy),
        strategies=strategies,
        sequence_lengths=[int(v) for v in cli.comparison_sequence_lengths] if cli.comparison_sequence_lengths else _comparison_sequence_lengths(config),
        comparison_datasets=resolved_datasets,
        comparison_trials=int(cli.comparison_trials if cli.comparison_trials is not None else _comparison_trials(config)),
        metric_name=cast(EvaluationMetricName, str(cli.metric or _comparison_metric_name(config))),
        resume_mode=str(cli.resume_mode),
        resume_file=cli.resume_file,
        comparison_save_gif=bool(cli.comparison_save_gif),
        comparison_trace_json=bool(cli.comparison_trace_json),
        comparison_trace_plot=bool(cli.comparison_trace_plot),
        comparison_plot_metric=None if str(cli.comparison_plot_metric) == "auto" else str(cli.comparison_plot_metric),
        comparison_record_interval=cli.comparison_record_interval,
        comparison_base_seed=cli.seed,
        strategy_labels=profile_labels if profile_labels else None,
        comparison_late_window_points=int(cli.late_window_points),
    )


def _trace_scores(cli: argparse.Namespace) -> None:
    config = _resolved_main_config()
    dataset = str(resolve_dataset_path(cli.dataset) if cli.dataset is not None else config["dataset"])
    overrides = _collect_override_params(cli)
    strategy_raw = _first_strategy_arg(cli.strategy)
    resolved_variant, strategy_name, env_mode, strategy_params = _resolve_runtime_selection(
        config,
        dataset=dataset,
        strategy=strategy_raw,
        profile=cli.profile,
        env_mode=cli.env_mode,
        override_params=overrides,
    )
    steps = int(cli.steps if cli.steps is not None else config["sequence_length"])
    record_every = int(cli.record_interval if cli.record_interval is not None else min(10000, max(10, steps // 10)))
    save_json = True if cli.save_json is None else bool(cli.save_json)
    save_plot = True if cli.save_plot is None else bool(cli.save_plot)
    json_token = _resolve_output_token(save_enabled=save_json, explicit_path=cli.json_path, label="json")
    plot_token = _resolve_output_token(save_enabled=save_plot, explicit_path=cli.plot_path, label="plot")
    trace_scores_task(
        dataset=dataset,
        strategy=strategy_name,
        env_mode=env_mode,
        strategy_params=dict(strategy_params),
        steps=steps,
        record_every=record_every,
        save_plot=save_plot,
        print_trace=bool(cli.print_trace),
        seed=int(cli.seed if cli.seed is not None else 0),
        json_out=json_token,
        plot_out=plot_token,
        plot_metric=_resolve_plot_metric(cli),
        theme=str(cli.theme),
        accent_color=str(cli.accent_color),
        resolved_variant=resolved_variant,
    )


def _scaling_law(cli: argparse.Namespace) -> None:
    if cli.trace_json in {None, ""} and cli.dataset in {None, ""}:
        raise SystemExit("task 7 (scaling_law) needs either --trace-json or --dataset")
    profile = cli.profile
    strategy_raw = _first_strategy_arg(cli.strategy)
    if (
        cli.trace_json in {None, ""}
        and profile in {None, ""}
        and strategy_raw in {None, ""}
    ):
        profile = "rg"
    scaling_law_task(
        trace_json=cli.trace_json,
        dataset=cli.dataset,
        profile=profile,
        strategy=strategy_raw,
        steps=int(cli.steps) if cli.steps else 110000,
        record_interval=int(cli.record_interval) if cli.record_interval else 2000,
        seed=int(cli.seed) if cli.seed is not None else 0,
        seeds_list=None,
        env_mode=cli.env_mode,
        metric=(cli.metric or "nmi"),
        loss_floor=1e-3,
        fit_min_step=None,
        fit_max_step=None,
        plot_path=cli.plot_path,
        json_path=cli.json_path,
        theme=str(cli.theme),
        accent=str(cli.accent_color),
    )


def _render_trace(cli: argparse.Namespace) -> None:
    if cli.trace_json in {None, ""}:
        raise SystemExit("--trace-json is required for task 5")
    save_plot = True if cli.save_plot is None else bool(cli.save_plot)
    plot_token = _resolve_output_token(save_enabled=save_plot, explicit_path=cli.plot_path, label="plot")
    render_trace_plot_task(
        str(cli.trace_json),
        plot_out=plot_token,
        plot_metric=_resolve_plot_metric(cli),
        theme=str(cli.theme),
        accent_color=str(cli.accent_color),
    )


def _sequence_info(cli: argparse.Namespace) -> None:
    config = _resolved_main_config()
    dataset = str(resolve_dataset_path(cli.dataset) if cli.dataset is not None else config["dataset"])
    overrides = _collect_override_params(cli)
    strategy_raw = _first_strategy_arg(cli.strategy)
    resolved_variant, strategy_name, env_mode, _ = _resolve_runtime_selection(
        config,
        dataset=dataset,
        strategy=strategy_raw,
        profile=cli.profile,
        env_mode=cli.env_mode,
        override_params=overrides,
    )
    save_json = True if cli.save_json is None else bool(cli.save_json)
    json_token = _resolve_output_token(save_enabled=save_json, explicit_path=cli.json_path, label="json")
    analyze_sequence_information_task(
        dataset=dataset,
        strategy=strategy_name,
        env_mode=env_mode,
        json_out=json_token,
        resolved_variant=resolved_variant,
    )


_TASK_HANDLERS: dict[int, Any] = {
    0: _run_single_test,
    1: _run_live,
    2: _run_all_experiments,
    4: _trace_scores,
    5: _render_trace,
    6: _sequence_info,
    7: _scaling_law,
}


def dispatch_cli(argv: Optional[list[str]] = None) -> int:
    args = sys.argv if argv is None else argv
    cli = _parse_cli(args)
    if bool(cli.list_profiles):
        print("Profiles:")
        for name in list_target_profiles():
            profile = VARIANT_CONFIGS[name]
            param_count = len(cast(dict[str, Any], profile.get("params", {})))
            print(f"- {name} -> strategy={profile['public_runtime']} params={param_count}")
            print(f"  {profile['summary']}")
        return 0
    task = _resolve_task(cli)
    _validate_cli_contract(cli, task)
    handler = _TASK_HANDLERS.get(task)
    if handler is None:
        raise SystemExit(f"Unsupported task: {task}")
    handler(cli)
    return 0


__all__ = [
    "DEFAULT_TASKNUM",
    "MAIN_CONFIG",
    "MEMORY_PROFILE_CHOICES",
    "VARIANT_CONFIGS",
    "dispatch_cli",
    "list_target_profiles",
    "resolve_profile_run_spec",
]


if __name__ == "__main__":
    raise SystemExit(dispatch_cli())
