from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np

from main import MAIN_CONFIG
from tools.rg_mst import estimate_rg_significance
from unified_experiments import infer_default_env_mode, looks_like_hierarchy_dataset, resolve_dataset_path


def _resolved_main_config_local() -> dict[str, Any]:
    config = deepcopy(MAIN_CONFIG)
    config["dataset"] = resolve_dataset_path(str(config["dataset"]))
    return config


def _strategy_params_for_local(
    config: dict[str, Any],
    strategy_name: str,
    *,
    env_mode: str | None = None,
) -> dict[str, Any]:
    strategy_defaults = config.get("strategy_defaults", {})
    defaults = dict(strategy_defaults.get(str(strategy_name), {})) if isinstance(strategy_defaults, dict) else {}
    if env_mode not in {None, "", "auto"}:
        defaults["env_mode"] = env_mode
    elif config.get("env_mode") not in {None, "", "auto"}:
        defaults["env_mode"] = config.get("env_mode")
    return defaults


def pairwise_cluster_geometry(positions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    valid_labels = sorted({int(label) for label in np.asarray(labels, dtype=int).tolist() if int(label) >= 0})
    if len(valid_labels) < 2:
        return {}

    positions = np.asarray(positions, dtype=float)
    centroids: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}
    points_by_label: dict[int, np.ndarray] = {}

    for label in valid_labels:
        pts = positions[np.asarray(labels, dtype=int) == label]
        points_by_label[label] = pts
        centroid = np.mean(pts, axis=0)
        centroids[label] = centroid
        radii[label] = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if len(pts) else 0.0

    centroid_dists: list[float] = []
    boundary_gaps: list[float] = []
    nearest_point_dists: list[float] = []

    for idx, label_i in enumerate(valid_labels):
        for label_j in valid_labels[idx + 1:]:
            pts_i = points_by_label[label_i]
            pts_j = points_by_label[label_j]
            centroid_dist = float(np.linalg.norm(centroids[label_j] - centroids[label_i]))
            centroid_dists.append(centroid_dist)
            boundary_gaps.append(centroid_dist - radii[label_i] - radii[label_j])
            inter = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=2)
            nearest_point_dists.append(float(np.min(inter)))

    return {
        "cluster_count": int(len(valid_labels)),
        "centroid_min": float(np.min(centroid_dists)),
        "boundary_gap_min": float(np.min(boundary_gaps)),
        "nearest_point_min": float(np.min(nearest_point_dists)),
    }


def level_position_stats(positions: np.ndarray) -> dict[str, float]:
    pos = np.asarray(positions, dtype=float)
    if pos.shape[0] < 2:
        return {
            "point_count": float(pos.shape[0]),
            "bbox_norm": float(np.max(np.linalg.norm(pos, axis=1))) if pos.size else 0.0,
        }
    diffs = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    tri = dist[np.triu_indices(pos.shape[0], 1)]
    finite_tri = tri[np.isfinite(tri)]
    return {
        "point_count": float(pos.shape[0]),
        "bbox_norm": float(np.max(np.linalg.norm(pos, axis=1))),
        "nearest_mean": float(np.mean(nearest)),
        "nearest_median": float(np.median(nearest)),
        "pairwise_mean": float(np.mean(finite_tri)),
        "pairwise_median": float(np.median(finite_tri)),
        "pairwise_max": float(np.max(finite_tri)),
    }


def level_support_position_stats(model: Any, level_index: int) -> dict[str, Any]:
    level = model.levels[level_index]
    support_mask = np.asarray(model._level_support_masks[level_index], dtype=bool)
    support_idx = np.flatnonzero(support_mask)
    stats = level_position_stats(np.asarray(level.syncmap[support_idx], dtype=float))
    stats["support_count"] = int(support_idx.size)
    return stats


def _pair_metrics_by_label(positions: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels, dtype=int).reshape(-1)
    valid_labels = sorted({int(label) for label in lab.tolist() if int(label) >= 0})
    rows: list[dict[str, Any]] = []
    for idx, label_i in enumerate(valid_labels):
        pts_i = pos[lab == label_i]
        if pts_i.shape[0] == 0:
            continue
        centroid_i = np.mean(pts_i, axis=0)
        radius_i = float(np.max(np.linalg.norm(pts_i - centroid_i, axis=1))) if pts_i.shape[0] > 0 else 0.0
        for label_j in valid_labels[idx + 1:]:
            pts_j = pos[lab == label_j]
            if pts_j.shape[0] == 0:
                continue
            centroid_j = np.mean(pts_j, axis=0)
            radius_j = float(np.max(np.linalg.norm(pts_j - centroid_j, axis=1))) if pts_j.shape[0] > 0 else 0.0
            inter = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=2)
            centroid_dist = float(np.linalg.norm(centroid_j - centroid_i))
            rows.append(
                {
                    "label_a": int(label_i),
                    "label_b": int(label_j),
                    "centroid_distance": centroid_dist,
                    "nearest_distance": float(np.min(inter)),
                    "boundary_gap": float(centroid_dist - radius_i - radius_j),
                }
            )
    return rows


def _within_group_rows(positions: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels, dtype=int).reshape(-1)
    valid_labels = sorted({int(label) for label in lab.tolist() if int(label) >= 0})
    rows: list[dict[str, Any]] = []
    for label in valid_labels:
        pts = pos[lab == label]
        if pts.shape[0] < 2:
            rows.append(
                {
                    "label": int(label),
                    "token_count": int(pts.shape[0]),
                    "nearest_median": None,
                    "pairwise_median": None,
                }
            )
            continue
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        nearest = np.min(dist, axis=1)
        tri = dist[np.triu_indices(pts.shape[0], 1)]
        finite_tri = tri[np.isfinite(tri)]
        rows.append(
            {
                "label": int(label),
                "token_count": int(pts.shape[0]),
                "nearest_median": float(np.median(nearest)),
                "pairwise_median": float(np.median(finite_tri)) if finite_tri.size > 0 else None,
            }
        )
    return rows


def summarize_true_group_geometry(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    same_target: float,
    diff_target: float,
    purity_values: list[float] | None = None,
) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels, dtype=int).reshape(-1)
    valid_labels = sorted({int(label) for label in lab.tolist() if int(label) >= 0})
    within_rows = _within_group_rows(pos, lab)
    between_rows = _pair_metrics_by_label(pos, lab)

    same_vals = [float(row["nearest_median"]) for row in within_rows if row.get("nearest_median") is not None]
    diff_vals = [float(row["nearest_distance"]) for row in between_rows if row.get("nearest_distance") is not None]
    same_rel_errors = [abs(val - same_target) / max(same_target, 1e-6) for val in same_vals]
    same_too_close = [max(0.0, same_target - val) / max(same_target, 1e-6) for val in same_vals]
    same_too_far = [max(0.0, val - same_target) / max(same_target, 1e-6) for val in same_vals]
    diff_deficit = [max(0.0, diff_target - val) / max(diff_target, 1e-6) for val in diff_vals]
    diff_surplus = [max(0.0, val - diff_target) / max(diff_target, 1e-6) for val in diff_vals]

    summary: dict[str, Any] = {
        "true_group_count": int(len(valid_labels)),
        "within_group_count": int(len(within_rows)),
        "between_group_pair_count": int(len(between_rows)),
        "same_group_target_rel_error_mean": float(np.mean(same_rel_errors)) if same_rel_errors else None,
        "same_group_target_rel_error_max": float(np.max(same_rel_errors)) if same_rel_errors else None,
        "same_group_target_rel_error_std": float(np.std(same_rel_errors)) if same_rel_errors else None,
        "same_group_too_close_rel_mean": float(np.mean(same_too_close)) if same_too_close else None,
        "same_group_too_far_rel_mean": float(np.mean(same_too_far)) if same_too_far else None,
        "cross_group_target_deficit_rel_mean": float(np.mean(diff_deficit)) if diff_deficit else None,
        "cross_group_target_deficit_rel_max": float(np.max(diff_deficit)) if diff_deficit else None,
        "cross_group_target_surplus_rel_mean": float(np.mean(diff_surplus)) if diff_surplus else None,
    }
    if purity_values:
        summary["dominant_true_purity_mean"] = float(np.mean(purity_values))
    return {
        "target_same_distance": float(same_target),
        "target_diff_distance": float(diff_target),
        "within_true_groups": within_rows,
        "between_true_groups": between_rows,
        "summary": summary,
    }


def true_leaf_level_geometry(model: Any, env: Any) -> dict[str, Any]:
    level0 = getattr(model, "levels", [getattr(model, "level0", None)])[0]
    positions = np.asarray(getattr(level0, "syncmap", getattr(model, "syncmap", np.zeros((0, 2)))), dtype=float)
    try:
        true_labels = np.asarray(env.trueLabel(level=0), dtype=int)
    except TypeError:
        true_labels = np.asarray(env.trueLabel(), dtype=int)
    if positions.ndim != 2 or true_labels.ndim != 1 or positions.shape[0] != true_labels.shape[0]:
        return {}

    same_target = float(model._level_equilibrium_distance(0))
    try:
        diff_target = float(model._level_equilibrium_distance(1))
    except Exception:
        diff_target = float(same_target * float(getattr(model, "inter_cluster_distance_ratio", 3.16)))

    result = summarize_true_group_geometry(
        positions,
        true_labels,
        same_target=same_target,
        diff_target=diff_target,
        purity_values=None,
    )
    result["true_level_index"] = 0
    return result


def true_group_support_geometry(
    model: Any,
    env: Any,
    lower_level_index: int,
    source_tokens: np.ndarray,
    support_tokens: np.ndarray,
) -> dict[str, Any]:
    true_level_index = int(lower_level_index + 1)
    hierarchy_depth = int(getattr(env, "hierarchy_depth", 1))
    if true_level_index >= hierarchy_depth:
        return {}

    true_labels = np.asarray(env.trueLabel(level=true_level_index), dtype=int)
    source_arr = np.asarray(source_tokens, dtype=int).reshape(-1)
    support_arr = np.asarray(support_tokens, dtype=int).reshape(-1)
    if source_arr.shape[0] != true_labels.shape[0] or support_arr.size == 0:
        return {}

    upper_positions = np.asarray(model.levels[lower_level_index + 1].syncmap[support_arr], dtype=float)
    token_label_counts: list[dict[str, Any]] = []
    dominant_labels = np.full(support_arr.shape[0], -1, dtype=int)
    purities: list[float] = []
    for idx, token in enumerate(support_arr.tolist()):
        members = np.where(source_arr == int(token))[0]
        counts = Counter(int(x) for x in true_labels[members].tolist() if int(x) >= 0)
        dominant = -1
        purity = 0.0
        if counts:
            dominant, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
            purity = float(count) / float(sum(counts.values()))
        dominant_labels[idx] = int(dominant)
        purities.append(float(purity))
        token_label_counts.append(
            {
                "token": int(token),
                "dominant_true_label": int(dominant),
                "purity": float(purity),
                "counts": {str(int(k)): int(v) for k, v in sorted(counts.items())},
            }
        )

    same_target = float(model._level_equilibrium_distance(lower_level_index + 1))
    try:
        diff_target = float(model._level_equilibrium_distance(lower_level_index + 2))
    except Exception:
        diff_target = float(same_target * float(getattr(model, "inter_cluster_distance_ratio", 3.16)))
    result = summarize_true_group_geometry(
        upper_positions,
        dominant_labels,
        same_target=same_target,
        diff_target=diff_target,
        purity_values=purities,
    )
    result.update(
        {
        "true_level_index": true_level_index,
        "token_true_label_counts": token_label_counts,
        }
    )
    summary = result.setdefault("summary", {})
    summary["dominant_true_group_count"] = int(len({int(x) for x in dominant_labels.tolist() if int(x) >= 0}))
    return result


def representative_upper_level_state(model: Any, env: Any, lower_level_index: int) -> dict[str, Any]:
    raw_labels = model._current_raw_labels(lower_level_index)
    stable_tokens = model._update_stable_tokens_for_level(lower_level_index, raw_labels)
    macro_tokens = model._update_macro_tokens_for_level(lower_level_index, stable_tokens)
    source_tokens = model._source_tokens_for_level(lower_level_index, raw_labels, stable_tokens, macro_tokens)
    active_tokens, token_labels = model._cluster_upper_level_tokens(lower_level_index, source_tokens)
    upper_index = int(lower_level_index + 1)
    upper_model = model.levels[upper_index]
    active_stable_tokens, stable_centroids, _ = model._stable_token_geometry_for_level(lower_level_index, stable_tokens)
    intrinsic_scales = model._stable_token_internal_scales_for_level(lower_level_index, stable_tokens)
    significance = estimate_rg_significance(
        active_stable_tokens,
        stable_centroids,
        knn_k=int(getattr(model, "rg_knn_k", 3)),
        jump_ratio=float(getattr(model, "rg_jump_ratio", 1.75)),
        fallback_scale=float(getattr(model, "rg_fallback_scale", 1.5)),
        local_scales_override=intrinsic_scales,
    )

    result: dict[str, Any] = {
        "lower_level_index": int(lower_level_index),
        "upper_level_index": upper_index,
        "target_distance": float(model._level_equilibrium_distance(upper_index)),
        "stable_token_count": int(len({int(x) for x in np.asarray(stable_tokens, dtype=int).tolist() if int(x) >= 0})),
        "macro_token_count": int(len({int(x) for x in np.asarray(macro_tokens, dtype=int).tolist() if int(x) >= 0})),
        "source_token_count": int(len({int(x) for x in np.asarray(source_tokens, dtype=int).tolist() if int(x) >= 0})),
        "active_token_count": int(np.asarray(active_tokens, dtype=int).size),
        "predicted_cluster_count": int(len({int(x) for x in np.asarray(token_labels, dtype=int).tolist() if int(x) >= 0})),
        "stable_tokens": np.asarray(stable_tokens, dtype=int).tolist(),
        "macro_tokens": np.asarray(macro_tokens, dtype=int).tolist(),
        "source_tokens": np.asarray(source_tokens, dtype=int).tolist(),
        "active_tokens": np.asarray(active_tokens, dtype=int).tolist(),
        "predicted_cluster_labels": np.asarray(token_labels, dtype=int).tolist(),
        "significance": {
            "threshold_norm": float(significance.threshold_norm),
            "threshold_distance": float(significance.threshold_distance),
            "best_jump_ratio": float(significance.best_jump_ratio),
            "component_count": int(significance.component_count),
            "component_labels": np.asarray(significance.component_labels, dtype=int).tolist(),
            "local_scales": np.asarray(significance.local_scales, dtype=float).tolist(),
            "intrinsic_scales": np.asarray(intrinsic_scales, dtype=float).tolist(),
            "mst_normalized_lengths": np.asarray(significance.mst_normalized_lengths, dtype=float).tolist(),
        },
    }
    evidence_state = getattr(model, "get_stable_token_evidence_state", None)
    if lower_level_index == 0 and callable(evidence_state):
        result["stable_token_evidence"] = evidence_state()

    support_tokens = np.array(
        sorted({int(x) for x in np.asarray(source_tokens, dtype=int).tolist() if int(x) >= 0}),
        dtype=int,
    )
    result["support_tokens"] = support_tokens.tolist()
    result["support_token_count"] = int(support_tokens.size)
    result["true_group_geometry"] = true_group_support_geometry(
        model,
        env,
        lower_level_index,
        source_tokens,
        support_tokens,
    )

    if np.asarray(active_tokens, dtype=int).size > 0:
        active_positions = np.asarray(upper_model.syncmap[np.asarray(active_tokens, dtype=int)], dtype=float)
        result["token_position_stats"] = level_position_stats(active_positions)
        result["token_cluster_geometry"] = pairwise_cluster_geometry(active_positions, np.asarray(token_labels, dtype=int))
    else:
        result["token_position_stats"] = {}
        result["token_cluster_geometry"] = {}

    raw_rows = getattr(upper_model, "_g_rows", None)
    raw_cols = getattr(upper_model, "_g_cols", None)
    raw_weights = getattr(upper_model, "_g_w", None)
    rows = np.asarray(raw_rows if raw_rows is not None else np.asarray([], dtype=int), dtype=int)
    cols = np.asarray(raw_cols if raw_cols is not None else np.asarray([], dtype=int), dtype=int)
    weights = np.asarray(raw_weights if raw_weights is not None else np.asarray([], dtype=float), dtype=float)
    support_set = {int(token) for token in support_tokens.tolist()}
    active_set = {int(token) for token in np.asarray(active_tokens, dtype=int).tolist()}
    support_edge_rows: list[int] = []
    support_edge_cols: list[int] = []
    support_edge_weights: list[float] = []
    active_edge_rows: list[int] = []
    active_edge_cols: list[int] = []
    active_edge_weights: list[float] = []
    for row, col, weight in zip(rows.tolist(), cols.tolist(), weights.tolist()):
        if int(row) in support_set and int(col) in support_set:
            support_edge_rows.append(int(row))
            support_edge_cols.append(int(col))
            support_edge_weights.append(float(weight))
        if int(row) in active_set and int(col) in active_set:
            active_edge_rows.append(int(row))
            active_edge_cols.append(int(col))
            active_edge_weights.append(float(weight))
    result["support_graph"] = {
        "edge_count": int(len(support_edge_rows)),
        "rows": support_edge_rows,
        "cols": support_edge_cols,
        "weights": support_edge_weights,
    }
    result["active_graph"] = {
        "edge_count": int(len(active_edge_rows)),
        "rows": active_edge_rows,
        "cols": active_edge_cols,
        "weights": active_edge_weights,
    }

    coact = getattr(upper_model, "_coact", None)
    raw_support_similarity_edges: list[dict[str, Any]] = []
    if coact is not None and support_tokens.size >= 2:
        try:
            sim = np.asarray(coact.similarity(), dtype=float)
        except Exception:
            sim = np.asarray([], dtype=float)
        if sim.size > 0:
            support_idx = np.asarray(support_tokens, dtype=int)
            sub = sim[np.ix_(support_idx, support_idx)]
            for local_i in range(sub.shape[0]):
                for local_j in range(local_i + 1, sub.shape[1]):
                    w = float(sub[local_i, local_j])
                    if w <= 0.0:
                        continue
                    raw_support_similarity_edges.append(
                        {
                            "row": int(support_idx[local_i]),
                            "col": int(support_idx[local_j]),
                            "weight": w,
                        }
                    )
    raw_support_similarity_edges.sort(key=lambda item: item["weight"], reverse=True)
    result["graph_build"] = {
        "knn": int(getattr(upper_model, "knn", 0)),
        "graph_theta": float(getattr(upper_model, "graph_theta", 0.0)),
        "raw_support_similarity_edges": raw_support_similarity_edges,
    }

    pair_evidence_map = {}
    if 0 <= int(lower_level_index) < len(getattr(model, "_pair_relation_evidence_by_level", [])):
        pair_evidence_map = dict(model._pair_relation_evidence_by_level[int(lower_level_index)])
    relation_edges: list[dict[str, Any]] = []
    threshold = float(getattr(model, "persistent_relation_threshold", 0.0))
    for (a, b), value in pair_evidence_map.items():
        if int(a) not in support_set or int(b) not in support_set:
            continue
        if float(value) >= threshold > 0.0:
            state = "same"
        elif float(value) <= -threshold < 0.0 or (threshold > 0.0 and float(value) <= -threshold):
            state = "diff"
        else:
            state = "undecided"
        relation_edges.append(
            {
                "row": int(a),
                "col": int(b),
                "evidence": float(value),
                "state": state,
            }
        )
    relation_edges.sort(key=lambda item: abs(float(item["evidence"])), reverse=True)
    result["persistent_relation"] = {
        "enabled": bool(getattr(model, "persistent_relation_force_enabled", False)),
        "threshold": threshold,
        "edges": relation_edges,
    }
    return result


def build_params(dataset: str, strategy: str, env_mode: str | None, *, seed: int | None = None) -> dict[str, Any]:
    mode = env_mode
    if mode in {None, "", "auto"}:
        mode = infer_default_env_mode(dataset, strategy)
        if looks_like_hierarchy_dataset(dataset) and str(strategy).lower() != "ci":
            mode = "hierarchy_sequence"
    config = _resolved_main_config_local()
    params = _strategy_params_for_local(config, strategy, env_mode=mode)
    params["env_mode"] = mode
    if seed is not None and "clustering_random_state" in params:
        params["clustering_random_state"] = int(seed)
    return params
