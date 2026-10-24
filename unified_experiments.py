#!/usr/bin/env python
from __future__ import annotations
"""
Unified Experiment Library for SyncMap
Runs single tests and batch strategy comparisons.

Terminology:
- "adaptation_rate" (code): Learning rate parameter in the algorithm

This module provides library functions for running SyncMap experiments.
Import and call the functions from your own script for type-checked usage.
"""

import importlib
import hashlib
import json
import networkx as nx
import numpy as np
import pandas as pd
import inspect
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Literal, Iterator, Callable, cast

from test.GraphWalkTest import GraphWalkTest
from test.GraphWalkTestTokenize import GraphWalkTestTokenize
from test.HierarchySequenceTest import HierarchySequenceTest
from test.HierarchySequenceTestTokenize import HierarchySequenceTestTokenize
from test.HierarchySequenceTestCI import HierarchySequenceTestCI
import syncmap.SyncMapNS as SyncMapNS
import syncmap.SyncMapX_NSGD as SyncMapX_NSGD
from tools.shared_defaults import DEFAULT_HIERARCHY_ORGANIZE_LEVEL
from test.hierarchy_levels import (
    canonicalize_hierarchy_outputs,
    canonicalize_organize_level,
    cluster_level_key_from_index,
    parse_hierarchy_label_index,
)
from tools.clustering_strategies import DBSCANClustering, create_clustering_strategy

try:
    from test.GraphWalkTestCI import GraphWalkTestCI
except ModuleNotFoundError:
    GraphWalkTestCI = None

try:
    import syncmap.SyncMapX_CI as SyncMapX_CI
except ModuleNotFoundError:
    SyncMapX_CI = None

try:
    import syncmap.SyncMap as SyncMapBaseline
except ModuleNotFoundError:
    SyncMapBaseline = None

try:
    import syncmap.SyncMapTSF as SyncMapTSF
except ModuleNotFoundError:
    SyncMapTSF = None

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import time
import re
from tools.metric_utils import clustering_metric_bundle, cophenetic_score as metric_cophenetic_score
from tools.result_paths import (
    artifact_result_path,
    experiment_artifact_path,
    experiment_archive_dir,
    experiment_archive_path,
    result_dir,
    trace_result_paths,
    timestamp_now,
)


def ensure_profile_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill ``profile_name`` for CSVs written before the identity cleanup.

    Current comparison rows carry exactly two identity columns:
    ``engine_strategy`` (which engine ran) and ``profile_name`` (which
    maintained preset/label it ran under). Older CSVs spread the same value
    over ``method`` / ``strategy`` / ``runtime_strategy`` / ``comparison_name``;
    this fills ``profile_name`` from those so old files keep summarizing.
    """
    if 'profile_name' not in df.columns:
        df = df.copy()
        df['profile_name'] = ''
    profile = df['profile_name'].fillna('').astype(str).str.strip()
    for column in ('method', 'strategy', 'comparison_name', 'engine_strategy'):
        empty = profile == ''
        if not empty.any():
            break
        if column in df.columns:
            profile = profile.mask(empty, df[column].fillna('').astype(str).str.strip())
    # Historical artifacts used a temporary code for the original SyncMap
    # baseline. Keep raw CSVs immutable, but normalize the displayed profile.
    df['profile_name'] = profile.str.lower().replace({'strict': 'syncmap'})
    return df

RG_SIGNIFICANT_RATIO = 3.16
DEFAULT_EVAL_READOUT_METHOD = "auto"
DEFAULT_EVAL_READOUT_MIN_SAMPLES = 2
DEFAULT_EVAL_READOUT_MAX_CLUSTERS = 16


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

StrategyName = Literal['ns', 'nsgd', 'tsf', 'ci', 'syncmap', 'rg', 'symmetric']

DEPRECATED_STRATEGY_HINTS: Dict[str, str] = {
    'strict': "Use --strategy syncmap for the original SyncMap baseline.",
}

StrategyParams = Dict[str, Any]
StrategyConfig = Dict[str, StrategyParams]
EvaluationMetricName = Literal['nmi', 'ari']

COMPARISON_RUNTIME_STRATEGY_KEY = "__runtime_strategy"
COMPARISON_ENGINE_STRATEGY_KEY = "__engine_strategy"
COMPARISON_PROFILE_NAME_KEY = "__profile_name"
COMPARISON_DISPLAY_NAME_KEY = "__display_name"


def canonicalize_strategy_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")


def known_strategy_names() -> List[str]:
    return ['ns', 'nsgd', 'tsf', 'ci', 'syncmap', 'rg', 'symmetric']


@lru_cache(maxsize=256)
def looks_like_hierarchy_dataset(dataset_path: str) -> bool:
    normalized = str(dataset_path).replace("\\", "/").lower()
    if "/hierarchy/" in normalized:
        return True
    try:
        from test.graph_dataset_loader import inspect_dot_dataset

        resolved = resolve_dataset_path(str(dataset_path))
        return bool(inspect_dot_dataset(resolved).is_hierarchical)
    except Exception:
        return False


def resolve_dataset_path(dataset_path: str) -> str:
    text = str(dataset_path).strip()
    path = Path(text)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path.resolve())
    repo_root = Path(__file__).resolve().parent
    repo_candidate = repo_root / text
    if repo_candidate.exists():
        return str(repo_candidate)
    dataset_candidate = repo_root / "datasets" / text
    if dataset_candidate.exists():
        return str(dataset_candidate)
    hierarchy_candidate = repo_root / "datasets" / "hierarchy" / text
    if hierarchy_candidate.exists():
        return str(hierarchy_candidate)
    return text


def classify_dataset_family(dataset_path: str) -> str:
    """Classify a dataset as flat or hierarchy for reporting."""
    return "hierarchy" if looks_like_hierarchy_dataset(dataset_path) else "flat"


def _relabel_cluster_ids(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    relabeled = np.full(arr.shape[0], -1, dtype=int)
    valid = sorted({int(v) for v in arr.tolist() if int(v) >= 0})
    mapping = {label: idx for idx, label in enumerate(valid)}
    for idx, label in enumerate(arr.tolist()):
        if int(label) >= 0:
            relabeled[idx] = mapping[int(label)]
    return relabeled


def _call_organize(model: Any, level: str | None = None) -> np.ndarray:
    organizer = getattr(model, "organize")
    if level is None:
        return np.asarray(organizer(), dtype=int)
    try:
        return np.asarray(organizer(level=level), dtype=int)
    except TypeError:
        return np.asarray(organizer(), dtype=int)


def _lift_cluster_centroids(
    node_positions: np.ndarray,
    lower_labels: np.ndarray,
    *,
    eps: float,
    min_samples: int = 2,
) -> np.ndarray:
    labels = np.asarray(lower_labels, dtype=int).reshape(-1)
    positions = np.asarray(node_positions, dtype=float)
    lifted = np.full(labels.shape[0], -1, dtype=int)
    valid = sorted({int(v) for v in labels.tolist() if int(v) >= 0})
    if not valid:
        return lifted
    if len(valid) == 1:
        lifted[labels >= 0] = 0
        return lifted

    centroids = []
    for label in valid:
        centroids.append(np.mean(positions[labels == label], axis=0))
    centroid_arr = np.asarray(centroids, dtype=float)
    upper = DBSCANClustering(eps=float(eps), min_samples=max(1, int(min_samples))).fit_predict(centroid_arr)
    upper = _relabel_cluster_ids(upper)
    for group_idx, label in enumerate(valid):
        lifted[labels == label] = int(upper[group_idx])
    return lifted


def generic_hierarchy_outputs(model: Any, env: Any) -> dict[str, np.ndarray]:
    """Dataset-driven hierarchy readout shared by live UI and formal trace scoring."""
    with _rng_neutral():
        return _generic_hierarchy_outputs_impl(model, env)


def _generic_hierarchy_outputs_impl(model: Any, env: Any) -> dict[str, np.ndarray]:
    base_labels = _relabel_cluster_ids(_call_organize(model))
    outputs: dict[str, np.ndarray] = {
        "cluster_level1": base_labels.copy(),
        "level0": base_labels.copy(),
    }

    depth = max(1, int(getattr(env, "hierarchy_depth", 1)))
    if not bool(getattr(env, "is_hierarchical", False)) or depth <= 1:
        return canonicalize_hierarchy_outputs(outputs)

    organizer = getattr(model, "organize_hierarchy", None)
    if callable(organizer):
        return canonicalize_hierarchy_outputs(organizer())

    syncmap = np.asarray(getattr(model, "syncmap"), dtype=float)
    base_eps = float(getattr(model, "dbscan_eps", 3.0))
    ratio = float(getattr(model, "inter_cluster_distance_ratio", RG_SIGNIFICANT_RATIO))
    min_samples = int(getattr(model, "dbscan_min_samples", 2))

    current = base_labels.copy()
    for level_idx in range(1, depth):
        scaled_eps = float(base_eps * (ratio ** level_idx))
        current = _lift_cluster_centroids(syncmap, current, eps=scaled_eps, min_samples=min_samples)
        outputs[f"cluster_level{level_idx + 1}"] = current.copy()
        outputs[f"level{level_idx}"] = current.copy()
    return canonicalize_hierarchy_outputs(outputs)


@dataclass(frozen=True)
class EvaluationReadoutConfig:
    method: str = DEFAULT_EVAL_READOUT_METHOD
    eps: float | None = None
    min_samples: int = DEFAULT_EVAL_READOUT_MIN_SAMPLES
    max_clusters: int = DEFAULT_EVAL_READOUT_MAX_CLUSTERS
    standardize: bool = True


def _pop_evaluation_readout_config(params: dict[str, Any]) -> EvaluationReadoutConfig:
    raw = dict(params)
    legacy_source = str(raw.pop("eval_readout_source", "")).strip().lower()
    method = str(raw.pop("eval_readout_method", DEFAULT_EVAL_READOUT_METHOD)).strip().lower() or DEFAULT_EVAL_READOUT_METHOD
    eps_raw = raw.pop("eval_readout_eps", None)
    min_samples_raw = raw.pop("eval_readout_min_samples", DEFAULT_EVAL_READOUT_MIN_SAMPLES)
    max_clusters_raw = raw.pop("eval_readout_max_clusters", DEFAULT_EVAL_READOUT_MAX_CLUSTERS)
    standardize_raw = raw.pop("eval_readout_standardize", True)

    params.clear()
    params.update(raw)

    eps = None if eps_raw in {None, "", "none", "auto"} else float(eps_raw)
    min_samples = max(1, int(min_samples_raw))
    max_clusters = max(2, int(max_clusters_raw))
    standardize = bool(standardize_raw)
    if legacy_source not in {"", "external", "unified"}:
        raise ValueError(
            "eval_readout_source was removed: formal evaluation has one unified "
            "geometry readout. Remove the parameter instead of selecting an "
            f"algorithm-specific readout (got {legacy_source!r})."
        )
    if method not in {"auto", "dbscan", "hdbscan", "louvain", "agglomerative", "spectral", "oracle", "mst_gaps", "mst_ladder"}:
        raise ValueError(
            f"Unsupported eval_readout_method={method!r}; expected one of auto/dbscan/hdbscan/louvain/agglomerative/spectral/oracle/mst_gaps/mst_ladder"
        )
    return EvaluationReadoutConfig(
        method=method,
        eps=eps,
        min_samples=min_samples,
        max_clusters=max_clusters,
        standardize=standardize,
    )


def _embedding_points(model: Any) -> np.ndarray:
    points = np.asarray(getattr(model, "syncmap"), dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D embedding array on model.syncmap, got shape={points.shape}")
    return points


def _standardize_embedding(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr.copy()
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return centered
    nn = min(centered.shape[0], 3)
    nbrs = NearestNeighbors(n_neighbors=nn)
    nbrs.fit(centered)
    distances, _ = nbrs.kneighbors(centered)
    if distances.shape[1] >= 2:
        local = distances[:, 1]
    else:
        local = np.linalg.norm(centered - centered[[0]], axis=1)
    scale = float(np.median(local[local > 0])) if np.any(local > 0) else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return centered / scale


def _estimate_external_dbscan_eps(points: np.ndarray, min_samples: int) -> float:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] <= 1:
        return 1.0
    k = min(arr.shape[0], max(2, int(min_samples) + 1))
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(arr)
    distances, _ = nbrs.kneighbors(arr)
    k_dist = distances[:, -1]
    positive = k_dist[k_dist > 0]
    if positive.size == 0:
        return 1.0
    return float(max(1e-6, np.median(positive) * 1.25))


def _estimate_spectral_cluster_count(points: np.ndarray, max_clusters: int) -> int:
    arr = np.asarray(points, dtype=float)
    n = int(arr.shape[0])
    if n <= 2:
        return max(1, n)
    nn = max(2, min(n - 1, int(round(np.sqrt(n)))))
    nbrs = NearestNeighbors(n_neighbors=nn + 1)
    nbrs.fit(arr)
    _, indices = nbrs.kneighbors(arr)
    graph = np.zeros((n, n), dtype=float)
    for row in range(n):
        for col in indices[row, 1:]:
            graph[row, int(col)] = 1.0
            graph[int(col), row] = 1.0
    degree = np.diag(np.sum(graph, axis=1))
    laplacian = degree - graph
    eigvals = np.linalg.eigvalsh(laplacian)
    upper = min(max_clusters, n - 1)
    if upper <= 1:
        return 1
    search = eigvals[: upper + 1]
    gaps = np.diff(search)
    if gaps.size == 0:
        return min(2, upper)
    best_k = int(np.argmax(gaps[:upper]) + 1)
    return max(2, min(upper, best_k))


def _resolve_external_readout_method(config: EvaluationReadoutConfig, env: Any) -> str:
    method = str(config.method).strip().lower()
    if method != "auto":
        return method
    if bool(getattr(env, "is_hierarchical", False)) and max(1, int(getattr(env, "hierarchy_depth", 1))) > 1:
        return "mst_gaps"
    return "dbscan"


def _external_mst_gap_hierarchy_outputs(
    points: np.ndarray,
    *,
    required_levels: int,
    min_samples: int = 2,
    min_gap_ratio: float = 1.05,
) -> dict[str, np.ndarray]:
    arr = np.asarray(points, dtype=float)
    node_count = int(arr.shape[0]) if arr.ndim == 2 else 0
    outputs: dict[str, np.ndarray] = {}
    if node_count <= 0 or required_levels <= 0:
        return outputs
    if node_count == 1:
        base = np.asarray([0], dtype=int)
        for level_idx in range(required_levels):
            outputs[f"cluster_level{level_idx + 1}"] = base.copy()
            outputs[f"level{level_idx}"] = base.copy()
        return outputs

    distances = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_weighted_edges_from(
        (i, j, float(distances[i, j]))
        for i in range(node_count)
        for j in range(i + 1, node_count)
    )
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    edge_lengths = sorted(float(data["weight"]) for _, _, data in mst.edges(data=True))
    if len(edge_lengths) < 2:
        one = np.zeros(node_count, dtype=int)
        for level_idx in range(required_levels):
            outputs[f"cluster_level{level_idx + 1}"] = one.copy()
            outputs[f"level{level_idx}"] = one.copy()
        return outputs

    def mst_cut_labels(eps: float) -> np.ndarray | None:
        if int(min_samples) > 2:
            return None
        cut_graph = nx.Graph()
        cut_graph.add_nodes_from(range(node_count))
        cut_graph.add_edges_from(
            (int(i), int(j))
            for i, j, data in mst.edges(data=True)
            if float(data["weight"]) <= float(eps)
        )
        labels = np.full(node_count, -1, dtype=int)
        next_label = 0
        for component in nx.connected_components(cut_graph):
            members = np.asarray(sorted(int(node) for node in component), dtype=int)
            if members.size < int(min_samples):
                continue
            labels[members] = next_label
            next_label += 1
        return labels

    candidates: list[tuple[float, int, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    ratio_floor = max(float(min_gap_ratio), 1.0)
    for lower, upper in zip(edge_lengths[:-1], edge_lengths[1:]):
        if lower <= 1e-12:
            continue
        jump_ratio = float(upper / lower)
        if jump_ratio < ratio_floor:
            continue
        eps = float(np.sqrt(lower * upper))
        labels = mst_cut_labels(eps)
        if labels is None:
            labels = np.asarray(
                create_clustering_strategy(
                    "dbscan",
                    eps=eps,
                    min_samples=max(1, int(min_samples)),
                ).fit_predict(arr),
                dtype=int,
            )
        if np.any(labels < 0):
            continue
        labels = _relabel_cluster_ids(labels)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_count = int(unique.size)
        if cluster_count <= 1 or cluster_count >= node_count:
            continue
        if np.any(counts < max(1, int(min_samples))):
            continue
        signature = tuple(int(value) for value in labels.tolist())
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((jump_ratio, cluster_count, labels))

    if not candidates:
        one = np.zeros(node_count, dtype=int)
        for level_idx in range(required_levels):
            outputs[f"cluster_level{level_idx + 1}"] = one.copy()
            outputs[f"level{level_idx}"] = one.copy()
        return outputs

    selected = sorted(candidates, key=lambda item: (-item[0], -item[1]))[:required_levels]
    selected.sort(key=lambda item: -item[1])
    labels_list = [np.asarray(labels, dtype=int).copy() for _, _, labels in selected]
    while len(labels_list) < required_levels:
        labels_list.append(labels_list[-1].copy())
    for level_idx, labels in enumerate(labels_list[:required_levels]):
        outputs[f"cluster_level{level_idx + 1}"] = labels.copy()
        outputs[f"level{level_idx}"] = labels.copy()
    return outputs


def _mst_gap_cut_candidates(
    points: np.ndarray,
    *,
    min_samples: int = 2,
    min_gap_ratio: float = 1.0,
    fill_small_components: bool = False,
) -> list[tuple[float, int, np.ndarray]]:
    arr = np.asarray(points, dtype=float)
    node_count = int(arr.shape[0]) if arr.ndim == 2 else 0
    if node_count <= 1:
        return []

    distances = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_weighted_edges_from(
        (i, j, float(distances[i, j]))
        for i in range(node_count)
        for j in range(i + 1, node_count)
    )
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    edge_lengths = sorted(float(data["weight"]) for _, _, data in mst.edges(data=True))
    if len(edge_lengths) < 2:
        return []

    candidates: list[tuple[float, int, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    ratio_floor = max(float(min_gap_ratio), 1.0)
    sample_floor = max(1, int(min_samples))
    for lower, upper in zip(edge_lengths[:-1], edge_lengths[1:]):
        if lower <= 1e-12:
            continue
        jump_ratio = float(upper / lower)
        if jump_ratio < ratio_floor:
            continue
        eps = float(np.sqrt(lower * upper))
        cut_graph = nx.Graph()
        cut_graph.add_nodes_from(range(node_count))
        cut_graph.add_edges_from(
            (int(i), int(j))
            for i, j, data in mst.edges(data=True)
            if float(data["weight"]) <= eps
        )

        components = [
            np.asarray(sorted(int(node) for node in component), dtype=int)
            for component in nx.connected_components(cut_graph)
        ]
        labels = np.full(node_count, -1, dtype=int)
        valid_components: list[np.ndarray] = []
        next_label = 0
        for component in components:
            if component.size < sample_floor:
                continue
            labels[component] = next_label
            valid_components.append(component)
            next_label += 1

        if fill_small_components and valid_components:
            centroids = np.asarray(
                [np.mean(arr[component], axis=0) for component in valid_components],
                dtype=float,
            )
            for component in components:
                if component.size >= sample_floor:
                    continue
                for node in component.tolist():
                    nearest = int(np.argmin(np.linalg.norm(centroids - arr[int(node)], axis=1)))
                    labels[int(node)] = nearest

        if np.any(labels < 0):
            continue
        labels = _relabel_cluster_ids(labels)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_count = int(unique.size)
        if cluster_count <= 1 or cluster_count >= node_count:
            continue
        if np.any(counts < sample_floor):
            continue
        signature = tuple(int(value) for value in labels.tolist())
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((jump_ratio, cluster_count, labels))
    return candidates


def _local_mst_ladder_refinement(
    points: np.ndarray,
    anchor_labels: np.ndarray,
    *,
    min_samples: int = 2,
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    anchor = _relabel_cluster_ids(np.asarray(anchor_labels, dtype=int).reshape(-1))
    refined = np.full(anchor.shape, -1, dtype=int)
    next_label = 0
    sample_floor = max(1, int(min_samples))

    for anchor_id in sorted({int(value) for value in anchor.tolist() if int(value) >= 0}):
        indices = np.where(anchor == anchor_id)[0]
        if indices.size < 2 * sample_floor:
            refined[indices] = next_label
            next_label += 1
            continue

        candidates = _mst_gap_cut_candidates(
            arr[indices],
            min_samples=sample_floor,
            min_gap_ratio=1.0,
            fill_small_components=True,
        )
        if not candidates:
            refined[indices] = next_label
            next_label += 1
            continue

        # Fine labels are read inside an already accepted macro slice.  In this
        # mode the most useful unsupervised signal is the finest non-singleton
        # MST cut, with tiny components attached to their nearest valid neighbor.
        _, _, local_labels = max(candidates, key=lambda item: (item[1], item[0]))
        local_labels = _relabel_cluster_ids(np.asarray(local_labels, dtype=int))
        for local_id in sorted({int(value) for value in local_labels.tolist() if int(value) >= 0}):
            refined[indices[local_labels == local_id]] = next_label
            next_label += 1

    return _relabel_cluster_ids(refined)


def _external_mst_ladder_hierarchy_outputs(
    points: np.ndarray,
    *,
    required_levels: int,
    min_samples: int = 2,
) -> dict[str, np.ndarray]:
    arr = np.asarray(points, dtype=float)
    node_count = int(arr.shape[0]) if arr.ndim == 2 else 0
    outputs: dict[str, np.ndarray] = {}
    if node_count <= 0 or required_levels <= 0:
        return outputs
    if required_levels == 1 or node_count <= 1:
        return _external_mst_gap_hierarchy_outputs(
            arr,
            required_levels=required_levels,
            min_samples=min_samples,
            min_gap_ratio=1.05,
        )

    base = _external_mst_gap_hierarchy_outputs(
        arr,
        required_levels=required_levels,
        min_samples=min_samples,
        min_gap_ratio=1.05,
    )
    anchor = np.asarray(base.get("cluster_level2"), dtype=int)
    if anchor.shape[0] != node_count or len(set(anchor.tolist())) <= 1:
        return base

    fine = _local_mst_ladder_refinement(arr, anchor, min_samples=min_samples)
    outputs["cluster_level1"] = fine.copy()
    outputs["level0"] = fine.copy()
    for level_idx in range(1, required_levels):
        key = f"cluster_level{level_idx + 1}"
        labels = np.asarray(base.get(key, anchor), dtype=int)
        if labels.shape[0] != node_count:
            labels = anchor.copy()
        outputs[key] = labels.copy()
        outputs[f"level{level_idx}"] = labels.copy()
    return outputs


def _cluster_points_external(
    points: np.ndarray,
    config: EvaluationReadoutConfig,
    *,
    oracle_cluster_count: int | None = None,
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.asarray([], dtype=int)
    if arr.shape[0] == 1:
        return np.asarray([0], dtype=int)
    work = _standardize_embedding(arr) if bool(config.standardize) else arr.copy()
    method = str(config.method).strip().lower()

    if method == "dbscan":
        eps = float(config.eps) if config.eps is not None else _estimate_external_dbscan_eps(work, config.min_samples)
        labels = create_clustering_strategy(
            "dbscan",
            eps=float(eps),
            min_samples=max(1, int(config.min_samples)),
        ).fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    if method == "hdbscan":
        labels = create_clustering_strategy(
            "hdbscan",
            min_cluster_size=max(2, int(config.min_samples)),
            min_samples=2,
        ).fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    if method == "louvain":
        labels = create_clustering_strategy(
            "louvain",
            distance_threshold=float(config.eps) if config.eps is not None else _estimate_external_dbscan_eps(work, config.min_samples) * 1.5,
            resolution=1.0,
            knn_k=max(0, min(work.shape[0] - 1, 12)),
            mutual_knn=True,
        ).fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    if method == "agglomerative":
        if oracle_cluster_count is not None:
            labels = create_clustering_strategy(
                "agglomerative",
                n_clusters=max(1, int(oracle_cluster_count)),
                distance_threshold=None,
                linkage="ward",
            ).fit_predict(work)
        else:
            labels = create_clustering_strategy(
                "agglomerative",
                n_clusters=None,
                distance_threshold=float(config.eps) if config.eps is not None else _estimate_external_dbscan_eps(work, config.min_samples) * 1.5,
                linkage="ward",
            ).fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    if method == "spectral":
        n_clusters = int(oracle_cluster_count) if oracle_cluster_count is not None else _estimate_spectral_cluster_count(work, config.max_clusters)
        n_clusters = max(1, min(n_clusters, work.shape[0]))
        if n_clusters == 1:
            return np.zeros(work.shape[0], dtype=int)
        nn = max(2, min(work.shape[0] - 1, int(round(np.sqrt(work.shape[0])))))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=nn,
            assign_labels="kmeans",
            random_state=0,
        )
        labels = model.fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    if method == "oracle":
        if oracle_cluster_count is None:
            raise ValueError("Oracle external readout requires oracle_cluster_count from the ground-truth level.")
        labels = create_clustering_strategy(
            "agglomerative",
            n_clusters=max(1, int(oracle_cluster_count)),
            distance_threshold=None,
            linkage="ward",
        ).fit_predict(work)
        return _relabel_cluster_ids(np.asarray(labels, dtype=int))

    raise ValueError(f"Unsupported external readout method: {config.method}")


def _lift_external_cluster_labels(
    node_positions: np.ndarray,
    lower_labels: np.ndarray,
    config: EvaluationReadoutConfig,
    *,
    oracle_cluster_count: int | None = None,
) -> np.ndarray:
    labels = np.asarray(lower_labels, dtype=int).reshape(-1)
    positions = np.asarray(node_positions, dtype=float)
    lifted = np.full(labels.shape[0], -1, dtype=int)
    valid = sorted({int(v) for v in labels.tolist() if int(v) >= 0})
    if not valid:
        return lifted
    if len(valid) == 1:
        lifted[labels >= 0] = 0
        return lifted

    centroids = np.asarray([np.mean(positions[labels == label], axis=0) for label in valid], dtype=float)
    upper = _cluster_points_external(centroids, config, oracle_cluster_count=oracle_cluster_count)
    upper = _relabel_cluster_ids(np.asarray(upper, dtype=int))
    for group_idx, label in enumerate(valid):
        lifted[labels == label] = int(upper[group_idx])
    return lifted


def unified_readout_outputs(
    model: Any,
    env: Any,
    config: EvaluationReadoutConfig,
) -> dict[str, np.ndarray]:
    with _rng_neutral():
        return _unified_readout_outputs_impl(model, env, config)


def _unified_readout_outputs_impl(
    model: Any,
    env: Any,
    config: EvaluationReadoutConfig,
) -> dict[str, np.ndarray]:
    points = _embedding_points(model)
    depth = max(1, int(getattr(env, "hierarchy_depth", 1))) if bool(getattr(env, "is_hierarchical", False)) else 1
    outputs: dict[str, np.ndarray] = {}
    resolved_method = _resolve_external_readout_method(config, env)
    resolved_config = EvaluationReadoutConfig(
        method=resolved_method,
        eps=config.eps,
        min_samples=config.min_samples,
        max_clusters=config.max_clusters,
        standardize=config.standardize,
    )

    oracle_counts = None
    if resolved_method == "oracle":
        oracle_counts = [
            len(np.unique(_get_env_true_labels(env, level_idx)))
            for level_idx in range(depth)
        ]

    if resolved_method == "mst_gaps" and depth > 1:
        return canonicalize_hierarchy_outputs(
            _external_mst_gap_hierarchy_outputs(
                points,
                required_levels=depth,
                min_samples=resolved_config.min_samples,
                min_gap_ratio=1.05,
            )
        )

    if resolved_method == "mst_ladder" and depth > 1:
        return canonicalize_hierarchy_outputs(
            _external_mst_ladder_hierarchy_outputs(
                points,
                required_levels=depth,
                min_samples=resolved_config.min_samples,
            )
        )

    current = _cluster_points_external(
        points,
        resolved_config,
        oracle_cluster_count=None if oracle_counts is None else int(oracle_counts[0]),
    )
    outputs["cluster_level1"] = current.copy()
    outputs["level0"] = current.copy()

    for level_idx in range(1, depth):
        current = _lift_external_cluster_labels(
            points,
            current,
            resolved_config,
            oracle_cluster_count=None if oracle_counts is None else int(oracle_counts[level_idx]),
        )
        outputs[f"cluster_level{level_idx + 1}"] = current.copy()
        outputs[f"level{level_idx}"] = current.copy()

    return canonicalize_hierarchy_outputs(outputs)


def infer_default_env_mode(dataset_path: str, strategy_name: str) -> str:
    strategy_key = canonicalize_strategy_name(strategy_name)
    if looks_like_hierarchy_dataset(dataset_path):
        return 'hierarchy_sequence_ci' if strategy_key == 'ci' else 'hierarchy_sequence'
    return 'graphwalk_ci' if strategy_key == 'ci' else 'graphwalk'


ENV_MODE_ALIASES: Dict[str, str] = {
    'improved_hierarchy': 'hierarchy_sequence',
    'improved_hierarchy_ci': 'hierarchy_sequence_ci',
    'graphwalktokenize': 'graphwalk_tokenize',
    'graphwalkci': 'graphwalk_ci',
    'hierarchysequence': 'hierarchy_sequence',
    'hierarchysequenceci': 'hierarchy_sequence_ci',
}


def canonicalize_env_mode(env_mode: str) -> str:
    return ENV_MODE_ALIASES.get(str(env_mode).strip().lower(), str(env_mode).strip().lower())


def _create_rg_model(base_kwargs: dict[str, Any], params: dict[str, Any]) -> Any:
    try:
        rg_module = importlib.import_module("syncmap.SyncMapRG")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Strategy 'rg' requires an available SyncMapRG module in this workspace."
        ) from exc

    creator = getattr(rg_module, "create_syncmap_rg", None)
    if callable(creator):
        return creator(**base_kwargs, **params)

    rg_cls = getattr(rg_module, "SyncMapX_RG", None)
    if rg_cls is None:
        raise ValueError(
            "SyncMapRG module is missing both 'create_syncmap_rg' and 'SyncMapX_RG'."
        )
    return rg_cls(**base_kwargs, **params)






def _create_syncmap_model(base_kwargs: dict[str, Any], params: dict[str, Any]) -> Any:
    if SyncMapBaseline is None:
        raise ValueError(
            "Strategy 'syncmap' requires syncmap/SyncMap.py in this workspace."
        )
    syncmap_cls = getattr(SyncMapBaseline, "SyncMap", None)
    if syncmap_cls is None:
        raise ValueError(
            "syncmap.SyncMap is missing class 'SyncMap'."
        )
    return syncmap_cls(**base_kwargs, **params)




def _create_symmetric_model(base_kwargs: dict[str, Any], params: dict[str, Any]) -> Any:
    try:
        module = importlib.import_module("syncmap.SyncMapX_Symmetric")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Strategy 'symmetric' requires SyncMapX_Symmetric.py in this workspace."
        ) from exc

    creator = getattr(module, "create_symmetric_syncmap", None)
    if callable(creator):
        return creator(**base_kwargs, **params)

    cls = getattr(module, "SymmetricalSyncMap", None)
    if cls is None:
        raise ValueError(
            "SyncMapX_Symmetric module is missing both 'create_symmetric_syncmap' and 'SymmetricalSyncMap'."
        )
    return cls(**base_kwargs, **params)


def _normalize_strategy_model_params(strategy_key: str, params: dict[str, Any], *, env: Any) -> dict[str, Any]:
    raw = dict(params)
    if strategy_key == 'ns':
        normalized = SyncMapNS.SyncMapNS.normalize_params(raw)
        return normalized
    if strategy_key == 'nsgd':
        return SyncMapX_NSGD.SyncMapX_NSGD.normalize_params(raw)
    if strategy_key == 'tsf':
        if SyncMapTSF is None:
            raise ValueError("Strategy 'tsf' requires SyncMapTSF.py in this workspace.")
        normalized = SyncMapTSF.SyncMapTSF.normalize_params(raw)
        if 'target_levels' not in raw and bool(getattr(env, 'is_hierarchical', False)):
            normalized['target_levels'] = max(1, _hierarchy_depth(env))
        if 'max_clusters' not in raw:
            normalized['max_clusters'] = min(int(getattr(env, 'output_size', getattr(env, 'n', 16)) or 16), 16)
        return normalized
    if strategy_key == 'ci':
        if SyncMapX_CI is None:
            raise ValueError("Strategy 'ci' requires SyncMapX_CI.py in this workspace.")
        return SyncMapX_CI.SyncMapX_CI.normalize_params(raw)
    if strategy_key == 'syncmap':
        if SyncMapBaseline is None:
            raise ValueError("Strategy 'syncmap' requires syncmap/SyncMap.py in this workspace.")
        syncmap_cls = getattr(SyncMapBaseline, 'SyncMap', None)
        if syncmap_cls is None:
            raise ValueError("syncmap.SyncMap is missing class 'SyncMap'.")
        return syncmap_cls.normalize_params(raw)
    if strategy_key == 'rg':
        rg_module = importlib.import_module('syncmap.SyncMapRG')
        rg_cls = getattr(rg_module, 'SyncMapX_RG', None)
        if rg_cls is None:
            raise ValueError("SyncMapRG module is missing 'SyncMapX_RG'.")
        normalized = rg_cls.normalize_params(raw)
        if 'level1_enabled' not in raw and (not bool(getattr(env, 'is_hierarchical', False))):
            normalized['level1_enabled'] = False
        if 'level_count' not in raw and bool(getattr(env, 'is_hierarchical', False)):
            normalized['level_count'] = max(2, _hierarchy_depth(env) + 1)
        return normalized
    if strategy_key == 'symmetric':
        try:
            module = importlib.import_module('syncmap.SyncMapX_Symmetric')
        except ModuleNotFoundError as exc:
            raise ValueError("Strategy 'symmetric' requires SyncMapX_Symmetric.py in this workspace.") from exc
        cls = getattr(module, 'SymmetricalSyncMap', None)
        if cls is None:
            raise ValueError("SyncMapX_Symmetric module is missing 'SymmetricalSyncMap'.")
        return cls.normalize_params(raw)
    return raw

@dataclass(frozen=True)
class EvaluationMetric:
    name: EvaluationMetricName
    display_name: str
    compute_fn: Callable[[Any, Any], float]

    @property
    def score_key(self) -> str:
        return f"{self.name}_score"

    @property
    def curve_key(self) -> str:
        return self.name

    @property
    def final_key(self) -> str:
        return f"final_{self.name}"

    @property
    def target_only_key(self) -> str:
        return f"target_only_{self.name}"

    @property
    def with_source_key(self) -> str:
        return f"with_source_{self.name}"

    def compute(self, labels: Any, true_labels: Any) -> float:
        return float(self.compute_fn(labels, true_labels))

    def format_score(self, value: float) -> str:
        return f"{self.display_name}={value:.4f}"


DEFAULT_EVAL_METRIC: EvaluationMetricName = 'ari'
_METRIC_REGISTRY: dict[EvaluationMetricName, EvaluationMetric] = {
    'nmi': EvaluationMetric('nmi', 'NMI', lambda a, b: float(normalized_mutual_info_score(a, b))),
    'ari': EvaluationMetric('ari', 'ARI', lambda a, b: float(adjusted_rand_score(a, b))),
}


def resolve_metric(metric: EvaluationMetricName | str | None = None) -> EvaluationMetric:
    """Return the metric configuration for the requested metric name."""
    if metric is None:
        metric = DEFAULT_EVAL_METRIC
    metric_key = str(metric).strip().lower()
    if metric_key not in _METRIC_REGISTRY:
        raise ValueError(f"Unsupported metric '{metric}'. Available: {list(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[metric_key]  # type: ignore[return-value]


def detect_metric_from_frame(df: pd.DataFrame, fallback: Optional[EvaluationMetric] = None) -> EvaluationMetric:
    """Infer which metric produced the dataframe results."""
    if 'metric_name' in df.columns:
        values = df['metric_name'].dropna().unique()
        if len(values):
            try:
                return resolve_metric(str(values[0]))
            except ValueError:
                pass
    return fallback or resolve_metric()


def build_metric_context(metric: EvaluationMetric) -> dict[str, Any]:
    """Return a serializable context for downstream (e.g., HTML) consumers."""
    return {
        'name': metric.name,
        'display_name': metric.display_name,
        'score_key': metric.score_key,
        'curve_key': metric.curve_key,
        'final_key': metric.final_key,
        'target_only_key': metric.target_only_key,
        'with_source_key': metric.with_source_key,
    }


def compute_all_metric_scores(labels: Any, true_labels: Any) -> dict[str, float]:
    """Compute every registered evaluation metric for a single labeling."""
    scores: dict[str, float] = {}
    for metric in _METRIC_REGISTRY.values():
        scores[metric.score_key] = metric.compute(labels, true_labels)
    return scores


def compute_noise_sensitive_metric_scores(labels: Any, true_labels: Any) -> dict[str, float]:
    """Supplemental diagnostics for DBSCAN/HDBSCAN-style noise labels.

    sklearn metrics treat ``-1`` as a normal cluster id. Keep the raw metrics as
    the formal score, but expose common alternative noise policies so reports can
    detect when a large noise bucket is driving the interpretation.
    """
    pred = np.asarray(labels, dtype=int).reshape(-1)
    truth = np.asarray(true_labels, dtype=int).reshape(-1)
    if pred.shape != truth.shape or pred.size == 0:
        return {}
    bundle = clustering_metric_bundle(truth, pred)
    return {
        'nmi_classified_only_score': float(bundle.get('nmi_classified_only', float('nan'))),
        'ari_classified_only_score': float(bundle.get('ari_classified_only', float('nan'))),
        'classified_covered_ratio': float(bundle.get('classified_covered_ratio', float('nan'))),
        'singleton_cluster_ratio': float(bundle.get('singleton_cluster_ratio', float('nan'))),
        'noise_covered_ratio': float(bundle.get('noise_covered_ratio', float('nan'))),
        'nmi_masked_noise_score': float(bundle.get('nmi_masked_noise', float('nan'))),
        'ari_masked_noise_score': float(bundle.get('ari_masked_noise', float('nan'))),
        'nmi_noise_singletons_score': float(bundle.get('nmi_noise_singletons', float('nan'))),
        'ari_noise_singletons_score': float(bundle.get('ari_noise_singletons', float('nan'))),
    }


def compute_hierarchy_metric_diagnostics(
    model: Any,
    env: Any,
    readout_config: EvaluationReadoutConfig,
    outputs: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Per-level ARI for the active readout, keyed by the public level names.

    Column naming follows the established ``cluster_level{N}`` convention
    (1-based, ``cluster_level1`` = finest = display ``L1``). ARI only: it is
    the formal label-correctness metric; NMI stays a whole-vector
    diagnostics elsewhere.

    ``outputs`` lets callers reuse readout labels they already computed (e.g.
    from ``_select_evaluation_prediction``) so all per-level scores come from
    the same readout pass.
    """
    if not bool(getattr(env, "is_hierarchical", False)):
        return {}

    try:
        depth = _hierarchy_depth(env)
        if outputs is None:
            outputs = unified_readout_outputs(model, env, readout_config)
    except Exception:
        return {}

    ari_metric = _METRIC_REGISTRY["ari"]
    diagnostics: dict[str, float] = {}
    scored_levels = 0
    for level_idx in range(depth):
        key = cluster_level_key_from_index(level_idx)
        if key not in outputs:
            continue
        pred = np.asarray(outputs[key], dtype=int).reshape(-1)
        truth = _get_env_true_labels(env, level_idx).reshape(-1)
        if pred.shape != truth.shape or pred.size == 0:
            continue
        scored_levels += 1
        diagnostics[f"{key}_ari_score"] = float(ari_metric.compute(pred, truth))
    diagnostics["hierarchy_scored_level_count"] = float(scored_levels)
    return diagnostics


def compute_oracle_readout_ceiling(
    model: Any,
    env: Any,
    readout_config: EvaluationReadoutConfig,
) -> dict[str, float]:
    """Diagnostic-only readout ceiling from ground-truth cluster counts.

    Cuts the learned geometry with agglomerative/ward at the true cluster
    count per level. High oracle ARI with low formal ARI means the geometry
    contains the structure and the unsupervised readout is the bottleneck;
    low oracle ARI means the dynamics themselves failed. This leaks the true
    cluster count, so it must never be used as the formal score.
    """
    is_hier = bool(getattr(env, "is_hierarchical", False))
    try:
        oracle_config = EvaluationReadoutConfig(
            method="oracle",
            eps=readout_config.eps,
            min_samples=readout_config.min_samples,
            max_clusters=readout_config.max_clusters,
            standardize=readout_config.standardize,
        )
        outputs = unified_readout_outputs(model, env, oracle_config)
    except Exception:
        return {}

    depth = _hierarchy_depth(env) if is_hier else 1
    ari_metric = _METRIC_REGISTRY["ari"]
    diagnostics: dict[str, float] = {}
    for level_idx in range(depth):
        key = cluster_level_key_from_index(level_idx)
        if key not in outputs:
            continue
        pred = np.asarray(outputs[key], dtype=int).reshape(-1)
        truth = _get_env_true_labels(env, level_idx if is_hier else None).reshape(-1)
        if pred.shape != truth.shape or pred.size == 0:
            continue
        diagnostics[f"oracle_{key}_ari_score"] = float(ari_metric.compute(pred, truth))
    return diagnostics


def compute_cophenetic_score(model: Any, env: Any) -> float:
    """Threshold-free tree-structure score of the final geometry.

    Spearman correlation between the ground-truth merge-depth matrix and the
    single-linkage cophenetic distances of the embedding. Independent of any
    readout cut, so it separates dynamics quality from readout quality. Not
    chance-corrected; diagnostic alongside ARI, never the formal score.
    Note: pair mass is dominated by the coarsest split, so missing coarse
    structure costs far more than missing fine structure.
    """
    try:
        if bool(getattr(env, "is_hierarchical", False)):
            depth = _hierarchy_depth(env)
            true_levels = np.stack([_get_env_true_labels(env, i) for i in range(depth)])
        else:
            true_levels = np.asarray(_get_env_true_labels(env, None), dtype=int).reshape(1, -1)
        points = _embedding_points(model)
        return float(metric_cophenetic_score(true_levels, points))
    except Exception:
        return float("nan")


def apply_hierarchy_total_scores(
    scores: dict[str, float],
    *,
    is_hierarchical: bool,
    depth: int = 1,
) -> dict[str, float]:
    """Make ``ari_score`` the equal-weight all-level total for hierarchy data.

    For flat datasets ``ari_score`` stays the single target partition score.
    For hierarchy datasets ``ari_score`` becomes the arithmetic mean of the
    required per-level ``cluster_level{N}_ari_score`` values. Any level the
    readout failed to produce counts as 0.0, so the total cannot improve by
    dropping difficult levels. The per-level columns remain the only other ARI
    columns; worst-level quality can be derived with ``min`` without storing a
    duplicate score family. ``nmi_score`` remains a fixed-level diagnostic.
    """
    if not bool(is_hierarchical):
        return scores

    updated = dict(scores)
    required_depth = max(1, int(depth))
    level_values: list[float] = []
    for level_index in range(1, required_depth + 1):
        value = updated.get(f"cluster_level{level_index}_ari_score", float("nan"))
        level_values.append(float(value) if np.isfinite(float(value)) else 0.0)
    arr = np.asarray(level_values, dtype=float)
    updated["ari_score"] = float(np.mean(arr))
    return updated


def _get_env_true_labels(env: Any, level: int | None = None) -> np.ndarray:
    true_label_fn = getattr(env, 'trueLabel')
    if level is None:
        return np.asarray(true_label_fn(), dtype=int)
    try:
        return np.asarray(true_label_fn(level=level), dtype=int)
    except TypeError:
        return np.asarray(true_label_fn(), dtype=int)


def _hierarchy_depth(env: Any) -> int:
    try:
        return max(1, int(getattr(env, 'hierarchy_depth', 1)))
    except Exception:
        return 1


def _infer_label_level(model: Any, env: Any, explicit_level: int | None = None) -> int:
    depth = _hierarchy_depth(env)
    if not bool(getattr(env, 'is_hierarchical', False)):
        return 0
    if explicit_level is not None:
        return int(np.clip(int(explicit_level), 0, depth - 1))
    return 0


def _organize_for_evaluation(model: Any) -> np.ndarray:
    with _rng_neutral():
        return np.asarray(model.organize(), dtype=int)


def _organize_for_fixed_hierarchy_level(model: Any, level_index: int) -> np.ndarray:
    level_key = cluster_level_key_from_index(int(level_index))
    with _rng_neutral():
        return _call_organize(model, level=level_key)


def _supports_keyword_argument(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


@contextmanager
def _rng_neutral() -> Any:
    """Run a block without advancing the global np.random stream.

    Evaluation/readout must not perturb training randomness; otherwise how
    often a run is scored would change what it learns.
    """
    saved_state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(saved_state)


@contextmanager
def _isolated_model_rng(model: Any) -> Any:
    """Swap global np.random to the model's private stream for the duration.

    Engines that consume global RNG during input() would otherwise change the
    future environment sequence, making interleaved (live) and pre-generated
    (formal) training diverge. Models without a reserved seed (created outside
    create_model) run unisolated for backward compatibility.
    """
    seed = getattr(model, "_isolated_model_rng_seed", None)
    if seed is None:
        yield
        return
    saved_state = np.random.get_state()
    model_state = getattr(model, "_isolated_model_rng_state", None)
    if model_state is None:
        np.random.seed(int(seed))
    else:
        np.random.set_state(model_state)
    try:
        yield
    finally:
        model._isolated_model_rng_state = np.random.get_state()
        np.random.set_state(saved_state)


@contextmanager
def _isolated_environment_rng(env: Any) -> Any:
    """Swap global ``np.random`` to an environment's private stream.

    Graph-walk environments currently consume the NumPy module-level RNG.
    Keeping their state on the environment makes a seeded input sequence
    independent of model construction, model updates, and strategy ordering.
    Environments created without an explicit ``environment_seed`` remain
    unisolated for backward compatibility with direct library callers.
    """
    seed = getattr(env, "_isolated_environment_rng_seed", None)
    if seed is None:
        yield
        return
    saved_state = np.random.get_state()
    env_state = getattr(env, "_isolated_environment_rng_state", None)
    if env_state is None:
        np.random.seed(int(seed))
    else:
        np.random.set_state(env_state)
    try:
        yield
    finally:
        env._isolated_environment_rng_state = np.random.get_state()
        np.random.set_state(saved_state)


def _advance_model_chunk(
    model: Any,
    batch: np.ndarray,
    *,
    phase_offset: int,
    reset_trajectory: bool,
    append_final_frame: bool,
) -> None:
    with _isolated_model_rng(model):
        input_general = getattr(model, "inputGeneral", None)
        if callable(input_general):
            kwargs: dict[str, Any] = {}
            if _supports_keyword_argument(input_general, "phase_offset"):
                kwargs["phase_offset"] = int(phase_offset)
            if _supports_keyword_argument(input_general, "reset_trajectory"):
                kwargs["reset_trajectory"] = bool(reset_trajectory)
            if _supports_keyword_argument(input_general, "append_final_frame"):
                kwargs["append_final_frame"] = bool(append_final_frame)
            if kwargs:
                input_general(batch, **kwargs)
                return
        model.input(batch)


def _cluster_level_candidate_keys(outputs: dict[str, np.ndarray]) -> list[str]:
    keys = {
        str(key).strip().lower()
        for key in outputs.keys()
        if re.fullmatch(r"cluster_level\d+", str(key).strip().lower())
    }
    return sorted(keys, key=lambda name: int(parse_hierarchy_label_index(name) or 0))


def _select_evaluation_prediction(
    model: Any,
    env: Any,
    metric: EvaluationMetric,
    *,
    readout_config: EvaluationReadoutConfig,
    label_level: int | None = None,
) -> dict[str, Any]:
    eval_level = _infer_label_level(model, env, label_level)
    true_labels = _get_env_true_labels(env, eval_level if bool(getattr(env, "is_hierarchical", False)) else None)
    if str(getattr(model, "formal_prediction_kind", "")).strip().lower() == "sequence_partition":
        with _rng_neutral():
            if bool(getattr(env, "is_hierarchical", False)):
                hierarchy_organizer = getattr(model, "organize_hierarchy", None)
                if callable(hierarchy_organizer):
                    outputs = canonicalize_hierarchy_outputs(
                        hierarchy_organizer()
                    )
                else:
                    base = _relabel_cluster_ids(_call_organize(model))
                    outputs = canonicalize_hierarchy_outputs(
                        {
                            "cluster_level1": base,
                            "level0": base,
                        }
                    )
            else:
                base = _relabel_cluster_ids(_call_organize(model))
                outputs = canonicalize_hierarchy_outputs(
                    {
                        "cluster_level1": base,
                        "level0": base,
                    }
                )
        output_key = (
            cluster_level_key_from_index(eval_level)
            if bool(getattr(env, "is_hierarchical", False))
            else "cluster_level1"
        )
        labels = np.asarray(
            outputs.get(
                output_key,
                outputs.get(
                    "cluster_level1",
                    np.full(true_labels.shape[0], -1, dtype=int),
                ),
            ),
            dtype=int,
        )
        result = {
            "labels": labels,
            "true_labels": true_labels,
            "score": float(metric.compute(labels, true_labels)),
            "output_key": output_key,
            "mode": (
                "sequence_partition_hierarchy_fixed_level"
                if bool(getattr(env, "is_hierarchical", False))
                else "sequence_partition_flat"
            ),
            "readout_method": str(
                getattr(model, "formal_readout_method", "sequence_partition")
            ),
            "outputs": outputs,
        }
        if bool(getattr(env, "is_hierarchical", False)):
            result["truth_level"] = eval_level
            result["truth_output_key"] = output_key
        return result
    outputs = unified_readout_outputs(model, env, readout_config)
    resolved_method = _resolve_external_readout_method(readout_config, env)
    output_key = cluster_level_key_from_index(eval_level) if bool(getattr(env, "is_hierarchical", False)) else "cluster_level1"
    labels = np.asarray(outputs.get(output_key, outputs.get("cluster_level1", np.full(true_labels.shape[0], -1, dtype=int))), dtype=int)
    score = float(metric.compute(labels, true_labels))
    result = {
        "labels": labels,
        "true_labels": true_labels,
        "score": score,
        "output_key": output_key,
        "mode": "unified_geometry_hierarchy_fixed_level" if bool(getattr(env, "is_hierarchical", False)) else "unified_geometry_flat",
        "readout_method": str(resolved_method),
        # Full readout label dict so callers can score per-level diagnostics
        # from the same readout pass as the fixed-level score.
        "outputs": outputs,
    }
    if bool(getattr(env, "is_hierarchical", False)):
        result["truth_level"] = eval_level
        result["truth_output_key"] = output_key
    return result


def summarize_predicted_labels(labels: Any) -> dict[str, float]:
    """Return simple diagnostics that are useful when comparing clustering quality."""
    arr = np.asarray(labels, dtype=int)
    cluster_labels = arr[arr >= 0]
    cluster_count = len(np.unique(cluster_labels)) if cluster_labels.size else 0
    noise_ratio = float(np.mean(arr < 0)) if arr.size else 0.0
    return {
        'cluster_count': float(cluster_count),
        'noise_ratio': noise_ratio,
    }


# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy Comparison Configuration (defaults)
# COMPARISON_SEQUENCE_LENGTHS = [5000, 10000, 50000]
DEFAULT_COMPARISON_SEQUENCE_LENGTHS: List[int] = [5000, 6000]
COMPARISON_SEQUENCE_LENGTHS = DEFAULT_COMPARISON_SEQUENCE_LENGTHS  # Backward compatibility
COMPARISON_TRIALS = 5

SAMPLE_INTERVAL = 1000  # Sample evaluation metric every N iterations for learning curves

# Common Configuration
TIME_DELAY = 10
MAP_DIMENSIONS = 3
RESULTS_DIR = result_dir()
RESULTS_CSV_DIR = result_dir("csv")
RESULTS_HTML_DIR = result_dir("html")
RESULTS_MD_DIR = result_dir("md")

# Temporal memory configuration for UpMRKV environment
MEMORY_STEPS = 3  # number of recent states to retain (including current)
MEMORY_DECAY_RATE = 0.1  # exponential decay applied per delay window
MEMORY_NOISE_INTENSITY = 0.0  # additive Gaussian noise level

TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
LEGACY_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M"
COMPARISON_CSV_PREFIX = "comparison_full_results"
COMPARISON_PARTIAL_RESULTS_PREFIX = "comparison_partial_results"
COMPARISON_SUMMARY_PREFIX = "comparison_summary_stats"
COMPARISON_REPORT_PREFIX = "comparison_report"

COMPARISON_CSV_FILENAME = f"{COMPARISON_CSV_PREFIX}.csv"
COMPARISON_PARTIAL_RESULTS_FILENAME = f"{COMPARISON_PARTIAL_RESULTS_PREFIX}.csv"
COMPARISON_SUMMARY_FILENAME = f"{COMPARISON_SUMMARY_PREFIX}.csv"
COMPARISON_REPORT_FILENAME = f"{COMPARISON_REPORT_PREFIX}.md"

RUN_SIGNATURE_COLUMN = "run_signature"
ENVIRONMENT_RNG_CONTRACT = "isolated_environment_v1"
HIERARCHY_SCORE_CONTRACT = "required_level_mean_ari_v1"

# Default strategies for batch comparison
DEFAULT_STRATEGIES: StrategyConfig = {'nsgd': {}, 'rg': {}, 'tsf': {}}

DEFAULT_HIERARCHY_TEST_DATASETS: List[str] = [
    'datasets/hierarchy/DCH1_fixed_chunk_6_3.dot',
    'datasets/hierarchy/RANDOM_4LEVEL_TREE_root_to_leaf.dot',
    'datasets/hierarchy/SMALL_5LEVEL_TREE.dot',
]


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _current_timestamp() -> str:
    """Return a consistent timestamp string for filenames."""
    return timestamp_now()


def _extract_timestamp_from_name(name: str) -> Optional[str]:
    """Extract timestamp from a filename, supporting current and legacy formats."""
    current_match = re.search(r"\d{14}", name)
    if current_match:
        return current_match.group(0)
    legacy_match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}", name)
    if legacy_match:
        return legacy_match.group(0)
    return None


def _parse_timestamp_seconds(text: str) -> Optional[float]:
    for fmt in (TIMESTAMP_FORMAT, LEGACY_TIMESTAMP_FORMAT):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _extract_run_timestamp_from_path(path: str | Path) -> Optional[str]:
    candidate = Path(path)
    ts = _extract_timestamp_from_name(candidate.name)
    if ts:
        return ts
    parent_name = candidate.parent.name
    if re.fullmatch(r"\d{14}", parent_name):
        return parent_name
    return None


def _timestamp_sort_key(path: Path) -> tuple[float, float]:
    """
    Create a sort key that prefers parsed timestamps and falls back to mtime.
    Returns (timestamp_seconds, mtime) for descending sorts.
    """
    ts = _extract_run_timestamp_from_path(path)
    if ts:
        parsed = _parse_timestamp_seconds(ts)
        if parsed is not None:
            return (parsed, path.stat().st_mtime)
    return (0.0, path.stat().st_mtime)


def _archive_filename(prefix: str, extension: str) -> str:
    normalized_prefix = str(prefix).strip().replace(" ", "_")
    return f"{normalized_prefix}.{extension.lstrip('.').lower()}"


def _iter_archived_result_candidates(prefix: str, extension: str) -> list[Path]:
    archive_root = result_dir("experiment")
    ext = extension.lstrip(".").lower()
    return list(archive_root.glob(f"*/{prefix}*.{ext}"))


def _legacy_result_dir_for_extension(extension: str) -> Path:
    ext = extension.lstrip(".").lower()
    if ext == "csv":
        return RESULTS_CSV_DIR
    if ext == "html":
        return RESULTS_HTML_DIR
    if ext == "md":
        return RESULTS_MD_DIR
    return RESULTS_DIR


def find_latest_result_file(prefix: str, extension: str) -> Optional[Path]:
    """Find the newest archived result file, with legacy flat-directory fallback."""
    ext = extension.lstrip(".").lower()
    legacy_dir = _legacy_result_dir_for_extension(ext)
    legacy_dir.mkdir(exist_ok=True)
    candidates = _iter_archived_result_candidates(prefix, ext)
    candidates.extend(legacy_dir.glob(f"{prefix}*.{ext}"))
    if not candidates:
        return None
    candidates.sort(key=_timestamp_sort_key, reverse=True)
    return candidates[0]


def build_timestamped_path(prefix: str, timestamp: Optional[str] = None, extension: str = "csv") -> Path:
    """Build a result path. Unified-experiment artifacts now archive by run folder."""
    ts = timestamp or _current_timestamp()
    ext = extension.lstrip(".").lower()
    if ext in {"csv", "html", "md"}:
        return experiment_archive_path(_archive_filename(prefix, ext), timestamp=ts)
    normalized_prefix = str(prefix).strip().replace(" ", "_")
    return RESULTS_DIR / f"{normalized_prefix}_{ts}.{ext}"


def _json_signature_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_signature_payload(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_signature_payload(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_comparison_run_signature(
    *,
    strategies: StrategyConfig,
    sequence_lengths: List[int],
    datasets: List[str],
    comparison_trials: int,
    metric_name: EvaluationMetricName,
    base_seed: Optional[int] = None,
) -> str:
    payload = {
        "kind": "comparison",
        "environment_rng_contract": ENVIRONMENT_RNG_CONTRACT,
        "hierarchy_score_contract": HIERARCHY_SCORE_CONTRACT,
        "strategies": [
            {"name": str(name), "params": _json_signature_payload(dict(params))}
            for name, params in strategies.items()
        ],
        "sequence_lengths": [int(v) for v in sequence_lengths],
        "datasets": [str(resolve_dataset_path(dataset)) for dataset in datasets],
        "comparison_trials": int(comparison_trials),
        "metric_name": str(metric_name),
        "base_seed": None if base_seed is None else int(base_seed),
    }
    raw = json.dumps(_json_signature_payload(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_comparison_partial_signature(path: Path) -> Optional[str]:
    try:
        sample = pd.read_csv(path, nrows=1)
    except Exception:
        return None
    if RUN_SIGNATURE_COLUMN not in sample.columns or sample.empty:
        return None
    value = sample.iloc[0].get(RUN_SIGNATURE_COLUMN)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def find_matching_comparison_partial_csv(prefix: str, expected_signature: str) -> Optional[Path]:
    candidates = _iter_archived_result_candidates(prefix, "csv")
    candidates.extend(RESULTS_CSV_DIR.glob(f"{prefix}*.csv"))
    if not candidates:
        return None
    candidates.sort(key=_timestamp_sort_key, reverse=True)
    for candidate in candidates:
        signature = _read_comparison_partial_signature(candidate)
        if signature == expected_signature:
            return candidate
    return None


def resolve_run_timestamp(timestamp: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> str:
    """
    Choose a consistent timestamp for a run.

    Preference order:
    1) Explicit argument
    2) DataFrame attrs['run_timestamp'] if present
    3) Current timestamp
    """
    if timestamp:
        return timestamp
    if df is not None:
        ts = df.attrs.get('run_timestamp')
        if isinstance(ts, str) and ts:
            return ts
    return _current_timestamp()


def build_comparison_csv_path(timestamp: Optional[str] = None) -> Path:
    """Build the timestamped comparison CSV path."""
    return build_timestamped_path(COMPARISON_CSV_PREFIX, timestamp, extension="csv")


def extract_timestamp_from_path(path: str | Path) -> Optional[str]:
    """Public helper to extract timestamp from a given filename or path."""
    return _extract_run_timestamp_from_path(path)


def find_latest_comparison_csv(metric: Optional[EvaluationMetric] = None) -> Optional[Path]:
    """Return the latest comparison CSV (timestamped or legacy), optionally filtered by metric."""
    candidates = _iter_archived_result_candidates(COMPARISON_CSV_PREFIX, "csv")
    candidates.extend(RESULTS_CSV_DIR.glob(f"{COMPARISON_CSV_PREFIX}*.csv"))
    if not candidates:
        return None
    candidates.sort(key=_timestamp_sort_key, reverse=True)
    if metric is None:
        return candidates[0]

    for path in candidates:
        try:
            metric_val = str(pd.read_csv(path, nrows=1)['metric_name'].iloc[0]).lower()  # type: ignore
        except Exception:
            continue
        if metric_val == metric.name:
            return path
    return None


def create_strategy_config(**strategies: StrategyParams) -> StrategyConfig:
    """Create a custom strategy configuration.

    Args:
        **strategies: Keyword arguments where keys are strategy names and values are parameter dicts.
                     Use empty dict {} for strategies with no parameters.

    Returns:
        A properly typed strategy configuration dictionary

    Example:
        >>> config = create_strategy_config(
        ...     nsgd={},
        ...     ns={'graph_spring_mode': 'rg_canonical'},
        ...     ci={'env_mode': 'graphwalk_ci'},
        ...     rg={'warmup_steps': 2000}
        ... )
    """
    # Validate strategy names
    valid_strategies = known_strategy_names()

    for name in strategies.keys():
        canonical_name = canonicalize_strategy_name(name)
        if canonical_name in DEPRECATED_STRATEGY_HINTS:
            raise ValueError(DEPRECATED_STRATEGY_HINTS[canonical_name])
        if canonical_name not in valid_strategies:
            raise ValueError(
                f"Unknown strategy: {name}. "
                f"Valid strategies are: {', '.join(valid_strategies)}"
            )

    return strategies  # type: ignore[return-value]


def resolve_strategy_execution_spec(
    strategy_name: str,
    strategy_params: Optional[StrategyParams] = None,
) -> tuple[str, str | None, str, StrategyParams]:
    """Split a comparison/profile entry into display identity and engine.

    Task-2 callers may key a strategy map by a profile name such as ``rg``
    while executing the existing engine ``rg``. Keep that distinction explicit:
    public result fields keep the selected profile, and ``engine_strategy`` records
    which implementation class was reused internally.
    """
    params = dict(strategy_params or {})
    engine_strategy = str(
        params.pop(
            COMPARISON_ENGINE_STRATEGY_KEY,
            params.pop(COMPARISON_RUNTIME_STRATEGY_KEY, strategy_name),
        )
    ).strip().lower()
    profile_raw = params.pop(COMPARISON_PROFILE_NAME_KEY, None)
    display_raw = params.pop(COMPARISON_DISPLAY_NAME_KEY, None)
    profile_name = None if profile_raw in {None, ""} else str(profile_raw).strip().lower()
    display_name = str(display_raw or profile_name or strategy_name).strip()
    if not display_name:
        display_name = engine_strategy
    return engine_strategy, profile_name, display_name, params


def _top_level_dot_datasets() -> list[Path]:
    datasets_path = Path("datasets")
    if not datasets_path.exists():
        return []
    return sorted(
        f for f in datasets_path.iterdir()
        if f.is_file() and f.suffix == ".dot" and 'imbalanced' not in f.name.lower()
    )


def get_all_datasets() -> list[str]:
    """Get flat .dot files from the top-level datasets directory."""
    datasets: List[str] = []
    for f in _top_level_dot_datasets():
        if classify_dataset_family(str(f)) == "flat":
            datasets.append(str(f))
    return sorted(datasets)


def get_hierarchy_datasets() -> list[str]:
    """Get all hierarchy .dot files, including legacy top-level hierarchy files."""
    hierarchy_root = Path("datasets") / "hierarchy"
    if not hierarchy_root.exists():
        hierarchy_files: list[Path] = []
    else:
        hierarchy_files = [f for f in hierarchy_root.rglob("*.dot") if f.is_file()]
    datasets: List[str] = []
    for f in hierarchy_files:
        datasets.append(str(f))
    for f in _top_level_dot_datasets():
        if classify_dataset_family(str(f)) == "hierarchy":
            datasets.append(str(f))
    return sorted(dict.fromkeys(datasets))


def get_all_comparison_datasets() -> list[str]:
    """Get all comparison datasets, including hierarchy graphs."""
    merged = list(get_all_datasets()) + list(get_hierarchy_datasets())
    return sorted(dict.fromkeys(merged))


def get_test_datasets(limit: int = 18) -> list[str]:
    """Return an evenly spaced, deterministic subset of the flat datasets for quick runs."""
    all_datasets = get_all_datasets()
    if len(all_datasets) <= limit:
        return all_datasets
    step = len(all_datasets) / float(limit)
    return [all_datasets[int(i * step)] for i in range(limit)]


def get_test_comparison_datasets() -> list[str]:
    """Get fast comparison subset plus representative hierarchy datasets."""
    merged = list(get_test_datasets()) + [resolve_dataset_path(path) for path in DEFAULT_HIERARCHY_TEST_DATASETS]
    return sorted(dict.fromkeys(merged))


# ============================================================================
# MODEL CREATION & TRAINING
# ============================================================================

def create_model(dataset_path: str, strategy_name: str, strategy_params: dict[str, Any], *, trajectory_interval: int=50, record_trajectory: bool=False, environment_seed: int | None=None) -> Tuple[Any, Any]:
    """
    Unified model factory.
    """

    def _coerce_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(fallback)

    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(fallback)

    def _coerce_bool(value: Any, fallback: bool) -> bool:
        try:
            return bool(value)
        except Exception:
            return bool(fallback)
    strategy_key = canonicalize_strategy_name(strategy_name)
    if strategy_key in DEPRECATED_STRATEGY_HINTS:
        raise ValueError(DEPRECATED_STRATEGY_HINTS[strategy_key])
    if strategy_key not in known_strategy_names():
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid strategies are: {', '.join(known_strategy_names())}")
    params = dict(strategy_params)
    model_strategy_key = strategy_key
    if environment_seed is not None:
        np.random.seed(int(environment_seed))
    env_time_delay = _coerce_int(params.pop('env_time_delay', TIME_DELAY), TIME_DELAY)
    env_mode_default = infer_default_env_mode(dataset_path, model_strategy_key)
    env_mode = canonicalize_env_mode(str(params.pop('env_mode', env_mode_default)))
    env_history_length_default = 2 if model_strategy_key == 'ci' else 3
    env_history_gain_default = 2.0 if model_strategy_key == 'ci' else 1.0
    env_history_length = _coerce_int(params.pop('env_history_length', env_history_length_default), env_history_length_default)
    env_decay_rate = _coerce_float(params.pop('env_decay_rate', 0.0), 0.0)
    env_current_gain = _coerce_float(params.pop('env_current_gain', 1.0), 1.0)
    env_history_gain = _coerce_float(params.pop('env_history_gain', env_history_gain_default), env_history_gain_default)
    env_noise_intensity = _coerce_float(params.pop('env_noise_intensity', 0.0), 0.0)
    env_normalize_observation = _coerce_bool(params.pop('env_normalize_observation', False), False)
    env_skip_current_duplicates = _coerce_bool(params.pop('env_skip_current_duplicates', True), True)
    has_leaf_switch_override = 'env_hierarchy_leaf_switch_prob' in params
    has_decay_override = 'env_hierarchy_level_switch_decay' in params
    env_hierarchy_leaf_switch_prob = _coerce_float(params.pop('env_hierarchy_leaf_switch_prob', 0.45), 0.45)
    env_hierarchy_level_switch_decay = _coerce_float(params.pop('env_hierarchy_level_switch_decay', 0.25), 0.25)
    env_hierarchy_support_threshold = _coerce_float(params.pop('env_hierarchy_support_threshold', 1e-06), 1e-06)
    legacy_same_factor = params.pop('env_hierarchy_same_factor', None)
    legacy_cross_weight = params.pop('env_hierarchy_cross_weight', None)
    if legacy_same_factor is not None and (not has_leaf_switch_override):
        env_hierarchy_leaf_switch_prob = _coerce_float(legacy_same_factor, env_hierarchy_leaf_switch_prob)
    if legacy_cross_weight is not None and (not has_decay_override):
        env_hierarchy_level_switch_decay = min(1.0, max(1e-06, 1.0 / max(_coerce_float(legacy_cross_weight, 1.0), 1e-06)))
    if env_mode in {'graphwalk_ci', 'ci', 'hard'}:
        if GraphWalkTestCI is None:
            raise ValueError('CI environment mode requires test/GraphWalkTestCI.py in this workspace.')
        env = GraphWalkTestCI(env_time_delay, dataset_path, history_length=env_history_length, decay_rate=env_decay_rate, current_gain=env_current_gain, history_gain=env_history_gain, noise_intensity=env_noise_intensity, normalize_observation=env_normalize_observation, skip_current_duplicates=env_skip_current_duplicates)
    elif env_mode in {'hierarchy_sequence_ci', 'hierarchy_ci'}:
        env = HierarchySequenceTestCI(env_time_delay, dataset_path, leaf_switch_prob=env_hierarchy_leaf_switch_prob, level_switch_decay=env_hierarchy_level_switch_decay, support_threshold=env_hierarchy_support_threshold, history_length=env_history_length, decay_rate=env_decay_rate, current_gain=env_current_gain, history_gain=env_history_gain, noise_intensity=env_noise_intensity, normalize_observation=env_normalize_observation, skip_current_duplicates=env_skip_current_duplicates)
    elif env_mode in {'hierarchy_sequence', 'hierarchy', 'graphwalk_hierarchy'}:
        env = HierarchySequenceTest(env_time_delay, dataset_path, leaf_switch_prob=env_hierarchy_leaf_switch_prob, level_switch_decay=env_hierarchy_level_switch_decay, support_threshold=env_hierarchy_support_threshold)
    elif env_mode in {'graphwalk'}:
        env = GraphWalkTest(env_time_delay, dataset_path)
    elif env_mode in {'graphwalk_tokenize', 'tokenize', 'hierarchy_sequence_tokenize', 'hierarchy_tokenize'}:
        if looks_like_hierarchy_dataset(dataset_path):
            env = HierarchySequenceTestTokenize(env_time_delay, dataset_path, leaf_switch_prob=env_hierarchy_leaf_switch_prob, level_switch_decay=env_hierarchy_level_switch_decay, support_threshold=env_hierarchy_support_threshold)
        else:
            env = GraphWalkTestTokenize(env_time_delay, dataset_path)
    else:
        valid_modes = ['graphwalk', 'graphwalk_tokenize', 'hierarchy_sequence_tokenize', 'graphwalk_ci', 'hierarchy_sequence', 'hierarchy_sequence_ci']
        raise ValueError(f"Unknown env_mode: {env_mode}. Valid modes are: {', '.join(valid_modes)}")
    output_size = env.getOutputSize()
    if environment_seed is not None:
        env._isolated_environment_rng_seed = int(environment_seed)
        env._isolated_environment_rng_state = np.random.get_state()
        np.random.seed(int(environment_seed))
    effective_env_params = {'env_mode': env_mode, 'env_time_delay': env_time_delay, 'env_history_length': env_history_length, 'env_decay_rate': env_decay_rate, 'env_current_gain': env_current_gain, 'env_history_gain': env_history_gain, 'env_noise_intensity': env_noise_intensity, 'env_normalize_observation': env_normalize_observation, 'env_skip_current_duplicates': env_skip_current_duplicates, 'env_hierarchy_leaf_switch_prob': env_hierarchy_leaf_switch_prob, 'env_hierarchy_level_switch_decay': env_hierarchy_level_switch_decay, 'env_hierarchy_support_threshold': env_hierarchy_support_threshold}
    params = _normalize_strategy_model_params(model_strategy_key, params, env=env)
    base_rate_calc = float(np.clip(0.001 * output_size, 0.05, 0.1))
    user_specified_rate = params.pop('adaptation_rate', None)
    adaptation_rate = _coerce_float(user_specified_rate, base_rate_calc) if user_specified_rate is not None else base_rate_calc
    input_size_val = _coerce_int(params.pop('input_size', output_size), output_size)
    dimensions_val = _coerce_int(params.pop('dimensions', 3), 3)
    trajectory_interval_val = _coerce_int(params.pop('trajectory_interval', trajectory_interval), trajectory_interval)
    record_trajectory_val = _coerce_bool(params.pop('record_trajectory', record_trajectory), record_trajectory)
    base_kwargs = {'input_size': input_size_val, 'dimensions': dimensions_val, 'adaptation_rate': adaptation_rate, 'trajectory_interval': trajectory_interval_val, 'record_trajectory': record_trajectory_val}
    effective_model_params = dict(base_kwargs)
    effective_model_params.update(params)
    if model_strategy_key == 'nsgd':
        model = SyncMapX_NSGD.create_syncmap_nsgd(**base_kwargs, **params)
        model.method_name = 'SyncMap_NSGD'
    elif model_strategy_key == 'tsf':
        if SyncMapTSF is None:
            raise ValueError("Strategy 'tsf' requires SyncMapTSF.py in this workspace.")
        model = SyncMapTSF.create_syncmap_tsf(**base_kwargs, **params)
        model.method_name = 'SyncMap_TSF'
    elif strategy_key == 'ns':
        model = SyncMapNS.create_syncmap_ns(**base_kwargs, **params)
        model.method_name = 'SyncMap_NS'
    elif model_strategy_key == 'ci':
        if SyncMapX_CI is None:
            raise ValueError("Strategy 'ci' requires SyncMapX_CI.py in this workspace.")
        model = SyncMapX_CI.create_syncmap_ci(**base_kwargs, **params)
    elif model_strategy_key == 'syncmap':
        model = _create_syncmap_model(base_kwargs, params)
    elif model_strategy_key == 'rg':
        model = _create_rg_model(base_kwargs, params)
    elif model_strategy_key == 'symmetric':
        model = _create_symmetric_model(base_kwargs, params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid strategies are: {', '.join(known_strategy_names())}")
    if not hasattr(model, 'update_params'):
        model.update_params = lambda **kwargs: None
    if hasattr(model, 'get_effective_params'):
        try:
            effective_model_params = dict(model.get_effective_params())
        except Exception:
            effective_model_params = dict(effective_model_params)
    model.effective_model_params = effective_model_params
    model.effective_env_params = effective_env_params
    model._isolated_model_rng_seed = int(np.random.randint(0, 2 ** 31 - 1))
    model._isolated_model_rng_state = None
    return (model, env)

def train_model(
    model: Any,
    env: GraphWalkTest,
    iterations: int,
    sample_curves: bool = True,
    metric: EvaluationMetric | None = None,
    readout_config: EvaluationReadoutConfig | None = None,
    label_level: int | None = None,
    sample_interval: int | None = None,
) -> list[dict[str, Any]]:
    """Train a model and optionally sample learning curves (With Stability Fix)."""
    metric = metric or resolve_metric()
    readout_config = readout_config or EvaluationReadoutConfig()
    with _isolated_environment_rng(env):
        input_sequence, _ = env.getSequence(iterations)
    learning_curve: list[dict[str, Any]] = []
    interval = max(1, int(sample_interval if sample_interval is not None else SAMPLE_INTERVAL))
    hierarchy_refresh_enabled = bool(
        getattr(env, 'is_hierarchical', False)
        or getattr(model, 'level1_enabled', False)
    )
    hierarchy_refresh_interval = (
        max(0, int(getattr(model, 'hierarchy_refresh_interval', 0) or 0))
        if hierarchy_refresh_enabled
        else 0
    )
    phase_offset = 0
    # Feed in the algorithm's own step_chunk (default 100), independent of the
    # 1000-step curve-sampling interval, so results don't depend on chunk size.
    feed_chunk = max(1, int(getattr(model, 'step_chunk', 100)))

    def _refresh_hierarchy_state_if_needed(completed_step: int) -> None:
        if hierarchy_refresh_interval <= 0:
            return
        if completed_step <= 0:
            return
        if completed_step % hierarchy_refresh_interval != 0 and completed_step < iterations:
            return
        refresher = getattr(model, 'refresh_hierarchy_state', None)
        if callable(refresher):
            with _isolated_model_rng(model):
                refresher()

    try:
        if sample_curves:
            # Sample metric periodically during training
            for start in range(0, iterations, interval):
                end = min(start + interval, iterations)
                
                # Stability Fix: Catch internal errors during specific batch updates
                try:
                    for fs in range(start, end, feed_chunk):
                        fe = min(fs + feed_chunk, end)
                        _advance_model_chunk(
                            model,
                            input_sequence[fs:fe],
                            phase_offset=fs,
                            reset_trajectory=(fs == 0),
                            append_final_frame=(fe >= iterations),
                        )
                        _refresh_hierarchy_state_if_needed(fe)
                    phase_offset = end
                except ValueError as e:
                    if "setting an array element with a sequence" in str(e):
                        print(f"  [Warn] Training Stability Error (Ragged Array) at iter {start}. Stopping trial early.")
                        break # Stop training this model, return current curve
                    raise e
                except Exception as e:
                    print(f"  [Warn] Training Step Error: {e}")
                    break

                # Evaluate
                eval_result = _select_evaluation_prediction(
                    model,
                    env,
                    metric,
                    readout_config=readout_config,
                    label_level=label_level,
                )
                score = float(eval_result['score'])
                point: dict[str, float] = {
                    'iteration': float(end),
                    metric.curve_key: float(score),
                }
                labels = np.asarray(eval_result['labels'], dtype=int)
                true_labels = np.asarray(eval_result['true_labels'], dtype=int)
                point_scores = compute_all_metric_scores(labels, true_labels)
                point_scores.update(compute_hierarchy_metric_diagnostics(model, env, readout_config, outputs=eval_result.get('outputs')))
                point_scores = apply_hierarchy_total_scores(
                    {name: float(value) for name, value in point_scores.items() if name.endswith('_score')},
                    is_hierarchical=bool(getattr(env, 'is_hierarchical', False)),
                    depth=_hierarchy_depth(env),
                )
                point.update(point_scores)
                if bool(getattr(env, 'is_hierarchical', False)):
                    point['ari_scope'] = 'hierarchy_level_mean'
                    point['nmi_scope'] = 'fixed_level'
                else:
                    point['ari_scope'] = 'single_partition'
                    point['nmi_scope'] = 'single_partition'
                for metric_obj in _METRIC_REGISTRY.values():
                    score_key = metric_obj.score_key
                    if score_key in point:
                        point[metric_obj.curve_key] = float(point[score_key])
                learning_curve.append(point)
        else:
            # Just train and get final result
            try:
                if hierarchy_refresh_interval > 0:
                    for start in range(0, iterations, hierarchy_refresh_interval):
                        end = min(start + hierarchy_refresh_interval, iterations)
                        _advance_model_chunk(
                            model,
                            input_sequence[start:end],
                            phase_offset=phase_offset,
                            reset_trajectory=(start == 0),
                            append_final_frame=(end >= iterations),
                        )
                        phase_offset = end
                        _refresh_hierarchy_state_if_needed(end)
                else:
                    _advance_model_chunk(
                        model,
                        input_sequence,
                        phase_offset=0,
                        reset_trajectory=True,
                        append_final_frame=True,
                    )
                eval_result = _select_evaluation_prediction(
                    model,
                    env,
                    metric,
                    readout_config=readout_config,
                    label_level=label_level,
                )
                score = float(eval_result['score'])
                point: dict[str, float] = {
                    'iteration': float(iterations),
                    metric.curve_key: float(score),
                }
                labels = np.asarray(eval_result['labels'], dtype=int)
                true_labels = np.asarray(eval_result['true_labels'], dtype=int)
                point_scores = compute_all_metric_scores(labels, true_labels)
                point_scores.update(compute_hierarchy_metric_diagnostics(model, env, readout_config, outputs=eval_result.get('outputs')))
                point_scores = apply_hierarchy_total_scores(
                    {name: float(value) for name, value in point_scores.items() if name.endswith('_score')},
                    is_hierarchical=bool(getattr(env, 'is_hierarchical', False)),
                    depth=_hierarchy_depth(env),
                )
                point.update(point_scores)
                if bool(getattr(env, 'is_hierarchical', False)):
                    point['ari_scope'] = 'hierarchy_level_mean'
                    point['nmi_scope'] = 'fixed_level'
                else:
                    point['ari_scope'] = 'single_partition'
                    point['nmi_scope'] = 'single_partition'
                for metric_obj in _METRIC_REGISTRY.values():
                    score_key = metric_obj.score_key
                    if score_key in point:
                        point[metric_obj.curve_key] = float(point[score_key])
                learning_curve.append(point)
            except ValueError as e:
                 if "setting an array element with a sequence" in str(e):
                     print(f"  [Warn] Training Stability Error (Ragged Array). returning {metric.display_name}=0.")
                     # Append a failure record so the experiment doesn't crash
                     learning_curve.append({'iteration': float(iterations), metric.curve_key: 0.0, 'nmi': 0.0, 'ari': 0.0})
                 else:
                     raise e

    except KeyboardInterrupt:
        raise # Allow user to Ctrl+C
    except Exception as e:
        print(f"  [Error] Critical Training Failure: {e}")
        # Return a zero-score curve to allow experiment loop to continue
        if not learning_curve:
            learning_curve.append({'iteration': float(iterations), metric.curve_key: 0.0, 'nmi': 0.0, 'ari': 0.0})

    # Ensure we return at least one result to prevent IndexError in caller
    if not learning_curve:
        learning_curve.append({'iteration': 0.0, metric.curve_key: 0.0, 'nmi': 0.0, 'ari': 0.0})

    return learning_curve

# ============================================================================
# STRATEGY COMPARISON EXPERIMENT
# ============================================================================

def run_single_comparison_test(
    dataset_path: str, 
    sequence_length: int, 
    strategy_name: str,
    strategy_params: dict[str, Any], 
    trial_num: int,
    custom_label: Optional[str] = None,
    profile_name: Optional[str] = None,
    comparison_name: Optional[str] = None,
    metric: EvaluationMetric | None = None,
    base_seed: Optional[int] = None,
    include_model: bool = False,
    record_trajectory: bool = False,
    sample_curves: bool = False,
    sample_interval: int | None = None,
) -> Optional[dict[str, Any]]:
    """Run a single comparison test.

    Seeding: with ``base_seed`` given, trial i uses ``base_seed + i - 1`` so
    every strategy sees the exact same sequence for that trial and the run is
    reproducible; without it, each trial draws fresh OS entropy (fully random).
    """
    trial_seed: int | None
    if base_seed is not None:
        trial_seed = int(base_seed) + int(trial_num) - 1
        np.random.seed(trial_seed)
    else:
        trial_seed = None
        np.random.seed()

    try:
        metric_obj = metric or resolve_metric()
        
        # 1. Run the experiment (create_model picks the engine from strategy_name).
        result = run_single_test(
            dataset_name=dataset_path,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            sequence_length=sequence_length,
            print_results=False,
            include_model=include_model,
            metric_name=metric_obj.name,
            record_trajectory=record_trajectory,
            sample_curves=sample_curves,
            sample_interval=sample_interval,
            seed=trial_seed,
        )

        base_strategy = result.pop('strategy')
        result.pop('runtime_strategy', None)

        # Exactly two identity columns: 'engine_strategy' (which engine ran,
        # set canonically by run_single_test) and 'profile_name' (which
        # maintained preset/label it ran under). Display casing is a consumer
        # concern, not a stored column.
        result['profile_name'] = str(
            profile_name or comparison_name or custom_label or base_strategy
        ).strip().lower()
        result['trial'] = trial_num

        return result
        
    except Exception as e:
        print(f"Error with {dataset_path} [{strategy_name}]: {e}")
        return None

def find_latest_partial_csv(prefix: str) -> Optional[Path]:
    """Find the most recent partial result file to resume from."""
    candidates = _iter_archived_result_candidates(prefix, "csv")
    candidates.extend(RESULTS_CSV_DIR.glob(f"{prefix}*.csv"))
    if not candidates:
        return None
    candidates.sort(key=_timestamp_sort_key, reverse=True)
    return candidates[0]


def _save_learning_curve_json(curve: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(curve, fh, ensure_ascii=False, indent=2)
    return output_path


def _save_learning_curve_plot(
    curve: list[dict[str, Any]],
    output_path: Path,
    *,
    metric: str = "nmi",
    theme: str = "dark",
    accent_color: str = "#6970ac",
    title: str | None = None,
) -> Path:
    from matplotlib import pyplot as plt
    from tools.plot_theme import apply_basic_plot_theme

    metric_key = str(metric).strip().lower()
    if metric_key not in {"nmi", "ari"}:
        raise ValueError(f"Unsupported learning-curve metric: {metric}")

    steps = np.asarray([int(float(row.get("iteration", 0))) for row in curve], dtype=int)
    values = np.asarray([float(row.get(metric_key, 0.0)) for row in curve], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    palette = apply_basic_plot_theme(fig, ax, theme=theme, accent=accent_color, grid_alpha=0.22)
    ax.plot(
        steps,
        values,
        marker="o",
        linewidth=1.8,
        markersize=3.8,
        color=str(palette["series_l1"]),
        label=metric_key.upper(),
    )
    ax.set_title(title or f"Learning Curve ({metric_key.upper()})")
    ax.set_xlabel("step")
    ax.set_ylabel(metric_key.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0.0)
    legend = ax.legend(frameon=False)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(palette["panel_bg"])
        frame.set_edgecolor(palette["inactive"])
        frame.set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(palette["fg"])
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_comparison_artifacts(
    result: dict[str, Any],
    *,
    dataset: str,
    strategy_name: str,
    sequence_length: int,
    trial_num: int,
    run_timestamp: str,
    save_gif: bool,
    save_trace_json: bool,
    save_trace_plot: bool,
    trace_plot_metric: str,
    theme: str = "dark",
    accent_color: str = "#6970ac",
) -> None:
    qualifier = f"trial{int(trial_num):02d}"

    if save_gif:
        from tools.visual_tools import visualize_trajectory_animation

        model = result.get("syncmap")
        env = result.get("graphwalk")
        if model is not None and env is not None:
            gif_path = experiment_artifact_path(
                strategy_name,
                dataset,
                "animation",
                "gif",
                step_count=int(sequence_length),
                timestamp=run_timestamp,
                qualifier=qualifier,
            )
            ok = visualize_trajectory_animation(
                model,
                graphwalk=env,
                dataset=dataset,
                strategy=strategy_name,
                save_path=str(gif_path),
                total_steps=int(sequence_length),
                show=False,
                theme=theme,
                accent_color=accent_color,
            )
            if ok:
                print(f"saved_gif: {gif_path}")

    curve = list(result.get("learning_curve", []))
    if not curve:
        return

    json_path = experiment_artifact_path(
        strategy_name,
        dataset,
        "score",
        "json",
        step_count=int(sequence_length),
        timestamp=run_timestamp,
        qualifier=qualifier,
    )
    plot_path = experiment_artifact_path(
        strategy_name,
        dataset,
        "curve",
        "png",
        step_count=int(sequence_length),
        timestamp=run_timestamp,
        qualifier=qualifier,
    )
    if save_trace_json:
        saved_json = _save_learning_curve_json(curve, json_path)
        print(f"saved_json: {saved_json}")
    if save_trace_plot:
        saved_plot = _save_learning_curve_plot(
            curve,
            plot_path,
            metric=trace_plot_metric,
            theme=theme,
            accent_color=accent_color,
            title=f"{str(strategy_name).upper()} {Path(dataset).name} {trace_plot_metric.upper()} curve",
        )
        print(f"saved_plot: {saved_plot}")

def _estimate_comparison_eta_seconds(
    strategies: StrategyConfig,
    datasets: List[str],
    sequence_lengths: List[int],
    trial_count: int,
    *,
    calibration_steps: int = 500,
) -> float:
    """Estimate total runtime by timing a short real run per (strategy, dataset).

    Calibration is RNG-neutral (global stream saved/restored), so it cannot
    change the experiment's random sequences.
    """
    steps_per_pair = float(sum(int(v) for v in sequence_lengths) * max(1, int(trial_count)))
    total_seconds = 0.0
    saved_state = np.random.get_state()
    try:
        for dataset in datasets:
            for strategy_name, strategy_params in strategies.items():
                try:
                    runtime_strategy, _profile, _display, runtime_params = resolve_strategy_execution_spec(
                        str(strategy_name),
                        strategy_params,
                    )
                    np.random.seed(0)
                    model_params = dict(runtime_params)
                    model_params.pop('label_level', None)
                    _pop_evaluation_readout_config(model_params)
                    model, env = create_model(dataset, runtime_strategy, model_params, record_trajectory=False)
                    seq, _cls = env.getSequence(int(calibration_steps))
                    start = time.perf_counter()
                    _advance_model_chunk(
                        model,
                        seq,
                        phase_offset=0,
                        reset_trajectory=True,
                        append_final_frame=False,
                    )
                    rate = (time.perf_counter() - start) / float(calibration_steps)
                    total_seconds += rate * steps_per_pair
                except Exception:
                    continue
    finally:
        np.random.set_state(saved_state)
    return total_seconds


def run_comparison_experiment(
    use_test_subset: bool = False,
    include_hierarchy: bool = False,
    strategies: Optional[StrategyConfig] = None,
    sequence_lengths: Optional[List[int]] = None,
    datasets: Optional[List[str]] = None,
    comparison_trials: Optional[int] = None,
    metric_name: EvaluationMetricName = DEFAULT_EVAL_METRIC,
    timestamp: Optional[str] = None,
    resume_mode: str = "auto",
    resume_file: Optional[str] = None,
    save_gif: bool = False,
    save_trace_json: bool = False,
    save_trace_plot: bool = False,
    trace_plot_metric: Optional[str] = None,
    record_interval: Optional[int] = None,
    base_seed: Optional[int] = None,
    strategy_labels: Optional[Dict[str, str]] = None,
    comparison_late_window_points: int = 5,
) -> pd.DataFrame:
    """Run strategy comparison experiment.

    ``base_seed``: when given, trial i (1-based) seeds np.random with
    ``base_seed + i - 1``; when None, trials are fully random.

    ``comparison_late_window_points`` defaults to the calibrated value of 5
    sampling points (see ``_handle_result``). Any run that uses a different value
    must record that value in the CSV ``late_window_points`` column.
    """

    if strategies is None:
        strategies = DEFAULT_STRATEGIES
    if sequence_lengths is None:
        sequence_lengths = DEFAULT_COMPARISON_SEQUENCE_LENGTHS
    trial_count = max(1, int(comparison_trials if comparison_trials is not None else COMPARISON_TRIALS))
    metric = resolve_metric(metric_name)

    if datasets is not None:
        datasets = [resolve_dataset_path(str(dataset)) for dataset in datasets]
        print(f"Using explicit dataset list: {len(datasets)} datasets")
    elif use_test_subset:
        datasets = get_test_comparison_datasets() if include_hierarchy else get_test_datasets()
        if include_hierarchy:
            print(f"Using test subset: {len(datasets)} datasets (including hierarchy)")
        else:
            print(f"Using test subset: {len(datasets)} datasets")
    else:
        datasets = get_all_comparison_datasets() if include_hierarchy else get_all_datasets()
        if include_hierarchy:
            print(f"Found {len(datasets)} comparison datasets (including hierarchy)")
        else:
            print(f"Found {len(datasets)} datasets")

    normalized_resume_mode = str(resume_mode).strip().lower()
    if normalized_resume_mode not in {"auto", "off"}:
        raise ValueError(f"Unsupported resume_mode={resume_mode!r}; expected one of: auto, off")

    run_signature = build_comparison_run_signature(
        strategies=strategies,
        sequence_lengths=sequence_lengths,
        datasets=datasets,
        comparison_trials=trial_count,
        metric_name=metric_name,
        base_seed=base_seed,
    )

    matched_partial: Optional[Path] = None
    if normalized_resume_mode == "off":
        print("Comparison resume disabled; starting a fresh run.")
    elif resume_file not in {None, ""}:
        candidate = Path(str(resume_file)).expanduser()
        if not candidate.is_absolute():
            candidate = (RESULTS_DIR / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requested resume_file does not exist: {candidate}")
        candidate_signature = _read_comparison_partial_signature(candidate)
        if candidate_signature is None:
            raise ValueError(
                f"Requested resume_file has no compatible {RUN_SIGNATURE_COLUMN} column: {candidate}"
            )
        if candidate_signature != run_signature:
            raise ValueError(
                "Requested resume_file belongs to a different comparison run signature and cannot be resumed safely: "
                f"{candidate}"
            )
        matched_partial = candidate
        print(f"Using explicit partial results file: {matched_partial.name}")
    else:
        matched_partial = find_matching_comparison_partial_csv(
            COMPARISON_PARTIAL_RESULTS_PREFIX,
            run_signature,
        )
        if matched_partial is not None:
            print(f"Found matching partial results: {matched_partial.name}. Resuming...")
        else:
            print("No matching signed partial results found; starting a fresh run.")

    if matched_partial is not None and matched_partial.exists():
        run_timestamp = _extract_run_timestamp_from_path(matched_partial)
        if not run_timestamp:
            run_timestamp = resolve_run_timestamp(timestamp)
    else:
        run_timestamp = resolve_run_timestamp(timestamp)

    resolved_trace_plot_metric = str(trace_plot_metric).strip().lower() if trace_plot_metric not in {None, "", "auto"} else str(metric.name)
    total_tests = len(datasets) * len(sequence_lengths) * len(strategies) * trial_count
    strategy_names = list(strategies.keys())
    artifact_interval = max(1, int(record_interval if record_interval is not None else min(10000, max(10, max(sequence_lengths) // 10))))
    print(f"Comparing Strategies: {strategy_names}")
    print(f"Total tests to run: {total_tests}")
    print("Calibrating runtime estimate...", flush=True)
    eta_seconds = _estimate_comparison_eta_seconds(strategies, datasets, sequence_lengths, trial_count)
    if eta_seconds > 0:
        print(f"Estimated total runtime: ~{eta_seconds / 3600:.1f}h ({eta_seconds / 60:.0f}min), training-time based; evaluation overhead extra")
    if save_gif or save_trace_json or save_trace_plot:
        print(
            "Integrated artifact export enabled: "
            f"gif={bool(save_gif)} json={bool(save_trace_json)} plot={bool(save_trace_plot)} "
            f"plot_metric={resolved_trace_plot_metric} record_interval={artifact_interval}"
        )
    print()

    checkpoint_file = build_timestamped_path(COMPARISON_PARTIAL_RESULTS_PREFIX, run_timestamp)
    all_results: list[dict[str, Any]] = []
    if checkpoint_file.exists():
        print(f"Loading previous results from {checkpoint_file}")
        loaded = pd.read_csv(checkpoint_file).to_dict('records')
        all_results = cast(list[dict[str, Any]], loaded)

    completed_tests: set[tuple[str, int, str, int]] = set()
    for res in all_results:
        identity = str(
            res.get('profile_name')
            or res.get('comparison_name')
            or res.get('method')
            or res.get('strategy', '')
        ).strip().lower()
        key = (res['dataset'], int(res['sequence_length']), identity, int(res['trial']))
        completed_tests.add(key)

    test_count = len(completed_tests)

    for dataset in datasets:
        dataset_name = Path(dataset).name
        print(f"\n=== Testing: {dataset_name} ===")

        for seq_len in sequence_lengths:
            for strategy_name, strategy_params in strategies.items():
                runtime_strategy, profile_name, display_name, runtime_params = resolve_strategy_execution_spec(
                    str(strategy_name),
                    strategy_params,
                )
                method_source = strategy_labels.get(strategy_name, display_name) if strategy_labels else display_name
                method_label = str(method_source).upper()
                profile_key = str(profile_name or strategy_name).strip().lower()

                for trial in range(1, trial_count + 1):
                    if (dataset_name, seq_len, profile_key, trial) in completed_tests:
                        continue

                    test_count += 1
                    print(f"[{test_count}/{total_tests}] {method_label} (L={seq_len}) Trial {trial}... ", end='', flush=True)
                    result = run_single_comparison_test(
                        dataset,
                        seq_len,
                        runtime_strategy,
                        runtime_params,
                        trial,
                        custom_label=method_label,
                        profile_name=profile_name,
                        comparison_name=str(strategy_name),
                        metric=metric,
                        base_seed=base_seed,
                        include_model=bool(save_gif),
                        record_trajectory=bool(save_gif),
                        sample_curves=bool(save_trace_json or save_trace_plot),
                        sample_interval=artifact_interval,
                    )
                    if result is not None and (save_gif or save_trace_json or save_trace_plot):
                        _save_comparison_artifacts(
                            result,
                            dataset=dataset,
                            strategy_name=display_name,
                            sequence_length=seq_len,
                            trial_num=trial,
                            run_timestamp=run_timestamp,
                            save_gif=bool(save_gif),
                            save_trace_json=bool(save_trace_json),
                            save_trace_plot=bool(save_trace_plot),
                            trace_plot_metric=resolved_trace_plot_metric,
                        )
                    _handle_result(
                        result,
                        all_results,
                        checkpoint_file,
                        metric,
                        run_signature,
                        late_window_points=int(comparison_late_window_points),
                    )

    final_df = pd.DataFrame(all_results)
    final_df.attrs['run_timestamp'] = run_timestamp
    final_path = build_comparison_csv_path(run_timestamp)
    try:
        checkpoint_file.rename(final_path)
    except OSError:
        final_df.to_csv(final_path, index=False)
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    return final_df

def _stable_tail_window(
    values: list[float],
    *,
    tol: float = 0.01,
    consec: int = 3,
) -> tuple[int, float, float]:
    """Find the smallest trailing window whose mean has converged.

    Starting from w=1, grow the window point by point. Once the running mean
    changes by less than ``tol`` for ``consec`` consecutive expansions, the
    tail mean is stable: further dilation only dilutes with older,
    pre-saturation samples. Returns (window_points, stable_mean, stable_std).
    """
    n = len(values)
    if n <= 0:
        return 0, float("nan"), float("nan")
    if n == 1:
        return 1, float(values[0]), 0.0
    prev: float | None = None
    run = 0
    for w in range(1, n + 1):
        m = float(np.mean(values[-w:]))
        if prev is not None and abs(m - prev) < tol:
            run += 1
        else:
            run = 0
        if run >= consec:
            return w, m, float(np.std(values[-w:]) if w >= 2 else 0.0)
        prev = m
    return n, float(np.mean(values)), float(np.std(values))


def _handle_result(
    result: Optional[dict[str, Any]],
    all_results: list[dict[str, Any]],
    checkpoint_file: Path | str,
    metric: EvaluationMetric,
    run_signature: str,
    late_window_points: int = 5,
) -> None:
    """Helper to process and save result.

    ``late_window_points`` default 5 is the calibrated tail-window length:
    growing the trailing window until the running mean stops changing by
    more than 5% converges at a median of 5 sampling points (median raw span
    ~1000 steps) across representative traces. The value is fixed; the
    calibration script is not part of this repository.
    """
    if result:
        enriched = dict(result)
        enriched[RUN_SIGNATURE_COLUMN] = run_signature
        try:
            curve = list(result.get("learning_curve") or [])
            if curve and late_window_points > 0:
                key = metric.curve_key
                curve_values = [float(point[key]) for point in curve if key in point]
                if len(curve) < late_window_points:
                    # Trajectory is shorter than the requested window: a tail
                    # slice would silently become a full-run average and hide
                    # early-unsaturated samples. Emit NaN instead of a
                    # misleading value so consumers know sampling was too coarse.
                    enriched[f"late_mean_{key}"] = float("nan")
                    enriched[f"late_std_{key}"] = float("nan")
                    enriched["late_window_points"] = int(late_window_points)
                    enriched["late_window_incomplete"] = True
                else:
                    tail = curve_values[-late_window_points:]
                    if tail:
                        enriched[f"late_mean_{key}"] = float(np.mean(tail))
                        enriched[f"late_std_{key}"] = float(np.std(tail))
                        enriched["late_window_points"] = int(late_window_points)
                        enriched["late_window_incomplete"] = False
                # Adaptive stable-tail window (user rule: grow the trailing
                # window until the running mean stops changing materially).
                stable_w, stable_mean, stable_std = _stable_tail_window(curve_values)
                if curve_values:
                    enriched[f"stable_mean_{key}"] = float(stable_mean)
                    enriched[f"stable_std_{key}"] = float(stable_std)
                    enriched["stable_window_points"] = int(stable_w)
                    enriched["stable_window_incomplete"] = bool(stable_w >= len(curve_values))
        except Exception:
            pass
        all_results.append(enriched)
        score_value = result.get(metric.score_key, 0.0)
        print(f"{metric.display_name}={score_value:.4f} ({result['elapsed_time']:.1f}s)")
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
    else:
        print("FAILED")


def comparison_summary_evaluation_aggregations(
    df: pd.DataFrame,
    group_keys: list[str],
) -> dict[str, list[str]]:
    """Return stable evaluation-contract fields or reject mixed cases."""

    aggregations: dict[str, list[str]] = {}
    for column in ('evaluation_mode', 'eval_readout_method', 'evaluation_output_key'):
        if column not in df.columns:
            continue
        distinct = df.groupby(group_keys, dropna=False)[column].nunique(dropna=False)
        if bool((distinct > 1).any()):
            offenders = distinct[distinct > 1].index.tolist()[:5]
            raise ValueError(
                f"Cannot summarize mixed {column} values within comparison cases: {offenders}"
            )
        aggregations[column] = ['first']
    return aggregations


def save_comparison_results(df: pd.DataFrame, timestamp: Optional[str]=None) -> pd.DataFrame:
    """Save and analyze comparison results"""
    RESULTS_CSV_DIR.mkdir(exist_ok=True)
    metric = detect_metric_from_frame(df)
    run_timestamp = resolve_run_timestamp(timestamp, df)
    csv_path = build_comparison_csv_path(run_timestamp)
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')
    agg_spec: dict[str, list[str]] = {'elapsed_time': ['mean'], 'output_size': ['first']}
    summary_group_keys = ['dataset', 'profile_name', 'sequence_length']
    agg_spec.update(comparison_summary_evaluation_aggregations(df, summary_group_keys))
    for metric_obj in _METRIC_REGISTRY.values():
        if metric_obj.score_key in df.columns:
            agg_spec[metric_obj.score_key] = ['mean', 'std', 'min', 'max']
    for diagnostic_score in ('nmi_classified_only_score', 'ari_classified_only_score', 'nmi_masked_noise_score', 'ari_masked_noise_score', 'nmi_noise_singletons_score', 'ari_noise_singletons_score'):
        if diagnostic_score in df.columns:
            agg_spec[diagnostic_score] = ['mean']
    if 'cluster_count' in df.columns:
        agg_spec['cluster_count'] = ['mean']
    if 'noise_ratio' in df.columns:
        agg_spec['noise_ratio'] = ['mean']
    if 'classified_covered_ratio' in df.columns:
        agg_spec['classified_covered_ratio'] = ['mean']
    if 'singleton_cluster_ratio' in df.columns:
        agg_spec['singleton_cluster_ratio'] = ['mean']
    if 'noise_covered_ratio' in df.columns:
        agg_spec['noise_covered_ratio'] = ['mean']
    for column in df.columns:
        column_name = str(column)
        if column_name.startswith('hierarchy_') and (column_name.endswith('_score') or column_name == 'hierarchy_scored_level_count'):
            agg_spec.setdefault(column_name, ['mean'])
        elif re.fullmatch('(?:oracle_)?cluster_level\\d+_ari_score', column_name):
            agg_spec.setdefault(column_name, ['mean', 'std'])
        elif column_name == 'cophenetic_score':
            agg_spec.setdefault(column_name, ['mean', 'std'])
    df = ensure_profile_name_column(df)
    summary = df.groupby(summary_group_keys).agg(agg_spec).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary_path = build_timestamped_path(COMPARISON_SUMMARY_PREFIX, run_timestamp)
    summary.to_csv(summary_path, index=False)
    print(f'Saved: {summary_path}')
    report_path = build_timestamped_path(COMPARISON_REPORT_PREFIX, run_timestamp, extension='md')
    generate_comparison_report(df, summary, metric, output_path=report_path)
    return summary


def generate_comparison_report(df: pd.DataFrame, summary: pd.DataFrame, metric: EvaluationMetric, output_path: Optional[Path]=None) -> None:
    """Generate markdown comparison report (Fix: Handle multiple sequence lengths)"""
    df = ensure_profile_name_column(df)
    report: list[str] = []
    report.append('# Strategy Comparison Report\n\n')
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    seq_lengths = sorted(summary['sequence_length'].unique())
    report.append(f'Sequence lengths: {seq_lengths}\n\n')
    report.append(f'## Overall Performance (Mean {metric.display_name})\n\n')
    for sl in seq_lengths:
        report.append(f'### Sequence Length: {sl}\n\n')
        subset = summary[summary['sequence_length'] == sl]
        try:
            pivot = subset.pivot(index='dataset', columns='profile_name', values=f'{metric.score_key}_mean')
            cols = list(pivot.columns)
            pivot = pivot[cols]
            report.append(pivot.to_markdown())
        except Exception as e:
            report.append(f'*(Table generation failed for length {sl}: {e})*')
        report.append('\n\n')
    report.append(f'## Performance by Dataset Family (Mean {metric.display_name})\n\n')
    family_df = df.copy()
    if 'dataset_family' not in family_df.columns:
        source_col = 'dataset_path' if 'dataset_path' in family_df.columns else 'dataset'
        family_df['dataset_family'] = family_df[source_col].astype(str).apply(classify_dataset_family)
    for sl in seq_lengths:
        report.append(f'### Sequence Length: {sl}\n\n')
        subset = family_df[family_df['sequence_length'] == sl]
        try:
            family_summary = subset.groupby(['dataset_family', 'profile_name'])[metric.score_key].mean().reset_index()
            pivot = family_summary.pivot(index='dataset_family', columns='profile_name', values=metric.score_key)
            ordered_rows = [name for name in ['flat', 'hierarchy'] if name in pivot.index]
            if ordered_rows:
                pivot = pivot.reindex(ordered_rows)
            cols = list(pivot.columns)
            pivot = pivot[cols]
            report.append(pivot.to_markdown())
        except Exception as e:
            report.append(f'*(Dataset family table generation failed for length {sl}: {e})*')
        report.append('\n\n')
    supplemental_metrics = [metric_obj for metric_obj in _METRIC_REGISTRY.values() if metric_obj.name != metric.name and f'{metric_obj.score_key}_mean' in summary.columns]
    for extra_metric in supplemental_metrics:
        report.append(f'## Supplemental {extra_metric.display_name} Summary\n\n')
        for sl in seq_lengths:
            report.append(f'### Sequence Length: {sl}\n\n')
            subset = summary[summary['sequence_length'] == sl]
            try:
                pivot = subset.pivot(index='dataset', columns='profile_name', values=f'{extra_metric.score_key}_mean')
                cols = list(pivot.columns)
                pivot = pivot[cols]
                report.append(pivot.to_markdown())
            except Exception as e:
                report.append(f'*(Table generation failed for length {sl}: {e})*')
            report.append('\n\n')
    diagnostic_cols = [col for col in ('cluster_count_mean', 'noise_ratio_mean', 'classified_covered_ratio_mean', 'singleton_cluster_ratio_mean', 'noise_covered_ratio_mean', 'nmi_classified_only_score_mean', 'ari_classified_only_score_mean', 'nmi_masked_noise_score_mean', 'ari_masked_noise_score_mean', 'nmi_noise_singletons_score_mean', 'ari_noise_singletons_score_mean') if col in summary.columns]
    if diagnostic_cols:
        report.append('## Clustering Diagnostics\n\n')
        for sl in seq_lengths:
            report.append(f'### Sequence Length: {sl}\n\n')
            subset = summary[summary['sequence_length'] == sl]
            try:
                diag_table = subset[['dataset', 'profile_name', *diagnostic_cols]].copy()
                report.append(diag_table.to_markdown(index=False))
            except Exception as e:
                report.append(f'*(Diagnostics table generation failed for length {sl}: {e})*')
            report.append('\n\n')
    report.append('## Win Rate Analysis (Aggregated)\n')
    methods = df['profile_name'].unique()
    if len(methods) > 1:
        for sl in seq_lengths:
            report.append(f'**Length {sl}:**\n')
            subset = summary[summary['sequence_length'] == sl]
            try:
                pivot = subset.pivot(index='dataset', columns='profile_name', values=f'{metric.score_key}_mean')
                best_methods = pivot.idxmax(axis=1)
                win_counts = best_methods.value_counts()
                report.append(win_counts.to_markdown(header=['Method', 'Wins']))
            except:
                report.append('(Could not calculate win rates)')
            report.append('\n\n')
    report_text = ''.join(report)
    try:
        output_file = output_path or build_timestamped_path(COMPARISON_REPORT_PREFIX, resolve_run_timestamp(None, df), extension='md')
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f'Saved: {output_file}')
    except Exception as e:
        print(f'Error saving report file: {e}')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_single_test(dataset_name: str, strategy_name: str, strategy_params: Optional[StrategyParams]=None, sequence_length: int=5000, print_results: bool=True, trajectory_interval: int=50, record_trajectory: bool=True, include_model: bool=False, metric_name: EvaluationMetricName=DEFAULT_EVAL_METRIC, sample_curves: bool=False, sample_interval: int | None=None, seed: int | None=None) -> Dict[str, Any]:
    """Run SyncMap on a single dataset with a specific strategy."""
    metric = resolve_metric(metric_name)
    dataset_text = str(dataset_name)
    dataset_path_obj = Path(dataset_text)
    if dataset_path_obj.is_absolute():
        dataset_path = dataset_text
    elif not dataset_text.startswith('datasets/'):
        dataset_path = f'datasets/{dataset_text}'
    else:
        dataset_path = dataset_text
    if strategy_params is None:
        strategy_key = canonicalize_strategy_name(strategy_name)
        if strategy_name in DEFAULT_STRATEGIES:
            strategy_params = DEFAULT_STRATEGIES[strategy_name]
        elif strategy_key in DEFAULT_STRATEGIES:
            strategy_params = DEFAULT_STRATEGIES[strategy_key]
        elif strategy_key in {'nsgd', 'ns', 'tsf', 'ci', 'syncmap', 'rg'}:
            strategy_params = {}
        else:
            strategy_params = {}
    raw_strategy_params = dict(strategy_params or {})
    parameter_seed = raw_strategy_params.get('seed')
    run_seed = int(seed) if seed is not None else int(parameter_seed) if parameter_seed is not None else None
    if run_seed is not None:
        np.random.seed(run_seed)
    model_strategy_params = dict(raw_strategy_params)
    model_strategy_params.pop('seed', None)
    label_level_override_raw = model_strategy_params.pop('label_level', None)
    label_level_override = int(label_level_override_raw) if label_level_override_raw is not None else None
    eval_readout_config = _pop_evaluation_readout_config(model_strategy_params)
    try:
        model, env = create_model(dataset_path, strategy_name, model_strategy_params, trajectory_interval=trajectory_interval, record_trajectory=record_trajectory, environment_seed=run_seed)
        actual_rate = getattr(model, 'adaptation_rate', None)
        rate_display = f'{actual_rate:.6f}' if isinstance(actual_rate, (float, int)) else 'N/A'
        start_time = time.time()
        eval_label_level = _infer_label_level(model, env, label_level_override)
        final_labels: Optional[np.ndarray] = None
        true_labels: Optional[np.ndarray] = None
        evaluation_output_key = cluster_level_key_from_index(eval_label_level)
        evaluation_target_key = cluster_level_key_from_index(eval_label_level)
        evaluation_mode = 'hierarchy_fixed_level' if bool(getattr(env, 'is_hierarchical', False)) else 'flat_organize'
        learning_curve = train_model(model, env, sequence_length, sample_curves=bool(sample_curves), metric=metric, readout_config=eval_readout_config, label_level=label_level_override, sample_interval=sample_interval)
        try:
            final_eval = _select_evaluation_prediction(model, env, metric, readout_config=eval_readout_config, label_level=label_level_override)
            final_labels = np.asarray(final_eval['labels'], dtype=int)
            true_labels = np.asarray(final_eval['true_labels'], dtype=int)
            evaluation_output_key = str(final_eval['output_key'])
            evaluation_target_key = str(final_eval.get('truth_output_key', evaluation_target_key))
            evaluation_mode = str(final_eval['mode'])
            eval_label_level = int(final_eval.get('truth_level', eval_label_level))
        except Exception:
            final_labels = None
        elapsed_time = time.time() - start_time
        if final_labels is not None and true_labels is not None:
            all_scores = compute_all_metric_scores(final_labels, true_labels)
            all_scores.update(compute_noise_sensitive_metric_scores(final_labels, true_labels))
            all_scores.update(compute_hierarchy_metric_diagnostics(model, env, eval_readout_config, outputs=final_eval.get('outputs')))
            all_scores.update(compute_oracle_readout_ceiling(model, env, eval_readout_config))
            all_scores['cophenetic_score'] = compute_cophenetic_score(model, env)
            all_scores = apply_hierarchy_total_scores(all_scores, is_hierarchical=bool(getattr(env, 'is_hierarchical', False)), depth=_hierarchy_depth(env))
            label_stats = summarize_predicted_labels(final_labels)
            final_score = float(all_scores[metric.score_key])
            learning_curve[-1][metric.curve_key] = final_score
        else:
            all_scores = {metric.score_key: float(learning_curve[-1][metric.curve_key])}
            label_stats = {'cluster_count': float('nan'), 'noise_ratio': float('nan')}
            final_score = float(learning_curve[-1][metric.curve_key])
        result: dict[str, Any] = {'dataset': Path(dataset_path).name, 'dataset_path': dataset_path, 'dataset_family': classify_dataset_family(dataset_path), 'strategy': strategy_name, 'runtime_strategy': strategy_name, 'engine_strategy': canonicalize_strategy_name(strategy_name), 'sequence_length': sequence_length, 'seed': run_seed, 'trajectory_interval': trajectory_interval, 'output_size': env.getOutputSize(), metric.score_key: final_score, 'elapsed_time': elapsed_time, 'metric_name': metric.name, 'adaptation_rate': actual_rate, 'strategy_params': raw_strategy_params, 'effective_model_params': getattr(model, 'effective_model_params', None), 'effective_env_params': getattr(model, 'effective_env_params', None), 'requested_label_level': label_level_override, 'label_level': eval_label_level, 'evaluation_output_key': evaluation_output_key, 'evaluation_target_key': evaluation_target_key, 'evaluation_mode': 'hierarchy_level_mean' if bool(getattr(env, 'is_hierarchical', False)) and metric.name == 'ari' else f'hierarchy_fixed_level_{metric.name}' if bool(getattr(env, 'is_hierarchical', False)) else evaluation_mode, 'eval_readout_method': str(final_eval.get('readout_method', eval_readout_config.method)) if final_labels is not None else eval_readout_config.method}
        result.update(all_scores)
        result.update(label_stats)
        diagnostics_fn = getattr(model, 'get_diagnostics', None)
        if callable(diagnostics_fn):
            model_diagnostics = dict(diagnostics_fn())
            for key, value in model_diagnostics.items():
                if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
                    result[f'algorithm_{key}'] = value
        result['learning_curve'] = learning_curve
        if final_labels is not None:
            result['predicted_labels'] = np.asarray(final_labels, dtype=int)
        if true_labels is not None:
            result['true_labels'] = np.asarray(true_labels, dtype=int)
        if include_model:
            result['syncmap'] = model
            result['graphwalk'] = env
        if print_results:
            print('=' * 60)
            print(f"Dataset: {result['dataset']}")
            print(f'Strategy: {strategy_name}')
            print(f'Parameters: {strategy_params}')
            print(f'Actual Rate: {rate_display}')
            print(f"Evaluation Output: {evaluation_output_key} -> {evaluation_target_key} ({result['evaluation_mode']})")
            print('-' * 60)
            print(f'{metric.display_name} Score: {final_score:.4f}')
            for metric_obj in _METRIC_REGISTRY.values():
                if metric_obj.name == metric.name:
                    continue
                alt_score = result.get(metric_obj.score_key)
                if alt_score is not None:
                    print(f'{metric_obj.display_name} Score: {float(alt_score):.4f}')
            if 'cluster_count' in result:
                print(f"Cluster Count: {result['cluster_count']:.0f}")
            if 'noise_ratio' in result:
                print(f"Noise Ratio: {result['noise_ratio']:.4f}")
            print(f'Elapsed Time: {elapsed_time:.2f}s')
            print('=' * 60)
        return result
    except Exception as e:
        print(f'Error running test on {dataset_path}: {e}')
        raise


def run_single_test_viewer(
    dataset_name: str,
    strategy_name: str,
    strategy_params: Optional[StrategyParams] = None,
    *,
    sequence_length: int = 5000,
    trajectory_interval: int = 50,
    dimension: int = 3,
    playback: bool = True,
    start_paused: bool = True,
    show: bool = True,
    metric_name: EvaluationMetricName = DEFAULT_EVAL_METRIC,
) -> Any:
    """Run one experiment and open the shared read-only viewer."""
    result = run_single_test(
        dataset_name,
        strategy_name,
        strategy_params=strategy_params,
        sequence_length=sequence_length,
        print_results=True,
        trajectory_interval=trajectory_interval,
        record_trajectory=True,
        include_model=True,
        metric_name=metric_name,
    )
    from live import launch_single_test_viewer

    return launch_single_test_viewer(
        result,
        dimension=dimension,
        playback=playback,
        start_paused=start_paused,
        show=show,
    )

# ============================================================================
# PUBLIC API
# ============================================================================

# Export types for external use
__all__ = [
    # Main experiment functions
    'run_comparison_experiment',
    'save_comparison_results',
    'run_all_experiments',
    # Utility functions
    'run_single_test',
    'run_single_test_viewer',
    'create_strategy_config',
    # Types
    'StrategyName',
    'StrategyParams',
    'StrategyConfig',
    'COMPARISON_ENGINE_STRATEGY_KEY',
    'COMPARISON_RUNTIME_STRATEGY_KEY',
    'COMPARISON_PROFILE_NAME_KEY',
    'COMPARISON_DISPLAY_NAME_KEY',
    # Result file helpers
    'find_latest_comparison_csv',
    'build_comparison_csv_path',
    'build_timestamped_path',
    'extract_timestamp_from_path',
    # Default configurations
    'DEFAULT_STRATEGIES',
    'DEFAULT_COMPARISON_SEQUENCE_LENGTHS',
]


def run_all_experiments(
    use_test_subset: bool = False,
    comparison_include_hierarchy: bool = False,
    strategies: Optional[StrategyConfig] = None,
    sequence_lengths: Optional[List[int]] = None,
    comparison_datasets: Optional[List[str]] = None,
    comparison_trials: Optional[int] = None,
    metric_name: EvaluationMetricName = DEFAULT_EVAL_METRIC,
    resume_mode: str = "auto",
    resume_file: Optional[str] = None,
    comparison_save_gif: bool = False,
    comparison_trace_json: bool = False,
    comparison_trace_plot: bool = False,
    comparison_plot_metric: Optional[str] = None,
    comparison_record_interval: Optional[int] = None,
    comparison_base_seed: Optional[int] = None,
    strategy_labels: Optional[Dict[str, str]] = None,
    comparison_late_window_points: int = 5,
) -> None:
    """Run the batch strategy comparison and save CSV/Markdown summaries."""
    print("=" * 80)
    print("STRATEGY COMPARISON EXPERIMENT")
    if use_test_subset and not comparison_datasets:
        print("(TEST MODE: using a representative dataset subset)")
    print("=" * 80)
    comparison_df = run_comparison_experiment(
        use_test_subset=use_test_subset,
        include_hierarchy=comparison_include_hierarchy,
        strategies=strategies,
        sequence_lengths=sequence_lengths,
        datasets=comparison_datasets,
        comparison_trials=comparison_trials,
        metric_name=metric_name,
        timestamp=_current_timestamp(),
        resume_mode=resume_mode,
        resume_file=resume_file,
        save_gif=bool(comparison_save_gif),
        save_trace_json=bool(comparison_trace_json),
        save_trace_plot=bool(comparison_trace_plot),
        trace_plot_metric=None if comparison_plot_metric in {None, "", "auto"} else str(comparison_plot_metric),
        record_interval=comparison_record_interval,
        base_seed=comparison_base_seed,
        strategy_labels=strategy_labels,
        comparison_late_window_points=int(comparison_late_window_points),
    )
    run_timestamp = resolve_run_timestamp(None, comparison_df)
    save_comparison_results(comparison_df, timestamp=run_timestamp)
    print("\n" + "=" * 80)
    print("BATCH EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {experiment_archive_dir(run_timestamp)}")
