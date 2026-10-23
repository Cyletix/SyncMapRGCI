#!/usr/bin/env python3
"""
SyncMapNS: minimal first-order kernel for recursive RG.

Design constraints:
- first-order position update, no velocity / inertia
- pulse / plasticity remain part of the NS core when enabled
- same spring law at every level; only natural distance scales
- optional geometric guards stay as projections, not as extra forces
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING, Union

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from tools.clustering_strategies import ClusteringStrategy, community_louvain, create_clustering_strategy
from tools.density_scale import DEFAULT_PACKING_FACTOR, compute_density_scale

if TYPE_CHECKING:
    from sklearn.neighbors import NearestNeighbors
else:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        class NearestNeighbors:
            def __init__(self, n_neighbors: int, algorithm: str = "auto") -> None:
                self.n_neighbors = int(n_neighbors)

            def fit(self, X: NDArray[np.float64]) -> "NearestNeighbors":
                self._X = np.asarray(X, dtype=float)
                return self

            def kneighbors(
                self,
                X: Optional[NDArray[np.float64]] = None,
                return_distance: bool = True,
            ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
                samples = self._X if X is None else np.asarray(X, dtype=float)
                diff = samples[:, None, :] - self._X[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                order = np.argsort(dist, axis=1)[:, : self.n_neighbors]
                rows = np.arange(samples.shape[0])[:, None]
                distances = dist[rows, order]
                return distances, order


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _stats_to_mask(
    n: int,
    active_indices: Iterable[int] | NDArray[np.bool_] | NDArray[np.int_],
) -> NDArray[np.bool_]:
    if isinstance(active_indices, np.ndarray) and active_indices.dtype == np.bool_:
        return np.asarray(active_indices, dtype=bool)
    if isinstance(active_indices, np.ndarray):
        arr = np.asarray(active_indices)
        if arr.ndim == 1 and arr.shape[0] == int(n) and np.issubdtype(arr.dtype, np.number):
            return np.asarray(arr, dtype=float) > 1e-12
    mask = np.zeros(int(n), dtype=bool)
    idx = np.asarray(list(active_indices), dtype=int)
    if idx.size > 0:
        mask[idx] = True
    return mask


def _stats_to_values(
    n: int,
    active_indices: Iterable[int] | NDArray[np.bool_] | NDArray[np.int_] | NDArray[np.float64],
) -> NDArray[np.float64]:
    if isinstance(active_indices, np.ndarray):
        arr = np.asarray(active_indices)
        if arr.ndim == 1 and arr.shape[0] == int(n) and arr.dtype == np.bool_:
            return np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == int(n) and np.issubdtype(arr.dtype, np.number):
            return np.clip(np.asarray(arr, dtype=float), 0.0, None)
    values = np.zeros(int(n), dtype=float)
    idx = np.asarray(list(active_indices), dtype=int)
    if idx.size > 0:
        values[idx] = 1.0
    return values


class _NSCoactivationStats:
    """NS-local EMA tracker so NS/RG semantics do not depend on shared module drift."""

    def __init__(self, n: int, memory_steps: int = 100, power_gain: float = 2.0) -> None:
        self.n = int(n)
        self.memory_steps = max(1, int(memory_steps))
        self.alpha = 1.0 / float(self.memory_steps)
        self.power_gain = float(power_gain)
        self._prev: NDArray[np.float64] | None = None
        self.coactivation_matrix: NDArray[np.float64] = np.zeros((self.n, self.n), dtype=float)

    def reset(self) -> None:
        self._prev = None
        self.coactivation_matrix.fill(0.0)

    def clear_window(self) -> None:
        return

    def update(self, active_indices: Iterable[int] | NDArray[np.bool_] | NDArray[np.int_]) -> None:
        curr_values = _stats_to_values(self.n, active_indices)
        curr = curr_values > 1e-12
        self.coactivation_matrix *= (1.0 - self.alpha)
        idx = np.flatnonzero(curr)
        if idx.size > 0:
            pair_weight = np.minimum(curr_values[idx][:, None], curr_values[idx][None, :])
            self.coactivation_matrix[np.ix_(idx, idx)] += self.alpha * pair_weight
        self._prev = curr_values

    def similarity(self) -> NDArray[np.float64]:
        strength = self.coactivation_matrix.copy()
        np.fill_diagonal(strength, 0.0)
        max_val = float(np.max(strength))
        if max_val > 1e-12:
            strength /= max_val
        return np.clip(strength, 0.0, 1.0)


class _NSTransitionTracker:
    """Bounded transition-preference tracker for sparse graph-force export."""

    def __init__(self, n: int, memory_steps: int = 100, power_gain: float = 2.0) -> None:
        self.n = int(n)
        self.memory_steps = max(1, int(memory_steps))
        self.power_gain = float(power_gain)
        self.decay = 1.0 - (1.0 / float(self.memory_steps))
        self._prev: NDArray[np.float64] | None = None
        self.C: NDArray[np.float64] = np.zeros((self.n, self.n), dtype=float)
        self.T = self.C
        self.cnt: NDArray[np.float64] = np.zeros(self.n, dtype=float)

    def reset(self) -> None:
        self._prev = None
        self.C.fill(0.0)
        self.cnt.fill(0.0)

    def clear_window(self) -> None:
        return

    def update(self, active_indices: Iterable[int] | NDArray[np.bool_] | NDArray[np.int_]) -> None:
        curr_values = _stats_to_values(self.n, active_indices)
        curr = curr_values > 1e-12
        if self.decay <= 0.0:
            self.C.fill(0.0)
            self.cnt.fill(0.0)
        else:
            self.C *= self.decay
            self.cnt *= self.decay
        if self._prev is not None and np.any(self._prev > 1e-12) and np.any(curr):
            prev_values = np.asarray(self._prev, dtype=float)
            prev_idx = np.flatnonzero(prev_values > 1e-12)
            curr_idx = np.flatnonzero(curr)
            seen_pairs: set[tuple[int, int]] = set()
            for i in prev_idx.tolist():
                for j in curr_idx.tolist():
                    a = int(i)
                    b = int(j)
                    if a == b:
                        continue
                    row = min(a, b)
                    col = max(a, b)
                    pair = (row, col)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    pair_weight = min(float(prev_values[a]), float(curr_values[b]))
                    if pair_weight <= 1e-12:
                        continue
                    self.C[row, col] += pair_weight
                    self.C[col, row] += pair_weight
        if np.any(curr):
            curr_idx = np.flatnonzero(curr)
            self.cnt[curr_idx] += curr_values[curr_idx]
        self._prev = curr_values

    def similarity(self) -> NDArray[np.float64]:
        denom = np.sqrt(np.outer(self.cnt, self.cnt)) + 1e-12
        sim = np.divide(self.C, denom, out=np.zeros_like(self.C), where=denom > 0.0)
        np.fill_diagonal(sim, 0.0)
        return np.clip(sim, 0.0, 1.0)

    def edge_strength(self) -> NDArray[np.float64]:
        strength = np.power(self.similarity(), max(float(self.power_gain), 1e-12))
        np.fill_diagonal(strength, 0.0)
        return np.clip(strength, 0.0, 2.0)


class SyncMapNS:
    DEFAULT_PARAMS: dict[str, Any] = {
        "base_wall": 100.0,
        "density_ref": 64.0,
        "packing_factor": DEFAULT_PACKING_FACTOR,
        "significant_distance_factor": 0.5,
        "adaptation_rate": 0.05,
        "max_step_norm": 1.0,
        "center_on_origin": True,
        "centering_strength": 0.1,
        "recenter_graph_support_only": False,
        "graph_spring_mode": "rg_canonical",
        "graph_r0_mode": "fixed",
        "graph_r0_fixed": 1.0,
        "graph_r0_rel": 1.0,
        "graph_degree_normalize": True,
        "macro_boundary_repulsion_enabled": False,
        "macro_boundary_scale": 1.0,
        "macro_cross_attr_scale": 0.001,
        "macro_boundary_same_attr_scale": 1.0,
        "macro_boundary_same_attr_size_power": 0.0,
        "macro_boundary_coactivation_gate_enabled": False,
        "macro_boundary_coactivation_scale": 0.05,
        "macro_boundary_coactivation_floor": 0.0,
        "macro_boundary_softness_rel": 0.0,
        "graph_spring_repulse_gain": 2.0,
        "graph_spring_repulse_power": 1.0,
        "graph_spring_repulse_shoulder_gain": 12.0,
        "graph_spring_extend_shoulder_gain": 0.0,
        "graph_spring_extend_shoulder_power": 2.0,
        "graph_spring_extend_shoulder_rel_start": 0.3,
        "graph_spring_soft_rel": 2.0,
        "graph_spring_hard_rel": 5.0,
        "graph_spring_near_slope": 0.2,
        "graph_spring_mid_slope": 0.6,
        "graph_spring_far_gain": 0.8,
        "graph_spring_far_power": 2.0,
        "line_obstacle_repulsion_enabled": False,
        "line_obstacle_radius_rel": 1.0,
        "line_obstacle_strength": 0.5,
        "primitive_contrastive_force_enabled": False,
        "primitive_active_attraction_strength": 0.0,
        "primitive_inactive_repulsion_strength": 0.0,
        "primitive_force_distance_floor_rel": 0.25,
        "primitive_force_max_step_ratio": 1.0,
        # Optional confidence scaling is disabled by default in NS/RG.
        "negative_evidence_confidence_enabled": False,
        "negative_evidence_confidence_level": 0.95,
        "k_attr": 0.03,
        "k_repulse": 1.0,
        "k_pulse": 0.0,
        "pulse_mode": "none",
        "plasticity_enabled": False,
        "plasticity_decay": 0.99,
        "plasticity_min": 0.01,
        "plasticity_recovery": 1.0,
        "plasticity_update_period": 10,
        "physics_batch_size": 1,
        "physics_steps_per_batch": 1,
        "inter_cluster_distance_ratio": 3.16,
        "cluster_distance_strength": 0.0,
        "macro_centroid_distance_ratio": 3.16,
        "k_macro_repulse": 0.0,
        "macro_centroid_max_step_ratio": 0.1,
        "macro_centroid_near_gain": 4.0,
        "macro_centroid_far_gain": 0.5,
        "macro_centroid_cutoff_ratio": 2.0,
        "macro_centroid_use_cluster_mass": True,
        "macro_centroid_mass_power": 1.0,
        "entity_boundary_repulsion_enabled": False,
        "entity_boundary_distance_ratio": 4.8,
        "k_entity_boundary_repulse": 0.0,
        "entity_boundary_min_gap_ratio": 0.25,
        "entity_boundary_max_step_ratio": 0.5,
        "entity_sibling_attraction_enabled": False,
        "entity_sibling_target_ratio": 4.8,
        "entity_sibling_capture_ratio": 3.16,
        "k_entity_sibling_attr": 0.0,
        "entity_sibling_max_step_ratio": 0.5,
        "entity_sibling_boundary_attenuation_enabled": True,
        "entity_sibling_boundary_release_ratio": 1.25,
        "entity_sibling_boundary_power": 2.0,
        "entity_sibling_deadband_ratio": 0.0,
        "cluster_exclusion_enabled": False,
        "cluster_exclusion_distance_ratio": 3.16,
        "cluster_exclusion_max_step_ratio": 0.1,
        "cluster_exclusion_use_cluster_mass": True,
        "cluster_exclusion_mass_power": 1.0,
        "cluster_exclusion_distance_mode": "nearest_point",
        "project_close_pairs": False,
        "project_close_pairs_gain": 1.0,
        "edge_stretch_guard_enabled": False,
        "edge_stretch_max_step": 0.0,
        "edge_stretch_max_step_rel": 0.0,
        "edge_detach_distance": 0.0,
        "edge_detach_distance_rel": 0.0,
        "graph_weight_confidence_enabled": False,
        "graph_weight_confidence_percentile": 90.0,
        "graph_weight_confidence_power": 1.0,
        "graph_weight_confidence_floor": 0.0,
        "graph_weight_confidence_gain": 1.0,
        "stats_mode": "coactivation",
        "stats_memory_steps": 100,
        "stats_max_lag": 0,
        "stats_power_gain": 2.0,
        "weighted_stats_input": False,
        "T_push": 10,
        "knn": 12,
        "graph_theta": 0.1,
        "dbscan_eps": None,
        "dbscan_readout_eps": None,
        "dbscan_auto_eps": False,
        "dbscan_auto_eps_source": "knn95",
        "dbscan_auto_eps_margin": 1.05,
        "dbscan_min_samples": 2,
        "dbscan_readout_min_samples": None,
        "embed_cluster_method": "dbscan",
        "graph_louvain_resolution": 1.0,
        "clustering_period": 100,
        "louvain_distance_factor": 0.15,
        "clustering_random_state": None,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        "input_size",
        "dimensions",
        "base_wall",
        "density_ref",
        "packing_factor",
        "significant_distance_factor",
        "space_size",
        "reference_local_scale",
        "effective_local_scale",
        "significant_distance",
        "scale_factor",
        "adaptation_rate",
        "max_step_norm",
        "center_on_origin",
        "centering_strength",
        "recenter_graph_support_only",
        "graph_spring_mode",
        "graph_r0_mode",
        "graph_r0_fixed",
        "graph_r0_rel",
        "graph_r0",
        "graph_degree_normalize",
        "macro_boundary_repulsion_enabled",
        "macro_boundary_scale",
        "macro_cross_attr_scale",
        "macro_boundary_same_attr_scale",
        "macro_boundary_same_attr_size_power",
        "macro_boundary_coactivation_gate_enabled",
        "macro_boundary_coactivation_scale",
        "macro_boundary_coactivation_floor",
        "macro_boundary_softness_rel",
        "graph_spring_repulse_gain",
        "graph_spring_repulse_power",
        "graph_spring_repulse_shoulder_gain",
        "graph_spring_extend_shoulder_gain",
        "graph_spring_extend_shoulder_power",
        "graph_spring_extend_shoulder_rel_start",
        "graph_spring_soft_rel",
        "graph_spring_hard_rel",
        "graph_spring_near_slope",
        "graph_spring_mid_slope",
        "graph_spring_far_gain",
        "graph_spring_far_power",
        "line_obstacle_repulsion_enabled",
        "line_obstacle_radius_rel",
        "line_obstacle_strength",
        "primitive_contrastive_force_enabled",
        "primitive_active_attraction_strength",
        "primitive_inactive_repulsion_strength",
        "primitive_force_distance_floor_rel",
        "primitive_force_max_step_ratio",
        "negative_evidence_confidence_enabled",
        "negative_evidence_confidence_level",
        "k_attr",
        "k_repulse",
        "k_pulse",
        "pulse_mode",
        "plasticity_enabled",
        "plasticity_decay",
        "plasticity_min",
        "plasticity_recovery",
        "plasticity_update_period",
        "physics_batch_size",
        "physics_steps_per_batch",
        "inter_cluster_distance_ratio",
        "cluster_distance_strength",
        "macro_centroid_distance_ratio",
        "k_macro_repulse",
        "macro_centroid_max_step_ratio",
        "macro_centroid_near_gain",
        "macro_centroid_far_gain",
        "macro_centroid_cutoff_ratio",
        "macro_centroid_use_cluster_mass",
        "macro_centroid_mass_power",
        "entity_boundary_repulsion_enabled",
        "entity_boundary_distance_ratio",
        "k_entity_boundary_repulse",
        "entity_boundary_min_gap_ratio",
        "entity_boundary_max_step_ratio",
        "entity_sibling_attraction_enabled",
        "entity_sibling_target_ratio",
        "entity_sibling_capture_ratio",
        "k_entity_sibling_attr",
        "entity_sibling_max_step_ratio",
        "entity_sibling_boundary_attenuation_enabled",
        "entity_sibling_boundary_release_ratio",
        "entity_sibling_boundary_power",
        "entity_sibling_deadband_ratio",
        "cluster_exclusion_enabled",
        "cluster_exclusion_distance_ratio",
        "cluster_exclusion_max_step_ratio",
        "cluster_exclusion_use_cluster_mass",
        "cluster_exclusion_mass_power",
        "cluster_exclusion_distance_mode",
        "project_close_pairs",
        "project_close_pairs_gain",
        "edge_stretch_guard_enabled",
        "edge_stretch_max_step",
        "edge_stretch_max_step_rel",
        "edge_detach_distance",
        "edge_detach_distance_rel",
        "graph_weight_confidence_enabled",
        "graph_weight_confidence_percentile",
        "graph_weight_confidence_power",
        "graph_weight_confidence_floor",
        "graph_weight_confidence_gain",
        "stats_mode",
        "stats_memory_steps",
        "stats_max_lag",
        "stats_power_gain",
        "weighted_stats_input",
        "T_push",
        "knn",
        "graph_theta",
        "dbscan_eps",
        "dbscan_readout_eps",
        "dbscan_auto_eps",
        "dbscan_auto_eps_source",
        "dbscan_auto_eps_margin",
        "dbscan_min_samples",
        "dbscan_readout_min_samples",
        "embed_cluster_method",
        "graph_louvain_resolution",
        "clustering_period",
        "louvain_distance_factor",
        "clustering_random_state",
    )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return dict(cls.DEFAULT_PARAMS)

    @classmethod
    def normalize_params(cls, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = cls.default_params()
        if params:
            merged.update(dict(params))
        return merged

    def __init__(
        self,
        input_size: int,
        dimensions: int = 3,
        adaptation_rate: float = 1.0,
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        self.input_size = int(input_size)
        self.dimensions = int(dimensions)
        self._step_count = 0
        self._labels_version = 0

        manual_dbscan_eps = strategy_kwargs.get("dbscan_eps")
        self._manual_dbscan_eps: Optional[float] = (
            float(manual_dbscan_eps) if manual_dbscan_eps is not None else None
        )
        self.dbscan_eps = float(self._manual_dbscan_eps) if self._manual_dbscan_eps is not None else 0.0
        manual_dbscan_readout_eps = strategy_kwargs.get("dbscan_readout_eps")
        self._manual_dbscan_readout_eps: Optional[float] = (
            float(manual_dbscan_readout_eps) if manual_dbscan_readout_eps is not None else None
        )
        self.dbscan_readout_eps: Optional[float] = (
            float(self._manual_dbscan_readout_eps)
            if self._manual_dbscan_readout_eps is not None
            else None
        )
        self.dbscan_auto_eps = bool(strategy_kwargs.get("dbscan_auto_eps", False))
        self.dbscan_auto_eps_source = str(
            strategy_kwargs.get("dbscan_auto_eps_source", "knn95")
        ).strip().lower()
        self.dbscan_auto_eps_margin = max(
            float(strategy_kwargs.get("dbscan_auto_eps_margin", 1.05)),
            1e-6,
        )

        self.base_wall = float(strategy_kwargs.get("base_wall", 100.0))
        self.density_ref = float(strategy_kwargs.get("density_ref", 64.0))
        self.packing_factor = float(strategy_kwargs.get("packing_factor", DEFAULT_PACKING_FACTOR))
        self.significant_distance_factor = float(strategy_kwargs.get("significant_distance_factor", 0.5))
        requested_space = strategy_kwargs.get("space_size", -1.0)
        self.space_size = float(requested_space)
        self.reference_local_scale = 0.0
        self.effective_local_scale = 0.0
        self.significant_distance = 0.0
        self.significant_k = 0
        self.scale_factor = 1.0
        self._recalc_space_params()

        self.adaptation_rate = float(strategy_kwargs.get("adaptation_rate", adaptation_rate))
        self.max_step_norm = float(strategy_kwargs.get("max_step_norm", 5.0))
        self.center_on_origin = bool(strategy_kwargs.get("center_on_origin", True))
        self.centering_strength = float(strategy_kwargs.get("centering_strength", 1.0))
        self.recenter_graph_support_only = bool(strategy_kwargs.get("recenter_graph_support_only", False))

        self.graph_spring_mode = str(strategy_kwargs.get("graph_spring_mode", "rg_canonical")).lower()
        self.graph_r0_mode = str(strategy_kwargs.get("graph_r0_mode", "fixed")).lower()
        self.graph_r0_fixed = float(
            strategy_kwargs.get("graph_r0_fixed", strategy_kwargs.get("graph_r0_abs", 1.0))
        )
        self.graph_r0_rel = float(strategy_kwargs.get("graph_r0_rel", 1.0))
        self.graph_degree_normalize = bool(strategy_kwargs.get("graph_degree_normalize", True))
        self.macro_boundary_repulsion_enabled = bool(
            strategy_kwargs.get("macro_boundary_repulsion_enabled", False)
        )
        self.macro_boundary_scale = max(float(strategy_kwargs.get("macro_boundary_scale", 1.0)), 1.0)
        self.macro_cross_attr_scale = float(
            np.clip(strategy_kwargs.get("macro_cross_attr_scale", 0.001), 0.0, 1.0)
        )
        self.macro_boundary_same_attr_scale = max(
            float(strategy_kwargs.get("macro_boundary_same_attr_scale", 1.0)),
            1.0,
        )
        self.macro_boundary_same_attr_size_power = max(
            float(strategy_kwargs.get("macro_boundary_same_attr_size_power", 0.0)),
            0.0,
        )
        self.macro_boundary_coactivation_gate_enabled = bool(
            strategy_kwargs.get("macro_boundary_coactivation_gate_enabled", False)
        )
        self.macro_boundary_coactivation_scale = max(
            float(strategy_kwargs.get("macro_boundary_coactivation_scale", 0.05)),
            1e-9,
        )
        self.macro_boundary_coactivation_floor = float(
            np.clip(strategy_kwargs.get("macro_boundary_coactivation_floor", 0.0), 0.0, 1.0)
        )
        self.macro_boundary_softness_rel = max(
            float(strategy_kwargs.get("macro_boundary_softness_rel", 0.0)),
            0.0,
        )
        self.graph_spring_repulse_gain = float(strategy_kwargs.get("graph_spring_repulse_gain", 2.0))
        self.graph_spring_repulse_power = float(strategy_kwargs.get("graph_spring_repulse_power", 1.0))
        self.graph_spring_repulse_shoulder_gain = float(
            strategy_kwargs.get("graph_spring_repulse_shoulder_gain", 12.0)
        )
        self.graph_spring_extend_shoulder_gain = float(
            strategy_kwargs.get("graph_spring_extend_shoulder_gain", 0.0)
        )
        self.graph_spring_extend_shoulder_power = float(
            strategy_kwargs.get("graph_spring_extend_shoulder_power", 2.0)
        )
        self.graph_spring_extend_shoulder_rel_start = float(
            strategy_kwargs.get("graph_spring_extend_shoulder_rel_start", 0.3)
        )
        self.graph_spring_soft_rel = float(strategy_kwargs.get("graph_spring_soft_rel", 2.0))
        self.graph_spring_hard_rel = float(strategy_kwargs.get("graph_spring_hard_rel", 5.0))
        self.graph_spring_near_slope = float(strategy_kwargs.get("graph_spring_near_slope", 0.2))
        self.graph_spring_mid_slope = float(strategy_kwargs.get("graph_spring_mid_slope", 0.6))
        self.graph_spring_far_gain = float(strategy_kwargs.get("graph_spring_far_gain", 0.8))
        self.graph_spring_far_power = float(strategy_kwargs.get("graph_spring_far_power", 2.0))
        self.line_obstacle_repulsion_enabled = bool(
            strategy_kwargs.get("line_obstacle_repulsion_enabled", False)
        )
        self.line_obstacle_radius_rel = max(
            float(strategy_kwargs.get("line_obstacle_radius_rel", 1.0)),
            1e-6,
        )
        self.line_obstacle_strength = max(
            float(strategy_kwargs.get("line_obstacle_strength", 0.5)),
            0.0,
        )
        self.primitive_contrastive_force_enabled = bool(
            strategy_kwargs.get("primitive_contrastive_force_enabled", False)
        )
        self.primitive_active_attraction_strength = max(
            float(strategy_kwargs.get("primitive_active_attraction_strength", 0.0)),
            0.0,
        )
        self.primitive_inactive_repulsion_strength = max(
            float(strategy_kwargs.get("primitive_inactive_repulsion_strength", 0.0)),
            0.0,
        )
        self.primitive_force_distance_floor_rel = max(
            float(strategy_kwargs.get("primitive_force_distance_floor_rel", 0.25)),
            1e-6,
        )
        self.primitive_force_max_step_ratio = max(
            float(strategy_kwargs.get("primitive_force_max_step_ratio", 1.0)),
            0.0,
        )
        self.negative_evidence_confidence_enabled = bool(
            strategy_kwargs.get("negative_evidence_confidence_enabled", False)
        )
        self.negative_evidence_confidence_level = float(
            np.clip(strategy_kwargs.get("negative_evidence_confidence_level", 0.95), 0.5, 1.0 - 1e-9)
        )
        self.k_attr = float(strategy_kwargs.get("k_attr", 0.03))
        self.k_repulse = float(strategy_kwargs.get("k_repulse", 0.0))
        self.k_pulse = float(strategy_kwargs.get("k_pulse", 0.0))
        self.pulse_mode = str(strategy_kwargs.get("pulse_mode", "none")).lower()
        self.plasticity_enabled = bool(strategy_kwargs.get("plasticity_enabled", True))
        self.plasticity_decay = float(strategy_kwargs.get("plasticity_decay", 0.99))
        self.plasticity_min = float(strategy_kwargs.get("plasticity_min", 0.01))
        self.plasticity_recovery = float(strategy_kwargs.get("plasticity_recovery", 1.0))
        self.plasticity_update_period = max(1, int(strategy_kwargs.get("plasticity_update_period", 10)))
        self.physics_batch_size = max(1, int(strategy_kwargs.get("physics_batch_size", 10)))
        self.physics_steps_per_batch = max(1, int(strategy_kwargs.get("physics_steps_per_batch", 3)))
        # Algorithm-intrinsic feed granularity: drivers (unified experiment,
        # multilive) feed in step_chunk-sized batches so behaviour does not
        # depend on an externally chosen chunk size.
        self.step_chunk = max(1, int(strategy_kwargs.get("step_chunk", 100)))
        self.plasticity: FloatArray = np.ones(self.input_size, dtype=np.float64)
        self.inter_cluster_distance_ratio = float(strategy_kwargs.get("inter_cluster_distance_ratio", 3.16))
        self.cluster_distance_strength = float(strategy_kwargs.get("cluster_distance_strength", 0.0))
        self.macro_centroid_distance_ratio = float(
            strategy_kwargs.get(
                "macro_centroid_distance_ratio",
                strategy_kwargs.get(
                    "cluster_exclusion_distance_ratio",
                    self.inter_cluster_distance_ratio,
                ),
            )
        )
        self.k_macro_repulse = float(
            strategy_kwargs.get("k_macro_repulse", self.cluster_distance_strength)
        )
        self.macro_centroid_max_step_ratio = float(
            strategy_kwargs.get(
                "macro_centroid_max_step_ratio",
                strategy_kwargs.get("cluster_exclusion_max_step_ratio", 0.1),
            )
        )
        self.macro_centroid_near_gain = max(
            float(strategy_kwargs.get("macro_centroid_near_gain", 4.0)),
            0.0,
        )
        self.macro_centroid_far_gain = max(
            float(strategy_kwargs.get("macro_centroid_far_gain", 0.5)),
            0.0,
        )
        self.macro_centroid_cutoff_ratio = max(
            float(strategy_kwargs.get("macro_centroid_cutoff_ratio", 2.0)),
            1.0 + 1e-6,
        )
        self.macro_centroid_use_cluster_mass = bool(
            strategy_kwargs.get("macro_centroid_use_cluster_mass", True)
        )
        self.macro_centroid_mass_power = max(
            float(strategy_kwargs.get("macro_centroid_mass_power", 1.0)),
            0.0,
        )
        self.entity_boundary_repulsion_enabled = bool(
            strategy_kwargs.get("entity_boundary_repulsion_enabled", False)
        )
        self.entity_boundary_distance_ratio = max(
            float(strategy_kwargs.get("entity_boundary_distance_ratio", 4.8)),
            1.0,
        )
        self.k_entity_boundary_repulse = max(
            float(strategy_kwargs.get("k_entity_boundary_repulse", 0.0)),
            0.0,
        )
        self.entity_boundary_min_gap_ratio = float(
            np.clip(strategy_kwargs.get("entity_boundary_min_gap_ratio", 0.25), 1e-6, 1.0)
        )
        self.entity_boundary_max_step_ratio = max(
            float(strategy_kwargs.get("entity_boundary_max_step_ratio", 0.5)),
            0.0,
        )
        self.entity_sibling_attraction_enabled = bool(
            strategy_kwargs.get("entity_sibling_attraction_enabled", False)
        )
        self.entity_sibling_target_ratio = max(
            float(strategy_kwargs.get("entity_sibling_target_ratio", 4.8)),
            1.0,
        )
        self.entity_sibling_capture_ratio = max(
            float(strategy_kwargs.get("entity_sibling_capture_ratio", 3.16)),
            1.0,
        )
        self.k_entity_sibling_attr = max(
            float(strategy_kwargs.get("k_entity_sibling_attr", 0.0)),
            0.0,
        )
        self.entity_sibling_max_step_ratio = max(
            float(strategy_kwargs.get("entity_sibling_max_step_ratio", 0.5)),
            0.0,
        )
        self.entity_sibling_boundary_attenuation_enabled = bool(
            strategy_kwargs.get("entity_sibling_boundary_attenuation_enabled", True)
        )
        self.entity_sibling_boundary_release_ratio = max(
            float(strategy_kwargs.get("entity_sibling_boundary_release_ratio", 1.25)),
            1.0 + 1e-6,
        )
        self.entity_sibling_boundary_power = max(
            float(strategy_kwargs.get("entity_sibling_boundary_power", 2.0)),
            0.0,
        )
        self.entity_sibling_deadband_ratio = max(
            float(strategy_kwargs.get("entity_sibling_deadband_ratio", 0.0)),
            0.0,
        )
        self.cluster_exclusion_enabled = bool(strategy_kwargs.get("cluster_exclusion_enabled", False))
        self.cluster_exclusion_distance_ratio = float(
            strategy_kwargs.get("cluster_exclusion_distance_ratio", self.inter_cluster_distance_ratio)
        )
        self.cluster_exclusion_max_step_ratio = float(
            strategy_kwargs.get("cluster_exclusion_max_step_ratio", 0.1)
        )
        self.cluster_exclusion_use_cluster_mass = bool(
            strategy_kwargs.get("cluster_exclusion_use_cluster_mass", True)
        )
        self.cluster_exclusion_mass_power = float(
            strategy_kwargs.get("cluster_exclusion_mass_power", 1.0)
        )
        self.cluster_exclusion_distance_mode = str(
            strategy_kwargs.get("cluster_exclusion_distance_mode", "nearest_point")
        ).lower()

        self.project_close_pairs = bool(strategy_kwargs.get("project_close_pairs", False))
        self.project_close_pairs_gain = float(strategy_kwargs.get("project_close_pairs_gain", 1.0))
        self.edge_stretch_guard_enabled = bool(strategy_kwargs.get("edge_stretch_guard_enabled", False))
        self.edge_stretch_max_step = float(strategy_kwargs.get("edge_stretch_max_step", 0.0))
        self.edge_stretch_max_step_rel = float(strategy_kwargs.get("edge_stretch_max_step_rel", 0.0))
        self.edge_detach_distance = float(strategy_kwargs.get("edge_detach_distance", 0.0))
        self.edge_detach_distance_rel = float(strategy_kwargs.get("edge_detach_distance_rel", 0.0))
        self.graph_weight_confidence_enabled = bool(
            strategy_kwargs.get("graph_weight_confidence_enabled", False)
        )
        self.graph_weight_confidence_percentile = float(
            strategy_kwargs.get("graph_weight_confidence_percentile", 90.0)
        )
        self.graph_weight_confidence_power = float(
            strategy_kwargs.get("graph_weight_confidence_power", 1.0)
        )
        self.graph_weight_confidence_floor = float(
            strategy_kwargs.get("graph_weight_confidence_floor", 0.0)
        )
        self.graph_weight_confidence_gain = float(
            strategy_kwargs.get("graph_weight_confidence_gain", 1.0)
        )
        self.stats_mode = self._canonical_stats_mode(str(strategy_kwargs.get("stats_mode", "coactivation")))
        stats_memory_steps_raw = strategy_kwargs.get("stats_memory_steps", None)
        if stats_memory_steps_raw is None:
            stats_alpha_raw = strategy_kwargs.get("stats_alpha", None)
            if stats_alpha_raw is not None:
                try:
                    stats_alpha = float(stats_alpha_raw)
                except Exception:
                    stats_alpha = 0.0
                if stats_alpha > 1e-12:
                    stats_memory_steps_raw = max(1, int(round(1.0 / stats_alpha)))
        self.stats_memory_steps = int(100 if stats_memory_steps_raw is None else stats_memory_steps_raw)
        self.stats_max_lag = max(0, int(strategy_kwargs.get("stats_max_lag", 0)))
        self.stats_power_gain = float(strategy_kwargs.get("stats_power_gain", 2.0))
        self.weighted_stats_input = bool(strategy_kwargs.get("weighted_stats_input", False))
        self._coact: Optional[Union[_NSCoactivationStats, _NSTransitionTracker]] = None
        self._reinit_stats()

        self.T_push = int(strategy_kwargs.get("T_push", 10))
        self.knn = int(strategy_kwargs.get("knn", 12))
        self.graph_theta = float(strategy_kwargs.get("graph_theta", 0.1))

        self.dbscan_min_samples = int(strategy_kwargs.get("dbscan_min_samples", 3))
        raw_dbscan_readout_min_samples = strategy_kwargs.get("dbscan_readout_min_samples", None)
        self.dbscan_readout_min_samples: Optional[int] = (
            None if raw_dbscan_readout_min_samples is None else max(1, int(raw_dbscan_readout_min_samples))
        )
        self.embed_cluster_method = str(strategy_kwargs.get("embed_cluster_method", "dbscan")).lower()
        self.graph_louvain_resolution = float(strategy_kwargs.get("graph_louvain_resolution", 1.0))
        self.clustering_period = int(strategy_kwargs.get("clustering_period", 100))
        self.louvain_distance_factor = float(strategy_kwargs.get("louvain_distance_factor", 0.15))
        clustering_random_state = strategy_kwargs.get("clustering_random_state", None)
        self.clustering_random_state = None if clustering_random_state is None else int(clustering_random_state)
        self._reinit_clustering_strategy()

        self.syncmap: FloatArray = (
            (np.random.rand(self.input_size, self.dimensions) - 0.5) * self.space_size
        )
        self._last_total_update = np.zeros_like(self.syncmap)
        self._g_rows: Optional[IntArray] = None
        self._g_cols: Optional[IntArray] = None
        self._g_w: Optional[FloatArray] = None
        self.graph_r0 = max(float(self.graph_r0_fixed), 1e-6) if self.graph_r0_mode == "fixed" else 1.0
        self.size0 = 1.0
        self._current_active_mask: Optional[NDArray[np.bool_]] = None
        self._current_active_values: Optional[FloatArray] = None
        self._current_pulse_center: Optional[FloatArray] = None
        self._macro_feedback_labels: Optional[IntArray] = None
        self._macro_feedback_strengths: Optional[FloatArray] = None
        self._entity_feedback_labels: Optional[IntArray] = None
        self._entity_feedback_parent_labels: Optional[IntArray] = None
        self._entity_feedback_strengths: Optional[FloatArray] = None
        self._entity_feedback_parent_strengths: Optional[FloatArray] = None

        self.labels: Optional[IntArray] = None
        self.trajectory_interval = max(1, int(trajectory_interval))
        self.trajectory: Optional[list[FloatArray]] = [] if record_trajectory else None
        self.trajectory_steps: Optional[list[int]] = [] if record_trajectory else None

        self.organize(persist=True)

    @property
    def graph(self) -> Optional[tuple[IntArray, IntArray, FloatArray]]:
        if self._g_rows is None or self._g_cols is None or self._g_w is None:
            return None
        return self._g_rows, self._g_cols, self._g_w

    def _recalc_space_params(self) -> None:
        explicit_space_size = self.space_size if self.space_size > 0.0 else None
        scale = compute_density_scale(
            input_size=self.input_size,
            dimensions=self.dimensions,
            base_wall=self.base_wall,
            density_ref=self.density_ref,
            significant_distance_factor=self.significant_distance_factor,
            packing_factor=self.packing_factor,
            explicit_space_size=explicit_space_size,
        )

        self.reference_local_scale = float(scale.reference_local_scale)
        self.effective_local_scale = float(scale.effective_local_scale)
        self.space_size = float(scale.space_size)
        self.significant_distance = float(scale.significant_distance)
        self.significant_k = int(scale.significant_k)
        self.scale_factor = float(scale.scale_factor)
        self.scale = max(self.scale_factor, 0.5)

        natural_radius = float(self.reference_local_scale)
        if explicit_space_size is None and self.scale < 1.0:
            auto_dbscan_eps = max(1e-4, float(natural_radius * self.scale))
        else:
            auto_dbscan_eps = max(1e-4, float(self.significant_distance))

        if self._manual_dbscan_eps is not None and not self.dbscan_auto_eps:
            self.dbscan_eps = float(self._manual_dbscan_eps)
            return
        if self._manual_dbscan_eps is not None and self.dbscan_auto_eps:
            self.dbscan_eps = float(self._manual_dbscan_eps)
            return
        self.dbscan_eps = auto_dbscan_eps

    def _reinit_clustering_strategy(self) -> None:
        try:
            if self.embed_cluster_method == "dbscan":
                self.clustering_strategy = create_clustering_strategy(
                    "dbscan",
                    eps=self.dbscan_eps,
                    min_samples=int(self.dbscan_min_samples),
                )
            elif self.embed_cluster_method == "louvain":
                self.clustering_strategy = create_clustering_strategy(
                    "louvain",
                    distance_factor=self.louvain_distance_factor,
                    random_state=self.clustering_random_state,
                )
            elif self.embed_cluster_method == "graph_louvain":
                # Graph-backed readout is handled directly in organize()/periodic clustering.
                # Keep DBSCAN as a safe fallback before the first graph push.
                self.clustering_strategy = create_clustering_strategy(
                    "dbscan",
                    eps=self.dbscan_eps,
                    min_samples=int(self.dbscan_min_samples),
                )
            elif self.embed_cluster_method == "hdbscan":
                self.clustering_strategy = create_clustering_strategy(
                    "hdbscan",
                    min_cluster_size=int(self.dbscan_min_samples),
                    min_samples=int(self.dbscan_min_samples),
                )
            else:
                self.clustering_strategy = create_clustering_strategy("dbscan", eps=self.dbscan_eps)
        except Exception:
            self.clustering_strategy = create_clustering_strategy("dbscan", eps=self.dbscan_eps)

    def _graph_louvain_labels(
        self,
        *,
        resolution: float | None = None,
        samples: Optional[FloatArray] = None,
    ) -> IntArray:
        graph = self.graph
        if graph is None:
            fallback = create_clustering_strategy(
                "dbscan",
                eps=float(self.dbscan_eps),
                min_samples=int(self.dbscan_min_samples),
            )
            sample_arr = np.asarray(self.syncmap if samples is None else samples, dtype=float)
            return np.asarray(fallback.fit_predict(sample_arr), dtype=int)

        rows, cols, weights = graph
        node_count = int(self.input_size)
        if rows.size == 0 or cols.size == 0 or weights.size == 0:
            return np.arange(node_count, dtype=int)

        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(node_count))
        for idx_i, idx_j, weight in zip(rows.tolist(), cols.tolist(), weights.tolist()):
            if int(idx_i) == int(idx_j):
                continue
            graph_nx.add_edge(int(idx_i), int(idx_j), weight=float(weight))

        if graph_nx.number_of_edges() == 0:
            return np.arange(node_count, dtype=int)

        resolved_resolution = float(
            self.graph_louvain_resolution if resolution is None else resolution
        )
        try:
            partition = community_louvain.best_partition(
                graph_nx,
                weight="weight",
                resolution=resolved_resolution,
                random_state=self.clustering_random_state,
            )
        except TypeError:
            partition = community_louvain.best_partition(
                graph_nx,
                weight="weight",
                resolution=resolved_resolution,
            )
        return np.asarray([int(partition.get(node, -1)) for node in range(node_count)], dtype=int)

    @staticmethod
    def _canonical_stats_mode(mode: str) -> str:
        raw = str(mode).strip().lower()
        alias_map = {
            "coactivation": "coactivation",
            "ema": "coactivation",
            "markov": "coactivation",
            "transition": "transition",
            "step_transition": "transition",
            "ema_transition": "transition",
        }
        if raw in alias_map:
            return alias_map[raw]
        raise ValueError(f"Unsupported stats_mode: {mode}. Valid values: coactivation, transition")

    def _reinit_stats(self) -> None:
        self.stats_mode = self._canonical_stats_mode(self.stats_mode)
        self.stats_memory_steps = max(1, int(self.stats_memory_steps))
        if self.stats_mode == "transition":
            self._coact = _NSTransitionTracker(
                self.input_size,
                memory_steps=int(self.stats_memory_steps),
                power_gain=float(self.stats_power_gain),
            )
        else:
            self._coact = _NSCoactivationStats(
                self.input_size,
                memory_steps=int(self.stats_memory_steps),
                power_gain=float(self.stats_power_gain),
            )

    @staticmethod
    def _cluster_group_geometry(
        positions: FloatArray,
        labels: IntArray,
    ) -> tuple[
        IntArray,
        list[NDArray[np.bool_]],
        FloatArray,
        FloatArray,
        dict[int, int],
    ]:
        valid_labels = np.array(sorted({int(label) for label in labels if int(label) >= 0}), dtype=int)
        centroids = np.zeros((valid_labels.size, positions.shape[1]), dtype=float)
        sizes = np.ones(valid_labels.size, dtype=float)
        group_masks: list[NDArray[np.bool_]] = []
        for idx, label in enumerate(valid_labels.tolist()):
            mask = np.asarray(labels == label, dtype=bool)
            group_masks.append(mask)
            centroids[idx] = np.mean(positions[mask], axis=0)
            sizes[idx] = max(1.0, float(np.sum(mask)))
        label_to_group = {int(label): idx for idx, label in enumerate(valid_labels.tolist())}
        return valid_labels, group_masks, centroids, sizes, label_to_group

    @staticmethod
    def _cluster_pair_distance_direction(
        positions: FloatArray,
        group_masks: list[NDArray[np.bool_]],
        centroids: FloatArray,
        idx_i: int,
        idx_j: int,
        *,
        distance_mode: str = "centroid",
    ) -> tuple[float, FloatArray]:
        mode = str(distance_mode).lower()
        if mode == "nearest_point":
            pts_i = positions[group_masks[idx_i]]
            pts_j = positions[group_masks[idx_j]]
            if pts_i.size > 0 and pts_j.size > 0:
                pairwise = pts_j[None, :, :] - pts_i[:, None, :]
                pairwise_dist = np.linalg.norm(pairwise, axis=2)
                flat_idx = int(np.argmin(pairwise_dist))
                local_i, local_j = np.unravel_index(flat_idx, pairwise_dist.shape)
                vector = pairwise[local_i, local_j]
                dist = float(np.linalg.norm(vector))
                if dist > 1e-9:
                    return dist, vector / dist

        vector = centroids[idx_j] - centroids[idx_i]
        dist = float(np.linalg.norm(vector))
        if dist <= 1e-9:
            return 0.0, np.zeros(positions.shape[1], dtype=float)
        return dist, vector / dist

    @staticmethod
    def _relative_cluster_mass(
        sizes: FloatArray,
        *,
        use_cluster_mass: bool,
        mass_power: float,
    ) -> FloatArray:
        if use_cluster_mass:
            mean_size = max(float(np.mean(sizes)), 1e-9)
            return np.power(sizes / mean_size, float(max(mass_power, 0.0)))
        return np.ones_like(sizes)

    @staticmethod
    def _cap_group_shift(
        group_shift: FloatArray,
        max_group_step: float | None,
    ) -> FloatArray:
        if max_group_step is not None and max_group_step > 0.0:
            norms = np.linalg.norm(group_shift, axis=1)
            over = norms > float(max_group_step)
            if np.any(over):
                group_shift = np.array(group_shift, copy=True)
                group_shift[over] *= (float(max_group_step) / norms[over])[:, None]
        return group_shift

    @staticmethod
    def _lift_group_shift(
        positions: FloatArray,
        labels: IntArray,
        group_masks: list[NDArray[np.bool_]],
        group_shift: FloatArray,
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        for idx, mask in enumerate(group_masks):
            if np.any(mask):
                shift[mask] += group_shift[idx]
        return shift

    @staticmethod
    def _smooth_boundary_release(value: float, *, power: float) -> float:
        t = float(np.clip(value, 0.0, 1.0))
        smooth = t * t * (3.0 - 2.0 * t)
        if power > 0.0:
            smooth = smooth ** float(power)
        return float(np.clip(smooth, 0.0, 1.0))

    @staticmethod
    def _clip_point_shift(
        shift: FloatArray,
        max_point_step: float,
    ) -> FloatArray:
        if max_point_step <= 0.0:
            return shift
        norms = np.linalg.norm(shift, axis=1)
        over = norms > float(max_point_step)
        if not np.any(over):
            return shift
        clipped = np.array(shift, copy=True)
        clipped[over] *= (float(max_point_step) / norms[over])[:, None]
        return clipped

    @staticmethod
    def _compute_cluster_exclusion_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        strength: float,
        exclusion_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = "centroid",
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapNS._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift
        relative_mass = SyncMapNS._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        group_shift = np.zeros_like(centroids)
        normalizer = float(max(1, valid_labels.size - 1))
        target = max(float(exclusion_distance), 1e-6)

        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                dist, direction = SyncMapNS._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode=distance_mode,
                )
                if dist <= 1e-9:
                    continue
                delta = target - dist
                if delta <= 0.0:
                    continue
                repel = (float(strength) * delta / normalizer) * direction
                group_shift[i] -= relative_mass[j] * repel
                group_shift[j] += relative_mass[i] * repel

        group_shift = SyncMapNS._cap_group_shift(group_shift, max_group_step)
        return SyncMapNS._lift_group_shift(positions, labels, group_masks, group_shift)

    @staticmethod
    def _compute_macro_centroid_repulsion_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        strength: float,
        target_distance: float,
        max_group_step: float | None = None,
        strengths: FloatArray | None = None,
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapNS._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift

        strengths_supplied = False
        group_confidence = np.ones(valid_labels.size, dtype=float)
        if strengths is not None:
            node_strengths = np.asarray(strengths, dtype=float).reshape(-1)
            if node_strengths.shape[0] == positions.shape[0]:
                strengths_supplied = True
                group_confidence = np.asarray(
                    [
                        float(np.min(np.clip(node_strengths[mask], 0.0, 1.0)))
                        for mask in group_masks
                    ],
                    dtype=float,
                )

        group_shift = np.zeros_like(centroids)
        target = max(float(target_distance), 1e-6)
        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                pair_confidence = min(float(group_confidence[i]), float(group_confidence[j]))
                if pair_confidence <= 0.0:
                    pair_confidence = 0.0
                pair_target = target
                dist, direction_i_to_j = SyncMapNS._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode="nearest_point",
                )
                if dist <= 1e-9 or dist >= pair_target:
                    continue
                penetration = (pair_target - dist) / pair_target
                force_mag = float(strength) * pair_confidence * penetration
                force_vec = force_mag * direction_i_to_j
                group_shift[i] -= force_vec / max(float(sizes[i]), 1.0)
                group_shift[j] += force_vec / max(float(sizes[j]), 1.0)

        group_shift = SyncMapNS._cap_group_shift(group_shift, max_group_step)
        return SyncMapNS._lift_group_shift(positions, labels, group_masks, group_shift)

    def _apply_cluster_distance_maintenance(self, disp: FloatArray) -> None:
        if not self.macro_boundary_repulsion_enabled:
            return
        if self._macro_feedback_labels is None:
            return
        labels = np.asarray(self._macro_feedback_labels, dtype=int).reshape(-1)
        source_strengths = self._macro_feedback_strengths
        if labels.shape[0] != self.syncmap.shape[0]:
            return

        # Labels and strengths only change at hierarchy refresh, but this force
        # runs every step. Cache the label-derived geometry (masks, sizes,
        # confidence) keyed on their exact bytes; only centroids and pair
        # distances depend on positions and are recomputed per step. All
        # arithmetic on positions is identical to the uncached path, so the
        # trajectory stays bit-identical.
        cache = getattr(self, "_macro_geometry_cache", None)
        strengths_arr = None
        if source_strengths is not None:
            strengths_arr = np.asarray(source_strengths, dtype=float).reshape(-1)
        cache_key = (
            labels.tobytes(),
            None if strengths_arr is None else strengths_arr.tobytes(),
        )
        if cache is None or cache["key"] != cache_key:
            positions_now = np.asarray(self.syncmap, dtype=float)
            valid_labels, group_masks, _centroids, sizes, _label_map = self._cluster_group_geometry(
                positions_now,
                labels,
            )
            group_confidence = np.ones(valid_labels.size, dtype=float)
            if strengths_arr is not None and strengths_arr.shape[0] == labels.shape[0]:
                group_confidence = np.asarray(
                    [
                        float(np.min(np.clip(strengths_arr[mask], 0.0, 1.0)))
                        for mask in group_masks
                    ],
                    dtype=float,
                )
            cache = {
                "key": cache_key,
                "labels": labels.copy(),
                "masks": group_masks,
                "indices": [np.flatnonzero(mask) for mask in group_masks],
                "sizes": sizes,
                "confidence": group_confidence,
                "count": int(valid_labels.size),
            }
            self._macro_geometry_cache = cache
        if cache["count"] < 2:
            return

        strength = max(float(self.k_macro_repulse), 0.0)
        if strength <= 0.0:
            return

        base_distance = max(float(getattr(self, "graph_r0", self.graph_r0_fixed)), 1e-6)
        target_distance = max(base_distance * float(self.macro_centroid_distance_ratio), 1e-6)
        max_step = max(base_distance * float(self.macro_centroid_max_step_ratio), 1e-6)
        positions = np.asarray(self.syncmap, dtype=float)

        group_masks = cache["masks"]
        group_indices = cache["indices"]
        sizes = cache["sizes"]
        group_confidence = cache["confidence"]
        group_count = cache["count"]

        # Vectorized group centroids: scatter-sum grouped positions by group id
        # then divide by sizes, replacing the per-group np.mean python loop
        # (the real per-step cost on many-cluster graphs like EC). Groups are
        # small (<128 members) so scatter-add summation matches np.mean's
        # pairwise sum bitwise; verified against the saved baselines.
        node_group = cache.get("node_group")
        grouped_nodes = cache.get("grouped_nodes")
        grouped_ids = cache.get("grouped_ids")
        if node_group is None:
            node_group = np.full(positions.shape[0], -1, dtype=int)
            for group_idx in range(group_count):
                node_group[group_indices[group_idx]] = group_idx
            grouped_nodes = np.flatnonzero(node_group >= 0)
            grouped_ids = node_group[grouped_nodes]
            cache["node_group"] = node_group
            cache["grouped_nodes"] = grouped_nodes
            cache["grouped_ids"] = grouped_ids
        centroid_sums = np.zeros((group_count, positions.shape[1]), dtype=float)
        np.add.at(centroid_sums, grouped_ids, positions[grouped_nodes])
        centroids = centroid_sums / sizes[:, None]

        target = max(float(target_distance), 1e-6)
        # Exact separation pre-gate: the nearest inter-group point distance is
        # bounded below by centroid_dist - radius_i - radius_j. If that bound is
        # >= target for every pair, no pair can be within target, so every pair
        # would hit the `dist >= target: continue` branch and group_shift stays
        # zero. Skipping then produces the identical (unchanged) disp without
        # the O(n^2 d) full distance matrix. Once the repulsion has separated
        # the macros (the common steady state) this skips the whole force.
        # Vectorized radii: per-grouped-node distance to its centroid, then a
        # segment-max by group id (replaces the per-group python loop).
        node_centroid_dist = np.linalg.norm(positions[grouped_nodes] - centroids[grouped_ids], axis=1)
        group_radii = np.zeros(group_count, dtype=float)
        np.maximum.at(group_radii, grouped_ids, node_centroid_dist)
        centroid_dist = np.linalg.norm(centroids[None, :, :] - centroids[:, None, :], axis=2)
        lower_bound = centroid_dist - group_radii[:, None] - group_radii[None, :]
        iu = np.triu_indices(group_count, k=1)
        if not np.any(lower_bound[iu] < target):
            return

        # One full pairwise computation per step replaces per-pair blocks; the
        # per-element subtraction/norm operands are identical, and argmin over
        # the sliced block preserves the original row-major tie-breaking.
        diff_full = positions[None, :, :] - positions[:, None, :]
        dist_full = np.linalg.norm(diff_full, axis=2)

        group_shift = np.zeros_like(centroids)
        # Vectorized nearest-distance gate: most group pairs sit beyond the
        # target distance and are skipped; computing all pair minima in one
        # scatter-min avoids the O(k^2) python loop for them. Exact: min over
        # the same values, and the per-pair argmin below preserves the
        # original row-major tie-breaking for pairs that do interact.
        # node_group/grouped_* were cached above when centroids were built.
        sub_dist = dist_full[np.ix_(grouped_nodes, grouped_nodes)]
        pair_min = np.full((group_count, group_count), np.inf, dtype=float)
        np.minimum.at(
            pair_min,
            (
                np.repeat(grouped_ids, grouped_ids.size),
                np.tile(grouped_ids, grouped_ids.size),
            ),
            sub_dist.ravel(),
        )
        near_i, near_j = np.where(np.triu(pair_min < target, k=1))
        for i, j in zip(near_i.tolist(), near_j.tolist()):
                pair_confidence = min(float(group_confidence[i]), float(group_confidence[j]))
                if pair_confidence <= 0.0:
                    pair_confidence = 0.0
                pair_target = target
                idx_i_arr = group_indices[i]
                idx_j_arr = group_indices[j]
                block = dist_full[np.ix_(idx_i_arr, idx_j_arr)]
                flat_idx = int(np.argmin(block))
                local_i, local_j = np.unravel_index(flat_idx, block.shape)
                node_i = int(idx_i_arr[local_i])
                node_j = int(idx_j_arr[local_j])
                dist = float(dist_full[node_i, node_j])
                if dist > 1e-9:
                    direction_i_to_j = diff_full[node_i, node_j] / dist
                else:
                    centroid_vector = centroids[j] - centroids[i]
                    centroid_dist = float(np.linalg.norm(centroid_vector))
                    if centroid_dist <= 1e-9:
                        continue
                    dist = centroid_dist
                    direction_i_to_j = centroid_vector / centroid_dist
                if dist <= 1e-9 or dist >= pair_target:
                    continue
                penetration = (pair_target - dist) / pair_target
                force_mag = float(strength) * pair_confidence * penetration
                force_vec = force_mag * direction_i_to_j
                group_shift[i] -= force_vec / max(float(sizes[i]), 1.0)
                group_shift[j] += force_vec / max(float(sizes[j]), 1.0)

        group_shift = self._cap_group_shift(group_shift, max_step)
        shift = self._lift_group_shift(positions, cache["labels"], group_masks, group_shift)
        if not np.any(np.abs(shift) > 1e-12):
            return
        disp += self._clip_point_shift(shift, max_step)

    @staticmethod
    def _compute_entity_boundary_repulsion_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        strength: float,
        target_distance: float,
        max_group_step: float | None = None,
        strengths: FloatArray | None = None,
        min_gap_ratio: float = 0.25,
    ) -> FloatArray:
        """Push separate extracted entities apart only while their surfaces overlap a scale boundary."""
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapNS._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift

        strengths_supplied = False
        group_confidence = np.ones(valid_labels.size, dtype=float)
        if strengths is not None:
            node_strengths = np.asarray(strengths, dtype=float).reshape(-1)
            if node_strengths.shape[0] == positions.shape[0]:
                strengths_supplied = True
                group_confidence = np.asarray(
                    [
                        float(np.min(np.clip(node_strengths[mask], 0.0, 1.0)))
                        for mask in group_masks
                    ],
                    dtype=float,
                )

        target = max(float(target_distance), 1e-6)
        group_shift = np.zeros_like(centroids)
        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                pair_confidence = min(float(group_confidence[i]), float(group_confidence[j]))
                dist, direction_i_to_j = SyncMapNS._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode="nearest_point",
                )
                if dist <= 1e-9 or dist >= target:
                    continue
                if pair_confidence <= 0.0:
                    continue
                penetration = (target - dist) / target
                floor = target * float(np.clip(min_gap_ratio, 1e-6, 1.0))
                safe_dist = max(float(dist), floor)
                effective_penetration = (target - safe_dist) / target
                force_mag = float(strength) * max(effective_penetration, penetration * pair_confidence)
                if force_mag <= 0.0:
                    continue
                force_vec = force_mag * direction_i_to_j
                group_shift[i] -= force_vec / max(float(sizes[i]), 1.0)
                group_shift[j] += force_vec / max(float(sizes[j]), 1.0)

        group_shift = SyncMapNS._cap_group_shift(group_shift, max_group_step)
        return SyncMapNS._lift_group_shift(positions, labels, group_masks, group_shift)

    def _apply_entity_boundary_maintenance(self, disp: FloatArray) -> None:
        if not self.entity_boundary_repulsion_enabled or self._entity_feedback_labels is None:
            return
        labels = np.asarray(self._entity_feedback_labels, dtype=int).reshape(-1)
        if labels.shape[0] != self.syncmap.shape[0]:
            return
        strength = max(float(self.k_entity_boundary_repulse), 0.0)
        if strength <= 0.0:
            return
        base_distance = max(float(getattr(self, "graph_r0", self.graph_r0_fixed)), 1e-6)
        entity_target = max(base_distance * float(self.entity_boundary_distance_ratio), 1e-6)
        macro_target = max(base_distance * float(self.macro_centroid_distance_ratio), 1e-6)
        max_step = max(base_distance * float(self.entity_boundary_max_step_ratio), 1e-6)
        strengths = self._entity_feedback_strengths

        parent_labels = self._entity_feedback_parent_labels
        if parent_labels is not None and parent_labels.shape[0] == self.syncmap.shape[0]:
            parent_arr = np.asarray(parent_labels, dtype=int).reshape(-1)
            unique_parents = np.unique(parent_arr[parent_arr >= 0])
            if unique_parents.size >= 1:
                shift_total = np.zeros_like(np.asarray(self.syncmap, dtype=float))
                positions = np.asarray(self.syncmap, dtype=float)
                # Same-parent entity repulsion at entity_target
                for parent_id in unique_parents.tolist():
                    child_mask = (labels >= 0) & (parent_arr == parent_id)
                    child_indices = np.flatnonzero(child_mask)
                    if child_indices.size < 2:
                        continue
                    child_entity_labels = labels[child_mask]
                    unique_child_labels = np.unique(child_entity_labels[child_entity_labels >= 0])
                    if unique_child_labels.size < 2:
                        continue
                    local_label_map = {int(lab): idx for idx, lab in enumerate(unique_child_labels.tolist())}
                    local_labels = np.full(labels.shape[0], -1, dtype=int)
                    for idx in child_indices.tolist():
                        orig = int(labels[idx])
                        if orig in local_label_map:
                            local_labels[idx] = local_label_map[orig]
                    shift = self._compute_entity_boundary_repulsion_shift(
                        positions, local_labels,
                        strength=strength,
                        target_distance=entity_target,
                        max_group_step=max_step,
                        strengths=strengths,
                        min_gap_ratio=self.entity_boundary_min_gap_ratio,
                    )
                    shift_total += shift
                # Cross-parent entity repulsion at macro_target
                # Only push entities with DIFFERENT parents apart to macro_target.
                # Same-parent entities already handled by the per-parent loop above.
                if unique_parents.size >= 2:
                    cross_labels = np.full(labels.shape[0], -1, dtype=int)
                    # Remap all entities to 0..N-1 (keep same global index space)
                    all_valid_entities = np.unique(labels[labels >= 0])
                    global_label_map = {int(lab): idx for idx, lab in enumerate(all_valid_entities.tolist())}
                    for global_idx in range(labels.shape[0]):
                        orig = int(labels[global_idx])
                        if orig in global_label_map:
                            cross_labels[global_idx] = global_label_map[orig]
                    # Push: all entity pairs → macro_target, BUT only pairs with
                    # DIFFERENT parents experience the force. We do this by building
                    # per-parent-pair shift contributions.
                    parents_list = unique_parents.tolist()
                    for pi in range(len(parents_list) - 1):
                        pa = parents_list[pi]
                        mask_a = (labels >= 0) & (parent_arr == pa)
                        for pj in range(pi + 1, len(parents_list)):
                            pb = parents_list[pj]
                            mask_b = (labels >= 0) & (parent_arr == pb)
                            pair_labels = np.full(labels.shape[0], -1, dtype=int)
                            pair_labels[mask_a] = cross_labels[mask_a]
                            pair_labels[mask_b] = cross_labels[mask_b]
                            shift_pair = self._compute_entity_boundary_repulsion_shift(
                                positions, pair_labels,
                                strength=strength,
                                target_distance=macro_target,
                                max_group_step=max_step,
                                strengths=strengths,
                                min_gap_ratio=self.entity_boundary_min_gap_ratio,
                            )
                            shift_total += shift_pair
                if np.any(np.abs(shift_total) > 1e-12):
                    disp += self._clip_point_shift(shift_total, max_step)
        else:
            shift = self._compute_entity_boundary_repulsion_shift(
                np.asarray(self.syncmap, dtype=float),
                labels,
                strength=strength,
                target_distance=entity_target,
                max_group_step=max_step,
                strengths=self._entity_feedback_strengths,
                min_gap_ratio=self.entity_boundary_min_gap_ratio,
            )
            if np.any(np.abs(shift) > 1e-12):
                disp += self._clip_point_shift(shift, max_step)

    @staticmethod
    def _compute_entity_sibling_attraction_shift(
        positions: FloatArray,
        entity_labels: IntArray,
        parent_labels: IntArray,
        *,
        strength: float,
        target_distance: float,
        capture_distance: float,
        max_group_step: float | None = None,
        parent_strengths: FloatArray | None = None,
        boundary_attenuation_enabled: bool = True,
        boundary_release_ratio: float = 1.25,
        boundary_power: float = 2.0,
        deadband_ratio: float = 0.25,
    ) -> FloatArray:
        """Pull distinct entities with one approved parent toward a bounded scale well."""
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapNS._cluster_group_geometry(
            positions,
            entity_labels,
        )
        if valid_labels.size < 2:
            return shift

        parent_arr = np.asarray(parent_labels, dtype=int).reshape(-1)
        if parent_arr.shape[0] != positions.shape[0]:
            return shift
        node_confidence = None
        if parent_strengths is not None:
            strength_arr = np.asarray(parent_strengths, dtype=float).reshape(-1)
            if strength_arr.shape[0] == positions.shape[0]:
                node_confidence = np.clip(strength_arr, 0.0, 1.0)

        group_parents = np.full(valid_labels.size, -1, dtype=int)
        group_confidence = np.ones(valid_labels.size, dtype=float)
        for idx, mask in enumerate(group_masks):
            parents = np.unique(parent_arr[mask])
            parents = parents[parents >= 0]
            if parents.size != 1:
                continue
            group_parents[idx] = int(parents[0])
            if node_confidence is not None:
                group_confidence[idx] = float(np.min(node_confidence[mask]))

        target = max(float(target_distance), 1e-6)
        attraction_target = target * (1.0 + max(float(deadband_ratio), 0.0))
        capture = max(float(capture_distance), target)
        max_extension = max((capture - attraction_target) / target, 0.0)
        release = max(target * float(boundary_release_ratio), target + 1e-6)
        attenuation_power = max(float(boundary_power), 0.0)
        group_shift = np.zeros_like(centroids)
        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                if group_parents[i] < 0 or group_parents[i] != group_parents[j]:
                    continue
                pair_confidence = min(float(group_confidence[i]), float(group_confidence[j]))
                if pair_confidence <= 0.0:
                    continue
                dist, direction_i_to_j = SyncMapNS._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode="nearest_point",
                )
                if dist <= attraction_target or dist <= 1e-9:
                    continue
                extension = min((dist - attraction_target) / target, max_extension)
                if extension <= 0.0:
                    continue
                attenuation = 1.0
                if boundary_attenuation_enabled:
                    candidate_shift = np.zeros_like(centroids)
                    candidate_shift[i] += direction_i_to_j / max(float(sizes[i]), 1.0)
                    candidate_shift[j] -= direction_i_to_j / max(float(sizes[j]), 1.0)
                    affected = (i, j)
                    for group_idx in affected:
                        for other_idx in range(valid_labels.size):
                            if other_idx == group_idx:
                                continue
                            near_dist, direction_group_to_other = SyncMapNS._cluster_pair_distance_direction(
                                positions,
                                group_masks,
                                centroids,
                                group_idx,
                                other_idx,
                                distance_mode="nearest_point",
                            )
                            if near_dist <= 1e-9 or near_dist >= release:
                                continue
                            delta_gap = float(
                                np.dot(
                                    candidate_shift[other_idx] - candidate_shift[group_idx],
                                    direction_group_to_other,
                                )
                            )
                            if delta_gap >= 0.0:
                                continue
                            clearance = (near_dist - target) / max(release - target, 1e-9)
                            attenuation = min(
                                attenuation,
                                SyncMapNS._smooth_boundary_release(
                                    clearance,
                                    power=attenuation_power,
                                ),
                            )
                            if attenuation <= 0.0:
                                break
                        if attenuation <= 0.0:
                            break
                if attenuation <= 0.0:
                    continue
                force_vec = float(strength) * pair_confidence * extension * attenuation * direction_i_to_j
                group_shift[i] += force_vec / max(float(sizes[i]), 1.0)
                group_shift[j] -= force_vec / max(float(sizes[j]), 1.0)

        group_shift = SyncMapNS._cap_group_shift(group_shift, max_group_step)
        return SyncMapNS._lift_group_shift(positions, entity_labels, group_masks, group_shift)

    def _apply_entity_sibling_attraction(self, disp: FloatArray) -> None:
        if (
            not self.entity_sibling_attraction_enabled
            or self._entity_feedback_labels is None
            or self._entity_feedback_parent_labels is None
        ):
            return
        strength = max(float(self.k_entity_sibling_attr), 0.0)
        if strength <= 0.0:
            return
        base_distance = max(float(getattr(self, "graph_r0", self.graph_r0_fixed)), 1e-6)
        target_distance = max(base_distance * float(self.entity_sibling_target_ratio), 1e-6)
        capture_distance = max(target_distance * float(self.entity_sibling_capture_ratio), target_distance)
        max_step = max(base_distance * float(self.entity_sibling_max_step_ratio), 1e-6)
        shift = self._compute_entity_sibling_attraction_shift(
            np.asarray(self.syncmap, dtype=float),
            np.asarray(self._entity_feedback_labels, dtype=int),
            np.asarray(self._entity_feedback_parent_labels, dtype=int),
            strength=strength,
            target_distance=target_distance,
            capture_distance=capture_distance,
            max_group_step=max_step,
            parent_strengths=self._entity_feedback_parent_strengths,
            boundary_attenuation_enabled=bool(self.entity_sibling_boundary_attenuation_enabled),
            boundary_release_ratio=float(self.entity_sibling_boundary_release_ratio),
            boundary_power=float(self.entity_sibling_boundary_power),
            deadband_ratio=float(self.entity_sibling_deadband_ratio),
        )
        if np.any(np.abs(shift) > 1e-12):
            disp += self._clip_point_shift(shift, max_step)

    def _graph_weight_confidence_multiplier(self, weights: FloatArray) -> FloatArray:
        edge_w = np.asarray(weights, dtype=float)
        if edge_w.size == 0:
            return edge_w
        positive = edge_w[np.isfinite(edge_w) & (edge_w > 0.0)]
        if positive.size == 0:
            return np.ones_like(edge_w, dtype=float)

        percentile = float(np.clip(self.graph_weight_confidence_percentile, 1.0, 100.0))
        scale = float(np.percentile(positive, percentile))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.max(positive))
        if not np.isfinite(scale) or scale <= 1e-12:
            return np.ones_like(edge_w, dtype=float)

        confidence = np.clip(edge_w / scale, 0.0, 1.0)
        power = max(float(self.graph_weight_confidence_power), 1e-6)
        if abs(power - 1.0) > 1e-9:
            confidence = np.power(confidence, power)

        floor = max(float(self.graph_weight_confidence_floor), 0.0)
        gain = max(float(self.graph_weight_confidence_gain), 0.0)
        return floor + (gain - floor) * confidence

    def _effective_graph_spring_weights(
        self,
        idx_i: IntArray,
        idx_j: IntArray,
        weights: FloatArray,
    ) -> FloatArray:
        confidence_params = (
            bool(self.graph_weight_confidence_enabled),
            float(self.graph_weight_confidence_percentile),
            float(self.graph_weight_confidence_power),
            float(self.graph_weight_confidence_floor),
            float(self.graph_weight_confidence_gain),
            bool(self.graph_degree_normalize),
        )
        spring_cache = getattr(self, "_spring_weight_cache", None)
        if (
            spring_cache is not None
            and spring_cache[0] is idx_i
            and spring_cache[1] is idx_j
            and spring_cache[2] is weights
            and spring_cache[3] == confidence_params
        ):
            return spring_cache[4]

        if self.graph_degree_normalize:
            degree = np.bincount(
                np.concatenate([idx_i, idx_j]),
                minlength=self.syncmap.shape[0],
            ).astype(float)
            w_eff = weights * (2.0 / np.maximum(degree[idx_i] + degree[idx_j], 1.0))
        else:
            w_eff = np.asarray(weights, dtype=float)

        if self.graph_weight_confidence_enabled:
            w_eff = w_eff * self._graph_weight_confidence_multiplier(weights)

        self._spring_weight_cache = (idx_i, idx_j, weights, confidence_params, w_eff)
        return w_eff

    def _apply_graph_springs(self, disp: FloatArray) -> None:
        if self._g_rows is None or self._g_cols is None or self._g_w is None:
            return
        idx_i = self._g_rows
        idx_j = self._g_cols
        weights = self._g_w
        if idx_i.size == 0:
            return

        w_eff = self._effective_graph_spring_weights(idx_i, idx_j, weights)

        diff = self.syncmap[idx_j] - self.syncmap[idx_i]
        dist = np.linalg.norm(diff, axis=1) + 1e-9
        target = max(float(self.graph_r0), 1e-6)
        macro_labels = self._macro_feedback_labels
        cross_macro = np.zeros(dist.shape, dtype=bool)
        if macro_labels is not None and macro_labels.shape[0] == self.syncmap.shape[0]:
            macro_i = macro_labels[idx_i]
            macro_j = macro_labels[idx_j]
            cross_macro = (
                (macro_i >= 0)
                & (macro_j >= 0)
                & (macro_i != macro_j)
            )

        if self.graph_spring_mode == "rg_canonical":
            tgt = np.broadcast_to(target, dist.shape)
            factor = np.empty_like(dist)
            far_mask = dist >= tgt
            if np.any(far_mask):
                detach_limit = self._edge_detach_distance_abs()
                tgt_far = tgt[far_mask]
                detach_far = np.where(
                    detach_limit <= tgt_far,
                    tgt_far * max(float(self.inter_cluster_distance_ratio), 1.0),
                    detach_limit,
                )
                extend_span = np.maximum(detach_far - tgt_far, 1e-6)
                extend_start = tgt_far + (
                    float(self.graph_spring_extend_shoulder_rel_start) * extend_span
                )
                extension = dist[far_mask] - tgt_far
                shoulder = np.zeros_like(extension)
                shoulder_mask = dist[far_mask] > extend_start
                if np.any(shoulder_mask):
                    rel = np.clip(
                        (dist[far_mask][shoulder_mask] - extend_start[shoulder_mask])
                        / np.maximum(detach_far[shoulder_mask] - extend_start[shoulder_mask], 1e-6),
                        0.0,
                        None,
                    )
                    shoulder[shoulder_mask] = (
                        max(float(self.graph_spring_extend_shoulder_gain), 0.0)
                        * tgt_far[shoulder_mask]
                        * np.power(
                            rel,
                            max(float(self.graph_spring_extend_shoulder_power), 1.0),
                        )
                    )
                factor[far_mask] = self.k_attr * w_eff[far_mask] * (extension + shoulder)

            near_mask = ~far_mask
            if np.any(near_mask):
                tgt_near = tgt[near_mask]
                safe_dist = np.maximum(dist[near_mask], 1e-9)
                compression = tgt_near - safe_dist
                factor[near_mask] = -(
                    self.k_attr
                    * w_eff[near_mask]
                    * (
                        max(float(self.graph_spring_repulse_gain), 0.0)
                        * tgt_near
                        * (
                            np.power(
                                tgt_near / safe_dist,
                                max(float(self.graph_spring_repulse_power), 1e-6),
                            )
                            - 1.0
                        )
                        + max(float(self.graph_spring_repulse_shoulder_gain), 0.0) * compression
                    )
                )
        elif self.graph_spring_mode == "stable_soft":
            delta = dist - target
            rel = delta / target
            abs_rel = np.abs(rel)
            soft_break = max(float(self.graph_spring_soft_rel), 1e-6)
            hard_break = max(float(self.graph_spring_hard_rel), soft_break + 1e-6)
            near_mag = max(float(self.graph_spring_near_slope), 0.0) * np.minimum(abs_rel, soft_break)
            mid_excess = np.maximum(np.minimum(abs_rel, hard_break) - soft_break, 0.0)
            mid_mag = (
                max(float(self.graph_spring_near_slope), 0.0) * soft_break
                + max(float(self.graph_spring_mid_slope), 0.0) * mid_excess
            )
            far_excess = np.maximum(abs_rel - hard_break, 0.0)
            far_mag = (
                max(float(self.graph_spring_near_slope), 0.0) * soft_break
                + max(float(self.graph_spring_mid_slope), 0.0) * (hard_break - soft_break)
                + max(float(self.graph_spring_far_gain), 0.0)
                * np.power(far_excess, max(float(self.graph_spring_far_power), 1.0))
            )
            rel_mag = np.where(
                abs_rel <= soft_break,
                near_mag,
                np.where(abs_rel <= hard_break, mid_mag, far_mag),
            )
            factor = self.k_attr * w_eff * target * np.sign(rel) * rel_mag
        elif self.graph_spring_mode == "nsgd_simple":
            # Legacy NSGD spring (mirrors SyncMapX_NSGD._apply_graph_springs):
            # constant-magnitude attraction along graph edges, active only when
            # stretched beyond the rest length `target` (no compression/repulsion
            # when close). Lets NS reproduce the NSGD force model purely via
            # config (pair with k_attr≈1.0). Additive option; default unchanged.
            factor = np.where(dist > target, self.k_attr * w_eff, 0.0)
        else:
            factor = self.k_attr * w_eff * (dist - target)

        if np.any(cross_macro):
            factor = np.array(factor, copy=True)
            cross_attr_mask = cross_macro & (factor > 0.0)
            if np.any(cross_attr_mask):
                factor[cross_attr_mask] *= float(self.macro_cross_attr_scale)

        move = diff * (factor[:, None] / dist[:, None])
        np.add.at(disp, idx_i, move)
        np.add.at(disp, idx_j, -move)

    def _fallback_obstacle_direction(self, axis: FloatArray) -> FloatArray:
        basis = np.zeros(self.dimensions, dtype=float)
        basis[int(np.argmin(np.abs(axis)))] = 1.0
        direction = basis - float(np.dot(basis, axis)) * axis
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.zeros(self.dimensions, dtype=float)
            direction[0] = 1.0
            return direction
        return direction / norm

    def _apply_active_edge_line_repulsion(self, disp: FloatArray) -> None:
        if not self.line_obstacle_repulsion_enabled:
            return
        if self._g_rows is None or self._g_cols is None or self._g_w is None:
            return
        if self._current_active_mask is None:
            return
        active_mask = np.asarray(self._current_active_mask, dtype=bool).reshape(-1)
        if active_mask.shape[0] != self.syncmap.shape[0]:
            return
        strength = max(float(self.line_obstacle_strength), 0.0)
        radius = max(float(self.graph_r0) * float(self.line_obstacle_radius_rel), 1e-6)
        if strength <= 0.0 or radius <= 0.0 or self.input_size <= 2:
            return

        idx_i = np.asarray(self._g_rows, dtype=int)
        idx_j = np.asarray(self._g_cols, dtype=int)
        if idx_i.size == 0:
            return
        active_edge_mask = active_mask[idx_i] & active_mask[idx_j]
        if not np.any(active_edge_mask):
            return

        positions = np.asarray(self.syncmap, dtype=float)
        edge_i = idx_i[active_edge_mask]
        edge_j = idx_j[active_edge_mask]
        edge_w = np.asarray(self._g_w, dtype=float)[active_edge_mask]
        target = max(float(self.graph_r0), 1e-6)

        for i, j, weight in zip(edge_i.tolist(), edge_j.tolist(), edge_w.tolist()):
            p_i = positions[int(i)]
            p_j = positions[int(j)]
            edge_vec = p_j - p_i
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= target:
                continue
            axis = edge_vec / edge_len
            for k in range(self.input_size):
                if k == int(i) or k == int(j):
                    continue
                offset = positions[k] - p_i
                projection = float(np.dot(offset, axis) / edge_len)
                if projection <= 0.05 or projection >= 0.95:
                    continue
                closest = p_i + projection * edge_vec
                radial = positions[k] - closest
                radial_dist = float(np.linalg.norm(radial))
                if radial_dist >= radius:
                    continue
                if radial_dist <= 1e-12:
                    direction = self._fallback_obstacle_direction(axis)
                else:
                    direction = radial / radial_dist
                penetration = radius - radial_dist
                magnitude = float(strength) * float(weight) * (penetration / radius)
                push = magnitude * direction
                disp[k] += push
                disp[int(i)] -= 0.5 * (1.0 - projection) * push
                disp[int(j)] -= 0.5 * projection * push

    def _observe_input_symbol(self, values: FloatArray, active_mask: NDArray[np.bool_]) -> None:
        """Optional symbol-stream observation hook.

        The base NS/RG path deliberately does nothing. Experimental engines
        may derive internal clocks or confidence only from ``values`` and
        ``active_mask``; environment state and labels are not available here.
        """
        return

    def _negative_relation_confidence(
        self,
        active_idx: IntArray,
        inactive_idx: IntArray,
        active_weights: FloatArray | None = None,
    ) -> FloatArray:
        """Return one exclusion-confidence multiplier per inactive symbol.

        NS/RG use an all-ones multiplier.
        """
        return np.ones(np.asarray(inactive_idx, dtype=int).size, dtype=float)

    def _primitive_inactive_repulsion_multiplier(
        self,
        active_idx: IntArray,
        inactive_idx: IntArray,
        active_weights: FloatArray | None = None,
    ) -> FloatArray:
        """Multiplier for the primitive inactive-force branch.

        The NS/RG implementation uses the same unity multiplier for
        immediate input contrast and graph-derived obstacle relations.
        """
        return self._negative_relation_confidence(
            active_idx,
            inactive_idx,
            active_weights=active_weights,
        )

    def _local_obstacle_repulsion_multiplier(
        self,
        active_idx: IntArray,
        inactive_idx: IntArray,
        active_weights: FloatArray | None = None,
    ) -> FloatArray:
        """Multiplier for graph-local obstacle repulsion."""
        return self._negative_relation_confidence(
            active_idx,
            inactive_idx,
            active_weights=active_weights,
        )

    def get_symbol_clock_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "symbol_clock": int(self._step_count),
            "pattern_clock": 0,
            "evidence_clock": 0,
        }

    def _apply_primitive_contrastive_force(self, disp: FloatArray) -> None:
        """Apply label-free per-observation attraction and negative evidence repulsion."""
        if not self.primitive_contrastive_force_enabled or self._current_active_mask is None:
            return
        active_mask = np.asarray(self._current_active_mask, dtype=bool).reshape(-1)
        if active_mask.shape[0] != self.syncmap.shape[0]:
            return
        active_idx = np.flatnonzero(active_mask)
        if active_idx.size == 0:
            return
        inactive_idx = np.flatnonzero(~active_mask)
        positions = np.asarray(self.syncmap, dtype=float)
        active_positions = positions[active_idx]
        values = self._current_active_values
        if values is not None and np.asarray(values).shape[0] == positions.shape[0]:
            active_weights = np.clip(np.asarray(values, dtype=float)[active_idx], 0.0, None)
        else:
            active_weights = np.ones(active_idx.size, dtype=float)
        weight_total = float(np.sum(active_weights))
        if weight_total > 1e-12:
            # Same multiply/sum/divide order as np.average without its wrapper.
            positive_center = (
                np.multiply(active_positions, active_weights[:, None]).sum(axis=0)
                / active_weights.sum()
            )
        else:
            positive_center = np.mean(active_positions, axis=0)

        base_distance = max(float(getattr(self, "graph_r0", self.graph_r0_fixed)), 1e-6)
        distance_floor = max(base_distance * float(self.primitive_force_distance_floor_rel), 1e-6)
        shift = np.zeros_like(positions)

        positive_strength = max(float(self.primitive_active_attraction_strength), 0.0)
        if positive_strength > 0.0 and active_idx.size > 1:
            toward_center = positive_center[None, :] - active_positions
            active_dist = np.linalg.norm(toward_center, axis=1)
            safe_active_dist = np.maximum(active_dist, distance_floor)
            shift[active_idx] += positive_strength * toward_center / safe_active_dist[:, None]

        negative_strength = max(float(self.primitive_inactive_repulsion_strength), 0.0)
        if negative_strength > 0.0 and inactive_idx.size > 0:
            away_from_center = positions[inactive_idx] - positive_center[None, :]
            inactive_dist = np.linalg.norm(away_from_center, axis=1)
            safe_inactive_dist = np.maximum(inactive_dist, distance_floor)
            repulsion = (
                negative_strength
                * base_distance
                * away_from_center
                / np.square(safe_inactive_dist)[:, None]
            )
            if self.negative_evidence_confidence_enabled:
                confidence = self._primitive_inactive_repulsion_multiplier(
                    active_idx.astype(int),
                    inactive_idx.astype(int),
                    active_weights=active_weights,
                )
                repulsion *= np.asarray(confidence, dtype=float)[:, None]
            shift[inactive_idx] += repulsion

        max_step = base_distance * float(self.primitive_force_max_step_ratio)
        if max_step > 0.0:
            shift = self._clip_point_shift(shift, max_step)
        disp += shift

    def _apply_local_obstacle_repulsion(self, disp: FloatArray) -> None:
        strength = max(float(self.k_repulse), 0.0)
        if strength <= 0.0:
            return
        if self._current_active_mask is None:
            return
        mask = np.asarray(self._current_active_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.syncmap.shape[0]:
            return

        if self._g_rows is None or self._g_cols is None:
            return
        cutoff = max(float(self.graph_r0) * float(self.inter_cluster_distance_ratio), 1e-6)
        inactive_idx = np.flatnonzero(~mask)
        if inactive_idx.size == 0:
            return
        idx_i = np.asarray(self._g_rows, dtype=int)
        idx_j = np.asarray(self._g_cols, dtype=int)
        active_edge_mask = mask[idx_i] & mask[idx_j]
        if not np.any(active_edge_mask):
            return

        inactive_pos = np.asarray(self.syncmap[inactive_idx], dtype=float)
        for node_idx in np.unique(np.concatenate([idx_i[active_edge_mask], idx_j[active_edge_mask]])).tolist():
            node = int(node_idx)
            node_pos = np.asarray(self.syncmap[node], dtype=float)
            diff = inactive_pos - node_pos[None, :]
            dist = np.linalg.norm(diff, axis=1)
            valid = (dist > 1e-9) & (dist < cutoff)
            if not np.any(valid):
                continue

            safe_dist = dist[valid]
            local_diff = diff[valid]
            local_inactive_idx = inactive_idx[valid]
            magnitude = strength * (cutoff - safe_dist) / cutoff
            unit = local_diff / safe_dist[:, None]
            pair_force = magnitude[:, None] * unit
            if self.negative_evidence_confidence_enabled:
                confidence = self._local_obstacle_repulsion_multiplier(
                    np.asarray([node], dtype=int),
                    local_inactive_idx.astype(int),
                )
                pair_force *= np.asarray(confidence, dtype=float)[:, None]
            disp[node] -= np.sum(pair_force, axis=0)
            np.add.at(disp, local_inactive_idx, pair_force)

    def _apply_active_pulse(self, disp: FloatArray) -> None:
        if self._current_pulse_center is None or self.k_pulse <= 0.0:
            return
        if self.pulse_mode in {"none", "off", "disabled"}:
            return
        diff = self.syncmap - self._current_pulse_center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        valid_mask = (dist > 1e-9).squeeze()
        if np.any(valid_mask):
            direction = diff[valid_mask] / dist[valid_mask]
            mag = self.k_pulse / (dist[valid_mask] + 1.0)
            disp[valid_mask] += direction * mag

    def _update_plasticity(self) -> None:
        if not self.plasticity_enabled:
            self.plasticity.fill(1.0)
            return
        if self.labels is None:
            return
        core_mask = self.labels >= 0
        self.plasticity[core_mask] *= self.plasticity_decay
        self.plasticity[core_mask] = np.maximum(self.plasticity[core_mask], self.plasticity_min)
        noise_mask = ~core_mask
        self.plasticity[noise_mask] = np.minimum(
            self.plasticity[noise_mask] * 1.5,
            self.plasticity_recovery,
        )

    def _project_close_graph_pairs(self) -> None:
        if not self.project_close_pairs or self._g_rows is None or self._g_cols is None:
            return
        if self._g_rows.size == 0:
            return

        target = max(float(self.graph_r0), 1e-6)
        diff = self.syncmap[self._g_cols] - self.syncmap[self._g_rows]
        dist = np.linalg.norm(diff, axis=1)
        valid_mask = dist > 1e-9
        if not np.any(valid_mask):
            return

        idx_i = self._g_rows[valid_mask]
        idx_j = self._g_cols[valid_mask]
        diff = diff[valid_mask]
        dist = dist[valid_mask]
        gap = target - dist
        close_mask = gap > 0.0
        if not np.any(close_mask):
            return

        gain = float(np.clip(self.project_close_pairs_gain, 0.0, 1.0))
        if gain <= 0.0:
            return

        direction = diff[close_mask] / dist[close_mask, None]
        correction = 0.5 * gain * gap[close_mask, None] * direction
        shift = np.zeros_like(self.syncmap)
        np.add.at(shift, idx_i[close_mask], -correction)
        np.add.at(shift, idx_j[close_mask], correction)
        self.syncmap += self._clip_update(shift)

    def _edge_stretch_budget_abs(self) -> float:
        abs_budget = max(float(self.edge_stretch_max_step), 0.0)
        if abs_budget > 0.0:
            return abs_budget
        rel_budget = max(float(self.edge_stretch_max_step_rel), 0.0)
        if rel_budget <= 0.0:
            return 0.0
        return rel_budget * max(float(self.graph_r0), 1e-6)

    def _edge_detach_distance_abs(self) -> float:
        abs_limit = max(float(self.edge_detach_distance), 0.0)
        if abs_limit > 0.0:
            return abs_limit
        rel_limit = max(float(self.edge_detach_distance_rel), 0.0)
        if rel_limit <= 0.0:
            return 0.0
        return rel_limit * max(float(self.graph_r0), 1e-6)

    def _project_graph_edge_stretch(self, previous_syncmap: FloatArray) -> None:
        if not self.edge_stretch_guard_enabled or self._g_rows is None or self._g_cols is None:
            return

        budget = self._edge_stretch_budget_abs()
        detach_limit = self._edge_detach_distance_abs()
        if budget <= 0.0 and detach_limit <= 0.0:
            return

        idx_i = np.asarray(self._g_rows, dtype=int)
        idx_j = np.asarray(self._g_cols, dtype=int)
        if idx_i.size == 0:
            return

        prev_diff = np.asarray(previous_syncmap[idx_j] - previous_syncmap[idx_i], dtype=float)
        prev_dist = np.linalg.norm(prev_diff, axis=1)
        curr_diff = np.asarray(self.syncmap[idx_j] - self.syncmap[idx_i], dtype=float)
        curr_dist = np.linalg.norm(curr_diff, axis=1)

        valid_mask = curr_dist > 1e-9
        if detach_limit > 0.0:
            valid_mask &= prev_dist < detach_limit
        if not np.any(valid_mask):
            return

        idx_i = idx_i[valid_mask]
        idx_j = idx_j[valid_mask]
        prev_dist = prev_dist[valid_mask]
        curr_diff = curr_diff[valid_mask]
        curr_dist = curr_dist[valid_mask]

        allowed = np.array(curr_dist, copy=True)
        if budget > 0.0:
            allowed = np.minimum(allowed, prev_dist + budget)
        if detach_limit > 0.0:
            allowed = np.minimum(allowed, detach_limit)

        over = curr_dist > allowed + 1e-9
        if not np.any(over):
            return

        idx_i = idx_i[over]
        idx_j = idx_j[over]
        curr_diff = curr_diff[over]
        curr_dist = curr_dist[over]
        allowed = allowed[over]
        direction = curr_diff / curr_dist[:, None]
        correction = 0.5 * (curr_dist - allowed)[:, None] * direction

        shift = np.zeros_like(self.syncmap)
        np.add.at(shift, idx_i, correction)
        np.add.at(shift, idx_j, -correction)
        self.syncmap += self._clip_update(shift)

    def _clip_update(self, update: FloatArray) -> FloatArray:
        norms = np.linalg.norm(update, axis=1, keepdims=True)
        over = (norms > self.max_step_norm).squeeze()
        if not np.any(over):
            return update
        clipped = np.array(update, copy=True)
        clipped[over] = clipped[over] / norms[over] * self.max_step_norm
        return clipped

    def _apply_spherical_constraints(self) -> None:
        if self.space_size <= 0:
            return
        dists = np.linalg.norm(self.syncmap, axis=1)
        outside = dists > self.space_size
        if np.any(outside):
            self.syncmap[outside] *= (self.space_size / dists[outside])[:, None]

    def _recenter_syncmap(self) -> None:
        if not self.center_on_origin or self.syncmap.size == 0:
            return
        strength = float(np.clip(self.centering_strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        if self.recenter_graph_support_only and self._g_rows is not None and self._g_cols is not None:
            support_idx = np.unique(np.concatenate([self._g_rows, self._g_cols])).astype(int)
            if support_idx.size > 0:
                centroid = np.mean(self.syncmap[support_idx], axis=0, keepdims=True)
                self.syncmap[support_idx] -= strength * centroid
                return
        centroid = np.mean(self.syncmap, axis=0, keepdims=True)
        self.syncmap -= strength * centroid

    def step_once(self) -> None:
        previous_syncmap = np.array(self.syncmap, copy=True)
        buffers = getattr(self, "_step_disp_buffers", None)
        if buffers is None or buffers[0].shape != self.syncmap.shape:
            buffers = (
                np.zeros_like(self.syncmap),
                np.zeros_like(self.syncmap),
                np.zeros_like(self.syncmap),
            )
            self._step_disp_buffers = buffers
        disp_micro, disp_entity, disp_macro = buffers
        disp_micro.fill(0.0)
        disp_entity.fill(0.0)
        disp_macro.fill(0.0)
        self._apply_graph_springs(disp_micro)
        self._apply_primitive_contrastive_force(disp_micro)
        self._apply_active_edge_line_repulsion(disp_micro)
        self._apply_local_obstacle_repulsion(disp_micro)
        self._apply_active_pulse(disp_micro)
        self._apply_entity_boundary_maintenance(disp_entity)
        self._apply_entity_sibling_attraction(disp_entity)
        self._apply_cluster_distance_maintenance(disp_macro)
        point_update = self._clip_update(disp_micro) + disp_entity + disp_macro
        if self._step_count % self.plasticity_update_period == 0:
            self._update_plasticity()
        if self.plasticity_enabled:
            point_update = self.adaptation_rate * self.plasticity[:, None] * point_update
        else:
            point_update = self.adaptation_rate * point_update
        self.syncmap += point_update
        self._project_graph_edge_stretch(previous_syncmap)
        self._project_close_graph_pairs()
        self._apply_spherical_constraints()
        self._recenter_syncmap()
        if not np.isfinite(self.syncmap).all():
            self.syncmap = np.nan_to_num(self.syncmap, nan=0.0, posinf=1e3, neginf=-1e3)
        self._last_total_update = np.asarray(self.syncmap - previous_syncmap, dtype=float)
        self._step_count += 1

    def inputGeneral(
        self,
        x: NDArray[np.float64],
        *,
        phase_offset: int = 0,
        reset_trajectory: bool = True,
        append_final_frame: bool = True,
    ) -> None:
        if reset_trajectory and self.trajectory is not None:
            self.trajectory.clear()
            self.trajectory_steps.clear()  # type: ignore[union-attr]
        if reset_trajectory or not hasattr(self, "_last_time_index"):
            self._last_time_index = -1

        if (
            self.trajectory is not None
            and int(phase_offset) % self.trajectory_interval == 0
            and self._last_time_index != int(phase_offset)
        ):
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(int(phase_offset))  # type: ignore[union-attr]
            self._last_time_index = int(phase_offset)

        batch = np.asarray(x, dtype=float)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        T = int(batch.shape[0])
        if T <= 0:
            return
        active_masks = batch > 0.1

        for t in range(T):
            global_t = phase_offset + t
            values_t = np.clip(np.asarray(batch[t], dtype=float).reshape(-1), 0.0, None)
            mask_t = np.asarray(active_masks[t], dtype=bool)
            self._current_active_mask = mask_t
            self._current_active_values = values_t
            if self.negative_evidence_confidence_enabled:
                self._observe_input_symbol(values_t, mask_t)
            if self._coact is not None:
                if self.weighted_stats_input:
                    self._coact.update(values_t)
                else:
                    self._coact.update(mask_t)
            # _current_pulse_center is consumed only by _apply_active_pulse,
            # which is a no-op for k_pulse <= 0; skip the per-step average then.
            if self.k_pulse > 0.0 and np.any(values_t > 1e-12):
                weight_sum = float(np.sum(values_t))
                if weight_sum > 1e-12:
                    self._current_pulse_center = np.average(self.syncmap, axis=0, weights=values_t)
                elif np.any(mask_t):
                    self._current_pulse_center = np.mean(self.syncmap[mask_t], axis=0)
                else:
                    self._current_pulse_center = None
            else:
                self._current_pulse_center = None

            if global_t % self.physics_batch_size == 0:
                for _ in range(self.physics_steps_per_batch):
                    self.step_once()
                if self.T_push > 0 and (self._step_count % self.T_push == 0):
                    self._update_graph_structure()
                if self.clustering_period > 0 and (self._step_count % self.clustering_period == 0):
                    self._cluster_embed_labels()

            completed_step = global_t + 1
            if self.trajectory is not None and (completed_step % self.trajectory_interval == 0):
                self.trajectory.append(self.syncmap.copy())
                self.trajectory_steps.append(completed_step)  # type: ignore[union-attr]
                self._last_time_index = completed_step

        final_step = phase_offset + T
        if append_final_frame and self.trajectory is not None and T > 0 and self._last_time_index != final_step:
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(final_step)  # type: ignore[union-attr]

    def input(self, x: NDArray[np.float64]) -> None:
        self.inputGeneral(x)

    def _export_graph_edges(self) -> Optional[tuple[IntArray, IntArray, FloatArray]]:
        if self._coact is None:
            return None
        clear_after_export = hasattr(self._coact, "clear_window")
        try:
            edge_strength_fn = getattr(self._coact, "edge_strength", None)
            if callable(edge_strength_fn):
                sim_raw = edge_strength_fn()
            else:
                sim_raw = self._coact.similarity()
        except Exception:
            return None
        if sim_raw is None:
            if clear_after_export:
                try:
                    self._coact.clear_window()
                except Exception:
                    pass
            return None
        sim = np.asarray(sim_raw, dtype=float)
        if sim.size == 0:
            if clear_after_export:
                try:
                    self._coact.clear_window()
                except Exception:
                    pass
            return None

        n = int(sim.shape[0])
        if n <= 1:
            if clear_after_export:
                try:
                    self._coact.clear_window()
                except Exception:
                    pass
            return None

        sim = np.asarray(sim, dtype=float)
        np.fill_diagonal(sim, 0.0)
        k = min(int(self.knn), n - 1)
        if k <= 0:
            if clear_after_export:
                try:
                    self._coact.clear_window()
                except Exception:
                    pass
            return None

        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = idx.reshape(-1)
        weights = sim[rows, cols].reshape(-1)
        keep = weights >= self.graph_theta
        rows = rows[keep]
        cols = cols[keep]
        weights = weights[keep]
        if rows.size == 0:
            if clear_after_export:
                try:
                    self._coact.clear_window()
                except Exception:
                    pass
            return None

        a = np.minimum(rows, cols)
        b = np.maximum(rows, cols)
        key = (a.astype(np.int64) * n) + b.astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        w_max = np.zeros_like(uniq, dtype=float)
        np.maximum.at(w_max, inv, weights)
        b_u = uniq % n
        a_u = uniq // n
        if clear_after_export:
            try:
                self._coact.clear_window()
            except Exception:
                pass
        return a_u.astype(int), b_u.astype(int), w_max

    def _update_graph_structure(self) -> None:
        edges = self._export_graph_edges()
        if edges is None:
            return
        stable_med = self.size0 if self.size0 > 1e-9 else 1.0
        self.set_graph_edges(edges[0], edges[1], edges[2], r0_rel=self.graph_r0_rel, nn1_median=stable_med)

    def set_graph(
        self,
        graph_rows: NDArray[np.int_],
        graph_cols: NDArray[np.int_],
        graph_w: Optional[NDArray[np.float64]] = None,
    ) -> None:
        rows = np.asarray(graph_rows, dtype=int)
        cols = np.asarray(graph_cols, dtype=int)
        if graph_w is None:
            weights = np.ones(rows.shape[0], dtype=float)
        else:
            weights = np.asarray(graph_w, dtype=float)
        self.set_graph_edges(rows, cols, weights, r0_rel=self.graph_r0_rel, nn1_median=self.size0)

    def set_graph_edges(
        self,
        rows: ArrayLike | NDArray[np.int_],
        cols: ArrayLike | NDArray[np.int_],
        weights: ArrayLike | NDArray[np.float64],
        r0_rel: float = 1.0,
        nn1_median: float | None = None,
    ) -> None:
        idx_i = np.asarray(rows, dtype=int)
        idx_j = np.asarray(cols, dtype=int)
        edge_w = np.asarray(weights, dtype=float)
        if idx_i.size == 0:
            self._g_rows = None
            self._g_cols = None
            self._g_w = None
            return
        self._g_rows = idx_i
        self._g_cols = idx_j
        self._g_w = edge_w
        if self.graph_r0_mode == "fixed":
            self.graph_r0 = max(float(self.graph_r0_fixed), 1e-6)
        else:
            median = float(nn1_median) if nn1_median is not None else 1.0
            self.graph_r0 = max(float(r0_rel) * median, 1e-6)

    def _measure_graph_connected_distance_scale(
        self,
        samples: FloatArray,
    ) -> float | None:
        if self._g_rows is None or self._g_cols is None:
            return None
        if samples.shape[0] != self.input_size:
            return None
        if self._g_rows.size == 0 or self._g_cols.size == 0:
            return None
        diff = samples[self._g_rows] - samples[self._g_cols]
        distances = np.linalg.norm(diff, axis=1)
        valid = distances[np.isfinite(distances) & (distances > 1e-9)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _measure_knn_distance_scale(
        self,
        samples: FloatArray,
    ) -> float | None:
        if samples.shape[0] < 2:
            return None
        try:
            k = max(1, int(self.dbscan_min_samples))
            nbrs = NearestNeighbors(
                n_neighbors=min(k + 1, self.input_size),
                algorithm="auto",
            ).fit(samples)
            distances, _ = nbrs.kneighbors(samples, return_distance=True)
            col = min(k, distances.shape[1] - 1)
            valid = distances[:, col]
            valid = valid[np.isfinite(valid) & (valid > 1e-9)]
            if valid.size == 0:
                return None
            return float(np.percentile(valid, 95))
        except Exception:
            return None

    def _compute_auto_dbscan_eps(
        self,
        samples: FloatArray,
    ) -> float:
        mode = str(self.dbscan_auto_eps_source).strip().lower()
        measured_scale: float | None = None
        if mode in {"graph_midpoint", "measured_midpoint", "graph_local"}:
            measured_scale = self._measure_graph_connected_distance_scale(samples)
        if mode in {"graph_midpoint", "measured_midpoint"}:
            if measured_scale is None:
                measured_scale = max(float(self.graph_r0), 1e-6)
            growth = max(float(self.inter_cluster_distance_ratio), 1.0)
            eps = measured_scale * np.sqrt(growth) * self.dbscan_auto_eps_margin
        else:
            if measured_scale is None:
                measured_scale = self._measure_knn_distance_scale(samples)
            if measured_scale is None:
                measured_scale = max(float(self.graph_r0), 1e-6)
            eps = measured_scale * self.dbscan_auto_eps_margin
        return max(float(eps), 1e-4)

    def _cluster_embed_labels(self) -> IntArray:
        samples = np.asarray(self.syncmap, dtype=float)
        if samples.size == 0:
            labels = np.array([], dtype=int)
            self.labels = labels
            self._labels_version += 1
            return labels

        if self.embed_cluster_method == "dbscan" and self.dbscan_auto_eps:
            self.dbscan_eps = self._compute_auto_dbscan_eps(samples)
            self._reinit_clustering_strategy()

        try:
            if self.embed_cluster_method == "graph_louvain":
                labels = self._graph_louvain_labels(samples=samples)
            else:
                labels = np.asarray(self.clustering_strategy.fit_predict(samples), dtype=int)
        except Exception:
            labels = np.zeros(self.input_size, dtype=int)
        self.labels = labels
        self._labels_version += 1
        return labels

    def organize(
        self,
        strategy: Optional[ClusteringStrategy] = None,
        persist: bool = False,
        **kwargs: Any,
    ) -> IntArray:
        default_eps = float(self.dbscan_eps)
        default_min_samples = int(self.dbscan_min_samples)
        if self.embed_cluster_method == "dbscan" and self.dbscan_auto_eps:
            auto_eps = self._compute_auto_dbscan_eps(np.asarray(self.syncmap, dtype=float))
            default_eps = float(auto_eps)
            if persist:
                self.dbscan_eps = float(auto_eps)
        if not persist:
            if self.dbscan_readout_eps is not None:
                default_eps = float(self.dbscan_readout_eps)
            if self.dbscan_readout_min_samples is not None:
                default_min_samples = int(self.dbscan_readout_min_samples)
        temp_strategy = strategy
        cluster_method = kwargs.get("cluster_method")
        graph_louvain_resolution = kwargs.get("graph_louvain_resolution", kwargs.get("resolution"))
        if temp_strategy is None and cluster_method is not None:
            method = str(cluster_method).lower()
            if method == "dbscan":
                temp_strategy = create_clustering_strategy(
                    "dbscan",
                    eps=float(kwargs.get("eps", default_eps)),
                    min_samples=int(kwargs.get("min_samples", default_min_samples)),
                )
            elif method == "louvain":
                temp_strategy = create_clustering_strategy(
                    "louvain",
                    distance_factor=float(kwargs.get("lv_dist", self.louvain_distance_factor)),
                    random_state=self.clustering_random_state,
                )
            elif method == "hdbscan":
                temp_strategy = create_clustering_strategy(
                    "hdbscan",
                    min_cluster_size=int(kwargs.get("min_cluster_size", default_min_samples)),
                    min_samples=int(kwargs.get("min_samples", default_min_samples)),
                )
            elif method == "graph_louvain":
                labels = self._graph_louvain_labels(
                    resolution=None if graph_louvain_resolution is None else float(graph_louvain_resolution)
                )
                if persist:
                    self.labels = labels
                    self._labels_version += 1
                return labels
        if temp_strategy is None:
            if self.embed_cluster_method == "graph_louvain":
                labels = self._graph_louvain_labels(
                    resolution=None if graph_louvain_resolution is None else float(graph_louvain_resolution)
                )
                if persist:
                    self.labels = labels
                    self._labels_version += 1
                return labels
            if self.embed_cluster_method == "dbscan":
                temp_strategy = create_clustering_strategy(
                    "dbscan",
                    eps=float(kwargs.get("eps", default_eps)),
                    min_samples=int(kwargs.get("min_samples", default_min_samples)),
                )
            else:
                temp_strategy = self.clustering_strategy

        labels = np.asarray(temp_strategy.fit_predict(self.syncmap), dtype=int)
        if persist:
            self.labels = labels
            self._labels_version += 1
            if strategy is not None:
                self.clustering_strategy = strategy
        return labels

    def update_params(self, **kwargs: Any) -> None:
        if "dbscan_auto_eps" in kwargs and "dbscan_eps" not in kwargs and not bool(kwargs["dbscan_auto_eps"]):
            self._manual_dbscan_eps = float(self.dbscan_eps)

        int_keys = {
            "dbscan_min_samples",
            "knn",
            "T_push",
            "stats_memory_steps",
        }
        int_min0_keys = {"stats_max_lag", "clustering_period"}
        bool_keys = {
            "dbscan_auto_eps",
            "graph_degree_normalize",
            "macro_boundary_repulsion_enabled",
            "entity_boundary_repulsion_enabled",
            "entity_sibling_attraction_enabled",
            "entity_sibling_boundary_attenuation_enabled",
            "project_close_pairs",
            "edge_stretch_guard_enabled",
            "graph_weight_confidence_enabled",
            "line_obstacle_repulsion_enabled",
            "primitive_contrastive_force_enabled",
            "negative_evidence_confidence_enabled",
            "center_on_origin",
            "recenter_graph_support_only",
            "cluster_exclusion_enabled",
            "cluster_exclusion_use_cluster_mass",
        }
        str_keys = {
            "graph_spring_mode",
            "graph_r0_mode",
            "pulse_mode",
            "embed_cluster_method",
            "stats_mode",
            "cluster_exclusion_distance_mode",
            "dbscan_auto_eps_source",
        }
        reinit_stats = False

        for key, value in kwargs.items():
            if key == "dbscan_eps":
                self._manual_dbscan_eps = float(value)
                self.dbscan_eps = float(value)
                continue
            if key == "dbscan_readout_eps":
                if value is None:
                    self._manual_dbscan_readout_eps = None
                    self.dbscan_readout_eps = None
                else:
                    self._manual_dbscan_readout_eps = float(value)
                    self.dbscan_readout_eps = float(value)
                continue
            if key == "dbscan_readout_min_samples":
                self.dbscan_readout_min_samples = None if value is None else max(1, int(value))
                continue
            if key == "dbscan_auto_eps_margin":
                self.dbscan_auto_eps_margin = max(float(value), 1e-6)
                continue
            if key in int_keys:
                setattr(self, key, max(1, int(value)))
                if key in {"stats_memory_steps", "stats_max_lag"}:
                    reinit_stats = True
                continue
            if key in int_min0_keys:
                setattr(self, key, max(0, int(value)))
                reinit_stats = True
                continue
            if key in bool_keys:
                setattr(self, key, bool(value))
                continue
            if key in str_keys:
                if key == "stats_mode":
                    setattr(self, key, self._canonical_stats_mode(str(value)))
                else:
                    setattr(self, key, str(value).lower())
                if key == "stats_mode":
                    reinit_stats = True
                continue
            if key == "clustering_random_state":
                self.clustering_random_state = None if value is None else int(value)
                continue
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, (float, np.floating)):
                    setattr(self, key, float(value))
                elif isinstance(current, (int, np.integer)):
                    setattr(self, key, int(value))
                else:
                    setattr(self, key, value)

        if self.graph_r0_mode == "fixed":
            self.graph_r0 = max(float(self.graph_r0_fixed), 1e-6)
        self._recalc_space_params()
        if reinit_stats:
            self._reinit_stats()
        self._reinit_clustering_strategy()

    def save_weights(self, filepath: Union[str, Path]) -> None:
        data = {
            "syncmap": self.syncmap,
            "step_count": self._step_count,
            "labels": self.labels,
            "labels_version": self._labels_version,
            "graph_rows": self._g_rows,
            "graph_cols": self._g_cols,
            "graph_weights": self._g_w,
            "graph_r0": self.graph_r0,
            "coact_state": self._coact.__dict__ if self._coact is not None else None,
        }
        with open(filepath, "wb") as handle:
            pickle.dump(data, handle)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        self.syncmap = np.asarray(data.get("syncmap", self.syncmap), dtype=float)
        self._step_count = int(data.get("step_count", self._step_count))
        labels = data.get("labels")
        self.labels = None if labels is None else np.asarray(labels, dtype=int)
        self._labels_version = int(data.get("labels_version", self._labels_version))
        self._g_rows = None if data.get("graph_rows") is None else np.asarray(data["graph_rows"], dtype=int)
        self._g_cols = None if data.get("graph_cols") is None else np.asarray(data["graph_cols"], dtype=int)
        self._g_w = None if data.get("graph_weights") is None else np.asarray(data["graph_weights"], dtype=float)
        self.graph_r0 = float(data.get("graph_r0", self.graph_r0))
        if self._coact is not None and data.get("coact_state") is not None:
            for key, value in data["coact_state"].items():
                if not hasattr(self._coact, key):
                    continue
                setattr(self._coact, key, value)

    def get_trajectory(self) -> NDArray[np.float64]:
        if not self.trajectory:
            return np.array([])
        return np.stack(self.trajectory, axis=0)

    def set_macro_feedback_labels(
        self,
        labels: IntArray | None,
        *,
        scale: float | None = None,
        enabled: bool | None = None,
        strengths: FloatArray | None = None,
    ) -> None:
        if enabled is not None:
            self.macro_boundary_repulsion_enabled = bool(enabled)
        if scale is not None:
            self.macro_boundary_scale = max(float(scale), 1.0)
        if labels is None:
            self._macro_feedback_labels = None
            self._macro_feedback_strengths = None
            return
        arr = np.asarray(labels, dtype=int).reshape(-1)
        if arr.shape[0] != self.syncmap.shape[0]:
            raise ValueError(
                f"macro feedback label shape mismatch: expected {self.syncmap.shape[0]}, got {arr.shape[0]}"
            )
        self._macro_feedback_labels = arr.copy()
        if strengths is None:
            self._macro_feedback_strengths = None
            return
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if strength_arr.shape[0] != self.syncmap.shape[0]:
            raise ValueError(
                f"macro feedback strength shape mismatch: expected {self.syncmap.shape[0]}, got {strength_arr.shape[0]}"
            )
        self._macro_feedback_strengths = np.clip(strength_arr, 0.0, 1.0).copy()

    def set_entity_feedback_labels(
        self,
        labels: IntArray | None,
        *,
        parent_labels: IntArray | None = None,
        enabled: bool | None = None,
        strengths: FloatArray | None = None,
        parent_strengths: FloatArray | None = None,
    ) -> None:
        if enabled is not None:
            self.entity_boundary_repulsion_enabled = bool(enabled)
        if labels is None:
            self._entity_feedback_labels = None
            self._entity_feedback_parent_labels = None
            self._entity_feedback_strengths = None
            self._entity_feedback_parent_strengths = None
            return
        arr = np.asarray(labels, dtype=int).reshape(-1)
        if arr.shape[0] != self.syncmap.shape[0]:
            raise ValueError(
                f"entity feedback label shape mismatch: expected {self.syncmap.shape[0]}, got {arr.shape[0]}"
            )
        self._entity_feedback_labels = arr.copy()
        if parent_labels is None:
            self._entity_feedback_parent_labels = None
        else:
            parent_arr = np.asarray(parent_labels, dtype=int).reshape(-1)
            if parent_arr.shape[0] != self.syncmap.shape[0]:
                raise ValueError(
                    "entity feedback parent label shape mismatch: "
                    f"expected {self.syncmap.shape[0]}, got {parent_arr.shape[0]}"
                )
            self._entity_feedback_parent_labels = parent_arr.copy()
        if strengths is None:
            self._entity_feedback_strengths = None
        else:
            strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
            if strength_arr.shape[0] != self.syncmap.shape[0]:
                raise ValueError(
                    f"entity feedback strength shape mismatch: expected {self.syncmap.shape[0]}, got {strength_arr.shape[0]}"
                )
            self._entity_feedback_strengths = np.clip(strength_arr, 0.0, 1.0).copy()
        if parent_strengths is None:
            self._entity_feedback_parent_strengths = None
            return
        parent_strength_arr = np.asarray(parent_strengths, dtype=float).reshape(-1)
        if parent_strength_arr.shape[0] != self.syncmap.shape[0]:
            raise ValueError(
                "entity feedback parent strength shape mismatch: "
                f"expected {self.syncmap.shape[0]}, got {parent_strength_arr.shape[0]}"
            )
        self._entity_feedback_parent_strengths = np.clip(parent_strength_arr, 0.0, 1.0).copy()

    def get_trajectory_steps(self) -> NDArray[np.int_]:
        if not self.trajectory_steps:
            return np.array([], dtype=int)
        return np.asarray(self.trajectory_steps, dtype=int)

    def get_syncmap(self) -> FloatArray:
        return np.array(self.syncmap, copy=True)

    def get_labels(self) -> Optional[IntArray]:
        return None if self.labels is None else np.array(self.labels, copy=True)

    def get_effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in self.EFFECTIVE_PARAM_KEYS:
            params[key] = getattr(self, key)
        params["trajectory_interval"] = int(self.trajectory_interval)
        params["record_trajectory"] = bool(self.trajectory is not None)
        params["stats_backend"] = (
            "transition" if isinstance(self._coact, _NSTransitionTracker) else "coactivation"
        )
        if self._coact is not None and hasattr(self._coact, "alpha"):
            params["stats_alpha"] = float(getattr(self._coact, "alpha"))
        elif self._coact is not None and hasattr(self._coact, "decay"):
            params["stats_alpha"] = float(1.0 - getattr(self._coact, "decay"))
        return params


def create_syncmap_ns(
    input_size: int,
    dimensions: int,
    adaptation_rate: float,
    **kwargs: Any,
) -> SyncMapNS:
    return SyncMapNS(input_size, dimensions, adaptation_rate=adaptation_rate, **kwargs)
