#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from tools.clustering_strategies import ClusteringStrategy, create_clustering_strategy
from tools.density_scale import DEFAULT_PACKING_FACTOR, compute_density_scale
from syncmap.SyncMapCI import build_asymmetric_node_features


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class SyncMapX_CI:
    """Prototype CI variant based on directed transition statistics.

    The model keeps a soft directed transition matrix between observed nodes,
    converts it into asymmetric node features, and repeatedly embeds those
    features into a low-dimensional space that can be clustered by the existing
    framework.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "adaptation_rate": 0.2,
        "transition_alpha": 0.0,
        "obs_threshold": 1e-8,
        "decode_mode": "soft",
        "feature_include_marginals": True,
        "base_wall": 100.0,
        "density_ref": 64.0,
        "packing_factor": DEFAULT_PACKING_FACTOR,
        "space_size": None,
        "reembed_period": None,
        "cluster_period": None,
        "min_transition_probability": 0.0,
        "min_transition_count": 1.0,
        "embed_cluster_method": "louvain",
        "significant_distance_factor": 1.0,
        "louvain_distance_factor": 1.0,
        "louvain_knn_k": None,
        "louvain_mutual_knn": True,
        "dbscan_eps": None,
        "dbscan_auto_eps": True,
        "dbscan_min_samples": 2,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        "input_size",
        "dimensions",
        "adaptation_rate",
        "transition_alpha",
        "obs_threshold",
        "decode_mode",
        "feature_include_marginals",
        "base_wall",
        "density_ref",
        "packing_factor",
        "space_size",
        "reembed_period",
        "cluster_period",
        "min_transition_probability",
        "min_transition_count",
        "embed_cluster_method",
        "significant_distance_factor",
        "louvain_distance_factor",
        "louvain_knn_k",
        "louvain_mutual_knn",
        "dbscan_eps",
        "dbscan_auto_eps",
        "dbscan_min_samples",
        "reference_local_scale",
        "effective_local_scale",
        "significant_distance",
        "significant_k",
        "scale_factor",
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

    @staticmethod
    def _default_reembed_period(input_size: int) -> int:
        if input_size <= 64:
            return 1
        if input_size <= 160:
            return 5
        if input_size <= 320:
            return 10
        return 20

    def _default_louvain_knn_k(self) -> int:
        if self.embed_cluster_method != "louvain" or self.input_size < 128:
            return 0
        return max(4, min(12, 2 * self.dimensions))

    def _normalize_periods(self) -> None:
        self.reembed_period = max(1, int(self.reembed_period))
        self.cluster_period = max(int(self.reembed_period), int(self.cluster_period))

    def __init__(
        self,
        input_size: int,
        dimensions: int = 3,
        adaptation_rate: float = 0.2,
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        self.input_size = int(input_size)
        self.dimensions = int(dimensions)
        self.adaptation_rate = float(strategy_kwargs.get("adaptation_rate", adaptation_rate))
        self.transition_alpha = float(strategy_kwargs.get("transition_alpha", 0.0))
        self.obs_threshold = float(strategy_kwargs.get("obs_threshold", 1e-8))
        self.decode_mode = str(strategy_kwargs.get("decode_mode", "soft")).lower()
        self.feature_include_marginals = bool(strategy_kwargs.get("feature_include_marginals", True))
        self.base_wall = float(strategy_kwargs.get("base_wall", 100.0))
        self.density_ref = float(strategy_kwargs.get("density_ref", 64.0))
        self.packing_factor = float(strategy_kwargs.get("packing_factor", DEFAULT_PACKING_FACTOR))
        manual_space_size = strategy_kwargs.get("space_size")
        self._manual_space_size = (
            float(manual_space_size) if manual_space_size is not None and float(manual_space_size) > 0.0 else None
        )
        self.space_size = float(self._manual_space_size) if self._manual_space_size is not None else 0.0
        reembed_period = strategy_kwargs.get("reembed_period")
        self.reembed_period = (
            max(1, int(reembed_period))
            if reembed_period is not None
            else self._default_reembed_period(self.input_size)
        )
        cluster_period = strategy_kwargs.get("cluster_period")
        self.cluster_period = self.reembed_period if cluster_period is None else max(1, int(cluster_period))
        self._normalize_periods()
        self.min_transition_probability = float(strategy_kwargs.get("min_transition_probability", 0.0))
        self.min_transition_count = float(strategy_kwargs.get("min_transition_count", 1.0))

        self.embed_cluster_method = str(strategy_kwargs.get("embed_cluster_method", "louvain")).lower()
        sig_factor = strategy_kwargs.get("significant_distance_factor", strategy_kwargs.get("louvain_distance_factor", 1.0))
        self.significant_distance_factor = float(sig_factor)
        self.louvain_distance_factor = float(sig_factor)
        louvain_knn_k = strategy_kwargs.get("louvain_knn_k")
        self.louvain_knn_k = int(louvain_knn_k) if louvain_knn_k is not None else self._default_louvain_knn_k()
        self.louvain_mutual_knn = bool(strategy_kwargs.get("louvain_mutual_knn", True))
        manual_dbscan_eps = strategy_kwargs.get("dbscan_eps")
        self._manual_dbscan_eps = float(manual_dbscan_eps) if manual_dbscan_eps is not None else None
        self.dbscan_eps = float(self._manual_dbscan_eps) if self._manual_dbscan_eps is not None else 0.0
        self.dbscan_auto_eps = bool(strategy_kwargs.get("dbscan_auto_eps", True))
        self.dbscan_min_samples = int(strategy_kwargs.get("dbscan_min_samples", 2))
        self.reference_local_scale = 0.0
        self.effective_local_scale = 0.0
        self.significant_distance = 0.0
        self.significant_k = 0
        self.scale_factor = 1.0
        self._recalc_density_params()

        self.transition_counts: FloatArray = np.zeros((self.input_size, self.input_size), dtype=float)
        self._prev_weights: Optional[FloatArray] = None
        self._step_count = 0
        self.syncmap: FloatArray = np.zeros((self.input_size, self.dimensions), dtype=float)
        self.labels: Optional[IntArray] = None
        self._embed_version = 0
        self._labels_version = -1

        self.trajectory_interval = max(1, int(trajectory_interval))
        if record_trajectory:
            self.trajectory: Optional[list[FloatArray]] = []
            self.trajectory_steps: Optional[list[int]] = []
        else:
            self.trajectory = None
            self.trajectory_steps = None

        self._reinit_clustering_strategy()
        self.organize(persist=True)

    def _reinit_clustering_strategy(self) -> None:
        if self.embed_cluster_method == "louvain":
            self.clustering_strategy = create_clustering_strategy(
                "louvain",
                distance_factor=self.louvain_distance_factor,
                distance_threshold=float(self.significant_distance),
                knn_k=int(self.louvain_knn_k),
                mutual_knn=bool(self.louvain_mutual_knn),
            )
            return
        self.clustering_strategy = create_clustering_strategy(
            "dbscan",
            eps=float(self.dbscan_eps),
            min_samples=int(self.dbscan_min_samples),
        )

    def _recalc_density_params(self) -> None:
        scale = compute_density_scale(
            input_size=self.input_size,
            dimensions=self.dimensions,
            base_wall=self.base_wall,
            density_ref=self.density_ref,
            significant_distance_factor=self.significant_distance_factor,
            packing_factor=self.packing_factor,
            explicit_space_size=self._manual_space_size,
        )
        self.reference_local_scale = float(scale.reference_local_scale)
        self.effective_local_scale = float(scale.effective_local_scale)
        self.space_size = float(scale.space_size)
        self.significant_distance = float(scale.significant_distance)
        self.significant_k = int(scale.significant_k)
        self.scale_factor = float(scale.scale_factor)
        if self.dbscan_auto_eps or self._manual_dbscan_eps is None:
            self.dbscan_eps = float(self.significant_distance)
        else:
            self.dbscan_eps = float(self._manual_dbscan_eps)

    def _weights_from_observation(self, x_t: FloatArray) -> FloatArray:
        x = np.asarray(x_t, dtype=float).reshape(-1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        x[x < self.obs_threshold] = 0.0

        if self.decode_mode == "argmax":
            weights = np.zeros_like(x)
            if np.any(x > 0.0):
                weights[int(np.argmax(x))] = 1.0
            return weights

        total = float(x.sum())
        if total <= 1e-12:
            return np.zeros_like(x)
        return x / total

    def _transition_probabilities(self) -> FloatArray:
        counts = np.asarray(self.transition_counts, dtype=float)
        row_sums = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    def get_estimated_adjacency(self) -> IntArray:
        probs = self._transition_probabilities()
        support = (self.transition_counts >= self.min_transition_count) & (probs >= self.min_transition_probability)
        np.fill_diagonal(support, False)
        return support.astype(int)

    def _compute_embedding(self) -> FloatArray:
        features = build_asymmetric_node_features(
            self.transition_counts,
            include_marginals=self.feature_include_marginals,
        )
        if features.size == 0:
            return np.zeros((self.input_size, self.dimensions), dtype=float)

        features = np.log1p(np.maximum(features, 0.0))
        features = features - np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        features = np.divide(features, np.maximum(std, 1e-6), out=np.zeros_like(features), where=std > 0)

        if not np.any(np.abs(features) > 1e-9):
            return np.zeros((self.input_size, self.dimensions), dtype=float)

        try:
            _, _, vt = np.linalg.svd(features, full_matrices=False)
            k = max(1, min(self.dimensions, vt.shape[0]))
            coords = features @ vt[:k].T
        except np.linalg.LinAlgError:
            coords = features[:, : min(self.dimensions, features.shape[1])]

        if coords.shape[1] < self.dimensions:
            coords = np.pad(coords, ((0, 0), (0, self.dimensions - coords.shape[1])))

        coords = np.asarray(coords[:, : self.dimensions], dtype=float)
        max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
        if max_abs > 1e-9:
            coords = coords / max_abs * (self.space_size * 0.5)
        return self._align_embedding_to_previous(coords)

    def _align_embedding_to_previous(self, coords: FloatArray) -> FloatArray:
        prev = np.asarray(self.syncmap, dtype=float)
        if prev.shape != coords.shape or not np.any(np.abs(prev) > 1e-9):
            return coords

        prev_center = np.mean(prev, axis=0, keepdims=True)
        coords_center = np.mean(coords, axis=0, keepdims=True)
        prev_zero = prev - prev_center
        coords_zero = coords - coords_center

        try:
            cross = coords_zero.T @ prev_zero
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rot = u @ vt
            return (coords_zero @ rot) + prev_center
        except np.linalg.LinAlgError:
            return coords

    def _refresh_embedding(self, *, should_cluster: bool = True) -> None:
        coords = self._compute_embedding()
        blend = float(np.clip(self.adaptation_rate, 0.0, 1.0))
        self.syncmap = (1.0 - blend) * self.syncmap + blend * coords
        self._embed_version += 1
        if should_cluster:
            self._cluster_embed_labels()

    def _cluster_embed_labels(self) -> IntArray:
        if self.syncmap.size == 0:
            self.labels = np.zeros(self.input_size, dtype=int)
            self._labels_version = self._embed_version
            return self.labels

        try:
            self.labels = np.asarray(self.clustering_strategy.fit_predict(self.syncmap), dtype=int)
        except Exception:
            self.labels = np.zeros(self.input_size, dtype=int)
        self._labels_version = self._embed_version
        return self.labels

    def inputGeneral(
        self,
        x: FloatArray,
        *,
        phase_offset: int = 0,
        reset_trajectory: bool = True,
    ) -> None:
        if reset_trajectory and self.trajectory is not None:
            self.trajectory.clear()
            self.trajectory_steps.clear()  # type: ignore[union-attr]
        if reset_trajectory or not hasattr(self, "_last_time_index"):
            self._last_time_index = -1

        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        T = int(X.shape[0])
        if T <= 0:
            return

        for t in range(T):
            global_t = phase_offset + t
            weights = self._weights_from_observation(X[t])

            if self.transition_alpha > 0.0:
                self.transition_counts *= (1.0 - self.transition_alpha)

            if self._prev_weights is not None and np.any(weights > 0.0):
                self.transition_counts += np.outer(self._prev_weights, weights)
            self._prev_weights = weights

            if global_t % self.reembed_period == 0:
                self._refresh_embedding(should_cluster=(global_t % self.cluster_period == 0))

            if self.trajectory is not None and (global_t % self.trajectory_interval == 0):
                self.trajectory.append(self.syncmap.copy())
                self.trajectory_steps.append(global_t)  # type: ignore[union-attr]
                self._last_time_index = global_t

            self._step_count += 1

        final_t = phase_offset + T - 1
        if self.trajectory is not None and self._last_time_index != final_t:
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(final_t)  # type: ignore[union-attr]

    def input(self, x: FloatArray) -> None:
        self.inputGeneral(x)

    def organize(self, strategy: ClusteringStrategy | None = None, persist: bool = False, **kwargs: Any) -> IntArray:
        if strategy is None:
            strategy = self.clustering_strategy
        try:
            labels = np.asarray(strategy.fit_predict(self.syncmap), dtype=int)
        except Exception:
            labels = np.zeros(self.input_size, dtype=int)
        if persist:
            self.labels = labels
            self._labels_version = self._embed_version
        return labels

    def update_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k == "dbscan_eps":
                self._manual_dbscan_eps = float(v)
                self.dbscan_eps = float(v)
                continue
            if k == "dbscan_auto_eps":
                next_auto = bool(v)
                if not next_auto and self._manual_dbscan_eps is None:
                    self._manual_dbscan_eps = float(self.dbscan_eps)
                if next_auto:
                    self._manual_dbscan_eps = None
                self.dbscan_auto_eps = next_auto
                continue
            if k == "space_size":
                value = float(v)
                if value > 0.0:
                    self._manual_space_size = value
                    self.space_size = value
                else:
                    self._manual_space_size = None
                continue
            if k in {"significant_distance_factor", "louvain_distance_factor"}:
                value = float(v)
                self.significant_distance_factor = value
                self.louvain_distance_factor = value
                continue
            if k in {"dbscan_min_samples", "reembed_period", "cluster_period"}:
                setattr(self, k, max(1, int(v)))
                continue
            if k == "louvain_knn_k":
                self.louvain_knn_k = max(0, int(v))
                continue
            if k in {"louvain_mutual_knn", "feature_include_marginals"}:
                setattr(self, k, bool(v))
                continue
            if k == "embed_cluster_method":
                self.embed_cluster_method = str(v).lower()
                continue
            if hasattr(self, k):
                try:
                    setattr(self, k, float(v))
                except Exception:
                    setattr(self, k, v)
        self._normalize_periods()
        self._recalc_density_params()
        self._reinit_clustering_strategy()
        self._refresh_embedding()

    def save_weights(self, filepath: Union[str, Path]) -> None:
        data = {
            "transition_counts": self.transition_counts,
            "syncmap": self.syncmap,
            "labels": self.labels,
            "prev_weights": self._prev_weights,
            "step_count": self._step_count,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.transition_counts = np.asarray(data.get("transition_counts", self.transition_counts), dtype=float)
        self.syncmap = np.asarray(data.get("syncmap", self.syncmap), dtype=float)
        self.labels = data.get("labels")
        prev = data.get("prev_weights")
        self._prev_weights = None if prev is None else np.asarray(prev, dtype=float)
        self._step_count = int(data.get("step_count", 0))

    def get_trajectory(self) -> FloatArray:
        if not self.trajectory:
            return np.array([])
        return np.stack(self.trajectory, axis=0)

    def get_trajectory_steps(self) -> IntArray:
        if not self.trajectory_steps:
            return np.array([], dtype=int)
        return np.asarray(self.trajectory_steps, dtype=int)

    def get_effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in self.EFFECTIVE_PARAM_KEYS:
            params[key] = getattr(self, key)
        params["trajectory_interval"] = int(self.trajectory_interval)
        params["record_trajectory"] = bool(self.trajectory is not None)
        return params


def create_syncmap_ci(input_size: int, dimensions: int, adaptation_rate: float, **kwargs: Any) -> SyncMapX_CI:
    return SyncMapX_CI(input_size, dimensions, adaptation_rate, **kwargs)
