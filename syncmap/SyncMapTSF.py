#!/usr/bin/env python3
"""
Independent TSFMap-style dynamics with a repository-compatible API.

This implementation follows the core paper mechanics:
- positive set / negative set split from the current decayed input
- centroid-based velocity update with Hebbian / anti-Hebbian terms
- fixed-size normalization after each update step
- hierarchy readout from agglomerative clustering of the learned space

It intentionally stays separate from RG/NS so we can compare kernels fairly
on the same environments and CLI surface.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _relabel_cluster_ids(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    relabeled = np.full(arr.shape[0], -1, dtype=int)
    valid = sorted({int(v) for v in arr.tolist() if int(v) >= 0})
    mapping = {label: idx for idx, label in enumerate(valid)}
    for idx, label in enumerate(arr.tolist()):
        if int(label) >= 0:
            relabeled[idx] = mapping[int(label)]
    return relabeled


class SyncMapTSF:
    DEFAULT_PARAMS: dict[str, Any] = {
        "space_size": 1.0,
        "activation_threshold": 0.1,
        "adaptation_rate": 1e-3,
        "velocity_decay": 0.999,
        "mu_pos": 6.0,
        "mu_neg": 3.0,
        "mu_repulse": 2.0,
        "normalize_each_step": True,
        "normalize_mode": "max_abs",
        "embed_cluster_method": "agglomerative",
        "cluster_linkage": "ward",
        "organize_cluster_count": None,
        "organize_distance_factor": 1.0,
        "target_levels": 3,
        "min_clusters": 2,
        "max_clusters": 16,
        "organize_level": "cluster_level1",
        "clustering_random_state": None,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        "input_size",
        "dimensions",
        "space_size",
        "activation_threshold",
        "adaptation_rate",
        "velocity_decay",
        "mu_pos",
        "mu_neg",
        "mu_repulse",
        "normalize_each_step",
        "normalize_mode",
        "embed_cluster_method",
        "cluster_linkage",
        "organize_cluster_count",
        "organize_distance_factor",
        "target_levels",
        "min_clusters",
        "max_clusters",
        "organize_level",
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
        adaptation_rate: float = 1e-3,
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        self.input_size = int(input_size)
        self.dimensions = int(dimensions)
        self.space_size = max(float(strategy_kwargs.get("space_size", 1.0)), 1e-6)
        self.activation_threshold = float(strategy_kwargs.get("activation_threshold", 0.1))
        self.adaptation_rate = float(strategy_kwargs.get("adaptation_rate", adaptation_rate))
        self.velocity_decay = float(strategy_kwargs.get("velocity_decay", 0.999))
        self.mu_pos = float(strategy_kwargs.get("mu_pos", 6.0))
        self.mu_neg = float(strategy_kwargs.get("mu_neg", 3.0))
        self.mu_repulse = float(strategy_kwargs.get("mu_repulse", 2.0))
        self.normalize_each_step = bool(strategy_kwargs.get("normalize_each_step", True))
        self.normalize_mode = str(strategy_kwargs.get("normalize_mode", "max_abs")).lower()
        self.embed_cluster_method = str(strategy_kwargs.get("embed_cluster_method", "agglomerative")).lower()
        self.cluster_linkage = str(strategy_kwargs.get("cluster_linkage", "ward")).lower()
        self.organize_cluster_count = strategy_kwargs.get("organize_cluster_count", None)
        self.organize_distance_factor = float(strategy_kwargs.get("organize_distance_factor", 1.0))
        self.target_levels = max(1, int(strategy_kwargs.get("target_levels", 3)))
        self.min_clusters = max(2, int(strategy_kwargs.get("min_clusters", 2)))
        self.max_clusters = max(self.min_clusters, int(strategy_kwargs.get("max_clusters", 16)))
        self.organize_level = str(strategy_kwargs.get("organize_level", "cluster_level1"))
        clustering_random_state = strategy_kwargs.get("clustering_random_state", None)
        self.clustering_random_state = None if clustering_random_state is None else int(clustering_random_state)

        self._step_count = 0
        self._labels_version = 0
        self._last_time_index = -1

        self.syncmap: FloatArray = (
            (np.random.rand(self.input_size, self.dimensions) * 2.0 - 1.0) * self.space_size
        )
        self.velocity: FloatArray = np.zeros((self.input_size, self.dimensions), dtype=float)
        self.labels: Optional[IntArray] = None
        self._last_hierarchy_outputs: dict[str, IntArray] | None = None

        self.trajectory_interval = max(1, int(trajectory_interval))
        self.trajectory: Optional[list[FloatArray]] = [] if record_trajectory else None
        self.trajectory_steps: Optional[list[int]] = [] if record_trajectory else None

        self.organize(persist=True)

    @property
    def graph(self) -> None:
        return None

    def _normalize_positions(self) -> None:
        if not self.normalize_each_step or self.syncmap.size == 0:
            return
        if self.normalize_mode == "max_abs":
            denom = float(np.max(np.abs(self.syncmap)))
            if denom > 1e-12:
                self.syncmap = (self.space_size / denom) * self.syncmap
            return
        radii = np.linalg.norm(self.syncmap, axis=1)
        max_radius = float(np.max(radii)) if radii.size else 0.0
        if max_radius > 1e-12:
            self.syncmap = (self.space_size / max_radius) * self.syncmap

    def _step_input(self, x_t: FloatArray) -> None:
        active = np.asarray(x_t >= self.activation_threshold, dtype=bool)
        inactive = ~active
        if int(np.sum(active)) <= 1 or int(np.sum(inactive)) <= 1:
            return

        cp = np.mean(self.syncmap[active], axis=0)
        cn = np.mean(self.syncmap[inactive], axis=0)

        d_cp = np.linalg.norm(self.syncmap - cp[None, :], axis=1)
        d_cn = np.linalg.norm(self.syncmap - cn[None, :], axis=1)
        d_cp = np.maximum(d_cp, 1e-9)
        d_cn = np.maximum(d_cn, 1e-9)

        update = np.zeros_like(self.syncmap)
        update[active] += self.mu_pos * (cp[None, :] - self.syncmap[active]) / d_cp[active, None]
        update[inactive] += self.mu_neg * (self.syncmap[inactive] - cn[None, :]) / d_cn[inactive, None]
        update[inactive] += self.mu_repulse * (self.syncmap[inactive] - cp[None, :]) / (d_cp[inactive, None] ** 2)

        self.velocity = (self.velocity_decay * self.velocity) + update
        self.syncmap += self.adaptation_rate * self.velocity
        self._normalize_positions()
        if not np.isfinite(self.syncmap).all():
            self.syncmap = np.nan_to_num(self.syncmap, nan=0.0, posinf=self.space_size, neginf=-self.space_size)
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
            self._last_time_index = -1

        batch = np.asarray(x, dtype=float)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.size == 0:
            return

        if (
            self.trajectory is not None
            and int(phase_offset) % self.trajectory_interval == 0
            and self._last_time_index != int(phase_offset)
        ):
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(int(phase_offset))  # type: ignore[union-attr]
            self._last_time_index = int(phase_offset)

        total_steps = int(batch.shape[0])
        for local_step in range(total_steps):
            self._step_input(np.asarray(batch[local_step], dtype=float))
            completed_step = int(phase_offset + local_step + 1)
            if self.trajectory is not None and completed_step % self.trajectory_interval == 0:
                self.trajectory.append(self.syncmap.copy())
                self.trajectory_steps.append(completed_step)  # type: ignore[union-attr]
                self._last_time_index = completed_step

        final_step = int(phase_offset + total_steps)
        if append_final_frame and self.trajectory is not None and self._last_time_index != final_step:
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(final_step)  # type: ignore[union-attr]
            self._last_time_index = final_step

    def input(self, x: NDArray[np.float64]) -> None:
        self.inputGeneral(x)

    def _build_linkage(self) -> Optional[np.ndarray]:
        samples = np.asarray(self.syncmap, dtype=float)
        if samples.shape[0] <= 1:
            return None
        if np.allclose(samples, samples[[0]]):
            return None
        method = self.cluster_linkage
        if method not in {"ward", "average", "complete", "single"}:
            method = "ward"
        # optimal_ordering=True would be O(N^3); False is sufficient for cluster assignments
        return linkage(samples, method=method, optimal_ordering=False)

    def _geometric_fallback_counts(self, max_count: int, level_count: int) -> list[int]:
        if level_count <= 1:
            return [max(self.min_clusters, min(max_count, self.input_size))]
        raw = np.geomspace(max_count, self.min_clusters, num=level_count)
        counts = [int(np.clip(round(value), self.min_clusters, max_count)) for value in raw.tolist()]
        ordered: list[int] = []
        seen: set[int] = set()
        for count in counts:
            if count not in seen:
                seen.add(count)
                ordered.append(count)
        if self.min_clusters not in seen:
            ordered.append(self.min_clusters)
        return sorted(ordered, reverse=True)

    def _hierarchy_cluster_counts(self, desired_levels: int) -> list[int]:
        level_count = max(1, int(desired_levels))
        max_count = min(self.max_clusters, self.input_size)
        if max_count < self.min_clusters:
            return [max(1, max_count)]
        tree = self._build_linkage()
        if tree is None or tree.shape[0] <= 1:
            return self._geometric_fallback_counts(max_count, level_count)

        distances = np.asarray(tree[:, 2], dtype=float)
        jumps = np.diff(distances)
        ranked = np.argsort(-jumps)
        counts: list[int] = []
        for jump_index in ranked.tolist():
            count = int(self.input_size - (jump_index + 1))
            if count < self.min_clusters or count > max_count:
                continue
            if count not in counts:
                counts.append(count)
            if len(counts) >= level_count:
                break
        if len(counts) < level_count:
            for fallback in self._geometric_fallback_counts(max_count, level_count):
                if fallback not in counts:
                    counts.append(fallback)
                if len(counts) >= level_count:
                    break
        return sorted(counts[:level_count], reverse=True)

    def _labels_for_cluster_count(self, n_clusters: int) -> IntArray:
        count = int(np.clip(n_clusters, 1, self.input_size))
        if count <= 1:
            return np.zeros(self.input_size, dtype=int)
        if count >= self.input_size:
            return np.arange(self.input_size, dtype=int)
        tree = self._build_linkage()
        if tree is None:
            if count <= 1:
                return np.zeros(self.input_size, dtype=int)
            return np.arange(self.input_size, dtype=int)
        labels = fcluster(tree, t=count, criterion="maxclust") - 1
        return np.asarray(_relabel_cluster_ids(labels), dtype=int)

    def _auto_flat_labels(self) -> IntArray:
        if self.organize_cluster_count is not None:
            return self._labels_for_cluster_count(int(self.organize_cluster_count))
        counts = self._hierarchy_cluster_counts(max(1, self.target_levels))
        if not counts:
            return np.zeros(self.input_size, dtype=int)
        return self._labels_for_cluster_count(int(counts[0]))

    def organize_hierarchy(self, **kwargs: Any) -> dict[str, IntArray]:
        target_levels = max(1, int(kwargs.get("target_levels", self.target_levels)))
        counts = self._hierarchy_cluster_counts(target_levels)
        outputs: dict[str, IntArray] = {}
        for index, count in enumerate(counts, start=1):
            outputs[f"cluster_level{index}"] = self._labels_for_cluster_count(int(count))
        if "cluster_level1" not in outputs:
            outputs["cluster_level1"] = self._auto_flat_labels()
        self._last_hierarchy_outputs = {key: np.asarray(value, dtype=int) for key, value in outputs.items()}
        return self._last_hierarchy_outputs

    def organize(
        self,
        level: str | None = None,
        persist: bool = False,
        **kwargs: Any,
    ) -> IntArray:
        requested_level = str(level or kwargs.get("level") or self.organize_level or "cluster_level1")
        if requested_level.startswith("level"):
            try:
                level_index = int(requested_level.replace("level", ""))
                requested_level = f"cluster_level{level_index + 1}"
            except Exception:
                requested_level = "cluster_level1"
        if requested_level.startswith("cluster_level"):
            outputs = self.organize_hierarchy(**kwargs)
            labels = np.asarray(outputs.get(requested_level, outputs["cluster_level1"]), dtype=int)
        elif self.embed_cluster_method == "agglomerative":
            labels = self._auto_flat_labels()
        else:
            samples = np.asarray(self.syncmap, dtype=float)
            if samples.shape[0] <= 1:
                labels = np.zeros(samples.shape[0], dtype=int)
            else:
                threshold = float(np.median(pdist(samples))) * float(self.organize_distance_factor)
                threshold = max(threshold, 1e-6)
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    linkage="ward" if self.cluster_linkage == "ward" else "average",
                )
                labels = np.asarray(clusterer.fit_predict(samples), dtype=int)
                labels = np.asarray(_relabel_cluster_ids(labels), dtype=int)
        if persist:
            self.labels = np.asarray(labels, dtype=int)
            self._labels_version += 1
        return np.asarray(labels, dtype=int)

    def update_params(self, **kwargs: Any) -> None:
        int_keys = {"target_levels", "min_clusters", "max_clusters"}
        bool_keys = {"normalize_each_step"}
        str_keys = {"normalize_mode", "embed_cluster_method", "cluster_linkage", "organize_level"}
        for key, value in kwargs.items():
            if key in int_keys:
                setattr(self, key, int(value))
                continue
            if key in bool_keys:
                setattr(self, key, bool(value))
                continue
            if key in str_keys:
                setattr(self, key, str(value))
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
        self.min_clusters = max(2, int(self.min_clusters))
        self.max_clusters = max(self.min_clusters, int(self.max_clusters))
        self.target_levels = max(1, int(self.target_levels))

    def save_weights(self, filepath: Union[str, Path]) -> None:
        data = {
            "syncmap": self.syncmap,
            "velocity": self.velocity,
            "step_count": self._step_count,
            "labels": self.labels,
            "labels_version": self._labels_version,
        }
        with open(filepath, "wb") as handle:
            pickle.dump(data, handle)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        self.syncmap = np.asarray(data.get("syncmap", self.syncmap), dtype=float)
        self.velocity = np.asarray(data.get("velocity", self.velocity), dtype=float)
        self._step_count = int(data.get("step_count", self._step_count))
        labels = data.get("labels")
        self.labels = None if labels is None else np.asarray(labels, dtype=int)
        self._labels_version = int(data.get("labels_version", self._labels_version))

    def get_trajectory(self) -> NDArray[np.float64]:
        if not self.trajectory:
            return np.array([])
        return np.stack(self.trajectory, axis=0)

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
        return params


def create_syncmap_tsf(
    input_size: int,
    dimensions: int,
    adaptation_rate: float,
    **kwargs: Any,
) -> SyncMapTSF:
    return SyncMapTSF(input_size, dimensions, adaptation_rate=adaptation_rate, **kwargs)
