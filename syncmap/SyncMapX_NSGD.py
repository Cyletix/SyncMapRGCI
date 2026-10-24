#!/usr/bin/env python3
"""
NSGD baseline implementation (first-order SyncMapX dynamics).
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, ArrayLike

if TYPE_CHECKING:
    from sklearn.neighbors import NearestNeighbors
else:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        class NearestNeighbors:
            def __init__(self, n_neighbors: int, algorithm: str) -> None: ...
            def fit(self, X: NDArray[np.float64]) -> None: ...
            def kneighbors(self, return_distance: bool) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
                return (np.array([[1.0]]), np.array([[0]]))

from tools.clustering_strategies import ClusteringStrategy, create_clustering_strategy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class _LegacyCoactivationStats:
    """Coactivation statistics used by the NSGD baseline."""

    def __init__(self, n: int, alpha: float = 0.01, max_lag: int = 0) -> None:
        self.n = int(n)
        self.alpha = float(alpha)
        self.max_lag = max(0, int(max_lag))
        self.C: NDArray[np.float64] = np.zeros((self.n, self.n), dtype=np.float64)
        self.cnt: NDArray[np.float64] = np.zeros(self.n, dtype=np.float64)
        self._lags: deque[NDArray[np.intp]] = deque(maxlen=self.max_lag)

    def reset(self) -> None:
        self.C.fill(0.0)
        self.cnt.fill(0.0)
        self._lags.clear()

    def _decay(self) -> None:
        if self.alpha <= 0.0:
            return
        decay = 1.0 - self.alpha
        self.C *= decay
        self.cnt *= decay

    def update(self, active_indices: Iterable[int] | NDArray[np.bool_] | NDArray[np.intp]) -> None:
        self._decay()
        idx: NDArray[np.intp]
        if isinstance(active_indices, np.ndarray) and active_indices.dtype == np.bool_:
            idx = np.where(active_indices)[0].astype(np.intp, copy=False)
        else:
            idx = np.asarray(list(active_indices), dtype=np.intp)
        if idx.size == 0:
            if self.max_lag > 0:
                self._lags.append(idx)
            return

        self.cnt[idx] += 1.0
        for i in idx:
            self.C[i, idx] += 1.0
            self.C[idx, i] += 1.0

        if self.max_lag > 0 and len(self._lags) > 0:
            for past in self._lags:
                if past.size == 0:
                    continue
                for i in idx:
                    self.C[i, past] += 1.0
                    self.C[past, i] += 1.0
            self._lags.append(idx)
        elif self.max_lag > 0:
            self._lags.append(idx)

    def similarity(self) -> NDArray[np.float64]:
        denom = np.sqrt(np.outer(self.cnt, self.cnt)) + 1e-12
        sim = np.divide(self.C, denom, out=np.zeros_like(self.C), where=denom > 0)
        np.fill_diagonal(sim, 0.0)
        return np.clip(sim, 0.0, 1.0)


class SyncMapX_NSGD:
    DEFAULT_PARAMS: dict[str, Any] = {
        'base_wall': 100.0,
        'density_ref': 64.0,
        'space_size': -1.0,
        'adaptation_rate': 0.05,
        'max_step_norm': 5.0,
        'k_repulse': 0.0,
        'k_pulse': 2.0,
        'k_attr': 1.0,
        'plasticity_decay': 0.99,
        'plasticity_min': 0.01,
        'plasticity_recovery': 1.0,
        'stats_alpha': 0.01,
        'stats_max_lag': 0,
        'T_push': 10,
        'knn': 12,
        'graph_theta': 0.1,
        'graph_r0_rel': 0.1,
        'dbscan_min_samples': 3,
        'dbscan_auto_eps': False,
        'embed_cluster_method': 'dbscan',
        'clustering_period': 100,
        'louvain_distance_factor': 0.15,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        'input_size',
        'dimensions',
        'base_wall',
        'density_ref',
        'space_size',
        'scale',
        'adaptation_rate',
        'max_step_norm',
        'k_repulse',
        'k_pulse',
        'k_attr',
        'plasticity_decay',
        'plasticity_min',
        'plasticity_recovery',
        'stats_memory_steps',
        'T_push',
        'knn',
        'graph_theta',
        'graph_r0_rel',
        'graph_r0',
        'dbscan_eps',
        'dbscan_min_samples',
        'dbscan_auto_eps',
        'embed_cluster_method',
        'clustering_period',
        'louvain_distance_factor',
    )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return dict(cls.DEFAULT_PARAMS)

    @classmethod
    def normalize_params(cls, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = cls.default_params()
        if params:
            merged.update(dict(params))
            if 'stats_alpha' in params and 'stats_memory_steps' not in params:
                merged.pop('stats_memory_steps', None)
        return merged

    def __init__(
        self,
        input_size: int,
        dimensions: int = 3,
        adaptation_rate: float = 1,
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        **strategy_kwargs: Any,
    ) -> None:

        self.input_size = int(input_size)
        self.dimensions = int(dimensions)
        self._step_count: int = 0

        self.base_wall = float(strategy_kwargs.get('base_wall', 100.0))
        self.density_ref = float(strategy_kwargs.get('density_ref', 64.0))
        self.space_size = float(strategy_kwargs.get('space_size', -1.0))
        self._recalc_space_params()

        self.adaptation_rate = float(strategy_kwargs.get('adaptation_rate', adaptation_rate))
        self.max_step_norm = float(strategy_kwargs.get('max_step_norm', 5.0))

        self.k_repulse = float(strategy_kwargs.get('k_repulse', 0.0))
        self.k_pulse = float(strategy_kwargs.get('k_pulse', 2.0))
        self.k_attr = float(strategy_kwargs.get('k_attr', 1))

        self.plasticity: NDArray[np.float64] = np.ones(self.input_size, dtype=np.float64)
        # Keep the archive-faithful values as defaults, while making the
        # already-advertised EFFECTIVE_PARAM_KEYS usable by opt-in ablations.
        # ``decay=1, min=1, recovery=1`` is the no-freeze control used by the
        # continual graph-switch probe; normal ``nsgd`` behavior is unchanged.
        self.plasticity_decay = float(strategy_kwargs.get('plasticity_decay', 0.99))
        self.plasticity_min = float(strategy_kwargs.get('plasticity_min', 0.01))
        self.plasticity_recovery = float(strategy_kwargs.get('plasticity_recovery', 1.0))

        stats_memory_steps_raw = strategy_kwargs.get('stats_memory_steps')
        if stats_memory_steps_raw is not None:
            self.stats_memory_steps = max(1, int(stats_memory_steps_raw))
            stats_alpha = 1.0 / float(self.stats_memory_steps)
        else:
            stats_alpha = float(strategy_kwargs.get('stats_alpha', 0.01))
            self.stats_memory_steps = (
                max(1, int(round(1.0 / stats_alpha)))
                if stats_alpha > 1e-12
                else 0
            )
        stats_max_lag = int(strategy_kwargs.get('stats_max_lag', 0))
        self._coact: Optional[_LegacyCoactivationStats] = _LegacyCoactivationStats(
            self.input_size,
            alpha=stats_alpha,
            max_lag=stats_max_lag,
        )

        self.T_push: int = int(strategy_kwargs.get('T_push', 10))
        self.knn: int = int(strategy_kwargs.get('knn', 12))
        self.graph_theta: float = float(strategy_kwargs.get('graph_theta', 0.1))
        self.graph_r0_rel: float = float(strategy_kwargs.get('graph_r0_rel', 0.1))

        self.dbscan_min_samples: int = int(strategy_kwargs.get('dbscan_min_samples', 3))
        self.dbscan_auto_eps: bool = bool(strategy_kwargs.get('dbscan_auto_eps', False))
        self.embed_cluster_method = str(strategy_kwargs.get('embed_cluster_method', 'dbscan')).lower()
        self.clustering_period: int = int(strategy_kwargs.get('clustering_period', 100))
        self.louvain_distance_factor = float(strategy_kwargs.get('louvain_distance_factor', 0.15))

        self._reinit_clustering_strategy()

        self.syncmap: NDArray[np.float64] = (
            (np.random.rand(self.input_size, self.dimensions) - 0.5) * self.space_size
        )

        self._g_rows: Optional[NDArray[np.int_]] = None
        self._g_cols: Optional[NDArray[np.int_]] = None
        self._g_w: Optional[NDArray[np.float64]] = None
        self.graph_r0: float = 1.0
        self.size0: float = 1.0
        self._current_pulse_center: Optional[NDArray[np.float64]] = None

        self.labels: Optional[NDArray[np.int_]] = None
        self.trajectory_interval = max(1, trajectory_interval)
        if record_trajectory:
            self.trajectory: Optional[list[NDArray[np.float64]]] = []
            self.trajectory_steps: Optional[list[int]] = []
        else:
            self.trajectory = None
            self.trajectory_steps = None

        self.organize(strategy=self.clustering_strategy, persist=True)

    def _recalc_space_params(self) -> None:
        n_points = max(1, self.input_size)
        dim = max(1, self.dimensions)
        packing_factor = 0.64

        if self.density_ref > 1e-6:
            radius_ratio = (packing_factor / self.density_ref) ** (1.0 / dim)
            natural_radius = (self.base_wall / 2.0) * radius_ratio
            scale_factor = (n_points / self.density_ref) ** (1.0 / dim)
        else:
            natural_radius = 1.0
            scale_factor = 1.0

        self.scale = max(scale_factor, 0.5)

        if getattr(self, 'space_size', -1.0) > 0:
            self.space_size = float(self.space_size)
        else:
            self.space_size = self.base_wall * self.scale

        if not hasattr(self, 'dbscan_eps') or getattr(self, 'dbscan_auto_eps', False):
            if self.scale < 1.0:
                self.dbscan_eps = natural_radius * self.scale
            else:
                self.dbscan_eps = natural_radius
        self.dbscan_eps = 1.0

    def _reinit_clustering_strategy(self) -> None:
        try:
            if self.embed_cluster_method == 'dbscan':
                self.clustering_strategy = create_clustering_strategy(
                    'dbscan',
                    eps=self.dbscan_eps,
                    min_samples=int(self.dbscan_min_samples)
                )
            elif self.embed_cluster_method == 'louvain':
                self.clustering_strategy = create_clustering_strategy(
                    'louvain',
                    distance_factor=self.louvain_distance_factor
                )
        except Exception as e:
            print(f"Error re-initializing strategy: {e}")

    def _apply_graph_springs(self, disp: FloatArray) -> None:
        if self._g_rows is None or self._g_cols is None or self._g_w is None:
            return
        i, j, w = self._g_rows, self._g_cols, self._g_w
        diff = self.syncmap[j] - self.syncmap[i]
        dist = np.linalg.norm(diff, axis=1) + 1e-9
        factor = self.k_attr * w
        mask = dist > self.graph_r0
        if not np.any(mask):
            return
        move_vec = diff[mask] * (factor[mask, None] / dist[mask, None])
        np.add.at(disp, i[mask], move_vec)
        np.add.at(disp, j[mask], -move_vec)

    def _apply_active_pulse(self, disp: FloatArray) -> None:
        if self._current_pulse_center is None or self.k_pulse <= 0.0:
            return
        diff = self.syncmap - self._current_pulse_center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        valid_mask = (dist > 1e-9).squeeze()
        if np.any(valid_mask):
            direction = diff[valid_mask] / dist[valid_mask]
            mag = self.k_pulse / (dist[valid_mask] + 1.0)
            disp[valid_mask] += direction * mag

    def _update_plasticity(self) -> None:
        if self.labels is None:
            return
        core_mask = self.labels >= 0
        self.plasticity[core_mask] *= self.plasticity_decay
        self.plasticity[core_mask] = np.maximum(self.plasticity[core_mask], self.plasticity_min)
        noise_mask = ~core_mask
        self.plasticity[noise_mask] = np.minimum(
            self.plasticity[noise_mask] * 1.5,
            self.plasticity_recovery
        )

    def _apply_spherical_constraints(self) -> None:
        limit = self.space_size
        dists = np.linalg.norm(self.syncmap, axis=1)
        mask_out = dists > limit
        if np.any(mask_out):
            self.syncmap[mask_out] *= (limit / dists[mask_out, None])

    def step_once(self) -> None:
        disp = np.zeros_like(self.syncmap)
        self._apply_graph_springs(disp)
        self._apply_active_pulse(disp)

        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        clip_mask = (disp_norm > self.max_step_norm).squeeze()
        if np.any(clip_mask):
            disp[clip_mask] = disp[clip_mask] / disp_norm[clip_mask] * self.max_step_norm

        if self._step_count % 10 == 0:
            self._update_plasticity()

        self.syncmap += self.adaptation_rate * self.plasticity[:, None] * disp

        if self.space_size > 0:
            self._apply_spherical_constraints()

        if not np.isfinite(self.syncmap).all():
            self.syncmap = np.nan_to_num(self.syncmap, nan=0.0, posinf=1e3, neginf=-1e3)

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
        T = int(x.shape[0])

        physics_batch_size = 10
        active_masks = x > 0.1

        for t in range(max(1, T)):
            global_t = phase_offset + t
            if self._coact is not None:
                self._coact.update(active_masks[t])

            mask_t = active_masks[t]
            if np.any(mask_t):
                self._current_pulse_center = np.mean(self.syncmap[mask_t], axis=0)
            else:
                self._current_pulse_center = None

            if global_t % physics_batch_size == 0:
                for _ in range(3):
                    self.step_once()
                if self.T_push > 0 and (self._step_count % self.T_push == 0):
                    self._update_graph_structure()
                if self.clustering_period > 0 and (self._step_count % self.clustering_period == 0):
                    self._cluster_embed_labels()

            if self.trajectory is not None and (global_t % self.trajectory_interval == 0):
                self.trajectory.append(self.syncmap.copy())
                self.trajectory_steps.append(global_t)  # type: ignore[union-attr]
                self._last_time_index = global_t

        final_t = phase_offset + T - 1
        if append_final_frame and self.trajectory is not None and (T > 0 and self._last_time_index != final_t):
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(final_t)  # type: ignore[union-attr]

    def input(self, x: NDArray[np.float64]) -> None:
        self.inputGeneral(x)

    def _update_graph_structure(self) -> None:
        edges = self._export_graph_edges()
        if edges is not None:
            stable_med = self.size0 if self.size0 > 1e-9 else 1.0
            self.set_graph_edges(
                edges[0], edges[1], edges[2],
                r0_rel=self.graph_r0_rel, nn1_median=stable_med
            )

    def _export_graph_edges(self) -> tuple[IntArray, IntArray, NDArray[np.float64]] | None:
        try:
            if self._coact is None:
                return None
            sim = self._coact.similarity()
            n = int(sim.shape[0])
            if n <= 1:
                return None
            np.fill_diagonal(sim, 0.0)
            k = min(int(self.knn), n - 1)
            if k <= 0:
                return None
            idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row = np.repeat(np.arange(n), k)
            col = idx.reshape(-1)
            w = sim[row, col].reshape(-1)
            mask = w >= self.graph_theta
            row, col, w = row[mask], col[mask], w[mask]
            if row.size == 0:
                return None
            a, b = np.minimum(row, col), np.maximum(row, col)
            key = (a.astype(np.int64) * n) + b.astype(np.int64)
            uniq, inv = np.unique(key, return_inverse=True)
            w_max = np.zeros_like(uniq, dtype=float)
            np.maximum.at(w_max, inv, w)
            b_u = uniq % n
            a_u = uniq // n
            return a_u.astype(int), b_u.astype(int), w_max
        except Exception:
            return None

    def set_graph_edges(
        self,
        rows: ArrayLike | NDArray[np.int_],
        cols: ArrayLike | NDArray[np.int_],
        weights: ArrayLike | NDArray[np.float64],
        r0_rel: float = 0.4,
        nn1_median: float | None = None,
    ) -> None:
        try:
            r = np.asarray(rows, dtype=int)
            c = np.asarray(cols, dtype=int)
            w = np.asarray(weights, dtype=float)
            if r.size == 0:
                self._g_rows, self._g_cols, self._g_w = None, None, None
                return
            self._g_rows, self._g_cols = r, c
            self._g_w = w
            med = float(nn1_median) if nn1_median is not None else 1.0
            self.graph_r0 = float(max(1e-6, float(r0_rel) * med))
        except Exception:
            self._g_rows = None

    def _cluster_embed_labels(self) -> Optional[IntArray]:
        X = self.syncmap
        if X.size == 0:
            return None

        if self.embed_cluster_method == 'dbscan' and self.dbscan_auto_eps:
            try:
                k = self.dbscan_min_samples
                nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
                distances, _ = nbrs.kneighbors(X)
                k_dists = distances[:, k]
                valid_dists = k_dists[k_dists > 1e-6]
                if len(valid_dists) > 0:
                    target_eps = np.percentile(valid_dists, 95) * 1.5
                    self.dbscan_eps = max(1e-4, target_eps)
                else:
                    self.dbscan_eps = 1e-4
                self._reinit_clustering_strategy()
            except Exception as e:
                print(f"[Warn] Auto-EPS failed: {e}")

        try:
            labels = self.clustering_strategy.fit_predict(X)
            self.labels = np.asarray(labels, dtype=int)
        except Exception:
            self.labels = np.zeros(self.input_size, dtype=int)

        return self.labels

    def organize(self, strategy: ClusteringStrategy | None = None, persist: bool = False, **kwargs: Any) -> IntArray:
        if strategy is None:
            strategy = self.clustering_strategy
        return np.asarray(strategy.fit_predict(self.syncmap), dtype=int)

    def update_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'space_size':
                    self.space_size = float(v)
                else:
                    try:
                        setattr(self, k, float(v))
                    except Exception:
                        setattr(self, k, v)
        if self._coact is not None:
            if 'stats_memory_steps' in kwargs:
                self.stats_memory_steps = max(1, int(kwargs['stats_memory_steps']))
                self._coact.alpha = 1.0 / float(self.stats_memory_steps)
            elif 'stats_alpha' in kwargs:
                self._coact.alpha = float(kwargs['stats_alpha'])
                self.stats_memory_steps = (
                    max(1, int(round(1.0 / self._coact.alpha)))
                    if self._coact.alpha > 1e-12
                    else 0
                )
        self._recalc_space_params()
        self._reinit_clustering_strategy()

    def save_weights(self, filepath: Union[str, Path]) -> None:
        data = {
            'syncmap': self.syncmap,
            'plasticity': self.plasticity,
            'step_count': self._step_count,
            'labels': self.labels,
            'coact_state': self._coact.__dict__ if self._coact else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if 'syncmap' in data:
            self.syncmap = data['syncmap']
        if 'plasticity' in data:
            self.plasticity = data['plasticity']
        self.labels = data.get('labels')
        self._step_count = data.get('step_count', 0)
        if self._coact and 'coact_state' in data:
            for k, v in data['coact_state'].items():
                setattr(self._coact, k, v)

    def get_trajectory(self) -> NDArray[np.float64]:
        if not self.trajectory:
            return np.array([])
        return np.stack(self.trajectory, axis=0)

    def get_trajectory_steps(self) -> NDArray[np.int_]:
        if not self.trajectory_steps:
            return np.array([], dtype=int)
        return np.asarray(self.trajectory_steps, dtype=int)

    def get_effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in self.EFFECTIVE_PARAM_KEYS:
            params[key] = getattr(self, key)
        params['trajectory_interval'] = int(self.trajectory_interval)
        params['record_trajectory'] = bool(self.trajectory is not None)
        if self._coact is not None:
            params['stats_alpha'] = float(self._coact.alpha)
            params['stats_max_lag'] = int(self._coact.max_lag)
        return params


def create_syncmap_nsgd(input_size: int, dimensions: int, adaptation_rate: float, **kwargs: Any) -> SyncMapX_NSGD:
    return SyncMapX_NSGD(input_size, dimensions, adaptation_rate, **kwargs)
