"""
Symmetrical SyncMap — from "Symmetrical SyncMap for Imbalanced General Chunking Problems"
(arXiv:2310.10045v1, Zhang & Vargas, 2023)

Key innovations over the original SyncMap:
1.  Memory Window — generalised state memory *m* (threshold *a*) instead of fixed m=2.
2.  Symmetrical Activation — equal numbers of positive and negative nodes are
    stochastically selected at each time step, balancing the two feedback loops.
3.  Moving-average readout — DBSCAN is applied to a running mean of node positions
    (window = 10 000 steps), not a single snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, DBSCAN


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_k_indices(n: int, k: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Sample *exactly* k distinct indices from [0, n) without replacement."""
    if k >= n:
        return np.arange(n, dtype=np.intp)
    return rng.choice(n, size=k, replace=False).astype(np.intp, copy=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Symmetrical SyncMap
# ═══════════════════════════════════════════════════════════════════════════════

class SymmetricalSyncMap:
    """
    Symmetrical SyncMap with a constant learning rate, as in the paper.

    Parameters
    ----------
    input_size : int
        Number of distinct state variables (nodes).
    dimensions : int
        Dimensionality of the SyncMap map space *k*.
    adaptation_rate : float
        Base learning rate α.
    state_memory : int
        Memory window *m* (number of most-recently-active states in PS_temp).
        Default 3 (recommended in the paper).
    stochastic_prob : float
        Probability *Pr* ∈ [0,1] of enabling stochastic positive-node selection
        when m > 2.  Default 0.3 (30 %).
    moving_average_window : int
        Steps over which node positions are averaged before clustering.
        Paper default: 10 000.
    space_size : float
        Radius of the normalisation hypersphere.  Paper default: 10.0.
    trajectory_interval : int
        Record a trajectory snapshot every *interval* steps.
    record_trajectory : bool
        Whether to record trajectory internally.
    **strategy_kwargs
        Optional space_size / organize_dbscan_* overrides.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "space_size": 10.0,
        "state_memory": 3,
        "stochastic_prob": 0.3,
        "moving_average_window": 10000,
        "random_seed": None,
        "organize_dbscan_eps": 4.5,
        "organize_dbscan_min_samples": 2,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        "input_size",
        "dimensions",
        "space_size",
        "state_memory",
        "stochastic_prob",
        "moving_average_window",
        "random_seed",
        "adaptation_rate",
        "organize_dbscan_eps",
        "organize_dbscan_min_samples",
        "trajectory_interval",
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
        dimensions: int,
        adaptation_rate: float,
        state_memory: int = 3,
        stochastic_prob: float = 0.3,
        moving_average_window: int = 10000,
        random_seed: Optional[int] = None,
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        strategy_kwargs = dict(strategy_kwargs)

        # ── core dimensions ───────────────────────────────────────────────
        self.input_size: int = int(input_size)
        self.dimensions: int = int(dimensions)
        self.space_size = float(strategy_kwargs.pop("space_size", self.DEFAULT_PARAMS["space_size"]))

        # ── Symmetrical-SyncMap specific parameters ────────────────────────
        self.state_memory: int = max(2, int(state_memory))
        self.stochastic_prob: float = float(np.clip(stochastic_prob, 0.0, 1.0))
        self.moving_average_window: int = max(1, int(moving_average_window))
        self.random_seed: Optional[int] = (
            None if random_seed is None else int(random_seed)
        )

        # ── clustering params ─────────────────────────────────────────────
        self.organize_dbscan_eps = float(
            strategy_kwargs.pop("organize_dbscan_eps", self.DEFAULT_PARAMS["organize_dbscan_eps"])
        )
        self.organize_dbscan_min_samples = int(
            strategy_kwargs.pop("organize_dbscan_min_samples", self.DEFAULT_PARAMS["organize_dbscan_min_samples"])
        )

        # ── trajectory ────────────────────────────────────────────────────
        self.trajectory: list[NDArray[np.float64]] = []
        self.trajectory_interval: int = max(1, int(trajectory_interval))
        self.trajectory_steps: list[int] = []
        self._last_time_index: int = -1
        _ = record_trajectory  # always recorded internally for moving-average

        # ── map state ─────────────────────────────────────────────────────
        self.syncmap: NDArray[np.float64] = np.random.rand(input_size, dimensions).astype(np.float64)
        self.synapses_matrix: NDArray[np.float64] = np.zeros([input_size, input_size], dtype=np.float64)
        self.organized: bool = False
        self.labels: Optional[NDArray[np.int_]] = None

        # ── moving-average buffer for readout ─────────────────────────────
        # Paper §5.2: DBSCAN is applied to the mean of node positions over the
        # last `moving_average_window` steps (a sliding window, default 10000),
        # NOT a cumulative mean over all steps. Ring buffer + running sum keeps
        # the readout on the converged steady state and out of the early
        # pre-convergence transient.
        from collections import deque as _deque
        self._ma_buffer: _deque[NDArray[np.float64]] = _deque(maxlen=self.moving_average_window)
        self._ma_sum: NDArray[np.float64] = np.zeros_like(self.syncmap)

        self.adaptation_rate: float = float(adaptation_rate)
        if strategy_kwargs:
            raise TypeError(f"Unexpected SymmetricalSyncMap parameters: {sorted(strategy_kwargs)}")

        # ── RNG for stochastic selection ──────────────────────────────────
        # Without an explicit seed, draw one from the global NumPy state so seeded runs stay reproducible.
        rng_seed = self.random_seed if self.random_seed is not None else int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(rng_seed)

    # ── update helpers ────────────────────────────────────────────────────

    def _apply_update(self, node_idx: int, update_vector: NDArray[np.float64]) -> None:
        self.syncmap[node_idx] += self.adaptation_rate * update_vector

    def _normalize(self) -> None:
        """Normalise all node vectors to lie inside a hypersphere of radius `space_size`."""
        maximum = float(np.max(np.abs(self.syncmap)))
        if maximum > 0:
            self.syncmap = self.space_size * self.syncmap / maximum

    def _update_moving_average(self) -> None:
        """Add current node positions to the sliding-window moving average
        (mean over the last `moving_average_window` steps, paper §5.2)."""
        if len(self._ma_buffer) == self._ma_buffer.maxlen:
            # the oldest frame is about to be evicted by the append below
            self._ma_sum = self._ma_sum - self._ma_buffer[0]
        self._ma_buffer.append(self.syncmap.copy())
        self._ma_sum = self._ma_sum + self.syncmap

    def _get_moving_average(self) -> NDArray[np.float64]:
        """Return the sliding-window moving-average node positions."""
        if not self._ma_buffer:
            return self.syncmap
        return self._ma_sum / len(self._ma_buffer)

    # ── main training loop ────────────────────────────────────────────────

    def inputGeneral(
        self,
        x: NDArray[np.float64],
        *,
        phase_offset: int = 0,
        reset_trajectory: bool = False,
        append_final_frame: bool = True,
    ) -> None:
        r"""
        Process an input sequence *x* of shape (T, N).

        At each time step:
        1. Build PS_temp from the *m* most-recently-active nodes (threshold logic).
        2. Stochastically select PS_t ⊆ PS_temp.
        3. Build NS_temp = all_nodes \\ PS_t.
        4. Stochastically select NS_t ⊆ NS_temp with |NS_t| = |PS_t|.
        5. Compute centroids cp, cn and apply attraction / repulsion.
        6. Normalise.
        """
        if reset_trajectory:
            self.trajectory.clear()
            self.trajectory_steps.clear()
            self._last_time_index = -1
        # Use explicit phase_offset if provided, else auto-compute from last state
        if phase_offset <= 0 and self._last_time_index >= 0:
            phase_offset = self._last_time_index + 1
        T = int(x.shape[0])
        N = self.input_size

        # ── rolling activation tracker ────────────────────────────────────
        # last_active[i] = global time step when node i was last active
        # Initialised to a large negative value so all nodes start inactive.
        last_active = np.full(N, -999999, dtype=np.int64)

        # decay constant as per paper Eq. 6: exp(-0.1 * (t - t_a))
        # threshold a = exp(-0.1 * m * tstep)  with tstep = 10
        #   m=2 → a = exp(-2)   ≈ 0.135
        #   m=3 → a = exp(-3)   ≈ 0.050
        #   m=4 → a = exp(-4)   ≈ 0.018
        #   m=5 → a = exp(-5)   ≈ 0.007
        tstep = 10
        threshold_a = float(np.exp(-0.1 * self.state_memory * tstep))

        for t in range(T):
            global_t = phase_offset + t
            current_input = x[t, :]  # shape (N,)

            # Update activation tracker
            active_mask = current_input > 0
            last_active[active_mask] = global_t

            # ── build PS_temp: nodes whose activation is above threshold ──
            # activation value = exp(-0.1 * (global_t - last_active))
            ages = global_t - last_active
            activation_values = np.exp(-0.1 * ages)
            # nodes that are currently active (input > 0) OR still within memory window
            in_window = activation_values > threshold_a
            ps_temp_indices = np.where(in_window)[0]

            if len(ps_temp_indices) == 0:
                # No positive nodes → skip update but still record trajectory
                if global_t % self.trajectory_interval == 0:
                    self.trajectory.append(self.syncmap.copy())
                    self.trajectory_steps.append(global_t)
                    self._last_time_index = global_t
                continue

            # ── stochastic selection into PS_t ────────────────────────────
            if self.state_memory > 2 and self._rng.random() < self.stochastic_prob:
                # Select exactly 2 positive nodes from PS_temp
                k_select = min(2, len(ps_temp_indices))
                selected_ps = self._rng.choice(ps_temp_indices, size=k_select, replace=False)
                ps_t_set = set(int(idx) for idx in selected_ps)
            else:
                ps_t_set = set(int(idx) for idx in ps_temp_indices)

            # ── build NS_temp = all nodes \ PS_t ──────────────────────────
            ns_temp_indices = np.array([i for i in range(N) if i not in ps_t_set], dtype=np.intp)

            if len(ps_t_set) == 0 or len(ns_temp_indices) == 0:
                if global_t % self.trajectory_interval == 0:
                    self.trajectory.append(self.syncmap.copy())
                    self.trajectory_steps.append(global_t)
                    self._last_time_index = global_t
                continue

            # ── stochastic selection into NS_t: |NS_t| = |PS_t| ───────────
            m_neg = len(ps_t_set)
            selected_ns = _sample_k_indices(len(ns_temp_indices), m_neg, self._rng)
            ns_t_indices = ns_temp_indices[selected_ns]
            ns_t_set = set(int(idx) for idx in ns_t_indices)

            # Quick validity check
            if len(ps_t_set) <= 1 or len(ns_t_set) == 0:
                if global_t % self.trajectory_interval == 0:
                    self.trajectory.append(self.syncmap.copy())
                    self.trajectory_steps.append(global_t)
                    self._last_time_index = global_t
                continue

            # ── centroids ─────────────────────────────────────────────────
            ps_list = list(ps_t_set)
            ns_list = list(ns_t_set)

            cp = np.mean(self.syncmap[ps_list], axis=0)
            cn = np.mean(self.syncmap[ns_list], axis=0)

            # ── update: attraction for PS, repulsion for NS ───────────────
            for i_idx in ps_list:
                # positive feedback: pull toward cp
                diff_p = cp - self.syncmap[i_idx]
                dist_p = float(np.linalg.norm(diff_p)) + 1e-8
                self._apply_update(i_idx, diff_p / dist_p)

            for i_idx in ns_list:
                # negative feedback: push away from cn
                diff_n = cn - self.syncmap[i_idx]
                dist_n = float(np.linalg.norm(diff_n)) + 1e-8
                self._apply_update(i_idx, -diff_n / dist_n)

            # ── normalise ─────────────────────────────────────────────────
            self._normalize()

            # ── update moving average ─────────────────────────────────────
            self._update_moving_average()

            # ── record trajectory ─────────────────────────────────────────
            if global_t % self.trajectory_interval == 0:
                self.trajectory.append(self.syncmap.copy())
                self.trajectory_steps.append(global_t)
                self._last_time_index = global_t

        # Ensure last step is recorded
        final_step = phase_offset + T - 1
        if T > 0 and self._last_time_index != final_step:
            self.trajectory.append(self.syncmap.copy())
            self.trajectory_steps.append(final_step)
            self._last_time_index = final_step

    def input(self, x: NDArray[np.float64]) -> None:
        """Main input method — delegates to inputGeneral."""
        self.inputGeneral(x)

    # ── clustering / readout ──────────────────────────────────────────────

    def organize(self) -> NDArray[np.int_]:
        """
        Cluster nodes using DBSCAN on the **moving-average** node positions
        (as recommended in the paper, Section 5).
        """
        self.organized = True
        ma_positions = self._get_moving_average()
        self.labels = DBSCAN(
            eps=float(self.organize_dbscan_eps),
            min_samples=int(self.organize_dbscan_min_samples),
        ).fit_predict(ma_positions)

        self.trajectory.append(self.syncmap.copy())
        if self._last_time_index >= 0:
            self.trajectory_steps.append(self._last_time_index)

        return self.labels

    def hierarchical_organize(self, n: int = 2) -> NDArray[np.int_]:
        """Cluster nodes using Agglomerative Clustering (ward linkage)."""
        self.organized = True
        ma_positions = self._get_moving_average()
        self.labels = AgglomerativeClustering(
            distance_threshold=None, n_clusters=n,
        ).fit_predict(ma_positions)
        return self.labels

    def activate(self, x: NDArray[np.float64]) -> Optional[int]:
        """Return the cluster label of the most-active input node."""
        if not self.organized:
            print("Activating a non-organized SymmetricalSyncMap")
            return None
        if self.labels is None:
            raise RuntimeError("Labels not initialized. Call organize() before activate().")
        max_index = int(np.argmax(x))
        return self.labels[max_index]

    # ── persistence ───────────────────────────────────────────────────────

    def save_weights(self, filepath: Union[str, Path]) -> None:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'syncmap': self.syncmap}, f)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if data['syncmap'].shape != self.syncmap.shape:
            raise ValueError(
                f"Size mismatch: loaded {data['syncmap'].shape}, "
                f"expected {self.syncmap.shape}"
            )
        self.syncmap = data['syncmap']

    def reset_adaptation(self) -> None:
        """Constant-rate update has no optimizer state to reset."""

    # ── parameter helpers ─────────────────────────────────────────────────

    def update_params(self, **kwargs: Any) -> None:
        if "space_size" in kwargs:
            self.space_size = float(kwargs["space_size"])
        if "organize_dbscan_eps" in kwargs:
            self.organize_dbscan_eps = float(kwargs["organize_dbscan_eps"])
        if "organize_dbscan_min_samples" in kwargs:
            self.organize_dbscan_min_samples = max(1, int(kwargs["organize_dbscan_min_samples"]))
        if "adaptation_rate" in kwargs:
            self.adaptation_rate = float(kwargs["adaptation_rate"])

    # ── trajectory access ─────────────────────────────────────────────────

    def get_trajectory(self) -> NDArray[np.float64]:
        if not self.trajectory:
            raise RuntimeError("Trajectory not recorded. Run input() first.")
        return np.stack(self.trajectory, axis=0)

    def get_trajectory_steps(self) -> NDArray[np.int_]:
        if not self.trajectory_steps:
            raise RuntimeError("Trajectory steps not recorded. Run input() first.")
        return np.asarray(self.trajectory_steps, dtype=int)

    def get_effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in self.EFFECTIVE_PARAM_KEYS:
            params[key] = getattr(self, key)
        params["record_trajectory"] = True
        return params


# ── convenience factory ────────────────────────────────────────────────────

def create_symmetric_syncmap(
    input_size: int,
    dimensions: int,
    adaptation_rate: float,
    state_memory: int = 3,
    stochastic_prob: float = 0.3,
    moving_average_window: int = 10000,
    random_seed: Optional[int] = None,
    trajectory_interval: int = 50,
    record_trajectory: bool = False,
    **kwargs: Any,
) -> SymmetricalSyncMap:
    """Create a constant-rate SymmetricalSyncMap."""
    return SymmetricalSyncMap(
        input_size,
        dimensions,
        adaptation_rate,
        state_memory=state_memory,
        stochastic_prob=stochastic_prob,
        moving_average_window=moving_average_window,
        random_seed=random_seed,
        trajectory_interval=trajectory_interval,
        record_trajectory=record_trajectory,
        **kwargs,
    )
