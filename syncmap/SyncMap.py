"""Original SyncMap baseline.

Port of the public reference implementation released with
"Continual General Chunking Problem and SyncMap" (Vargas & Asabuki, AAAI 2021),
https://github.com/zweifel/SyncMap (MIT License, Copyright (c) 2021 zweifel).
The update rule and DBSCAN readout follow that release; this module only adds
the trajectory recording and parameter helpers used by the experiment runner.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering, DBSCAN


class SyncMap:
    """Constant-rate SyncMap: active nodes move toward the active centroid and away from the inactive one."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "space_size": 10.0,
        "organize_dbscan_eps": 3.0,
        "organize_dbscan_min_samples": 2,
    }

    EFFECTIVE_PARAM_KEYS: tuple[str, ...] = (
        "input_size",
        "dimensions",
        "adaptation_rate",
        "space_size",
        "trajectory_interval",
        "organize_dbscan_eps",
        "organize_dbscan_min_samples",
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
        trajectory_interval: int = 50,
        record_trajectory: bool = False,
        space_size: float = 10.0,
        organize_dbscan_eps: float = 3.0,
        organize_dbscan_min_samples: int = 2,
    ) -> None:
        self.organized: bool = False
        self.input_size: int = int(input_size)
        self.dimensions: int = int(dimensions)
        self.adaptation_rate: float = float(adaptation_rate)
        self.space_size: float = float(space_size)
        self.organize_dbscan_eps: float = float(organize_dbscan_eps)
        self.organize_dbscan_min_samples: int = int(organize_dbscan_min_samples)
        self.syncmap: NDArray[np.float64] = np.random.rand(self.input_size, self.dimensions)
        self.labels: Optional[NDArray[np.int_]] = None

        # The runner always reads the trajectory, so it is recorded regardless of the flag.
        _ = record_trajectory
        self.trajectory_interval: int = max(1, int(trajectory_interval))
        self.trajectory: list[NDArray[np.float64]] = []
        self.trajectory_steps: list[int] = []
        self._last_time_index: int = -1

    def _record(self, global_t: int) -> None:
        self.trajectory.append(self.syncmap.copy())
        self.trajectory_steps.append(global_t)
        self._last_time_index = global_t

    def inputGeneral(self, x: NDArray[np.float64]) -> None:
        plus = x > 0.1
        minus = ~plus
        phase_offset = self._last_time_index + 1 if self._last_time_index >= 0 else 0

        T = int(x.shape[0])
        for t in range(T):
            global_t = phase_offset + t
            vplus = plus[t, :]
            vminus = minus[t, :]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass > 1 and minus_mass > 1:
                center_plus = np.dot(vplus, self.syncmap) / plus_mass
                center_minus = np.dot(vminus, self.syncmap) / minus_mass
                dist_plus = distance.cdist(center_plus[None, :], self.syncmap, "euclidean").T
                dist_minus = distance.cdist(center_minus[None, :], self.syncmap, "euclidean").T

                update_plus = vplus[:, np.newaxis] * ((center_plus - self.syncmap) / dist_plus)
                update_minus = vminus[:, np.newaxis] * ((center_minus - self.syncmap) / dist_minus)
                self.syncmap += self.adaptation_rate * (update_plus - update_minus)

                maximum = self.syncmap.max()
                if maximum > 0:
                    self.syncmap = self.space_size * self.syncmap / maximum

            if global_t % self.trajectory_interval == 0:
                self._record(global_t)

        final_step = phase_offset + T - 1
        if T > 0 and self._last_time_index != final_step:
            self._record(final_step)

    def input(self, x: NDArray[np.float64]) -> None:
        self.inputGeneral(x)

    def organize(self) -> NDArray[np.int_]:
        self.organized = True
        self.labels = DBSCAN(
            eps=float(self.organize_dbscan_eps),
            min_samples=int(self.organize_dbscan_min_samples),
        ).fit_predict(self.syncmap)
        self.trajectory.append(self.syncmap.copy())
        if self._last_time_index >= 0:
            self.trajectory_steps.append(self._last_time_index)
        return self.labels

    def hierarchical_organize(self, n: int = 2) -> NDArray[np.int_]:
        self.organized = True
        self.labels = AgglomerativeClustering(distance_threshold=None, n_clusters=n).fit_predict(self.syncmap)
        return self.labels

    def activate(self, x: NDArray[np.float64]) -> Optional[int]:
        """Return the label of the index with maximum input value."""
        if not self.organized:
            print("Activating a non-organized SyncMap")
            return None
        if self.labels is None:
            raise RuntimeError("Labels not initialized. Call organize() before activate().")
        return self.labels[int(np.argmax(x))]

    def save_weights(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({"syncmap": self.syncmap}, f)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        if data["syncmap"].shape != self.syncmap.shape:
            raise ValueError(f"Size mismatch: loaded {data['syncmap'].shape}, expected {self.syncmap.shape}")
        self.syncmap = data["syncmap"]

    def reset_adaptation(self) -> None:
        """Constant-rate update has no optimizer state to reset."""

    def update_params(self, **kwargs: Any) -> None:
        if "space_size" in kwargs:
            self.space_size = float(kwargs["space_size"])
        if "organize_dbscan_eps" in kwargs:
            self.organize_dbscan_eps = float(kwargs["organize_dbscan_eps"])
        if "organize_dbscan_min_samples" in kwargs:
            self.organize_dbscan_min_samples = max(1, int(kwargs["organize_dbscan_min_samples"]))
        if "adaptation_rate" in kwargs:
            self.adaptation_rate = float(kwargs["adaptation_rate"])

    def get_trajectory(self) -> NDArray[np.float64]:
        if not self.trajectory:
            raise RuntimeError("Trajectory not recorded. Run inputGeneral() first.")
        return np.stack(self.trajectory, axis=0)

    def get_trajectory_steps(self) -> NDArray[np.int_]:
        if not self.trajectory_steps:
            raise RuntimeError("Trajectory steps not recorded. Run inputGeneral() first.")
        return np.asarray(self.trajectory_steps, dtype=int)

    def get_effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {key: getattr(self, key) for key in self.EFFECTIVE_PARAM_KEYS}
        params["record_trajectory"] = True
        return params
