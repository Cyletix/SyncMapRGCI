"""
Clustering strategy utilities for SyncMapX.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, Protocol, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering
try:
    from hdbscan import HDBSCAN as ExternalHDBSCAN
except Exception:  # pragma: no cover - optional dependency
    ExternalHDBSCAN = None

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


class _LouvainModule(Protocol):
    def best_partition(
        self,
        graph: nx.Graph[Any, Any],
        weight: str | None = ...,
        resolution: float = ...,
        random_state: int | None = ...,
    ) -> dict[int, int]:
        ...


try:
    _community_module = importlib.import_module("community.community_louvain")
except ModuleNotFoundError as exc:  # pragma: no cover - import error surface at runtime
    raise ModuleNotFoundError(
        "The 'python-louvain' package is required for Louvain clustering. Please install it via 'pip install python-louvain'."
    ) from exc

community_louvain = cast(_LouvainModule, _community_module)


class ClusteringStrategy(Protocol):
    """Protocol for clustering strategy implementations."""

    def fit_predict(self, samples: FloatArray) -> IntArray:
        ...


@dataclass
class DBSCANClustering:
    """Thin wrapper around scikit-learn's DBSCAN."""

    eps: float = 3.0
    min_samples: int = 2

    def fit_predict(self, samples: FloatArray) -> IntArray:
        if samples.size == 0:
            return np.array([], dtype=int)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(samples)
        return cast(IntArray, labels)


@dataclass
class LouvainClustering:
    """
    Louvain community detection on a similarity graph derived from sample distances.
    """

    distance_factor: float = 0.5
    distance_threshold: float | None = None
    weight_mode: Literal["exp", "inverse"] = "exp"
    # Resolution parameter forwarded to python-louvain; higher => more/smaller communities
    resolution: float = 1.0
    # If > 0, build a kNN graph (scale-invariant) instead of threshold-by-distance graph
    knn_k: int = 0
    # If using kNN graph, keep only mutual kNN edges (reduces spurious crossings)
    mutual_knn: bool = True
    random_state: int | None = None

    def fit_predict(self, samples: FloatArray) -> IntArray:
        n_samples = samples.shape[0]
        if n_samples == 0:
            return np.array([], dtype=int)
        if n_samples == 1:
            return np.array([0], dtype=int)
        if np.allclose(samples, samples[[0]]):
            return np.zeros(n_samples, dtype=int)

        pairwise_distances: FloatArray | None = None
        if self.distance_threshold is None:
            pairwise_distances = distance.pdist(samples, metric="euclidean")
            mean_distance = float(np.mean(pairwise_distances))
            threshold = max(1e-6, mean_distance * self.distance_factor)
            weight_scale = max(mean_distance, 1e-9)
        else:
            threshold = max(1e-6, float(self.distance_threshold))
            weight_scale = threshold
        graph = nx.Graph()
        graph.add_nodes_from(range(n_samples))

        if int(self.knn_k) > 0:
            k = int(self.knn_k)
            k = max(1, min(k, n_samples - 1))
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
            nbrs.fit(samples)
            dists, inds = nbrs.kneighbors(return_distance=True)
            # Build neighbor maps for mutual check
            if bool(self.mutual_knn):
                neigh_sets = [set(inds[i, 1:].tolist()) for i in range(n_samples)]
            # inds[i, 0] = i, skip self
            for i in range(n_samples):
                neigh = inds[i, 1:]
                nd = dists[i, 1:]
                local = {int(j): float(d) for j, d in zip(neigh.tolist(), nd.tolist())}
                for j in local.keys():
                    if int(j) <= i:
                        continue
                    if bool(self.mutual_knn) and (i not in neigh_sets[int(j)]):
                        continue
                    dist_ij = local[int(j)]
                    if dist_ij >= threshold:
                        continue
                    if self.weight_mode == "inverse":
                        weight = 1.0 / (float(dist_ij) + 1e-5)
                    else:
                        weight = float(np.exp(-float(dist_ij) / weight_scale))
                    graph.add_edge(i, int(j), weight=weight)
        else:
            condensed = pairwise_distances if pairwise_distances is not None else distance.pdist(samples, metric="euclidean")
            dist_matrix = distance.squareform(condensed)
            rows, cols = np.where(np.triu(dist_matrix < threshold, k=1))
            if rows.size > 0:
                edge_distances = dist_matrix[rows, cols]
                if self.weight_mode == "inverse":
                    weights = 1.0 / (edge_distances + 1e-5)
                else:
                    weights = np.exp(-edge_distances / weight_scale)
                graph.add_weighted_edges_from(
                    (int(i), int(j), float(w))
                    for i, j, w in zip(rows.tolist(), cols.tolist(), weights.tolist())
                )

        if graph.number_of_edges() == 0:
            # No edges created; fall back to each node as its own cluster
            return np.arange(n_samples, dtype=int)

        try:
            partition = community_louvain.best_partition(
                graph,
                weight="weight",
                resolution=float(self.resolution),
                random_state=None if self.random_state is None else int(self.random_state),
            )
        except TypeError:
            partition = community_louvain.best_partition(
                graph,
                weight="weight",
                resolution=float(self.resolution),
            )
        labels = np.array([partition.get(i, -1) for i in range(n_samples)], dtype=int)
        return labels


@dataclass
class AgglomerativeClusteringStrategy:
    """
    Wrapper for scikit-learn AgglomerativeClustering.

    Note: For linkage='ward', metric is fixed to Euclidean by sklearn.
    """

    n_clusters: int | None = 2
    distance_threshold: float | None = None
    linkage: str = "ward"

    def fit_predict(self, samples: FloatArray) -> IntArray:
        if samples.size == 0:
            return np.array([], dtype=int)

        model = SkAgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
        )
        labels = model.fit_predict(samples)
        return cast(IntArray, labels)


@dataclass
class HDBSCANClusteringStrategy:
    """Wrapper for the optional hdbscan package."""

    min_cluster_size: int = 2
    min_samples: int | None = None

    def fit_predict(self, samples: FloatArray) -> IntArray:
        if samples.size == 0:
            return np.array([], dtype=int)
        if ExternalHDBSCAN is None:
            raise ModuleNotFoundError(
                "The 'hdbscan' package is required for HDBSCAN clustering. Please install it via 'pip install hdbscan'."
            )
        clusterer = ExternalHDBSCAN(
            min_cluster_size=max(2, int(self.min_cluster_size)),
            min_samples=None if self.min_samples is None else max(1, int(self.min_samples)),
        )
        labels = clusterer.fit_predict(samples)
        return cast(IntArray, labels)


def create_clustering_strategy(name: str, **kwargs: Any) -> ClusteringStrategy:
    """Factory helper that instantiates a clustering strategy by name."""
    strategy_name = name.lower()

    if strategy_name == "dbscan":
        return DBSCANClustering(**kwargs)
    if strategy_name == "louvain":
        return LouvainClustering(**kwargs)
    if strategy_name in ("agglomerative", "hierarchical"):
        return AgglomerativeClusteringStrategy(**kwargs)
    if strategy_name == "hdbscan":
        return HDBSCANClusteringStrategy(**kwargs)

    raise ValueError(f"Unknown clustering strategy: {name}")
