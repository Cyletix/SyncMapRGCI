from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score


def _as_1d_int(labels: Any) -> NDArray[np.integer[Any]]:
    return np.asarray(labels, dtype=int).reshape(-1)


def arithmetic_nmi_score(true_labels: Any, predicted_labels: Any) -> float:
    """Arithmetic NMI using the ``2 * I / (H_true + H_pred)`` normalization.

    This helper keeps that normalization but uses one consistent logarithm base,
    so a perfect match still scores ``1.0``.
    """
    truth = _as_1d_int(true_labels)
    pred = _as_1d_int(predicted_labels)
    if truth.shape != pred.shape:
        raise ValueError(
            f"Metric inputs must have equal shape, got true={truth.shape} pred={pred.shape}"
        )
    if truth.size == 0:
        return 0.0
    mi = float(mutual_info_score(truth, pred))
    _, truth_counts = np.unique(truth, return_counts=True)
    _, pred_counts = np.unique(pred, return_counts=True)

    def _entropy(counts: NDArray[np.integer[Any]]) -> float:
        probs = counts.astype(float) / float(np.sum(counts))
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    h_sum = _entropy(truth_counts) + _entropy(pred_counts)
    if h_sum <= 0.0:
        return 0.0
    return float(2.0 * mi / h_sum)


def noise_as_singletons(labels: Any) -> NDArray[np.integer[Any]]:
    """Return labels where every ``-1`` noise point becomes its own cluster."""
    arr = _as_1d_int(labels).copy()
    if not np.any(arr < 0):
        return arr
    next_label = int(np.max(arr[arr >= 0])) + 1 if np.any(arr >= 0) else 0
    for idx in np.where(arr < 0)[0]:
        arr[int(idx)] = next_label
        next_label += 1
    return arr


def classified_mask(labels: Any, *, min_cluster_size: int = 2) -> NDArray[np.bool_]:
    """Return points counted as classified for SyncMap NMI.

    SyncMap visual readouts may encode isolated unclassified points as ordinary
    singleton cluster ids rather than ``-1``. Treat both explicit negative labels
    and clusters smaller than ``min_cluster_size`` as unclassified.
    """
    arr = _as_1d_int(labels)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    valid = arr >= 0
    if min_cluster_size <= 1 or not np.any(valid):
        return valid
    labels_valid, counts = np.unique(arr[valid], return_counts=True)
    large_labels = set(labels_valid[counts >= int(min_cluster_size)].astype(int).tolist())
    return np.array([int(label) in large_labels if int(label) >= 0 else False for label in arr], dtype=bool)


def nmi_iso_score(
    true_labels: Any,
    predicted_labels: Any,
    *,
    min_cluster_size: int = 2,
    coverage_power: float = 1.0,
) -> float:
    """Isolation-aware NMI. Legacy diagnostic only.

    Removed from the formal pipeline on 2026-06-12: ARI is the formal metric
    and is naturally robust to the singleton-inflation problem this metric was
    built to expose. Kept for standalone diagnostic tools.

    The score first removes unclassified predictions (``-1`` and singleton
    predicted clusters by default), computes arithmetic NMI on the covered
    subset, then multiplies by coverage. This prevents a screen full of isolated
    singleton labels from receiving a high NMI.
    """
    truth = _as_1d_int(true_labels)
    pred = _as_1d_int(predicted_labels)
    if truth.shape != pred.shape:
        raise ValueError(
            f"Metric inputs must have equal shape, got true={truth.shape} pred={pred.shape}"
        )
    if pred.size == 0:
        return 0.0
    mask = classified_mask(pred, min_cluster_size=min_cluster_size)
    coverage = float(np.mean(mask))
    if coverage <= 0.0 or not np.any(mask):
        return 0.0
    covered_nmi = arithmetic_nmi_score(truth[mask], pred[mask])
    return float((coverage ** float(coverage_power)) * covered_nmi)


def clustering_metric_bundle(true_labels: Any, predicted_labels: Any) -> dict[str, float]:
    """Compute raw and noise-policy-aware clustering metrics.

    Raw sklearn NMI/ARI treat ``-1`` and singleton clusters as normal labels.
    """
    truth = _as_1d_int(true_labels)
    pred = _as_1d_int(predicted_labels)
    if truth.shape != pred.shape:
        raise ValueError(
            f"Metric inputs must have equal shape, got true={truth.shape} pred={pred.shape}"
        )

    classified = classified_mask(pred)
    coverage = float(np.mean(classified)) if classified.size else 0.0
    bundle = {
        'classified_covered_ratio': coverage,
        'singleton_cluster_ratio': float(np.mean(~classified & (pred >= 0))) if pred.size else 0.0,
        'nmi_raw': float(normalized_mutual_info_score(truth, pred)),
        'ari_raw': float(adjusted_rand_score(truth, pred)),
    }
    if np.any(classified):
        bundle['nmi_classified_only'] = arithmetic_nmi_score(truth[classified], pred[classified])
        bundle['ari_classified_only'] = float(adjusted_rand_score(truth[classified], pred[classified]))
    else:
        bundle['nmi_classified_only'] = float('nan')
        bundle['ari_classified_only'] = float('nan')

    if not np.any(pred < 0):
        return bundle

    valid = pred >= 0
    bundle['noise_covered_ratio'] = float(np.mean(valid))
    if np.any(valid):
        bundle['nmi_masked_noise'] = float(normalized_mutual_info_score(truth[valid], pred[valid]))
        bundle['ari_masked_noise'] = float(adjusted_rand_score(truth[valid], pred[valid]))
    else:
        bundle['nmi_masked_noise'] = float('nan')
        bundle['ari_masked_noise'] = float('nan')

    singleton_pred = noise_as_singletons(pred)
    bundle['nmi_noise_singletons'] = float(normalized_mutual_info_score(truth, singleton_pred))
    bundle['ari_noise_singletons'] = float(adjusted_rand_score(truth, singleton_pred))
    return bundle


def cophenetic_from_labels(label_levels: Any) -> NDArray[np.float64]:
    """Cophenetic merge-depth matrix from hierarchical label vectors.

    Args:
        label_levels: ``(n_levels, n_nodes)`` int array; row 0 is the finest
            level, the last row the coarsest.

    Returns:
        ``(n_nodes, n_nodes)`` float array. ``M[i, j]`` is the COUNT of levels
        at which ``i`` and ``j`` share a label (nested tree: same fine = depth,
        same coarse only = 1, never = 0). Higher value = merged earlier.
    """
    levels = np.asarray(label_levels, dtype=int)
    if levels.ndim != 2:
        raise ValueError(f"label_levels must be 2D (n_levels, n_nodes), got {levels.shape}")
    n_levels, n_nodes = levels.shape
    merged = np.zeros((n_nodes, n_nodes), dtype=float)
    # COUNT the agreeing levels. The original proposal pseudocode took the max
    # 1-based level index where labels agree, but with nested labels that maps
    # same-fine pairs and same-coarse-only pairs to the same value, erasing
    # the fine tier and silently biasing the score toward coarse-only
    # geometries (verified: a 2-blob no-fine embedding scored 0.94 while a
    # 6-cluster no-coarse embedding scored 0.04 on a 6x2 tree).
    for level_idx in range(n_levels):
        same = levels[level_idx][:, None] == levels[level_idx][None, :]
        merged += same.astype(float)
    return merged


def cophenetic_from_geometry(points: Any) -> NDArray[np.float64]:
    """Single-linkage cophenetic distance matrix of an embedding.

    Equivalent to the max edge weight on the unique MST path between each
    node pair (the ultrametric the ``mst_gaps`` readout family operates on),
    but threshold-free.
    """
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import pdist, squareform

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        raise ValueError(f"points must be 2D with >=3 rows, got {arr.shape}")
    condensed = pdist(arr)
    tree = linkage(condensed, method="single")
    return np.asarray(squareform(cophenet(tree)), dtype=float)


def cophenetic_score(true_label_levels: Any, points: Any) -> float:
    """Spearman correlation between true tree merge order and geometry tree.

    Threshold-free: compares the pairwise merge-depth ORDER the geometry
    induces (single-linkage ultrametric) against the ground-truth tree.
    ``1.0`` = the embedding encodes the exact tree merge order. Independent
    of any readout cut, so it separates dynamics quality from readout
    quality. Not corrected for chance; report alongside ARI, not instead.
    """
    from scipy.stats import spearmanr

    true_merge = cophenetic_from_labels(true_label_levels)
    model_distance = cophenetic_from_geometry(points)
    if true_merge.shape != model_distance.shape:
        raise ValueError(
            f"Shape mismatch: truth {true_merge.shape} vs geometry {model_distance.shape}"
        )
    n_nodes = true_merge.shape[0]
    upper = np.triu_indices(n_nodes, k=1)
    # true_merge is a similarity (deeper merge = closer); negate to compare
    # distance-vs-distance. Spearman is invariant to monotone rescaling.
    truth_values = -true_merge[upper]
    model_values = model_distance[upper]
    if np.allclose(truth_values, truth_values[0]) or np.allclose(model_values, model_values[0]):
        return float("nan")
    rho, _p = spearmanr(truth_values, model_values)
    return float(rho)
