from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - dense fallback preserves runtime behavior
    NearestNeighbors = None

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

_EPS = 1e-9
_NO_CLIFF_THRESHOLD = 1.0 - 1e-6


@dataclass(frozen=True)
class RGSignificanceResult:
    active_tokens: IntArray
    local_scales: FloatArray
    threshold_norm: float
    threshold_distance: float
    best_jump_ratio: float
    mst_edges: IntArray
    mst_lengths: FloatArray
    mst_normalized_lengths: FloatArray
    component_labels: IntArray
    component_count: int


def _pairwise_distances(points: FloatArray) -> FloatArray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _dense_knn_query_from_distances(distances: FloatArray, knn_k: int) -> tuple[IntArray, FloatArray]:
    n = int(distances.shape[0])
    if n <= 1:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0), dtype=float)
    k = min(max(1, int(knn_k)), n - 1)
    finite = np.asarray(distances, dtype=float).copy()
    np.fill_diagonal(finite, np.inf)
    idx = np.argpartition(finite, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(finite, idx, axis=1)
    order = np.argsort(part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return idx.astype(int), part.astype(float)


def _dense_knn_query(points: FloatArray, knn_k: int) -> tuple[IntArray, FloatArray]:
    distances = _pairwise_distances(points)
    return _dense_knn_query_from_distances(distances, knn_k)


def _knn_query(points: FloatArray, knn_k: int) -> tuple[IntArray, FloatArray]:
    n = int(points.shape[0])
    if n <= 1:
        return np.zeros((n, 0), dtype=int), np.zeros((n, 0), dtype=float)
    k = min(max(1, int(knn_k)), n - 1)
    if NearestNeighbors is None:
        return _dense_knn_query(points, k)
    query = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    query.fit(points)
    distances, indices = query.kneighbors(points, return_distance=True)
    return indices[:, 1:].astype(int), distances[:, 1:].astype(float)


def _estimate_local_scales_from_neighbors(neighbor_distances: FloatArray, node_count: int) -> FloatArray:
    if node_count <= 1:
        return np.ones(node_count, dtype=float)
    if neighbor_distances.size == 0:
        return np.ones(node_count, dtype=float)
    positive = np.where(np.isfinite(neighbor_distances) & (neighbor_distances > _EPS), neighbor_distances, np.inf)
    local = np.min(positive, axis=1)
    valid = local[np.isfinite(local) & (local > _EPS)]
    fallback = float(np.median(valid)) if valid.size else 1.0
    local = np.where(np.isfinite(local) & (local > _EPS), local, fallback)
    return np.maximum(local.astype(float), _EPS)


def _dense_mst(distances: FloatArray) -> tuple[IntArray, FloatArray]:
    n = int(distances.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)

    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    best = distances[0].copy()
    parent = np.zeros(n, dtype=int)
    parent[0] = -1

    edges: list[tuple[int, int]] = []
    lengths: list[float] = []

    for _ in range(n - 1):
        candidates = np.where(visited, np.inf, best)
        nxt = int(np.argmin(candidates))
        if not np.isfinite(candidates[nxt]):
            break
        prev = int(parent[nxt])
        edges.append((prev, nxt))
        lengths.append(float(distances[prev, nxt]))
        visited[nxt] = True

        improved = distances[nxt] < best
        parent[improved] = nxt
        best = np.minimum(best, distances[nxt])

    return np.asarray(edges, dtype=int), np.asarray(lengths, dtype=float)


def _mutual_knn_pairs(distances: FloatArray, knn_k: int) -> IntArray:
    n = int(distances.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=int)

    k = min(max(1, int(knn_k)), n - 1)
    finite = distances.copy()
    np.fill_diagonal(finite, np.inf)
    idx = np.argpartition(finite, kth=k - 1, axis=1)[:, :k]
    neighbor_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    neighbor_mask[rows, idx] = True
    mutual = np.triu(neighbor_mask & neighbor_mask.T, k=1)
    return np.argwhere(mutual).astype(int)


def _mutual_knn_pairs_from_neighbors(neighbor_indices: IntArray, knn_k: int) -> IntArray:
    n = int(neighbor_indices.shape[0])
    if n <= 1 or neighbor_indices.shape[1] == 0:
        return np.zeros((0, 2), dtype=int)
    k = min(max(1, int(knn_k)), neighbor_indices.shape[1])
    neighbor_sets = [set(map(int, row[:k].tolist())) for row in neighbor_indices]
    mutual: list[tuple[int, int]] = []
    for u in range(n):
        for v in neighbor_sets[u]:
            if v <= u:
                continue
            if u in neighbor_sets[v]:
                mutual.append((u, v))
    if not mutual:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(mutual, dtype=int)


def _candidate_edges_from_neighbors(
    neighbor_indices: IntArray,
    neighbor_distances: FloatArray,
) -> tuple[IntArray, FloatArray]:
    edge_weights: dict[tuple[int, int], float] = {}
    for u in range(int(neighbor_indices.shape[0])):
        for slot, v in enumerate(neighbor_indices[u].tolist()):
            weight = float(neighbor_distances[u, slot])
            if not np.isfinite(weight):
                continue
            a = min(u, int(v))
            b = max(u, int(v))
            if a == b:
                continue
            prev = edge_weights.get((a, b))
            if prev is None or weight < prev:
                edge_weights[(a, b)] = weight
    if not edge_weights:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
    pairs = np.asarray(list(edge_weights.keys()), dtype=int)
    lengths = np.asarray(list(edge_weights.values()), dtype=float)
    order = np.argsort(lengths)
    return pairs[order], lengths[order]


def _candidate_edges_from_mask(distances: FloatArray, edge_mask: NDArray[np.bool_]) -> tuple[IntArray, FloatArray]:
    n = int(distances.shape[0])
    if n <= 1 or edge_mask.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
    edge_weights: dict[tuple[int, int], float] = {}
    for u in range(n):
        for v in np.where(edge_mask[u])[0].tolist():
            a = min(u, int(v))
            b = max(u, int(v))
            if a == b:
                continue
            weight = float(distances[a, b])
            if not np.isfinite(weight):
                continue
            prev = edge_weights.get((a, b))
            if prev is None or weight < prev:
                edge_weights[(a, b)] = weight
    if not edge_weights:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
    pairs = np.asarray(list(edge_weights.keys()), dtype=int)
    lengths = np.asarray(list(edge_weights.values()), dtype=float)
    order = np.argsort(lengths)
    return pairs[order], lengths[order]


def _sparse_mst(
    node_count: int,
    candidate_edges: IntArray,
    candidate_lengths: FloatArray,
) -> tuple[IntArray, FloatArray]:
    if node_count <= 1 or candidate_edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)

    parent = np.arange(node_count, dtype=int)
    rank = np.zeros(node_count, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst_edges: list[tuple[int, int]] = []
    mst_lengths: list[float] = []
    for idx, (u, v) in enumerate(candidate_edges.tolist()):
        if union(int(u), int(v)):
            mst_edges.append((int(u), int(v)))
            mst_lengths.append(float(candidate_lengths[idx]))
            if len(mst_edges) == node_count - 1:
                break
    return np.asarray(mst_edges, dtype=int), np.asarray(mst_lengths, dtype=float)


def _sparse_mst_from_points(points: FloatArray, base_knn_k: int) -> tuple[IntArray, FloatArray]:
    n = int(points.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
    current_k = min(max(1, int(base_knn_k)), n - 1)
    while True:
        neighbor_indices, neighbor_distances = _knn_query(points, current_k)
        candidate_edges, candidate_lengths = _candidate_edges_from_neighbors(neighbor_indices, neighbor_distances)
        mst_edges, mst_lengths = _sparse_mst(n, candidate_edges, candidate_lengths)
        if mst_edges.shape[0] == n - 1:
            return mst_edges, mst_lengths
        if current_k >= n - 1:
            break
        next_k = min(n - 1, max(current_k + 1, current_k * 2))
        if next_k == current_k:
            break
        current_k = next_k
    distances = _pairwise_distances(points)
    return _dense_mst(distances)


def _semantic_adjusted_distances(
    geometry_distances: FloatArray,
    semantic_similarity: FloatArray,
    *,
    semantic_penalty_scale: float,
    semantic_similarity_floor: float,
    semantic_penalty_mode: str = "absolute",
    semantic_relative_threshold: float = 0.5,
    semantic_anchor_floor: float = 0.3,
) -> FloatArray:
    geom = np.asarray(geometry_distances, dtype=float)
    sim = np.asarray(semantic_similarity, dtype=float)
    if geom.shape != sim.shape:
        raise ValueError("semantic_similarity must match the pairwise geometry distance matrix shape.")
    floor = float(np.clip(semantic_similarity_floor, 0.0, 1.0))
    scale = max(float(semantic_penalty_scale), 0.0)
    if scale <= 0.0:
        return geom.copy()
    mode = str(semantic_penalty_mode).strip().lower()
    semantic_support = np.clip((sim - floor) / max(1.0 - floor, 1e-9), 0.0, 1.0)
    if mode == "contrastive":
        sim_wo_diag = sim.copy()
        np.fill_diagonal(sim_wo_diag, 0.0)
        anchors = np.max(sim_wo_diag, axis=1)
        anchor_floor = float(np.clip(semantic_anchor_floor, 0.0, 1.0))
        anchor_strength = np.clip((anchors - anchor_floor) / max(1.0 - anchor_floor, 1e-9), 0.0, 1.0)
        anchor_matrix = np.maximum.outer(anchors, anchors)
        relative_support = np.divide(
            sim,
            np.maximum(anchor_matrix, 1e-9),
            out=np.zeros_like(sim),
            where=anchor_matrix > 1e-9,
        )
        rel_threshold = max(float(semantic_relative_threshold), 1e-9)
        relative_gap = np.clip((rel_threshold - relative_support) / rel_threshold, 0.0, 1.0)
        edge_anchor_strength = np.maximum.outer(anchor_strength, anchor_strength)
        penalty = 1.0 + scale * relative_gap * edge_anchor_strength
    else:
        penalty = 1.0 + scale * (1.0 - semantic_support)
    penalty = np.asarray(penalty, dtype=float)
    np.fill_diagonal(penalty, 0.0)
    adjusted = geom * penalty
    np.fill_diagonal(adjusted, 0.0)
    return adjusted


def _neighbor_mask_from_indices(neighbor_indices: IntArray, node_count: int, knn_k: int) -> NDArray[np.bool_]:
    mask = np.zeros((node_count, node_count), dtype=bool)
    if node_count <= 1 or neighbor_indices.shape[1] == 0:
        return mask
    k = min(max(1, int(knn_k)), neighbor_indices.shape[1])
    for u in range(node_count):
        for v in neighbor_indices[u, :k].tolist():
            v_int = int(v)
            if 0 <= v_int < node_count and v_int != u:
                mask[u, v_int] = True
    return mask


def _semantic_topk_mask(
    semantic_similarity: FloatArray,
    semantic_topk: int,
    min_similarity: float,
    *,
    semantic_filter_mode: str = "absolute",
    semantic_relative_threshold: float = 0.5,
    semantic_anchor_floor: float = 0.3,
) -> NDArray[np.bool_]:
    n = int(semantic_similarity.shape[0])
    mask = np.zeros((n, n), dtype=bool)
    if n <= 1:
        return mask
    k = min(max(0, int(semantic_topk)), n - 1)
    if k <= 0:
        return mask
    sim = np.asarray(semantic_similarity, dtype=float).copy()
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    threshold = float(min_similarity)
    mode = str(semantic_filter_mode).strip().lower()
    anchors = np.max(np.where(np.isfinite(sim), sim, -np.inf), axis=1)
    anchors = np.where(np.isfinite(anchors), anchors, 0.0)
    rel_threshold = max(float(semantic_relative_threshold), 1e-9)
    anchor_floor = float(np.clip(semantic_anchor_floor, 0.0, 1.0))
    for u in range(n):
        for slot, v in enumerate(idx[u].tolist()):
            support = float(part[u, slot])
            if not (np.isfinite(support) and support > threshold):
                continue
            if mode == "contrastive":
                v_int = int(v)
                anchor = max(float(anchors[u]), float(anchors[v_int]))
                if anchor <= anchor_floor:
                    continue
                relative_support = support / max(anchor, 1e-9)
                if relative_support < rel_threshold:
                    continue
                mask[u, int(v)] = True
                continue
            mask[u, int(v)] = True
    return mask


def _select_threshold(norm_lengths: FloatArray, jump_ratio: float, fallback_scale: float) -> tuple[float, float]:
    if norm_lengths.size == 0:
        return float(max(1.0, fallback_scale)), 1.0
    ordered = np.sort(np.maximum(norm_lengths.astype(float), _EPS))
    if ordered.size == 1:
        return float(_NO_CLIFF_THRESHOLD), 1.0

    ratios = ordered[1:] / ordered[:-1]
    best_idx = int(np.argmax(ratios))
    best_ratio = float(ratios[best_idx])
    if best_ratio >= float(jump_ratio):
        threshold = float(np.sqrt(ordered[best_idx] * ordered[best_idx + 1]))
        return threshold, best_ratio

    # No clear cliff means there is not enough evidence for a coarser merge yet.
    return float(_NO_CLIFF_THRESHOLD), best_ratio


def _component_labels(node_count: int, edges: IntArray) -> IntArray:
    parent = np.arange(node_count, dtype=int)
    rank = np.zeros(node_count, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for u, v in edges.tolist():
        union(int(u), int(v))

    roots = np.array([find(i) for i in range(node_count)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


def estimate_rg_significance(
    active_tokens: IntArray,
    centroids: FloatArray,
    *,
    knn_k: int = 3,
    jump_ratio: float = 1.75,
    fallback_scale: float = 1.5,
    local_scales_override: FloatArray | None = None,
    semantic_similarity: FloatArray | None = None,
    semantic_penalty_scale: float = 0.0,
    semantic_similarity_floor: float = 0.0,
    semantic_penalty_mode: str = "absolute",
    semantic_relative_threshold: float = 0.5,
    semantic_anchor_floor: float = 0.3,
    semantic_edge_mode: str = "none",
    semantic_topk: int = 0,
) -> RGSignificanceResult:
    tokens = np.asarray(active_tokens, dtype=int)
    points = np.asarray(centroids, dtype=float)
    n = int(points.shape[0])
    if n == 0:
        return RGSignificanceResult(
            active_tokens=tokens,
            local_scales=np.zeros(0, dtype=float),
            threshold_norm=float(fallback_scale),
            threshold_distance=0.0,
            best_jump_ratio=1.0,
            mst_edges=np.zeros((0, 2), dtype=int),
            mst_lengths=np.zeros(0, dtype=float),
            mst_normalized_lengths=np.zeros(0, dtype=float),
            component_labels=np.zeros(0, dtype=int),
            component_count=0,
        )
    if n == 1:
        return RGSignificanceResult(
            active_tokens=tokens,
            local_scales=np.ones(1, dtype=float),
            threshold_norm=float(fallback_scale),
            threshold_distance=float(fallback_scale),
            best_jump_ratio=1.0,
            mst_edges=np.zeros((0, 2), dtype=int),
            mst_lengths=np.zeros(0, dtype=float),
            mst_normalized_lengths=np.zeros(0, dtype=float),
            component_labels=np.zeros(1, dtype=int),
            component_count=1,
        )

    neighbor_k = min(max(1, int(knn_k)), n - 1)
    geometry_distances = _pairwise_distances(points)
    semantic_edge_mode_norm = str(semantic_edge_mode).strip().lower()
    semantic_topk_value = max(0, int(semantic_topk))
    use_semantic_penalty = semantic_similarity is not None and max(float(semantic_penalty_scale), 0.0) > 0.0
    use_semantic_edge_filter = (
        semantic_similarity is not None
        and semantic_edge_mode_norm != "none"
        and semantic_topk_value > 0
    )
    if use_semantic_penalty:
        effective_distances = _semantic_adjusted_distances(
            geometry_distances,
            np.asarray(semantic_similarity, dtype=float),
            semantic_penalty_scale=float(semantic_penalty_scale),
            semantic_similarity_floor=float(semantic_similarity_floor),
            semantic_penalty_mode=str(semantic_penalty_mode),
            semantic_relative_threshold=float(semantic_relative_threshold),
            semantic_anchor_floor=float(semantic_anchor_floor),
        )
    else:
        effective_distances = geometry_distances

    if use_semantic_penalty or use_semantic_edge_filter:
        neighbor_indices, neighbor_distances = _dense_knn_query_from_distances(effective_distances, neighbor_k)
        local_neighbor_distances = _dense_knn_query_from_distances(geometry_distances, neighbor_k)[1]
    else:
        neighbor_indices, neighbor_distances = _knn_query(points, neighbor_k)
        local_neighbor_distances = neighbor_distances

    topology_candidate_edges = np.zeros((0, 2), dtype=int)
    topology_candidate_lengths = np.zeros(0, dtype=float)
    if use_semantic_edge_filter:
        if semantic_edge_mode_norm != "intersect":
            raise ValueError(f"Unsupported semantic_edge_mode: {semantic_edge_mode}")
        geometry_neighbor_mask = _neighbor_mask_from_indices(neighbor_indices, n, neighbor_k)
        semantic_neighbor_mask = _semantic_topk_mask(
            np.asarray(semantic_similarity, dtype=float),
            semantic_topk_value,
            float(semantic_similarity_floor),
            semantic_filter_mode=str(semantic_penalty_mode),
            semantic_relative_threshold=float(semantic_relative_threshold),
            semantic_anchor_floor=float(semantic_anchor_floor),
        )
        candidate_mask = geometry_neighbor_mask & semantic_neighbor_mask & semantic_neighbor_mask.T
        topology_candidate_edges, topology_candidate_lengths = _candidate_edges_from_mask(
            effective_distances,
            candidate_mask,
        )
        mst_edges, mst_lengths = _sparse_mst(n, topology_candidate_edges, topology_candidate_lengths)
    elif use_semantic_penalty:
        mst_edges, mst_lengths = _dense_mst(effective_distances)
    else:
        mst_edges, mst_lengths = _sparse_mst_from_points(points, max(neighbor_k, 3))

    if local_scales_override is None:
        local_scales = _estimate_local_scales_from_neighbors(local_neighbor_distances, n)
    else:
        local_scales = np.asarray(local_scales_override, dtype=float).reshape(-1)
        if local_scales.shape[0] != n:
            raise ValueError("local_scales_override must match the number of active tokens.")
        local_scales = np.maximum(local_scales, _EPS)
    mst_norm = np.array(
        [
            mst_lengths[idx] / max(local_scales[int(u)], local_scales[int(v)], _EPS)
            for idx, (u, v) in enumerate(mst_edges.tolist())
        ],
        dtype=float,
    )
    threshold_norm, best_jump_ratio = _select_threshold(mst_norm, jump_ratio, fallback_scale)
    threshold_distance = float(threshold_norm * np.median(local_scales))

    keep_edges: set[tuple[int, int]] = set()
    for idx, (u, v) in enumerate(mst_edges.tolist()):
        if mst_norm[idx] <= threshold_norm:
            keep_edges.add((min(int(u), int(v)), max(int(u), int(v))))

    if use_semantic_edge_filter:
        for idx, (u, v) in enumerate(topology_candidate_edges.tolist()):
            raw = float(topology_candidate_lengths[idx])
            norm = raw / max(local_scales[int(u)], local_scales[int(v)], _EPS)
            if norm <= threshold_norm:
                keep_edges.add((min(int(u), int(v)), max(int(u), int(v))))
    else:
        for u, v in _mutual_knn_pairs_from_neighbors(neighbor_indices, neighbor_k).tolist():
            raw = float(effective_distances[int(u), int(v)])
            norm = raw / max(local_scales[int(u)], local_scales[int(v)], _EPS)
            if norm <= threshold_norm:
                keep_edges.add((min(int(u), int(v)), max(int(u), int(v))))

    component_labels = _component_labels(
        n,
        np.asarray(sorted(keep_edges), dtype=int) if keep_edges else np.zeros((0, 2), dtype=int),
    )
    component_count = int(len(np.unique(component_labels)))
    return RGSignificanceResult(
        active_tokens=tokens,
        local_scales=local_scales,
        threshold_norm=float(threshold_norm),
        threshold_distance=threshold_distance,
        best_jump_ratio=float(best_jump_ratio),
        mst_edges=mst_edges,
        mst_lengths=mst_lengths,
        mst_normalized_lengths=mst_norm,
        component_labels=component_labels,
        component_count=component_count,
    )
