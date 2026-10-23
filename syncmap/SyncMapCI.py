#!/usr/bin/env python3
"""
Research scaffold for a causal / directed-graph version of SyncMap.

This module does not yet implement a full "causal SyncMap". Instead it
provides the minimal pieces needed to study the problem rigorously:

1. Generate synthetic DAGs.
2. Sample directed trajectories on the DAG.
3. Recover directed edges from observed transitions as a baseline.
4. Convert trajectories into edge tokens / line-graph structure for future
   relation-centric SyncMap experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.intp]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EdgeRecoveryMetrics:
    precision: float
    recall: float
    f1: float
    shd: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class GraphWalkBenchmarkResult:
    dataset: str
    sequence_length: int
    node_count: int
    true_edge_count: int
    is_dag: bool
    argmax_state_accuracy: float
    direct_metrics: EdgeRecoveryMetrics
    argmax_metrics: EdgeRecoveryMetrics


def _as_binary_adjacency(adjacency: Sequence[Sequence[float]] | NDArray[np.float64]) -> IntArray:
    arr = np.asarray(adjacency)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    support = (arr > 0).astype(int)
    np.fill_diagonal(support, 0)
    return support


def _row_normalize(matrix: Sequence[Sequence[float]] | NDArray[np.float64]) -> FloatArray:
    arr = np.asarray(matrix, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)


class DirectedTransitionStats:
    """Online directed transition counts.

    Negative state IDs are treated as explicit boundary markers, so flattened
    trajectory sequences can be passed in without creating spurious restart
    edges across trajectory boundaries.
    """

    def __init__(self, n_nodes: int) -> None:
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        self.n_nodes = int(n_nodes)
        self.transition_counts: FloatArray = np.zeros((self.n_nodes, self.n_nodes), dtype=float)

    def reset(self) -> None:
        self.transition_counts.fill(0.0)

    def update_transition(self, src: int, dst: int, weight: float = 1.0) -> None:
        if src < 0 or dst < 0:
            return
        if src >= self.n_nodes or dst >= self.n_nodes:
            raise ValueError(f"state out of range: src={src}, dst={dst}, n_nodes={self.n_nodes}")
        self.transition_counts[int(src), int(dst)] += float(weight)

    def update_sequence(self, states: Sequence[int] | IntArray) -> None:
        arr = np.asarray(states, dtype=int).reshape(-1)
        if arr.size < 2:
            return
        for src, dst in zip(arr[:-1].tolist(), arr[1:].tolist()):
            self.update_transition(int(src), int(dst))

    def update_trajectories(self, trajectories: Iterable[Sequence[int] | IntArray]) -> None:
        for traj in trajectories:
            self.update_sequence(traj)

    def transition_probabilities(self, smoothing: float = 0.0) -> FloatArray:
        counts = np.asarray(self.transition_counts, dtype=float)
        if smoothing > 0.0:
            counts = counts + float(smoothing)
            np.fill_diagonal(counts, np.diag(self.transition_counts))
        return _row_normalize(counts)

    def adjacency_support(
        self,
        *,
        min_count: float = 1.0,
        min_probability: float = 0.0,
        exclude_self_loops: bool = True,
    ) -> IntArray:
        support = self.transition_counts >= float(min_count)
        if min_probability > 0.0:
            support &= self.transition_probabilities() >= float(min_probability)
        if exclude_self_loops:
            np.fill_diagonal(support, False)
        return support.astype(int)


def sample_random_dag(
    n_nodes: int,
    edge_probability: float = 0.3,
    *,
    random_state: int | None = None,
    ensure_backbone: bool = True,
) -> IntArray:
    """Sample a DAG in topological order 0..n-1.

    If ``ensure_backbone=True``, a chain ``0->1->...->n-1`` is inserted so the
    graph is connected from source to sink before adding extra forward edges.
    """

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be in [0, 1]")

    rng = np.random.default_rng(random_state)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)

    if ensure_backbone and n_nodes >= 2:
        for i in range(n_nodes - 1):
            adjacency[i, i + 1] = 1

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] == 1:
                continue
            if rng.random() < edge_probability:
                adjacency[i, j] = 1

    return adjacency


def dag_sources(adjacency: Sequence[Sequence[float]] | NDArray[np.float64]) -> IntArray:
    support = _as_binary_adjacency(adjacency)
    indegree = support.sum(axis=0)
    return np.where(indegree == 0)[0].astype(int)


def dag_sinks(adjacency: Sequence[Sequence[float]] | NDArray[np.float64]) -> IntArray:
    support = _as_binary_adjacency(adjacency)
    outdegree = support.sum(axis=1)
    return np.where(outdegree == 0)[0].astype(int)


def is_acyclic_adjacency(adjacency: Sequence[Sequence[float]] | NDArray[np.float64]) -> bool:
    """Return True iff the directed graph is acyclic."""

    support = _as_binary_adjacency(adjacency)
    indegree = support.sum(axis=0).astype(int)
    queue = [int(node) for node in np.where(indegree == 0)[0].tolist()]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        children = np.where(support[node] > 0)[0]
        for child in children.tolist():
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))

    return visited == support.shape[0]


def sample_dag_trajectories(
    adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
    num_trajectories: int,
    *,
    random_state: int | None = None,
    start_nodes: Sequence[int] | None = None,
    max_steps: int | None = None,
) -> list[IntArray]:
    """Sample source-to-sink trajectories on a DAG.

    The adjacency may be binary or weighted. Positive row entries are
    interpreted as outgoing transition weights.
    """

    if num_trajectories < 0:
        raise ValueError("num_trajectories must be non-negative")

    weights = np.asarray(adjacency, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    n_nodes = weights.shape[0]
    rng = np.random.default_rng(random_state)
    sources = np.asarray(start_nodes if start_nodes is not None else dag_sources(weights), dtype=int)
    if sources.size == 0:
        raise ValueError("DAG has no source nodes")

    trajectories: list[IntArray] = []
    for _ in range(num_trajectories):
        current = int(rng.choice(sources))
        traj = [current]

        while True:
            if max_steps is not None and len(traj) >= int(max_steps):
                break
            outgoing = np.where(weights[current] > 0)[0]
            if outgoing.size == 0:
                break
            probs = weights[current, outgoing].astype(float)
            probs = probs / probs.sum()
            current = int(rng.choice(outgoing, p=probs))
            traj.append(current)

        trajectories.append(np.asarray(traj, dtype=int))

    return trajectories


def flatten_trajectories(
    trajectories: Iterable[Sequence[int] | IntArray],
    *,
    boundary_token: int = -1,
) -> IntArray:
    flat: list[int] = []
    first = True
    for traj in trajectories:
        arr = np.asarray(traj, dtype=int).reshape(-1)
        if arr.size == 0:
            continue
        if not first:
            flat.append(int(boundary_token))
        flat.extend(arr.tolist())
        first = False
    return np.asarray(flat, dtype=int)


def recover_adjacency_from_trajectories(
    trajectories: Iterable[Sequence[int] | IntArray],
    n_nodes: int,
    *,
    min_count: float = 1.0,
    min_probability: float = 0.0,
) -> IntArray:
    stats = DirectedTransitionStats(n_nodes)
    stats.update_trajectories(trajectories)
    return stats.adjacency_support(
        min_count=min_count,
        min_probability=min_probability,
        exclude_self_loops=True,
    )


def recover_adjacency_from_sequence(
    sequence: Sequence[int] | IntArray,
    n_nodes: int,
    *,
    min_count: float = 1.0,
    min_probability: float = 0.0,
) -> IntArray:
    stats = DirectedTransitionStats(n_nodes)
    stats.update_sequence(sequence)
    return stats.adjacency_support(
        min_count=min_count,
        min_probability=min_probability,
        exclude_self_loops=True,
    )


def structural_hamming_distance(
    true_adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
    predicted_adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
) -> int:
    truth = _as_binary_adjacency(true_adjacency).astype(bool)
    pred = _as_binary_adjacency(predicted_adjacency).astype(bool)
    if truth.shape != pred.shape:
        raise ValueError("true and predicted adjacency must have the same shape")
    return int(np.logical_xor(truth, pred).sum())


def evaluate_edge_recovery(
    true_adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
    predicted_adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
) -> EdgeRecoveryMetrics:
    truth = _as_binary_adjacency(true_adjacency).astype(bool)
    pred = _as_binary_adjacency(predicted_adjacency).astype(bool)
    if truth.shape != pred.shape:
        raise ValueError("true and predicted adjacency must have the same shape")

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return EdgeRecoveryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        shd=structural_hamming_distance(truth, pred),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def build_asymmetric_node_features(
    transition_counts: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    include_marginals: bool = True,
) -> FloatArray:
    """Build directed context features for each node.

    For node ``i``, the feature is

    ``phi(i) = [P(next | i), P(prev | i), out_count(i), in_count(i)]``.

    The outgoing and incoming parts preserve directionality, unlike symmetric
    co-activation matrices.
    """

    counts = np.asarray(transition_counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("transition_counts must be a square matrix")

    outgoing = _row_normalize(counts)
    incoming = _row_normalize(counts.T)
    parts = [outgoing, incoming]

    if include_marginals:
        out_mass = counts.sum(axis=1, keepdims=True)
        in_mass = counts.sum(axis=0, keepdims=True).T
        parts.extend([out_mass, in_mass])

    return np.concatenate(parts, axis=1)


def trajectory_to_edge_tokens(trajectory: Sequence[int] | IntArray) -> list[tuple[int, int]]:
    arr = np.asarray(trajectory, dtype=int).reshape(-1)
    if arr.size < 2:
        return []
    tokens: list[tuple[int, int]] = []
    for src, dst in zip(arr[:-1].tolist(), arr[1:].tolist()):
        if src < 0 or dst < 0:
            continue
        tokens.append((int(src), int(dst)))
    return tokens


def trajectories_to_edge_tokens(
    trajectories: Iterable[Sequence[int] | IntArray],
) -> list[list[tuple[int, int]]]:
    return [trajectory_to_edge_tokens(traj) for traj in trajectories]


def build_directed_line_graph(
    adjacency: Sequence[Sequence[float]] | NDArray[np.float64],
) -> tuple[IntArray, list[tuple[int, int]]]:
    """Return the directed line graph of a directed graph.

    Each original edge ``(i, j)`` becomes a line-graph node. A directed edge is
    created from token ``(i, j)`` to token ``(j, k)`` whenever both edges exist
    in the original graph.
    """

    support = _as_binary_adjacency(adjacency)
    edge_array = np.argwhere(support > 0)
    edge_tokens = [(int(src), int(dst)) for src, dst in edge_array.tolist()]
    token_index = {token: idx for idx, token in enumerate(edge_tokens)}

    line_graph = np.zeros((len(edge_tokens), len(edge_tokens)), dtype=int)
    for idx, (_, dst) in enumerate(edge_tokens):
        next_nodes = np.where(support[dst] > 0)[0]
        for next_node in next_nodes.tolist():
            next_idx = token_index.get((dst, int(next_node)))
            if next_idx is not None:
                line_graph[idx, next_idx] = 1

    return line_graph, edge_tokens


def benchmark_graphwalk_dataset(
    filename: str,
    *,
    sequence_length: int = 5000,
    time_delay: int = 10,
    random_state: int = 0,
    min_count: float = 1.0,
) -> GraphWalkBenchmarkResult:
    """Benchmark a GraphWalk dataset under the SyncMapCI recovery view.

    This uses two recoveries:

    1. Direct state sequence from ``GraphWalkTest.input_class``.
    2. Naive decoding by ``argmax(input_sequence)``.

    If (2) is already near-perfect, the observation model is probably too easy
    to justify a more complex latent-state method.
    """

    from test.GraphWalkTest import GraphWalkTest

    np.random.seed(int(random_state))
    env = GraphWalkTest(time_delay=int(time_delay), filename=filename)
    input_sequence, input_class = env.getSequence(int(sequence_length))

    true_adjacency = (np.asarray(env.a) > 0).astype(int)
    observed_states = np.asarray(input_class, dtype=int).reshape(-1)
    decoded_states = np.asarray(np.argmax(np.asarray(input_sequence), axis=1), dtype=int).reshape(-1)

    direct_pred = recover_adjacency_from_sequence(
        observed_states,
        env.output_size,
        min_count=min_count,
    )
    argmax_pred = recover_adjacency_from_sequence(
        decoded_states,
        env.output_size,
        min_count=min_count,
    )

    return GraphWalkBenchmarkResult(
        dataset=str(filename),
        sequence_length=int(sequence_length),
        node_count=int(env.output_size),
        true_edge_count=int(true_adjacency.sum()),
        is_dag=is_acyclic_adjacency(true_adjacency),
        argmax_state_accuracy=float(np.mean(decoded_states == observed_states)),
        direct_metrics=evaluate_edge_recovery(true_adjacency, direct_pred),
        argmax_metrics=evaluate_edge_recovery(true_adjacency, argmax_pred),
    )
