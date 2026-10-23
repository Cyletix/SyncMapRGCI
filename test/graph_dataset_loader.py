from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class LoadedGraphDataset:
    graph: nx.DiGraph
    transition: np.ndarray[Any, Any]
    label_levels: np.ndarray[Any, Any]
    observed_nodes: list[str]
    is_hierarchical: bool


@dataclass(frozen=True)
class ObservedLabelDataset:
    observed_nodes: list[str]
    node_labels: dict[str, Any]
    label_levels: np.ndarray[Any, Any]
    is_hierarchical: bool


@dataclass(frozen=True)
class DotDatasetSignature:
    observed_node_count: int
    label_depth: int
    has_internal_nodes: bool
    is_hierarchical: bool


def _normalize_label(raw: Any) -> str:
    text = str(raw).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return text.strip()


def _parse_label_tuple(raw: Any) -> tuple[int, ...]:
    text = _normalize_label(raw)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Empty label is not supported: {raw!r}")
    return tuple(int(part) for part in parts)


def _sort_key(node_id: str) -> tuple[int, Any]:
    text = str(node_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _encode_levels(label_tuples: list[tuple[int, ...]]) -> np.ndarray[Any, Any]:
    width = max(len(parts) for parts in label_tuples)
    raw = np.zeros((len(label_tuples), width), dtype=int)
    for row_idx, parts in enumerate(label_tuples):
        raw[row_idx, : len(parts)] = np.asarray(parts, dtype=int)

    encoded = np.zeros_like(raw)
    for col_idx in range(raw.shape[1]):
        values = raw[:, col_idx]
        uniq = sorted(set(int(v) for v in values.tolist()))
        mapping = {value: idx for idx, value in enumerate(uniq)}
        encoded[:, col_idx] = np.asarray([mapping[int(v)] for v in values.tolist()], dtype=int)
    return encoded


def _shared_suffix_length(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> int:
    shared = 0
    for av, bv in zip(a[::-1], b[::-1]):
        if int(av) != int(bv):
            break
        shared += 1
    return shared


def _build_hierarchy_transition(
    label_levels: np.ndarray[Any, Any],
    *,
    same_factor: float,
    cross_weight: float,
) -> np.ndarray[Any, Any]:
    n = int(label_levels.shape[0])
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    a = np.zeros((n, n), dtype=np.float64)
    base = float(max(cross_weight, 1e-6))
    growth = float(max(same_factor, 1.0))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            shared = _shared_suffix_length(label_levels[i], label_levels[j])
            a[i, j] = base * (growth ** shared)

    row_sums = a.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return a / row_sums


def _build_observed_subgraph(
    raw_graph: nx.DiGraph,
    observed_nodes: list[str],
    transition: np.ndarray[Any, Any],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    label_map = dict(raw_graph.nodes(data="label"))
    for node in observed_nodes:
        graph.add_node(node, label=label_map.get(node))
    rows, cols = np.where(transition > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(observed_nodes[int(i)], observed_nodes[int(j)], weight=float(transition[int(i), int(j)]))
    return graph


def load_dot_observed_labels(filename: str) -> ObservedLabelDataset:
    raw_graph = nx.DiGraph(nx.nx_agraph.read_dot(filename))
    label_map = dict(raw_graph.nodes(data="label"))

    labeled_nodes = [str(node) for node, label in raw_graph.nodes(data="label") if label is not None]
    observed_nodes = sorted(labeled_nodes, key=_sort_key)
    if not observed_nodes:
        raise ValueError(f"Dataset does not contain labeled observable nodes: {filename}")

    label_tuples = [_parse_label_tuple(label_map[node]) for node in observed_nodes]
    label_levels = _encode_levels(label_tuples)
    is_hierarchical = (len(observed_nodes) != raw_graph.number_of_nodes()) or any(len(parts) > 1 for parts in label_tuples)
    observed_label_map = {node: label_map[node] for node in observed_nodes}
    return ObservedLabelDataset(
        observed_nodes=observed_nodes,
        node_labels=observed_label_map,
        label_levels=label_levels,
        is_hierarchical=is_hierarchical,
    )


def inspect_dot_dataset(filename: str) -> DotDatasetSignature:
    raw_graph = nx.DiGraph(nx.nx_agraph.read_dot(filename))
    observed = load_dot_observed_labels(filename)
    observed_node_count = int(len(observed.observed_nodes))
    label_depth = int(observed.label_levels.shape[1])
    has_internal_nodes = bool(raw_graph.number_of_nodes() != observed_node_count)
    return DotDatasetSignature(
        observed_node_count=observed_node_count,
        label_depth=label_depth,
        has_internal_nodes=has_internal_nodes,
        is_hierarchical=bool(observed.is_hierarchical),
    )


def load_dot_graph_dataset(
    filename: str,
    *,
    hierarchy_same_factor: float = 4.0,
    hierarchy_cross_weight: float = 1.0,
) -> LoadedGraphDataset:
    raw_graph = nx.DiGraph(nx.nx_agraph.read_dot(filename))
    observed = load_dot_observed_labels(filename)
    observed_nodes = observed.observed_nodes
    label_levels = observed.label_levels
    is_hierarchical = observed.is_hierarchical

    if is_hierarchical:
        transition = _build_hierarchy_transition(
            label_levels,
            same_factor=hierarchy_same_factor,
            cross_weight=hierarchy_cross_weight,
        )
        graph = _build_observed_subgraph(raw_graph, observed_nodes, transition)
        return LoadedGraphDataset(
            graph=graph,
            transition=transition,
            label_levels=label_levels,
            observed_nodes=observed_nodes,
            is_hierarchical=True,
        )

    adj_matrix = nx.adjacency_matrix(raw_graph, nodelist=observed_nodes)
    transition = np.array(adj_matrix.toarray(), dtype=np.float64)  # type: ignore[arg-type]
    for row_idx in range(transition.shape[0]):
        accum = float(transition[row_idx].sum())
        if accum <= 0:
            raise ValueError(f"Node {observed_nodes[row_idx]!r} has no outgoing transitions in {filename}")
        transition[row_idx] /= accum

    return LoadedGraphDataset(
        graph=raw_graph,
        transition=transition,
        label_levels=label_levels,
        observed_nodes=observed_nodes,
        is_hierarchical=False,
    )
