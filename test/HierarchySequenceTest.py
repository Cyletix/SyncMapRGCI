from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .graph_dataset_loader import load_dot_observed_labels
from .GraphWalkTest import GraphWalkTest


FloatArray = np.ndarray[Any, Any]


def _default_level_switch_probs(depth: int, leaf_switch_prob: float, level_switch_decay: float) -> FloatArray:
    probs = np.zeros(depth, dtype=np.float64)
    leaf_prob = float(np.clip(leaf_switch_prob, 1e-6, 0.95))
    decay = float(np.clip(level_switch_decay, 1e-6, 1.0))
    for level in range(depth):
        probs[level] = leaf_prob * (decay ** level)
    return np.clip(probs, 1e-6, 0.95)


def _build_exact_hierarchy_transition(
    label_levels: FloatArray,
    level_switch_probs: FloatArray,
) -> np.ndarray[Any, Any]:
    n_leaves, depth = label_levels.shape
    if n_leaves <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    if depth <= 0:
        return np.eye(n_leaves, dtype=np.float64)

    coarse_levels = np.asarray(label_levels[:, ::-1], dtype=int)
    coarse_switch_probs = np.asarray(level_switch_probs[::-1], dtype=np.float64)

    children: list[dict[int, int]] = [dict()]
    terminal_leaf_lists: list[list[int]] = [[]]
    leaf_terminal_nodes = np.zeros(n_leaves, dtype=int)
    leaf_values = np.zeros((n_leaves, depth), dtype=int)

    for leaf_idx in range(n_leaves):
        node_idx = 0
        for coarse_idx, raw_value in enumerate(coarse_levels[leaf_idx]):
            value = int(raw_value)
            leaf_values[leaf_idx, coarse_idx] = value
            child_idx = children[node_idx].get(value)
            if child_idx is None:
                child_idx = len(children)
                children[node_idx][value] = child_idx
                children.append({})
                terminal_leaf_lists.append([])
            node_idx = child_idx
        terminal_leaf_lists[node_idx].append(int(leaf_idx))
        leaf_terminal_nodes[leaf_idx] = node_idx

    node_count = len(children)
    resample_distributions = np.zeros((node_count, n_leaves), dtype=np.float64)

    for node_idx in range(node_count - 1, -1, -1):
        leaf_list = terminal_leaf_lists[node_idx]
        if leaf_list:
            weight = 1.0 / float(len(leaf_list))
            resample_distributions[node_idx, np.asarray(leaf_list, dtype=int)] = weight
            continue

        child_nodes = list(children[node_idx].values())
        if not child_nodes:
            continue
        child_weight = 1.0 / float(len(child_nodes))
        for child_idx in child_nodes:
            resample_distributions[node_idx] += resample_distributions[child_idx] * child_weight

    transition = np.zeros((n_leaves, n_leaves), dtype=np.float64)
    for leaf_idx in range(n_leaves):
        row = transition[leaf_idx]
        continue_prob = 1.0
        node_idx = 0

        for coarse_idx in range(depth):
            child_map = children[node_idx]
            current_value = int(leaf_values[leaf_idx, coarse_idx])
            candidate_count = len(child_map)

            if candidate_count > 1:
                p_switch = float(np.clip(coarse_switch_probs[coarse_idx], 0.0, 1.0))
                if p_switch > 0.0:
                    switch_prob = continue_prob * p_switch / float(candidate_count - 1)
                    for value, child_idx in child_map.items():
                        if int(value) == current_value:
                            continue
                        row += resample_distributions[child_idx] * switch_prob
                    continue_prob *= (1.0 - p_switch)

            node_idx = child_map[current_value]

        leaf_list = terminal_leaf_lists[leaf_terminal_nodes[leaf_idx]]
        if leaf_list:
            row[np.asarray(leaf_list, dtype=int)] += continue_prob / float(len(leaf_list))

        total = float(row.sum())
        if total <= 0.0:
            row[:] = 1.0 / float(n_leaves)
        elif abs(total - 1.0) > 1e-12:
            row /= total

    return transition


def _build_support_graph(
    observed_nodes: list[str],
    node_labels: dict[str, Any],
    transition: FloatArray,
    support_threshold: float,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in observed_nodes:
        graph.add_node(node, label=node_labels[node])
    rows, cols = np.where(transition > float(support_threshold))
    for row_idx, col_idx in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(
            observed_nodes[int(row_idx)],
            observed_nodes[int(col_idx)],
            weight=float(transition[int(row_idx), int(col_idx)]),
        )
    return graph


class HierarchySequenceTest(GraphWalkTest):
    true_label_levels: np.ndarray[Any, Any]
    is_hierarchical: bool
    hierarchy_depth: int
    level_switch_probs: FloatArray

    def __init__(
        self,
        time_delay: int,
        filename: str = "graph.dot",
        *,
        leaf_switch_prob: float = 0.45,
        level_switch_decay: float = 0.25,
        support_threshold: float = 1e-6,
    ) -> None:
        self._hierarchy_params = {
            "leaf_switch_prob": float(leaf_switch_prob),
            "level_switch_decay": float(level_switch_decay),
            "support_threshold": float(support_threshold),
        }
        super().__init__(time_delay=time_delay, filename=filename)

    def _load_graph(self, filename: str, allow_resize: bool) -> None:
        filename_resolved = self._resolve_path(filename)
        observed = load_dot_observed_labels(filename_resolved)
        if not observed.is_hierarchical:
            raise ValueError(f"HierarchySequenceTest expects a hierarchy dataset: {filename}")

        label_levels_new = np.asarray(observed.label_levels, dtype=int)
        output_size_new = int(label_levels_new.shape[0])
        level_switch_probs = _default_level_switch_probs(
            label_levels_new.shape[1],
            self._hierarchy_params["leaf_switch_prob"],
            self._hierarchy_params["level_switch_decay"],
        )
        transition = _build_exact_hierarchy_transition(label_levels_new, level_switch_probs)
        graph = _build_support_graph(
            observed.observed_nodes,
            observed.node_labels,
            transition,
            support_threshold=self._hierarchy_params["support_threshold"],
        )

        if hasattr(self, "output_size") and not allow_resize and output_size_new != self.output_size:
            raise ValueError(
                f"switch_dataset requires same output_size ({self.output_size}), got {output_size_new}"
            )

        self.g = graph
        self.true_label_levels = label_levels_new
        self.true_label = np.asarray(label_levels_new[:, 0], dtype=int)
        self.output_size = output_size_new
        self.a = transition
        self.is_hierarchical = True
        self.hierarchy_depth = int(label_levels_new.shape[1])
        self.level_switch_probs = level_switch_probs

    def trueLabel(self, level: int = 0) -> np.ndarray[Any, Any]:
        level_idx = int(np.clip(level, 0, self.true_label_levels.shape[1] - 1))
        return self.true_label_levels[:, level_idx]

    def get_true_label_levels(self) -> np.ndarray[Any, Any]:
        return self.true_label_levels.copy()

    def get_level_switch_probs(self) -> np.ndarray[Any, Any]:
        return self.level_switch_probs.copy()
