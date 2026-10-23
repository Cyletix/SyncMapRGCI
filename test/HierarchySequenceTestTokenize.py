from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .GraphWalkTest import to_categorical
from .HierarchySequenceTest import HierarchySequenceTest
from .graph_dataset_loader import load_dot_observed_labels


class HierarchySequenceTestTokenize(HierarchySequenceTest):
    """Hierarchy sequence env with an explicit segmentation token.

    This preserves HierarchySequenceTest's leaf-level transition semantics, but
    whenever a transition switches to a different leaf, the environment emits a
    shared extra token first. The actual switch node name is tracked separately
    for diagnostics/live display.
    """

    reset_token_index: int
    leaf_output_size: int
    boundary_mask: np.ndarray[Any, Any]
    episode_ids: np.ndarray[Any, Any]
    switch_node_names: np.ndarray[Any, Any]
    _leaf_transition: np.ndarray[Any, Any]
    _leaf_output_class: int
    _pending_leaf_class: int | None
    _pending_switch_name: str | None
    _current_switch_name: str | None
    _episode_id: int
    _last_boundary: bool
    _leaf_paths: dict[int, list[str]]

    def __init__(
        self,
        time_delay: int,
        filename: str = "graph.dot",
        *,
        leaf_switch_prob: float = 0.45,
        level_switch_decay: float = 0.25,
        support_threshold: float = 1e-6,
    ) -> None:
        self.boundary_mask = np.empty(0, dtype=bool)
        self.episode_ids = np.empty(0, dtype=int)
        self.switch_node_names = np.empty(0, dtype=object)
        self._pending_leaf_class = None
        self._pending_switch_name = None
        self._current_switch_name = None
        self._episode_id = 0
        self._last_boundary = False
        self._leaf_paths = {}
        super().__init__(
            time_delay=time_delay,
            filename=filename,
            leaf_switch_prob=leaf_switch_prob,
            level_switch_decay=level_switch_decay,
            support_threshold=support_threshold,
        )
        self._leaf_output_class = int(np.random.randint(self.leaf_output_size))
        self.output_class = self._leaf_output_class
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0, dtype=int)

    def _follow_unique_path(self, raw_graph: nx.DiGraph, start_node: str) -> list[str]:
        path = [str(start_node)]
        current = str(start_node)
        seen = {current}
        while True:
            succ = list(raw_graph.successors(current))
            if not succ:
                break
            if len(succ) != 1:
                raise ValueError(
                    f"HierarchySequenceTestTokenize expects a tree-like upward graph. "
                    f"Node {current!r} has {len(succ)} successors."
                )
            current = str(succ[0])
            if current in seen:
                raise ValueError(f"Cycle detected while resolving hierarchy path from {start_node!r}")
            path.append(current)
            seen.add(current)
        return path

    def _switch_node_name(self, src_leaf_idx: int, dst_leaf_idx: int) -> str | None:
        src_path = self._leaf_paths.get(int(src_leaf_idx))
        dst_path = self._leaf_paths.get(int(dst_leaf_idx))
        if not src_path or not dst_path:
            return None
        shared_name: str | None = None
        for src_name, dst_name in zip(src_path[::-1], dst_path[::-1]):
            if str(src_name) != str(dst_name):
                break
            shared_name = str(src_name)
        return shared_name

    def _build_tokenized_support(self) -> np.ndarray[Any, Any]:
        n = int(self.leaf_output_size)
        token = int(self.reset_token_index)
        support = np.zeros((n + 1, n + 1), dtype=np.float64)
        offdiag = np.asarray(self._leaf_transition, dtype=np.float64).copy()
        np.fill_diagonal(offdiag, 0.0)

        for src in range(n):
            stay_prob = float(self._leaf_transition[src, src])
            switch_prob = float(offdiag[src].sum())
            support[src, src] = stay_prob
            support[src, token] = switch_prob

        token_row = offdiag.sum(axis=0)
        total = float(token_row.sum())
        if total > 0.0:
            support[token, :n] = token_row / total
        else:
            support[token, :n] = 1.0 / float(max(n, 1))
        return support

    def _load_graph(self, filename: str, allow_resize: bool) -> None:
        expected_size = int(getattr(self, "output_size", 0))
        super()._load_graph(filename, allow_resize=allow_resize)

        filename_resolved = self._resolve_path(filename)
        raw_graph = nx.DiGraph(nx.nx_agraph.read_dot(filename_resolved))
        observed = load_dot_observed_labels(filename_resolved)
        if not observed.is_hierarchical:
            raise ValueError(
                f"HierarchySequenceTestTokenize expects a hierarchy dataset with internal switch nodes: {filename}"
            )

        self.leaf_output_size = int(self.output_size)
        self.reset_token_index = int(self.leaf_output_size)
        self._leaf_transition = np.asarray(self.a, dtype=np.float64).copy()
        base_levels = np.asarray(self.true_label_levels, dtype=int)
        extra_levels = (np.max(base_levels, axis=0) + 1).reshape(1, -1)

        tokenized_output_size = int(self.leaf_output_size + 1)
        if expected_size > 0 and not allow_resize and tokenized_output_size != expected_size:
            raise ValueError(
                f"switch_dataset requires same output_size ({expected_size}), got {tokenized_output_size}"
            )

        self.true_label_levels = np.vstack([base_levels, extra_levels])
        self.true_label = np.asarray(self.true_label_levels[:, 0], dtype=int)
        self.output_size = tokenized_output_size
        self.a = self._build_tokenized_support()
        self.is_hierarchical = True
        self.hierarchy_depth = int(self.true_label_levels.shape[1])
        self._leaf_paths = {
            idx: self._follow_unique_path(raw_graph, node_name)
            for idx, node_name in enumerate(observed.observed_nodes)
        }

    def switch_dataset(self, filename: str) -> None:
        self._load_graph(filename, allow_resize=False)
        self.time_counter = 0
        self._leaf_output_class = int(np.random.randint(self.leaf_output_size))
        self.output_class = self._leaf_output_class
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0, dtype=int)
        self.boundary_mask = np.empty(0, dtype=bool)
        self.episode_ids = np.empty(0, dtype=int)
        self.switch_node_names = np.empty(0, dtype=object)
        self._pending_leaf_class = None
        self._pending_switch_name = None
        self._current_switch_name = None
        self._episode_id = 0
        self._last_boundary = False

    def currentSwitchNodeName(self) -> str | None:
        return self._current_switch_name

    def getInput(self, reset: bool = False) -> np.ndarray[Any, Any]:
        if reset:
            self.time_counter = 0
            self._leaf_output_class = int(np.random.randint(self.leaf_output_size))
            self.output_class = self._leaf_output_class
            self.previous_output_class = None
            self.previous_previous_output_class = None
            self._pending_leaf_class = None
            self._pending_switch_name = None
            self._current_switch_name = None
            self._episode_id = 0
            self._last_boundary = False

        update = self.updateTimeDelay()
        if not update:
            pass
        elif self._pending_leaf_class is not None:
            self.output_class = int(self._pending_leaf_class)
            self._leaf_output_class = int(self._pending_leaf_class)
            self._pending_leaf_class = None
            self._pending_switch_name = None
            self._current_switch_name = None
            self._last_boundary = False
        else:
            probs = np.asarray(self._leaf_transition[self._leaf_output_class], dtype=float)
            next_leaf = int(np.random.choice(self.leaf_output_size, p=probs))
            if next_leaf != self._leaf_output_class:
                self._pending_leaf_class = next_leaf
                self._pending_switch_name = self._switch_node_name(self._leaf_output_class, next_leaf)
                self.output_class = int(self.reset_token_index)
                self._current_switch_name = self._pending_switch_name
                self._episode_id += 1
                self._last_boundary = True
            else:
                self._leaf_output_class = next_leaf
                self.output_class = next_leaf
                self._current_switch_name = None
                self._last_boundary = False

        noise_intensity = 0.0
        if self.previous_output_class is None or self.previous_output_class == self.output_class:
            input_value = (
                to_categorical(self.output_class, self.output_size) * np.exp(-0.1 * self.time_counter)
                + np.random.randn(self.output_size) * noise_intensity
            )
        else:
            input_value = (
                to_categorical(self.output_class, self.output_size) * np.exp(-0.1 * self.time_counter)
                + np.random.randn(self.output_size) * noise_intensity
                + to_categorical(self.previous_output_class, self.output_size)
                * np.exp(-0.1 * (self.time_counter + self.time_delay))
            )
        return input_value

    def getSequence(self, sequence_size: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        self.input_sequence = np.empty((sequence_size, self.output_size))
        self.input_class = np.empty(sequence_size, dtype=int)
        self.boundary_mask = np.zeros(sequence_size, dtype=bool)
        self.episode_ids = np.zeros(sequence_size, dtype=int)
        self.switch_node_names = np.empty(sequence_size, dtype=object)

        for i in range(sequence_size):
            input_value = self.getInput()
            self.input_class[i] = int(self.output_class)
            self.input_sequence[i] = input_value
            self.boundary_mask[i] = bool(self._last_boundary)
            self.episode_ids[i] = int(self._episode_id)
            self.switch_node_names[i] = self._current_switch_name

        return self.input_sequence, self.input_class

    def getSequenceWithBoundaryLabels(
        self,
        sequence_size: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        sequence, classes = self.getSequence(sequence_size)
        return sequence, classes, self.boundary_mask.copy(), self.episode_ids.copy()

    def trueBoundaryMask(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.boundary_mask, dtype=bool).copy()

    def trueBoundaryIndices(self) -> np.ndarray[Any, Any]:
        return np.where(np.asarray(self.boundary_mask, dtype=bool))[0].astype(int)

    def trueEpisodeIds(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.episode_ids, dtype=int).copy()
