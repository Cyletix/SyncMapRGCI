from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .GraphWalkTest import GraphWalkTest, to_categorical
from .graph_dataset_loader import load_dot_observed_labels


class GraphWalkTestTokenize(GraphWalkTest):
    """Graph-walk environment with hidden sink-to-source resets.

    The observable sequence contains only node observations. When the walk
    reaches a sink node, the next transition silently restarts from a source
    node. Ground-truth boundaries are stored for change-point evaluation, but
    no extra reset symbol is emitted into the observation vector.
    """

    source_nodes: np.ndarray[Any, Any]
    sink_nodes: np.ndarray[Any, Any]
    boundary_mask: np.ndarray[Any, Any]
    episode_ids: np.ndarray[Any, Any]
    _episode_id: int
    _last_boundary: bool

    def __init__(self, time_delay: int, filename: str = "graph.dot") -> None:
        self.time_delay = int(time_delay)
        self.time_counter = 0
        self._load_graph(filename, allow_resize=True)

        if self.source_nodes.size == 0 or self.sink_nodes.size == 0:
            raise ValueError(
                "GraphWalkTestTokenize requires an implicit-reset graph with at least one "
                f"source node (indegree=0) and one sink node (outdegree=0). "
                f"Got {self.source_nodes.size} sources and {self.sink_nodes.size} sinks: {filename}"
            )

        self.output_class = int(np.random.choice(self.source_nodes))
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0, dtype=int)
        self.boundary_mask = np.empty(0, dtype=bool)
        self.episode_ids = np.empty(0, dtype=int)
        self._episode_id = 0
        self._last_boundary = False

    def _resolve_path(self, filename: str) -> str:
        fpath = Path(filename)
        if fpath.exists():
            return str(fpath)
        base = Path(__file__).resolve().parents[1]
        alt = base / "datasets" / filename
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Cannot locate dataset file: {filename}")

    def _load_graph(self, filename: str, allow_resize: bool) -> None:
        filename_resolved = self._resolve_path(filename)
        g_new = nx.DiGraph(nx.nx_agraph.read_dot(filename_resolved))
        observed = load_dot_observed_labels(filename_resolved)
        if observed.is_hierarchical:
            raise ValueError(
                "GraphWalkTestTokenize is only defined for non-hierarchical graph-walk datasets. "
                "Hierarchy datasets use derived transition dynamics, so forcing them through hidden-reset "
                "tokenize changes the sequence semantics and breaks comparability. "
                f"Use hierarchy_sequence / hierarchy_sequence_ci instead: {filename}"
            )
        observed_nodes = list(observed.observed_nodes)
        true_label_new = np.asarray(observed.label_levels[:, 0], dtype=int)

        output_size_new = len(observed_nodes)
        adj_matrix = nx.adjacency_matrix(g_new, nodelist=observed_nodes)
        a_new = np.array(adj_matrix.toarray(), dtype=np.float64)  # type: ignore

        row_sums = np.sum(a_new, axis=1)
        non_sink = row_sums > 0.0
        if np.any(non_sink):
            a_new[non_sink] = a_new[non_sink] / row_sums[non_sink, None]

        if hasattr(self, "output_size") and not allow_resize and output_size_new != self.output_size:
            raise ValueError(f"switch_dataset requires same output_size ({self.output_size}), got {output_size_new}")

        indegree = np.asarray(np.sum(a_new > 0.0, axis=0), dtype=int)
        outdegree = np.asarray(np.sum(a_new > 0.0, axis=1), dtype=int)

        self.g = g_new
        self.true_label = true_label_new
        self.output_size = output_size_new
        self.a = a_new
        self.source_nodes = np.where(indegree == 0)[0].astype(int)
        self.sink_nodes = np.where(outdegree == 0)[0].astype(int)
        self.is_hierarchical = bool(observed.is_hierarchical)

    def switch_dataset(self, filename: str) -> None:
        self._load_graph(filename, allow_resize=False)
        self.time_counter = 0
        self.output_class = int(np.random.choice(self.source_nodes))
        self.previous_output_class = None
        self.previous_previous_output_class = None
        self.input_sequence = np.empty((0, self.output_size))
        self.input_class = np.empty(0, dtype=int)
        self.boundary_mask = np.empty(0, dtype=bool)
        self.episode_ids = np.empty(0, dtype=int)
        self._episode_id = 0
        self._last_boundary = False

    def getInput(self, reset: bool = False) -> np.ndarray[Any, Any]:
        if reset:
            self.time_counter = 0
            self.output_class = int(np.random.choice(self.source_nodes))
            self.previous_output_class = None
            self.previous_previous_output_class = None
            self._episode_id = 0
            self._last_boundary = False

        update = self.updateTimeDelay()
        self._last_boundary = False

        if update:
            self.previous_output_class = self.output_class
            outgoing = np.where(self.a[self.output_class] > 0.0)[0]
            if outgoing.size == 0:
                self.output_class = int(np.random.choice(self.source_nodes))
                self._episode_id += 1
                self._last_boundary = True
            else:
                probs = np.asarray(self.a[self.output_class, outgoing], dtype=float)
                probs = probs / np.sum(probs)
                self.output_class = int(np.random.choice(outgoing, p=probs))

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

        for i in range(sequence_size):
            input_value = self.getInput()
            self.input_class[i] = int(self.output_class)
            self.input_sequence[i] = input_value
            self.boundary_mask[i] = bool(self._last_boundary)
            self.episode_ids[i] = int(self._episode_id)

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
