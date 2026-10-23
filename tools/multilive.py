"""Multi-strategy SyncMap grid and live metric-curve viewer.

N strategies run on one shared input sequence. One window shows the embedding
cells; a second near-square grid records each strategy's NMI and ARI at the
configured interval while retaining and rescaling the complete time history.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

from tools.gui_runtime import configure_gui_environment

configure_gui_environment()

import numpy as np
from matplotlib.colors import to_rgba
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tools.plot_theme import resolve_plot_palette
from tools.result_paths import result_path
from unified_experiments import (
    create_model,
    _advance_model_chunk,
    _isolated_environment_rng,
    _select_evaluation_prediction,
    resolve_metric,
    EvaluationReadoutConfig,
)

try:
    from tools.visual_tools import animate_trajectory, visualize_input_sequence
except ImportError:
    visualize_input_sequence = None
    animate_trajectory = None

_SINGLETON = to_rgba("#8a8f98")


def _required_level_mean_ari(
    entries: list[tuple[str, float, float]],
    expected_depth: int,
) -> float:
    """Mean required-level ARI; invalid or missing required levels count as 0."""

    depth = max(1, int(expected_depth))
    finite = [float(entry[2]) for entry in entries if np.isfinite(float(entry[2]))]
    return float(sum(finite[:depth]) / depth)


def _colors_for_labels(labels, *, singleton_gray: bool = False) -> np.ndarray:
    """tab20 by sorted label rank; noise/singletons gray — matches LiveTuner."""
    import matplotlib.pyplot as plt
    arr = np.asarray(labels, dtype=int).reshape(-1)
    if arr.size == 0:
        return np.zeros((0, 4))
    cmap = plt.get_cmap("tab20")
    n = max(int(getattr(cmap, "N", 20)), 1)
    valid = sorted({int(l) for l in arr.tolist() if l >= 0})
    out = np.zeros((arr.size, 4))
    for i, l in enumerate(valid):
        out[arr == l] = cmap(i % n)
    if singleton_gray and valid:
        u, c = np.unique(arr[arr >= 0], return_counts=True)
        singles = {int(l) for l, cnt in zip(u.tolist(), c.tolist()) if cnt <= 1}
        if singles:
            out[np.array([int(l) in singles for l in arr])] = _SINGLETON
    out[arr < 0] = _SINGLETON
    return out


def grid_shape(n: int) -> tuple[int, int]:
    """Smallest near-square grid (rows, cols): 1→1x1 2→1x2 4→2x2 5/6→2x3 8→2x4 9→3x3."""
    n = max(1, int(n))
    rows = max(1, int(math.isqrt(n)))
    cols = math.ceil(n / rows)
    return rows, cols


def _feed(model: Any, chunk: np.ndarray, phase_offset: int) -> None:
    ig = getattr(model, "inputGeneral", None)
    if callable(ig):
        try:
            ig(chunk, phase_offset=phase_offset, reset_trajectory=False)
            return
        except TypeError:
            pass
    model.input(chunk)


def _pos3(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return np.zeros((0, 3))
    if pos.shape[1] >= 3:
        return pos[:, :3]
    out = np.zeros((pos.shape[0], 3))
    out[:, : pos.shape[1]] = pos
    return out


class MultiStrategyViewer:
    def __init__(
        self,
        dataset: str,
        strategies: list[dict[str, Any]],
        *,
        total_steps: int | None = None,
        record_interval: int | None = None,
        trajectory_interval: int = 50,     # unused; continuous sim
        theme: str = "dark",
        accent_color: str = "#6970ac",
        show: bool = True,
        seed: int = 0,
    ) -> None:
        import matplotlib.pyplot as plt

        self.n = len(strategies)
        self.strategies = strategies
        self._dataset_path = str(dataset)
        self._dataset_name = os.path.basename(str(dataset))
        self._theme = theme
        self._accent = accent_color
        self.pal = resolve_plot_palette(theme=theme, accent=accent_color)
        self._timer = None
        self._use_pred = False
        self._paused = True
        self._t = 0
        self._total_steps = None if total_steps is None else max(0, int(total_steps))
        self._chunk = 100   # algorithm step granularity (set from the model below)
        self._seed = int(seed)

        # ── models on a shared input sequence ────────────────────────────
        self.env = None
        self.models: list[Any] = []
        eff_params: list[dict[str, Any]] = []
        for info in strategies:
            np.random.seed(self._seed)
            model, env = create_model(
                dataset_path=dataset,
                strategy_name=info["strategy_name"],
                strategy_params=info.get("params", {}),
                record_trajectory=False,
                environment_seed=self._seed,
            )
            self.models.append(model)
            if self.env is None:
                self.env = env
            ep = getattr(model, "get_effective_params", lambda: {})()
            eff_params.append(dict(ep) if isinstance(ep, dict) else {})
        self.true_labels = np.asarray(self.env.trueLabel(), dtype=int)
        self._display_label_level = 0
        self._score_display_mode = "summary"
        # feed granularity from the algorithm itself (same as the unified experiment)
        self._chunk = int(getattr(self.models[0], "step_chunk", 100)) if self.models else 100
        # Every algorithm uses the same formal geometry readout. Ground truth
        # is only consumed by the metric after labels have been produced.
        self._metric = resolve_metric("ari")
        self._readout_cfg = EvaluationReadoutConfig(method="auto")
        self._label_cache: list = [None] * self.n
        self._readout_outputs_cache: list[dict[str, np.ndarray] | None] = [None] * self.n
        self._last_readout = 0
        self._readout_interval = max(1, int(record_interval if record_interval is not None else 200))
        self._curve_steps: list[int] = []
        self._curve_nmi: list[list[float]] = [[] for _ in range(self.n)]
        self._curve_ari: list[list[float]] = [[] for _ in range(self.n)]
        self._seq = None
        self._seq_pos = 0
        self._refill_sequence()
        self._common = self._common_params(eff_params)

        # ── figure + grid ────────────────────────────────────────────────
        self._dim = 3
        self._show_edges = False
        self._show_mesh = False
        self.input_fig: Any = None
        self.graph_fig: Any = None
        self.curve_fig: Any = None
        self.curve_axes: list[Any] = []
        self.curve_nmi_lines: list[Any] = []
        self.curve_ari_lines: list[Any] = []
        self.b_export_picture: Any = None
        self.b_export_curves: Any = None
        self.b_show_input: Any = None
        self.b_show_graph: Any = None
        self.b_live_curve: Any = None
        self.b_mesh: Any = None
        self.b_level: Any = None
        self.b_score: Any = None
        self.b_dataset_prev: Any = None
        self.b_dataset_next: Any = None
        self.dataset_text_artist: Any = None
        self._dataset_paths: list[str] = []
        self._dataset_index = 0
        self._legend: Any = None
        self._init_dataset_list()
        # each algorithm's own best eps (its configured value, else None → adapt to scale)
        self._base_eps = [self._compute_base_eps(i) for i in range(self.n)]
        rows, cols = grid_shape(self.n)
        self.fig = plt.figure(figsize=(cols * 4.2 + 2.8, max(7.0, rows * 4.2)),
                              facecolor=self.pal["fig_bg"])
        self.fig.suptitle(self._dataset_name, fontsize=12, fontweight="bold",
                          color=self.pal["fg"], x=0.39, y=0.985)
        self._build_cells()
        self._build_legend()
        self._build_panel()
        self._record_metric_point()
        self._redraw()

    def _build_cells(self) -> None:
        """(Re)create the grid of plot cells for the current view dimension."""
        for ax in getattr(self, "axes", []):
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        rows, cols = grid_shape(self.n)
        gl, gr, gt, gb = 0.02, 0.76, 0.94, 0.03
        self.axes, self.scatters, self.metric_txts, self.edge_cols = [], [], [], []
        for idx in range(self.n):
            r, c = divmod(idx, cols)
            w = (gr - gl) / cols - 0.015
            h = (gt - gb) / rows - 0.06
            left = gl + c * (gr - gl) / cols + 0.008
            bottom = gb + (rows - 1 - r) * (gt - gb) / rows + 0.02
            is3d = self._dim == 3
            ax = self.fig.add_axes([left, bottom, w, h], projection="3d" if is3d else None)
            pos = _pos3(self.models[idx].syncmap)
            if is3d:
                self._style_3d(ax)
                sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, depthshade=False,
                                linewidths=0.0, edgecolors="none")
                ax.view_init(elev=20, azim=-65)
                txt = ax.text2D(0.02, 0.02, "NMI/ARI: --/--", transform=ax.transAxes,
                                fontsize=8, color=self.pal["fg"], va="bottom", ha="left")
            else:
                ax.set_facecolor(self.pal["ax_bg"]); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(self.pal["inactive"])
                sc = ax.scatter(pos[:, 0], pos[:, 1], s=20, linewidths=0.0, edgecolors="none")
                txt = ax.text(0.02, 0.02, "NMI/ARI: --/--", transform=ax.transAxes,
                              fontsize=8, color=self.pal["fg"], va="bottom", ha="left")
            ax.set_title(self.strategies[idx]["display_name"].upper(), fontsize=10,
                         color=self.pal["fg"], pad=2)
            if is3d:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                # Matplotlib 3.10 cannot autoscale a newly registered empty
                # Line3DCollection, while 3.8 has no ``autolim`` keyword to
                # disable that autoscale.  Register a zero-length segment at
                # an existing point, then clear it; this works on both APIs
                # without expanding the scatter's data limits.
                anchor = np.asarray(pos[0], dtype=float)
                ec = Line3DCollection(
                    [[anchor, anchor]],
                    colors=self.pal["line"],
                    linewidths=0.4,
                    alpha=0.5,
                )
                ax.add_collection3d(ec)
                ec.set_segments([])
            else:
                from matplotlib.collections import LineCollection
                ec = LineCollection([], colors=self.pal["line"], linewidths=0.4, alpha=0.5)
                ax.add_collection(ec)
            self.axes.append(ax)
            self.scatters.append(sc)
            self.metric_txts.append(txt)
            self.edge_cols.append(ec)

        if getattr(self, "_timer", None) is None:   # create once; survives cell rebuilds
            self._timer = self.fig.canvas.new_timer(interval=60)
            self._timer.add_callback(self._step)
            self._timer.start()
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig.canvas.mpl_connect("close_event", self._on_main_close)

    @staticmethod
    def _format_step_tick(value: float, _position: int | None = None) -> str:
        magnitude = abs(float(value))
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:.2g}M"
        if magnitude >= 1_000:
            return f"{value / 1_000:.3g}k"
        return f"{value:.0f}"

    @staticmethod
    def _figure_is_open(fig: Any) -> bool:
        if fig is None:
            return False
        try:
            import matplotlib.pyplot as plt
            return bool(plt.fignum_exists(fig.number))
        except Exception:
            return False

    @staticmethod
    def _show_auxiliary_figure(fig: Any) -> None:
        try:
            fig.canvas.draw_idle()
            manager = getattr(fig.canvas, "manager", None)
            show = getattr(manager, "show", None)
            if callable(show):
                show()
            else:
                fig.show()
        except Exception:
            pass

    def _set_auxiliary_button(self, kind: str, is_open: bool) -> None:
        button = {
            "input": self.b_show_input,
            "graph": self.b_show_graph,
            "curve": self.b_live_curve,
        }.get(kind)
        if button is not None:
            button.label.set_text(f"{kind}:{'on' if is_open else 'off'}")
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _close_auxiliary_figure(self, kind: str) -> None:
        attr = f"{kind}_fig"
        fig = getattr(self, attr, None)
        setattr(self, attr, None)
        if self._figure_is_open(fig):
            import matplotlib.pyplot as plt
            try:
                plt.close(fig)
            except Exception:
                pass
        if kind == "curve":
            self.curve_axes = []
            self.curve_nmi_lines = []
            self.curve_ari_lines = []
        self._set_auxiliary_button(kind, False)

    def _register_auxiliary_figure(self, kind: str, fig: Any) -> None:
        setattr(self, f"{kind}_fig", fig)

        def _on_close(_event: Any) -> None:
            if getattr(self, f"{kind}_fig", None) is fig:
                setattr(self, f"{kind}_fig", None)
                if kind == "curve":
                    self.curve_axes = []
                    self.curve_nmi_lines = []
                    self.curve_ari_lines = []
                self._set_auxiliary_button(kind, False)

        fig.canvas.mpl_connect("close_event", _on_close)
        self._set_auxiliary_button(kind, True)

    def _build_curve_figure(self) -> None:
        """Create or restyle the per-strategy NMI/ARI history grid."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        rows, cols = grid_shape(self.n)
        curve_fig = getattr(self, "curve_fig", None)
        if not self._figure_is_open(curve_fig):
            curve_fig = plt.figure(figsize=(cols * 4.8, max(3.2, rows * 3.4)))
            self.curve_fig = curve_fig
            self._register_auxiliary_figure("curve", curve_fig)
            self.curve_fig.canvas.mpl_connect("key_press_event", self._on_key)
            manager = getattr(self.curve_fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title(f"{self._dataset_name} · live NMI/ARI")
        else:
            curve_fig.clear()

        curve_fig.set_facecolor(self.pal["fig_bg"])
        curve_fig.suptitle(
            f"{self._dataset_name} · NMI/ARI every {self._readout_interval:g} steps",
            fontsize=12,
            fontweight="bold",
            color=self.pal["fg"],
        )
        self.curve_axes: list[Any] = []
        self.curve_nmi_lines: list[Any] = []
        self.curve_ari_lines: list[Any] = []
        for idx in range(rows * cols):
            ax = curve_fig.add_subplot(rows, cols, idx + 1)
            if idx >= self.n:
                ax.set_visible(False)
                continue
            ax.set_facecolor(self.pal["ax_bg"])
            for spine in ax.spines.values():
                spine.set_color(self.pal["inactive"])
            ax.tick_params(colors=self.pal["fg"], labelsize=8)
            ax.grid(True, color=self.pal["grid"], alpha=0.45, linewidth=0.7)
            ax.set_title(
                str(self.strategies[idx]["display_name"]).upper(),
                fontsize=10,
                color=self.pal["fg"],
            )
            ax.set_xlabel("steps", fontsize=8, color=self.pal["fg"])
            ax.set_ylabel("score", fontsize=8, color=self.pal["fg"])
            ax.set_ylim(-0.10, 1.05)
            ax.xaxis.set_major_formatter(FuncFormatter(self._format_step_tick))
            hierarchy = bool(getattr(self.env, "is_hierarchical", False))
            nmi_label = "L1 NMI" if hierarchy else "NMI"
            ari_label = "mean-level ARI" if hierarchy else "ARI"
            nmi_line, = ax.plot([], [], color="#4c9be8", linewidth=1.7, label=nmi_label)
            ari_line, = ax.plot([], [], color="#f28e2b", linewidth=1.7, label=ari_label)
            legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
            for text in legend.get_texts():
                text.set_color(self.pal["fg"])
            self.curve_axes.append(ax)
            self.curve_nmi_lines.append(nmi_line)
            self.curve_ari_lines.append(ari_line)
        curve_fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.90, wspace=0.24, hspace=0.34)

    def _metric_values(self, labels: np.ndarray) -> tuple[float, float]:
        pred = np.asarray(labels, dtype=int).reshape(-1)
        return (
            float(normalized_mutual_info_score(self.true_labels, pred)),
            float(adjusted_rand_score(self.true_labels, pred)),
        )

    def _record_metric_point(self) -> None:
        values: list[tuple[float, float]] = []
        for idx in range(self.n):
            try:
                fixed = self._metric_values(self._cell_labels(idx))
                if bool(getattr(self.env, "is_hierarchical", False)):
                    entries = self._hierarchy_metric_entries(idx)
                    mean_ari = _required_level_mean_ari(
                        entries,
                        self._hierarchy_display_depth(),
                    )
                    values.append((fixed[0], mean_ari))
                else:
                    values.append(fixed)
            except Exception:
                values.append((float("nan"), float("nan")))
        if self._curve_steps and self._curve_steps[-1] == self._t:
            for idx, (nmi, ari) in enumerate(values):
                self._curve_nmi[idx][-1] = nmi
                self._curve_ari[idx][-1] = ari
            return
        self._curve_steps.append(int(self._t))
        for idx, (nmi, ari) in enumerate(values):
            self._curve_nmi[idx].append(nmi)
            self._curve_ari[idx].append(ari)

    def _redraw_curves(self) -> None:
        if not getattr(self, "curve_axes", None):
            return
        x_values = np.asarray(self._curve_steps, dtype=float)
        right = max(float(self._readout_interval), float(self._t), 1.0)
        for idx, ax in enumerate(self.curve_axes):
            self.curve_nmi_lines[idx].set_data(x_values, np.asarray(self._curve_nmi[idx], dtype=float))
            self.curve_ari_lines[idx].set_data(x_values, np.asarray(self._curve_ari[idx], dtype=float))
            ax.set_xlim(0.0, right)
        if self.curve_fig is not None:
            self.curve_fig.canvas.draw_idle()

    def _on_live_curve(self, _: Any = None) -> None:
        if self._figure_is_open(self.curve_fig):
            self._close_auxiliary_figure("curve")
            return
        self._build_curve_figure()
        self._redraw_curves()
        if self.curve_fig is not None:
            self._show_auxiliary_figure(self.curve_fig)

    @staticmethod
    def _inspect_dataset_signature(candidate: Path) -> tuple[int, bool] | None:
        try:
            content = candidate.read_text(encoding="utf-8")
        except Exception:
            return None
        node_ids: set[int] = set()
        labeled_nodes: set[int] = set()
        saw_multilevel_label = False
        saw_unlabeled_node = False
        node_pattern = re.compile(r'^\s*(\d+)\s*\[(.*?)\]\s*;?\s*$')
        edge_pattern = re.compile(r'^\s*(\d+)\s*->\s*(\d+)')
        label_pattern = re.compile(r'label\s*=\s*(".*?"|[^,\]]+)')
        for line in content.splitlines():
            node_match = node_pattern.match(line)
            if node_match:
                node_id = int(node_match.group(1))
                node_ids.add(node_id)
                label_match = label_pattern.search(node_match.group(2))
                if label_match is None:
                    saw_unlabeled_node = True
                    continue
                raw_label = label_match.group(1).strip().strip('"')
                if raw_label:
                    labeled_nodes.add(node_id)
                    if "," in raw_label:
                        saw_multilevel_label = True
                continue
            edge_match = edge_pattern.match(line)
            if edge_match:
                node_ids.add(int(edge_match.group(1)))
                node_ids.add(int(edge_match.group(2)))
        total_nodes = len(node_ids)
        observed_nodes = len(labeled_nodes) if labeled_nodes else total_nodes
        is_hierarchical = (
            saw_multilevel_label
            or bool(labeled_nodes and len(labeled_nodes) != total_nodes)
            or saw_unlabeled_node
        )
        return observed_nodes, is_hierarchical

    def _is_dataset_switch_compatible(self, candidate: Path) -> bool:
        signature = self._inspect_dataset_signature(candidate)
        if signature is None:
            return False
        observed_count, is_hierarchical = signature
        return (
            int(observed_count) == int(len(self.true_labels))
            and bool(is_hierarchical) == bool(getattr(self.env, "is_hierarchical", False))
        )

    def _init_dataset_list(self) -> None:
        try:
            current = Path(self._dataset_path).resolve()
            directory = current.parent if current.parent.exists() else Path("datasets").resolve()
            candidates = sorted(path.resolve() for path in directory.glob("*.dot") if path.is_file())
            compatible = [path for path in candidates if self._is_dataset_switch_compatible(path)]
            if current not in compatible:
                compatible.insert(0, current)
            self._dataset_paths = [str(path) for path in compatible] or [str(current)]
            self._dataset_index = self._dataset_paths.index(str(current))
            self._dataset_path = str(current)
        except Exception:
            self._dataset_paths = [str(self._dataset_path)]
            self._dataset_index = 0

    def _dataset_switch_enabled(self) -> bool:
        return (
            callable(getattr(self.env, "switch_dataset", None))
            and len(self._dataset_paths) > 1
        )

    def _sync_model_hierarchy_capacity(self) -> None:
        if not bool(getattr(self.env, "is_hierarchical", False)):
            return
        depth = max(1, int(getattr(self.env, "hierarchy_depth", 1)))
        for model in self.models:
            updates: dict[str, int] = {}
            if hasattr(model, "target_levels"):
                updates["target_levels"] = depth
            if hasattr(model, "levels"):
                updates["level_count"] = max(2, depth + 1)
            update_params = getattr(model, "update_params", None)
            if updates and callable(update_params):
                try:
                    update_params(**updates)
                except Exception as exc:
                    print(
                        f"[Warn] Failed to align {type(model).__name__} "
                        f"after dataset switch: {exc}"
                    )

    def _switch_dataset_path(self, target: str) -> bool:
        switch = getattr(self.env, "switch_dataset", None)
        if not callable(switch):
            return False
        try:
            switch(target)
        except Exception as exc:
            print(f"[Warn] Failed to switch dataset to {target}: {exc}")
            return False
        self._dataset_path = str(Path(target).resolve())
        self._dataset_name = Path(self._dataset_path).name
        self.true_labels = np.asarray(self.env.trueLabel(), dtype=int)
        self._display_label_level %= self._hierarchy_display_depth()
        self._sync_model_hierarchy_capacity()
        self._refill_sequence()
        self._label_cache = [None] * self.n
        self._readout_outputs_cache = [None] * self.n
        self._refresh_labels()
        self._last_readout = self._t
        self._record_metric_point()
        self._close_auxiliary_figure("input")
        self._close_auxiliary_figure("graph")
        if self.dataset_text_artist is not None:
            self.dataset_text_artist.set_text(self._dataset_name)
        if self.b_level is not None:
            self.b_level.label.set_text(self._level_button_label())
        self._build_legend()
        self.fig.suptitle(
            self._dataset_name,
            fontsize=12,
            fontweight="bold",
            color=self.pal["fg"],
            x=0.39,
            y=0.985,
        )
        if self._figure_is_open(self.curve_fig):
            self._build_curve_figure()
            self._redraw_curves()
        self._redraw()
        return True

    def _advance_dataset(self, step: int) -> None:
        if not self._dataset_switch_enabled():
            return
        direction = 1 if step >= 0 else -1
        start = int(self._dataset_index)
        for offset in range(1, len(self._dataset_paths) + 1):
            index = (start + direction * offset) % len(self._dataset_paths)
            if self._switch_dataset_path(self._dataset_paths[index]):
                self._dataset_index = index
                return
        print("[Warn] No compatible dataset found in current folder.")

    # ── sequence (continuous: refill instead of stopping) ────────────────
    def _refill_sequence(self) -> None:
        with _isolated_environment_rng(self.env):
            seq, _ = self.env.getSequence(20000)
        self._seq = np.asarray(seq, dtype=float)
        self._seq_pos = 0

    def _next_chunk(self, *, max_steps: int | None = None) -> np.ndarray:
        chunk_size = self._chunk if max_steps is None else min(self._chunk, max(0, int(max_steps)))
        if self._seq_pos + chunk_size > self._seq.shape[0]:
            self._refill_sequence()
        chunk = self._seq[self._seq_pos:self._seq_pos + chunk_size]
        self._seq_pos += chunk_size
        return chunk

    # ── parameters common to ALL strategies ──────────────────────────────
    @staticmethod
    def _common_params(eff_params: list[dict[str, Any]]) -> dict[str, Any]:
        if not eff_params:
            return {}
        keys = set(eff_params[0])
        for ep in eff_params[1:]:
            keys &= set(ep)
        skip = {"input_size", "record_trajectory", "trajectory_interval"}
        common: dict[str, Any] = {}
        for k in sorted(keys - skip):
            vals = [ep.get(k) for ep in eff_params]
            common[k] = vals[0] if all(v == vals[0] for v in vals) else "varies"
        return common

    # ── rendering ────────────────────────────────────────────────────────
    def _style_3d(self, ax) -> None:
        ax.set_facecolor(self.pal["ax_bg"])
        try:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color(to_rgba(self.pal["ax_bg"]))
                axis.line.set_color(self.pal["inactive"])
                axis._axinfo["grid"]["color"] = self.pal["grid"]
        except Exception:
            pass
        self._apply_mesh_to_axis(ax, 1.0)

    def _apply_mesh_to_axis(self, ax: Any, limit: float) -> None:
        if not self._show_mesh:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            return
        bound = max(float(limit), 1.0)
        ticks = np.linspace(-bound, bound, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis._axinfo["grid"]["color"] = to_rgba(self.pal["grid"], 0.55)
                axis._axinfo["grid"]["linewidth"] = 0.7
            except Exception:
                pass

    def _build_legend(self) -> None:
        from matplotlib.lines import Line2D
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:
                pass
            self._legend = None
        true_labels = self._display_true_labels()
        labels = sorted({int(l) for l in true_labels.tolist() if l >= 0})
        cols = _colors_for_labels(np.array(labels)) if labels else np.zeros((0, 4))
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5,
                          markerfacecolor=cols[i], markeredgecolor="none",
                          label=f"class {l + 1}") for i, l in enumerate(labels)]
        if handles and self.axes:
            self._legend = self.fig.legend(
                handles=handles,
                loc="upper left",
                ncol=2,
                fontsize=7,
                frameon=False,
                labelcolor=self.pal["fg"],
                bbox_to_anchor=(0.01, 0.99),
            )

    def _hierarchy_display_depth(self) -> int:
        levels = getattr(self.env, "true_label_levels", None)
        if hasattr(levels, "shape") and len(levels.shape) >= 2:
            return max(1, int(levels.shape[1]))
        return max(1, int(getattr(self.env, "hierarchy_depth", 1)))

    def _display_true_labels(self) -> np.ndarray:
        if bool(getattr(self.env, "is_hierarchical", False)):
            try:
                return np.asarray(
                    self.env.trueLabel(level=self._display_label_level),
                    dtype=int,
                )
            except TypeError:
                pass
        return np.asarray(self.true_labels, dtype=int)

    def _level_button_label(self) -> str:
        return f"level:L{int(self._display_label_level)}"

    def _score_button_label(self) -> str:
        return (
            "score:overall"
            if self._score_display_mode == "summary"
            else "score:levels"
        )

    def _compute_base_eps(self, idx: int):
        """Each algorithm's own best eps: its configured value, else None
        (None → adapt to the cell's current embedding scale at readout time)."""
        p = self.strategies[idx].get("params", {}) or {}
        for k in ("organize_dbscan_eps", "dbscan_eps", "eval_readout_eps"):
            v = p.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    @staticmethod
    def _label_summary(labels: np.ndarray) -> tuple[int, float]:
        arr = np.asarray(labels, dtype=int).reshape(-1)
        if arr.size == 0:
            return 0, 0.0
        valid = arr[arr >= 0]
        cluster_count = len(np.unique(valid)) if valid.size else 0
        noise_ratio = float(np.mean(arr < 0))
        return int(cluster_count), noise_ratio

    def _readout_labels(self, idx: int) -> np.ndarray:
        """Apply the same formal unsupervised readout to every embedding."""
        m = self.models[idx]
        pos = np.asarray(m.syncmap, dtype=float)
        r = _select_evaluation_prediction(
            m,
            self.env,
            self._metric,
            readout_config=self._readout_cfg,
        )
        raw_outputs = r.get("outputs", {})
        self._readout_outputs_cache[idx] = {
            str(key): np.asarray(value, dtype=int).reshape(-1)
            for key, value in raw_outputs.items()
        }
        lab = np.asarray(r["labels"], dtype=int).reshape(-1)
        if lab.shape[0] != pos.shape[0]:
            raise ValueError(
                f"readout returned {lab.shape[0]} labels for {pos.shape[0]} points"
            )
        return lab

    def _refresh_labels(self) -> None:
        self._readout_outputs_cache = [None] * self.n
        self._label_cache = [self._readout_labels(i) for i in range(self.n)]

    def _cell_labels(self, idx: int) -> np.ndarray:
        if self._label_cache[idx] is None:
            self._label_cache[idx] = self._readout_labels(idx)
        return self._label_cache[idx]

    def _cell_display_labels(self, idx: int) -> np.ndarray:
        formal = self._cell_labels(idx)
        outputs = self._readout_outputs_cache[idx] or {}
        key = f"cluster_level{int(self._display_label_level) + 1}"
        labels = np.asarray(outputs.get(key, formal), dtype=int).reshape(-1)
        return labels if labels.shape == formal.shape else formal

    def _hierarchy_metric_entries(self, idx: int) -> list[tuple[str, float, float]]:
        self._cell_labels(idx)
        outputs = self._readout_outputs_cache[idx] or {}
        entries: list[tuple[str, float, float]] = []
        for level_idx in range(self._hierarchy_display_depth()):
            pred = outputs.get(f"cluster_level{level_idx + 1}")
            if pred is None:
                continue
            true = np.asarray(self.env.trueLabel(level=level_idx), dtype=int)
            pred_arr = np.asarray(pred, dtype=int)
            if pred_arr.shape != true.shape:
                continue
            entries.append(
                (
                    f"L{level_idx + 1}",
                    float(normalized_mutual_info_score(true, pred_arr)),
                    float(adjusted_rand_score(true, pred_arr)),
                )
            )
        return entries

    def _format_metric_text(self, idx: int) -> str:
        if not bool(getattr(self.env, "is_hierarchical", False)):
            pred = self._cell_labels(idx)
            nmi = normalized_mutual_info_score(self.true_labels, pred)
            ari = adjusted_rand_score(self.true_labels, pred)
            cluster_count, noise_ratio = self._label_summary(pred)
            return (
                f"NMI/ARI: {nmi:.3f}/{ari:.3f} | "
                f"k={cluster_count} noise={noise_ratio:.2f} | t={self._t}"
            )

        entries = self._hierarchy_metric_entries(idx)
        if not entries:
            return f"NMI/ARI: --/-- | t={self._t}"
        mean_ari = _required_level_mean_ari(
            entries,
            self._hierarchy_display_depth(),
        )
        if self._score_display_mode == "levels":
            level_lines = [
                "  ".join(
                    f"{name}:{nmi:.2f}/{ari:.2f}"
                    for name, nmi, ari in entries[start:start + 2]
                )
                for start in range(0, len(entries), 2)
            ]
            return "\n".join(
                [f"ALL(mean ARI): {mean_ari:.3f}", *level_lines]
            )
        fixed_name, fixed_nmi, fixed_ari = entries[0]
        return "\n".join(
            [
                f"ALL(mean ARI): {mean_ari:.3f} | t={self._t}",
                f"Target {fixed_name}: {fixed_nmi:.3f}/{fixed_ari:.3f}",
            ]
        )

    def _redraw_cell(self, idx: int) -> None:
        pos = _pos3(self.models[idx].syncmap)
        if pos.shape[0] == 0:
            return
        # shared formal readout (cached; recomputed every N steps)
        pred_score = self._cell_display_labels(idx)
        if self._use_pred:
            colors = _colors_for_labels(pred_score, singleton_gray=True)
        else:
            colors = _colors_for_labels(self._display_true_labels())
        sc = self.scatters[idx]
        if self._dim == 3:
            sc._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        else:
            sc.set_offsets(pos[:, :2])
        sc.set_color(colors)
        # auto-scale each cell to its OWN extent (TSF lives in [-1,1], RG much larger)
        lim = max(float(np.max(np.abs(pos[:, : self._dim]))) * 1.1, 1e-3)
        self.axes[idx].set_xlim(-lim, lim)
        self.axes[idx].set_ylim(-lim, lim)
        if self._dim == 3:
            self.axes[idx].set_zlim(-lim, lim)
            self._apply_mesh_to_axis(self.axes[idx], lim)
        # cluster edges: spokes from each node to its cluster centroid
        segs = []
        if self._show_edges:
            lab = pred_score
            d = self._dim
            for cl in {int(l) for l in lab.tolist() if l >= 0}:
                pts = pos[lab == cl][:, :d]
                if pts.shape[0] >= 2:
                    cen = pts.mean(0)
                    segs.extend(np.vstack([p, cen]) for p in pts)
        try:
            self.edge_cols[idx].set_segments(segs)
        except Exception:
            pass
        try:
            self.metric_txts[idx].set_text(self._format_metric_text(idx))
        except Exception:
            self.metric_txts[idx].set_text(f"NMI/ARI: --/-- | t={self._t}")

    def _redraw(self) -> None:
        for idx in range(self.n):
            self._redraw_cell(idx)
        self.fig.canvas.draw_idle()

    # ── simulation step (continuous, mirrors LiveTuner._step) ────────────
    def _step(self) -> None:
        if self._paused:
            return
        remaining = None if self._total_steps is None else self._total_steps - self._t
        if remaining is not None and remaining <= 0:
            self._paused = True
            self.b_toggle.label.set_text("Play")
            self.fig.canvas.draw_idle()
            return
        until_readout = max(1, self._readout_interval - (self._t - self._last_readout))
        step_limit = until_readout if remaining is None else min(remaining, until_readout)
        chunk = self._next_chunk(max_steps=step_limit)
        if chunk.shape[0] == 0:
            return
        for model in self.models:
            # feed EXACTLY like the unified experiment (_advance_model_chunk:
            # isolated RNG, phase_offset, no trajectory) so behaviour matches.
            _advance_model_chunk(model, chunk, phase_offset=self._t,
                                 reset_trajectory=False, append_final_frame=False)
            ref = getattr(model, "refresh_hierarchy_state", None)  # RG hierarchy refresh
            if callable(ref):
                try:
                    ref()
                except Exception:
                    pass
        self._t += int(chunk.shape[0])
        reached_target = self._total_steps is not None and self._t >= self._total_steps
        if reached_target or self._t - self._last_readout >= self._readout_interval:
            self._refresh_labels()
            self._last_readout = self._t
            self._record_metric_point()
            self._redraw_curves()
        if reached_target:
            self._paused = True
            self.b_toggle.label.set_text("Play")
        self._redraw()

    # ── panel (LiveTuner-style controls, drive all cells) ────────────────
    def _build_panel(self) -> None:
        from matplotlib.widgets import Slider, Button
        pal, fg = self.pal, self.pal["fg"]
        px = 0.785
        panel = self.fig.add_axes([0.772, 0.03, 0.222, 0.91])
        panel.set_facecolor(pal["panel_bg"]); panel.set_xticks([]); panel.set_yticks([])
        for sp in panel.spines.values():
            sp.set_visible(False)
        panel.set_title(f"Controls ({self.n} strategies)", fontsize=10, color=fg, pad=6)

        def label(y, t):
            self.fig.text(px, y, t, fontsize=9, color=fg, fontweight="bold")

        def mkbtn(rect, text, cb):
            b = Button(self.fig.add_axes(rect), text,
                       color=pal["btn_bg"], hovercolor=pal["btn_hover"])
            b.on_clicked(cb)
            b.color = pal["btn_bg"]
            b.hovercolor = pal["btn_hover"]
            b.ax.set_facecolor(pal["btn_bg"])
            for sp in b.ax.spines.values():
                sp.set_edgecolor(pal["btn_edge"])
            b.label.set_color(fg); b.label.set_fontsize(9)
            return b

        def mkslider(rect, name, lo, hi, val, step, cb):
            sax = self.fig.add_axes(rect)
            sax.set_facecolor(pal["slider_rest"])
            try:
                s = Slider(sax, name, lo, hi, valinit=val, valstep=step, color=pal["slider_label"])
            except TypeError:
                s = Slider(sax, name, lo, hi, valinit=val, valstep=step)
            s.on_changed(cb)
            s.label.set_color(fg); s.valtext.set_color(fg)
            s.label.set_position((0.5, 0.5))
            s.label.set_transform(s.ax.transAxes)
            s.label.set_horizontalalignment("center")
            s.label.set_verticalalignment("center")
            s.valtext.set_position((0.98, 0.5))
            s.valtext.set_transform(s.ax.transAxes)
            s.valtext.set_horizontalalignment("right")
            s.valtext.set_verticalalignment("center")
            for attr in ("track", "poly", "hline", "vline", "_handle", "handle"):
                w = getattr(s, attr, None)
                if w is None:
                    continue
                c = pal["slider_rest"] if attr == "track" else pal["slider_label"]
                try: w.set_color(c)
                except Exception:
                    try: w.set_facecolor(c)
                    except Exception: pass
            return s

        w = 0.195
        y = [0.915]

        def place(h, gap=0.008):
            y[0] -= h
            bottom = y[0]
            y[0] -= gap
            return bottom

        def sect(name):
            label(place(0.020) + 0.002, name)

        def one_btn(text, cb):
            return mkbtn([px, place(0.030), w, 0.030], text, cb)

        def two_btn(t1, cb1, t2, cb2):
            b = place(0.030); bw = (w - 0.012) / 2
            return (mkbtn([px, b, bw, 0.030], t1, cb1),
                    mkbtn([px + bw + 0.012, b, bw, 0.030], t2, cb2))

        def sld(name, lo, hi, val, step, cb):
            return mkslider([px, place(0.020), w, 0.020], name, lo, hi, val, step, cb)

        sect("Dataset")
        self.dataset_text_artist = self.fig.text(
            px,
            place(0.030) + 0.015,
            self._dataset_name,
            fontsize=7.5,
            color=fg,
            va="center",
            ha="left",
        )
        if self._dataset_switch_enabled():
            self.b_dataset_prev, self.b_dataset_next = two_btn(
                "Prev",
                lambda _e: self._advance_dataset(-1),
                "Next",
                lambda _e: self._advance_dataset(1),
            )

        sect("Playback")
        self.b_toggle = one_btn("Play", self._on_toggle)
        self.b_reset, self.b_step = two_btn("Reset", self._on_reset, "Step", self._on_step)
        self.s_chunk = sld("chunk", 1, 1000, self._chunk, 1, self._on_chunk)

        sect("View")
        self.b_view, self.b_theme = two_btn(
            f"view:{self._dim}D", self._on_view, self._theme, self._on_theme)

        sect("Display")
        self.b_edges, self.b_mesh = two_btn(
            f"edges:{'on' if self._show_edges else 'off'}",
            self._on_edges,
            f"mesh:{'on' if self._show_mesh else 'off'}",
            self._on_mesh,
        )

        sect("Evaluation")
        if bool(getattr(self.env, "is_hierarchical", False)):
            b = place(0.030)
            gap = 0.006
            bw = (w - 2 * gap) / 3
            self.b_colors = mkbtn(
                [px, b, bw, 0.030],
                "colors:pred" if self._use_pred else "colors:true",
                self._on_colors,
            )
            self.b_level = mkbtn(
                [px + bw + gap, b, bw, 0.030],
                self._level_button_label(),
                self._on_level,
            )
            self.b_score = mkbtn(
                [px + 2 * (bw + gap), b, bw, 0.030],
                self._score_button_label(),
                self._on_score,
            )
            for button in (self.b_colors, self.b_level, self.b_score):
                button.label.set_fontsize(7.2)
        else:
            self.b_colors = one_btn(
                "colors:pred" if self._use_pred else "colors:true",
                self._on_colors,
            )
            self.b_level = None
            self.b_score = None

        sect("Export")
        self.b_export_picture, self.b_export_curves = two_btn(
            "save pair", self._on_export_picture, "save curves", self._on_export_curves)

        sect("Actions")
        b = place(0.030)
        gap = 0.006
        bw = (w - 2 * gap) / 3
        self.b_show_input = mkbtn(
            [px, b, bw, 0.030],
            f"input:{'on' if self._figure_is_open(self.input_fig) else 'off'}",
            self._on_show_input,
        )
        self.b_show_graph = mkbtn(
            [px + bw + gap, b, bw, 0.030],
            f"graph:{'on' if self._figure_is_open(self.graph_fig) else 'off'}",
            self._on_show_graph,
        )
        self.b_live_curve = mkbtn(
            [px + 2 * (bw + gap), b, bw, 0.030],
            f"curve:{'on' if self._figure_is_open(self.curve_fig) else 'off'}",
            self._on_live_curve,
        )
        for button in (self.b_show_input, self.b_show_graph, self.b_live_curve):
            button.label.set_fontsize(7.2)

        sect("Common params")
        lines = [f"{k} = {v:.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                 else f"{k} = {v}" for k, v in list(self._common.items())[:10]]
        self.fig.text(px, max(0.03, place(0.0) - 0.14), "\n".join(lines), fontsize=7.5,
                      color=fg, family="monospace", va="bottom")

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_toggle(self, _: Any = None) -> None:
        self._paused = not self._paused
        self.b_toggle.label.set_text("Play" if self._paused else "Pause")
        self.fig.canvas.draw_idle()

    def _on_reset(self, _: Any = None) -> None:
        for m in self.models:
            if hasattr(m, "syncmap"):
                m.syncmap = np.random.rand(*np.asarray(m.syncmap).shape).astype(float)
        self._refill_sequence()
        self._t = 0
        self._last_readout = 0
        self._label_cache = [None] * self.n
        self._readout_outputs_cache = [None] * self.n
        self._curve_steps.clear()
        for values in self._curve_nmi:
            values.clear()
        for values in self._curve_ari:
            values.clear()
        self._record_metric_point()
        self._close_auxiliary_figure("input")
        self._close_auxiliary_figure("graph")
        self._paused = True
        self.b_toggle.label.set_text("Play")
        self._redraw()
        self._redraw_curves()

    def _on_step(self, _: Any = None) -> None:
        was = self._paused
        self._paused = False
        self._step()
        self._paused = was

    def _on_chunk(self, val: float) -> None:
        self._chunk = max(1, int(val))

    def _on_colors(self, _: Any = None) -> None:
        self._use_pred = not self._use_pred
        self.b_colors.label.set_text("colors:pred" if self._use_pred else "colors:true")
        self._redraw()

    def _on_level(self, _: Any = None) -> None:
        depth = self._hierarchy_display_depth()
        self._display_label_level = (self._display_label_level + 1) % depth
        if self.b_level is not None:
            self.b_level.label.set_text(self._level_button_label())
        self._build_legend()
        self._redraw()

    def _on_score(self, _: Any = None) -> None:
        self._score_display_mode = (
            "levels"
            if self._score_display_mode == "summary"
            else "summary"
        )
        if self.b_score is not None:
            self.b_score.label.set_text(self._score_button_label())
        self._redraw()

    def _on_view(self, _: Any = None) -> None:
        self._dim = 2 if self._dim == 3 else 3
        self._build_cells()
        self.b_view.label.set_text(f"view:{self._dim}D")
        self._redraw()

    def _on_edges(self, _: Any = None) -> None:
        self._show_edges = not self._show_edges
        self.b_edges.label.set_text("edges:on" if self._show_edges else "edges:off")
        self._redraw()

    def _on_mesh(self, _: Any = None) -> None:
        self._show_mesh = not self._show_mesh
        if self.b_mesh is not None:
            self.b_mesh.label.set_text(f"mesh:{'on' if self._show_mesh else 'off'}")
        self._redraw()

    def _on_theme(self, _: Any = None) -> None:
        curve_open = self._figure_is_open(self.curve_fig)
        self._theme = "light" if self._theme == "dark" else "dark"
        self.pal = resolve_plot_palette(theme=self._theme, accent=self._accent)
        self.fig.clear()
        self.fig.set_facecolor(self.pal["fig_bg"])
        self.fig.suptitle(self._dataset_name, fontsize=12, fontweight="bold",
                          color=self.pal["fg"], x=0.39, y=0.985)
        self._build_cells()
        self._build_legend()
        self._build_panel()
        if curve_open:
            self._build_curve_figure()
        self._redraw()
        self._redraw_curves()

    def _export_picture_paths(self, timestamp: str) -> tuple[Any, Any]:
        name = self._dataset_name.replace(".", "_")
        stem = f"multilive_{name}_step{int(self._t)}_{timestamp}"
        return (
            result_path(f"{stem}_layout.png", kind="picture"),
            result_path(f"{stem}_curves.png", kind="picture"),
        )

    def _save_layout_picture(self, target: Any) -> None:
        self.fig.savefig(
            target,
            dpi=200,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
        )

    def _save_curve_picture(self, target: Any) -> None:
        curve_was_open = self._figure_is_open(self.curve_fig)
        if not curve_was_open:
            self._build_curve_figure()
            self._redraw_curves()
        try:
            self.curve_fig.savefig(
                target,
                dpi=200,
                bbox_inches="tight",
                facecolor=self.curve_fig.get_facecolor(),
            )
        finally:
            if not curve_was_open:
                self._close_auxiliary_figure("curve")

    def _on_export_picture(self, _: Any = None) -> None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target, curve_target = self._export_picture_paths(ts)
        try:
            self._save_layout_picture(target)
            self._save_curve_picture(curve_target)
        except Exception as exc:
            print(f"[Warn] Paired picture export failed: {exc}")
            return
        print(f"Saved picture to {target}")
        print(f"Saved metric curves to {curve_target}")

    def _on_export_curves(self, _: Any = None) -> None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _layout_target, curve_target = self._export_picture_paths(ts)
        try:
            self._save_curve_picture(curve_target)
        except Exception as exc:
            print(f"[Warn] Metric-curve export failed: {exc}")
            return
        print(f"Saved metric curves to {curve_target}")

    def _on_show_input(self, _: Any = None) -> None:
        if self._figure_is_open(self.input_fig):
            self._close_auxiliary_figure("input")
            return
        if visualize_input_sequence is None:
            print("[Warn] visualize_input_sequence not available (visual_tools not found).")
            return
        try:
            axes = visualize_input_sequence(self.env, show=False)
        except Exception as exc:
            print(f"[Warn] Failed to show input sequence: {exc}")
            return
        fig = axes.figure
        manager = getattr(fig.canvas, "manager", None)
        setter = getattr(manager, "set_window_title", None)
        if callable(setter):
            setter(f"{self._dataset_name} · input")
        self._register_auxiliary_figure("input", fig)
        self._show_auxiliary_figure(fig)

    def _on_show_graph(self, _: Any = None) -> None:
        if self._figure_is_open(self.graph_fig):
            self._close_auxiliary_figure("graph")
            return
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            graph = getattr(self.env, "g", None)
            if graph is None:
                raise RuntimeError("environment graph is unavailable")
            graph = graph.copy()
            fig, ax = plt.subplots(figsize=(7.0, 7.0))
            layout = nx.spring_layout(graph, seed=0, weight="weight")
            colors = _colors_for_labels(self._display_true_labels())
            nx.draw_networkx(
                graph,
                pos=layout,
                ax=ax,
                node_color=colors,
                node_size=220,
                font_size=7,
                arrows=bool(graph.is_directed()),
                with_labels=True,
            )
            ax.set_title(f"Environment graph [{self._dataset_name}]")
            ax.set_axis_off()
            fig.tight_layout()
        except Exception as exc:
            print(f"[Warn] Failed to show graph preview: {exc}")
            return
        self._register_auxiliary_figure("graph", fig)
        self._show_auxiliary_figure(fig)

    def _on_main_close(self, _event: Any = None) -> None:
        if self._timer is not None:
            self._timer.stop()
        for kind in ("input", "graph", "curve"):
            self._close_auxiliary_figure(kind)

    def _on_key(self, event: Any) -> None:
        if event.key == " ":
            self._on_toggle()


def launch_multi_strategy_viewer(
    dataset: str,
    strategies: list[dict[str, Any]],
    *,
    total_steps: int | None = None,
    record_interval: int | None = None,
    theme: str = "dark",
    accent_color: str = "#6970ac",
    show: bool = True,
    seed: int = 0,
) -> "MultiStrategyViewer":
    import matplotlib.pyplot as plt
    viewer = MultiStrategyViewer(
        dataset, strategies, total_steps=total_steps, record_interval=record_interval,
        theme=theme, accent_color=accent_color, show=show, seed=seed,
    )
    if show:
        plt.show()
    return viewer
