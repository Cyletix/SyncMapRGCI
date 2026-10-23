#!/usr/bin/env python3
"""
Interactive SyncMap tuner for the focused RG branch.
Compatible with the unified experiment entry point.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, TypeAlias, cast
from numbers import Real
import math
import warnings

from tools.gui_runtime import configure_gui_environment

configure_gui_environment()

import matplotlib
matplotlib.use("Qt5Agg") # 保持交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.colors import Normalize, to_rgba
if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
else:
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        class _Axes3DDummy(Axes): pass
        Axes3D = _Axes3DDummy
    try:
        # 3D 线段集合，用于 3D 模式下的聚类连边
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except Exception:  # pragma: no cover - 极端环境下回退为 2D 类型
        Line3DCollection = LineCollection  # type: ignore[assignment]

from numpy.typing import NDArray
import numpy as np

from tools.metric_utils import clustering_metric_bundle
from tools.result_paths import artifact_result_path, result_path
from tools.plot_theme import resolve_plot_palette
from tools.shared_defaults import DEFAULT_HIERARCHY_ORGANIZE_LEVEL
from test.hierarchy_levels import (
    canonicalize_hierarchy_outputs,
    canonicalize_organize_level,
    cluster_level_display,
    cluster_level_key_from_index,
    parse_hierarchy_label_index,
)
# 移除旧的特定引用，只保留统一入口
from unified_experiments import (
    EvaluationReadoutConfig,
    _advance_model_chunk,
    _isolated_environment_rng,
    _rng_neutral,
    _select_evaluation_prediction,
    create_model,
    unified_readout_outputs,
    _resolve_external_readout_method,
    resolve_metric,
)
from tools.clustering_strategies import create_clustering_strategy
from sklearn.cluster import DBSCAN
from tools.visual_tools import animate_trajectory, default_anim_path, visualize_input_sequence

ClusterCollection: TypeAlias = LineCollection | Line3DCollection

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

DATASET = "datasets/chain_mixed5_6.dot" # 默认值

DISPLAY_LABELS: dict[str, str] = {
    'base_wall': 'wall',
    'density_ref': 'rho_ref',
    'space_size': 'space',
    'louvain_k_dist': 'lv_dist',
    'adaptation_rate': 'rate',
    'step_chunk': 'chunk',
    'graph_theta': 'theta',
    'graph_r0': 'r0',
    'graph_r0_rel': 'r0_rel',
    'graph_r0_fixed': 'r0',
    'embed_cluster_method': 'cluster',
}

# Legend: no artificial class-count limit — always use native legend.

if hasattr(Axes3D, "_button_release") and not getattr(Axes3D, "_elrion_safe_button_release", False):
    def _safe_button_release(self: Any, _event: Any) -> None:
        self.button_pressed = None
        fig = getattr(self, "figure", None)
        canvas = getattr(fig, "canvas", None)
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar and self.get_navigate_mode() is None:
            toolbar.push_current()

    Axes3D._button_release = _safe_button_release  # type: ignore[attr-defined]
    Axes3D._elrion_safe_button_release = True  # type: ignore[attr-defined]

class LiveTuner:
    def __init__(
        self,
        dataset: str,
        strategy_name: str,
        strategy_params: dict[str, Any],
        dimension: int,
        *,
        display_name: str | None = None,
        preloaded_model: Any | None = None,
        preloaded_env: Any | None = None,
        precomputed_trajectory: FloatArray | None = None,
        precomputed_steps: IntArray | None = None,
        start_paused: bool = False,
        readonly: bool = False,
        auto_pause_steps: int | None = None,
        record_interval: int | None = None,
    ) -> None:
        self._ds_path = dataset
        self._strategy_name = strategy_name
        self._variant_name: str | None = display_name  # profile/variant display label override
        self._strategy_params = strategy_params or {}
        self._preloaded_model = preloaded_model
        self._preloaded_env = preloaded_env
        self._precomputed_traj = None if precomputed_trajectory is None else np.asarray(precomputed_trajectory, dtype=float)
        self._precomputed_steps = None if precomputed_steps is None else np.asarray(precomputed_steps, dtype=int)
        self._traj_cursor: int = 0
        self._start_paused = bool(start_paused)
        self.readonly = bool(readonly)
        self._auto_pause_steps = (
            None
            if auto_pause_steps is None
            else max(0, int(auto_pause_steps))
        )
        self._requested_record_interval = (
            None
            if record_interval is None
            else max(1, int(record_interval))
        )
        self._auto_pause_announced = False
        self.model: Any = None
        self.env: Any = None
        self.output_size: int = 0
        self._label_level_override: int | None = None
        self._eval_label_level: int = 0
        self._display_label_level: int = 0
        self._score_display_mode: str = 'summary'
        self._hierarchy_outputs_cache: dict[str, NDArray[np.integer[Any]]] | None = None
        self._hierarchy_outputs_step: int = -10**9
        self._formal_eval_cache: dict[str, Any] | None = None
        self._formal_eval_step: int = -10**9

        self._slider_apply = lambda *_: None
        self._seed: int = int(self._strategy_params.get('seed', 0))
        self._initial_seed: int = 0
        self._formal_eval_metric = resolve_metric("ari")
        self._formal_eval_readout_config = self._build_live_eval_readout_config()
        configured_eval_method = str(self._strategy_params.get("eval_readout_method", "")).strip().lower()
        self._formal_readout_method_locked = configured_eval_method not in {"", "auto"}
        self._formal_readout_eps_locked = "eval_readout_eps" in self._strategy_params
        self._formal_readout_min_samples_locked = "eval_readout_min_samples" in self._strategy_params
        # Live display uses the same single formal readout as task 0/task 2.

        self.dbscan_min_samples: int = 3
        self.embed_cluster_method = str(
            self._strategy_params.get('embed_cluster_method', 'dbscan')
        ).lower()
        self.dbscan_eps: float = float(
            self._strategy_params.get('dbscan_eps', 1.5)
        )
        raw_readout_eps = self._strategy_params.get('dbscan_readout_eps', None)
        self.dbscan_readout_eps: float | None = None if raw_readout_eps is None else float(raw_readout_eps)
        self.dbscan_auto_eps: bool = False
        self.louvain_distance_factor: float = 0.5
        self.louvain_knn_k: int = 0
        self.louvain_mutual_knn: bool = True
        
        # UI 初始化需要的变量，防止 setup 前报错
        self.init_space_size = float(self._strategy_params.get('space_size', 0.0))

        # 初始化环境和模型
        self._setup_simulation(dataset)
        self._initial_seed = self._seed
        self._space_override_enabled: bool = self.init_space_size > 0.0
        self._reset_playback()

        self.step_chunk = 100
        self._step_chunk_options = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        self._last_labels: NDArray[np.integer[Any]] | None = None
        self._last_cluster_step: int = -10**9

        syncmap_arr = np.asarray(getattr(self.model, 'syncmap', np.zeros((0, 2))), dtype=float)
        syncmap_dims = int(syncmap_arr.shape[1]) if syncmap_arr.ndim == 2 and syncmap_arr.size > 0 else 2
        self.cluster_dimension = max(
            1,
            min(int(getattr(self.model, 'dimensions', syncmap_dims or 1)), max(syncmap_dims, 1)),
        )
        self.max_view_dimension = 3 if syncmap_dims >= 3 else 2
        requested_view_dimension = 2 if int(dimension) <= 2 else 3
        self._view_mode = 'xy'
        self.dimension = 2 if self.max_view_dimension < 3 else requested_view_dimension
        if self.max_view_dimension >= 3 and requested_view_dimension >= 3:
            self._view_mode = '3d'
        self.cluster_refresh_steps = (
            self._default_cluster_refresh_steps()
            if self._requested_record_interval is None
            else self._requested_record_interval
        )
        self._force_monitor_refresh_every = 5
        self._frame_index = 0
        self._interaction_active = False
        self._current_axis_limit: float | None = None

        dataset_name = Path(self._ds_path).stem

        # === 绘图初始化 ===
        if self.readonly:
            figure_size = (10, 7.9)
            self._plot_rect = (0.06, 0.07, 0.68, 0.88)
            self._panel_rect = (0.805, 0.07, 0.165, 0.86)
        else:
            figure_size = (10, 8.1)
            self._plot_rect = (0.055, 0.06, 0.67, 0.89)
            self._panel_rect = (0.79, 0.05, 0.19, 0.90)
        self.fig = plt.figure(figsize=figure_size)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            setter = getattr(manager, "set_window_title", None)
            if callable(setter):
                setter(f"{self._impl_name} – {dataset_name}")

        self.ax: Axes | Axes3D
        self.ax_panel: Axes = self.fig.add_axes(self._panel_rect)
        self.ax_panel.set_xticks([]); self.ax_panel.set_yticks([])
        for spine in self.ax_panel.spines.values():
            spine.set_visible(False)
        panel_title = "Viewer" if self.readonly else "Controls"
        self.ax_panel.set_title(f"{panel_title} ({self._impl_name})", fontsize=10, pad=6)

        self._theme: str = 'dark'
        self._accent_color: str = "#6970ac"
        # 初始化调色盘到实例属性，便于跨方法统一取色
        self._palette: dict[str, Any] = {}
        self._color_fig_bg = ''
        self._color_ax_bg = ''
        self._color_panel_bg = ''
        self._color_fg = ''
        self._color_inactive = ''
        self._color_slider_rest = ''
        self._color_slider_label = ''
        self._color_btn_bg = ''
        self._color_btn_edge = ''
        self._color_btn_hover = ''
        self._color_line = ''
        self._theme_palette()
        self._all_buttons: list[Button] = []
        self._hover_cids: dict[str, int] = {}
        self.max_relation_edges: int = 1500
        self.max_relation_arrows: int = 300

        self.scatter: PathCollection
        # 2D 使用 LineCollection，3D 使用 Line3DCollection
        self.cluster_lines: ClusterCollection | None = None
        self.edge_arrows: Artist | None = None

        # ... (UI Layout 变量初始化)
        self._panel_items: list[tuple[Axes, tuple[float, float, float, float]]] = []
        self._panel_all_axes: set[Axes] = {self.ax_panel}
        self._panel_cursor_y = 0.0
        self._panel_scroll = 0.0
        self._slider_axes_map: dict[Axes, Slider] = {}
        self._slider_steps: dict[Slider, float] = {}
        self._last_active_slider: Slider | None = None
        self._panel_margin_x = 0.02
        self._suppress_slider_events = False
        self._radio_groups: list[RadioButtons] = []
        self._check_groups: list[CheckButtons] = []
        self._label_axes: list[Axes] = []
        self.use_pred_colors: bool = False
        self._singleton_color = np.asarray(to_rgba("#8a8f98"), dtype=float)
        self.show_cluster_edges: bool = False
        self.show_mesh: bool = False
        self.embed_cluster_method: str = getattr(self, 'embed_cluster_method', 'dbscan')
        self.r_embed_method: RadioButtons | None = None
        self.b_pred: Button | None = None
        self.b_auto_eps: Button | None = None
        self.b_edges: Button | None = None
        self.b_embed: Button | None = None
        self.b_score: Button | None = None
        self.b_focus: Button | None = None
        self.b_theme: Button | None = None
        self.b_view: Button | None = None
        self.b_export_gif: Button | None = None
        self.b_export_picture: Button | None = None
        self.b_export_json: Button | None = None
        self.b_show_input: Button | None = None
        self.b_show_graph: Button | None = None
        self.b_live_curve: Button | None = None
        self.b_mesh: Button | None = None
        self.ax_pred: Axes | None = None
        self.ax_auto: Axes | None = None
        self.ax_edges: Axes | None = None
        self.ax_embed: Axes | None = None
        self.ax_score: Axes | None = None
        self.ax_focus: Axes | None = None
        self.ax_dataset: Axes | None = None
        self.ax_dataset_text: Axes | None = None
        self.dataset_text_artist: Artist | None = None
        self.b_dataset_prev: Button | None = None
        self.b_dataset_next: Button | None = None
        self.ax_eps: Axes | None = None
        self.ax_readout_eps: Axes | None = None
        self.ax_minp: Axes | None = None
        self.ax_louvain_k_dist: Axes | None = None
        self.ax_louvain_knn_k: Axes | None = None
        self.ax_louvain_mutual: Axes | None = None
        self.s_eps: Slider | None = None
        self.s_readout_eps: Slider | None = None
        self.s_minp: Slider | None = None
        self.s_louvain_k_dist: Slider | None = None
        self.s_louvain_knn_k: Slider | None = None
        self.s_memory_steps: Slider | None = None
        self.s_stable_token_evidence_memory_scale: Slider | None = None
        self.s_stable_token_evidence_min_members: Slider | None = None
        self.s_stable_token_evidence_min_stable_refreshes: Slider | None = None
        self.s_stable_token_evidence_semantic_z_score: Slider | None = None
        self.s_stable_token_evidence_min_self_mass: Slider | None = None
        self.s_stable_token_evidence_self_cross_ratio: Slider | None = None
        self.s_k_repulse: Slider | None = None
        self.s_primitive_active_attraction_strength: Slider | None = None
        self.s_primitive_inactive_repulsion_strength: Slider | None = None
        self.s_k_macro_repulse: Slider | None = None
        self.s_macro_cross_attr_scale: Slider | None = None
        self.s_macro_centroid_distance_ratio: Slider | None = None
        self.s_macro_centroid_max_step_ratio: Slider | None = None
        self.s_k_entity_boundary_repulse: Slider | None = None
        self.s_entity_boundary_distance_ratio: Slider | None = None
        self.s_entity_boundary_max_step_ratio: Slider | None = None
        self.s_k_entity_sibling_attr: Slider | None = None
        self.s_entity_sibling_target_ratio: Slider | None = None
        self.s_entity_sibling_max_step_ratio: Slider | None = None
        self.s_entity_sibling_boundary_release_ratio: Slider | None = None
        self.s_entity_sibling_boundary_power: Slider | None = None
        self.s_line_obstacle_strength: Slider | None = None
        self.s_line_obstacle_radius_rel: Slider | None = None
        self.b_louvain_mutual: Button | None = None

        # 可选数据集列表（用于 live 切换）
        self._dataset_paths: list[str] = []
        self._dataset_index: int = 0
        self._legend = None
        self._legend_info_ax: Axes | None = None
        self.force_fig: plt.Figure | None = None
        self.force_axes: list[Axes] = []
        self.force_display_cap: float = 10.0
        self.enable_force_monitor: bool = False
        self.input_fig: plt.Figure | None = None
        self.graph_fig: plt.Figure | None = None
        self.curve_fig: plt.Figure | None = None
        self.curve_ax: Axes | None = None
        self.curve_nmi_line: Artist | None = None
        self.curve_ari_line: Artist | None = None
        self._curve_steps: list[int] = []
        self._curve_nmi: list[float] = []
        self._curve_ari: list[float] = []

        self._init_dataset_list()
        self._create_plot_axes()

        self._build_panel()
        self._sync_ui_from_model(reset_display_state=False)
        self._init_force_monitor()
        self._update_force_monitor()

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("close_event", self._on_main_figure_close)
        
        self._set_running_state(not self._start_paused)

        self.ani = FuncAnimation(self.fig, self._on_frame, interval=80, blit=False, cache_frame_data=False)

    def _uses_direct_model_labels(self) -> bool:
        if str(getattr(self, '_strategy_name', '')).lower() == 'hclust':
            return True
        return (
            str(getattr(self.model, 'formal_prediction_kind', '')).lower()
            == 'sequence_partition'
        )

    def _live_adaptation_rate(self, default: float = 0.1) -> float:
        """Return a finite value for the dynamics-only live control.

        Partition-native models deliberately expose ``adaptation_rate=None``:
        they have no coordinate-dynamics learning rate. Matplotlib sliders and
        snapshot JSON still require a real scalar, so represent that
        not-applicable value as 0.0 instead of attempting ``float(None)``.
        """
        raw_value = getattr(
            self._base_dynamics_model(),
            'adaptation_rate',
            default,
        )
        if raw_value is None:
            return 0.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _should_show_force_monitor(self) -> bool:
        return bool(self.enable_force_monitor) and hasattr(self.model, 'levels') and len(getattr(self.model, 'levels', [])) > 0

    def _force_monitor_levels(self) -> list[tuple[str, Any]]:
        levels = getattr(self.model, 'levels', None)
        if levels:
            return [(f"L{idx}", level) for idx, level in enumerate(levels)]
        level0 = getattr(self.model, 'level0', None)
        if level0 is not None:
            return [("L0", level0)]
        return []

    @staticmethod
    def _representative_edge_weight(level: Any) -> float:
        rows = getattr(level, '_g_rows', None)
        cols = getattr(level, '_g_cols', None)
        weights = getattr(level, '_g_w', None)
        if rows is None or cols is None or weights is None:
            return 1.0
        rows_arr = np.asarray(rows, dtype=int)
        cols_arr = np.asarray(cols, dtype=int)
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.size == 0:
            return 1.0
        if bool(getattr(level, 'graph_degree_normalize', False)):
            degree = np.bincount(
                np.concatenate([rows_arr, cols_arr]),
                minlength=int(getattr(level, 'input_size', 0)),
            ).astype(float)
            weights_arr = weights_arr * (2.0 / np.maximum(degree[rows_arr] + degree[cols_arr], 1.0))
        return float(np.median(weights_arr)) if weights_arr.size else 1.0

    @staticmethod
    def _spring_signed_force(level: Any, dist: np.ndarray, weight_eff: float) -> np.ndarray:
        d = np.asarray(dist, dtype=float)
        target = max(float(getattr(level, 'graph_r0', getattr(level, 'graph_r0_fixed', 1.0))), 1e-6)
        mode = str(getattr(level, 'graph_spring_mode', 'rg_canonical')).lower()
        k_attr = float(getattr(level, 'k_attr', 0.03))
        toward = np.zeros_like(d)

        if mode == 'rg_canonical':
            far_mask = d >= target
            if np.any(far_mask):
                detach_limit = float(getattr(level, '_edge_detach_distance_abs', lambda: 0.0)())
                if detach_limit <= target:
                    detach_limit = target * max(float(getattr(level, 'inter_cluster_distance_ratio', 3.16)), 1.0)
                extend_span = max(detach_limit - target, 1e-6)
                extend_start = target + (
                    float(getattr(level, 'graph_spring_extend_shoulder_rel_start', 0.3)) * extend_span
                )
                extension = d[far_mask] - target
                shoulder = np.zeros_like(extension)
                shoulder_mask = d[far_mask] > extend_start
                if np.any(shoulder_mask):
                    rel = np.clip(
                        (d[far_mask][shoulder_mask] - extend_start) / max(detach_limit - extend_start, 1e-6),
                        0.0,
                        None,
                    )
                    shoulder[shoulder_mask] = (
                        max(float(getattr(level, 'graph_spring_extend_shoulder_gain', 0.0)), 0.0)
                        * target
                        * np.power(
                            rel,
                            max(float(getattr(level, 'graph_spring_extend_shoulder_power', 2.0)), 1.0),
                        )
                    )
                toward[far_mask] = k_attr * weight_eff * (extension + shoulder)
            if np.any(~far_mask):
                safe_d = np.maximum(d[~far_mask], 1e-9)
                compression = target - safe_d
                toward[~far_mask] = -(
                    k_attr
                    * weight_eff
                    * (
                        max(float(getattr(level, 'graph_spring_repulse_gain', 2.0)), 0.0)
                        * target
                        * (
                            np.power(
                                target / safe_d,
                                max(float(getattr(level, 'graph_spring_repulse_power', 1.0)), 1e-6),
                            ) - 1.0
                        )
                        + max(float(getattr(level, 'graph_spring_repulse_shoulder_gain', 12.0)), 0.0)
                        * compression
                    )
                )
        elif mode == 'stable_soft':
            rel = (d - target) / target
            abs_rel = np.abs(rel)
            soft_break = max(float(getattr(level, 'graph_spring_soft_rel', 2.0)), 1e-6)
            hard_break = max(float(getattr(level, 'graph_spring_hard_rel', 5.0)), soft_break + 1e-6)
            near_mag = max(float(getattr(level, 'graph_spring_near_slope', 0.2)), 0.0) * np.minimum(abs_rel, soft_break)
            mid_excess = np.maximum(np.minimum(abs_rel, hard_break) - soft_break, 0.0)
            mid_mag = (
                max(float(getattr(level, 'graph_spring_near_slope', 0.2)), 0.0) * soft_break
                + max(float(getattr(level, 'graph_spring_mid_slope', 0.6)), 0.0) * mid_excess
            )
            far_excess = np.maximum(abs_rel - hard_break, 0.0)
            far_mag = (
                max(float(getattr(level, 'graph_spring_near_slope', 0.2)), 0.0) * soft_break
                + max(float(getattr(level, 'graph_spring_mid_slope', 0.6)), 0.0) * (hard_break - soft_break)
                + max(float(getattr(level, 'graph_spring_far_gain', 0.8)), 0.0)
                * np.power(far_excess, max(float(getattr(level, 'graph_spring_far_power', 2.0)), 1.0))
            )
            toward = k_attr * weight_eff * target * np.sign(rel) * np.where(
                abs_rel <= soft_break,
                near_mag,
                np.where(abs_rel <= hard_break, mid_mag, far_mag),
            )
        else:
            toward = k_attr * weight_eff * (d - target)

        return -toward

    def _effective_signed_force_curve(self, level: Any, dist: np.ndarray) -> tuple[np.ndarray, float, float]:
        weight_eff = self._representative_edge_weight(level)
        raw_force = self._spring_signed_force(level, dist, weight_eff)
        raw_adaptation_rate = getattr(level, 'adaptation_rate', 1.0)
        adaptation_rate = (
            0.0
            if raw_adaptation_rate is None
            else float(raw_adaptation_rate)
        )
        max_step = max(float(getattr(level, 'max_step_norm', 10.0)), 1e-6)
        effective = np.clip(adaptation_rate * raw_force, -max_step, max_step)
        return effective, max_step, weight_eff

    def _init_force_monitor(self) -> None:
        if not self._should_show_force_monitor():
            return
        levels = self._force_monitor_levels()
        if not levels:
            return
        if self.force_fig is not None:
            return
        height = max(2.6, 2.35 * len(levels))
        self.force_fig, axes = plt.subplots(len(levels), 1, figsize=(6.8, height), squeeze=False)
        manager = getattr(self.force_fig.canvas, "manager", None)
        if manager is not None:
            setter = getattr(manager, "set_window_title", None)
            if callable(setter):
                setter(f"{self._impl_name} force monitor")
        self.force_axes = [ax for ax in axes[:, 0]]

        def _on_close(_event: Any) -> None:
            self.force_fig = None
            self.force_axes = []

        self.force_fig.canvas.mpl_connect("close_event", _on_close)

    def _update_force_monitor(self) -> None:
        if self.force_fig is None or not self.force_axes:
            return
        levels = self._force_monitor_levels()
        if len(levels) != len(self.force_axes):
            try:
                plt.close(self.force_fig)
            except Exception:
                pass
            self.force_fig = None
            self.force_axes = []
            self._init_force_monitor()
            if self.force_fig is None:
                return
            levels = self._force_monitor_levels()

        for ax, (name, level) in zip(self.force_axes, levels):
            ax.clear()
            target = max(float(getattr(level, 'graph_r0', getattr(level, 'graph_r0_fixed', 1.0))), 1e-6)
            ratio_grid = np.linspace(0.05, 3.0, 900)
            dist_grid = ratio_grid * target
            effective, max_step, _weight_eff = self._effective_signed_force_curve(level, dist_grid)
            display_cap = min(self.force_display_cap, max(max_step, 1.0))
            display = np.clip(effective, -display_cap, display_cap)

            focus_mask = (ratio_grid >= 0.35) & (ratio_grid <= 2.2)
            local_amp = float(np.max(np.abs(display[focus_mask]))) if np.any(focus_mask) else float(np.max(np.abs(display)))
            ylim = max(0.08, min(display_cap, local_amp * 1.25))

            ax.set_facecolor("#fbfbfd")
            ax.axvspan(0.0, 1.0, color="#f7dada", alpha=0.55, lw=0.0)
            ax.axvspan(1.0, 3.0, color="#dff3df", alpha=0.55, lw=0.0)
            ax.plot(ratio_grid, display, color="#2e6fd8", linewidth=2.6)
            ax.scatter([1.0], [0.0], s=42, color="#1d8f3a", zorder=4)
            ax.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.75)
            ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.axhline(-min(max_step, display_cap), color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.85)
            ax.grid(True, alpha=0.16, linewidth=0.8)
            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(-ylim, ylim)
            ax.set_ylabel("applied step")
            ax.set_title(f"{name} force curve (r0={target:.3f})", fontsize=11)
            ax.text(0.18, ylim * 0.82, "Compression", color="#7a4242", fontsize=9, ha="left", va="center")
            ax.text(1.24, ylim * 0.82, "Extension", color="#3d6a3d", fontsize=9, ha="left", va="center")

        self.force_axes[-1].set_xlabel("normalized distance r = d / r0")
        if self.force_fig is not None:
            self.force_fig.patch.set_facecolor("white")
            self.force_fig.tight_layout()
            self.force_fig.canvas.draw_idle()

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, complex): return float(value.real)
        if isinstance(value, Real): return float(value)
        raise TypeError(f"Cannot convert {type(value).__name__} to float.")

    @staticmethod
    def _require_slider(slider: Slider | None, name: str) -> Slider:
        if slider is None: raise RuntimeError(f"Slider '{name}' not init.")
        return slider

    def _is_minimal_rg_profile(self) -> bool:
        # `rg` (engine SyncMapRG) is the minimal mainline engine and gets the
        # minimal control surface.
        names = {
            str(getattr(self, '_strategy_name', '') or '').strip().lower(),
            str(getattr(self, '_variant_name', '') or '').strip().lower(),
        }
        return 'rg' in names

    @staticmethod
    def _require_checkbuttons(cb: CheckButtons | None, name: str) -> CheckButtons:
        if cb is None: raise RuntimeError(f"CheckButtons '{name}' not init.")
        return cb

    def _default_cluster_refresh_steps(self) -> int:
        if self.output_size >= 240:
            return 400
        if self.output_size >= 120:
            return 200
        return max(100, self.step_chunk)

    def _ui_label(self, raw_label: str) -> str:
        return DISPLAY_LABELS.get(raw_label, raw_label)

    def _project_positions(self, samples: Any, dims: int | None = None) -> FloatArray:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2:
            return np.zeros((0, max(1, dims or 2)), dtype=float)
        target_dims = max(1, int(dims or self.dimension))
        if target_dims >= 3:
            if arr.shape[1] < target_dims:
                arr = np.pad(arr, ((0, 0), (0, target_dims - arr.shape[1])))
            return np.asarray(arr[:, :target_dims], dtype=float)

        axes = self._current_projection_axes()
        required_dims = max(axes) + 1 if axes else target_dims
        if arr.shape[1] < required_dims:
            arr = np.pad(arr, ((0, 0), (0, required_dims - arr.shape[1])))
        return np.asarray(arr[:, list(axes[:target_dims])], dtype=float)

    def _current_projection_axes(self) -> tuple[int, ...]:
        if self.dimension >= 3 or self.max_view_dimension < 3:
            return (0, 1, 2) if self.dimension >= 3 else (0, 1)
        if self._view_mode == 'xz':
            return (0, 2)
        if self._view_mode == 'yz':
            return (1, 2)
        return (0, 1)

    def _current_axis_labels(self) -> tuple[str, ...]:
        axes = self._current_projection_axes()
        return tuple(f"dim {axis + 1}" for axis in axes[: self.dimension])

    def _view_button_label(self) -> str:
        if self.dimension >= 3:
            return "view:3D"
        return f"view:{self._view_mode.upper()}"

    def _clear_legend_box(self) -> None:
        legend_box = getattr(self, '_legend_info_ax', None)
        if legend_box is None:
            return
        try:
            legend_box.remove()
        except Exception:
            pass
        self._legend_info_ax = None

    def _clear_plot_axes(self) -> None:
        self._clear_edge_arrows()
        legend = getattr(self, '_legend', None)
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                pass
            self._legend = None
        self._clear_legend_box()
        for attr in ('ax_theme_btn', 'text_metrics', 'text_switch', 'scatter', 'cluster_lines', 'ax'):
            artist = getattr(self, attr, None)
            if artist is None:
                continue
            try:
                artist.remove()
            except Exception:
                pass
            if attr != 'ax':
                setattr(self, attr, None)

    def _create_plot_axes(self) -> None:
        if hasattr(self, 'ax'):
            self._clear_plot_axes()
        self._current_axis_limit = None
        if self.dimension == 2:
            self.ax = self.fig.add_subplot()
        else:
            self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_position(self._plot_rect)

        pos = self._project_positions(getattr(self.model, 'syncmap', np.zeros((0, self.dimension))), self.dimension)
        colors = self._colors_for_labels(
            self._display_labels_for_current_mode(),
            singleton_gray=bool(self.use_pred_colors),
        )
        if self.dimension == 2:
            self.scatter = self.ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=colors,
                s=20,
                linewidths=0.0,
                edgecolors='none',
            )
            self.cluster_lines = LineCollection([], colors='tab:gray', linewidths=0.3, alpha=0.55, zorder=0.5)
            self.ax.add_collection(self.cluster_lines)
            ax2d = cast(Axes, self.ax)
            self.text_metrics = ax2d.text(
                0.02,
                0.02,
                s="NMI/ARI: --/--",
                transform=ax2d.transAxes,
                fontsize=9,
                va='bottom',
                ha='left',
                linespacing=1.15,
            )
            self.text_switch = ax2d.text(0.02, 0.16, s="", transform=ax2d.transAxes, fontsize=8.5, va='bottom', ha='left')
            self.text_switch.set_visible(False)
        else:
            ax3d = cast(Axes3D, self.ax)
            scatter_fn = cast(Callable[..., PathCollection], ax3d.scatter)
            self.scatter = scatter_fn(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                c=colors,
                s=20,
                depthshade=False,
                linewidths=0.0,
                edgecolors='none',
            )
            self.cluster_lines = Line3DCollection([], colors='tab:gray', linewidths=0.3, alpha=0.55, zorder=0.5)
            ax3d.add_collection3d(self.cluster_lines)
            self.text_metrics = ax3d.text2D(
                0.02,
                0.02,
                s="NMI/ARI: --/--",
                transform=self.ax.transAxes,
                fontsize=9,
                va='bottom',
                ha='left',
                linespacing=1.15,
            )
            self.text_switch = ax3d.text2D(0.02, 0.16, s="", transform=self.ax.transAxes, fontsize=8.5, va='bottom', ha='left')
            self.text_switch.set_visible(False)
            ax3d.view_init(elev=20, azim=-65)

        axis_labels = self._current_axis_labels()
        self.ax.set_xlabel(axis_labels[0] if len(axis_labels) >= 1 else "dim 1")
        self.ax.set_ylabel(axis_labels[1] if len(axis_labels) >= 2 else "dim 2")
        if isinstance(self.ax, Axes3D):
            cast(Axes3D, self.ax).set_zlabel(axis_labels[2] if len(axis_labels) >= 3 else "dim 3")
        self._apply_mesh_visibility()
        self._set_title()
        self._build_legend(self._legend_labels_for_current_mode())
        self._build_theme_button()

    def _toggle_view_dimension(self) -> None:
        if self.max_view_dimension < 3:
            return
        cycle = ('3d', 'xy', 'xz', 'yz')
        current = self._view_mode if self._view_mode in cycle else ('3d' if self.dimension >= 3 else 'xy')
        next_idx = (cycle.index(current) + 1) % len(cycle)
        self._view_mode = cycle[next_idx]
        self.dimension = 3 if self._view_mode == '3d' else 2
        if self.b_view is not None:
            self.b_view.label.set_text(self._view_button_label())
        self._create_plot_axes()
        self._apply_theme()
        self._update_metrics_and_plot(recompute_labels=False)

    def _toggle_mesh(self) -> None:
        self.show_mesh = not bool(self.show_mesh)
        if self.b_mesh is not None:
            self.b_mesh.label.set_text(self._format_toggle_label("mesh", self.show_mesh))
        self._apply_mesh_visibility()
        self.fig.canvas.draw_idle()

    def _apply_mesh_visibility(self, limit: float | None = None) -> None:
        """Show a readable cubic reference mesh only in the 3D space view."""
        if not isinstance(getattr(self, "ax", None), Axes3D):
            return
        ax3d = cast(Axes3D, self.ax)
        if not bool(getattr(self, "show_mesh", False)):
            ax3d.grid(False)
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            return

        if limit is None:
            current = getattr(self, "_current_axis_limit", None)
            if current is None:
                xlim = ax3d.get_xlim3d()
                limit = max(abs(float(xlim[0])), abs(float(xlim[1])), 1.0)
            else:
                limit = float(current)
        ticks = np.linspace(-max(float(limit), 1.0), max(float(limit), 1.0), 5)
        ax3d.set_xticks(ticks)
        ax3d.set_yticks(ticks)
        ax3d.set_zticks(ticks)
        ax3d.grid(True)
        grid_color = getattr(self, "_palette", {}).get("grid", "#777777")
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            try:
                axis._axinfo["grid"]["color"] = to_rgba(grid_color, 0.55)
                axis._axinfo["grid"]["linewidth"] = 0.7
            except Exception:
                pass

    def _on_mouse_press(self, event: Any) -> None:
        if getattr(event, 'inaxes', None) is self.ax:
            self._interaction_active = True

    def _on_mouse_release(self, event: Any) -> None:
        if self._interaction_active:
            self._interaction_active = False

    def _dataset_switch_enabled(self) -> bool:
        return (
            not self.readonly
            and callable(getattr(self.env, 'switch_dataset', None))
            and len(self._available_datasets()) > 1
        )

    def _inspect_dataset_signature(self, candidate: Path) -> tuple[int, bool] | None:
        try:
            content = candidate.read_text(encoding='utf-8')
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
                attrs = node_match.group(2)
                label_match = label_pattern.search(attrs)
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
        is_hierarchical = saw_multilevel_label or (labeled_nodes and len(labeled_nodes) != total_nodes) or saw_unlabeled_node
        return observed_nodes, is_hierarchical

    def _is_dataset_switch_compatible(self, candidate: Path) -> bool:
        signature = self._inspect_dataset_signature(candidate)
        if signature is None:
            return False
        observed_count, is_hierarchical = signature
        if int(observed_count) != int(self.output_size):
            return False
        env_is_hierarchical = bool(getattr(self.env, 'is_hierarchical', False))
        if bool(is_hierarchical) != env_is_hierarchical:
            return False
        return True

    def _update_dataset_readout(self) -> None:
        if self.dataset_text_artist is None:
            return
        current_name = Path(self._dataset_path()).name
        self.dataset_text_artist.set_text(current_name)

    def _infer_eval_label_level(self) -> int:
        depth = max(1, int(getattr(self.env, 'hierarchy_depth', 1)))
        if not bool(getattr(self.env, 'is_hierarchical', False)):
            return 0
        if self._label_level_override is not None:
            return int(np.clip(self._label_level_override, 0, depth - 1))
        return 0

    def _hierarchy_display_depth(self) -> int:
        if hasattr(self.env, 'true_label_levels') and hasattr(self.env.true_label_levels, 'shape'):
            return max(1, int(self.env.true_label_levels.shape[1]))
        if hasattr(self.env, 'label_levels') and hasattr(self.env.label_levels, 'shape'):
            return max(1, int(self.env.label_levels.shape[1]))
        if hasattr(self.env, 'hierarchy_depth'):
            return max(1, int(getattr(self.env, 'hierarchy_depth', 1)))
        return 1

    def _clamp_display_label_level(self, level_idx: int | None = None) -> int:
        depth = self._hierarchy_display_depth()
        current = self._display_label_level if level_idx is None else int(level_idx)
        return int(np.clip(current, 0, depth - 1))

    def _true_labels_for_metric(self) -> IntArray:
        if bool(getattr(self.env, 'is_hierarchical', False)):
            try:
                return np.asarray(self.env.trueLabel(level=self._eval_label_level), dtype=int)
            except TypeError:
                pass
        return np.asarray(self.env.trueLabel(), dtype=int)

    def _true_labels_for_display(self) -> IntArray:
        if bool(getattr(self.env, 'is_hierarchical', False)):
            try:
                return np.asarray(self.env.trueLabel(level=self._clamp_display_label_level()), dtype=int)
            except TypeError:
                pass
        return self._true_labels_for_metric()

    def _display_level_key(self, level_idx: int | None = None) -> str:
        return cluster_level_key_from_index(self._clamp_display_label_level(level_idx))

    def _display_level_text(self, level_idx: int | None = None) -> str:
        return f"L{self._clamp_display_label_level(level_idx)}"

    def _display_button_label(self) -> str:
        return f"level:{self._display_level_text()}"

    def _label_color_lookup(self, labels: Any) -> dict[int, tuple[float, float, float, float]]:
        labels_arr = np.asarray(labels, dtype=int).reshape(-1)
        valid_labels = sorted({int(label) for label in labels_arr.tolist() if int(label) >= 0})
        cmap = plt.get_cmap('tab20')
        palette_size = max(int(getattr(cmap, 'N', 20)), 1)
        color_lookup: dict[int, tuple[float, float, float, float]] = {}
        for idx, label in enumerate(valid_labels):
            rgba = cmap(idx % palette_size)
            color_lookup[label] = (
                float(rgba[0]),
                float(rgba[1]),
                float(rgba[2]),
                float(rgba[3]),
            )
        return color_lookup

    def _colors_for_labels(
        self,
        labels: Any,
        *,
        singleton_gray: bool,
    ) -> NDArray[np.float64]:
        labels_arr = np.asarray(labels, dtype=int).reshape(-1)
        if labels_arr.size == 0:
            return np.zeros((0, 4), dtype=float)

        colors = np.zeros((labels_arr.size, 4), dtype=float)
        color_lookup = self._label_color_lookup(labels_arr)
        for label, rgba in color_lookup.items():
            colors[labels_arr == label] = rgba

        if singleton_gray:
            valid = labels_arr >= 0
            if np.any(valid):
                valid_labels = labels_arr[valid]
                unique, counts = np.unique(valid_labels, return_counts=True)
                singleton_labels = {int(label) for label, count in zip(unique.tolist(), counts.tolist()) if int(count) <= 1}
                if singleton_labels:
                    singleton_mask = np.array([int(label) in singleton_labels for label in labels_arr], dtype=bool)
                    colors[singleton_mask] = self._singleton_color

        colors[labels_arr < 0] = self._singleton_color
        return colors

    def _colors_for_true_labels(self, labels: Any) -> NDArray[np.float64]:
        return self._colors_for_labels(labels, singleton_gray=False)

    def _configured_organize_level(self) -> str:
        configured = self._strategy_params.get('organize_level')
        if configured is None:
            configured = getattr(self.model, 'organize_level', None)
        return canonicalize_organize_level(configured, default=DEFAULT_HIERARCHY_ORGANIZE_LEVEL)

    def _build_live_eval_readout_config(self) -> EvaluationReadoutConfig:
        default = EvaluationReadoutConfig()
        method = str(self._strategy_params.get("eval_readout_method", default.method)).strip().lower() or default.method
        eps_raw = self._strategy_params.get("eval_readout_eps", default.eps)
        min_samples_raw = self._strategy_params.get("eval_readout_min_samples", default.min_samples)
        max_clusters_raw = self._strategy_params.get("eval_readout_max_clusters", default.max_clusters)
        standardize_raw = self._strategy_params.get("eval_readout_standardize", default.standardize)
        eps = None if eps_raw in {None, "", "none", "auto"} else float(eps_raw)
        min_samples = max(1, int(min_samples_raw))
        max_clusters = max(2, int(max_clusters_raw))
        standardize = bool(standardize_raw)
        return EvaluationReadoutConfig(
            method=method,
            eps=eps,
            min_samples=min_samples,
            max_clusters=max_clusters,
            standardize=standardize,
        )

    def _formal_readout_label(self) -> str:
        cfg = self._formal_eval_readout_config
        return f"unified:{_resolve_external_readout_method(cfg, self.env)}"

    def _display_readout_label(self) -> str:
        return self._formal_readout_label()

    def _update_formal_readout_controls(
        self,
        *,
        method: str | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> None:
        cfg = self._formal_eval_readout_config
        next_method = cfg.method
        if method is not None:
            method_key = str(method).strip().lower()
            if method_key in {"auto", "dbscan", "hdbscan", "louvain", "agglomerative", "spectral", "mst_gaps", "mst_ladder"}:
                next_method = method_key
        self._formal_eval_readout_config = replace(
            cfg,
            method=next_method,
            eps=cfg.eps if eps is None else float(eps),
            min_samples=cfg.min_samples if min_samples is None else max(1, int(min_samples)),
        )

    def _organize_level_key(self) -> str:
        return self._configured_organize_level()

    def _hierarchy_focus_options(self) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        depth = self._hierarchy_display_depth()

        for level_idx in range(depth):
            level_key = self._display_level_key(level_idx)
            level_display = f"L{level_idx}"
            options.append((level_key, level_display))

        return options

    def _focus_button_label(self) -> str:
        return self._display_button_label()

    def _score_button_label(self) -> str:
        return "score:overall" if self._score_display_mode == 'summary' else "score:levels"

    def _cycle_hierarchy_focus(self) -> None:
        if not bool(getattr(self.env, 'is_hierarchical', False)):
            return
        depth = self._hierarchy_display_depth()
        self._display_label_level = (self._clamp_display_label_level() + 1) % max(depth, 1)
        self._eval_label_level = self._infer_eval_label_level()
        if self.b_focus is not None:
            self.b_focus.label.set_text(self._focus_button_label())
        self._build_legend(self._legend_labels_for_current_mode())
        self._update_metrics_and_plot(recompute_labels=False)

    def _formal_evaluation(self, *, force: bool = False) -> dict[str, Any] | None:
        if not hasattr(self.model, 'organize'):
            return None
        if (
            not force
            and self._formal_eval_cache is not None
            and self._formal_eval_step == int(self._last_cluster_step)
        ):
            return self._formal_eval_cache
        try:
            result = _select_evaluation_prediction(
                self.model,
                self.env,
                self._formal_eval_metric,
                readout_config=self._formal_eval_readout_config,
                label_level=self._label_level_override,
            )
        except Exception:
            return None
        self._formal_eval_cache = dict(result)
        self._formal_eval_step = int(self._last_cluster_step)
        return self._formal_eval_cache

    def _hierarchy_outputs(self, *, force: bool = False) -> dict[str, NDArray[np.integer[Any]]]:
        if not bool(getattr(self.env, 'is_hierarchical', False)) or not hasattr(self.model, 'organize'):
            return {}
        if (
            not force
            and self._hierarchy_outputs_cache is not None
            and self._hierarchy_outputs_step == self._last_cluster_step
        ):
            return self._hierarchy_outputs_cache

        outputs: dict[str, NDArray[np.integer[Any]]] = {}
        try:
            outputs = cast(
                dict[str, NDArray[np.integer[Any]]],
                unified_readout_outputs(self.model, self.env, self._formal_eval_readout_config),
            )
        except Exception:
            outputs = {}

        self._hierarchy_outputs_cache = outputs
        self._hierarchy_outputs_step = self._last_cluster_step
        return outputs

    def _current_predicted_labels(self, *, force: bool = False) -> NDArray[np.integer[Any]] | None:
        if bool(getattr(self.env, 'is_hierarchical', False)) and hasattr(self.model, 'organize'):
            outputs = self._hierarchy_outputs(force=force)
            labels = outputs.get(self._organize_level_key())
            if labels is not None:
                return labels
        formal = self._formal_evaluation(force=force)
        if formal is not None:
            return np.asarray(formal.get("labels"), dtype=int)
        return None

    def _nearest_hierarchy_output(
        self,
        outputs: dict[str, NDArray[np.integer[Any]]],
        level_idx: int,
    ) -> NDArray[np.integer[Any]] | None:
        desired = self._clamp_display_label_level(level_idx)
        desired_key = self._display_level_key(desired)
        labels = outputs.get(desired_key)
        if labels is not None:
            return labels

        available: list[tuple[int, NDArray[np.integer[Any]]]] = []
        for key, value in outputs.items():
            parsed = parse_hierarchy_label_index(key)
            if parsed is not None:
                available.append((int(parsed), value))
        if not available:
            return None

        available.sort(key=lambda item: (abs(item[0] - desired), item[0]))
        return available[0][1]

    def _predicted_labels_for_display(self, *, force: bool = False) -> NDArray[np.integer[Any]] | None:
        if bool(getattr(self.env, 'is_hierarchical', False)) and hasattr(self.model, 'organize'):
            outputs = self._hierarchy_outputs(force=force)
            labels = self._nearest_hierarchy_output(outputs, self._display_label_level)
            if labels is not None:
                return np.asarray(labels, dtype=int)
        labels = self._current_predicted_labels(force=force)
        if labels is None:
            return None
        return np.asarray(labels, dtype=int)

    def _display_labels_for_current_mode(self, *, force: bool = False) -> NDArray[np.integer[Any]]:
        if bool(getattr(self, 'use_pred_colors', False)):
            predicted = self._predicted_labels_for_display(force=force)
            if predicted is not None:
                return predicted
        return self._true_labels_for_display()

    def _legend_labels_for_current_mode(self) -> NDArray[np.integer[Any]]:
        return self._display_labels_for_current_mode(force=False)

    @staticmethod
    def _legend_label_name_map(labels: NDArray[np.integer[Any]]) -> dict[int, str]:
        uniq = [int(v) for v in np.unique(np.asarray(labels, dtype=int)).tolist()]
        names: dict[int, str] = {}
        next_class = 1
        for label in uniq:
            if label < 0:
                names[int(label)] = "noise"
                continue
            names[int(label)] = f"class {next_class}"
            next_class += 1
        return names

    def _metric_payload(self, true_y: Any, pred_y: Any) -> dict[str, float] | None:
        true_arr = np.asarray(true_y, dtype=int)
        pred_arr = np.asarray(pred_y, dtype=int)
        if true_arr.ndim != 1 or pred_arr.ndim != 1 or true_arr.shape[0] != pred_arr.shape[0]:
            return None
        payload = clustering_metric_bundle(true_arr, pred_arr)
        payload.setdefault('nmi_noise_singletons', float(payload['nmi_raw']))
        payload.setdefault('ari_noise_singletons', float(payload['ari_raw']))
        payload.setdefault('classified_covered_ratio', 1.0)
        payload.setdefault('singleton_cluster_ratio', 0.0)
        return payload

    def _canonical_hierarchy_metric_lines(
        self,
        outputs: dict[str, NDArray[np.integer[Any]]],
    ) -> list[tuple[str, dict[str, float]]]:
        entries: list[tuple[str, dict[str, float]]] = []
        depth = 1
        if hasattr(self.env, 'label_levels') and hasattr(self.env.label_levels, 'shape'):
            depth = max(1, int(self.env.label_levels.shape[1]))
        elif hasattr(self.env, 'hierarchy_depth'):
            depth = max(1, int(getattr(self.env, 'hierarchy_depth', 1)))
        else:
            depth = len([k for k in outputs.keys() if k.startswith('cluster_level')])
            depth = max(1, depth)

        for level_idx in range(depth):
            pred = outputs.get(cluster_level_key_from_index(level_idx))
            if pred is None:
                continue
            payload = self._metric_payload(self.env.trueLabel(level=level_idx), pred)
            if payload is not None:
                entries.append((cluster_level_display(level_idx), payload))
        return entries

    @staticmethod
    def _hierarchy_mean_metric_payload(
        entries: list[tuple[str, dict[str, float]]],
        *,
        expected_depth: int,
    ) -> dict[str, float] | None:
        if not entries:
            return None
        payloads = [payload for _, payload in entries]
        result: dict[str, float] = {}
        values = [float(payload.get("ari_raw", float("nan"))) for payload in payloads]
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            return None
        missing = max(0, int(expected_depth) - len(finite))
        result["ari_raw"] = float(np.mean(finite[:expected_depth] + [0.0] * missing))
        return result

    def _format_metric_text(self, labels_pred: NDArray[np.integer[Any]] | None) -> str:
        formal_eval = self._formal_evaluation(force=False)
        if formal_eval is None:
            return f"NMI/ARI: --/-- | t={self.total_steps}"

        true_y = np.asarray(formal_eval.get("true_labels"), dtype=int)
        formal_pred = np.asarray(formal_eval.get("labels"), dtype=int)
        formal_payload = self._metric_payload(true_y, formal_pred)
        if formal_payload is None:
            return f"NMI/ARI: --/-- | t={self.total_steps}"

        if not bool(getattr(self.env, 'is_hierarchical', False)):
            return (
                f"NMI/ARI: "
                f"{formal_payload['nmi_raw']:.3f}/"
                f"{formal_payload['ari_raw']:.3f} "
                f"| t={self.total_steps}"
            )

        outputs = self._hierarchy_outputs(force=False)
        entries = self._canonical_hierarchy_metric_lines(outputs)
        expected_depth = max(1, int(getattr(self.env, 'hierarchy_depth', len(entries) or 1)))
        if self._score_display_mode == 'levels' and entries:
            total_payload = self._hierarchy_mean_metric_payload(
                entries,
                expected_depth=expected_depth,
            )
            lines = [
                (
                    f"ALL(mean ARI): {total_payload['ari_raw']:.3f} "
                    f"| t={self.total_steps}"
                )
                if total_payload is not None
                else f"ALL(mean ARI): -- | t={self.total_steps}",
                "NMI/ARI",
                *[
                    f"{label}: {payload['nmi_raw']:.3f}/{payload['ari_raw']:.3f}"
                    for label, payload in entries
                ],
            ]
            return "\n".join(lines)

        formal_level = cluster_level_display(int(formal_eval.get("truth_level", self._eval_label_level)))
        total_payload = self._hierarchy_mean_metric_payload(
            entries,
            expected_depth=expected_depth,
        )
        if total_payload is None:
            return (
                f"Target {formal_level} (fixed): "
                f"{formal_payload['nmi_raw']:.3f}/"
                f"{formal_payload['ari_raw']:.3f} "
                f"| t={self.total_steps}"
            )
        return "\n".join(
            [
                (
                    f"Hierarchy total ARI(mean): "
                    f"{total_payload['ari_raw']:.3f} | t={self.total_steps}"
                ),
                (
                    f"Target {formal_level} (fixed): "
                    f"{formal_payload['nmi_raw']:.3f}/"
                    f"{formal_payload['ari_raw']:.3f}"
                ),
            ]
        )

    def _format_switch_text(self) -> str:
        getter = getattr(self.env, 'currentSwitchNodeName', None)
        if not callable(getter):
            return ""
        try:
            switch_node = getter()
        except Exception:
            return ""
        if not switch_node:
            return ""
        return f"switch: {switch_node}"

    def _sync_ui_from_model(self, *, reset_display_state: bool) -> None:
        syncmap_arr = np.asarray(getattr(self.model, 'syncmap', np.zeros((0, 2))), dtype=float)
        syncmap_dims = int(syncmap_arr.shape[1]) if syncmap_arr.ndim == 2 and syncmap_arr.size > 0 else 2
        self.cluster_dimension = max(
            1,
            min(int(getattr(self.model, 'dimensions', syncmap_dims or 1)), max(syncmap_dims, 1)),
        )
        self.max_view_dimension = 3 if syncmap_dims >= 3 else 2
        if self.max_view_dimension < 3:
            self._view_mode = 'xy'
            self.dimension = 2
        else:
            if self._view_mode not in {'3d', 'xy', 'xz', 'yz'}:
                self._view_mode = '3d' if self.dimension >= 3 else 'xy'
            self.dimension = 3 if self._view_mode == '3d' else 2
        self._eval_label_level = self._infer_eval_label_level()
        self._display_label_level = self._clamp_display_label_level(self._display_label_level)

        self._suppress_slider_events = True
        if not self.readonly:
            graph_model = self._graph_control_target_model()
            curr_base_wall = getattr(graph_model, 'base_wall', getattr(self.model, 'base_wall', 100.0))
            curr_density = getattr(graph_model, 'density_ref', getattr(self.model, 'density_ref', 75.0))
            curr_k_attr = min(self._k_attr_slider_value(), self._k_attr_slider_limit())

            def _set_optional_slider(slider: Slider | None, name: str, value: float) -> None:
                if slider is not None:
                    self._require_slider(slider, name).set_val(value)

            self._require_slider(self.s_base_wall, 'base_wall').set_val(curr_base_wall)
            self._require_slider(self.s_density_ref, 'density_ref').set_val(curr_density)
            slider_space = self._require_slider(self.s_space_size, 'space_size')
            self._space_override_enabled = self.init_space_size > 0.0
            slider_space.set_val(self.init_space_size if self._space_override_enabled else 0.0)
            self._update_space_slider_display()

            self._require_slider(
                self.s_adaptation_rate,
                'adaptation_rate',
            ).set_val(self._live_adaptation_rate())
            self._require_slider(self.s_memory_steps, 'memory_steps').set_val(
                float(getattr(graph_model, 'stats_memory_steps', getattr(self.model, 'stats_memory_steps', 1000)))
            )
            _set_optional_slider(self.s_stable_token_evidence_memory_scale, 'entity_mem_scale',
                float(getattr(self.model, 'stable_token_evidence_memory_scale', 10.0))
            )
            _set_optional_slider(self.s_stable_token_evidence_min_members, 'entity_members',
                float(getattr(self.model, 'stable_token_evidence_min_members', 2))
            )
            _set_optional_slider(self.s_stable_token_evidence_min_stable_refreshes, 'entity_age',
                float(getattr(self.model, 'stable_token_evidence_min_stable_refreshes', 5))
            )
            _set_optional_slider(self.s_stable_token_evidence_semantic_z_score, 'entity_sem_z',
                float(getattr(self.model, 'stable_token_evidence_semantic_z_score', 1.645))
            )
            _set_optional_slider(self.s_stable_token_evidence_min_self_mass, 'entity_self_mass',
                float(getattr(self.model, 'stable_token_evidence_min_self_mass', 5.0))
            )
            _set_optional_slider(self.s_stable_token_evidence_self_cross_ratio, 'entity_self_ratio',
                float(getattr(self.model, 'stable_token_evidence_self_cross_ratio', 1.25))
            )
            step_chunk_slider = self._require_slider(self.s_step_chunk, 'step_chunk')
            step_chunk_slider.set_val(float(self._step_chunk_slider_index()))
            if getattr(step_chunk_slider, 'valtext', None) is not None:
                step_chunk_slider.valtext.set_text(str(self.step_chunk))

            self._update_k_attr_slider_range(curr_k_attr)
            self._require_slider(self.s_k_attr, 'k_attr').set_val(curr_k_attr)
            self._require_slider(self.s_k_repulse, 'k_repulse').set_val(
                float(getattr(graph_model, 'k_repulse', getattr(self.model, 'k_repulse', 0.0)))
            )
            self._require_slider(self.s_primitive_active_attraction_strength, 'primitive_active_attraction_strength').set_val(
                float(getattr(graph_model, 'primitive_active_attraction_strength', getattr(self.model, 'primitive_active_attraction_strength', 0.0)))
            )
            self._require_slider(self.s_primitive_inactive_repulsion_strength, 'primitive_inactive_repulsion_strength').set_val(
                float(getattr(graph_model, 'primitive_inactive_repulsion_strength', getattr(self.model, 'primitive_inactive_repulsion_strength', 0.0)))
            )
            _set_optional_slider(self.s_k_macro_repulse, 'k_macro_repulse',
                float(getattr(graph_model, 'k_macro_repulse', getattr(self.model, 'k_macro_repulse', 0.0)))
            )
            _set_optional_slider(self.s_macro_cross_attr_scale, 'macro_cross_attr_scale',
                float(getattr(graph_model, 'macro_cross_attr_scale', getattr(self.model, 'macro_cross_attr_scale', 0.0)))
            )
            _set_optional_slider(self.s_macro_centroid_distance_ratio, 'macro_centroid_distance_ratio',
                float(getattr(graph_model, 'macro_centroid_distance_ratio', getattr(self.model, 'macro_centroid_distance_ratio', 4.0)))
            )
            _set_optional_slider(self.s_macro_centroid_max_step_ratio, 'macro_centroid_max_step_ratio',
                float(getattr(graph_model, 'macro_centroid_max_step_ratio', getattr(self.model, 'macro_centroid_max_step_ratio', 1.0)))
            )
            _set_optional_slider(self.s_k_entity_boundary_repulse, 'k_entity_boundary_repulse',
                float(getattr(graph_model, 'k_entity_boundary_repulse', getattr(self.model, 'k_entity_boundary_repulse', 0.0)))
            )
            _set_optional_slider(self.s_entity_boundary_distance_ratio, 'entity_boundary_distance_ratio',
                float(getattr(graph_model, 'entity_boundary_distance_ratio', getattr(self.model, 'entity_boundary_distance_ratio', 3.16)))
            )
            _set_optional_slider(self.s_entity_boundary_max_step_ratio, 'entity_boundary_max_step_ratio',
                float(getattr(graph_model, 'entity_boundary_max_step_ratio', getattr(self.model, 'entity_boundary_max_step_ratio', 0.5)))
            )
            _set_optional_slider(self.s_k_entity_sibling_attr, 'k_entity_sibling_attr',
                float(getattr(graph_model, 'k_entity_sibling_attr', getattr(self.model, 'k_entity_sibling_attr', 0.0)))
            )
            _set_optional_slider(self.s_entity_sibling_target_ratio, 'entity_sibling_target_ratio',
                float(getattr(graph_model, 'entity_sibling_target_ratio', getattr(self.model, 'entity_sibling_target_ratio', 3.16)))
            )
            _set_optional_slider(self.s_entity_sibling_max_step_ratio, 'entity_sibling_max_step_ratio',
                float(getattr(graph_model, 'entity_sibling_max_step_ratio', getattr(self.model, 'entity_sibling_max_step_ratio', 0.5)))
            )
            _set_optional_slider(self.s_entity_sibling_boundary_release_ratio, 'entity_sibling_boundary_release_ratio',
                float(getattr(graph_model, 'entity_sibling_boundary_release_ratio', getattr(self.model, 'entity_sibling_boundary_release_ratio', 1.25)))
            )
            _set_optional_slider(self.s_entity_sibling_boundary_power, 'entity_sibling_boundary_power',
                float(getattr(graph_model, 'entity_sibling_boundary_power', getattr(self.model, 'entity_sibling_boundary_power', 2.0)))
            )
            _set_optional_slider(self.s_line_obstacle_strength, 'line_obstacle_strength',
                float(getattr(graph_model, 'line_obstacle_strength', getattr(self.model, 'line_obstacle_strength', 0.0)))
            )
            _set_optional_slider(self.s_line_obstacle_radius_rel, 'line_obstacle_radius_rel',
                float(getattr(graph_model, 'line_obstacle_radius_rel', getattr(self.model, 'line_obstacle_radius_rel', 1.0)))
            )
            self._require_slider(self.s_knn, 'knn').set_val(getattr(graph_model, 'knn', getattr(self, 'knn', 12)))
            self._require_slider(self.s_graphTheta, 'graph_theta').set_val(
                getattr(graph_model, 'graph_theta', getattr(self, 'graph_min_weight', 0.1))
            )
            self._require_slider(self.s_graphR0, 'graph_r0').set_val(self._graph_r0_slider_value())
            self._require_slider(self.s_graphPush, 'T_push').set_val(
                getattr(graph_model, 'T_push', getattr(self, 'T_push', 10))
            )

            self._require_slider(self.s_eps, 'eps').set_val(float(self.dbscan_eps))
            self._require_slider(self.s_readout_eps, 'readout_eps').set_val(
                float(self.dbscan_eps if self.dbscan_readout_eps is None else self.dbscan_readout_eps)
            )
            self._require_slider(self.s_minp, 'minPts').set_val(float(self.dbscan_min_samples))
            self._require_slider(self.s_louvain_k_dist, 'louvain_k_dist').set_val(float(self.louvain_distance_factor))
            self._require_slider(self.s_louvain_knn_k, 'louvain_knn_k').set_val(float(self.louvain_knn_k))
        self._suppress_slider_events = False

        self._sync_live_clustering_state_from_model()
        if self.b_embed is not None:
            self.b_embed.label.set_text(f"cluster:{self.embed_cluster_method}")

        if reset_display_state:
            self.use_pred_colors = False
            self.show_cluster_edges = False

        if self.b_pred is not None:
            self.b_pred.label.set_text(self._format_toggle_label('predict', self.use_pred_colors))
        if self.b_edges is not None:
            self.b_edges.label.set_text(self._format_toggle_label('edges', self.show_cluster_edges))
        if self.b_auto_eps is not None:
            self.b_auto_eps.label.set_text(self._format_toggle_label('auto_eps', self.dbscan_auto_eps))
        if self.b_louvain_mutual is not None:
            self.b_louvain_mutual.label.set_text(self._format_toggle_label('mutual_knn', self.louvain_mutual_knn))
        if self.b_view is not None:
            self.b_view.label.set_text(self._view_button_label())
        if self.b_theme is not None:
            self.b_theme.label.set_text(self._theme_button_label())
        if self.b_focus is not None:
            self.b_focus.label.set_text(self._focus_button_label())
        if self.b_score is not None:
            self.b_score.label.set_text(self._score_button_label())

        self._create_plot_axes()
        self._apply_discrete_controls()
        self._last_labels = None
        self._last_cluster_step = -10**9
        self._update_dataset_readout()
        self._apply_theme()
        self._update_metrics_and_plot(recompute_labels=True)
        self._update_force_monitor()
        self.fig.canvas.draw_idle()

    def _set_running_state(self, running: bool) -> None:
        self.running = bool(running)
        if self.b_toggle is not None:
            self.b_toggle.label.set_text('Pause' if self.running else 'Start')

    def _handle_reset(self) -> None:
        was_running = bool(getattr(self, 'running', False))
        use_pred_colors = bool(getattr(self, 'use_pred_colors', False))
        show_cluster_edges = bool(getattr(self, 'show_cluster_edges', False))

        self._seed = self._initial_seed
        self._setup_simulation(self._ds_path)
        self._auto_pause_announced = False
        self._reset_playback()
        self._close_auxiliary_figure("input")
        self._close_auxiliary_figure("graph")
        self._reset_curve_history()
        self._sync_ui_from_model(reset_display_state=False)

        self.use_pred_colors = use_pred_colors
        self.show_cluster_edges = show_cluster_edges
        if self.b_pred is not None:
            self.b_pred.label.set_text(self._format_toggle_label('predict', self.use_pred_colors))
        if self.b_edges is not None:
            self.b_edges.label.set_text(self._format_toggle_label('edges', self.show_cluster_edges))

        self._set_running_state(was_running)
        self._update_metrics_and_plot(recompute_labels=True)
        self._update_force_monitor()
        self.fig.canvas.draw_idle()

    def _graph_r0_slider_field(self) -> str:
        level0_model = getattr(self.model, 'level0', self.model)
        if str(getattr(level0_model, 'graph_r0_mode', 'relative')).lower() == 'fixed':
            return 'graph_r0_fixed'
        return 'graph_r0_rel'

    def _base_dynamics_model(self) -> Any:
        return getattr(self.model, 'level0', self.model)

    def _graph_control_target_model(self) -> Any:
        return self._base_dynamics_model()

    def _graph_control_support(self) -> dict[str, bool]:
        target_model = self._graph_control_target_model()
        return {
            'k_attr': hasattr(target_model, 'k_attr'),
            'knn': hasattr(target_model, 'knn'),
            'graph_theta': hasattr(target_model, 'graph_theta'),
            'graph_r0': (
                hasattr(target_model, 'graph_r0')
                or hasattr(target_model, 'graph_r0_fixed')
                or hasattr(target_model, 'graph_r0_rel')
            ),
            'T_push': hasattr(target_model, 'T_push'),
        }

    def _apply_graph_control_highlights(self, active_color: str, inactive_color: str) -> None:
        support = self._graph_control_support()
        slider_rest = getattr(self, '_color_slider_rest', inactive_color)
        graph_widgets = {
            'k_attr': getattr(self, 's_k_attr', None),
            'knn': getattr(self, 's_knn', None),
            'graph_theta': getattr(self, 's_graphTheta', None),
            'graph_r0': getattr(self, 's_graphR0', None),
            'T_push': getattr(self, 's_graphPush', None),
        }
        for key, widget in graph_widgets.items():
            enabled = bool(support.get(key, False))
            self._color_slider_handle(widget, active_color if enabled else slider_rest)
            self._set_widget_enabled(widget, enabled)

    def _k_attr_slider_value(self) -> float:
        level0_model = self._graph_control_target_model()
        fallback = getattr(self.model, 'k_attr', 0.1)
        return float(getattr(level0_model, 'k_attr', fallback))

    def _k_attr_slider_limit(self) -> float:
        base_value = max(float(getattr(self, '_k_attr_slider_default', self._k_attr_slider_value())), 0.0)
        return max(1.2, base_value * 4.0)

    def _update_k_attr_slider_range(self, target_value: float | None = None) -> None:
        slider = getattr(self, 's_k_attr', None)
        if slider is None:
            return
        desired_max = self._k_attr_slider_limit()
        if abs(float(slider.valmax) - desired_max) > 1e-9:
            slider.valmax = desired_max
            slider.ax.set_xlim(slider.valmin, slider.valmax)
        current = max(float(target_value if target_value is not None else slider.val), 0.0)
        if current > desired_max:
            prev_state = self._suppress_slider_events
            try:
                self._suppress_slider_events = True
                slider.set_val(desired_max)
            finally:
                self._suppress_slider_events = prev_state

    def _graph_r0_slider_value(self) -> float:
        level0_model = self._graph_control_target_model()
        field = self._graph_r0_slider_field()
        fallback = getattr(level0_model, 'graph_r0_rel', 1.0)
        return float(getattr(level0_model, field, fallback))

    def _setup_simulation(self, dataset: str) -> None:
        """
        初始化仿真环境和模型。
        核心逻辑：完全依赖统一工厂 create_model，不再进行手动分发。
        """
        use_preloaded = self._preloaded_model is not None and self._preloaded_env is not None
        if use_preloaded:
            self.model = self._preloaded_model
            self.env = self._preloaded_env
        else:
            np.random.seed(int(self._seed))
            try:
                # === 唯一入口：调用工厂 ===
                model_params = {
                    key: value
                    for key, value in self._strategy_params.items()
                    if key != "seed"
                }
                self.model, self.env = create_model(
                    dataset_path=dataset,
                    strategy_name=self._strategy_name,
                    strategy_params=model_params,
                    trajectory_interval=10000,
                    environment_seed=int(self._seed),
                )
            except Exception as e:
                print(f"Error creating model: {e}")
                raise e
            
        self.output_size = self.env.getOutputSize()
        self._impl_name = (self._variant_name or self._strategy_name).upper() # 用于显示标题
        self._eval_label_level = self._infer_eval_label_level()
        self._hierarchy_outputs_cache = None
        self._hierarchy_outputs_step = -10**9
        self._formal_eval_cache = None
        self._formal_eval_step = -10**9

        # 读取一些默认值用于 UI 初始化
        ADAPTATION_RATE = 0.1
        self.default_step_scale = self._live_adaptation_rate(ADAPTATION_RATE)
        self.total_steps = 0
        
        # 图参数默认值
        graph_model = self._graph_control_target_model()
        self.T_push: int = getattr(graph_model, 'T_push', getattr(self.model, 'T_push', 10))
        self.knn: int = getattr(graph_model, 'knn', getattr(self.model, 'knn', 3))
        self.graph_min_weight: float = getattr(
            graph_model,
            'graph_theta',
            getattr(self.model, 'graph_theta', 0.10),
        )
        self.graph_wscale = getattr(graph_model, 'k_attr', getattr(self.model, 'k_attr', 0.1))
        self.graph_r0_rel: float = self._graph_r0_slider_value()
        self._k_attr_slider_default = self._k_attr_slider_value()

        # 聚类参数默认值直接从当前目标模型读取；不再重写为 level-specific 参数。
        self._sync_live_clustering_state_from_model()
        
        # 补丁：确保 update_params 方法存在
        if not hasattr(self.model, "update_params"):
            print(f"[Warn] Model {type(self.model)} missing update_params, adding dummy.")
            self.model.update_params = lambda **kwargs: None

        # Live UI should reflect the model defaults here, not rewrite them.

    def _live_cluster_target_model(self) -> Any:
        if self._uses_direct_model_labels():
            return self.model
        return getattr(self.model, 'level0', self.model)

    def _live_cluster_update_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def _invalidate_clustering_cache(self) -> None:
        self._last_labels = None
        self._last_cluster_step = -10**9
        self._hierarchy_outputs_cache = None
        self._hierarchy_outputs_step = -10**9
        self._formal_eval_cache = None
        self._formal_eval_step = -10**9

    def _sync_live_clustering_state_from_model(self) -> None:
        if self._uses_direct_model_labels():
            self.embed_cluster_method = 'model'
            return

        cluster_model = self._live_cluster_target_model()
        self.embed_cluster_method = str(getattr(cluster_model, 'embed_cluster_method', self.embed_cluster_method)).lower()
        model_min_samples = int(getattr(cluster_model, 'dbscan_min_samples', self.dbscan_min_samples))
        self.dbscan_min_samples = (
            int(self._formal_eval_readout_config.min_samples)
            if self._formal_readout_min_samples_locked
            else model_min_samples
        )
        cluster_eps = getattr(cluster_model, 'dbscan_eps', getattr(self, 'dbscan_eps', 1.5))
        if cluster_eps is not None:
            self.dbscan_eps = self._to_float(cluster_eps)
        cluster_readout_eps = getattr(cluster_model, 'dbscan_readout_eps', self.dbscan_readout_eps)
        self.dbscan_readout_eps = (
            None if cluster_readout_eps is None else self._to_float(cluster_readout_eps)
        )
        if self._formal_readout_eps_locked and self._formal_eval_readout_config.eps is not None:
            self.dbscan_readout_eps = float(self._formal_eval_readout_config.eps)
        self.dbscan_auto_eps = bool(getattr(cluster_model, 'dbscan_auto_eps', self.dbscan_auto_eps))
        self.louvain_distance_factor = float(
            getattr(cluster_model, 'louvain_distance_factor', self.louvain_distance_factor)
        )
        self.louvain_knn_k = int(getattr(cluster_model, 'louvain_knn_k', self.louvain_knn_k))
        self.louvain_mutual_knn = bool(getattr(cluster_model, 'louvain_mutual_knn', self.louvain_mutual_knn))
        self._update_formal_readout_controls(
            method=None if self._formal_readout_method_locked else self.embed_cluster_method,
            eps=None
            if self._formal_readout_eps_locked
            else self.dbscan_readout_eps if self.dbscan_readout_eps is not None else self.dbscan_eps,
            min_samples=None if self._formal_readout_min_samples_locked else self.dbscan_min_samples,
        )

    # --- UI 构建代码 ---

    def _panel_add_axes(self, height: float) -> Axes:
        px, py, pw, ph = self._panel_rect
        margin = self._panel_margin_x * pw
        ax = self.fig.add_axes((px + margin, 0.0, pw - 2 * margin, height))
        base = (0.0, self._panel_cursor_y, 1.0, height)
        self._panel_items.append((ax, base))
        self._panel_all_axes.add(ax)
        self._panel_cursor_y += (height + 0.008)
        return ax

    def _panel_add_row_axes(self, count: int, height: float = 0.036, gap: float = 0.04) -> list[Axes]:
        if count <= 0:
            return []
        row_y = self._panel_cursor_y
        total_gap = gap * max(0, count - 1)
        width = max(0.0, (1.0 - total_gap) / count)
        axes: list[Axes] = []
        for idx in range(count):
            ax = self.fig.add_axes((0.0, 0.0, 0.0, height))
            base = (idx * (width + gap), row_y, width, height)
            self._panel_items.append((ax, base))
            self._panel_all_axes.add(ax)
            axes.append(ax)
        self._panel_cursor_y += (height + 0.008)
        return axes

    def _panel_add_label(self, text: str) -> None:
        ax = self._panel_add_axes(0.025)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(self._color_panel_bg or self._theme_palette().get('panel_bg', '#f5f5f5'))
        ax.text(
            0.0,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            va='center',
            ha='left',
            color=self._color_fg or self._theme_palette().get('fg', '#101010'),
        )
        for spine in ax.spines.values(): spine.set_visible(False)
        self._label_axes.append(ax)

    def _prime_button_theme(self, btn: Button | None) -> None:
        if btn is None:
            return
        try:
            btn.ax.set_facecolor(self._color_btn_bg)
            for sp in btn.ax.spines.values():
                sp.set_edgecolor(self._color_btn_edge)
            btn.label.set_color(self._color_fg)
            btn.color = self._color_btn_bg
            btn.hovercolor = self._color_btn_hover
        except Exception:
            pass

    def _panel_reflow(self) -> None:
        px, py, pw, ph = self._panel_rect
        margin = self._panel_margin_x * pw
        content_h = self._panel_cursor_y
        max_scroll = max(0.0, content_h - ph)
        self._panel_scroll = min(max(self._panel_scroll, 0.0), max_scroll)
        for ax, base in self._panel_items:
            xrel, yb, wrel, h = base
            y_fig = py + ph - (yb - self._panel_scroll) - h
            usable_w = pw - 2 * margin
            x_fig = px + margin + xrel * usable_w
            w_fig = max(0.0, wrel * usable_w)
            ax.set_position((x_fig, y_fig, w_fig, h))
            visible = (y_fig + h) > py and y_fig < (py + ph)
            if visible and getattr(ax, '_panel_manual_visible', True):
                ax.set_visible(True)
            else:
                ax.set_visible(False)
        self.fig.canvas.draw_idle()

    def _inactive_control_color(self) -> str:
        try:
            return self._color_inactive
        except Exception:
            return self._theme_palette().get('inactive', '#a0b4d0')
    
    def _color_slider_handle(self, slider: Slider | None, color: str) -> None:
        """
        Normalize slider knob coloring across Matplotlib versions.
        Older releases expose `handle`, newer ones use `handles` or `_handle`.
        """
        if slider is None: return
        candidates: list[Artist] = []
        for attr in ('handle', '_handle'):
            art = getattr(slider, attr, None)
            if isinstance(art, Artist): candidates.append(art)
        handles = getattr(slider, 'handles', None)
        if handles:
            for art in handles:
                if isinstance(art, Artist): candidates.append(art)
        for art in candidates:
            try:
                art_any = cast(Any, art)
                art_any.set_facecolor(color)
                art_any.set_edgecolor(color)
            except Exception: pass
        for attr in ('poly', 'hline', 'vline'):
            artist = getattr(slider, attr, None)
            if artist is not None:
                try: artist.set_color(color)
                except Exception: pass

    def _set_widget_enabled(self, widget: Any, enabled: bool) -> None:
        if widget is None:
            return
        setter = getattr(widget, 'set_active', None)
        if callable(setter):
            try:
                setter(bool(enabled))
            except Exception:
                pass
        if hasattr(widget, 'eventson'):
            try:
                widget.eventson = bool(enabled)
            except Exception:
                pass
        ax = getattr(widget, 'ax', None)
        alpha = 1.0 if enabled else 0.38
        if ax is not None:
            try:
                ax.patch.set_alpha(alpha)
                for spine in ax.spines.values():
                    spine.set_alpha(alpha)
            except Exception:
                pass
        for attr in ('label', 'valtext'):
            artist = getattr(widget, attr, None)
            if artist is not None:
                try:
                    artist.set_alpha(alpha)
                except Exception:
                    pass

    def _layout_slider_text(self, slider: Slider | None) -> None:
        if slider is None:
            return
        label = getattr(slider, 'label', None)
        if label is not None:
            try:
                label.set_position((0.5, 0.5))
                label.set_ha('center')
                label.set_va('center')
                label.set_transform(slider.ax.transAxes)
                label.set_zorder(10)
                label.set_clip_on(False)
            except Exception:
                pass
        value_text = getattr(slider, 'valtext', None)
        if value_text is not None:
            try:
                value_text.set_position((0.985, 0.5))
                value_text.set_ha('right')
                value_text.set_va('center')
                value_text.set_transform(slider.ax.transAxes)
                value_text.set_zorder(11)
            except Exception:
                pass

    def _apply_algorithm_highlights(self, active_color: str, inactive_color: str) -> None:
        if self._uses_direct_model_labels():
            slider_rest = getattr(self, '_color_slider_rest', inactive_color)
            self._color_slider_handle(getattr(self, 's_eps', None), slider_rest)
            self._color_slider_handle(getattr(self, 's_readout_eps', None), slider_rest)
            self._color_slider_handle(getattr(self, 's_minp', None), slider_rest)
            self._color_slider_handle(getattr(self, 's_louvain_k_dist', None), slider_rest)
            self._color_slider_handle(getattr(self, 's_louvain_knn_k', None), slider_rest)
            self._set_widget_enabled(getattr(self, 's_eps', None), False)
            self._set_widget_enabled(getattr(self, 's_readout_eps', None), False)
            self._set_widget_enabled(getattr(self, 's_minp', None), False)
            self._set_widget_enabled(getattr(self, 'b_auto_eps', None), False)
            self._set_widget_enabled(getattr(self, 's_louvain_k_dist', None), False)
            self._set_widget_enabled(getattr(self, 's_louvain_knn_k', None), False)
            self._set_widget_enabled(getattr(self, 'b_louvain_mutual', None), False)
            self._set_widget_enabled(getattr(self, 'b_embed', None), False)
            self._apply_graph_control_highlights(active_color, inactive_color)
            return

        method = str(getattr(self, 'embed_cluster_method', 'dbscan')).lower()
        slider_rest = getattr(self, '_color_slider_rest', inactive_color)

        s_eps = getattr(self, 's_eps', None)
        s_readout_eps = getattr(self, 's_readout_eps', None)
        s_minp = getattr(self, 's_minp', None)
        s_louvain = getattr(self, 's_louvain_k_dist', None)
        s_louvain_knn = getattr(self, 's_louvain_knn_k', None)

        if method == 'dbscan':
            eps_color = active_color
            minp_color = active_color
            louvain_color = slider_rest
        elif method == 'hdbscan':
            eps_color = slider_rest
            minp_color = active_color
            louvain_color = slider_rest
        elif method == 'louvain':
            eps_color = slider_rest
            minp_color = slider_rest
            louvain_color = active_color
        else:
            eps_color = slider_rest
            minp_color = slider_rest
            louvain_color = slider_rest

        self._color_slider_handle(s_eps, eps_color)
        self._color_slider_handle(s_readout_eps, eps_color)
        self._color_slider_handle(s_minp, minp_color)
        self._color_slider_handle(s_louvain, louvain_color)
        self._color_slider_handle(s_louvain_knn, louvain_color)
        dbscan_active = method == 'dbscan'
        hdbscan_active = method == 'hdbscan'
        louvain_active = method == 'louvain'
        self._set_widget_enabled(s_eps, dbscan_active)
        self._set_widget_enabled(s_readout_eps, dbscan_active)
        self._set_widget_enabled(s_minp, dbscan_active or hdbscan_active)
        self._set_widget_enabled(getattr(self, 'b_auto_eps', None), dbscan_active)
        self._set_widget_enabled(s_louvain, louvain_active)
        self._set_widget_enabled(s_louvain_knn, louvain_active)
        self._set_widget_enabled(getattr(self, 'b_louvain_mutual', None), louvain_active)
        self._apply_graph_control_highlights(active_color, inactive_color)

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes not in self._panel_all_axes and event.inaxes is not self.ax_panel: return
        raw_step = float(getattr(event, 'step', 0) or 0)
        if raw_step == 0.0:
            btn = getattr(event, 'button', '')
            raw_step = 1.0 if btn == 'up' else -1.0
        increment = 0.08 * self._panel_rect[3]
        self._panel_scroll -= raw_step * increment
        content_h = self._panel_cursor_y
        _, py, _, ph = self._panel_rect
        max_scroll = max(0.0, content_h - ph)
        self._panel_scroll = min(max(self._panel_scroll, 0.0), max_scroll)
        self._panel_reflow()

    def _on_key_press(self, event: Any) -> None:
        key = str(getattr(event, 'key', ''))
        if key == " ":
            self._set_running_state(not self.running)
            return
        if key in ("left", "right"):
            slider: Slider | None = None
            if event.inaxes is not None: slider = self._slider_axes_map.get(event.inaxes)
            if slider is None: slider = self._last_active_slider
            if slider is None: return
            step = self._slider_steps.get(slider)
            if step is None or step <= 0:
                range_val = self._to_float(slider.valmax) - self._to_float(slider.valmin)
                fallback = range_val * 0.01
                step = fallback if fallback > 0 else 1e-4
            else: step = self._to_float(step)
            delta = step if key == "right" else -step
            slider.set_val(np.clip(self._to_float(slider.val) + delta, self._to_float(slider.valmin), self._to_float(slider.valmax)))
            self._last_active_slider = slider
            return
        if key in ("pageup", "pagedown", "home", "end"):
            content_h = self._panel_cursor_y
            _, py, _, ph = self._panel_rect
            max_scroll = max(0.0, content_h - ph)
            inc = 0.25 * ph
            if key == "pageup": self._panel_scroll = max(0.0, self._panel_scroll - inc)
            elif key == "pagedown": self._panel_scroll = min(max_scroll, self._panel_scroll + inc)
            elif key == "home": self._panel_scroll = 0.0
            elif key == "end": self._panel_scroll = max_scroll
            self._panel_reflow()

    def _on_step_chunk_change(self, val: float) -> None:
        idx = int(np.clip(round(float(val)), 0, len(self._step_chunk_options) - 1))
        self.step_chunk = int(self._step_chunk_options[idx])
        if getattr(self, 's_step_chunk', None) is not None and getattr(self.s_step_chunk, 'valtext', None) is not None:
            self.s_step_chunk.valtext.set_text(str(self.step_chunk))
        self.cluster_refresh_steps = (
            self._default_cluster_refresh_steps()
            if self._requested_record_interval is None
            else self._requested_record_interval
        )

    def _step_chunk_slider_index(self, chunk: int | float | None = None) -> int:
        value = int(self.step_chunk if chunk is None else max(1, int(chunk)))
        diffs = [abs(value - option) for option in self._step_chunk_options]
        return int(np.argmin(diffs))

    def _init_dataset_list(self) -> None:
        """扫描 datasets/ 目录，构建可用数据集列表，并记录当前索引。"""
        try:
            ds_path = Path(self._dataset_path())
            ds_dir = ds_path.parent if ds_path.parent.exists() else Path("datasets")
            candidates = sorted(p for p in ds_dir.glob("*.dot") if p.is_file())
            compatible = [p for p in candidates if self._is_dataset_switch_compatible(p)]
            if ds_path not in compatible:
                compatible.insert(0, ds_path)
            self._dataset_paths = [str(p) for p in compatible] or [str(ds_path)]
            current_name = ds_path.name
            try:
                self._dataset_index = next(i for i, p in enumerate(self._dataset_paths) if Path(p).name == current_name)
            except StopIteration:
                self._dataset_index = 0
        except Exception:
            self._dataset_paths = [self._dataset_path()]
            self._dataset_index = 0

    def _available_datasets(self) -> list[str]:
        paths = getattr(self, "_dataset_paths", None)
        if not paths:
            return [self._dataset_path()]
        return paths

    def _advance_dataset(self, step: int = 1) -> None:
        """在可用数据集中循环切换，跳过不兼容数据集。"""
        paths = self._available_datasets()
        if not paths or not self._dataset_switch_enabled():
            return
        idx = getattr(self, "_dataset_index", 0) or 0
        direction = 1 if step >= 0 else -1
        for offset in range(1, len(paths) + 1):
            new_idx = (idx + direction * offset) % len(paths)
            target = paths[new_idx]
            if self._switch_dataset_path(target):
                self._dataset_index = new_idx
                self._update_dataset_readout()
                return
        print("[Warn] No compatible dataset found in current folder.")

    def _sync_model_hierarchy_capacity_after_dataset_switch(self) -> None:
        """Keep model hierarchy readout capacity aligned with the switched environment."""
        if not bool(getattr(self.env, 'is_hierarchical', False)):
            return
        depth = max(1, int(self._hierarchy_display_depth()))
        update_params = getattr(self.model, "update_params", None)
        updates: dict[str, Any] = {}

        if hasattr(self.model, "target_levels"):
            updates["target_levels"] = depth
            self._strategy_params["target_levels"] = depth

        if hasattr(self.model, "levels"):
            requested_level_count = max(2, depth + 1)
            updates["level_count"] = requested_level_count
            self._strategy_params["level_count"] = requested_level_count

        if updates and callable(update_params):
            update_params(**updates)

    def _switch_dataset_path(self, new_path: str) -> bool:
        """
        切换 env 的底层图到新的 .dot 文件，但不重置模型权重。
        要求输出维度相同（GraphWalkTest.switch_dataset 内部会检查）。
        """
        env = getattr(self, "env", None)
        if env is None:
            return False
        switch_fn = getattr(env, "switch_dataset", None)
        if switch_fn is None or not callable(switch_fn):
            print("[Warn] Current environment does not support dataset switching.")
            return False
        try:
            switch_fn(new_path)
        except Exception as e:
            print(f"[Warn] Failed to switch dataset to {new_path}: {e}")
            return False

        # 更新当前路径并强制重新计算聚类 / 指标
        self._ds_path = new_path
        self._sync_model_hierarchy_capacity_after_dataset_switch()
        self._eval_label_level = self._infer_eval_label_level()
        self._display_label_level = self._clamp_display_label_level(self._display_label_level)
        self._hierarchy_outputs_cache = None
        self._hierarchy_outputs_step = -10**9
        self._last_labels = None
        self._last_cluster_step = -10**9
        self._close_auxiliary_figure("input")
        self._close_auxiliary_figure("graph")
        if self.b_focus is not None:
            self.b_focus.label.set_text(self._focus_button_label())
        self._update_metrics_and_plot(recompute_labels=True)
        if self._figure_is_open(self.curve_fig):
            self._build_curve_figure()
            self._redraw_live_curve()
        self._set_title()
        self._update_dataset_readout()
        return True

    def _build_panel(self) -> None:
        self._panel_add_label("Dataset")
        self.ax_dataset_text = self._panel_add_axes(0.055)
        self.ax_dataset_text.set_xticks([])
        self.ax_dataset_text.set_yticks([])
        for spine in self.ax_dataset_text.spines.values():
            spine.set_visible(False)
        self.dataset_text_artist = self.ax_dataset_text.text(
            0.0,
            0.5,
            Path(self._dataset_path()).name,
            transform=self.ax_dataset_text.transAxes,
            fontsize=8.5,
            va='center',
            ha='left',
            wrap=True,
        )
        if self._dataset_switch_enabled():
            dataset_axes = self._panel_add_row_axes(2)
            self.ax_dataset = dataset_axes[0]
            self.b_dataset_prev = Button(dataset_axes[0], "Prev")
            self._all_buttons.append(self.b_dataset_prev)
            self.b_dataset_prev.on_clicked(lambda _e: self._advance_dataset(-1))
            self.b_dataset_next = Button(dataset_axes[1], "Next")
            self._all_buttons.append(self.b_dataset_next)
            self.b_dataset_next.on_clicked(lambda _e: self._advance_dataset(1))

        self._panel_add_label("Playback")
        ax_toggle = self._panel_add_axes(0.036)
        reset_step_axes = self._panel_add_row_axes(2)
        ax_reset, ax_step = reset_step_axes
        self.b_toggle = Button(ax_toggle, 'Pause')
        self.b_reset = Button(ax_reset, 'Reset')
        self.b_step = Button(ax_step, 'Step')
        self._all_buttons += [self.b_toggle, self.b_reset, self.b_step]

        export_axes = self._panel_add_row_axes(2)
        ax_export_picture, ax_export_json = export_axes
        self.b_export_picture = Button(ax_export_picture, 'picture')
        self.b_export_json = Button(ax_export_json, 'export json')
        self._all_buttons.extend([self.b_export_picture, self.b_export_json])

        def on_toggle(_e: Any) -> None:
            self._set_running_state(not self.running)
        
        def on_reset(_e: Any) -> None:
            self._handle_reset()

        def on_step(_e: Any) -> None:
            self._simulate_chunk()
            self._update_metrics_and_plot(recompute_labels=bool(self._should_recluster()))
            if self.force_fig is not None:
                self._update_force_monitor()
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        self.b_toggle.on_clicked(on_toggle)
        self.b_reset.on_clicked(on_reset)
        self.b_step.on_clicked(on_step)
        self.b_export_picture.on_clicked(lambda _e: self._export_current_picture())
        self.b_export_json.on_clicked(lambda _e: self._export_layout_snapshot())
        self._on_reset_callback = on_reset

        def _configure_slider(ax: Axes, label: str, vmin: float, vmax: float, vinit: float, step: float) -> Slider:
            edge_color = getattr(self, '_color_btn_bg', self._accent_color)
            handle_style = {'facecolor': self._accent_color, 'edgecolor': edge_color}
            try:
                s = Slider(
                    ax,
                    self._ui_label(label),
                    vmin,
                    vmax,
                    valinit=vinit,
                    valstep=step,
                    handle_style=handle_style,
                    dragging=False,
                )
            except TypeError:
                try:
                    s = Slider(ax, self._ui_label(label), vmin, vmax, valinit=vinit, valstep=step, dragging=False)
                except TypeError:
                    s = Slider(ax, self._ui_label(label), vmin, vmax, valinit=vinit, valstep=step)
            s.label.set_fontsize(8.5)
            s.valtext.set_fontsize(8.0)
            self._layout_slider_text(s)
            self._slider_axes_map[ax] = s
            self._slider_steps[s] = step
            return s

        if not self.readonly:
            ax_chunk = self._panel_add_axes(0.035)
            self.s_step_chunk = _configure_slider(
                ax_chunk,
                'step_chunk',
                0.0,
                float(len(self._step_chunk_options) - 1),
                float(self._step_chunk_slider_index()),
                1.0,
            )
            self.s_step_chunk.valtext.set_text(str(self.step_chunk))
            self.s_step_chunk.on_changed(self._on_step_chunk_change)

        self._panel_add_label("View")
        view_axes = self._panel_add_row_axes(2)
        self.b_view = Button(view_axes[0], self._view_button_label())
        self.b_theme = Button(view_axes[1], self._theme_button_label())
        self._all_buttons.extend([self.b_view, self.b_theme])
        self._prime_button_theme(self.b_view)
        self._prime_button_theme(self.b_theme)
        self.b_view.on_clicked(lambda _e: self._toggle_view_dimension())
        self.b_theme.on_clicked(lambda _e: self._toggle_theme())

        self._panel_add_label("Display")
        display_axes = self._panel_add_row_axes(2)
        self.ax_edges = display_axes[0]
        self.b_edges = Button(self.ax_edges, self._format_toggle_label('edges', self.show_cluster_edges))
        self.b_mesh = Button(display_axes[1], self._format_toggle_label('mesh', self.show_mesh))
        self._all_buttons.extend([self.b_edges, self.b_mesh])
        self._prime_button_theme(self.b_edges)
        self._prime_button_theme(self.b_mesh)

        def _on_edges_toggle(_e: Any) -> None:
            self.show_cluster_edges = not bool(self.show_cluster_edges)
            if self.b_edges is not None:
                self.b_edges.label.set_text(self._format_toggle_label('edges', self.show_cluster_edges))
            self._update_metrics_and_plot(recompute_labels=False)

        self.b_edges.on_clicked(_on_edges_toggle)
        self.b_mesh.on_clicked(lambda _e: self._toggle_mesh())

        self._panel_add_label("Evaluation")
        is_hierarchical = bool(getattr(self.env, 'is_hierarchical', False))
        evaluation_axes = self._panel_add_row_axes(3 if is_hierarchical else 1)
        self.ax_pred = evaluation_axes[0]
        self.b_pred = Button(
            self.ax_pred,
            self._format_toggle_label('predict', self.use_pred_colors),
        )
        self._all_buttons.append(self.b_pred)
        self._prime_button_theme(self.b_pred)

        def _on_pred_toggle(_e: Any) -> None:
            self.use_pred_colors = not bool(self.use_pred_colors)
            if self.b_pred is not None:
                self.b_pred.label.set_text(self._format_toggle_label('predict', self.use_pred_colors))
            self._update_metrics_and_plot(recompute_labels=True)
        self.b_pred.on_clicked(_on_pred_toggle)

        if is_hierarchical:
            self.ax_focus = evaluation_axes[1]
            self.ax_score = evaluation_axes[2]
            self.b_focus = Button(self.ax_focus, self._focus_button_label())
            self.b_score = Button(self.ax_score, self._score_button_label())
            self._all_buttons.extend([self.b_focus, self.b_score])
            self._prime_button_theme(self.b_focus)
            self._prime_button_theme(self.b_score)

            def _on_focus_toggle(_e: Any) -> None:
                self._cycle_hierarchy_focus()

            def _on_score_toggle(_e: Any) -> None:
                self._score_display_mode = 'levels' if self._score_display_mode == 'summary' else 'summary'
                if self.b_score is not None:
                    self.b_score.label.set_text(self._score_button_label())
                self._update_metrics_and_plot(recompute_labels=False)

            self.b_focus.on_clicked(_on_focus_toggle)
            self.b_score.on_clicked(_on_score_toggle)

        self._panel_add_label("Actions")
        auxiliary_axes = self._panel_add_row_axes(3)
        self.b_show_input = Button(auxiliary_axes[0], self._format_toggle_label("input", False))
        self.b_show_graph = Button(auxiliary_axes[1], self._format_toggle_label("graph", False))
        self.b_live_curve = Button(auxiliary_axes[2], self._format_toggle_label("curve", False))
        self._all_buttons.extend([self.b_show_input, self.b_show_graph, self.b_live_curve])
        for button in [self.b_pred, self.b_focus, self.b_score, self.b_show_input, self.b_show_graph, self.b_live_curve]:
            if button is not None:
                button.label.set_fontsize(7.5)
        self._prime_button_theme(self.b_show_input)
        self._prime_button_theme(self.b_show_graph)
        self._prime_button_theme(self.b_live_curve)
        self.b_show_input.on_clicked(lambda _e: self._show_input_sequence())
        self.b_show_graph.on_clicked(lambda _e: self._show_graph_preview())
        self.b_live_curve.on_clicked(lambda _e: self._toggle_curve_window())

        if self.readonly:
            ax_gif = self._panel_add_axes(0.04)
            self.b_export_gif = Button(ax_gif, "save gif")
            self._all_buttons.append(self.b_export_gif)
            self.b_export_gif.on_clicked(lambda _e: self._export_current_gif())

            self._panel_reflow()
            return

        def add_slider(label: str, vmin: float, vmax: float, vinit: float, step: float) -> Slider:
            ax = self._panel_add_axes(0.035)
            return _configure_slider(ax, label, vmin, vmax, vinit, step)

        # Space Sliders
        self._panel_add_label("Space")
        graph_model = self._graph_control_target_model()
        self.s_base_wall = add_slider('base_wall', 10.0, 300.0, getattr(graph_model, 'base_wall', getattr(self.model, 'base_wall', 100.0)), 5.0)
        self.s_density_ref = add_slider('density_ref', 10.0, 500.0, getattr(graph_model, 'density_ref', getattr(self.model, 'density_ref', 75.0)), 5.0)
        vmax_space = max(500.0, float(getattr(graph_model, 'space_size', getattr(self.model, 'space_size', 100.0))) * 2.0 or 500.0)
        space_init = self.init_space_size if getattr(self, '_space_override_enabled', False) else 0.0
        self.s_space_size = add_slider('space_size', 0.0, vmax_space, space_init, 1.0)
        self._update_space_slider_display()

        # Clustering & Diag
        self._panel_add_label("Cluster")

        vinit_eps = self._to_float(self.dbscan_eps)
        vmax_eps = max(20.0, vinit_eps * 1.5)
        vinit_minp = float(self.dbscan_min_samples)
        vinit_distance = float(self.louvain_distance_factor)

        cluster_button_axes = self._panel_add_row_axes(2)
        self.ax_auto = cluster_button_axes[0]
        self.ax_embed = cluster_button_axes[1]
        self.b_auto_eps = Button(self.ax_auto, self._format_toggle_label('auto_eps', self.dbscan_auto_eps))
        self._all_buttons.append(self.b_auto_eps)
        def _on_auto_toggle(_e: Any) -> None:
            if str(getattr(self, 'embed_cluster_method', 'dbscan')).lower() != 'dbscan':
                return
            self.dbscan_auto_eps = not bool(self.dbscan_auto_eps)
            if self.b_auto_eps is not None:
                self.b_auto_eps.label.set_text(self._format_toggle_label('auto_eps', self.dbscan_auto_eps))
            try:
                self.model.update_params(**self._live_cluster_update_kwargs(dbscan_auto_eps=self.dbscan_auto_eps))
            except Exception:
                pass
            self._invalidate_clustering_cache()
            self._update_metrics_and_plot(recompute_labels=True)
        self.b_auto_eps.on_clicked(_on_auto_toggle)

        self.b_embed = Button(self.ax_embed, f"cluster:{str(getattr(self, 'embed_cluster_method', 'dbscan')).lower()}")
        self._all_buttons.append(self.b_embed)
        def _on_embed_toggle(_e: Any) -> None:
            if self._uses_direct_model_labels():
                return
            current = str(getattr(self, 'embed_cluster_method', 'dbscan')).lower()
            order = ['dbscan', 'hdbscan', 'louvain']
            try:
                next_idx = (order.index(current) + 1) % len(order)
            except ValueError:
                next_idx = 0
            self.embed_cluster_method = order[next_idx]
            if self.b_embed is not None:
                self.b_embed.label.set_text(f"cluster:{self.embed_cluster_method}")
            self._formal_readout_method_locked = False
            self._update_formal_readout_controls(method=self.embed_cluster_method)
            try:
                self.model.update_params(**self._live_cluster_update_kwargs(embed_cluster_method=self.embed_cluster_method))
            except Exception:
                pass
            self._invalidate_clustering_cache()
            self._update_metrics_and_plot(recompute_labels=True)
            self._apply_algorithm_highlights(self._accent_color, self._inactive_control_color())
        self.b_embed.on_clicked(_on_embed_toggle)

        self._panel_add_label("Thresholds")
        self.s_eps = add_slider('eps', 0.1, vmax_eps, vinit_eps, 0.1)
        self.ax_eps = self.s_eps.ax
        vinit_readout_eps = float(vinit_eps if self.dbscan_readout_eps is None else self.dbscan_readout_eps)
        self.s_readout_eps = add_slider('readout_eps', 0.1, vmax_eps * 1.5, vinit_readout_eps, 0.1)
        self.ax_readout_eps = self.s_readout_eps.ax
        max_minp = max(12.0, min(80.0, float(max(2, self.output_size)) / 4.0))
        self.s_minp = add_slider('minPts', 1.0, max_minp, vinit_minp, 1.0)
        self.ax_minp = self.s_minp.ax
        self.s_louvain_k_dist = add_slider('louvain_k_dist', 0.05, 2.0, vinit_distance, 0.05)
        self.ax_louvain_k_dist = self.s_louvain_k_dist.ax
        self.s_louvain_knn_k = add_slider('louvain_knn_k', 0.0, 32.0, float(self.louvain_knn_k), 1.0)
        self.ax_louvain_knn_k = self.s_louvain_knn_k.ax

        louvain_button_axes = self._panel_add_row_axes(1)
        self.ax_louvain_mutual = louvain_button_axes[0]
        self.b_louvain_mutual = Button(
            self.ax_louvain_mutual,
            self._format_toggle_label('mutual_knn', self.louvain_mutual_knn),
        )
        self._all_buttons.append(self.b_louvain_mutual)

        def _on_louvain_mutual_toggle(_e: Any) -> None:
            if str(getattr(self, 'embed_cluster_method', 'dbscan')).lower() != 'louvain':
                return
            self.louvain_mutual_knn = not bool(self.louvain_mutual_knn)
            if self.b_louvain_mutual is not None:
                self.b_louvain_mutual.label.set_text(
                    self._format_toggle_label('mutual_knn', self.louvain_mutual_knn)
                )
            try:
                self.model.update_params(
                    **self._live_cluster_update_kwargs(louvain_mutual_knn=self.louvain_mutual_knn)
                )
            except Exception:
                pass
            self._invalidate_clustering_cache()
            self._update_metrics_and_plot(recompute_labels=True)

        self.b_louvain_mutual.on_clicked(_on_louvain_mutual_toggle)

        graph_model = self._graph_control_target_model()
        core_profile = self._is_minimal_rg_profile()

        self._panel_add_label("Dynamics")
        self.s_adaptation_rate = add_slider('adaptation_rate', 0.0, 1, self.default_step_scale, 0.01)
        self.s_memory_steps = add_slider(
            'memory_steps',
            1.0,
            5000.0,
            float(getattr(graph_model, 'stats_memory_steps', getattr(self.model, 'stats_memory_steps', 1000))),
            1.0,
        )
        if not core_profile:
            self.s_stable_token_evidence_memory_scale = add_slider(
                'entity_mem_scale',
                1.0,
                100.0,
                float(getattr(self.model, 'stable_token_evidence_memory_scale', 10.0)),
                1.0,
            )
            self.s_stable_token_evidence_min_members = add_slider(
                'entity_members',
                1.0,
                8.0,
                float(getattr(self.model, 'stable_token_evidence_min_members', 2)),
                1.0,
            )
            self.s_stable_token_evidence_min_stable_refreshes = add_slider(
                'entity_age',
                1.0,
                50.0,
                float(getattr(self.model, 'stable_token_evidence_min_stable_refreshes', 5)),
                1.0,
            )
            self.s_stable_token_evidence_semantic_z_score = add_slider(
                'entity_sem_z',
                0.0,
                4.0,
                float(getattr(self.model, 'stable_token_evidence_semantic_z_score', 1.645)),
                0.01,
            )
            self.s_stable_token_evidence_min_self_mass = add_slider(
                'entity_self_mass',
                0.0,
                50.0,
                float(getattr(self.model, 'stable_token_evidence_min_self_mass', 5.0)),
                0.5,
            )
            self.s_stable_token_evidence_self_cross_ratio = add_slider(
                'entity_self_ratio',
                1.0,
                10.0,
                float(getattr(self.model, 'stable_token_evidence_self_cross_ratio', 1.25)),
                0.05,
            )

        self._panel_add_label("Graph")
        k_attr_init = self._k_attr_slider_value()
        self.s_k_attr = add_slider(
            'k_attr',
            0.0,
            self._k_attr_slider_limit(),
            min(k_attr_init, self._k_attr_slider_limit()),
            0.01,
        )
        self.s_knn = add_slider('knn', 1.0, 12.0, float(getattr(graph_model, 'knn', 12)), 1.0)
        self.s_graphTheta = add_slider('graph_theta', 0.0, 0.5, float(getattr(graph_model, 'graph_theta', 0.1)), 0.01)
        self.s_graphR0 = add_slider('graph_r0', 0.1, 5.0, self._graph_r0_slider_value(), 0.01)
        self.s_graphPush = add_slider('T_push', 1.0, 100.0, float(getattr(graph_model, 'T_push', 10)), 1.0)

        self._panel_add_label("Forces")
        self.s_k_repulse = add_slider(
            'k_repulse',
            0.0,
            10.0,
            float(getattr(graph_model, 'k_repulse', getattr(self.model, 'k_repulse', 0.0))),
            0.01,
        )
        self.s_primitive_active_attraction_strength = add_slider(
            'raw_attr',
            0.0,
            1.0,
            float(getattr(graph_model, 'primitive_active_attraction_strength', getattr(self.model, 'primitive_active_attraction_strength', 0.0))),
            0.01,
        )
        self.s_primitive_inactive_repulsion_strength = add_slider(
            'raw_repulse',
            0.0,
            1.0,
            float(getattr(graph_model, 'primitive_inactive_repulsion_strength', getattr(self.model, 'primitive_inactive_repulsion_strength', 0.0))),
            0.01,
        )
        if not core_profile:
            self.s_k_macro_repulse = add_slider(
                'k_macro_repulse',
                0.0,
                5.0,
                float(getattr(graph_model, 'k_macro_repulse', getattr(self.model, 'k_macro_repulse', 0.0))),
                0.05,
            )
            self.s_macro_cross_attr_scale = add_slider(
                'cross_attr',
                0.0,
                1.0,
                float(getattr(graph_model, 'macro_cross_attr_scale', getattr(self.model, 'macro_cross_attr_scale', 0.0))),
                0.01,
            )
            self.s_macro_centroid_distance_ratio = add_slider(
                'macro_dist',
                1.0,
                16.0,
                float(getattr(graph_model, 'macro_centroid_distance_ratio', getattr(self.model, 'macro_centroid_distance_ratio', 4.0))),
                0.1,
            )
            self.s_macro_centroid_max_step_ratio = add_slider(
                'macro_step',
                0.0,
                5.0,
                float(getattr(graph_model, 'macro_centroid_max_step_ratio', getattr(self.model, 'macro_centroid_max_step_ratio', 1.0))),
                0.05,
            )
            self.s_k_entity_boundary_repulse = add_slider(
                'entity_push',
                0.0,
                3.0,
                float(getattr(graph_model, 'k_entity_boundary_repulse', getattr(self.model, 'k_entity_boundary_repulse', 0.0))),
                0.02,
            )
            self.s_entity_boundary_distance_ratio = add_slider(
                'entity_dist',
                1.0,
                8.0,
                float(getattr(graph_model, 'entity_boundary_distance_ratio', getattr(self.model, 'entity_boundary_distance_ratio', 3.16))),
                0.05,
            )
            self.s_entity_boundary_max_step_ratio = add_slider(
                'entity_step',
                0.0,
                3.0,
                float(getattr(graph_model, 'entity_boundary_max_step_ratio', getattr(self.model, 'entity_boundary_max_step_ratio', 0.5))),
                0.02,
            )
            self.s_k_entity_sibling_attr = add_slider(
                'sibling_attr',
                0.0,
                3.0,
                float(getattr(graph_model, 'k_entity_sibling_attr', getattr(self.model, 'k_entity_sibling_attr', 0.0))),
                0.02,
            )
            self.s_entity_sibling_target_ratio = add_slider(
                'sibling_target',
                1.0,
                8.0,
                float(getattr(graph_model, 'entity_sibling_target_ratio', getattr(self.model, 'entity_sibling_target_ratio', 3.16))),
                0.05,
            )
            self.s_entity_sibling_max_step_ratio = add_slider(
                'sibling_step',
                0.0,
                3.0,
                float(getattr(graph_model, 'entity_sibling_max_step_ratio', getattr(self.model, 'entity_sibling_max_step_ratio', 0.5))),
                0.02,
            )
            self.s_entity_sibling_boundary_release_ratio = add_slider(
                'sibling_release',
                1.0,
                2.0,
                float(getattr(graph_model, 'entity_sibling_boundary_release_ratio', getattr(self.model, 'entity_sibling_boundary_release_ratio', 1.25))),
                0.01,
            )
            self.s_entity_sibling_boundary_power = add_slider(
                'sibling_power',
                0.0,
                6.0,
                float(getattr(graph_model, 'entity_sibling_boundary_power', getattr(self.model, 'entity_sibling_boundary_power', 2.0))),
                0.1,
            )
            self.s_line_obstacle_strength = add_slider(
                'line_repulse',
                0.0,
                5.0,
                float(getattr(graph_model, 'line_obstacle_strength', getattr(self.model, 'line_obstacle_strength', 0.0))),
                0.05,
            )
            self.s_line_obstacle_radius_rel = add_slider(
                'line_radius',
                0.1,
                3.0,
                float(getattr(graph_model, 'line_obstacle_radius_rel', getattr(self.model, 'line_obstacle_radius_rel', 1.0))),
                0.05,
            )

        def on_any_slider(_val: float | None = None) -> None:
            if getattr(self, '_suppress_slider_events', False): return

            # 读取滑块值
            val_base_wall = self._to_float(self._require_slider(self.s_base_wall, 'base_wall').val)
            val_density = self._to_float(self._require_slider(self.s_density_ref, 'density_ref').val)
            val_space = self._to_float(self._require_slider(self.s_space_size, 'space_size').val)
            
            # [Fix 1: 更新 LiveTuner 的本地状态]
            self.dbscan_eps = self._to_float(self._require_slider(self.s_eps, 'eps').val)
            self.dbscan_readout_eps = self._to_float(self._require_slider(self.s_readout_eps, 'readout_eps').val)
            self.dbscan_min_samples = int(round(self._to_float(self._require_slider(self.s_minp, 'minPts').val)))
            self._formal_readout_eps_locked = False
            self._formal_readout_min_samples_locked = False
            self._update_formal_readout_controls(
                eps=self.dbscan_readout_eps,
                min_samples=self.dbscan_min_samples,
            )
            self.louvain_distance_factor = self._to_float(self._require_slider(self.s_louvain_k_dist, 'louvain_k_dist').val)
            self.louvain_knn_k = max(
                0,
                int(round(self._to_float(self._require_slider(self.s_louvain_knn_k, 'louvain_knn_k').val))),
            )

            s_space_size = self._require_slider(self.s_space_size, 'space_size')
            manual_space_enabled = getattr(self, '_space_override_enabled', False)
            if getattr(self, '_last_active_slider', None) is s_space_size:
                manual_space_enabled = val_space > 0.0
                self._space_override_enabled = manual_space_enabled
                if not manual_space_enabled:
                    prev_state = self._suppress_slider_events
                    try:
                        self._suppress_slider_events = True
                        s_space_size.set_val(0.0)
                    finally:
                        self._suppress_slider_events = prev_state

            # 构建 update_params 参数字典
            k_attr_value = min(
                self._to_float(self._require_slider(self.s_k_attr, 'k_attr').val),
                self._k_attr_slider_limit(),
            )
            update_kwargs = dict(
                base_wall=val_base_wall,
                density_ref=val_density,
                adaptation_rate=self._to_float(self._require_slider(self.s_adaptation_rate, 'adaptation_rate').val),
                stats_memory_steps=int(round(self._to_float(self._require_slider(self.s_memory_steps, 'memory_steps').val))),
                k_repulse=self._to_float(self._require_slider(self.s_k_repulse, 'k_repulse').val),
                primitive_active_attraction_strength=self._to_float(self._require_slider(self.s_primitive_active_attraction_strength, 'primitive_active_attraction_strength').val),
                primitive_inactive_repulsion_strength=self._to_float(self._require_slider(self.s_primitive_inactive_repulsion_strength, 'primitive_inactive_repulsion_strength').val),
            )
            optional_slider_updates = {
                'stable_token_evidence_memory_scale': (self.s_stable_token_evidence_memory_scale, 'entity_mem_scale', float),
                'stable_token_evidence_min_members': (self.s_stable_token_evidence_min_members, 'entity_members', int),
                'stable_token_evidence_min_stable_refreshes': (self.s_stable_token_evidence_min_stable_refreshes, 'entity_age', int),
                'stable_token_evidence_semantic_z_score': (self.s_stable_token_evidence_semantic_z_score, 'entity_sem_z', float),
                'stable_token_evidence_min_self_mass': (self.s_stable_token_evidence_min_self_mass, 'entity_self_mass', float),
                'stable_token_evidence_self_cross_ratio': (self.s_stable_token_evidence_self_cross_ratio, 'entity_self_ratio', float),
                'k_macro_repulse': (self.s_k_macro_repulse, 'k_macro_repulse', float),
                'macro_cross_attr_scale': (self.s_macro_cross_attr_scale, 'macro_cross_attr_scale', float),
                'macro_centroid_distance_ratio': (self.s_macro_centroid_distance_ratio, 'macro_centroid_distance_ratio', float),
                'macro_centroid_max_step_ratio': (self.s_macro_centroid_max_step_ratio, 'macro_centroid_max_step_ratio', float),
                'k_entity_boundary_repulse': (self.s_k_entity_boundary_repulse, 'k_entity_boundary_repulse', float),
                'entity_boundary_distance_ratio': (self.s_entity_boundary_distance_ratio, 'entity_boundary_distance_ratio', float),
                'entity_boundary_max_step_ratio': (self.s_entity_boundary_max_step_ratio, 'entity_boundary_max_step_ratio', float),
                'k_entity_sibling_attr': (self.s_k_entity_sibling_attr, 'k_entity_sibling_attr', float),
                'entity_sibling_target_ratio': (self.s_entity_sibling_target_ratio, 'entity_sibling_target_ratio', float),
                'entity_sibling_max_step_ratio': (self.s_entity_sibling_max_step_ratio, 'entity_sibling_max_step_ratio', float),
                'entity_sibling_boundary_release_ratio': (self.s_entity_sibling_boundary_release_ratio, 'entity_sibling_boundary_release_ratio', float),
                'entity_sibling_boundary_power': (self.s_entity_sibling_boundary_power, 'entity_sibling_boundary_power', float),
                'line_obstacle_strength': (self.s_line_obstacle_strength, 'line_obstacle_strength', float),
                'line_obstacle_radius_rel': (self.s_line_obstacle_radius_rel, 'line_obstacle_radius_rel', float),
            }
            for param_name, (slider, slider_name, caster) in optional_slider_updates.items():
                if slider is None:
                    continue
                raw_value = self._to_float(self._require_slider(slider, slider_name).val)
                update_kwargs[param_name] = int(round(raw_value)) if caster is int else raw_value
            graph_support = self._graph_control_support()
            if graph_support.get('k_attr', False):
                update_kwargs['k_attr'] = k_attr_value
            if graph_support.get('knn', False):
                update_kwargs['knn'] = max(1, int(round(self._to_float(self._require_slider(self.s_knn, 'knn').val))))
            if graph_support.get('graph_theta', False):
                update_kwargs['graph_theta'] = self._to_float(self._require_slider(self.s_graphTheta, 'graph_theta').val)
            if graph_support.get('T_push', False):
                update_kwargs['T_push'] = int(max(1.0, self._to_float(self._require_slider(self.s_graphPush, 'T_push').val)))
            update_kwargs.update(
                self._live_cluster_update_kwargs(
                    dbscan_eps=self.dbscan_eps,
                    dbscan_readout_eps=self.dbscan_readout_eps,
                    dbscan_min_samples=self.dbscan_min_samples,
                    louvain_distance_factor=self.louvain_distance_factor,
                    louvain_knn_k=self.louvain_knn_k,
                    louvain_mutual_knn=self.louvain_mutual_knn,
                    dbscan_auto_eps=self.dbscan_auto_eps,
                )
            )
            if graph_support.get('graph_r0', False):
                update_kwargs[self._graph_r0_slider_field()] = self._to_float(
                    self._require_slider(self.s_graphR0, 'graph_r0').val
                )
            
            if manual_space_enabled:
                update_kwargs['space_size'] = val_space
            
            self.model.update_params(**update_kwargs)
            self._invalidate_clustering_cache()
            self._update_k_attr_slider_range(k_attr_value)
            self._update_space_slider_display()
            self._update_metrics_and_plot(recompute_labels=True)
            self._update_force_monitor()

        def _bind_slider(slider: Slider) -> None:
            def _handler(val: float, _slider: Slider = slider) -> None:
                self._last_active_slider = _slider
                on_any_slider(val)
            slider.on_changed(_handler)

        bound_sliders = (
            self.s_base_wall, self.s_density_ref, self.s_space_size,
            self.s_adaptation_rate, self.s_memory_steps, self.s_k_attr,
            self.s_stable_token_evidence_memory_scale, self.s_stable_token_evidence_min_members,
            self.s_stable_token_evidence_min_stable_refreshes,
            self.s_stable_token_evidence_semantic_z_score,
            self.s_stable_token_evidence_min_self_mass,
            self.s_stable_token_evidence_self_cross_ratio,
            self.s_k_repulse, self.s_primitive_active_attraction_strength,
            self.s_primitive_inactive_repulsion_strength, self.s_k_macro_repulse, self.s_macro_cross_attr_scale,
            self.s_macro_centroid_distance_ratio, self.s_macro_centroid_max_step_ratio,
            self.s_k_entity_boundary_repulse, self.s_entity_boundary_distance_ratio,
            self.s_entity_boundary_max_step_ratio,
            self.s_k_entity_sibling_attr, self.s_entity_sibling_target_ratio,
            self.s_entity_sibling_max_step_ratio,
            self.s_entity_sibling_boundary_release_ratio, self.s_entity_sibling_boundary_power,
            self.s_line_obstacle_strength, self.s_line_obstacle_radius_rel,
            self.s_knn, self.s_graphTheta, self.s_graphR0,
            self.s_graphPush, self.s_eps, self.s_readout_eps, self.s_minp, self.s_louvain_k_dist,
            self.s_louvain_knn_k,
        )
        for s in bound_sliders:
            if s is not None:
                _bind_slider(s)

        self._slider_apply = on_any_slider
        self._apply_discrete_controls()
        self._panel_reflow()

    def _update_space_slider_display(self, target_space: float | None = None) -> None:
        slider = getattr(self, 's_space_size', None)
        if slider is None: return
        if target_space is None:
            target_space = getattr(self.model, 'space_size', None)
        if target_space is None: return
        
        target_val = float(target_space)
        if target_val > slider.valmax:
            slider.valmax = target_val * 1.2
            slider.ax.set_xlim(slider.valmin, slider.valmax)
        
        override_enabled = getattr(self, '_space_override_enabled', False)
        if override_enabled:
            current = self._to_float(slider.val)
            if abs(current - target_val) > 1e-6:
                prev_state = self._suppress_slider_events
                try:
                    self._suppress_slider_events = True
                    slider.set_val(target_val)
                finally: self._suppress_slider_events = prev_state
            text = f"{current:.1f}"
        else:
            if abs(self._to_float(slider.val)) > 1e-9:
                prev_state = self._suppress_slider_events
                try:
                    self._suppress_slider_events = True
                    slider.set_val(0.0)
                finally: self._suppress_slider_events = prev_state
            text = f"auto({target_val:.1f})"
        if getattr(slider, 'valtext', None) is not None:
            slider.valtext.set_text(text)

    def _format_toggle_label(self, base: str, value: bool) -> str:
        is_on = bool(value)
        if base == "predict":
            return "colors:pred" if is_on else "colors:true"
        if base in ("edges", "cluster_edges"):
            return f"edges:{'on' if is_on else 'off'}"
        if base == "auto_eps":
            return "eps:auto" if is_on else "eps:manual"
        if base == "mutual_knn":
            return f"mutual:{'on' if is_on else 'off'}"
        return f"{base}:{'on' if is_on else 'off'}"

    def _apply_discrete_controls(self) -> None:
        accent = getattr(self, '_accent_color', '#3b82f6')
        inactive = self._inactive_control_color() if hasattr(self, '_inactive_control_color') else '#9ca3af'
        self._apply_algorithm_highlights(accent, inactive)

    @staticmethod
    def _distance_summary(samples: Any) -> dict[str, float | int]:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            return {
                'count': int(arr.shape[0]) if arr.ndim == 2 else 0,
                'pair_count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'nearest_mean': 0.0,
                'nearest_median': 0.0,
            }
        distances: list[float] = []
        nearest: list[float] = []
        for i in range(arr.shape[0]):
            row_distances: list[float] = []
            for j in range(i + 1, arr.shape[0]):
                dist = float(np.linalg.norm(arr[i] - arr[j]))
                distances.append(dist)
                row_distances.append(dist)
            for j in range(i):
                row_distances.append(float(np.linalg.norm(arr[i] - arr[j])))
            if row_distances:
                nearest.append(min(row_distances))
        if not distances:
            return {
                'count': int(arr.shape[0]),
                'pair_count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'nearest_mean': 0.0,
                'nearest_median': 0.0,
            }
        dist_arr = np.asarray(distances, dtype=float)
        return {
            'count': int(arr.shape[0]),
            'pair_count': int(dist_arr.size),
            'mean': float(np.mean(dist_arr)),
            'median': float(np.median(dist_arr)),
            'min': float(np.min(dist_arr)),
            'max': float(np.max(dist_arr)),
            'nearest_mean': float(np.mean(nearest)) if nearest else 0.0,
            'nearest_median': float(np.median(nearest)) if nearest else 0.0,
        }

    @staticmethod
    def _cluster_distance_payload(samples: Any, labels: Any) -> dict[str, Any]:
        pos = np.asarray(samples, dtype=float)
        lab = np.asarray(labels, dtype=int).reshape(-1)
        if pos.ndim != 2 or lab.ndim != 1 or pos.shape[0] != lab.shape[0]:
            return {'cluster_count': 0, 'centroids': [], 'pairwise_distances': []}
        valid_labels = sorted({int(label) for label in lab.tolist() if int(label) >= 0})
        if not valid_labels:
            return {'cluster_count': 0, 'centroids': [], 'pairwise_distances': []}

        centroids: list[dict[str, Any]] = []
        centroid_rows: list[np.ndarray[Any, Any]] = []
        for label in valid_labels:
            mask = lab == label
            if not np.any(mask):
                continue
            centroid = np.mean(pos[mask], axis=0)
            centroid_rows.append(np.asarray(centroid, dtype=float))
            centroids.append(
                {
                    'label': int(label),
                    'size': int(np.sum(mask)),
                    'centroid': np.asarray(centroid, dtype=float).tolist(),
                }
            )

        pairwise_distances: list[dict[str, Any]] = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = float(np.linalg.norm(centroid_rows[i] - centroid_rows[j]))
                pairwise_distances.append(
                    {
                        'src_label': int(centroids[i]['label']),
                        'dst_label': int(centroids[j]['label']),
                        'distance': dist,
                    }
                )

        return {
            'cluster_count': int(len(centroids)),
            'centroids': centroids,
            'pairwise_distances': pairwise_distances,
        }

    def _level_snapshot(
        self,
        level_name: str,
        level_model: Any,
        labels: Any | None = None,
        *,
        level_index: int | None = None,
    ) -> dict[str, Any]:
        positions = np.asarray(getattr(level_model, 'syncmap', np.empty((0, max(1, self.dimension)))), dtype=float)
        label_array: NDArray[np.integer[Any]] | None = None
        if labels is not None:
            label_array = np.asarray(labels, dtype=int)
        else:
            organizer = getattr(level_model, 'organize', None)
            if callable(organizer):
                try:
                    label_array = np.asarray(organizer(persist=True), dtype=int)
                except Exception:
                    label_array = None

        payload: dict[str, Any] = {
            'name': level_name,
            'node_count': int(positions.shape[0]) if positions.ndim == 2 else 0,
            'dimensions': int(positions.shape[1]) if positions.ndim == 2 and positions.size > 0 else 0,
            'positions': positions.tolist() if positions.ndim == 2 else [],
            'distance_summary': self._distance_summary(positions),
        }
        if level_index is not None:
            if hasattr(self.model, '_level_equilibrium_distance'):
                try:
                    payload['target_distance'] = float(self.model._level_equilibrium_distance(level_index))
                except Exception:
                    payload['target_distance'] = float(
                        getattr(level_model, 'graph_r0', getattr(level_model, 'graph_r0_fixed', 0.0))
                    )
            else:
                payload['target_distance'] = float(
                    getattr(level_model, 'graph_r0', getattr(level_model, 'graph_r0_fixed', 0.0))
                )

        readout_config: dict[str, Any] = {
            'embed_cluster_method': str(getattr(level_model, 'embed_cluster_method', 'unknown')),
            'dbscan_eps': float(getattr(level_model, 'dbscan_eps', 0.0)),
            'dbscan_min_samples': int(getattr(level_model, 'dbscan_min_samples', 0)),
            'louvain_distance_factor': float(getattr(level_model, 'louvain_distance_factor', 0.0)),
            'graph_r0': float(getattr(level_model, 'graph_r0', getattr(level_model, 'graph_r0_fixed', 0.0))),
        }
        if positions.ndim == 2 and positions.shape[0] > 1 and readout_config['embed_cluster_method'] == 'louvain':
            pair_mean = float(payload['distance_summary'].get('mean', 0.0))
            readout_config['effective_distance_threshold'] = (
                pair_mean * float(readout_config['louvain_distance_factor'])
            )
        payload['readout_config'] = readout_config
        if label_array is not None and label_array.ndim == 1 and positions.ndim == 2 and label_array.shape[0] == positions.shape[0]:
            payload['labels'] = label_array.tolist()
            payload['cluster_distances'] = self._cluster_distance_payload(positions, label_array)
        return payload

    def _snapshot_export_path(self) -> Path:
        return result_path(self._unified_result_filename('json', artifact='snapshot'), kind='snapshots')

    def _picture_export_path(self) -> Path:
        return result_path(self._unified_result_filename('png', artifact='picture'), kind='picture')

    def _unified_result_filename(self, extension: str, *, artifact: str) -> str:
        impl_name = str(getattr(self, '_impl_name', self._strategy_name) or self._strategy_name or 'rg')
        target = artifact_result_path(
            impl_name,
            self._dataset_path(),
            artifact,
            extension,
            step_count=int(getattr(self, 'total_steps', 0)),
            kind=None,
        )
        return target.name

    def _export_current_picture(self) -> None:
        target = self._picture_export_path()
        try:
            self.fig.savefig(target, dpi=200, bbox_inches='tight', facecolor=self.fig.get_facecolor())
        except Exception as exc:
            print(f'[Warn] Picture export failed: {exc}')
            return
        print(f'Saved picture to {target}')

    def _export_layout_snapshot(self) -> None:
        try:
            hierarchy_outputs = self._hierarchy_outputs(force=True)
        except Exception:
            hierarchy_outputs = {}

        true_labels = self._true_labels_for_display()
        pred_labels = self._predicted_labels_for_display(force=True)
        level0_model = self._base_dynamics_model()
        graph_model = self._graph_control_target_model()
        cluster_model = self._live_cluster_target_model()
        level0_positions = np.asarray(
            getattr(level0_model, 'syncmap', np.empty((0, max(1, self.dimension)))),
            dtype=float,
        )
        projected_positions = self._project_positions(level0_positions, self.dimension) if level0_positions.ndim == 2 else np.empty((0, self.dimension), dtype=float)

        true_label_levels: dict[str, list[int]] = {}
        label_levels = getattr(self.env, 'label_levels', None)
        if hasattr(label_levels, 'shape'):
            arr = np.asarray(label_levels, dtype=int)
            if arr.ndim == 2:
                for idx in range(arr.shape[1]):
                    true_label_levels[f'level_{idx}'] = arr[:, idx].tolist()

        node_records: list[dict[str, Any]] = []
        pred_arr = None if pred_labels is None else np.asarray(pred_labels, dtype=int).reshape(-1)
        for idx in range(level0_positions.shape[0] if level0_positions.ndim == 2 else 0):
            record: dict[str, Any] = {
                'node_index': int(idx),
                'position': level0_positions[idx].tolist(),
                'projected_position': projected_positions[idx].tolist() if idx < projected_positions.shape[0] else [],
                'true_label': int(true_labels[idx]) if idx < true_labels.shape[0] else -1,
            }
            if pred_arr is not None and idx < pred_arr.shape[0]:
                record['predicted_label'] = int(pred_arr[idx])
            for key, labels in hierarchy_outputs.items():
                labels_arr = np.asarray(labels, dtype=int).reshape(-1)
                if idx < labels_arr.shape[0]:
                    record[key] = int(labels_arr[idx])
            node_records.append(record)

        level_payloads: list[dict[str, Any]] = []
        upper_level_states: dict[str, Any] = {}
        levels = getattr(self.model, 'levels', None)
        if levels:
            representative_upper_level_state = None
            try:
                from tools.archive.rg_hierarchy_diagnostics import representative_upper_level_state as _representative_upper_level_state
                representative_upper_level_state = _representative_upper_level_state
            except Exception:
                representative_upper_level_state = None
            for level_index, level_model in enumerate(levels):
                level_name = f'level_{level_index}'
                level_labels = None
                if level_index == 0:
                    level_labels = pred_arr
                level_payloads.append(
                    self._level_snapshot(level_name, level_model, level_labels, level_index=level_index)
                )
            if callable(representative_upper_level_state):
                for lower_level_index in range(len(levels) - 1):
                    key = f'L{lower_level_index + 1}_to_L{lower_level_index + 2}'
                    try:
                        upper_level_states[key] = representative_upper_level_state(
                            self.model,
                            self.env,
                            lower_level_index,
                        )
                    except Exception:
                        continue
        else:
            level_payloads.append(self._level_snapshot('level_0', self.model, pred_arr, level_index=0))

        lifted_cluster_distances: dict[str, Any] = {}
        for key, labels in hierarchy_outputs.items():
            labels_arr = np.asarray(labels, dtype=int).reshape(-1)
            if level0_positions.ndim == 2 and labels_arr.shape[0] == level0_positions.shape[0]:
                lifted_cluster_distances[key] = self._cluster_distance_payload(level0_positions, labels_arr)

        payload = {
            'metadata': {
                'dataset': self._dataset_path(),
                'strategy': self._strategy_name,
                'impl': getattr(self, '_impl_name', self._strategy_name.upper()),
                'total_steps': int(getattr(self, 'total_steps', 0)),
                'running': bool(getattr(self, 'running', False)),
                'dimension': int(self.dimension),
                'view_mode': str(getattr(self, '_view_mode', 'xy')),
                'organize_level': self._organize_level_key() if hasattr(self.model, 'organize_level') else None,
                'formal_eval_readout': self._formal_readout_label(),
                'display_readout': self._display_readout_label(),
                'formal_eval_level': int(self._eval_label_level),
                'display_level': self._display_level_text(),
                'exported_at': datetime.now().isoformat(timespec='seconds'),
            },
            'params': {
                'embed_cluster_method': str(getattr(cluster_model, 'embed_cluster_method', self.embed_cluster_method)),
                'dbscan_eps': float(self.dbscan_eps),
                'dbscan_readout_eps': None if self.dbscan_readout_eps is None else float(self.dbscan_readout_eps),
                'dbscan_min_samples': int(self.dbscan_min_samples),
                'dbscan_auto_eps': bool(self.dbscan_auto_eps),
                'louvain_distance_factor': float(self.louvain_distance_factor),
                'louvain_knn_k': int(self.louvain_knn_k),
                'louvain_mutual_knn': bool(self.louvain_mutual_knn),
                'base_wall': float(getattr(graph_model, 'base_wall', getattr(self.model, 'base_wall', 0.0))),
                'density_ref': float(getattr(graph_model, 'density_ref', getattr(self.model, 'density_ref', 0.0))),
                'space_size': float(getattr(graph_model, 'space_size', getattr(self.model, 'space_size', 0.0))),
                'adaptation_rate': self._live_adaptation_rate(default=0.0),
                'stats_mode': str(getattr(graph_model, 'stats_mode', getattr(self.model, 'stats_mode', 'unknown'))),
                'stats_memory_steps': int(getattr(graph_model, 'stats_memory_steps', getattr(self.model, 'stats_memory_steps', 0))),
                'stats_memory_scale': float(getattr(self.model, 'stats_memory_scale', 0.0)),
                'stable_token_evidence_enabled': bool(getattr(self.model, 'stable_token_evidence_enabled', False)),
                'stable_token_evidence_memory_scale': float(getattr(self.model, 'stable_token_evidence_memory_scale', 0.0)),
                'stable_token_evidence_min_members': int(getattr(self.model, 'stable_token_evidence_min_members', 0)),
                'stable_token_evidence_semantic_gate_enabled': bool(getattr(self.model, 'stable_token_evidence_semantic_gate_enabled', True)),
                'stable_token_evidence_semantic_source': str(getattr(self.model, 'stable_token_evidence_semantic_source', 'slow_raw')),
                'stable_token_evidence_min_stable_refreshes': int(getattr(self.model, 'stable_token_evidence_min_stable_refreshes', 0)),
                'stable_token_evidence_semantic_z_score': float(getattr(self.model, 'stable_token_evidence_semantic_z_score', 0.0)),
                'stable_token_evidence_min_self_mass': float(getattr(self.model, 'stable_token_evidence_min_self_mass', 0.0)),
                'stable_token_evidence_self_cross_ratio': float(getattr(self.model, 'stable_token_evidence_self_cross_ratio', 0.0)),
                'stats_max_lag': int(getattr(graph_model, 'stats_max_lag', getattr(self.model, 'stats_max_lag', 0))),
                'seed': int(getattr(self, '_seed', 0)),
                'k_attr': float(getattr(level0_model, 'k_attr', getattr(self.model, 'k_attr', 0.0))),
                'cluster_distance_strength': float(getattr(level0_model, 'cluster_distance_strength', getattr(self.model, 'cluster_distance_strength', 0.0))),
                'graph_spring_repulse_shoulder_gain': float(getattr(level0_model, 'graph_spring_repulse_shoulder_gain', 0.0)),
                'knn': int(getattr(graph_model, 'knn', getattr(self.model, 'knn', 0))),
                'graph_theta': float(getattr(graph_model, 'graph_theta', getattr(self.model, 'graph_theta', 0.0))),
                'graph_r0': float(getattr(level0_model, 'graph_r0', getattr(self.model, 'graph_r0', 0.0))),
                'T_push': int(getattr(graph_model, 'T_push', getattr(self.model, 'T_push', 0))),
                'renormalize_interval': int(getattr(self.model, 'renormalize_interval', 0)),
                'rg_input_level': str(getattr(self.model, 'rg_input_level', '')),
                'source_unit_motion_coherence_enabled': bool(getattr(self.model, 'source_unit_motion_coherence_enabled', False)),
                'source_unit_motion_coherence_strength': float(getattr(self.model, 'source_unit_motion_coherence_strength', 0.0)),
                'source_unit_motion_coherence_max_step_ratio': float(getattr(self.model, 'source_unit_motion_coherence_max_step_ratio', 0.0)),
                'inter_level_support_reanchor_enabled': bool(getattr(self.model, 'inter_level_support_reanchor_enabled', False)),
                'inter_level_support_reanchor_strength': float(getattr(self.model, 'inter_level_support_reanchor_strength', 0.0)),
                'source_unit_barrier_enabled': bool(getattr(self.model, 'source_unit_barrier_enabled', False)),
                'source_unit_barrier_strength_ratio': float(getattr(self.model, 'source_unit_barrier_strength_ratio', 0.0)),
                'unit_retention_enabled': bool(getattr(self.model, 'unit_retention_enabled', False)),
                'unit_merge_hysteresis_steps': int(getattr(self.model, 'unit_merge_hysteresis_steps', 0)),
                'unit_merge_distance_ratio': float(getattr(self.model, 'unit_merge_distance_ratio', 0.0)),
            },
            'true_labels_display': true_labels.tolist(),
            'true_label_levels': true_label_levels,
            'predicted_labels_display': None if pred_arr is None else pred_arr.tolist(),
            'hierarchy_outputs': {key: np.asarray(value, dtype=int).tolist() for key, value in hierarchy_outputs.items()},
            'nodes': node_records,
            'levels': level_payloads,
            'upper_level_states': upper_level_states,
            'level0_distance_summary': self._distance_summary(level0_positions),
            'lifted_cluster_distances': lifted_cluster_distances,
        }

        target = self._snapshot_export_path()
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            print(f'[Warn] JSON snapshot export failed: {exc}')
            return
        print(f'Saved layout snapshot to {target}')

    def _export_current_gif(self) -> None:
        if self._precomputed_traj is None or self._precomputed_traj.ndim != 3 or self._precomputed_traj.shape[0] <= 1:
            print("[Warn] No recorded trajectory available for GIF export.")
            return
        final_step = None
        if self._precomputed_steps is not None and len(self._precomputed_steps) > 0:
            final_step = int(self._precomputed_steps[-1])
        target = default_anim_path(
            self._dataset_path(),
            self._strategy_name,
            method_name=self._impl_name,
            step_count=final_step,
        )
        try:
            animate_trajectory(
                self._precomputed_traj,
                labels=np.asarray(self.env.trueLabel(), dtype=int),
                steps=self._precomputed_steps,
                interval=80,
                save_path=target,
                show=False,
                method_name=self._impl_name,
                theme=self._theme,
                accent_color=self._accent_color,
            )
        except Exception as exc:
            print(f"[Warn] GIF export failed: {exc}")
            return
        print(f"Saved GIF to {target}")

    @staticmethod
    def _figure_is_open(fig: plt.Figure | None) -> bool:
        if fig is None:
            return False
        try:
            return bool(plt.fignum_exists(fig.number))
        except Exception:
            return False

    @staticmethod
    def _show_auxiliary_figure(fig: plt.Figure) -> None:
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
            button.label.set_text(self._format_toggle_label(kind, is_open))
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _close_auxiliary_figure(self, kind: str) -> None:
        attr = f"{kind}_fig"
        fig: plt.Figure | None = getattr(self, attr, None)
        setattr(self, attr, None)
        if self._figure_is_open(fig):
            try:
                plt.close(fig)
            except Exception:
                pass
        if kind == "curve":
            self.curve_ax = None
            self.curve_nmi_line = None
            self.curve_ari_line = None
        self._set_auxiliary_button(kind, False)

    def _register_auxiliary_figure(self, kind: str, fig: plt.Figure) -> None:
        setattr(self, f"{kind}_fig", fig)

        def _on_close(_event: Any) -> None:
            if getattr(self, f"{kind}_fig", None) is fig:
                setattr(self, f"{kind}_fig", None)
                if kind == "curve":
                    self.curve_ax = None
                    self.curve_nmi_line = None
                    self.curve_ari_line = None
                self._set_auxiliary_button(kind, False)

        fig.canvas.mpl_connect("close_event", _on_close)
        self._set_auxiliary_button(kind, True)

    @staticmethod
    def _format_curve_step(value: float, _position: int | None = None) -> str:
        magnitude = abs(float(value))
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:.2g}M"
        if magnitude >= 1_000:
            return f"{value / 1_000:.3g}k"
        return f"{value:.0f}"

    def _build_curve_figure(self) -> None:
        from matplotlib.ticker import FuncFormatter

        fig = self.curve_fig
        if not self._figure_is_open(fig):
            fig = plt.figure(figsize=(7.2, 4.6))
            self._register_auxiliary_figure("curve", fig)
            manager = getattr(fig.canvas, "manager", None)
            setter = getattr(manager, "set_window_title", None)
            if callable(setter):
                setter(f"{self._impl_name} · live NMI/ARI")
        else:
            fig.clear()
        self.curve_fig = fig
        palette = self._theme_palette()
        fig.set_facecolor(palette["fig_bg"])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(palette["ax_bg"])
        for spine in ax.spines.values():
            spine.set_color(palette["inactive"])
        ax.tick_params(colors=palette["fg"], labelsize=8)
        ax.grid(True, color=palette["grid"], alpha=0.45, linewidth=0.7)
        ax.set_title(
            f"{Path(self._dataset_path()).name} · {self._impl_name}",
            fontsize=10,
            color=palette["fg"],
        )
        ax.set_xlabel("steps", fontsize=8, color=palette["fg"])
        ax.set_ylabel("score", fontsize=8, color=palette["fg"])
        ax.set_ylim(-0.10, 1.05)
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_curve_step))
        hierarchy = bool(getattr(self.env, "is_hierarchical", False))
        nmi_label = "L1 NMI" if hierarchy else "NMI"
        ari_label = "mean-level ARI" if hierarchy else "ARI"
        nmi_line, = ax.plot([], [], color="#4c9be8", linewidth=1.7, label=nmi_label)
        ari_line, = ax.plot([], [], color="#f28e2b", linewidth=1.7, label=ari_label)
        legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(palette["fg"])
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.90)
        self.curve_ax = ax
        self.curve_nmi_line = cast(Artist, nmi_line)
        self.curve_ari_line = cast(Artist, ari_line)

    def _record_live_curve_point(self) -> None:
        formal_eval = self._formal_evaluation(force=False)
        if formal_eval is None:
            return
        payload = self._metric_payload(
            formal_eval.get("true_labels"),
            formal_eval.get("labels"),
        )
        if payload is None:
            return
        step = int(self.total_steps)
        nmi = float(payload["nmi_raw"])
        ari = float(payload["ari_raw"])
        if bool(getattr(self.env, "is_hierarchical", False)):
            outputs = self._hierarchy_outputs(force=False)
            entries = self._canonical_hierarchy_metric_lines(outputs)
            expected_depth = max(
                1,
                int(getattr(self.env, "hierarchy_depth", len(entries) or 1)),
            )
            total_payload = self._hierarchy_mean_metric_payload(
                entries,
                expected_depth=expected_depth,
            )
            if total_payload is not None:
                ari = float(total_payload["ari_raw"])
        if self._curve_steps and self._curve_steps[-1] == step:
            self._curve_nmi[-1] = nmi
            self._curve_ari[-1] = ari
        else:
            self._curve_steps.append(step)
            self._curve_nmi.append(nmi)
            self._curve_ari.append(ari)
        self._redraw_live_curve()

    def _redraw_live_curve(self) -> None:
        if (
            self.curve_ax is None
            or self.curve_nmi_line is None
            or self.curve_ari_line is None
            or not self._figure_is_open(self.curve_fig)
        ):
            return
        x_values = np.asarray(self._curve_steps, dtype=float)
        cast(Any, self.curve_nmi_line).set_data(x_values, np.asarray(self._curve_nmi, dtype=float))
        cast(Any, self.curve_ari_line).set_data(x_values, np.asarray(self._curve_ari, dtype=float))
        right = max(float(self.cluster_refresh_steps), float(self.total_steps), 1.0)
        self.curve_ax.set_xlim(0.0, right)
        self.curve_fig.canvas.draw_idle()

    def _toggle_curve_window(self) -> None:
        if self._figure_is_open(self.curve_fig):
            self._close_auxiliary_figure("curve")
            return
        self._build_curve_figure()
        self._redraw_live_curve()
        if self.curve_fig is not None:
            self._show_auxiliary_figure(self.curve_fig)

    def _reset_curve_history(self) -> None:
        self._curve_steps.clear()
        self._curve_nmi.clear()
        self._curve_ari.clear()
        self._redraw_live_curve()

    def _on_main_figure_close(self, _event: Any) -> None:
        for kind in ("input", "graph", "curve"):
            self._close_auxiliary_figure(kind)

    def _show_input_sequence(self) -> None:
        if self._figure_is_open(self.input_fig):
            self._close_auxiliary_figure("input")
            return
        try:
            axes = visualize_input_sequence(self.env, show=False)
        except Exception as exc:
            print(f"[Warn] Unable to show input sequence: {exc}")
            return
        fig = axes.figure
        manager = getattr(fig.canvas, "manager", None)
        setter = getattr(manager, "set_window_title", None)
        if callable(setter):
            setter(f"{Path(self._dataset_path()).name} · input")
        self._register_auxiliary_figure("input", fig)
        self._show_auxiliary_figure(fig)

    def _show_graph_preview(self) -> None:
        if self._figure_is_open(self.graph_fig):
            self._close_auxiliary_figure("graph")
            return
        try:
            import networkx as nx
        except Exception as exc:
            print(f"[Warn] Unable to import networkx for graph preview: {exc}")
            return

        previews = self._graph_preview_objects(nx)
        if not previews:
            print("[Warn] No learned or environment graph is available for preview.")
            return
        try:
            panel_count = len(previews)
            fig, axes = plt.subplots(1, panel_count, figsize=(7.0 * panel_count, 7.0))
            axes_arr = np.atleast_1d(axes)
            fig.suptitle(f"Graph Comparison [{Path(self._dataset_path()).name}]", y=0.98)
            for ax, preview in zip(axes_arr.tolist(), previews):
                self._draw_graph_preview_panel(nx, cast(Axes, ax), preview)
                print(
                    "graph_preview: "
                    f"edges={preview['edge_source']} "
                    f"labels={preview['label_source']} "
                    f"nodes={preview['graph'].number_of_nodes()} "
                    f"edges_count={preview['graph'].number_of_edges()}"
                )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            self._register_auxiliary_figure("graph", fig)
            self._show_auxiliary_figure(fig)
        except Exception as exc:
            print(f"[Warn] Unable to render graph preview: {exc}")

    def _draw_graph_preview_panel(
        self,
        nx_module: Any,
        ax: Axes,
        preview: dict[str, Any],
    ) -> None:
        graph_obj = preview['graph']
        labels = preview.get('labels')
        label_source = str(preview.get('label_source', 'none'))
        node_colors = None
        if labels is not None and np.asarray(labels, dtype=int).shape[0] == graph_obj.number_of_nodes():
            node_colors = self._colors_for_labels(labels, singleton_gray=False)
        node_text = self._graph_preview_node_text(graph_obj, labels, label_source=label_source)
        pos = preview.get('layout')
        if pos is None:
            pos = nx_module.spring_layout(graph_obj, seed=0, weight='weight')
        draw_kwargs: dict[str, Any] = {
            'ax': ax,
            'node_size': 220,
            'font_size': 7,
            'arrows': bool(graph_obj.is_directed()),
            'with_labels': True,
            'labels': node_text,
        }
        if node_colors is not None:
            draw_kwargs['node_color'] = node_colors
        nx_module.draw_networkx(graph_obj, pos=pos, **draw_kwargs)
        ax.set_title(
            f"{preview['edge_source']}\nlabels={label_source}"
        )
        ax.set_axis_off()

    def _graph_preview_node_names(self, node_count: int) -> list[str]:
        graph_obj = getattr(self.env, 'g', None)
        if graph_obj is not None:
            try:
                names = [str(node) for node in list(graph_obj.nodes())]
                if len(names) == int(node_count):
                    return names
            except Exception:
                pass
        return [str(idx) for idx in range(int(node_count))]

    def _graph_preview_labels(
        self,
        node_count: int,
    ) -> tuple[NDArray[np.integer[Any]] | None, str]:
        predicted = self._predicted_labels_for_display(force=True)
        if predicted is not None:
            pred_arr = np.asarray(predicted, dtype=int).reshape(-1)
            if pred_arr.shape[0] == int(node_count):
                if bool(getattr(self.env, 'is_hierarchical', False)):
                    return pred_arr, f"predicted[{self._display_level_text()}]"
                return pred_arr, "predicted"

        true_labels = self._true_labels_for_display()
        true_arr = np.asarray(true_labels, dtype=int).reshape(-1)
        if true_arr.shape[0] == int(node_count):
            if bool(getattr(self.env, 'is_hierarchical', False)):
                return true_arr, f"true[{self._display_level_text()}]"
            return true_arr, "true"
        return None, "none"

    def _graph_preview_node_text(
        self,
        graph_obj: Any,
        labels: NDArray[np.integer[Any]] | None,
        *,
        label_source: str,
    ) -> dict[Any, str]:
        names = self._graph_preview_node_names(graph_obj.number_of_nodes())
        text_map: dict[Any, str] = {}
        nodes = list(graph_obj.nodes())
        labels_arr = None if labels is None else np.asarray(labels, dtype=int).reshape(-1)
        if labels_arr is None:
            for idx, node in enumerate(nodes):
                text_map[node] = names[idx]
            return text_map

        prefix = "p" if label_source.startswith("predicted") else "t"
        for idx, node in enumerate(nodes):
            label_value = int(labels_arr[idx]) if idx < labels_arr.shape[0] else -1
            text_map[node] = f"{names[idx]}\n{prefix}{label_value}"
        return text_map

    def _graph_preview_objects(
        self,
        nx_module: Any,
    ) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        level0_model = self._base_dynamics_model()
        graph_tuple = getattr(level0_model, 'graph', None)
        if graph_tuple is not None:
            try:
                rows, cols, weights = graph_tuple
                rows_arr = np.asarray(rows, dtype=int).reshape(-1)
                cols_arr = np.asarray(cols, dtype=int).reshape(-1)
                weights_arr = np.asarray(weights, dtype=float).reshape(-1)
                if rows_arr.size > 0 and cols_arr.size > 0 and weights_arr.size > 0:
                    names = self._graph_preview_node_names(int(level0_model.input_size))
                    graph_obj = nx_module.Graph()
                    graph_obj.add_nodes_from(names)
                    for row, col, weight in zip(rows_arr.tolist(), cols_arr.tolist(), weights_arr.tolist()):
                        if int(row) < 0 or int(col) < 0 or int(row) >= len(names) or int(col) >= len(names):
                            continue
                        if float(weight) <= 0.0:
                            continue
                        graph_obj.add_edge(names[int(row)], names[int(col)], weight=float(weight))
                    if graph_obj.number_of_edges() > 0 or graph_obj.number_of_nodes() > 0:
                        labels, label_source = self._graph_preview_labels(graph_obj.number_of_nodes())
                        previews.append(
                            {
                                'graph': graph_obj,
                                'edge_source': 'learned_graph',
                                'labels': labels,
                                'label_source': label_source,
                            }
                        )
            except Exception:
                pass

        env_graph = getattr(self.env, 'g', None)
        if env_graph is not None:
            try:
                env_graph_obj = env_graph.copy()
            except Exception:
                env_graph_obj = env_graph
            labels, label_source = self._graph_preview_labels(env_graph_obj.number_of_nodes())
            preview: dict[str, Any] = {
                'graph': env_graph_obj,
                'edge_source': 'environment_graph',
                'labels': labels,
                'label_source': label_source,
            }
            if previews:
                learned_nodes = list(previews[0]['graph'].nodes())
                env_nodes = list(env_graph_obj.nodes())
                if learned_nodes == env_nodes:
                    try:
                        preview['layout'] = nx_module.spring_layout(env_graph_obj, seed=0, weight='weight')
                        previews[0]['layout'] = preview['layout']
                    except Exception:
                        pass
            previews.append(preview)
        return previews

    def _set_title(self) -> None:
        try:
            ds_name = Path(self._dataset_path()).name
            ds_part = f" [{ds_name}]"
        except Exception: ds_part = ""
        self.ax.set_title(f"Live {self._impl_name}{ds_part}", pad=8)

    def _build_legend(self, labels: Any | None = None) -> None:
        legend = getattr(self, '_legend', None)
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                pass
            self._legend = None
        self._clear_legend_box()

        try:
            labels_arr = np.asarray(
                self._legend_labels_for_current_mode() if labels is None else labels,
                dtype=int,
            ).reshape(-1)
        except Exception:
            return
        if labels_arr.size == 0:
            return
        uniq = np.unique(labels_arr)
        color_lookup = self._label_color_lookup(labels_arr)
        label_name_map = self._legend_label_name_map(labels_arr)

        # Always use native matplotlib legend — no class-count limit.
        # For very large class counts, shrink font & marker to keep it compact.
        n_classes = int(uniq.size)
        if n_classes <= 12:
            marker_size, font_size, n_col = 6, 8, 1
        elif n_classes <= 24:
            marker_size, font_size, n_col = 5, 7, 2
        else:
            marker_size, font_size, n_col = 4, 6.5, 2

        handles = []
        fg_color = self._theme_palette().get('fg', '#000000')
        for label_idx in uniq.tolist():
            color = color_lookup.get(int(label_idx), tuple(self._singleton_color.tolist()))
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='none',
                    markerfacecolor=color,
                    markersize=marker_size,
                    label=label_name_map.get(int(label_idx), f"class {int(label_idx)}"),
                )
            )
        self._legend = self.ax.legend(
            handles=handles,
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            fontsize=font_size,
            frameon=False,
            title=None,
            ncol=n_col,
            handletextpad=0.3,
            columnspacing=0.8,
            labelspacing=0.25,
        )
        try:
            for text in self._legend.get_texts():
                text.set_color(fg_color)
        except Exception:
            pass

    def _theme_palette(self) -> dict[str, Any]:
        palette = resolve_plot_palette(theme=self._theme, accent=self._accent_color)
        # Persist palette onto self for cross-method reuse
        self._palette = palette
        self._color_fig_bg = str(palette['fig_bg'])
        self._color_ax_bg = str(palette['ax_bg'])
        self._color_panel_bg = str(palette['panel_bg'])
        self._color_fg = str(palette['fg'])
        self._color_inactive = str(palette['inactive'])
        self._color_slider_rest = str(palette['slider_rest'])
        self._color_slider_label = str(palette['slider_label'])
        self._color_btn_bg = str(palette['btn_bg'])
        self._color_btn_edge = str(palette['btn_edge'])
        self._color_btn_hover = str(palette['btn_hover'])
        self._color_line = str(palette['line'])
        return palette

    def _theme_button_label(self) -> str:
        return self._theme

    def apply_visual_theme(
        self,
        *,
        theme: str | None = None,
        accent_color: str | None = None,
    ) -> None:
        if theme not in {None, ""}:
            normalized = str(theme).strip().lower()
            if normalized in {"light", "dark"}:
                self._theme = normalized
        if accent_color not in {None, ""}:
            self._accent_color = str(accent_color)
        self._apply_theme()

    def _toggle_theme(self) -> None:
        self._theme = 'dark' if self._theme == 'light' else 'light'
        if getattr(self, 'b_theme', None) is not None:
            self.b_theme.label.set_text(self._theme_button_label())
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._theme_palette()  # refresh and persist palette on self
        fig_bg = self._color_fig_bg
        ax_bg = self._color_ax_bg
        panel_bg = self._color_panel_bg
        fg = self._color_fg
        accent = self._accent_color
        slider_rest = self._color_slider_rest
        slider_label_color = self._color_slider_label
        btn_bg = self._color_btn_bg
        btn_edge = self._color_btn_edge
        btn_hover = self._color_btn_hover
        inactive = self._color_inactive
        line_color = self._color_line
        self.fig.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        for spine in self.ax.spines.values(): spine.set_color(fg)
        self.ax.tick_params(colors=fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        if isinstance(self.ax, Axes3D):
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                label = getattr(axis, 'label', None)
                if label is not None: label.set_color(fg)
                line = getattr(axis, 'line', None)
                if line is not None: line.set_color(fg)
                pane = getattr(axis, 'pane', None)
                if pane is not None: pane.set_facecolor(ax_bg)
        if getattr(self, 'ax_panel', None) is not None:
            self.ax_panel.set_facecolor(panel_bg)
            for sp in self.ax_panel.spines.values(): sp.set_color(fg)
            try: self.ax_panel.title.set_color(fg)
            except Exception: pass
        if getattr(self, 'text_metrics', None) is not None: self.text_metrics.set_color(fg)
        if getattr(self, 'text_switch', None) is not None: self.text_switch.set_color(fg)
        try:
            for s_ax, sld in self._slider_axes_map.items():
                s_ax.set_facecolor(panel_bg)
                for sp in s_ax.spines.values(): sp.set_edgecolor(btn_edge)
                self._layout_slider_text(sld)
                sld.label.set_color(slider_label_color)
                sld.label.set_bbox(
                    {
                        'facecolor': 'none',
                        'edgecolor': 'none',
                        'alpha': 0.0,
                        'pad': 0.0,
                    }
                )
                sld.valtext.set_color(fg)
                self._color_slider_handle(sld, accent)
                for attr in ('poly', 'hline', 'vline'):
                    artist = getattr(sld, attr, None)
                    if artist: artist.set_color(accent)
                track = getattr(sld, 'track', None)
                if track:
                    track.set_facecolor(slider_rest)
                    track.set_edgecolor(slider_rest)
                else: sld.ax.set_facecolor(slider_rest)
        except Exception: pass
        for btn in self._all_buttons:
            if btn is None: continue
            try:
                btn.ax.set_facecolor(btn_bg)
                btn.color = btn_bg 
                for sp in btn.ax.spines.values(): sp.set_edgecolor(btn_edge)
                btn.label.set_color(fg)
                btn.hovercolor = btn_hover
            except Exception: pass
        for rb in getattr(self, '_radio_groups', []):
            if rb is None: continue
            try:
                rb.ax.set_facecolor(panel_bg)
                for sp in rb.ax.spines.values(): sp.set_edgecolor(btn_edge)
                for lab in rb.labels: lab.set_color(fg)
                rb.activecolor = accent
                for circ in getattr(rb, 'circles', []): circ.set_edgecolor(accent)
            except Exception: pass
        for cb in getattr(self, '_check_groups', []):
            if cb is None: continue
            try:
                cb.ax.set_facecolor(panel_bg)
                for sp in cb.ax.spines.values(): sp.set_edgecolor(btn_edge)
                for lab in cb.labels: lab.set_color(fg)
                self._set_check_colors(cb, accent)
                for lab in cb.labels:
                    x, y = lab.get_position()
                    lab.set_position((x + 0.03, y))
            except Exception: pass
        if self.cluster_lines is not None: self.cluster_lines.set_color(line_color)
        leg = getattr(self, '_legend', None)
        if leg:
            for text in leg.get_texts(): text.set_color(fg)
        legend_box = getattr(self, '_legend_info_ax', None)
        if legend_box is not None:
            legend_box.set_facecolor(panel_bg)
            legend_box.patch.set_alpha(0.88)
            for text in legend_box.texts:
                text.set_color(fg)
        for lax in getattr(self, '_label_axes', []):
            lax.set_facecolor(panel_bg)
            for sp in lax.spines.values(): sp.set_color(fg)
            for t in getattr(lax, 'texts', []): t.set_color(fg)
        if getattr(self, 'ax_dataset_text', None) is not None:
            self.ax_dataset_text.set_facecolor(panel_bg)
            if self.dataset_text_artist is not None:
                cast(Any, self.dataset_text_artist).set_color(fg)
        circ = getattr(self, 'theme_circle', None)
        if circ:
            face = '#000000' if self._theme == 'dark' else '#ffffff'
            circ.set_facecolor(face)
            circ.set_edgecolor(fg)
        lab = getattr(self, 'theme_label', None)
        if lab:
            lab.set_text('D' if self._theme == 'dark' else 'L')
            lab.set_color(fg)
        self._apply_algorithm_highlights(accent, inactive)
        self._apply_mesh_visibility()
        if self._figure_is_open(getattr(self, "curve_fig", None)):
            self._build_curve_figure()
            self._redraw_live_curve()
        self.fig.canvas.draw_idle()

    def _set_check_colors(self, cb: CheckButtons, color: str) -> None:
        try:
            if hasattr(cb, 'check_props'):
                props = getattr(cb, 'check_props')
                props.update({'facecolor': color, 'edgecolor': color})
            if hasattr(cb, 'frame_props'):
                fprops = getattr(cb, 'frame_props')
                fprops.update({'edgecolor': color})
        except Exception: pass
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for rect in getattr(cb, 'rectangles', []):
                    rect.set_facecolor('none')
                    rect.set_edgecolor(color)
                for lines in getattr(cb, 'lines', []):
                    try:
                        for line in lines: line.set_color(color)
                    except TypeError: lines.set_color(color)
        except Exception: pass

    def _clear_edge_arrows(self) -> None:
        if self.edge_arrows is None:
            return
        try:
            self.edge_arrows.remove()
        except Exception:
            pass
        self.edge_arrows = None

    def _edge_display_support(self) -> tuple[np.ndarray[Any, Any] | None, bool]:
        samples = getattr(self.model, 'syncmap', None)
        if samples is None:
            return None, False
        n_nodes = int(np.asarray(samples).shape[0])
        if n_nodes <= 0:
            return None, False

        if bool(getattr(self, 'use_pred_colors', False)):
            getter = getattr(self.model, 'get_estimated_adjacency', None)
            support = None
            if callable(getter):
                try:
                    support = np.asarray(getter(), dtype=int)
                except Exception:
                    support = None
            if support is None:
                support = self._predicted_cluster_skeleton_support(np.asarray(samples, dtype=float))
                if support is None:
                    return None, False
        else:
            getter = getattr(self.env, 'get_true_adjacency_support', None)
            if callable(getter):
                support = np.asarray(getter(), dtype=int)
            else:
                base_adj = getattr(self.env, 'a', None)
                if base_adj is None:
                    return None, False
                support = np.asarray(np.asarray(base_adj) > 0, dtype=int)

        if support.ndim != 2 or support.shape != (n_nodes, n_nodes):
            return None, False
        support = np.asarray(support > 0, dtype=int)
        np.fill_diagonal(support, 0)
        if not np.any(support):
            return support, False

        if bool(getattr(self, 'use_pred_colors', False)):
            directed = bool(not np.array_equal(support, support.T))
        else:
            directed = bool(getattr(self.env, 'is_effectively_directed', lambda: not np.array_equal(support, support.T))())
        return support, directed

    def _predicted_cluster_skeleton_support(
        self,
        samples: NDArray[np.float64],
    ) -> np.ndarray[Any, Any] | None:
        labels = self._predicted_labels_for_display(force=False)
        if labels is None or labels.ndim != 1 or labels.shape[0] != samples.shape[0]:
            return None

        n_nodes = int(samples.shape[0])
        support = np.zeros((n_nodes, n_nodes), dtype=int)
        valid_labels = sorted({int(label) for label in labels.tolist() if int(label) >= 0})
        if not valid_labels:
            return support

        def add_edge(i: int, j: int) -> None:
            if i == j:
                return
            support[i, j] = 1
            support[j, i] = 1

        for label in valid_labels:
            members = np.where(labels == label)[0]
            m = int(members.size)
            if m < 2:
                continue
            if m == 2:
                add_edge(int(members[0]), int(members[1]))
                continue

            local = np.asarray(samples[members], dtype=float)
            diff = local[:, None, :] - local[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)

            k = min(2, m - 1)
            order = np.argsort(dist, axis=1)[:, :k]
            for i_local in range(m):
                for j_local in order[i_local].tolist():
                    add_edge(int(members[i_local]), int(members[int(j_local)]))

            visited = {0}
            while len(visited) < m:
                best_i = -1
                best_j = -1
                best_d = float('inf')
                for i_local in visited:
                    row = dist[i_local]
                    for j_local in range(m):
                        if j_local in visited:
                            continue
                        d = float(row[j_local])
                        if d < best_d:
                            best_d = d
                            best_i = i_local
                            best_j = j_local
                if best_i < 0 or best_j < 0:
                    break
                add_edge(int(members[best_i]), int(members[best_j]))
                visited.add(best_j)
        return support

    def _edge_focus_labels(self) -> np.ndarray[Any, Any] | None:
        labels = self._predicted_labels_for_display(force=False)
        if labels is None:
            return None
        samples = getattr(self.model, 'syncmap', None)
        if samples is None:
            return None
        n_nodes = int(np.asarray(samples).shape[0])
        if labels.ndim != 1 or labels.shape[0] != n_nodes:
            return None
        return labels

    def _filter_edges_to_focus_layer(
        self,
        src: NDArray[np.int_],
        dst: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        labels = self._edge_focus_labels()
        if labels is None or src.size == 0:
            return src, dst
        src_labels = labels[src]
        dst_labels = labels[dst]
        keep = (src_labels >= 0) & (dst_labels >= 0) & (src_labels == dst_labels)
        if not np.any(keep):
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        return src[keep], dst[keep]

    def _select_display_edges(
        self,
        pos: NDArray[np.float64],
        support: np.ndarray[Any, Any],
        directed: bool,
    ) -> tuple[np.ndarray[Any, Any], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        edge_index = np.argwhere(np.asarray(support) > 0)
        if edge_index.size == 0:
            empty_pairs = np.empty((0, 2), dtype=int)
            empty_pos = np.empty((0, pos.shape[1]), dtype=float)
            empty_dist = np.empty(0, dtype=float)
            return empty_pairs, empty_pos, empty_pos, empty_dist

        src = edge_index[:, 0].astype(int)
        dst = edge_index[:, 1].astype(int)
        src, dst = self._filter_edges_to_focus_layer(src, dst)
        if src.size == 0:
            empty_pairs = np.empty((0, 2), dtype=int)
            empty_pos = np.empty((0, pos.shape[1]), dtype=float)
            empty_dist = np.empty(0, dtype=float)
            return empty_pairs, empty_pos, empty_pos, empty_dist
        if not directed:
            keep = src < dst
            src = src[keep]
            dst = dst[keep]
        if src.size == 0:
            empty_pairs = np.empty((0, 2), dtype=int)
            empty_pos = np.empty((0, pos.shape[1]), dtype=float)
            empty_dist = np.empty(0, dtype=float)
            return empty_pairs, empty_pos, empty_pos, empty_dist

        src_pos = np.asarray(pos[src], dtype=float)
        dst_pos = np.asarray(pos[dst], dtype=float)
        dist2 = np.sum((dst_pos - src_pos) ** 2, axis=1)
        order = np.argsort(dist2, kind='stable')
        if order.size > self.max_relation_edges:
            order = order[: self.max_relation_edges]
        pairs = np.column_stack((src[order], dst[order]))
        return pairs, src_pos[order], dst_pos[order], np.sqrt(dist2[order])

    def _edge_distance_scale(self, distances: NDArray[np.float64]) -> float:
        for attr in ('significant_distance', 'graph_r0'):
            value = getattr(self.model, attr, None)
            if value is None:
                continue
            try:
                scale = float(value)
            except Exception:
                continue
            if scale > 1e-9:
                return scale

        local_scale = getattr(self.model, 'effective_local_scale', None)
        if local_scale is not None:
            try:
                scale = 2.0 * float(local_scale)
            except Exception:
                scale = 0.0
            if scale > 1e-9:
                return scale

        if distances.size == 0:
            return 1.0
        return max(1e-6, float(np.median(distances)))

    def _edge_visual_strength(
        self,
        distances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        scale = self._edge_distance_scale(distances)
        ratio = distances / max(scale, 1e-6)
        strength = 1.0 / (1.0 + ratio * ratio)
        alpha = 0.05 + 0.45 * strength
        linewidth = 0.08 + 0.28 * np.sqrt(strength)
        return strength, alpha, linewidth, scale

    def _update_edge_overlay(self, pos: NDArray[np.float64]) -> None:
        show_edges = bool(getattr(self, 'show_cluster_edges', False))
        if self.cluster_lines is None:
            self._clear_edge_arrows()
            return
        if not show_edges:
            self.cluster_lines.set_segments([])
            self.cluster_lines.set_visible(False)
            self._clear_edge_arrows()
            return

        support, directed = self._edge_display_support()
        if support is None:
            self.cluster_lines.set_segments([])
            self.cluster_lines.set_visible(False)
            self._clear_edge_arrows()
            return

        edge_pairs, src_pos, dst_pos, distances = self._select_display_edges(pos, support, directed)
        if edge_pairs.shape[0] == 0:
            self.cluster_lines.set_segments([])
            self.cluster_lines.set_visible(False)
            self._clear_edge_arrows()
            return

        segments = [[tuple(src), tuple(dst)] for src, dst in zip(src_pos.tolist(), dst_pos.tolist())]
        _strength, alpha, linewidth, _scale = self._edge_visual_strength(distances)
        base_color = np.asarray(to_rgba(getattr(self, '_color_line', 'tab:gray')), dtype=float)
        edge_colors = np.tile(base_color, (edge_pairs.shape[0], 1))
        edge_colors[:, 3] = alpha
        self.cluster_lines.set_segments(segments)
        self.cluster_lines.set_color(edge_colors)
        self.cluster_lines.set_linewidth(linewidth.tolist())
        self.cluster_lines.set_visible(True)

        self._clear_edge_arrows()
        if not directed:
            return

        arrow_count = min(self.max_relation_arrows, edge_pairs.shape[0])
        if arrow_count <= 0:
            return
        src_arrow = src_pos[:arrow_count]
        dst_arrow = dst_pos[:arrow_count]
        delta = dst_arrow - src_arrow
        start = src_arrow + 0.72 * delta
        vec = 0.18 * delta
        keep = np.linalg.norm(vec, axis=1) > 1e-9
        if not np.any(keep):
            return
        start = start[keep]
        vec = vec[keep]
        arrow_alpha = alpha[:arrow_count][keep]
        line_color = np.asarray(to_rgba(getattr(self, '_color_line', 'tab:gray')), dtype=float)
        arrow_colors = np.tile(line_color, (start.shape[0], 1))
        arrow_colors[:, 3] = arrow_alpha

        try:
            if self.dimension == 2:
                ax2d = cast(Axes, self.ax)
                self.edge_arrows = cast(
                    Artist,
                    ax2d.quiver(
                        start[:, 0],
                        start[:, 1],
                        vec[:, 0],
                        vec[:, 1],
                        angles='xy',
                        scale_units='xy',
                        scale=1.0,
                        color=arrow_colors,
                        width=0.0014,
                        headwidth=2.6,
                        headlength=3.2,
                        headaxislength=2.8,
                        zorder=0.6,
                    ),
                )
            else:
                ax3d = cast(Axes3D, self.ax)
                self.edge_arrows = cast(
                    Artist,
                    ax3d.quiver(
                        start[:, 0],
                        start[:, 1],
                        start[:, 2],
                        vec[:, 0],
                        vec[:, 1],
                        vec[:, 2],
                        color=arrow_colors,
                        linewidth=0.28,
                        arrow_length_ratio=0.45,
                    ),
                )
        except Exception:
            self._clear_edge_arrows()

    def _build_theme_button(self) -> None:
        try:
            if getattr(self, 'ax_theme_btn', None):
                self.ax_theme_btn.remove()
        except Exception:
            pass
        self.ax_theme_btn = None
        self.theme_circle = None
        self.theme_label = None

    def _dataset_path(self) -> str:
        return getattr(self, "_ds_path", DATASET)

    def _reset_playback(self) -> None:
        """重置预录制轨迹的播放游标，并将模型同步到第 0 帧。"""
        self._traj_cursor = 0
        if self._precomputed_traj is None:
            return
        if self._precomputed_traj.shape[0] == 0:
            self._precomputed_traj = None
            return
        first = self._precomputed_traj[0]
        try:
            self.model.syncmap = np.array(first, copy=True)
        except Exception:
            self.model.syncmap = first
        if self._precomputed_steps is not None and self._precomputed_steps.size > 0:
            try:
                self.total_steps = int(self._precomputed_steps[0])
            except Exception:
                self.total_steps = 0
        else:
            self.total_steps = 0

    def _simulate_chunk(self) -> None:
        if self._precomputed_traj is not None:
            if self._traj_cursor >= self._precomputed_traj.shape[0]:
                self.running = False
                return
            frame = self._precomputed_traj[self._traj_cursor]
            try:
                self.model.syncmap = np.array(frame, copy=True)
            except Exception:
                self.model.syncmap = frame
            if self._precomputed_steps is not None and self._traj_cursor < self._precomputed_steps.shape[0]:
                self.total_steps = int(self._precomputed_steps[self._traj_cursor])
            else:
                self.total_steps = self._traj_cursor
            self._traj_cursor += 1
            return

        # Keep live training semantically aligned with task0/task2. Two
        # alignment constraints apply at once:
        # 1) one environment sample per model update, because models that
        #    consume global np.random during input() would otherwise train on
        #    a different environment sequence than formal pre-generated runs;
        # 2) phase_offset must be the cumulative step count, because engines
        #    that gate periodic work on (phase_offset + t) % period (e.g. the
        #    NS physics_batch_size gate) change dynamics entirely when every
        #    call restarts at phase 0.
        steps_to_run = int(self.step_chunk)
        if self._auto_pause_steps is not None:
            remaining = int(self._auto_pause_steps) - int(self.total_steps)
            if remaining <= 0:
                self._set_running_state(False)
                if not self._auto_pause_announced:
                    print(
                        f"Auto-paused at step {self.total_steps} "
                        f"(target={self._auto_pause_steps})."
                    )
                    self._auto_pause_announced = True
                return
            steps_to_run = min(steps_to_run, remaining)

        with _isolated_environment_rng(self.env):
            for _ in range(steps_to_run):
                x = np.asarray(self.env.getInput(), dtype=np.float64).reshape(1, self.output_size)
                _advance_model_chunk(
                    self.model,
                    x,
                    phase_offset=self.total_steps,
                    reset_trajectory=(self.total_steps == 0),
                    append_final_frame=False,
                )
                self.total_steps += 1

        if (
            self._auto_pause_steps is not None
            and self.total_steps >= self._auto_pause_steps
        ):
            self._set_running_state(False)
            if not self._auto_pause_announced:
                print(
                    f"Auto-paused at step {self.total_steps} "
                    f"(target={self._auto_pause_steps})."
                )
                self._auto_pause_announced = True

    def _should_recluster(self) -> bool:
        if self._last_labels is None:
            return True
        if self.total_steps < self._last_cluster_step:
            return True
        if (
            self._auto_pause_steps is not None
            and self.total_steps >= self._auto_pause_steps
            and self._last_cluster_step < self.total_steps
        ):
            return True
        return (self.total_steps - self._last_cluster_step) >= self.cluster_refresh_steps

    def _update_metrics_and_plot(self, recompute_labels: bool = True) -> None:
        labels_pred: NDArray[np.integer[Any]] | None
        
        # hierarchy 模型也要尊重重算节流，否则每帧都会强制 organize_hierarchy()
        if bool(getattr(self.env, 'is_hierarchical', False)) and hasattr(self.model, 'organize'):
            if recompute_labels or self._last_labels is None:
                try:
                    if recompute_labels:
                        self._invalidate_clustering_cache()
                    self._last_cluster_step = int(self.total_steps)
                    labels_pred = self._current_predicted_labels(force=recompute_labels)
                    self._last_labels = labels_pred
                    self._sync_live_clustering_state_from_model()
                except Exception:
                    labels_pred = None
            else:
                labels_pred = self._last_labels
        else:
            # 非 hierarchy 模型也使用和正式实验相同的统一读出。
            if recompute_labels or self._last_labels is None:
                self._last_cluster_step = int(self.total_steps)
                labels_pred = self._current_predicted_labels(force=recompute_labels)
                self._last_labels = labels_pred
                if labels_pred is not None:
                    self._last_cluster_step = int(self.total_steps)
            else:
                labels_pred = self._last_labels

        pos = self._project_positions(getattr(self.model, 'syncmap', np.zeros((0, self.dimension))), self.dimension)
        if self.dimension == 2:
            self.scatter.set_offsets(pos)
        else:
            cast(Any, self.scatter)._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # === [新增] 可视化质量 (Mass) ===
        # 获取质量并归一化到合适的显示尺寸范围 (例如 10 ~ 100)
        if hasattr(self.model, 'mass'):
            mass = self.model.mass
            # 简单的线性映射: size = base_size + mass * factor
            # 假设 mass 范围 1.0 ~ 50.0
            sizes = 20.0 + (mass - 1.0) * 5.0 
            # 限制一下最大显示尺寸，防止遮挡
            sizes = np.clip(sizes, 20.0, 150.0)
            self.scatter.set_sizes(sizes)
        # ==============================

        self._update_edge_overlay(pos)

        display_labels = self._display_labels_for_current_mode()
        try:
            scatter_artist = cast(Any, self.scatter)
            colors = self._colors_for_labels(display_labels, singleton_gray=bool(self.use_pred_colors))
            scatter_artist.set_array(None)
            scatter_artist.set_facecolors(colors)
            scatter_artist.set_edgecolors(colors)
            scatter_artist.set_linewidths(0.0)
        except Exception:
            pass

        if recompute_labels or (self._legend is None and self._legend_info_ax is None):
            try:
                self._build_legend(self._legend_labels_for_current_mode())
            except Exception:
                pass

        try:
            self.text_metrics.set_text(self._format_metric_text(labels_pred))
        except Exception:
            self.text_metrics.set_text(f"NMI/ARI: --/-- | t={self.total_steps}")
        if recompute_labels:
            self._record_live_curve_point()
        try:
            switch_text = self._format_switch_text()
            self.text_switch.set_text(switch_text)
            self.text_switch.set_visible(bool(switch_text))
        except Exception:
            self.text_switch.set_text("")
            self.text_switch.set_visible(False)

        span = float(np.max(np.abs(pos))) if pos.size else 1.0
        limit = max(1.0, span * 1.05)
        self._update_axis_limits(limit)

    def _update_axis_limits(self, limit: float, *, force: bool = False) -> None:
        target = float(max(1.0, limit))
        current = self._current_axis_limit
        if not force and current is not None:
            if target <= current * 1.02 and target >= current * 0.65:
                return
        self.ax.set_xlim(-target, target)
        self.ax.set_ylim(-target, target)
        if isinstance(self.ax, Axes3D):
            self.ax.set_zlim(-target, target)
            self._apply_mesh_visibility(target)
        self._current_axis_limit = target

    def _cluster_embed_labels(self) -> NDArray[np.integer[Any]] | None:
        # Display clustering must not advance the global RNG stream, or the
        # live environment sequence would diverge from formal pre-generated
        # sequences whenever a readout method consumes randomness.
        with _rng_neutral():
            return self._cluster_embed_labels_impl()

    def _cluster_embed_labels_impl(self) -> NDArray[np.integer[Any]] | None:
        if self._uses_direct_model_labels():
            try:
                labels = np.asarray(self._current_predicted_labels(force=True), dtype=int)
                setattr(self.model, 'labels', labels)
                return labels
            except Exception:
                return cast(Any, getattr(self.model, 'labels', None))

        organizer = getattr(self.model, 'organize', None)
        if callable(organizer):
            try:
                labels = np.asarray(organizer(), dtype=int)
                setattr(self.model, 'labels', labels)
                return labels
            except Exception:
                pass

        if self.dbscan_auto_eps:
            cluster_model = self._live_cluster_target_model()
            new_eps_raw = getattr(cluster_model, 'dbscan_eps', self.dbscan_eps)
            new_eps = self._to_float(self.dbscan_eps if new_eps_raw is None else new_eps_raw)
            if abs(new_eps - self._to_float(self.dbscan_eps)) > 1e-6:
                self.dbscan_eps = new_eps
                if self.s_eps is not None:
                    try:
                        self._suppress_slider_events = True
                        self._require_slider(self.s_eps, 'eps').set_val(new_eps)
                    finally: self._suppress_slider_events = False
        
        new_space = getattr(self.model, 'space_size', None)
        if new_space is not None:
            self._update_space_slider_display(float(new_space))
        
        samples = getattr(self.model, 'syncmap', None)
        if samples is None:
            return cast(Any, getattr(self.model, 'labels', None))
        samples_arr = np.asarray(samples, dtype=float)
        if samples_arr.ndim != 2:
            return cast(Any, getattr(self.model, 'labels', None))

        cached_labels = getattr(self.model, 'labels', None)
        if self._precomputed_traj is None and cached_labels is not None:
            embed_version = getattr(self.model, '_embed_version', None)
            labels_version = getattr(self.model, '_labels_version', None)
            cached_arr = np.asarray(cached_labels, dtype=int)
            if (
                embed_version is not None
                and labels_version == embed_version
                and cached_arr.shape[0] == samples_arr.shape[0]
            ):
                return cached_arr

        method = str(getattr(self, 'embed_cluster_method', 'dbscan')).lower()
        try:
            if method == 'graph_louvain':
                cluster_model = self._live_cluster_target_model()
                graph_louvain = getattr(cluster_model, '_graph_louvain_labels', None)
                if callable(graph_louvain):
                    labels = np.asarray(graph_louvain(), dtype=int)
                    setattr(cluster_model, 'labels', labels)
                    return labels
                method = 'louvain'
            if method == 'louvain':
                distance_threshold = getattr(self.model, 'significant_distance', None)
                strategy = create_clustering_strategy(
                    'louvain',
                    distance_factor=float(self.louvain_distance_factor),
                    distance_threshold=None if distance_threshold is None else float(distance_threshold),
                    knn_k=int(self.louvain_knn_k),
                    mutual_knn=bool(self.louvain_mutual_knn),
                )
            elif method == 'hdbscan':
                strategy = create_clustering_strategy(
                    'hdbscan',
                    min_cluster_size=int(self.dbscan_min_samples),
                    min_samples=2,
                )
            else:
                method = 'dbscan'
                strategy = create_clustering_strategy(
                    'dbscan',
                    eps=float(self.dbscan_eps),
                    min_samples=int(self.dbscan_min_samples),
                )
            cluster_dims = max(1, min(self.cluster_dimension, samples_arr.shape[1]))
            labels = strategy.fit_predict(samples_arr[:, :cluster_dims])
            setattr(self.model, 'labels', labels)
            return labels
        except Exception:
            return cast(Any, getattr(self.model, 'labels', None))

    def _on_frame(self, _frame: int) -> tuple[Artist, ...]:
        self._frame_index += 1
        if self._interaction_active:
            return (self.scatter, self.text_metrics)
        if self.running:
            self._simulate_chunk()
            self._update_metrics_and_plot(recompute_labels=bool(self._should_recluster()))
        should_refresh_force = self.running and (self.force_fig is not None) and (
            self._frame_index % max(1, int(self._force_monitor_refresh_every)) == 0
        )
        if should_refresh_force:
            self._update_force_monitor()
        return (self.scatter, self.text_metrics)

    def _colors_for_predicted_labels(self, labels: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
        return self._colors_for_labels(labels, singleton_gray=True)


def launch_live_from_result(
    result: dict[str, Any],
    *,
    dimension: int = 3,
    playback: bool = True,
    start_paused: bool = True,
    show: bool = True,
    readonly: bool = False,
    theme: str | None = None,
    accent_color: str | None = None,
    display_name: str | None = None,
) -> LiveTuner:
    """
    使用 run_single_test 的返回结果快速启动 Live GUI。

    Args:
        result: run_single_test(include_model=True) 的返回字典，需要包含 syncmap 与 graphwalk。
        dimension: Live 视图维度（2/3）。
        playback: 若可用则使用记录的 trajectory 进行回放。
        start_paused: 是否以暂停状态打开（方便拖动/观察）。
        show: 是否立即调用 plt.show()。
    """
    syncmap = result.get('syncmap')
    env = result.get('graphwalk')
    if syncmap is None or env is None:
        raise ValueError("Result缺少 syncmap / graphwalk，请在 run_single_test 时设置 include_model=True。")

    dataset = result.get('dataset')
    dataset_path = result.get('dataset_path')
    if dataset_path:
        dataset_path = str(dataset_path)
    elif dataset:
        dataset_path = dataset if str(dataset).startswith("datasets/") else f"datasets/{dataset}"
    else:
        dataset_path = DATASET
    strategy_params = result.get('strategy_params', {})
    if not isinstance(strategy_params, dict):
        strategy_params = {}

    traj = None
    steps = None
    if playback:
        if hasattr(syncmap, "get_trajectory"):
            try:
                traj = np.asarray(syncmap.get_trajectory(), dtype=float)
            except Exception as exc:
                print(f"[Warn] 无法读取轨迹，改为静态显示: {exc}")
        if traj is None and hasattr(syncmap, "syncmap"):
            try:
                traj = np.expand_dims(np.asarray(syncmap.syncmap, dtype=float), axis=0)
            except Exception:
                traj = None
        if hasattr(syncmap, "get_trajectory_steps"):
            try:
                steps = np.asarray(syncmap.get_trajectory_steps(), dtype=int)
            except Exception:
                steps = None
        if traj is not None and steps is not None and len(traj) != len(steps):
            steps = None

    tuner = LiveTuner(
        dataset_path,
        strategy_name=str(result.get('strategy') or 'rg'),
        strategy_params=strategy_params,
        dimension=dimension,
        display_name=display_name or result.get('resolved_variant') or None,
        preloaded_model=syncmap,
        preloaded_env=env,
        precomputed_trajectory=traj if playback else None,
        precomputed_steps=steps if playback else None,
        start_paused=start_paused,
        readonly=readonly,
    )
    if theme not in {None, ""} or accent_color not in {None, ""}:
        tuner.apply_visual_theme(theme=theme, accent_color=accent_color)
    if show:
        plt.show()
    return tuner


def launch_single_test_viewer(
    result: dict[str, Any],
    *,
    dimension: int = 3,
    playback: bool = True,
    start_paused: bool = True,
    show: bool = True,
    theme: str | None = None,
    accent_color: str | None = None,
    display_name: str | None = None,
) -> LiveTuner:
    """Open a read-only playback viewer for a finished single-test result."""
    return launch_live_from_result(
        result,
        dimension=dimension,
        playback=playback,
        start_paused=start_paused,
        show=show,
        readonly=True,
        theme=theme,
        accent_color=accent_color,
        display_name=display_name,
    )

class MultiStrategyViewer:
    """Grid viewer: N strategies on the same dataset, side-by-side live playback."""

    def __init__(
        self,
        dataset: str,
        strategies: list[dict[str, Any]],
        *,
        theme: str = "dark",
        accent_color: str = "#6970ac",
    ) -> None:
        import matplotlib.pyplot as plt

        self.dataset = dataset
        self.strategies = strategies
        self.n = len(strategies)
        self.theme = theme
        self.accent_color = accent_color

        # Compute grid layout: smallest rows × cols ≥ n
        cols = int(math.ceil(math.sqrt(self.n)))
        rows = int(math.ceil(self.n / cols))
        self.rows = rows
        self.cols = cols

        # Create the shared env (same dataset, same input sequence)
        self.env = None
        self.models: list[Any] = []
        self._setup_models()

        # Pre-generate shared input sequence
        self.total_steps = 5000
        self._phase_offset = 0
        self._gen_sequence()

        # Build figure
        self.fig, self.axes_grid = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
        # Normalise axes_grid to 2D array
        if rows == 1 and cols == 1:
            self.axes_grid = np.array([[self.axes_grid]])
        elif rows == 1:
            self.axes_grid = self.axes_grid.reshape(1, -1)
        elif cols == 1:
            self.axes_grid = self.axes_grid.reshape(-1, 1)

        # Hide unused axes
        for idx in range(self.n, rows * cols):
            r, c = divmod(idx, cols)
            self.axes_grid[r, c].set_visible(False)

        self._palette = resolve_plot_palette(theme=theme, accent=accent_color)
        self.fig.patch.set_facecolor(self._palette["fig_bg"])

        # Scatter artists per subplot
        self._scatters: list[PathCollection] = []
        self._init_scatters()

        # Control panel (right side)
        self._setup_control_panel()

        # Timer for playback
        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._step)
        self._paused = True
        self._timer.start()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self._timer.stop())

    def _setup_models(self) -> None:
        for info in self.strategies:
            np.random.seed(int(info.get("params", {}).get("seed", 0)))
            try:
                model, env = create_model(
                    dataset_path=self.dataset,
                    strategy_name=info["strategy_name"],
                    strategy_params=info.get("params", {}),
                    trajectory_interval=10000,
                    record_trajectory=False,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create model for '{info['display_name']}': {e}") from e
            if self.env is None:
                self.env = env
            self.models.append(model)

    def _gen_sequence(self) -> None:
        assert self.env is not None
        self.input_seq, _ = self.env.getSequence(self.total_steps)

    def _init_scatters(self) -> None:
        true_labels = self.env.trueLabel() if self.env is not None else None
        unique_labels = sorted(set(true_labels)) if true_labels is not None else []
        cmap = plt.cm.get_cmap("tab10", max(1, len(unique_labels)))

        for idx in range(self.n):
            r, c = divmod(idx, self.cols)
            ax = self.axes_grid[r, c]
            model = self.models[idx]
            name = self.strategies[idx]["display_name"].upper()

            pos = model.syncmap
            if pos.ndim != 2 or pos.shape[0] == 0:
                sc = ax.scatter([], [])
            else:
                colors = [cmap(int(l) % cmap.N) for l in true_labels] if true_labels is not None else None
                sc = ax.scatter(
                    pos[:, 0], pos[:, 1],
                    c=colors, s=30, alpha=0.85, edgecolors="none",
                )
            self._scatters.append(sc)

            ax.set_title(name, fontsize=10, color=self._palette["fg"])
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(self._palette["ax_bg"])
            for spine in ax.spines.values():
                spine.set_color(self._palette["inactive"])

    def _setup_control_panel(self) -> None:
        # Use the rightmost column area for controls
        self._panel_ax = self.fig.add_axes([0.84, 0.05, 0.14, 0.90])
        self._panel_ax.set_xticks([])
        self._panel_ax.set_yticks([])
        for spine in self._panel_ax.spines.values():
            spine.set_visible(False)
        self._panel_ax.set_facecolor(self._palette["panel_bg"])
        self._panel_ax.set_title("Controls", fontsize=10, color=self._palette["fg"], pad=6)

        # Shared step counter text
        self._info_text = self._panel_ax.text(
            0.5, 0.92, f"Step: 0 / {self.total_steps}",
            transform=self._panel_ax.transAxes,
            ha="center", va="top", fontsize=9,
            color=self._palette["fg"],
        )
        # Speed slider
        from matplotlib.widgets import Slider
        ax_speed = self.fig.add_axes([0.86, 0.80, 0.10, 0.03])
        self._speed_slider = Slider(ax_speed, "Speed", 1, 50, valinit=10, valstep=1)
        self._speed_slider.label.set_color(self._palette["fg"])

        # Pause / Play button
        from matplotlib.widgets import Button
        ax_btn = self.fig.add_axes([0.86, 0.70, 0.10, 0.04])
        self._pause_btn = Button(ax_btn, "Pause")
        self._pause_btn.label.set_fontsize(9)
        self._pause_btn.on_clicked(self._toggle_pause)

        # DBSCAN eps override
        ax_eps = self.fig.add_axes([0.86, 0.60, 0.10, 0.03])
        self._eps_slider = Slider(ax_eps, "EPS", 0.1, 10.0, valinit=4.5, valstep=0.1)
        self._eps_slider.label.set_color(self._palette["fg"])

    def _toggle_pause(self, _event: Any) -> None:
        self._paused = not self._paused
        self._pause_btn.label.set_text("Play" if self._paused else "Pause")
        self.fig.canvas.draw_idle()

    def _on_key(self, event: Any) -> None:
        if event.key == " ":
            self._toggle_pause(None)

    def _step(self) -> None:
        if self._paused:
            return

        chunk_size = int(self._speed_slider.val)
        end = min(self._phase_offset + chunk_size, self.total_steps)
        if end <= self._phase_offset:
            self._paused = True
            self._pause_btn.label.set_text("Play")
            return

        chunk = self.input_seq[self._phase_offset:end]

        # Run one chunk through each model
        for model in self.models:
            input_general = getattr(model, "inputGeneral", None)
            if callable(input_general):
                try:
                    input_general(chunk, phase_offset=self._phase_offset, reset_trajectory=False)
                except TypeError:
                    model.input(chunk)
            else:
                model.input(chunk)

        self._phase_offset = end

        # Update all scatters and re-cluster with shared eps
        eps = float(self._eps_slider.val)
        true_labels = self.env.trueLabel() if self.env is not None else None
        unique_labels = sorted(set(true_labels)) if true_labels is not None else []
        cmap = plt.cm.get_cmap("tab10", max(1, len(unique_labels)))

        for idx in range(self.n):
            model = self.models[idx]
            pos = model.syncmap

            # Quick DBSCAN
            try:
                c_labels = DBSCAN(eps=eps, min_samples=2).fit_predict(pos)
            except Exception:
                c_labels = np.zeros(pos.shape[0], dtype=int)

            colors = [cmap(int(l) % cmap.N) for l in c_labels]
            self._scatters[idx].set_offsets(pos[:, :2] if pos.shape[1] >= 2 else np.column_stack([pos[:, 0], np.zeros(pos.shape[0])]))
            self._scatters[idx].set_color(colors)

        self._info_text.set_text(f"Step: {end} / {self.total_steps}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    raise SystemExit("live.py is backend-only. Use `python main.py --task 1 ...` or `python main.py --task 0 --live true ...`.")
