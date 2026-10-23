#!/usr/bin/env python3
"""
Visualization helpers for SyncMap / GraphWalk outputs.
Author: Cyletix
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
import os
import warnings

import matplotlib

if not os.environ.get("MPLBACKEND"):  # An explicit MPLBACKEND (e.g. Agg on headless machines) wins.
    try:
        matplotlib.use("Qt5Agg")  # Prefer the interactive backend when a Qt session exists.
    except Exception:
        matplotlib.use("Agg")  # Fallback so headless GIF/static export still works.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import os
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.widgets import Slider
from numpy.typing import NDArray

from tools.plot_theme import resolve_plot_palette
from tools.result_paths import artifact_result_path, build_run_stem

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
AnyArray = NDArray[np.generic]


def _ensure_numpy_array(data: Any) -> AnyArray:
    """Convert incoming data to a numpy array without copying if possible."""
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def _extract_graphwalk_sequence(
    graphwalk: Any,
    fallback_length: int = 200,
) -> tuple[FloatArray, IntArray]:
    """Fetch the cached GraphWalk sequence or trigger a short regeneration."""
    seq = getattr(graphwalk, "input_sequence", None)
    cls = getattr(graphwalk, "input_class", None)

    if seq is None or not hasattr(seq, "size") or int(getattr(seq, "size", 0)) == 0:
        if not hasattr(graphwalk, "getSequence"):
            raise ValueError("GraphWalk instance does not provide an input sequence.")
        seq, cls = graphwalk.getSequence(fallback_length)

    seq_arr = _ensure_numpy_array(seq)
    if seq_arr.ndim != 2:
        raise ValueError("input_sequence must be a 2D array shaped (T, N).")

    if cls is None:
        cls_arr = np.zeros(seq_arr.shape[0], dtype=int)
    else:
        cls_arr = _ensure_numpy_array(cls).astype(int, copy=False)

    return cast(FloatArray, seq_arr), cast(IntArray, cls_arr)


def _extract_graphwalk_labels(graphwalk: Any | None) -> IntArray | None:
    """Return the graphwalk true labels if available."""
    if graphwalk is None:
        return None
    if not hasattr(graphwalk, "trueLabel"):
        return None
    labels = graphwalk.trueLabel()
    if labels is None:
        return None
    return cast(IntArray, _ensure_numpy_array(labels).astype(int, copy=False))


def _label_color_lookup(labels: Any) -> dict[int, tuple[float, float, float, float]]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    valid_labels = sorted({int(label) for label in labels_arr.tolist() if int(label) >= 0})
    cmap = plt.get_cmap("tab20")
    palette_size = max(int(getattr(cmap, "N", 20)), 1)
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


def _label_name_map(labels: Any) -> dict[int, str]:
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


def _colors_for_labels(labels: Any, *, singleton_gray: bool = True) -> NDArray[np.float64]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    colors = np.zeros((labels_arr.shape[0], 4), dtype=float)
    colors[:, :] = (0.65, 0.65, 0.65, 1.0)
    if labels_arr.size == 0:
        return colors
    uniq_nonneg = sorted({int(v) for v in labels_arr.tolist() if int(v) >= 0})
    if singleton_gray:
        counts = {label: int(np.sum(labels_arr == label)) for label in uniq_nonneg}
        active_labels = [label for label in uniq_nonneg if counts[label] > 1]
    else:
        active_labels = uniq_nonneg
    color_lookup = _label_color_lookup(active_labels)
    for idx, label in enumerate(labels_arr.tolist()):
        if int(label) < 0:
            continue
        colors[idx, :] = color_lookup.get(int(label), (0.65, 0.65, 0.65, 1.0))
    return colors


def _build_discrete_label_legend(
    ax: Axes,
    labels: Any,
    *,
    fg_color: str = "#000000",
) -> Any | None:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return None
    uniq = np.unique(labels_arr)
    if uniq.size == 0:
        return None
    color_lookup = _label_color_lookup(labels_arr)
    label_name_map = _label_name_map(labels_arr)
    n_classes = int(uniq.size)
    if n_classes <= 12:
        marker_size, font_size, n_col = 6, 8, 1
    elif n_classes <= 24:
        marker_size, font_size, n_col = 5, 7, 2
    else:
        marker_size, font_size, n_col = 4, 6.5, 2
    handles = []
    for label_idx in uniq.tolist():
        color = color_lookup.get(int(label_idx), (0.65, 0.65, 0.65, 1.0))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=marker_size,
                label=label_name_map.get(int(label_idx), f"class {int(label_idx)}"),
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
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
        for text in legend.get_texts():
            text.set_color(fg_color)
    except Exception:
        pass
    return legend


def _format_syncmap_method_label(candidate: str, default: str = "SyncMap") -> str:
    """Normalize various method/variant names to a concise label for titles/files.

    Examples:
    - "SyncMapX_Nbody" -> "SyncMap_Nbody"
    - "SyncMapX" -> "SyncMap"
    - "package.SyncMap" -> "SyncMap"
    - "Nbody" -> "Nbody" (caller may prepend if desired)
    """
    s = str(candidate).strip()
    if not s:
        return default
    if "." in s:
        s = s.split(".")[-1]
    if s.startswith("SyncMapX_"):
        s = "SyncMap_" + s[len("SyncMapX_"):]
    elif s == "SyncMapX":
        s = "SyncMap"
    elif s.startswith("SyncMapX"):
        s = s.replace("SyncMapX", "SyncMap", 1)
    return s


def resolve_syncmap_method_name(syncmap: Any, default: str = "SyncMap") -> str:
    """Best-effort extraction of the SyncMap variant/method name.

    Priority:
    1) Common name-like attributes on the instance
    2) Class's module or class name
    Fallback to `default` if nothing usable is found.
    """
    for attr in ("method_name", "name", "variant_name", "variant", "algo_name"):
        try:
            if hasattr(syncmap, attr):
                val = getattr(syncmap, attr)
                if isinstance(val, str) and val.strip():
                    return _format_syncmap_method_label(val, default=default)
        except Exception:
            pass

    try:
        cls = type(syncmap)
        module_name = getattr(cls, "__module__", "") or ""
        label = module_name
        if label in ("__main__", "builtins", ""):
            label = getattr(cls, "__name__", "") or default
        return _format_syncmap_method_label(label, default=default)
    except Exception:
        return default


def _extract_syncmap_positions(
    syncmap: Any,
) -> tuple[FloatArray, IntArray | None, Literal["dynamic", "static"]]:
    """Return positions, optional steps, and mode for downstream plotting."""
    trajectory: FloatArray | None = None
    steps: IntArray | None = None

    if hasattr(syncmap, "get_trajectory"):
        try:
            raw_traj = syncmap.get_trajectory()
        except Exception:
            raw_traj = None
        if raw_traj is not None:
            trajectory_arr = _ensure_numpy_array(raw_traj).astype(float, copy=False)
            if trajectory_arr.ndim == 3:
                trajectory = cast(FloatArray, trajectory_arr)
                print(f"[Info] Using dynamic trajectory: shape={trajectory.shape}")
                try:
                    raw_steps = syncmap.get_trajectory_steps()
                except Exception:
                    raw_steps = None
                if raw_steps is not None:
                    steps_arr = _ensure_numpy_array(raw_steps).astype(int, copy=False)
                    steps = cast(IntArray, steps_arr)
                return trajectory, steps, "dynamic"

    for attr in ("positions", "w", "syncmap", "synapses_matrix"):
        candidate = getattr(syncmap, attr, None)
        if candidate is None:
            continue
        arr = _ensure_numpy_array(candidate).astype(float, copy=False)
        if arr.ndim == 2:
            print(f"[Info] Using static matrix: {attr}, shape={arr.shape}")
            return cast(FloatArray, arr), None, "static"

    raise ValueError("Unable to extract positional data from SyncMap instance.")


def default_anim_path(
    dataset: str,
    strategy: str,
    base_dir: str | Path = "results/picture",
    ext: str = ".gif",
    *,
    method_name: str = "SyncMap",
    step_count: int | None = None,
) -> Path:
    """Return canonical GIF path under results/picture with full timestamp precision."""
    base_path = Path(base_dir)
    if base_path == Path("results/picture"):
        return artifact_result_path(
            strategy,
            dataset,
            "animation",
            ext,
            step_count=step_count,
            kind="picture",
        )
    base_path.mkdir(parents=True, exist_ok=True)
    stem = build_run_stem(strategy, dataset, "animation", step_count=step_count)
    candidate = base_path / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        alt = base_path / f"{stem}_{counter:02d}{ext}"
        if not alt.exists():
            return alt
        counter += 1


def plot_input_sequence(
    input_sequence: FloatArray,
    input_class: IntArray,
    ax: Axes | None = None,
    max_nodes: int | None = None,
    *,
    show: bool = True,
) -> Axes:
    """Plot the input signal as a heatmap over time."""
    _ = input_class  # Reserved for potential future use
    if ax is None:
        fig, actual_ax = plt.subplots(figsize=(7, 4))
    else:
        fig_from_ax = ax.figure
        if not isinstance(fig_from_ax, Figure):
            raise RuntimeError("Provided Axes is not associated with a Figure.")
        fig = fig_from_ax
        actual_ax = ax
    if max_nodes is not None:
        seq_to_show = input_sequence[:, :max_nodes]
    else:
        seq_to_show = input_sequence
    img = actual_ax.imshow(seq_to_show.T, aspect="auto", cmap="viridis", interpolation="nearest")
    actual_ax.set_title("Input Sequence (GraphWalk Activation)")
    actual_ax.set_xlabel("Time step")
    actual_ax.set_ylabel("Node index")
    fig.colorbar(img, ax=actual_ax, label="Activation")
    fig.tight_layout()
    fig.canvas.draw_idle()

    if show:
        plt.show()

    return actual_ax


def plot_syncmap_embedding(
    positions: FloatArray,
    labels: IntArray | None = None,
    ax: Axes | None = None,
    *,
    method_name: str = "SyncMap",
) -> Axes:
    """Plot the SyncMap embedding in space."""
    if ax is None:
        _, actual_ax = plt.subplots(figsize=(6, 6))
    else:
        actual_ax = ax
    actual_ax.set_title(f"{method_name} Embedding")
    if positions.shape[1] >= 2:
        x, y = positions[:, 0], positions[:, 1]
        scatter_colors = _colors_for_labels(labels, singleton_gray=False) if labels is not None else "gray"
        sc = actual_ax.scatter(x, y, c=scatter_colors, s=20)
        if labels is not None:
            _build_discrete_label_legend(actual_ax, labels)
        actual_ax.set_xlabel("dim 1")
        actual_ax.set_ylabel("dim 2")
    else:
        actual_ax.plot(positions[:, 0], "o")
        actual_ax.set_ylabel("1D position")
    return actual_ax


def plot_syncmap_over_time(
    positions_over_time: FloatArray,
    labels: IntArray | None,
    steps: IntArray | None = None,
    *,
    show: bool = True,
    method_name: str = "SyncMap",
    theme: str = "dark",
    accent_color: str = "#6970ac",
    title_suffix: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot SyncMap positions in 2D with an optional time slider.
    If positions_over_time is (T, N, D), enable the slider to step through time.
    If it is (N, D), show the final static layout.
    """
    steps_arr: IntArray | None = None
    if positions_over_time.ndim == 3:
        frame_count, node_count, dim_count = positions_over_time.shape
        if dim_count < 2:
            raise ValueError("positions_over_time needs at least two dimensions for a scatter plot.")
        data_to_show = positions_over_time[-1, :, :2]
        print(f"[Info] Detected dynamic trajectory with {frame_count} time steps, {node_count} nodes, {dim_count} dimensions.")
        all_positions = positions_over_time[:, :, :2]
        if steps is not None:
            steps_arr = np.asarray(steps, dtype=int)
            if len(steps_arr) != frame_count:
                warnings.warn("steps length does not match frame count; ignoring steps.", UserWarning, stacklevel=2)
                steps_arr = None
    elif positions_over_time.ndim == 2:
        node_count, dim_count = positions_over_time.shape
        if dim_count < 2:
            raise ValueError("positions_over_time needs at least two dimensions for a scatter plot.")
        frame_count = 1
        data_to_show = positions_over_time[:, :2]
        all_positions = positions_over_time[np.newaxis, :, :2]
        print(f"[Info] Detected static positions with {node_count} nodes, {dim_count} dimensions.")
        if steps is not None:
            steps_arr = np.asarray(steps, dtype=int).reshape(-1)
            if len(steps_arr) != 1:
                warnings.warn("static positions accept at most one step value; ignoring steps.", UserWarning, stacklevel=2)
                steps_arr = None
    else:
        raise ValueError("positions_over_time must be shaped (N, D) or (T, N, D).")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    palette = resolve_plot_palette(theme=theme, accent=accent_color)

    x_min = float(np.min(all_positions[:, :, 0]))
    x_max = float(np.max(all_positions[:, :, 0]))
    y_min = float(np.min(all_positions[:, :, 1]))
    y_max = float(np.max(all_positions[:, :, 1]))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.patch.set_facecolor(palette["fig_bg"])
    ax.set_facecolor(palette["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(palette["fg"])
    ax.tick_params(colors=palette["fg"])

    scatter_colors = _colors_for_labels(labels, singleton_gray=False) if labels is not None else "gray"
    sc = ax.scatter(data_to_show[:, 0], data_to_show[:, 1], c=scatter_colors, s=12.5)

    title_text = f"{method_name} Embedding"
    if title_suffix:
        title_text = f"{title_text} — {title_suffix}"
    if steps_arr is not None and len(steps_arr) > 0:
        initial_step = int(steps_arr[-1])
        ax.set_title(f"{title_text} (t={initial_step})")
    elif frame_count > 1:
        ax.set_title(f"{title_text} (t={frame_count - 1})")
    else:
        ax.set_title(title_text)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.title.set_color(palette["fg"])
    ax.xaxis.label.set_color(palette["fg"])
    ax.yaxis.label.set_color(palette["fg"])
    ax.grid(True, color=palette["grid"], alpha=0.22, linewidth=0.8)

    if labels is not None:
        _build_discrete_label_legend(ax, labels, fg_color=palette["fg"])

    if frame_count > 1:
        slider_ax = fig.add_axes((0.2, 0.02, 0.6, 0.04))
        slider_ax.set_facecolor(palette["panel_bg"])
        slider = Slider(slider_ax, "t", 0, frame_count - 1, valinit=frame_count - 1, valstep=1)
        slider.label.set_color(palette["slider_label"])
        slider.valtext.set_color(palette["fg"])
        if hasattr(slider, "track"):
            slider.track.set_facecolor(palette["slider_rest"])
        if hasattr(slider, "poly"):
            slider.poly.set_facecolor(palette["accent"])

        def update(_value: float) -> None:
            idx = int(round(slider.val))
            idx = max(0, min(frame_count - 1, idx))
            sc.set_offsets(all_positions[idx])
            real_t = int(steps_arr[idx]) if steps_arr is not None and len(steps_arr) > 0 else idx
            ax.set_title(f"{title_text} (t={real_t})")
            slider.valtext.set_text(str(real_t))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        last_t = int(steps_arr[-1]) if steps_arr is not None and len(steps_arr) > 0 else (frame_count - 1)
        ax.set_title(f"{title_text} (t={last_t})")
        fig._syncmap_slider = slider  # type: ignore[attr-defined]

        def on_key(event: Event) -> None:
            key = getattr(event, "key", None)
            if key == "left":
                slider.set_val(max(slider.val - 1, slider.valmin))
            elif key == "right":
                slider.set_val(min(slider.val + 1, slider.valmax))

        fig.canvas.mpl_connect("key_press_event", on_key)
        update(slider.val)

    if show:
        plt.show()

    return fig, ax


def plot_combined(
    input_sequence: FloatArray,
    input_class: IntArray,
    positions_over_time: FloatArray,
    labels: IntArray | None,
) -> None:
    """Deprecated wrapper; use the individual plotting functions instead."""
    warnings.warn(
        "plot_combined is deprecated. Call plot_input_sequence and plot_syncmap_over_time separately.",
        DeprecationWarning,
        stacklevel=2,
    )
    plot_input_sequence(input_sequence, input_class)
    plot_syncmap_over_time(positions_over_time, labels)


def animate_trajectory(
    positions_over_time: FloatArray,
    labels: IntArray | None = None,
    steps: IntArray | None = None,
    interval: int = 100,
    save_path: str | Path | None = None,
    show: bool = True,
    *,
    save_after_preview: bool = False,
    method_name: str = "SyncMap",
    theme: str = "light",
    accent_color: str = "#6970ac",
) -> None:
    """
    Play or save an animation of SyncMap node trajectories.
    positions_over_time: np.ndarray (T, N, D)
    labels: np.ndarray (N,)
    interval: frame interval in milliseconds
    save_path: optional path to store an mp4/gif; otherwise preview only
    save_after_preview: if True, save after the preview window closes
    """
    frame_count, _, dimension_count = positions_over_time.shape
    if dimension_count < 2:
        raise ValueError("positions_over_time must be at least two-dimensional.")

    if steps is not None:
        steps_array = np.asarray(steps, dtype=int)
        if len(steps_array) != frame_count:
            warnings.warn("steps length does not match frame count; ignoring steps.", UserWarning, stacklevel=2)
            steps = None
        else:
            steps = steps_array

    if save_after_preview and not show and save_path is not None:
        warnings.warn("save_after_preview=True requires show=True; saving immediately instead.", UserWarning, stacklevel=2)
        save_after_preview = False

    def create_animation() -> tuple[Figure, FuncAnimation]:
        fig, ax = plt.subplots(figsize=(6, 6))
        palette = resolve_plot_palette(theme=theme, accent=accent_color)

        x_min = float(np.min(positions_over_time[:, :, 0]))
        x_max = float(np.max(positions_over_time[:, :, 0]))
        y_min = float(np.min(positions_over_time[:, :, 1]))
        y_max = float(np.max(positions_over_time[:, :, 1]))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        else:
            margin_x = 0.05 * (x_max - x_min)
            x_min -= margin_x
            x_max += margin_x
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        else:
            margin_y = 0.05 * (y_max - y_min)
            y_min -= margin_y
            y_max += margin_y

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.patch.set_facecolor(palette["fig_bg"])
        ax.set_facecolor(palette["ax_bg"])
        for spine in ax.spines.values():
            spine.set_color(palette["fg"])
        ax.tick_params(colors=palette["fg"])
        init_t = int(steps[0]) if steps is not None else 0
        ax.set_title(f"{method_name} Node Evolution (t={init_t})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.title.set_color(palette["fg"])
        ax.xaxis.label.set_color(palette["fg"])
        ax.yaxis.label.set_color(palette["fg"])
        ax.grid(True, color=palette["grid"], alpha=0.22, linewidth=0.8)

        scatter_colors = _colors_for_labels(labels, singleton_gray=False) if labels is not None else "gray"
        sc = ax.scatter(positions_over_time[0, :, 0], positions_over_time[0, :, 1], c=scatter_colors, s=10)

        if labels is not None:
            _build_discrete_label_legend(ax, labels, fg_color=palette["fg"])

        def update(frame: int) -> tuple[PathCollection]:
            sc.set_offsets(positions_over_time[frame, :, :2])
            real_t = int(steps[frame]) if steps is not None else frame
            ax.set_title(f"{method_name} Node Evolution (t={real_t})")
            return (sc,)

        ani = FuncAnimation(fig, update, frames=frame_count, interval=interval, blit=False, repeat=True)
        return fig, ani

    def save_animation(ani_obj: FuncAnimation, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(int(round(1000 / interval)), 1)
        suffix = target_path.suffix.lower()

        last_percent = -1

        def progress_callback(frame_number: int, total: int | None) -> None:
            nonlocal last_percent
            total_frames = int(total) if total is not None and total > 0 else frame_count
            current = min(frame_number + 1, total_frames)
            percent = int(round(current * 100 / max(total_frames, 1)))
            if percent != last_percent or current == total_frames:
                print(f"\rSaving animation: {current}/{total_frames} frames ({percent}%)", end="", flush=True)
                last_percent = percent
            if current >= total_frames:
                print("", flush=True)

        if suffix == ".gif":
            writer = PillowWriter(fps=fps)
            ani_obj.save(target_path, writer=writer, progress_callback=progress_callback)
        else:
            ffmpeg_path = os.environ.get("FFMPEG_PATH")
            if ffmpeg_path:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            try:
                from matplotlib.animation import FFMpegWriter

                writer = FFMpegWriter(fps=fps)
                ani_obj.save(target_path, writer=writer, progress_callback=progress_callback)
            except Exception as e:
                print("Could not save mp4 with FFMpegWriter. Install ffmpeg, set FFMPEG_PATH, or save as .gif.")
                raise e

        print(f"Animation saved to {target_path}")

    fig, ani = create_animation()

    if show:
        plt.show()
        plt.close(fig)

    if save_path is not None:
        save_path = Path(save_path)
        if save_after_preview and show:
            # Rebuild fresh figure/animation for saving to avoid closed figure issues.
            fig, ani = create_animation()
        save_animation(ani, save_path)
        plt.close(fig)
    elif not show:
        plt.close(fig)


def visualize_input_sequence(
    graphwalk: Any,
    *,
    max_nodes: int | None = None,
    fallback_length: int = 200,
    show: bool = True,
) -> Axes:
    """Convenience wrapper that pulls data from GraphWalk and plots the input sequence."""
    seq, cls = _extract_graphwalk_sequence(graphwalk, fallback_length=fallback_length)
    return plot_input_sequence(seq, cls, max_nodes=max_nodes, show=show)


def visualize_syncmap_evolution(
    syncmap: Any,
    *,
    graphwalk: Any | None = None,
    labels: IntArray | None = None,
    show: bool = True,
) -> tuple[Figure, Axes] | None:
    """Plot the latest SyncMap positions, automatically handling trajectory extraction."""
    resolved_labels = labels if labels is not None else _extract_graphwalk_labels(graphwalk)
    try:
        positions, steps, _mode = _extract_syncmap_positions(syncmap)
    except ValueError as exc:
        print(f"⚠️ 无法绘制 SyncMap 位置: {exc}")
        return None

    try:
        method_name = resolve_syncmap_method_name(syncmap)
        fig_ax = plot_syncmap_over_time(positions, resolved_labels, steps=steps, show=show, method_name=method_name)
    except Exception as exc:
        print(f"⚠️ 绘制 SyncMap 轨迹时出错: {exc}")
        return None
    return fig_ax


def visualize_trajectory_animation(
    syncmap: Any,
    *,
    graphwalk: Any | None = None,
    labels: IntArray | None = None,
    interval: int = 100,
    save_path: str | Path | None = None,
    dataset: str | None = None,
    strategy: str | None = None,
    total_steps: int | None = None,
    show: bool = True,
    save_after_preview: bool = False,
    theme: str = "light",
    accent_color: str = "#6970ac",
) -> bool:
    """Animate the SyncMap trajectory with sensible defaults and fallbacks."""
    resolved_labels = labels if labels is not None else _extract_graphwalk_labels(graphwalk)

    try:
        positions, steps, mode = _extract_syncmap_positions(syncmap)
    except ValueError as exc:
        print(f"⚠️ 无法提取轨迹用于动画: {exc}")
        return False

    if mode != "dynamic":
        print("⚠️ 当前 SyncMap 没有可用于动画的时间序列数据。")
        return False

    if save_path is None and dataset is not None and strategy is not None:
        final_step = int(total_steps) if total_steps is not None else None
        if final_step is None and steps is not None and len(steps) > 0:
            final_step = int(steps[-1])
        save_path = default_anim_path(
            dataset,
            strategy,
            method_name=resolve_syncmap_method_name(syncmap),
            step_count=final_step,
        )

    try:
        animate_trajectory(
            positions,
            labels=resolved_labels,
            steps=steps,
            interval=interval,
            save_path=save_path,
            show=show,
            save_after_preview=save_after_preview,
            method_name=resolve_syncmap_method_name(syncmap),
            theme=theme,
            accent_color=accent_color,
        )
    except Exception as exc:
        print(f"⚠️ 动画播放失败: {exc}")
        return False
    return True
