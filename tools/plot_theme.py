from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def resolve_plot_palette(theme: str = "light", accent: str = "#6970ac") -> dict[str, Any]:
    normalized = str(theme or "light").strip().lower()
    if normalized not in {"light", "dark"}:
        normalized = "light"

    if normalized == "dark":
        palette = {
            "theme": "dark",
            "fig_bg": "#0d0f14",
            "ax_bg": "#11131c",
            "panel_bg": "#0f111a",
            "fg": "#f2f2f2",
            "inactive": "#2f3a56",
            "slider_rest": "#1f2330",
            "slider_label": "#93c5fd",
            "btn_bg": "#1c2233",
            "btn_edge": "#2f3a56",
            "btn_hover": "#29314a",
            "line": "#b0b5c7",
            "grid": "#2f3a56",
            "series_l1": "#8ab4ff",
            "series_l2": accent,
            "series_l3": "#f4b860",
        }
    else:
        palette = {
            "theme": "light",
            "fig_bg": "#ffffff",
            "ax_bg": "#ffffff",
            "panel_bg": "#f5f5f5",
            "fg": "#101010",
            "inactive": "#a0b4d0",
            "slider_rest": "#d4d7e0",
            "slider_label": "#1e3a8a",
            "btn_bg": "#A0B4D0",
            "btn_edge": "#a0b4d0",
            "btn_hover": "#dce7ff",
            "line": "#6b7280",
            "grid": "#d4d7e0",
            "series_l1": "#2e6fd8",
            "series_l2": accent,
            "series_l3": "#d97706",
        }

    palette["accent"] = accent
    return palette


def apply_basic_plot_theme(
    fig: Figure,
    ax: Axes,
    *,
    theme: str = "light",
    accent: str = "#6970ac",
    grid_alpha: float = 0.25,
) -> dict[str, Any]:
    palette = resolve_plot_palette(theme=theme, accent=accent)
    fig.patch.set_facecolor(palette["fig_bg"])
    ax.set_facecolor(palette["ax_bg"])
    for spine in ax.spines.values():
        spine.set_color(palette["fg"])
    ax.tick_params(colors=palette["fg"])
    ax.xaxis.label.set_color(palette["fg"])
    ax.yaxis.label.set_color(palette["fg"])
    ax.title.set_color(palette["fg"])
    ax.grid(True, color=palette["grid"], alpha=float(grid_alpha), linewidth=0.8)
    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(palette["panel_bg"])
        frame.set_edgecolor(palette["inactive"])
        frame.set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(palette["fg"])
    return palette
