#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.result_paths import classify_output_token, trace_plot_path_from_json
from tools.trace_rg_hierarchy_scores import save_score_plot


def render_one(
    json_path: Path,
    plot_out: str | None,
    *,
    metric: str = "nmi",
    theme: str = "light",
    accent_color: str = "#6970ac",
) -> Path:
    record: Any = json.loads(json_path.read_text(encoding="utf-8"))
    plot_mode = classify_output_token(plot_out)
    if plot_mode == "disabled":
        return None
    target = Path(str(plot_out)).expanduser() if plot_mode == "path" else trace_plot_path_from_json(json_path)
    if isinstance(record, list):
        from unified_experiments import _save_learning_curve_plot

        return _save_learning_curve_plot(
            record,
            target,
            metric=metric,
            theme=theme,
            accent_color=accent_color,
            title=f"Learning Curve ({str(metric).upper()})",
        )
    if not isinstance(record, dict):
        raise ValueError(
            "Trace JSON must be either a task-0 learning-curve list or a "
            "task-4 trace record."
        )
    return save_score_plot(record, target, metric=metric, theme=theme, accent_color=accent_color)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate the standard RG NMI plot from an existing score-trace JSON."
    )
    parser.add_argument("json_paths", nargs="+", help="One or more rg_*_score_*.json files")
    parser.add_argument(
        "--plot-out",
        default="auto",
        help="Output PNG path. Only valid when rendering a single JSON. Default: results/picture/<trace-picture-name>.png.",
    )
    parser.add_argument("--plot-metric", default="nmi", choices=["nmi", "ari"], help="Metric to render from the trace JSON.")
    parser.add_argument("--theme", default="light", help="Plot theme: light or dark.")
    parser.add_argument("--accent-color", default="#6970ac", help="Accent color shared with the live plot theme.")
    args = parser.parse_args()

    json_paths = [Path(p) for p in args.json_paths]
    if len(json_paths) > 1 and classify_output_token(args.plot_out) == "path":
        raise SystemExit("--plot-out can only be used with a single JSON input")

    for json_path in json_paths:
        saved = render_one(
            json_path,
            args.plot_out if len(json_paths) == 1 else None,
            metric=str(args.plot_metric),
            theme=str(args.theme),
            accent_color=str(args.accent_color),
        )
        print(f"saved_plot: {saved if saved is not None else 'false'}")


if __name__ == "__main__":
    main()
