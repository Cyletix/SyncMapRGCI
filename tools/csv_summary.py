#!/usr/bin/env python3
"""Summarize SyncMapRGCI comparison CSV files as mean +/- std tables.

The formal metric dimension is one of nmi or ari. The default summary metric
is ari because it directly scores exact partition agreement and is robust to
singleton inflation. NMI remains a diagnostic; it can look acceptable when
fine labels are merged into coarse groups. Evaluation mode, readout method,
and output level are separate grouping dimensions so incompatible hierarchy
score contracts are not mixed accidentally.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


METRIC_COLUMNS = {
    "nmi": "nmi_score",
    "ari": "ari_score",
}


def _float_or_nan(value: str | None) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan"), float("nan")
    mean = sum(xs) / len(xs)
    variance = sum((v - mean) ** 2 for v in xs) / len(xs)
    return mean, math.sqrt(variance)


def _fmt(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(std):
        return f"{mean:.4f}+/-nan"
    return f"{mean:.4f}+/-{std:.4f}"


def _display_name(value: str) -> str:
    text = str(value)
    return "syncmap" if text.strip().lower() == "strict" else text


def _group_key(row: dict[str, str], *, include_profile: bool, include_readout: bool) -> tuple[str, ...]:
    key = [
        row.get("dataset", ""),
        row.get("sequence_length", ""),
        _display_name(
            row.get("engine_strategy") or row.get("runtime_strategy") or row.get("strategy", "")
        ),
    ]
    if include_profile:
        key.append(_display_name(
            row.get("profile_name")
            or row.get("comparison_name")
            or row.get("method", "").lower()
        ))
    if include_readout:
        key.extend(
            [
                row.get("evaluation_mode", ""),
                row.get("eval_readout_method", ""),
                row.get("evaluation_output_key", ""),
            ]
        )
    return tuple(key)


def summarize_csv(
    csv_path: str,
    *,
    metric: str = "ari",
    compact: bool = False,
    include_profile: bool = True,
    include_readout: bool = True,
) -> int:
    metric_key = METRIC_COLUMNS[metric]
    path = Path(csv_path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"ERROR: CSV is empty: {path}", file=sys.stderr)
        return 1
    if metric_key not in rows[0]:
        print(f"ERROR: metric column missing: {metric_key}", file=sys.stderr)
        return 1

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, include_profile=include_profile, include_readout=include_readout)].append(row)

    expected_trials = max(
        (int(float(row.get("trial", "0"))) for row in rows if row.get("trial", "")),
        default=0,
    )

    print(f"Summary: {path}")
    print(f"  metric={metric} rows={len(rows)} groups={len(groups)}")
    engines = sorted(
        {
            _display_name(
                row.get("engine_strategy")
                or row.get("runtime_strategy")
                or row.get("strategy", "")
            )
            for row in rows
        }
    )
    print(f"  engines={engines}")
    print()

    for dataset in sorted({key[0] for key in groups}):
        dataset_rows = [row for row in rows if row.get("dataset", "") == dataset]
        nodes = dataset_rows[0].get("output_size", "?") if dataset_rows else "?"
        print(f"{dataset} (nodes={nodes})")
        dataset_keys = sorted(
            [key for key in groups if key[0] == dataset],
            key=lambda key: (
                int(key[1]) if str(key[1]).isdigit() else str(key[1]),
                key[2:],
            ),
        )
        for key in dataset_keys:
            values = groups[key]
            dataset_name, seq_len, engine = key[:3]
            del dataset_name
            profile = key[3] if include_profile else ""
            readout_offset = 4 if include_profile else 3
            if include_readout:
                evaluation_mode, readout_method, output_key = key[readout_offset:readout_offset + 3]
            else:
                evaluation_mode, readout_method, output_key = "", "", ""

            scores = [_float_or_nan(row.get(metric_key)) for row in values]
            aris = [_float_or_nan(row.get("ari_score")) for row in values]
            nmis = [_float_or_nan(row.get("nmi_score")) for row in values]
            times = [_float_or_nan(row.get("elapsed_time")) for row in values]
            clusters = [_float_or_nan(row.get("cluster_count")) for row in values]
            level_columns = sorted(
                {
                    column
                    for row in values
                    for column in row
                    if re.fullmatch(r"cluster_level\d+_ari_score", column)
                },
                key=lambda column: int(re.findall(r"\d+", column)[0]),
            )
            level_stats = [
                (
                    int(re.findall(r"\d+", column)[0]),
                    *_mean_std([_float_or_nan(row.get(column)) for row in values]),
                )
                for column in level_columns
            ]
            score_mu, score_std = _mean_std(scores)
            ari_mu, ari_std = _mean_std(aris)
            nmi_mu, nmi_std = _mean_std(nmis)
            time_mu, _ = _mean_std(times)
            cluster_mu, _ = _mean_std(clusters)
            collapsed = sum(1 for value in scores if math.isfinite(value) and abs(value) < 1e-12)
            expected = expected_trials or len(values)

            label_parts = [f"seq={seq_len:>7s}", f"{engine:<8s}", f"n={len(values)}/{expected}"]
            if profile and profile != engine:
                label_parts.append(f"profile={profile}")
            if include_readout and evaluation_mode:
                label_parts.append(f"evaluation={evaluation_mode}")
            if include_readout and (readout_method or output_key):
                label_parts.append(f"readout={readout_method}:{output_key}")
            line = "  " + "  ".join(label_parts)
            line += f"  {metric}={_fmt(score_mu, score_std)}"
            if not compact:
                if metric != "nmi":
                    line += f"  nmi={_fmt(nmi_mu, nmi_std)}"
                if metric != "ari":
                    line += f"  ari={_fmt(ari_mu, ari_std)}"
                for level_number, level_mu, level_std in level_stats:
                    if math.isfinite(level_mu):
                        line += f"  L{level_number}={_fmt(level_mu, level_std)}"
                line += f"  k={cluster_mu:.1f}" if math.isfinite(cluster_mu) else "  k=nan"
                if collapsed:
                    line += f"  COLLAPSED={collapsed}/{len(values)}"
                line += f"  time={time_mu:.1f}s" if math.isfinite(time_mu) else "  time=nan"
            print(line)
        print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to comparison_partial_results.csv or comparison_full_results.csv")
    parser.add_argument("--metric", default="ari", choices=sorted(METRIC_COLUMNS))
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--no-profile", action="store_true", help="Do not split groups by profile_name")
    parser.add_argument("--no-readout", action="store_true", help="Do not split groups by evaluation mode/readout/output")
    args = parser.parse_args()
    raise SystemExit(
        summarize_csv(
            args.csv,
            metric=str(args.metric),
            compact=bool(args.compact),
            include_profile=not bool(args.no_profile),
            include_readout=not bool(args.no_readout),
        )
    )


if __name__ == "__main__":
    main()
