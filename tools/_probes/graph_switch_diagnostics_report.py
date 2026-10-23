#!/usr/bin/env python3
"""Summarize and plot graph-switch plasticity probe outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NSGD_ARMS = [
    "nsgd",
    "nsgd_unfreeze_at_switch",
    "nsgd_no_freeze",
    "nsgd_mem300",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def build_report(probe_dir: Path) -> None:
    payload = json.loads((probe_dir / "results.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(probe_dir / "summary.csv")
    curves = pd.read_csv(probe_dir / "curves.csv")
    fairness = pd.read_csv(probe_dir / "fairness_audit.csv")
    freeze_effect = pd.read_csv(probe_dir / "nsgd_freeze_effect.csv")
    memory_effect = pd.read_csv(probe_dir / "nsgd_memory_effect.csv")

    ready = summary[summary["source_ready"]].copy()
    strategy_summary = (
        ready.groupby("strategy", as_index=False)
        .agg(
            source_ready_cases=("source_ready", "size"),
            ari_post_auc_mean=("ari_post_auc_mean", "mean"),
            ari_post_auc_std=("ari_post_auc_mean", "std"),
            ari_target_only_auc_mean=("ari_target_only_auc_mean", "mean"),
            ari_normalized_auc_ratio_mean=("ari_normalized_auc_ratio", "mean"),
            ari_post_late_mean=("ari_post_late_mean", "mean"),
            ari_post_late_std_mean=("ari_post_late_std", "mean"),
            nmi_post_auc_mean=("nmi_post_auc_mean", "mean"),
            elapsed_time_mean=("elapsed_time", "mean"),
        )
    )
    total_readiness = (
        summary.groupby("strategy", as_index=False)
        .agg(source_ready_count=("source_ready", "sum"), total_cases=("source_ready", "size"))
    )
    strategy_summary = strategy_summary.merge(total_readiness, on="strategy", how="outer")
    strategy_summary.to_csv(probe_dir / "strategy_summary.csv", index=False)

    readiness_keys = ready[["source", "target", "strategy", "seed"]]
    post = curves[curves["condition"] == "post_switch"].merge(
        readiness_keys,
        on=["source", "target", "strategy", "seed"],
        how="inner",
    )
    mechanism_columns = [
        "ari",
        "nmi",
        "plasticity_mean",
        "plasticity_frozen_fraction",
        "plasticity_fully_mobile_fraction",
        "actual_motion_mean_per_node_step",
        "coactivation_cosine_to_reference",
        "coactivation_relative_change_from_reference",
        "relation_edge_jaccard_to_reference",
        "coordinate_from_reference_mean_rel_space",
    ]
    mechanism = (
        post[post["strategy"].isin(NSGD_ARMS)]
        .groupby(["strategy", "iteration"], as_index=False)[mechanism_columns]
        .mean()
    )
    mechanism.to_csv(probe_dir / "mechanism_summary.csv", index=False)

    colors = {
        "nsgd": "#3b82f6",
        "nsgd_unfreeze_at_switch": "#ef4444",
        "nsgd_no_freeze": "#f59e0b",
        "nsgd_mem300": "#8b5cf6",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for strategy in NSGD_ARMS:
        group = mechanism[mechanism["strategy"] == strategy]
        if group.empty:
            continue
        color = colors[strategy]
        axes[0, 0].plot(group["iteration"], group["ari"], label=strategy, color=color)
        axes[0, 1].plot(
            group["iteration"],
            group["plasticity_frozen_fraction"],
            label=strategy,
            color=color,
        )
        axes[1, 0].plot(
            group["iteration"],
            group["actual_motion_mean_per_node_step"],
            label=strategy,
            color=color,
        )
        axes[1, 1].plot(
            group["iteration"],
            group["relation_edge_jaccard_to_reference"],
            label=strategy,
            color=color,
        )
    axes[0, 0].set_ylabel("ARI")
    axes[0, 0].set_title("Post-switch target ARI")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].set_title("Mobility <= 0.05")
    axes[1, 0].set_ylabel("mean movement / node / physics step")
    axes[1, 0].set_title("Actual coordinate motion")
    axes[1, 0].set_yscale("symlog", linthresh=1e-6)
    axes[1, 1].set_ylabel("edge Jaccard")
    axes[1, 1].set_title("Relation edges retained from switch")
    for axis in axes[1]:
        axis.set_xlabel("post-switch input steps")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(probe_dir / "nsgd_switch_mechanism.png", dpi=170)
    plt.close(fig)

    ready_freeze = freeze_effect[freeze_effect["source_ready_nsgd"]].copy()
    freeze_summary = (
        ready_freeze.groupby("candidate", as_index=False)
        .agg(
            case_count=("ari_post_auc_delta_candidate_minus_nsgd", "size"),
            ari_auc_delta_mean=("ari_post_auc_delta_candidate_minus_nsgd", "mean"),
            ari_auc_delta_median=("ari_post_auc_delta_candidate_minus_nsgd", "median"),
            ari_late_delta_mean=("ari_post_late_mean_delta_candidate_minus_nsgd", "mean"),
            ari_late_std_delta_mean=("ari_post_late_std_delta_candidate_minus_nsgd", "mean"),
            nmi_auc_delta_mean=("nmi_post_auc_delta_candidate_minus_nsgd", "mean"),
        )
    )
    freeze_summary.to_csv(probe_dir / "freeze_effect_summary.csv", index=False)

    ready_memory = memory_effect[
        memory_effect["source_ready_nsgd"] & memory_effect["source_ready_nsgd_mem300"]
    ]
    memory_summary = pd.DataFrame([{
        "case_count": int(len(ready_memory)),
        "ari_auc_delta_mem300_minus_mem100_mean": float(
            ready_memory["ari_post_auc_delta_mem300_minus_mem100"].mean()
        ),
        "ari_late_delta_mem300_minus_mem100_mean": float(
            ready_memory["ari_post_late_mean_delta_mem300_minus_mem100"].mean()
        ),
        "nmi_auc_delta_mem300_minus_mem100_mean": float(
            ready_memory["nmi_post_auc_delta_mem300_minus_mem100"].mean()
        ),
    }])
    memory_summary.to_csv(probe_dir / "memory_effect_summary.csv", index=False)

    switch_columns = [
        "strategy",
        "switch_plasticity_mean_before",
        "switch_plasticity_mean_after",
        "switch_frozen_fraction_before",
        "switch_frozen_fraction_after",
        "stats_memory_steps_effective",
        "env_time_delay",
        "env_history_length",
    ]
    switch_summary = (
        summary[summary["strategy"].isin(NSGD_ARMS)]
        .groupby("strategy", as_index=False)[switch_columns[1:]]
        .mean()
    )
    memory_rows: list[dict[str, object]] = []
    seen_strategies: set[str] = set()
    for record in payload["records"]:
        strategy = str(record["strategy"])
        if strategy in seen_strategies:
            continue
        seen_strategies.add(strategy)
        model_params = record.get("effective_model_params") or {}
        env_params = record.get("effective_env_params") or {}
        memory_steps = model_params.get("stats_memory_steps")
        memory_rows.append({
            "strategy": strategy,
            "env_time_delay": env_params.get("env_time_delay"),
            "env_history_length": env_params.get("env_history_length"),
            "stats_memory_steps": memory_steps,
            "stats_memory_scale": model_params.get("stats_memory_scale"),
            "level1_enabled": model_params.get("level1_enabled"),
            "stats_half_life_steps": (
                float(np.log(0.5) / np.log(1.0 - 1.0 / float(memory_steps)))
                if memory_steps is not None and float(memory_steps) > 1.0
                else np.nan
            ),
            "velocity_decay": model_params.get("velocity_decay"),
            "moving_average_window": model_params.get("moving_average_window"),
            "state_memory": model_params.get("state_memory"),
        })
    memory_contract = pd.DataFrame(memory_rows)
    memory_contract.to_csv(probe_dir / "memory_contract.csv", index=False)

    identity_columns = [
        "ari",
        "nmi",
        "plasticity_mean",
        "plasticity_frozen_fraction",
        "coactivation_similarity_mean",
        "relation_edge_count",
    ]
    source_identity = curves[
        (curves["condition"] == "source")
        & curves["strategy"].isin(["nsgd", "nsgd_unfreeze_at_switch"])
    ].pivot_table(
        index=["source", "target", "seed", "iteration"],
        columns="strategy",
        values=identity_columns,
    )
    identity_rows: list[dict[str, object]] = []
    for column in identity_columns:
        base = source_identity[(column, "nsgd")]
        candidate = source_identity[(column, "nsgd_unfreeze_at_switch")]
        identity_rows.append({
            "field": column,
            "max_abs_source_delta": float(np.nanmax(np.abs(base - candidate))),
        })
    causal_identity = pd.DataFrame(identity_rows)
    causal_identity.to_csv(probe_dir / "causal_source_identity.csv", index=False)

    report = f"""# Detailed NSGD graph-switch report

## Objective

- Test whether maintained NSGD can learn a changed target partition.
- Verify whether the switch-time unfreeze is complete and quantify its effect.
- Audit exact-sequence, step-budget, environment, and memory fairness.

## Controls

- Maintained `nsgd`.
- `nsgd_unfreeze_at_switch`: source-identical causal intervention, then
  mobility and all plasticity bounds set to 1 at the switch.
- `nsgd_no_freeze`: no freezing during either phase.
- `nsgd_mem300`: maintained freezing with 300-step coactivation memory.
- Maintained `rg` and original `syncmap`.
- Every transfer arm is compared with a fresh target-only model receiving the
  exact saved target sequence.

## Fairness audit

- Pair/seed groups: {len(fairness)}.
- Passing groups: {int(fairness["fairness_pass"].sum())}/{len(fairness)}.
- Maximum target-sequence digest count: {int(fairness["target_sequence_digest_count"].max())}.

## Source-ready strategy results

{_markdown_table(strategy_summary.round(4))}

## Switch intervention

{_markdown_table(switch_summary.round(4))}

The causal unfreeze arm must be source-identical to maintained NSGD:

{_markdown_table(causal_identity.round(12))}

## Paired freezing effects

Candidate minus maintained NSGD, restricted to source-ready NSGD cases:

{_markdown_table(freeze_summary.round(4))}

## Memory-length effect

300-step NSGD minus maintained 100-step NSGD:

{_markdown_table(memory_summary.round(4))}

The algorithms do not have one interchangeable notion of memory. The shared
environment contract is matched; internal mechanisms are disclosed below:

{_markdown_table(memory_contract.round(4))}

For these flat switches RG reports `level1_enabled=false`, so its active
coactivation horizon is the 300-step base window; the geometric memory scale
is not an active upper-level advantage in this probe.

## Mechanism trace

- `mechanism_summary.csv` contains ARI/NMI, mobility distribution, actual
  coordinate motion, coactivation change, and relation-edge turnover at every
  recorded checkpoint.
- `nsgd_switch_mechanism.png` plots the aggregate source-ready trajectories.

## Interpretation boundary

- A one-way switch tests new-target learning, not old-task retention or
  switching back.
- Source readiness is required before interpreting adaptation.
- The probe isolates full permanent unfreezing. It does not yet implement a
  confidence-triggered temporary unfreeze.
"""
    (probe_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path)
    args = parser.parse_args()
    build_report(args.probe_dir.resolve())
    print(f"saved={args.probe_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
