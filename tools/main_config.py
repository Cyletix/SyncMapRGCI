"""Visible run-control configuration for SyncMap experiments.

Extracted from main.py so the large parameter panel does not bury the task
logic. Edit this file to change the default task/theme/metric, the
strategy/profile parameter panel (VARIANT_CONFIGS), and the MAIN_CONFIG block.
main.py imports these names back, so code importing them from `main`
(e.g. tools) keeps working.
"""
from __future__ import annotations

from typing import Any

from unified_experiments import EvaluationMetricName


DEFAULT_TASKNUM = 1
DEFAULT_THEME = "dark"
DEFAULT_ACCENT_COLOR = "#6970ac"
DEFAULT_METRIC: EvaluationMetricName = "ari"

TASK_ALIASES: dict[str, int] = {
    "0": 0,
    "single_test": 0,
    "1": 1,
    "run_live": 1,
    "2": 2,
    "all_experiments": 2,
    "4": 4,
    "trace_scores": 4,
    "5": 5,
    "render_trace": 5,
    "6": 6,
    "analyze_sequence_information": 6,
    "7": 7,
    "scaling_law": 7,
}


# Canonical full-suite hierarchy datasets. `--datasets hierarchy` expands to
# this list; `--datasets flat` expands at runtime to every top-level flat
# .dot file (unified_experiments.get_all_datasets).
HIERARCHY_DATASET_SUITE: list[str] = [
    "datasets/hierarchy/DCH1_fixed_chunk_6_3.dot",
    "datasets/hierarchy/DCH2_fixed_chunk_6_3_dynamic.dot",
    "datasets/hierarchy/DIH2_fixed_chunk_branches_2.dot",
    "datasets/hierarchy/EC_fixed_chunk_15_3.dot",
    "datasets/hierarchy/EH_fixed_chunk_15_3_deep.dot",
    "datasets/hierarchy/HB_fixed_chunk_branches.dot",
    "datasets/hierarchy/IEH_fixed_chunk_deep_hierarchy.dot",
    "datasets/hierarchy/IH_fixed_chunk_unbalanced.dot",
    "datasets/hierarchy/fixed_chunk_15_3.dot",
    "datasets/hierarchy/fixed_chunk_branches_extra.dot",
]


# Visible strategy/profile control panel.
# Edit this file when you want to change what this branch runs by default.
#
# Important manual-check datasets:
# - chain_mixed5_6.dot
# - chain_mixed5_60.dot
# - chain_mixed60_5.dot
# - fixed5_6.dot
# - hierarchy/DCH1_fixed_chunk_6_3.dot
VARIANT_CONFIGS: dict[str, dict[str, Any]] = {'syncmap': {'public_runtime': 'syncmap', 'status': 'active', 'summary': 'Original SyncMap baseline preserved as its own runtime.', 'params': {}}, 'symmetric': {'public_runtime': 'symmetric', 'status': 'active', 'summary': 'Symmetrical SyncMap baseline using the maintained runtime defaults from SymmetricalSyncMap.DEFAULT_PARAMS.', 'params': {}}, 'nsgd': {'public_runtime': 'nsgd', 'status': 'active', 'summary': 'NSGD relation-graph dynamics and plasticity preset.', 'params': {'base_wall': 100.0, 'density_ref': 64.0, 'adaptation_rate': 0.05, 'max_step_norm': 5.0, 'k_repulse': 0.0, 'k_pulse': 2.0, 'k_attr': 1.0, 'plasticity_decay': 0.99, 'plasticity_min': 0.01, 'plasticity_recovery': 1.0, 'T_push': 10, 'knn': 12, 'graph_theta': 0.1, 'graph_r0_rel': 0.1, 'dbscan_eps': 1.0, 'dbscan_min_samples': 3, 'dbscan_auto_eps': False, 'embed_cluster_method': 'dbscan', 'clustering_period': 100, 'louvain_distance_factor': 0.15, 'stats_memory_steps': 100, 'stats_max_lag': 0}}, 'tsf': {'public_runtime': 'tsf', 'status': 'active', 'summary': 'Independent TSFMap-style centroid dynamics with post-hoc hierarchical clustering, kept separate from RG recursion for direct hierarchy comparisons.', 'params': {'adaptation_rate': 0.001, 'space_size': 1.0, 'activation_threshold': 0.1, 'velocity_decay': 0.999, 'mu_pos': 6.0, 'mu_neg': 3.0, 'mu_repulse': 2.0, 'normalize_each_step': True, 'normalize_mode': 'max_abs', 'embed_cluster_method': 'agglomerative', 'cluster_linkage': 'ward', 'target_levels': 3, 'min_clusters': 2, 'max_clusters': 16, 'organize_level': 'cluster_level1'}}, 'rg': {'public_runtime': 'rg', 'status': 'mainline', 'summary': 'Mainline Recursive Grouping (RG), engine SyncMapRG. Primary dynamics: primitive contrastive force (co-activation attraction toward the active centroid + 1/r repulsion of inactive points from it). Secondary: graph spring (k_attr) + local obstacle (k_repulse). Geometric patch forces (entity_boundary, sibling, line_obstacle) off; macro_centroid repulsion via runtime feedback.', 'params': {}}}

# 2026-06-17: Memory-length controlled variants for the TSF-vs-RG fairness
# ablation. Two independent memory knobs:
#   - env_time_delay  : shared INPUT window (paper's m=10). Per-strategy override.
#   - stats_memory_scale : RG's per-level INTERNAL memory growth (default 10×/level).
# Defined by spreading the base profile params so they stay in sync automatically.
VARIANT_CONFIGS["rg_shortmem"] = {
    "public_runtime": "rg",
    "status": "experimental",
    "summary": (
        'RG sensitivity preset with stats_memory_scale=1.0. Local-event capacities are equal across levels, but raw-input horizons still differ with token cadence.'
    ),
    "params": {**VARIANT_CONFIGS["rg"]["params"], "stats_memory_scale": 1.0},
}

# 2026-06-24: stricter memory-length control. rg_shortmem only
# shortened per-level GROWTH (L0 base stayed 300). This shrinks the L0 BASE window
# too: stats_memory_scale=1.0 AND stats_memory_steps=30 -> every level uses a 30-step
# window (10x shorter than rg_shortmem's 300, 1000x shorter than rg's L2=30000).
# If RG's coarse (L2) holds at all-30, RG's coarse advantage is memory-independent.
VARIANT_CONFIGS["rg_shortmem_l0"] = {
    "public_runtime": "rg",
    "status": "experimental",
    "summary": (
        'RG sensitivity preset with base relation memory 30 and scale 1.0. Each level counts its own received events; this is not raw-clock matching.'
    ),
    "params": {**VARIANT_CONFIGS["rg"]["params"], "stats_memory_scale": 1.0, "stats_memory_steps": 30},
}

VARIANT_CONFIGS["tsf_longmem"] = {
    "public_runtime": "tsf",
    "status": "experimental",
    "summary": (
        'TSF sensitivity preset with env_time_delay=100. This changes the shared observation history, not just model-internal memory.'
    ),
    "params": {**VARIANT_CONFIGS["tsf"]["params"], "env_time_delay": 100},
}

VARIANT_CONFIGS["tsf_velocity_longmem"] = {
    "public_runtime": "tsf",
    "status": "experimental",
    "summary": (
        "TSF internal-memory ablation: keep the shared input encoder at "
        "env_time_delay=10 and extend only TSF velocity inertia from "
        "0.999 (~1000-step EMA) to 0.9999 (~10000-step EMA)."
    ),
    "params": {**VARIANT_CONFIGS["tsf"]["params"], "velocity_decay": 0.9999},
}

MEMORY_PROFILE_CHOICES: tuple[str, ...] = (
    "historical",
    "coactivation",
    "transition",
    "ema",
)


MAIN_CONFIG: dict[str, Any] = {'dataset': 'datasets/hierarchy/DCH1_fixed_chunk_6_3.dot', 'default_strategy': 'rg', 'comparison': {'profiles': ['syncmap', 'nsgd', 'rg', 'tsf'], 'strategies': ['syncmap', 'nsgd', 'rg', 'tsf'], 'strategy_defaults': {'syncmap': {}, 'nsgd': {}, 'rg': dict(VARIANT_CONFIGS['rg']['params']), 'tsf': {}}, 'sequence_lengths': [5000, 100000], 'trials': 5, 'target_iterations': [5000], 'metric_name': 'ari'}, 'comparison_profiles': ['syncmap', 'nsgd', 'rg', 'tsf'], 'comparison_strategies': ['syncmap', 'nsgd', 'rg', 'tsf'], 'strategy_defaults': {'syncmap': {}, 'nsgd': {}, 'rg': dict(VARIANT_CONFIGS['rg']['params'])}, 'sequence_length': 100000, 'all_sequence_lengths': [5000, 100000], 'comparison_trials': 5, 'all_target_iterations': [5000], 'all_metric_name': 'ari'}
