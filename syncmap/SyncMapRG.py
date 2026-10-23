#!/usr/bin/env python3
"""
SyncMapX_RG: Minimal Recursive Hebbian Tokenization engine (rg algorithm).
----------------------------------------------------------------------------------
Independent engine for the ``rg`` algorithm (distinct from ``rg`` / SyncMapRG.py).
Core dynamics:
- Primitive contrastive force: co-activation attraction + inverse-square repulsion.
- Secondary: graph spring (k_attr) + local obstacle (k_repulse).
- RG recursion with stable-token / macro-token readout (physical_mst_gaps / dbscan).
All geometric patch forces (entity_boundary, sibling, line_obstacle, macro_centroid)
are intentionally absent from this engine.
"""
from __future__ import annotations

from collections import deque
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from syncmap.SyncMapNS import SyncMapNS
from tools.clustering_strategies import create_clustering_strategy
from tools.shared_defaults import DEFAULT_HIERARCHY_ORGANIZE_LEVEL
from test.hierarchy_levels import (
    canonicalize_hierarchy_outputs,
    canonicalize_organize_level,
    parse_cluster_level_index,
)
from tools.rg_mst import RGSignificanceResult, _dense_mst, estimate_rg_significance

IntArray = NDArray[np.intp]
FloatArray = NDArray[np.float64]


class _StableTokenEvidenceMemory:
    """Compact transition evidence for reversible stable-token retention."""

    def __init__(self, capacity: int, memory_steps: int) -> None:
        self.capacity = max(1, int(capacity))
        self.counts = np.zeros((self.capacity, self.capacity), dtype=float)
        self._previous_tokens = np.zeros(0, dtype=int)
        self.set_memory_steps(memory_steps)

    def set_memory_steps(self, memory_steps: int) -> None:
        self.memory_steps = max(1, int(memory_steps))
        self.decay = 1.0 - (1.0 / float(self.memory_steps))

    def reset_token(self, token: int) -> None:
        token_id = int(token)
        if token_id < 0 or token_id >= self.capacity:
            return
        self.counts[token_id, :] = 0.0
        self.counts[:, token_id] = 0.0
        self._previous_tokens = self._previous_tokens[self._previous_tokens != token_id]

    def observe(self, active_tokens: IntArray) -> None:
        self.counts *= self.decay
        current = np.asarray(active_tokens, dtype=int).reshape(-1)
        current = np.unique(current[(current >= 0) & (current < self.capacity)])
        previous = self._previous_tokens
        if previous.size > 0 and current.size > 0:
            weight = 1.0 / float(previous.size * current.size)
            self.counts[np.ix_(previous, current)] += weight
        self._previous_tokens = current

    def supports_distinct_tokens(
        self,
        tokens: tuple[int, ...],
        *,
        min_self_mass: float,
        self_cross_ratio: float,
    ) -> bool:
        token_arr = np.asarray(tokens, dtype=int)
        if token_arr.size < 2 or np.any((token_arr < 0) | (token_arr >= self.capacity)):
            return False
        for token in token_arr.tolist():
            others = token_arr[token_arr != int(token)]
            self_mass = float(self.counts[int(token), int(token)])
            cross_mass = 0.0
            if others.size:
                cross_mass = float(
                    np.max(
                        np.maximum(
                            self.counts[int(token), others],
                            self.counts[others, int(token)],
                        )
                    )
                )
            if self_mass < float(min_self_mass):
                return False
            if self_mass < float(self_cross_ratio) * cross_mass:
                return False
        return True


class SyncMapX_RG:
    # Subclasses may replace only the per-level engine while reusing RG's
    # recursive orchestration. The maintained RG class continues to instantiate
    # SyncMapNS exactly as before.
    LEVEL_MODEL_CLASS = SyncMapNS

    DEFAULT_PARAMS: dict[str, Any] = {
        # 2026-06-24: the mainline rg configuration now lives HERE (the class is
        # the single source of truth, like every other strategy's DEFAULT_PARAMS).
        # These were previously injected from main_config VARIANT_CONFIGS["rg"]
        # ["params"], so a bare SyncMapX_RG() / run_single_test('rg') ran an
        # unconfigured engine (all forces off, native readout) and scored ~0.
        # The block below overrides the SyncMapNS base defaults:
        'adaptation_rate': 0.05,
        'max_step_norm': 2.0,
        'T_push': 100,
        'clustering_period': 1000,
        'stats_memory_steps': 300,
        'k_attr': 0.07,
        'k_repulse': 1.0,
        'embed_cluster_method': 'dbscan',
        'dbscan_min_samples': 2,
        'level_count': 2,
        'rg_scale_factor': 3.16,
        'rg_time_scale_mode': 'geometric',
        'rg_time_scale_growth': 3.16,
        'stats_memory_scale': 10.0,
        'warmup_steps': 2000,
        'renormalize_interval': 50,
        # Keep the historical nested global-modulo scheduler as the default.
        # `elapsed_geometric` is an experimental scheduler that gives every
        # inter-level interface its own raw-input-step deadline.
        'rg_inter_level_schedule_mode': 'nested_modulo',
        'level1_enabled': True,
        'organize_level': DEFAULT_HIERARCHY_ORGANIZE_LEVEL,
        'rg_input_level': 'raw',
        'feedback_mode': 'none',
        'rg_knn_k': 3,
        'rg_jump_ratio': 1.75,
        'upper_rg_jump_ratio': None,
        'rg_jump_ratio_final': None,
        'rg_jump_ratio_anneal_start': 0,
        'rg_jump_ratio_anneal_end': 0,
        'macro_boundary_gap_gate_enabled': False,
        'macro_boundary_gap_gate_floor': 0.1,
        'macro_boundary_gap_gate_full_ratio': 3.0,
        'macro_boundary_min_stable_refreshes': 0,
        'macro_boundary_token_purity_enabled': False,
        'macro_boundary_token_purity_threshold': 0.6,
        'macro_boundary_token_purity_floor': 0.2,
        'rg_semantic_graph_enabled': False,
        'rg_semantic_graph_strength': 4.0,
        'rg_semantic_graph_similarity_floor': 0.2,
        'rg_semantic_graph_penalty_mode': 'contrastive',
        'rg_semantic_graph_relative_threshold': 0.5,
        'rg_semantic_graph_anchor_floor': 0.3,
        'rg_semantic_graph_edge_mode': 'none',
        'rg_semantic_graph_topk': 0,
        'rg_fallback_scale': 1.5,
        'macro_component_proposal_mode': 'semantic',
        'upper_cluster_proposal_mode': 'semantic',
        'upper_cluster_semantic_source': 'lower',
        'upper_cluster_semantic_method': 'louvain',
        'upper_cluster_semantic_threshold': 0.2,
        'upper_cluster_semantic_resolution': 1.0,
        'upper_cluster_semantic_self_loop_weight': 1.0,
        'upper_cluster_semantic_sparsify_mode': 'none',
        'upper_cluster_semantic_topk': 0,
        'rg_semantic_sharpen_power': 1.0,
        'upper_cluster_refresh_interval': 25,
        'hierarchy_refresh_interval': 200,
        'macro_split_hysteresis_ratio': 1.0,
        'inter_level_rigid_shift_enabled': True,
        'inter_level_rigid_shift_strength': 1.0,
        'inter_level_rigid_shift_max_step_ratio': 0.5,
        'inter_level_centroid_anchor_enabled': False,
        'inter_level_centroid_anchor_strength': 0.2,
        'inter_level_centroid_anchor_max_step_ratio': 0.25,
        'source_unit_motion_coherence_enabled': False,
        'source_unit_motion_coherence_strength': 0.5,
        'source_unit_motion_coherence_max_step_ratio': 0.5,
        'inter_level_support_reanchor_enabled': False,
        'inter_level_support_reanchor_strength': 0.5,
        'source_unit_barrier_enabled': False,
        'source_unit_barrier_strength_ratio': 0.25,
        'support_unit_repulsion_enabled': False,
        'support_unit_repulsion_strength_ratio': 1.0,
        'cluster_distance_strength': 0.1,
        'macro_centroid_distance_ratio': 3.16,
        'k_macro_repulse': 0.1,
        'macro_centroid_max_step_ratio': 0.1,
        'entity_boundary_repulsion_enabled': False,
        'entity_boundary_distance_ratio': 4.8,
        'k_entity_boundary_repulse': 0.0,
        'entity_boundary_min_gap_ratio': 0.25,
        'entity_boundary_max_step_ratio': 0.5,
        'entity_sibling_attraction_enabled': False,
        'entity_sibling_target_ratio': 4.8,
        'entity_sibling_capture_ratio': 3.16,
        'k_entity_sibling_attr': 0.0,
        'entity_sibling_max_step_ratio': 0.5,
        'entity_sibling_boundary_attenuation_enabled': True,
        'entity_sibling_boundary_release_ratio': 1.25,
        'entity_sibling_boundary_power': 2.0,
        'entity_sibling_deadband_ratio': 0.0,
        'macro_cross_attr_scale': 0.001,
        'line_obstacle_repulsion_enabled': False,
        'line_obstacle_radius_rel': 1.0,
        'line_obstacle_strength': 0.5,
        'primitive_contrastive_force_enabled': True,
        'primitive_active_attraction_strength': 0.5,
        'primitive_inactive_repulsion_strength': 2.0,
        'primitive_force_distance_floor_rel': 0.15,
        'primitive_force_max_step_ratio': 3.0,
        'graph_weight_confidence_enabled': False,
        'graph_weight_confidence_percentile': 90.0,
        'graph_weight_confidence_power': 1.0,
        'graph_weight_confidence_floor': 0.0,
        'graph_weight_confidence_gain': 1.0,
        'rg_fill_noise_labels': True,
        'stable_token_extraction_mode': 'physical_mst_gaps',
        'hierarchy_readout_mode': 'physical_mst_gaps',
        'hierarchy_readout_min_gap_ratio': 1.05,
        'dbscan_eps_scale_mode': 'geometric_midpoint',
        'dbscan_eps_margin': 1.05,
        'dbscan_readout_eps': None,
        'dbscan_readout_slack': 1.0,
        'dbscan_readout_min_samples': None,
        'rg_upper_input_active_gain': 1.0,
        'rg_upper_input_mass_gain': 0.5,
        # rg is a transition-based engine: its graph spring reads the
        # prev->curr transition operator. On single-active-token streams (which
        # the real benchmarks are) the same-step coactivation operator is blind,
        # so default the base level to transition rather than inheriting NS's
        # coactivation default. Paths that pass stats_mode explicitly are unaffected.
        'stats_mode': 'transition',
        'rg_upper_stats_mode': 'transition',
        # Upper-level macro edge weight source. 'transition' = the upper level's own
        # token->token transition stats (default, legacy). 'coactivation_density' =
        # aggregate the LOWER level's node co-activation matrix by token membership
        # (RG coarse-graining of "volume"): the macro structure is a node co-activation
        # community that token transitions wash out, so this lets the upper spring see
        # it. Read-only wrt the lower level.
        'bottomup_force_architecture': False,
        'rg_upper_edge_mode': 'transition',
        'rg_upper_history_window': 6,
        'persistent_relation_force_enabled': False,
        'persistent_relation_decay': 0.98,
        'persistent_relation_threshold': 0.03466981625132058,
        'persistent_relation_signal_gain': 3.5010740011433463,
        'persistent_relation_surprise_low': 0.25,
        'persistent_relation_surprise_mid': 0.5,
        'persistent_relation_surprise_high': 1.0,
        'persistent_relation_same_force_strength_ratio': 2.9412014246974474,
        'persistent_relation_diff_force_strength_ratio': 1.0,
    }

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        params = SyncMapNS.default_params()
        params.update(cls.DEFAULT_PARAMS)
        return params

    @classmethod
    def normalize_params(cls, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = cls.default_params()
        if params:
            merged.update(dict(params))
        return merged

    @staticmethod
    def _is_level_specific_param(name: str) -> bool:
        stem, marker, suffix = str(name).rpartition('_l')
        return bool(stem) and marker == '_l' and suffix.isdigit()

    @classmethod
    def _reject_level_specific_params(cls, kwargs: dict[str, Any]) -> None:
        illegal = sorted(str(k) for k in kwargs.keys() if cls._is_level_specific_param(str(k)))
        if illegal:
            joined = ', '.join(illegal)
            raise ValueError(
                f"RG no longer accepts level-specific parameter overrides: {joined}. "
                "Provide one shared base parameter set and let RG expand it recursively."
            )

    def __init__(self, input_size: int, dimensions: int = 3, **kwargs: Any) -> None:
        self._reject_level_specific_params(kwargs)
        self.input_size = int(input_size)
        self.dimensions = int(dimensions)
        shared_space_kwargs: dict[str, Any] = {}
        for key in ('base_wall', 'density_ref', 'packing_factor', 'significant_distance_factor'):
            if key in kwargs:
                shared_space_kwargs[key] = kwargs[key]
        if 'space_size' in kwargs:
            shared_space_kwargs['space_size'] = kwargs['space_size']

        # RG核心设计：每个层级遵循相同的动力学，只是尺度不同（×scale_factor）
        self.rg_scale_factor = float(kwargs.get('rg_scale_factor', 3.16))
        self.stats_memory_scale = max(1.0, float(kwargs.get('stats_memory_scale', 10.0)))
        self.stable_token_evidence_enabled = False
        self.stable_token_evidence_memory_scale = 1.0
        self.stable_token_evidence_min_members = 2
        self.stable_token_evidence_semantic_gate_enabled = False
        self.stable_token_evidence_semantic_source = 'slow_raw'
        self.stable_token_evidence_min_stable_refreshes = 0
        self.stable_token_evidence_semantic_z_score = 0.0
        self.stable_token_evidence_min_self_mass = 0.0
        self.stable_token_evidence_self_cross_ratio = 1.0
        self.stable_token_merge_debounce_enabled = False
        self.stable_token_merge_debounce_steps = 0
        self.stable_token_merge_debounce_min_age = 0
        self.stable_token_merge_debounce_activation_steps = 0
        self.stable_token_merge_debounce_compactness_enabled = False
        self.stable_token_merge_debounce_compactness_ratio = 1.0

        self.level_count = max(2, int(kwargs.get('level_count', 2)))
        self._shared_space_kwargs = dict(shared_space_kwargs)
        self._init_level_kwargs = dict(kwargs)
        self.levels: list[SyncMapNS] = []
        self._level_phases: list[int] = []
        self.inter_level_unit_maps: list[IntArray] = []
        self.inter_level_meta_maps: list[IntArray] = []
        self._last_lifted_labels_by_level: list[Optional[IntArray]] = []
        self._level_support_masks: list[NDArray[np.bool_]] = []
        self._merge_evidence_by_level: list[dict[tuple[int, ...], int]] = []
        self._stable_token_evidence_by_level: list[Any] = []
        self._stable_token_evidence_decisions: dict[str, int] = {}
        self._stable_token_evidence_semantic_ratios: deque[float] = deque(maxlen=1)
        self._stable_token_evidence_semantic_z_margins: deque[float] = deque(maxlen=1)
        self._stable_token_evidence_confirmed_signatures_by_level: list[dict[int, tuple[int, ...]]] = []
        self._stable_token_identity_ages_by_level: list[IntArray] = []
        self._stable_token_identity_signatures_by_level: list[list[tuple[int, ...] | None]] = []
        self._stable_token_evidence_matured_by_level: list[NDArray[np.bool_]] = []
        self._stable_token_semantic_raw_memory = _StableTokenEvidenceMemory(self.input_size, 1)
        self._pair_relation_evidence_by_level: list[dict[tuple[int, int], float]] = []
        self._last_source_support_tokens_by_level: list[tuple[int, ...]] = []
        self._last_meta_support_tokens_by_level: list[tuple[int, ...]] = []
        self._last_upper_cluster_tokens_by_level: list[tuple[int, ...]] = []
        self._last_upper_cluster_labels_by_level: list[Optional[IntArray]] = []
        self._last_upper_cluster_refresh_step_by_level: list[int] = []
        self._last_upper_cluster_provenance_by_level: list[Optional[dict[str, Any]]] = []
        self._last_upper_input_by_level: list[NDArray[np.float64]] = []
        self._upper_history_by_level: list[deque[int]] = []
        self._last_macro_feedback_strengths_by_level: list[FloatArray] = []
        self._macro_stability_counts_by_level: list[IntArray] = []
        self._macro_member_signatures_by_level: list[list[tuple[int, ...] | None]] = []
        self._next_inter_level_due_step_by_level: list[int] = []
        self._ensure_level_count(self.level_count)

        self.step_count = 0
        self.warmup_steps = int(kwargs.get('warmup_steps', 2000))
        self.renormalize_interval = max(1, int(kwargs.get('renormalize_interval', 50)))
        self.rg_time_scale_mode = str(kwargs.get('rg_time_scale_mode', 'geometric')).strip().lower()
        self.rg_time_scale_growth = float(kwargs.get('rg_time_scale_growth', 0.0))
        self.rg_inter_level_schedule_mode = self._normalize_inter_level_schedule_mode(
            kwargs.get('rg_inter_level_schedule_mode', 'nested_modulo')
        )
        self.level1_enabled = bool(kwargs.get('level1_enabled', True))
        self.organize_level = canonicalize_organize_level(
            kwargs.get('organize_level'),
            default=DEFAULT_HIERARCHY_ORGANIZE_LEVEL,
        )
        self.rg_input_level = str(kwargs.get('rg_input_level', 'stable')).lower()
        self.feedback_mode = str(kwargs.get('feedback_mode', 'none')).strip().lower()
        self.rg_knn_k = max(1, int(kwargs.get('rg_knn_k', 3)))
        self.rg_jump_ratio = float(kwargs.get('rg_jump_ratio', 1.75))
        upper_jump = kwargs.get('upper_rg_jump_ratio', None)
        self.upper_rg_jump_ratio = None if upper_jump is None else float(upper_jump)
        jump_final = kwargs.get('rg_jump_ratio_final', None)
        self.rg_jump_ratio_final = None if jump_final is None else float(jump_final)
        self.rg_jump_ratio_anneal_start = max(0, int(kwargs.get('rg_jump_ratio_anneal_start', 0)))
        self.rg_jump_ratio_anneal_end = max(0, int(kwargs.get('rg_jump_ratio_anneal_end', 0)))
        self.macro_boundary_gap_gate_enabled = bool(kwargs.get('macro_boundary_gap_gate_enabled', False))
        self.macro_boundary_gap_gate_floor = float(np.clip(kwargs.get('macro_boundary_gap_gate_floor', 0.1), 0.0, 1.0))
        self.macro_boundary_gap_gate_full_ratio = max(
            float(kwargs.get('macro_boundary_gap_gate_full_ratio', 3.0)),
            self.rg_jump_ratio + 1e-6,
        )
        self.macro_boundary_min_stable_refreshes = max(
            0,
            int(kwargs.get('macro_boundary_min_stable_refreshes', 0)),
        )
        self.macro_boundary_token_purity_enabled = bool(
            kwargs.get('macro_boundary_token_purity_enabled', False)
        )
        self.macro_boundary_token_purity_threshold = float(
            np.clip(kwargs.get('macro_boundary_token_purity_threshold', 0.6), 0.0, 1.0)
        )
        self.macro_boundary_token_purity_floor = float(
            np.clip(kwargs.get('macro_boundary_token_purity_floor', 0.2), 0.0, 1.0)
        )
        self.rg_semantic_graph_enabled = bool(kwargs.get('rg_semantic_graph_enabled', False))
        self.rg_semantic_graph_strength = max(float(kwargs.get('rg_semantic_graph_strength', 4.0)), 0.0)
        self.rg_semantic_graph_similarity_floor = float(
            np.clip(kwargs.get('rg_semantic_graph_similarity_floor', 0.2), 0.0, 1.0)
        )
        self.rg_semantic_graph_penalty_mode = str(
            kwargs.get('rg_semantic_graph_penalty_mode', 'contrastive')
        ).strip().lower()
        self.rg_semantic_graph_relative_threshold = float(
            np.clip(kwargs.get('rg_semantic_graph_relative_threshold', 0.5), 0.0, 1.0)
        )
        self.rg_semantic_graph_anchor_floor = float(
            np.clip(kwargs.get('rg_semantic_graph_anchor_floor', 0.3), 0.0, 1.0)
        )
        self.rg_semantic_graph_edge_mode = str(
            kwargs.get('rg_semantic_graph_edge_mode', 'none')
        ).strip().lower()
        self.rg_semantic_graph_topk = max(int(kwargs.get('rg_semantic_graph_topk', 0)), 0)
        self.rg_fallback_scale = float(kwargs.get('rg_fallback_scale', 1.5))
        self.macro_component_proposal_mode = str(
            kwargs.get('macro_component_proposal_mode', 'significance')
        ).strip().lower()
        self.upper_cluster_proposal_mode = str(
            kwargs.get('upper_cluster_proposal_mode', 'semantic')
        ).strip().lower()
        self.upper_cluster_semantic_source = str(
            kwargs.get('upper_cluster_semantic_source', 'upper')
        ).strip().lower()
        self.upper_cluster_semantic_method = str(
            kwargs.get('upper_cluster_semantic_method', 'louvain')
        ).strip().lower()
        self.upper_cluster_semantic_threshold = float(
            np.clip(kwargs.get('upper_cluster_semantic_threshold', 0.2), 0.0, 1.0)
        )
        self.upper_cluster_semantic_resolution = max(
            float(kwargs.get('upper_cluster_semantic_resolution', 1.0)),
            1e-6,
        )
        self.upper_cluster_semantic_self_loop_weight = max(
            float(kwargs.get('upper_cluster_semantic_self_loop_weight', 1.0)),
            0.0,
        )
        self.upper_cluster_semantic_sparsify_mode = str(
            kwargs.get('upper_cluster_semantic_sparsify_mode', 'none')
        ).strip().lower()
        self.upper_cluster_semantic_topk = max(int(kwargs.get('upper_cluster_semantic_topk', 0)), 0)
        self.rg_semantic_sharpen_power = max(
            float(kwargs.get('rg_semantic_sharpen_power', 1.0)),
            1.0,
        )
        self.upper_cluster_refresh_interval = max(1, int(kwargs.get('upper_cluster_refresh_interval', 25)))
        self.hierarchy_refresh_interval = max(0, int(kwargs.get('hierarchy_refresh_interval', 0)))
        self.macro_split_hysteresis_ratio = max(float(kwargs.get('macro_split_hysteresis_ratio', 1.0)), 1.0)
        self.inter_level_rigid_shift_enabled = bool(kwargs.get('inter_level_rigid_shift_enabled', True))
        self.inter_level_rigid_shift_strength = float(kwargs.get('inter_level_rigid_shift_strength', 1.0))
        self.inter_level_rigid_shift_max_step_ratio = float(kwargs.get('inter_level_rigid_shift_max_step_ratio', 0.5))
        self.inter_level_centroid_anchor_enabled = bool(kwargs.get('inter_level_centroid_anchor_enabled', False))
        self.inter_level_centroid_anchor_strength = float(kwargs.get('inter_level_centroid_anchor_strength', 0.2))
        self.inter_level_centroid_anchor_max_step_ratio = float(
            kwargs.get(
                'inter_level_centroid_anchor_max_step_ratio',
                kwargs.get('inter_level_rigid_shift_max_step_ratio', 0.25),
            )
        )
        self.source_unit_motion_coherence_enabled = bool(kwargs.get('source_unit_motion_coherence_enabled', False))
        self.source_unit_motion_coherence_strength = float(kwargs.get('source_unit_motion_coherence_strength', 0.5))
        self.source_unit_motion_coherence_max_step_ratio = float(kwargs.get('source_unit_motion_coherence_max_step_ratio', 0.5))
        self.inter_level_support_reanchor_enabled = bool(kwargs.get('inter_level_support_reanchor_enabled', False))
        self.inter_level_support_reanchor_strength = float(kwargs.get('inter_level_support_reanchor_strength', 0.5))
        self.source_unit_barrier_enabled = bool(kwargs.get('source_unit_barrier_enabled', False))
        self.source_unit_barrier_strength_ratio = float(kwargs.get('source_unit_barrier_strength_ratio', 0.25))
        self.support_unit_repulsion_enabled = bool(kwargs.get('support_unit_repulsion_enabled', False))
        self.support_unit_repulsion_strength_ratio = float(
            kwargs.get('support_unit_repulsion_strength_ratio', 1.0)
        )
        self.rg_fill_noise_labels = bool(kwargs.get('rg_fill_noise_labels', True))
        self.stable_token_extraction_mode = str(
            kwargs.get('stable_token_extraction_mode', 'native')
        ).strip().lower()
        self.hierarchy_readout_mode = str(
            kwargs.get('hierarchy_readout_mode', 'native_levels')
        ).strip().lower()
        self.hierarchy_readout_min_gap_ratio = max(
            float(kwargs.get('hierarchy_readout_min_gap_ratio', 1.05)),
            1.0,
        )
        self.dbscan_eps_scale_mode = str(
            kwargs.get('dbscan_eps_scale_mode', 'geometric_midpoint')
        ).strip().lower()
        self.dbscan_eps_margin = max(float(kwargs.get('dbscan_eps_margin', 1.05)), 1e-6)
        raw_dbscan_readout_eps = kwargs.get('dbscan_readout_eps', None)
        self.dbscan_readout_eps = None if raw_dbscan_readout_eps is None else float(raw_dbscan_readout_eps)
        self.dbscan_readout_slack = max(float(kwargs.get('dbscan_readout_slack', 1.0)), 1e-6)
        raw_dbscan_readout_min_samples = kwargs.get('dbscan_readout_min_samples', None)
        self.dbscan_readout_min_samples = (
            None if raw_dbscan_readout_min_samples is None else max(1, int(raw_dbscan_readout_min_samples))
        )
        self.rg_upper_input_active_gain = float(kwargs.get('rg_upper_input_active_gain', 1.0))
        self.rg_upper_input_mass_gain = float(kwargs.get('rg_upper_input_mass_gain', 0.5))
        self.rg_upper_stats_mode = str(kwargs.get('rg_upper_stats_mode', 'transition')).strip().lower()
        self.bottomup_force_architecture = bool(kwargs.get('bottomup_force_architecture', False))
        self.rg_upper_edge_mode = str(kwargs.get('rg_upper_edge_mode', 'transition')).strip().lower()
        self.rg_upper_history_window = max(1, int(kwargs.get('rg_upper_history_window', 6)))
        if self._upper_history_by_level:
            self._upper_history_by_level = [
                deque(history, maxlen=self.rg_upper_history_window)
                for history in self._upper_history_by_level
            ]
        self.persistent_relation_force_enabled = bool(kwargs.get('persistent_relation_force_enabled', False))
        self.persistent_relation_decay = float(kwargs.get('persistent_relation_decay', 0.98))
        self.persistent_relation_threshold = float(kwargs.get('persistent_relation_threshold', 0.03466981625132058))
        self.persistent_relation_signal_gain = float(kwargs.get('persistent_relation_signal_gain', 3.5010740011433463))
        self.persistent_relation_surprise_low = float(kwargs.get('persistent_relation_surprise_low', 0.25))
        self.persistent_relation_surprise_mid = float(kwargs.get('persistent_relation_surprise_mid', 0.5))
        self.persistent_relation_surprise_high = float(kwargs.get('persistent_relation_surprise_high', 1.0))
        self.persistent_relation_same_force_strength_ratio = float(
            kwargs.get('persistent_relation_same_force_strength_ratio', 2.9412014246974474)
        )
        self.persistent_relation_diff_force_strength_ratio = float(
            kwargs.get('persistent_relation_diff_force_strength_ratio', 1.0)
        )
        self.unit_retention_enabled = False
        self.unit_merge_hysteresis_steps = 1
        self.unit_merge_distance_ratio = 1.0

        self._last_level0_raw_labels: Optional[IntArray] = None
        self._last_macro_tokens: Optional[IntArray] = None
        self._last_rg_significance: Optional[RGSignificanceResult] = None

        # Expose the current RG scale state for live/debug tooling.
        self.base_wall = float(kwargs.get('base_wall', 100.0))
        self.density_ref = float(kwargs.get('density_ref', 64.0))
        self.space_size = float(kwargs.get('space_size', 0.0)) if 'space_size' in kwargs else 0.0
        self.effective_local_scale: float = 0.0
        self.reference_local_scale: float = 0.0
        self.significant_distance: float = 0.0
        self.macro_token_count: int = 0

        # 注意：某些参数（centering_strength, max_step_norm等）已经在 _build_level_kwargs 中
        # 按 scale_factor 缩放过了。需要防止 update_params 用通用版本覆盖它们。
        
        # 需要被保护的参数（已被缩放）
        scaled_params = {'centering_strength', 'max_step_norm', 'dbscan_eps', 'graph_r0_rel', 'graph_r0_fixed'}
        
        # 过滤掉已缩放参数与所有 level-specific 覆盖，RG 只接受统一基准参数。
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in scaled_params}
        
        if filtered_kwargs:
            self.update_params(**filtered_kwargs)

    def _sync_public_level_aliases(self) -> None:
        for level_index, level_model in enumerate(self.levels):
            setattr(self, f'level{level_index}', level_model)
        self.level0 = self.levels[0]
        self.level1 = self.levels[1]
        self._level0_phase = self._level_phases[0]
        self._level1_phase = self._level_phases[1]
        self.node_to_cluster_map = self.inter_level_unit_maps[0]
        self.token_to_meta_map = self.inter_level_meta_maps[0]
        self._last_level1_lifted_labels = (
            self._last_lifted_labels_by_level[0]
            if self._last_lifted_labels_by_level
            else None
        )

    def _ensure_level_count(self, min_count: int) -> None:
        target = max(2, int(min_count))
        while len(self.levels) < target:
            level_index = len(self.levels)
            if level_index == 0:
                level_size = self.input_size
            else:
                level_size = self.levels[level_index - 1].input_size
            upper_size = level_size
            level_kwargs = self._build_level_kwargs(
                self._init_level_kwargs,
                level=level_index,
                scale_factor=self.rg_scale_factor ** level_index,
            )
            self.levels.append(
                self.LEVEL_MODEL_CLASS(
                    input_size=level_size,
                    dimensions=self.dimensions,
                    **level_kwargs,
                    **self._shared_space_kwargs,
                )
            )
            self._level_phases.append(0)
            if level_index == 0:
                self._level_support_masks.append(np.ones(level_size, dtype=bool))
            else:
                self._level_support_masks.append(np.zeros(level_size, dtype=bool))
            if level_index > 0:
                lower_size = self.levels[level_index - 1].input_size
                self.inter_level_unit_maps.append(np.full(lower_size, -1, dtype=int))
                self.inter_level_meta_maps.append(np.full(upper_size, -1, dtype=int))
                self._last_lifted_labels_by_level.append(None)
                self._merge_evidence_by_level.append({})
                self._pair_relation_evidence_by_level.append({})
                self._last_source_support_tokens_by_level.append(tuple())
                self._last_meta_support_tokens_by_level.append(tuple())
                self._last_upper_cluster_tokens_by_level.append(tuple())
                self._last_upper_cluster_labels_by_level.append(None)
                self._last_upper_cluster_refresh_step_by_level.append(-1)
                self._last_upper_cluster_provenance_by_level.append(None)
                self._last_upper_input_by_level.append(np.zeros(upper_size, dtype=float))
                self._upper_history_by_level.append(deque(maxlen=max(1, int(getattr(self, 'rg_upper_history_window', 6)))))
                self._last_macro_feedback_strengths_by_level.append(np.zeros(lower_size, dtype=float))
                self._macro_stability_counts_by_level.append(np.zeros(upper_size, dtype=int))
                self._macro_member_signatures_by_level.append([None] * upper_size)
                self._next_inter_level_due_step_by_level.append(-1)
        self.level_count = len(self.levels)
        self._sync_public_level_aliases()

    def _stable_token_evidence_memory_steps_for_level(self, level_index: int) -> int:
        return 0

    def _refresh_stable_token_evidence_memory_steps(self) -> None:
        return

    @staticmethod
    def _normalize_level_index(level: str | int) -> int:
        if isinstance(level, (int, np.integer)):
            return max(0, int(level))
        text = str(level).strip().lower()
        if text.startswith('l') and text[1:].isdigit():
            return int(text[1:])
        if text.startswith('level') and text[5:].isdigit():
            return int(text[5:])
        raise ValueError(f"Unsupported level specifier: {level}")

    @staticmethod
    def _build_level_kwargs(
        kwargs: dict[str, Any],
        level: str | int = 'l0',
        scale_factor: float = 1.0,
    ) -> dict[str, Any]:
        """构建层级参数：共享同一动力学，只对允许的尺度参数做统一缩放。"""
        level_index = SyncMapX_RG._normalize_level_index(level)
        ns_defaults = SyncMapNS.default_params()
        rg_defaults = SyncMapX_RG.default_params()
        result: dict[str, Any] = {}

        # === 基础参数 ===
        result['adaptation_rate'] = float(kwargs.get('adaptation_rate', rg_defaults['adaptation_rate']))
        result['record_trajectory'] = bool(kwargs.get('record_trajectory', False)) if level_index == 0 else False

        # === 聚类和嵌入参数 ===
        result['embed_cluster_method'] = str(kwargs.get('embed_cluster_method', ns_defaults['embed_cluster_method']))
        time_scale_mode = str(kwargs.get('rg_time_scale_mode', rg_defaults['rg_time_scale_mode'])).strip().lower()
        time_scale_growth = float(kwargs.get('rg_time_scale_growth', rg_defaults['rg_time_scale_growth']))
        if time_scale_growth <= 1.0:
            time_scale_growth = max(float(kwargs.get('rg_scale_factor', rg_defaults['rg_scale_factor'])), 1.0)
        time_scale_multiplier = 1.0
        if time_scale_mode != 'uniform':
            time_scale_multiplier = float(time_scale_growth) ** int(level_index)
        base_t_push = max(1, int(kwargs.get('T_push', rg_defaults['T_push'])))
        base_clustering_period = max(0, int(kwargs.get('clustering_period', ns_defaults['clustering_period'])))
        result['T_push'] = max(1, int(round(base_t_push * time_scale_multiplier)))
        result['clustering_period'] = (
            0
            if base_clustering_period <= 0
            else max(1, int(round(base_clustering_period * time_scale_multiplier)))
        )
        base_graph_r0_fixed = float(
            kwargs.get(
                'graph_r0_fixed',
                kwargs.get('graph_r0_abs', kwargs.get('graph_r0', rg_defaults['graph_r0_fixed'])),
            )
        )
        level_graph_r0_fixed = max(base_graph_r0_fixed * scale_factor, 1e-6)
        dbscan_eps_scale_mode = str(
            kwargs.get('dbscan_eps_scale_mode', rg_defaults['dbscan_eps_scale_mode'])
        ).strip().lower()
        dbscan_eps_margin = max(
            float(kwargs.get('dbscan_eps_margin', rg_defaults['dbscan_eps_margin'])),
            1e-6,
        )
        readout_slack = max(float(kwargs.get('dbscan_readout_slack', rg_defaults['dbscan_readout_slack'])), 1e-6)

        # DBSCAN 参数：优先按层级距离律给出锚点，并允许在运行时用当前图实际形成的局部距离自适应修正。
        raw_dbscan_eps = kwargs.get('dbscan_eps', rg_defaults['dbscan_eps'])
        if dbscan_eps_scale_mode in {'geometric_midpoint', 'measured_midpoint'}:
            midpoint_gain = np.sqrt(max(time_scale_growth, 1.0))
            result['dbscan_eps'] = level_graph_r0_fixed * midpoint_gain * dbscan_eps_margin
        elif raw_dbscan_eps is None:
            result['dbscan_eps'] = None
        else:
            base_dbscan_eps = float(raw_dbscan_eps)
            if dbscan_eps_scale_mode == 'relative_to_r0':
                base_ratio = base_dbscan_eps / max(base_graph_r0_fixed, 1e-6)
                result['dbscan_eps'] = base_ratio * level_graph_r0_fixed
            else:
                result['dbscan_eps'] = base_dbscan_eps
        result['dbscan_auto_eps'] = False
        result['dbscan_auto_eps_source'] = 'knn95'
        result['dbscan_auto_eps_margin'] = dbscan_eps_margin
        if dbscan_eps_scale_mode == 'measured_midpoint':
            result['dbscan_auto_eps'] = True
            result['dbscan_auto_eps_source'] = 'graph_midpoint'
        raw_dbscan_readout_eps = kwargs.get('dbscan_readout_eps', rg_defaults['dbscan_readout_eps'])
        if raw_dbscan_readout_eps is not None:
            base_readout_eps = float(raw_dbscan_readout_eps)
            if dbscan_eps_scale_mode in {'geometric_midpoint', 'measured_midpoint'}:
                midpoint_gain = np.sqrt(max(time_scale_growth, 1.0))
                result['dbscan_readout_eps'] = level_graph_r0_fixed * midpoint_gain * dbscan_eps_margin
            elif dbscan_eps_scale_mode == 'relative_to_r0':
                base_ratio = base_readout_eps / max(base_graph_r0_fixed, 1e-6)
                result['dbscan_readout_eps'] = base_ratio * level_graph_r0_fixed
            else:
                result['dbscan_readout_eps'] = base_readout_eps
        elif dbscan_eps_scale_mode == 'measured_midpoint':
            result['dbscan_readout_eps'] = None
        elif result['dbscan_eps'] is None:
            result['dbscan_readout_eps'] = None
        else:
            result['dbscan_readout_eps'] = float(result['dbscan_eps']) * readout_slack
        result['dbscan_min_samples'] = int(kwargs.get('dbscan_min_samples', rg_defaults['dbscan_min_samples']))
        raw_dbscan_readout_min_samples = kwargs.get(
            'dbscan_readout_min_samples',
            rg_defaults['dbscan_readout_min_samples'],
        )
        result['dbscan_readout_min_samples'] = (
            None if raw_dbscan_readout_min_samples is None else int(raw_dbscan_readout_min_samples)
        )

        result['louvain_distance_factor'] = float(kwargs.get('louvain_distance_factor', ns_defaults['louvain_distance_factor']))
        result['graph_louvain_resolution'] = float(
            kwargs.get('graph_louvain_resolution', ns_defaults['graph_louvain_resolution'])
        )
        clustering_random_state = kwargs.get('clustering_random_state', ns_defaults['clustering_random_state'])
        result['clustering_random_state'] = (
            None if clustering_random_state is None else int(clustering_random_state)
        )

        # === 力学参数 ===
        k_pulse_base = float(kwargs.get('k_pulse', rg_defaults['k_pulse']))
        result['k_pulse'] = k_pulse_base
        
        # k_attr（弹簧刚度）在各层保持同形同参。
        # 层级差异只通过自然距离 graph_r0_fixed = base_r0 * scale_factor^level 体现，
        # 不再额外把高层弹簧软化。这样 L1/L2/L3 的力学形式完全一致。
        k_attr_base = float(kwargs.get('k_attr', rg_defaults['k_attr']))
        result['k_attr'] = k_attr_base
        
        # === 图spring参数 ===
        result['graph_spring_mode'] = str(kwargs.get('graph_spring_mode', rg_defaults['graph_spring_mode']))
        result['graph_degree_normalize'] = bool(kwargs.get('graph_degree_normalize', ns_defaults['graph_degree_normalize']))
        result['graph_weight_confidence_enabled'] = bool(
            kwargs.get(
                'graph_weight_confidence_enabled',
                rg_defaults['graph_weight_confidence_enabled'],
            )
        )
        result['graph_weight_confidence_percentile'] = float(
            kwargs.get(
                'graph_weight_confidence_percentile',
                rg_defaults['graph_weight_confidence_percentile'],
            )
        )
        result['graph_weight_confidence_power'] = float(
            kwargs.get(
                'graph_weight_confidence_power',
                rg_defaults['graph_weight_confidence_power'],
            )
        )
        result['graph_weight_confidence_floor'] = float(
            kwargs.get(
                'graph_weight_confidence_floor',
                rg_defaults['graph_weight_confidence_floor'],
            )
        )
        result['graph_weight_confidence_gain'] = float(
            kwargs.get(
                'graph_weight_confidence_gain',
                rg_defaults['graph_weight_confidence_gain'],
            )
        )
        result['macro_boundary_same_attr_scale'] = float(
            kwargs.get('macro_boundary_same_attr_scale', ns_defaults['macro_boundary_same_attr_scale'])
        )
        result['macro_boundary_same_attr_size_power'] = float(
            kwargs.get(
                'macro_boundary_same_attr_size_power',
                ns_defaults['macro_boundary_same_attr_size_power'],
            )
        )
        result['macro_boundary_coactivation_gate_enabled'] = bool(
            kwargs.get(
                'macro_boundary_coactivation_gate_enabled',
                ns_defaults['macro_boundary_coactivation_gate_enabled'],
            )
        )
        result['macro_boundary_coactivation_scale'] = float(
            kwargs.get(
                'macro_boundary_coactivation_scale',
                ns_defaults['macro_boundary_coactivation_scale'],
            )
        )
        result['macro_boundary_coactivation_floor'] = float(
            kwargs.get(
                'macro_boundary_coactivation_floor',
                ns_defaults['macro_boundary_coactivation_floor'],
            )
        )
        result['macro_boundary_softness_rel'] = float(
            kwargs.get('macro_boundary_softness_rel', ns_defaults['macro_boundary_softness_rel'])
        )
        result['graph_spring_repulse_gain'] = float(kwargs.get('graph_spring_repulse_gain', ns_defaults['graph_spring_repulse_gain']))
        result['graph_spring_repulse_power'] = float(kwargs.get('graph_spring_repulse_power', ns_defaults['graph_spring_repulse_power']))
        result['graph_spring_repulse_shoulder_gain'] = float(kwargs.get('graph_spring_repulse_shoulder_gain', ns_defaults['graph_spring_repulse_shoulder_gain']))
        result['graph_spring_extend_shoulder_gain'] = float(kwargs.get('graph_spring_extend_shoulder_gain', ns_defaults['graph_spring_extend_shoulder_gain']))
        result['graph_spring_extend_shoulder_power'] = float(kwargs.get('graph_spring_extend_shoulder_power', ns_defaults['graph_spring_extend_shoulder_power']))
        result['graph_spring_extend_shoulder_rel_start'] = float(kwargs.get('graph_spring_extend_shoulder_rel_start', ns_defaults['graph_spring_extend_shoulder_rel_start']))
        result['graph_r0_mode'] = str(kwargs.get('graph_r0_mode', ns_defaults['graph_r0_mode']))
        result['pulse_mode'] = str(kwargs.get('pulse_mode', rg_defaults['pulse_mode']))
        
        # graph_r0_rel: RG缩放时需要调整以维持合适的距离
        # graph_r0 = graph_r0_rel * median(nearest_neighbor_distances)
        # L1中，邻近距离会自动扩大（因为L1节点间隔更远），所以我们缩放graph_r0_rel
        base_graph_r0_rel = float(kwargs.get('graph_r0_rel', rg_defaults['graph_r0_rel']))
        if result['graph_r0_mode'] == 'fixed':
            result['graph_r0_rel'] = base_graph_r0_rel
        else:
            result['graph_r0_rel'] = base_graph_r0_rel * scale_factor
        base_graph_r0_fixed = float(kwargs.get('graph_r0_fixed', kwargs.get('graph_r0_abs', kwargs.get('graph_r0', rg_defaults['graph_r0_fixed']))))
        result['graph_r0_fixed'] = base_graph_r0_fixed * scale_factor

        # === 距离维持参数（RG核心）===
        # inter_cluster_distance_ratio：簇间距离与簇内距离的比例
        base_distance_ratio = float(kwargs.get('inter_cluster_distance_ratio', kwargs.get('rg_scale_factor', rg_defaults['inter_cluster_distance_ratio'])))
        result['inter_cluster_distance_ratio'] = base_distance_ratio  # 比例本身不缩放

        result['center_on_origin'] = bool(kwargs.get('center_on_origin', ns_defaults['center_on_origin']))
        result['recenter_graph_support_only'] = bool(kwargs.get('recenter_graph_support_only', level_index > 0))
        result['centering_strength'] = float(kwargs.get('centering_strength', rg_defaults['centering_strength'])) * scale_factor
        result['cluster_distance_strength'] = float(kwargs.get('cluster_distance_strength', rg_defaults['cluster_distance_strength']))
        result['macro_centroid_distance_ratio'] = float(
            kwargs.get(
                'macro_centroid_distance_ratio',
                kwargs.get(
                    'cluster_exclusion_distance_ratio',
                    kwargs.get('inter_cluster_distance_ratio', rg_defaults['macro_centroid_distance_ratio']),
                ),
            )
        )
        result['k_macro_repulse'] = float(
            kwargs.get('k_macro_repulse', kwargs.get('cluster_distance_strength', rg_defaults['k_macro_repulse']))
        )
        result['macro_centroid_max_step_ratio'] = float(
            kwargs.get(
                'macro_centroid_max_step_ratio',
                kwargs.get('cluster_exclusion_max_step_ratio', rg_defaults['macro_centroid_max_step_ratio']),
            )
        )
        result['entity_boundary_repulsion_enabled'] = bool(
            kwargs.get('entity_boundary_repulsion_enabled', rg_defaults['entity_boundary_repulsion_enabled'])
        )
        result['entity_boundary_distance_ratio'] = float(
            kwargs.get(
                'entity_boundary_distance_ratio',
                kwargs.get('rg_scale_factor', rg_defaults['entity_boundary_distance_ratio']),
            )
        )
        result['k_entity_boundary_repulse'] = float(
            kwargs.get('k_entity_boundary_repulse', rg_defaults['k_entity_boundary_repulse'])
        )
        result['entity_boundary_min_gap_ratio'] = float(
            kwargs.get('entity_boundary_min_gap_ratio', rg_defaults['entity_boundary_min_gap_ratio'])
        )
        result['entity_boundary_max_step_ratio'] = float(
            kwargs.get('entity_boundary_max_step_ratio', rg_defaults['entity_boundary_max_step_ratio'])
        )
        result['entity_sibling_attraction_enabled'] = bool(
            kwargs.get('entity_sibling_attraction_enabled', rg_defaults['entity_sibling_attraction_enabled'])
        )
        result['entity_sibling_target_ratio'] = float(
            kwargs.get(
                'entity_sibling_target_ratio',
                kwargs.get('rg_scale_factor', rg_defaults['entity_sibling_target_ratio']),
            )
        )
        result['entity_sibling_capture_ratio'] = float(
            kwargs.get(
                'entity_sibling_capture_ratio',
                kwargs.get('rg_scale_factor', rg_defaults['entity_sibling_capture_ratio']),
            )
        )
        result['k_entity_sibling_attr'] = float(
            kwargs.get('k_entity_sibling_attr', rg_defaults['k_entity_sibling_attr'])
        )
        result['entity_sibling_max_step_ratio'] = float(
            kwargs.get('entity_sibling_max_step_ratio', rg_defaults['entity_sibling_max_step_ratio'])
        )
        result['entity_sibling_boundary_attenuation_enabled'] = bool(
            kwargs.get(
                'entity_sibling_boundary_attenuation_enabled',
                rg_defaults['entity_sibling_boundary_attenuation_enabled'],
            )
        )
        result['entity_sibling_boundary_release_ratio'] = float(
            kwargs.get(
                'entity_sibling_boundary_release_ratio',
                rg_defaults['entity_sibling_boundary_release_ratio'],
            )
        )
        result['entity_sibling_boundary_power'] = float(
            kwargs.get('entity_sibling_boundary_power', rg_defaults['entity_sibling_boundary_power'])
        )
        result['entity_sibling_deadband_ratio'] = float(
            kwargs.get('entity_sibling_deadband_ratio', rg_defaults['entity_sibling_deadband_ratio'])
        )
        result['macro_cross_attr_scale'] = float(
            kwargs.get('macro_cross_attr_scale', rg_defaults['macro_cross_attr_scale'])
        )
        result['line_obstacle_repulsion_enabled'] = bool(
            kwargs.get('line_obstacle_repulsion_enabled', ns_defaults['line_obstacle_repulsion_enabled'])
        )
        result['line_obstacle_radius_rel'] = float(
            kwargs.get('line_obstacle_radius_rel', ns_defaults['line_obstacle_radius_rel'])
        )
        result['line_obstacle_strength'] = float(
            kwargs.get('line_obstacle_strength', ns_defaults['line_obstacle_strength'])
        )
        result['primitive_contrastive_force_enabled'] = bool(
            kwargs.get('primitive_contrastive_force_enabled', rg_defaults['primitive_contrastive_force_enabled'])
        )
        result['primitive_active_attraction_strength'] = float(
            kwargs.get('primitive_active_attraction_strength', rg_defaults['primitive_active_attraction_strength'])
        )
        result['primitive_inactive_repulsion_strength'] = float(
            kwargs.get('primitive_inactive_repulsion_strength', rg_defaults['primitive_inactive_repulsion_strength'])
        )
        result['primitive_force_distance_floor_rel'] = float(
            kwargs.get('primitive_force_distance_floor_rel', rg_defaults['primitive_force_distance_floor_rel'])
        )
        result['primitive_force_max_step_ratio'] = float(
            kwargs.get('primitive_force_max_step_ratio', rg_defaults['primitive_force_max_step_ratio'])
        )
        result['negative_evidence_confidence_enabled'] = bool(
            kwargs.get(
                'negative_evidence_confidence_enabled',
                ns_defaults['negative_evidence_confidence_enabled'],
            )
        )
        result['negative_evidence_confidence_level'] = float(
            kwargs.get(
                'negative_evidence_confidence_level',
                ns_defaults['negative_evidence_confidence_level'],
            )
        )
        result['project_close_pairs'] = bool(kwargs.get('project_close_pairs', ns_defaults['project_close_pairs']))
        result['project_close_pairs_gain'] = float(kwargs.get('project_close_pairs_gain', ns_defaults['project_close_pairs_gain']))
        result['edge_stretch_guard_enabled'] = bool(kwargs.get('edge_stretch_guard_enabled', ns_defaults['edge_stretch_guard_enabled']))
        result['edge_stretch_max_step'] = float(kwargs.get('edge_stretch_max_step', ns_defaults['edge_stretch_max_step']))
        result['edge_stretch_max_step_rel'] = float(kwargs.get('edge_stretch_max_step_rel', ns_defaults['edge_stretch_max_step_rel']))
        result['edge_detach_distance'] = float(kwargs.get('edge_detach_distance', ns_defaults['edge_detach_distance']))
        result['edge_detach_distance_rel'] = float(kwargs.get('edge_detach_distance_rel', ns_defaults['edge_detach_distance_rel']))
        
        # === 可塑性参数 ===
        base_stats_memory_steps = int(kwargs.get('stats_memory_steps', ns_defaults['stats_memory_steps']))
        stats_memory_scale = max(1.0, float(kwargs.get('stats_memory_scale', rg_defaults['stats_memory_scale'])))
        if level_index > 0:
            result['stats_mode'] = str(kwargs.get('rg_upper_stats_mode', rg_defaults['rg_upper_stats_mode']))
        else:
            result['stats_mode'] = str(kwargs.get('stats_mode', rg_defaults['stats_mode']))
        result['stats_memory_steps'] = max(
            1,
            int(
                int(round(base_stats_memory_steps * (stats_memory_scale ** level_index)))
            ),
        )
        result['stats_power_gain'] = float(kwargs.get('stats_power_gain', ns_defaults['stats_power_gain']))
        result['weighted_stats_input'] = bool(kwargs.get('weighted_stats_input', level_index > 0))

        # === 步长限制（尺度参数） ===
        result['max_step_norm'] = float(kwargs.get('max_step_norm', rg_defaults['max_step_norm'])) * scale_factor

        return result

    @property
    def syncmap(self) -> NDArray[np.float64]:
        """Expose Level 0 embedding for compatibility with existing tools."""
        return self.level0.syncmap

    @syncmap.setter
    def syncmap(self, value: NDArray[np.float64]) -> None:
        """Set Level 0 embedding (for trajectory playback)."""
        self.level0.syncmap = value

    @property
    def labels(self) -> NDArray[np.integer[Any]] | None:
        """Expose Level 0 cluster labels for compatibility with live UI."""
        return self.level0.labels

    @labels.setter
    def labels(self, value: Any) -> None:
        self.level0.labels = None if value is None else np.asarray(value, dtype=int)

    def _level_model(self, level: str | int) -> SyncMapNS:
        return self.levels[self._normalize_level_index(level)]

    def _resolved_time_scale_growth(self) -> float:
        growth = float(self.rg_time_scale_growth)
        if growth <= 1.0:
            growth = max(float(self.rg_scale_factor), 1.0)
        return growth

    @staticmethod
    def _normalize_inter_level_schedule_mode(value: Any) -> str:
        mode = str(value).strip().lower()
        if mode not in {'nested_modulo', 'elapsed_geometric'}:
            raise ValueError(
                "rg_inter_level_schedule_mode must be 'nested_modulo' or "
                f"'elapsed_geometric', got {value!r}."
            )
        return mode

    def _time_scale_multiplier(self, level_index: int) -> float:
        if str(self.rg_time_scale_mode).strip().lower() == 'uniform':
            return 1.0
        return float(self._resolved_time_scale_growth()) ** int(max(0, level_index))

    def _renormalize_interval_for_level(self, lower_level_index: int) -> int:
        base_interval = max(1, int(self.renormalize_interval))
        multiplier = self._time_scale_multiplier(lower_level_index)
        return max(1, int(round(base_interval * multiplier)))

    def _reset_inter_level_schedule(self) -> None:
        target = max(0, len(self.levels) - 1)
        self._next_inter_level_due_step_by_level = [-1] * target

    def _inter_level_schedule_due(self, lower_level_index: int) -> bool:
        """Consume one interface deadline when the current raw step is due.

        The legacy mode deliberately preserves the historical nested global
        modulo behavior.  The experimental elapsed mode keeps an absolute
        raw-input-step deadline for every interface.  A deeper interface is
        still evaluated only when its lower interface emits, so deadlines are
        quantized to the lower interface's event grid, but they no longer need
        to share an exact integer divisor with that grid.
        """
        level_index = max(0, int(lower_level_index))
        interval = self._renormalize_interval_for_level(level_index)
        if self.rg_inter_level_schedule_mode == 'nested_modulo':
            return int(self.step_count) % interval == 0

        target = max(0, len(self.levels) - 1)
        if len(self._next_inter_level_due_step_by_level) < target:
            self._next_inter_level_due_step_by_level.extend(
                [-1] * (target - len(self._next_inter_level_due_step_by_level))
            )
        if level_index >= target:
            return False

        next_due = int(self._next_inter_level_due_step_by_level[level_index])
        if next_due < 0:
            warmup_anchor = max(0, int(self.warmup_steps))
            if level_index == 0:
                next_due = ((warmup_anchor + interval - 1) // interval) * interval
            else:
                next_due = warmup_anchor + interval

        current_step = int(self.step_count)
        if current_step < next_due:
            self._next_inter_level_due_step_by_level[level_index] = next_due
            return False

        # Advance from the ideal deadline, rather than from the quantized
        # firing step, so 150/200-step alternation can average to 158 instead
        # of drifting to a fixed 200-step cadence.
        elapsed_intervals = ((current_step - next_due) // interval) + 1
        self._next_inter_level_due_step_by_level[level_index] = (
            next_due + elapsed_intervals * interval
        )
        return True

    def _level_equilibrium_distance(self, level: str | int) -> float:
        model = self._level_model(level)
        if str(getattr(model, 'graph_r0_mode', 'relative')).lower() == 'fixed':
            return max(float(getattr(model, 'graph_r0_fixed', getattr(model, 'graph_r0', 1.0))), 1e-6)
        return max(float(getattr(model, 'graph_r0', 1.0)), 1e-6)

    def input(self, x_batch: NDArray[np.float64]) -> None:
        """Consume raw input while keeping Level 0 stepping comparable to plain NSGD."""
        T = int(x_batch.shape[0])
        interval = self._renormalize_interval_for_level(0)
        
        t = 0
        while t < T:
            if self.level1_enabled and (self.step_count % interval == 0):
                steps_to_process = 1
            else:
                steps_to_process = interval - (self.step_count % interval)
                
            steps_to_process = min(steps_to_process, T - t)
            
            chunk = x_batch[t : t + steps_to_process]
            self.level0.inputGeneral(
                chunk,
                phase_offset=self._level_phases[0],
                reset_trajectory=(self._level_phases[0] == 0),
                append_final_frame=False,
            )
            self._level_phases[0] += steps_to_process
            self._level0_phase = self._level_phases[0]

            if self.level1_enabled and (self.step_count % interval == 0):
                active_indices = np.where(chunk[0] > 0.1)[0]
                self._propagate_between_levels(0, active_indices)

            self.step_count += steps_to_process
            t += steps_to_process

    def _observe_stable_token_evidence_from_raw_chunk(self, chunk: FloatArray) -> None:
        return

    @staticmethod
    def _fill_noise_labels_from_geometry(
        positions: FloatArray,
        labels: IntArray,
    ) -> IntArray:
        pos = np.asarray(positions, dtype=float)
        arr = np.asarray(labels, dtype=int).reshape(-1)
        if pos.shape[0] != arr.shape[0]:
            return arr.copy()
        if not np.any(arr < 0):
            return arr.copy()

        valid_labels = np.array(sorted({int(label) for label in arr.tolist() if int(label) >= 0}), dtype=int)
        if valid_labels.size == 0:
            return arr.copy()

        filled = arr.copy()
        centroids = np.zeros((valid_labels.size, pos.shape[1]), dtype=float)
        for idx, label in enumerate(valid_labels.tolist()):
            members = filled == int(label)
            if not np.any(members):
                continue
            centroids[idx] = np.mean(pos[members], axis=0)

        noise_indices = np.flatnonzero(filled < 0)
        if noise_indices.size:
            delta = centroids[None, :, :] - pos[noise_indices][:, None, :]
            distances = np.linalg.norm(delta, axis=2)
            best_idx = np.argmin(distances, axis=1)
            filled[noise_indices] = valid_labels[best_idx]
        return filled

    def _normalized_feedback_mode(self) -> str:
        mode = str(self.feedback_mode).strip().lower()
        if mode not in {'none', 'soft', 'rigid'}:
            return 'soft'
        return mode

    def _feedback_strength_scale(self) -> float:
        mode = self._normalized_feedback_mode()
        if mode == 'none':
            return 0.0
        if mode == 'soft':
            return 0.35
        return 1.0

    def _feedback_allows_barrier(self) -> bool:
        return self._normalized_feedback_mode() == 'rigid'

    def _current_raw_labels(self, level_index: int) -> IntArray:
        self._ensure_level_count(level_index + 1)
        level_model = self.levels[level_index]
        if level_index == 0:
            physical_labels = None
            if self.stable_token_extraction_mode == 'physical_mst_gaps' and bool(getattr(self, 'level1_enabled', False)):
                physical_levels = self._physical_mst_hierarchy_readouts(max(1, self.level_count - 1))
                if physical_levels:
                    physical_labels = np.asarray(physical_levels[0], dtype=int)
            if physical_labels is not None and physical_labels.shape[0] == level_model.input_size:
                raw = physical_labels
                level_model.labels = raw.copy()
                level_model._labels_version += 1
            elif level_model.labels is not None:
                raw = np.asarray(level_model.labels, dtype=int)
            else:
                raw = np.asarray(level_model.organize(persist=True), dtype=int)
        else:
            support_mask = np.asarray(self._level_support_masks[level_index], dtype=bool)
            raw = np.full(level_model.input_size, -1, dtype=int)
            support_idx = np.flatnonzero(support_mask)
            if support_idx.size == 1:
                raw[support_idx[0]] = 0
                level_model.labels = raw
                level_model._labels_version += 1
            elif support_idx.size > 1:
                support_positions = np.asarray(level_model.syncmap[support_idx], dtype=float)
                try:
                    support_labels = np.asarray(
                        self._cluster_positions_for_level(
                            level_model,
                            support_positions,
                            support_indices=support_idx,
                            persist=False,
                        ),
                        dtype=int,
                    )
                except Exception:
                    support_labels = np.zeros(support_idx.size, dtype=int)
                if np.any(support_labels < 0):
                    next_label = int(np.max(support_labels[support_labels >= 0])) + 1 if np.any(support_labels >= 0) else 0
                    support_labels = support_labels.copy()
                    for idx in np.flatnonzero(support_labels < 0):
                        support_labels[int(idx)] = int(next_label)
                        next_label += 1
                raw[support_idx] = support_labels
                level_model.labels = raw
                level_model._labels_version += 1
            else:
                level_model.labels = raw
                level_model._labels_version += 1
        if self.rg_fill_noise_labels:
            raw = self._fill_noise_labels_from_geometry(level_model.syncmap, raw)
            level_model.labels = raw.copy()
        if level_index == 0:
            self._last_level0_raw_labels = raw.copy()
        return raw

    def _current_level0_raw_labels(self) -> IntArray:
        return self._current_raw_labels(0)

    def _lift_native_level_labels_to_nodes(self, level_index: int) -> IntArray:
        self._ensure_level_count(level_index + 1)
        if level_index <= 0:
            labels = np.asarray(self._current_level0_readout_labels(), dtype=int)
            self._last_level0_raw_labels = labels.copy()
            return labels.copy()

        source_maps: list[IntArray] = []
        for lower_index in range(level_index):
            raw_labels = self._current_raw_labels(lower_index)
            stable_tokens = self._update_stable_tokens_for_level(lower_index, raw_labels)
            source_maps.append(
                np.asarray(
                    self._resolved_source_tokens_for_level(lower_index, raw_labels, stable_tokens),
                    dtype=int,
                )
            )

        level_model = self.levels[level_index]
        support_idx = np.unique(source_maps[level_index - 1][source_maps[level_index - 1] >= 0]).astype(int)
        level_labels = np.full(level_model.input_size, -1, dtype=int)
        if support_idx.size == 1:
            level_labels[support_idx[0]] = 0
        elif support_idx.size > 1:
            support_positions = np.asarray(level_model.syncmap[support_idx], dtype=float)
            support_labels = np.asarray(
                self._cluster_positions_for_level(
                    level_model,
                    support_positions,
                    support_indices=support_idx,
                    persist=False,
                    readout=True,
                ),
                dtype=int,
            )
            if self.rg_fill_noise_labels and np.any(support_labels < 0):
                support_labels = self._fill_noise_labels_from_geometry(support_positions, support_labels)
            level_labels[support_idx] = np.asarray(support_labels, dtype=int)
        elif level_model.labels is not None:
            level_labels = np.asarray(level_model.labels, dtype=int).copy()

        lifted = self._lift_labels_through_source_tokens(source_maps[level_index - 1], level_labels)
        for lower_index in range(level_index - 2, -1, -1):
            lifted = self._lift_labels_through_source_tokens(source_maps[lower_index], lifted)

        self._last_lifted_labels_by_level[level_index - 1] = np.asarray(lifted, dtype=int).copy()
        self._sync_public_level_aliases()
        return np.asarray(lifted, dtype=int).copy()

    def _current_level0_readout_labels(self) -> IntArray:
        level_model = self.levels[0]
        readout_eps = getattr(level_model, 'dbscan_readout_eps', None)
        readout_min_samples = getattr(level_model, 'dbscan_readout_min_samples', None)
        if readout_eps is None and readout_min_samples is None:
            labels = self._current_raw_labels(0)
            self._last_level0_raw_labels = labels.copy()
            return labels.copy()
        labels = np.asarray(
            self._cluster_positions_for_level(
                level_model,
                np.asarray(level_model.syncmap, dtype=float),
                support_indices=None,
                persist=False,
                readout=True,
            ),
            dtype=int,
        )
        if self.rg_fill_noise_labels:
            labels = self._fill_noise_labels_from_geometry(level_model.syncmap, labels)
        self._last_level0_raw_labels = labels.copy()
        return labels.copy()

    def _empirical_level_distance_scale(
        self,
        level_model: SyncMapNS,
        samples: FloatArray,
        *,
        support_indices: IntArray | None = None,
    ) -> float | None:
        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.ndim != 2 or sample_arr.shape[0] <= 1:
            return None
        rows = getattr(level_model, '_g_rows', None)
        cols = getattr(level_model, '_g_cols', None)
        if rows is not None and cols is not None:
            row_arr = np.asarray(rows, dtype=int).reshape(-1)
            col_arr = np.asarray(cols, dtype=int).reshape(-1)
            if support_indices is not None:
                support_arr = np.asarray(support_indices, dtype=int).reshape(-1)
                support_set = {int(idx) for idx in support_arr.tolist()}
                if support_set:
                    pairs = [
                        (int(i), int(j))
                        for i, j in zip(row_arr.tolist(), col_arr.tolist())
                        if int(i) in support_set and int(j) in support_set
                    ]
                    if pairs:
                        index_map = {int(src): local_idx for local_idx, src in enumerate(support_arr.tolist())}
                        distances = []
                        for src, dst in pairs:
                            local_i = index_map.get(int(src))
                            local_j = index_map.get(int(dst))
                            if local_i is None or local_j is None:
                                continue
                            distances.append(
                                float(np.linalg.norm(sample_arr[local_i] - sample_arr[local_j]))
                            )
                        valid = np.asarray([d for d in distances if d > 1e-9], dtype=float)
                        if valid.size > 0:
                            return float(np.median(valid))
            else:
                if row_arr.size > 0 and col_arr.size > 0 and sample_arr.shape[0] == level_model.input_size:
                    diff = sample_arr[row_arr] - sample_arr[col_arr]
                    distances = np.linalg.norm(diff, axis=1)
                    valid = distances[np.isfinite(distances) & (distances > 1e-9)]
                    if valid.size > 0:
                        return float(np.median(valid))

        if sample_arr.shape[0] >= 2:
            diff = sample_arr[:, None, :] - sample_arr[None, :, :]
            dmat = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dmat, np.inf)
            nearest = np.min(dmat, axis=1)
            valid = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
            if valid.size > 0:
                return float(np.median(valid))
        return None

    def _dynamic_cluster_eps(
        self,
        level_model: SyncMapNS,
        samples: FloatArray,
        *,
        support_indices: IntArray | None = None,
        readout: bool = False,
    ) -> float:
        measured = self._empirical_level_distance_scale(
            level_model,
            samples,
            support_indices=support_indices,
        )
        if measured is None:
            measured = max(float(getattr(level_model, 'graph_r0', getattr(level_model, 'graph_r0_fixed', 1.0))), 1e-6)
        growth = max(float(getattr(level_model, 'inter_cluster_distance_ratio', self.rg_scale_factor)), 1.0)
        margin = max(float(getattr(self, 'dbscan_eps_margin', 1.05)), 1e-6)
        eps = float(measured) * np.sqrt(growth) * margin
        if readout:
            slack = max(float(getattr(self, 'dbscan_readout_slack', 1.0)), 1e-6)
            eps *= slack
        return max(float(eps), 1e-4)

    @staticmethod
    def _canonical_readout_labels(labels: IntArray) -> IntArray:
        values = np.asarray(labels, dtype=int).reshape(-1)
        result = np.full(values.shape, -1, dtype=int)
        next_label = 0
        mapping: dict[int, int] = {}
        for index, value in enumerate(values.tolist()):
            if int(value) < 0:
                continue
            if int(value) not in mapping:
                mapping[int(value)] = next_label
                next_label += 1
            result[index] = int(mapping[int(value)])
        return result

    @staticmethod
    def _mst_cut_labels_for_low_min_samples(
        mst: nx.Graph,
        node_count: int,
        eps: float,
        min_samples: int,
    ) -> IntArray | None:
        """Return DBSCAN-equivalent epsilon components for min_samples <= 2.

        Thresholding a complete-graph MST preserves the epsilon-connected
        components of the full distance graph. With DBSCAN min_samples of
        one or two, those components differ only by singleton noise handling.
        """
        if int(min_samples) > 2:
            return None
        cut_graph = nx.Graph()
        cut_graph.add_nodes_from(range(int(node_count)))
        cut_graph.add_edges_from(
            (int(i), int(j))
            for i, j, data in mst.edges(data=True)
            if float(data['weight']) <= float(eps)
        )
        labels = np.full(int(node_count), -1, dtype=int)
        next_label = 0
        for component in nx.connected_components(cut_graph):
            members = np.asarray(sorted(int(node) for node in component), dtype=int)
            if members.size < int(min_samples):
                continue
            labels[members] = next_label
            next_label += 1
        return labels

    @staticmethod
    def _mst_threshold_component_labels(
        mst_edges: IntArray,
        mst_lengths: FloatArray,
        node_count: int,
        eps: float,
        min_samples: int,
    ) -> IntArray:
        """Epsilon components of the complete distance graph via MST threshold.

        Same contract as _mst_cut_labels_for_low_min_samples but on a dense
        Prim MST edge list (no networkx): thresholding any MST of the complete
        graph yields the epsilon-connected components of the full graph, and
        component ids are assigned in first-member order like
        nx.connected_components.
        """
        n = int(node_count)
        parent = np.arange(n, dtype=int)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        keep = np.asarray(mst_lengths, dtype=float) <= float(eps)
        for a, b in np.asarray(mst_edges, dtype=int)[keep]:
            root_a, root_b = find(int(a)), find(int(b))
            if root_a != root_b:
                parent[root_b] = root_a
        roots = parent
        while True:
            hopped = roots[roots]
            if np.array_equal(hopped, roots):
                break
            roots = hopped
        counts = np.bincount(roots, minlength=n)
        sample_floor = int(min_samples)
        root_to_label = np.full(n, -1, dtype=int)
        next_label = 0
        for node in range(n):
            root = int(roots[node])
            if root_to_label[root] == -1 and counts[root] >= sample_floor:
                root_to_label[root] = next_label
                next_label += 1
        return np.where(counts[roots] >= sample_floor, root_to_label[roots], -1)

    def _physical_mst_hierarchy_readouts(self, required_levels: int) -> list[IntArray]:
        """Read hierarchy cuts from distance gaps in the final physical embedding."""
        positions = np.asarray(self.levels[0].syncmap, dtype=float)
        node_count = int(positions.shape[0]) if positions.ndim == 2 else 0
        if node_count < 2 or required_levels <= 0:
            return []

        # Same-state memoization: this readout is a pure function of the
        # level-0 geometry, yet propagation, refresh, and organize each call
        # it within the same step (421 calls per 10k steps profiled on DCH1).
        cache = getattr(self, "_mst_readout_cache", None)
        if cache is None:
            cache = {}
            self._mst_readout_cache = cache
        cache_key = (int(required_levels), hash(positions.tobytes()))
        cached = cache.get(cache_key)
        if cached is not None:
            return [np.asarray(arr, dtype=int).copy() for arr in cached]

        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2,
        )
        mst_edges, mst_lengths = _dense_mst(distances)
        if mst_lengths.shape[0] < 2:
            return []
        edge_lengths = np.sort(np.asarray(mst_lengths, dtype=float)).tolist()

        min_samples = max(1, int(getattr(self.levels[0], 'dbscan_min_samples', 2)))
        min_ratio = max(float(self.hierarchy_readout_min_gap_ratio), 1.0)
        candidates: list[tuple[float, int, IntArray]] = []
        seen: set[tuple[int, ...]] = set()
        for lower, upper in zip(edge_lengths[:-1], edge_lengths[1:]):
            if lower <= 1e-12:
                continue
            jump_ratio = float(upper / lower)
            if jump_ratio < min_ratio:
                continue
            eps = float(np.sqrt(lower * upper))
            labels = None
            if min_samples <= 2:
                labels = self._mst_threshold_component_labels(
                    mst_edges,
                    mst_lengths,
                    node_count,
                    eps,
                    min_samples,
                )
            if labels is None:
                labels = np.asarray(
                    create_clustering_strategy(
                        'dbscan',
                        eps=eps,
                        min_samples=min_samples,
                    ).fit_predict(positions),
                    dtype=int,
                )
            if np.any(labels < 0):
                # Fill noise into the nearest cluster instead of discarding the cut.
                # Rejecting any cut with a noise point threw away exactly the
                # fine-scale cuts (which routinely leave a singleton), starving the
                # tokenizer of fine tokens and leaving L1 with too few degrees of
                # freedom to separate macros (the 3-token "three-body" deadlock).
                labels = self._fill_noise_labels_from_geometry(positions, labels)
            labels = self._canonical_readout_labels(labels)
            unique, counts = np.unique(labels, return_counts=True)
            cluster_count = int(unique.size)
            if cluster_count <= 1 or cluster_count >= node_count:
                continue
            if np.any(counts < min_samples):
                continue
            signature = tuple(int(value) for value in labels.tolist())
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((jump_ratio, cluster_count, labels))

        if not candidates:
            if len(cache) > 8:
                cache.clear()
            cache[cache_key] = []
            return []
        # The coarse hierarchy levels come from the largest distance jumps (macro
        # scale). But the FINEST level (index 0) is ALSO the tokenizer's fine
        # segmentation fed to L1, so it must capture the fine scale (~K_fine
        # clusters), not just the coarse jumps — otherwise L1 receives only a handful
        # of tokens and cannot form macro structure (a fully-connected 3-token graph
        # cannot be split into 2). Force the finest cut (max cluster count) to be the
        # fine level; fill the coarser levels from the largest jumps.
        finest = max(candidates, key=lambda item: item[1])
        ordered = [finest] + sorted(candidates, key=lambda item: (-item[0], -item[1]))
        result: list[tuple[float, int, IntArray]] = []
        seen_counts: set[int] = set()
        for cand in sorted(ordered, key=lambda item: -item[1]):
            if cand[1] in seen_counts:
                continue
            seen_counts.add(cand[1])
            result.append(cand)
            if len(result) >= required_levels:
                break
        result_labels = [np.asarray(labels, dtype=int).copy() for _, _, labels in result]
        if len(cache) > 8:
            cache.clear()
        cache[cache_key] = [arr.copy() for arr in result_labels]
        return result_labels

    def _cluster_positions_for_level(
        self,
        level_model: SyncMapNS,
        samples: FloatArray,
        *,
        support_indices: IntArray | None,
        persist: bool,
        readout: bool = False,
    ) -> IntArray:
        method = str(getattr(level_model, 'embed_cluster_method', 'dbscan')).strip().lower()
        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.size == 0:
            return np.asarray([], dtype=int)
        if method == 'dbscan':
            eps = self._dynamic_cluster_eps(
                level_model,
                sample_arr,
                support_indices=support_indices,
                readout=readout,
            )
            min_samples = int(
                getattr(
                    level_model,
                    'dbscan_readout_min_samples' if readout else 'dbscan_min_samples',
                    getattr(level_model, 'dbscan_min_samples', 2),
                ) or getattr(level_model, 'dbscan_min_samples', 2)
            )
            strategy = create_clustering_strategy(
                'dbscan',
                eps=float(eps),
                min_samples=max(1, int(min_samples)),
            )
            labels = np.asarray(strategy.fit_predict(sample_arr), dtype=int)
            if persist:
                level_model.dbscan_eps = float(eps)
            return labels
        if method == 'hdbscan':
            min_samples = int(
                getattr(
                    level_model,
                    'dbscan_readout_min_samples' if readout else 'dbscan_min_samples',
                    getattr(level_model, 'dbscan_min_samples', 2),
                ) or getattr(level_model, 'dbscan_min_samples', 2)
            )
            strategy = create_clustering_strategy(
                'hdbscan',
                min_cluster_size=max(2, int(min_samples)),
                min_samples=max(1, int(min_samples)),
            )
            return np.asarray(strategy.fit_predict(sample_arr), dtype=int)
        if readout:
            full_labels = np.asarray(level_model.organize(persist=False), dtype=int)
            if support_indices is not None:
                idx = np.asarray(support_indices, dtype=int).reshape(-1)
                if idx.size < full_labels.shape[0] and np.all((idx >= 0) & (idx < full_labels.shape[0])):
                    return full_labels[idx]
            return full_labels
        return np.asarray(level_model.clustering_strategy.fit_predict(sample_arr), dtype=int)

    def _set_level_support(self, level_index: int, active_units: IntArray) -> None:
        self._ensure_level_count(level_index + 1)
        level_model = self.levels[level_index]
        mask = np.zeros(level_model.input_size, dtype=bool)
        active = np.asarray(active_units, dtype=int).reshape(-1)
        valid = active[(active >= 0) & (active < level_model.input_size)]
        if level_index == 0 and valid.size == 0:
            mask[:] = True
        elif valid.size > 0:
            mask[np.unique(valid)] = True
        self._level_support_masks[level_index] = mask

        if level_model.labels is not None and level_model.labels.shape[0] == mask.size:
            labels = np.asarray(level_model.labels, dtype=int).copy()
            labels[~mask] = -1
            level_model.labels = labels

    def _allocate_slot(self, capacity: int, forbidden: set[int], previous_ids: IntArray) -> int:
        previous_active = {int(token) for token in previous_ids if token >= 0}
        for token in range(capacity):
            if token not in forbidden and token not in previous_active:
                return token
        for token in range(capacity):
            if token not in forbidden:
                return token
        return 0

    def _should_keep_raw_merge_split(
        self,
        level_index: int,
        previous_tokens: IntArray,
        members: IntArray,
        merge_key: tuple[int, ...],
        next_merge_evidence: dict[tuple[int, ...], int],
    ) -> bool:
        if not self.unit_retention_enabled or len(merge_key) < 2:
            return False

        lower_model = self.levels[level_index]
        positions = np.asarray(lower_model.syncmap, dtype=float)
        merge_threshold = max(
            float(self._level_equilibrium_distance(level_index)) * float(self.unit_merge_distance_ratio),
            1e-6,
        )
        min_cross_distance = float("inf")
        for idx_i, token_i in enumerate(merge_key[:-1]):
            pts_i = positions[members[previous_tokens[members] == token_i]]
            if pts_i.size == 0:
                continue
            for token_j in merge_key[idx_i + 1:]:
                pts_j = positions[members[previous_tokens[members] == token_j]]
                if pts_j.size == 0:
                    continue
                pairwise = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=2)
                if pairwise.size:
                    min_cross_distance = min(min_cross_distance, float(np.min(pairwise)))

        collapsed = np.isfinite(min_cross_distance) and min_cross_distance <= merge_threshold
        previous_evidence = self._merge_evidence_by_level[level_index].get(merge_key, 0)
        current_evidence = previous_evidence + 1 if collapsed else 0
        next_merge_evidence[merge_key] = current_evidence
        return current_evidence < self.unit_merge_hysteresis_steps

    def _should_debounce_stable_token_merge(
        self,
        level_index: int,
        previous_tokens: IntArray,
        members: IntArray,
        merge_key: tuple[int, ...],
        next_merge_evidence: dict[tuple[int, ...], int],
    ) -> bool:
        return False

    def _stable_token_is_compact_for_merge_debounce(
        self,
        level_index: int,
        previous_tokens: IntArray,
        token_id: int,
    ) -> bool:
        return False

    def _should_keep_evidence_supported_split(
        self,
        level_index: int,
        previous_tokens: IntArray,
        members: IntArray,
        merge_key: tuple[int, ...],
    ) -> bool:
        return False

    def _stable_token_semantic_evidence_matrices(self, level_index: int) -> tuple[FloatArray, FloatArray | None]:
        lower_model = self.levels[level_index]
        coact = getattr(lower_model, '_coact', None)
        if coact is None:
            return np.zeros((lower_model.input_size, lower_model.input_size), dtype=float), None
        try:
            similarity = np.asarray(coact.similarity(), dtype=float)
        except Exception:
            return np.zeros((lower_model.input_size, lower_model.input_size), dtype=float), None
        transition_mass = getattr(coact, 'C', None)
        if transition_mass is not None:
            transition_mass = np.asarray(transition_mass, dtype=float)
        return similarity, transition_mass

    def _refresh_stable_token_semantic_confirmations(self, level_index: int, stable_tokens: IntArray) -> None:
        return

    def _update_stable_token_identity_ages(self, level_index: int, stable_tokens: IntArray) -> None:
        return

    def get_stable_token_evidence_state(self) -> dict[str, Any]:
        return {
            'enabled': False,
            'removed': True,
            'reason': 'stable-token debounce and evidence-memory retention were removed from the active RG path',
        }

    def _current_transition_semantics_support_split(
        self,
        level_index: int,
        stable_tokens: IntArray,
        merge_key: tuple[int, ...],
    ) -> bool:
        lower_model = self.levels[level_index]
        sim, transition_mass = self._stable_token_semantic_evidence_matrices(level_index)
        if sim.shape != (lower_model.input_size, lower_model.input_size):
            return False

        stable_arr = np.asarray(stable_tokens, dtype=int)
        return all(
            self._current_transition_semantics_support_token(
                level_index,
                stable_arr,
                int(token),
                tuple(other for other in merge_key if int(other) != int(token)),
                similarity=sim,
                transition_mass=transition_mass,
            )
            for token in merge_key
        )

    def _current_transition_semantics_support_token(
        self,
        level_index: int,
        stable_tokens: IntArray,
        token: int,
        competitors: tuple[int, ...],
        *,
        similarity: FloatArray | None = None,
        transition_mass: FloatArray | None = None,
    ) -> bool:
        sim = similarity
        if sim is None:
            sim, transition_mass = self._stable_token_semantic_evidence_matrices(level_index)
        stable_arr = np.asarray(stable_tokens, dtype=int)
        if sim.shape != (self.levels[level_index].input_size, self.levels[level_index].input_size):
            return False
        own_members = np.flatnonzero(stable_arr == int(token))
        if own_members.size < self.stable_token_evidence_min_members:
            return False
        own_block = np.asarray(sim[np.ix_(own_members, own_members)], dtype=float).copy()
        np.fill_diagonal(own_block, -np.inf)
        cohesion = float(np.min(np.max(own_block, axis=1)))
        boundary = 0.0
        for other in competitors:
            other_members = np.flatnonzero(stable_arr == int(other))
            forward = np.asarray(sim[np.ix_(own_members, other_members)], dtype=float)
            reverse = np.asarray(sim[np.ix_(other_members, own_members)], dtype=float)
            if forward.size:
                boundary = max(boundary, float(np.max(forward)))
            if reverse.size:
                boundary = max(boundary, float(np.max(reverse)))
        observed_ratio = cohesion / boundary if boundary > 1e-12 else (1e6 if cohesion > 0.0 else 0.0)
        self._stable_token_evidence_semantic_ratios.append(float(observed_ratio))
        ratio_supported = bool(
            np.isfinite(cohesion)
            and cohesion > 0.0
            and observed_ratio >= float(self.stable_token_evidence_self_cross_ratio)
        )
        if not ratio_supported:
            return False
        required_z = float(self.stable_token_evidence_semantic_z_score)
        if required_z <= 0.0:
            return True
        if transition_mass is None:
            return False
        transition_mass = np.asarray(transition_mass, dtype=float)
        if transition_mass.shape != sim.shape:
            return False
        competitor_parts = [
            np.flatnonzero(stable_arr == int(other))
            for other in competitors
            if np.any(stable_arr == int(other))
        ]
        competitor_members = (
            np.concatenate(competitor_parts)
            if competitor_parts else np.zeros(0, dtype=int)
        )
        member_z_margins: list[float] = []
        for member in own_members.tolist():
            peers = own_members[own_members != int(member)]
            internal_mass = float(np.max(transition_mass[int(member), peers])) if peers.size else 0.0
            external_mass = (
                float(np.max(transition_mass[int(member), competitor_members]))
                if competitor_members.size else 0.0
            )
            denominator = np.sqrt(max(internal_mass + external_mass, 1e-12))
            member_z_margins.append((internal_mass - external_mass) / denominator)
        bottleneck_z_margin = float(np.min(member_z_margins)) if member_z_margins else float('-inf')
        self._stable_token_evidence_semantic_z_margins.append(bottleneck_z_margin)
        return bool(bottleneck_z_margin >= required_z)

    def _update_stable_tokens_for_level(self, level_index: int, raw_labels: IntArray) -> IntArray:
        self._ensure_level_count(level_index + 2)
        lower_model = self.levels[level_index]
        upper_model = self.levels[level_index + 1]
        stable = np.full(lower_model.input_size, -1, dtype=int)
        previous_tokens = self.inter_level_unit_maps[level_index].copy()
        assigned_tokens: set[int] = set()
        next_merge_evidence: dict[tuple[int, ...], int] = {}

        for raw_cluster in (int(c) for c in np.unique(raw_labels) if c >= 0):
            members = np.where(raw_labels == raw_cluster)[0]
            if members.size == 0:
                continue
            token = -1
            previous_for_members = previous_tokens[members]
            previous_for_members = previous_for_members[previous_for_members >= 0]
            previous_unique = tuple(sorted({int(token_id) for token_id in previous_for_members.tolist()}))

            if len(previous_unique) > 1 and self._should_keep_raw_merge_split(
                level_index,
                previous_tokens,
                members,
                previous_unique,
                next_merge_evidence,
            ):
                for previous_token in previous_unique:
                    previous_members = members[previous_tokens[members] == previous_token]
                    if previous_members.size == 0:
                        continue
                    stable[previous_members] = int(previous_token)
                    assigned_tokens.add(int(previous_token))
                continue

            if previous_for_members.size:
                candidates, counts = np.unique(previous_for_members, return_counts=True)
                order = np.argsort(-counts)
                for idx in order:
                    candidate = int(candidates[idx])
                    if candidate not in assigned_tokens:
                        token = candidate
                        break

            if token < 0:
                token = self._allocate_slot(upper_model.input_size, assigned_tokens, previous_tokens)

            stable[members] = token
            assigned_tokens.add(token)

        self.inter_level_unit_maps[level_index] = stable
        self._merge_evidence_by_level[level_index] = next_merge_evidence
        self._sync_public_level_aliases()
        return stable

    def _update_stable_tokens(self, raw_labels: IntArray) -> IntArray:
        return self._update_stable_tokens_for_level(0, raw_labels)

    def _stable_token_evidence_node_strengths(self, level_index: int, stable_tokens: IntArray) -> FloatArray | None:
        return None

    def _stable_token_geometry_for_level(
        self,
        level_index: int,
        stable_tokens: IntArray,
    ) -> tuple[IntArray, FloatArray, IntArray]:
        lower_model = self.levels[level_index]
        active_tokens = np.array(sorted({int(token) for token in stable_tokens if token >= 0}), dtype=int)
        if active_tokens.size == 0:
            return active_tokens, np.zeros((0, self.dimensions), dtype=float), np.zeros(0, dtype=int)

        centroids = np.zeros((active_tokens.size, self.dimensions), dtype=float)
        sizes = np.zeros(active_tokens.size, dtype=int)
        for idx, token in enumerate(active_tokens.tolist()):
            members = np.where(stable_tokens == token)[0]
            sizes[idx] = int(members.size)
            centroids[idx] = np.mean(lower_model.syncmap[members], axis=0)
        return active_tokens, centroids, sizes

    def _stable_token_internal_scales_for_level(
        self,
        level_index: int,
        stable_tokens: IntArray,
    ) -> FloatArray:
        lower_model = self.levels[level_index]
        active_tokens = np.array(sorted({int(token) for token in stable_tokens if token >= 0}), dtype=int)
        if active_tokens.size == 0:
            return np.zeros(0, dtype=float)

        base_scale = max(float(self._level_equilibrium_distance(level_index)), 1e-6)
        scales = np.full(active_tokens.size, base_scale, dtype=float)
        for idx, token in enumerate(active_tokens.tolist()):
            members = np.where(stable_tokens == token)[0]
            if members.size < 2:
                continue
            pos = np.asarray(lower_model.syncmap[members], dtype=float)
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            nearest = np.min(dist, axis=1)
            finite = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
            if finite.size == 0:
                tri = dist[np.triu_indices(dist.shape[0], 1)]
                finite = tri[np.isfinite(tri) & (tri > 1e-9)]
            if finite.size > 0:
                scales[idx] = max(base_scale, float(np.median(finite)))
        return scales

    def _stable_token_geometry(self, stable_tokens: IntArray) -> tuple[IntArray, FloatArray, IntArray]:
        return self._stable_token_geometry_for_level(0, stable_tokens)

    def _current_macro_jump_ratio(self) -> float:
        base_ratio = float(self.rg_jump_ratio)
        final_ratio = self.rg_jump_ratio_final
        if final_ratio is None:
            return base_ratio
        start = max(0, int(self.rg_jump_ratio_anneal_start))
        end = max(start, int(self.rg_jump_ratio_anneal_end))
        if end <= start:
            return float(final_ratio) if int(self.step_count) >= start else base_ratio
        progress = (float(self.step_count) - float(start)) / float(end - start)
        progress = min(1.0, max(0.0, progress))
        return float((1.0 - progress) * base_ratio + progress * float(final_ratio))

    def _macro_gap_confidence(self, significance: RGSignificanceResult) -> float:
        observed_ratio = max(float(getattr(significance, "best_jump_ratio", 1.0)), 1.0)
        return self._gap_ratio_to_confidence(observed_ratio)

    def _gap_ratio_to_confidence(self, observed_ratio: float) -> float:
        if not self.macro_boundary_gap_gate_enabled:
            return 1.0
        current_ratio = max(float(self._current_macro_jump_ratio()), 1.0)
        if observed_ratio < current_ratio:
            return 0.0
        floor = float(np.clip(self.macro_boundary_gap_gate_floor, 0.0, 1.0))
        full_ratio = max(float(self.macro_boundary_gap_gate_full_ratio), current_ratio + 1e-6)
        if observed_ratio >= full_ratio:
            return 1.0
        progress = (observed_ratio - current_ratio) / (full_ratio - current_ratio)
        progress = min(1.0, max(0.0, progress))
        return float(floor + (1.0 - floor) * progress)

    def _component_local_gap_confidences(
        self,
        significance: RGSignificanceResult,
        component_labels: IntArray,
    ) -> dict[int, float]:
        labels = np.asarray(component_labels, dtype=int).reshape(-1)
        component_ids = sorted({int(label) for label in labels.tolist() if int(label) >= 0})
        if not component_ids:
            return {}

        default_confidence = self._macro_gap_confidence(significance)
        if not self.macro_boundary_gap_gate_enabled:
            return {component_id: 1.0 for component_id in component_ids}

        mst_edges = np.asarray(
            getattr(significance, "mst_edges", np.zeros((0, 2), dtype=int)),
            dtype=int,
        )
        mst_norm = np.asarray(
            getattr(significance, "mst_normalized_lengths", np.zeros(0, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if mst_edges.ndim != 2 or mst_edges.shape[1] != 2 or mst_edges.shape[0] != mst_norm.shape[0]:
            return {component_id: default_confidence for component_id in component_ids}

        threshold_norm = max(float(getattr(significance, "threshold_norm", 1.0)), 1.0)
        confidences: dict[int, float] = {}
        for component_id in component_ids:
            internal_norms: list[float] = []
            external_norms: list[float] = []
            for edge_idx, (u, v) in enumerate(mst_edges.tolist()):
                if not (0 <= int(u) < labels.shape[0] and 0 <= int(v) < labels.shape[0]):
                    continue
                label_u = int(labels[int(u)])
                label_v = int(labels[int(v)])
                if label_u < 0 or label_v < 0:
                    continue
                norm = max(float(mst_norm[edge_idx]), 1e-9)
                if label_u == component_id and label_v == component_id:
                    internal_norms.append(norm)
                elif label_u == component_id or label_v == component_id:
                    external_norms.append(norm)

            internal_ref = max(max(internal_norms) if internal_norms else threshold_norm, 1e-9)
            if not external_norms:
                confidences[component_id] = default_confidence
                continue
            local_ratio = min(external_norms) / internal_ref
            confidences[component_id] = self._gap_ratio_to_confidence(local_ratio)
        return confidences

    def _token_purity_to_confidence(self, purity: float) -> float:
        if not self.macro_boundary_token_purity_enabled:
            return 1.0
        purity_val = float(np.clip(purity, 0.0, 1.0))
        threshold = float(np.clip(self.macro_boundary_token_purity_threshold, 0.0, 1.0))
        floor = float(np.clip(self.macro_boundary_token_purity_floor, 0.0, 1.0))
        if purity_val <= threshold:
            return floor
        progress = (purity_val - threshold) / max(1.0 - threshold, 1e-9)
        progress = min(1.0, max(0.0, progress))
        return float(floor + (1.0 - floor) * progress)

    @staticmethod
    def _summarize_token_coupling(values: FloatArray, *, top_k: int = 2) -> float:
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr) & (arr > 1e-12)]
        if finite.size == 0:
            return 0.0
        if finite.size <= max(1, int(top_k)):
            return float(np.mean(finite))
        top = np.partition(finite, finite.size - int(top_k))[-int(top_k):]
        return float(np.mean(top))

    def _component_token_semantic_confidences(
        self,
        level_index: int,
        stable_tokens: IntArray,
        active_tokens: IntArray,
        component_labels: IntArray,
    ) -> dict[int, float]:
        labels = np.asarray(component_labels, dtype=int).reshape(-1)
        component_ids = sorted({int(label) for label in labels.tolist() if int(label) >= 0})
        if not component_ids or not self.macro_boundary_token_purity_enabled:
            return {component_id: 1.0 for component_id in component_ids}

        token_pair, token_self = self._stable_token_semantic_similarity_matrix(
            level_index,
            stable_tokens,
            active_tokens,
        )
        if token_pair is None or token_self is None:
            return {component_id: 1.0 for component_id in component_ids}
        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        token_indices: dict[int, int] = {int(token): idx for idx, token in enumerate(active_arr.tolist())}

        confidences: dict[int, float] = {}
        all_token_indices = np.arange(active_arr.size, dtype=int)
        for component_id in component_ids:
            member_tokens = active_arr[labels == component_id]
            member_indices = np.asarray(
                [token_indices[int(token)] for token in member_tokens.tolist() if int(token) in token_indices],
                dtype=int,
            )
            if member_indices.size <= 1:
                confidences[component_id] = 1.0
                continue

            intra_block = token_pair[np.ix_(member_indices, member_indices)]
            intra_vals = intra_block[np.triu_indices(member_indices.size, k=1)]
            intra_strength = self._summarize_token_coupling(intra_vals, top_k=max(1, member_indices.size - 1))
            if intra_strength <= 1e-12:
                intra_strength = self._summarize_token_coupling(token_self[member_indices], top_k=member_indices.size)

            outside_indices = all_token_indices[~np.isin(all_token_indices, member_indices)]
            if outside_indices.size == 0:
                confidences[component_id] = 1.0
                continue
            external_strength = self._summarize_token_coupling(
                token_pair[np.ix_(member_indices, outside_indices)],
                top_k=min(2, max(1, outside_indices.size)),
            )
            if intra_strength <= 1e-12 and external_strength <= 1e-12:
                confidences[component_id] = 1.0
                continue
            purity = intra_strength / max(intra_strength + external_strength, 1e-9)
            confidences[component_id] = self._token_purity_to_confidence(purity)
        return confidences

    def _stable_token_semantic_similarity_matrix(
        self,
        level_index: int,
        stable_tokens: IntArray,
        active_tokens: IntArray,
    ) -> tuple[FloatArray | None, FloatArray | None]:
        self._ensure_level_count(level_index + 1)
        lower_model = self.levels[level_index]
        coact = getattr(lower_model, "_coact", None)
        if coact is None:
            return None, None
        try:
            sim = np.asarray(coact.similarity(), dtype=float)
        except Exception:
            return None, None
        if sim.shape != (lower_model.input_size, lower_model.input_size):
            return None, None

        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        token_members = [np.where(np.asarray(stable_tokens, dtype=int) == int(token))[0] for token in active_arr.tolist()]
        token_pair = np.zeros((active_arr.size, active_arr.size), dtype=float)
        token_self = np.zeros(active_arr.size, dtype=float)
        for idx_i, members_i in enumerate(token_members):
            if members_i.size >= 2:
                block = np.asarray(sim[np.ix_(members_i, members_i)], dtype=float)
                mask = ~np.eye(block.shape[0], dtype=bool)
                token_self[idx_i] = self._summarize_token_coupling(
                    block[mask],
                    top_k=max(1, block.shape[0] - 1),
                )
            for idx_j in range(idx_i + 1, active_arr.size):
                members_j = token_members[idx_j]
                if members_i.size == 0 or members_j.size == 0:
                    continue
                block_ij = np.asarray(sim[np.ix_(members_i, members_j)], dtype=float)
                block_ji = np.asarray(sim[np.ix_(members_j, members_i)], dtype=float)
                forward_strength = self._summarize_token_coupling(block_ij, top_k=2)
                backward_strength = self._summarize_token_coupling(block_ji, top_k=2)
                token_pair[idx_i, idx_j] = 0.5 * (forward_strength + backward_strength)
                token_pair[idx_j, idx_i] = token_pair[idx_i, idx_j]
        return token_pair, token_self

    def _apply_macro_split_hysteresis(
        self,
        active_tokens: IntArray,
        centroids: FloatArray,
        component_labels: IntArray,
        previous_meta: IntArray,
        *,
        threshold_distance: float,
    ) -> IntArray:
        hysteresis_ratio = max(float(self.macro_split_hysteresis_ratio), 1.0)
        if hysteresis_ratio <= 1.0:
            return np.asarray(component_labels, dtype=int)

        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        labels = np.asarray(component_labels, dtype=int).reshape(-1).copy()
        previous_arr = np.asarray(previous_meta, dtype=int).reshape(-1)
        if active_arr.size == 0 or labels.shape[0] != active_arr.shape[0]:
            return labels

        split_barrier = max(float(threshold_distance), 1e-6) * hysteresis_ratio
        if split_barrier <= 0.0:
            return labels

        previous_ids = sorted(
            {
                int(previous_arr[int(token)])
                for token in active_arr.tolist()
                if 0 <= int(token) < previous_arr.shape[0] and int(previous_arr[int(token)]) >= 0
            }
        )
        for meta_token in previous_ids:
            member_mask = np.array(
                [
                    0 <= int(token) < previous_arr.shape[0] and int(previous_arr[int(token)]) == int(meta_token)
                    for token in active_arr.tolist()
                ],
                dtype=bool,
            )
            if int(np.sum(member_mask)) <= 1:
                continue
            member_components = sorted({int(label) for label in labels[member_mask].tolist() if int(label) >= 0})
            if len(member_components) <= 1:
                continue

            allow_split = False
            for idx_i, comp_i in enumerate(member_components[:-1]):
                comp_i_centroid = np.mean(centroids[member_mask & (labels == int(comp_i))], axis=0)
                for comp_j in member_components[idx_i + 1:]:
                    comp_j_centroid = np.mean(centroids[member_mask & (labels == int(comp_j))], axis=0)
                    if float(np.linalg.norm(comp_j_centroid - comp_i_centroid)) > split_barrier:
                        allow_split = True
                        break
                if allow_split:
                    break

            if not allow_split:
                labels[member_mask] = int(member_components[0])
        return labels

    def _update_macro_tokens_for_level(self, level_index: int, stable_tokens: IntArray) -> IntArray:
        self._ensure_level_count(level_index + 2)
        lower_model = self.levels[level_index]
        upper_model = self.levels[level_index + 1]
        macro_tokens = np.full(lower_model.input_size, -1, dtype=int)
        active_tokens, centroids, _ = self._stable_token_geometry_for_level(level_index, stable_tokens)
        intrinsic_scales = self._stable_token_internal_scales_for_level(level_index, stable_tokens)
        token_semantic_similarity = None
        if self.rg_semantic_graph_enabled:
            token_semantic_similarity, _ = self._stable_token_semantic_similarity_matrix(
                level_index,
                stable_tokens,
                active_tokens,
            )

        if active_tokens.size == 0:
            self.inter_level_meta_maps[level_index] = np.full(upper_model.input_size, -1, dtype=int)
            self._macro_stability_counts_by_level[level_index] = np.zeros(upper_model.input_size, dtype=int)
            self._macro_member_signatures_by_level[level_index] = [None] * upper_model.input_size
            self._last_macro_feedback_strengths_by_level[level_index] = np.zeros(lower_model.input_size, dtype=float)
            if level_index == 0:
                self._last_macro_tokens = macro_tokens
            self._last_rg_significance = None
            self.reference_local_scale = 0.0
            self.significant_distance = 0.0
            self.macro_token_count = 0
            self._sync_public_level_aliases()
            return macro_tokens

        significance = estimate_rg_significance(
            active_tokens,
            centroids,
            knn_k=self.rg_knn_k,
            jump_ratio=self._current_macro_jump_ratio(),
            fallback_scale=self.rg_fallback_scale,
            local_scales_override=intrinsic_scales,
            semantic_similarity=token_semantic_similarity,
            semantic_penalty_scale=self.rg_semantic_graph_strength,
            semantic_similarity_floor=self.rg_semantic_graph_similarity_floor,
            semantic_penalty_mode=self.rg_semantic_graph_penalty_mode,
            semantic_relative_threshold=self.rg_semantic_graph_relative_threshold,
            semantic_anchor_floor=self.rg_semantic_graph_anchor_floor,
            semantic_edge_mode=self.rg_semantic_graph_edge_mode,
            semantic_topk=self.rg_semantic_graph_topk,
        )
        component_labels = np.asarray(significance.component_labels, dtype=int)
        semantic_macro_proposal = str(self.macro_component_proposal_mode).strip().lower() == 'semantic'
        if semantic_macro_proposal:
            semantic_component_labels = self._cluster_upper_level_semantic_tokens(
                level_index,
                upper_model,
                stable_tokens,
                active_tokens,
                semantic_source='lower',
            )
            if semantic_component_labels is not None and semantic_component_labels.shape == component_labels.shape:
                component_labels = np.asarray(semantic_component_labels, dtype=int)
        else:
            relation_component_labels = self._persistent_relation_component_labels(level_index, active_tokens)
            if relation_component_labels is not None and relation_component_labels.shape == component_labels.shape:
                component_labels = np.asarray(relation_component_labels, dtype=int)
        previous_meta = self.inter_level_meta_maps[level_index].copy()
        component_labels = self._apply_macro_split_hysteresis(
            active_tokens,
            centroids,
            component_labels,
            previous_meta,
            threshold_distance=float(significance.threshold_distance),
        )
        assigned_meta: set[int] = set()
        next_meta = np.full(upper_model.input_size, -1, dtype=int)
        if semantic_macro_proposal:
            component_confidences = {
                int(component): 1.0
                for component in np.unique(component_labels).tolist()
                if int(component) >= 0
            }
        else:
            component_confidences = self._component_local_gap_confidences(significance, component_labels)
        semantic_confidences = self._component_token_semantic_confidences(
            level_index,
            stable_tokens,
            active_tokens,
            component_labels,
        )
        next_signatures: list[tuple[int, ...] | None] = [None] * upper_model.input_size
        meta_confidences = np.zeros(upper_model.input_size, dtype=float)

        for component in (int(c) for c in np.unique(component_labels)):
            member_tokens = active_tokens[component_labels == component]
            member_signature = tuple(sorted(int(token) for token in member_tokens.tolist()))
            meta_token = -1
            previous_for_members = previous_meta[member_tokens]
            previous_for_members = previous_for_members[previous_for_members >= 0]

            if previous_for_members.size:
                candidates, counts = np.unique(previous_for_members, return_counts=True)
                order = np.argsort(-counts)
                for idx in order:
                    candidate = int(candidates[idx])
                    if candidate not in assigned_meta:
                        meta_token = candidate
                        break

            if meta_token < 0:
                meta_token = self._allocate_slot(upper_model.input_size, assigned_meta, previous_meta)

            next_meta[member_tokens] = meta_token
            next_signatures[meta_token] = member_signature
            meta_confidences[meta_token] = min(
                float(component_confidences.get(component, 1.0)),
                float(semantic_confidences.get(component, 1.0)),
            )
            assigned_meta.add(meta_token)

        previous_counts = np.asarray(self._macro_stability_counts_by_level[level_index], dtype=int).reshape(-1)
        previous_signatures = list(self._macro_member_signatures_by_level[level_index])
        if len(previous_signatures) != upper_model.input_size:
            previous_signatures = [None] * upper_model.input_size
        next_counts = np.zeros_like(next_meta, dtype=int)
        for meta_token, signature in enumerate(next_signatures):
            if signature is None:
                continue
            if previous_signatures[meta_token] == signature:
                next_counts[meta_token] = int(previous_counts[meta_token]) + 1
            else:
                next_counts[meta_token] = 1

        self.inter_level_meta_maps[level_index] = next_meta
        self._macro_stability_counts_by_level[level_index] = next_counts
        self._macro_member_signatures_by_level[level_index] = next_signatures
        node_strengths = np.zeros(lower_model.input_size, dtype=float)
        for node_idx, stable_token in enumerate(stable_tokens.tolist()):
            if stable_token >= 0:
                macro_token = int(next_meta[int(stable_token)])
                macro_tokens[node_idx] = macro_token
                if macro_token < 0:
                    continue
                if self.macro_boundary_min_stable_refreshes > 0:
                    age_gate = (
                        1.0
                        if int(next_counts[macro_token]) >= int(self.macro_boundary_min_stable_refreshes)
                        else 0.0
                    )
                else:
                    age_gate = 1.0
                node_strengths[node_idx] = float(meta_confidences[macro_token]) * float(age_gate)

        self._last_macro_feedback_strengths_by_level[level_index] = node_strengths.copy()

        if level_index == 0:
            self._last_macro_tokens = macro_tokens.copy()
        self._last_rg_significance = significance
        self.reference_local_scale = float(np.median(significance.local_scales)) if significance.local_scales.size else 0.0
        self.significant_distance = float(significance.threshold_distance)
        self.macro_token_count = int(len({int(x) for x in component_labels.tolist() if int(x) >= 0}))
        self._sync_public_level_aliases()
        return macro_tokens

    def _update_macro_tokens(self, stable_tokens: IntArray) -> IntArray:
        return self._update_macro_tokens_for_level(0, stable_tokens)

    def _current_macro_tokens_for_level(self, level_index: int) -> IntArray:
        stable_tokens = self._update_stable_tokens_for_level(level_index, self._current_raw_labels(level_index))
        return self._update_macro_tokens_for_level(level_index, stable_tokens)

    def _current_macro_tokens(self) -> IntArray:
        return self._current_macro_tokens_for_level(0)

    def _lift_stable_tokens_through_cached_meta_map(
        self,
        level_index: int,
        stable_tokens: IntArray,
    ) -> IntArray:
        self._ensure_level_count(level_index + 2)
        lower_model = self.levels[level_index]
        upper_model = self.levels[level_index + 1]
        cached_meta = np.asarray(self.inter_level_meta_maps[level_index], dtype=int).reshape(-1)
        if cached_meta.shape[0] != upper_model.input_size:
            return self._update_macro_tokens_for_level(level_index, stable_tokens)

        macro_tokens = np.full(lower_model.input_size, -1, dtype=int)
        for node_idx, stable_token in enumerate(np.asarray(stable_tokens, dtype=int).tolist()):
            if 0 <= int(stable_token) < cached_meta.shape[0]:
                macro_tokens[node_idx] = int(cached_meta[int(stable_token)])
        return macro_tokens

    def _macro_feedback_tokens_for_level(
        self,
        level_index: int,
        stable_tokens: IntArray,
    ) -> IntArray:
        refresh_interval = max(0, int(self.hierarchy_refresh_interval))
        should_refresh = refresh_interval <= 0 or (self.step_count % refresh_interval == 0)
        if should_refresh:
            return self._update_macro_tokens_for_level(level_index, stable_tokens)
        macro_tokens = self._lift_stable_tokens_through_cached_meta_map(level_index, stable_tokens)
        if np.any(np.asarray(macro_tokens, dtype=int) >= 0):
            return macro_tokens
        return self._update_macro_tokens_for_level(level_index, stable_tokens)

    def _source_tokens_for_level(
        self,
        level_index: int,
        raw_labels: IntArray,
        stable_tokens: IntArray,
        macro_tokens: IntArray,
    ) -> IntArray:
        mode = str(self.rg_input_level).lower()
        if mode in {'raw', 'raw_labels', 'current'}:
            return raw_labels
        if mode in {'stable', 'stable_tokens'}:
            return stable_tokens
        return macro_tokens

    def _source_mode_uses_macro_tokens(self) -> bool:
        mode = str(self.rg_input_level).lower()
        return mode not in {'raw', 'raw_labels', 'current', 'stable', 'stable_tokens'}

    def _resolved_source_tokens_for_level(
        self,
        level_index: int,
        raw_labels: IntArray,
        stable_tokens: IntArray,
    ) -> IntArray:
        if str(self.rg_input_level).lower() in {'raw', 'raw_labels', 'current'}:
            return np.asarray(raw_labels, dtype=int)
        if not self._source_mode_uses_macro_tokens():
            return np.asarray(stable_tokens, dtype=int)
        macro_tokens = self._macro_feedback_tokens_for_level(level_index, stable_tokens)
        return np.asarray(
            self._source_tokens_for_level(level_index, raw_labels, stable_tokens, macro_tokens),
            dtype=int,
        )

    def _level1_source_tokens(self, stable_tokens: IntArray, macro_tokens: IntArray) -> IntArray:
        return self._source_tokens_for_level(
            0,
            self._current_level0_raw_labels(),
            stable_tokens,
            macro_tokens,
        )

    def _build_upper_level_input(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        active_indices: IntArray,
    ) -> tuple[FloatArray, IntArray, IntArray]:
        self._ensure_level_count(lower_level_index + 2)
        upper_model = self.levels[lower_level_index + 1]
        source_arr = np.asarray(source_tokens, dtype=int).reshape(-1)
        active_arr = np.asarray(active_indices, dtype=int).reshape(-1)
        active_arr = active_arr[(active_arr >= 0) & (active_arr < source_arr.shape[0])]
        active_tokens = source_arr[active_arr] if active_arr.size > 0 else np.zeros(0, dtype=int)
        valid_tokens = np.unique(active_tokens[active_tokens >= 0])
        support_tokens = np.unique(source_arr[source_arr >= 0])

        upper_input = np.zeros(upper_model.input_size, dtype=float)
        if valid_tokens.size == 0:
            self._last_upper_input_by_level[lower_level_index] = upper_input.copy()
            return upper_input, valid_tokens, support_tokens

        active_gain = max(float(self.rg_upper_input_active_gain), 0.0)
        mass_gain = max(float(self.rg_upper_input_mass_gain), 0.0)

        history = self._upper_history_by_level[lower_level_index]
        for token in active_tokens.tolist():
            if int(token) >= 0:
                history.append(int(token))

        active_counts = np.bincount(active_tokens[active_tokens >= 0], minlength=upper_model.input_size).astype(float)
        history_arr = np.asarray(list(history), dtype=int)
        history_counts = (
            np.bincount(history_arr[history_arr >= 0], minlength=upper_model.input_size).astype(float)
            if history_arr.size > 0
            else np.zeros(upper_model.input_size, dtype=float)
        )
        active_norm = active_counts / max(float(np.max(active_counts)), 1.0)
        history_norm = history_counts / max(float(np.max(history_counts)), 1.0)

        history_tokens = np.flatnonzero(history_counts > 0.0)
        weighted_tokens = np.unique(np.concatenate([valid_tokens, history_tokens])).astype(int)
        upper_input[weighted_tokens] = mass_gain * history_norm[weighted_tokens]
        upper_input[valid_tokens] += active_gain * active_norm[valid_tokens]

        max_weight = float(np.max(upper_input[weighted_tokens])) if weighted_tokens.size > 0 else 0.0
        if max_weight > 1e-12:
            upper_input[weighted_tokens] /= max_weight
        upper_input = np.clip(upper_input, 0.0, 1.0)
        self._last_upper_input_by_level[lower_level_index] = upper_input.copy()
        return upper_input, valid_tokens, support_tokens

    def _reset_upper_cluster_cache(self) -> None:
        for level_index in range(len(self._last_upper_cluster_tokens_by_level)):
            self._last_upper_cluster_tokens_by_level[level_index] = tuple()
            self._last_upper_cluster_labels_by_level[level_index] = None
            self._last_upper_cluster_refresh_step_by_level[level_index] = -1
            self._last_upper_cluster_provenance_by_level[level_index] = None

    def last_upper_cluster_provenance(self, lower_level_index: int = 0) -> Optional[dict[str, Any]]:
        self._ensure_level_count(lower_level_index + 2)
        provenance = self._last_upper_cluster_provenance_by_level[lower_level_index]
        if provenance is None:
            return None
        return dict(provenance)

    @staticmethod
    def _relabel_token_labels(labels: IntArray) -> IntArray:
        arr = np.asarray(labels, dtype=int).reshape(-1)
        relabeled = np.full(arr.shape[0], -1, dtype=int)
        valid = sorted({int(v) for v in arr.tolist() if int(v) >= 0})
        mapping = {label: idx for idx, label in enumerate(valid)}
        for idx, label in enumerate(arr.tolist()):
            if int(label) >= 0:
                relabeled[idx] = mapping[int(label)]
        return relabeled

    def _cluster_upper_level_graph_tokens(
        self,
        upper_model: SyncMapNS,
        active_tokens: IntArray,
    ) -> IntArray | None:
        rows = getattr(upper_model, '_g_rows', None)
        cols = getattr(upper_model, '_g_cols', None)
        weights = getattr(upper_model, '_g_w', None)
        if rows is None or cols is None or weights is None:
            return None
        row_arr = np.asarray(rows, dtype=int).reshape(-1)
        col_arr = np.asarray(cols, dtype=int).reshape(-1)
        weight_arr = np.asarray(weights, dtype=float).reshape(-1)
        if row_arr.size == 0 or col_arr.size == 0 or weight_arr.size == 0:
            return None

        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        active_set = {int(token) for token in active_arr.tolist()}
        graph = nx.Graph()
        graph.add_nodes_from(int(token) for token in active_arr.tolist())
        for i, j, w in zip(row_arr.tolist(), col_arr.tolist(), weight_arr.tolist()):
            if int(i) in active_set and int(j) in active_set and float(w) > 0.0:
                graph.add_edge(int(i), int(j), weight=float(w))
        if graph.number_of_edges() == 0:
            return None

        try:
            from community.community_louvain import best_partition

            partition = best_partition(
                graph,
                weight='weight',
                resolution=float(getattr(upper_model, 'graph_louvain_resolution', 1.0)),
            )
            labels = np.asarray([int(partition.get(int(token), -1)) for token in active_arr.tolist()], dtype=int)
            return self._relabel_token_labels(labels)
        except Exception:
            labels = np.full(active_arr.shape[0], -1, dtype=int)
            for comp_idx, component in enumerate(nx.connected_components(graph)):
                for token in component:
                    matches = np.where(active_arr == int(token))[0]
                    labels[matches] = int(comp_idx)
            return self._relabel_token_labels(labels)

    def _cluster_upper_level_semantic_tokens(
        self,
        lower_level_index: int,
        upper_model: SyncMapNS,
        source_tokens: IntArray,
        active_tokens: IntArray,
        *,
        semantic_source: str | None = None,
    ) -> IntArray | None:
        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        if active_arr.size <= 1:
            return None
        sim: FloatArray | None = None
        source_mode = self.upper_cluster_semantic_source if semantic_source is None else semantic_source
        if str(source_mode).strip().lower() == "upper":
            coact = getattr(upper_model, "_coact", None)
            if coact is not None:
                try:
                    upper_sim = np.asarray(coact.similarity(), dtype=float)
                    if upper_sim.shape == (upper_model.input_size, upper_model.input_size):
                        block = np.asarray(upper_sim[np.ix_(active_arr, active_arr)], dtype=float)
                        sim = 0.5 * (block + block.T)
                except Exception:
                    sim = None
        if sim is None:
            token_pair, _ = self._stable_token_semantic_similarity_matrix(
                lower_level_index,
                np.asarray(source_tokens, dtype=int),
                active_arr,
            )
            if token_pair is None:
                return None
            sim = np.asarray(token_pair, dtype=float)
        if sim.shape != (active_arr.size, active_arr.size):
            return None
        sim = self._sharpen_semantic_similarity_for_proposal(sim)
        sim = self._sparsify_semantic_similarity_for_proposal(sim)
        np.fill_diagonal(sim, 0.0)
        method = str(self.upper_cluster_semantic_method).strip().lower()
        threshold = float(np.clip(self.upper_cluster_semantic_threshold, 0.0, 1.0))

        if method == "threshold":
            graph = nx.Graph()
            graph.add_nodes_from(range(active_arr.size))
            self_loop_weight = max(float(self.upper_cluster_semantic_self_loop_weight), 0.0)
            if self_loop_weight > 0.0:
                for idx in range(active_arr.size):
                    graph.add_edge(idx, idx, weight=self_loop_weight)
            ii, jj = np.where(np.triu(sim > threshold, k=1))
            for i, j in zip(ii.tolist(), jj.tolist()):
                graph.add_edge(int(i), int(j), weight=float(sim[int(i), int(j)]))
            labels = np.full(active_arr.size, -1, dtype=int)
            for comp_idx, component in enumerate(nx.connected_components(graph)):
                for idx in component:
                    labels[int(idx)] = int(comp_idx)
            return self._relabel_token_labels(labels)

        graph = nx.Graph()
        graph.add_nodes_from(range(active_arr.size))
        self_loop_weight = max(float(self.upper_cluster_semantic_self_loop_weight), 0.0)
        if self_loop_weight > 0.0:
            for idx in range(active_arr.size):
                graph.add_edge(idx, idx, weight=self_loop_weight)
        ii, jj = np.where(np.triu(sim > threshold, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            graph.add_edge(int(i), int(j), weight=float(sim[int(i), int(j)]))
        if graph.number_of_edges() == 0:
            return np.arange(active_arr.size, dtype=int)

        try:
            from community.community_louvain import best_partition

            partition = best_partition(
                graph,
                weight="weight",
                resolution=float(self.upper_cluster_semantic_resolution),
            )
            labels = np.asarray([int(partition.get(i, -1)) for i in range(active_arr.size)], dtype=int)
            return self._relabel_token_labels(labels)
        except Exception:
            labels = np.full(active_arr.size, -1, dtype=int)
            for comp_idx, component in enumerate(nx.connected_components(graph)):
                for idx in component:
                    labels[int(idx)] = int(comp_idx)
            return self._relabel_token_labels(labels)

    def _sharpen_semantic_similarity_for_proposal(self, similarity: FloatArray) -> FloatArray:
        sim = np.asarray(similarity, dtype=float).copy()
        np.fill_diagonal(sim, 0.0)
        power = max(float(self.rg_semantic_sharpen_power), 1.0)
        sharpened = np.power(np.clip(sim, 0.0, 1.0), power)
        np.fill_diagonal(sharpened, 0.0)
        max_val = float(np.max(sharpened))
        if max_val > 1e-12:
            sharpened /= max_val
        return np.clip(sharpened, 0.0, 1.0)

    def _sparsify_semantic_similarity_for_proposal(self, similarity: FloatArray) -> FloatArray:
        sim = np.asarray(similarity, dtype=float).copy()
        np.fill_diagonal(sim, 0.0)
        mode = str(self.upper_cluster_semantic_sparsify_mode).strip().lower()
        topk = max(int(self.upper_cluster_semantic_topk), 0)
        n = int(sim.shape[0])
        if mode == "none" or topk <= 0 or n <= 1:
            return np.clip(sim, 0.0, 1.0)

        k = min(topk, n - 1)
        row_mask = np.zeros((n, n), dtype=bool)
        for idx in range(n):
            row = sim[idx].copy()
            row[idx] = -np.inf
            neighbors = np.argpartition(-row, kth=k - 1)[:k]
            for nbr in neighbors.tolist():
                if np.isfinite(row[int(nbr)]) and row[int(nbr)] > 0.0:
                    row_mask[idx, int(nbr)] = True

        if mode == "mutual_topk":
            keep_mask = row_mask & row_mask.T
        elif mode == "row_topk":
            keep_mask = row_mask | row_mask.T
        else:
            return np.clip(sim, 0.0, 1.0)

        sparse = np.where(keep_mask, sim, 0.0)
        np.fill_diagonal(sparse, 0.0)
        return np.clip(sparse, 0.0, 1.0)

    def _cluster_upper_level_significance_tokens(
        self,
        upper_model: SyncMapNS,
        active_tokens: IntArray,
    ) -> IntArray | None:
        active_arr = np.asarray(active_tokens, dtype=int).reshape(-1)
        if active_arr.size <= 1:
            return None

        token_positions = np.asarray(upper_model.syncmap[active_arr], dtype=float)
        jump_ratio = self.rg_jump_ratio if self.upper_rg_jump_ratio is None else self.upper_rg_jump_ratio
        significance = estimate_rg_significance(
            active_arr,
            token_positions,
            knn_k=int(self.rg_knn_k),
            jump_ratio=float(jump_ratio),
            fallback_scale=float(self.rg_fallback_scale),
        )
        labels = np.asarray(significance.component_labels, dtype=int).reshape(-1)
        if labels.shape[0] != active_arr.shape[0]:
            return None
        return self._relabel_token_labels(labels)

    def _cluster_upper_level_tokens(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        *,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> tuple[IntArray, IntArray]:
        self._ensure_level_count(lower_level_index + 2)
        upper_model = self.levels[lower_level_index + 1]
        active_tokens = np.array(sorted({int(token) for token in source_tokens if token >= 0}), dtype=int)
        if active_tokens.size == 0:
            return active_tokens, np.zeros(0, dtype=int)
        if active_tokens.size == 1:
            token_labels = np.array([0], dtype=int)
            self._last_upper_cluster_tokens_by_level[lower_level_index] = tuple(int(token) for token in active_tokens.tolist())
            self._last_upper_cluster_labels_by_level[lower_level_index] = token_labels.copy()
            self._last_upper_cluster_refresh_step_by_level[lower_level_index] = int(self.step_count)
            self._last_upper_cluster_provenance_by_level[lower_level_index] = {
                'mode': 'singleton',
                'source': 'singleton',
                'group_count': 1,
                'step_count': int(self.step_count),
                'active_tokens': tuple(int(token) for token in active_tokens.tolist()),
            }
            return active_tokens, token_labels

        active_tuple = tuple(int(token) for token in active_tokens.tolist())
        refresh_interval = max(1, int(self.upper_cluster_refresh_interval))
        cached_tokens = self._last_upper_cluster_tokens_by_level[lower_level_index]
        cached_labels = self._last_upper_cluster_labels_by_level[lower_level_index]
        cached_step = self._last_upper_cluster_refresh_step_by_level[lower_level_index]
        if (
            not force_refresh
            and cached_labels is not None
            and cached_tokens == active_tuple
            and (int(self.step_count) - int(cached_step)) < refresh_interval
        ):
            return active_tokens, np.asarray(cached_labels, dtype=int).copy()

        strategy = kwargs.get('strategy', upper_model.clustering_strategy)
        token_positions = upper_model.syncmap[active_tokens]
        proposal_mode = str(
            kwargs.get('upper_cluster_proposal_mode', self.upper_cluster_proposal_mode)
        ).strip().lower()
        # Macro proposal now prefers sequence statistics over geometry. Geometry
        # remains as a fallback/readout path when statistical evidence is missing.
        token_labels = None
        provenance_source = 'strategy'
        semantic_group_count: int | None = None
        significance_group_count: int | None = None
        graph_group_count: int | None = None
        if proposal_mode == 'semantic':
            token_labels = self._cluster_upper_level_semantic_tokens(
                lower_level_index,
                upper_model,
                source_tokens,
                active_tokens,
            )
            if token_labels is not None:
                semantic_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                if semantic_group_count <= 1:
                    token_labels = self._cluster_upper_level_significance_tokens(upper_model, active_tokens)
                    if token_labels is not None:
                        significance_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                        if significance_group_count <= 1:
                            token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                            if token_labels is not None:
                                graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                            provenance_source = 'semantic_fallback_graph'
                        else:
                            provenance_source = 'semantic_fallback_significance'
                    else:
                        token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                        if token_labels is not None:
                            graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                        provenance_source = 'semantic_fallback_graph'
                else:
                    provenance_source = 'semantic'
            else:
                token_labels = self._cluster_upper_level_significance_tokens(upper_model, active_tokens)
                if token_labels is not None:
                    significance_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                    if significance_group_count <= 1:
                        token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                        if token_labels is not None:
                            graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                        provenance_source = 'semantic_missing_fallback_graph'
                    else:
                        provenance_source = 'semantic_missing_fallback_significance'
                else:
                    token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                    if token_labels is not None:
                        graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                    provenance_source = 'semantic_missing_fallback_graph'
        else:
            token_labels = self._cluster_upper_level_significance_tokens(upper_model, active_tokens)
            if token_labels is not None:
                significance_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                if significance_group_count <= 1:
                    token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                    if token_labels is not None:
                        graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                    provenance_source = 'significance_fallback_graph'
                else:
                    provenance_source = 'significance'
            else:
                token_labels = self._cluster_upper_level_graph_tokens(upper_model, active_tokens)
                if token_labels is not None:
                    graph_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
                provenance_source = 'significance_missing_fallback_graph'
        if token_labels is None:
            token_labels = np.asarray(strategy.fit_predict(token_positions), dtype=int)
            provenance_source = 'strategy'
        if np.any(token_labels < 0):
            next_label = int(np.max(token_labels[token_labels >= 0])) + 1 if np.any(token_labels >= 0) else 0
            token_labels = token_labels.copy()
            for idx in np.flatnonzero(token_labels < 0):
                token_labels[int(idx)] = int(next_label)
                next_label += 1
        final_group_count = len({int(label) for label in token_labels.tolist() if int(label) >= 0})
        self._last_upper_cluster_tokens_by_level[lower_level_index] = active_tuple
        self._last_upper_cluster_labels_by_level[lower_level_index] = token_labels.copy()
        self._last_upper_cluster_refresh_step_by_level[lower_level_index] = int(self.step_count)
        self._last_upper_cluster_provenance_by_level[lower_level_index] = {
            'mode': proposal_mode,
            'source': provenance_source,
            'group_count': int(final_group_count),
            'semantic_group_count': None if semantic_group_count is None else int(semantic_group_count),
            'significance_group_count': None if significance_group_count is None else int(significance_group_count),
            'graph_group_count': None if graph_group_count is None else int(graph_group_count),
            'step_count': int(self.step_count),
            'active_tokens': active_tuple,
        }
        return active_tokens, token_labels

    def _cluster_level1_tokens(
        self,
        source_tokens: IntArray,
        **kwargs: Any,
    ) -> tuple[IntArray, IntArray]:
        return self._cluster_upper_level_tokens(0, source_tokens, **kwargs)

    def _lift_token_labels_to_nodes(
        self,
        source_tokens: IntArray,
        active_tokens: IntArray,
        token_labels: IntArray,
    ) -> IntArray:
        lifted = np.full(self.input_size, -1, dtype=int)
        token_to_label = {
            int(token): int(label)
            for token, label in zip(active_tokens.tolist(), token_labels.tolist())
        }
        for node_idx, token in enumerate(source_tokens.tolist()):
            if token >= 0:
                lifted[node_idx] = token_to_label.get(int(token), -1)
        return lifted

    @staticmethod
    def _lift_labels_through_source_tokens(
        source_tokens: IntArray,
        upper_labels: IntArray,
    ) -> IntArray:
        lifted = np.full(len(source_tokens), -1, dtype=int)
        for idx, token in enumerate(np.asarray(source_tokens, dtype=int).tolist()):
            if token >= 0 and int(token) < len(upper_labels):
                lifted[idx] = int(upper_labels[int(token)])
        return lifted

    @staticmethod
    def _cluster_group_geometry(
        positions: FloatArray,
        labels: IntArray,
    ) -> tuple[
        IntArray,
        list[NDArray[np.bool_]],
        FloatArray,
        FloatArray,
        dict[int, int],
    ]:
        valid_labels = np.array(sorted({int(label) for label in labels if label >= 0}), dtype=int)
        centroids = np.zeros((valid_labels.size, positions.shape[1]), dtype=float)
        sizes = np.ones(valid_labels.size, dtype=float)
        group_masks: list[NDArray[np.bool_]] = []
        for idx, label in enumerate(valid_labels.tolist()):
            mask = labels == label
            group_masks.append(mask)
            centroids[idx] = np.mean(positions[mask], axis=0)
            sizes[idx] = max(1.0, float(np.sum(mask)))
        label_to_group = {int(label): idx for idx, label in enumerate(valid_labels.tolist())}
        return valid_labels, group_masks, centroids, sizes, label_to_group

    @staticmethod
    def _cluster_pair_distance_direction(
        positions: FloatArray,
        group_masks: list[NDArray[np.bool_]],
        centroids: FloatArray,
        idx_i: int,
        idx_j: int,
        *,
        distance_mode: str = 'centroid',
    ) -> tuple[float, FloatArray]:
        mode = str(distance_mode).lower()
        if mode == 'nearest_point':
            pts_i = positions[group_masks[idx_i]]
            pts_j = positions[group_masks[idx_j]]
            if pts_i.size > 0 and pts_j.size > 0:
                pairwise = pts_j[None, :, :] - pts_i[:, None, :]
                pairwise_dist = np.linalg.norm(pairwise, axis=2)
                flat_idx = int(np.argmin(pairwise_dist))
                local_i, local_j = np.unravel_index(flat_idx, pairwise_dist.shape)
                vector = pairwise[local_i, local_j]
                dist = float(np.linalg.norm(vector))
                if dist > 1e-9:
                    return dist, vector / dist

        vector = centroids[idx_j] - centroids[idx_i]
        dist = float(np.linalg.norm(vector))
        if dist <= 1e-9:
            return 0.0, np.zeros(positions.shape[1], dtype=float)
        return dist, vector / dist

    @staticmethod
    def _relative_cluster_mass(
        sizes: FloatArray,
        *,
        use_cluster_mass: bool,
        mass_power: float,
    ) -> FloatArray:
        if use_cluster_mass:
            mean_size = max(float(np.mean(sizes)), 1e-9)
            return np.power(sizes / mean_size, float(max(mass_power, 0.0)))
        return np.ones_like(sizes)

    @staticmethod
    def _cap_group_shift(
        group_shift: FloatArray,
        max_group_step: float | None,
    ) -> FloatArray:
        if max_group_step is not None and max_group_step > 0.0:
            norms = np.linalg.norm(group_shift, axis=1)
            over = norms > float(max_group_step)
            if np.any(over):
                group_shift = np.array(group_shift, copy=True)
                group_shift[over] *= (float(max_group_step) / norms[over])[:, None]
        return group_shift

    @staticmethod
    def _lift_group_shift(
        positions: FloatArray,
        labels: IntArray,
        group_masks: list[NDArray[np.bool_]],
        group_shift: FloatArray,
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        for idx, mask in enumerate(group_masks):
            if np.any(mask):
                shift[mask] += group_shift[idx]
        return shift

    @staticmethod
    def _compute_rigid_cluster_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        strength: float,
        target_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = 'centroid',
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapX_RG._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift
        relative_mass = SyncMapX_RG._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        group_shift = np.zeros_like(centroids)
        normalizer = float(max(1, valid_labels.size - 1))
        target = max(float(target_distance), 1e-6)

        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                dist, direction = SyncMapX_RG._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode=distance_mode,
                )
                if dist <= 1e-9:
                    continue
                delta = dist - target
                base_move = (float(strength) * delta / normalizer) * direction
                group_shift[i] += relative_mass[j] * base_move
                group_shift[j] -= relative_mass[i] * base_move

        group_shift = SyncMapX_RG._cap_group_shift(group_shift, max_group_step)
        return SyncMapX_RG._lift_group_shift(positions, labels, group_masks, group_shift)

    @staticmethod
    def _compute_cluster_exclusion_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        strength: float,
        exclusion_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = 'centroid',
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapX_RG._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift
        relative_mass = SyncMapX_RG._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        group_shift = np.zeros_like(centroids)
        normalizer = float(max(1, valid_labels.size - 1))
        target = max(float(exclusion_distance), 1e-6)

        for i in range(valid_labels.size - 1):
            for j in range(i + 1, valid_labels.size):
                dist, direction = SyncMapX_RG._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode=distance_mode,
                )
                if dist <= 1e-9:
                    continue
                delta = target - dist
                if delta <= 0.0:
                    continue
                repel = (float(strength) * delta / normalizer) * direction
                group_shift[i] -= relative_mass[j] * repel
                group_shift[j] += relative_mass[i] * repel

        group_shift = SyncMapX_RG._cap_group_shift(group_shift, max_group_step)
        return SyncMapX_RG._lift_group_shift(positions, labels, group_masks, group_shift)

    @staticmethod
    def _compute_conditional_cluster_exclusion_shift(
        positions: FloatArray,
        source_labels: IntArray,
        source_to_super_labels: dict[int, int],
        *,
        strength: float,
        exclusion_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = 'centroid',
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0:
            return shift

        valid_labels, group_masks, centroids, sizes, _ = SyncMapX_RG._cluster_group_geometry(positions, source_labels)
        if valid_labels.size < 2:
            return shift

        relative_mass = SyncMapX_RG._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        group_shift = np.zeros_like(centroids)
        normalizer = float(max(1, valid_labels.size - 1))
        target = max(float(exclusion_distance), 1e-6)

        for i in range(valid_labels.size - 1):
            src_i = int(valid_labels[i])
            super_i = int(source_to_super_labels.get(src_i, -1))
            if super_i < 0:
                continue
            for j in range(i + 1, valid_labels.size):
                src_j = int(valid_labels[j])
                super_j = int(source_to_super_labels.get(src_j, -1))
                if super_j < 0 or super_i == super_j:
                    continue

                dist, direction = SyncMapX_RG._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode=distance_mode,
                )
                if dist <= 1e-9:
                    continue
                delta = target - dist
                if delta <= 0.0:
                    continue
                repel = (float(strength) * delta / normalizer) * direction
                group_shift[i] -= relative_mass[j] * repel
                group_shift[j] += relative_mass[i] * repel

        group_shift = SyncMapX_RG._cap_group_shift(group_shift, max_group_step)
        return SyncMapX_RG._lift_group_shift(positions, source_labels, group_masks, group_shift)

    def _cluster_relation_edges(
        self,
        labels: IntArray,
    ) -> tuple[IntArray, IntArray, FloatArray]:
        rows = getattr(self.level0, '_g_rows', None)
        cols = getattr(self.level0, '_g_cols', None)
        weights = getattr(self.level0, '_g_w', None)
        if rows is None or cols is None or weights is None:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        valid_labels, _group_masks, _centroids, _sizes, label_to_group = self._cluster_group_geometry(
            self.level0.syncmap,
            labels,
        )
        if valid_labels.size < 2:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        edge_map: dict[tuple[int, int], float] = {}
        for node_i, node_j, weight in zip(rows.tolist(), cols.tolist(), np.asarray(weights, dtype=float).tolist()):
            label_i = int(labels[int(node_i)])
            label_j = int(labels[int(node_j)])
            if label_i < 0 or label_j < 0 or label_i == label_j:
                continue
            gi = label_to_group.get(label_i)
            gj = label_to_group.get(label_j)
            if gi is None or gj is None or gi == gj:
                continue
            a, b = (gi, gj) if gi < gj else (gj, gi)
            prev = edge_map.get((a, b), 0.0)
            edge_map[(a, b)] = max(prev, float(weight))

        if not edge_map:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        keys = sorted(edge_map.keys())
        src = np.array([a for a, _ in keys], dtype=int)
        dst = np.array([b for _, b in keys], dtype=int)
        w = np.array([edge_map[key] for key in keys], dtype=float)
        return src, dst, w

    @staticmethod
    def _compute_cluster_graph_shift(
        positions: FloatArray,
        labels: IntArray,
        edge_src: IntArray,
        edge_dst: IntArray,
        edge_weights: FloatArray,
        *,
        strength: float,
        target_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = 'centroid',
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0 or edge_src.size == 0:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapX_RG._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift
        relative_mass = SyncMapX_RG._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )

        weights = np.asarray(edge_weights, dtype=float)
        mean_weight = max(float(np.mean(weights)), 1e-6)
        norm_weights = np.clip(weights / mean_weight, 0.25, 4.0)

        degree = np.zeros(valid_labels.size, dtype=float)
        for i, j, w in zip(edge_src.tolist(), edge_dst.tolist(), norm_weights.tolist()):
            degree[int(i)] += float(w)
            degree[int(j)] += float(w)

        target = max(float(target_distance), 1e-6)
        group_shift = np.zeros_like(centroids)
        for i, j, w in zip(edge_src.tolist(), edge_dst.tolist(), norm_weights.tolist()):
            dist, direction = SyncMapX_RG._cluster_pair_distance_direction(
                positions,
                group_masks,
                centroids,
                int(i),
                int(j),
                distance_mode=distance_mode,
            )
            if dist <= 1e-9:
                continue
            delta = dist - target
            base = float(strength) * float(w) * delta * direction
            group_shift[int(i)] += (relative_mass[int(j)] * base) / max(degree[int(i)], 1.0)
            group_shift[int(j)] -= (relative_mass[int(i)] * base) / max(degree[int(j)], 1.0)

        group_shift = SyncMapX_RG._cap_group_shift(group_shift, max_group_step)
        return SyncMapX_RG._lift_group_shift(positions, labels, group_masks, group_shift)

    @staticmethod
    def _clip_point_shift(
        shift: FloatArray,
        max_point_step: float,
    ) -> FloatArray:
        if max_point_step <= 0.0:
            return shift
        norms = np.linalg.norm(shift, axis=1)
        over = norms > float(max_point_step)
        if not np.any(over):
            return shift
        clipped = np.array(shift, copy=True)
        clipped[over] *= (float(max_point_step) / norms[over])[:, None]
        return clipped

    def _level0_cluster_target_distance(self) -> float:
        # 兼容旧诊断入口：第一层 coarse 的目标距离现在直接等于该层自身的平衡距离，
        # 不再由 L0 额外乘一遍 inter_cluster_distance_ratio 推出来。
        return self._level_equilibrium_distance(1)

    def _apply_cluster_force_to_level0(
        self,
        labels: IntArray,
        *,
        exclusion_strength: float,
        target_distance: float,
        max_step_ratio: float,
        use_cluster_mass: bool,
        mass_power: float,
        graph_strength: float = 0.0,
    ) -> None:
        if len({int(label) for label in labels if label >= 0}) < 2:
            return

        max_step = max(
            float(getattr(self.level0, 'graph_r0', 1.0)) * float(max_step_ratio),
            1e-6,
        )
        shift = np.zeros_like(self.level0.syncmap)
        if graph_strength > 0.0:
            edge_src, edge_dst, edge_w = self._cluster_relation_edges(labels)
            shift += self._compute_cluster_graph_shift(
                self.level0.syncmap,
                labels,
                edge_src,
                edge_dst,
                edge_w,
                strength=float(graph_strength),
                target_distance=target_distance,
                max_group_step=max_step,
                use_cluster_mass=bool(use_cluster_mass),
                mass_power=float(mass_power),
                distance_mode='nearest_point',
            )
        if exclusion_strength > 0.0:
            shift += self._compute_cluster_exclusion_shift(
                self.level0.syncmap,
                labels,
                strength=float(exclusion_strength),
                exclusion_distance=target_distance,
                max_group_step=max_step,
                use_cluster_mass=bool(use_cluster_mass),
                mass_power=float(mass_power),
                distance_mode='nearest_point',
            )
        if not np.any(np.abs(shift) > 1e-12):
            return

        shift = self._clip_point_shift(shift, max_step)
        self.level0.syncmap += shift
        if self.level0.space_size > 0:
            self.level0._apply_spherical_constraints()
        self.level0._recenter_syncmap()

    def _level1_target_distance(self) -> float:
        """Return the intrinsic equilibrium distance for the first coarse layer."""
        return self._level_equilibrium_distance(1)

    def _apply_inter_level_rigid_shift(
        self,
        lower_model: SyncMapNS,
        upper_model: SyncMapNS,
        lower_to_upper_tokens: IntArray,
        previous_upper_syncmap: FloatArray,
    ) -> None:
        if not self.inter_level_rigid_shift_enabled:
            return
        strength = float(self.inter_level_rigid_shift_strength) * float(self._feedback_strength_scale())
        if strength <= 0.0:
            return

        token_map = np.asarray(lower_to_upper_tokens, dtype=int).reshape(-1)
        if token_map.shape[0] != lower_model.syncmap.shape[0]:
            return

        active_tokens = np.array(sorted({int(token) for token in token_map if token >= 0}), dtype=int)
        if active_tokens.size == 0:
            return

        previous_upper = np.asarray(previous_upper_syncmap, dtype=float)
        current_upper = np.asarray(upper_model.syncmap, dtype=float)
        if previous_upper.shape != current_upper.shape:
            return

        token_shift = strength * (current_upper[active_tokens] - previous_upper[active_tokens])
        if not np.any(np.abs(token_shift) > 1e-12):
            return

        base_distance = max(
            float(
                getattr(
                    lower_model,
                    'graph_r0_fixed',
                    getattr(lower_model, 'graph_r0', 1.0),
                )
            ),
            1e-6,
        )
        max_step = max(
            float(base_distance) * float(self.inter_level_rigid_shift_max_step_ratio),
            1e-6,
        )
        token_norm = np.linalg.norm(token_shift, axis=1)
        over = token_norm > max_step
        if np.any(over):
            token_shift = np.array(token_shift, copy=True)
            token_shift[over] *= (max_step / token_norm[over])[:, None]

        lower_shift = np.zeros_like(lower_model.syncmap)
        for token, shift in zip(active_tokens.tolist(), token_shift):
            members = token_map == int(token)
            if np.any(members):
                lower_shift[members] += shift

        if not np.any(np.abs(lower_shift) > 1e-12):
            return

        lower_model.syncmap += lower_shift
        if lower_model.space_size > 0:
            lower_model._apply_spherical_constraints()
        lower_model._recenter_syncmap()

    def _apply_inter_level_centroid_anchor(
        self,
        lower_model: SyncMapNS,
        upper_model: SyncMapNS,
        lower_to_upper_tokens: IntArray,
    ) -> None:
        if not self.inter_level_centroid_anchor_enabled:
            return
        strength = float(self.inter_level_centroid_anchor_strength) * float(self._feedback_strength_scale())
        if strength <= 0.0:
            return

        token_map = np.asarray(lower_to_upper_tokens, dtype=int).reshape(-1)
        if token_map.shape[0] != lower_model.syncmap.shape[0]:
            return

        active_tokens = np.array(sorted({int(token) for token in token_map if token >= 0}), dtype=int)
        if active_tokens.size == 0:
            return

        base_distance = max(
            float(
                getattr(
                    lower_model,
                    'graph_r0_fixed',
                    getattr(lower_model, 'graph_r0', 1.0),
                )
            ),
            1e-6,
        )
        max_step = max(float(base_distance) * float(self.inter_level_centroid_anchor_max_step_ratio), 1e-6)
        lower_shift = np.zeros_like(lower_model.syncmap)

        for token in active_tokens.tolist():
            members = token_map == int(token)
            if not np.any(members):
                continue
            anchor = np.asarray(upper_model.syncmap[int(token)], dtype=float)
            # Elastic feedback must act pointwise so wrongly merged members can still separate later.
            member_shift = strength * (anchor[None, :] - np.asarray(lower_model.syncmap[members], dtype=float))
            lower_shift[members] += member_shift

        if not np.any(np.abs(lower_shift) > 1e-12):
            return

        lower_shift = self._clip_point_shift(lower_shift, max_step)
        lower_model.syncmap += lower_shift
        if lower_model.space_size > 0:
            lower_model._apply_spherical_constraints()
        lower_model._recenter_syncmap()

    def _apply_cluster_force_to_upper_level(
        self,
        lower_level_index: int,
        labels: IntArray,
        *,
        exclusion_strength: float,
        target_distance: float,
        max_step_ratio: float = 0.1,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
    ) -> None:
        self._ensure_level_count(lower_level_index + 2)
        upper_model = self.levels[lower_level_index + 1]
        if exclusion_strength <= 0.0 or target_distance <= 0.0:
            return

        positions = upper_model.syncmap
        labels_arr = np.asarray(labels, dtype=int).reshape(-1)
        valid_labels = np.array(sorted({int(label) for label in labels_arr if label >= 0}), dtype=int)
        if valid_labels.size < 2:
            return

        max_step = max(float(target_distance) * float(max_step_ratio), 1e-6)
        shift = self._compute_rigid_cluster_shift(
            positions,
            labels_arr,
            strength=exclusion_strength,
            target_distance=target_distance,
            max_group_step=max_step,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        if not np.any(np.abs(shift) > 1e-12):
            return
        shift = self._clip_point_shift(shift, max_step)

        upper_model.syncmap += shift
        if upper_model.space_size > 0:
            upper_model._apply_spherical_constraints()
        upper_model._recenter_syncmap()

    def _apply_cluster_force_to_level1(
        self,
        labels: IntArray,
        *,
        exclusion_strength: float,
        target_distance: float,
        max_step_ratio: float = 0.1,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
    ) -> None:
        """对Level1应用簇间距离维持力。
        
        这是RG设计的关键部分：确保L1上的聚类簇之间维持正确的距离。
        """
        self._apply_cluster_force_to_upper_level(
            0,
            labels,
            exclusion_strength=exclusion_strength,
            target_distance=target_distance,
            max_step_ratio=max_step_ratio,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )

    @staticmethod
    def _component_labels_from_model_graph(
        model: SyncMapNS,
        tokens: IntArray,
    ) -> IntArray:
        token_arr = np.asarray(tokens, dtype=int).reshape(-1)
        if token_arr.size == 0:
            return np.zeros(0, dtype=int)
        if token_arr.size == 1:
            return np.array([0], dtype=int)

        token_list = [int(token) for token in token_arr.tolist() if int(token) >= 0]
        if not token_list:
            return np.zeros(0, dtype=int)
        token_set = set(token_list)
        parent = {token: token for token in token_list}

        def find(token: int) -> int:
            root = token
            while parent[root] != root:
                root = parent[root]
            while parent[token] != token:
                nxt = parent[token]
                parent[token] = root
                token = nxt
            return root

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        raw_rows = getattr(model, '_g_rows', None)
        raw_cols = getattr(model, '_g_cols', None)
        raw_weights = getattr(model, '_g_w', None)
        rows = np.asarray(raw_rows if raw_rows is not None else np.asarray([], dtype=int), dtype=int)
        cols = np.asarray(raw_cols if raw_cols is not None else np.asarray([], dtype=int), dtype=int)
        weights = np.asarray(raw_weights if raw_weights is not None else np.asarray([], dtype=float), dtype=float)
        for row, col, weight in zip(rows.tolist(), cols.tolist(), weights.tolist()):
            if float(weight) <= 0.0:
                continue
            a = int(row)
            b = int(col)
            if a in token_set and b in token_set:
                union(a, b)

        root_to_label: dict[int, int] = {}
        labels = np.full(token_arr.size, -1, dtype=int)
        next_label = 0
        for idx, token in enumerate(token_arr.tolist()):
            token_id = int(token)
            if token_id < 0:
                continue
            root = find(token_id)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[idx] = root_to_label[root]
        return labels

    @staticmethod
    def _pair_key(a: int, b: int) -> tuple[int, int]:
        ia = int(a)
        ib = int(b)
        return (ia, ib) if ia < ib else (ib, ia)

    def _persistent_relation_state(self, evidence: float) -> str:
        threshold = max(float(self.persistent_relation_threshold), 1e-6)
        if float(evidence) >= threshold:
            return 'same'
        if float(evidence) <= -threshold:
            return 'diff'
        return 'undecided'

    def _persistent_relation_surprise(self, state: str, signal: int) -> float:
        if signal == 0:
            return 0.0
        if state == 'undecided':
            return float(self.persistent_relation_surprise_mid)
        if (state == 'same' and signal > 0) or (state == 'diff' and signal < 0):
            return float(self.persistent_relation_surprise_low)
        return float(self.persistent_relation_surprise_high)

    def _source_token_same_pairs_from_lower_graph(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        support_tokens: IntArray,
    ) -> list[tuple[int, int]]:
        """Lift lower-level graph support edges into source-token same-pair evidence.

        This lets a single lower-level adjacency between two source units seed a
        provisional upper-level relation even before the upper graph has grown a
        persistent edge of its own.
        """
        self._ensure_level_count(lower_level_index + 1)
        lower_model = self.levels[lower_level_index]
        labels = np.asarray(source_tokens, dtype=int).reshape(-1)
        if labels.shape[0] != lower_model.syncmap.shape[0]:
            return []
        support_set = {int(token) for token in np.asarray(support_tokens, dtype=int).tolist() if int(token) >= 0}
        if len(support_set) < 2:
            return []

        raw_rows = getattr(lower_model, "_g_rows", None)
        raw_cols = getattr(lower_model, "_g_cols", None)
        raw_weights = getattr(lower_model, "_g_w", None)
        rows = np.asarray(raw_rows if raw_rows is not None else np.asarray([], dtype=int), dtype=int)
        cols = np.asarray(raw_cols if raw_cols is not None else np.asarray([], dtype=int), dtype=int)
        weights = np.asarray(raw_weights if raw_weights is not None else np.asarray([], dtype=float), dtype=float)
        if rows.size == 0 or cols.size == 0 or weights.size == 0:
            return []

        same_pairs: set[tuple[int, int]] = set()
        for row, col, weight in zip(rows.tolist(), cols.tolist(), weights.tolist()):
            if float(weight) <= 0.0:
                continue
            a = int(labels[int(row)])
            b = int(labels[int(col)])
            if a < 0 or b < 0 or a == b:
                continue
            if a not in support_set or b not in support_set:
                continue
            same_pairs.add(self._pair_key(a, b))
        return sorted(same_pairs)

    def _persistent_relation_component_labels(
        self,
        lower_level_index: int,
        tokens: IntArray,
    ) -> IntArray | None:
        if not self.persistent_relation_force_enabled:
            return None
        token_arr = np.asarray(tokens, dtype=int).reshape(-1)
        token_arr = token_arr[token_arr >= 0]
        if token_arr.size < 2:
            return None
        if not (0 <= lower_level_index < len(self._pair_relation_evidence_by_level)):
            return None

        evidence_map = dict(self._pair_relation_evidence_by_level[lower_level_index])
        if not evidence_map:
            return None

        threshold = max(float(self.persistent_relation_threshold), 1e-6)
        token_list = [int(token) for token in token_arr.tolist()]
        token_set = set(token_list)
        parent = {token: token for token in token_list}
        same_edge_count = 0

        def find(token: int) -> int:
            root = token
            while parent[root] != root:
                root = parent[root]
            while parent[token] != token:
                nxt = parent[token]
                parent[token] = root
                token = nxt
            return root

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for (a, b), value in evidence_map.items():
            if float(value) < threshold:
                continue
            ia = int(a)
            ib = int(b)
            if ia not in token_set or ib not in token_set:
                continue
            union(ia, ib)
            same_edge_count += 1

        if same_edge_count <= 0:
            return None

        root_to_label: dict[int, int] = {}
        labels = np.full(token_arr.size, -1, dtype=int)
        next_label = 0
        for idx, token in enumerate(token_list):
            root = find(int(token))
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[idx] = root_to_label[root]

        if next_label >= token_arr.size:
            return None
        return labels

    def _update_persistent_relation_evidence(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        support_tokens: IntArray,
        active_tokens: IntArray,
    ) -> None:
        if not self.persistent_relation_force_enabled:
            return
        support_arr = np.asarray(support_tokens, dtype=int)
        support_arr = np.unique(support_arr[support_arr >= 0])
        if support_arr.size < 2:
            if 0 <= lower_level_index < len(self._pair_relation_evidence_by_level):
                self._pair_relation_evidence_by_level[lower_level_index] = {}
            return

        support_set = {int(token) for token in support_arr.tolist()}
        source_arr = np.asarray(source_tokens, dtype=int).reshape(-1)
        active_arr = np.asarray(active_tokens, dtype=int)
        active_set = {int(token) for token in active_arr.tolist() if int(token) in support_set}
        evidence_map = dict(self._pair_relation_evidence_by_level[lower_level_index])
        next_map: dict[tuple[int, int], float] = {}
        decay = float(np.clip(self.persistent_relation_decay, 0.0, 1.0))

        for key, value in evidence_map.items():
            if key[0] not in support_set or key[1] not in support_set:
                continue
            decayed = decay * float(value)
            if abs(decayed) > 1e-6:
                next_map[key] = decayed

        upper_model = self.levels[lower_level_index + 1]
        gain = float(self.persistent_relation_signal_gain)
        active_list = sorted(active_set)

        same_groups: list[list[int]] = []
        if len(active_list) >= 2:
            component_labels = self._component_labels_from_model_graph(
                upper_model,
                np.asarray(active_list, dtype=int),
            )
            groups: dict[int, list[int]] = {}
            for token, label in zip(active_list, np.asarray(component_labels, dtype=int).tolist()):
                if int(label) < 0:
                    continue
                groups.setdefault(int(label), []).append(int(token))
            same_groups = [
                sorted(tokens)
                for tokens in groups.values()
                if len(tokens) >= 2
            ]

        same_pairs: set[tuple[int, int]] = set(
            self._source_token_same_pairs_from_lower_graph(
                lower_level_index,
                source_arr,
                support_arr,
            )
        )
        if len(active_list) >= 2:
            for i in range(len(active_list) - 1):
                for j in range(i + 1, len(active_list)):
                    same_pairs.add(self._pair_key(active_list[i], active_list[j]))
        same_anchor_tokens: set[int] = set()
        for tokens in same_groups:
            same_anchor_tokens.update(int(token) for token in tokens)
            for i in range(len(tokens) - 1):
                for j in range(i + 1, len(tokens)):
                    same_pairs.add(self._pair_key(tokens[i], tokens[j]))
        for a, b in same_pairs:
            same_anchor_tokens.add(int(a))
            same_anchor_tokens.add(int(b))

        diff_pairs: list[tuple[int, int]] = []
        if same_anchor_tokens:
            inactive_list = sorted(support_set - same_anchor_tokens)
            for a in sorted(same_anchor_tokens):
                for b in inactive_list:
                    diff_pairs.append(self._pair_key(a, b))

        same_pair_list = sorted(same_pairs)
        same_scale = gain / float(max(1, len(same_pair_list)))
        diff_scale = gain / float(max(1, len(diff_pairs)))

        for key in same_pair_list:
            prev = float(next_map.get(key, 0.0))
            surprise = self._persistent_relation_surprise(
                self._persistent_relation_state(prev),
                1,
            )
            updated = prev + same_scale * surprise
            if abs(updated) > 1e-6:
                next_map[key] = updated
            elif key in next_map:
                next_map.pop(key, None)

        for key in diff_pairs:
            prev = float(next_map.get(key, 0.0))
            surprise = self._persistent_relation_surprise(
                self._persistent_relation_state(prev),
                -1,
            )
            updated = prev - diff_scale * surprise
            if abs(updated) > 1e-6:
                next_map[key] = updated
            elif key in next_map:
                next_map.pop(key, None)

        self._pair_relation_evidence_by_level[lower_level_index] = next_map

    @staticmethod
    def _compute_selected_cluster_exclusion_shift(
        positions: FloatArray,
        labels: IntArray,
        *,
        selected_pairs: set[tuple[int, int]],
        strength: float,
        exclusion_distance: float,
        max_group_step: float | None = None,
        use_cluster_mass: bool = True,
        mass_power: float = 1.0,
        distance_mode: str = 'centroid',
    ) -> FloatArray:
        shift = np.zeros_like(positions)
        if strength <= 0.0 or not selected_pairs:
            return shift
        valid_labels, group_masks, centroids, sizes, _ = SyncMapX_RG._cluster_group_geometry(positions, labels)
        if valid_labels.size < 2:
            return shift
        relative_mass = SyncMapX_RG._relative_cluster_mass(
            sizes,
            use_cluster_mass=use_cluster_mass,
            mass_power=mass_power,
        )
        group_shift = np.zeros_like(centroids)
        normalizer = float(max(1, len(selected_pairs)))
        target = max(float(exclusion_distance), 1e-6)

        for i in range(valid_labels.size - 1):
            label_i = int(valid_labels[i])
            for j in range(i + 1, valid_labels.size):
                label_j = int(valid_labels[j])
                pair = (label_i, label_j) if label_i < label_j else (label_j, label_i)
                if pair not in selected_pairs:
                    continue
                dist, direction = SyncMapX_RG._cluster_pair_distance_direction(
                    positions,
                    group_masks,
                    centroids,
                    i,
                    j,
                    distance_mode=distance_mode,
                )
                if dist <= 1e-9:
                    continue
                delta = target - dist
                if delta <= 0.0:
                    continue
                repel = (float(strength) * delta / normalizer) * direction
                group_shift[i] -= relative_mass[j] * repel
                group_shift[j] += relative_mass[i] * repel

        group_shift = SyncMapX_RG._cap_group_shift(group_shift, max_group_step)
        return SyncMapX_RG._lift_group_shift(positions, labels, group_masks, group_shift)

    def _apply_persistent_relation_forces_to_upper_level(
        self,
        lower_level_index: int,
        support_tokens: IntArray,
    ) -> None:
        if not self.persistent_relation_force_enabled:
            return
        self._ensure_level_count(lower_level_index + 2)
        support_arr = np.asarray(support_tokens, dtype=int)
        support_arr = np.unique(support_arr[support_arr >= 0])
        if support_arr.size < 2:
            return

        evidence_map = self._pair_relation_evidence_by_level[lower_level_index]
        if not evidence_map:
            return

        upper_model = self.levels[lower_level_index + 1]
        positions = np.asarray(upper_model.syncmap[support_arr], dtype=float)
        local_labels = np.arange(support_arr.size, dtype=int)

        parent = list(range(support_arr.size))

        def find(idx: int) -> int:
            root = idx
            while parent[root] != root:
                root = parent[root]
            while parent[idx] != idx:
                nxt = parent[idx]
                parent[idx] = root
                idx = nxt
            return root

        def union(a_idx: int, b_idx: int) -> None:
            ra = find(a_idx)
            rb = find(b_idx)
            if ra != rb:
                parent[rb] = ra

        token_to_local = {int(token): idx for idx, token in enumerate(support_arr.tolist())}
        same_src: list[int] = []
        same_dst: list[int] = []
        same_weights: list[float] = []
        threshold = max(float(self.persistent_relation_threshold), 1e-6)
        for (a, b), value in evidence_map.items():
            if float(value) < threshold:
                continue
            ia = token_to_local.get(int(a))
            ib = token_to_local.get(int(b))
            if ia is None or ib is None:
                continue
            union(int(ia), int(ib))
            same_src.append(int(ia))
            same_dst.append(int(ib))
            same_weights.append(float(np.clip(abs(float(value)) / threshold, 0.25, 4.0)))

        root_to_label: dict[int, int] = {}
        component_labels = np.full(support_arr.size, -1, dtype=int)
        next_label = 0
        for idx in range(support_arr.size):
            root = find(idx)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            component_labels[idx] = root_to_label[root]

        diff_pairs: set[tuple[int, int]] = set()
        for (a, b), value in evidence_map.items():
            if float(value) > -threshold:
                continue
            ia = token_to_local.get(int(a))
            ib = token_to_local.get(int(b))
            if ia is None or ib is None:
                continue
            ca = int(component_labels[int(ia)])
            cb = int(component_labels[int(ib)])
            if ca == cb:
                continue
            diff_pairs.add((ca, cb) if ca < cb else (cb, ca))

        shift = np.zeros_like(positions)
        same_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        same_strength *= max(float(self.persistent_relation_same_force_strength_ratio), 0.0)
        diff_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        diff_strength *= max(float(self.persistent_relation_diff_force_strength_ratio), 0.0)

        same_target = max(float(self._level_equilibrium_distance(lower_level_index + 1)), 1e-6)
        next_level_index = lower_level_index + 2
        if next_level_index < len(self.levels):
            diff_target = max(float(self._level_equilibrium_distance(next_level_index)), 1e-6)
        else:
            diff_target = max(
                float(self._level_equilibrium_distance(lower_level_index + 1) * self.rg_scale_factor),
                1e-6,
            )
        max_step = max(
            max(same_target, diff_target) * float(self.inter_level_rigid_shift_max_step_ratio),
            1e-6,
        )

        if same_src and same_strength > 0.0:
            shift += self._compute_cluster_graph_shift(
                positions,
                local_labels,
                np.asarray(same_src, dtype=int),
                np.asarray(same_dst, dtype=int),
                np.asarray(same_weights, dtype=float),
                strength=same_strength,
                target_distance=same_target,
                max_group_step=max_step,
                use_cluster_mass=False,
                mass_power=1.0,
                distance_mode='nearest_point',
            )
        if diff_pairs and diff_strength > 0.0 and len({int(x) for x in component_labels.tolist() if int(x) >= 0}) >= 2:
            shift += self._compute_selected_cluster_exclusion_shift(
                positions,
                component_labels,
                selected_pairs=diff_pairs,
                strength=diff_strength,
                exclusion_distance=diff_target,
                max_group_step=max_step,
                use_cluster_mass=True,
                mass_power=1.0,
                distance_mode='nearest_point',
            )

        if not np.any(np.abs(shift) > 1e-12):
            return
        shift = self._clip_point_shift(shift, max_step)
        upper_model.syncmap[support_arr] += shift
        if upper_model.space_size > 0:
            upper_model._apply_spherical_constraints()
        upper_model._recenter_syncmap()

    def _apply_upper_cluster_separation(
        self,
        lower_level_index: int,
        upper_active_tokens: IntArray,
        upper_cluster_labels: IntArray,
    ) -> None:
        self._ensure_level_count(lower_level_index + 2)
        upper_model = self.levels[lower_level_index + 1]
        exclusion_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        if exclusion_strength <= 0.0:
            return

        active_tokens = np.asarray(upper_active_tokens, dtype=int).reshape(-1)
        graph_component_labels = self._component_labels_from_model_graph(upper_model, active_tokens)
        cluster_labels = np.asarray(upper_cluster_labels, dtype=int).reshape(-1)
        if active_tokens.size == 0:
            return
        if graph_component_labels.size == active_tokens.size:
            graph_group_count = len({int(label) for label in graph_component_labels.tolist() if int(label) >= 0})
        else:
            graph_group_count = 0
        if cluster_labels.size == active_tokens.size:
            cluster_group_count = len({int(label) for label in cluster_labels.tolist() if int(label) >= 0})
        else:
            cluster_group_count = 0
        if graph_group_count >= 2:
            separation_labels = graph_component_labels
        elif cluster_group_count >= 2:
            separation_labels = cluster_labels
        else:
            return

        labels = np.full(upper_model.input_size, -1, dtype=int)
        valid = (active_tokens >= 0) & (active_tokens < upper_model.input_size) & (separation_labels >= 0)
        if not np.any(valid):
            return
        labels[active_tokens[valid]] = separation_labels[valid]

        next_level_index = lower_level_index + 2
        if next_level_index < len(self.levels):
            target_distance = max(float(self._level_equilibrium_distance(next_level_index)), 1e-6)
        else:
            target_distance = max(
                float(self._level_equilibrium_distance(lower_level_index + 1) * self.rg_scale_factor),
                1e-6,
            )
        self._apply_cluster_force_to_upper_level(
            lower_level_index,
            labels,
            exclusion_strength=exclusion_strength,
            target_distance=target_distance,
            max_step_ratio=0.1,
            use_cluster_mass=True,
            mass_power=1.0,
        )

    def _apply_support_barrier_to_lower_level(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        upper_cluster_labels: IntArray,
        upper_active_tokens: IntArray,
    ) -> None:
        self._ensure_level_count(lower_level_index + 2)
        lower_model = self.levels[lower_level_index]
        upper_model = self.levels[lower_level_index + 1]

        labels = np.asarray(source_tokens, dtype=int).reshape(-1)
        if labels.shape[0] != lower_model.syncmap.shape[0]:
            return
        valid_count = len({int(token) for token in labels.tolist() if int(token) >= 0})
        if valid_count < 2:
            return

        if not self.source_unit_barrier_enabled:
            return
        if not self._feedback_allows_barrier():
            return

        exclusion_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        exclusion_strength *= max(float(self.source_unit_barrier_strength_ratio), 0.0)
        if exclusion_strength <= 0.0:
            return

        source_to_super_labels = {
            int(token): int(label)
            for token, label in zip(
                np.asarray(upper_active_tokens, dtype=int).tolist(),
                np.asarray(upper_cluster_labels, dtype=int).tolist(),
            )
            if int(token) >= 0 and int(label) >= 0
        }
        if len(set(source_to_super_labels.values())) < 2:
            return

        next_level_index = lower_level_index + 2
        if next_level_index < len(self.levels):
            target_distance = max(float(self._level_equilibrium_distance(next_level_index)), 1e-6)
        else:
            target_distance = max(
                float(self._level_equilibrium_distance(lower_level_index + 1) * self.rg_scale_factor),
                1e-6,
            )
        max_step = max(float(getattr(lower_model, 'max_step_norm', target_distance)), 1e-6)
        shift = self._compute_conditional_cluster_exclusion_shift(
            np.asarray(lower_model.syncmap, dtype=float),
            labels,
            source_to_super_labels,
            strength=exclusion_strength,
            exclusion_distance=target_distance,
            max_group_step=max_step,
            use_cluster_mass=True,
            mass_power=1.0,
            distance_mode='nearest_point',
        )
        if not np.any(np.abs(shift) > 1e-12):
            return

        shift = self._clip_point_shift(shift, max_step)
        lower_model.syncmap += shift
        if lower_model.space_size > 0:
            lower_model._apply_spherical_constraints()
        lower_model._recenter_syncmap()

    def _apply_source_unit_barrier_to_lower_level(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
    ) -> None:
        self._ensure_level_count(lower_level_index + 2)
        lower_model = self.levels[lower_level_index]
        upper_model = self.levels[lower_level_index + 1]

        labels = np.asarray(source_tokens, dtype=int).reshape(-1)
        if labels.shape[0] != lower_model.syncmap.shape[0]:
            return
        valid_count = len({int(token) for token in labels.tolist() if int(token) >= 0})
        if valid_count < 2:
            return
        if not self.source_unit_barrier_enabled:
            return
        if not self._feedback_allows_barrier():
            return

        exclusion_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        exclusion_strength *= float(self._feedback_strength_scale())
        if exclusion_strength <= 0.0:
            return

        target_distance = max(float(self._level_equilibrium_distance(lower_level_index + 1)), 1e-6)
        base_distance = max(
            float(
                getattr(
                    lower_model,
                    'graph_r0_fixed',
                    getattr(lower_model, 'graph_r0', 1.0),
                )
            ),
            1e-6,
        )
        max_step = max(float(base_distance) * float(self.inter_level_rigid_shift_max_step_ratio), 1e-6)
        shift = self._compute_cluster_exclusion_shift(
            np.asarray(lower_model.syncmap, dtype=float),
            labels,
            strength=exclusion_strength,
            exclusion_distance=target_distance,
            max_group_step=max_step,
            use_cluster_mass=True,
            mass_power=1.0,
            distance_mode='nearest_point',
        )
        if not np.any(np.abs(shift) > 1e-12):
            return

        shift = self._clip_point_shift(shift, max_step)
        lower_model.syncmap += shift
        if lower_model.space_size > 0:
            lower_model._apply_spherical_constraints()
        lower_model._recenter_syncmap()

    def _apply_source_unit_motion_coherence(
        self,
        lower_level_index: int,
        previous_source_tokens: IntArray,
        lower_active_indices: IntArray,
    ) -> None:
        if not self.source_unit_motion_coherence_enabled:
            return
        if self._feedback_strength_scale() <= 0.0:
            return
        self._ensure_level_count(lower_level_index + 1)
        lower_model = self.levels[lower_level_index]
        labels = np.asarray(previous_source_tokens, dtype=int).reshape(-1)
        if labels.shape[0] != lower_model.syncmap.shape[0]:
            return
        if lower_active_indices.size == 0:
            return

        last_update = np.asarray(getattr(lower_model, '_last_total_update', np.zeros_like(lower_model.syncmap)), dtype=float)
        if last_update.shape != lower_model.syncmap.shape:
            return
        if not np.any(np.abs(last_update) > 1e-12):
            return

        active_labels = labels[np.asarray(lower_active_indices, dtype=int)]
        active_tokens = np.array(sorted({int(token) for token in active_labels.tolist() if int(token) >= 0}), dtype=int)
        if active_tokens.size == 0:
            return

        strength = float(np.clip(self.source_unit_motion_coherence_strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        max_step = max(
            float(self._level_equilibrium_distance(lower_level_index)) * float(self.source_unit_motion_coherence_max_step_ratio),
            1e-6,
        )
        correction = np.zeros_like(lower_model.syncmap)
        for token in active_tokens.tolist():
            members = np.where(labels == int(token))[0]
            if members.size < 2:
                continue
            mean_update = np.mean(last_update[members], axis=0, keepdims=True)
            correction[members] = strength * (mean_update - last_update[members])
        if not np.any(np.abs(correction) > 1e-12):
            return

        correction = self._clip_point_shift(correction, max_step)
        lower_model.syncmap += correction
        if lower_model.space_size > 0:
            lower_model._apply_spherical_constraints()
        lower_model._recenter_syncmap()

    def _apply_support_unit_repulsion_to_upper_level(
        self,
        lower_level_index: int,
        support_tokens: IntArray,
    ) -> None:
        if not self.support_unit_repulsion_enabled:
            return
        support_arr = np.asarray(support_tokens, dtype=int)
        support_arr = support_arr[support_arr >= 0]
        if support_arr.size < 2:
            return

        upper_model = self.levels[lower_level_index + 1]
        exclusion_strength = max(float(getattr(upper_model, 'cluster_distance_strength', 0.0)), 0.0)
        exclusion_strength *= max(float(self.support_unit_repulsion_strength_ratio), 0.0)
        exclusion_strength *= float(self._feedback_strength_scale())
        if exclusion_strength <= 0.0:
            return

        labels = np.full(upper_model.input_size, -1, dtype=int)
        labels[support_arr] = support_arr
        target_distance = max(float(self._level_equilibrium_distance(lower_level_index + 1)), 1e-6)
        max_step = max(
            float(target_distance) * float(self.inter_level_rigid_shift_max_step_ratio),
            1e-6,
        )
        shift = self._compute_cluster_exclusion_shift(
            np.asarray(upper_model.syncmap, dtype=float),
            labels,
            strength=exclusion_strength,
            exclusion_distance=target_distance,
            max_group_step=max_step,
            use_cluster_mass=False,
            mass_power=1.0,
            distance_mode='nearest_point',
        )
        if not np.any(np.abs(shift) > 1e-12):
            return

        shift = self._clip_point_shift(shift, max_step)
        upper_model.syncmap += shift
        if upper_model.space_size > 0:
            upper_model._apply_spherical_constraints()
        upper_model._recenter_syncmap()

    def _reanchor_upper_level_from_source_geometry(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        support_tokens: IntArray,
    ) -> None:
        if not self.inter_level_support_reanchor_enabled:
            return
        support_tuple = tuple(int(token) for token in np.asarray(support_tokens, dtype=int).tolist())
        if len(support_tuple) < 2:
            self._last_source_support_tokens_by_level[lower_level_index] = support_tuple
            return

        lower_model = self.levels[lower_level_index]
        upper_model = self.levels[lower_level_index + 1]
        strength = float(np.clip(self.inter_level_support_reanchor_strength, 0.0, 1.0))
        strength *= float(self._feedback_strength_scale())
        if strength <= 0.0:
            self._last_source_support_tokens_by_level[lower_level_index] = support_tuple
            return

        positions = np.asarray(lower_model.syncmap, dtype=float)
        source_labels = np.asarray(source_tokens, dtype=int)
        target_positions = np.zeros((len(support_tuple), self.dimensions), dtype=float)
        for idx, token in enumerate(support_tuple):
            members = np.where(source_labels == int(token))[0]
            if members.size == 0:
                continue
            target_positions[idx] = np.mean(positions[members], axis=0)

        target_positions -= np.mean(target_positions, axis=0, keepdims=True)
        diffs = target_positions[:, None, :] - target_positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        valid = dists[np.triu_indices_from(dists, k=1)]
        valid = valid[valid > 1e-9]
        if valid.size > 0:
            current_scale = float(np.median(valid))
            desired_scale = max(float(self._level_equilibrium_distance(lower_level_index + 1)), 1e-6)
            target_positions *= desired_scale / max(current_scale, 1e-6)

        support_arr = np.asarray(support_tuple, dtype=int)
        upper_model.syncmap[support_arr] = (
            (1.0 - strength) * np.asarray(upper_model.syncmap[support_arr], dtype=float)
            + strength * target_positions
        )
        if upper_model.space_size > 0:
            upper_model._apply_spherical_constraints()
        upper_model._recenter_syncmap()
        self._last_source_support_tokens_by_level[lower_level_index] = support_tuple

    def _set_upper_graph_from_coactivation_density(
        self,
        lower_level_index: int,
        source_tokens: IntArray,
        upper_model: SyncMapNS,
    ) -> bool:
        """RG coarse-graining of the macro edge weights: aggregate the LOWER level's
        node co-activation matrix by token membership and set the upper level's graph
        from it. The macro structure is a node co-activation community that token
        transitions wash out; this carries that 'volume' up so the upper spring can
        separate macros. Read-only wrt the lower level (only reads its co-occurrence
        statistic). Returns True if a graph was set."""
        lower = self.levels[lower_level_index]
        coact = getattr(getattr(lower, '_coact', None), 'coactivation_matrix', None)
        if coact is None:
            coact = getattr(getattr(lower, '_coact', None), 'C', None)
        if coact is None:
            return False
        node_coact = np.asarray(coact, dtype=float)
        src = np.asarray(source_tokens, dtype=int).reshape(-1)
        if src.shape[0] != node_coact.shape[0] or node_coact.ndim != 2:
            return False
        tokens = sorted({int(t) for t in src.tolist() if 0 <= int(t) < int(upper_model.input_size)})
        if len(tokens) < 2:
            return False
        members = {t: np.flatnonzero(src == t) for t in tokens}
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        for idx, a in enumerate(tokens):
            ma = members[a]
            if ma.size == 0:
                continue
            for b in tokens[idx + 1:]:
                mb = members[b]
                if mb.size == 0:
                    continue
                w = float(node_coact[np.ix_(ma, mb)].mean())
                if w > 1e-12:
                    rows.append(int(a))
                    cols.append(int(b))
                    weights.append(w)
        if not rows:
            return False
        weight_arr = np.asarray(weights, dtype=float)
        max_w = float(np.max(weight_arr))
        if max_w > 1e-12:
            weight_arr = weight_arr / max_w
        stable_med = upper_model.size0 if upper_model.size0 > 1e-9 else 1.0
        upper_model.set_graph_edges(
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            weight_arr,
            r0_rel=upper_model.graph_r0_rel,
            nn1_median=stable_med,
        )
        return True

    def _propagate_between_levels(self, lower_level_index: int, lower_active_indices: IntArray) -> None:
        if lower_level_index + 1 >= len(self.levels):
            return
        if self.step_count < self.warmup_steps:
            return
        interval = self._renormalize_interval_for_level(lower_level_index)
        if not self._inter_level_schedule_due(lower_level_index):
            return

        active_indices = np.asarray(lower_active_indices, dtype=int)
        if active_indices.size == 0:
            return

        raw_labels = self._current_raw_labels(lower_level_index)
        stable_tokens = self._update_stable_tokens_for_level(lower_level_index, raw_labels)
        macro_tokens = self._macro_feedback_tokens_for_level(lower_level_index, stable_tokens)
        source_tokens = self._resolved_source_tokens_for_level(
            lower_level_index,
            raw_labels,
            stable_tokens,
        )
        lower_model = self.levels[lower_level_index]
        setter = getattr(lower_model, 'set_macro_feedback_labels', None)
        if callable(setter):
            if lower_level_index == 0:
                strength_arr = None
                if lower_level_index < len(self._last_macro_feedback_strengths_by_level):
                    strength_arr = np.asarray(
                        self._last_macro_feedback_strengths_by_level[lower_level_index],
                        dtype=float,
                    )
                setter(
                    np.asarray(macro_tokens, dtype=int),
                    enabled=not self.bottomup_force_architecture,
                    scale=float(max(self.rg_scale_factor, 1.0)),
                    strengths=strength_arr,
                )
            else:
                setter(None, enabled=False)
        entity_setter = getattr(lower_model, 'set_entity_feedback_labels', None)
        if callable(entity_setter):
            if lower_level_index == 0:
                entity_strengths = self._stable_token_evidence_node_strengths(
                    lower_level_index,
                    stable_tokens,
                )
                parent_strengths = None
                if lower_level_index < len(self._last_macro_feedback_strengths_by_level):
                    parent_strengths = np.asarray(
                        self._last_macro_feedback_strengths_by_level[lower_level_index],
                        dtype=float,
                    )
                entity_setter(
                    np.asarray(stable_tokens, dtype=int),
                    parent_labels=np.asarray(macro_tokens, dtype=int),
                    enabled=bool(getattr(lower_model, 'entity_boundary_repulsion_enabled', False)),
                    strengths=entity_strengths,
                    parent_strengths=parent_strengths,
                )
            else:
                entity_setter(None, enabled=False)
        upper_input, valid_tokens, support_tokens = self._build_upper_level_input(
            lower_level_index,
            source_tokens,
            active_indices,
        )
        self._set_level_support(lower_level_index + 1, support_tokens)
        if valid_tokens.size == 0:
            return

        upper_model = self.levels[lower_level_index + 1]
        # STARVATION FIX (physics relaxation, NOT input repetition). The upper level
        # receives macro observations at the slow renormalize cadence (correct: it
        # filters lower-frequency structure), but feeding ONE timestep per
        # observation gave its solver ~1/interval the iterations of the level below,
        # so its spring never converged. Re-feeding the SAME observation would turn a
        # single sharp transition into a sustained blurred co-activation and destroy
        # the macro statistic; instead we present ONE sharp observation and let the
        # solver relax `interval` iterations toward equilibrium. That reaches solver
        # parity with the level below without polluting the statistic. `interval` is
        # the existing per-level renormalize interval; no new parameter is introduced.
        relax_steps = max(1, int(interval))
        prev_steps_per_batch = int(getattr(upper_model, 'physics_steps_per_batch', 1))
        upper_model.physics_steps_per_batch = relax_steps
        # MACRO EDGE WEIGHT (RG coarse-graining). When enabled, set the upper graph
        # from the LOWER level's node co-activation density aggregated by token
        # membership, instead of the upper level's own token->token transitions. The
        # macro structure is a node co-activation community that token transitions
        # wash out (within~cross uniform); this carries the "volume" up. Read-only wrt
        # the lower level. We pin T_push=0 so the relaxation uses this externally-set
        # graph rather than rebuilding it from token transitions.
        prev_tpush = int(getattr(upper_model, 'T_push', 0))
        graph_set = False
        if self.rg_upper_edge_mode == 'coactivation_density':
            graph_set = self._set_upper_graph_from_coactivation_density(
                lower_level_index, source_tokens, upper_model,
            )
            if graph_set:
                upper_model.T_push = 0
        try:
            upper_model.inputGeneral(
                upper_input.reshape(1, -1),
                phase_offset=self._level_phases[lower_level_index + 1],
                reset_trajectory=(self._level_phases[lower_level_index + 1] == 0),
                append_final_frame=False,
            )
        finally:
            upper_model.physics_steps_per_batch = prev_steps_per_batch
            if graph_set:
                upper_model.T_push = prev_tpush
        self._level_phases[lower_level_index + 1] += 1
        self._sync_public_level_aliases()
        self._cluster_upper_level_tokens(
            lower_level_index,
            source_tokens,
        )

        self._propagate_between_levels(lower_level_index + 1, valid_tokens)

    def _propagate_to_level1(self, x_t: NDArray[np.float64]) -> None:
        active_indices = np.where(x_t > 0.1)[0]
        self._propagate_between_levels(0, active_indices)

    def _lift_level_labels_to_nodes(self, level_index: int, **kwargs: Any) -> IntArray:
        _ = kwargs
        if self.hierarchy_readout_mode == 'physical_mst_gaps' and bool(getattr(self, 'level1_enabled', False)):
            physical_levels = self._physical_mst_hierarchy_readouts(max(level_index + 1, 2, self.level_count))
            if level_index < len(physical_levels):
                return np.asarray(physical_levels[level_index], dtype=int).copy()
        if level_index <= 0:
            return self._lift_native_level_labels_to_nodes(0)
        return self._lift_native_level_labels_to_nodes(level_index)

    def _lift_level1_labels_to_nodes(self, **kwargs: Any) -> IntArray:
        return self._lift_level_labels_to_nodes(1, **kwargs)

    def organize(self, level: str | None = None, **kwargs: Any) -> IntArray:
        """Return labels at the requested level.

        cluster_level1 -> Level 0 clustering over original nodes
        cluster_levelN -> Level N-1 clustering lifted back to original nodes
        stable_tokens  -> persistent token IDs over original nodes
        macro_tokens   -> coarse token IDs over original nodes

        Legacy aliases remain accepted for compatibility:
        level0 / cluster_level0 -> cluster_level1
        level1 / cluster_level2_refined / top -> cluster_level2
        """
        target = canonicalize_organize_level(level or self.organize_level)
        cluster_level_index = parse_cluster_level_index(target)
        if cluster_level_index is not None:
            self._ensure_level_count(cluster_level_index + 1)
            return self._lift_level_labels_to_nodes(cluster_level_index, **kwargs)
        if target == 'stable_tokens':
            return self._update_stable_tokens(self._current_level0_raw_labels()).copy()
        if target == 'macro_tokens':
            return self._current_macro_tokens().copy()
        raise ValueError(f"Unsupported organize level: {target}")

    def organize_hierarchy(self, **kwargs: Any) -> dict[str, IntArray]:
        stable_tokens = self.organize(level='stable_tokens', **kwargs)
        macro_tokens = self.organize(level='macro_tokens', **kwargs)
        outputs: dict[str, IntArray] = {
            'stable_tokens': stable_tokens,
            'macro_tokens': macro_tokens,
        }
        for level_index in range(len(self.levels)):
            key = f'cluster_level{level_index + 1}'
            labels = self._lift_level_labels_to_nodes(level_index, **kwargs)
            outputs[key] = labels
            outputs[f'level{level_index}'] = labels
        outputs['cluster_level2_refined'] = np.asarray(macro_tokens, dtype=int).copy()
        return canonicalize_hierarchy_outputs(outputs)

    def refresh_hierarchy_state(self) -> None:
        """Refresh hierarchy bookkeeping from the current embedding state."""
        if not bool(getattr(self, 'level1_enabled', False)):
            return
        self.organize_hierarchy()

    def get_rg_significance_state(self) -> dict[str, Any]:
        significance = self._last_rg_significance
        if significance is None:
            return {'available': False}
        return {
            'available': True,
            'active_tokens': tuple(int(token) for token in np.asarray(significance.active_tokens, dtype=int).tolist()),
            'component_count': int(significance.component_count),
            'best_jump_ratio': float(significance.best_jump_ratio),
            'threshold_distance': float(significance.threshold_distance),
            'threshold_norm': float(significance.threshold_norm),
        }

    def update_params(self, **kwargs: Any) -> None:
        cadence_changed = False
        if 'warmup_steps' in kwargs:
            warmup_steps = int(kwargs['warmup_steps'])
            cadence_changed = cadence_changed or warmup_steps != self.warmup_steps
            self.warmup_steps = warmup_steps
        if 'renormalize_interval' in kwargs:
            renormalize_interval = max(1, int(kwargs['renormalize_interval']))
            cadence_changed = cadence_changed or renormalize_interval != self.renormalize_interval
            self.renormalize_interval = renormalize_interval
        if 'rg_time_scale_mode' in kwargs:
            time_scale_mode = str(kwargs['rg_time_scale_mode']).strip().lower()
            cadence_changed = cadence_changed or time_scale_mode != self.rg_time_scale_mode
            self.rg_time_scale_mode = time_scale_mode
        if 'rg_time_scale_growth' in kwargs:
            time_scale_growth = float(kwargs['rg_time_scale_growth'])
            cadence_changed = cadence_changed or time_scale_growth != self.rg_time_scale_growth
            self.rg_time_scale_growth = time_scale_growth
        if 'rg_inter_level_schedule_mode' in kwargs:
            schedule_mode = self._normalize_inter_level_schedule_mode(
                kwargs['rg_inter_level_schedule_mode']
            )
            if schedule_mode != self.rg_inter_level_schedule_mode:
                self.rg_inter_level_schedule_mode = schedule_mode
                cadence_changed = True
        if cadence_changed:
            self._reset_inter_level_schedule()
        if 'organize_level' in kwargs:
            self.organize_level = canonicalize_organize_level(kwargs['organize_level'])
        if 'rg_input_level' in kwargs:
            self.rg_input_level = str(kwargs['rg_input_level']).lower()
        if 'feedback_mode' in kwargs:
            self.feedback_mode = str(kwargs['feedback_mode']).strip().lower()
        if 'rg_fill_noise_labels' in kwargs:
            self.rg_fill_noise_labels = bool(kwargs['rg_fill_noise_labels'])
        if 'hierarchy_readout_mode' in kwargs:
            self.hierarchy_readout_mode = str(kwargs['hierarchy_readout_mode']).strip().lower()
        if 'stable_token_extraction_mode' in kwargs:
            self.stable_token_extraction_mode = str(kwargs['stable_token_extraction_mode']).strip().lower()
        if 'hierarchy_readout_min_gap_ratio' in kwargs:
            self.hierarchy_readout_min_gap_ratio = max(float(kwargs['hierarchy_readout_min_gap_ratio']), 1.0)
        if 'dbscan_eps_scale_mode' in kwargs:
            self.dbscan_eps_scale_mode = str(kwargs['dbscan_eps_scale_mode']).strip().lower()
        if 'dbscan_eps_margin' in kwargs:
            self.dbscan_eps_margin = max(float(kwargs['dbscan_eps_margin']), 1e-6)
        if 'dbscan_readout_eps' in kwargs:
            raw_readout_eps = kwargs['dbscan_readout_eps']
            self.dbscan_readout_eps = None if raw_readout_eps is None else float(raw_readout_eps)
        if 'dbscan_readout_slack' in kwargs:
            self.dbscan_readout_slack = max(float(kwargs['dbscan_readout_slack']), 1e-6)
        if 'dbscan_readout_min_samples' in kwargs:
            raw_readout_min_samples = kwargs['dbscan_readout_min_samples']
            self.dbscan_readout_min_samples = (
                None if raw_readout_min_samples is None else max(1, int(raw_readout_min_samples))
            )
        if 'rg_knn_k' in kwargs:
            self.rg_knn_k = max(1, int(kwargs['rg_knn_k']))
        if 'rg_jump_ratio' in kwargs:
            self.rg_jump_ratio = float(kwargs['rg_jump_ratio'])
        if 'upper_rg_jump_ratio' in kwargs:
            upper_jump = kwargs['upper_rg_jump_ratio']
            self.upper_rg_jump_ratio = None if upper_jump is None else float(upper_jump)
        if 'rg_jump_ratio_final' in kwargs:
            jump_final = kwargs['rg_jump_ratio_final']
            self.rg_jump_ratio_final = None if jump_final is None else float(jump_final)
        if 'rg_jump_ratio_anneal_start' in kwargs:
            self.rg_jump_ratio_anneal_start = max(0, int(kwargs['rg_jump_ratio_anneal_start']))
        if 'rg_jump_ratio_anneal_end' in kwargs:
            self.rg_jump_ratio_anneal_end = max(0, int(kwargs['rg_jump_ratio_anneal_end']))
        if 'macro_boundary_gap_gate_enabled' in kwargs:
            self.macro_boundary_gap_gate_enabled = bool(kwargs['macro_boundary_gap_gate_enabled'])
        if 'macro_boundary_gap_gate_floor' in kwargs:
            self.macro_boundary_gap_gate_floor = float(np.clip(kwargs['macro_boundary_gap_gate_floor'], 0.0, 1.0))
        if 'macro_boundary_gap_gate_full_ratio' in kwargs:
            self.macro_boundary_gap_gate_full_ratio = max(
                float(kwargs['macro_boundary_gap_gate_full_ratio']),
                self.rg_jump_ratio + 1e-6,
            )
        if 'macro_boundary_min_stable_refreshes' in kwargs:
            self.macro_boundary_min_stable_refreshes = max(0, int(kwargs['macro_boundary_min_stable_refreshes']))
        if 'macro_boundary_token_purity_enabled' in kwargs:
            self.macro_boundary_token_purity_enabled = bool(kwargs['macro_boundary_token_purity_enabled'])
        if 'macro_boundary_token_purity_threshold' in kwargs:
            self.macro_boundary_token_purity_threshold = float(
                np.clip(kwargs['macro_boundary_token_purity_threshold'], 0.0, 1.0)
            )
        if 'macro_boundary_token_purity_floor' in kwargs:
            self.macro_boundary_token_purity_floor = float(
                np.clip(kwargs['macro_boundary_token_purity_floor'], 0.0, 1.0)
            )
        if 'rg_semantic_graph_enabled' in kwargs:
            self.rg_semantic_graph_enabled = bool(kwargs['rg_semantic_graph_enabled'])
        if 'rg_semantic_graph_strength' in kwargs:
            self.rg_semantic_graph_strength = max(float(kwargs['rg_semantic_graph_strength']), 0.0)
        if 'rg_semantic_graph_similarity_floor' in kwargs:
            self.rg_semantic_graph_similarity_floor = float(
                np.clip(kwargs['rg_semantic_graph_similarity_floor'], 0.0, 1.0)
            )
        if 'rg_semantic_graph_penalty_mode' in kwargs:
            self.rg_semantic_graph_penalty_mode = str(kwargs['rg_semantic_graph_penalty_mode']).strip().lower()
        if 'rg_semantic_graph_relative_threshold' in kwargs:
            self.rg_semantic_graph_relative_threshold = float(
                np.clip(kwargs['rg_semantic_graph_relative_threshold'], 0.0, 1.0)
            )
        if 'rg_semantic_graph_anchor_floor' in kwargs:
            self.rg_semantic_graph_anchor_floor = float(
                np.clip(kwargs['rg_semantic_graph_anchor_floor'], 0.0, 1.0)
            )
        if 'rg_semantic_graph_edge_mode' in kwargs:
            self.rg_semantic_graph_edge_mode = str(kwargs['rg_semantic_graph_edge_mode']).strip().lower()
        if 'rg_semantic_graph_topk' in kwargs:
            self.rg_semantic_graph_topk = max(int(kwargs['rg_semantic_graph_topk']), 0)
        if 'rg_fallback_scale' in kwargs:
            self.rg_fallback_scale = float(kwargs['rg_fallback_scale'])
        if 'upper_cluster_proposal_mode' in kwargs:
            self.upper_cluster_proposal_mode = str(kwargs['upper_cluster_proposal_mode']).strip().lower()
        if 'macro_component_proposal_mode' in kwargs:
            self.macro_component_proposal_mode = str(kwargs['macro_component_proposal_mode']).strip().lower()
        if 'upper_cluster_semantic_source' in kwargs:
            self.upper_cluster_semantic_source = str(kwargs['upper_cluster_semantic_source']).strip().lower()
        if 'upper_cluster_semantic_method' in kwargs:
            self.upper_cluster_semantic_method = str(kwargs['upper_cluster_semantic_method']).strip().lower()
        if 'upper_cluster_semantic_threshold' in kwargs:
            self.upper_cluster_semantic_threshold = float(
                np.clip(kwargs['upper_cluster_semantic_threshold'], 0.0, 1.0)
            )
        if 'upper_cluster_semantic_resolution' in kwargs:
            self.upper_cluster_semantic_resolution = max(float(kwargs['upper_cluster_semantic_resolution']), 1e-6)
        if 'upper_cluster_semantic_self_loop_weight' in kwargs:
            self.upper_cluster_semantic_self_loop_weight = max(float(kwargs['upper_cluster_semantic_self_loop_weight']), 0.0)
        if 'upper_cluster_semantic_sparsify_mode' in kwargs:
            self.upper_cluster_semantic_sparsify_mode = str(kwargs['upper_cluster_semantic_sparsify_mode']).strip().lower()
        if 'upper_cluster_semantic_topk' in kwargs:
            self.upper_cluster_semantic_topk = max(int(kwargs['upper_cluster_semantic_topk']), 0)
        if 'rg_semantic_sharpen_power' in kwargs:
            self.rg_semantic_sharpen_power = max(float(kwargs['rg_semantic_sharpen_power']), 1.0)
        if 'upper_cluster_refresh_interval' in kwargs:
            self.upper_cluster_refresh_interval = max(1, int(kwargs['upper_cluster_refresh_interval']))
        if 'hierarchy_refresh_interval' in kwargs:
            self.hierarchy_refresh_interval = max(0, int(kwargs['hierarchy_refresh_interval']))
        if 'macro_split_hysteresis_ratio' in kwargs:
            self.macro_split_hysteresis_ratio = max(float(kwargs['macro_split_hysteresis_ratio']), 1.0)
        if 'inter_level_rigid_shift_enabled' in kwargs:
            self.inter_level_rigid_shift_enabled = bool(kwargs['inter_level_rigid_shift_enabled'])
        if 'inter_level_rigid_shift_strength' in kwargs:
            self.inter_level_rigid_shift_strength = float(kwargs['inter_level_rigid_shift_strength'])
        if 'inter_level_rigid_shift_max_step_ratio' in kwargs:
            self.inter_level_rigid_shift_max_step_ratio = float(kwargs['inter_level_rigid_shift_max_step_ratio'])
        if 'inter_level_centroid_anchor_enabled' in kwargs:
            self.inter_level_centroid_anchor_enabled = bool(kwargs['inter_level_centroid_anchor_enabled'])
        if 'inter_level_centroid_anchor_strength' in kwargs:
            self.inter_level_centroid_anchor_strength = float(kwargs['inter_level_centroid_anchor_strength'])
        if 'inter_level_centroid_anchor_max_step_ratio' in kwargs:
            self.inter_level_centroid_anchor_max_step_ratio = float(kwargs['inter_level_centroid_anchor_max_step_ratio'])
        if 'source_unit_motion_coherence_enabled' in kwargs:
            self.source_unit_motion_coherence_enabled = bool(kwargs['source_unit_motion_coherence_enabled'])
        if 'source_unit_motion_coherence_strength' in kwargs:
            self.source_unit_motion_coherence_strength = float(kwargs['source_unit_motion_coherence_strength'])
        if 'source_unit_motion_coherence_max_step_ratio' in kwargs:
            self.source_unit_motion_coherence_max_step_ratio = float(kwargs['source_unit_motion_coherence_max_step_ratio'])
        if 'inter_level_support_reanchor_enabled' in kwargs:
            self.inter_level_support_reanchor_enabled = bool(kwargs['inter_level_support_reanchor_enabled'])
        if 'inter_level_support_reanchor_strength' in kwargs:
            self.inter_level_support_reanchor_strength = float(kwargs['inter_level_support_reanchor_strength'])
        if 'source_unit_barrier_enabled' in kwargs:
            self.source_unit_barrier_enabled = bool(kwargs['source_unit_barrier_enabled'])
        if 'source_unit_barrier_strength_ratio' in kwargs:
            self.source_unit_barrier_strength_ratio = float(kwargs['source_unit_barrier_strength_ratio'])
        if 'support_unit_repulsion_enabled' in kwargs:
            self.support_unit_repulsion_enabled = bool(kwargs['support_unit_repulsion_enabled'])
        if 'support_unit_repulsion_strength_ratio' in kwargs:
            self.support_unit_repulsion_strength_ratio = float(kwargs['support_unit_repulsion_strength_ratio'])
        if 'persistent_relation_force_enabled' in kwargs:
            self.persistent_relation_force_enabled = bool(kwargs['persistent_relation_force_enabled'])
        if 'persistent_relation_decay' in kwargs:
            self.persistent_relation_decay = float(kwargs['persistent_relation_decay'])
        if 'persistent_relation_threshold' in kwargs:
            self.persistent_relation_threshold = float(kwargs['persistent_relation_threshold'])
        if 'persistent_relation_signal_gain' in kwargs:
            self.persistent_relation_signal_gain = float(kwargs['persistent_relation_signal_gain'])
        if 'persistent_relation_surprise_low' in kwargs:
            self.persistent_relation_surprise_low = float(kwargs['persistent_relation_surprise_low'])
        if 'persistent_relation_surprise_mid' in kwargs:
            self.persistent_relation_surprise_mid = float(kwargs['persistent_relation_surprise_mid'])
        if 'persistent_relation_surprise_high' in kwargs:
            self.persistent_relation_surprise_high = float(kwargs['persistent_relation_surprise_high'])
        if 'persistent_relation_same_force_strength_ratio' in kwargs:
            self.persistent_relation_same_force_strength_ratio = float(kwargs['persistent_relation_same_force_strength_ratio'])
        if 'persistent_relation_diff_force_strength_ratio' in kwargs:
            self.persistent_relation_diff_force_strength_ratio = float(kwargs['persistent_relation_diff_force_strength_ratio'])
        if 'unit_retention_enabled' in kwargs:
            self.unit_retention_enabled = bool(kwargs['unit_retention_enabled'])
        if 'unit_merge_hysteresis_steps' in kwargs:
            self.unit_merge_hysteresis_steps = max(1, int(kwargs['unit_merge_hysteresis_steps']))
        if 'unit_merge_distance_ratio' in kwargs:
            self.unit_merge_distance_ratio = float(kwargs['unit_merge_distance_ratio'])
        if 'level_count' in kwargs:
            requested_level_count = max(2, int(kwargs['level_count']))
            if requested_level_count > len(self.levels):
                self._ensure_level_count(requested_level_count)
            self.level_count = len(self.levels)
            self._sync_public_level_aliases()

        level_params: list[dict[str, Any]] = [dict() for _ in self.levels]
        scaled_level_keys = {
            'centering_strength',
            'max_step_norm',
            'dbscan_eps',
            'graph_r0_rel',
            'graph_r0_fixed',
        }
        local_keys = {
            'warmup_steps',
            'level_count',
            'renormalize_interval',
            'rg_time_scale_mode',
            'rg_time_scale_growth',
            'rg_inter_level_schedule_mode',
            'level1_enabled',
            'stats_memory_scale',
            'organize_level',
            'rg_input_level',
            'rg_upper_edge_mode',
            'feedback_mode',
            'rg_fill_noise_labels',
            'hierarchy_readout_mode',
            'stable_token_extraction_mode',
            'hierarchy_readout_min_gap_ratio',
            'dbscan_eps_scale_mode',
            'dbscan_eps_margin',
            'dbscan_readout_slack',
            'rg_knn_k',
            'rg_jump_ratio',
            'upper_rg_jump_ratio',
            'rg_jump_ratio_final',
            'rg_jump_ratio_anneal_start',
            'rg_jump_ratio_anneal_end',
            'macro_boundary_gap_gate_enabled',
            'macro_boundary_gap_gate_floor',
            'macro_boundary_gap_gate_full_ratio',
            'macro_boundary_min_stable_refreshes',
            'macro_boundary_token_purity_enabled',
            'macro_boundary_token_purity_threshold',
            'macro_boundary_token_purity_floor',
            'rg_semantic_graph_enabled',
            'rg_semantic_graph_strength',
            'rg_semantic_graph_similarity_floor',
            'rg_semantic_graph_penalty_mode',
            'rg_semantic_graph_relative_threshold',
            'rg_semantic_graph_anchor_floor',
            'rg_semantic_graph_edge_mode',
            'rg_semantic_graph_topk',
            'rg_fallback_scale',
            'upper_cluster_proposal_mode',
            'macro_component_proposal_mode',
            'upper_cluster_semantic_source',
            'upper_cluster_semantic_method',
            'upper_cluster_semantic_threshold',
            'upper_cluster_semantic_resolution',
            'upper_cluster_semantic_self_loop_weight',
            'upper_cluster_semantic_sparsify_mode',
            'upper_cluster_semantic_topk',
            'rg_semantic_sharpen_power',
            'upper_cluster_refresh_interval',
            'hierarchy_refresh_interval',
            'macro_split_hysteresis_ratio',
            'inter_level_rigid_shift_enabled',
            'inter_level_rigid_shift_strength',
            'inter_level_rigid_shift_max_step_ratio',
            'inter_level_centroid_anchor_enabled',
            'inter_level_centroid_anchor_strength',
            'inter_level_centroid_anchor_max_step_ratio',
            'source_unit_motion_coherence_enabled',
            'source_unit_motion_coherence_strength',
            'source_unit_motion_coherence_max_step_ratio',
            'inter_level_support_reanchor_enabled',
            'inter_level_support_reanchor_strength',
            'source_unit_barrier_enabled',
            'source_unit_barrier_strength_ratio',
            'support_unit_repulsion_enabled',
            'support_unit_repulsion_strength_ratio',
            'persistent_relation_force_enabled',
            'persistent_relation_decay',
            'persistent_relation_threshold',
            'persistent_relation_signal_gain',
            'persistent_relation_surprise_low',
            'persistent_relation_surprise_mid',
            'persistent_relation_surprise_high',
            'persistent_relation_same_force_strength_ratio',
            'persistent_relation_diff_force_strength_ratio',
            'unit_retention_enabled',
            'unit_merge_hysteresis_steps',
            'unit_merge_distance_ratio',
        }

        base_stats_memory_steps: int | None = None
        if 'stats_memory_steps' in kwargs:
            base_stats_memory_steps = max(1, int(kwargs['stats_memory_steps']))

        for k, v in kwargs.items():
            if k in local_keys:
                continue
            if hasattr(self, k):
                try:
                    setattr(self, k, v)
                except Exception:
                    pass

            if k in scaled_level_keys:
                for level_index, params in enumerate(level_params):
                    params[k] = float(v) * (self.rg_scale_factor ** level_index)
                continue

            if k == 'T_push':
                base_value = max(1, int(v))
                for level_index, params in enumerate(level_params):
                    params[k] = max(1, int(round(base_value * self._time_scale_multiplier(level_index))))
                continue
            if k == 'clustering_period':
                base_value = max(0, int(v))
                for level_index, params in enumerate(level_params):
                    params[k] = (
                        0
                        if base_value <= 0
                        else max(1, int(round(base_value * self._time_scale_multiplier(level_index))))
                    )
                continue

            if k == 'stats_memory_steps':
                continue

            if k in ('stats_mode', 'rg_upper_stats_mode'):
                # stats_mode is level-specific in RG: L0 uses `stats_mode`, the
                # upper levels use `rg_upper_stats_mode`. The generic uniform path
                # below would clobber that split (forcing every level to the L0
                # mode), leaving L1+ deaf to the sharp transition operator that
                # _build_level_kwargs had correctly assigned at construction.
                base_mode = str(kwargs.get('stats_mode', getattr(self, 'stats_mode', 'coactivation')))
                upper_mode = str(kwargs.get('rg_upper_stats_mode', getattr(self, 'rg_upper_stats_mode', base_mode)))
                for level_index, params in enumerate(level_params):
                    params['stats_mode'] = base_mode if level_index == 0 else upper_mode
                continue

            for params in level_params:
                params[k] = v

        if base_stats_memory_steps is not None:
            for level_index, params in enumerate(level_params):
                if 'stats_memory_steps' in params:
                    continue
                params['stats_memory_steps'] = max(
                    1,
                    int(round(base_stats_memory_steps * (self.stats_memory_scale ** level_index))),
                )

        for level_model, params in zip(self.levels, level_params):
            if params:
                level_model.update_params(**params)
        self._reset_upper_cluster_cache()

        if self.levels:
            level0 = self.levels[0]
            self.base_wall = float(getattr(level0, 'base_wall', self.base_wall))
            self.density_ref = float(getattr(level0, 'density_ref', self.density_ref))
            self.space_size = float(getattr(level0, 'space_size', self.space_size))
            self.reference_local_scale = float(getattr(level0, 'reference_local_scale', self.reference_local_scale))
            self.effective_local_scale = float(getattr(level0, 'effective_local_scale', self.effective_local_scale))
            self.significant_distance = float(getattr(level0, 'significant_distance', self.significant_distance))

    def get_effective_params(self) -> dict[str, Any]:
        return {
            'input_size': self.input_size,
            'dimensions': self.dimensions,
            'level_count': self.level_count,
            'rg_scale_factor': self.rg_scale_factor,
            'rg_time_scale_mode': self.rg_time_scale_mode,
            'rg_time_scale_growth': self.rg_time_scale_growth,
            'rg_inter_level_schedule_mode': self.rg_inter_level_schedule_mode,
            'stats_memory_scale': self.stats_memory_scale,
            'warmup_steps': self.warmup_steps,
            'renormalize_interval': self.renormalize_interval,
            'level1_enabled': self.level1_enabled,
            'organize_level': self.organize_level,
            'rg_input_level': self.rg_input_level,
            'feedback_mode': self.feedback_mode,
            'rg_fill_noise_labels': self.rg_fill_noise_labels,
            'stable_token_extraction_mode': self.stable_token_extraction_mode,
            'hierarchy_readout_mode': self.hierarchy_readout_mode,
            'hierarchy_readout_min_gap_ratio': self.hierarchy_readout_min_gap_ratio,
            'dbscan_eps_scale_mode': self.dbscan_eps_scale_mode,
            'dbscan_eps_margin': self.dbscan_eps_margin,
            'dbscan_readout_slack': self.dbscan_readout_slack,
            'dbscan_readout_eps': self.dbscan_readout_eps,
            'dbscan_readout_min_samples': self.dbscan_readout_min_samples,
            'rg_knn_k': self.rg_knn_k,
            'rg_jump_ratio': self.rg_jump_ratio,
            'upper_rg_jump_ratio': self.upper_rg_jump_ratio,
            'rg_jump_ratio_final': self.rg_jump_ratio_final,
            'rg_jump_ratio_anneal_start': self.rg_jump_ratio_anneal_start,
            'rg_jump_ratio_anneal_end': self.rg_jump_ratio_anneal_end,
            'macro_boundary_gap_gate_enabled': self.macro_boundary_gap_gate_enabled,
            'macro_boundary_gap_gate_floor': self.macro_boundary_gap_gate_floor,
            'macro_boundary_gap_gate_full_ratio': self.macro_boundary_gap_gate_full_ratio,
            'macro_boundary_min_stable_refreshes': self.macro_boundary_min_stable_refreshes,
            'macro_boundary_token_purity_enabled': self.macro_boundary_token_purity_enabled,
            'macro_boundary_token_purity_threshold': self.macro_boundary_token_purity_threshold,
            'macro_boundary_token_purity_floor': self.macro_boundary_token_purity_floor,
            'rg_semantic_graph_enabled': self.rg_semantic_graph_enabled,
            'rg_semantic_graph_strength': self.rg_semantic_graph_strength,
            'rg_semantic_graph_similarity_floor': self.rg_semantic_graph_similarity_floor,
            'rg_semantic_graph_penalty_mode': self.rg_semantic_graph_penalty_mode,
            'rg_semantic_graph_relative_threshold': self.rg_semantic_graph_relative_threshold,
            'rg_semantic_graph_anchor_floor': self.rg_semantic_graph_anchor_floor,
            'rg_semantic_graph_edge_mode': self.rg_semantic_graph_edge_mode,
            'rg_semantic_graph_topk': self.rg_semantic_graph_topk,
            'rg_fallback_scale': self.rg_fallback_scale,
            'macro_component_proposal_mode': self.macro_component_proposal_mode,
            'upper_cluster_proposal_mode': self.upper_cluster_proposal_mode,
            'upper_cluster_semantic_source': self.upper_cluster_semantic_source,
            'upper_cluster_semantic_method': self.upper_cluster_semantic_method,
            'upper_cluster_semantic_threshold': self.upper_cluster_semantic_threshold,
            'upper_cluster_semantic_resolution': self.upper_cluster_semantic_resolution,
            'upper_cluster_semantic_self_loop_weight': self.upper_cluster_semantic_self_loop_weight,
            'upper_cluster_semantic_sparsify_mode': self.upper_cluster_semantic_sparsify_mode,
            'upper_cluster_semantic_topk': self.upper_cluster_semantic_topk,
            'rg_semantic_sharpen_power': self.rg_semantic_sharpen_power,
            'upper_cluster_refresh_interval': self.upper_cluster_refresh_interval,
            'hierarchy_refresh_interval': self.hierarchy_refresh_interval,
            'macro_split_hysteresis_ratio': self.macro_split_hysteresis_ratio,
            'base_wall': self.base_wall,
            'density_ref': self.density_ref,
            'space_size': self.space_size,
            'reference_local_scale': self.reference_local_scale,
            'effective_local_scale': self.effective_local_scale,
            'significant_distance': self.significant_distance,
            'inter_level_rigid_shift_enabled': self.inter_level_rigid_shift_enabled,
            'inter_level_rigid_shift_strength': self.inter_level_rigid_shift_strength,
            'inter_level_rigid_shift_max_step_ratio': self.inter_level_rigid_shift_max_step_ratio,
            'inter_level_centroid_anchor_enabled': self.inter_level_centroid_anchor_enabled,
            'inter_level_centroid_anchor_strength': self.inter_level_centroid_anchor_strength,
            'inter_level_centroid_anchor_max_step_ratio': self.inter_level_centroid_anchor_max_step_ratio,
            'source_unit_motion_coherence_enabled': self.source_unit_motion_coherence_enabled,
            'source_unit_motion_coherence_strength': self.source_unit_motion_coherence_strength,
            'source_unit_motion_coherence_max_step_ratio': self.source_unit_motion_coherence_max_step_ratio,
            'inter_level_support_reanchor_enabled': self.inter_level_support_reanchor_enabled,
            'inter_level_support_reanchor_strength': self.inter_level_support_reanchor_strength,
            'source_unit_barrier_enabled': self.source_unit_barrier_enabled,
            'source_unit_barrier_strength_ratio': self.source_unit_barrier_strength_ratio,
            'support_unit_repulsion_enabled': self.support_unit_repulsion_enabled,
            'support_unit_repulsion_strength_ratio': self.support_unit_repulsion_strength_ratio,
            'k_attr': float(getattr(self.levels[0], 'k_attr', 0.0)) if self.levels else 0.0,
            'k_repulse': float(getattr(self.levels[0], 'k_repulse', 0.0)) if self.levels else 0.0,
            'macro_cross_attr_scale': float(getattr(self.levels[0], 'macro_cross_attr_scale', 1.0)) if self.levels else 1.0,
            'cluster_distance_strength': float(getattr(self.levels[0], 'cluster_distance_strength', 0.0)) if self.levels else 0.0,
            'macro_centroid_distance_ratio': float(getattr(self.levels[0], 'macro_centroid_distance_ratio', 3.16)) if self.levels else 3.16,
            'k_macro_repulse': float(getattr(self.levels[0], 'k_macro_repulse', 0.0)) if self.levels else 0.0,
            'macro_centroid_max_step_ratio': float(getattr(self.levels[0], 'macro_centroid_max_step_ratio', 0.0)) if self.levels else 0.0,
            'entity_boundary_repulsion_enabled': bool(getattr(self.levels[0], 'entity_boundary_repulsion_enabled', False)) if self.levels else False,
            'entity_boundary_distance_ratio': float(getattr(self.levels[0], 'entity_boundary_distance_ratio', 4.8)) if self.levels else 4.8,
            'k_entity_boundary_repulse': float(getattr(self.levels[0], 'k_entity_boundary_repulse', 0.0)) if self.levels else 0.0,
            'entity_boundary_min_gap_ratio': float(getattr(self.levels[0], 'entity_boundary_min_gap_ratio', 0.25)) if self.levels else 0.25,
            'entity_boundary_max_step_ratio': float(getattr(self.levels[0], 'entity_boundary_max_step_ratio', 0.0)) if self.levels else 0.0,
            'entity_sibling_attraction_enabled': bool(getattr(self.levels[0], 'entity_sibling_attraction_enabled', False)) if self.levels else False,
            'entity_sibling_target_ratio': float(getattr(self.levels[0], 'entity_sibling_target_ratio', 4.8)) if self.levels else 4.8,
            'entity_sibling_capture_ratio': float(getattr(self.levels[0], 'entity_sibling_capture_ratio', 3.16)) if self.levels else 3.16,
            'k_entity_sibling_attr': float(getattr(self.levels[0], 'k_entity_sibling_attr', 0.0)) if self.levels else 0.0,
            'entity_sibling_max_step_ratio': float(getattr(self.levels[0], 'entity_sibling_max_step_ratio', 0.0)) if self.levels else 0.0,
            'entity_sibling_boundary_attenuation_enabled': bool(getattr(self.levels[0], 'entity_sibling_boundary_attenuation_enabled', True)) if self.levels else True,
            'entity_sibling_boundary_release_ratio': float(getattr(self.levels[0], 'entity_sibling_boundary_release_ratio', 1.25)) if self.levels else 1.25,
            'entity_sibling_boundary_power': float(getattr(self.levels[0], 'entity_sibling_boundary_power', 2.0)) if self.levels else 2.0,
            'entity_sibling_deadband_ratio': float(getattr(self.levels[0], 'entity_sibling_deadband_ratio', 0.0)) if self.levels else 0.0,
            'primitive_contrastive_force_enabled': bool(getattr(self.levels[0], 'primitive_contrastive_force_enabled', False)) if self.levels else False,
            'primitive_active_attraction_strength': float(getattr(self.levels[0], 'primitive_active_attraction_strength', 0.0)) if self.levels else 0.0,
            'primitive_inactive_repulsion_strength': float(getattr(self.levels[0], 'primitive_inactive_repulsion_strength', 0.0)) if self.levels else 0.0,
            'line_obstacle_repulsion_enabled': bool(getattr(self.levels[0], 'line_obstacle_repulsion_enabled', False)) if self.levels else False,
            'line_obstacle_radius_rel': float(getattr(self.levels[0], 'line_obstacle_radius_rel', 1.0)) if self.levels else 1.0,
            'line_obstacle_strength': float(getattr(self.levels[0], 'line_obstacle_strength', 0.0)) if self.levels else 0.0,
            'stats_mode': str(getattr(self.levels[0], 'stats_mode', '')) if self.levels else '',
            'stats_memory_steps': int(getattr(self.levels[0], 'stats_memory_steps', 0)) if self.levels else 0,
            'embed_cluster_method': str(getattr(self.levels[0], 'embed_cluster_method', '')) if self.levels else '',
            'dbscan_eps': float(getattr(self.levels[0], 'dbscan_eps', 0.0)) if self.levels else 0.0,
            'rg_upper_input_active_gain': self.rg_upper_input_active_gain,
            'rg_upper_input_mass_gain': self.rg_upper_input_mass_gain,
            'rg_upper_stats_mode': self.rg_upper_stats_mode,
            'rg_upper_history_window': self.rg_upper_history_window,
            'persistent_relation_force_enabled': self.persistent_relation_force_enabled,
            'persistent_relation_decay': self.persistent_relation_decay,
            'persistent_relation_threshold': self.persistent_relation_threshold,
            'persistent_relation_signal_gain': self.persistent_relation_signal_gain,
            'persistent_relation_surprise_low': self.persistent_relation_surprise_low,
            'persistent_relation_surprise_mid': self.persistent_relation_surprise_mid,
            'persistent_relation_surprise_high': self.persistent_relation_surprise_high,
            'persistent_relation_same_force_strength_ratio': self.persistent_relation_same_force_strength_ratio,
            'persistent_relation_diff_force_strength_ratio': self.persistent_relation_diff_force_strength_ratio,
            'level_params': [level.get_effective_params() for level in self.levels],
        }

    def save_weights(self, filepath: Union[str, Path]) -> None:
        data = {
            'levels_state': [level.__dict__ for level in self.levels],
            'level0_state': self.level0.__dict__,
            'level1_state': self.level1.__dict__,
            'rg_step_count': self.step_count,
            'rg_level0_phase': self._level0_phase,
            'rg_level1_phase': self._level1_phase,
            'rg_level_phases': list(self._level_phases),
            'rg_warmup_steps': self.warmup_steps,
            'rg_renormalize_interval': self.renormalize_interval,
            'rg_inter_level_schedule_mode': self.rg_inter_level_schedule_mode,
            'rg_inter_level_next_due_steps': list(self._next_inter_level_due_step_by_level),
            'rg_stable_tokens': self.node_to_cluster_map,
            'rg_meta_tokens': self.token_to_meta_map,
            'rg_organize_level': self.organize_level,
            'rg_level1_enabled': self.level1_enabled,
            'rg_input_level': self.rg_input_level,
            'feedback_mode': self.feedback_mode,
            'rg_fill_noise_labels': self.rg_fill_noise_labels,
            'stable_token_extraction_mode': self.stable_token_extraction_mode,
            'hierarchy_readout_mode': self.hierarchy_readout_mode,
            'hierarchy_readout_min_gap_ratio': self.hierarchy_readout_min_gap_ratio,
            'dbscan_eps_scale_mode': self.dbscan_eps_scale_mode,
            'dbscan_eps_margin': self.dbscan_eps_margin,
            'dbscan_readout_slack': self.dbscan_readout_slack,
            'dbscan_readout_eps': self.dbscan_readout_eps,
            'dbscan_readout_min_samples': self.dbscan_readout_min_samples,
            'rg_knn_k': self.rg_knn_k,
            'rg_jump_ratio': self.rg_jump_ratio,
            'upper_rg_jump_ratio': self.upper_rg_jump_ratio,
            'rg_jump_ratio_final': self.rg_jump_ratio_final,
            'rg_jump_ratio_anneal_start': self.rg_jump_ratio_anneal_start,
            'rg_jump_ratio_anneal_end': self.rg_jump_ratio_anneal_end,
            'macro_boundary_gap_gate_enabled': self.macro_boundary_gap_gate_enabled,
            'macro_boundary_gap_gate_floor': self.macro_boundary_gap_gate_floor,
            'macro_boundary_gap_gate_full_ratio': self.macro_boundary_gap_gate_full_ratio,
            'macro_boundary_min_stable_refreshes': self.macro_boundary_min_stable_refreshes,
            'macro_boundary_token_purity_enabled': self.macro_boundary_token_purity_enabled,
            'macro_boundary_token_purity_threshold': self.macro_boundary_token_purity_threshold,
            'macro_boundary_token_purity_floor': self.macro_boundary_token_purity_floor,
            'rg_semantic_graph_enabled': self.rg_semantic_graph_enabled,
            'rg_semantic_graph_strength': self.rg_semantic_graph_strength,
            'rg_semantic_graph_similarity_floor': self.rg_semantic_graph_similarity_floor,
            'rg_semantic_graph_penalty_mode': self.rg_semantic_graph_penalty_mode,
            'rg_semantic_graph_relative_threshold': self.rg_semantic_graph_relative_threshold,
            'rg_semantic_graph_anchor_floor': self.rg_semantic_graph_anchor_floor,
            'rg_semantic_graph_edge_mode': self.rg_semantic_graph_edge_mode,
            'rg_semantic_graph_topk': self.rg_semantic_graph_topk,
            'rg_fallback_scale': self.rg_fallback_scale,
            'macro_component_proposal_mode': self.macro_component_proposal_mode,
            'upper_cluster_proposal_mode': self.upper_cluster_proposal_mode,
            'upper_cluster_semantic_source': self.upper_cluster_semantic_source,
            'upper_cluster_semantic_method': self.upper_cluster_semantic_method,
            'upper_cluster_semantic_threshold': self.upper_cluster_semantic_threshold,
            'upper_cluster_semantic_resolution': self.upper_cluster_semantic_resolution,
            'upper_cluster_semantic_self_loop_weight': self.upper_cluster_semantic_self_loop_weight,
            'upper_cluster_semantic_sparsify_mode': self.upper_cluster_semantic_sparsify_mode,
            'upper_cluster_semantic_topk': self.upper_cluster_semantic_topk,
            'rg_semantic_sharpen_power': self.rg_semantic_sharpen_power,
            'rg_time_scale_mode': self.rg_time_scale_mode,
            'rg_time_scale_growth': self.rg_time_scale_growth,
            'upper_cluster_refresh_interval': self.upper_cluster_refresh_interval,
            'hierarchy_refresh_interval': self.hierarchy_refresh_interval,
            'macro_split_hysteresis_ratio': self.macro_split_hysteresis_ratio,
            'inter_level_rigid_shift_enabled': self.inter_level_rigid_shift_enabled,
            'inter_level_rigid_shift_strength': self.inter_level_rigid_shift_strength,
            'inter_level_rigid_shift_max_step_ratio': self.inter_level_rigid_shift_max_step_ratio,
            'inter_level_centroid_anchor_enabled': self.inter_level_centroid_anchor_enabled,
            'inter_level_centroid_anchor_strength': self.inter_level_centroid_anchor_strength,
            'inter_level_centroid_anchor_max_step_ratio': self.inter_level_centroid_anchor_max_step_ratio,
            'source_unit_motion_coherence_enabled': self.source_unit_motion_coherence_enabled,
            'source_unit_motion_coherence_strength': self.source_unit_motion_coherence_strength,
            'source_unit_motion_coherence_max_step_ratio': self.source_unit_motion_coherence_max_step_ratio,
            'inter_level_support_reanchor_enabled': self.inter_level_support_reanchor_enabled,
            'inter_level_support_reanchor_strength': self.inter_level_support_reanchor_strength,
            'source_unit_barrier_enabled': self.source_unit_barrier_enabled,
            'source_unit_barrier_strength_ratio': self.source_unit_barrier_strength_ratio,
            'support_unit_repulsion_enabled': self.support_unit_repulsion_enabled,
            'support_unit_repulsion_strength_ratio': self.support_unit_repulsion_strength_ratio,
            'persistent_relation_force_enabled': self.persistent_relation_force_enabled,
            'persistent_relation_decay': self.persistent_relation_decay,
            'persistent_relation_threshold': self.persistent_relation_threshold,
            'persistent_relation_signal_gain': self.persistent_relation_signal_gain,
            'persistent_relation_surprise_low': self.persistent_relation_surprise_low,
            'persistent_relation_surprise_mid': self.persistent_relation_surprise_mid,
            'persistent_relation_surprise_high': self.persistent_relation_surprise_high,
            'persistent_relation_same_force_strength_ratio': self.persistent_relation_same_force_strength_ratio,
            'persistent_relation_diff_force_strength_ratio': self.persistent_relation_diff_force_strength_ratio,
            'unit_retention_enabled': self.unit_retention_enabled,
            'unit_merge_hysteresis_steps': self.unit_merge_hysteresis_steps,
            'unit_merge_distance_ratio': self.unit_merge_distance_ratio,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if 'levels_state' in data:
            for level_model, state in zip(self.levels, data['levels_state']):
                for k, v in state.items():
                    setattr(level_model, k, v)
        elif 'level0_state' in data:
            for k, v in data['level0_state'].items():
                setattr(self.level0, k, v)
        if 'levels_state' not in data and 'level1_state' in data:
            for k, v in data['level1_state'].items():
                setattr(self.level1, k, v)

        self.step_count = int(data.get('rg_step_count', 0))
        self._level0_phase = int(data.get('rg_level0_phase', self.step_count))
        self._level1_phase = int(data.get('rg_level1_phase', 0))
        raw_level_phases = data.get('rg_level_phases')
        if isinstance(raw_level_phases, (list, tuple)):
            restored_phases = [max(0, int(value)) for value in raw_level_phases]
            restored_phases = restored_phases[: len(self._level_phases)]
            restored_phases.extend([0] * (len(self._level_phases) - len(restored_phases)))
            self._level_phases = restored_phases
            if self._level_phases:
                self._level0_phase = self._level_phases[0]
            if len(self._level_phases) > 1:
                self._level1_phase = self._level_phases[1]
        else:
            if self._level_phases:
                self._level_phases[0] = self._level0_phase
            if len(self._level_phases) > 1:
                self._level_phases[1] = self._level1_phase
        self.warmup_steps = int(data.get('rg_warmup_steps', self.warmup_steps))
        self.renormalize_interval = max(
            1,
            int(data.get('rg_renormalize_interval', self.renormalize_interval)),
        )
        if 'rg_stable_tokens' in data:
            self.node_to_cluster_map = np.asarray(data['rg_stable_tokens'], dtype=int)
        if 'rg_meta_tokens' in data:
            self.token_to_meta_map = np.asarray(data['rg_meta_tokens'], dtype=int)
        self.organize_level = canonicalize_organize_level(data.get('rg_organize_level', self.organize_level))
        self.level1_enabled = bool(data.get('rg_level1_enabled', self.level1_enabled))
        self.rg_input_level = str(data.get('rg_input_level', self.rg_input_level)).lower()
        self.feedback_mode = str(data.get('feedback_mode', self.feedback_mode)).strip().lower()
        self.rg_fill_noise_labels = bool(data.get('rg_fill_noise_labels', self.rg_fill_noise_labels))
        self.stable_token_extraction_mode = str(
            data.get('stable_token_extraction_mode', self.stable_token_extraction_mode)
        ).strip().lower()
        self.hierarchy_readout_mode = str(
            data.get('hierarchy_readout_mode', self.hierarchy_readout_mode)
        ).strip().lower()
        self.hierarchy_readout_min_gap_ratio = max(
            float(data.get('hierarchy_readout_min_gap_ratio', self.hierarchy_readout_min_gap_ratio)),
            1.0,
        )
        self.dbscan_eps_scale_mode = str(
            data.get('dbscan_eps_scale_mode', self.dbscan_eps_scale_mode)
        ).strip().lower()
        self.dbscan_eps_margin = max(
            float(data.get('dbscan_eps_margin', self.dbscan_eps_margin)),
            1e-6,
        )
        self.dbscan_readout_slack = max(
            float(data.get('dbscan_readout_slack', self.dbscan_readout_slack)),
            1e-6,
        )
        raw_readout_eps = data.get('dbscan_readout_eps', self.dbscan_readout_eps)
        self.dbscan_readout_eps = None if raw_readout_eps is None else float(raw_readout_eps)
        raw_readout_min_samples = data.get(
            'dbscan_readout_min_samples',
            self.dbscan_readout_min_samples,
        )
        self.dbscan_readout_min_samples = (
            None if raw_readout_min_samples is None else max(1, int(raw_readout_min_samples))
        )
        self.rg_knn_k = max(1, int(data.get('rg_knn_k', self.rg_knn_k)))
        self.rg_jump_ratio = float(data.get('rg_jump_ratio', self.rg_jump_ratio))
        upper_jump = data.get('upper_rg_jump_ratio', self.upper_rg_jump_ratio)
        self.upper_rg_jump_ratio = None if upper_jump is None else float(upper_jump)
        jump_final = data.get('rg_jump_ratio_final', self.rg_jump_ratio_final)
        self.rg_jump_ratio_final = None if jump_final is None else float(jump_final)
        self.rg_jump_ratio_anneal_start = max(
            0,
            int(data.get('rg_jump_ratio_anneal_start', self.rg_jump_ratio_anneal_start)),
        )
        self.rg_jump_ratio_anneal_end = max(
            0,
            int(data.get('rg_jump_ratio_anneal_end', self.rg_jump_ratio_anneal_end)),
        )
        self.macro_boundary_gap_gate_enabled = bool(
            data.get('macro_boundary_gap_gate_enabled', self.macro_boundary_gap_gate_enabled)
        )
        self.macro_boundary_gap_gate_floor = float(
            np.clip(data.get('macro_boundary_gap_gate_floor', self.macro_boundary_gap_gate_floor), 0.0, 1.0)
        )
        self.macro_boundary_gap_gate_full_ratio = max(
            float(data.get('macro_boundary_gap_gate_full_ratio', self.macro_boundary_gap_gate_full_ratio)),
            self.rg_jump_ratio + 1e-6,
        )
        self.macro_boundary_min_stable_refreshes = max(
            0,
            int(data.get('macro_boundary_min_stable_refreshes', self.macro_boundary_min_stable_refreshes)),
        )
        self.macro_boundary_token_purity_enabled = bool(
            data.get('macro_boundary_token_purity_enabled', self.macro_boundary_token_purity_enabled)
        )
        self.macro_boundary_token_purity_threshold = float(
            np.clip(
                data.get('macro_boundary_token_purity_threshold', self.macro_boundary_token_purity_threshold),
                0.0,
                1.0,
            )
        )
        self.macro_boundary_token_purity_floor = float(
            np.clip(
                data.get('macro_boundary_token_purity_floor', self.macro_boundary_token_purity_floor),
                0.0,
                1.0,
            )
        )
        self.rg_semantic_graph_enabled = bool(
            data.get('rg_semantic_graph_enabled', self.rg_semantic_graph_enabled)
        )
        self.rg_semantic_graph_strength = max(
            float(data.get('rg_semantic_graph_strength', self.rg_semantic_graph_strength)),
            0.0,
        )
        self.rg_semantic_graph_similarity_floor = float(
            np.clip(
                data.get('rg_semantic_graph_similarity_floor', self.rg_semantic_graph_similarity_floor),
                0.0,
                1.0,
            )
        )
        self.rg_semantic_graph_penalty_mode = str(
            data.get('rg_semantic_graph_penalty_mode', self.rg_semantic_graph_penalty_mode)
        ).strip().lower()
        self.rg_semantic_graph_relative_threshold = float(
            np.clip(
                data.get('rg_semantic_graph_relative_threshold', self.rg_semantic_graph_relative_threshold),
                0.0,
                1.0,
            )
        )
        self.rg_semantic_graph_anchor_floor = float(
            np.clip(
                data.get('rg_semantic_graph_anchor_floor', self.rg_semantic_graph_anchor_floor),
                0.0,
                1.0,
            )
        )
        self.rg_semantic_graph_edge_mode = str(
            data.get('rg_semantic_graph_edge_mode', self.rg_semantic_graph_edge_mode)
        ).strip().lower()
        self.rg_semantic_graph_topk = max(
            int(data.get('rg_semantic_graph_topk', self.rg_semantic_graph_topk)),
            0,
        )
        self.rg_fallback_scale = float(data.get('rg_fallback_scale', self.rg_fallback_scale))
        self.macro_component_proposal_mode = str(
            data.get('macro_component_proposal_mode', self.macro_component_proposal_mode)
        ).strip().lower()
        self.upper_cluster_proposal_mode = str(
            data.get('upper_cluster_proposal_mode', self.upper_cluster_proposal_mode)
        ).strip().lower()
        self.upper_cluster_semantic_source = str(
            data.get('upper_cluster_semantic_source', self.upper_cluster_semantic_source)
        ).strip().lower()
        self.upper_cluster_semantic_method = str(
            data.get('upper_cluster_semantic_method', self.upper_cluster_semantic_method)
        ).strip().lower()
        self.upper_cluster_semantic_threshold = float(
            np.clip(
                data.get('upper_cluster_semantic_threshold', self.upper_cluster_semantic_threshold),
                0.0,
                1.0,
            )
        )
        self.upper_cluster_semantic_resolution = max(
            float(data.get('upper_cluster_semantic_resolution', self.upper_cluster_semantic_resolution)),
            1e-6,
        )
        self.upper_cluster_semantic_self_loop_weight = max(
            float(data.get('upper_cluster_semantic_self_loop_weight', self.upper_cluster_semantic_self_loop_weight)),
            0.0,
        )
        self.upper_cluster_semantic_sparsify_mode = str(
            data.get('upper_cluster_semantic_sparsify_mode', self.upper_cluster_semantic_sparsify_mode)
        ).strip().lower()
        self.upper_cluster_semantic_topk = max(
            int(data.get('upper_cluster_semantic_topk', self.upper_cluster_semantic_topk)),
            0,
        )
        self.rg_semantic_sharpen_power = max(
            float(data.get('rg_semantic_sharpen_power', self.rg_semantic_sharpen_power)),
            1.0,
        )
        self.rg_time_scale_mode = str(data.get('rg_time_scale_mode', self.rg_time_scale_mode)).strip().lower()
        self.rg_time_scale_growth = float(data.get('rg_time_scale_growth', self.rg_time_scale_growth))
        self.rg_inter_level_schedule_mode = self._normalize_inter_level_schedule_mode(
            data.get('rg_inter_level_schedule_mode', self.rg_inter_level_schedule_mode)
        )
        raw_next_due = data.get('rg_inter_level_next_due_steps')
        schedule_size = max(0, len(self.levels) - 1)
        if isinstance(raw_next_due, (list, tuple)):
            restored_next_due = [int(value) for value in raw_next_due[:schedule_size]]
            restored_next_due.extend([-1] * (schedule_size - len(restored_next_due)))
            self._next_inter_level_due_step_by_level = restored_next_due
        else:
            self._next_inter_level_due_step_by_level = [-1] * schedule_size
        self.upper_cluster_refresh_interval = max(
            1,
            int(data.get('upper_cluster_refresh_interval', self.upper_cluster_refresh_interval)),
        )
        self.hierarchy_refresh_interval = max(
            0,
            int(data.get('hierarchy_refresh_interval', self.hierarchy_refresh_interval)),
        )
        self.macro_split_hysteresis_ratio = max(
            float(data.get('macro_split_hysteresis_ratio', self.macro_split_hysteresis_ratio)),
            1.0,
        )
        self._reset_upper_cluster_cache()
        self.inter_level_rigid_shift_enabled = bool(
            data.get('inter_level_rigid_shift_enabled', self.inter_level_rigid_shift_enabled)
        )
        self.inter_level_rigid_shift_strength = float(
            data.get('inter_level_rigid_shift_strength', self.inter_level_rigid_shift_strength)
        )
        self.inter_level_rigid_shift_max_step_ratio = float(
            data.get('inter_level_rigid_shift_max_step_ratio', self.inter_level_rigid_shift_max_step_ratio)
        )
        self.inter_level_centroid_anchor_enabled = bool(
            data.get('inter_level_centroid_anchor_enabled', self.inter_level_centroid_anchor_enabled)
        )
        self.inter_level_centroid_anchor_strength = float(
            data.get('inter_level_centroid_anchor_strength', self.inter_level_centroid_anchor_strength)
        )
        self.inter_level_centroid_anchor_max_step_ratio = float(
            data.get('inter_level_centroid_anchor_max_step_ratio', self.inter_level_centroid_anchor_max_step_ratio)
        )
        self.source_unit_motion_coherence_enabled = bool(
            data.get('source_unit_motion_coherence_enabled', self.source_unit_motion_coherence_enabled)
        )
        self.source_unit_motion_coherence_strength = float(
            data.get('source_unit_motion_coherence_strength', self.source_unit_motion_coherence_strength)
        )
        self.source_unit_motion_coherence_max_step_ratio = float(
            data.get('source_unit_motion_coherence_max_step_ratio', self.source_unit_motion_coherence_max_step_ratio)
        )
        self.inter_level_support_reanchor_enabled = bool(
            data.get('inter_level_support_reanchor_enabled', self.inter_level_support_reanchor_enabled)
        )
        self.inter_level_support_reanchor_strength = float(
            data.get('inter_level_support_reanchor_strength', self.inter_level_support_reanchor_strength)
        )
        self.source_unit_barrier_enabled = bool(
            data.get('source_unit_barrier_enabled', self.source_unit_barrier_enabled)
        )
        self.source_unit_barrier_strength_ratio = float(
            data.get('source_unit_barrier_strength_ratio', self.source_unit_barrier_strength_ratio)
        )
        self.support_unit_repulsion_enabled = bool(
            data.get('support_unit_repulsion_enabled', self.support_unit_repulsion_enabled)
        )
        self.support_unit_repulsion_strength_ratio = float(
            data.get('support_unit_repulsion_strength_ratio', self.support_unit_repulsion_strength_ratio)
        )
        self.persistent_relation_force_enabled = bool(
            data.get('persistent_relation_force_enabled', self.persistent_relation_force_enabled)
        )
        self.persistent_relation_decay = float(
            data.get('persistent_relation_decay', self.persistent_relation_decay)
        )
        self.persistent_relation_threshold = float(
            data.get('persistent_relation_threshold', self.persistent_relation_threshold)
        )
        self.persistent_relation_signal_gain = float(
            data.get('persistent_relation_signal_gain', self.persistent_relation_signal_gain)
        )
        self.persistent_relation_surprise_low = float(
            data.get('persistent_relation_surprise_low', self.persistent_relation_surprise_low)
        )
        self.persistent_relation_surprise_mid = float(
            data.get('persistent_relation_surprise_mid', self.persistent_relation_surprise_mid)
        )
        self.persistent_relation_surprise_high = float(
            data.get('persistent_relation_surprise_high', self.persistent_relation_surprise_high)
        )
        self.persistent_relation_same_force_strength_ratio = float(
            data.get('persistent_relation_same_force_strength_ratio', self.persistent_relation_same_force_strength_ratio)
        )
        self.persistent_relation_diff_force_strength_ratio = float(
            data.get('persistent_relation_diff_force_strength_ratio', self.persistent_relation_diff_force_strength_ratio)
        )
        self.unit_retention_enabled = bool(
            data.get('unit_retention_enabled', self.unit_retention_enabled)
        )
        self.unit_merge_hysteresis_steps = max(
            1,
            int(data.get('unit_merge_hysteresis_steps', self.unit_merge_hysteresis_steps)),
        )
        self.unit_merge_distance_ratio = float(
            data.get('unit_merge_distance_ratio', self.unit_merge_distance_ratio)
        )

    def get_trajectory(self) -> NDArray[np.float64]:
        return self.levels[0].get_trajectory()

    def get_trajectory_steps(self) -> IntArray:
        return self.levels[0].get_trajectory_steps()

    def visualize_hierarchy(self) -> None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        pos0 = self.level0.syncmap
        raw_labels = self.organize(level=DEFAULT_HIERARCHY_ORGANIZE_LEVEL)
        source_tokens = self.organize(level='macro_tokens')

        ax.scatter(
            pos0[:, 0], pos0[:, 1], pos0[:, 2],
            c=raw_labels, cmap='tab20', s=20, alpha=0.6, label='Level 0 (Tokens)'
        )

        if self.step_count > self.warmup_steps:
            pos1 = self.level1.syncmap
            active_tokens = np.array(sorted({int(token) for token in source_tokens if token >= 0}), dtype=int)
            if active_tokens.size > 0:
                valid_pos1 = pos1[active_tokens]
                ax.scatter(
                    valid_pos1[:, 0], valid_pos1[:, 1], valid_pos1[:, 2],
                    c='red', marker='^', s=200, label='Level 1 (Concepts)'
                )
                for node_idx, token in enumerate(source_tokens.tolist()):
                    if token < 0:
                        continue
                    p0 = pos0[node_idx]
                    p1 = pos1[token]
                    ax.plot(
                        [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color='gray', alpha=0.3, linewidth=0.5,
                    )

        ax.set_title(f"SyncMap RG Prototype (Step {self.step_count})")
        ax.legend()
        plt.show()


def create_syncmap_rg(input_size: int, dimensions: int, **kwargs: Any) -> SyncMapX_RG:
    """Factory function for unified_experiments."""
    return SyncMapX_RG(input_size, dimensions, **kwargs)
