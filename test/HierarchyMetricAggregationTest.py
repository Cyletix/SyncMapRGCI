from __future__ import annotations

import unittest
from unittest.mock import patch

from unified_experiments import (
    apply_hierarchy_total_scores,
    build_comparison_run_signature,
)


class HierarchyMetricAggregationTest(unittest.TestCase):
    def test_flat_score_is_unchanged(self) -> None:
        scores = {"ari_score": 0.4, "cluster_level1_ari_score": 0.8}
        self.assertEqual(
            apply_hierarchy_total_scores(scores, is_hierarchical=False),
            scores,
        )

    def test_hierarchy_uses_equal_weight_level_mean(self) -> None:
        result = apply_hierarchy_total_scores(
            {
                "ari_score": 0.1,
                "cluster_level1_ari_score": 0.6,
                "cluster_level2_ari_score": 1.0,
            },
            is_hierarchical=True,
            depth=2,
        )
        self.assertAlmostEqual(result["ari_score"], 0.8)
        self.assertNotIn("hierarchy_joint_min_ari_score", result)

    def test_missing_required_level_contributes_zero(self) -> None:
        result = apply_hierarchy_total_scores(
            {"ari_score": 0.9, "cluster_level1_ari_score": 0.9},
            is_hierarchical=True,
            depth=3,
        )
        self.assertAlmostEqual(result["ari_score"], 0.3)

    def test_no_reported_level_scores_zeroes_hierarchy_total(self) -> None:
        result = apply_hierarchy_total_scores(
            {"ari_score": 0.9},
            is_hierarchical=True,
            depth=2,
        )
        self.assertEqual(result["ari_score"], 0.0)

    def test_only_current_one_based_level_columns_enter_total(self) -> None:
        result = apply_hierarchy_total_scores(
            {
                "ari_score": 0.1,
                "cluster_level0_ari_score": 1.0,
                "cluster_level2_ari_score": 0.4,
                "cluster_level1_ari_score": 0.8,
                "cluster_level3_ari_score": 1.0,
            },
            is_hierarchical=True,
            depth=2,
        )
        self.assertAlmostEqual(result["ari_score"], 0.6)

    def test_missing_interior_level_is_not_replaced_by_extra_level(self) -> None:
        result = apply_hierarchy_total_scores(
            {
                "ari_score": 1.0,
                "cluster_level2_ari_score": 0.4,
                "cluster_level3_ari_score": 1.0,
            },
            is_hierarchical=True,
            depth=2,
        )
        self.assertAlmostEqual(result["ari_score"], 0.2)

    def test_comparison_signature_versions_hierarchy_score_contract(self) -> None:
        kwargs = {
            "strategies": {"rg": {}},
            "sequence_lengths": [1000],
            "datasets": [],
            "comparison_trials": 1,
            "metric_name": "ari",
            "base_seed": 0,
        }
        with patch("unified_experiments.HIERARCHY_SCORE_CONTRACT", "mean_v1"):
            first = build_comparison_run_signature(**kwargs)
        with patch("unified_experiments.HIERARCHY_SCORE_CONTRACT", "mean_v2"):
            second = build_comparison_run_signature(**kwargs)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
