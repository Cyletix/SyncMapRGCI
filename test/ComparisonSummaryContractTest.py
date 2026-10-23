from __future__ import annotations

import unittest

import pandas as pd

from unified_experiments import comparison_summary_evaluation_aggregations


class ComparisonSummaryContractTest(unittest.TestCase):
    def test_preserves_stable_evaluation_contract(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "hier.dot",
                    "profile_name": "rg",
                    "sequence_length": 100,
                    "evaluation_mode": "hierarchy_level_mean",
                    "eval_readout_method": "mst_gaps",
                    "evaluation_output_key": "cluster_level1",
                },
                {
                    "dataset": "hier.dot",
                    "profile_name": "rg",
                    "sequence_length": 100,
                    "evaluation_mode": "hierarchy_level_mean",
                    "eval_readout_method": "mst_gaps",
                    "evaluation_output_key": "cluster_level1",
                },
            ]
        )
        aggregations = comparison_summary_evaluation_aggregations(
            frame,
            ["dataset", "profile_name", "sequence_length"],
        )
        self.assertEqual(
            aggregations,
            {
                "evaluation_mode": ["first"],
                "eval_readout_method": ["first"],
                "evaluation_output_key": ["first"],
            },
        )

    def test_rejects_mixed_evaluation_modes_within_case(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "hier.dot",
                    "profile_name": "rg",
                    "sequence_length": 100,
                    "evaluation_mode": "hierarchy_joint_min",
                },
                {
                    "dataset": "hier.dot",
                    "profile_name": "rg",
                    "sequence_length": 100,
                    "evaluation_mode": "hierarchy_level_mean",
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "mixed evaluation_mode"):
            comparison_summary_evaluation_aggregations(
                frame,
                ["dataset", "profile_name", "sequence_length"],
            )


if __name__ == "__main__":
    unittest.main()
