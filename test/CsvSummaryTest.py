from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from tools.csv_summary import summarize_csv


class CsvSummaryTest(unittest.TestCase):
    def test_evaluation_mode_and_readout_are_reported_without_unpack_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            row = {
                "dataset": "DCH1.dot",
                "sequence_length": "1000",
                "engine_strategy": "rg",
                "profile_name": "rg",
                "eval_readout_method": "mst_gaps",
                "evaluation_output_key": "cluster_level1",
                "evaluation_mode": "hierarchy_level_mean",
                "ari_score": "0.8",
                "nmi_score": "0.9",
                "trial": "1",
                "elapsed_time": "1.0",
                "cluster_count": "6",
            }
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = summarize_csv(str(path), metric="ari")
            rendered = output.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("evaluation=hierarchy_level_mean", rendered)
            self.assertIn("readout=mst_gaps:cluster_level1", rendered)

    def test_historical_baseline_code_is_displayed_as_syncmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            row = {
                "dataset": "flat.dot",
                "sequence_length": "1000",
                "engine_strategy": "strict",
                "profile_name": "strict",
                "ari_score": "0.8",
                "trial": "1",
            }
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = summarize_csv(str(path), metric="ari")
            rendered = output.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("engines=['syncmap']", rendered)
            self.assertIn("syncmap", rendered)
            self.assertNotIn("strict", rendered)


if __name__ == "__main__":
    unittest.main()
