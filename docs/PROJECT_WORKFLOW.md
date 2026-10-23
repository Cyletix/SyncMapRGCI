# Project workflow

Run from the repository root in a Python 3.11 environment with requirements.txt installed. Use [CLI reference](CLI_REFERENCE.md) for commands and [project manifest](../project_manifest.yaml) for the supported runtime/profile inventory.

Keep algorithm implementations in syncmap/, environment logic in test/, shared helpers in tools/, and explanations in docs/. New experiment outputs belong under results/ using the existing output helpers. See [output governance](OUTPUT_GOVERNANCE.md).

Before interpreting a comparison, check dataset family, input-step/compute budget, effective parameters, evaluation mode and sampling window. Use the effective parameters stored in each result row rather than assuming current defaults.

Focused checks:

```bash
python -m unittest test.HierarchyMetricAggregationTest test.ComparisonSummaryContractTest test.CsvSummaryTest
python main.py --help
```

These checks cover score aggregation, report metadata and CLI accessibility. A short NSGD/RG smoke run checks integration; python tools/benchmark.py run reproduces the benchmark results.
