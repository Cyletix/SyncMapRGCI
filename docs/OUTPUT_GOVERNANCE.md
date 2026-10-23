# Output locations

Use tools/result_paths.py for output names.

| Content | Location |
|---|---|
| Batch comparison archive | results/experiment/<timestamp>/ |
| Single-run state/trace JSON | results/snapshots/ |
| Static plots | results/picture/ |
| Animations | results/animations/ |
| Representative benchmark (tracked) | results/benchmark/ |

Output directories are created when an export is requested and are ignored by git, except results/benchmark/, which holds the curated benchmark produced by tools/benchmark.py. Regenerate it with that script instead of editing its files by hand.

Each result keeps its metric/evaluation context and effective parameters, so a run can be interpreted without the current defaults.
