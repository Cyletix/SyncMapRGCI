# Output locations

Use tools/result_paths.py for output names.

| Content | Location |
|---|---|
| Batch comparison archive | results/experiment/<timestamp>/ |
| Single-run state/trace JSON | results/snapshots/ |
| Static plots | results/picture/ |
| Animations | results/animations/ |
| CSV summaries | results/csv/ |
| Representative benchmark (tracked) | results/benchmark/ |
| Supplementary experiments (tracked) | results/supp/ |
| Presentation and its diagrams (tracked) | docs/presentation/ |

main.py creates the results/ run folders (experiment, picture, animations, csv, html, md, snapshots) when it starts; html/ and md/ are currently left empty. Everything under results/ is ignored by git except results/benchmark/, built by tools/benchmark.py, and results/supp/, built by tools/supp_run.py and tools/supp_report.py. Regenerate those with their scripts instead of editing their files by hand.

Files too large for the repository (full result tables with learning curves, the single-file presentation and its PDF) are attached to the GitHub releases.

Each result keeps its metric/evaluation context and effective parameters, so a run can be interpreted without the current defaults.
