# Project structure

The repository keeps its original code layout.

| Path | Role |
|---|---|
| main.py | Command-line entry point for every experiment task |
| unified_experiments.py | Model/environment construction, experiment execution and evaluation |
| live.py | Interactive single-model viewer |
| syncmap/ | NSGD, RG and their supporting/baseline implementations |
| test/ | Runtime graph and hierarchy environments, plus focused evaluation checks |
| tools/ | Shared configuration, readout, CSV summaries, plotting, the benchmark and supplementary-experiment drivers, and the presentation diagrams |
| datasets/ | Synthetic graph definitions; hierarchy graphs under datasets/hierarchy/, supplementary graphs under datasets/supp/ |
| results/benchmark/ | Representative benchmark: report, comparison CSV, figures, GIFs (tools/benchmark.py) |
| results/supp/ | Supplementary experiments: report, comparison CSV, figures (tools/supp_run.py, tools/supp_report.py) |
| docs/ | Documentation map, design, usage and evaluation notes |
| docs/presentation/ | HTML slide deck and its diagrams (tools/deck_figures.py) |
| project_manifest.yaml | Runtime/profile inventory and evaluation contract |
| requirements.txt | Python dependency declarations |

NSGD is implemented in syncmap/SyncMapX_NSGD.py. RG is implemented in syncmap/SyncMapRG.py and uses syncmap/SyncMapNS.py for each level. The baseline files are SyncMap.py (original SyncMap, constant learning rate), SyncMapX_Symmetric.py (Symmetrical SyncMap, constant learning rate) and SyncMapTSF.py. SyncMapCI.py and SyncMapX_CI.py contain exploratory directed-transition work.

The test directory is needed at runtime: it is not solely a unit-test suite. Support implementations under tools/archive/ are retained only where the current runtime still imports them.
