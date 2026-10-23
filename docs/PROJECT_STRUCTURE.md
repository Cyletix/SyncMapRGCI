# Project structure

The repository keeps its original code layout.

| Path | Role |
|---|---|
| main.py | Public command-line entry point |
| unified_experiments.py | Model/environment construction, experiment execution and evaluation |
| live.py | Interactive single-model viewer |
| syncmap/ | NSGD, RG and their supporting/baseline implementations |
| test/ | Runtime graph and hierarchy environments, plus focused evaluation checks |
| tools/ | Shared configuration, readout, CSV summaries, plotting and selected experiment helpers |
| datasets/ | Synthetic graph definitions; hierarchy graphs remain under datasets/hierarchy/ |
| results/benchmark/ | Representative benchmark: report, comparison CSV, figures, GIFs (tools/benchmark.py) |
| docs/ | Design, usage and evaluation notes |
| project_manifest.yaml | Runtime/profile inventory and evaluation contract |
| requirements.txt | Python dependency declarations |

NSGD is implemented in syncmap/SyncMapX_NSGD.py. RG is implemented in syncmap/SyncMapRG.py and uses syncmap/SyncMapNS.py for each level. The baseline files are SyncMap.py (original SyncMap, constant learning rate), SyncMapX_Symmetric.py (Symmetrical SyncMap, constant learning rate) and SyncMapTSF.py. SyncMapCI.py and SyncMapX_CI.py contain exploratory directed-transition work.

The test directory is needed at runtime: it is not solely a unit-test suite. Support implementations under tools/archive/ are retained only where the current runtime still imports them.
