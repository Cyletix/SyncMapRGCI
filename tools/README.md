# Tools

Shared runtime support remains at its original paths.

| Files | Purpose |
|---|---|
| main_config.py | Task defaults, profile presets and comparison configuration |
| clustering_strategies.py, rg_mst.py, metric_utils.py | Readout and scoring |
| density_scale.py | Geometry support |
| result_paths.py | Output routing |
| gui_runtime.py, multilive.py, plot_theme.py, visual_tools.py | Viewer and plotting support |
| trace_rg_hierarchy_scores.py, render_rg_trace_plot.py | Sample and render score trajectories |
| csv_summary.py | CSV summaries |
| benchmark.py | Representative benchmark runner and report builder (results/benchmark/) |
| make_supp_datasets.py, supp_run.py, supp_report.py | Supplementary experiments: graphs in datasets/supp/, the resumable run grid and its report (results/supp/) |
| deck_figures.py | Diagrams of the presentation (docs/presentation/) and its single-file bundle |
| sequence_information_analysis.py | Inspect input/environment statistics |
| scaling_law.py | Scaling and trace analysis |
| switch.py | Structural-switch probe and report dispatcher |
| conv.py | Convergence-by-step or wall-clock dispatcher |

The two dispatchers use retained implementations in _probes/. The representative
hierarchy snapshot helper remains in archive/rg_hierarchy_diagnostics.py and is
imported explicitly by the viewer/trace code.

Use each command's --help to inspect its arguments.
