# Documentation map

Everything public about SyncMapRGCI, by purpose. The repository README is available in [English](../README.md), [简体中文](../README.zh-CN.md) and [日本語](../README.ja.md); everything else is in English.

## Start here

| If you want to | Read |
|---|---|
| know what the project is and run it | [README](../README.md) |
| understand RG step by step, with results | [Presentation](presentation/index.html) (HTML slides, see below) |
| see the main comparison | [Benchmark report](../results/benchmark/REPORT.md) |
| see the targeted follow-up experiments | [Supplementary report](../results/supp/REPORT.md) |

## Presentation

[`presentation/index.html`](presentation/index.html) is a self-contained slide deck (English, dark theme, speaker notes included): the problem, the RG mechanism piece by piece, the evaluation protocol, the benchmark, the supplementary experiments and the conclusions.

- GitHub shows HTML as source. Open it from a clone (`docs/presentation/index.html` in any browser), or download the single-file version or the PDF from the [releases page](https://github.com/Cyletix/SyncMapRGCI/releases).
- Keys: `→` / `Space` next, `←` previous, `N` speaker notes, `F` full screen. `#12` in the address opens slide 12.
- Its diagrams are drawn by [`tools/deck_figures.py`](../tools/deck_figures.py); its result figures are the ones in `results/benchmark/` and `results/supp/`.

## Results

- [Benchmark report](../results/benchmark/REPORT.md): 8 scenarios × 5 models × 5 seeds at 100,000 steps, with per-level scores, 3-D maps, score curves and training GIFs. Built by [`tools/benchmark.py`](../tools/benchmark.py).
- [Supplementary report](../results/supp/REPORT.md): a community-orientation sweep on 300-state graphs, harder hierarchies at three mixing levels, and RG with recursion switched off. Built by [`tools/make_supp_datasets.py`](../tools/make_supp_datasets.py), [`tools/supp_run.py`](../tools/supp_run.py) and [`tools/supp_report.py`](../tools/supp_report.py).
- Large files (full result tables with learning curves, the standalone deck, the deck as PDF) are attached to the [releases](https://github.com/Cyletix/SyncMapRGCI/releases) instead of the repository.

## Design and usage

- [Design reference](DESIGN_REFERENCE.md): algorithms and the evaluation contract.
- [CLI reference](CLI_REFERENCE.md): tasks, flags and runnable examples.
- [Project structure](PROJECT_STRUCTURE.md): paths and responsibilities.
- [Output governance](OUTPUT_GOVERNANCE.md): where generated outputs go.
- [Project workflow](PROJECT_WORKFLOW.md): running and checking the code.

## Status

- [Current plan](PLAN.md): remaining research questions.
- [Known limitations](ISSUES_LOG.md): interpretation and implementation limits.

## Data

- `datasets/`: flat graph-walk graphs (`.dot`).
- `datasets/hierarchy/`: hierarchical graphs used by the benchmark.
- `datasets/supp/`: graphs generated for the supplementary experiments by `tools/make_supp_datasets.py`.

## Background

- Vargas & Asabuki, "Continual General Chunking Problem and SyncMap", AAAI 2021.
- Zhang & Vargas, "Symmetrical SyncMap for Imbalanced General Chunking Problems", arXiv:2310.10045.
- Vargas, Tham & Zhang, "Dynamical Equations with Bottom-up Self-Organizing Properties Learn Accurate Dynamical Hierarchies without any Loss Function", arXiv:2302.02140.
