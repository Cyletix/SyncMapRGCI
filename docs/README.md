# Documentation map

Everything public about SyncMapRGCI, by purpose. The repository README is available in [English](../README.md), [简体中文](../README.zh-CN.md) and [日本語](../README.ja.md); everything else is in English.

## Start here

| If you want to | Read |
|---|---|
| know what the project is and run it | [README](../README.md) |
| get the idea without any SyncMap background | [Beginner presentation](presentation/beginner.html) (HTML slides, see below) |
| understand RG step by step, with results | [Presentation](presentation/index.html) (detailed version) |
| see the main comparison | [Benchmark report](../results/benchmark/REPORT.md) |
| see the targeted follow-up experiments | [Supplementary report](../results/supp/REPORT.md) |

## Presentations

Two self-contained slide decks (English, dark theme, speaker notes included):

- [`presentation/beginner.html`](presentation/beginner.html), for newcomers with no SyncMap background (18 main slides plus backup): why chunking matters, the task as a game, why choosing the scale is the hard part, SyncMap and RG in one picture each, how a grouping is scored, and the main results in plain words.
- [`presentation/index.html`](presentation/index.html), the detailed version for SyncMap readers (53 slides): the problem, the RG mechanism piece by piece, the evaluation protocol, the benchmark, the supplementary experiments and the conclusions.

How to use them:

- GitHub shows HTML as source. Open them from a clone (`docs/presentation/*.html` in any browser), or download the single-file versions or the PDFs from the [releases page](https://github.com/Cyletix/SyncMapRGCI/releases).
- Keys: `→` / `Space` next, `←` previous, `N` speaker notes, `F` full screen. `#12` in the address opens slide 12.
- Their diagrams are drawn by [`tools/deck_figures.py`](../tools/deck_figures.py) (`f*.png` for the detailed deck, `b*.png` for the beginner deck); their result figures are the ones in `results/benchmark/` and `results/supp/`.

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
