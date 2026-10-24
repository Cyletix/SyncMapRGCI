# CLI reference

Run commands from the repository root in a Python 3.11 environment with requirements.txt installed. The public interface uses named flags. Parameter presets are selected with --profile; implementations are selected with --strategy. Repeat --set key=value to override a model parameter.

```bash
python main.py --help
python main.py --list-profiles
```

## Runtime and profiles

Primary profiles: nsgd, rg. Comparison profiles: syncmap, symmetric, tsf.
Additional runtimes without a preset: ns (RG's single-level kernel) and ci (directed-transition exploration). Select them with --strategy ns / --strategy ci (task 2: --strategies); --profile ns is rejected with a hint.
Memory sensitivity presets: rg_shortmem, rg_shortmem_l0, tsf_longmem, tsf_velocity_longmem. These are controlled variants, not separate established algorithms.

The no-argument default is task 1, with RG on DCH1. Explicitly select task 0 for a batch run without opening the viewer.

## Tasks

| Task | Purpose |
|---|---|
| 0 | Single experiment; optional JSON, plot, animation or post-run viewer |
| 1 | Interactive live viewer; multiple --strategy values open a comparison view |
| 2 | Batch comparison (CSV and Markdown summaries) |
| 4 | Sample an experiment's score trajectory |
| 5 | Render an existing score-trace JSON |
| 6 | Analyze the input sequence/environment |
| 7 | Analyze scaling or fit a supplied trace |

## NSGD and RG examples

```bash
python main.py --task 0 --profile nsgd --dataset datasets/fixed5_6.dot --steps 5000 --seed 0 --save-gif false --save-png false
python main.py --task 0 --profile rg --dataset datasets/hierarchy/DCH1_fixed_chunk_6_3.dot --steps 5000 --seed 0 --save-gif false --save-png false
python main.py --task 1 --strategy nsgd rg --dataset datasets/hierarchy/DCH1_fixed_chunk_6_3.dot --steps 5000
python main.py --task 2 --profiles nsgd rg --datasets datasets/fixed5_6.dot datasets/hierarchy/DCH1_fixed_chunk_6_3.dot --comparison-sequence-lengths 5000 --comparison-trials 1 --seed 0
```

Task 2 uses --profiles/--strategies and --datasets for lists. --steps controls the single-run/live/trace tasks; use --comparison-sequence-lengths for batch comparison.

--profiles applies the maintained presets. --strategies runs the named runtimes with their class defaults, so for example NSGD keeps its class readout settings instead of the profile's, and preset names such as rg_shortmem are accepted only by --profiles. Without either flag, task 2 runs the MAIN_CONFIG strategy list with class defaults. In tasks 0/1, --strategy also accepts preset names and resolves them like --profile.

`--datasets flat hierarchy` expands to the 39 flat graphs in datasets/ plus the 10-graph hierarchy suite (49 graphs). Task 0 accepts but ignores task-2 list flags such as --datasets and --comparison-sequence-lengths.

## Output controls

Task 0/4/6 JSON output is enabled by default. Use --save-json false to disable it or --json-path to choose a destination. --save-png and --save-gif control the static and animated single-run exports. GUI viewing is opt-in for task 0 through --live. A GUI requires Qt and a display server.

For task 0, --trace samples the score curve during the same run and exports it (JSON and a curve PNG); it does not rerun the experiment. Adding --print-trace reruns task 0 in trace mode and prints per-level trace summaries. --record-interval and --trace-record-interval control sampling.

## Evaluation

Every model is scored by the same shared readout of its current coordinates: DBSCAN on standardised coordinates for flat graphs (eps = 1.25 × median k-distance, min_samples 2) and Euclidean-MST gap cuts for hierarchy graphs. The models' own organize() readouts are not used for scoring. Hierarchical ARI is the equal-weight mean across every required hierarchy level, with a missing required level scored as zero. NMI remains a fixed-level diagnostic; per-level scores, oracle-K per-level scores (true cluster count given to an agglomerative readout) and cophenetic correlation provide additional context.

## Benchmark and supplementary experiments

python tools/benchmark.py run reruns the representative benchmark (eight graphs, five models, 100,000 steps, five seeds) through main.py and rebuilds results/benchmark/. python tools/benchmark.py report only rebuilds the figures and REPORT.md from the saved runs.

python tools/make_supp_datasets.py writes the graphs in datasets/supp/; python tools/supp_run.py run runs the supplementary grid (resumable, one main.py call per cell); python tools/supp_run.py showcase records the seed-0 maps; python tools/supp_report.py builds the tables and figures in results/supp/.

python tools/deck_figures.py redraws the diagrams of the presentation in docs/presentation/.
