# CLI reference

Run commands from the repository root in a Python 3.11 environment with requirements.txt installed. The public interface uses named flags. Parameter presets are selected with --profile; implementations are selected with --strategy. Repeat --set key=value to override a model parameter.

```bash
python main.py --help
python main.py --list-profiles
```

## Runtime and profiles

Primary profiles: nsgd, rg. Comparison profiles: syncmap, symmetric, tsf.
Additional runtimes: ns (RG's single-level kernel), ci (directed-transition exploration).
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

Task 2 uses --profiles/--strategies and --datasets for lists. --steps controls the single-run/live/trace tasks; use --comparison-sequence-lengths for batch comparison. The CLI rejects silently ignored list selections.

## Output controls

Task 0/4/6 JSON output is enabled by default. Use --save-json false to disable it or --json-path to choose a destination. --save-png and --save-gif control the static and animated single-run exports. GUI viewing is opt-in for task 0 through --live. A GUI requires Qt and a display server.

For task 0, --trace runs a separate trace experiment. It is not a free export from the first run. --record-interval and --trace-record-interval control sampling.

## Evaluation

Flat graphs use the shared unsupervised geometric readout. Hierarchical ARI is the equal-weight mean across every required hierarchy level, with a missing required level scored as zero. NMI remains a fixed-level diagnostic; per-level scores and cophenetic correlation provide additional context.

## Benchmark

python tools/benchmark.py run reruns the representative benchmark (eight graphs, five models, 100,000 steps, five seeds) through main.py and rebuilds results/benchmark/. python tools/benchmark.py report only rebuilds the figures and REPORT.md from the saved runs.
