# Design and evaluation reference

## Scope

The main research lines are NSGD and RG. SyncMap, Symmetric and TSF provide comparison baselines. NS is the single-level learning kernel used by RG. CI records exploratory work on directed transitions; it is not a completed causal-identification method.

## NSGD

Symbols are represented by coordinates. Exponentially weighted co-activity statistics, normalized by activity counts, define a thresholded nearest-neighbor relation graph. Position updates combine relation attraction and activity-dependent displacement, with per-point plasticity and bounded steps.

The maintained profile uses a 100-observation relation memory, plasticity decay 0.99 and a positive floor of 0.01. Freezing therefore slows coordinate motion rather than making the model permanently immutable. Relation statistics continue updating. The generic repulsion coefficient being zero does not eliminate activity-dependent displacement.

The implementation is in [SyncMapX_NSGD.py](../syncmap/SyncMapX_NSGD.py); profile overrides are in [main_config.py](../tools/main_config.py).

## RG

[SyncMapRG.py](../syncmap/SyncMapRG.py) builds recursive levels from the [SyncMapNS.py](../syncmap/SyncMapNS.py) kernel. Lower-level groups produce tokens for the next level. Local contrastive dynamics, relation springs, token membership and configured feedback contribute to the representation.

NSGD and the NS kernel are distinct implementations. RG does not repeatedly instantiate the NSGD class. RG defaults are defined by SyncMapX_RG.DEFAULT_PARAMS; the bare rg profile adds no parameter overrides.

At the reviewed defaults, distance scales grow by 3.16 and statistics-memory capacities by a factor of 10 per level from a base of 300. Each level receives its own token sequence. Nested scheduling and token cadence mean that these capacities are not directly comparable in raw-input time.

Flat RG disables the upper layer by default. On hierarchy datasets the factory configures sufficient recursive levels for the dataset depth. Neither internal token groups nor a favorable final snapshot alone establish successful hierarchical recovery.

## Evaluation

All retained representation-learning methods use the shared unsupervised geometry evaluator. Ground-truth labels are used to score predictions, not to choose the best predicted level.

The formal hierarchy ARI is the arithmetic mean of cluster_level1_ari_score through the required depth. A missing required level contributes zero. The worst-level score is diagnostic. NMI remains fixed-level; cophenetic correlation and individual level scores describe complementary properties.
## Fairness and limitations

Equal input steps do not imply equal computation or equal memory. Input history, co-activity/transition statistics, plasticity, coordinates, optimizer state, caches and upper-level token memories are distinct channels. NSGD's co-activity evidence and RG's transition evidence also differ even when their time constants match.

With an explicit seed, the input sequence is drawn from an environment-private random stream, independent of model construction and updates; the benchmark seeds every trial this way. Upper-level raw-clock control and late-window stability remain open research questions.

## Relation to the research theme

The experiments concern synthetic graph-walk symbol sequences. Hierarchical visual chunking and causal relations motivated the research; a validated visual causal-inference system has not been established. The directed-transition source files are exploratory components.

See the [benchmark report](../results/benchmark/REPORT.md) for the quantitative evidence.
