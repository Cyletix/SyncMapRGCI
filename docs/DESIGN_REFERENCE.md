# Design and evaluation reference

## Scope

The main research lines are NSGD and RG. SyncMap, Symmetric and TSF provide comparison baselines. NS is the single-level learning kernel used by RG. CI records exploratory work on directed transitions; it is not a completed causal-identification method.

## NSGD

Symbols are represented by coordinates. Exponentially weighted co-activity statistics, normalized by activity counts, define a thresholded nearest-neighbor relation graph. Position updates combine relation attraction and activity-dependent displacement, with per-point plasticity and bounded steps.

The maintained profile uses a 100-observation relation memory, plasticity decay 0.99 and a positive floor of 0.01. Freezing therefore slows coordinate motion rather than making the model permanently immutable. Relation statistics continue updating. The generic repulsion coefficient being zero does not eliminate activity-dependent displacement.

The implementation is in [SyncMapX_NSGD.py](../syncmap/SyncMapX_NSGD.py); profile overrides are in [main_config.py](../tools/main_config.py).

## RG

[SyncMapRG.py](../syncmap/SyncMapRG.py) builds recursive levels from the [SyncMapNS.py](../syncmap/SyncMapNS.py) kernel. Lower-level groups produce tokens for the next level. Local contrastive dynamics, relation springs, token membership and configured feedback contribute to the representation.

Each level keeps an undirected relation memory from consecutive thresholded observations: the Cartesian pairs of the previous and current active sets are added symmetrically to a decayed count matrix C (memory M, decay 1 − 1/M), alongside decayed activity counts n. The similarity is C_ij / sqrt(n_i · n_j); after a power gain, the top-k neighbours above a threshold become spring edges with rest length r_L. The primitive contrastive force pulls active points toward their weighted centre and pushes inactive points away from it with magnitude proportional to 1/distance (bounded by a distance floor and a step cap).

NSGD and the NS kernel are distinct implementations. RG does not repeatedly instantiate the NSGD class. RG defaults are defined by SyncMapX_RG.DEFAULT_PARAMS; the bare rg profile adds no parameter overrides.

At the reviewed defaults, distance scales grow by 3.16 and statistics-memory capacities by a factor of 10 per level from a base of 300. Each level receives its own token sequence. Nested scheduling and token cadence mean that these capacities are not directly comparable in raw-input time.

Flat RG disables the upper layer by default. On hierarchy datasets the factory builds depth + 1 levels (DCH1: 3, IEH: 5). With the default nested_modulo scheduler a level only checks its cadence when the level below fires, so the deep cadences alias to the least common multiple of the per-level targets (50 → 3,950 → about 1,971,050 raw steps); within 100,000 steps the levels above L1 receive few observations. Neither internal token groups nor a favorable final snapshot alone establish successful hierarchical recovery.

## Input environments

Flat graphs are walked over their .dot edges. Hierarchy graphs ignore the .dot edges: the stream is generated from the multi-level node labels. At each level, from the top down, the walk jumps to a sibling group with probability 0.45 × 0.25^(level above the finest) (0.45 between chunks, 0.1125 between the groups one level up, and so on); otherwise it moves to a uniformly chosen member of its current finest chunk (env_hierarchy_leaf_switch_prob and env_hierarchy_level_switch_decay). In both cases each node is held for 11 input steps (TIME_DELAY = 10) and the observation carries an exponentially decaying trace of the previous node.

## Evaluation

All retained representation-learning methods use the shared unsupervised geometry evaluator on their current coordinates: DBSCAN on standardised coordinates for flat graphs (eps = 1.25 × median k-distance, min_samples 2) and cuts of the Euclidean minimum spanning tree at large neighbouring-edge gaps (ratio ≥ 1.05) for hierarchy graphs, one cut per required level. The models' own organize() readouts are not used for scoring. Ground-truth labels are used to score predictions, not to choose the best predicted level.

The formal hierarchy ARI is the arithmetic mean of cluster_level1_ari_score through the required depth. A missing required level contributes zero. The weakest level can be read from the per-level columns (there is no separate worst-level column). NMI remains fixed-level; cophenetic correlation and individual level scores describe complementary properties.

The oracle_cluster_levelN_ari_score columns cut the same geometry with Ward agglomerative clustering at the true cluster count of each level. They leak the true count and are never a formal score: high oracle ARI with low formal ARI means the structure is in the map and the readout is the bottleneck; low oracle ARI means the dynamics did not form it.

## Fairness and limitations

Equal input steps do not imply equal computation or equal memory. Input history, co-activity/transition statistics, plasticity, coordinates, optimizer state, caches and upper-level token memories are distinct channels. NSGD's co-activity evidence and RG's transition evidence also differ even when their time constants match.

With an explicit seed, the input sequence is drawn from an environment-private random stream, independent of model construction and updates; the benchmark seeds every trial this way. Upper-level raw-clock control and late-window stability remain open research questions.

## Relation to the research theme

The experiments concern synthetic graph-walk symbol sequences. Hierarchical visual chunking and causal relations motivated the research; a validated visual causal-inference system has not been established. The directed-transition source files are exploratory components.

See the [benchmark report](../results/benchmark/REPORT.md) and the [supplementary report](../results/supp/REPORT.md) for the quantitative evidence, and the [presentation](presentation/index.html) for a step-by-step explanation.
