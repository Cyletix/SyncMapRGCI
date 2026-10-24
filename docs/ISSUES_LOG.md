# Known limitations

- The benchmark compares equal input-step budgets. It does not establish memory-matched or compute-matched comparisons; RG costs about 3-4x NSGD per step on flat graphs and 14-17x on hierarchies.
- RG upper-level clocks alias under the default nested_modulo scheduler (target 50 / 158 / 499 raw steps become 50 / 3,950 / about 1,971,050), so levels above L1 see few observations within 100,000 steps. The elapsed_geometric option also changes event counts and force exposure and is not validated; scheduler superiority is not established.
- The hierarchy readout is a single-checkpoint MST gap cut. It can jump between neighbouring checkpoints, and it can miss coarse groups that the map separates clearly when a model collapses each group to a point; the oracle-K columns show where the readout, rather than the map, is the bottleneck.
- Hierarchy datasets are detected by path (datasets/hierarchy/) or by .dot content (internal nodes or multi-level labels); --env-mode overrides the detection.
- Claims about full causal identification are not supported by the included experiments; CI is exploratory.
- Interactive viewing (task 1, task 0 --live) requires Qt and a display server; batch operation is available without a GUI.
