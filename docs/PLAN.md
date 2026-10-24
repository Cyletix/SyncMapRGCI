# Research plan

The benchmark supports review of NSGD and RG on flat and hierarchical synthetic sequences.

Remaining work:
1. Validate hierarchy readout stability over multiple checkpoints and seeds, and close the gap the oracle-K diagnostic exposes between what a map separates and what the MST gap cut recovers.
2. Fix or replace the nested_modulo level scheduler so that levels above L1 receive their intended cadence, then repeat the hierarchy experiments.
3. Separate input, relation, optimizer and upper-level token memory in controlled comparisons.
4. Compare compute budgets explicitly alongside input-step budgets.
5. Test adaptation when the graph structure switches, including return-to-source sequences, before making retention claims.
6. Develop the directed-transition (CI) line into a tested causal-structure method.

These are proposed directions, not completed results.
