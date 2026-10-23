# Known limitations

- The benchmark compares equal input-step budgets. It does not establish memory-matched or compute-matched comparisons; RG costs more per step.
- RG upper-level clocks and token inputs complicate raw-time comparisons; scheduler superiority is not established.
- Hierarchy auto-detection is path-based (datasets/hierarchy/); other hierarchical graphs need --env-mode hierarchy_sequence.
- Claims about full causal identification are not supported by the included experiments; CI is exploratory.
- Interactive viewing requires Qt and a display server; batch operation is available without a GUI.
