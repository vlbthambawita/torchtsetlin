Records produced by the P0 harness *before* the M2 per-clause match-count diagnostic
(diagnostics.match_count) was added at the orchestrator's request mid-P0. Their accuracies
are unaffected -- M2 is a read-only probe that consumes no RNG -- but they no longer satisfy
record.validate(), which requires match_count for any arm with more than one patch.
Superseded by the re-run of the same (arm, seed) pairs in results/.
