# Live risk register

Seeded from Section 13 of PLAN.md. Updated whenever a risk fires, is retired, or a new one appears.

| # | Risk | Likelihood | Status | Mitigation / stop condition |
|---|---|---|---|---|
| R1 | Vanilla CTM does not reproduce ~60% (T0 fails) | Medium | open | Hard stop at G3; engineer audits config against the paper; re-baseline the ladder with the user |
| R2 | Nothing beats 60.7% | Medium-high | open | Acceptable outcome; report becomes a rigorous negative with the accuracy decomposition |
| R3 | Compute overruns | Medium | open | P0 measures true costs; P5 screening filters; ask user before any single experiment >10 GPU-h |
| R4 | An agent decides by argument | Medium | open | Claim tags; notetaker flags `[HYPOTHESIS]`-backed decisions; orchestrator rejects the DR |
| R5 | Silent test-set leakage | Medium | open | Validation-only selection; engineer greps selection paths at every gate |
| R6 | An arm is a strawman | High | open | Matched budgets on all axes; documented tuning allowance; theorist reviews every arm vs its paper |
| R7 | Someone edits `src/` | Low | open | C1 guard at every gate and commit; hard stop |
| R8 | GPU OOM / queue collapse | Medium | open | Per-GPU lock; P0 memory table; return codes logged |
| R9 | Results outpace understanding | Medium | open | Rounds are mandatory synchronisation points |
| R10 | Report drifts from results | Low | open | C4; `check.sh`; engineer audits numbers-to-code at G7 |
