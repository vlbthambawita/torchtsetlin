#!/bin/bash
# Clause sweep on GPU slot 2: team M, batch 64, 16 000 ... 128 000 clauses per class and member.
# Safe to re-run: it resumes. Options: see run_clauses_gpu.sh.
exec "$(dirname "$0")/run_clauses_gpu.sh" 2
