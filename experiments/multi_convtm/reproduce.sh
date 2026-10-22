#!/usr/bin/env bash
# One-line reproduction of experiments/multi_convtm (TM Composites, CIFAR-10).
#
#   ./reproduce.sh                       # everything: data, tmu env, torchtsetlin (hard + soft
#                                        #   budget, 3 seeds), tmu (authors' scripts), report
#   ./reproduce.sh --quick               # 1-epoch smoke test into .cache/quick/ (~1 h)
#   ./reproduce.sh --stages reference    # Table 2 from the authors' class sums (seconds)
#   ./reproduce.sh --stages report       # rebuild REPORT.md / PROPOSAL.md from results/
#   ./reproduce.sh --stages torchtsetlin --variants soft --seeds 0 --gpus 1
#
# Re-running resumes: finished runs (status "done" in results/) are skipped.
# Full run: roughly 2 days on an RTX 3090 + RTX 3080 (tmu is the long pole).
set -euo pipefail
cd "$(dirname "$0")/code"
exec python reproduce.py "$@"
