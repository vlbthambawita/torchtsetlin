# TM query-key attention on MNIST

Tests the spec [tm-query-key-attention-mnist-implementation-plan.md](tm-query-key-attention-mnist-implementation-plan.md)
against a flat TM and a convolutional TM. Plan: [PLAN.md](PLAN.md). Deviations: [DECISIONS.md](DECISIONS.md).
Results: [report/REPORT.md](report/REPORT.md) (generated from `results/**/*.json`).

Run from the repository root (`src/torchtsetlin` is read-only here):

```bash
python -m experiments.transformer_like.code.cli prepare-data            # manifests (seed 314159)
pytest experiments/transformer_like/tests -q                           # 65 tests, CPU (+CUDA)
python -m experiments.transformer_like.code.cli smoke --device cuda     # 1k/200 smoke, never reads test
python -m experiments.transformer_like.code.cli probe --device cuda     # 2-epoch cost probe
python -m experiments.transformer_like.code.cli search --gpus 0,1 --per-gpu 2   # validation search + S3 search
python -m experiments.transformer_like.code.cli freeze-experiment      # writes results/frozen_experiment.json
python -m experiments.transformer_like.code.cli benchmark --stage final --gpus 0,1 --per-gpu 1
python -m experiments.transformer_like.code.cli benchmark --stage secondary --gpus 0,1 --per-gpu 2
python -m experiments.transformer_like.code.add_interventions secondary_s1   # D24
CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_like.code.cli timing --device cuda
python -m experiments.transformer_like.code.cli report
```

Every stage is resumable: a completed record is never rerun. `benchmark` refuses to run without
the frozen manifest, and every test-set read is appended to `results/test_access_log.jsonl`.
Checkpoints and feature caches live in `artifacts/` (git-ignored; SHA-256 in each record).
