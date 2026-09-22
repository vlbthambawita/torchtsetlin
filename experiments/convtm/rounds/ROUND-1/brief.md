# ROUND-1 brief

**Convened**: 2026-09-20
**Question**: *Which existing approaches to convolution in Tsetlin machines do we re-implement, in
what order, and what does each cost? And what, from the CNN side, is the mechanism inventory we will
mine for candidates in P5?*

## Why this round exists

P3 spends 80-120 GPU-hours re-implementing published methods. Choosing that list well is the highest
-leverage decision in the programme, and it is the one decision we are allowed to take **before** we
have our own measurements — because it is a decision about *what to measure*, not about what works.
Everything after this round needs `[MEASURED]` evidence.

## Inputs

- `PLAN.md` (the contract), `CHARTER.md` (the rules card)
- `source_documents/papers/**` (~45 PDFs), `source_documents/book_chapters/**`
- `experiments/mctm/` — this repo's prior programme, summarised in PLAN.md Section 3
- `src/torchtsetlin/` — what the library can express today (READ-ONLY)
- `code/CONTRACT.md` — the harness interface

## Required outputs

| Member | File | Content |
|---|---|---|
| tm-theorist | `LITERATURE_TM.md`, `THEORY.md`, `rounds/ROUND-1/tm-theorist.md` | see agent definition |
| dl-expert | `LITERATURE_CNN.md`, `MECHANISMS.md`, `rounds/ROUND-1/dl-expert.md` | see agent definition |
| research-engineer | `rounds/ROUND-1/research-engineer.md` | feasibility: for each proposed arm, can the harness express it, what does it cost, what would have to be written |

Position papers use the fixed format in PLAN.md Section 5.1 (answer in one sentence; claims with
tags; what I need measured, ranked with cost; what would change my mind).

## What the round must produce

`DR-001`: the ranked P3 shortlist — minimum 6 arms, target ~12 — each with a paper, a reported
CIFAR-10 number to reproduce against, an owner, a command line and a cost estimate. Plus the
matched-budget controls each arm needs.
