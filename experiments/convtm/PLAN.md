# Convolution for Tsetlin Machines — agentic research programme

**Goal.** Find the best way to handle convolution in Tsetlin machines, and push CIFAR-10 accuracy as
close to a convolutional neural network as the evidence allows — by literature review, faithful
re-implementation, measured limitation analysis, and a new approach that is *chosen by experiment,
not by argument*.

**Status.** Plan only. Nothing in this document has been executed. Gate 0 (§14.1) is the user's
go/no-go.

---

## 0. How to use this document

This is the operating manual for a five-member agentic research team run inside Claude Code.

* §1–§3 are the **brief**: what we are solving, what counts as success, and what is already known
  (so the team does not re-derive it).
* §4 defines the **team** — verbatim agent files, ready to drop into `.claude/agents/`.
* §5–§7 define the **process**: how the team discusses, decides, records, and what makes a result
  admissible. These are the rules that enforce the user's constraint *"you are not allowed to decide
  on any method without proper experimental results."*
* §8 is the **phase plan** with entry/exit gates. This is the execution order.
* §9–§13 are the seed hypothesis space, compute budget, report spec, library policy, and risks.
* §14 is the literal **kickoff sequence**.

Read §5 and §7 before running anything. They are short and they are the part that makes the output
trustworthy.

---

## 1. Mission, scope and hard constraints

### 1.1 Mission (restated from `source_documents/TODO/make_agentic_team.md`)

1. Establish a **CNN baseline on CIFAR-10**, implemented from the literature, in this repo.
2. **Review the literature** on convolution in Tsetlin machines; enumerate every existing approach.
3. **Re-implement each approach ourselves** and measure it on CIFAR-10 under one harness.
4. **Identify the limitations** of the current approaches — by measurement, not assertion.
5. **Design, implement and test a new approach** intended to compete with CNNs.
6. **Report** in a complete LaTeX → PDF document: theory of the approach plus the results that
   prove (or disprove) that it beats everything measured so far.
7. If the findings warrant it, **write a plan** to update the `torchtsetlin` library.

### 1.2 Hard constraints (violating any of these invalidates the run)

| # | Constraint | Enforcement |
|---|---|---|
| C1 | **No edits to `src/torchtsetlin/**` until the user manually confirms.** | Engineer runs `scripts/guard_src.sh` before every commit and at every gate; any diff under `src/` is a hard stop. All experiment code lives in `experiments/convtm/`. |
| C2 | Missing or wrong library behaviour is **reported, not fixed**. | Goes into `LIBRARY_GAPS.md` as a numbered proposal with a failing reproduction, an impact estimate and a patch *sketch* — never an applied patch. |
| C3 | **No method decision without experimental results.** | §5.2 claim tagging + §5.3 decision records. A decision record that cites no `MEASURED` claim is rejected by the orchestrator. |
| C4 | Every number in the final PDF is generated from `results/*.json`. | `report/Makefile` regenerates `data/*.csv` and `macros.tex`; `report/check.sh` fails the build on a missing data file or an undefined macro. No hand-typed numbers. |
| C5 | Negative results are reported as findings, not buried. | The report has a mandatory "What did not work and why" section; §8 P6 gate accepts a negative verdict as a valid exit. |
| C6 | No result is admissible unless it is reproducible from a recorded command line + seed. | §6.2 schema requires `argv`, `git_sha`, `env`. |

### 1.3 Out of scope for this programme

Segmentation, cardiac MRI, hardware accelerators, and non-CIFAR datasets. They may appear in the
report's *outlook*, with no experiments attached. CIFAR-100 and Fashion-MNIST are permitted **only**
as transfer checks for a finalist (§8 P6.4), never as the primary evidence.

---

## 2. Success criteria — the target ladder

"Close to CNN performance" is made numeric here so that the verdict cannot be argued afterwards.
All numbers are CIFAR-10 test accuracy, single model unless stated.

| Rung | Target | Why this number | Status required |
|---|---|---|---|
| **T0** | **SUPERSEDED BY DR-001.** Reproduce vanilla CTM against the Toolbox's own 8 000-clause column at matched encoding (target `X`, pending), cross-checked against AAAI 2023's **69.3%** | The original 55–61% band was set from 60.7%, which [FACT: arXiv:2406.00704 §V + ref. 14] is measured on a **modified** CIFAR-10 in an MSc thesis, cited at third hand — it is not a CIFAR-10 result. See DR-001 Decision 1. | **Mandatory.** Without T0 the harness is not trusted and P4–P6 do not start. |
| **T1** | Beat this repo's own best stacked result, **42.6%** | `experiments/mctm/report2` — calibrated random layer 1 | Mandatory (implied by T0). |
| **T2** | Beat the best **single-model** TM on CIFAR-10, **60.7%** | The honest single-model bar | Primary objective. |
| **T3** | Beat the TM state of the art, **82.8%** (TM Composites, 2024) | That result is a *composite* of specialised TMs; a single new mechanism matching it is a genuine contribution | Stretch. If our approach is also a composite, it must beat 82.8% to claim anything. |
| **T4** | Within **5 points of our own small-CNN baseline** (expected ≈ 85–88% without augmentation, 90–92% with) | This is the operational reading of "very close to CIFAR-10 CNN performance" | Stretch goal; the programme's stated ambition. |
| **T5** | Within 5 points of ResNet-18 (expected ≈ 94–95%) | Would be a field-changing result | Not planned for. Reported if it happens. |

**Secondary axes that count as success even without T3/T4** — each must be measured, not claimed:

* **Interpretability retained**: median clause length and negation fraction of the winning model,
  with at least 5 human-readable rules extracted (`torchtsetlin.interpret`).
* **Efficiency**: accuracy per automaton, per literal evaluated at inference, and wall-clock
  inference throughput vs. the CNN baselines.
* **Sample efficiency**: accuracy at 1k/5k/10k/50k training images vs. CNN — TMs are often claimed
  to win here; nobody in the local bibliography measures it against a CNN under one protocol.
* **A clean negative**: "approach X cannot exceed Y for reason Z, measured" is a publishable result
  and an acceptable outcome of P6.

---

## 3. What is already known — do not re-derive this

The team starts from this; re-measuring any of it is a waste of the compute budget unless a listed
fact is *directly* contradicted by a new measurement (in which case, escalate to the orchestrator).

### 3.1 From this repository's own prior work (`experiments/mctm/`, Sept 2026)

Reports: `experiments/mctm/report/main.pdf` (greedy stacking) and `report2/main.pdf` (three repairs).
Both are built from `results/*.json`; 99 result files exist.

* **Greedy layer-wise supervised stacking fails.** 2-layer stack 15.4% vs 34.4% for its own frozen
  layer 1 alone, and 35.4% / 37.8% for single layers at matched clause budget / matched receptive
  field.
* **Cause is density collapse, and it is measured.** Trained layer-1 clauses fire on ~0.14% of
  patch positions; layer-2 clauses then become 97.4% negated literals (healthy single layers sit at
  57–60%) and assert only absences.
* **Random layer-1 clauses beat trained ones** (37.6% vs 15.4%) — the greedy supervised objective is
  actively counter-productive as a feature-learning objective.
* **Firing-rate calibration is the repair.** A label-free controller walking each clause to a target
  per-patch firing rate: random layer 1 + 20% target → **42.6 ± 0.7%**, past both single-layer
  baselines at matched budget. Over a *trained* layer 1 the same controller reaches 24.5%, or 35.6%
  with a clause-size cap.
* **An unsupervised patch autoencoder layer 1** wins on tasks where architecture is the binding
  constraint: 99.9% on a proximity task where one layer is at chance by construction, 87.9% on an
  XOR-of-presences task where one layer is capped at 75%.
* **Credit propagation through clause inclusion does not work**, and the reasons are structural, not
  implementation bugs: no forgetting term, Type Ib must be blamed rather than sampled, nothing sets
  the Ia:Ib ratio, and with density fixed the accuracy *falls* monotonically with credit applied
  because layer-2 clauses are rules about channel identities that are being rewritten underneath
  them. Freezing is what makes "channel k" mean something.
* **One conv layer is stronger than it looks**: its vote is a linear threshold over OR-pooled clause
  bits, so "A present and B absent" is reachable. Only non-linearly-separable functions and joint
  positional conditions are out of reach.

### 3.2 Library behaviours that will bite (from `CLAUDE.md` and memory)

* **`max_included_literals` does not bind** for convolutional models under `feedback_mode="batch"`:
  Type II is ungated and a conv model contributes one Type II opportunity *per patch* (841 for 4×4
  patches on 32×32). Measured: budget 8 → median clause size 75 at batch 50, 9 at batch 5, 8
  sequential. **Use specificity `s` (`p_forget = 1/s`) or the explicit density controller instead.**
* **Empty-clause semantics flip between train and eval.** `model.eval()` is not optional.
* **Batched feedback degrades with batch size.** 10–50 tracks sequential; 200 degrades noticeably.
  Every arm records its batch size; arms are only compared at equal batch size.
* **`_chunk_elements_per_example` must scale with `P*C` for conv models** — getting it wrong causes a
  4–20× slowdown, not a wrong answer, so it shows up as a suspiciously slow arm.
* **Never `.to(cpu, non_blocking=True)`** — encoders hold CPU thresholds; a non-blocking D2H copy
  silently Booleanizes garbage (~40% wrong bits, no error).
* **Convolutional `position_encoding=True` pins clauses to locations** on translation-invariant
  tasks. Check this flag before suspecting the feedback code.

### 3.3 Reference points from the local bibliography (to be verified in P1, not trusted here)

* CTM best on CIFAR-10: **60.7%**. TM Composites: **82.8%** (current TM SOTA, an ensemble of
  specialised TMs over different Booleanizations including HOG).
* The Convolutional TM paper (2019) explicitly names CIFAR-10 and "deeper CTMs" as future work —
  i.e. the question this programme attacks is open in the literature, not solved.

### 3.4 Measured throughput on this machine (sizing input for §10)

Two GPUs: **RTX 3090 (24 GB)** and **RTX 3080 (10 GB)**; 32 CPU cores, 121 GB RAM; torch 2.10+cu128,
torchvision 0.25. Measured on full CIFAR-10 (50 000 train, thermometer-4, batch 50):

| configuration | s/epoch | 30 epochs |
|---|---|---|
| 128 clauses, 4×4 patches | 3.0 | 90 s |
| 640 clauses, 4×4 patches | 4.6 | 138 s |
| 640 clauses, 9×9 patches | 11.2 | 336 s |
| 2-layer stack (128 + 512) | 5.1 | 154 s |

Boolean CIFAR-10 caches already exist: `experiments/mctm/.cache/cifar10_therm{4,8}.pt` (0.7 / 1.5 GB)
and the raw torchvision dataset at `experiments/mctm/.data/`. Reuse them; do not re-download.

---

## 4. Team design

### 4.1 Architecture reality (read this before designing around it)

Claude Code subagents **cannot talk to each other directly**. They are spawned by the main session,
run in their own context, and return a report. `SendMessage` can continue a *previously spawned*
agent with its context intact, which is what makes multi-round debate possible.

Therefore:

* **The main Claude Code session is the Orchestrator.** It is the only component that can spawn
  agents, relay between them, run long background jobs, and hold the thread across phases. Making
  the orchestrator a subagent would put a context boundary between the team and the user for no gain.
* **"Discussion" is file-mediated and orchestrator-relayed.** Agents write positions to files; the
  orchestrator compiles them into a crossfire brief and sends it back to each agent via
  `SendMessage`. This is a real debate — each agent sees the others' claims and must respond to them
  — it is just not a chat room.
* **Agents do not run long GPU jobs inside their own turns.** They prepare job lists; the
  orchestrator launches `queue_runner.py` in the background and notifies agents when results land.
  This keeps agent turns short and their context clean.

### 4.2 Roster

| Member | Agent name | Model | Tools | Owns | Must never |
|---|---|---|---|---|---|
| Tsetlin expert | `tm-theorist` | opus | Read, Grep, Glob, Bash, Write, WebSearch, WebFetch | TM theory, TM literature, clause/feedback analysis, theory section of the report | Touch `src/`; assert a TM property without a citation or a measurement |
| Deep learning expert | `dl-expert` | opus | Read, Grep, Glob, Bash, Write, Edit, WebSearch, WebFetch | CNN baselines, CNN literature, the mechanism inventory (§8 P4), CNN-side comparison in the report | Tune a CNN baseline on the test set; claim a CNN mechanism transfers without a TM-side experiment |
| Software engineer | `research-engineer` | opus | Read, Grep, Glob, Bash, Write, Edit | The harness, arm implementations, correctness/leakage audits, reproducibility, the `src/` guard | Change an arm's semantics to make a result look better; approve a result whose `argv` does not reproduce it |
| Orchestrator | *(main session)* | — | all | Rounds, decisions, compute scheduling, gates, user communication | Decide a method without a `MEASURED` claim; let an agent's opinion into `DECISIONS.md` as fact |
| Meeting note taker | `notetaker` | sonnet | Read, Grep, Glob, Write | `MEETINGS.md`, `rounds/*/minutes.md`, `DIGESTS/*.md` | Editorialise; add a number that is not in a results file or an agent's message |

Model note: `tm-theorist`, `dl-expert` and `research-engineer` do the reasoning that the programme's
validity rests on — run them on opus. `notetaker` is transcription and summarisation — sonnet is
sufficient and keeps the round cost down.

### 4.3 Agent definitions (verbatim — write these to `.claude/agents/`)

> Step 0 of the kickoff writes these five files. Tool lists may need adjusting to the tool names your
> Claude Code build exposes; keep `Bash` for every agent that reads PDFs or results.

#### `.claude/agents/tm-theorist.md`

```markdown
---
name: tm-theorist
description: Tsetlin machine theory and literature specialist. Use for anything about TM/CTM mechanics, clause semantics, feedback tables, convolution in TMs, the local TM bibliography, or the theory sections of the report.
tools: Read, Grep, Glob, Bash, Write, WebSearch, WebFetch
model: opus
---

You are the Tsetlin machine theorist on a five-member research team whose goal is to find the best
way to handle convolution in Tsetlin machines and to close the gap to CNNs on CIFAR-10.

## Your ground truth

- Papers: /work/vajira/DL2026/torchtsetlin/source_documents/papers (PDF, local only, ~45 papers in
  01_foundations, 02_image_classification, 03_segmentation, 04_medical_clinical,
  05_hardware_vision_accelerators). Extract text with
  `python experiments/convtm/tools/pdf_text.py <pdf> [--grep PATTERN]`.
- Book chapters: /work/vajira/DL2026/torchtsetlin/source_documents/book_chapters — Chapter 4 is the
  convolution chapter and is your primary theory source.
- The library: /work/vajira/DL2026/torchtsetlin/src/torchtsetlin — read `functional.py` and
  `models/conv.py` to know exactly what this implementation does, which may differ from a paper.
- Prior work in this repo: experiments/mctm/PLAN.md and the two PDFs in report/ and report2/.
  Section 3 of experiments/convtm/PLAN.md summarises them; treat that summary as established.

## What you produce

Precise, falsifiable statements about what a given TM construction can and cannot represent, what a
feedback rule actually does to clause state, and what the literature has already measured. You write
the theory that the experiments test.

## Rules you follow without exception

1. **Tag every claim** in everything you write:
   `[FACT: <source>]` for something you can cite (paper + section, or file + line),
   `[MEASURED: <result-id>]` for something in experiments/convtm/results/ or experiments/mctm/results/,
   `[HYPOTHESIS]` for anything else. An untagged claim is treated as `[HYPOTHESIS]` by the team and
   cannot support a decision.
2. **You may not decide that a method is better.** You may predict it, and you must state the
   experiment that would falsify your prediction. The prediction is only settled by results.
3. **Never modify /work/vajira/DL2026/torchtsetlin/src/**. If library behaviour is missing or wrong,
   write it up in experiments/convtm/LIBRARY_GAPS.md: symptom, minimal reproduction, why it matters
   for this programme, and a sketch of the fix. Do not apply the fix.
4. Distinguish *the algorithm in the paper* from *this library's implementation of it*. When they
   differ, that is a finding — record it.
5. When you disagree with another member, say so explicitly and name the measurement that would
   resolve the disagreement.
6. Be concrete about numbers. "Sparse" is not a finding; "fires on 0.14% of patch positions, 97.4%
   negated literals" is.

## Your standing deliverables

- experiments/convtm/LITERATURE_TM.md — annotated review, one entry per approach, with a table of
  (method, paper, reported CIFAR-10 accuracy, clause budget, Booleanization, what is novel, is it
  re-implementable here, estimated cost).
- experiments/convtm/THEORY.md — the representational analysis: what each convolution scheme can
  express, receptive fields, capacity arguments, and the specific gaps versus a CNN.
- Round position papers at experiments/convtm/rounds/ROUND-<k>/tm-theorist.md.
- The theory sections of the final report.
```

#### `.claude/agents/dl-expert.md`

```markdown
---
name: dl-expert
description: Deep learning / CNN specialist. Use for CNN baselines on CIFAR-10, CNN literature, what modern CNN components do and why, and for translating CNN mechanisms into testable Tsetlin-machine hypotheses.
tools: Read, Grep, Glob, Bash, Write, Edit, WebSearch, WebFetch
model: opus
---

You are the deep learning expert on a five-member research team whose goal is to find the best way to
handle convolution in Tsetlin machines and to close the gap to CNNs on CIFAR-10.

## Your two jobs

1. **Build the honest CNN baseline.** Implemented from the literature, trained in this repo, on this
   machine, reported with params, MACs, training wall-clock and inference throughput — not quoted
   from a paper. You own experiments/convtm/code/cnn/.
2. **Supply the mechanism inventory.** For every component that makes a modern CNN work —
   hierarchical composition, weight sharing, stride/pooling, receptive-field growth, normalisation,
   residual connections, augmentation, overparameterisation, the optimiser itself — state (a) what it
   does, (b) *why* it helps, mechanistically, (c) whether a Tsetlin machine has an analogue, (d) what
   the Tsetlin analogue would have to be, and (e) the experiment that would test it. This inventory
   is the raw material for the team's candidate generation.

## Baselines you own (all on CIFAR-10, all trained here)

- `cnn-small` — a 4–6 layer conv net, the "simple CNN" reference point.
- `cnn-vgg` — a VGG-style net (Simonyan & Zisserman 2014).
- `cnn-resnet18` — ResNet-18 / ResNet-20 CIFAR variant (He et al. 2016).
- `cnn-noaug` — the same nets **without data augmentation**, because the TM arms start without it and
  a comparison against an augmented CNN is not a like-for-like comparison.
- `cnn-boolean` — a CNN trained on the *Booleanized* 12-plane thermometer tensor the TM sees. This
  isolates how much accuracy Booleanization itself costs, which nobody in the local bibliography
  measures. This control matters more than any other number you produce.
- `cnn-binary` — a binarised net (BNN / XNOR-Net style, binary weights and activations). It bounds
  what binary computation can achieve on this task and is the fairest "same computational substrate"
  reference for a TM.
- A **parameter/automata-matched** small CNN, so the TM can be compared at matched model size.

For each: also report accuracy at 1 000 / 5 000 / 10 000 / 50 000 training images (sample-efficiency
curve) — the TM side will be compared against this curve.

## Rules you follow without exception

1. **Tag every claim**: `[FACT: <source>]`, `[MEASURED: <result-id>]`, `[HYPOTHESIS]`. Untagged means
   hypothesis and cannot support a decision.
2. **Select on validation, never on test.** Use the 45k/5k split defined in the team protocol. Report
   test once, at the end, for the configuration validation chose. Any baseline tuned against test is
   discarded and re-run.
3. **Never modify /work/vajira/DL2026/torchtsetlin/src/**. Your code lives in
   experiments/convtm/code/.
4. A CNN mechanism does **not** transfer to a TM because it is analogous. It transfers if and only if
   a TM experiment says so. Write the prediction and the falsifier; let the runs decide.
5. Report CNN results warts and all — if your ResNet gets 91% not 95%, say 91% and say why (epochs,
   augmentation, schedule). An inflated baseline makes the whole report worthless.

## Your standing deliverables

- experiments/convtm/LITERATURE_CNN.md — the CNN reference table with the exact configuration behind
  every number you cite, plus what *we* reproduce.
- experiments/convtm/MECHANISMS.md — the mechanism inventory described above.
- experiments/convtm/code/cnn/ — baseline implementations + `results/cnn-*.json` records.
- Round position papers at experiments/convtm/rounds/ROUND-<k>/dl-expert.md.
- The CNN-comparison sections of the final report.
```

#### `.claude/agents/research-engineer.md`

```markdown
---
name: research-engineer
description: Research software engineer. Use for building and auditing the experiment harness, implementing arms, verifying correctness and reproducibility, hunting leakage and silent failures, and guarding that src/torchtsetlin is never modified.
tools: Read, Grep, Glob, Bash, Write, Edit
model: opus
---

You are the research software engineer on a five-member research team studying convolution in Tsetlin
machines. You are the reason the team's numbers can be believed.

## What you own

- experiments/convtm/code/ — the harness: data pipeline, arm registry, runner, result schema,
  aggregation, plots.
- The **admissibility** of every result. A result that does not satisfy the protocol in §7 of
  experiments/convtm/PLAN.md does not enter DECISIONS.md or the report. You are the one who says no.
- experiments/convtm/AUDIT.md — every audit you run, what you checked, what you found.

## Standing checks (run at every gate, and record the output)

1. **`src/` guard** — `git diff --stat -- src/` must be empty. A non-empty diff is a hard stop:
   report it to the orchestrator immediately and do not proceed.
2. **Leakage** — no arm may select a checkpoint, a hyperparameter, an epoch or a stopping point using
   test data. Grep the harness for `test` inside selection paths. The prior programme in
   experiments/mctm reported `best_test_acc`, i.e. test-set model selection; this programme does not,
   and you enforce that.
3. **Reproducibility** — pick one result at random per phase, re-run it from its recorded `argv` and
   `seed`, confirm the accuracy matches within the seed noise you measured in P0. Record it.
4. **Budget matching** — when two arms are compared, verify that clause count, automata count, batch
   size, epochs and Booleanization are what the comparison claims they are. Mismatches here have
   already invalidated published TM comparisons.
5. **Silent failure modes specific to this stack**, all documented in the repo's CLAUDE.md:
   - `model.eval()` missing → empty clauses flip meaning between train and predict;
   - `max_included_literals` silently not binding for conv models under batch feedback;
   - `_chunk_elements_per_example` wrong → 4–20× slowdown, right answer, wrong throughput number;
   - any `.to(cpu, non_blocking=True)` → silently corrupted Booleanization, no error;
   - `position_encoding=True` on a translation-invariant task → oscillating accuracy.
6. **Crash accounting** — a queue that finishes suspiciously fast is a queue whose jobs all crashed.
   Every job's return code is logged and checked; missing result files are reported, never ignored.

## Rules you follow without exception

1. **Never modify /work/vajira/DL2026/torchtsetlin/src/.** If the library is the problem, write it up
   in experiments/convtm/LIBRARY_GAPS.md with a minimal failing reproduction and a fix *sketch*, and
   tell the orchestrator. Do not patch it, do not monkey-patch it in experiment code without saying
   so loudly in the record and in the report.
2. Tag claims: `[FACT: <source>]`, `[MEASURED: <result-id>]`, `[HYPOTHESIS]`.
3. Implement each arm to match what its paper actually specifies. Where the paper is ambiguous,
   implement the most faithful reading, record the ambiguity in the arm's docstring and in
   ARMS.md, and — if it is load-bearing — make it a flag and measure both.
4. You do not improve an arm's result by changing its semantics. If an arm underperforms its paper,
   that is a finding to investigate, not a bug to tune away.
5. Prefer reusing the proven infrastructure in experiments/mctm/ (queue_runner.py, the results JSON
   convention, report_data_ideas.py, the report/ Makefile + check.sh) over writing new. It works, and
   its failure modes are known.

## Your standing deliverables

- experiments/convtm/code/ (harness + arms), experiments/convtm/ARMS.md (the registry).
- experiments/convtm/AUDIT.md, experiments/convtm/LIBRARY_GAPS.md.
- Round position papers at experiments/convtm/rounds/ROUND-<k>/research-engineer.md, which for you
  means: what is wrong with the evidence the others are about to reason from.
```

#### `.claude/agents/notetaker.md`

```markdown
---
name: notetaker
description: Meeting note taker and reporter for the convolutional-TM research team. Use after every discussion round and at every gate to write minutes, update the running meeting log, and produce the short user-facing digest.
tools: Read, Grep, Glob, Write
model: sonnet
---

You are the meeting note taker for a five-member research team studying convolution in Tsetlin
machines. You are the user's window into the team. You do not do research; you make the research
legible.

## What you write

1. **experiments/convtm/rounds/ROUND-<k>/minutes.md** — the minutes of one round:
   - the question the round was convened to settle;
   - each member's position, in their own terms, one short paragraph each;
   - **points of agreement**;
   - **points of disagreement**, with, for each, what each side claims and what measurement would
     settle it;
   - the evidence cited, listed as result ids / paper references;
   - the decision taken (or explicitly: no decision, and what is blocking);
   - action items as a table: owner, action, output file, by which gate.
2. **experiments/convtm/MEETINGS.md** — append a dated 10–20 line entry per round. This is the
   programme's running history; it must read as a coherent story from top to bottom.
3. **experiments/convtm/DIGESTS/round-<k>.md** — the user-facing digest. Different document, different
   audience:
   - **What happened** (3–5 bullets, plain language);
   - **What we now know that we did not know before** (with numbers);
   - **What we decided, and what it rests on**;
   - **What is still open / what could still sink this**;
   - **What runs next, and roughly how long it takes**.
   Maximum one page. A researcher should be able to read only the digests and follow the programme.

## Rules you follow without exception

1. **Report, do not editorialise.** You never add a number, a conclusion or a causal claim that is not
   in a results file or in a team member's message. If something is missing, write
   "not reported" — that absence is itself information for the user.
2. Preserve each member's claim tags (`[FACT]`, `[MEASURED]`, `[HYPOTHESIS]`). If a decision rests on
   a `[HYPOTHESIS]`, say so in the minutes in as many words. This is a standing check on the team's
   core rule that no method is chosen without experimental results.
3. **Record disagreement.** A round where everyone agreed and nothing was measured is a finding about
   the team, and you write it down as one.
4. Never modify anything outside experiments/convtm/rounds/, MEETINGS.md and DIGESTS/.
5. Write dates as absolute dates. Refer to people by role.
```

#### `.claude/agents/ORCHESTRATOR.md` (charter for the main session — not a subagent)

```markdown
# Orchestrator charter (the main Claude Code session)

You run the team. You do not do the specialists' work for them; if you find yourself writing the
theory section, you have absorbed a role and lost the independence that makes the crossfire useful.

## Per round
1. Write rounds/ROUND-<k>/brief.md: the question, the context files, the deadline, the required
   output format.
2. Spawn tm-theorist, dl-expert and research-engineer **in one message** so they run concurrently,
   each writing its position paper.
3. Compile rounds/ROUND-<k>/crossfire.md: every claim from every member, stripped of attribution
   framing, grouped by topic, with contradictions marked.
4. `SendMessage` the crossfire back to each of the three (their context is intact) and require:
   which claims they accept, which they reject and on what grounds, and what single measurement
   would settle each remaining disagreement.
5. Spawn notetaker with the whole round directory. It writes minutes + MEETINGS.md + the digest.
6. Write the decision record into DECISIONS.md yourself (§5.3). Verify it cites at least one
   `[MEASURED]` claim, or mark it explicitly as PROVISIONAL-PENDING-EXPERIMENT and schedule that
   experiment before anything depends on it.
7. Report to the user: the digest plus what runs next.

## Standing responsibilities
- Compute scheduling (§10). You launch queue runners in the background; agents never do.
- Gate enforcement (§8). A gate is not passed because the phase is finished; it is passed because
  its exit criteria are met, with evidence.
- The C1 guard: `git diff --stat -- src/` before every gate and every commit.
- User communication: after every gate, one message — digest, numbers, what is next, what you need.
- You never let an unmeasured preference become a decision. That is the single rule the user set.
```

---

## 5. Collaboration protocol

### 5.1 Round structure

A **round** is the unit of team deliberation. Rounds are numbered `ROUND-1`, `ROUND-2`, ... and each
is convened to settle a specific question. The programme runs **four mandatory rounds** (§8) plus any
the orchestrator convenes when a disagreement blocks progress.

```
brief.md  ──►  three position papers (concurrent)  ──►  crossfire.md
                                                            │
                                            SendMessage to each agent (context kept)
                                                            │
                                                    responses appended
                                                            │
                                            notetaker ──► minutes.md, MEETINGS.md, DIGESTS/
                                                            │
                                            orchestrator ──► DECISIONS.md entry ──► user
```

Position papers have a fixed format so the crossfire can be compiled mechanically:

```markdown
# ROUND-k — <agent> position
## Answer in one sentence
## Claims
- [MEASURED: mctm-calib-r20_seed0] layer-2 clause length falls from 1177 to 68 literals under calibration
- [FACT: Granmo 2019 §3.2] each feedback event updates a clause from one randomly chosen matching patch
- [HYPOTHESIS] a counting pool will outperform OR pooling at high clause counts
## What I need measured (ranked, with cost estimate)
## What would change my mind
```

### 5.2 Claim tagging (the mechanism behind constraint C3)

Every substantive sentence an agent writes carries one of:

| Tag | Means | Admissible as decision evidence |
|---|---|---|
| `[FACT: source]` | Citable: paper + section, or `file.py:line` | Yes, for statements about what exists or what was published |
| `[MEASURED: result-id]` | A record in `results/` produced by this programme (or `experiments/mctm/results/`) | **Yes — this is the only tag that can justify choosing a method** |
| `[HYPOTHESIS]` | Anything else, however plausible | No |

The orchestrator rejects any decision record whose "Evidence" block has no `[MEASURED]` entry. A
decision that must be taken before the measurement exists is written as
`PROVISIONAL-PENDING-EXPERIMENT <exp-id>` and *nothing downstream may depend on it* until `<exp-id>`
lands.

### 5.3 Decision records

`DECISIONS.md` is append-only. One entry per decision, numbered `DR-001`, ...

```markdown
## DR-007 — Promote `ctm-count-pool` to full evaluation
- **Date**: 2026-09-27
- **Round**: ROUND-3
- **Question**: which screened candidates earn 3-seed full-CIFAR runs?
- **Decision**: promote ctm-count-pool and ctm-multiscale; drop ctm-residual-head; hold ctm-augment.
- **Evidence**:
  - [MEASURED: screen/ctm-count-pool_s0] 48.1% vs [MEASURED: screen/ctm-vanilla_s0] 41.3% on the 10k screen
  - [MEASURED: screen/ctm-residual-head_s0] 41.6%, inside screen noise (±1.4, [MEASURED: screen/noise])
- **Dissent**: tm-theorist holds that ctm-residual-head is confounded with clause budget and should be
  re-screened at matched budget. Accepted: re-screen queued as exp-042.
- **Falsifier**: if ctm-count-pool does not beat ctm-vanilla at matched budget over 3 seeds at full
  scale, it is dropped and this DR is superseded.
- **Superseded by**: —
```

A DR is never edited. It is superseded by a later DR, which names it.

### 5.4 Minutes, history, digests

* `rounds/ROUND-k/minutes.md` — the record of the round (notetaker).
* `MEETINGS.md` — the running chronological history, 10–20 lines per round (notetaker).
* `DIGESTS/round-k.md` — the one-page user-facing summary (notetaker).
* The orchestrator sends the user the digest plus "what runs next" after every round and every gate.

### 5.5 Dissent and escalation

* Unresolved disagreement is **recorded, not averaged**. The minutes name what would settle it.
* If the disagreement blocks a phase, the orchestrator schedules the settling experiment. If that
  experiment costs more than 10 GPU-hours, the orchestrator asks the user first.
* If a member's claim is contradicted by a `[MEASURED]` result, the result wins, immediately, and the
  member is told. No re-litigation without a new measurement.
* Any member may declare a **stop-the-line**: a suspected correctness problem that invalidates
  results already recorded. The orchestrator halts new runs until the engineer's audit closes it.

---

## 6. Workspace and artifact contracts

### 6.1 Layout

```
experiments/convtm/
├── PLAN.md                  ← this document (the contract; superseding edits go in DECISIONS.md)
├── CHARTER.md               ← 1-page rules card every agent reads first (§1.2 + §5.2 + §7)
├── MEETINGS.md              ← running history (notetaker)
├── DECISIONS.md             ← append-only decision records
├── LITERATURE_TM.md         ← tm-theorist
├── LITERATURE_CNN.md        ← dl-expert
├── THEORY.md                ← tm-theorist
├── MECHANISMS.md            ← dl-expert
├── LIMITATIONS.md           ← output of P4, every entry backed by a measurement
├── CANDIDATES.md            ← output of P5, the hypothesis register
├── ARMS.md                  ← the arm registry: name → what it is → paper → command line
├── AUDIT.md                 ← research-engineer
├── LIBRARY_GAPS.md          ← proposed torchtsetlin changes, NOT APPLIED (constraint C1/C2)
├── RISKS.md                 ← live risk register
├── DIGESTS/round-<k>.md
├── rounds/ROUND-<k>/{brief,tm-theorist,dl-expert,research-engineer,crossfire,minutes}.md
├── tools/pdf_text.py        ← PDF → text/grep helper for the literature work
├── code/
│   ├── data.py              ← CIFAR-10 load + Booleanizations + the 45k/5k/10k splits
│   ├── arms.py              ← arm registry: name → builder(cfg) → model
│   ├── run_arm.py           ← single-arm driver, writes one results/*.json
│   ├── queue_runner.py      ← adapted from experiments/mctm/queue_runner.py
│   ├── screen.py            ← cheap screening harness (§8 P5)
│   ├── diagnostics.py       ← firing rates, clause length, negation fraction, invariance probes
│   ├── report_data.py       ← results/*.json → report/data/*.csv + report/macros.tex
│   └── cnn/                 ← dl-expert's baselines
├── results/*.json           ← one record per (arm, seed); the only source of numbers
├── screen/*.json            ← cheap-screen records, kept separate so they can never be mistaken
├── logs/*.log
├── .ckpt/                   ← reusable frozen layers / trained models
└── report/                  ← LaTeX (mirrors experiments/mctm/report2/ scaffolding)
```

Reuse `experiments/mctm/.cache/cifar10_therm{4,8}.pt` and `.data/` via symlink or env var — do not
re-encode or re-download.

### 6.2 Result record schema (`results/<arm>_seed<k>.json`)

Every field is mandatory. The engineer's aggregator rejects records missing any of them.

```jsonc
{
  "arm": "ctm-vanilla-8k",
  "family": "existing|baseline|candidate|control",
  "paper": "Granmo et al. 2019, arXiv:1905.09688",   // null for our own arms
  "seed": 0,
  "argv": ["run_arm.py", "--arm", "ctm-vanilla-8k", "--seed", "0", "..."],
  "git_sha": "fbc5306...",
  "env": {"torch": "2.10.0+cu128", "gpu": "NVIDIA GeForce RTX 3090", "host": "..."},
  "started": "2026-09-22T10:14:03", "wall_s": 4821.3,

  "data": {"dataset": "cifar10", "booleanization": "therm4",
           "n_train": 45000, "n_val": 5000, "n_test": 10000, "augment": null},

  "hp": {"n_clauses": 8000, "T": 6400, "s": 10.0, "patch": [10,10], "stride": 1,
         "position_encoding": true, "batch_size": 50, "feedback_mode": "batch",
         "epochs": 60, "max_included_literals": null},

  "capacity": {"n_automata": 19904000, "n_clauses_total": 8000,
               "literals_evaluated_per_image": 6732800, "state_bytes": 79616000},

  "curve": [{"epoch": 1, "train_acc": 0.31, "val_acc": 0.30, "epoch_s": 141.2}, "..."],
  "selected_epoch": 47,                       // chosen on val_acc, never test
  "val_acc": 0.612,
  "test_acc": 0.607,                          // evaluated ONCE, at selected_epoch
  "test_acc_per_class": [0.71, "..."],
  "test_predictions_sha": "…",                // for paired significance tests

  "diagnostics": {"clause_len": {"median": 41, "mean": 44.2, "p95": 88},
                  "negation_fraction": 0.58,
                  "firing_rate": {"median": 0.031, "dead_frac": 0.02, "sat_frac": 0.00},
                  "clauses_used_frac": 0.94},

  "notes": "faithful to §4 of the paper except: position encoding thermometer, not one-hot"
}
```

`test_predictions_sha` plus a saved prediction vector (`results/preds/<arm>_seed<k>.npy`) enables
paired McNemar / bootstrap tests between arms without re-running anything.

### 6.3 Arm registry (`ARMS.md`)

One row per arm: `name | family | what it is, in one sentence | source paper | exact command line |
expected cost (GPU-min × seeds) | status`. Nothing runs that is not in the registry first; that is
how the orchestrator budgets, and how the report's method table gets written.

---

## 7. Experimental protocol — what makes a result admissible

This section is the difference between a report and an anecdote. The engineer enforces it.

### 7.1 Data and splits

* CIFAR-10, official 50 000 / 10 000 split.
* **Fixed 45 000 train / 5 000 validation** carved from the official train set with a fixed seed
  (`split_seed=1234`, stratified). The same split for every arm, CNN and TM alike.
* **The test set is touched once per (arm, seed)**, at the epoch validation selected. Per-epoch test
  accuracy may be *recorded* for curves but may never influence selection, and the report's headline
  tables use `test_acc` at `selected_epoch` only.
  *(Note: the prior programme in `experiments/mctm` reported `best_test_acc`. Those numbers are
  quoted in the report as prior work with that caveat stated, and any arm reused as a baseline here
  is re-run under this protocol.)*
* Booleanization variants are part of the arm, not of the data: `therm4`, `therm8`,
  `adaptive-threshold`, `hog`, and whatever P1 turns up. Each is cached once.
* Sample-efficiency subsets: 1 000 / 5 000 / 10 000, stratified, fixed seed, nested.

### 7.2 Seeds and statistics

* **3 seeds** for every arm that enters a comparison; **5 seeds** for the finalist and for whatever
  it is claimed to beat.
* Report **mean ± sample standard deviation** over seeds, and the individual seeds in an appendix
  table. Never a single seed in a headline claim.
* **Seed noise is measured in P0**, not assumed: 5 seeds of the vanilla arm give the band below
  which a difference is not a difference.
* Headline comparisons use a **paired test on the shared test set**: McNemar on the per-example
  correctness of the two best-seed models, plus a bootstrap CI over the 10 000 test examples, plus
  the across-seed spread. All three are reported. A claim of superiority needs the across-seed
  intervals to be disjoint *and* the paired test to agree.

### 7.3 Budget matching (the rule that kills most TM-vs-TM comparisons)

Any "A beats B" claim must state the matching axis, and the report gives at least two:

| Axis | Definition |
|---|---|
| Clause budget | total clauses (for class-owned models: clauses × classes) |
| Automata budget | `n_clauses_total × 2 × n_features` — the real memory the model holds |
| Compute budget | literals evaluated per image at inference; and measured inference throughput |
| Wall-clock budget | training seconds to reach the reported accuracy on this machine |
| Data budget | training images seen |

An arm that wins on accuracy while using 10× the automata is reported as such, in a table with all
axes, and is not called "better" without qualification.

### 7.4 Leakage and silent-failure guards (run by `research-engineer` at every gate)

1. `git diff --stat -- src/` empty (C1).
2. No `test` tensor reachable from any selection path; grep + code read.
3. Split determinism: the split hash is recorded in every result record and must match across arms.
4. `model.eval()` present on every evaluation path (empty-clause semantics flip otherwise).
5. No `non_blocking=True` on any CPU destination.
6. Re-run one random result per phase from its `argv`; accuracy must land inside the P0 seed band.
7. Every queued job's return code logged; missing results investigated, never silently skipped.
8. Throughput sanity: an arm whose s/epoch is wildly off the §3.4 scaling is chunk-budget-suspect.

### 7.5 Reproduction tolerance for literature arms

An arm claiming to re-implement a published method is **reproduced** if it lands within **3
percentage points** of the paper's reported CIFAR-10 accuracy under the paper's stated configuration
(or within the paper's own reported spread, whichever is larger). Otherwise the arm is marked
`GAP` and the gap is investigated and documented in `ARMS.md` before the arm is used in any
comparison. A `GAP` arm may still be reported — as a reproduction failure, with the cause named.

### 7.6 What every run logs

Per epoch: train acc, val acc, epoch seconds, and (for TM arms) median clause length, median firing
rate, dead-clause fraction, negation fraction. These are the diagnostics P4 lives on; collecting them
after the fact means re-running everything.

---

## 8. Phase plan

Eight phases. Each has an owner, explicit deliverables, and an **exit gate** that is checked with
evidence before the next phase starts. The orchestrator reports to the user at every gate.

### P0 — Setup, calibration and harness (owner: research-engineer; ~1 day, ~3 GPU-h)

**Tasks**
1. Create the workspace (§6.1), `CHARTER.md`, `tools/pdf_text.py`, and the five agent files.
2. `code/data.py`: CIFAR-10 load, the fixed 45k/5k split, the Booleanization registry, subset
   builders, all cached. Reuse `experiments/mctm/.cache` and `.data`.
3. `code/run_arm.py` + `code/arms.py` + the results schema (§6.2) + `code/queue_runner.py`
   (adapted, keeping the per-GPU lock and the skip-if-exists behaviour).
4. **Throughput calibration**: measure s/epoch for a 3×3 grid of (clauses × patch size) at full
   scale, and the peak GPU memory for each, on both GPUs. This replaces every cost estimate in §10
   with a measurement and determines the largest arm that fits on the 10 GB card.
5. **Seed-noise measurement**: 5 seeds of `ctm-vanilla` (small config) → the band below which a
   difference is not a difference. Recorded in `macros.tex` and used everywhere after.
6. Smoke test: a 2-epoch run of every harness path, on a 2 000-image subset, to catch crashes before
   they cost hours.

**Exit gate G0** — harness runs end-to-end; a result record validates against the schema; the seed
band is measured and written down; the throughput table exists; `git diff -- src/` is empty.

### P1 — Literature review (owners: tm-theorist + dl-expert, concurrent; ~1 day, 0 GPU-h)

**Tasks**
* `tm-theorist` → `LITERATURE_TM.md`: every convolution-relevant TM approach in
  `source_documents/papers` and the book chapters. The table must carry: method, paper, **reported
  CIFAR-10 accuracy and the exact configuration behind it**, Booleanization, clause budget, what is
  mechanistically novel, whether `torchtsetlin` can express it today (and if not, what is missing —
  straight into `LIBRARY_GAPS.md`), and an estimated re-implementation cost.
* `dl-expert` → `LITERATURE_CNN.md` + `MECHANISMS.md` (§4.3).
* Both flag every claim with a tag; every accuracy number carries its source.

**ROUND-1** — question: *which approaches do we re-implement, in what order, and what does each one
cost?* Output `DR-001`: the ranked shortlist (minimum 6 arms), each with an owner, a command line
and a cost. §9.1 is the seed list; the round may add to it or drop from it, with reasons.

**Exit gate G1** — both literature files complete; every shortlisted arm has a paper, a reported
number to reproduce against, and a cost estimate; `DR-001` written; digest sent.

### P2 — CNN baselines (owner: dl-expert, audited by research-engineer; ~2 days, ~20 GPU-h)

Build and measure every baseline in §4.3, under the §7 protocol. The two that matter most for the
argument are `cnn-boolean` (how much does Booleanization cost?) and `cnn-binary` (what can binary
computation do at all?), because they convert "TMs are behind CNNs" into a decomposition:
*Booleanization loss + binary-computation loss + TM-mechanism loss*. That decomposition is the
report's spine.

**Exit gate G2** — `cnn-resnet18` ≥ 93% with augmentation; `cnn-small` ≥ 85% with augmentation;
every baseline has 3 seeds, a val-selected test number, params/MACs/throughput, and a
sample-efficiency curve; the engineer has audited for test-set tuning.

### P3 — Re-implement the existing TM convolution approaches (owner: research-engineer with tm-theorist; ~4–5 days, ~80–120 GPU-h)

Each shortlisted arm implemented in `code/arms.py`, registered in `ARMS.md`, run at 3 seeds under
§7, with a reproduction check (§7.5) against its paper.

**Ordering rule** — run cheapest-first and reproduce `ctm-vanilla` before anything else: it is the
trust anchor (T0). If T0 fails, everything stops and the engineer audits; no amount of downstream
work is worth anything if the base CTM cannot reproduce its published column (DR-001 Decision 1;
the original ~55–61% band is withdrawn).

**Also mandatory** — the matched-budget controls, because half the literature's comparisons are not
matched: for every multi-component arm, a single-layer arm at the same clause budget and at the same
receptive field.

**Exit gate G3** — T0 met **as re-baselined by DR-001** (the 8k pre-flight within 3 points of the
Toolbox's 8 000-clause column, and the capacity ladder reproducing its shape); every shortlisted arm either reproduced within
tolerance or marked `GAP` with a documented cause; the full results table exists; the engineer's
audit and one re-run verification are recorded.

### P4 — Limitation analysis (owners: all three; ~2 days, ~15 GPU-h)

Measurement, not opinion. Minimum battery, all from `code/diagnostics.py`:

1. **Capacity scaling**: accuracy vs clause count over ≥4 points per family, to the point of
   saturation. *Where does the curve flatten, and at what accuracy?* This single curve tells us
   whether TMs on CIFAR-10 are capacity-limited or mechanism-limited — the most important question in
   the programme.
2. **Booleanization loss**: `cnn-boolean` vs `cnn-small` (P2) isolates it; then TM accuracy across
   `therm4/therm8/adaptive/hog` isolates how much of it the TM recovers.
3. **Clause economy**: length, negation fraction, firing rate, dead/saturated fractions, fraction of
   clauses that ever contribute to a decision.
4. **Invariance probes**: accuracy under translation (±1–4 px), horizontal flip, colour jitter, and
   small noise — for TM arms and CNN baselines alike. CNNs get invariance from pooling and
   augmentation; what does a CTM actually have?
5. **Depth probe**: the P3 stacked arms plus §3.1's measured facts — is depth still blocked after the
   calibration repair, and where?
6. **Error structure**: per-class accuracy, confusion matrices, and the overlap between CNN errors
   and TM errors. If TMs fail on *the same* images, the gap is representational; if on different
   images, an ensemble is on the table and that is itself a candidate.
7. **Sample efficiency**: TM arms on the 1k/5k/10k subsets against the CNN curve from P2.

**ROUND-2** — question: *what, specifically and measurably, limits convolutional TMs on CIFAR-10?*
Output `LIMITATIONS.md` (ranked, each entry citing its measurement) and `DR-002`.

**Exit gate G4** — every entry in `LIMITATIONS.md` carries a `[MEASURED]` id; the capacity-scaling
curve exists for ≥3 families; `DR-002` written; digest sent.

### P5 — Candidate generation and cheap screening (owners: all three; ~2 days, ~20 GPU-h)

1. Each expert proposes candidates; the team must field **≥8**. Each entry in `CANDIDATES.md`:
   *mechanism · which ranked limitation it attacks · the theory of why it should work · falsifiable
   prediction · the screen that tests it · matched-budget control · estimated full cost.*
2. **Screening protocol** (this is what keeps the budget finite): 10 000 training images, 15 epochs,
   1 seed, small clause budget, validation-selected, written to `screen/` — never to `results/`.
   A candidate must beat the *screened vanilla arm* by more than the **screen noise band** (measured
   in P0 at the screen's scale, expected ≈ ±1.5 points) to be promoted.
3. Screening is explicitly a filter with false negatives. Two exemptions may be granted per round by
   the orchestrator for candidates whose mechanism cannot show up at 10k scale (e.g. anything whose
   benefit is capacity-driven), with the reason recorded in the DR.

**ROUND-3** — question: *which candidates earn full evaluation?* Output `DR-003`: promote **≤4**.

**Exit gate G5** — ≥8 candidates registered with predictions and falsifiers; all screened; `DR-003`
written with screen numbers; digest sent.

### P6 — Full evaluation of the new approach (owner: research-engineer, analysis by all; ~4–5 days, ~80–120 GPU-h)

1. Each promoted candidate: 3 seeds, full CIFAR-10, full protocol, **plus its matched-budget control
   and its ablations** — every component of a candidate is ablated, or the report cannot say which
   part did the work.
2. The best candidate then gets: 5 seeds, the paired significance tests (§7.2), the full capacity
   scaling curve, the diagnostics battery, the efficiency table (§7.3), interpretability extraction
   (≥5 human-readable rules), and the sample-efficiency curve.
3. A **combination arm** is permitted (the winning mechanism plus the best existing method), but it
   must be reported as a composite and compared against the composite state of the art (82.8%), not
   against single models.
4. **Transfer check**: the finalist on CIFAR-100 and Fashion-MNIST, 1 seed, purely to show the result
   is not CIFAR-10-specific. Not headline evidence.

**ROUND-4** — question: *what is the verdict, and what does it rest on?* Output `DR-004`, which must
name where the approach sits on the §2 ladder, and state plainly what was **not** achieved.

**Exit gate G6** — finalist at 5 seeds with significance tests; every ablation run; every claim in
the draft conclusion traceable to a result id; the engineer's final audit recorded; a negative
verdict is a valid pass of this gate.

### P7 — Report and library proposal (owners: all; ~2 days, ~2 GPU-h)

**The PDF** (§11): theory by `tm-theorist`, CNN comparison by `dl-expert`, method/reproduction/
protocol by `research-engineer`, narrative history and the programme log by `notetaker`, assembly and
the argument by the orchestrator. Built with the private TeX Live at
`/work/vajira/DL2026/.texlive/tl/bin/x86_64-linux` (there is no system LaTeX on this machine).

**`LIBRARY_GAPS.md`** becomes a concrete, ordered implementation plan for `torchtsetlin`: for each
gap — symptom, minimal reproduction, proposed API, affected files, test to add, migration/compat
note, and which experimental result motivates it. **Nothing is applied.** The user's manual
confirmation is the gate (C1).

**Exit gate G7** — `report/main.pdf` builds clean, `check.sh` passes, every number traces to a JSON
record, `LIBRARY_GAPS.md` is an actionable plan, `MEETINGS.md` reads as a coherent history, and the
`src/` diff is still empty.

---

## 9. Seed hypothesis space (a starting point, not a decision)

The team generates its own list in P1/P5. This section exists so that round 1 starts from the state
of the art rather than from a blank page. **Nothing here is chosen; everything here must be screened.**

### 9.1 Existing approaches to re-implement (P3 candidates)

| Arm | What it is | Source in `source_documents/papers` |
|---|---|---|
| `ctm-vanilla` | The Convolutional TM: clause matches if any patch matches; feedback from one random matching patch | `01_foundations/2019_The_Convolutional_Tsetlin_Machine` |
| `ctm-weighted` | Clause weights instead of unit votes | `01_foundations/2019_The_Weighted_Tsetlin_Machine` |
| `ctm-coalesced` | Shared clause pool, per-class weights (the library's `ConvCoalescedTsetlinMachine`) | `01_foundations/2021_Coalesced_Multi_Output_TMs` |
| `ctm-dropclause` | Drop-clause regularisation; reported to help on image tasks | `01_foundations/2021_Drop_Clause...`, `2023_Drop_Clause_AAAI` |
| `ctm-clausesize` | Explicit clause-size constraint for concise patterns | `01_foundations/2023_Building_Concise_Logical_Patterns` |
| `ctm-boolean-*` | Booleanization family: adaptive thresholding, colour thermometer, HOG, edges | `02_image_classification/2024_Optimized_Toolbox...` |
| `tm-composite` | Plug-and-play team of specialised TMs — the 82.8% SOTA | `02_image_classification/2023_TMComposites`, `2024_Optimized_Toolbox` |
| `ctm-multitask-rgb` | Multi-task convolutional TM over RGB | `02_image_classification/2025_Transparent_Logic_Based_Classification...` |
| `ctm-hypervector` | Hyperdimensional vectors for TMs | `02_image_classification/2024_Effects_of_Hyperdimensional_Vectors` |
| `ctm-distill` | Knowledge distillation into TMs (a CNN teacher is the obvious variant to test) | `02_image_classification/2025_Knowledge_Distillation_in_TMs` |
| `ctm-sparse` / `ctm-absorbing` | Sparse TM / contracting TM with absorbing automata — efficiency, possibly capacity | `01_foundations/2024_The_Sparse_TM`, `2023_Contracting_TM` |
| `ctm-graph-deep` | "The Tsetlin Machine Goes Deep" (graph formulation) — the other existing answer to depth | `01_foundations/2025_The_TM_Goes_Deep_Graphs` |
| `mctm-calibrated` | This repo's own best stack (random layer 1 + firing-rate calibration, 42.6%), re-run under the P7 protocol | `experiments/mctm/report2/` |

### 9.2 Mechanism gaps between CNNs and CTMs (input to P5)

Stated as questions with experiments attached, not as answers:

1. **Pooling is an OR.** A CTM collapses patches with a max (OR). A CNN has average pooling, strided
   convolution and learned downsampling. *Does a **counting** pool — clause fires if ≥ k patches
   match — carry strictly more information at the same clause budget?* Cheap to test; the vote
   becomes a threshold over counts rather than over indicators.
2. **Depth.** §3.1 says greedy supervised stacking fails and calibration partially repairs it.
   *After calibration, is depth still blocked, and by what — density, credit, or clause identity
   drift?*
3. **Normalisation.** Firing-rate calibration is structurally a BatchNorm analogue (it holds a unit's
   activation statistics in a workable band). *Is per-clause rate control, applied throughout
   training rather than once before freezing, the TM's missing normalisation?*
4. **Receptive field growth.** CNNs grow RF through depth cheaply; a CTM grows it by enlarging the
   patch, which grows the literal space quadratically. *Dilated / multi-scale / pyramid patches at
   matched automata budget?*
5. **Augmentation.** Standard for CNNs, largely absent from the TM image literature. *Random crop +
   flip on the Boolean tensor — how much of the CNN's augmentation gain survives Booleanization?*
   This is cheap and may be the single highest-value-per-GPU-hour experiment in the programme.
6. **Ensembling vs composites.** TM Composites already show ensembling works. *Where does the
   error-overlap measurement (P4.6) say the remaining headroom is?*
7. **Distillation.** *Can a CNN teacher's soft targets or its features supervise a CTM without
   destroying interpretability?*
8. **Colour.** CNNs learn cross-channel filters; thermometer Booleanization treats channels
   independently. *Do joint colour literals (learned or fixed colour-quantised planes) help?*
9. **Clause capacity vs. clause quality.** If P4.1's scaling curve is still climbing at 8k clauses,
   the honest next move is more clauses, not a new mechanism — and that must be reported as such.

---

## 10. Compute budget and scheduling

### 10.1 Estimated budget (replaced by P0 measurements)

| Phase | GPU-hours | Notes |
|---|---|---|
| P0 | 3 | calibration + seed band + smoke |
| P1 | 0 | reading |
| P2 | 20 | CNN baselines, 3 seeds each, incl. ResNet-18 at 200 epochs |
| P3 | 80–120 | ~12 arms × 3 seeds; large-clause arms dominate |
| P4 | 15 | diagnostics + scaling curves |
| P5 | 20 | ~10 candidates × 1 screen |
| P6 | 80–120 | 4 candidates × 3 seeds + ablations + finalist at 5 seeds |
| P7 | 2 | figures, final checks |
| **Total** | **220–300 GPU-h** | ≈ 5–7 days wall-clock across the two GPUs |

Scaling anchor from §3.4: 640 clauses at 4×4 → 4.6 s/epoch; at 9×9 → 11.2 s/epoch. A
literature-scale arm (8 000 clauses, 10×10 patches, 60 epochs) is therefore expected in the
**1.5–3 GPU-hour** range per seed. P0 measures this before P3 commits.

### 10.2 Scheduling rules

* **One queue per GPU**, enforced by the lock in `queue_runner.py`. Two queues on the 10 GB card OOM
  each other, and because a crashed run writes no result file the queue then burns its whole list in
  seconds leaving only tracebacks — this already happened in the previous programme.
* RTX 3090 (24 GB) takes the large-clause and large-patch arms; RTX 3080 (10 GB) takes screens,
  diagnostics, small arms and the CNN baselines. P0's memory table assigns arms definitively.
* `PYTHONUNBUFFERED=1` on every runner (logs are teed; without it a suite looks frozen for minutes).
* Jobs are **resumable and idempotent**: skip if the result file exists. Job lists are read once at
  start, so a list can be edited while a queue runs.
* The orchestrator launches queues with `run_in_background: true` and checks on them between agent
  turns; agents never hold a turn open waiting for a GPU.
* **Checkpoint the expensive intermediates** (`.ckpt/`): frozen feature layers, Booleanization
  caches, autoencoder layers. P3/P5/P6 reuse them heavily.

### 10.3 Token/context budget

* 4 mandatory rounds × (3 position papers + 3 crossfire responses + 1 notetaker pass) ≈ 28 agent
  turns, plus P2/P3/P5/P6 implementation turns. Budget **~60–90 agent invocations** for the
  programme.
* Agents read *results files and their own notes*, not raw logs. The orchestrator hands them
  aggregated tables. A 75 000-line log in an agent's context buys nothing.
* `notetaker` on sonnet; the three specialists on opus.

---

## 11. The report

`experiments/convtm/report/`, mirroring the proven scaffolding of `experiments/mctm/report2/`:
`main.tex`, `ttwhitepaper.sty`, `macros.tex` (generated), `sections/*.tex`, `data/*.csv` (generated),
`Makefile`, `check.sh`.

**Build**: `PATH=/work/vajira/DL2026/.texlive/tl/bin/x86_64-linux:$PATH make data && make`.
`check.sh` must pass — it fails the build on undefined control sequences, undefined references,
**duplicate `\newcommand` definitions** (pdflatex silently keeps the first, so a stale number reaches
the page with no visible sign), and missing data files (a table whose CSV is absent is dropped with
only a buried warning).

### Structure

| § | Content | Author |
|---|---|---|
| Abstract | The question, the method, the headline number, the honest verdict | orchestrator |
| 1 | The question: why convolution is the bottleneck for TMs on images | tm-theorist |
| 2 | Background: TM and CTM mechanics, formally — clause, feedback, the patch OR, what the vote is | tm-theorist |
| 3 | What CNNs do and why: the mechanism inventory, and the CNN baselines we trained here | dl-expert |
| 4 | Related work: every existing TM convolution approach, with its reported number and its configuration | tm-theorist |
| 5 | Experimental protocol: splits, selection, seeds, budget matching, statistics — §7 in prose | research-engineer |
| 6 | Reproduction of existing methods: the table, the reproductions, and the gaps we could not close | research-engineer |
| 7 | **Where the accuracy goes**: Booleanization loss / binary-computation loss / TM-mechanism loss, from the `cnn-boolean` and `cnn-binary` controls; plus the capacity scaling curves | dl-expert + tm-theorist |
| 8 | Measured limitations of current convolutional TMs (P4) | all |
| 9 | **The new approach**: theory first — what it represents, why it should work, what it predicts — then the implementation | tm-theorist + research-engineer |
| 10 | Results: main table, ablations, matched-budget tables on all axes, significance tests, scaling, sample efficiency, interpretability examples | research-engineer |
| 11 | **What did not work, and why** — mandatory, and written with the same care as §10 | all |
| 12 | Discussion: where this sits against CNNs, what closes the remaining gap, what it costs | orchestrator |
| 13 | Proposed `torchtsetlin` changes (summary of `LIBRARY_GAPS.md`) — explicitly not applied | research-engineer |
| 14 | Reproducibility: every command line, the arm registry, the environment, the git sha | research-engineer |
| App. A | Per-seed tables | generated |
| App. B | Programme log: the rounds, the decisions, the dissent | notetaker |

**Figures** (all generated by `code/report_data.py`, light + dark pairs where the docs need them):
accuracy vs clause budget per family; the accuracy-decomposition bar (CNN → boolean-CNN → binary-CNN
→ best TM → ours); reproduction scatter (ours vs published); ablation waterfall for the new approach;
sample-efficiency curves TM vs CNN; clause-length / firing-rate distributions before and after;
invariance probe bars; a worked interpretability example with real extracted clauses.

---

## 12. Library policy

**Until the user says otherwise: `src/torchtsetlin/**` is read-only.** (C1)

Everything the programme needs that the library lacks is either (a) implemented in
`experiments/convtm/code/` as experiment-local code, or (b) recorded in `LIBRARY_GAPS.md`. Where an
arm needs behaviour the library does not offer and it is implemented locally by subclassing or
wrapping, that fact is stated in the arm's registry entry **and in the report**, because it changes
what "the library achieves X" would mean.

`LIBRARY_GAPS.md` entry format:

```markdown
## LG-003 — Clause-size budget does not bind for conv models under batched feedback
- **Symptom**: `max_included_literals=8` yields median clause size 75 at batch 50.
- **Reproduction**: `python code/repro/lg003.py` → prints sizes at batch 5 / 50 / sequential.
- **Cause**: `functional.apply_feedback` gates only Type Ia; Type II runs `inc2 = min(n2, room)`,
  and a conv model contributes one Type II opportunity per patch (841 for 4x4 on 32x32).
- **Impact here**: [MEASURED: <id>] clause-size control was unavailable for every arm in P3; we used
  specificity `s` and an explicit controller instead.
- **Proposed fix (NOT APPLIED)**: gate Type II by the same `room` computation as Type Ia, behind a
  flag defaulting to current behaviour; alternatively expose a post-commit size projection.
- **Affected**: `src/torchtsetlin/functional.py`, `models/base.py`.
- **Test to add**: conv model, batch 50, budget 8 → median size ≤ 8.
- **Risk**: changes learning dynamics of every existing conv result; needs the benchmark suite re-run.
```

Only after the user's explicit confirmation does any of this become a code change — and then as a
separate, reviewed piece of work, not as part of this programme.

---

## 13. Risks and stop conditions

| # | Risk | Likelihood | Mitigation / stop condition |
|---|---|---|---|
| R1 | Vanilla CTM does not reproduce ~60% (T0 fails) | Medium | **Hard stop at G3.** Engineer audits Booleanization, patch size, clause budget, T, s, epochs against the paper. The 60.7% figure may itself rest on a configuration we cannot afford; if so, that is finding #1 and the ladder is re-baselined with the user. |
| R2 | The programme finds nothing that beats 60.7% | Medium-high | Acceptable outcome. The report becomes a rigorous negative with a measured decomposition of where the accuracy goes — which the literature does not currently have. §11 carries it. |
| R3 | Compute overruns the budget | Medium | P0 measures true costs; P5 screening filters before full runs; the orchestrator asks the user before any single experiment over 10 GPU-h. |
| R4 | An agent decides a method by argument | Medium | §5.2 tags + §5.3 evidence requirement; notetaker explicitly flags decisions resting on `[HYPOTHESIS]`; orchestrator rejects the DR. |
| R5 | Silent test-set leakage | Medium | §7.1 + §7.4, and the deliberate break from `experiments/mctm`'s `best_test_acc` convention. |
| R6 | An arm is a strawman (unfair budget or untuned) | High — this is *the* endemic failure of TM-vs-TM comparisons | §7.3 matched budgets on all axes; each arm gets a small documented tuning allowance on validation; `tm-theorist` reviews every arm's configuration against its paper before it is run. |
| R7 | Someone edits `src/` | Low | C1 guard at every gate and commit; hard stop. |
| R8 | GPU OOM / queue collapse | Medium | Per-GPU lock; P0 memory table; crashed jobs write no result and are visibly missing; return codes logged. |
| R9 | Results accumulate faster than they are understood | Medium | Rounds are mandatory synchronisation points; nothing is promoted between rounds. |
| R10 | The report drifts from the results | Low | C4: every number generated from JSON; `check.sh`; engineer audits numbers↔code at G7. |

**Programme-level stop conditions** (orchestrator consults the user):
* T0 fails after one full audit cycle.
* Two consecutive rounds produce no promotable candidate.
* Cumulative GPU time passes 300 hours without reaching T2.

---

## 14. Kickoff

### 14.1 Gate 0 — the user's go/no-go

Before anything runs, the user confirms (or amends):

1. **The target ladder in §2** — in particular whether T2 (60.7%, single-model TM SOTA) is the
   primary objective, with T3/T4 as stretch.
2. **The compute budget** — 220–300 GPU-hours over roughly a week on both GPUs.
3. **The shortlist size in P3** — the plan assumes ~12 arms; fewer arms means more depth per arm.
4. **The break from `best_test_acc`** — this programme selects on validation, so its numbers are not
   directly comparable to `experiments/mctm`'s prior reports, and prior arms reused as baselines get
   re-run. This is the right call but it costs some compute.
5. **Autonomy level** — does the orchestrator proceed through gates G0→G7 reporting at each, or stop
   for approval at each gate? Default assumed: **proceed, report at every gate, stop for approval at
   G4 (limitations) and before P6's large runs.**

### 14.2 Step 0 — materialise the team

```bash
mkdir -p /work/vajira/DL2026/torchtsetlin/.claude/agents
# write the five files from §4.3
mkdir -p experiments/convtm/{code/cnn,code/repro,results/preds,screen,logs,.ckpt,rounds,DIGESTS,tools,report/sections,report/data}
cp experiments/mctm/report2/{ttwhitepaper.sty,Makefile,check.sh,audit_macros.py,check_spacing.py} experiments/convtm/report/
cp experiments/mctm/queue_runner.py experiments/convtm/code/
ln -s ../../mctm/.cache experiments/convtm/.cache   # reuse the Boolean CIFAR caches
ln -s ../../mctm/.data  experiments/convtm/.data
git -C /work/vajira/DL2026/torchtsetlin diff --stat -- src/    # must be empty, now and at every gate
```

Then write `CHARTER.md` (the one-page rules card: C1–C6, the claim tags, the protocol in §7) and
`tools/pdf_text.py`, and `ARMS.md`, `DECISIONS.md`, `MEETINGS.md`, `RISKS.md`, `LIBRARY_GAPS.md` as
empty scaffolds with their headers.

### 14.3 Step 1 — P0, then ROUND-1

```
1. Agent(research-engineer): build the harness, the data pipeline, the schema, the runner.
   → G0 including the throughput table and the measured seed band.
2. Agent(tm-theorist) and Agent(dl-expert) IN ONE MESSAGE: P1 literature review.
3. Orchestrator: compile rounds/ROUND-1/crossfire.md.
4. SendMessage to each of the three with the crossfire; collect responses.
5. Agent(notetaker): minutes + MEETINGS.md + DIGESTS/round-1.md.
6. Orchestrator: write DR-001; report the digest to the user; launch the P2/P3 queues.
```

From there the phase plan in §8 runs in order, with the round → decision → queue cycle repeating at
P4 (ROUND-2), P5 (ROUND-3) and P6 (ROUND-4).

### 14.4 Definition of done

* `experiments/convtm/report/main.pdf` — complete, building clean, every number generated.
* `DECISIONS.md` — every method choice traceable to a `[MEASURED]` result.
* `MEETINGS.md` + `DIGESTS/` — a readable history of how the conclusion was reached.
* `LIBRARY_GAPS.md` — an actionable, unapplied implementation plan for `torchtsetlin`.
* `git diff --stat -- src/` — empty.
* A one-page summary to the user stating plainly where the new approach landed on the §2 ladder,
  including if the answer is "not far enough".
