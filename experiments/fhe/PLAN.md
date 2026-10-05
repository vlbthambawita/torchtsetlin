# Encrypted Inference for Tsetlin Machines with Bit-wise FHE: full research plan

**Version:** 2 · **Written:** 2026-10-05 · **Author of the study:** Vajira Thambawita (Simula)
**Purpose:** a self-contained specification for running the whole study **in a fresh GitHub
repository**, from an empty directory to an IEEE-format paper. Nothing outside this file is needed,
except the public packages and papers it names.

Conventions used throughout:
- `[VERIFIED 2026-10-05]` — executed while this plan was written (Ryzen 9 3950X, 32 threads,
  Python 3.10, `torchtsetlin` 0.3.0, `concrete-python` 2.11.0, `openfhe` 1.5.1.0.22.4).
- `[CHECK]` — an API or fact that must be confirmed in Phase 0 before relying on it.
- `[HYPOTHESIS]` — a claim the experiments must confirm or refute.

---

## 0. Instructions to whoever executes this plan (human or agent)

1. **Work in phases, in order, and stop at every gate (G0–G4).** Write the gate outcome into
   `FINDINGS.md` before starting the next phase.
2. **No number is typed by hand into the paper.** Every table and figure is generated from
   `results/*.json` by a script in `paper/scripts/`.
3. **Every result record carries provenance** (§10). A record without `git_sha`, `argv`, `seed`
   and `env` is not admissible.
4. **Tag every claim** in `FINDINGS.md` and in paper drafts: `[MEASURED: <record-id>]`,
   `[FACT: <citation>]` or `[HYPOTHESIS]`. Only `[MEASURED]` can justify a design choice.
5. **Negative results are results.** If a scheme cannot run a formulation, or the clause-budget
   hypothesis fails, report it with the same care as a success.
6. **Never modify `torchtsetlin`.** It is a pinned dependency. Library problems go into
   `LIBRARY_GAPS.md` (symptom, minimal repro, impact, suggested fix) and are reported upstream to
   https://github.com/vlbthambawita/torchtsetlin/issues.
7. **Selection on validation, test once.** Hyperparameters and "best model" choices use a fixed
   validation split. The test set is evaluated once per (arm, seed).
8. **Ask before installing system-level software** (CUDA toolkit, compilers). User-space installs
   (`rustup`, conda/pip) are fine.

---

## 1. Background (enough to write the introduction and related work)

### 1.1 Tsetlin machines (TMs)
A TM classifier learns **conjunctive clauses** over Boolean literals (each input bit and its
negation). Clause *j* fires iff every literal it includes is 1. Each class owns clauses of positive
and negative polarity, and the class score is the (optionally weighted) signed count of firing
clauses, clamped to [−T, T]. The prediction is the argmax. Learning uses Tsetlin automata with
Type I/II feedback and needs no gradients. Variants used here:
- **Vanilla multi-class TM** (Granmo 2018, arXiv 1804.01508).
- **Weighted TM**: integer clause weights (arXiv 1911.12607).
- **Coalesced TM (CoTM)**: one shared clause pool, and each clause has a signed integer weight per
  class (arXiv 2108.07594). Under FHE this means **each clause is evaluated once for all classes**.
- **Clause-size constraint**: a hard cap *b* on literals per clause (Abeyrathna et al. 2023,
  arXiv 2301.08190). In `torchtsetlin` ≥ 0.3.0 the argument is `max_included_literals`, and it is a
  **hard cap**: no clause ends an update with more than *b* literals.
- **Convolutional TM** (arXiv 1905.09688): a clause is an OR over image patches of an AND over patch literals.

### 1.2 Bit-wise FHE
Schemes from the "bit-wise / lookup-table FHE" family (catalogued in PREMAL_Bench,
https://vajira.info/PREMAL_Bench/index.html, `data/schemes.json` ids 20–25; it accompanies
*SoK: Private LLM Inference using Approximate HE*, Al Badawi, Alexandru, Polyakov,
Vaikuntanathan, 2026, https://eprint.iacr.org/2026/935):

| id | Scheme | Plaintext | Key property for TMs | Library |
|---|---|---|---|---|
| 20 | GSW (CRYPTO 2013) | bits | basis of blind rotation | — (inside FHEW/TFHE) |
| 21 | FHEW / DM (EUROCRYPT 2015) | bits, small ints via LUT | fast gate bootstrapping | OpenFHE BinFHE (`AP`, `GINX`) |
| 22 | TFHE / CGGI (ASIACRYPT 2016, JoC 2020) | bits, small ints via LUT | **programmable bootstrapping (PBS)**: any LUT on a small message, noise refreshed | TFHE-rs, Concrete, OpenFHE |
| 23 | FHEW with small keys, LMKCDEY (EUROCRYPT 2023) | bits | small evaluation keys | OpenFHE BinFHE (`LMKCDEY`) |
| 24 | FINAL, NTRU+LWE (ASIACRYPT 2022) | bits | NTRU gate bootstrapping | github.com/KULeuven-COSIC/FINAL (C++) |
| 25 | SIMD-ALU FHE, O(1) amortised bootstrapping (CRYPTO 2026) | machine words | SIMD for word logic | no public library → excluded |

Bit-level CKKS variants (ids 10 BLEACH, 11 Bit-CKKS, 13 CKKS functional bootstrapping) are
stretch/excluded arms (§5).

In all of these, **adding ciphertexts and multiplying by a plaintext integer are cheap** (no
bootstrap). **Every non-linear step costs one bootstrap**, which dominates latency.

### 1.3 State of the art (as of 2026-10-05)
- **No prior work runs a TM under homomorphic encryption.** Search: arXiv, OpenAlex, all ISTM
  proceedings titles in Crossref, IACR ePrint ("Tsetlin" → 0 hits), GitHub. TM privacy work is
  federated learning only: FedTM (ISTM 2023, doi 10.1109/ISTM58889.2023.10454982), TPFL (arXiv
  2409.10392), decentralised TM ensembles (arXiv 2607.20124), none with encryption.
- **Closest baseline: EI-DDLGN** (arXiv 2609.13636, Sep 2026). Differentiable logic-gate networks
  under TFHE-rs, MNIST thresholded at 0.5, i9-10900 (10 cores):

  | Model | Acc. | s/image |
  |---|---|---|
  | EI-DDLGN Small (2×4K gates) | 91.55 % | 6.58 |
  | EI-DDLGN Medium (4×6K) | 95.87 % | 19.33 |
  | EI-DDLGN Large (6×8K) | 97.20 % | 34.14 |
  | QAT-FCNN-4 (Concrete-style arithmetic baseline) | 91.54 % | 88.24 |

- Other Boolean/LUT-native FHE ML: TT-TFHE (truth-table nets, arXiv 2302.01584), tree models in
  Concrete ML (arXiv 2303.01254), Homomorphic WiSARDs (weightless networks *trained* under TFHE,
  arXiv 2403.20190), TAPAS (binary networks, arXiv 1806.03461), FHE-DiNN (ePrint 2017/1114),
  XONN (garbled circuits, arXiv 1902.07342), and TsetlinWiSARD (Tsetlin automata training WiSARD
  LUTs, arXiv 2603.24186).

### 1.4 The core idea `[HYPOTHESIS]`
Under a **public model and an encrypted input**, the server knows which literals each clause
includes. A clause with *k* included literals is

    clause = PBS( Σ_{i∈clause} lit_i  ==  k )      — one bootstrap, if k fits the message space

The sum is free. Class sums are plaintext-weighted sums of clause bits, which are also free. So
**TM inference costs about one PBS per clause, at circuit depth 1**, against one PBS per gate over
several layers for logic-gate networks. The **message precision of that PBS is set by the largest
clause**, and the clause budget *b* bounds it. A pilot measurement supports this:

**Pilot `[VERIFIED 2026-10-05]`** (Concrete 2.11, random TM-shaped circuits, F = 784 inputs,
10 classes, 32 threads, median of 3 runs, all outputs bit-exact):

| clauses | max clause size | PBS | latency / sample |
|---|---|---|---|
| 200 | 15 | 200 | 0.35 s |
| 1000 | 3 | 1000 | 1.65 s |
| 1000 | 8 | 1000 | 1.80 s |
| 1000 | 15 | 1000 | 1.76 s |
| 1000 | 31 | 1000 | 2.37 s |
| 1000 | 63 | 1000 | 7.22 s |

That is roughly 1.7 ms per PBS amortised while the largest clause stays ≤ 15 (4-bit), then the
cost grows steeply with precision. Re-measure these in Phase 0; they are pilot numbers only.

---

## 2. Research questions

- **RQ1 — Exactness.** Does encrypted inference reproduce `model.predict` bit-exactly for each
  scheme and formulation? What is the observed bootstrap failure rate?
- **RQ2 — Cost.** What are the latency per sample (1 thread and all threads; GPU for TFHE-rs),
  key sizes, ciphertext sizes and client↔server bytes for each scheme?
- **RQ3 — Structure → cost.** How do clause count, the clause budget *b*, weighting and the
  vanilla-vs-coalesced choice trade accuracy against bootstraps and latency? Is *b* the knob that
  sets PBS precision?
- **RQ4 — Comparison.** Where does an encrypted TM sit on MNIST's accuracy–latency plane against
  EI-DDLGN (reported, and one point reproduced on our hardware if feasible)?
- **RQ5 — Convolution.** Up to what patch × clause size is an encrypted convolutional TM feasible?

Out of scope: encrypted training, encrypted model (except optional Phase 4), federated secure aggregation.

---

## 3. Threat model and output modes

**TM-1 (main).** The server holds a plaintext TM. The client Booleanises its input locally (the
encoder parameters are public), encrypts the F input bits and sends them. The server evaluates the
circuit and returns ciphertexts, which the client decrypts. The goal is confidentiality of the client
input against an honest-but-curious server, at ≥128-bit security.

**Output modes.** Each mode is reported separately because each reveals something different to the client:

| Mode | Server returns | Client learns | Extra server cost |
|---|---|---|---|
| O1 | C clause bits | which clauses fired (model structure) | none |
| O2 | K class sums | per-class scores | additions only |
| O3 | argmax label | label only | clamp (K PBS) + argmax tournament (≈K−1 comparisons, multi-PBS) |

**Clamp semantics.** `torchtsetlin` predicts with `argmax(clamp(votes, −T, T))` and the first index
wins ties. O3 must implement the clamp. O2 decrypts the raw sums and the client applies clamp and
argmax, which is then exact. Verified: the simulator in §8 reproduces `predict` exactly with the
clamp `[VERIFIED]`. Report how often the clamp changes the argmax.

**TM-2 (optional, Phase 4).** The model is also encrypted, so each clause becomes an AND over all 2F
of (¬inc ∨ lit) and costs O(C·F) bootstraps. Report it, and state that it sacrifices
interpretability for the server.

---

## 4. Circuit formulations

All formulations consume one **IR** (§8.2), exported from a trained model in `eval()` semantics.
Empty clauses and contradictory clauses (containing both x and ¬x) are always False in prediction,
so they are dropped. Clauses with all-zero weights are dropped too. NOT is free in every scheme
(negate the LWE ciphertext), so the client encrypts only F bits.

- **F1 — gate tree.** AND over the k literals as a balanced binary tree → k−1 bootstrapped gates,
  depth ⌈log₂k⌉. Works on every scheme with gate bootstrapping (FHEW, LMKCDEY, TFHE, FINAL).
- **F2 — threshold PBS.** clause = PBS(Σ lits == k). This needs a scheme with addition plus a PBS
  on ≥⌈log₂(k+1)⌉-bit messages (TFHE-rs shortint/integer, Concrete, OpenFHE C++). For k > m (the
  largest supported sum), split into ⌈k/m⌉ groups, PBS each group, then one combining PBS.
- **Aggregation.** O2 computes `votes = clause_bits @ W` (plaintext integer W), which is free.
  Allow enough message bits for Σ|W| (radix/integer types in TFHE-rs; automatic in Concrete).
- **Convolution (Phase 3).** A per-patch clause uses F2 at each patch, then OR over P patches = PBS(Σ ≥ 1)
  (or a tree when P exceeds the message space). Cost ≈ P·C + C·⌈P/m⌉ PBS.

**Bootstrap-count model** (the simulator outputs it, and it is validated against measured latency):
F1 = Σⱼ(kⱼ−1); F2 = Σⱼ (1 if kⱼ ≤ m else ⌈kⱼ/m⌉+1); O3 adds the clamp and the comparisons.

---

## 5. Arms (scheme × backend × formulation)

| Arm | Scheme | Backend | Lang | F1 | F2 | Outputs | Priority |
|---|---|---|---|---|---|---|---|
| A1 | TFHE/CGGI | **Concrete** `concrete-python==2.11.0` | Python | — | ✓ `[VERIFIED]` | O1, O2 | core |
| A2 | TFHE/CGGI | **TFHE-rs** (`tfhe` crate, pinned) | Rust | ✓ (`boolean`) | ✓ (`shortint`) | O1, O2, O3 | core |
| A3 | TFHE/CGGI | TFHE-rs **CUDA** (RTX 3090) | Rust | — | ✓ | O1, O2 | core if `gpu` feature builds |
| A4 | FHEW (GINX) | **OpenFHE** `openfhe==1.5.1.0.22.4` | Python | ✓ `[VERIFIED]` | ✗ in Python* | O1 | core |
| A5 | FHEW (AP) | OpenFHE | Python | ✓ `[VERIFIED]` | ✗* | O1 | core |
| A6 | LMKCDEY | OpenFHE | Python | ✓ `[VERIFIED]` | ✗* | O1 | core |
| A7 | FINAL | KULeuven-COSIC/FINAL | C++ | ✓ | ✗ | O1 | secondary |
| A8 | CKKS functional bootstrapping | OpenFHE C++ | C++ | — | ✓ batched over slots | O2 | stretch |
| — | Bit-CKKS, BLEACH, SIMD-ALU | no public library | — | — | — | — | excluded (state why) |

\* `[VERIFIED]`: the OpenFHE 1.5.1 Python bindings expose only 2-input gates (`OR, AND, NOR, NAND,
XOR(_FAST), XNOR(_FAST)`), `EvalNOT`, `EvalFunc`/`GenerateLUTviaFunction`, `EvalSign`,
`EvalFloor` and `EvalDecomp`, but **no ciphertext addition**, so F2 is impossible from Python. The
LUT plaintext space with `GenerateBinFHEContext(STD128, True, 12, 0, GINX, False)` is 8. An F2
OpenFHE arm, if wanted, is C++ (`[CHECK]` C++ `EvalBinGate` with vector inputs / `AND3`/`AND4`,
and LWE addition).

Security: use each library's **default 128-bit parameter set** (OpenFHE `STD128`,
`STD128_LMKCDEY`; Concrete defaults; TFHE-rs default 128-bit parameters for the chosen message/carry
size). Record the parameter names, and estimate security with the lattice estimator
(https://github.com/malb/lattice-estimator) for every parameter set used.

---

## 6. Repository setup (fresh repo)

### 6.1 Layout
```
tm-fhe/
  README.md            how to reproduce everything (one command per phase)
  PLAN.md              this file
  FINDINGS.md          gate outcomes + claim-tagged findings (append-only log)
  LIBRARY_GAPS.md      torchtsetlin / backend problems found
  environment.yml      conda env "tmfhe"
  pyproject.toml       package "tmfhe" (src layout), ruff + pytest config
  src/tmfhe/
    __init__.py
    provenance.py      record() helper (§10)
    data.py            dataset loaders + Booleanisation (via torchtsetlin.data)
    train.py           train/select/save TMs (torchtsetlin)
    ir.py              export_circuit()            (§8.2, verified code)
    simulate.py        plaintext circuit + bootstrap counts (§8.3, verified code)
    backends/
      base.py          Backend protocol (§8.4)
      concrete_be.py   A1   (§8.5, verified pattern)
      openfhe_be.py    A4–A6 (§8.6, verified API)
      tfhers.py        thin wrapper calling the Rust CLI (A2, A3)
  rust/tm_tfhe/        Cargo crate: CLI "tm-tfhe" (§8.7)
  cpp/final/           FINAL wrapper (A7, optional)
  scripts/
    p0_microbench.py  p0_ir_check.py
    p1_train_small.py p1_encrypted.py
    p2_train_mnist.py p2_encrypted.py
    p3_conv_sim.py    p3_conv_encrypted.py
  results/             one JSON per run (committed)
  models/              checkpoints + IR (git-ignored; IR JSON committed for finalists)
  tests/               pytest: IR == predict, backend round-trips on toy circuits
  paper/
    main.tex  refs.bib  sections/*.tex  figures/  tables/
    scripts/make_figures.py  make_tables.py
    Makefile
  .github/workflows/ci.yml   ruff + pytest (CPU, toy sizes) + LaTeX build
```

### 6.2 Environment
```bash
conda create -n tmfhe python=3.10 -y && conda activate tmfhe
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build for faster training
pip install "torchtsetlin[vision,sklearn,viz]==0.3.0"
pip install "concrete-python==2.11.0" "setuptools<81"   # concrete imports pkg_resources [VERIFIED]
pip install "openfhe==1.5.1.0.22.4"
pip install numpy pandas matplotlib scikit-learn pytest ruff
# Rust for TFHE-rs (user space)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```
- `concrete-python` without `setuptools<81` fails with `ModuleNotFoundError: pkg_resources`
  `[VERIFIED]`.
- GPU Concrete is an optional extra wheel index (`[CHECK]` https://docs.zama.ai/concrete for the
  current GPU install line). It is not required.
- TFHE-rs: pin the crate version in `Cargo.toml` (latest 1.x at start). `[CHECK]` feature flags on
  https://docs.rs/tfhe (`boolean`, `shortint`, `integer`, and `gpu` for CUDA, which needs the CUDA
  toolkit + `nvcc`; ask before installing).
- FINAL: `git clone https://github.com/KULeuven-COSIC/FINAL`. Its README lists the build
  dependencies (`[CHECK]`, likely NTL/GMP/FFTW). Skip A7 if the build costs more than half a day,
  and record that.
- Record `pip freeze`, `cargo tree -d`, `lscpu`, `nvidia-smi` into `results/env_<date>.txt`.

### 6.3 Thread control
Latency is reported **twice**: single-thread and all-threads. Set `OMP_NUM_THREADS`,
`RAYON_NUM_THREADS` (TFHE-rs) and `taskset -c 0` for the single-thread runs. Concrete uses its own
parallelism, so `[CHECK]` the configuration flags (`fhe.Configuration(...)`, e.g. dataflow / loop
parallelism) and record them.

---

## 7. Datasets, Booleanisation and models (with `torchtsetlin` 0.3.0)

| Dataset | Source | Booleanisation | F (bits) | Why |
|---|---|---|---|---|
| Noisy XOR | `tt.data.make_noisy_xor(n, n_features=12, noise=0.4, seed=…)` (train), `noise=0.0` (test) | already Boolean | 12 | sanity check, TM paper benchmark |
| Breast Cancer Wisconsin | `sklearn.datasets.load_breast_cancer` | `tt.data.ThermometerEncoder(n_bits=4).fit(x_train)` | 120 | tabular medical; used by Homomorphic WiSARDs |
| UCI Phishing Websites | `sklearn.datasets.fetch_openml("PhishingWebsites", version=1)` `[CHECK]` (30 ternary features {−1,0,1}) | map to {0,1,2} → `tt.data.OneHotEncoder()` | 90 | identical encoding to EI-DDLGN (30 → 90 bits) |
| MNIST | `tt.data.load_mnist_boolean(root, train=…, threshold=0.5, flatten=True)` | pixel > 0.5 (`[CHECK]` pixels are scaled to [0,1] before thresholding) | 784 | same input as EI-DDLGN |
| 2D Noisy XOR (conv) | `tt.data.make_2d_noisy_xor(...)` | Boolean images | — | conv feasibility (Phase 3) |

Splits: a stratified 80/20 train/validation split of the official training data with
`split_seed=1234` (MNIST: 55 000 / 5 000), plus the official test set (Breast Cancer and Phishing:
stratified 60/20/20 train/val/test, `split_seed=1234`). Keep split indices in `results/splits/*.json`.

**Training (torchtsetlin API `[VERIFIED]`):**
```python
import torch, torchtsetlin as tt

tt.seed_everything(seed)
model = tt.TsetlinMachine(n_features=F, n_classes=K, n_clauses=n_clauses, T=T, s=s,
                          weighted=weighted, max_included_literals=b)    # b=None: no cap
# or: tt.CoalescedTsetlinMachine(n_features=F, n_outputs=K, n_clauses=n_clauses, T=T, s=s,
#                                max_included_literals=b)
trainer = tt.Trainer(model, device="cuda" if torch.cuda.is_available() else "cpu",
                     batch_size=32, verbose=False)     # keep batch 10–50: batched feedback
hist = trainer.fit((x_tr, y_tr), epochs=E, val_data=(x_va, y_va))  # degrades at large batch sizes
model.eval()                                           # REQUIRED: empty clauses -> False
val_acc = (model.predict(x_va) == y_va).float().mean()
torch.save(model.state_dict(), path)
```
Notes `[VERIFIED]` from the library's behaviour:
- **Call `model.eval()` before predicting or exporting.** In training mode empty clauses evaluate to
  True, and predictions change silently.
- **Batch size:** the default `feedback_mode="batch"` aggregates feedback per mini-batch. Keep the
  batch at ≤50 (batch 200 measurably degrades learning), or use `feedback_mode="sequential"` for small data.
- `max_included_literals` is a hard cap in 0.3.0. Pin `torchtsetlin==0.3.0` so the meaning of *b* is stable.

**Starting hyperparameters** (tune T, s and epochs on validation only):

| Dataset | Model | clauses | T | s | epochs |
|---|---|---|---|---|---|
| Noisy XOR | TM | 10 / class | 15 | 3.9 | 200 |
| Breast Cancer | TM, CoTM | 50–200 / class (TM), 100–400 (CoTM) | 10–50 | 3–10 | 100 |
| Phishing | TM, CoTM | 100–500 / class, 200–1000 (CoTM) | 20–100 | 3–10 | 100 |
| MNIST | TM | {100, 500, 2000} / class | 50 (scale with clauses) | 10 | 50 |
| MNIST | CoTM | {500, 2000, 5000} total | 50–5000 `[CHECK]` | 10 | 50 |

The **clause budget grid** for RQ3: *b* ∈ {None, 32, 15, 8, 4}, weighted ∈ {False, True},
3 seeds {0, 1, 2}.

---

## 8. Code that is already written and tested

### 8.1 Status
The exporter (§8.2) and simulator (§8.3) were executed against `torchtsetlin` 0.3.0 on Noisy XOR
for a vanilla TM, a weighted TM and a coalesced TM. **The simulator matched `model.predict` on 5000/5000 test
examples in every case `[VERIFIED]`.** The Concrete pattern (§8.5) compiled to exactly one PBS per
clause and decrypted bit-exactly `[VERIFIED]`.

### 8.2 `src/tmfhe/ir.py`
```python
"""Export a trained flat torchtsetlin classifier to a backend-neutral circuit (the IR)."""
import numpy as np
import torchtsetlin as tt

IR_VERSION = 1


def export_circuit(model) -> dict:
    if isinstance(model, (tt.ConvTsetlinMachine, tt.ConvCoalescedTsetlinMachine,
                          tt.SegmentationTsetlinMachine, tt.CoalescedSegmentationTsetlinMachine)):
        raise TypeError("flat models only; conv models use export_conv_circuit (Phase 3)")
    model.eval()                                            # prediction semantics: empty clause = False
    inc = (model.ta_state >= model.n_states).cpu().numpy()  # (C, 2F); literals are [x, ~x]
    C, L = inc.shape
    F = L // 2
    if isinstance(model, tt.CoalescedTsetlinMachine):
        if model.multi_label:
            raise TypeError("multi-label not supported")
        W = model.weights.cpu().numpy().astype(np.int64)    # (C, K) signed
    elif isinstance(model, tt.TsetlinMachine):
        sw = model.clause_polarity.cpu().numpy().astype(np.int64)
        if model.weighted:
            sw = sw * model.weights.cpu().numpy().astype(np.int64)
        W = np.zeros((C, model.n_classes), np.int64)
        W[np.arange(C), model.clause_class.cpu().numpy()] = sw
    else:
        raise TypeError(type(model).__name__)
    nonempty = inc.any(1)
    contradictory = (inc[:, :F] & inc[:, F:]).any(1)       # x AND ~x: always False
    live = nonempty & ~contradictory & (W != 0).any(1)
    clauses = [np.flatnonzero(inc[j]).tolist() for j in np.flatnonzero(live)]
    return {
        "ir_version": IR_VERSION,
        "model_type": type(model).__name__,
        "n_features": int(F),
        "n_classes": int(W.shape[1]),
        "T": int(model.T) if float(model.T).is_integer() else float(model.T),
        "clauses": clauses,                                  # literal ids: i<F -> x_i, i>=F -> NOT x_{i-F}
        "weights": W[live].tolist(),                         # (C_live, K)
        "stats": {"n_clauses_total": int(C), "n_live": int(live.sum()),
                  "n_empty": int((~nonempty).sum()),
                  "n_contradictory": int((nonempty & contradictory).sum()),
                  "max_k": int(max((len(c) for c in clauses), default=0))},
    }
```
Save as JSON (`json.dump`). Keep the encoder parameters next to it (thermometer thresholds, one-hot
category counts) so the client can Booleanise.

### 8.3 `src/tmfhe/simulate.py`
```python
"""Plaintext evaluation of the IR with exact circuit semantics + bootstrap counts."""
import math
import numpy as np


def clause_bits(ir, x):                      # x: (B, F) in {0,1}
    lits = np.concatenate([x, 1 - x], axis=1).astype(np.int64)
    out = np.zeros((x.shape[0], len(ir["clauses"])), np.int64)
    for j, c in enumerate(ir["clauses"]):
        out[:, j] = (lits[:, c].sum(1) == len(c))
    return out


def predict(ir, x, clamp=True):
    v = clause_bits(ir, x) @ np.asarray(ir["weights"], np.int64)
    if clamp:
        v = np.clip(v, -ir["T"], ir["T"])
    return v.argmax(1), v                    # np.argmax: first max wins, same as torch


def bootstrap_counts(ir, m):
    """F1: AND tree, k-1 gates per clause. F2: groups of <= m literals, one PBS per group,
    plus one combining PBS when a clause needs more than one group."""
    ks = [len(c) for c in ir["clauses"]]
    f1 = sum(max(k - 1, 0) for k in ks)
    f2 = sum(1 if k <= m else math.ceil(k / m) + 1 for k in ks)
    return {"F1_gates": f1, "F2_pbs": f2, "clauses": len(ks)}
```
`tests/test_ir.py` must assert `simulate.predict(ir, x)[0] == model.predict(x)` on the full test set
for every trained model (vanilla, weighted, coalesced). This is gate G0's correctness test.

### 8.4 Backend protocol (`src/tmfhe/backends/base.py`)
```python
class Backend(Protocol):
    name: str                                   # e.g. "concrete-2.11.0/F2"
    def setup(self, ir: dict, formulation: str, output: str) -> dict: ...   # compile + keygen; returns sizes/timings
    def encrypt(self, x_bits: np.ndarray) -> Any: ...                       # client side, one sample
    def run(self, ct: Any) -> Any: ...                                      # server side; THIS is timed as latency
    def decrypt(self, ct_out: Any) -> np.ndarray: ...                       # client: clause bits (O1) or sums (O2) or label (O3)
    def sizes(self) -> dict: ...                                            # key/ciphertext bytes
```
`scripts/p1_encrypted.py` loops: encrypt → run (timed) → decrypt → compare with
`simulate.predict` for that sample, and writes one record per (arm, model, seed).

### 8.5 Concrete backend pattern (A1) `[VERIFIED]`
```python
import numpy as np
from concrete import fhe

def build_concrete(ir, inputset):
    F = ir["n_features"]; C = len(ir["clauses"])
    INC = np.zeros((2 * F, C), np.int64)
    for j, c in enumerate(ir["clauses"]):
        INC[c, j] = 1
    k = INC.sum(0)
    W = np.asarray(ir["weights"], np.int64)

    def tm(x):                                   # x: (F,) encrypted bits
        lits = np.concatenate((x, 1 - x))        # NOT is free
        s = lits @ INC                           # clear-matrix product: no PBS
        c = s == k                               # one PBS per clause
        return c @ W                             # O2 class sums (return c for O1)

    circuit = fhe.Compiler(tm, {"x": "encrypted"}).compile(inputset)  # inputset: ~100 real samples
    circuit.keygen()
    return circuit
# circuit.programmable_bootstrap_count == C  [VERIFIED]
# ct = circuit.encrypt(x); out = circuit.run(ct); circuit.decrypt(out)
```
Record `circuit.programmable_bootstrap_count`, `circuit.complexity`, compile time, keygen time and
the key sizes (`[CHECK]` serialisation API for sizes: `circuit.client.keys.serialize()` /
`circuit.server.save()`). For large circuits, `[CHECK]` compile memory. If needed, compile per
block of clauses and sum the partial class sums on the client (O2-partial). Record that if used.

### 8.6 OpenFHE backend (A4–A6, F1) `[VERIFIED API]`
```python
from openfhe import BinFHEContext, STD128, STD128_LMKCDEY, GINX, AP, LMKCDEY, AND

def make_ctx(method):
    cc = BinFHEContext()
    cc.GenerateBinFHEContext(STD128_LMKCDEY if method == LMKCDEY else STD128, method)
    sk = cc.KeyGen(); cc.BTKeyGen(sk)
    return cc, sk

def eval_clause_f1(cc, ct_lits, clause):          # ct_lits[i]: encrypted literal i (NOT via cc.EvalNOT)
    layer = [ct_lits[i] for i in clause]
    while len(layer) > 1:                         # balanced AND tree, depth ceil(log2 k)
        nxt = [cc.EvalBinGate(AND, layer[i], layer[i + 1]) for i in range(0, len(layer) - 1, 2)]
        if len(layer) % 2: nxt.append(layer[-1])
        layer = nxt
    return layer[0]                               # k == 1 costs 0 gates
```
Single-gate latency measured during the pilot (Python, 2-input AND, mean of 20, no thread
pinning): GINX ≈ 113 ms, AP ≈ 137 ms, LMKCDEY ≈ 105 ms; keygen 0.3 s / 3.6 s / 0.2 s
`[VERIFIED, unpinned]`. Re-measure in Phase 0 with thread control. These are slow enough that A4–A6
run on Phase 1 datasets and the smallest MNIST models only. Literals are encrypted once per sample
(F ciphertexts; negations by `EvalNOT`, which is free). Parallelise clauses with a process pool only
if the bindings release the GIL (`[CHECK]`). Otherwise report single-thread figures and parallel
projections, labelled as projections.

### 8.7 TFHE-rs CLI (A2, A3) — to be written in Phase 0, `[CHECK]` all APIs on docs.rs
- Crate `rust/tm_tfhe`, binary `tm-tfhe` with subcommands
  `keygen | encrypt | run --ir model.json --form F1|F2 --out O1|O2|O3 | decrypt | bench`.
  Ciphertexts and keys are exchanged as `bincode` files, so Python only orchestrates and is never
  inside the timed region.
- F1: `tfhe::boolean` (`gen_keys()`, `server_key.and(&a, &b)`, `server_key.not(&a)`).
- F2: `tfhe::shortint` with the default 128-bit parameter set for **message 2 bits + carry 2 bits**
  (4-bit space, sums up to 15). Clause = `unchecked_add` over its literal ciphertexts, then
  `apply_lookup_table(&acc, &lut_eq_k)`. **`[CHECK]` that the LUT is evaluated on the full
  message+carry value**, not message-only. Unit-test it: the sum of k encrypted ones → 1, k−1 → 0, for
  every k ≤ 15. For larger budgets use a 3+3 or 4+4 parameter set, or grouping. Class sums (O2) with
  `tfhe::integer` radix ciphertexts or client-side summation of decrypted clause bits. Pick one and
  record it.
- Parallelism: rayon over clauses (`par_iter`). GPU: `[CHECK]` `gpu` feature, CUDA server key types.
- Reproduce the Phase-0 microbenchmarks inside Rust (`criterion` or manual `Instant` timing over ≥100
  iterations).

---

## 9. Phases, tasks and gates

### Phase 0 — infrastructure and primitive costs (estimate 1 week)
1. Create the repo (§6.1), environment (§6.2), CI (ruff + pytest on toy sizes + `latexmk` build).
2. Implement `ir.py`, `simulate.py` and their tests (copy §8.2–8.3).
3. Backends A1 (Concrete) and A4–A6 (OpenFHE). Build the TFHE-rs CLI (A2), and A3 if CUDA is available.
4. **Microbenchmarks** (`scripts/p0_microbench.py`, per backend, 1 thread and all threads):
   keygen time; key sizes (client, server/bootstrap); ciphertext size per bit; one gate bootstrap;
   PBS at 2, 3, 4, 5, 6, 7 message bits; ≥200 repetitions → median, IQR.
5. **Failure rate:** ≥10⁵ PBS on random inputs per parameter set where feasible (A1/A2). Count wrong
   decryptions, and report the observed rate with a 95 % Clopper–Pearson upper bound.
6. **Toy end-to-end:** a 2-class TM with 6 clauses on 12 bits (the §8.5 smoke test) through every
   backend and every supported formulation/output mode.

**G0 passes when:** (a) the IR/simulator tests pass for all model types; (b) every core backend gives
bit-exact output on the toy; (c) the microbenchmark table is in `results/`. Otherwise document the
failure and drop or fix the arm.

### Phase 1 — exactness on small models (RQ1; estimate 1 week)
1. Train Noisy XOR, Breast Cancer and Phishing models: TM and CoTM, *b* ∈ {None, 8}, 3 seeds.
   Select on validation, then evaluate test once.
2. Export IR, and verify simulator == predict on 100 % of test (G0 test, re-run).
3. Encrypted inference on the **entire test set** (Noisy XOR: 1000 samples) for every core arm ×
   formulation × output mode the arm supports.
4. Record exact-match count, latency (median/IQR/p95), bytes up/down, keys and bootstrap counts.

**G1 passes when:** ≥1 arm is bit-exact on all three datasets, and every mismatch in any arm is
explained (parameter precision, carry overflow, bootstrap failure, clamp). Write a mismatch
table in `FINDINGS.md`.

### Phase 2 — MNIST accuracy–latency frontier and the clause budget (RQ2–RQ4; estimate 2–3 weeks)
1. Train on MNIST (Boolean, threshold 0.5): TM {100, 500, 2000}/class × *b* {None, 32, 15, 8, 4} ×
   weighted {F, T} × seeds {0,1,2}, and CoTM {500, 2000, 5000} × *b* × seeds. Log the plaintext
   validation and test accuracy, `max_k`, live clauses and the F2 bootstrap count from the simulator.
2. **Latency is input-independent in FHE** (the circuit is fixed), so measure the latency of **every
   configuration (seed 0)** on 30 samples with A1 and A2. Check **exactness on 1000 test images** for
   the Pareto-front configurations, and on the full 10 000 for the two finalists.
3. Fit the latency model: latency ≈ α·PBS(precision) + β, with α per message bit-width from Phase 0.
   Report the prediction error (target ≤25 %).
4. **Comparison with EI-DDLGN:** overlay its four reported points on our plot as "reported, i9-10900".
   If time allows, reproduce EI-DDLGN-Small locally (difflogic + their TFHE-rs parameters) for a
   same-hardware comparison. Otherwise state clearly that the comparison is cross-hardware.
5. Pilot-based budget: ~1.7 ms/PBS amortised (32 threads, ≤4-bit) → a 2000-clause/class TM (20 000
   PBS) ≈ 35 s/image, while a 2000-clause CoTM ≈ 3.5 s/image `[HYPOTHESIS]`. Plan compute so the
   whole grid at 30 samples/config fits in about 24 h of machine time; trim the grid if not.

**G2 passes when** RQ3 is answered either way with `[MEASURED]` records: does tightening *b* reduce
latency at bounded accuracy loss? Is the CoTM cheaper per accuracy point than the vanilla TM?

### Phase 3 — convolutional TM feasibility (RQ5; estimate 1 week)
1. Write `export_conv_circuit` for `tt.ConvTsetlinMachine` (patch size, stride, padding,
   `position_encoding`; reuse the model's own patch extraction from `torchtsetlin` to build
   per-patch literal index maps). Verify it against `model.predict` with a conv simulator first.
   **Note:** `position_encoding=True` makes clauses location-specific. Train with it off on
   translation-invariant toys (`make_2d_noisy_xor`), where it otherwise hurts accuracy.
2. Simulator-only PBS counts vs P (patches), C, patch size.
3. Encrypt the smallest configurations that the latency model predicts under 10 min/sample (A1 or A2).

**G3:** a feasibility frontier (P × C vs latency), measured at ≥2 points and predicted elsewhere.

### Phase 4 — optional (each needs a go-ahead after G2)
- TM-2 (encrypted model) on Breast Cancer / Phishing with A2 F1.
- A8: CKKS functional bootstrapping in OpenFHE (C++), F2 batched across slots, amortised per sample.
- A7: FINAL vs FHEW vs TFHE gate cost at equal security, F1.

### Phase 5 — paper (estimate 2 weeks, overlapping Phase 3–4)
See §11. **G4:** every number in the PDF is traceable to a `results/*.json` record, and the paper
builds in CI.

---

## 10. Result record format (`src/tmfhe/provenance.py`)

One JSON file per run: `results/<phase>/<arm>__<dataset>__<model-id>__<form>__<out>__s<seed>.json`
```json
{
  "record_id": "p2-A1-mnist-tm2000-b15-w0-F2-O2-s0",
  "phase": "p2", "arm": "A1", "scheme": "TFHE/CGGI", "backend": "concrete-python 2.11.0",
  "dataset": "mnist", "split_seed": 1234,
  "model": {"type": "TsetlinMachine", "n_clauses": 2000, "T": 50, "s": 10, "weighted": false,
            "max_included_literals": 15, "seed": 0, "epochs": 50, "batch_size": 32,
            "plaintext_val_acc": 0.0, "plaintext_test_acc": 0.0},
  "ir": {"n_live": 0, "max_k": 0, "F1_gates": 0, "F2_pbs": 0, "sha256": "..."},
  "fhe_params": {"name": "...", "security_bits_estimated": 0, "message_bits": 4},
  "formulation": "F2", "output_mode": "O2", "threads": 32,
  "timing_s": {"compile": 0, "keygen": 0, "encrypt_median": 0, "run_median": 0, "run_iqr": [0, 0],
               "run_p95": 0, "decrypt_median": 0, "n_timed": 30},
  "sizes_bytes": {"client_key": 0, "server_key": 0, "ct_in": 0, "ct_out": 0},
  "exactness": {"n_samples": 1000, "n_match_vs_simulator": 1000, "n_match_vs_predict": 1000},
  "argv": ["..."], "git_sha": "...", "env": {"cpu": "...", "gpu": "...", "python": "...",
  "packages": {"torchtsetlin": "0.3.0", "concrete-python": "2.11.0", "openfhe": "...", "tfhe": "..."}},
  "timestamp": "..."
}
```

---

## 11. The paper (IEEE format, LaTeX)

### 11.1 Target and template
- IEEE conference format: `\documentclass[conference]{IEEEtran}` (from CTAN, or `pip`/`tlmgr install
  ieeetran`). Default is 8 pages + references. Candidate venues: ISTM 2026/2027 (IEEE, Tsetlin
  community), IEEE TrustCom / IEEE S&P workshops, or IEEE Access (journal, `\documentclass[journal]{IEEEtran}`
  with the Access template) for a longer version.
- Build with `latexmk -pdf main.tex`, or `tectonic main.tex`. CI: GitHub Action
  `xu-cheng/latex-action@v3` with `root_file: paper/main.tex`.

### 11.2 Working title
*Encrypted Inference for Tsetlin Machines: One Bootstrap per Clause with Bit-wise Fully Homomorphic Encryption*

### 11.3 Skeleton (`paper/main.tex`)
```latex
\documentclass[conference]{IEEEtran}
\usepackage{cite,amsmath,amssymb,graphicx,booktabs,siunitx,xcolor,algorithm,algpseudocode}
\usepackage[hidelinks]{hyperref}
\begin{document}
\title{Encrypted Inference for Tsetlin Machines:\\One Bootstrap per Clause with Bit-wise FHE}
\author{\IEEEauthorblockN{Vajira Thambawita}\IEEEauthorblockA{Simula, Oslo, Norway\\vajira@simula.no}}
\maketitle
\begin{abstract} ... \end{abstract}
\begin{IEEEkeywords} Tsetlin machine, fully homomorphic encryption, TFHE, FHEW,
privacy-preserving machine learning, interpretable AI \end{IEEEkeywords}
\input{sections/introduction}
\input{sections/background}
\input{sections/method}
\input{sections/setup}
\input{sections/results}
\input{sections/discussion}
\input{sections/related}
\input{sections/conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```
(Fix the author block and affiliation before submission.)

### 11.4 Section contents
1. **Introduction.** Motivation (private inference on sensitive tabular/medical data; TMs are
   Boolean, interpretable and cheap). The gap (no encrypted TM exists, §1.3). Contributions:
   (i) the first mapping of TM inference to bit-wise FHE, with the one-PBS-per-clause formulation;
   (ii) the clause budget as a direct FHE-cost control; (iii) a multi-scheme evaluation (TFHE via
   Concrete/TFHE-rs, FHEW, LMKCDEY, ±FINAL) on four datasets; (iv) MNIST comparison with
   logic-gate networks; (v) open-source code and records.
2. **Background.** TM inference (Eq.: clause, class sum, clamp, argmax), CoTM, clause budget. LWE,
   gate bootstrapping, PBS, and the cost model (linear ops free, non-linear = bootstrap).
3. **Method.** Threat model and output modes (§3), IR, F1/F2 (+ Algorithm 1: encrypted TM
   inference), bootstrap-count model, precision analysis (bits = ⌈log₂(max_k+1)⌉, plus carry for
   the sums), convolution extension.
4. **Experimental setup.** Hardware, libraries with exact versions, parameter sets + estimated
   security, datasets/encodings/splits, TM hyperparameters, metrics, timing protocol.
5. **Results.** RQ1–RQ5, each with its table/figure (§11.5).
6. **Discussion.** What leaks in O1/O2/O3; interpretability vs model confidentiality (TM-2);
   limits (no training, conv cost); when to choose which scheme.
7. **Related work.** Boolean-native FHE ML (EI-DDLGN, TT-TFHE, Concrete ML trees, Homomorphic
   WiSARDs, TAPAS, FHE-DiNN), MPC (XONN), TM privacy/FL (FedTM, TPFL), TM hardware.
8. **Conclusion.**

### 11.5 Figures and tables (all generated by `paper/scripts/`)
| ID | Content | Source |
|---|---|---|
| Fig. 1 | Pipeline diagram: client Booleanise → encrypt → server clause layer (Σ + PBS) → sums → client | hand-drawn TikZ (no numbers) |
| Tab. I | Schemes/backends/parameters/security/key sizes | p0 records |
| Tab. II | Primitive costs: gate, PBS at 2–7 bits, 1 vs all threads | p0 records |
| Tab. III | Exactness + latency on Noisy XOR, Breast Cancer, Phishing per arm | p1 records |
| Fig. 2 | Latency vs message precision (max_k) at fixed C, pilot-style sweep | p0/p2 records |
| Fig. 3 | MNIST accuracy vs latency (log x), TM vs CoTM, coloured by *b*; EI-DDLGN overlay (reported) | p2 records + constants table with citation |
| Fig. 4 | Predicted vs measured latency (cost-model validation) | p2 records |
| Tab. IV | Clause-budget ablation: accuracy, max_k, PBS, latency | p2 records |
| Fig. 5 | Conv feasibility frontier (P × C vs latency) | p3 records |
| Tab. V | Leakage per output mode (qualitative) | text |

Style: vector PDFs, IEEE single-column width (3.5 in), consistent colours per scheme, numbers via
`siunitx`. Report mean ± sd over seeds for accuracy, and median [IQR] for latency.

### 11.6 Reference list for `refs.bib` (verify every entry against its DOI/ePrint before submission)
| Key | Reference |
|---|---|
| granmo2018tm | O.-C. Granmo, "The Tsetlin Machine — A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic," arXiv:1804.01508, 2018 |
| granmo2019ctm | O.-C. Granmo et al., "The Convolutional Tsetlin Machine," arXiv:1905.09688, 2019 |
| phoulady2020wtm | A. Phoulady et al., "The Weighted Tsetlin Machine: Compressed Representations with Weighted Clauses," arXiv:1911.12607 |
| glimsdal2021cotm | S. Glimsdal, O.-C. Granmo, "Coalesced Multi-Output Tsetlin Machines with Clause Sharing," arXiv:2108.07594 |
| abeyrathna2023clausesize | K. D. Abeyrathna et al., "Building Concise Logical Patterns by Constraining Tsetlin Machine Clause Size," IJCAI 2023, arXiv:2301.08190 |
| gentry2009 | C. Gentry, "Fully Homomorphic Encryption Using Ideal Lattices," STOC 2009 |
| gsw2013 | C. Gentry, A. Sahai, B. Waters, CRYPTO 2013, ePrint 2013/340 |
| ducas2015fhew | L. Ducas, D. Micciancio, "FHEW: Bootstrapping Homomorphic Encryption in Less Than a Second," EUROCRYPT 2015, ePrint 2014/816 |
| chillotti2020tfhe | I. Chillotti, N. Gama, M. Georgieva, M. Izabachène, "TFHE: Fast Fully Homomorphic Encryption over the Torus," J. Cryptology 33, 2020, doi:10.1007/s00145-019-09319-x |
| chillotti2021pbs | I. Chillotti, M. Joye, P. Paillier, "Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks," CSCML 2021, ePrint 2021/091 |
| lee2023lmkcdey | Y. Lee et al., "Efficient FHEW Bootstrapping with Small Evaluation Keys, and Applications to Threshold HE," EUROCRYPT 2023, ePrint 2022/198 |
| bonte2022final | C. Bonte et al., "FINAL: Faster FHE Instantiated with NTRU and LWE," ASIACRYPT 2022, ePrint 2022/074 |
| openfhe2022 | A. Al Badawi et al., "OpenFHE: Open-Source Fully Homomorphic Encryption Library," WAHC 2022, ePrint 2022/915 |
| tfhers | Zama, "TFHE-rs," https://github.com/zama-ai/tfhe-rs (cite the version used) |
| concrete | Zama, "Concrete," https://github.com/zama-ai/concrete (cite the version used) |
| albrecht2015lwe | M. R. Albrecht, R. Player, S. Scott, "On the concrete hardness of Learning with Errors," J. Math. Cryptology 2015 (lattice estimator) |
| yassin2026eiddlgn | M. Y. M. Yassin, M. A. Sayed, M. Taha, "EI-DDLGN: Efficient Encrypted Inference with Deep Differentiable Logic Gate Networks under TFHE," arXiv:2609.13636, 2026 |
| petersen2022difflogic | F. Petersen et al., "Deep Differentiable Logic Gate Networks," NeurIPS 2022, arXiv:2210.08277 |
| benamira2023tttfhe | A. Benamira et al., "TT-TFHE: a Torus FHE-Friendly Neural Network Architecture," arXiv:2302.01584 |
| frery2023trees | J. Frery et al., "Privacy-Preserving Tree-Based Inference with TFHE," arXiv:2303.01254 |
| neumann2024wisard | L. Neumann et al., "Homomorphic WiSARDs: Efficient Weightless Neural Network Training over Encrypted Data," arXiv:2403.20190 |
| duan2026tsetlinwisard | S. Duan et al., "TsetlinWiSARD," arXiv:2603.24186 |
| sanyal2018tapas | A. Sanyal et al., "TAPAS," ICML 2018, arXiv:1806.03461 |
| bourse2018fhedinn | F. Bourse et al., "Fast Homomorphic Evaluation of Deep Discretized Neural Networks," CRYPTO 2018, ePrint 2017/1114 |
| riazi2019xonn | M. S. Riazi et al., "XONN," USENIX Security 2019, arXiv:1902.07342 |
| how2023fedtm | S. S. Q. How et al., "FedTM," ISTM 2023, doi:10.1109/ISTM58889.2023.10454982 |
| gohari2024tpfl | R. Jafari Gohari et al., "TPFL," arXiv:2409.10392 / Cluster Computing 2025 |
| albadawi2026sok | A. Al Badawi, A. Alexandru, Y. Polyakov, V. Vaikuntanathan, "SoK: Private LLM Inference using Approximate Homomorphic Encryption," ePrint 2026/935 |
| torchtsetlin | V. Thambawita, "torchtsetlin," https://github.com/vlbthambawita/torchtsetlin, v0.3.0 |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Concrete compile time/memory on 20 000-clause circuits | Compile per clause block and sum on the client (O2-partial). Use A2 for the largest models. |
| OpenFHE Python has no F2 and slow gates (~100 ms) | Restrict A4–A6 to Phase 1 and the small MNIST models. Report projections as projections. Optional C++ F2. |
| TFHE-rs LUT semantics (message vs message+carry) | Phase-0 unit test over all k (§8.7) before any measurement. |
| Bootstrap failures → rare mismatches | Phase-0 failure rate; attribute every Phase-1 mismatch. |
| Clamp/argmax disagreement | Simulator with clamp; O2 does clamp+argmax on the client; O3 implements the clamp. |
| Cross-hardware comparison with EI-DDLGN | Label "reported". Reproduce one point if possible. Compare on PBS counts as a hardware-free axis. |
| CoTM T/s settings unclear on MNIST | Tune on validation in Phase 2 step 1 before the grid. Report the tuning budget. |
| Grid too expensive | Latency on 30 samples (input-independent), exactness on subsets, finalists in full (§9 Phase 2). |
| Library change mid-study | Pin every version. A change requires re-running G0. |

---

## 13. Timeline (indicative)

| Week | Work | Gate |
|---|---|---|
| 1 | Phase 0 | G0 |
| 2 | Phase 1 | G1 |
| 3–5 | Phase 2 (training on GPU while backends run on CPU) | G2 |
| 6 | Phase 3 | G3 |
| 6–8 | Phase 4 (optional) + paper | G4 |

---

## 14. Pilot reproduction script (for Phase 0 step 6 and §1.4)
```python
# pip install "concrete-python==2.11.0" "setuptools<81" numpy
import time, numpy as np
from concrete import fhe
F, K = 784, 10
rng = np.random.default_rng(0)
for C, kmax in [(200, 15), (1000, 3), (1000, 8), (1000, 15), (1000, 31), (1000, 63)]:
    INC = np.zeros((2 * F, C), np.int64)
    for j in range(C):
        INC[rng.choice(2 * F, size=rng.integers(1, kmax + 1), replace=False), j] = 1
    k = INC.sum(0); W = rng.integers(-3, 4, size=(C, K))
    def tm(x):
        return ((np.concatenate((x, 1 - x)) @ INC) == k) @ W
    circ = fhe.Compiler(tm, {"x": "encrypted"}).compile([rng.integers(0, 2, size=F) for _ in range(50)])
    circ.keygen()
    x = rng.integers(0, 2, size=F); enc = circ.encrypt(x)
    lat = []
    for _ in range(3):
        t = time.time(); r = circ.run(enc); lat.append(time.time() - t)
    print(C, kmax, circ.programmable_bootstrap_count, f"{np.median(lat):.2f}s",
          np.array_equal(circ.decrypt(r), tm(x)))
```
