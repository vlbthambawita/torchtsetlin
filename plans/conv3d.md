# 3D convolutional Tsetlin machine: feasibility, proofs and implementation plan

*Status: **implemented**, 2026-10-12. Every step below is done. See "Implementation notes"
at the end for what the measurements changed.*

## TL;DR

**Yes. It is feasible, and most of the existing machinery carries over unchanged.** The
convolutional Tsetlin machine (CTM, Granmo et al., 2019) never uses the fact that its patches
come from a 2D grid. Clause evaluation, the OR over patches, Type I/II selection and the
random-matching-patch feedback all work on an abstract **set of patches**: in this code base,
the `(B, P, 2F)` literal tensor that `_ConvMixin._encode` produces. A 3D CTM changes only how
that set is built:

* windows slide over `(D, H, W)` instead of `(H, W)`,
* each patch has `Z·kd·kh·kw` voxel features, and
* position literals use three thermometers instead of two.

In code, a new `_Conv3dMixin` overrides the geometry methods (`_grid`, `conv_features`,
`_coerce_input`, `_encode`, `_patches_per_example`, and the interpretation helpers). It
inherits `_evaluate` and `_feedback_counts` from `_ConvMixin` without changes. The proofs below
show this is correct, show that it reduces bit for bit to the 2D model when `D = kd = 1`, and
show that it is strictly more expressive per clause than the obvious workaround (stacking depth
into channels). One real engineering problem needs work: **memory**. For volumes the size of
cardiac MRI, the per-example intermediates exceed the default chunk budget, so the plan
includes a patch-axis chunking step (Step 6) with its own correctness lemma.

Motivating use cases: short-axis cardiac MRI stacks `(slices, H, W)`, cine MRI treated as
`(time, H, W)` (2D+t), and CT/MRI volumes in general.

---

## 1. Definitions

**Input.** A Boolean volume $X \in \{0,1\}^{Z \times D \times H \times W}$ with $Z$ bit planes
(channels after Booleanization).

**Windows.** Kernel $k = (k_d, k_h, k_w)$ and stride $s = (s_d, s_h, s_w)$, with no padding,
which matches the 2D classifier. Along an axis of length $N$ with kernel $k_a \le N$ and
stride $s_a$, the window start positions are $\{0, s_a, 2s_a, \dots\}$ such that the window
fits. The patch grid is $P_d \times P_h \times P_w$ and $P = P_d P_h P_w$.

**Patch features.** For patch position $p = (u, v, w)$, define the feature vector
$\phi_p(X) \in \{0,1\}^F$ as

$$
\phi_p(X)_{(z,i,j,l)} = X[z,\; u s_d + i,\; v s_h + j,\; w s_w + l],
\qquad 0 \le i < k_d,\ 0 \le j < k_h,\ 0 \le l < k_w,
$$

flattened in $(z, i, j, l)$ row-major order. With `position_encoding=True`, three thermometer
blocks follow it:

$$
\tau_d(u)_m = [u > m]\ (0 \le m < P_d - 1),\quad
\tau_h(v)_m = [v > m]\ (0 \le m < P_h - 1),\quad
\tau_w(w)_m = [w > m]\ (0 \le m < P_w - 1),
$$

so $F = Z k_d k_h k_w + (P_d - 1) + (P_h - 1) + (P_w - 1)$. The literal vector is
$\ell_p(X) = [\phi_p(X), \neg\phi_p(X)] \in \{0,1\}^{2F}$, which is exactly what
`_ConvMixin._encode` builds in 2D.

**Clause.** Clause $j$ has an include set $I_j \subseteq \{1,\dots,2F\}$. It matches a patch
when $c_j(\ell) = \bigwedge_{t \in I_j} \ell_t$, and its image-level output is

$$
C_j(X) = \bigvee_{p \in \mathcal{P}} c_j(\ell_p(X)),
$$

where an empty $I_j$ evaluates to `True` while learning and `False` while predicting, as in
2D.

**Feedback.** This is unchanged from the CTM. Vote sums, the clamp at $T$ and the Type I/II
selection probabilities depend only on the vector of clause outputs $C(X)$. When a clause that
fired is selected for Type Ia or Type II feedback, it is updated with the literals
$\ell_{p^\*}(X)$ of a patch $p^\*$ drawn uniformly from its matching set
$M_j(X) = \{p : c_j(\ell_p(X)) = 1\}$. Type Ib uses no patch.

---

## 2. Mathematical results

### Lemma 1: grid size

Along one axis, the number of window positions is $P_a = \lfloor (N_a - k_a)/s_a \rfloor + 1$
whenever $k_a \le N_a$.

*Proof.* Start $t s_a$ is valid iff $t s_a + k_a \le N_a$, i.e.
$t \le (N_a - k_a)/s_a$. The valid $t$ are the integers $0,\dots,\lfloor (N_a-k_a)/s_a \rfloor$,
which is $\lfloor (N_a-k_a)/s_a\rfloor + 1$ values. ∎

This is the same formula as `_ConvMixin._grid`, applied to three axes.

### Lemma 2: chained `Tensor.unfold` builds $\phi_p$ exactly

`torch.nn.functional.unfold` only accepts 4D input, so the 3D encoder uses

```python
u = x.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)   # (B, Z, Pd, Ph, Pw, kd, kh, kw)
patches = u.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(B, Pd * Ph * Pw, Z * kd * kh * kw)
```

*Claim.* `patches[b, (u·Ph + v)·Pw + w, ((z·kd + i)·kh + j)·kw + l] = X[b, z, u·sd+i, v·sh+j, w·sw+l]`.

*Proof.* By definition, `Tensor.unfold(dim, size, step)` replaces axis `dim` (length $N$) by an
axis of length $\lfloor (N - \text{size})/\text{step}\rfloor + 1$ (Lemma 1). It appends a new
trailing axis of length `size`, and element $[\dots, t, \dots, r]$ equals the original element
at index $t\cdot\text{step} + r$ on that axis. The three calls act on distinct original axes
(2, 3, 4). Each call leaves the other axes' indexing unchanged, so the composition gives
`u[b, z, u_, v, w, i, j, l] = X[b, z, u_·sd + i, v·sh + j, w·sw + l]`. The permutation moves
$z$ next to $(i, j, l)$. A row-major reshape flattens the leading $(u, v, w)$ into the patch
index $(u P_h + v) P_w + w$, and the trailing $(z, i, j, l)$ into the feature index
$((z k_d + i) k_h + j) k_w + l$. ∎

*Numerical check (done while writing this plan):* exhaustive comparison against explicit
slicing for `(B,Z,D,H,W) = (2,3,5,6,7)`, `k = (2,3,4)`, `s = (1,2,1)`. All entries match.

The patch order is row-major over $(u, v, w)$, so `patch_clause_outputs` can reshape the
`(B, P, C)` matches to `(B, Pd, Ph, Pw, C)`.

### Theorem 3: exact reduction to the 2D model

If $D = k_d = 1$ (so $s_d$ is irrelevant), then for every input the 3D literal tensor
`(B, P, 2F)` is **bitwise identical** to the 2D literal tensor of `ConvTsetlinMachine` with
`patch_size=(kh, kw)`, `stride=(sh, sw)` on `X[:, :, 0]`. The same holds with
`position_encoding=True`.

*Proof.*
(i) $P_d = 1$ by Lemma 1, so $P = P_h P_w$ and the patch index $(0\cdot P_h + v)P_w + w = vP_w + w$
is the same row-major order that `TF.unfold` uses.
(ii) $k_d = 1$ forces $i = 0$, so the feature index is $(z k_h + j) k_w + l$. This is `TF.unfold`'s
channel-major order `(z, j, l)`, and both reference the voxel $X[z, 0, v s_h + j, w s_w + l]$.
(iii) The depth thermometer has $P_d - 1 = 0$ bits, so the position block is
$[\tau_h(v), \tau_w(w)]$, which is exactly `_position_bits(Py, Px)`.
(iv) Negated literals are appended the same way in both. ∎

*Corollary 3a.* The two models have identical `include` matrices after identical `ta_state`
initialisation, and `_evaluate` and `_feedback_counts` are shared code acting on identical
tensors. So under the same RNG seed, the **whole training trajectory is identical**. Numerically
confirmed for (i)–(ii) with `k = (3, 4)`, `s = (2, 1)`: `torch.equal` holds. This gives the
strongest regression test available (Step 8, test T3).

### Theorem 4: the shared learning code is correct for any patch set

`_ConvMixin._evaluate` and `_ConvMixin._feedback_counts` implement the CTM semantics of §1 for
**any** finite patch set given as a `(B, P, 2F)` literal tensor. So they are correct for the 3D
patch set without changes.

*Proof.* (a) *Evaluation.* `clause_outputs` on the `(B·P, 2F)` rows returns
$c_j(\ell_p)$ for every $(b, p, j)$: a clause matches iff its included literals have zero
violations (`clause_violations == 0`), with the empty-clause rule applied per mode.
`.any(dim=1)` is $\bigvee_p$. Neither step reads $P$'s geometry.

(b) *Random matching patch.* For a selected event $(b, j)$, the code forms
$\text{cand}_p = m_p \in \{0,1\}$ (the match indicator), draws $U_p \overset{iid}{\sim} \mathcal{U}[0,1)$,
and returns $p^\* = \arg\max_p U_p m_p$. Event selection is gated on $C_j(X)=1$, so
$M = M_j(X) \neq \varnothing$. If $p \notin M$, then $U_p m_p = 0$. If $p \in M$, then
$U_p m_p = U_p > 0$ almost surely, so $p^\* \in M$ a.s. Conditioned on the values, the
$\{U_p\}_{p\in M}$ are exchangeable and ties have probability 0 for continuous $U$, so
$\Pr[p^\* = p] = 1/|M|$ for each $p \in M$. ∎

(c) *Counts.* `index_add_` of $\ell_{p^\*}$ and $1 - \ell_{p^\*}$ into `n_true`/`n_false`
(Type Ia) and of $1-\ell_{p^\*}$ into `n2` (Type II) is the per-event CTM update that
`functional.apply_feedback` expects. The arguments in (a)–(c) use only the axis $P$.

*Remark (floating point, inherited from 2D, not new).* `torch.rand` is a float on a finite grid.
Ties among matched patches have probability $O(|M|^2 2^{-24})$ and resolve to the lowest
index. If every matched $U_p$ is exactly 0, `argmax` returns patch 0 even when it does not
match. Both effects are negligible and shared with 2D. They are recorded for completeness, not
fixed here.

*Consequence.* Any property of the CTM that is stated in terms of the abstract patch multiset
$\{\ell_p(X)\}_{p}$ transfers to 3D. This covers the per-event feedback rule, the vote and $T$
dynamics, and the batched-feedback fidelity trade-off in `docs/concepts/batching.md`. The only
thing that changes is which patches exist.

### Lemma 5: position literals express exactly the axis-aligned boxes

Let a clause include position literals only. The set of patch positions it admits is
$[a_d, b_d] \times [a_h, b_h] \times [a_w, b_w]$ (possibly empty). Conversely, every box of
patch coordinates can be expressed this way with at most 6 literals.

*Proof.* On one axis, the literal $[u > m]$ admits $u \ge m+1$ and $\neg[u > m]$ admits
$u \le m$. A conjunction of such literals is an intersection of half-lines, which is an
interval $[\max(m_i + 1), \min(m'_k)]$ (empty if lo > hi). Literals on different axes constrain
independent coordinates, so the conjunction is the product of the three intervals. Conversely,
$[a, b]$ on an axis with $P_a$ positions is $[u > a-1]$ (omitted if $a = 0$) $\wedge$
$\neg[u > b]$ (omitted if $b = P_a - 1$). That is at most 2 literals per axis. ∎

This is the 3D version of `clause_region`, which gains a `"d"` key decoded with the same
`rng` helper. The encoding costs $(P_d - 1) + (P_h - 1) + (P_w - 1)$ features. A one-hot
position code would cost $P_d P_h P_w$ (33,640 in the example of §3) and could not express a
box as one conjunction.

### Proposition 6: translation invariance (`position_encoding=False`)

Let $X'$ be $X$ shifted by $\Delta = (\delta_d s_d, \delta_h s_h, \delta_w s_w)$ with integer
$\delta$. Suppose every patch that matches clause $j$ on $X$ still lies inside the volume after
the shift. Then $C_j(X') \ge C_j(X)$. If the shift is invertible on the support of the matches
(the content moves without being cut off), $C_j(X') = C_j(X)$.

*Proof.* Without position bits, $\ell_p$ depends only on the voxels under window $p$. The
window at $p + \delta$ in $X'$ covers exactly the voxels that window $p$ covers in $X$, so
$\ell_{p+\delta}(X') = \ell_p(X)$. A match at $p$ in $X$ therefore gives a match at $p+\delta$ in
$X'$, and $\bigvee$ is preserved. Applying the same argument to $-\Delta$ gives the converse. ∎

The CLAUDE.md caveat transfers to 3D: with position literals on, translation-invariant toy
tasks oscillate. Use `position_encoding=False` for 3D toy tests.

### Proposition 7: 3D is strictly more expressive per clause than "depth as channels"

The obvious workaround needs no new code: reshape $X$ to $(Z\cdot D, H, W)$ and use the 2D
CTM.

*(a) Equivalence.* "Depth as channels" with window $(k_h, k_w)$ is exactly the 3D CTM with
$k_d = D$ (so $P_d = 1$). Both patch sets index the voxels
$\{X[z, i, \cdot, \cdot]\}_{z, i}$ at each in-plane position, in the same order up to a fixed
permutation of features. A fixed feature permutation is a relabelling of TA indices and leaves
the learning dynamics unchanged.

*(b) Strict separation.* Let $Z = 1$, $H = W = k_h = k_w = 1$, depth $D \ge 2$, and consider
the concept "some voxel is on", $y = \bigvee_{d} x_d$.

* The 3D CTM with $k_d = 1$ represents it with **one** clause, $x$ (the single positive
  literal): the OR over the $D$ depth patches is $\bigvee_d x_d$.
* Any depth-as-channels CTM needs **at least $D$ clauses** voting for the positive class.
  It has a single patch with features $(x_1, \dots, x_D)$, so each clause is one conjunction.
  For the all-zero input to be rejected, a positive clause must be false at $\mathbf{0}$
  (non-empty clauses only, since empty clauses are false at prediction and so never help
  cover a positive). A conjunction is false at $\mathbf{0}$ iff it contains some positive
  literal $x_{d}$. Such a clause is true only on inputs with $x_{d} = 1$. The positives
  $e_1, \dots, e_D$ (one voxel on) each need a clause that is true on them, and the clause
  true on $e_d$ must contain $x_d$ and no other positive literal. So the $D$ positives need
  $D$ pairwise distinct clauses. ∎

More generally, a 3D clause with $k_d < D$ reuses one learned pattern at all $P_d$ depth
offsets. A depth-as-channels model has to learn a separate copy per offset, which costs more
clauses and more data. That is the same argument that justifies 2D convolution over flat TMs,
whose demo is `examples/shapes_conv.py`.

### Lemma 8: patch-chunked evaluation and drawing are exact (needed for Step 6)

Partition $\mathcal{P}$ into blocks $\mathcal{P}_1, \dots, \mathcal{P}_r$.

*(a)* $C_j(X) = \bigvee_{k} \bigvee_{p \in \mathcal{P}_k} c_j(\ell_p)$. This holds because OR is
associative and commutative.

*(b)* With $U_p$ iid as in Theorem 4, let $(v_k, q_k) = (\max_{p\in\mathcal{P}_k} U_p m_p,\ \arg\max_{p \in \mathcal{P}_k} U_p m_p)$.
Then $\arg\max_p U_p m_p = q_{k^\*}$ with $k^\* = \arg\max_k v_k$. The maximum over a union is
the maximum of the block maxima. So a running "best value, best row" carried across patch
chunks gives a draw with **the same distribution** as the unchunked one: uniform on $M_j(X)$ by
Theorem 4(b). ∎

*Caveat.* The draw is equal in distribution, not bitwise. Calling `torch.rand` once per chunk
consumes the RNG stream in a different order. When $r = 1$, the code path must be the existing
one so that Corollary 3a (bitwise identity with 2D) still holds.

---

## 3. Cost model and why memory is the real issue

Per example, the 2D `_chunk_elements_per_example` estimate is
$3PC + 2C\cdot 2F + P\cdot 2F$. It is dimension-agnostic and is inherited as is.

Worked example: a cardiac short-axis stack downsampled to $D=12$, $H=W=64$, $Z=4$ thermometer
planes, kernel $3\times7\times7$, $C = 10 \times 200 = 2000$ clauses.

| stride | $P$ | $F$ | elements / example | fits $2^{27}$ default? | matmul / volume |
|---|---|---|---|---|---|
| (1,1,1) | 10·58·58 = 33,640 | 588 + 123 = 711 | ≈ 255 M | **no** (≈1.9×) | ≈ 0.19 TFLOP |
| (1,2,2) | 10·29·29 = 8,410 | 588 + 65 = 653 | ≈ 66 M | yes, chunk = 2 | ≈ 0.04 TFLOP |

Even at chunk size 1, the stride-1 configuration is over budget. `_chunk_size` floors at 1, so
the code would silently exceed the budget (the estimate is ~1 GB of float32; it is a worst
case, see the implementation notes). Hence Step 6. Note
also that the full-volume `matches` tensor is `bool` (P·C bytes ≈ 67 MB above) and is cheap.
The large intermediates are the float violation matrix and the `(n_sel, P)` random draw.

---

## 4. Implementation plan

Constraints from CLAUDE.md that apply throughout:

* New tensor attributes are **non-persistent buffers**, so the checkpoint format does not change.
* Use `typing.Optional`/`Tuple` (py39 ruff config).
* `_chunk_elements_per_example` must reflect real intermediates.
* Never use `non_blocking` to CPU.
* The random patch is drawn in `_feedback_counts`, not `_evaluate`.

### Step 1: geometry and constructor (`models/conv.py`)

* Add `IntOrTriple` and `_triple()`, alongside `_pair()`.
* Add `class _Conv3dMixin(_ConvMixin)` with `_init_conv3d(patch_size, stride, position_encoding, input_shape)`.
  It stores 3-tuples and validates positivity.
* `_grid(D, H, W) -> (Pd, Ph, Pw)` (Lemma 1). Raise if any axis is smaller than the kernel.
* `conv_features((Z, D, H, W))` returns $F$ from §1.

*Signature note:* `_grid` changes arity relative to the parent. Every caller inside the mixin is
overridden, so nothing inherited calls the 2D signature. Step 8 test T7 asserts this.

### Step 2: encoder

* `_position_bits(Pd, Ph, Pw, device)`: concatenate three thermometers. Each coordinate is
  repeated row-major: `u` uses `repeat_interleave(Ph*Pw)`, `v` uses
  `repeat_interleave(Pw).repeat(Pd)`, and `w` uses `repeat(Pd*Ph)`. Cache per
  `((Pd,Ph,Pw), device)`, as `segmentation.py::_position_bits_table` does, because the table
  is large in 3D.
* `_encode`: chained `Tensor.unfold` + permute + reshape (Lemma 2), append the position bits,
  check the width against `n_features`, and return `cat([patches, 1 - patches])`.

### Step 3: input coercion

`(B, Z, D, H, W)` is canonical. Accept `(D, H, W)` as a single 1-plane volume. Accept a 4D
input as a single `(Z, D, H, W)` volume when it equals `input_shape`, else as `(B, D, H, W)`
with one plane. This mirrors the 2D 3-dim rule.

**Pitfall:** `_ConvMixin._coerce_input` reshapes to 4D, so the override must skip it with
`super(_ConvMixin, self)._coerce_input(x)`. `_patches_per_example` returns `Pd*Ph*Pw`.
`_infer_n_features` is inherited and works through `conv_features`.

### Step 4: learning path (no new code)

Inherit `_evaluate` and `_feedback_counts` as they are (Theorem 4) and inherit
`_chunk_elements_per_example`. Add a comment pointing to this plan's Theorem 4 so nobody "fixes"
them later.

### Step 5: public classes and exports

* `Conv3dTsetlinMachine(_Conv3dMixin, TsetlinMachine)`
* `Conv3dCoalescedTsetlinMachine(_Conv3dMixin, CoalescedTsetlinMachine)`
* `Conv3dRegressionTsetlinMachine(_Conv3dMixin, RegressionTsetlinMachine)`

Constructors match the 2D ones, with `patch_size: IntOrTriple = (3, 7, 7)`, `stride=1`, and
`input_shape: Optional[Tuple[int,int,int,int]]`. Export them from `models/__init__.py` and
`torchtsetlin/__init__.py` (`__all__` in both). Write Google-style docstrings, since they are
the API reference, and add a `:::` stub to `docs/api/`.

### Step 6: patch-axis chunking (needed for realistic volumes)

Add an optional `patches_per_chunk: Optional[int] = None` to the 3D mixin, defaulting to
automatic: on when one example exceeds `max_chunk_elements`.

* **Evaluate:** loop over patch blocks, compute `clause_outputs` per block, and write into a
  preallocated `bool (B, P, C)` (Lemma 8a). Peak float memory drops from
  $B\cdot P\cdot C$ to $B\cdot|\mathcal{P}_k|\cdot C$.
* **Draw:** in `_feedback_counts`, carry `(best_val, best_p)` of shape `(n_sel,)` across
  blocks (Lemma 8b), then gather the rows once.
* **Bit-exactness guard:** when the plan has a single block, call the inherited code path
  unchanged (Corollary 3a must still hold).
* **Budget:** `_chunk_elements_per_example` uses the block size instead of $P$ for the float
  terms and keeps $P\cdot C/4$ (bool bytes in float-element units) for `matches`.

This is a generic improvement that would benefit 2D CTMs on large images too. Keep it in the 3D
mixin first, and promote it to `_ConvMixin` only after it has been benchmarked in 2D.

### Step 7: interpretation and visualisation

* `default_feature_names`: `p[z,i,j,l]`, then `d>k`, `y>k`, `x>k`.
* `clause_patch(j) -> (Z, kd, kh, kw)` int8.
* `clause_region(j) -> {"d","y","x"}` (Lemma 5).
* `patch_clause_outputs(x) -> (B, Pd, Ph, Pw, C)`.
* `viz.plot_conv_clauses` currently does `clause_patch(c)[plane]`, which would give a 3D array
  for this model. Add a `depth: Optional[int] = None` argument. `None` means show a montage of
  the `kd` slices, otherwise show one slice. Keep the 2D behaviour unchanged.
* `interpret.explain` relies on `input_shape` equality and `hasattr(model, "patch_size")`. Both
  work unchanged, but cover them with test T8.

### Step 8: tests (`tests/test_conv3d.py`, `device` fixture)

| id | what | proves |
|---|---|---|
| T1 | `_grid`, `conv_features` against the closed form on random shapes | Lemma 1 |
| T2 | `_encode` against explicit slicing loops, on random tensors **built on the device** | Lemma 2 |
| T3 | `D=kd=1`: literal tensors `torch.equal` to `ConvTsetlinMachine`; then same seed, 20 `update` steps ⇒ identical `ta_state` | Thm 3, Cor 3a |
| T4 | position bits: decode every patch's thermometer back to `(u,v,w)` | §1 encoding |
| T5 | `clause_region` round-trip: set TA states for a random box, decode it, compare | Lemma 5 |
| T6 | χ² test of the chunked draw against uniform on a hand-built match pattern; single-block path bitwise equals the unchunked path | Thm 4b, Lemma 8 |
| T7 | learns a 3D toy (§Step 9) to ≥ 0.95 with `position_encoding=False`; a depth-as-channels 2D CTM with the same clause budget scores lower | Prop 6, 7 |
| T8 | `model.eval()` empty-clause semantics, `state_dict` round-trip incl. lazy init, `explain` on one volume, coalesced and regression smoke tests | API parity |
| T9 | translation: shifting the volume by stride multiples leaves `forward` unchanged (no position bits) | Prop 6 |

Also run `ruff check src tests` and confirm that `tests/test_package.py` still passes, since it
checks exports.

### Step 9: toy data, example and docs

* `data/`: `make_3d_shapes(n, size=(8,16,16), ...)` generates a small cube, cross or bar placed
  at a random `(d, y, x)`. This is the 3D version of `make_shapes` and has a known
  translation-invariant ground truth.
* `examples/shapes_conv3d.py`: compare 3D CTM, depth-as-channels 2D CTM and flat TM under the
  same clause budget. This demonstrates Prop 7 empirically.
* `docs/concepts/convolution.md`: add a "3D volumes" section (patch tube, 3-axis thermometer,
  memory note). Add figures to `scripts/make_concept_figures.py` only if they help; do not
  hand-edit the SVGs.
* Changelog entry. No version bump in this step (versions live in two places; bump at release).

### Step 10: benchmark (optional, after merge)

Add a `conv3d` suite to `benchmarks/bench_device.py` using `PROG.plan(n)` / `PROG.item(label)`.
It should measure throughput vs `P`, stride and `patches_per_chunk` on CPU and GPU. Regenerate
`docs/cpu-vs-gpu.md` through `run_benchmarks.sh`, never by hand.

### Out of scope (follow-ups)

* **3D dense segmentation.** A `_Dense3dMixin` on top of `segmentation.py`. The same lemmas
  apply, but `B' = B·P` becomes very large for volumes, so `patches_per_commit` matters even
  more.
* **Anisotropic voxels.** MRI slice spacing is often much larger than the in-plane spacing.
  Handle it with `kd` and `sd` choices rather than resampling. No code change is needed, but the
  example should mention it.
* **2D+t with separate time semantics.** For example, a cyclic thermometer for cardiac phase.
  This is a position-encoding variant and only needs a different `_position_bits`.

---

## 5. Risks

| risk | mitigation |
|---|---|
| Memory blow-up on real volumes (§3) | Step 6; document stride guidance |
| Patch draw cost $O(n_\text{sel}\cdot P)$ dominates training | Step 6 blocks bound memory. Time is still linear in $P$, so benchmark it before advertising MRI-scale use |
| Position literals hurt translation-invariant tasks | Default `position_encoding=True` for parity with 2D, but tests and toys use `False` (CLAUDE.md caveat) |
| Batched-feedback fidelity drops when $P$ is large (more Type Ia events per commit) | Unchanged from 2D. Use `feedback_mode="sequential"` as the reference in T7 if batched results look off |
| Silent shape ambiguity for 4D input | Prefer an explicit `input_shape`. The error messages name the expected `(B, Z, D, H, W)` |

## 6. References

* O.-C. Granmo et al., *The Convolutional Tsetlin Machine*, arXiv:1905.09688 (2019): patch
  semantics, random matching patch and thermometer position features.
* `src/torchtsetlin/models/conv.py` (2D reference implementation) and
  `models/segmentation.py` (position-table caching, dense variant).
* `docs/concepts/convolution.md` and `docs/concepts/batching.md`.

---

## 7. Implementation notes (2026-10-12)

What was built: `models/conv.py::_Conv3dMixin` and the three `Conv3d*` classes;
`data.make_3d_shapes`; 3D support in `viz.plot_conv_clause(s)`; `tests/test_conv3d.py`
(T1–T9, 37 cases on CPU + CUDA); `examples/shapes_conv3d.py`; the "3D volumes" section and
two figures (`conv3d-sliding`, `conv3d-depth`) in `docs/concepts/convolution.md`; and a
`conv3d` benchmark suite. Where the measurements disagree with the plan, the measurements
win:

* **Step 6 saves less memory than §3 suggests.** The `3PC` term in the chunk estimate assumes
  every clause draws a patch for every example. In a real update far fewer events are
  selected. Measured on an RTX 3090 (12×64×64, 3×7×7, 2 000 clauses, batch 2): peak memory
  was **450 MB in one block against 369 MB with automatic blocking**, an 18% saving, not a
  1 GB → 0.5 GB one. With 200 clauses, blocking buys nothing, because ordinary batch
  chunking already keeps the update within `max_chunk_elements`. In that regime it can even
  raise peak memory, since a smaller per-example estimate lets more examples share a chunk.
  The literal tensor `(B, P, 2F)` and the bool match tensor set the floor.
* **The best default lever is stride, not blocking.** `stride=(1, 2, 2)` on the same volume
  is 3.0× faster (GPU training).
* **Toy task (Step 9).** `make_3d_shapes` uses shapes that differ *only* in slice order
  (spindle = plus/dot/plus, 3D cross = dot/plus/dot). The 3D model scored 1.000 on 5/5 seeds.
  Stacking slices as channels scored 0.83–0.98, so its gap varies a lot from seed to seed,
  and a flat machine scored 0.73–0.80. T7 therefore asserts 3D ≥ 0.95 and flat < 3D − 0.1,
  but **not** the stacked-channels gap, which would make the test flaky. That comparison
  lives in the example and the docs.
* `position_encoding=True` drops the toy to 0.48, the same caveat as 2D.
* The default `patch_size` is `(3, 7, 7)`, as planned, so pass `patch_size` for small volumes.
