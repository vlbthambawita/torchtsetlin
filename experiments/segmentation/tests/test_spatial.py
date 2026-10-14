"""H4 energy monotonicity / exhaustive check and H5 direct-solve / stability checks."""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import spatial as S  # noqa: E402


def _tiny(B=1, K=2, H=3, W=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, K, H, W, generator=g) * 3
    img = torch.rand(B, 3, H, W, generator=g)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    return scores, img, valid


def test_potts_energy_monotone_and_local_min():
    for seed in range(20):
        scores, img, valid = _tiny(K=3, H=6, W=7, seed=seed)
        lbl, hist = S.potts_icm(scores, img, valid, T=3.0, lam=1.0, sigma=0.3, max_sweeps=10, log=True)
        e = hist["energy"]
        assert all(b <= a + 1e-6 for a, b in zip(e, e[1:])), e


def test_potts_binary_exhaustive_on_tiny_graph():
    # ICM reaches a local optimum whose energy is never below the global optimum and is a
    # single-flip local minimum (verified by enumeration of all 2^9 labelings)
    for seed in range(10):
        scores, img, valid = _tiny(K=2, H=3, W=3, seed=seed)
        lam = 0.7
        lbl, _ = S.potts_icm(scores, img, valid, T=2.0, lam=lam, sigma=0.5, max_sweeps=50)
        U = -scores / 2.0
        wh, wv = S.edge_weights(img, 0.5, valid)
        e_icm = float(S.potts_energy(U, lbl, wh, wv, lam))
        best = min(
            float(S.potts_energy(U, torch.tensor(lab).view(1, 3, 3), wh, wv, lam))
            for lab in itertools.product([0, 1], repeat=9)
        )
        assert e_icm >= best - 1e-6
        for i in range(9):
            flip = lbl.clone().view(-1)
            flip[i] = 1 - flip[i]
            assert float(S.potts_energy(U, flip.view(1, 3, 3), wh, wv, lam)) >= e_icm - 1e-6


def test_lambda_zero_is_raw_argmax():
    scores, img, valid = _tiny(K=4, H=8, W=8)
    lbl, _ = S.potts_icm(scores, img, valid, T=1.0, lam=0.0, sigma=0.1)
    assert torch.equal(lbl, scores.argmax(1))


def _dense_L(wh, wv, H, W):
    n = H * W
    L = torch.zeros(n, n)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            if x + 1 < W:
                w = float(wh[0, y, x])
                j = i + 1
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w
            if y + 1 < H:
                w = float(wv[0, y, x])
                j = i + W
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w
    return L


def test_diffusion_matches_direct_solve_and_stays_a_distribution():
    scores, img, valid = _tiny(K=3, H=4, W=5, seed=3)
    lam = 1.0
    wh, wv = S.edge_weights(img, 0.3, valid)
    L = _dense_L(wh, wv, 4, 5)
    P = torch.softmax(scores / 2.0, 1)
    direct = torch.linalg.solve(torch.eye(20) + lam * L, P[0].reshape(3, -1).t()).t().reshape(1, 3, 4, 5)
    for method, it in (("explicit", 5000), ("cg", 200)):
        Q, info = S.diffuse(scores, img, valid, T=2.0, lam=lam, sigma=0.3, method=method, max_iter=it, tol=1e-6)
        assert info["converged"], (method, info)
        assert torch.allclose(Q, direct, atol=1e-4), method
    Q, _ = S.diffuse(scores, img, valid, T=2.0, lam=10.0, sigma=0.3, method="explicit", max_iter=30)
    assert bool((Q >= -1e-7).all())
    assert torch.allclose(Q.sum(1), torch.ones(1, 4, 5), atol=1e-5)


def test_explicit_objective_decreases():
    scores, img, valid = _tiny(K=2, H=5, W=5, seed=4)
    lam = 2.0
    wh, wv = S.edge_weights(img, None, valid)
    P = torch.softmax(scores, 1)

    def obj(Q):
        return 0.5 * ((Q - P) ** 2).sum() + 0.5 * lam * (Q * S.laplacian_apply(Q, wh, wv)).sum()

    prev = float(obj(P))
    for it in (1, 2, 5, 10, 20):
        Q, _ = S.diffuse(scores, img, valid, T=1.0, lam=lam, sigma=None, method="explicit", max_iter=it, tol=0)
        cur = float(obj(Q))
        assert cur <= prev + 1e-6
        prev = cur
