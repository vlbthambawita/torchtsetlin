"""Verify the synthetic tasks carry no ADDITIVE-in-positions signal.

A single convolutional TM layer realises exactly threshold(f(pos_A) + g(pos_B)); if that
function class can already predict the label, the task does not test compositionality.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import synthetic as S


def slots_and_labels(n, task, seed):
    g = torch.Generator().manual_seed(seed)
    A, B, Y = [], [], []
    for _ in range(n):
        want = bool(torch.randint(0, 2, (1,), generator=g))
        gaps = S.NEAR_GAP if want else S.FAR_GAP
        sa = int(torch.randint(0, S.N_SLOTS, (1,), generator=g))
        gap = gaps[int(torch.randint(0, len(gaps), (1,), generator=g))]
        sign = 1 if bool(torch.randint(0, 2, (1,), generator=g)) else -1
        A.append(sa); B.append((sa + sign * gap) % S.N_SLOTS); Y.append(int(want))
    return torch.tensor(A), torch.tensor(B), torch.tensor(Y).float()


def onehot(P, k):
    M = torch.zeros(len(P), k); M[torch.arange(len(P)), P] = 1.0
    return M


def main():
    n = 20000
    A, B, Y = slots_and_labels(n, "near", 0)
    print(f"balance {float(Y.mean()):.3f}")
    for cls in (0, 1):
        h = torch.bincount(A[Y == cls], minlength=S.N_SLOTS).float()
        print(f"  class {cls} slot marginal (A): min {h.min()/h.sum():.4f} max {h.max()/h.sum():.4f}")
    X = torch.cat([onehot(A, S.N_SLOTS), onehot(B, S.N_SLOTS)], dim=1)
    ntr = int(0.7 * n)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(800):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X[:ntr] @ w + b, Y[:ntr])
        loss.backward(); opt.step()
    acc = (((X[ntr:] @ w + b) > 0).float() == Y[ntr:]).float().mean()
    print(f"best additive-in-positions predictor (single-layer ceiling): {float(acc)*100:.2f}%")
    gap = torch.tensor([min(abs(int(a) - int(bb)) % S.N_SLOTS,
                            S.N_SLOTS - abs(int(a) - int(bb)) % S.N_SLOTS)
                        for a, bb in zip(A, B)]).float()
    print(f"ring-gap threshold (the JOINT condition): "
          f"{float((((gap <= max(S.NEAR_GAP)).float()) == Y).float().mean())*100:.2f}%")


if __name__ == "__main__":
    main()
