"""Synthetic datasets with *known* conditional probabilities.

The whole point of gate G0 is that ``P(y=1 | context)`` is known exactly, so calibration
error is measured, not estimated from finite-sample bins.
"""
from __future__ import annotations

import torch

P_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def distinct_patterns(n: int, n_features: int, generator: torch.Generator, device) -> torch.Tensor:
    """``n`` distinct Boolean patterns of width ``n_features`` as a (n, F) bool tensor."""
    if n_features <= 22 and n <= (1 << n_features):
        idx = torch.randperm(1 << n_features, generator=generator, device="cpu")[:n]
        bits = ((idx.unsqueeze(1) >> torch.arange(n_features)) & 1).bool()
        return bits.to(device)
    seen, out = set(), []
    while len(out) < n:
        cand = torch.randint(0, 2, (n * 2, n_features), generator=generator, dtype=torch.uint8)
        for row in cand:
            key = bytes(row.numpy())
            if key not in seen:
                seen.add(key)
                out.append(row)
                if len(out) == n:
                    break
    return torch.stack(out).bool().to(device)


class BernoulliContexts:
    """``n_contexts`` distinct contexts, each with its own known ``P(y=1)``."""

    def __init__(self, n_features, n_contexts, seed, device, p_levels=P_LEVELS):
        g = torch.Generator().manual_seed(seed)
        self.contexts = distinct_patterns(n_contexts, n_features, g, device)
        lv = torch.tensor(p_levels, dtype=torch.float32)
        self.p_true = lv[torch.arange(n_contexts) % len(p_levels)].to(device)
        self.n_features, self.n_contexts, self.device = n_features, n_contexts, device

    def sample(self, n, generator):
        idx = torch.randint(0, self.n_contexts, (n,), generator=generator, device=self.device)
        x = self.contexts[idx]
        u = torch.rand(n, generator=generator, device=self.device)
        return x, (u < self.p_true[idx]).long()


class SignalPlusNoise:
    """``P(y=1)`` depends only on the first ``n_signal`` bits; ``n_noise`` bits are irrelevant.

    This is the MNIST window-vs-canvas question in miniature: does padding the context with
    uninformative bits destroy calibration?
    """

    def __init__(self, n_signal, n_noise, seed, device, p_levels=P_LEVELS):
        g = torch.Generator().manual_seed(seed)
        self.n_signal, self.n_noise = n_signal, n_noise
        self.n_features = n_signal + n_noise
        self.n_groups = 1 << n_signal
        lv = torch.tensor(p_levels, dtype=torch.float32)
        perm = torch.randperm(self.n_groups, generator=g)
        self.p_group = lv[perm % len(p_levels)].to(device)
        self.device = device
        self._pow = (1 << torch.arange(n_signal, device=device)).long()

    def _group_of(self, sig):
        return (sig.long() * self._pow).sum(dim=1)

    def sample(self, n, generator):
        sig = torch.randint(0, 2, (n, self.n_signal), generator=generator, device=self.device).bool()
        x = sig
        if self.n_noise:
            noise = torch.randint(
                0, 2, (n, self.n_noise), generator=generator, device=self.device
            ).bool()
            x = torch.cat([sig, noise], dim=1)
        p = self.p_group[self._group_of(sig)]
        u = torch.rand(n, generator=generator, device=self.device)
        return x, (u < p).long()

    def probes(self, n_completions, generator):
        """Every signal pattern crossed with random noise completions -> (X, p_true, group)."""
        sig = ((torch.arange(self.n_groups, device=self.device).unsqueeze(1) >> torch.arange(
            self.n_signal, device=self.device)) & 1).bool()
        sig = sig.repeat_interleave(n_completions, dim=0)
        x = sig
        if self.n_noise:
            noise = torch.randint(
                0, 2, (sig.shape[0], self.n_noise), generator=generator, device=self.device
            ).bool()
            x = torch.cat([sig, noise], dim=1)
        grp = self._group_of(sig)
        return x, self.p_group[grp], grp


class CategoricalContexts:
    """``n_contexts`` contexts, each with a known sparse categorical distribution over ``K``."""

    def __init__(self, n_features, n_contexts, K, support, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.contexts = distinct_patterns(n_contexts, n_features, g, device)
        w = torch.tensor([0.5, 0.25, 0.15, 0.10])[:support]
        w = (w / w.sum()).to(device)
        probs = torch.zeros(n_contexts, K, device=device)
        for i in range(n_contexts):
            tok = torch.randperm(K, generator=g)[:support].to(device)
            probs[i, tok] = w
        self.p_true, self.K = probs, K
        self.n_features, self.n_contexts, self.device = n_features, n_contexts, device

    def sample(self, n, generator):
        idx = torch.randint(0, self.n_contexts, (n,), generator=generator, device=self.device)
        y = torch.multinomial(self.p_true[idx], 1, generator=generator).squeeze(1)
        return self.contexts[idx], y
