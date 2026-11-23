"""The benchmark's metrics for a torchtsetlin model, computed on the GPU in two chunked passes
over the clause outputs (so a 2 x 10 000-clause model does not need an (N, C) matrix).

``checks.py`` M1 compares this with ``metrics_np`` (the reference script's formulas).
"""

import numpy as np
import torch
from metrics_np import summarise


def clause_targets(model):
    """The label each clause votes for: its class for a positive clause, the other class for a
    negative one (two classes)."""
    assert model.n_classes == 2
    cls = model.clause_class.long()
    return torch.where(model.clause_polarity > 0, cls, 1 - cls)


@torch.no_grad()
def evaluate(model, x, y, chunk=4000, with_clauses=True):
    """Accuracy and (optionally) the clause metrics on one data set, in prediction mode.
    Returns a dict of floats (percentages), plus the raw arrays under ``_arrays`` for tests."""
    was_training = model.training
    model.eval()
    try:
        target = clause_targets(model)
        C, N = target.numel(), x.shape[0]
        match = torch.zeros(C, dtype=torch.float64, device=x.device)
        agree = torch.zeros(C, dtype=torch.float64, device=x.device)
        votes = torch.zeros(N, dtype=torch.long, device=x.device)
        correct = 0
        for i in range(0, N, chunk):
            xb, yb = x[i:i + chunk], y[i:i + chunk]
            correct += int((model(xb).argmax(1) == yb).sum())
            if with_clauses:
                co = model.evaluate_clauses(xb)                            # (B, C) bool
                match += co.sum(0)
                agree += (co & (yb.view(-1, 1) == target.view(1, -1))).sum(0)
                votes[i:i + chunk] = co.sum(1)
        out = {"acc": 100.0 * correct / N}
        if not with_clauses:
            return out
        precision = torch.where(match > 0, agree / match.clamp(min=1), torch.zeros_like(match))
        n_target = torch.stack([(y == 0).sum(), (y == 1).sum()]).double()[target].clamp(min=1)
        recall = agree / n_target
        max_precision = torch.zeros(N, dtype=torch.float64, device=x.device)
        for i in range(0, N, chunk):                                       # second pass: needs the precisions
            co = model.evaluate_clauses(x[i:i + chunk])
            max_precision[i:i + chunk] = torch.where(co, precision.view(1, -1), torch.zeros((), dtype=torch.float64,
                                                                                            device=x.device)).amax(1)
        arrays = tuple(t.cpu().numpy() for t in (precision, recall, votes, max_precision))
        out.update(summarise(*arrays))
        out["_arrays"] = arrays
        return out
    finally:
        model.train(was_training)


def naive(model, x, y):
    """The same through ``metrics_np.clause_metrics`` on the full clause-output matrix (check M1)."""
    from metrics_np import clause_metrics
    was_training = model.training
    model.eval()
    with torch.no_grad():
        co = model.evaluate_clauses(x).cpu().numpy()
    model.train(was_training)
    arrays = clause_metrics(co, y.cpu().numpy(), clause_targets(model).cpu().numpy())
    return summarise(*arrays), tuple(np.asarray(a) for a in arrays)
