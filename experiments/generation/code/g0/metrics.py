"""Calibration metrics. ``p_true`` is known exactly, so the error is measured, not binned."""
from __future__ import annotations

import torch


@torch.no_grad()
def binary_calibration(v1, p_true, T):
    """Vote sums -> calibration summary against the analytic map p = (v + T) / 2T."""
    p_hat = ((v1 + T) / (2.0 * T)).clamp(0.0, 1.0)
    err = (p_hat - p_true).abs()
    x = 2.0 * p_true - 1.0
    y = v1 / T
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum().clamp_min(1e-12)
    slope = (xc * yc).sum() / denom
    intercept = y.mean() - slope * x.mean()
    resid = y - (slope * x + intercept)
    ss_tot = (yc * yc).sum().clamp_min(1e-12)
    return {
        "mae": err.mean().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "max_err": err.max().item(),
        "slope": slope.item(),
        "intercept": intercept.item(),
        "r2": (1.0 - (resid * resid).sum() / ss_tot).item(),
        "p_hat_mean": p_hat.mean().item(),
        "p_true_mean": p_true.mean().item(),
    }


@torch.no_grad()
def per_level(p_hat, p_true):
    """Mean p_hat at each distinct true level -> the reliability table."""
    out = []
    for lv in sorted(set(p_true.tolist())):
        m = p_true == lv
        out.append({
            "p_true": lv,
            "n": int(m.sum().item()),
            "p_hat_mean": p_hat[m].mean().item(),
            "p_hat_std": p_hat[m].std(unbiased=False).item(),
        })
    return out


@torch.no_grad()
def categorical_calibration(p_hat, p_true):
    """Full-distribution error over all K outputs, not just the argmax."""
    p_hat = p_hat.clamp_min(1e-9)
    p_hat = p_hat / p_hat.sum(dim=1, keepdim=True)
    mae_all = (p_hat - p_true).abs().mean(dim=1)
    tv = 0.5 * (p_hat - p_true).abs().sum(dim=1)
    sup = p_true > 0
    kl = (p_true[sup] * (p_true[sup].log() - p_hat[sup].log())).sum() / p_true.shape[0]
    return {
        "mae_all_outputs": mae_all.mean().item(),
        "total_variation": tv.mean().item(),
        "kl_true_to_hat": kl.item(),
        "top1_match": (p_hat.argmax(1) == p_true.argmax(1)).float().mean().item(),
        "support_mass": (p_hat * sup.float()).sum(dim=1).mean().item(),
        "offsupport_max": (p_hat * (~sup).float()).max().item(),
    }


# ------------------------------------------------------------------ fitted calibrators
def fit_calibrators(v, y):
    """Fit isotonic and Platt maps from vote sums to probabilities on *sampled* labels.

    This is the honest protocol: a real generator never sees ``p_true``, only outcomes.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    vn = v.detach().cpu().numpy().astype("float64")
    yn = y.detach().cpu().numpy().astype("float64")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(vn, yn)
    platt = LogisticRegression(max_iter=1000).fit(vn.reshape(-1, 1), yn.astype(int))

    def f_iso(x):
        return np.clip(iso.predict(x.detach().cpu().numpy().astype("float64")), 0.0, 1.0)

    def f_platt(x):
        return platt.predict_proba(
            x.detach().cpu().numpy().astype("float64").reshape(-1, 1))[:, 1]

    return f_iso, f_platt


def calibrated_errors(v_probe, p_true, T, f_iso, f_platt):
    """Analytic vs isotonic vs Platt, all against the known p_true."""
    import numpy as np
    pt = p_true.detach().cpu().numpy().astype("float64")
    ana = np.clip((v_probe.detach().cpu().numpy() + T) / (2.0 * T), 0.0, 1.0)
    out = {}
    for name, p in (("analytic", ana), ("isotonic", f_iso(v_probe)), ("platt", f_platt(v_probe))):
        e = np.abs(p - pt)
        out[f"mae_{name}"] = float(e.mean())
        out[f"max_err_{name}"] = float(e.max())
    out["vote_range"] = float(v_probe.max() - v_probe.min())
    out["vote_abs_max"] = float(v_probe.abs().max())
    return out
