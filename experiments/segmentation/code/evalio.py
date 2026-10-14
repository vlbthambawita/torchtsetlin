"""Load a frozen run's checkpoint and predict any split / transformed inputs."""

from __future__ import annotations

import json
import os

import torch

import baselines
import record
import run as R


def load_run(run_id):
    with open(os.path.join(record.RESULTS, run_id + ".json")) as f:
        return json.load(f)


def load_model(rec, dev, d):
    cfg = {**R.DEFAULTS, **rec["config"]}
    K = R.K_OF[cfg["dataset"]]
    state = torch.load(os.path.join(record.HERE, rec["checkpoint"]["path"]))
    if cfg["family"] in ("tm", "coalesced"):
        m = R.build_tm(cfg, K, dev, None)
        ex = d["extra"][:1] if d.get("extra") is not None else None
        m._prepare(R.model_input(cfg, d["x"][:1], dev, ex))
        m.load_state_dict(state)
    else:
        in_ch = R.model_input(cfg, d["x"][:1], "cpu").shape[1]
        if cfg["family"] == "unet":
            m = baselines.UNet(in_ch, K, cfg["width"])
        elif cfg["family"] == "localcnn":
            m = baselines.LocalCNN(in_ch, K, 64, cfg.get("rf", 3))
        else:
            m = baselines.PixelLinear(in_ch, K)
        m = m.to(dev)
        m.load_state_dict(state)
    m.eval()
    return cfg, m


def predict(rec, split, dev, transform=None):
    """Scores for ``split`` (``transform(x_uint8, y) -> (x, y)`` applied first). Returns (cfg, d, scores)."""
    cfg = {**R.DEFAULTS, **rec["config"]}
    if split == "final":
        os.environ["SEG_UNSEAL"] = "1"
    d = dict(R.load_data(cfg, split))
    if transform is not None:
        d["x"], d["y"] = transform(d["x"], d["y"])
    if cfg.get("stack"):
        import hierarchy

        d["extra"] = hierarchy.deploy_features(cfg, rec, d, R.K_OF[cfg["dataset"]], dev)
    cfg, m = load_model(rec, dev, d)
    return cfg, d, R.predict_votes(cfg, m, d, dev)
