"""Add inference interventions to completed N0 records that lack them (from the saved checkpoint;
no retraining). Marked ``interventions_added_posthoc``. Test access is logged."""
import sys

import torch

from . import data as D
from .common import ART, RESULTS, read_json, write_json
from .models import TMQKAttention
from .run import INTERVENTIONS
from .training import metrics_from_votes


def main(stage: str, device: str = "cuda") -> None:
    va = D.load_split(D.MAIN, "valid")
    te = None
    for p in sorted((RESULTS / stage).glob("N0_*.json")):
        r = read_json(p)
        if r.get("status") != "completed" or "interventions_test" in r:
            continue
        if te is None:
            te = D.load_split(D.MAIN, "test", unseal=True, purpose=f"{stage}: post-hoc interventions")
        m = TMQKAttention.load(ART.parent / r["checkpoint"], device)
        for part, name in ((va, "valid"), (te, "test")):
            y = torch.as_tensor(part.y).to(device)
            r[f"interventions_{name}"] = {k: metrics_from_votes(m.votes_array(part.images, k), y)["accuracy"] for k in INTERVENTIONS}
        assert abs(metrics_from_votes(m.votes_array(te.images), torch.as_tensor(te.y).to(device))["accuracy"] - r["test"]["accuracy"]) < 1e-12
        r["interventions_added_posthoc"] = True
        write_json(p, r)
        print(r["run_id"], r["interventions_test"])


if __name__ == "__main__":
    main(sys.argv[1])
