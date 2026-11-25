"""The job graph, walked through every phase with fabricated records (no training): each
selection must make the next stage appear, tags must be unique, and every ``needs`` must name a
job that exists."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import common  # noqa: E402
import evals  # noqa: E402
import jobs as J  # noqa: E402


def test_the_graph_reaches_the_sealed_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    monkeypatch.setattr(common, "record_path", lambda tag: tmp_path / "runs" / f"{tag}.json")
    for mod in (J, evals):
        monkeypatch.setattr(mod, "record_path", common.record_path)
    monkeypatch.setattr(J, "is_done", lambda tag: (common.read_json(common.record_path(tag)) or {}).get("status") == "done")
    rng = random.Random(0)
    counts = {}
    for _ in range(40):                              # passes until nothing new appears
        new = 0
        js, blocked = J.all_jobs()
        tags = [j["tag"] for j in js]
        assert len(tags) == len(set(tags))
        known = set(tags)
        for j in js:
            assert all(t in known for t in j.get("needs", [])), (j["tag"], [t for t in j["needs"] if t not in known])
            if J.is_done(j["tag"]) or not J.runnable(j):
                continue
            new += 1
            if j["kind"] == "select":
                res = evals.do_select(j)
            else:
                v = rng.random()
                res = {"best": v, "metrics": {"mask_AP50": v, "semantic_threshold": 0.0}}
            common.write_json({"tag": j["tag"], "status": "done", "job": j, **res}, common.record_path(j["tag"]))
            counts[(j["phase"], j["kind"])] = counts.get((j["phase"], j["kind"]), 0) + 1
        if new == 0:
            break
    assert not blocked, blocked
    js, _ = J.all_jobs()
    assert all(J.is_done(j["tag"]) for j in js)
    per_phase = {ph: sum(v for (p, _), v in counts.items() if p == ph) for ph in J.PHASES}
    assert all(per_phase[ph] > 0 for ph in J.PHASES), per_phase
    assert counts[("final", "sealed")] >= 5 + 3 + 5 + 12 and counts[("p3", "props")] == 2
    for head in J.SEARCH_HEADS:                      # every head went through the whole search
        for st in ("A", "C1", "C", "D", "aug"):
            assert J.sel(f"{head}-{st}") is not None, (head, st)
    assert J.sel("pipeline")["best"]["variant"] in ("full", "norefine", "maskbox", "objrank", "densemask")
    screening = sum(v for (p, k), v in counts.items() if p in ("p1", "p2") and k == "train")
    print(per_phase, "train runs in p1+p2:", screening)


def test_tags_identify_runs_and_smoke_is_complete():
    a = J.J("p1", "mask", [J.DEF["mask"]], 0)
    b = J.J("p2", "mask", [J.DEF["mask"]], 0, stage="D")
    assert a["tag"] == b["tag"]                       # the same run, whatever phase lists it
    assert J.J("p1", "mask", [J.DEF["mask"]], 1)["tag"] != a["tag"]
    assert J.J("p1", "mask", [J.DEF["mask"]], 0, aug="alb")["tag"] != a["tag"]
    assert J.J("p1", "mask", [J.DEF["mask"]], 0, fold={"cv": 2, "oof": 1})["tag"].endswith("__cv2__oof1__s0")
    kinds = {j["kind"] for j in J.smoke_jobs()}
    assert kinds == {"train", "combine", "select", "props", "e2e", "baseline"}
