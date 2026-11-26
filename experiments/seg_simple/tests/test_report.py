"""The status, the cost table and the report run on fabricated records of every job (no GPU)."""

import random

import common
import jobs as JOBS
import report


def _fake(j, rng):
    m = lambda b: {"mIoU": b, "IoU_polyp": b - 0.2, "IoU_background": b + 0.1, "Dice": b, "pixel_acc": 0.9}   # noqa: E731
    if j["kind"] == "team":
        return {"tag": j["tag"], "status": "done", "members": j["members"][:2], "clauses_per_class": 800, "best": 0.6,
                "val": m(0.6 + rng.random() / 50), "test": m(0.59), "best_member_val_mIoU": 0.58,
                "mean_pairwise_disagreement": 0.05}
    b = 0.5 + 0.02 * (j["n"] > 100) + 0.01 * j["k"] ** 0.5 + rng.random() / 100
    return {"tag": j["tag"], "status": "done", "best": b, "best_epoch": 40, "epochs": 70, "val": m(b), "test": m(b - 0.01),
            "train": m(b + 0.1), "gap_train_minus_val": 0.1 + j["n"] / 1e5 + rng.random() / 100, "s_per_epoch": 12.0,
            "wall_s": 840.0, "history": [{"train_s": 9.0, "eval_s": 1.0}, {"train_s": 5.0, "eval_s": 1.0},
                                         {"train_s": 5.0, "eval_s": 1.0}]}


def test_status_price_and_report_on_fabricated_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(report, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(report, "PROG", tmp_path)
    monkeypatch.setattr(report, "CACHE", tmp_path / ".cache")
    report.status()                                                   # nothing recorded yet
    report.report()
    assert "not enough runs yet" in (tmp_path / "report" / "REPORT.md").read_text()
    rng = random.Random(0)
    for j in JOBS.all_jobs(JOBS.PHASES)[0]:
        common.write_json(_fake(j, rng), common.record_path(j["tag"]))
    report.status()
    out = capsys.readouterr().out
    assert "price complete" in out and "price3d complete" in out and "run(s) without a price" not in out
    assert f"{'main':8s} {54:4d} / {54:4d} done" in out
    report.report()
    text = (tmp_path / "report" / "REPORT.md").read_text()
    for h in ("H1:", "H2:", "H3:", "H4:", "H5:", "H6:", "H7 (T8, patch 3)", "H8:", "H9:", "H10:"):
        assert h in text, h
    assert "—" not in text.split("## Hypotheses")[0].split("## Side arms")[0]      # lines and grid are complete
    assert (tmp_path / "report" / "figures" / "clause_line.png").exists()
