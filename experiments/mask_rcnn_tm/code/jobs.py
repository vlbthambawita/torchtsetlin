"""The job graph (PLAN_KVASIR §3, §7, §12): which runs exist, in which phase, and what each waits for.

A job is a dict with a unique ``tag``; its record is ``results/runs/<tag>.json`` whatever phase
lists it, so a run needed by two phases is done once. ``needs`` names the tags that must be done
first. Later stages depend on *selection* jobs (``kind="select"``): they read finished records,
apply a pre-registered rule and write the chosen configuration, and the jobs that build on that
choice only appear once it exists. Every selection is mechanical and uses validation data only.

Phases (opened by hand through ``KV_PHASES``, a gate is read between them):

    price  3 epochs of every head and baseline (sym and alb), oracle ceilings K0b
    base   baselines (i), (ii-s), (iii), (v-b) and the dense TM (v-a)
    p1     stage A: single members of every head, the controls, baseline (iv) heads
    p2     stage B (wide heads), stage C (capacity), seeds for the best configurations, K3
    alb    the AlbumentationsX arm on the final configuration of every head
    p3     proposals (out-of-fold), stage 2 retrained on them, end-to-end on validation
    final  cross-validation, oracle swaps, the sealed evaluation
"""

import copy
import os

from common import is_done, read_json, record_path

PHASES = ["price", "base", "p1", "p2", "alb", "p3", "final"]
SEEDS3, SEEDS5 = [0, 1, 2], [0, 1, 2, 3, 4]
WIDE = (1, 2, 4, 8, 16, 32)

DEF = {
    "rpn_obj": {"enc": "T4", "conv": "flat", "patch": 3, "scales": list(WIDE), "clauses": 300, "T": 60, "s": 10.0},
    "rpn_delta": {"enc": "T4", "conv": "ml", "patch": 3, "scales": list(WIDE), "clauses": 1200, "T": 120, "s": 10.0},
    "cls": {"enc": "T4", "conv": "conv2d", "patch": 5, "crop": 16, "factor": 1.0, "clauses": 1000, "T": 100, "s": 10.0},
    "box": {"enc": "T4", "conv": "convml", "patch": 5, "crop": 32, "factor": 1.5, "validity": True, "clauses": 2000,
            "T": 200, "s": 10.0},
    "mask": {"enc": "T4", "conv": "flat", "patch": 3, "scales": [1, 2, 4, 8], "pos": True, "clauses": 300, "T": 60,
             "s": 10.0},
    # the segmentation programme's tuned Kvasir configuration (configs/e4kv.json, wide pyramid)
    "dense": {"enc": "T4", "conv": "flat", "patch": 3, "scales": list(WIDE), "clauses": 600, "T": 120, "s": 20.0},
}
JOINT_HEADS = ("rpn_obj", "cls", "mask")             # two-class heads: the joint arm exists
TEAM_HEADS = ("rpn_obj", "cls", "mask", "box")       # heads whose members can be added
SEARCH_HEADS = ("rpn_obj", "rpn_delta", "cls", "box", "mask")
BASELINES = ("mrcnn_i", "mrcnn_iis", "mrcnn_iii", "unet")


def seeds(default):
    env = os.environ.get("KV_SEEDS")
    return [int(v) for v in env.split(",")] if env else list(default)


# --------------------------------------------------------------------------- tags
def mdesc(m):
    if "scales" in m:
        d = f"{m['enc']}.p{m['patch']}x{max(m['scales'])}"
    else:
        d = f"{m['enc']}.k{m['patch']}c{m['crop']}" + (f"d{m['kd']}" if m["conv"] == "conv3d" else "")
    if m["conv"] == "grad":
        return d + ".grad"
    d += f".n{m['clauses']}.T{float(m['T']):g}"
    if float(m.get("s", 10.0)) != 10.0:
        d += f".s{float(m['s']):g}"
    if m.get("pos") is False:
        d += ".nopos"
    return d


def tag_of(job):
    t = f"{job['head']}__{'+'.join(mdesc(m) for m in job['members'])}"
    if job.get("combine", "single") != "single":
        t += "__" + {"joint": "joint", "composite": "comp"}[job["combine"]]
    b = job.get("bins")
    if b and (float(b.get("width", 0.1)) != 0.1 or not b.get("ordinal", True)):
        t += f"__w{float(b.get('width', 0.1)):g}{'o' if b.get('ordinal', True) else 'b'}"
    if job.get("per_shape"):
        t += "__perA"
    if job.get("aug", "sym") != "sym":
        t += f"__{job['aug']}"
    if job.get("src", "gt") != "gt":
        t += f"__on-{job['src']}"
    f = job.get("fold") or {}
    if f.get("cv") is not None:
        t += f"__cv{f['cv']}"
    if f.get("oof") is not None:
        t += f"__oof{f['oof']}"
    return job.get("prefix", "") + t + f"__s{job['seed']}"


def J(phase, head, members, seed=0, **kw):
    job = {"phase": phase, "kind": "train", "head": head, "members": copy.deepcopy(members), "combine": "single",
           "seed": int(seed), "aug": "sym", "src": "gt", "needs": []}
    job.update({k: v for k, v in kw.items() if v is not None})
    job["tag"] = tag_of(job)
    return job


def cfg_of(members, combine="single", **extra):
    return {"members": copy.deepcopy(members), "combine": combine, **extra}


def runs_for(phase, head, cfg, seed, **kw):
    """The jobs that realise configuration ``cfg`` at ``seed``, and the tag of the result. A
    composite is one training run per member plus a combine job."""
    extra = {k: v for k, v in cfg.items() if k not in ("members", "combine")}
    extra.update(kw)
    if cfg["combine"] != "composite":
        j = J(phase, head, cfg["members"], seed, combine=cfg["combine"], **extra)
        return [j], j["tag"]
    panel_refs = extra.pop("pipeline", None)
    js = [J(phase, head, [m], seed, pipeline=panel_refs, **extra) for m in cfg["members"]]
    c = J(phase, head, cfg["members"], seed, combine="composite", **extra)
    c.update(kind="combine", member_runs=[j["tag"] for j in js], needs=[j["tag"] for j in js] + list(extra.get("needs", [])))
    return js + [c], c["tag"]


def scaled(cfg, f):
    out = copy.deepcopy(cfg)
    for m in out["members"]:
        m["clauses"] = max(10, int(round(m["clauses"] * f)))
        m["T"] = max(2.0, round(float(m["T"]) * f))
    return out


def with_s(cfg, s):
    out = copy.deepcopy(cfg)
    for m in out["members"]:
        m["s"] = float(s)
    return out


def select_job(phase, name, rule, candidates, **kw):
    """``candidates``: list of ``{"name", "cfg", "tags"}`` (one tag per seed)."""
    needs = sorted({t for c in candidates for t in c["tags"]})
    return {"phase": phase, "kind": "select", "tag": f"select-{name}", "rule": rule, "candidates": candidates,
            "needs": needs, **kw}


def sel(name):
    r = read_json(record_path(f"select-{name}"))
    return r if r and r.get("status") == "done" else None


def baseline_job(phase, model, seed, aug="sym", fold=None, **kw):
    t = f"base-{model}" + (f"__{aug}" if aug != "sym" else "")
    f = fold or {}
    if f.get("cv") is not None:
        t += f"__cv{f['cv']}"
    job = {"phase": phase, "kind": "baseline", "model": model, "seed": int(seed), "aug": aug, "fold": fold,
           "needs": [], **kw}
    job["tag"] = kw.get("prefix", "") + t + f"__s{seed}"
    return job


# --------------------------------------------------------------------------- stage A lists
def stage_a(head):
    """``(members, extra)`` of every stage-A run of a head (seed 0)."""
    D = DEF[head]
    out = []
    if head == "mask":
        for enc in ("T4", "T8", "G8", "AG", "T4+AG"):
            for patch, sc in ((3, [1, 2, 4, 8]), (5, [1, 2, 4]), (3, [1, 2, 4, 8, 16])):
                out.append(([dict(D, enc=enc, patch=patch, scales=sc)], {}))
    elif head == "cls":
        for enc in ("T4", "T8", "AG", "T4+AG"):
            for patch in (3, 5, 7, 10):
                out.append(([dict(D, enc=enc, patch=patch)], {}))
        for enc in ("T8", "G8"):
            for kd in (2, 4):
                out.append(([dict(D, enc=enc, conv="conv3d", kd=kd, patch=5)], {}))
    elif head == "rpn_obj":
        for enc in ("T4", "T8", "AG", "T4+AG"):
            for patch in (3, 5):
                out.append(([dict(D, enc=enc, patch=patch)], {}))
    elif head == "rpn_delta":
        base = sel("rpn_obj-A")
        if base is None:
            return None
        f = base["best"]["members"][0]
        m = dict(D, enc=f["enc"], patch=f["patch"], scales=f["scales"])
        for width in (0.2, 0.1, 0.05):
            for ordinal in (True, False):
                out.append(([m], {"bins": {"width": width, "ordinal": ordinal}}))
    elif head == "box":
        for enc in ("T4", "T4+AG"):
            for patch in (5, 7):
                out.append(([dict(D, enc=enc, patch=patch)], {}))
    return out


def cand(name, cfg, tags):
    return {"name": name, "cfg": cfg, "tags": list(tags)}


def chain(head):
    """Jobs of the search of one head, as far as the selections so far allow:
    ``{"p1": [...], "p2": [...], "alb": [...]}`` and the reasons a later stage is not listed yet."""
    out, blocked = {"p1": [], "p2": [], "alb": []}, []
    A = stage_a(head)
    if A is None:
        return out, [f"{head}: stage A waits for select-rpn_obj-A"]
    cands = []
    for members, extra in A:
        j = J("p1", head, members, 0, stage="A", **extra)
        out["p1"].append(j)
        cands.append(cand(j["tag"], cfg_of(members, **extra), [j["tag"]]))
    if head == "cls":                                # the best three again on the 24 x 24 crop of box x 1.5
        out["p1"].append(select_job("p1", "cls-A1", "rank", cands))
        a1 = sel("cls-A1")
        if a1 is None:
            return out, [f"{head}: the 24 x 24 crops wait for select-cls-A1"]
        for c in a1["top"][:3]:
            m = [dict(c["members"][0], crop=24, factor=1.5)]
            j = J("p1", head, m, 0, stage="A")
            out["p1"].append(j)
            cands.append(cand(j["tag"], cfg_of(m), [j["tag"]]))
    out["p1"].append(select_job("p1", f"{head}-A", "families", cands))
    a = sel(f"{head}-A")
    if a is None:
        return out, [f"{head}: stage B waits for select-{head}-A"]

    # ---- stage B: wide heads -------------------------------------------------
    best, fam = a["best"], a["families"]
    extra = {k: v for k, v in best.items() if k not in ("members", "combine")}
    if head in TEAM_HEADS and len(fam) >= 2:
        bc = {"single": cfg_of(best["members"], **extra)}
        M = min(3, len(fam))
        bc["team2"] = cfg_of([f["members"][0] for f in fam[:2]], "composite", **extra)
        if M == 3:
            bc["team3"] = cfg_of([f["members"][0] for f in fam[:3]], "composite", **extra)
        if head in JOINT_HEADS:
            kinds = {f["members"][0]["conv"] for f in fam[:M]}
            if kinds <= {"flat"} or kinds <= {"conv2d"}:
                bc["joint"] = cfg_of([f["members"][0] for f in fam[:M]], "joint", **extra)
        bc["W"] = scaled(cfg_of(best["members"], **extra), M)
        cands = []
        for name, cfg in bc.items():
            tags = []
            for s in SEEDS3:
                js, t = runs_for("p2", head, cfg, s, stage="B")
                out["p2"] += js
                tags.append(t)
            cands.append(cand(name, cfg, tags))
        out["p2"].append(select_job("p2", f"{head}-B", "team", cands))
        b = sel(f"{head}-B")
        if b is None:
            return out, [f"{head}: stage C waits for select-{head}-B"]
        base = b["best"]
    else:
        base = cfg_of(best["members"], **extra)

    # ---- stage C: capacity, then s ------------------------------------------
    cands = []
    for f in (0.5, 1, 2, 4):
        cfg = base if f == 1 else scaled(base, f)
        js, t = runs_for("p2", head, cfg, 0, stage="C")
        out["p2"] += js
        cands.append(cand(f"x{f:g}", cfg, [t]))
    out["p2"].append(select_job("p2", f"{head}-C1", "rank", cands))
    c1 = sel(f"{head}-C1")
    if c1 is None:
        return out, [f"{head}: the s runs wait for select-{head}-C1"]
    for s_ in (5.0, 20.0):
        cfg = with_s(c1["best"], s_)
        js, t = runs_for("p2", head, cfg, 0, stage="C")
        out["p2"] += js
        cands.append(cand(f"s{s_:g}", cfg, [t]))
    out["p2"].append(select_job("p2", f"{head}-C", "rank", cands))
    c = sel(f"{head}-C")
    if c is None:
        return out, [f"{head}: the seeds wait for select-{head}-C"]

    # ---- the best two on three seeds, the finalist on five ---------------------
    cands = []
    for i, cfg in enumerate(c["top"][:2]):
        tags = []
        for s in SEEDS3:
            js, t = runs_for("p2", head, cfg, s, stage="D")
            out["p2"] += js
            tags.append(t)
        cands.append(cand(f"top{i + 1}", cfg, tags))
    out["p2"].append(select_job("p2", f"{head}-D", "rank", cands))
    d = sel(f"{head}-D")
    if d is None:
        return out, [f"{head}: the finalist waits for select-{head}-D"]
    final = d["best"]
    for s in SEEDS5:
        js, _ = runs_for("p2", head, final, s, stage="final")
        out["p2"] += js

    # ---- the AlbumentationsX arm ------------------------------------------------
    cands = [cand("sym", final, [runs_for("p2", head, final, s)[1] for s in SEEDS3])]
    tags = []
    for s in SEEDS3:
        js, t = runs_for("alb", head, final, s, stage="alb", aug="alb")
        out["alb"] += js
        tags.append(t)
    cands.append(cand("alb", final, tags))
    out["alb"].append(select_job("alb", f"{head}-aug", "wins_all", cands))
    if head == "mask":
        for part in ("alb-photo", "alb-geo"):
            out["alb"] += runs_for("alb", head, final, 0, stage="alb", aug=part)[0]
    return out, blocked


def final_cfg(head):
    """The final configuration of a head and its augmentation level, or ``None``."""
    if head == "dense":
        return cfg_of([DEF["dense"]]), "sym"
    d = sel(f"{head}-D")
    if d is None:
        return None
    a = sel(f"{head}-aug")
    return d["best"], (a["level"] if a else "sym")


def grad_cfg(head):
    return cfg_of([dict(DEF[head], conv="grad")])


# --------------------------------------------------------------------------- phases
def pipeline_jobs(phase, name, cfgs, seeds_, fold=None, variants=("full",), grad=False, stage="S6"):
    """Proposals and stage 2 for one set of head configurations ``cfgs`` (``{head: (cfg, aug)}``):
    the two RPN models on all training images and on each half (out of fold), the proposal
    cache, the three RoI heads retrained on the proposals, and the end-to-end evaluations."""
    out = []
    f0 = dict(fold or {})

    def rpn(seed, oof=None):
        refs = {}
        for h in ("rpn_obj", "rpn_delta"):
            cfg, aug = cfgs[h]
            fd = dict(f0, **({"oof": oof} if oof is not None else {})) or None
            js, t = runs_for(phase, h, cfg, seed, aug=aug, fold=fd, stage="rpn")
            out.extend(js)
            refs[h] = t
        return refs

    full = rpn(0)
    oof = [rpn(0, j) for j in (0, 1)]
    props = {"phase": phase, "kind": "props", "tag": f"props-{name}", "name": name, "full": full, "oof": oof,
             "fold": fold, "needs": list(full.values()) + [t for o in oof for t in o.values()]}
    out.append(props)
    for s in seeds_:
        refs = dict(full)
        for h in ("cls", "box", "mask"):
            cfg, aug = cfgs[h]
            extra = {"pipeline": dict(refs)} if h == "mask" else {}
            js, t = runs_for(phase, h, cfg, s, aug=aug, src=name, fold=fold, stage=stage,
                             needs=[props["tag"]] + (list(refs.values()) if h == "mask" else []), **extra)
            out.extend(js)
            refs[h] = t
        for v in variants:
            r = dict(refs)
            opts = {"full": {}, "norefine": {"refine": "none"}, "maskbox": {"refine": "mask"},
                    "objrank": {"rank": "obj"}, "densemask": {"mask_from": "dense"}}[v]
            if v == "densemask":
                r["dense"] = J("base", "dense", [DEF["dense"]], s, fold=fold)["tag"]
            out.append({"phase": phase, "kind": "e2e", "tag": f"e2e-{name}-{v}__s{s}", "name": name, "variant": v,
                        "refs": r, "opts": opts, "fold": fold, "seed": s, "needs": list(r.values()), "grad": grad})
    return out


def dense_e2e(phase, seed, fold=None, aug="sym"):
    d = J("base", "dense", [DEF["dense"]], seed, fold=fold, aug=aug)
    f = fold or {}
    name = "dense_cc" + (f"-{aug}" if aug != "sym" else "") + (f"-cv{f['cv']}" if f.get("cv") is not None else "")
    return {"phase": phase, "kind": "e2e", "tag": f"e2e-{name}__s{seed}", "name": name, "variant": "dense_cc",
            "refs": {"dense": d["tag"]}, "opts": {}, "fold": fold, "seed": seed, "needs": [d["tag"]]}


def jobs(phase):
    """``(jobs, blocked)`` of a phase: the jobs that can be listed now, and why others cannot."""
    out, blocked = [], []
    if phase == "price":
        out.append({"phase": "price", "kind": "oracle", "tag": "oracle-K0b", "needs": []})
        for head in ("rpn_obj", "rpn_delta", "cls", "box", "mask", "dense"):
            for aug in ("sym", "alb"):
                out.append(J("price", head, [DEF[head]], 0, aug=aug, prefix="price-", max_epochs=3, stage="price"))
        out.append(J("price", "mask", [dict(DEF["mask"], conv="grad")], 0, prefix="price-", max_epochs=3, stage="price"))
        for m in BASELINES:
            out.append(baseline_job("price", m, 0, prefix="price-", max_epochs=3))
    elif phase == "base":
        for m in BASELINES:
            for s in seeds(SEEDS3):
                out.append(baseline_job("base", m, s))
        out.append(baseline_job("base", "mrcnn_i", 0, schedule="steps"))        # the original step schedule, once
        out[-1]["tag"] = "base-mrcnn_i__steps__s0"
        for s in seeds(SEEDS5):
            out.append(J("base", "dense", [DEF["dense"]], s, stage="base"))
            out.append(dense_e2e("base", s))
    elif phase in ("p1", "p2", "alb"):
        for head in SEARCH_HEADS:
            js, bl = chain(head)
            out += js[phase]
            blocked += bl
        if phase == "p1":
            out.append(J("p1", "rpn_obj", [DEF["rpn_obj"]], 0, per_shape=True, stage="control"))         # RPN-A
            out.append(J("p1", "mask", [dict(DEF["mask"], pos=False)], 0, stage="control"))             # arm C
            for head in SEARCH_HEADS:                                                                   # baseline (iv)
                for s in seeds(SEEDS3):
                    out += runs_for("p1", head, grad_cfg(head), s, stage="iv")[0]
        if phase == "p2":
            fm = final_cfg("mask")
            if fm is None:
                blocked.append("K3 waits for select-mask-D")
            else:
                arms = {"A": [runs_for("p2", "mask", fm[0], s)[1] for s in SEEDS5]}
                nopos = copy.deepcopy(fm[0])
                for m in nopos["members"]:
                    m["pos"] = False
                arms["C"] = []
                for s in SEEDS5:
                    js, t = runs_for("p2", "mask", nopos, s, stage="K3")
                    out += js
                    arms["C"].append(t)
                arms["B"] = [J("base", "dense", [DEF["dense"]], s)["tag"] for s in SEEDS5]
                arms["iv"] = [runs_for("p1", "mask", grad_cfg("mask"), s)[1] for s in SEEDS3]
                out.append({"phase": "p2", "kind": "k3", "tag": "k3", "arms": arms,
                            "needs": sorted({t for v in arms.values() for t in v})})
        if phase == "alb":
            for s in seeds(SEEDS3):
                out.append(J("alb", "dense", [DEF["dense"]], s, aug="alb", stage="alb"))
                out.append(dense_e2e("alb", s, aug="alb"))
                for m in ("mrcnn_i", "mrcnn_iii"):
                    out.append(baseline_job("alb", m, s, aug="alb"))
    elif phase in ("p3", "final"):
        cfgs = {h: final_cfg(h) for h in SEARCH_HEADS}
        missing = [h for h, c in cfgs.items() if c is None]
        if missing:
            return out, [f"{phase} waits for select-{h}-D" for h in missing]
        variants = ("full", "norefine", "maskbox", "objrank", "densemask")
        main = pipeline_jobs("p3", "main", cfgs, SEEDS5, variants=variants)
        iv = pipeline_jobs("p3", "iv", {h: (grad_cfg(h), "sym") for h in SEARCH_HEADS}, SEEDS3, grad=True)
        cands = [cand(v, {"variant": v}, [f"e2e-main-{v}__s{s}" for s in SEEDS5]) for v in variants]
        pick = select_job("p3", "pipeline", "rank", cands, metric="mask_AP50")
        if phase == "p3":
            return main + iv + [pick], blocked
        p = sel("pipeline")
        if p is None:
            return out, ["final waits for select-pipeline"]
        v = p["best"]["variant"]
        for k in range(5):                           # cross-validation of the final arms
            fold = {"cv": k}
            out += pipeline_jobs("final", f"cv{k}", cfgs, [0], fold=fold, variants=(v,))
            out.append(J("final", "dense", [DEF["dense"]], 0, fold=fold, stage="cv"))
            out.append(dense_e2e("final", 0, fold=fold))
            out.append(baseline_job("final", "mrcnn_iii", 0, fold=fold))
        for o in ("proposals", "class", "box", "mask"):                         # oracle swaps on validation
            ref = next(j for j in main if j["tag"] == f"e2e-main-{v}__s0")
            out.append(dict(ref, phase="final", tag=f"e2e-main-{v}-oracle-{o}__s0", oracle=[o]))
        arms = [(f"tm-{v}", [f"e2e-main-{v}__s{s}" for s in SEEDS5]), ("iv", [f"e2e-iv-full__s{s}" for s in SEEDS3]),
                ("dense_cc", [f"e2e-dense_cc__s{s}" for s in SEEDS5])]
        for arm, tags in arms:
            for t in tags:
                out.append({"phase": "final", "kind": "sealed", "tag": f"sealed-{t}", "source": t, "arm": arm,
                            "needs": [t]})
        for m in BASELINES:
            for s in SEEDS3:
                b = baseline_job("base", m, s)
                out.append({"phase": "final", "kind": "sealed", "tag": f"sealed-{b['tag']}", "source": b["tag"],
                            "arm": m, "needs": [b["tag"]]})
    else:
        raise ValueError(f"unknown phase {phase!r}")
    seen, uniq = set(), []
    for j in out:                                    # a run listed twice is one run
        if j["tag"] not in seen:
            seen.add(j["tag"])
            uniq.append(j)
    return uniq, blocked


def smoke_jobs():
    """A tiny end-to-end pass through every kind of job (64 images, 2 epochs)."""
    kw = dict(prefix="smoke-", max_epochs=2, subset=64, stage="smoke")
    out = [J("smoke", h, [DEF[h]], 0, **kw) for h in ("rpn_obj", "rpn_delta", "cls", "box", "mask", "dense")]
    m2 = dict(DEF["mask"], patch=5, scales=[1, 2, 4])
    single = J("smoke", "mask", [m2], 0, **kw)
    out.append(single)
    out.append(J("smoke", "mask", [DEF["mask"], m2], 0, combine="joint", **kw))
    comp = J("smoke", "mask", [DEF["mask"], m2], 0, combine="composite", **kw)
    comp.update(kind="combine", member_runs=[out[4]["tag"], single["tag"]], needs=[out[4]["tag"], single["tag"]])
    out.append(comp)
    out.append(J("smoke", "cls", [dict(DEF["cls"], enc="T8", conv="conv3d", kd=2)], 0, **kw))
    out.append(J("smoke", "mask", [dict(DEF["mask"], conv="grad")], 0, **kw))
    out.append(J("smoke", "mask", [DEF["mask"]], 0, aug="alb", **kw))
    full = {"rpn_obj": out[0]["tag"], "rpn_delta": out[1]["tag"]}
    out.append({"phase": "smoke", "kind": "props", "tag": "props-smoke", "name": "smoke", "full": full,
                "oof": [full, full], "fold": None, "needs": list(full.values()), "subset": 64})
    refs = dict(full)
    for h in ("cls", "box", "mask"):
        extra = {"pipeline": dict(refs)} if h == "mask" else {}
        j = J("smoke", h, [DEF[h]], 0, src="smoke", needs=["props-smoke"] + list(refs.values()), **extra, **kw)
        out.append(j)
        refs[h] = j["tag"]
    out.append({"phase": "smoke", "kind": "e2e", "tag": "e2e-smoke-full__s0", "name": "smoke", "variant": "full",
                "refs": refs, "opts": {}, "fold": None, "seed": 0, "needs": list(refs.values()), "subset": 64})
    out.append({"phase": "smoke", "kind": "e2e", "tag": "e2e-smoke-dense_cc__s0", "name": "smoke", "variant": "dense_cc",
                "refs": {"dense": out[5]["tag"]}, "opts": {}, "fold": None, "seed": 0, "needs": [out[5]["tag"]], "subset": 64})
    out.append(select_job("smoke", "smoke", "rank", [cand(j["tag"], cfg_of(j["members"]), [j["tag"]])
                                                    for j in (out[4], single)]))
    for m in ("unet", "mrcnn_iis"):
        out.append(baseline_job("smoke", m, 0, prefix="smoke-", max_epochs=2, subset=64))
    return out


def all_jobs(phases=None):
    out, blocked, seen = [], [], set()
    for ph in phases or PHASES:
        js, bl = jobs(ph)
        for j in js:                                 # a run listed by two phases is one run
            if j["tag"] not in seen:
                seen.add(j["tag"])
                out.append(j)
        blocked += bl
    return out, blocked


def find(tag):
    for j in smoke_jobs():
        if j["tag"] == tag:
            return j
    for ph in PHASES:
        for j in jobs(ph)[0]:
            if j["tag"] == tag:
                return j
    raise SystemExit(f"no job with tag {tag!r} (it may wait for a selection that is not done yet)")


def runnable(job):
    return all(is_done(t) for t in job.get("needs", []))


if __name__ == "__main__":
    for ph in PHASES:
        js, bl = jobs(ph)
        kinds = {}
        for j in js:
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
        print(f"{ph:6s} {len(js):4d} jobs {kinds}" + (f"   blocked: {bl}" if bl else ""))
