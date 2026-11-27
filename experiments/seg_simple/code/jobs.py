"""The job list of the programme (plan sections 5 and 12). One dict per job; ``tag`` names its record.

Phases, in the order they are meant to run:
    price     3 epochs of every distinct dense shape (seconds per epoch on this GPU)
    main      clause line, patch line and grid, 3 seeds                      (54 runs)
    side      arms T, nopos, bal, shuffle, sym, bits, t8, px128              (113 runs)
    wide      extra member seeds, then the teams (evaluation only)
    price3d   3 epochs of every distinct window-model shape
    conv3d    dense references, 2D and 3D window models                      (69 runs)
    improve   pixels sampled across images, wide OR pyramid, adaptive-threshold bits (15 runs)

Tags: ``n<n>.k<k>[.<arm>...]__s<seed>`` for dense models, ``win2d.<enc>.k<k>.n<n>__s<seed>`` and
``win3d.<enc>.k<k>.d<kd>.n<n>__s<seed>`` for window models, ``pool.<enc>.s<largest scale>.n<n>__s<seed>`` for the
models of phase ``improve``, ``team-<name>__<repeat>`` for teams.
"""

from common import SAMPLED, WINDOW, is_done
from data import DEPTH

PHASES = ["price", "main", "side", "wide", "price3d", "conv3d", "improve"]
SEEDS = (0, 1, 2)
CLAUSE_LINE = [(n, 3) for n in (50, 100, 200, 400, 800, 1600, 3200)]
PATCH_LINE = [(400, k) for k in (1, 3, 5, 7, 9, 13)]
GRID_N, GRID_K = (100, 400, 1600), (1, 3, 5, 9)
GRID = [(n, k) for n in GRID_N for k in GRID_K]
CELLS = sorted(set(CLAUSE_LINE + PATCH_LINE + GRID))                 # 18
ARM_N = (100, 400, 1600)                                             # side arms at k = 3
WIN_K, WIN_KD, WIN_ENC, WIN_N = (3, 5, 7), (2, 4), ("T8", "B8"), 400
WIDE = (1, 2, 4, 8, 16, 32)                                          # scales of the wide OR pyramid
POOL = dict(pool_n=512, pool_a=0.25, pool_b=0.25)                    # pixels per image and epoch; class and boundary shares
IMPROVE_N = 400
IMPROVE_ARMS = (("T4", (1,)), ("T4", WIDE), ("AG", WIDE), ("T4+AG", WIDE), ("T4+AG", (1,)))
ASSUMED_EPOCHS = {"none": 150, "sym": 200}                           # cost projection only


def J(phase, group, model="dense", n=400, k=3, kd=None, enc="T4", pos=True, t_ratio=0.2, aug="none", bal=False,
      order="raster", sampled=False, px=64, seed=0, prefix="", scales=(1,), **kw):
    if model == "pool":
        t = f"pool.{enc}.s{max(scales)}.n{n}" + ("" if pos else ".nopos") + (".sym" if aug == "sym" else "")
        k, kw = 3, {**POOL, "scales": list(scales), **kw}
    elif model == "dense":
        t = f"n{n}.k{k}" + (f".{enc}" if enc != "T4" else "") + ("" if pos else ".nopos") \
            + (f".T{t_ratio}" if t_ratio != 0.2 else "") + (".sym" if aug == "sym" else "") + (".bal" if bal else "") \
            + (".shuffle" if order == "shuffle" else "") + (".sampled" if sampled else "") + (f".px{px}" if px != 64 else "")
    else:
        t = f"{model}.{enc}.k{k}" + (f".d{kd}" if model == "win3d" else "") + f".n{n}"
    job = {"phase": phase, "group": group, "kind": "train", "model": model, "n": n, "k": k, "kd": kd, "enc": enc,
           "pos": pos, "t_ratio": t_ratio, "aug": aug, "bal": bal, "order": order, "sampled": sampled, "px": px,
           "seed": int(seed), "needs": [], **kw}
    job["cell"] = prefix + t
    job["tag"] = f"{prefix}{t}__s{seed}"
    return job


def features(j):
    D = DEPTH[j["enc"]]
    if j["model"] == "dense":
        return 3 * D * j["k"] ** 2 + (16 if j["pos"] else 0)
    if j["model"] == "pool":
        return 3 * D * 9 * len(j["scales"]) + (16 if j["pos"] else 0)
    kd = j["kd"] if j["model"] == "win3d" else D
    return 3 * kd * j["k"] ** 2


def cost(j):
    """Relative cost of one epoch (ordering only; the pricing phase measures the real one)."""
    if j["kind"] != "train":
        return 0.0
    if j["model"] == "dense":
        return 2 * j["n"] * features(j) * j["px"] ** 2 * 700
    if j["model"] == "pool":
        return 2 * j["n"] * features(j) * (700 * j["pool_n"] + 150 * j["px"] ** 2)
    D = DEPTH[j["enc"]]
    kd = j["kd"] if j["model"] == "win3d" else D
    patches = (D - kd + 1) * (WINDOW - j["k"] + 1) ** 2
    return 2 * j["n"] * features(j) * patches * (700 * SAMPLED + 150 * j["px"] ** 2)


def price_key(j):
    return (j["model"], j["enc"], j["n"], j["k"], j["kd"], j["px"], j["sampled"], j["aug"], j["pos"])


def member(n, k, seed, **kw):
    return J("x", "x", n=n, k=k, seed=seed, **kw)["tag"]


def team(phase, name, repeat, members, **kw):
    return {"phase": phase, "group": "teams", "kind": "team", "name": name, "members": members, "needs": list(members),
            "cell": f"team-{name}", "tag": f"team-{name}__{repeat}", "seed": repeat, **kw}


def main_jobs():
    return [J("main", "clause-line" if (n, k) in CLAUSE_LINE else "patch-line" if (n, k) in PATCH_LINE else "grid",
              n=n, k=k, seed=s) for n, k in CELLS for s in SEEDS]


def side_jobs():
    out = []
    for s in SEEDS:
        for n in ARM_N:
            out += [J("side", "arm-T", n=n, t_ratio=r, seed=s) for r in (0.1, 0.4)]
            out += [J("side", "arm-nopos", n=n, pos=False, seed=s), J("side", "arm-bal", n=n, bal=True, seed=s),
                    J("side", "arm-shuffle", n=n, order="shuffle", seed=s)]
        out += [J("side", "arm-sym", n=n, k=k, aug="sym", seed=s) for n, k in sorted(set(CLAUSE_LINE + PATCH_LINE))]
        out += [J("side", "arm-bits", n=400, k=k, enc="B8", seed=s) for k in (1, 3, 5, 7, 9)]
        out += [J("side", "arm-bits", n=n, k=3, enc="B8", seed=s) for n in (100, 1600)]
        out += [J("side", "arm-t8", n=400, k=k, enc="T8", seed=s) for k in (3, 9)]
    out += [J("side", "arm-px128", n=n, k=3, px=128) for n in ARM_N]
    out += [J("side", "arm-px128", n=400, k=k, px=128) for k in (5, 9)]
    return out


TRIPLES = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def wide_jobs():
    out = [J("wide", "wide-members", n=400, k=k, seed=s) for k in (3, 5, 9) for s in range(3, 9)]
    for s in SEEDS:
        out += [team("wide", "W4x100", s, [member(100, k, s) for k in GRID_K]),
                team("wide", "W2x400", s, [member(400, k, s) for k in (3, 9)]),
                team("wide", "W4x400", s, [member(400, k, s) for k in GRID_K]),
                team("wide", "W6x400", s, [member(n, k, s) for n, k in PATCH_LINE]),
                team("wide", "W4x1600", s, [member(1600, k, s) for k in GRID_K]),
                team("wide", "W4x400.sym", s, [member(400, k, s, aug="sym") for k in GRID_K]),
                team("wide", "Wgreedy", s, [member(n, k, s) for n, k in CELLS], greedy=6)]
    for r, t in enumerate(TRIPLES):
        out.append(team("wide", "Wmix", r, [member(400, k, t[i]) for i, k in enumerate((3, 5, 9))]))
        out += [team("wide", f"Wsame-k{k}", r, [member(400, k, s) for s in t]) for k in (3, 5, 9)]
    return out


def conv3d_jobs():
    out = []
    for s in SEEDS:
        for e in WIN_ENC:
            out += [J("conv3d", "dense-ref", n=WIN_N, k=9, enc=e, pos=False, seed=s),
                    J("conv3d", "dense-ref", n=WIN_N, k=9, enc=e, pos=False, sampled=True, seed=s)]
            out += [J("conv3d", "win2d", model="win2d", n=WIN_N, k=k, enc=e, seed=s) for k in WIN_K]
            out += [J("conv3d", "win3d", model="win3d", n=WIN_N, k=k, kd=kd, enc=e, seed=s) for k in WIN_K for kd in WIN_KD]
        out.append(J("conv3d", "win3d-check", model="win3d", n=WIN_N, k=3, kd=8, enc="T8", seed=s))
        cand = [member(n, k, s) for n, k in CELLS] + [j["tag"] for j in out if j["seed"] == s and j.get("model") in ("win2d", "win3d")]
        out.append(team("conv3d", "Wgreedy3d", s, cand, greedy=6))
    return out


def improve_jobs():
    """Section "improve" of DECISIONS (D8): every arm at 400 clauses per class, three seeds."""
    return [J("improve", "improve", model="pool", n=IMPROVE_N, enc=e, scales=sc, seed=s) for s in SEEDS for e, sc in IMPROVE_ARMS]


def price_jobs(phase):
    src = (main_jobs() + side_jobs() + wide_jobs()) if phase == "price" else conv3d_jobs()
    seen, out = set(), []
    for j in src:
        if j["kind"] == "train" and price_key(j) not in seen:
            seen.add(price_key(j))
            keep = {k: j[k] for k in ("model", "n", "k", "kd", "enc", "pos", "aug", "sampled", "px")}
            out.append(J(phase, "price", seed=0, prefix="price-", max_epochs=3, **keep))
    return out


_BUILD = {"price": lambda: price_jobs("price"), "main": main_jobs, "side": side_jobs, "wide": wide_jobs,
          "price3d": lambda: price_jobs("price3d"), "conv3d": conv3d_jobs, "improve": improve_jobs}


def jobs(phase):
    """``(jobs, blocked)`` of a phase, cheap jobs first. ``blocked`` is always empty here: no job creates another."""
    js = _BUILD[phase]()
    return sorted(js, key=lambda j: (j["kind"] != "train", cost(j), j["tag"])), []


def all_jobs(phases):
    seen, out = set(), []
    for ph in phases:
        for j in jobs(ph)[0]:
            if j["tag"] not in seen:
                seen.add(j["tag"])
                out.append(j)
    return out, []


def smoke_jobs():
    kw = dict(limit=16, max_epochs=2, prefix="smoke-")
    a = [J("smoke", "smoke", n=50, k=3, **kw), J("smoke", "smoke", n=100, k=9, aug="sym", **kw),
         J("smoke", "smoke", n=50, k=9, enc="B8", pos=False, sampled=True, **kw),
         J("smoke", "smoke", n=50, k=3, bal=True, order="shuffle", t_ratio=0.4, **kw),
         J("smoke", "smoke", model="pool", n=50, enc="T4+AG", scales=WIDE, **kw),
         J("smoke", "smoke", model="win2d", n=50, k=5, enc="T8", **kw),
         J("smoke", "smoke", model="win3d", n=50, k=3, kd=2, enc="B8", **kw)]
    return a + [team("smoke", "smoke-W", 0, [j["tag"] for j in a], limit=16),
                team("smoke", "smoke-Wgreedy", 0, [j["tag"] for j in a], greedy=3, limit=16)]


def runnable(j):
    return all(is_done(t) for t in j["needs"])


def find(tag):
    for j in smoke_jobs() + all_jobs(PHASES)[0]:
        if j["tag"] == tag:
            return j
    raise SystemExit(f"no job with tag {tag}")
