"""Teams of 2D and 3D convolutional members trained on one joint vote sum (plan §3, §4).

The joint update is ``experiments/joint_wide_tm/code/joint.py`` (checked there, T0-T8); this file
adds the 3D member and builds / saves / reloads CIFAR-10 teams. The library is not modified.
"""

import time

import torch
from common import ENC_SHAPE, MEMBERS, MODEL_KW, N_CLASSES, REPO, RULE, TEAMS, _load

import torchtsetlin as tt

joint = _load("jw_joint", REPO / "experiments" / "joint_wide_tm" / "code" / "joint.py")
JointConv, JointTeam, learning_votes = joint.JointConv, joint.JointTeam, joint.learning_votes


class JointConv3d(joint._JointMixin, tt.Conv3dTsetlinMachine):
    """A 3D convolutional member. Without ``_joint_on`` it is the library model plus a counter."""


def member_cfgs(team, clauses):
    return {n: dict(MEMBERS[n], clauses=clauses) for n in TEAMS[team]}


def build_member(c, max_batch, dev):
    shape = ENC_SHAPE[c["enc"]]
    if c["conv"] == "3d":
        def make(**extra):
            return JointConv3d(N_CLASSES, c["clauses"], c["T"], c["s"],
                               patch_size=(c["kd"], c["patch"], c["patch"]), stride=1, input_shape=shape,
                               **MODEL_KW, **extra)
        m = make()
        # FB-1 (GitHub #17): the chunk budget must hold one whole example as ONE patch block, or
        # the model leaves the fused CUDA path (as cifar10_batch_clauses.train.chunk_budget).
        P = (shape[1] - c["kd"] + 1) * (shape[2] - c["patch"] + 1) * (shape[3] - c["patch"] + 1)
        C, L = m.n_clauses_total, m.n_literals
        need = 3 * P * C + 2 * C * L + P * L
        if need > m.max_chunk_elements:
            m = make(max_chunk_elements=need)
    else:
        m = JointConv(N_CLASSES, c["clauses"], c["T"], c["s"], patch_size=c["patch"], stride=1,
                      input_shape=shape, **MODEL_KW)
    return m.joint_setup(max_batch).to(dev)


def build_team(cfgs, max_batch, dev, rule=RULE):
    return JointTeam(list(cfgs), [build_member(c, max_batch, dev) for c in cfgs.values()], rule,
                     shared_negative=True)


def member_states(team):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in team.members]


def save_team(path, team, rec, row):
    """Self-contained team file (``hp`` holds what ``load_team`` needs); .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"members": dict(zip(team.names, member_states(team))), "hp": rec["hp"], "epoch": row["epoch"],
                "val_acc": row["val_acc"], "test_acc": row["test_acc"],
                "member_test_acc": {n: m["test_acc"] for n, m in row["members"].items()},
                "tag": rec["tag"], "phase": rec["phase"], "git_sha": rec["git_sha"], "env": rec["env"],
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, tmp)
    tmp.replace(path)
    return path


def load_team(path, dev):
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfgs = {n: {k: c[k] for k in ("enc", "conv", "patch", "T", "s", "clauses", "kd") if k in c}
            for n, c in st["hp"]["members"].items()}
    team = build_team(cfgs, 1, dev, st["hp"]["rule"])
    for m, n in zip(team.members, team.names):
        m.load_state_dict(st["members"][n])
    return team, st
