"""Graph Tsetlin machine (plans/graph-tm.md): data container, evaluation against a naive
reference, the depth-1 reduction to the convolutional coalesced TM, and learning."""

import pytest
import torch

import torchtsetlin as tt
from torchtsetlin.data import GraphBatch, SymbolTable

# ----------------------------------------------------------------------------- helpers


def _random_graphs(n_graphs, F, n_edge_types=2, n_node_types=1, max_nodes=6, gen=None):
    feats, edges, types, ntypes = [], [], [], []
    for _ in range(n_graphs):
        n = int(torch.randint(1, max_nodes + 1, (1,), generator=gen))
        feats.append(torch.rand(n, F, generator=gen) < 0.4)
        E = int(torch.randint(0, 3 * n, (1,), generator=gen))
        edges.append(torch.randint(0, n, (E, 2), generator=gen))
        types.append(torch.randint(0, n_edge_types, (E,), generator=gen))
        ntypes.append(torch.randint(0, n_node_types, (n,), generator=gen))
    return GraphBatch.from_graphs(feats, edges, types, ntypes if n_node_types > 1 else None,
                                  n_edge_types=n_edge_types, n_node_types=n_node_types)


def _sparse_includes(model, p=0.06, gen=None):
    """Random automata with a few included literals, so that clauses sometimes match."""
    N = model.n_states
    inc = torch.rand(model.ta_state.shape, generator=gen) < p
    model.ta_state.copy_(torch.where(inc, N, N - 1).to(model.ta_state.dtype))
    model._refresh_include()


def _naive_node_matches(model, g, empty_value):
    """The GraphTM definition, node by node and edge by edge (paper App. A.3 / reference
    kernels): a list over graphs of (n_i, C) Boolean last-layer matches. An empty component
    matches; with ``empty_value=False`` an entirely empty clause is then masked out."""
    inc = model.included_mask().cpu()
    Fn = int(model.n_features)
    C = inc.shape[0]
    D, M = model.depth, model.message_size
    bounds = model._layer_bounds()
    bits = model.msg_bits.cpu()
    g = g.to("cpu")
    out = []
    start = 0
    for b in range(len(g)):
        n = int(g.n_nodes[b])
        E = int(g.edges_per_graph[b])
        ed, et = g.edges[start:start + E], g.edge_type[start:start + E]
        start += E

        def comp(j, i, vec):
            a, c = bounds[i], bounds[i + 1]
            pos, neg = inc[j, a:c], inc[j, Fn + a:Fn + c]
            return bool((vec[pos]).all()) and not bool(vec[neg].any())

        x0 = g.node_features[b, :n]
        Mm = torch.zeros(n, C, dtype=torch.bool)
        for q in range(n):
            for j in range(C):
                ok = comp(j, 0, x0[q])
                if model.n_node_types > 1 and g.node_type is not None:
                    ok = ok and int(g.node_type[b, q]) == j % model.n_node_types
                Mm[q, j] = ok
        for i in range(1, D):
            inbox = torch.zeros(n, M, dtype=torch.bool)
            for (s, d), t in zip(ed.tolist(), et.tolist()):
                for j in range(C):
                    if Mm[s, j]:
                        inbox[d, (bits[j] + t) % M] = True
            new = torch.zeros_like(Mm)
            for q in range(n):
                for j in range(C):
                    new[q, j] = Mm[q, j] and comp(j, i, inbox[q])
            Mm = new
        if not empty_value:  # an empty clause does not vote, but its messages still flow
            Mm &= inc.any(dim=1).unsqueeze(0)
        out.append(Mm)
    return out


# ----------------------------------------------------------------------------- symbols


def test_symbol_table_encodings_match_the_reference():
    names = [f"s{i}" for i in range(300)]
    dh = SymbolTable(names, dim=128, encoding="double_hashing")
    i = torch.arange(300)
    assert torch.equal(dh.positions, torch.stack([i % 128, 127 - (i % 127)], dim=1))  # prevprime(128)=127
    oh = SymbolTable(["a", "b", "c"], encoding="one_hot")
    assert oh.dim == 3 and torch.equal(oh.vector(["c", "a"]), torch.tensor([True, False, True]))
    rnd = SymbolTable(names, dim=64, bits=3, seed=1)
    assert rnd.positions.shape == (300, 3)
    assert all(len(set(r)) == 3 for r in rnd.positions.tolist())
    assert torch.equal(rnd.positions, SymbolTable(names, dim=64, bits=3, seed=1).positions)
    assert set(rnd.decode(rnd.vector(["s5", "s9"]))) >= {"s5", "s9"}
    with pytest.raises(ValueError):
        SymbolTable(["a", "a"])


# ----------------------------------------------------------------------------- GraphBatch


def test_graph_batch_slicing_keeps_edges_inside_their_graphs():
    gen = torch.Generator().manual_seed(3)
    g = _random_graphs(7, 5, gen=gen)
    idx = torch.tensor([4, 0, 4, 6])  # duplicates are allowed
    sub = g[idx]
    assert len(sub) == 4 and sub.shape[1] == int(g.n_nodes[idx].max())
    start = torch.cumsum(g.edges_per_graph, 0) - g.edges_per_graph
    s0 = 0
    for k, b in enumerate(idx.tolist()):
        e = int(g.edges_per_graph[b])
        assert torch.equal(sub.edges[s0:s0 + e], g.edges[start[b]:start[b] + e])
        assert torch.equal(sub.node_features[k, : g.n_nodes[b]], g.node_features[b, : g.n_nodes[b]])
        s0 += e
    back = GraphBatch.cat([g[:3], g[3:]])
    assert torch.equal(back.node_features, g.node_features) and torch.equal(back.edges, g.edges)
    assert g[-1].n_nodes.item() == g.n_nodes[-1].item()
    with pytest.raises(ValueError):
        GraphBatch(torch.zeros(1, 2, 3), torch.tensor([2]), torch.tensor([[0, 5]]))


def test_from_sequences_builds_left_and_right_edges():
    sym = SymbolTable(["A", "B"], encoding="one_hot")
    g = GraphBatch.from_sequences([["A", "B", "A"]], sym)
    pairs = sorted(zip(map(tuple, g.edges.tolist()), g.edge_type.tolist()))
    assert pairs == [((0, 1), 1), ((1, 0), 0), ((1, 2), 1), ((2, 1), 0)]
    assert g.edge_type_names == ["left", "right"] and g[0].edge_type_names == ["left", "right"]


# ----------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize("encoding", ["random", "double_hashing", "one_hot"])
def test_evaluation_matches_the_naive_definition(device, encoding):
    gen = torch.Generator().manual_seed(11)
    g = _random_graphs(6, 7, n_edge_types=3, n_node_types=2, gen=gen)
    m = tt.GraphTsetlinMachine(7, 3, 9, T=5, depth=3, message_size=12, message_bits=2,
                               message_encoding=encoding, n_edge_types=3, n_node_types=2)
    _sparse_includes(m, p=0.05, gen=gen)
    m = m.to(device)
    for empty_value in (True, False):
        ref = _naive_node_matches(m, g, empty_value)
        got = m.node_clause_outputs(g.to(device), empty_value=empty_value).cpu()
        for b, r in enumerate(ref):
            assert torch.equal(got[b, : r.shape[0]], r), (encoding, empty_value, b)
            assert not got[b, r.shape[0]:].any()  # padding never matches
        m.eval()
        clause_out = m.evaluate_clauses(g.to(device)).cpu()
        assert torch.equal(clause_out, torch.stack([r.any(0) for r in _naive_node_matches(m, g, False)]))
        m.train()


def test_paper_worked_message_example():
    """Paper App. A.3, Algorithm 2 / Table 15: sequence 'BAAAE', node hypervector size 8 with
    'A' on bits 0 and 1, one clause C^0 = A with message bits I_cl = [4, 5], message size 8,
    left edges of type 0 and right edges of type 1: node X2 receives H^1_2 = [00001110]."""
    feats = torch.zeros(5, 8, dtype=torch.bool)
    feats[1:4, 0:2] = True  # X1..X3 = A
    # e_{n,n+1} = 0 (from X_n to its right neighbour), e_{n,n-1} = 1
    edges = [(n, n + 1) for n in range(4)] + [(n, n - 1) for n in range(1, 5)]
    types = [0] * 4 + [1] * 4
    g = GraphBatch.from_graphs([feats], [edges], [types], n_edge_types=2)
    m = tt.GraphTsetlinMachine(8, 2, 1, T=5, depth=2, message_size=8, message_bits=2, n_edge_types=2)
    m.msg_bits.copy_(torch.tensor([[4, 5]]))
    st = torch.full_like(m.ta_state, m.n_states - 1)
    st[0, 0] = st[0, 1] = m.n_states  # C^0 = A (both feature bits of A)
    m.ta_state.copy_(st)
    m._refresh_include()
    inbox = m.node_messages(g, 1)[0]
    assert inbox[2].int().tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    # X1: its left neighbour X0 = B sends nothing, its right neighbour X2 = A sends over type 1
    assert inbox[1].int().tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert m.message_collision_rate() == 0.0



def test_empty_clauses_still_send_messages_when_predicting():
    """Clause 0 is empty, clause 1 = 'a message from clause 0 arrived over type 1'. Clause 1
    learned to read "has a neighbour on that side" while clause 0 was empty during learning,
    so it must keep working in eval(); only clause 0's own vote disappears."""
    feats = torch.zeros(3, 2, dtype=torch.bool)
    g = GraphBatch.from_graphs([feats, feats[:1]], [[(0, 1), (1, 2)], []], [[1, 1], []], n_edge_types=2)
    m = tt.GraphTsetlinMachine(2, 2, 2, T=5, depth=2, message_encoding="one_hot", n_edge_types=2)
    st = torch.full_like(m.ta_state, m.n_states - 1)
    bit = 2 + 0 * 2 + 1  # layer-1 inbox starts after the 2 node features; clause 0, type 1
    st[1, bit] = m.n_states
    m.ta_state.copy_(st)
    m._refresh_include()
    m.eval()
    assert m.evaluate_clauses(g).tolist() == [[False, True], [False, False]]
    assert m.clause_expression(1) == "e11:0"
    learning = m.node_clause_outputs(g, empty_value=True).any(dim=1)
    assert learning.tolist() == [[True, True], [True, False]]

def test_padding_and_batching_do_not_change_outputs(device):
    gen = torch.Generator().manual_seed(5)
    g = _random_graphs(8, 6, gen=gen).to(device)
    m = tt.GraphTsetlinMachine(6, 2, 12, T=5, depth=2, message_size=16, n_edge_types=2).to(device)
    _sparse_includes(m, gen=gen)
    m.eval()
    whole = m.evaluate_clauses(g)
    one_by_one = torch.cat([m.evaluate_clauses(g[i]) for i in range(len(g))])
    assert torch.equal(whole, one_by_one)


# ----------------------------------------------------------------------------- depth 1 = CoTM


def _images(n=12, gen=None):
    return torch.rand(n, 1, 8, 8, generator=gen) < 0.3


def test_from_images_matches_the_convolutional_patch_features():
    gen = torch.Generator().manual_seed(2)
    x = _images(gen=gen)
    conv = tt.ConvCoalescedTsetlinMachine(3, 10, T=5, patch_size=3, input_shape=(1, 8, 8))
    lit = conv._encode(x.to(conv._literal_dtype()))
    g = GraphBatch.from_images(x, 3)
    F = conv.n_features
    assert torch.equal(g.node_features, lit[..., :F] > 0.5)


def test_depth_one_is_the_convolutional_coalesced_machine():
    """Same automata, weights and random stream -> identical votes and identical learning."""
    gen = torch.Generator().manual_seed(4)
    x = _images(16, gen=gen)
    y = torch.randint(0, 3, (16,), generator=gen)
    conv = tt.ConvCoalescedTsetlinMachine(3, 20, T=8, s=3.0, patch_size=3, input_shape=(1, 8, 8))
    graph = tt.GraphTsetlinMachine(conv.n_features, 3, 20, T=8, s=3.0, depth=1)
    graph.weights.copy_(conv.weights)
    g = GraphBatch.from_images(x, 3)
    for step in range(6):
        torch.manual_seed(100 + step)
        v1 = conv.update(x, y, sequential=step % 2 == 1)
        torch.manual_seed(100 + step)
        v2 = graph.update(g, y, sequential=step % 2 == 1)
        assert torch.equal(v1, v2)
        assert torch.equal(conv.ta_state, graph.ta_state)
        assert torch.equal(conv.weights, graph.weights)
    assert (conv.ta_state != conv.n_states - 1).sum() > 50  # the machines did learn
    conv.eval()
    graph.eval()
    assert torch.equal(conv(x), graph(g))


# ----------------------------------------------------------------------------- learning


def _aaa_task(n, gen, L=5):
    """Paper Sec. 3.1: five-letter sequences, positive iff they contain 'AAA'."""
    seqs, ys = [], []
    for _ in range(n):
        s = [("ABCDE")[i] for i in torch.randint(0, 5, (L,), generator=gen)]
        if bool(torch.rand(1, generator=gen) < 0.5):
            k = int(torch.randint(0, L - 2, (1,), generator=gen))
            s[k:k + 3] = ["A"] * 3
        seqs.append(s)
        ys.append(int("AAA" in "".join(s)))
    return seqs, torch.tensor(ys)


@pytest.mark.parametrize("encoding,shared", [("one_hot", False), ("random", True)])
def test_learns_three_consecutive_a(device, encoding, shared):
    gen = torch.Generator().manual_seed(1)
    sym = SymbolTable(list("ABCDE"), encoding="one_hot")
    xs, ys = _aaa_task(3000, gen)
    xt, yt = _aaa_task(1000, gen)
    gtr = GraphBatch.from_sequences(xs, sym).to(device)
    gte = GraphBatch.from_sequences(xt, sym).to(device)
    ytr, yte = ys.to(device), yt.to(device)
    m = tt.GraphTsetlinMachine(5, 2, 8, T=40, s=2.0, depth=2, message_encoding=encoding, message_size=64,
                               n_edge_types=2, shared_node_draw=shared,
                               edge_type_names=["left", "right"]).to(device)
    for _ in range(3):
        perm = torch.randperm(len(gtr), device=device)
        for i in range(0, len(gtr), 25):
            m.update(gtr[perm[i:i + 25]], ytr[perm[i:i + 25]])
    m.eval()
    acc = (m.predict(gte) == yte).float().mean().item()
    assert acc > 0.99
    assert any("A" in m.clause_expression(j, symbols=sym) for j in range(8))


def test_depth_one_cannot_see_neighbours():
    """The same task needs messages: a single node cannot tell 'AAA' from 'ABA'."""
    gen = torch.Generator().manual_seed(1)
    sym = SymbolTable(list("ABCDE"), encoding="one_hot")
    xs, ys = _aaa_task(2000, gen)
    g = GraphBatch.from_sequences(xs, sym)
    m = tt.GraphTsetlinMachine(5, 2, 8, T=40, s=2.0, depth=1, n_edge_types=2)
    for _ in range(3):
        for i in range(0, len(g), 25):
            m.update(g[i:i + 25], ys[i:i + 25])
    m.eval()
    assert (m.predict(g) == ys).float().mean().item() < 0.9


def test_sequential_equals_batches_of_one():
    gen = torch.Generator().manual_seed(8)
    g = _random_graphs(6, 5, gen=gen)
    y = torch.randint(0, 2, (6,), generator=gen)
    a = tt.GraphTsetlinMachine(5, 2, 10, T=6, depth=2, message_size=16, n_edge_types=2)
    b = tt.GraphTsetlinMachine(5, 2, 10, T=6, depth=2, message_size=16, n_edge_types=2)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(9)
    a.update(g, y, sequential=True)
    torch.manual_seed(9)
    for i in range(len(g)):
        b.update(g[i], y[i:i + 1])
    assert torch.equal(a.ta_state, b.ta_state) and torch.equal(a.weights, b.weights)


# ----------------------------------------------------------------------------- API


def test_state_dict_round_trip_and_trainer(tmp_path):
    gen = torch.Generator().manual_seed(6)
    sym = SymbolTable(list("ABCDE"), encoding="one_hot")
    xs, ys = _aaa_task(400, gen)
    g = GraphBatch.from_sequences(xs, sym)
    m = tt.GraphTsetlinMachine(None, 2, 8, T=40, s=2.0, depth=2, message_size=32, n_edge_types=2)
    trainer = tt.Trainer(m, batch_size=25, verbose=False)
    trainer.fit((g, ys), epochs=2)
    assert trainer.evaluate((g, ys))["accuracy"] > 0.5
    pred = trainer.predict(g)
    m2 = tt.GraphTsetlinMachine(None, 2, 8, T=40, s=2.0, depth=2, message_size=32, n_edge_types=2)
    torch.save(m.state_dict(), tmp_path / "g.pt")
    m2.load_state_dict(torch.load(tmp_path / "g.pt"))
    m2.eval()
    assert torch.equal(m2.msg_bits, m.msg_bits)
    assert torch.equal(m2.predict(g), pred)
    assert m2.n_node_features == 5


def test_input_checks():
    m = tt.GraphTsetlinMachine(4, 2, 4, T=5, depth=2, n_edge_types=1)
    with pytest.raises(TypeError):
        m.update(torch.zeros(3, 4, dtype=torch.bool), torch.tensor([0, 1, 0]))
    g = GraphBatch.from_graphs([torch.zeros(2, 4, dtype=torch.bool)], [[(0, 1)]], [[1]])
    with pytest.raises(ValueError):
        m.update(g, torch.tensor([0]))
    with pytest.raises(ValueError):
        tt.GraphTsetlinMachine(4, 2, 4, T=5, message_encoding="bloom")
