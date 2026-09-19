"""Regenerate the white paper's data/*.csv from benchmarks/results/cpu_vs_gpu.json.

Run from the repository root after ./benchmarks/run_benchmarks.sh:

    python whitepaper/data/regen.py

The figures read these CSVs directly, so a rebuild of main.pdf then picks the new numbers up.
The hand-written tables and the prose figures in sections/08-benchmarks.tex are NOT derived
from the CSVs -- check them against benchmarks/results/tables.md by hand.
"""
import json, os
J = json.load(open('benchmarks/results/cpu_vs_gpu.json'))
R = J['records']
OUT = 'whitepaper/data'

def pick(**kw):
    out = []
    for r in R:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out

def val(rs):
    assert len(rs) == 1, (len(rs), rs[:1])
    return rs[0]['examples_per_s']

def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(','.join(str(c) for c in row) + '\n')
    print(name, len(rows), 'rows')

# ---- batch (flat, 784 feat, 500 clauses/class) and the tiny "small" suite
def batch_rows(suite, model=None):
    bs = sorted({r['batch_size'] for r in R if r['suite'] == suite})
    rows = []
    for b in bs:
        sel = dict(suite=suite, batch_size=b)
        ct = val(pick(**sel, phase='train', device='cpu'))
        gt = val(pick(**sel, phase='train', device='cuda:0'))
        ci = val(pick(**sel, phase='infer', device='cpu'))
        gi = val(pick(**sel, phase='infer', device='cuda:0'))
        rows.append([b, round(ct), round(gt), round(ci), round(gi),
                     round(gt/ct, 1), round(gi/ci, 1)])
    return rows

write('batch-mnist.csv',
      'batch,cpu_train,gpu_train,cpu_infer,gpu_infer,train_speedup,infer_speedup',
      batch_rows('batch'))
write('batch-xor.csv',
      'batch,cpu_train,gpu_train,cpu_infer,gpu_infer',
      [r[:5] for r in batch_rows('small')])

# ---- clause budget / feature count sweeps
def sweep(suite, key):
    vals = sorted({r[key] for r in R if r['suite'] == suite})
    rows = []
    for v in vals:
        sel = {'suite': suite, key: v}
        ct = val(pick(**sel, phase='train', device='cpu'))
        gt = val(pick(**sel, phase='train', device='cuda:0'))
        ci = val(pick(**sel, phase='infer', device='cpu'))
        gi = val(pick(**sel, phase='infer', device='cuda:0'))
        rows.append([v, round(ct), round(gt), round(gt/ct, 1),
                     round(ci), round(gi), round(gi/ci, 1)])
    return rows

write('clauses.csv',
      'clauses,cpu_train,gpu_train,train_speedup,cpu_infer,gpu_infer,infer_speedup',
      sweep('clauses', 'n_clauses_per_class'))
write('features.csv',
      'features,cpu_train,gpu_train,train_speedup,cpu_infer,gpu_infer,infer_speedup',
      sweep('features', 'n_features'))

# ---- thread scaling
th = sorted({r['threads'] for r in R if r['suite'] == 'threads' and r['device'] == 'cpu'})
write('threads.csv', 'threads,train,infer',
      [[t, round(val([r for r in pick(suite='threads', threads=t, phase='train', device='cpu')
                      if r['extra'].get('threads_axis')])),
           round(val([r for r in pick(suite='threads', threads=t, phase='infer', device='cpu')
                      if r['extra'].get('threads_axis')]))] for t in th])

# ---- update cost breakdown (the 'phases' suite)
ph = [r for r in R if r['suite'] == 'phases']
stages = {}
for r in ph:
    stages.setdefault(r['extra'].get('stage', r['phase']), {})[r['device']] = \
        r['sec_per_batch'] * 1000.0
order = ['encode', 'evaluate', 'votes+select', 'feedback_counts', 'accumulator_alloc',
         'refresh_include', 'apply_feedback']
rows = [[i + 1, s, round(stages[s]['cpu'], 3), round(stages[s]['cuda:0'], 3)]
        for i, s in enumerate(order) if s in stages]
write('breakdown.csv', 'idx,stage,cpu_ms,gpu_ms', rows)
print('\nall phase keys:', sorted(stages))
for s in sorted(stages):
    print(f'  {s:24s} cpu {stages[s].get("cpu", float("nan")):9.3f}  gpu {stages[s].get("cuda:0", float("nan")):7.3f}')
