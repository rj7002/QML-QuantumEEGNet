#!/usr/bin/env python3
"""Summarize a condor sweep: per-subject test accuracy (mean over seeds) for each model, plus paired tests."""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

log_dir = sys.argv[1]
test, valid = defaultdict(dict), defaultdict(dict)
for path in glob.glob(os.path.join(log_dir, '*.log')):
    m = re.match(r'(.+)_s(\d)_seed(\d)\.log', os.path.basename(path))
    text = open(path).read()
    t = re.findall(r'Test accuracy: ([\d.]+)%', text)
    v = re.findall(r'Best validation accuracy: ([\d.]+)%', text)
    if m and t:
        key = (int(m[2]), int(m[3]))
        test[m[1]][key], valid[m[1]][key] = float(t[-1]), float(v[-1])

order = [m for m in ('quantum_pooled', 'classical', 'ablate_none', 'ablate_mlp', 'ablate_frozen') if m in test]
subjects = range(1, 10)
print('Test accuracy (%), mean over seeds; n = finished runs\n')
print('| Model | ' + ' | '.join(f'S{s}' for s in subjects) + ' | Avg | Valid avg | n |')
print('|---' * 13 + '|')
for model in order:
    row = []
    for s in subjects:
        vals = [a for (subj, _), a in test[model].items() if subj == s]
        row.append(f'{np.mean(vals):.1f}' if vals else '-')
    print(f'| {model} | ' + ' | '.join(row) + f' | {np.mean(list(test[model].values())):.1f} | '
          f'{np.mean(list(valid[model].values())):.1f} | {len(test[model])} |')

print('\nPaired comparison vs quantum_pooled over matching (subject, seed) runs:')
for model in order[1:]:
    keys = sorted(set(test['quantum_pooled']) & set(test[model]))
    if len(keys) < 6:
        continue
    q = np.array([test['quantum_pooled'][k] for k in keys])
    o = np.array([test[model][k] for k in keys])
    p = wilcoxon(q, o).pvalue if np.any(q != o) else 1.0
    print(f'  quantum_pooled - {model}: {np.mean(q - o):+.1f} points (n={len(keys)}, Wilcoxon p={p:.3f})')
