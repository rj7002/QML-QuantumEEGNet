#!/usr/bin/env python3
"""
Mean training curves per model for a sweep: train vs validation accuracy and loss, averaged over every
(subject, seed) run, with the mean test accuracy as a reference line.

Usage: python results/scripts/plot_curves.py experiments results/figures/fig6_training_curves.png
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = [('quantum_pooled', 'Quantum'), ('classical', 'Classical EEGNet'), ('ablate_none', 'No circuit'),
          ('ablate_mlp', 'Classical layer\nin place of circuit'), ('ablate_frozen', 'Frozen circuit')]
TRAIN, VALID = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'

exp_dir, out = sys.argv[1], sys.argv[2]
runs = defaultdict(list)
for path in glob.glob(os.path.join(exp_dir, '*', 'results.json')):
    r = json.load(open(path))
    s = r['config'].get('run_settings', {})
    if s.get('epochs') == 200 and s.get('align') and s.get('augment') and len(r['train_losses']) == 200:
        runs[s['model']].append(r)

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(2, len(MODELS), figsize=(14, 5.6), sharex=True, sharey='row')
epochs = np.arange(1, 201)

for col, (key, title) in enumerate(MODELS):
    rs = runs[key]
    for row, (tr, va, label) in enumerate([('train_accuracies', 'valid_accuracies', 'Accuracy (%)'),
                                           ('train_losses', 'valid_losses', 'Loss')]):
        ax = axes[row, col]
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.plot(epochs, np.mean([r[tr] for r in rs], axis=0), color=TRAIN, linewidth=2, label='Train')
        ax.plot(epochs, np.mean([r[va] for r in rs], axis=0), color=VALID, linewidth=2, label='Validation')
        if row == 0:
            test = np.mean([r['test_accuracy'] for r in rs])
            ax.axhline(test, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(198, test - 1.5, f'Test {test:.1f}%', color=INK, ha='right', va='top', fontsize=8)
            ax.set_title(f'{title}\n(n={len(rs)})', color=INK, fontsize=9.5)
        if col == 0:
            ax.set_ylabel(label)
        if row == 1:
            ax.set_xlabel('Epoch')

axes[0, 0].legend(frameon=False, loc='lower right', fontsize=8, bbox_to_anchor=(1, 0.08))
fig.suptitle('BCI IV 2a, align + augment: mean over 9 subjects x 3 seeds', color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f'Saved {out}; runs per model:', {k: len(v) for k, v in runs.items()})
