import json
import re
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments' / 'final_results'
OUT.mkdir(exist_ok=True)
SWEEPS = {'quantum': (ROOT / 'experiments/sweep_20261015_031134', 'quantum_pooled'),
          'classical': (ROOT / 'experiments/sweep_20261015_031154', 'classical')}
SUBJECTS, SEEDS = range(1, 10), (1, 2, 3)
PAPER = {'QEEGNet (paper)': [49.3, 30.2, 44.1, 30.6, 26.7, 31.9, 36.8, 28.1, 65.3],
         'EEGNet (paper)': [47.9, 22.9, 46.5, 32.6, 26.4, 28.8, 33.7, 32.3, 62.2]}

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE})


def runs(model):
    sweep_dir, name = SWEEPS[model]
    out = {}
    for s in SUBJECTS:
        for k in SEEDS:
            log = (sweep_dir / f'{name}_subject{s}_seed{k}.log').read_text()
            run_dir = re.findall(r'Results saved to: (\S+)', log)[-1]
            out[s, k] = json.loads((ROOT / run_dir / 'results.json').read_text())
    return out


R = {m: runs(m) for m in SWEEPS}
test = {m: np.array([[R[m][s, k]['test_accuracy'] for k in SEEDS] for s in SUBJECTS]) for m in R}
mean = {m: test[m].mean(axis=1) for m in R}
sd = {m: test[m].std(axis=1, ddof=1) for m in R}


def style(ax):
    ax.grid(True, axis='y', color=GRID, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


# Figure 1: per-subject test accuracy
fig, ax = plt.subplots(figsize=(11, 5))
style(ax)
xs = np.arange(10)
labels = [f'S{s}' for s in SUBJECTS] + ['Mean']
offsets = {'QEEGNet (paper)': -0.27, 'EEGNet (paper)': -0.09, 'quantum': 0.09, 'classical': 0.27}
for name, marker in (('QEEGNet (paper)', 'o'), ('EEGNet (paper)', 's')):
    vals = PAPER[name] + [st.mean(PAPER[name])]
    ax.plot(xs + offsets[name], vals, marker, color=MUTED, markerfacecolor=SURFACE, markersize=8,
            markeredgewidth=1.6, linestyle='', label=name)
for m, color, label in (('quantum', BLUE, 'Quantum EEGNet (ours, tuned)'),
                        ('classical', ORANGE, 'EEGNet (ours, tuned)')):
    vals = list(mean[m]) + [mean[m].mean()]
    errs = list(sd[m]) + [test[m].mean(axis=0).std(ddof=1)]
    ax.errorbar(xs + offsets[m], vals, yerr=errs, fmt='o', color=color, markersize=8, elinewidth=1.5,
                capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
ax.axvline(8.5, color=AXIS, linewidth=1)
ax.axhline(25, color=MUTED, linewidth=1, linestyle=':')
ax.text(-0.45, 22.5, 'chance (25%)', color=MUTED, fontsize=8, va='top')
ax.set_xticks(xs, labels)
ax.set_ylim(15, 95)
ax.set_ylabel('Test accuracy (%)')
ax.legend(loc='upper left', ncol=4, frameon=False, bbox_to_anchor=(0, 1.1), fontsize=9)
ax.set_title('BCI IV 2a test accuracy (session 2) by subject', loc='left', pad=34, fontsize=12)
fig.text(0.125, -0.02, 'Ours: mean of 3 seeds, error bars = SD across seeds (Mean column: SD of the 3 seed averages). '
         'Paper: single reported run.', color=INK2, fontsize=8)
save(fig, 'fig1_test_accuracy_by_subject')

# Figure 2: quantum vs classical per subject
fig, ax = plt.subplots(figsize=(5.8, 5.8))
style(ax)
ax.grid(True, axis='x', color=GRID, linewidth=0.8)
ax.plot([30, 92], [30, 92], color=MUTED, linewidth=1, linestyle='--')
ax.errorbar(mean['quantum'], mean['classical'], xerr=sd['quantum'], yerr=sd['classical'], fmt='o',
            color=INK2, markerfacecolor=BLUE, markeredgecolor=SURFACE, markersize=9, elinewidth=1,
            ecolor=AXIS, capsize=2)
for s, x, y in zip(SUBJECTS, mean['quantum'], mean['classical']):
    ax.annotate(f'S{s}', (x, y), xytext=(7, -10), textcoords='offset points', color=INK2, fontsize=9)
ax.text(33, 88, 'Classical better', color=MUTED, fontsize=9)
ax.text(89, 33, 'Quantum better', color=MUTED, fontsize=9, ha='right')
ax.set_xlim(30, 92)
ax.set_ylim(30, 92)
ax.set_aspect('equal')
ax.set_xlabel('Quantum EEGNet test accuracy (%)')
ax.set_ylabel('EEGNet test accuracy (%)')
ax.set_title('Per-subject comparison (mean of 3 seeds)', loc='left', fontsize=12)
save(fig, 'fig2_quantum_vs_classical')

# Figure 3: learning curves averaged over all 27 runs per model
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
for ax in (ax_acc, ax_loss):
    style(ax)
    ax.grid(True, axis='x', color=GRID, linewidth=0.8)
epochs = np.arange(1, 201)
for m, color, label in (('quantum', BLUE, 'Quantum EEGNet'), ('classical', ORANGE, 'EEGNet')):
    for ax, key, train_key in ((ax_acc, 'valid_accuracies', 'train_accuracies'),
                               (ax_loss, 'valid_losses', 'train_losses')):
        # average seeds within each subject, then show mean and SE across the 9 subjects
        per_subject = np.array([np.mean([R[m][s, k][key] for k in SEEDS], axis=0) for s in SUBJECTS])
        mu, se = per_subject.mean(axis=0), per_subject.std(axis=0, ddof=1) / 3
        ax.plot(epochs, mu, color=color, linewidth=2, label=f'{label}, validation')
        ax.fill_between(epochs, mu - se, mu + se, color=color, alpha=0.15, linewidth=0)
        train = np.mean([R[m][s, k][train_key] for s in SUBJECTS for k in SEEDS], axis=0)
        ax.plot(epochs, train, color=color, linewidth=1.2, linestyle='--', alpha=0.6, label=f'{label}, training')
ax_acc.axhline(25, color=MUTED, linewidth=1, linestyle=':')
ax_acc.text(3, 26.5, 'chance', color=MUTED, fontsize=8)
ax_acc.set_ylabel('Accuracy (%)')
ax_loss.set_ylabel('Cross-entropy loss')
for ax in (ax_acc, ax_loss):
    ax.set_xlabel('Epoch')
ax_acc.set_title('Accuracy', loc='left')
ax_loss.set_title('Loss', loc='left')
ax_acc.legend(loc='lower right', frameon=False, fontsize=9)
fig.suptitle('Learning curves, averaged over 9 subjects x 3 seeds (band = SE across subjects)',
             x=0.06, ha='left', fontsize=12)
fig.tight_layout()
save(fig, 'fig3_learning_curves')
print(OUT)
