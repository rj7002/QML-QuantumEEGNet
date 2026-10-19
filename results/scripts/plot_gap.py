import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; E = ROOT / 'experiments'
def run(log):
    d = re.findall(r'Results saved to: (\S+)', Path(log).read_text())[-1]
    return json.loads((ROOT / d / 'results.json').read_text())
RUNS = {
  'quantum': {'Tuned only': [E/f'search_20261008_012811/quantum_pooled_trial10_subject{s}.log' for s in (1,2,3)]
                              + [E/f'sweep_20261019_152903_bib3op3o/quantum_pooled_subject{s}_seed2.log' for s in (1,2,3)],
              '+ alignment + augmentation': [E/f'sweep_20261019_142231/quantum_pooled_subject{s}_seed{k}.log' for s in (1,2,3) for k in (1,2)]},
  'classical': {'Tuned only': [E/f'search_20261008_012811/classical_trial05_subject{s}.log' for s in (1,2,3)]
                              + [E/f'sweep_20261019_152903_82kfs3g4/classical_subject{s}_seed2.log' for s in (1,2,3)],
                '+ alignment + augmentation': [E/f'sweep_20261019_142235/classical_subject{s}_seed{k}.log' for s in (1,2,3) for k in (1,2)]},
}
BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
ep = np.arange(1, 201)
for ax, (model, title) in zip(axes, (('quantum', 'Quantum EEGNet'), ('classical', 'EEGNet'))):
    ax.grid(True, color=GRID, linewidth=0.8); ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True)
    for cfg, color in (('Tuned only', MUTED), ('+ alignment + augmentation', BLUE if model == 'quantum' else ORANGE)):
        R = [run(l) for l in RUNS[model][cfg]]
        tr = np.mean([r['train_accuracies'] for r in R], 0); va = np.mean([r['valid_accuracies'] for r in R], 0)
        ax.plot(ep, va, color=color, linewidth=2, label=f'{cfg}: validation')
        ax.plot(ep, tr, color=color, linewidth=1.2, linestyle='--', label=f'{cfg}: training')
        print(f'{model:9s} {cfg:28s} last-20-epoch train {tr[-20:].mean():.1f} valid {va[-20:].mean():.1f} gap {tr[-20:].mean()-va[-20:].mean():.1f} | best-saved valid {np.mean([r["best_valid_accuracy"] for r in R]):.1f}')
    ax.set_title(title, loc='left'); ax.set_xlabel('Epoch'); ax.set_ylim(20, 100)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
axes[0].set_ylabel('Accuracy (%)')
fig.suptitle('Training vs validation accuracy: effect of alignment + augmentation (subjects 1-3, seeds 1-2)', x=0.06, ha='left', fontsize=12)
fig.tight_layout()
out = E / 'final_results' / 'fig5_align_augment_gap'
for ext in ('png', 'pdf'): fig.savefig(f'{out}.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
print(f'{out}.png')
