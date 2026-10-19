import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments' / 'final_results'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE})

acc, agree, exact = defaultdict(dict), defaultdict(list), {}
for f in glob.glob(str(ROOT / 'experiments/hardware_eval/*aer_noise_p2-*.json')):
    r = json.loads(Path(f).read_text())
    n = r['noise']['p2'] * 100
    acc[n][int(r['subject'])] = r['accuracy']
    agree[n].append(r['prediction_agreement_with_exact'])
    exact[int(r['subject'])] = r['exact_accuracy_same_trials']
ionq = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / 'experiments/hardware_eval/*ionq_sim*.json'))]
ionq = [r for r in ionq if r['trials'] == 48 and r['seed'] == 1]
ionq_agree = st.mean(r['prediction_agreement_with_exact'] for r in ionq)

levels = sorted(acc)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
for ax in (ax1, ax2):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel('Noise level s (%): 2-qubit and readout error s, 1-qubit error s/10')

label_y = {}
for s in sorted(range(1, 10), key=lambda s: acc[levels[-1]][s]):
    y = acc[levels[-1]][s]
    label_y[s] = max(y, max(label_y.values(), default=-99) + 2.6)
for s in range(1, 10):
    ax1.plot(levels, [acc[n][s] for n in levels], color=BLUE, alpha=0.22, linewidth=1)
    ax1.text(levels[-1] + 0.4, label_y[s], f'S{s}', color=MUTED, fontsize=7.5, va='center')
mean_acc = [st.mean(acc[n].values()) for n in levels]
ax1.plot(levels, mean_acc, 'o-', color=BLUE, linewidth=2, markersize=8, markeredgecolor=SURFACE,
         markeredgewidth=1.5, label='Mean over 9 subjects')
ax1.axhline(st.mean(exact.values()), color=INK2, linewidth=1, linestyle='--', label='Noise-free (exact)')
ax1.axhline(25, color=MUTED, linewidth=1, linestyle=':')
ax1.text(0.3, 26.5, 'chance', color=MUTED, fontsize=8)
ax1.set_ylim(20, 92)
ax1.set_ylabel('Test accuracy (%)')
ax1.set_title('Accuracy under simulated hardware noise', loc='left')
ax1.legend(loc='upper left', frameon=False, fontsize=9)

ax2.plot(levels, [st.mean(agree[n]) for n in levels], 'o-', color=BLUE, linewidth=2, markersize=8,
         markeredgecolor=SURFACE, markeredgewidth=1.5, label='Local noise model (288 trials/subject)')
ax2.axhline(ionq_agree, color=ORANGE, linewidth=2, label=f'IonQ simulator, Aria-1 noise model (48 trials/subject): {ionq_agree:.1f}%')
ax2.set_ylim(80, 101)
ax2.set_ylabel('Predictions matching noise-free (%)')
ax2.set_title('Agreement with noise-free predictions', loc='left')
ax2.legend(loc='lower left', frameon=False, fontsize=9)

fig.suptitle('Noise robustness of the trained quantum EEGNet (6 qubits, 1 layer; 1,000 shots; seed 1)',
             x=0.06, ha='left', fontsize=12)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'fig4_noise_robustness.{ext}', dpi=200, facecolor=SURFACE, bbox_inches='tight')
print(OUT / 'fig4_noise_robustness.png')
