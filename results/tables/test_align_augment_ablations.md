# bnci2014_001 test accuracy (%), seeds [1, 2, 3], --align --augment, 200 epochs

quantum_pooled and ablate_*: --batch_size 32 --weight_decay 0.001 --n_qubits 6 --n_layers 1; classical: --batch_size 16 --weight_decay 0.01; all: --learning_rate 0.001 --dropout 0.5

ablate_none: no circuit; ablate_mlp: Linear+tanh in place of the circuit; ablate_frozen: circuit weights left untrained (moabb_bench/ablations.py)

Test accuracy (%), mean over seeds; n = finished runs

| Model | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Avg | Valid avg | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quantum_pooled | 81.5 | 55.1 | 89.2 | 60.0 | 50.9 | 51.3 | 87.0 | 81.2 | 78.8 | 70.6 | 77.7 | 27 |
| classical | 82.4 | 56.5 | 90.3 | 64.6 | 55.8 | 52.9 | 83.6 | 83.3 | 81.7 | 72.3 | 79.5 | 27 |
| ablate_none | 82.3 | 61.1 | 90.5 | 66.5 | 52.0 | 50.8 | 87.0 | 81.1 | 81.7 | 72.6 | 80.6 | 27 |
| ablate_mlp | 82.1 | 61.2 | 90.4 | 63.7 | 47.7 | 44.4 | 86.8 | 82.4 | 83.0 | 71.3 | 79.7 | 27 |
| ablate_frozen | 80.6 | 54.6 | 88.2 | 60.9 | 47.2 | 50.5 | 84.7 | 81.9 | 78.1 | 69.6 | 77.1 | 27 |

Paired comparison vs quantum_pooled over matching (subject, seed) runs:
  quantum_pooled - classical: -1.8 points (n=27, Wilcoxon p=0.107)
  quantum_pooled - ablate_none: -2.0 points (n=27, Wilcoxon p=0.032)
  quantum_pooled - ablate_mlp: -0.7 points (n=27, Wilcoxon p=0.209)
  quantum_pooled - ablate_frozen: +0.9 points (n=27, Wilcoxon p=0.196)
