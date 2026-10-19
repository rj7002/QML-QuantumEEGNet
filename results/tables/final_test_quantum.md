# bnci2014_001 test accuracy (%), seeds [1, 2, 3]

Extra train.py options: --epochs 200 --learning_rate 0.001 --batch_size 32 --dropout 0.5 --weight_decay 0.001 --n_qubits 6 --n_layers 1

| Model | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Avg | SD across seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QEEGNet (paper) | 49.3 | 30.2 | 44.1 | 30.6 | 26.7 | 31.9 | 36.8 | 28.1 | 65.3 | 38.1 |  |
| EEGNet (paper) | 47.9 | 22.9 | 46.5 | 32.6 | 26.4 | 28.8 | 33.7 | 32.3 | 62.2 | 37.0 |  |
| quantum_pooled (ours) | 73.0 | 39.2 | 85.1 | 52.2 | 39.7 | 44.7 | 68.2 | 74.3 | 77.8 | 61.6 | 1.7 |
