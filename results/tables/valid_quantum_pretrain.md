# bnci2014_001 best validation accuracy (%), seeds [1, 2]

Extra train.py options: --epochs 100 --pretrain_epochs 50 --learning_rate 0.001 --batch_size 32 --dropout 0.5 --weight_decay 0.001 --n_qubits 6 --n_layers 1 --align --augment

| Model | S1 | S2 | S3 | Avg | SD across seeds |
|---|---|---|---|---|---|
| quantum_pooled (ours) | 87.5 | 59.8 | 83.9 | 77.1 | 2.1 |
