# bnci2014_001 test accuracy (%), seeds [1, 2, 3]

Extra train.py options: --epochs 200 --learning_rate 0.001 --batch_size 16 --dropout 0.5 --weight_decay 0.01

| Model | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Avg | SD across seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QEEGNet (paper) | 49.3 | 30.2 | 44.1 | 30.6 | 26.7 | 31.9 | 36.8 | 28.1 | 65.3 | 38.1 |  |
| EEGNet (paper) | 47.9 | 22.9 | 46.5 | 32.6 | 26.4 | 28.8 | 33.7 | 32.3 | 62.2 | 37.0 |  |
| classical (ours) | 80.0 | 43.2 | 85.7 | 50.5 | 48.8 | 41.2 | 71.5 | 76.4 | 77.8 | 63.9 | 0.9 |
