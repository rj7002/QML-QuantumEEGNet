#!/usr/bin/env python3
"""
Run train.py over models x subjects x seeds and summarize per-subject test accuracy.

Defaults reproduce the QEEGNet paper's BCI IV 2a setup: train on session 1 (1/5 held out for
validation), test on session 2, 128 Hz, 4-38 Hz band-pass, 0.5-4 s after the cue, 100 epochs,
batch size 32, AdamW at 1e-3, 9 qubits / 4 circuit layers, best-validation-accuracy checkpoint.

Any other options are passed through to train.py (quantum-only flags are dropped for other models).
Use --metric valid while comparing settings, so choices are made without looking at test accuracy.

Usage:
    python moabb_bench/sweep.py
    python moabb_bench/sweep.py --models quantum_pooled,classical --seeds 1,2,3
    python moabb_bench/sweep.py --subjects 1,3,7 --metric valid --align --augment
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
SCRIPTS = {'train': TRAIN_SCRIPT,
           'pretrain': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrain.py')}

# Table III of the QEEGNet paper (arXiv:2407.19214): BCI IV 2a test accuracy, subjects 1-9
PAPER_TABLE_III = {
    'QEEGNet (paper)': [49.3, 30.2, 44.1, 30.6, 26.7, 31.9, 36.8, 28.1, 65.3],
    'EEGNet (paper)': [47.9, 22.9, 46.5, 32.6, 26.4, 28.8, 33.7, 32.3, 62.2],
}

PAPER_SETUP = [
    '--fmin', '4', '--fmax', '38', '--resample', '128', '--tmin', '0.5', '--tmax', '4',
    '--epochs', '100', '--batch_size', '32', '--learning_rate', '1e-3',
    '--n_qubits', '9', '--n_layers', '4',
]


# Lines train.py prints at the end of every run, one per metric
METRIC_LINES = {'test': 'Test accuracy', 'valid': 'Best validation accuracy'}

# train.py flags that only apply to quantum_pooled -> number of values each one takes
QUANTUM_ONLY_FLAGS = {'--reupload': 0, '--rotations': 1}


def args_for_model(model, extra_args):
    if model == 'quantum_pooled':
        return extra_args
    kept, skip = [], 0
    for arg in extra_args:
        if skip:
            skip -= 1
            continue
        name = arg.split('=')[0]
        if name in QUANTUM_ONLY_FLAGS:
            skip = 0 if '=' in arg else QUANTUM_ONLY_FLAGS[name]
            continue
        kept.append(arg)
    return kept


def log_path_for(sweep_dir, model, subject, seed):
    return os.path.join(sweep_dir, f'{model}_subject{subject}_seed{seed}.log')


def completed_accuracy(log_path, metric='test'):
    """
    Accuracy (test or best validation) printed in a finished run's own log, or None if the run
    never finished. Read from the log rather than the run's results folder, which older runs
    could share with another run started in the same second.
    """
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        log = f.read()
    if 'Results saved to:' not in log:
        return None
    return float(re.findall(rf'^{METRIC_LINES[metric]}: ([\d.]+)%', log, re.MULTILINE)[-1])


def run_one(dataset, model, subject, seed, setup_args, sweep_dir, metric, script=TRAIN_SCRIPT):
    log_path = log_path_for(sweep_dir, model, subject, seed)
    cmd = [sys.executable, '-W', 'ignore', script, '--dataset', dataset, '--model', model,
           '--subjects', str(subject), '--seed', str(seed), *setup_args]
    with open(log_path, 'w') as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'{model} subject {subject} seed {seed} failed; see {log_path}')
    return completed_accuracy(log_path, metric)


def format_summary(results, models, subjects, seeds, include_paper):
    header = ['Model'] + [f'S{s}' for s in subjects] + ['Avg']
    if len(seeds) > 1:
        header.append('SD across seeds')
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]

    if include_paper:
        for name, accs in PAPER_TABLE_III.items():
            row = [accs[s - 1] for s in subjects]
            cells = [name] + [f'{a:.1f}' for a in row] + [f'{statistics.mean(row):.1f}']
            if len(seeds) > 1:
                cells.append('')
            lines.append('| ' + ' | '.join(cells) + ' |')

    for model in models:
        per_subject = []
        for s in subjects:
            accs = [results[(model, s, seed)] for seed in seeds if (model, s, seed) in results]
            per_subject.append(statistics.mean(accs) if accs else None)

        cells = [f'{model} (ours)'] + ['-' if a is None else f'{a:.1f}' for a in per_subject]
        done = [a for a in per_subject if a is not None]
        cells.append(f'{statistics.mean(done):.1f}' if len(done) == len(subjects) else '-')

        if len(seeds) > 1:
            seed_avgs = [statistics.mean(results[(model, s, seed)] for s in subjects)
                         for seed in seeds if all((model, s, seed) in results for s in subjects)]
            cells.append(f'{statistics.stdev(seed_avgs):.1f}' if len(seed_avgs) > 1 else '-')
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Sweep models x subjects x seeds')
    parser.add_argument('--dataset', type=str, default='bnci2014_001')
    parser.add_argument('--models', type=str, default='quantum_pooled,classical',
                        help='Comma-separated: quantum, quantum_pooled, classical')
    parser.add_argument('--subjects', type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--no_paper_setup', action='store_true',
                        help="Use train.py's defaults instead of the paper's preprocessing/training settings")
    parser.add_argument('--resume', type=str, default=None,
                        help='Existing sweep directory to continue (pass the same options); '
                             'finished runs are reused, not retrained')
    parser.add_argument('--metric', type=str, default='test', choices=list(METRIC_LINES),
                        help='Accuracy to report: test, or best validation (use valid while tuning)')
    parser.add_argument('--script', type=str, default='train', choices=list(SCRIPTS),
                        help='train: per-subject training; pretrain: cross-subject pretraining + fine-tuning')
    args, extra_args = parser.parse_known_args()

    models = args.models.split(',')
    subjects = [int(s) for s in args.subjects.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    setup_args = ([] if args.no_paper_setup else PAPER_SETUP) + extra_args
    include_paper = args.dataset == 'bnci2014_001' and not args.no_paper_setup and args.metric == 'test'
    metric_label = 'test' if args.metric == 'test' else 'best validation'

    if args.resume:
        sweep_dir = os.path.abspath(args.resume)
    else:
        os.makedirs(os.path.join(REPO_ROOT, 'experiments'), exist_ok=True)
        # random suffix: sweeps started in the same second never share a directory
        sweep_dir = tempfile.mkdtemp(prefix=f'sweep_{datetime.now():%Y%m%d_%H%M%S}_',
                                     dir=os.path.join(REPO_ROOT, 'experiments'))
    summary_path = os.path.join(sweep_dir, 'summary.md')

    results = {}
    total = len(models) * len(subjects) * len(seeds)

    def write_summary():
        with open(summary_path, 'w') as f:
            f.write(f'# {args.dataset} {metric_label} accuracy (%), seeds {seeds}\n\n')
            f.write(f'Extra train.py options: {" ".join(extra_args) or "(none)"}\n\n')
            f.write(format_summary(results, models, subjects, seeds, include_paper) + '\n')
    for seed in seeds:
        for subject in subjects:
            for model in models:
                previous = completed_accuracy(log_path_for(sweep_dir, model, subject, seed), args.metric)
                if previous is not None:
                    results[(model, subject, seed)] = previous
                    print(f'[{len(results)}/{total}] {model} subject {subject} seed {seed}: '
                          f'{metric_label} {previous:.1f}% (already done)', flush=True)
                    continue

                start = datetime.now()
                acc = run_one(args.dataset, model, subject, seed,
                              args_for_model(model, setup_args), sweep_dir, args.metric, SCRIPTS[args.script])
                results[(model, subject, seed)] = acc
                minutes = (datetime.now() - start).total_seconds() / 60
                print(f'[{len(results)}/{total}] {model} subject {subject} seed {seed}: '
                      f'{metric_label} {acc:.1f}% ({minutes:.1f} min)', flush=True)
                write_summary()

    write_summary()
    print('\n' + format_summary(results, models, subjects, seeds, include_paper))
    print(f'\nSummary and per-run logs: {sweep_dir}')


if __name__ == '__main__':
    main()
