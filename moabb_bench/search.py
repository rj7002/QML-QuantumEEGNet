#!/usr/bin/env python3
"""
Random hyperparameter search on BCI IV 2a, scored on validation accuracy only.

Each model gets the same number of trials. Trial 0 is the QEEGNet paper's settings, for reference;
the rest are sampled at random (reproducibly) from the search space below. Every configuration is
trained on each search subject and scored by its best validation accuracy averaged over those
subjects, so the test sessions stay untouched until the final evaluation. Data processing follows
the paper (see sweep.PAPER_SETUP); no alignment, augmentation or circuit variants.

Finished runs are reused on --resume, so an interrupted search can be continued.

Usage:
    python moabb_bench/search.py
    python moabb_bench/search.py --trials 20 --workers 3 --epochs 200
    python moabb_bench/search.py --resume experiments/search_<timestamp>
"""
import argparse
import os
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from sweep import PAPER_SETUP, REPO_ROOT, TRAIN_SCRIPT, completed_accuracy

SHARED_SPACE = {
    'learning_rate': [3e-4, 1e-3, 3e-3],
    'batch_size': [16, 32, 64],
    'dropout': [0.25, 0.5],
    'weight_decay': [0.0, 1e-3, 1e-2],
}
QUANTUM_SPACE = {
    'n_qubits': [4, 6, 9],
    'n_layers': [1, 2, 4],
}
PAPER_CONFIG = {'learning_rate': 1e-3, 'batch_size': 32, 'dropout': 0.25, 'weight_decay': 0.0,
                'n_qubits': 9, 'n_layers': 4}


def search_space(model):
    return {**SHARED_SPACE, **(QUANTUM_SPACE if model.startswith('quantum') else {})}


def sample_configs(model, n_trials, search_seed):
    """Trial 0 is the paper's settings; the rest are distinct random draws, identical on every call."""
    space = search_space(model)
    configs = [{k: PAPER_CONFIG[k] for k in space}]
    n_possible = 1
    for values in space.values():
        n_possible *= len(values)

    rng = random.Random(f'{search_seed}-{model}')
    while len(configs) < min(n_trials, n_possible):
        config = {k: rng.choice(values) for k, values in space.items()}
        if config not in configs:
            configs.append(config)
    return configs


def config_args(config):
    return [arg for k, v in config.items() for arg in (f'--{k}', str(v))]


def run_one(model, trial, config, subject, args, search_dir):
    log_path = os.path.join(search_dir, f'{model}_trial{trial:02d}_subject{subject}.log')
    previous = completed_accuracy(log_path, 'valid')
    if previous is not None:
        return previous, True

    cmd = [sys.executable, '-W', 'ignore', TRAIN_SCRIPT, '--dataset', 'bnci2014_001',
           '--model', model, '--subjects', str(subject), '--seed', str(args.seed),
           *PAPER_SETUP, '--epochs', str(args.epochs), *config_args(config)]
    with open(log_path, 'w') as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'failed; see {log_path}')
    return completed_accuracy(log_path, 'valid'), False


def format_summary(configs_by_model, results, subjects, args):
    lines = [f'# Hyperparameter search: BCI IV 2a, best validation accuracy (%)',
             f'Subjects {subjects}, seed {args.seed}, {args.epochs} epochs, paper preprocessing. '
             f'Trial 0 = paper settings.', '']
    best = {}

    for model, configs in configs_by_model.items():
        keys = list(search_space(model))
        header = ['Rank', 'Trial'] + keys + [f'S{s}' for s in subjects] + ['Mean']
        rows = []
        for trial, config in enumerate(configs):
            accs = [results.get((model, trial, s)) for s in subjects]
            done = [a for a in accs if isinstance(a, float)]
            mean = statistics.mean(done) if len(done) == len(subjects) else None
            rows.append((mean, trial, config, accs))
        rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))

        lines += [f'## {model}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        for rank, (mean, trial, config, accs) in enumerate(rows, 1):
            cells = [str(rank) if mean is not None else '-', str(trial)] + [str(config[k]) for k in keys]
            cells += [f'{a:.1f}' if isinstance(a, float) else (a or '-') for a in accs]
            cells.append(f'{mean:.1f}' if mean is not None else '-')
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
        if rows and rows[0][0] is not None:
            best[model] = rows[0]

    if best:
        lines += ['## Final evaluation commands (best configuration per model; test accuracy, all subjects)', '']
        for model, (mean, trial, config, _) in best.items():
            lines.append(f'- {model} (trial {trial}, mean validation {mean:.1f}%):')
            lines.append(f'  `python3 -W ignore moabb_bench/sweep.py --models {model} --seeds 1,2,3 '
                         f'--epochs {args.epochs} {" ".join(config_args(config))}`')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Random hyperparameter search on validation accuracy')
    parser.add_argument('--models', type=str, default='quantum_pooled,classical')
    parser.add_argument('--subjects', type=str, default='1,2,3', help='Subjects each configuration is scored on')
    parser.add_argument('--trials', type=int, default=20, help='Configurations per model, including the paper\'s')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1, help='Training seed for every run')
    parser.add_argument('--search_seed', type=int, default=0, help='Seed for drawing configurations')
    parser.add_argument('--workers', type=int, default=3, help='Trainings to run in parallel')
    parser.add_argument('--resume', type=str, default=None,
                        help='Existing search directory to continue (pass the same options)')
    args = parser.parse_args()

    models = args.models.split(',')
    subjects = [int(s) for s in args.subjects.split(',')]
    configs_by_model = {m: sample_configs(m, args.trials, args.search_seed) for m in models}

    if args.resume:
        search_dir = os.path.abspath(args.resume)
    else:
        search_dir = os.path.join(REPO_ROOT, 'experiments', f'search_{datetime.now():%Y%m%d_%H%M%S}')
        os.makedirs(search_dir)
    summary_path = os.path.join(search_dir, 'summary.md')

    # Quantum runs are slowest, so start them first to keep the workers busy at the end
    jobs = [(m, t, c, s) for m in sorted(models, key=lambda m: not m.startswith('quantum'))
            for t, c in enumerate(configs_by_model[m]) for s in subjects]
    results = {}
    print(f'{len(jobs)} runs ({args.trials} trials x {len(subjects)} subjects x {len(models)} models), '
          f'{args.workers} at a time. Summary: {summary_path}', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, m, t, c, s, args, search_dir): (m, t, s) for m, t, c, s in jobs}
        for done_count, future in enumerate(as_completed(futures), 1):
            model, trial, subject = futures[future]
            try:
                acc, reused = future.result()
                results[(model, trial, subject)] = acc
                status = f'{acc:.1f}%' + (' (already done)' if reused else '')
            except RuntimeError as e:
                results[(model, trial, subject)] = 'failed'
                status = str(e)
            print(f'[{done_count}/{len(jobs)}] {datetime.now():%H:%M} {model} trial {trial} '
                  f'subject {subject}: {status}', flush=True)
            with open(summary_path, 'w') as f:
                f.write(format_summary(configs_by_model, results, subjects, args))

    print('\n' + format_summary(configs_by_model, results, subjects, args))


if __name__ == '__main__':
    main()
