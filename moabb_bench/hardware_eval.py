#!/usr/bin/env python3
"""
Evaluate a trained PooledQuantumEEGNet with its circuit executed on another quantum backend.

The convolution layers and the classifier run in PyTorch exactly as in training; only the circuit
evaluations (one per test trial per filter) go to the chosen backend. Any change in accuracy is
therefore caused by shot noise and hardware noise alone. Nothing is retrained.

Backends:
  exact      PennyLane statevector, as in training (reproduces the run's test accuracy)
  shots      noise-free sampling with a finite number of shots (Qiskit Aer)
  aer_noise  local Aer simulation with depolarizing gate errors and readout errors set by
             --p1/--p2/--readout (a simple stand-in for hardware noise)
  ionq_sim   IonQ cloud simulator with an IonQ hardware noise model (needs IONQ_API_KEY)
  ionq_qpu   real IonQ hardware (needs IONQ_API_KEY; billed per shot; needs --confirm_qpu)

Usage:
  python moabb_bench/hardware_eval.py --sweep_dir experiments/sweep_X --subject 3 --seed 1 --backend shots
  python moabb_bench/hardware_eval.py --run_dir experiments/quantum_pooled_... --backend ionq_sim --max_trials 48
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch

from loader import MOABBDataset
from quantum_variants import PooledQuantumEEGNet

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_dotenv(path=os.path.join(REPO_ROOT, '.env')):
    """Set KEY=VALUE lines from the repo's .env file, without overriding variables already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.removeprefix('export ').split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def resolve_run_dir(args):
    if args.run_dir:
        return os.path.join(REPO_ROOT, args.run_dir) if not os.path.isabs(args.run_dir) else args.run_dir
    log = os.path.join(args.sweep_dir, f'quantum_pooled_subject{args.subject}_seed{args.seed}.log')
    with open(log) as f:
        return os.path.join(REPO_ROOT, re.findall(r'Results saved to: (\S+)', f.read())[-1])


def load_run(run_dir):
    """Rebuild the trained model and its test data from a finished train.py run."""
    with open(os.path.join(run_dir, 'results.json')) as f:
        results = json.load(f)
    s = results['config']['run_settings']
    if s['model'] != 'quantum_pooled':
        raise ValueError(f"{run_dir} is a '{s['model']}' run; only quantum_pooled runs can be evaluated")

    subjects = [int(x) for x in s['subjects'].split(',')]
    data = MOABBDataset(dataset_name=s['dataset'], subjects=subjects, seed=s['seed'], fmin=s['fmin'],
                        fmax=s['fmax'], resample=s['resample'], tmin=s['tmin'], tmax=s['tmax'],
                        align=s.get('align', False))
    _, channels, length = data.x_test.shape[1:]

    model = PooledQuantumEEGNet(
        num_classes=len(data.classes), n_qubits=s['n_qubits'], n_layers=s['n_layers'],
        input_channels=channels, input_length=length, dropout_rate=s.get('dropout', 0.25),
        reupload=s.get('reupload', False), rotations=s.get('rotations', 'ry'))
    checkpoint = torch.load(os.path.join(run_dir, 'model.pth'), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, data, s, results['test_accuracy']


def subsample(y, max_trials, seed=0):
    """Indices of a class-balanced subset of at most max_trials trials (all trials if None)."""
    if max_trials is None or max_trials >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max_trials // len(classes)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in classes]))


def build_circuit(model):
    """Qiskit version of the trained circuit, with the trained weights fixed and the inputs left free."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    layer = model.quantum_layer
    n = layer.n_qubits
    weights = layer.q_layer.weights.detach().numpy()
    inputs = ParameterVector('x', n)
    qc = QuantumCircuit(n)
    for j in range(layer.n_layers):
        if j == 0 or layer.reupload:
            for i in range(n):
                qc.ry(inputs[i], i)
        for i in range(n):
            qc.cx(i, (i + 1) % n)
        for i in range(n):
            if layer.rotations == 'full':
                phi, theta, omega = weights[j, i]  # qml.Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
                qc.rz(float(phi), i)
                qc.ry(float(theta), i)
                qc.rz(float(omega), i)
            else:
                qc.ry(float(weights[j, i]), i)
    qc.measure_all()
    return qc, inputs


def expvals_from_counts(counts, n_qubits):
    """<Z_i> for each qubit; Qiskit bitstrings list qubit 0 as the rightmost character."""
    shots = sum(counts.values())
    z = np.zeros(n_qubits)
    for bits, c in counts.items():
        bits = bits.replace(' ', '')
        for i in range(n_qubits):
            z[i] += c if bits[-1 - i] == '0' else -c
    return z / shots


def make_noise_model(p1, p2, readout):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    noise = NoiseModel()
    if p1 > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['ry', 'rz'])
    if p2 > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    if readout > 0:
        noise.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return noise


def run_circuits(model, angles, args):
    """Circuit outputs <Z_i> for every row of angles (M, n_qubits) on the chosen backend."""
    if args.backend == 'exact':
        with torch.no_grad():
            return model.quantum_layer(torch.tensor(angles, dtype=torch.float32)).numpy()

    template, inputs = build_circuit(model)
    circuits = [template.assign_parameters(dict(zip(inputs, row))) for row in angles]
    n = template.num_qubits

    if args.backend in ('shots', 'aer_noise'):
        from qiskit_aer import AerSimulator
        noise = make_noise_model(args.p1, args.p2, args.readout) if args.backend == 'aer_noise' else None
        backend = AerSimulator(noise_model=noise, seed_simulator=args.sample_seed)
    else:
        from qiskit_ionq import IonQProvider
        load_dotenv()
        if not os.environ.get('IONQ_API_KEY'):
            sys.exit('IONQ_API_KEY is not set. Add it to .env in the repo root or export it.')
        provider = IonQProvider()
        if args.backend == 'ionq_sim':
            backend = provider.get_backend('ionq_simulator')
            backend.set_options(noise_model=args.ionq_noise_model)
        else:
            backend = provider.get_backend(args.qpu)

    out = []
    for start in range(0, len(circuits), args.chunk):
        chunk = circuits[start:start + args.chunk]
        result = backend.run(chunk, shots=args.shots).result()
        out.extend(expvals_from_counts(result.get_counts(i), n) for i in range(len(chunk)))
        print(f'  circuits {start + len(chunk)}/{len(circuits)}', flush=True)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained quantum model on another backend')
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--run_dir', type=str, help='Folder of a finished quantum_pooled train.py run')
    which.add_argument('--sweep_dir', type=str, help='Sweep folder; pick the run with --subject and --seed')
    parser.add_argument('--subject', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--backend', type=str, default='shots',
                        choices=['exact', 'shots', 'aer_noise', 'ionq_sim', 'ionq_qpu'])
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Evaluate a class-balanced subset of test trials (keeps cloud jobs small)')
    parser.add_argument('--chunk', type=int, default=500, help='Circuits per submitted job')
    parser.add_argument('--p1', type=float, default=0.001, help='aer_noise: 1-qubit depolarizing probability')
    parser.add_argument('--p2', type=float, default=0.01, help='aer_noise: 2-qubit (CNOT) depolarizing probability')
    parser.add_argument('--readout', type=float, default=0.01, help='aer_noise: readout bit-flip probability')
    parser.add_argument('--ionq_noise_model', type=str, default='aria-1', help='ionq_sim noise model')
    parser.add_argument('--qpu', type=str, default='qpu.forte-1', help='ionq_qpu device name')
    parser.add_argument('--confirm_qpu', action='store_true', help='Required for ionq_qpu (billed per shot)')
    parser.add_argument('--sample_seed', type=int, default=0, help='Seed for local shot sampling')
    args = parser.parse_args()
    if args.sweep_dir and (args.subject is None or args.seed is None):
        parser.error('--sweep_dir needs --subject and --seed')

    run_dir = resolve_run_dir(args)
    model, data, settings, saved_test_acc = load_run(run_dir)
    idx = subsample(data.y_test.numpy(), args.max_trials)
    x, y = data.x_test[idx], data.y_test[idx].numpy()

    with torch.no_grad():
        angles = model.circuit_inputs(x).numpy()  # (trials, filters, qubits)
    trials, filters, qubits = angles.shape
    n_circuits = trials * filters
    print(f'Run: {run_dir}\nSubject {settings["subjects"]}, seed {settings["seed"]}: {trials} test trials x '
          f'{filters} filters = {n_circuits} circuits of {qubits} qubits, {args.shots} shots each, '
          f'backend {args.backend}')

    if args.backend == 'ionq_qpu' and not args.confirm_qpu:
        sys.exit(f'Real hardware is billed per shot: this would run {n_circuits} circuits x {args.shots} '
                 f'shots = {n_circuits * args.shots:,} shots. Re-run with --confirm_qpu to submit.')

    with torch.no_grad():
        exact_vals = model.quantum_layer(torch.tensor(angles.reshape(-1, qubits), dtype=torch.float32)).numpy()
        exact_pred = model.classify(torch.tensor(exact_vals.reshape(trials, filters, qubits))).argmax(1).numpy()
        vals = run_circuits(model, angles.reshape(-1, qubits), args)
        pred = model.classify(torch.tensor(vals.reshape(trials, filters, qubits), dtype=torch.float32)).argmax(1).numpy()

    acc, exact_acc = 100 * (pred == y).mean(), 100 * (exact_pred == y).mean()
    summary = {
        'run_dir': run_dir, 'subject': settings['subjects'], 'seed': settings['seed'], 'backend': args.backend,
        'shots': args.shots, 'trials': int(trials), 'circuits': int(n_circuits),
        'noise': ({'p1': args.p1, 'p2': args.p2, 'readout': args.readout} if args.backend == 'aer_noise'
                  else args.ionq_noise_model if args.backend == 'ionq_sim'
                  else args.qpu if args.backend == 'ionq_qpu' else None),
        'accuracy': acc, 'exact_accuracy_same_trials': exact_acc, 'saved_test_accuracy_all_trials': saved_test_acc,
        'prediction_agreement_with_exact': 100 * (pred == exact_pred).mean(),
        'mean_abs_expval_error': float(np.abs(vals - exact_vals).mean()),
    }
    print(f'\nAccuracy on {args.backend}: {acc:.1f}%  (noise-free on the same trials: {exact_acc:.1f}%)')
    print(f'Predictions matching noise-free: {summary["prediction_agreement_with_exact"]:.1f}%  |  '
          f'mean |<Z> error|: {summary["mean_abs_expval_error"]:.4f}')

    out_dir = os.path.join(REPO_ROOT, 'experiments', 'hardware_eval')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'_p2-{args.p2}' if args.backend == 'aer_noise' else ''
    out = os.path.join(out_dir, f'subject{settings["subjects"]}_seed{settings["seed"]}_{args.backend}{tag}_'
                                f'{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
