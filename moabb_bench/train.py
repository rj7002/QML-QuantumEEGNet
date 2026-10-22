#!/usr/bin/env python3
"""
Train QuantumEEGNet, PooledQuantumEEGNet or ClassicalEEGNet on any MOABB-registered
motor-imagery dataset.

Usage:
    python moabb_bench/train.py --dataset bnci2014_004 --epochs 50
    python moabb_bench/train.py --dataset bnci2014_004 --model quantum_pooled --subjects 1
    python moabb_bench/train.py --dataset physionetmi --model classical --subjects 1,2,3
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import torch

from src import QuantumEEGNet, ClassicalEEGNet, Trainer
from loader import MOABBDataset, SUPPORTED_DATASETS
from quantum_variants import PooledQuantumEEGNet
from ablations import ABLATIONS, build_ablation


def get_device(device_config: str) -> torch.device:
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)


def main():
    parser = argparse.ArgumentParser(description='Train on a MOABB motor-imagery dataset')

    parser.add_argument('--dataset', type=str, default='bnci2014_004',
                        choices=list(SUPPORTED_DATASETS.keys()),
                        help='MOABB dataset to train on')
    parser.add_argument('--model', type=str, default='quantum',
                        choices=['quantum', 'quantum_pooled', 'classical', *ABLATIONS],
                        help='quantum: original QEEGNet; quantum_pooled: circuit sees the whole trial; '
                             'ablate_*: quantum_pooled with the circuit swapped out (see ablations.py)')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated subject numbers (default: all subjects in the dataset)')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate in the convolution layers')
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--reupload', action='store_true',
                        help='quantum_pooled only: re-encode the inputs before every circuit layer')
    parser.add_argument('--rotations', type=str, default='ry', choices=['ry', 'full'],
                        help="quantum_pooled only: trainable gates, 'ry' (original) or 'full' (3-angle rotations)")

    parser.add_argument('--align', action='store_true',
                        help='Euclidean alignment per recording session (uses signals, not labels)')
    parser.add_argument('--augment', action='store_true',
                        help='Random time shift, amplitude scaling and noise on training trials')

    parser.add_argument('--fmin', type=float, default=8.0, help='Band-pass low cutoff (Hz)')
    parser.add_argument('--fmax', type=float, default=32.0, help='Band-pass high cutoff (Hz)')
    parser.add_argument('--resample', type=float, default=None, help='Resample to this rate (Hz)')
    parser.add_argument('--tmin', type=float, default=0.0, help='Window start, seconds after cue')
    parser.add_argument('--tmax', type=float, default=None, help='Window end, seconds after cue')

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for weight init, data split and shuffling (use the same one across models)')

    args = parser.parse_args()
    if args.model != 'quantum_pooled' and (args.reupload or args.rotations != 'ry'):
        parser.error('--reupload and --rotations only apply to --model quantum_pooled')

    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    subjects = [int(s) for s in args.subjects.split(',')] if args.subjects else None

    print(f"\nLoading MOABB dataset '{args.dataset}'...")
    dataset = MOABBDataset(
        dataset_name=args.dataset, subjects=subjects, seed=args.seed,
        fmin=args.fmin, fmax=args.fmax, resample=args.resample, tmin=args.tmin, tmax=args.tmax,
        align=args.align
    )
    info = dataset.get_data_info()
    print(f"Dataset info: {info}")

    train_loader, valid_loader, test_loader = dataset.get_dataloaders(
        batch_size=args.batch_size, augment=args.augment)

    _, input_channels, input_length = info['input_shape']  # (1, channels, time)
    num_classes = info['num_classes']

    print(f"\nCreating {args.model} model "
          f"(channels={input_channels}, length={input_length}, classes={num_classes})...")

    if args.model in ('quantum', 'quantum_pooled', *ABLATIONS):
        model_kwargs = dict(
            num_classes=num_classes, n_qubits=args.n_qubits, n_layers=args.n_layers,
            input_channels=input_channels, input_length=input_length, dropout_rate=args.dropout
        )
        if args.model == 'quantum':
            model = QuantumEEGNet(**model_kwargs)
        elif args.model in ABLATIONS:
            model = build_ablation(args.model, **model_kwargs)
        else:
            model = PooledQuantumEEGNet(reupload=args.reupload, rotations=args.rotations, **model_kwargs)
    else:
        model = ClassicalEEGNet(
            num_classes=num_classes, input_channels=input_channels, input_length=input_length,
            dropout_rate=args.dropout
        )

    print(f"Total parameters: {model.count_parameters():,}")

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config={
            'model_name': f'{args.model}_eegnet_{args.dataset}',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'optimizer': 'adamw',
            'patience': args.patience,
            'num_classes': num_classes,
            'run_settings': vars(args),
        }
    )

    print("\nStarting training...")
    results = trainer.train()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_valid_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Results saved to: {trainer.experiment_dir}")


if __name__ == '__main__':
    main()
