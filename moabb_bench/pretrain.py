#!/usr/bin/env python3
"""
Cross-subject pretraining, then fine-tuning on the target subject.

Stage 1 trains on the training sessions of every other subject in the dataset (all of their
session-1 trials; their test sessions are never loaded into training). Stage 2 continues from the
best stage-1 checkpoint on the target subject's own training trials. Both stages pick their
checkpoint on the target subject's validation trials, and only the final model is tested on the
target subject's test session.

Takes the same options as train.py (so sweep.py can run it with --script pretrain); --epochs is the
number of fine-tuning epochs. Use --align: it puts every subject's recordings on a common scale.

Usage:
    python moabb_bench/pretrain.py --dataset bnci2014_001 --model quantum_pooled --subjects 1 --align
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import torch
import torch.utils.data as Data

from src import ClassicalEEGNet, Trainer
from loader import AugmentedDataset, MOABBDataset, SUPPORTED_DATASETS
from quantum_variants import PooledQuantumEEGNet
from train import get_device


def build_parser():
    p = argparse.ArgumentParser(description='Cross-subject pretraining + fine-tuning')
    p.add_argument('--dataset', type=str, default='bnci2014_001', choices=list(SUPPORTED_DATASETS))
    p.add_argument('--model', type=str, default='quantum_pooled', choices=['quantum_pooled', 'classical'])
    p.add_argument('--subjects', type=str, required=True, help='The single target subject')
    p.add_argument('--epochs', type=int, default=100, help='Fine-tuning epochs on the target subject')
    p.add_argument('--pretrain_epochs', type=int, default=50, help='Epochs on the other subjects')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-3, help='Pretraining learning rate')
    p.add_argument('--finetune_lr', type=float, default=3e-4, help='Fine-tuning learning rate')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--n_qubits', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--reupload', action='store_true')
    p.add_argument('--rotations', type=str, default='ry', choices=['ry', 'full'])
    p.add_argument('--align', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--fmin', type=float, default=8.0)
    p.add_argument('--fmax', type=float, default=32.0)
    p.add_argument('--resample', type=float, default=None)
    p.add_argument('--tmin', type=float, default=0.0)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    target = int(args.subjects)
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    def load(subject):
        return MOABBDataset(dataset_name=args.dataset, subjects=[subject], seed=args.seed, fmin=args.fmin,
                            fmax=args.fmax, resample=args.resample, tmin=args.tmin, tmax=args.tmax,
                            align=args.align)

    print(f'Loading target subject {target}...')
    target_data = load(target)
    sources = [s for s in target_data.moabb_dataset.subject_list if s != target]
    print(f'Loading source subjects {sources} (training sessions only)...')
    source_x, source_y = [], []
    for s in sources:
        d = load(s)
        source_x += [d.x_train, d.x_valid]
        source_y += [d.y_train, d.y_valid]
    source_x, source_y = torch.cat(source_x), torch.cat(source_y)
    print(f'Pretraining trials: {len(source_y)} | target train/valid/test: '
          f'{len(target_data.y_train)}/{len(target_data.y_valid)}/{len(target_data.y_test)}')

    _, channels, length = target_data.x_train.shape[1:]
    num_classes = len(target_data.classes)
    if args.model == 'quantum_pooled':
        model = PooledQuantumEEGNet(num_classes=num_classes, n_qubits=args.n_qubits, n_layers=args.n_layers,
                                    input_channels=channels, input_length=length, dropout_rate=args.dropout,
                                    reupload=args.reupload, rotations=args.rotations)
    else:
        model = ClassicalEEGNet(num_classes=num_classes, input_channels=channels, input_length=length,
                                dropout_rate=args.dropout)

    def train_loader(x, y):
        dataset = AugmentedDataset(x, y) if args.augment else Data.TensorDataset(x, y)
        return Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    _, valid_loader, test_loader = target_data.get_dataloaders(batch_size=args.batch_size)
    base_config = {'optimizer': 'adamw', 'weight_decay': args.weight_decay, 'patience': args.patience,
                   'num_classes': num_classes, 'run_settings': vars(args)}

    print(f'\nStage 1: pretraining for {args.pretrain_epochs} epochs on {len(sources)} other subjects')
    # The target's validation set stands in for the test loader so stage 1 never touches test data
    Trainer(model=model, device=device, train_loader=train_loader(source_x, source_y), valid_loader=valid_loader,
            test_loader=valid_loader,
            config={**base_config, 'model_name': f'{args.model}_pretrain_{args.dataset}_s{target}',
                    'epochs': args.pretrain_epochs, 'learning_rate': args.learning_rate}).train()

    print(f'\nStage 2: fine-tuning for {args.epochs} epochs on subject {target}')
    trainer = Trainer(model=model, device=device,
                      train_loader=train_loader(target_data.x_train, target_data.y_train),
                      valid_loader=valid_loader, test_loader=test_loader,
                      config={**base_config, 'model_name': f'{args.model}_finetune_{args.dataset}',
                              'epochs': args.epochs, 'learning_rate': args.finetune_lr})
    results = trainer.train()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_valid_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Results saved to: {trainer.experiment_dir}")


if __name__ == '__main__':
    main()
