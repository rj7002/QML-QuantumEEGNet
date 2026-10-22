"""
Ablations of PooledQuantumEEGNet: the same model with the circuit swapped out, to test whether the
circuit itself contributes. Everything else (convolutions, pooling to n_qubits angles per filter,
the pi*tanh squashing, the classifier) is unchanged.

  ablate_none    no circuit: the angles go straight to the classifier
  ablate_mlp     a small trainable classical layer in place of the circuit (Linear + tanh, shared
                 across filters like the circuit is, outputs in [-1, 1] like <Z>)
  ablate_frozen  the original circuit with its weights left at their random initialization
"""
import torch
import torch.nn as nn

from quantum_variants import PooledQuantumEEGNet

ABLATIONS = ('ablate_none', 'ablate_mlp', 'ablate_frozen')


class _ClassicalBottleneck(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.linear = nn.Linear(n_qubits, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


def build_ablation(kind: str, **kwargs) -> PooledQuantumEEGNet:
    if kind not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{kind}', expected one of {ABLATIONS}")
    model = PooledQuantumEEGNet(**kwargs)
    if kind == 'ablate_none':
        model.quantum_layer = nn.Identity()
    elif kind == 'ablate_mlp':
        model.quantum_layer = _ClassicalBottleneck(model.n_qubits)
    else:
        model.quantum_layer.q_layer.weights.requires_grad_(False)
    return model
