"""Quantum model variants for the MOABB benchmarks; the originals in src/models are left unchanged."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from src.models.quantum_eegnet import QuantumEEGNet


class BatchedQuantumLayer(nn.Module):
    """
    The circuit from src.models.QuantumLayer, evaluated a whole batch at a time, with two options:

    reupload: re-encode the inputs before every circuit layer instead of only the first
        ("data re-uploading", Perez-Salinas et al., 2020), which makes small circuits more expressive.
    rotations: 'ry' trains one RY angle per qubit per layer (the original circuit, which can only
        reach real-valued states); 'full' trains a general rotation (RZ-RY-RZ, 3 angles) instead.

    With reupload=False and rotations='ry' the circuit is gate-for-gate the original.
    """

    def __init__(self, n_qubits: int, n_layers: int, reupload: bool = False, rotations: str = "ry",
                 device_name: str = "default.qubit"):
        super().__init__()
        if rotations not in ("ry", "full"):
            raise ValueError(f"rotations must be 'ry' or 'full', got '{rotations}'")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.rotations = rotations
        self.device_name = device_name

        dev = qml.device(device_name, wires=n_qubits)

        # inputs[..., i] (not inputs[i]) so PennyLane can broadcast over the batch dimension
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            for j in range(n_layers):
                if j == 0 or reupload:
                    for i in range(n_qubits):
                        qml.RY(inputs[..., i], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                for i in range(n_qubits):
                    if rotations == "full":
                        qml.Rot(*weights[j, i], wires=i)
                    else:
                        qml.RY(weights[j, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shape = (n_layers, n_qubits, 3) if rotations == "full" else (n_layers, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_layer(x)

    def get_circuit_info(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "reupload": self.reupload,
            "rotations": self.rotations,
            "device": self.device_name,
            "total_parameters": self.q_layer.weights.numel(),
        }


class PooledQuantumEEGNet(QuantumEEGNet):
    """
    QuantumEEGNet whose circuit sees the whole trial.

    The original model passes each filter's full time series to a circuit that only encodes
    the first n_qubits values, so everything after the start of the trial is discarded. Here
    each filter's time series is average-pooled down to n_qubits values (one per segment of the
    trial) and squashed into [-pi, pi], since RY rotations wrap around outside that range.
    """

    def __init__(self, reupload: bool = False, rotations: str = "ry", **kwargs):
        super().__init__(**kwargs)
        self.quantum_layer = BatchedQuantumLayer(self.n_qubits, self.n_layers,
                                                 reupload=reupload, rotations=rotations)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = F.elu(self.batchnorm2(self.conv2(x)))
        x = self.dropout(F.avg_pool2d(x, (1, 4)))
        x = F.elu(self.batchnorm3(self.conv3(x)))
        x = F.elu(self.batchnorm4(self.conv4(x)))
        return self.dropout(F.avg_pool2d(x, (1, 8)))

    def circuit_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation angles fed to the circuit: (batch, filters, n_qubits)."""
        x = self._conv_features(x)
        batch, filters = x.shape[0], x.shape[1]
        x = F.adaptive_avg_pool1d(x.view(batch, filters, -1), self.n_qubits)
        return torch.pi * torch.tanh(x)

    def classify(self, expvals: torch.Tensor) -> torch.Tensor:
        """Class logits from circuit outputs shaped (batch, filters, n_qubits)."""
        return self.fc1(expvals.reshape(expvals.shape[0], -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.circuit_inputs(x)
        batch, filters = angles.shape[0], angles.shape[1]
        expvals = self.quantum_layer(angles.reshape(batch * filters, self.n_qubits))
        return self.classify(expvals.reshape(batch, filters, self.n_qubits))

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["model_type"] = "PooledQuantumEEGNet"
        return info
