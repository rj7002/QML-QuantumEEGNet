"""
Generic loader for MOABB-registered motor-imagery EEG datasets.

This module is intentionally kept outside `src/` so the main repo's dataset code
(BCIDataset/SyntheticDataset) is untouched — it's a standalone adapter that plugs
any MOABB motor-imagery dataset into the existing QuantumEEGNet/ClassicalEEGNet/Trainer
classes via the same get_dataloaders()/get_data_info() interface.
"""
import importlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as Data


# key -> (moabb module, class name). Subject/channel/class/rate info from MOABB's docs,
# current as of this integration; verify against `moabb.datasets.<Class>().subject_list`
# if a specific dataset's numbers matter for your analysis.
SUPPORTED_DATASETS = {
    'bnci2014_001': ('moabb.datasets', 'BNCI2014_001'),        # BCI IV 2a: 9 subj, 22ch, 4-class, 250Hz
    'bnci2014_004': ('moabb.datasets', 'BNCI2014_004'),        # BCI IV 2b: 9 subj, 3ch, 2-class, 250Hz
    'physionetmi': ('moabb.datasets', 'PhysionetMI'),          # 109 subj, 64ch, 4-class, 160Hz
    'lee2019_mi': ('moabb.datasets', 'Lee2019_MI'),            # 54 subj, 62ch, 2-class, 1000Hz
    'cho2017': ('moabb.datasets', 'Cho2017'),                  # 52 subj, 64ch, 2-class, 512Hz
    'schirrmeister2017': ('moabb.datasets', 'Schirrmeister2017'),  # 14 subj, 128ch, 4-class, 500Hz
    'weibo2014': ('moabb.datasets', 'Weibo2014'),              # 10 subj, 60ch, 7-class, 200Hz
}


def _import_dataset_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def euclidean_align(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Euclidean alignment (He & Wu, 2019): whiten each group's trials by that group's mean
    spatial covariance, so every recording session has identity mean covariance. Uses only
    the signals, never the labels.
    """
    aligned = np.empty_like(X)
    for group in np.unique(groups):
        idx = groups == group
        mean_cov = np.einsum('nct,ndt->cd', X[idx], X[idx]) / (idx.sum() * X.shape[2])
        eigvals, eigvecs = np.linalg.eigh(mean_cov)
        # Floor tiny eigenvalues: average-referenced recordings have a rank-deficient covariance
        eigvals = np.maximum(eigvals, eigvals.max() * 1e-6)
        inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        aligned[idx] = np.einsum('cd,ndt->nct', inv_sqrt, X[idx])
    return aligned


class AugmentedDataset(Data.Dataset):
    """Training-set wrapper that draws a fresh random time shift, amplitude scale and noise per sample."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_shift_frac: float = 0.05,
                 scale_range: Tuple[float, float] = (0.9, 1.1), noise_std: float = 0.05):
        self.x, self.y = x, y
        self.max_shift = int(x.shape[-1] * max_shift_frac)
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]

        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        if shift > 0:
            x = torch.cat([torch.zeros_like(x[..., :shift]), x[..., :-shift]], dim=-1)
        elif shift < 0:
            x = torch.cat([x[..., -shift:], torch.zeros_like(x[..., :-shift])], dim=-1)

        low, high = self.scale_range
        x = x * (low + (high - low) * torch.rand(1))
        x = x + self.noise_std * x.std() * torch.randn_like(x)
        return x, self.y[i]


class MOABBDataset:
    """
    Loads a MOABB motor-imagery dataset and exposes it through the same interface as
    this repo's BCIDataset/SyntheticDataset (get_dataloaders / get_data_info), so any
    MOABB-registered dataset in SUPPORTED_DATASETS can be swapped in without touching
    the model or training code.

    Requires the `moabb` package (`pip install moabb`). First use of a given dataset
    downloads and caches its raw files under `~/mne_data`.
    """

    def __init__(self, dataset_name: str, subjects: Optional[List[int]] = None,
                 validation_ratio: float = 0.2, test_ratio: float = 0.2,
                 fmin: float = 8.0, fmax: float = 32.0, resample: Optional[float] = None,
                 tmin: float = 0.0, tmax: Optional[float] = None, align: bool = False,
                 seed: int = 42):
        """
        Args:
            dataset_name: key into SUPPORTED_DATASETS (e.g. 'bnci2014_004')
            subjects: subset of subject numbers to load (default: the dataset's full subject list)
            validation_ratio: fraction of training trials held out for validation
            test_ratio: fraction of all trials held out as a test set; only used for datasets
                without official test sessions
            fmin/fmax: bandpass filter range (Hz) applied by MOABB's MotorImagery paradigm
            resample: optional resampling rate (Hz); leave None to keep the dataset's native rate
            tmin/tmax: trial window in seconds relative to the cue (tmax=None: the dataset's
                default trial end)
            align: apply Euclidean alignment separately to each subject's recording session
            seed: seed for the within-class shuffles
        """
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(SUPPORTED_DATASETS.keys())}"
            )

        try:
            import moabb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'moabb' package is required for MOABBDataset. Install it with: "
                "pip install -r moabb_bench/requirements.txt"
            ) from e

        from moabb.paradigms import MotorImagery

        self.dataset_name = dataset_name
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        module_name, class_name = SUPPORTED_DATASETS[dataset_name]
        dataset_cls = _import_dataset_class(module_name, class_name)
        self.moabb_dataset = dataset_cls()

        self.subjects = subjects or list(self.moabb_dataset.subject_list)

        paradigm = MotorImagery(fmin=fmin, fmax=fmax, resample=resample, tmin=tmin, tmax=tmax)
        X, y, _metadata = paradigm.get_data(dataset=self.moabb_dataset, subjects=self.subjects)
        # X: (n_trials, n_channels, n_times) ; y: array of string class labels

        self.align = align
        if align:
            groups = (_metadata['subject'].astype(str) + '/' + _metadata['session'].astype(str)).values
            X = euclidean_align(X, groups)

        self.classes = sorted(set(y))
        self.label_map = {label: i for i, label in enumerate(self.classes)}

        x = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, channels, time)
        y_int = torch.tensor([self.label_map[label] for label in y], dtype=torch.long)

        generator = torch.Generator().manual_seed(seed)
        sessions = _metadata['session'].astype(str).values
        is_test = torch.tensor(['test' in s for s in sessions])

        # Datasets with official train/test sessions (e.g. BCI IV 2a/2b) are split on them,
        # matching how the repo's BCIDataset uses the T/E files; others get a random split.
        if is_test.any() and not is_test.all():
            self.split_mode = 'session'
            self.test_sessions = sorted({s for s in sessions if 'test' in s})
            self.x_test, self.y_test = x[is_test], y_int[is_test]
            self.x_train, self.y_train, self.x_valid, self.y_valid = self._split_by_class(
                x[~is_test], y_int[~is_test], validation_ratio, generator)
        else:
            self.split_mode = 'random'
            self.test_sessions = []
            x_rest, y_rest, self.x_test, self.y_test = self._split_by_class(
                x, y_int, test_ratio, generator)
            self.x_train, self.y_train, self.x_valid, self.y_valid = self._split_by_class(
                x_rest, y_rest, validation_ratio, generator)

    def _split_by_class(self, x: torch.Tensor, y: torch.Tensor, held_ratio: float,
                        generator: torch.Generator):
        """Shuffle within each class and hold out `held_ratio` of it; returns (kept, held) pairs."""
        kept_x, kept_y, held_x, held_y = [], [], [], []

        for label in torch.unique(y):
            idx = torch.nonzero(y == label).flatten()
            idx = idx[torch.randperm(len(idx), generator=generator)]
            n_held = int(len(idx) * held_ratio)

            held_x.append(x[idx[:n_held]])
            held_y.append(y[idx[:n_held]])
            kept_x.append(x[idx[n_held:]])
            kept_y.append(y[idx[n_held:]])

        return torch.cat(kept_x), torch.cat(kept_y), torch.cat(held_x), torch.cat(held_y)

    def get_dataloaders(self, batch_size: int = 64, num_workers: int = 0, augment: bool = False):
        """Get PyTorch DataLoaders for training, validation, and test sets (augmentation: training only)."""
        if augment:
            train_dataset = AugmentedDataset(self.x_train, self.y_train)
        else:
            train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)

        train_loader = Data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        valid_loader = Data.DataLoader(
            valid_dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = Data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        return train_loader, valid_loader, test_loader

    def get_data_info(self) -> Dict:
        """Get information about the loaded dataset."""
        return {
            "dataset_type": self.dataset_name,
            "subjects": self.subjects,
            "split_mode": self.split_mode,
            "test_sessions": self.test_sessions,
            "euclidean_alignment": self.align,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "train_samples": len(self.x_train),
            "valid_samples": len(self.x_valid),
            "test_samples": len(self.x_test),
            "input_shape": tuple(self.x_train.shape[1:]),
            "class_distribution": {
                "train": torch.bincount(self.y_train).tolist(),
                "valid": torch.bincount(self.y_valid).tolist(),
                "test": torch.bincount(self.y_test).tolist(),
            }
        }
