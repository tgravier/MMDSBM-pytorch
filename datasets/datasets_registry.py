import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_circles, make_moons
from typing import Optional, Union, List

# ============================
# === Dataset Registry Core ===
# ============================

ARTIFICIAL_DATASETS = ["gaussian", "spiral", "moon", "circle", "gaussian_mixture"]
REAL_DATASETS = ["mnist", "cifar10"]

class DatasetConfig:
    def __init__(self, name: str, time: float, input_dim: Optional[int] = None, **params):
        if not self.is_valid_name(name):
            raise ValueError(f"Dataset '{name}' is not recognized.")
        if not isinstance(time, (float, int)):
            raise TypeError(f"'time' must be a number (float or int), got {type(time)}")

        self.name = name
        self.type = self.get_type(name)
        self.input_dim = input_dim
        self.time = float(time)
        self.params = params

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return name in ARTIFICIAL_DATASETS or name in REAL_DATASETS

    @staticmethod
    def get_type(name: str) -> str:
        if name in ARTIFICIAL_DATASETS:
            return "artificial"
        elif name in REAL_DATASETS:
            return "real"
        else:
            raise ValueError(f"Unknown dataset type for '{name}'")

    def get_time(self) -> float:
        return self.time

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "input_dim": self.input_dim,
            "time": self.time,
            "params": self.params,
        }

    def __repr__(self) -> str:
        return (f"DatasetConfig(name={self.name}, type={self.type}, "
                f"input_dim={self.input_dim}, time={self.time}, "
                f"params={self.params})")

# ============================
# === Dataset Variants ===
# ============================

class GaussianConfig(DatasetConfig):
    def __init__(self, time, mean=0.0, std=1.0, n_samples=1000, dim=2):
        mean_vec = self._ensure_vector(mean, dim)
        std_vec = self._ensure_vector(std, dim)
        super().__init__("gaussian", time, input_dim=dim,
                         mean=mean_vec, std=std_vec, n_samples=n_samples)

    @staticmethod
    def _ensure_vector(val, dim):
        if np.isscalar(val):
            return [val] * dim
        elif isinstance(val, (list, tuple, np.ndarray)):
            val = list(val)
            if len(val) != dim:
                raise ValueError(f"Expected {dim}-dimensional vector, got {len(val)}")
            return val
        else:
            raise TypeError("mean and std must be scalars or sequences")

class CircleConfig(DatasetConfig):
    def __init__(self, time, n_samples=1000, center=(0.0, 0.0), radius=1.0, thickness=0.05):
        super().__init__(
            "circle", time, input_dim=2,
            n_samples=n_samples, center=center, radius=radius, thickness=thickness
        )

class MoonConfig(DatasetConfig):
    def __init__(self, time, n_samples=1000, noise=0.1):
        self._validate_noise(noise)
        super().__init__(
            "moon", time, input_dim=2,
            n_samples=n_samples, noise=noise
        )

    @staticmethod
    def _validate_noise(noise):
        if noise < 0:
            raise ValueError(f"'noise' must be >= 0, got {noise}")


class GaussianMixtureConfig(DatasetConfig):
    def __init__(self, time, means, stds, weights=None, n_samples=1000):
        means = np.array(means)
        stds = np.array(stds)

        n_components, dim = means.shape

        if stds.shape != (n_components, dim):
            raise ValueError(f"Expected stds of shape ({n_components}, {dim}), got {stds.shape}")

        if weights is None:
            weights = np.ones(n_components) / n_components
        else:
            weights = np.array(weights)
            if weights.shape != (n_components,):
                raise ValueError(f"weights must have shape ({n_components},), got {weights.shape}")
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("weights must sum to 1")

        super().__init__(
            name="gaussian_mixture",
            time=time,
            input_dim=dim,
            means=means.tolist(),
            stds=stds.tolist(),
            weights=weights.tolist(),
            n_samples=n_samples
        )
