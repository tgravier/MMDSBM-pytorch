import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles
import numpy as np

from datasets.datasets_registry import DatasetConfig  

#datasets/datasets.py

# ============================
# === Timed Dataset Class ===
# ============================

class TimedDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, time: float):
        if not isinstance(time, (float, int)):
            raise TypeError(f"'time' must be a float or int, got {type(time)}")
        self.data = data
        self.labels = labels
        self.time = float(time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_time(self) -> float:
        return self.time
    
    def get_all(self):
        return self.data

    def __repr__(self):
        return f"TimedDataset(len={len(self)}, time={self.time})"


# ============================
# === Dataset Loader Func ===
# ============================

def load_dataset(config: DatasetConfig) -> TimedDataset:
    name = config.name
    params = config.params
    dim = config.input_dim
    n = params.get("n_samples", 1000)
    time = config.time

    if name == "gaussian":
        mean = np.array(params.get("mean", [0.0] * dim))
        std = np.array(params.get("std", [1.0] * dim))

        if mean.shape[0] != dim:
            raise ValueError(f"Mean vector has incorrect dimension: expected {dim}, got {mean.shape[0]}")
        if std.shape[0] != dim:
            raise ValueError(f"Std vector has incorrect dimension: expected {dim}, got {std.shape[0]}")

        data = np.random.normal(loc=mean, scale=std, size=(n, dim))
        labels = np.zeros(n)
        return TimedDataset(torch.tensor(data, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long),
                            time)

    elif name == "circle":
        center = np.array(params.get("center", [0.0, 0.0]))
        radius = params.get("radius", 1.0)
        thickness = params.get("thickness", 0.05)
        n = params.get("n_samples", 1000)
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.normal(radius, thickness, n)
        X = np.stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ], axis=1)
        y = np.zeros(n)
        return TimedDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long),
                            time)

    elif name == "moon":
        X, y = make_moons(n_samples=n,
                          noise=params.get("noise", 0.1))
        return TimedDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long),
                            time)

    else:
        raise ValueError(f"Unknown dataset name: {name}")


# ============================
# === Debug/Test Section ===
# ============================

if __name__ == "__main__":
    import sys
    import os

    print("Debug mode")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Exemple rapide de test :
    from datasets.datasets_registry import GaussianConfig
    config = GaussianConfig(time=1.0, mean=0, std=1, dim=2, n_samples=5)
    dataset = load_dataset(config)
    print(dataset)
    print(dataset[0])  # Affiche un échantillon
    print("time:", dataset.get_time())
