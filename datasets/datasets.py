import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles
import numpy as np
from sklearn.datasets import make_s_curve
from datasets.datasets_registry import DatasetConfig
import os
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

# datasets/datasets.py

# ============================
# === Timed Dataset Class ===
# ============================


class TimedDataset(Dataset):
    def __init__(self, data: torch.Tensor, time: float):
        if not isinstance(time, (float, int)):
            raise TypeError(f"'time' must be a float or int, got {type(time)}")
        self.data = data
        self.time = float(time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_time(self) -> float:
        return self.time

    def get_all(self):
        return self.data
    
    def get_variance(self):

        return self.data.var(dim=0, unbiased=False)

    def __repr__(self):
        return f"TimedDataset(len={len(self)}, time={self.time})"


# ============================
# === Dataset Loader Func ===
# ============================


def load_dataset(
    config: DatasetConfig,
    separation_train_test: bool = False,
    nb_points_test: int = 0
) -> Union[TimedDataset, Tuple[TimedDataset, TimedDataset]]:
    name = config.name
    params = config.params
    dim = config.input_dim
    n = params.get("n_samples", 1000)
    time = config.time

    if name == "gaussian":
        mean = np.array(params.get("mean", [0.0] * dim))
        std = np.array(params.get("std", [1.0] * dim))

        if mean.shape[0] != dim:
            raise ValueError(
                f"Mean vector has incorrect dimension: expected {dim}, got {mean.shape[0]}"
            )
        if std.shape[0] != dim:
            raise ValueError(
                f"Std vector has incorrect dimension: expected {dim}, got {std.shape[0]}"
            )

        data = np.random.normal(loc=mean, scale=std, size=(n, dim))
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(data, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(data, dtype=torch.float32), time)

    elif name == "circle":
        center = np.array(params.get("center", [0.0, 0.0]))
        radius = params.get("radius", 1.0)
        thickness = params.get("thickness", 0.05)
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.normal(radius, thickness, n)
        X = np.stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
            axis=1,
        )
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(X, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(X, dtype=torch.float32), time)

    elif name == "moon":
        X, _ = make_moons(n_samples=n, noise=params.get("noise", 0.1))
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(X, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(X, dtype=torch.float32), time)

    elif name == "gaussian_mixture":
        means = np.array(params["means"])  # shape (K, D)
        stds = np.array(params["stds"])  # shape (K, D)
        weights = np.array(params["weights"])  # shape (K,)
        k = len(weights)

        if means.shape != stds.shape or means.shape[1] != dim:
            raise ValueError(
                f"Mismatch in shapes: means {means.shape}, stds {stds.shape}, expected dim {dim}"
            )

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.")

        component_indices = np.random.choice(k, size=n, p=weights)
        data = np.zeros((n, dim), dtype=np.float32)
        for i in range(k):
            idx = component_indices == i
            data[idx] = np.random.normal(loc=means[i], scale=stds[i], size=(np.sum(idx), dim))
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(data, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(data, dtype=torch.float32), time)

    elif name == "s_curve":
        noise = params.get("noise", 0.0)
        X, _ = make_s_curve(n_samples=n, noise=noise)
        X_2d = X[:, [0, 2]]
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(X_2d, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(X_2d, dtype=torch.float32), time)

    elif name == "phate_traj_dim2":
        path = params["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )
        data_npz = np.load(path)
        if "pcs" not in data_npz:
            raise ValueError(f"File '{path}' must contain key 'pcs'.")
        pcs = data_npz["pcs"]
        pcs = pcs[:,:dim]
        if separation_train_test:
            data_train, data_test = random_split(torch.tensor(pcs, dtype=torch.float32), nb_points_test)
            return (
                TimedDataset(data_train, time),
                TimedDataset(data_test, time)
            )
        return TimedDataset(torch.tensor(pcs, dtype=torch.float32), time)

    else:
        raise ValueError(f"Unknown dataset name: {name}")


def random_split(data_tensor, nb_points_test):
    # Function to train test split , in a random way
    if nb_points_test <= 0 or nb_points_test >= len(data_tensor):
        print(nb_points_test)
        print(len(data_tensor))
        raise ValueError("Invalid nb_points_test value.")

    indices = np.random.permutation(len(data_tensor))
    test_indices = indices[:nb_points_test]
    train_indices = indices[nb_points_test:]

    data_train = data_tensor[train_indices]
    data_test = data_tensor[test_indices]

    return (data_train, data_test)


