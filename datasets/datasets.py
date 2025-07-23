import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles
import numpy as np
from sklearn.datasets import make_s_curve
from datasets.datasets_registry import DatasetConfig 
import os
from sklearn.preprocessing import StandardScaler

# datasets/datasets.py

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
            raise ValueError(
                f"Mean vector has incorrect dimension: expected {dim}, got {mean.shape[0]}"
            )
        if std.shape[0] != dim:
            raise ValueError(
                f"Std vector has incorrect dimension: expected {dim}, got {std.shape[0]}"
            )

        data = np.random.normal(loc=mean, scale=std, size=(n, dim))
        labels = np.zeros(n)
        return TimedDataset(
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            time,
        )

    elif name == "circle":
        center = np.array(params.get("center", [0.0, 0.0]))
        radius = params.get("radius", 1.0)
        thickness = params.get("thickness", 0.05)
        n = params.get("n_samples", 1000)
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.normal(radius, thickness, n)
        X = np.stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
            axis=1,
        )
        y = np.zeros(n)
        return TimedDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            time,
        )

    elif name == "moon":
        X, y = make_moons(n_samples=n, noise=params.get("noise", 0.1))
        return TimedDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            time,
        )
    

    elif name == "gaussian_mixture":
        means = np.array(params["means"])         # shape (K, D)
        stds = np.array(params["stds"])           # shape (K, D)
        weights = np.array(params["weights"])     # shape (K,)
        n = params.get("n_samples", 1000)
        time = config.time
        k = len(weights)
        dim = config.input_dim

        if means.shape != stds.shape or means.shape[1] != dim:
            raise ValueError(f"Mismatch in shapes: means {means.shape}, stds {stds.shape}, expected dim {dim}")

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.")


        component_indices = np.random.choice(k, size=n, p=weights)


        data = np.zeros((n, dim), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(k):
            idx = component_indices == i
            n_i = np.sum(idx)
            if n_i > 0:
                data[idx] = np.random.normal(loc=means[i], scale=stds[i], size=(n_i, dim))
                labels[idx] = i

        return TimedDataset(
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            time,
        )
    
    elif name == "s_curve":
        

        noise = params.get("noise", 0.0)
        X, _ = make_s_curve(n_samples=n, noise=noise)

        # On garde uniquement les colonnes x et z pour rester en 2D
        X_2d = X[:, [0, 2]]
        y = np.zeros(n)

        return TimedDataset(
            torch.tensor(X_2d, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            time,
        )
    
    elif name == "phate_traj_dim2":
        

        ## Standart scaler precendently for this dataset, I save the data in scaler.pkl

        path = params["file_path"]
        dim = dim




        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file '{path}' for time={time} not found.")

        # Load the PHATE embedding
        data_npz = np.load(path)
        if "pcs" not in data_npz:
            raise ValueError(f"File '{path}' must contain key 'pcs'.")

        pcs = data_npz["pcs"][:, dim]
        n_samples = pcs.shape[0]
        dummy_labels = np.zeros(n_samples, dtype=np.int64)  # Unused, but required by TimedDataset

        return TimedDataset(
            torch.tensor(pcs, dtype=torch.float32),
            torch.tensor(dummy_labels, dtype=torch.long),
            time
        )


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
