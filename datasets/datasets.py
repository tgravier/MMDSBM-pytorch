import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles
import numpy as np
from sklearn.datasets import make_s_curve
from datasets.datasets_registry import DatasetConfig
import os
import glob
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional
import joblib
from PIL import Image


# datasets/datasets.py

# ============================
# === Timed Dataset Class ===
# ============================


class TimedDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        time: float,
        path: Optional[str] = None,
        type: Optional[str] = None,
    ):
        if not isinstance(time, (float, int)):
            raise TypeError(f"'time' must be a float or int, got {type(time)}")
        self.data = data
        self.time = float(time)
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_time(self) -> float:
        return self.time

    def get_all(self, rescale_data=False):
        if not rescale_data:
            return self.data

        if rescale_data:
            if self.path is None:
                raise ValueError(
                    "Path to scaler.pkl must be provided to inverse rescale data."
                )

            path = os.path.dirname(self.path)
            scaler_path = os.path.join(path, "scaler.pkl")
            print(f"Loaded scaler from: {scaler_path}")

            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

            scaler = joblib.load(scaler_path)
            inversed_data = torch.tensor(
                scaler.inverse_transform(self.data.cpu().numpy())
            )
            return inversed_data

    def get_path(self):
        return self.path

    def get_variance(self):
        return self.data.var(dim=0, unbiased=False)

    def __repr__(self):
        return f"TimedDataset(len={len(self)}, time={self.time})"


# ============================
# === Dataset Loader Func ===
# ============================


def load_dataset(
    config: DatasetConfig, separation_train_test: bool = False, nb_points_test: int = 0
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
            data_train, data_test = random_split(
                torch.tensor(data, dtype=torch.float32), nb_points_test
            )
            return (TimedDataset(data_train, time), TimedDataset(data_test, time))
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
            data_train, data_test = random_split(
                torch.tensor(X, dtype=torch.float32), nb_points_test
            )
            return (TimedDataset(data_train, time), TimedDataset(data_test, time))
        return TimedDataset(torch.tensor(X, dtype=torch.float32), time)

    elif name == "moon":
        X, _ = make_moons(n_samples=n, noise=params.get("noise", 0.1))
        if separation_train_test:
            data_train, data_test = random_split(
                torch.tensor(X, dtype=torch.float32), nb_points_test
            )
            return (TimedDataset(data_train, time), TimedDataset(data_test, time))
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
            data[idx] = np.random.normal(
                loc=means[i], scale=stds[i], size=(np.sum(idx), dim)
            )
        if separation_train_test:
            data_train, data_test = random_split(
                torch.tensor(data, dtype=torch.float32), nb_points_test
            )
            return (TimedDataset(data_train, time), TimedDataset(data_test, time))
        return TimedDataset(torch.tensor(data, dtype=torch.float32), time)

    elif name == "s_curve":
        noise = params.get("noise", 0.0)
        X, _ = make_s_curve(n_samples=n, noise=noise)
        X_2d = X[:, [0, 2]]
        if separation_train_test:
            data_train, data_test = random_split(
                torch.tensor(X_2d, dtype=torch.float32), nb_points_test
            )
            return (TimedDataset(data_train, time), TimedDataset(data_test, time))
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
        pcs = pcs[:, :dim]

        if separation_train_test:
            data_train, data_test = random_split(
                torch.tensor(pcs, dtype=torch.float32), nb_points_test
            )
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(torch.tensor(pcs, dtype=torch.float32), time, path)

    elif name == "biotine_latent_flatten":
        path = params["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )
        data_pt = torch.load(path)
        data_pt = data_pt.view(data_pt.shape[0], -1)

        if separation_train_test:
            data_train, data_test = random_split(data_pt, nb_points_test)
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(data_pt, time, path, type="large_scale")

    elif name == "biotine_latent":
        path = params["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )
        data_pt = torch.load(path)

        if separation_train_test:
            data_train, data_test = random_split(data_pt, nb_points_test)
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(data_pt, time, path, type="large_scale")

    elif name == "mnist":
        # Récupérer le chemin du fichier .pt directement depuis les paramètres
        path = params["file_path"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )

        # Charger le fichier .pt
        data_pt = torch.load(path)

        # Flatten en (num_samples, -1)
        data_pt = data_pt.view(data_pt.shape[0], -1)

        if separation_train_test:
            data_train, data_test = random_split(data_pt, nb_points_test)
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(data_pt, time, path, type="large_scale")

    elif name == "mnist_unet":
        # Récupérer le chemin du fichier .pt directement depuis les paramètres
        path = params["file_path"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )

        # Charger le fichier .pt
        data_pt = torch.load(path)

        if separation_train_test:
            data_train, data_test = random_split(data_pt, nb_points_test)
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(data_pt, time, path, type="large_scale")

    elif name == "multi":
        # MULTI dataset (Kaggle Open Problems Multimodal) - preprocessed via TrajectoryNet pipeline
        path = params["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file '{path}' for time={time} not found."
            )
        data_npz = np.load(path)
        if "pcs" not in data_npz:
            raise ValueError(f"File '{path}' must contain key 'pcs'.")
        pcs = data_npz["pcs"]
        pcs = pcs[:, :dim]

        if separation_train_test:
            data_train, data_test = random_split(
                torch.tensor(pcs, dtype=torch.float32), nb_points_test
            )
            return (
                TimedDataset(data_train, time, path),
                TimedDataset(data_test, time, path),
            )
        return TimedDataset(torch.tensor(pcs, dtype=torch.float32), time, path)

    elif name == "kth":
        frame_dir = params["frame_dir"]
        frame_rank = params["frame_rank"]
        grayscale = params.get("grayscale", True)
        normalize_per_image = params.get("normalize_per_image", False)

        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(f"KTH frame directory not found: {frame_dir}")

        # Frames are named: {video}_seq{i}_frame{orig:04d}.png
        # We need to pick the frame_rank-th file per subsequence when sorted.
        # Group by subsequence, then pick the frame_rank-th entry.
        all_pngs = sorted(glob.glob(os.path.join(frame_dir, "*_seq*_frame*.png")))
        if not all_pngs:
            raise FileNotFoundError(f"No PNG files found in {frame_dir}")

        # Group by subsequence prefix (everything before _frame)
        from collections import defaultdict

        subseq_frames: dict = defaultdict(list)
        for p in all_pngs:
            base = os.path.basename(p)
            # Split at last "_frame" to get subsequence key
            prefix = base.rsplit("_frame", 1)[0]
            subseq_frames[prefix].append(p)

        # For each subsequence, pick the frame at position frame_rank
        selected_paths = []
        for prefix in sorted(subseq_frames):
            frames_sorted = sorted(subseq_frames[prefix])
            if frame_rank >= len(frames_sorted):
                raise IndexError(
                    f"KTH subsequence {prefix!r} has {len(frames_sorted)} frames, "
                    f"but frame_rank={frame_rank} requested"
                )
            selected_paths.append(frames_sorted[frame_rank])
        print(
            f"Selected {len(selected_paths)} frames for KTH dataset at time={time}; exerpt: {[p.split('/')[-1] for p in selected_paths[:5]]}"
        )

        # Load images into tensor, normalised to [-1, 1]
        tensors = []
        for p in selected_paths:
            img = Image.open(p)
            if grayscale:
                img = img.convert("L")
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # (H, W) in [-1, 1]
                tensors.append(torch.tensor(arr).unsqueeze(0))  # (1, H, W)
            else:
                arr = (
                    np.array(img, dtype=np.float32) / 127.5 - 1.0
                )  # (H, W, 3) in [-1, 1]
                tensors.append(torch.tensor(arr).permute(2, 0, 1))  # (3, H, W)

        data_pt = torch.stack(tensors)  # (N, C, H, W)

        if normalize_per_image:
            # Per-image min-max normalization to [-1, 1]
            for i in range(data_pt.shape[0]):
                img_min = data_pt[i].min()
                img_max = data_pt[i].max()
                if img_max - img_min > 0:
                    data_pt[i] = (data_pt[i] - img_min) / (img_max - img_min) * 2.0 - 1.0

        if dim is not None:
            # Flatten if dim is set (for MLP-based models)
            data_pt = data_pt.view(data_pt.shape[0], -1)
            if data_pt.shape[1] != dim:
                raise ValueError(
                    f"KTH flattened dim {data_pt.shape[1]} != expected dim {dim}"
                )

        if separation_train_test:
            data_train, data_test = random_split(data_pt, nb_points_test)
            return (
                TimedDataset(data_train, time, frame_dir),
                TimedDataset(data_test, time, frame_dir),
            )
        return TimedDataset(data_pt, time, frame_dir, type="large_scale")

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
