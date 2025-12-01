# utils/metrics
from typing import List, Tuple
import torch
from torch import Tensor
import ot  # POT: Python Optimal Transport
import numpy as np
from utils.kernels import MMDLoss, RBF


# ---------- Classic Statistics ----------


def get_classic_metrics(data: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns the mean and standard deviation of a dataset along the batch dimension.

    Args:
        data (Tensor): Input tensor of shape (B, D)

    Returns:
        Tuple[Tensor, Tensor]: (mean, std) both of shape (D,)
    """
    return data.mean(dim=0), data.std(dim=0, unbiased=False)


# ---------- Sliced Wasserstein Distance using POT ----------


def compute_swd_pot(x: Tensor, y: Tensor, n_proj: int = 50) -> float:
    """
    Computes the sliced Wasserstein distance between two empirical distributions using POT.

    Args:
        x (Tensor): Tensor of shape (N, D)
        y (Tensor): Tensor of shape (N, D)
        n_proj (int): Number of projections to use (default: 100)

    Returns:
        float: Sliced Wasserstein distance (W2)
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    return ot.sliced.sliced_wasserstein_distance(x_np, y_np, n_projections=n_proj)


# ---------- Evaluate Trajectory vs. Reference Datasets ----------


def evaluate_swd_over_time(
    generated: List[Tensor],
    time: List[float],
    datasets_inference: List,
    direction_tosample: str,
    n_proj: int = 50,
    rescale_data: bool = False,
) -> List[Tuple[float, float]]:
    """
    Evaluates Sliced Wasserstein Distance (SWD) between generated samples and reference datasets.

    Args:
        generated (List[Tensor]): Generated samples at different times (list of (N, D) tensors)
        time (List[float]): List of times associated with generated samples
        datasets_train (List[TimedDataset]): List of reference datasets
        direction_tosample (str): "forward" or "backward"
        n_proj (int): Number of projections for SWD

    Returns:
        List[Tuple[float, float]]: List of (reference_time, SWD)
    """
    assert direction_tosample in ["forward", "backward"], (
        "Direction must be 'forward' or 'backward'"
    )

    # Sort datasets by time depending on direction
    sorted_datasets = sorted(
        datasets_inference,
        key=lambda d: d.get_time(),
        reverse=(direction_tosample == "backward"),
    )

    time_tensor = torch.tensor(time)
    swd_scores = []

    for dataset in sorted_datasets:
        ref_time = dataset.get_time()
        ref_data = dataset.get_all(rescale_data=rescale_data).to(generated[0].device)

        # Find closest generated time
        closest_idx = int(torch.argmin(torch.abs(time_tensor - ref_time)).item())
        gen_samples = generated[closest_idx]

        # Match sample sizes
        num_samples = min(len(ref_data), len(gen_samples))
        ref_samples = ref_data[torch.randint(0, len(ref_data), (num_samples,))]
        gen_samples = gen_samples[:num_samples]

        # Compute SWD using POT
        swd = compute_swd_pot(gen_samples, ref_samples, n_proj=n_proj)
        swd_scores.append((ref_time, swd))

    return swd_scores


def evaluate_energy_over_time(
    generated: List[Tensor],
    time: List[float],
) -> List[Tuple[float, float]]:
    """
    Computes the path energy over time, defined as the squared L2 displacement between steps.

    Args:
        generated (List[Tensor]): List of tensors (N, D) for each time step.
        time (List[float]): Associated times (must be sorted in order).

    Returns:
        List[Tuple[float, float]]: List of (t_i, energy_i), energy at step i (between t_i and t_{i+1})
    """
    energy_scores = []
    for i in range(len(generated) - 1):
        x1 = generated[i]
        x2 = generated[i + 1]
        dt = time[i + 1] - time[i]

        # Ensure equal number of samples
        n = min(len(x1), len(x2))
        x1 = x1[:n]
        x2 = x2[:n]

        # Energy per step = average L2 norm squared / delta_t
        displacement = x2 - x1
        energy = (displacement.norm(p=2, dim=1) ** 2).mean().item() / (
            dt if dt != 0 else 1.0
        )

        energy_scores.append((time[i], energy))

    return energy_scores


def evaluate_wd_over_time(
    generated: List[Tensor],
    time: List[float],
    datasets_inference: List,
    direction_tosample: str,
) -> List[Tuple[float, float]]:
    """
    Evaluates true Wasserstein-2 Distance (WD) between generated samples and reference datasets.

    Args:
        generated (List[Tensor]): Generated samples at different times (list of (N, D) tensors)
        time (List[float]): List of times associated with generated samples
        datasets_train (List[TimedDataset]): List of reference datasets
        direction_tosample (str): "forward" or "backward"

    Returns:
        List[Tuple[float, float]]: List of (reference_time, WD)
    """
    assert direction_tosample in ["forward", "backward"]

    sorted_datasets = sorted(
        datasets_inference,
        key=lambda d: d.get_time(),
        reverse=(direction_tosample == "backward"),
    )

    time_tensor = torch.tensor(time)
    wd_scores = []

    for dataset in sorted_datasets:
        ref_time = dataset.get_time()
        ref_data = dataset.get_all().to(generated[0].device)

        # Find closest generated time
        closest_idx = int(torch.argmin(torch.abs(time_tensor - ref_time)).item())
        gen_samples = generated[closest_idx]

        # Match sample sizes
        n = min(len(ref_data), len(gen_samples))
        ref_samples = ref_data[torch.randint(0, len(ref_data), (n,))]
        gen_samples = gen_samples[:n]

        # Convert to NumPy
        x = gen_samples.detach().cpu().numpy()
        y = ref_samples.detach().cpu().numpy()

        # Uniform weights
        a = b = ot.unif(n)

        # Cost matrix (squared Euclidean distance)
        M = ot.dist(x, y, metric="euclidean") ** 2

        # Solve OT problem (returns the WD²)
        wd2 = ot.emd2(a, b, M)

        wd_scores.append((ref_time, wd2**0.5))  # Return sqrt for actual WD

    return wd_scores


def evaluate_w1_over_time(
    generated: List[Tensor],
    time: List[float],
    datasets_inference: List,
    direction_tosample: str,
    rescale_data: bool = False,
) -> List[Tuple[float, float]]:
    """
    Evaluates Wasserstein-1 Distance (W1) between generated samples and reference datasets.
    Uses unsquared Euclidean distance (L1 ground metric in the transport sense).

    Args:
        generated (List[Tensor]): Generated samples at different times (list of (N, D) tensors)
        time (List[float]): List of times associated with generated samples
        datasets_inference (List[TimedDataset]): List of reference datasets
        direction_tosample (str): "forward" or "backward"
        rescale_data (bool): Whether to rescale data using saved scaler

    Returns:
        List[Tuple[float, float]]: List of (reference_time, W1)
    """
    assert direction_tosample in ["forward", "backward"]

    sorted_datasets = sorted(
        datasets_inference,
        key=lambda d: d.get_time(),
        reverse=(direction_tosample == "backward"),
    )

    time_tensor = torch.tensor(time)
    w1_scores = []

    for dataset in sorted_datasets:
        ref_time = dataset.get_time()
        ref_data = dataset.get_all(rescale_data=rescale_data).to(generated[0].device)

        # Find closest generated time
        closest_idx = int(torch.argmin(torch.abs(time_tensor - ref_time)).item())
        gen_samples = generated[closest_idx]

        # Match sample sizes
        n = min(len(ref_data), len(gen_samples))
        ref_samples = ref_data[torch.randint(0, len(ref_data), (n,))]
        gen_samples = gen_samples[:n]

        # Convert to NumPy
        x = gen_samples.detach().cpu().numpy()
        y = ref_samples.detach().cpu().numpy()

        # Uniform weights
        a = b = ot.unif(n)

        # Cost matrix: unsquared Euclidean distance for W1
        M = ot.dist(x, y, metric="euclidean")  # No squaring!

        # Solve OT problem (returns W1 directly)
        w1 = ot.emd2(a, b, M)

        w1_scores.append((ref_time, w1))

    return w1_scores


def evaluate_w1_consecutive(
    generated: List[Tensor],
    time: List[float],
    datasets_inference: List,
    direction_tosample: str,
    rescale_data: bool = False,
) -> List[Tuple[float, float]]:
    """
    Evaluates W1 between generated samples at time t and reference data at t-1 (for forward)
    or t+1 (for backward).

    For forward: compare p_gen(t) with p_test(t-1)
    For backward: compare p_gen(t) with p_test(t+1)

    Args:
        generated (List[Tensor]): Generated samples at different times
        time (List[float]): List of times associated with generated samples
        datasets_inference (List[TimedDataset]): List of reference datasets
        direction_tosample (str): "forward" or "backward"
        rescale_data (bool): Whether to rescale data using saved scaler

    Returns:
        List[Tuple[float, float]]: List of (generated_time, W1) excluding starting point
    """
    assert direction_tosample in ["forward", "backward"]

    # Sort datasets by time (ascending for forward, descending for backward)
    sorted_datasets = sorted(
        datasets_inference,
        key=lambda d: d.get_time(),
        reverse=(direction_tosample == "backward"),
    )

    time_tensor = torch.tensor(time)
    w1_scores = []

    # Skip first dataset (starting point) - we compare gen[t] with test[t-1]
    for i in range(1, len(sorted_datasets)):
        current_dataset = sorted_datasets[i]
        prev_dataset = sorted_datasets[i - 1]

        gen_time = current_dataset.get_time()
        prev_time = prev_dataset.get_time()

        # Get generated samples at current time
        closest_idx = int(torch.argmin(torch.abs(time_tensor - gen_time)).item())
        gen_samples = generated[closest_idx]

        # Get reference data from PREVIOUS time point
        prev_data = prev_dataset.get_all(rescale_data=rescale_data).to(
            generated[0].device
        )

        # Match sample sizes
        n = min(len(prev_data), len(gen_samples))
        prev_samples = prev_data[torch.randint(0, len(prev_data), (n,))]
        gen_samples = gen_samples[:n]

        # Convert to NumPy
        x = gen_samples.detach().cpu().numpy()
        y = prev_samples.detach().cpu().numpy()

        # Uniform weights
        a = b = ot.unif(n)

        # Cost matrix: unsquared Euclidean distance for W1
        M = ot.dist(x, y, metric="euclidean")

        # Solve OT problem
        w1 = ot.emd2(a, b, M)

        w1_scores.append((gen_time, w1))

    return w1_scores


def evaluate_mmd_over_time(
    generated: List[Tensor],
    time: List[float],
    datasets_inference: List,
    direction_tosample: str,
    kernel_type: str = "gaussian",
    blur: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Computes MMD over time using either GeomLoss or custom RBF kernel.

    Args:
        ...
        kernel_type (str): "gaussian", "energy", "laplacian", or "rbf"
    """
    assert direction_tosample in ["forward", "backward"]

    sorted_datasets = sorted(
        datasets_inference,
        key=lambda d: d.get_time(),
        reverse=(direction_tosample == "backward"),
    )

    time_tensor = torch.tensor(time, device=generated[0].device)
    mmd_scores = []

    if kernel_type == "rbf":
        mmd_fn = MMDLoss(kernel=RBF())
    else:
        # GeomLoss path
        kernel_map = {
            "gaussian": "kernel",
            "energy": "energy",
            "laplacian": "laplacian",
        }

        if kernel_type not in kernel_map:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        from geomloss import SamplesLoss

        mmd_fn = SamplesLoss(loss=kernel_map[kernel_type], p=2, blur=blur)

    for dataset in sorted_datasets:
        ref_time = dataset.get_time()
        ref_data = dataset.get_all().to(generated[0].device)

        closest_idx = int(torch.argmin(torch.abs(time_tensor - ref_time)).item())
        gen_samples = generated[closest_idx]

        n = min(len(ref_data), len(gen_samples))
        ref_samples = ref_data[torch.randint(0, len(ref_data), (n,))]
        gen_samples = gen_samples[:n]

        # Evaluate
        if kernel_type == "rbf":
            mmd2 = mmd_fn(gen_samples, ref_samples).item()
        else:
            mmd2 = mmd_fn(gen_samples, ref_samples).item()

        mmd_scores.append((ref_time, mmd2))

    return mmd_scores
