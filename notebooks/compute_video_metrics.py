#!/usr/bin/env python
"""
Compute FID/KID metrics for generated video frames using torch-fidelity.

Converts .pt files containing generated frames to PNG images and computes
FID/KID scores against ground truth images for each time step.

Requires torch-fidelity >= 0.4.0 (install from source for DINOv2 support), eg:
    pip install git+https://github.com/toshas/torch-fidelity.git
"""

from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch_fidelity


# ============================================================================
# Configuration - EDIT THESE VALUES
# ============================================================================

# Path to inference results containing .pt files
BASE_PATH = Path(
    "/projects/static2dynamic/Gravier/schron/experiments_debug/biotine_7d_unet_image128_02/inference_28-11_14-04"
)

# Output directory for images (set to None to use BASE_PATH / "images")
OUTPUT_DIR = Path(
    "/projects/static2dynamic/Thomas/experiments/schron/biotine_7d_unet_image128_02/inference_28-11_14-04"
)

# Path to real/ground truth images
REAL_IMAGES_BASE_PATH = Path(
    "/projects/static2dynamic/datasets/biotine/3_channels_min_99_perc_normalized_rgb_stacks_png/patches_255"
)

# Device and batch size
DEVICE = "cuda:0"
BATCH_SIZE = 1024

# Feature extractor options:
#   - "inception-v3-compat" (default InceptionV3, compatible with TF)
#   - "dinov2-vit-s-14", "dinov2-vit-b-14", "dinov2-vit-l-14", "dinov2-vit-g-14"
#   - "clip-vit-b-32", "clip-vit-b-16", "clip-vit-l-14"
FEATURE_EXTRACTOR = "dinov2-vit-b-14"

# Metrics to compute
COMPUTE_FID = True
COMPUTE_KID = True

# Skip conversion if images already exist
SKIP_CONVERSION = True

# Whether to compute and save pixel value histograms
COMPUTE_HIST = True

# Whether to compute metrics (FID/KID) and save CSVs + plots
COMPUTE_METRICS = False

# Whether to regenerate metric plots from existing CSVs (useful for tweaking plot styling)
# Only used if COMPUTE_METRICS=False
REPLOT_METRICS = False

# Comparison mode: "gen_vs_true" or "true_vs_true"
# - "gen_vs_true": compute matrix of generated times (time1..time6) vs real folders (4,7,10,13,16,19)
# - "true_vs_true": compute matrix of real folders vs real folders (4,7,10,13,16,19)
# - "both": compute both matrices
MODE = "gen_vs_true"

# Mapping between generated times and real times
# Original selection: times = list(range(0, 19, 3)) = [0, 3, 6, 9, 12, 15, 18] (0-indexed)
# so with the 1-index of true data at /projects/static2dynamic/datasets/biotine/3_channels_min_99_perc_normalized_rgb_stacks_png/patches_255
# that means: [1, 4, 7, 10, 13, 16, 19] (1-indexed)
# First frame (index 0) is the input/conditioning frame, so 6 generated times are:
GEN_TO_REAL_FOLDER = {
    1: "4",  # gen time 1 -> real folder 4
    2: "7",  # gen time 2 -> real folder 7
    3: "10",  # gen time 3 -> real folder 10
    4: "13",  # gen time 4 -> real folder 13
    5: "16",  # gen time 5 -> real folder 16
    6: "19",  # gen time 6 -> real folder 19
}


# ============================================================================
# Helper functions
# ============================================================================


def denormalize(tensor):
    """
    Denormalize from [-1, 1] back to [0, 1] range.
    Original transform: Normalize(mean=[0.5]*3, std=[0.5]*3)
    This means: normalized = (x - 0.5) / 0.5 = 2*x - 1
    So to reverse: x = (normalized + 1) / 2
    """
    return (tensor + 1) / 2


def compute_image_histogram(
    dir_path: Path, name: str, sample_size: int = 1000
) -> np.ndarray:
    """
    Compute histogram of pixel values from PNG images in a directory.
    Returns histogram array of shape (256,) counting occurrences of each value 0-255.
    """
    images = list(dir_path.glob("*.png"))
    if not images:
        raise RuntimeError(f"{name}: No PNG images found in {dir_path}")

    # Sample images if there are too many
    if len(images) > sample_size:
        rng = np.random.default_rng(42)
        images = [
            images[i] for i in rng.choice(len(images), sample_size, replace=False)
        ]

    histogram = np.zeros(256, dtype=np.int64)

    for img_path in tqdm(images, desc=f"Computing histogram for {name}"):
        img = Image.open(img_path)

        # Check it's RGB
        if img.mode != "RGB":
            raise RuntimeError(
                f"{name}: Image {img_path.name} is mode '{img.mode}', expected 'RGB'"
            )

        # Get pixel values as uint8 and accumulate histogram
        arr = np.array(img, dtype=np.uint8).flatten()
        histogram += np.bincount(arr, minlength=256)

    return histogram


def plot_all_histograms(
    images_dir: Path,
    real_images_path: Path,
    output_dir: Path,
    mode: str,
    gen_to_real_folder: dict[int, str],
    sample_size: int = 1000,
):
    """
    Compute and plot pixel value histograms for all time steps.

    Creates a figure with rows = time steps, cols = [generated, real].
    Each subplot shows the pixel value distribution (0-255) for
    a sample of images.

    Args:
        images_dir: Directory containing generated images (time1/, time2/, etc.)
        real_images_path: Directory containing real images (4/, 7/, etc.)
        output_dir: Directory to save the histogram plot
        mode: "gen_vs_true", "true_vs_true", or "both"
        gen_to_real_folder: Mapping from gen time (1-6) to real folder name
        sample_size: Number of images to sample per directory
    """
    # Collect all histograms: key = (gen_time, "gen"/"real") -> histogram
    all_histograms = {}

    for gen_time, real_folder in gen_to_real_folder.items():
        if mode in ("gen_vs_true", "both"):
            gen_dir = images_dir / f"time{gen_time}"
            all_histograms[(gen_time, "gen")] = compute_image_histogram(
                gen_dir, f"Gen time{gen_time}", sample_size=sample_size
            )

        real_dir = real_images_path / real_folder
        all_histograms[(gen_time, "real")] = compute_image_histogram(
            real_dir, f"Real {real_folder}", sample_size=sample_size
        )

    # Create figure: rows = time steps, cols = [gen, real]
    n_times = len(gen_to_real_folder)
    has_gen = mode in ("gen_vs_true", "both")
    n_cols = 2 if has_gen else 1

    fig, axes = plt.subplots(n_times, n_cols, figsize=(6 * n_cols, 4 * n_times))
    if n_times == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (gen_time, real_folder) in enumerate(gen_to_real_folder.items()):
        col_idx = 0

        # Plot generated histogram
        if has_gen and (gen_time, "gen") in all_histograms:
            ax = axes[row_idx, col_idx]
            hist = all_histograms[(gen_time, "gen")]
            _plot_single_histogram(ax, hist, f"Generated time{gen_time}", color="C0")
            col_idx += 1

        # Plot real histogram
        real_col = col_idx if has_gen else 0
        if (gen_time, "real") in all_histograms:
            ax = axes[row_idx, real_col]
            hist = all_histograms[(gen_time, "real")]
            _plot_single_histogram(ax, hist, f"Real folder {real_folder}", color="C1")

    # Dynamic title based on mode
    if has_gen:
        title = "Pixel Value Distributions: Generated vs Real (1k samples each)"
    else:
        title = "Pixel Value Distributions: Real Images (1k samples each)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    histogram_plot_path = output_dir / "pixel_histograms_all_times.png"
    plt.savefig(histogram_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram plot saved to: {histogram_plot_path}")


def _plot_single_histogram(ax, hist: np.ndarray, title: str, color: str = "C0"):
    """Plot a single histogram on the given axes."""
    total = hist.sum()
    freq = hist / total if total > 0 else hist

    ax.bar(range(256), freq, width=1.0, alpha=0.7, color=color)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} (n={total:,} px)")
    ax.set_xlim(-5, 260)

    # Stats
    values = np.arange(256)
    mean = np.sum(values * hist) / total if total > 0 else 0
    nonzero = np.nonzero(hist)[0]
    val_min, val_max = (nonzero[0], nonzero[-1]) if len(nonzero) > 0 else (0, 0)
    ax.text(
        0.95,
        0.95,
        f"[{val_min}, {val_max}]\nμ={mean:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def plot_gen_vs_true_barplot(
    fid_values: list[float],
    kid_means: list[float] | None,
    kid_stds: list[float] | None,
    time_labels: list[str],
    real_labels: list[str],
    output_dir: Path,
    fe_short: str,
):
    """
    Plot bar charts for gen_vs_true mode (diagonal comparison only).

    Args:
        fid_values: FID values for each time step (diagonal of matrix)
        kid_means: KID mean values for each time step (or None)
        kid_stds: KID std values for each time step (or None)
        time_labels: Labels like ["time1", "time2", ...]
        real_labels: Labels like ["4", "7", ...]
        output_dir: Directory to save plots
        fe_short: Feature extractor short name for filename
    """
    n_times = len(time_labels)
    x = np.arange(n_times)

    # Create x-axis labels combining gen and real
    x_labels = [f"{t}\n(vs {r})" for t, r in zip(time_labels, real_labels)]

    # Determine number of subplots
    has_kid = kid_means is not None and len(kid_means) > 0
    n_plots = 2 if has_kid else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # FID bar plot
    ax = axes[0]
    bars = ax.bar(x, fid_values, color="C0", alpha=0.8)
    ax.set_xlabel("Generated Time (vs Real Folder)")
    ax.set_ylabel("FID Score")
    ax.set_title("FID: Generated vs Real (matched times)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Add value labels on bars
    for bar, val in zip(bars, fid_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # KID bar plot with error bars
    if has_kid:
        ax = axes[1]
        bars = ax.bar(x, kid_means, yerr=kid_stds, color="C1", alpha=0.8, capsize=4)
        ax.set_xlabel("Generated Time (vs Real Folder)")
        ax.set_ylabel("KID Score")
        ax.set_title("KID: Generated vs Real (matched times)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Add value labels
        for bar, val in zip(bars, kid_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = output_dir / f"{fe_short}_gen_vs_true_barplot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gen_vs_true bar plot to: {plot_path}")


def plot_true_vs_true_heatmaps(
    fid_df: pd.DataFrame,
    kid_mean_df: pd.DataFrame | None,
    output_dir: Path,
    fe_short: str,
):
    """
    Plot heatmaps for true_vs_true mode.

    Args:
        fid_df: DataFrame with FID matrix (rows and cols are real folder labels)
        kid_mean_df: DataFrame with KID mean matrix (or None)
        output_dir: Directory to save plots
        fe_short: Feature extractor short name for filename
    """
    has_kid = kid_mean_df is not None
    n_plots = 2 if has_kid else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    # FID heatmap
    ax = axes[0]
    sns.heatmap(
        fid_df.astype(float),
        annot=True,
        fmt=".1f",
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "FID"},
        square=True,
    )
    ax.set_title("FID: Real vs Real")
    ax.set_xlabel("Real Folder")
    ax.set_ylabel("Real Folder")

    # KID heatmap
    if has_kid:
        ax = axes[1]
        sns.heatmap(
            kid_mean_df.astype(float),
            annot=True,
            fmt=".4f",
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": "KID Mean"},
            square=True,
        )
        ax.set_title("KID Mean: Real vs Real")
        ax.set_xlabel("Real Folder")
        ax.set_ylabel("Real Folder")

    plt.tight_layout()
    plot_path = output_dir / f"{fe_short}_true_vs_true_heatmaps.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true_vs_true heatmaps to: {plot_path}")


# ============================================================================
# Main functions
# ============================================================================


def convert_pt_to_images(base_path: Path, images_dir: Path):
    """Convert .pt files to PNG images organized by time step."""
    all_files = list(base_path.rglob("*.pt"))
    print(f"Found {len(all_files)} .pt files.")

    # Filter files with exactly 6 times (exclude videos with different frame counts)
    valid_files = []
    for file in tqdm(all_files, desc="Filtering files"):
        d = torch.load(file, mmap=True)
        if len(d["time"]) == 6:
            valid_files.append(file)
        else:
            print(f"Skipping {file.name}: has {len(d['time'])} frames instead of 6")

    print(f"\nFound {len(valid_files)} valid files with 6 time steps.")

    # Create output directory structure
    images_dir.mkdir(exist_ok=True, parents=True)
    for i in range(1, 7):
        time_dir = images_dir / f"time{i}"
        time_dir.mkdir(exist_ok=True)

    # Save images for each file and time step
    for file_idx, file in enumerate(tqdm(valid_files, desc="Processing files")):
        d = torch.load(file)
        generated = d["generated"]
        times = d["time"]
        batch_idx = d["batch_index"]

        for time_idx in range(len(times)):
            time_dir = images_dir / f"time{time_idx + 1}"
            img_data = generated[time_idx]

            # Denormalize and clamp
            img_data = denormalize(img_data)
            img_data = torch.clamp(img_data, 0, 1)

            if img_data.dim() == 4:  # [B, C, H, W]
                for img_idx in range(img_data.shape[0]):
                    img_name = (
                        f"file{file_idx:04d}_batch{batch_idx}_img{img_idx:04d}.png"
                    )
                    save_image(img_data[img_idx], time_dir / img_name)
            elif img_data.dim() == 3:  # [C, H, W] single image
                img_name = f"file{file_idx:04d}_batch{batch_idx}.png"
                save_image(img_data, time_dir / img_name)
            else:
                print(f"Unexpected tensor shape: {img_data.shape}")

    print(f"\nImages saved to {images_dir}")
    return valid_files


def compute_metrics_for_dirs(
    gen_dir: Path,
    real_dir: Path,
    feature_extractor: str,
    compute_fid: bool = True,
    compute_kid: bool = True,
    batch_size: int = 64,
    cuda: bool = True,
) -> dict:
    """Compute FID and/or KID between two directories using torch-fidelity."""
    metrics = torch_fidelity.calculate_metrics(
        input1=str(gen_dir),
        input2=str(real_dir),
        cuda=cuda,
        batch_size=batch_size,
        fid=compute_fid,
        kid=compute_kid,
        feature_extractor=feature_extractor,
        verbose=False,
    )
    return metrics


def compute_all_metrics(
    images_dir: Path,
    real_images_base_path: Path,
    output_dir: Path,
    feature_extractor: str,
    compute_fid: bool = True,
    compute_kid: bool = True,
    batch_size: int = 64,
    mode: str = "gen_vs_true",
) -> dict:
    """Compute metrics for each time step and optionally pairwise matrices.

    mode: one of 'gen_vs_true' or 'true_vs_true' (or 'both')
    """
    results = {}
    print(f"Feature extractor: {feature_extractor}")
    print(f"Compute FID: {compute_fid}")
    print(f"Compute KID: {compute_kid}")
    print(f"Batch size: {batch_size}")
    print(f"Mode: {mode}")

    # Per-time metrics (gen time i vs mapped real folder) - only for gen_vs_true or both
    if mode in ("gen_vs_true", "both"):
        for gen_time_idx, real_folder in GEN_TO_REAL_FOLDER.items():
            gen_dir = images_dir / f"time{gen_time_idx}"
            real_dir = real_images_base_path / real_folder

            if not real_dir.exists():
                print(
                    f"WARNING: Real images directory {real_dir} does not exist, skipping time{gen_time_idx}"
                )
                continue

            if not gen_dir.exists():
                print(
                    f"WARNING: Generated images directory {gen_dir} does not exist, skipping time{gen_time_idx}"
                )
                continue

            print(
                f"\nComputing metrics for gen_time{gen_time_idx} vs real_folder {real_folder}..."
            )

            metrics = compute_metrics_for_dirs(
                gen_dir=gen_dir,
                real_dir=real_dir,
                feature_extractor=feature_extractor,
                compute_fid=compute_fid,
                compute_kid=compute_kid,
                batch_size=batch_size,
            )

            time_key = f"time{gen_time_idx}"
            results[time_key] = {"real_folder": real_folder}

            if compute_fid and "frechet_inception_distance" in metrics:
                results[time_key]["fid"] = metrics["frechet_inception_distance"]
                print(f"  FID: {metrics['frechet_inception_distance']:.4f}")

            if compute_kid and "kernel_inception_distance_mean" in metrics:
                results[time_key]["kid_mean"] = metrics[
                    "kernel_inception_distance_mean"
                ]
                results[time_key]["kid_std"] = metrics["kernel_inception_distance_std"]
                print(
                    f"  KID: {metrics['kernel_inception_distance_mean']:.6f} ± {metrics['kernel_inception_distance_std']:.6f}"
                )

        # Print per-time results summary
        print("\n" + "=" * 50)
        print("Results per Time Step:")
        for time_name, m in results.items():
            line = f"  {time_name} (real:{m['real_folder']})"
            if "fid" in m:
                line += f": FID={m['fid']:.4f}"
            if "kid_mean" in m:
                line += f", KID={m['kid_mean']:.6f}±{m['kid_std']:.6f}"
            print(line)

    fe_short = feature_extractor.replace("-", "_")

    # --- Pairwise matrix (only for true_vs_true) ---
    real_folders = [GEN_TO_REAL_FOLDER[i] for i in sorted(GEN_TO_REAL_FOLDER.keys())]

    # true_vs_true: rows=real folders, cols=real folders
    if mode in ("true_vs_true", "both"):
        print("\nComputing true_vs_true matrix (real vs real)...")
        labels = real_folders
        fid_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
        kid_mean_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
        kid_std_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
        kid_mean_path = None
        kid_std_path = None

        for a in labels:
            for b in labels:
                real_a = real_images_base_path / a
                real_b = real_images_base_path / b
                if not real_a.exists() or not real_b.exists():
                    fid_df.loc[a, b] = float("nan")
                    if compute_kid:
                        kid_mean_df.loc[a, b] = float("nan")
                        kid_std_df.loc[a, b] = float("nan")
                    continue

                try:
                    m = compute_metrics_for_dirs(
                        gen_dir=real_a,
                        real_dir=real_b,
                        feature_extractor=feature_extractor,
                        compute_fid=compute_fid,
                        compute_kid=compute_kid,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    print(f"Error computing true_vs_true cell {a} vs {b}: {e}")
                    fid_df.loc[a, b] = float("nan")
                    if compute_kid:
                        kid_mean_df.loc[a, b] = float("nan")
                        kid_std_df.loc[a, b] = float("nan")
                    continue

                if compute_fid and "frechet_inception_distance" in m:
                    fid_df.loc[a, b] = m["frechet_inception_distance"]
                else:
                    fid_df.loc[a, b] = float("nan")

                if compute_kid and "kernel_inception_distance_mean" in m:
                    kid_mean_df.loc[a, b] = m["kernel_inception_distance_mean"]
                    kid_std_df.loc[a, b] = m["kernel_inception_distance_std"]
                elif compute_kid:
                    kid_mean_df.loc[a, b] = float("nan")
                    kid_std_df.loc[a, b] = float("nan")

        fid_mat_path = output_dir / f"{fe_short}_true_vs_true_fid_matrix.csv"
        fid_df.to_csv(fid_mat_path)
        print(f"Saved true_vs_true FID matrix to {fid_mat_path}")

        if compute_kid:
            kid_mean_path = output_dir / f"{fe_short}_true_vs_true_kid_mean_matrix.csv"
            kid_std_path = output_dir / f"{fe_short}_true_vs_true_kid_std_matrix.csv"
            kid_mean_df.to_csv(kid_mean_path)
            kid_std_df.to_csv(kid_std_path)
            print(
                f"Saved true_vs_true KID mean/std matrices to {kid_mean_path} and {kid_std_path}"
            )

        results["true_vs_true_matrices"] = {
            "fid": str(fid_mat_path),
            "kid_mean": str(kid_mean_path) if kid_mean_path is not None else None,
            "kid_std": str(kid_std_path) if kid_std_path is not None else None,
        }

    return results


def main():
    # Computed paths
    output_dir = OUTPUT_DIR if OUTPUT_DIR else BASE_PATH / "output"
    images_dir = output_dir / "images"
    fe_short = FEATURE_EXTRACTOR.replace("-", "_")

    # Print configuration BEFORE safety checks
    print("=" * 60)
    print("Video Metrics Computation (torch-fidelity)")
    print("=" * 60)
    print(f"Base path: {BASE_PATH}")
    print(f"Real images path: {REAL_IMAGES_BASE_PATH}")
    print(f"Output dir: {output_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Feature extractor: {FEATURE_EXTRACTOR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Compute FID: {COMPUTE_FID}")
    print(f"Compute KID: {COMPUTE_KID}")
    print(f"Mode: {MODE}")
    print(f"Skip conversion: {SKIP_CONVERSION}")
    print(f"Compute histograms: {COMPUTE_HIST}")
    print(f"Compute metrics: {COMPUTE_METRICS}")
    print(f"Replot metrics: {REPLOT_METRICS}")
    print("=" * 60)

    # Warn if nothing will be computed
    if not COMPUTE_HIST and not COMPUTE_METRICS and not REPLOT_METRICS:
        print(
            "\nWARNING: No operations enabled (COMPUTE_HIST, COMPUTE_METRICS, REPLOT_METRICS all False)."
        )
        print("Nothing will be computed. Exiting.")
        return

    # Warn if conversion is enabled but not needed for true_vs_true mode
    if not SKIP_CONVERSION and MODE == "true_vs_true" and not COMPUTE_HIST:
        print(
            "\nWARNING: SKIP_CONVERSION=False but MODE='true_vs_true' and COMPUTE_HIST=False."
        )
        print(
            "Conversion creates generated images that won't be used. Consider SKIP_CONVERSION=True."
        )

    # --- Safety checks: prevent overwriting data ---
    # Check output_dir is not inside base_path or real_images_path
    try:
        output_dir.resolve().relative_to(BASE_PATH.resolve())
        raise RuntimeError(
            f"OUTPUT_DIR ({output_dir}) is inside BASE_PATH ({BASE_PATH}). "
            "This risks overwriting source data. Aborting."
        )
    except ValueError:
        pass  # Good: output_dir is not inside base_path

    try:
        output_dir.resolve().relative_to(REAL_IMAGES_BASE_PATH.resolve())
        raise RuntimeError(
            f"OUTPUT_DIR ({output_dir}) is inside REAL_IMAGES_BASE_PATH ({REAL_IMAGES_BASE_PATH}). "
            "This risks overwriting source data. Aborting."
        )
    except ValueError:
        pass  # Good: output_dir is not inside real_images_path

    # Check if output files already exist
    files_to_write = []

    # Histogram file
    if COMPUTE_HIST:
        files_to_write.append(output_dir / "pixel_histograms_all_times.png")

    # Metrics CSVs and plots (if computing)
    if COMPUTE_METRICS:
        # Per-time metrics CSV only written for gen_vs_true mode
        if MODE in ("gen_vs_true", "both"):
            files_to_write.append(output_dir / f"metrics_results_{fe_short}.csv")
            files_to_write.append(output_dir / f"{fe_short}_gen_vs_true_barplot.png")
        if MODE in ("true_vs_true", "both"):
            files_to_write.append(
                output_dir / f"{fe_short}_true_vs_true_fid_matrix.csv"
            )
            if COMPUTE_KID:
                files_to_write.append(
                    output_dir / f"{fe_short}_true_vs_true_kid_mean_matrix.csv"
                )
                files_to_write.append(
                    output_dir / f"{fe_short}_true_vs_true_kid_std_matrix.csv"
                )
            files_to_write.append(output_dir / f"{fe_short}_true_vs_true_heatmaps.png")

    # Replot only checks plot files (CSVs must exist)
    if REPLOT_METRICS and not COMPUTE_METRICS:
        if MODE in ("gen_vs_true", "both"):
            files_to_write.append(output_dir / f"{fe_short}_gen_vs_true_barplot.png")
        if MODE in ("true_vs_true", "both"):
            files_to_write.append(output_dir / f"{fe_short}_true_vs_true_heatmaps.png")

    existing_files = [f for f in files_to_write if f.exists()]
    if existing_files:
        raise RuntimeError(
            "Output files already exist and would be overwritten:\n"
            + "\n".join(f"  - {f}" for f in existing_files)
            + "\nDelete them manually or change OUTPUT_DIR. Aborting."
        )

    # Check if images_dir has content when not skipping conversion
    if not SKIP_CONVERSION and images_dir.exists() and any(images_dir.iterdir()):
        raise RuntimeError(
            f"Images directory {images_dir} already exists and is not empty. "
            "Set SKIP_CONVERSION=True or delete it. Aborting."
        )

    # Now safe to create output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input paths
    assert BASE_PATH.exists(), f"Base path {BASE_PATH} does not exist."
    assert REAL_IMAGES_BASE_PATH.exists(), (
        f"Real images path {REAL_IMAGES_BASE_PATH} does not exist."
    )

    # Step 1: Convert .pt to images
    if not SKIP_CONVERSION:
        print("\n[Step 1/3] Converting .pt files to PNG images...")
        convert_pt_to_images(BASE_PATH, images_dir)
    else:
        print("\n[Step 1/3] Skipping conversion (SKIP_CONVERSION=True)")

    if COMPUTE_HIST:
        # Step 2: Compute and save histograms for visual inspection
        print("\n[Step 2/3] Computing pixel histograms...")
        plot_all_histograms(
            images_dir=images_dir,
            real_images_path=REAL_IMAGES_BASE_PATH,
            output_dir=output_dir,
            mode=MODE,
            gen_to_real_folder=GEN_TO_REAL_FOLDER,
            sample_size=1000,
        )
    else:
        print("\n[Step 2/3] Skipping histograms (COMPUTE_HIST=False)")

    # Step 3: Compute metrics and plot
    if COMPUTE_METRICS:
        print("\n[Step 3/3] Computing metrics...")
        results = compute_all_metrics(
            images_dir,
            REAL_IMAGES_BASE_PATH,
            output_dir=output_dir,
            feature_extractor=FEATURE_EXTRACTOR,
            compute_fid=COMPUTE_FID,
            compute_kid=COMPUTE_KID,
            batch_size=BATCH_SIZE,
            mode=MODE,
        )

        # Build results dataframe (only for gen_vs_true per-time metrics)
        rows = []
        for time_key, metrics in results.items():
            # Skip matrix path entries
            if "matrices" in time_key:
                continue
            row = {"Time": time_key}
            if "real_folder" in metrics:
                row["Real Folder"] = metrics["real_folder"]
            if "fid" in metrics:
                row["FID"] = metrics["fid"]
            if "kid_mean" in metrics:
                row["KID Mean"] = metrics["kid_mean"]
                row["KID Std"] = metrics["kid_std"]
            rows.append(row)

        if rows:
            results_df = pd.DataFrame(rows)
            print("\n" + "=" * 60)
            print("RESULTS PER TIME STEP")
            print("=" * 60)
            print(results_df.to_string(index=False))

            # Save results
            results_path = output_dir / f"metrics_results_{fe_short}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to {results_path}")

        # Plot metrics (part of compute_metrics)
        print("\n[Step 3/3] Generating metric plots...")

        # Plot gen_vs_true bar charts
        if MODE in ("gen_vs_true", "both"):
            results_csv = output_dir / f"metrics_results_{fe_short}.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df = df[df["Time"].str.startswith("time", na=False)]
                if not df.empty:
                    plot_gen_vs_true_barplot(
                        fid_values=df["FID"].tolist(),
                        kid_means=df["KID Mean"].tolist()
                        if "KID Mean" in df.columns
                        else None,
                        kid_stds=df["KID Std"].tolist()
                        if "KID Std" in df.columns
                        else None,
                        time_labels=df["Time"].tolist(),
                        real_labels=df["Real Folder"].astype(str).tolist(),
                        output_dir=output_dir,
                        fe_short=fe_short,
                    )

        # Plot true_vs_true heatmaps
        if MODE in ("true_vs_true", "both"):
            fid_csv = output_dir / f"{fe_short}_true_vs_true_fid_matrix.csv"
            kid_csv = output_dir / f"{fe_short}_true_vs_true_kid_mean_matrix.csv"
            if fid_csv.exists():
                plot_true_vs_true_heatmaps(
                    fid_df=pd.read_csv(fid_csv, index_col=0),
                    kid_mean_df=pd.read_csv(kid_csv, index_col=0)
                    if kid_csv.exists()
                    else None,
                    output_dir=output_dir,
                    fe_short=fe_short,
                )

    elif REPLOT_METRICS:
        # Replot from existing CSVs
        print("\n[Step 3/3] Regenerating metric plots from existing CSVs...")

        if MODE in ("gen_vs_true", "both"):
            results_csv = output_dir / f"metrics_results_{fe_short}.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df = df[df["Time"].str.startswith("time", na=False)]
                if not df.empty:
                    plot_gen_vs_true_barplot(
                        fid_values=df["FID"].tolist(),
                        kid_means=df["KID Mean"].tolist()
                        if "KID Mean" in df.columns
                        else None,
                        kid_stds=df["KID Std"].tolist()
                        if "KID Std" in df.columns
                        else None,
                        time_labels=df["Time"].tolist(),
                        real_labels=df["Real Folder"].astype(str).tolist(),
                        output_dir=output_dir,
                        fe_short=fe_short,
                    )
            else:
                print(f"WARNING: Cannot plot gen_vs_true - {results_csv} not found")

        if MODE in ("true_vs_true", "both"):
            fid_csv = output_dir / f"{fe_short}_true_vs_true_fid_matrix.csv"
            kid_csv = output_dir / f"{fe_short}_true_vs_true_kid_mean_matrix.csv"
            if fid_csv.exists():
                plot_true_vs_true_heatmaps(
                    fid_df=pd.read_csv(fid_csv, index_col=0),
                    kid_mean_df=pd.read_csv(kid_csv, index_col=0)
                    if kid_csv.exists()
                    else None,
                    output_dir=output_dir,
                    fe_short=fe_short,
                )
            else:
                print(f"WARNING: Cannot plot true_vs_true - {fid_csv} not found")

    else:
        print(
            "\n[Step 3/3] Skipping metrics (COMPUTE_METRICS=False, REPLOT_METRICS=False)"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
