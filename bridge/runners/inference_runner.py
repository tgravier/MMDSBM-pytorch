# bridge/runners/inference_runner.py

import torch
import os
import numpy as np
from bridge.runners.train_runner import trainer_bridges
from conf.conf_loader import load_test_datasets
from utils.metrics import evaluate_swd_over_time, evaluate_mmd_over_time


def inference_bridges(
    config_classes,
    tracking_logger,
    logger,
    experiment_path: str,
    weight_epoch: int,
    n_runs: int = 1,
    
):
    """
    Run inference using a trained Schrödinger Bridge model.

    Args:
        config_classes: Configuration classes
        tracking_logger: Weights & Biases logger
        logger: Standard logger
        experiment_path: Path to the experiment containing trained model
        weight_epoch: Epoch number to load (e.g. 25 for 0025_forward.pth)
        n_runs: Number of inference runs to compute metrics with uncertainty
    """
    logger.info("Starting inference...")
    logger.info(f"Loading model from: {experiment_path}")
    logger.info(f"Using weights from epoch: {weight_epoch}")

    # Ensure inference runs on CPU
    config_classes.device = "cpu"

    # Create a trainer instance (which extends N_Bridges)
    trainer = trainer_bridges(
        config_classes=config_classes,
        tracking_logger=tracking_logger,
        logger=logger,
        resume_train=False,
        inference=True,
    )

    # Load the trained model checkpoint from specific epoch
    weights_dir = os.path.join(experiment_path, "network_weight")
    if not os.path.exists(weights_dir):
        logger.error(f"No network_weight directory found in {experiment_path}")
        raise FileNotFoundError(
            f"No network_weight directory found in {experiment_path}"
        )

    # Format epoch number with leading zeros (4 digits)
    epoch_str = f"{weight_epoch:04d}"

    # Load only forward checkpoint since we test forcément en forward
    forward_checkpoint_path = os.path.join(weights_dir, f"{epoch_str}_forward.pth")

    if not os.path.isfile(forward_checkpoint_path):
        logger.error(f"No forward checkpoint found at {forward_checkpoint_path}")
        raise FileNotFoundError(
            f"No forward checkpoint found at {forward_checkpoint_path}"
        )

    logger.info(f"Loading forward checkpoint from {forward_checkpoint_path}")
    forward_checkpoint = torch.load(
        forward_checkpoint_path, map_location=config_classes.device, weights_only=False
    )

    # Load forward model state
    trainer.net_fwd.load_state_dict(forward_checkpoint['state_dict'])
    if trainer.experiment_config.ema and hasattr(trainer, "net_fwd_ema"):
        # For .pth files, the checkpoint might be just the state dict or include ema
        if (
            isinstance(forward_checkpoint, dict)
            and "ema_state_dict" in forward_checkpoint
        ):
            trainer.net_fwd_ema.load_state_dict(forward_checkpoint["state_dict"])
        else:
            trainer.net_fwd_ema.load_state_dict(forward_checkpoint["state_dict"])

    # Set forward model to eval mode
    trainer.net_fwd.eval()
    if trainer.experiment_config.ema and hasattr(trainer, "net_fwd_ema"):
        trainer.net_fwd_ema.eval()

    # For inference in forward direction, we only need the forward network
    # Create net_dict with only forward model (as we test forcément en forward)
    if trainer.experiment_config.ema and hasattr(trainer, "net_fwd_ema"):
        net_dict = {"forward": trainer.net_fwd_ema}
    else:
        net_dict = {"forward": trainer.net_fwd}

    # Run inference
    logger.info("Running inference test...")

    # Load test datasets from experiment directory if they exist
    try:
        datasets_inference = load_test_datasets(experiment_path)
        logger.info(
            f"Loaded {len(datasets_inference)} test datasets from experiment directory"
        )
    except FileNotFoundError as e:
        logger.warning(f"Could not load test datasets from experiment: {e}")
        logger.info("Falling back to trainer's datasets...")
        # Use trainer's datasets if available, otherwise use train datasets
        datasets_inference = getattr(trainer, "datasets_test", trainer.datasets_train)

        # Ensure datasets_inference is a list
        if not isinstance(datasets_inference, list):
            datasets_inference = (
                [datasets_inference]
                if datasets_inference is not None
                else trainer.datasets_train
            )

    # Store metrics for multiple runs
    swd_scores = []
    mmd_scores = []

    logger.info(
        f"Running {n_runs} inference run(s) to compute metrics with uncertainty..."
    )

    with torch.no_grad():
        # Cast to suppress type warnings - datasets_inference should be a list
        datasets_for_inference = (
            list(datasets_inference)
            if isinstance(datasets_inference, (list, tuple))
            else [datasets_inference]
        )

        datasets_for_inference = trainer.leave_out_datasets(datasets_for_inference)



        for run_idx in range(n_runs):
            logger.info(f"Starting inference run {run_idx + 1}/{n_runs}")

            # Run inference
            generated, time = trainer.inference_test(
                args=config_classes,
                datasets_inference=datasets_for_inference,
                direction_tosample="forward",
                net_dict=net_dict,
                outer_iter_idx=0,
                num_samples=getattr(config_classes, "n_sample", 1000),
                sigma=getattr(config_classes, "sigma_inference", 0.0),
            )

            # Compute metrics for this run
            logger.info(f"Computing metrics for run {run_idx + 1}")

            # Compute SWD
            try:
                swd_results = evaluate_swd_over_time(
                    generated=generated,
                    time=time,
                    datasets_inference=datasets_for_inference,
                    direction_tosample="forward",
                    n_proj=50,
                )
                # Store all SWD values for this run (one per time point)
                run_swd_values = [swd for t, swd in swd_results]
                swd_scores.append(run_swd_values)

                # Log individual time points for this run
                for t, swd in swd_results:
                    logger.info(f"Run {run_idx + 1} - SWD @ t={t:.2f}: {swd:.6f}")

            except Exception as e:
                logger.warning(f"Failed to compute SWD for run {run_idx + 1}: {e}")

            # Compute MMD
            try:
                mmd_results = evaluate_mmd_over_time(
                    generated=generated,
                    time=time,
                    datasets_inference=datasets_for_inference,
                    direction_tosample="forward",
                    kernel_type=getattr(config_classes, "mmd_kernel", "rbf"),
                    blur=getattr(config_classes, "mmd_blur", 1.0),
                )
                # Store all MMD values for this run (one per time point)
                run_mmd_values = [mmd for t, mmd in mmd_results]
                mmd_scores.append(run_mmd_values)

                # Log individual time points for this run
                for t, mmd in mmd_results:
                    logger.info(f"Run {run_idx + 1} - MMD @ t={t:.2f}: {mmd:.6f}")

            except Exception as e:
                logger.warning(f"Failed to compute MMD for run {run_idx + 1}: {e}")

    # Compute final statistics
    if swd_scores:
        # swd_scores is now a list of lists: [run1_values, run2_values, ...]
        # where each run_values is [swd_t0, swd_t1, swd_t2, ...]

        # Convert to numpy array for easier manipulation
        swd_array = np.array(swd_scores)  # Shape: (n_runs, n_time_points)

        # Compute statistics across runs for each time point
        swd_means_per_time = np.mean(
            swd_array, axis=0
        )  # Mean across runs for each time
        swd_stds_per_time = (
            np.std(swd_array, axis=0)
            if len(swd_scores) > 1
            else np.zeros_like(swd_means_per_time)
        )

        # Also compute overall statistics (mean across all time points and runs)
        swd_overall_mean = np.mean(swd_array)
        swd_overall_std = np.std(swd_array) if len(swd_scores) > 1 else 0.0

        logger.info("=== SWD Results ===")
        logger.info(f"Overall SWD: {swd_overall_mean:.6f} ± {swd_overall_std:.6f}")
        print(f"Overall SWD: {swd_overall_mean:.6f} ± {swd_overall_std:.6f}")

        # Log per-time statistics
        time_values = [ds.get_time() for ds in datasets_for_inference]
        for i, (t, mean_swd, std_swd) in enumerate(
            zip(time_values, swd_means_per_time, swd_stds_per_time)
        ):
            logger.info(f"SWD @ t={t:.2f}: {mean_swd:.6f} ± {std_swd:.6f}")
            print(f"SWD @ t={t:.2f}: {mean_swd:.6f} ± {std_swd:.6f}")

    if mmd_scores:
        # mmd_scores is now a list of lists: [run1_values, run2_values, ...]
        # where each run_values is [mmd_t0, mmd_t1, mmd_t2, ...]

        # Convert to numpy array for easier manipulation
        mmd_array = np.array(mmd_scores)  # Shape: (n_runs, n_time_points)

        # Compute statistics across runs for each time point
        mmd_means_per_time = np.mean(
            mmd_array, axis=0
        )  # Mean across runs for each time
        mmd_stds_per_time = (
            np.std(mmd_array, axis=0)
            if len(mmd_scores) > 1
            else np.zeros_like(mmd_means_per_time)
        )

        # Also compute overall statistics (mean across all time points and runs)
        mmd_overall_mean = np.mean(mmd_array)
        mmd_overall_std = np.std(mmd_array) if len(mmd_scores) > 1 else 0.0

        logger.info("=== MMD Results ===")
        logger.info(f"Overall MMD: {mmd_overall_mean:.6f} ± {mmd_overall_std:.6f}")
        print(f"Overall MMD: {mmd_overall_mean:.6f} ± {mmd_overall_std:.6f}")

        # Log per-time statistics
        time_values = [ds.get_time() for ds in datasets_for_inference]
        for i, (t, mean_mmd, std_mmd) in enumerate(
            zip(time_values, mmd_means_per_time, mmd_stds_per_time)
        ):
            logger.info(f"MMD @ t={t:.2f}: {mean_mmd:.6f} ± {std_mmd:.6f}")
            print(f"MMD @ t={t:.2f}: {mean_mmd:.6f} ± {std_mmd:.6f}")

    # Log to wandb if available
    if tracking_logger and swd_scores:
        # Log overall statistics
        tracking_logger.log(
            {
                "inference/swd_overall_mean": swd_overall_mean,
                "inference/swd_overall_std": swd_overall_std,
                "inference/n_runs": n_runs,
            }
        )

        # Log per-time statistics
        time_values = [ds.get_time() for ds in datasets_for_inference]
        for i, (t, mean_swd, std_swd) in enumerate(
            zip(time_values, swd_means_per_time, swd_stds_per_time)
        ):
            tracking_logger.log(
                {
                    f"inference/swd_t_{t:.2f}_mean": mean_swd,
                    f"inference/swd_t_{t:.2f}_std": std_swd,
                }
            )

    if tracking_logger and mmd_scores:
        # Log overall statistics
        tracking_logger.log(
            {
                "inference/mmd_overall_mean": mmd_overall_mean,
                "inference/mmd_overall_std": mmd_overall_std,
                "inference/n_runs": n_runs,
            }
        )

        # Log per-time statistics
        time_values = [ds.get_time() for ds in datasets_for_inference]
        for i, (t, mean_mmd, std_mmd) in enumerate(
            zip(time_values, mmd_means_per_time, mmd_stds_per_time)
        ):
            tracking_logger.log(
                {
                    f"inference/mmd_t_{t:.2f}_mean": mean_mmd,
                    f"inference/mmd_t_{t:.2f}_std": std_mmd,
                }
            )

    # Save results to Excel
    save_results_to_excel(
        experiment_path,
        weight_epoch,
        n_runs,
        swd_scores,
        mmd_scores,
        time_values,
        swd_means_per_time,
        swd_stds_per_time,
        mmd_means_per_time,
        mmd_stds_per_time,
        swd_overall_mean,
        swd_overall_std,
        mmd_overall_mean,
        mmd_overall_std,
        logger,
    )

    logger.info("Inference completed successfully!")
    return trainer


def save_results_to_excel(
    experiment_path: str,
    weight_epoch: int,
    n_runs: int,
    swd_scores: list,
    mmd_scores: list,
    time_values: list,
    swd_means_per_time: list,
    swd_stds_per_time: list,
    mmd_means_per_time: list,
    mmd_stds_per_time: list,
    swd_overall_mean,  # Can be numpy type
    swd_overall_std,  # Can be numpy type
    mmd_overall_mean,  # Can be numpy type
    mmd_overall_std,  # Can be numpy type
    logger,
):
    """
    Save inference results to Excel files with detailed breakdown.
    """
    import pandas as pd
    from datetime import datetime

    # Convert numpy values to standard Python floats
    swd_overall_mean = float(swd_overall_mean) if swd_overall_mean is not None else 0.0
    swd_overall_std = float(swd_overall_std) if swd_overall_std is not None else 0.0
    mmd_overall_mean = float(mmd_overall_mean) if mmd_overall_mean is not None else 0.0
    mmd_overall_std = float(mmd_overall_std) if mmd_overall_std is not None else 0.0

    # Create results directory
    results_dir = os.path.join(experiment_path, "inference_results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_epoch_{weight_epoch:04d}_{n_runs}runs_{timestamp}.xlsx"
    filepath = os.path.join(results_dir, filename)

    try:
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # ===== Summary Sheet =====
            summary_data = {
                "Metric": ["SWD Overall", "MMD Overall"],
                "Mean": [swd_overall_mean, mmd_overall_mean],
                "Std": [swd_overall_std, mmd_overall_std],
                "N_Runs": [n_runs, n_runs],
                "Weight_Epoch": [weight_epoch, weight_epoch],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # ===== Per-Time Results Sheet =====
            if time_values and swd_means_per_time and mmd_means_per_time:
                per_time_data = {
                    "Time": time_values,
                    "SWD_Mean": swd_means_per_time,
                    "SWD_Std": swd_stds_per_time,
                    "MMD_Mean": mmd_means_per_time,
                    "MMD_Std": mmd_stds_per_time,
                }
                per_time_df = pd.DataFrame(per_time_data)
                per_time_df.to_excel(writer, sheet_name="Per_Time_Results", index=False)

            # ===== Raw SWD Data Sheet =====
            if swd_scores:
                # Create DataFrame with runs as columns and time points as rows
                swd_data = {}
                for run_idx, run_scores in enumerate(swd_scores):
                    swd_data[f"Run_{run_idx + 1}"] = run_scores

                # Add time values as index
                swd_df = pd.DataFrame(swd_data)
                if time_values and len(time_values) == len(swd_df):
                    swd_df.insert(0, "Time", time_values)
                swd_df.to_excel(writer, sheet_name="Raw_SWD_Data", index=False)

            # ===== Raw MMD Data Sheet =====
            if mmd_scores:
                # Create DataFrame with runs as columns and time points as rows
                mmd_data = {}
                for run_idx, run_scores in enumerate(mmd_scores):
                    mmd_data[f"Run_{run_idx + 1}"] = run_scores

                # Add time values as index
                mmd_df = pd.DataFrame(mmd_data)
                if time_values and len(time_values) == len(mmd_df):
                    mmd_df.insert(0, "Time", time_values)
                mmd_df.to_excel(writer, sheet_name="Raw_MMD_Data", index=False)

            # ===== Metadata Sheet =====
            metadata = {
                "Parameter": [
                    "Experiment_Path",
                    "Weight_Epoch",
                    "N_Runs",
                    "Timestamp",
                    "N_Time_Points",
                ],
                "Value": [
                    experiment_path,
                    weight_epoch,
                    n_runs,
                    timestamp,
                    len(time_values) if time_values else 0,
                ],
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        logger.info(f"Results saved to Excel: {filepath}")
        print(f"Results saved to Excel: {filepath}")

    except Exception as e:
        logger.warning(f"Failed to save results to Excel: {e}")
        print(f"Warning: Failed to save results to Excel: {e}")

        # Fallback: save as CSV
        try:
            csv_filepath = filepath.replace(".xlsx", "_summary.csv")
            summary_df.to_csv(csv_filepath, index=False)
            logger.info(f"Saved summary to CSV as fallback: {csv_filepath}")
        except Exception as csv_e:
            logger.warning(f"Failed to save CSV fallback: {csv_e}")
