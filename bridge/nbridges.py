# bridge/trainer_dsbm.py

import torch
import os
import glob
import random
import numpy as np
from typing import Tuple, Dict, List, Optional, Sequence
import json
import wandb

from bridge.core_dsbm import IMF_DSBM
from datasets.datasets_registry import DatasetConfig
from datasets.datasets import load_dataset, TimedDataset
from bridge.sde.bridge_sampler import inference_sample_sde
from utils.visualization import make_trajectory
from typing import Union, List, Tuple
from utils.metrics import (
    evaluate_swd_over_time,
    evaluate_energy_over_time,
    evaluate_wd_over_time,
    evaluate_mmd_over_time,
)


class N_Bridges(IMF_DSBM):
    def __init__(
        self,
        args,
        net_fwd,
        net_bwd,
        net_fwd_ema,
        net_bwd_ema,
        optimizer,
        tracking_logger,
        n_distribution: int,
        distributions: Sequence[DatasetConfig],
        inference: bool
    ):
        self.args = args
        self.args.inference = inference
        
        self.net_fwd_ema = net_fwd_ema
        self.net_bwd_ema = net_bwd_ema

        self.tracking_logger = tracking_logger

        self._validate_config_time_unique(distributions, "train")

        if len(distributions) != n_distribution:
            raise ValueError("Length of distributions must match n_distribution")

        self.n_distribution = n_distribution
        self.distributions = distributions

        if self.args.separation_train_test and not self.args.inference:
            self.datasets_train, self.datasets_test = self.prepare_dataset(
                distributions, separation_train_test=self.args.separation_train_test
            )
            self.datasets_train = self.leave_out_datasets(self.datasets_train)

            # Save test datasets to experiment directory
            self.save_test_datasets()

        else:
            self.datasets_train = self.prepare_dataset(distributions)
            self.datasets_train = self.leave_out_datasets(self.datasets_train)

        self.min_time, self.max_time = self.min_max_time(self.datasets_train)

        super().__init__(
            args=args,
            num_simulation_steps=args.num_simulation_steps,
            optimizer=optimizer,
            min_time=self.min_time,
            max_time=self.max_time,
            net_fwd=net_fwd,
            net_bwd=net_bwd,
            net_fwd_ema=net_fwd_ema,
            net_bwd_ema=net_bwd_ema,
            sig=args.sigma,
            eps=args.eps,
        )

    def leave_out_datasets(self, time_datasets_list):
        datasets_train_filtered = []
        for ds in time_datasets_list:
            if ds.get_time() in self.args.leave_out_list:
                self.args.leave_out_list.append(ds)
            else:
                datasets_train_filtered.append(ds)
        return datasets_train_filtered

    def min_max_time(self, datasets):
        max_time = max(float(ds.get_time()) for ds in datasets)
        min_time = min(float(ds.get_time()) for ds in datasets)

        return min_time, max_time

    def _validate_config_time_unique(
        self, distributions: List[DatasetConfig], mode: str
    ):
        times = [d.time for d in distributions]
        duplicates = {t for t in times if times.count(t) > 1 and t is not None}
        if duplicates:
            dups = [d for d in distributions if d.time in duplicates]
            raise ValueError(
                f"Duplicate time values found in {mode} distributions: {duplicates}\n"
                + "\n".join(f"- {repr(d)}" for d in dups)
            )

    def prepare_dataset(
        self, distributions: List[DatasetConfig], separation_train_test: bool = False
    ) -> Union[List[TimedDataset], Tuple[List[TimedDataset], List[TimedDataset]]]:
        if separation_train_test:
            # load_dataset returns (train, test) tuple for each config
            train_test_pairs = [
                load_dataset(
                    cfg,
                    separation_train_test=self.args.separation_train_test,
                    nb_points_test=self.args.nb_points_test,
                )
                for cfg in distributions
            ]
            train_datasets = [pair[0] for pair in train_test_pairs]
            test_datasets = [pair[1] for pair in train_test_pairs]
            return train_datasets, test_datasets
        else:
            return [load_dataset(cfg) for cfg in distributions]

    def generate_dataset_pairs(
        self, forward_pairs: List[Tuple[TimedDataset, TimedDataset]]
    ):
        time_dataset_init = [pair[0] for pair in forward_pairs]
        time_dataset_target = [pair[1] for pair in forward_pairs]

        x0 = [
            d.get_all().to(self.args.accelerator.device) for d in time_dataset_init
        ]  # (n_times - 1, n_points, data_dim)

        x1 = [
            d.get_all().to(self.args.accelerator.device) for d in time_dataset_target
        ]  # (n_times - 1, n_points, data_dim)

        dim = x0[0].size()[-1]

        assert len(x0) == len(x1), f"{len(x0)} != {len(x1)}"

        all_left_of_pairs = []
        all_right_of_pairs = []
        all_times = []

        for bridge_idx in range(len(x0)):
            nb_points_left = len(x0[bridge_idx])
            nb_points_right = len(x1[bridge_idx])
            nb_pairs = max(nb_points_left, nb_points_right)

            # sample with replacement in left marginal
            pairs_left_idx = torch.randint(nb_points_left, (nb_pairs,))
            pairs_left = x0[bridge_idx][pairs_left_idx]  # (nb_pairs, data_dim)
            all_left_of_pairs.append(pairs_left)
            # sample with replacement in right marginal
            pairs_right_idx = torch.randint(nb_points_right, (nb_pairs,))
            pairs_right = x1[bridge_idx][pairs_right_idx]  # (nb_pairs, data_dim)
            all_right_of_pairs.append(pairs_right)

            # times
            t_left = time_dataset_init[bridge_idx].get_time()
            t_right = time_dataset_target[bridge_idx].get_time()
            t_left = torch.full((nb_pairs,), t_left).to(self.args.accelerator.device)
            t_right = torch.full((nb_pairs,), t_right).to(self.args.accelerator.device)
            times = torch.stack([t_left, t_right], dim=1)  # (nb_pairs, 2)
            all_times.append(times)

        all_left_of_pairs = torch.cat(all_left_of_pairs)
        all_right_of_pairs = torch.cat(all_right_of_pairs)

        # all_left_of_pairs = torch.stack(all_left_of_pairs)
        # # (n_times-1, nb_pairs, data_dim)
        # all_right_of_pairs = torch.stack(all_right_of_pairs)
        # # (n_times-1, nb_pairs, data_dim)

        x_pairs = torch.stack([all_left_of_pairs, all_right_of_pairs], dim=1)
        # (n_times - 1, nb_pairs, 2, data_dim)
        x_pairs = x_pairs.reshape(
            -1, 2, dim
        )  # dim extract from x0 at the beginning of this function
        # ((n_times - 1) * nb_pairs, 2, data_dim)

        all_times = torch.cat(all_times, dim=0)  # ((n_times - 1) * nb_pairs, 2)

        return x_pairs, all_times

    def train(self):
        skip_forward = False
        if getattr(self.args, "resume_train", False):
            if self.args.previous_direction_to_train == "forward":
                skip_forward = True

        for outer_iter_idx in range(
            self.args.resume_train_nb_outer_iterations, self.args.nb_outer_iterations
        ):
            print(f"\n[Epoch {outer_iter_idx}]")

            forward_pairs = list(zip(self.datasets_train[:-1], self.datasets_train[1:]))

            if not (
                skip_forward
                and outer_iter_idx == self.args.resume_train_nb_outer_iterations
            ):
                print("Training FORWARD bridge")

                direction_to_train = "forward"
                x_pairs, t_pairs = self.generate_dataset_pairs(forward_pairs)

                loss_curve, grad_curve, net_dict, ema_dict = self.train_one_direction(
                    direction=direction_to_train,
                    x_pairs=x_pairs,
                    t_pairs=t_pairs,
                    outer_iter_idx=outer_iter_idx,
                )

                self.orchestrate_experiment(
                    args=self.args,
                    outer_iter_idx=outer_iter_idx,
                    direction_to_train=direction_to_train,
                    net_dict=ema_dict,  # TODO create an intermediary pointer if ema_dict is false : like net_inference -> ema_dict or net_dict
                    loss_curve=loss_curve,
                    grad_curve=grad_curve,
                )

            else:
                print("Skipping FORWARD training for first resumed iteration")

            print("Training BACKWARD bridge")

            direction_to_train = "backward"

            x_pairs, t_pairs = self.generate_dataset_pairs(forward_pairs)
            loss_curve, grad_curve, net_dict, ema_dict = self.train_one_direction(
                direction=direction_to_train,
                x_pairs=x_pairs,
                t_pairs=t_pairs,
                outer_iter_idx=outer_iter_idx,
            )

            self.orchestrate_experiment(
                args=self.args,
                outer_iter_idx=outer_iter_idx,
                direction_to_train=direction_to_train,
                net_dict=ema_dict,
                loss_curve=loss_curve,
                grad_curve=grad_curve,
            )

    def inference_test(
        self,
        args,
        direction_tosample: str,
        net_dict: dict,
        datasets_inference: list,
        outer_iter_idx: int,
        num_samples: int = 100,
        num_steps: int = 200,
        sigma: float = 1.0,
    ):
        device = next(net_dict[direction_tosample].parameters()).device

        min_idx = 0
        max_idx = -1

        z0 = (
            datasets_inference[min_idx if direction_tosample == "forward" else max_idx]
            .get_all()
            .to(device)
        )

        idx = torch.randint(0, z0.shape[0], (num_samples,))
        z0_sampled = z0[idx]

        # Build t_pairs as tensor of shape (n_bridge, 2)
        t_pairs_datasets = torch.tensor(
            [
                [datasets_inference[i].get_time(), datasets_inference[i + 1].get_time()]
                for i in range(len(datasets_inference) - 1)
            ],
            dtype=torch.float32,
            device=device,
        )

        generated, time = inference_sample_sde(
            zstart=z0_sampled,
            net_dict=net_dict,
            t_pairs=t_pairs_datasets,
            direction_tosample=direction_tosample,
            N=num_steps,
            sig=sigma,
            device=device,
        )

        return generated, time

    def save_test_datasets(self):
        """
        Save test datasets to the experiment directory when separation_train_test is enabled.
        Each dataset is saved as a .pt file with the time value in the filename.
        """
        if not hasattr(self, "datasets_test") or self.datasets_test is None:
            return

        base_dir = os.path.join(self.args.experiment_dir, self.args.experiment_name)
        datasets_test_dir = os.path.join(base_dir, "datasets_test")
        os.makedirs(datasets_test_dir, exist_ok=True)

        # Handle the case where datasets_test is a list of TimedDataset
        if isinstance(self.datasets_test, list):
            datasets_to_save = self.datasets_test
        else:
            datasets_to_save = [self.datasets_test]

        for dataset in datasets_to_save:
            # Check if dataset is a TimedDataset
            if hasattr(dataset, "get_time") and hasattr(dataset, "get_all"):
                time_value = dataset.get_time()
                # Format time value to avoid issues with float precision in filename
                time_str = f"{time_value:.6f}".replace(".", "_")
                filename = f"test_dataset_time_{time_str}.pt"
                filepath = os.path.join(datasets_test_dir, filename)

                # Save the dataset data and time
                dataset_dict = {
                    "data": dataset.get_all(),
                    "time": time_value,
                    "length": len(dataset),
                }

                torch.save(dataset_dict, filepath)
                print(f"Saved test dataset for time {time_value} to {filepath}")
            else:
                print(
                    f"Warning: Dataset {dataset} is not a TimedDataset, skipping save."
                )

    def save_networks(self, net_dict, direction_tosample: str, outer_iter_idx: int):
        base_dir = os.path.join(self.args.experiment_dir, self.args.experiment_name)
        weights_dir = os.path.join(base_dir, "network_weight")
        os.makedirs(weights_dir, exist_ok=True)

        # Remove only previous checkpoint for the same direction
        last_ckpt_pattern = f"last_checkpoint_{direction_tosample}.pt"
        old_ckpts = glob.glob(os.path.join(weights_dir, last_ckpt_pattern))
        for file_path in old_ckpts:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint for {direction_tosample}: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

        # Convert RNG state to serializable format
        torch_rng = torch.get_rng_state().cpu().tolist()
        cuda_rng = (
            [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )

        ckpt = {
            "state_dict": net_dict[direction_tosample].state_dict(),
            "optimizer_state": self.optimizer[direction_tosample].state_dict(),
            "outer_iter_idx": outer_iter_idx,
            "direction_tosample": direction_tosample,
            "rng_state": {
                "torch": torch_rng,
                "cuda": cuda_rng,
                "random": random.getstate(),
                "numpy": np.random.get_state(),
            },
        }

        last_ckpt_path = os.path.join(
            weights_dir, f"last_checkpoint_{direction_tosample}.pt"
        )
        archive_path = os.path.join(
            weights_dir, f"{outer_iter_idx:04d}_{direction_tosample}.pth"
        )

        torch.save(ckpt, last_ckpt_path)
        torch.save(ckpt, archive_path)

        print(f"Saved new checkpoint for {direction_tosample}: {last_ckpt_path}")
        print(f"Archived checkpoint at: {archive_path}")

        # Save meta information to track last direction
        meta_path = os.path.join(weights_dir, "latest_checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "last_saved_direction": direction_tosample,
                    "outer_iter_idx": outer_iter_idx,
                },
                f,
            )
        return last_ckpt_path, archive_path

    def orchestrate_experiment(
        self,
        args,
        outer_iter_idx,
        direction_to_train,
        net_dict,
        loss_curve,
        grad_curve,
    ):
        if self.args.separation_train_test:
            datasets_inference = self.datasets_test
            print("MODE EVAL WITH DATASET_TEST")

        else:
            datasets_inference = self.datasets_train
            print("MODE EVAL WITH DATASET_TRAIN")

        # ───── Log average loss and gradient per direction
        self.tracking_logger.log(
            {
                f"loss/{direction_to_train}": float(np.mean(loss_curve))
                if loss_curve
                else None,
                f"grad/{direction_to_train}": float(np.mean(grad_curve))
                if grad_curve
                else None,
                "epoch": outer_iter_idx,
            },
            step=outer_iter_idx,
        )

        # ───── Determine what should be executed this iteration
        do_plot = args.plot_vis
        do_swd = args.display_swd and outer_iter_idx % args.display_swd_n_epoch == 0
        do_mmd = args.display_mmd and outer_iter_idx % args.display_mmd_n_epoch == 0
        do_energy = (
            args.display_energy and outer_iter_idx % args.display_energy_n_epoch == 0
        )
        do_save = (
            args.save_networks and outer_iter_idx % args.save_networks_n_epoch == 0
        )

        # ───── Only run inference if needed
        if do_plot or do_swd or do_mmd or do_energy:
            datasets_for_generation = self.leave_out_datasets(datasets_inference)
            generated, time = self.inference_test(
                args=args,
                direction_tosample=direction_to_train,
                net_dict=net_dict,
                datasets_inference=datasets_for_generation,  # We use datasets_for_generation to keep the good dt
                outer_iter_idx=outer_iter_idx,
                num_samples=args.num_sample_metric,
                num_steps=args.num_simulation_steps,
                sigma=args.sigma_inference,
            )
        else:
            generated, time = None, None

        # ───── Trajectory plot
        if do_plot and generated is not None:
            save_path_trajectory = make_trajectory(
                args=args,
                net_dict=net_dict,
                generated=generated,
                time=time,
                direction_tosample=direction_to_train,
                dataset_train=datasets_inference,
                outer_iter_idx=outer_iter_idx,
                fps=args.fps,
                plot_traj=args.plot_traj,
                number_traj=args.number_traj,
            )

            if getattr(args, "log_wandb_traj", False):
                self.tracking_logger.log(
                    {
                        f"video/{direction_to_train}/epoch_{outer_iter_idx:04d}": wandb.Video(
                            save_path_trajectory, fps=args.fps, format="mp4"
                        )
                    },
                    step=outer_iter_idx,
                )

        # ───── SWD and WD
        if do_swd and generated is not None:
            swd_scores = evaluate_swd_over_time(
                generated=generated,
                time=time,
                datasets_inference=datasets_inference,
                direction_tosample=direction_to_train,
            )
            for t, swd in swd_scores:
                print(f"[SWD @ t={t:.2f}] = {swd:.4f}")
                if args.log_wandb_swd:
                    self.tracking_logger.log(
                        {
                            f"swd/{direction_to_train}/t={t:.2f}": swd,
                            "epoch": outer_iter_idx,
                        },
                        step=outer_iter_idx,
                    )

            if args.dim <= 3:
                wd_scores = evaluate_wd_over_time(
                    generated=generated,
                    time=time,
                    datasets_inference=datasets_inference,
                    direction_tosample=direction_to_train,
                )
                for t, wd in wd_scores:
                    print(f"[WD @ t={t:.2f}] = {wd:.4f}")
                    if args.log_wandb_swd:
                        self.tracking_logger.log(
                            {
                                f"wd/{direction_to_train}/t={t:.2f}": wd,
                                "epoch": outer_iter_idx,
                            },
                            step=outer_iter_idx,
                        )

        # ───── MMD
        if do_mmd and generated is not None:
            mmd_scores = evaluate_mmd_over_time(
                generated=generated,
                time=time,
                datasets_inference=datasets_inference,
                direction_tosample=direction_to_train,
                kernel_type=args.mmd_kernel,
            )
            for t, mmd in mmd_scores:
                print(f"[MMD @ t={t:.2f}] = {mmd:.4f}")
                if args.log_wandb_mmd:
                    self.tracking_logger.log(
                        {
                            f"mmd/{direction_to_train}/t={t:.2f}": mmd,
                            "epoch": outer_iter_idx,
                        },
                        step=outer_iter_idx,
                    )

        # ───── Energy
        if do_energy and generated is not None:
            energy_scores = evaluate_energy_over_time(
                generated=generated,
                time=time,
            )
            for t, energy in energy_scores:
                print(f"[ENERGY @ t={t:.2f}] = {energy:.4f}")
                if args.log_wandb_energy:
                    self.tracking_logger.log(
                        {
                            f"energy/{direction_to_train}/t={t:.2f}": energy,
                            "epoch": outer_iter_idx,
                        },
                        step=outer_iter_idx,
                    )

        # ───── Save networks
        if do_save:
            _, archive_path = (
                self.save_networks(  # TODO maybe save also current step and not just EMA ?
                    net_dict, direction_to_train, outer_iter_idx
                )
            )
            self.tracking_logger.run.save(archive_path)
