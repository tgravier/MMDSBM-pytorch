# bridge/trainer_dsbm.py

import torch
from torch import Tensor
from typing import Tuple, Dict, List, Optional, Sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os

from utils.visualization import make_trajectory
from bridge.core_dsbm import IMF_DSBM
from datasets.datasets_registry import DatasetConfig
from datasets.datasets import load_dataset, TimedDataset
from bridge.sde.bridge_sampler import inference_sample_sde


""" In this file we implement the main class for the training of the Schrodinger Bridge framework with N constraints
"""


class N_Bridges(IMF_DSBM):
    def __init__(
        self,
        args,
        net_fwd,
        net_bwd,
        optimizer,
        n_distribution: int,
        distributions_train: Sequence[DatasetConfig],
        distributions_test: Optional[Sequence[DatasetConfig]] = None,
    ):
        self.args = args
        super().__init__(
            args=args,
            num_simulation_steps=args.num_simulation_steps,
            optimizer=optimizer,
            net_fwd=net_fwd,
            net_bwd=net_bwd,
            sig=args.sigma,
            eps=args.eps,
        )
        self._validate_config_time_unique(distributions_train, "train")

        if distributions_test is not None:
            self._validate_config_time_unique(distributions_test, "test")
            if len(distributions_test) != n_distribution:
                raise ValueError(
                    "Length of distributions_test must match n_distribution"
                )
            for d_train, d_test in zip(distributions_train, distributions_test):
                if d_train.name != d_test.name:
                    raise ValueError(
                        f"Mismatch between training and test dataset types: {d_train.name} vs {d_test.name}"
                    )

        if len(distributions_train) != n_distribution:
            raise ValueError("Length of distributions_train must match n_distribution")

        self.n_distribution = n_distribution
        self.distributions_train = distributions_train
        self.distributions_test = distributions_test

        self.datasets_train = self.prepare_dataset(distributions_train)
        self.datasets_test = (
            self.prepare_dataset(distributions_test) if distributions_test else None
        )

    def _validate_config_time_unique(
        self, distributions: List[DatasetConfig], mode: str
    ):
        times = [d.time for d in distributions]
        duplicates = {t for t in times if times.count(t) > 1 and t is not None}
        if duplicates:
            dups = [d for d in distributions if d.time in duplicates]
            raise ValueError(
                f"Duplicate time values found in {mode} distributions: {duplicates}\n"
                f"Conflicting DatasetConfigs:\n"
                + "\n".join(f"- {repr(d)}" for d in dups)
            )

    def prepare_dataset(self, distributions: List[DatasetConfig]) -> List[TimedDataset]:
        return [load_dataset(config) for config in distributions]

    def generate_dataset_pairs(
        self,
        forward_pairs: List[Tuple[TimedDataset, TimedDataset]],
    ):
        time_dataset_init = [pair[0] for pair in forward_pairs]
        time_dataset_target = [pair[1] for pair in forward_pairs]

        n_samples = (
            time_dataset_init[0].get_all().shape[0]
        )  # Assuming all datasets have the same number of samples
        # shape x0  (time, num_sample,dim)
        x0 = torch.stack(
            [
                time_dataset.get_all().to(self.args.accelerator.device)
                for time_dataset in time_dataset_init
            ]
        )
        x1 = torch.stack(
            [
                time_dataset.get_all().to(self.args.accelerator.device)
                for time_dataset in time_dataset_target
            ]
        )

        # shape of x_pairs after bellow : (n_samples * n_times, dim)

        x_pairs = torch.stack([x0, x1], dim=2)  # (n_times, n_samples, 2, dim)
        x_pairs = x_pairs.reshape(
            -1, 2, x_pairs.shape[3]
        )  # (n_times * n_samples, 2, dim)

        t_pairs_init = torch.tensor(
            [time_dataset.get_time() for time_dataset in time_dataset_init],
            device=self.args.accelerator.device,
        )  # shape: n_times

        t_pairs_target = torch.tensor(
            [time_dataset.get_time() for time_dataset in time_dataset_target],
            device=self.args.accelerator.device,
        )  # shape: n_times
        t_pairs = torch.stack(  # shape: (n_times, 2)
            [
                t_pairs_init,
                t_pairs_target,
            ],
            dim=1,
        )
        t_pairs = t_pairs.repeat_interleave(
            n_samples, dim=0
        )  # shape: (n_times * n_samples, 2)

        return x_pairs, t_pairs

    def train(self):
        for outer_iter_idx in range(self.args.nb_outer_iterations):
            print(f"\n[Epoch {outer_iter_idx}]")

            # Get forward pairs: (D1, D2), (D2, D3), ...
            forward_pairs = list(zip(self.datasets_train[:-1], self.datasets_train[1:]))

            # === FORWARD ===
            print("Training FORWARD bridge")
            x_pairs, t_pairs = self.generate_dataset_pairs(forward_pairs)

            loss_curve, net_dict = self.train_one_direction(
                direction="forward",
                x_pairs=x_pairs,
                t_pairs=t_pairs,
                outer_iter_idx=outer_iter_idx,
            )

            if self.args.dim == 2 and outer_iter_idx % self.args.plot_vis_n_epoch == 0  and outer_iter_idx != 0:

                generated, time = self.inference_test(
                    args=self.args,
                    direction_tosample="forward",
                    net_dict=net_dict,
                    dataset_train=self.datasets_train,
                    outer_iter_idx=outer_iter_idx,
                    num_samples=1000,
                    num_steps=self.args.num_simulation_steps,
                    sigma=self.args.sigma,
                )

                make_trajectory(
                    args=self.args,
                    net_dict=net_dict,
                    generated=generated,
                    time=time,
                    direction_tosample="forward",
                    dataset_train=self.datasets_train,
                    outer_iter_idx=outer_iter_idx,
                    fps=self.args.fps,
                    plot_traj=self.args.plot_traj,
                    number_traj=self.args.number_traj,
                )

            self.save_networks(
                net_dict=net_dict,
                direction_tosample="forward",
                outer_iter_idx=outer_iter_idx,
            )

            # === BACKWARD ===
            print("Training BACKWARD bridge")
            x_pairs, t_pairs = self.generate_dataset_pairs(forward_pairs)
            loss_curve, net_dict = self.train_one_direction(
                direction="backward",
                x_pairs=x_pairs,
                t_pairs=t_pairs,
                outer_iter_idx=outer_iter_idx,
            )
            if self.args.dim == 2 and outer_iter_idx % self.args.plot_vis_n_epoch == 0  and outer_iter_idx != 0:

                generated, time = self.inference_test(
                    args=self.args,
                    direction_tosample="backward",
                    net_dict=net_dict,
                    dataset_train=self.datasets_train,
                    outer_iter_idx=outer_iter_idx,
                    num_samples=1000,
                    num_steps=self.args.num_simulation_steps,
                    sigma=self.args.sigma,
                )

                make_trajectory(
                    args=self.args,
                    net_dict=net_dict,
                    generated=generated,
                    time=time,
                    direction_tosample="backward",
                    dataset_train=self.datasets_train,
                    outer_iter_idx=outer_iter_idx,
                    fps=self.args.fps,
                    plot_traj=self.args.plot_traj,
                    number_traj=self.args.number_traj,
                )

            self.save_networks(
                net_dict=net_dict,
                direction_tosample="backward",
                outer_iter_idx=outer_iter_idx,
            )

    def inference_test(
        self,
        args,
        direction_tosample: str,
        net_dict: dict,
        dataset_train: list,
        outer_iter_idx: int,
        num_samples: int = 100,
        num_steps: int = 200,
        sigma: float = 1.0,
        one_bridge: bool = False,
    ):
        device = next(net_dict[direction_tosample].parameters()).device

        # Select the dataset with the minimum time value and the maximum time value (fix the bug of permute distrib in the config)
        max_time_idx = max(
            range(len(dataset_train)), key=lambda i: float(dataset_train[i].get_time())
        )
        min_time_idx = min(
            range(len(dataset_train)), key=lambda i: float(dataset_train[i].get_time())
        )


        # === Initial sample points
        if direction_tosample == "forward":
            z0 = dataset_train[min_time_idx].get_all().to(device)
        elif direction_tosample == "backward":
            z0 = dataset_train[max_time_idx].get_all().to(device)
        else:
            raise ValueError("Invalid direction")

        idx = torch.randint(0, z0.shape[0], (num_samples,))
        z0_sampled = z0[idx]

        t_pairs = [
            dataset_train[min_time_idx].get_time(),
            dataset_train[max_time_idx].get_time(),
        ]
        if not one_bridge:
            num_steps = num_steps * (len(dataset_train) - 1)

        # === Simulate trajectory

        generated, time = inference_sample_sde(
            zstart=z0_sampled,
            net_dict=net_dict,
            t_pairs=t_pairs,
            direction_tosample=direction_tosample,
            N=num_steps,
            sig=sigma,
            device=device,
        )

        return generated, time

    def save_networks(self, net_dict, direction_tosample: str, outer_iter_idx: int):
        base_dir = os.path.join(self.args.experiment_dir, self.args.experiment_name)
        weights_dir = os.path.join(base_dir, "network_weight")
        os.makedirs(weights_dir, exist_ok=True)

        filename = f"{outer_iter_idx:04d}_{direction_tosample}.pth"
        path = os.path.join(weights_dir, filename)

        # Save state_dict
        torch.save(net_dict[direction_tosample].state_dict(), path)
        print(
            f"Saved network {direction_tosample} weights at {outer_iter_idx} to {path}"
        )
