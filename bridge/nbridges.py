#bridge/trainer_dsbm.py

import torch
from torch import Tensor
from typing import Tuple, Dict, List, Optional, Sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

from bridge.core_dsbm import IMF_DSBM
from datasets.datasets_registry import DatasetConfig
from datasets.datasets import load_dataset, TimedDataset






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
            num_simulation_steps= args.num_simulation_steps,
            optimizer=optimizer,
            net_fwd = net_fwd,
            net_bwd= net_bwd,
            sig = args.sigma,
            eps = args.eps
        )
        self._validate_config_time_unique(distributions_train, "train")

        if distributions_test is not None:
            self._validate_config_time_unique(distributions_test, "test")
            if len(distributions_test) != n_distribution:
                raise ValueError("Length of distributions_test must match n_distribution")
            for d_train, d_test in zip(distributions_train, distributions_test):
                if d_train.name != d_test.name:
                    raise ValueError(f"Mismatch between training and test dataset types: {d_train.name} vs {d_test.name}")

        if len(distributions_train) != n_distribution:
            raise ValueError("Length of distributions_train must match n_distribution")

        self.n_distribution = n_distribution
        self.distributions_train = distributions_train
        self.distributions_test = distributions_test

        self.datasets_train = self.prepare_dataset(distributions_train)
        self.datasets_test = self.prepare_dataset(distributions_test) if distributions_test else None


    def _validate_config_time_unique(self, distributions: List[DatasetConfig], mode: str):
        times = [d.time for d in distributions]
        duplicates = {t for t in times if times.count(t) > 1 and t is not None}
        if duplicates:
            dups = [d for d in distributions if d.time in duplicates]
            raise ValueError(
                f"Duplicate time values found in {mode} distributions: {duplicates}\n"
                f"Conflicting DatasetConfigs:\n" +
                "\n".join(f"- {repr(d)}" for d in dups)
            )

    def prepare_dataset(
        self,
        distributions: List[DatasetConfig]
    ) -> List[TimedDataset]:
        
        return [load_dataset(config) for config in distributions]

    
    def generate_dataset_pairs(
            self,
            time_dataset_init:TimedDataset,
            time_dataset_target:TimedDataset ,

    ) -> Tuple[Tensor, List[float]]:
        
        time_dataset_init = time_dataset_init
        time_dataset_target = time_dataset_target

        x0 = time_dataset_init.get_all().to(self.args.accelerator.device)
        x1 = time_dataset_target.get_all().to(self.args.accelerator.device)

        x_pairs = torch.stack([x0,x1],dim=1)
    
        return x_pairs, [time_dataset_init.get_time(), time_dataset_target.get_time()]
    
    def train(self, method: str):
        if method == "direct":


            for outer_iter_idx in range(self.args.nb_outer_iterations):
                print(f"\n[Epoch {outer_iter_idx}]")

                # Get forward pairs: (D1, D2), (D2, D3), ...
                forward_pairs = list(zip(self.datasets_train[:-1], self.datasets_train[1:]))

                # === FORWARD ===
                for num_bridges, (dataset_init, dataset_target) in enumerate(forward_pairs):
                    print(f"Training FORWARD bridge {num_bridges} from t={dataset_init.get_time()} to t={dataset_target.get_time()}")
                    x_pairs, t_pairs = self.generate_dataset_pairs(dataset_init, dataset_target)

                    self.train_one_direction(
                        direction="forward",
                        num_bridges=num_bridges,
                        x_pairs=x_pairs,
                        t_pairs=t_pairs,
                        outer_iter_idx=outer_iter_idx
                    )

                # === BACKWARD ===
                for num_bridges, (dataset_init, dataset_target) in enumerate(forward_pairs):  # inverse pairs
                    print(f"Training BACKWARD bridge {num_bridges} from t={dataset_init.get_time()} to t={dataset_target.get_time()}")
                    x_pairs, t_pairs = self.generate_dataset_pairs(dataset_init, dataset_target)
                    self.train_one_direction(
                        direction="backward",
                        num_bridges=num_bridges,
                        x_pairs=x_pairs,
                        t_pairs=t_pairs,
                        outer_iter_idx=outer_iter_idx
                    )

        elif method == 'bouncing':
            for outer_iter_idx in range(self.args.nb_outer_iterations):

                # Get forward pairs: (D1, D2), (D2, D3), ...
                forward_pairs = list(zip(self.datasets_train[:-1], self.datasets_train[1:]))

                for num_bridges, (dataset_init, dataset_target) in enumerate(forward_pairs):
                    print(f"Training FORWARD bridge {num_bridges} from t={dataset_init.get_time()} to t={dataset_target.get_time()}")
                    x_pairs, t_pairs = self.generate_dataset_pairs(dataset_init, dataset_target)

                    self.train_one_direction(
                        direction="forward",
                        num_bridges=num_bridges,
                        x_pairs=x_pairs,
                        t_pairs=t_pairs,
                        outer_iter_idx=outer_iter_idx
                    )

                    print(f"Training BACKWARD bridge {num_bridges} from t={dataset_init.get_time()} to t={dataset_target.get_time()}")
                    x_pairs, t_pairs = self.generate_dataset_pairs(dataset_init, dataset_target)

                    self.train_one_direction(
                        direction="backward",
                        num_bridges=num_bridges,
                        x_pairs=x_pairs,
                        t_pairs=t_pairs,
                        outer_iter_idx=outer_iter_idx
                    )





