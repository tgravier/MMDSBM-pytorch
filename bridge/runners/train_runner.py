# bridge/train_runner.py

from bridge.nbridges import N_Bridges
from utils.tracking_logger import WandbLogger
from conf.conf_loader import load_config
import torch
from bridge.models.networks import ScoreNetwork
import torch.nn as nn
import os

class trainer_bridges(N_Bridges):
    def __init__(self, config_classes, tracking_logger, logger):
        # Load and store config
        self.experiment_config = config_classes
        self.distribution_config = self.experiment_config.distribution_cfg

        self.instance_gpu_config()

        net_fwd, net_bwd = self.instance_network(
            net_fwd_layers=self.experiment_config.net_fwd_layers,
            net_fwd_time_dim=self.experiment_config.net_fwd_time_dim,
            net_bwd_layers=self.experiment_config.net_bwd_layers,
            net_bwd_time_dim=self.experiment_config.net_bwd_time_dim,
        )

        # instanciation optimizer

        optimizer = self.instance_optimizer(net_fwd, net_bwd)

        # Instanciation of N_Bridges classes
        super().__init__(
            args=self.experiment_config,
            net_fwd=net_fwd,
            net_bwd=net_bwd,
            optimizer=optimizer,
            n_distribution=self.experiment_config.n_distributions,
            distributions_train=self.distribution_config.distributions_train,
        )

        self.launch_experiment()


    def instance_network(
        self, net_fwd_layers, net_fwd_time_dim, net_bwd_layers, net_bwd_time_dim
    ):
        """
        Instantiate forward and backward ScoreNetworks using parameters from self.config_class.
        Returns:
            net_fwd, net_bwd
        """
        input_dim = self.experiment_config.dim
        max_time = max(
            distribution.time
            for distribution in self.distribution_config.distributions_train
        )

        activation = nn.SiLU()  # TODO change to modulable activation function

        net_fwd = ScoreNetwork(
            input_dim=input_dim,
            layers_widths=net_fwd_layers + [input_dim],
            activation_fn=activation,
            time_dim=net_fwd_time_dim,
            max_time=max_time,
        )

        net_bwd = ScoreNetwork(
            input_dim=input_dim,
            layers_widths=net_bwd_layers + [input_dim],
            activation_fn=activation,
            time_dim=net_bwd_time_dim,
            max_time=max_time,
        )

        return net_fwd, net_bwd

    def instance_optimizer(self, net_fwd, net_bwd):
        """
        Instantiate optimizers for forward and backward networks based on config.
        Returns:
            dict with 'forward' and 'backward' optimizers.
        """
        lr = self.experiment_config.lr
        opt_type = self.experiment_config.optimizer_type.lower()
        opt_params = self.experiment_config.optimizer_params

        # Combine lr into optimizer parameters
        optimizer_args = {"lr": lr, **opt_params}

        if opt_type == "adam":
            optimizer_cls = torch.optim.Adam
        elif opt_type == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        optimizer = {
            "forward": optimizer_cls(net_fwd.parameters(), **optimizer_args),
            "backward": optimizer_cls(net_bwd.parameters(), **optimizer_args),
        }

        return optimizer
    
    def instance_gpu_config(self):

        gpu_id = self.experiment_config.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("Using device:", self.experiment_config.accelerator.device)

    def launch_experiment(self):
        super().train()
