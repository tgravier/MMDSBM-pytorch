# bridge/train_runner.py

from bridge.nbridges import N_Bridges
from utils.tracking_logger import WandbLogger
from conf.conf_loader import load_config
from bridge.models.networks import ScoreNetwork, ScoreNetworkResNet, print_trainable_params
from bridge.runners.ema import EMA

import torch
import torch.nn as nn
import os
import random
import numpy as np
import pickle
import json


class trainer_bridges(N_Bridges):
    def __init__(
        self, config_classes, tracking_logger, logger, resume_train: bool = False, inference = False,
    ):
        self.resume_train = resume_train
        self.tracking_logger = tracking_logger # WanDB
        self.logger = logger

        # Load and store config
        self.experiment_config = config_classes
        self.distribution_config = self.experiment_config.distribution_cfg

        self.instance_gpu_config()

        net_fwd, net_bwd = self.instance_network(
            model_name = self.experiment_config.model_name,
            net_fwd_layers=self.experiment_config.net_fwd_layers,
            net_fwd_time_dim=self.experiment_config.net_fwd_time_dim,
            net_bwd_layers=self.experiment_config.net_bwd_layers,
            net_bwd_time_dim=self.experiment_config.net_bwd_time_dim,
        )

       
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd



        if self.experiment_config.ema:

            self.net_fwd_ema, self.net_bwd_ema = self.instance_ema_config()

        optimizer = self.instance_optimizer(net_fwd, net_bwd)
        self.optimizer = optimizer

        

        # === Initialiser les flags pour la reprise
        self.set_resume_flags()

        # Instanciation de N_Bridges
        super().__init__(
            args=self.experiment_config,
            net_fwd=net_fwd,
            net_bwd=net_bwd,
            net_fwd_ema = self.net_fwd_ema,
            net_bwd_ema = self.net_bwd_ema,
            optimizer=optimizer,
            n_distribution=self.experiment_config.n_distributions,
            distributions=self.distribution_config.distributions,
            tracking_logger = self.tracking_logger,
            inference = inference
        )

        if not self.resume_train and not inference:
            self.launch_experiment()
        elif self.resume_train:
            self.load_and_resume_training()

    def set_resume_flags(self):
        # Inject flags in experiment_config (self.args)
        self.experiment_config.resume_train = self.resume_train
        if not hasattr(self.experiment_config, "resume_train_nb_outer_iterations"):
            self.experiment_config.resume_train_nb_outer_iterations = 0
        if not hasattr(self.experiment_config, "previous_direction_to_train"):
            self.experiment_config.previous_direction_to_train = "forward"  # default

    def instance_network(
        self, model_name, net_fwd_layers, net_fwd_time_dim, net_bwd_layers, net_bwd_time_dim
    ):  
        
        input_dim = self.experiment_config.dim
        max_time = max(
            distribution.time
            for distribution in self.distribution_config.distributions
        )



        activation = nn.SiLU()
        if model_name == "mlp":

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
        elif model_name == "resnet":

            net_fwd  = ScoreNetworkResNet(
                input_dim=input_dim,
                hidden_dim=128,
                activation_fn=activation,
                num_blocks=5,
                time_dim=16, 
                output_dim=input_dim,
                max_time=max_time,
            )

            net_bwd  = ScoreNetworkResNet(
                input_dim=input_dim,
                hidden_dim=128,
                num_blocks=5,
                activation_fn=activation,
                time_dim=16,
                output_dim=input_dim,  
                max_time=max_time,
            )

        print_trainable_params(net_fwd, "net_fwd")
        print_trainable_params(net_bwd, "net_bwd")



        return net_fwd, net_bwd

    def instance_optimizer(self, net_fwd, net_bwd):
        lr = self.experiment_config.lr
        opt_type = self.experiment_config.optimizer_type.lower()
        opt_params = self.experiment_config.optimizer_params
        optimizer_args = {"lr": lr, **opt_params}

        if opt_type == "adam":
            optimizer_cls = torch.optim.Adam
        elif opt_type == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        return {
            "forward": optimizer_cls(net_fwd.parameters(), **optimizer_args),
            "backward": optimizer_cls(net_bwd.parameters(), **optimizer_args),
        }

    def instance_gpu_config(self):
        gpu_id = self.experiment_config.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("Using device:", self.experiment_config.accelerator.device)
    
    def instance_ema_config(self):

        net_fwd_ema = EMA(self.net_fwd, decay = self.experiment_config.decay_ema)
        net_bwd_ema = EMA(self.net_fwd, decay = self.experiment_config.decay_ema)

        return net_fwd_ema, net_bwd_ema

    def launch_experiment(self):
        super().train()

    def load_and_resume_training(self):
        print("=> Attempting to resume training with both directions...")

        weights_dir = os.path.join(
            self.experiment_config.experiment_dir,
            self.experiment_config.experiment_name,
            "network_weight",
        )

        directions = ["forward", "backward"]
        checkpoints = {}

        # === Load available checkpoints (unsafe but practical way)
        for direction in directions:
            checkpoint_path = os.path.join(
                weights_dir, f"last_checkpoint_{direction}.pt"
            )
            if os.path.isfile(checkpoint_path):
                print(f"Found checkpoint for '{direction}' at {checkpoint_path}")
                checkpoints[direction] = torch.load(
                    checkpoint_path,
                    map_location=self.experiment_config.accelerator.device,
                )
            else:
                print(
                    f"WARNING: No checkpoint found for '{direction}' at {checkpoint_path}"
                )

        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in the weight directory.")

        # === Load networks and optimizers
        if "forward" in checkpoints:
            self.net_fwd.load_state_dict(checkpoints["forward"]["state_dict"])
            self.optimizer["forward"].load_state_dict(
                checkpoints["forward"]["optimizer_state"]
            )
        if "backward" in checkpoints:
            self.net_bwd.load_state_dict(checkpoints["backward"]["state_dict"])
            self.optimizer["backward"].load_state_dict(
                checkpoints["backward"]["optimizer_state"]
            )

        # === Load meta info to know which was last saved
        meta_path = os.path.join(weights_dir, "latest_checkpoint_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            prev_direction = meta["last_saved_direction"]
            last_iter = meta["outer_iter_idx"]
            print(f"=> Meta: last direction was '{prev_direction}', epoch {last_iter}")
        else:
            # fallback: use forward if exists
            prev_direction = "forward" if "forward" in checkpoints else "backward"
            last_iter = checkpoints[prev_direction]["outer_iter_idx"]

        # === Restore RNG
        rng_state = checkpoints[prev_direction]["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        random.setstate(rng_state["random"])
        np.random.set_state(rng_state["numpy"])

        # === Set resume values
        self.experiment_config.resume_train_nb_outer_iterations = (
            last_iter + 1 if prev_direction == "backward" else last_iter
        )
        self.experiment_config.previous_direction_to_train = (
            "backward" if prev_direction == "forward" else "forward"
        )

        print(
            f"=> Resuming from outer_iter_idx {self.experiment_config.resume_train_nb_outer_iterations}"
        )
        print(
            f"=> Switching direction to: {self.experiment_config.previous_direction_to_train}"
        )

        # Start training loop
        self.launch_experiment()

