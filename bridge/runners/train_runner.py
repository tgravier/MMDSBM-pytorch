# bridge/train_runner.py

from bridge.nbridges import N_Bridges
from utils.tracking_logger import WandbLogger
from conf.conf_loader import load_config
from bridge.models.networks import (
    ScoreNetwork,
    ScoreNetworkFILM,
    ScoreNetworkResNet,
    print_trainable_params,
)
import types
from bridge.runners.ema import EMA
from diffusers import UNet2DModel
import torch
import torch.nn as nn
import os
import random
import numpy as np
import pickle
import json


class trainer_bridges(N_Bridges):
    def __init__(
        self,
        config_classes,
        tracking_logger,
        logger,
        resume_train: bool = False,
        inference=False,
    ):
        self.resume_train = resume_train
        self.tracking_logger = tracking_logger  # WanDB
        self.logger = logger

        # Load and store config
        self.experiment_config = config_classes
        self.distribution_config = self.experiment_config.distribution_cfg

        self.instance_gpu_config()
        if self.experiment_config.model_name == "hf_unet":
            net_fwd, net_bwd = self.instance_network(
                model_name=self.experiment_config.model_name,
            )
        else:
            net_fwd, net_bwd = self.instance_network(
                model_name=self.experiment_config.model_name,
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
            net_fwd_ema=self.net_fwd_ema,
            net_bwd_ema=self.net_bwd_ema,
            optimizer=optimizer,
            n_distribution=self.experiment_config.n_distributions,
            distributions=self.distribution_config.distributions,
            tracking_logger=self.tracking_logger,
            inference=inference,
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
        self,
        model_name,
        net_fwd_layers=None,
        net_fwd_time_dim=None,
        net_bwd_layers=None,
        net_bwd_time_dim=None,
    ):
        input_dim = self.experiment_config.dim
        max_time = max(
            distribution.time for distribution in self.distribution_config.distributions
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
            net_fwd = ScoreNetworkResNet(
                input_dim=input_dim,
                layers_widths=net_fwd_layers + [input_dim],
                activation_fn=activation,
                time_dim=net_fwd_time_dim,
                max_time=max_time,
            )

            net_bwd = ScoreNetworkResNet(
                input_dim=input_dim,
                layers_widths=net_bwd_layers + [input_dim],
                activation_fn=activation,
                time_dim=net_bwd_time_dim,
                max_time=max_time,
            )

        elif model_name == "mlp_film":
            net_fwd = ScoreNetworkFILM(
                input_dim=input_dim,
                layers_widths=net_fwd_layers,
                activation_fn=activation,
                time_dim=net_fwd_time_dim,
                max_time=max_time,
            )

            net_bwd = ScoreNetworkFILM(
                input_dim=input_dim,
                layers_widths=net_bwd_layers,
                activation_fn=activation,
                time_dim=net_bwd_time_dim,
                max_time=max_time,
            )

        elif model_name == "hf_unet":
            net_fwd = UNet2DModel(
                sample_size=self.experiment_config.sample_size,
                in_channels=self.experiment_config.nb_channels,
                out_channels=self.experiment_config.nb_channels,
                time_embedding_type=self.experiment_config.time_embedding_type,
                down_block_types=self.experiment_config.down_block_types,
                block_out_channels=self.experiment_config.block_out_channels,
                up_block_types=self.experiment_config.up_block_types,
                layers_per_block=self.experiment_config.layers_per_block,
            )

            net_bwd = UNet2DModel(
                sample_size=self.experiment_config.sample_size,
                in_channels=self.experiment_config.nb_channels,
                out_channels=self.experiment_config.nb_channels,
                time_embedding_type=self.experiment_config.time_embedding_type,
                down_block_types=self.experiment_config.down_block_types,
                block_out_channels=self.experiment_config.block_out_channels,
                up_block_types=self.experiment_config.up_block_types,
                layers_per_block=self.experiment_config.layers_per_block,
            )

            net_fwd._orig_forward = net_fwd.forward
            net_bwd._orig_forward = net_bwd.forward

            def wrapped_forward(
                self, sample, timestep, class_labels=None, return_dict=None
            ):
                out = self._orig_forward(sample, timestep.squeeze(), return_dict=False)

                return out[0]

            net_fwd.forward = types.MethodType(wrapped_forward, net_fwd)
            net_bwd.forward = types.MethodType(wrapped_forward, net_bwd)

        else:
            raise ValueError(f"Unknown model name: {model_name}")

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
        net_fwd_ema = EMA(self.net_fwd, decay=self.experiment_config.decay_ema)
        net_bwd_ema = EMA(self.net_bwd, decay=self.experiment_config.decay_ema)

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
                    weights_only=False,
                )
            else:
                print(
                    f"WARNING: No checkpoint found for '{direction}' at {checkpoint_path}"
                )

        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in the weight directory.")

        # === Load networks and optimizers
        # Checkpoints are saved from EMA models (unwrapped), but at resume
        # time the networks may already be wrapped by torch.compile via
        # Accelerator (OptimizedModule with _orig_mod. prefix).  Load into
        # the inner module when a wrapper is present.
        def _load_net(net, state_dict):
            target = getattr(net, "_orig_mod", net)
            target.load_state_dict(state_dict)

        if "forward" in checkpoints:
            _load_net(self.net_fwd, checkpoints["forward"]["state_dict"])
            self.optimizer["forward"].load_state_dict(
                checkpoints["forward"]["optimizer_state"]
            )
        if "backward" in checkpoints:
            _load_net(self.net_bwd, checkpoints["backward"]["state_dict"])
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
        torch_rng = rng_state["torch"]
        if isinstance(torch_rng, list):
            torch_rng = torch.ByteTensor(torch_rng)
        torch.set_rng_state(torch_rng)
        if torch.cuda.is_available() and rng_state["cuda"]:
            cuda_states = [
                torch.ByteTensor(s) if isinstance(s, list) else s
                for s in rng_state["cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(rng_state["random"])
        np.random.set_state(rng_state["numpy"])

        # === Set resume values
        # Each epoch trains first_direction then opposite. Determine where we stopped.
        first_dir = self.experiment_config.first_direction
        second_dir = "forward" if first_dir == "backward" else "backward"

        if prev_direction == second_dir:
            # Second half of the epoch completed → full epoch done
            resume_epoch = last_iter + 1
            resume_skip_first_half = False
        else:
            # Only first half completed → resume mid-epoch
            resume_epoch = last_iter
            resume_skip_first_half = True

        self.experiment_config.resume_train_nb_outer_iterations = resume_epoch
        self.experiment_config.resume_skip_first_half = resume_skip_first_half

        print(f"=> Resuming from outer_iter_idx {resume_epoch}")
        if resume_skip_first_half:
            print(f"=> Skipping already-completed {first_dir} half, starting with {second_dir}")
        else:
            print(f"=> Starting fresh epoch with {first_dir}")

        # Start training loop
        self.launch_experiment()
