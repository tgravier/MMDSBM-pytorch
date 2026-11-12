import os
import random

from accelerate import Accelerator
from accelerate.utils import DynamoBackend, TorchDynamoPlugin

from datasets.datasets_registry import KTHConfig


class ExperimentConfig:
    def __init__(self):
        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_KTH_v2"
        self.experiment_dir = "DSBM_N_BRIDGES_KTH_v2"
        self.experiment_name = "bigattn_5_frames_sigma_0.3"
        self.experiment_type = ""
        self.seed = random.randint(1_000, 1_000_000)

        # ───── Data Parameters
        self.dim = None  # keep image shape (C, H, W) for UNet
        self.batch_size = 64  # <------------------- here
        self.n_distributions = 5
        self.separation_train_test = True
        self.nb_points_test = 30  # <------------------- here
        self.leave_out_list = []
        self.max_pairs = None  # <------------------- here
        self.recreate_dataset = False

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(n_frames=5, dim=self.dim)

        # ───── Simulation Parameters
        self.first_direction = "forward"
        self.coeff_sigma = 1
        self.first_coupling = "ind"
        self.sigma = 0.3  # <------------------- here
        self.sigma_mode = "mono"
        self.sigma_linspace = None
        self.num_simulation_steps = 400  # <------------------- here
        self.nb_inner_opt_steps = 10_000  # <------------------- here
        self.nb_outer_iterations = 30  # <------------------- here
        self.eps = 1e-3
        self.loss_scale = True

        # ───── EMA Parameters
        self.ema = True
        self.decay_ema = 0.9999

        # Warmup epoch
        self.warmup = True
        self.warmup_nb_inner_opt_steps = 100_000  # <------------------- here
        self.warmup_epoch = 0
        self.warmup_log_video_every = 20_000

        # ───── Optimization
        self.lr = 2e-4
        self.grad_clip = 1
        self.optimizer_type = "adamw"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.01}

        # --- Network General
        self.model_name = "hf_unet"

        # ───── Network: UNet config (1 channel grayscale, 120x160)
        self.sample_size = (120, 160)
        self.nb_channels = 1
        self.time_embedding_type = "positional"
        self.down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        )
        self.up_block_types = (
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
        self.block_out_channels = (64, 128, 256)
        self.layers_per_block = 2

        # ----- Inference
        self.sigma_inference = self.sigma
        self.num_sample_metric = 10
        self.chunk_size = 64  # <------------------- here

        # ───── Visualisation
        # ~ 10 seconds videos
        self.fps = self.num_simulation_steps // 10

        self.plot_vis = False
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 64
        self.plot_traj = False
        self.number_traj = 20

        # ───── Metric
        self.rescale = False

        self.log_wandb_loss = True

        self.display_swd = False
        self.log_wandb_swd = False
        self.display_swd_n_epoch = 1

        self.display_mmd = False
        self.log_wandb_mmd = False
        self.display_mmd_n_epoch = 1
        self.mmd_kernel = "rbf"
        self.mmd_blur = 1.0

        self.display_energy = False
        self.log_wandb_energy = False
        self.display_energy_n_epoch = 1

        # ───── Save Networks
        self.save_networks = True
        self.save_networks_n_epoch = 5

        # ------- Save Generation
        self.save_generation = False  # wandb videos replace .pt trajectory saving

        # ───── GPU
        self.gpu_id = 1
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # ───── Accelerator (must come after CUDA_VISIBLE_DEVICES is set)
        self.accelerator = Accelerator(
            dynamo_plugin=TorchDynamoPlugin(
                DynamoBackend.INDUCTOR, "max-autotune", True, False
            ),
            mixed_precision="no",
        )

        # ───── Debug
        self.debug = False


class DistributionConfig:
    def __init__(self, n_frames: int, dim=None):
        self.dim = dim

        frame_dir = f"/projects/static2dynamic/datasets/KTH_Actions/{n_frames}_frames"
        print(f"Using KTH dataset at {frame_dir} ")

        self.normalize_per_image = False

        self.distributions = [
            KTHConfig(
                time=float(rank),
                frame_dir=frame_dir,
                frame_rank=rank,
                dim=dim,
                grayscale=True,
                normalize_per_image=self.normalize_per_image,
            )
            for rank in range(n_frames)
        ]
