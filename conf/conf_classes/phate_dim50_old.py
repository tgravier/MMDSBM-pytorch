from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, GaussianMixtureConfig, PhateFromTrajectoryConfig


from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, CircleConfig


# Commentary :


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_PHATE"
        self.experiment_dir = "experiments"
        self.experiment_name = "dim50_norms2_01"

        # ───── Data Parameters
        self.dim = 50
        self.batch_size = 512
        self.n_distributions = 5
        self.separation_train_test = True
        self.nb_points_test = 1000
        self.leave_out_list = []

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters

        self.first_coupling = "ind"
        self.sigma = 0.7
        self.num_simulation_steps = 80*4
        self.nb_inner_opt_steps = 10000
        self.nb_outer_iterations = 10
        self.eps = 1e-3

        # ───── EMA Parameters

        self.ema = True
        self.decay_ema = 0.999
        # Warmup epoch

        self.warmup = True
        self.warmup_nb_inner_opt_steps = 5000
        self.warmup_epoch = 0
        # ───── Optimization
        self.lr = 2e-4
        self.grad_clip = 20
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ----  Network : General

        self.model_name = "resnet"

        # ───── Network: Forward score model
        self.net_fwd_layers = [256, 256,]
        self.net_fwd_time_dim = 128

        # ───── Network: Backward score model
        self.net_bwd_layers = [256, 256]
        self.net_bwd_time_dim = 128

        # ----- Inference

        self.sigma_inference = self.sigma
        self.num_sample_metric = 1000
        # ───── Visualisation
        self.fps = 20

        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 1024
        self.plot_traj = False
        self.number_traj = 20

        # ───── Metric

        self.log_wandb_loss = True

        self.display_swd = True
        self.log_wandb_swd = True
        self.display_swd_n_epoch = 1

    
        self.display_mmd = True
        self.log_wandb_mmd = True
        self.display_mmd_n_epoch = 1
        self.mmd_kernel = "rbf"  # Options: "gaussian", "laplacian", "energy", "rbf"
        self.mmd_blur = 1.0 
        
        self.display_energy = False
        self.log_wandb_energy = True
        self.display_energy_n_epoch = 1


        # ───── Save Networks

        self.save_networks = True
        self.save_networks_n_epoch = 1

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = False





class DistributionConfig:
    def __init__(self, dim:int, n_samples: int = 2381):
        self.dim = dim


        self.distributions = [
            PhateFromTrajectoryConfig(
                time=0,
                dim=dim,
                file_path="datasets/data/phate_multi_dim/pcs_label_0.npz",
            ),

                        PhateFromTrajectoryConfig(
                time=1,
                dim=dim,
                file_path="datasets/data/phate_multi_dim/pcs_label_1.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=2,
                dim=dim,
                file_path="datasets/data/phate_multi_dim/pcs_label_2.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=3,
                dim=dim,
                file_path="datasets/data/phate_multi_dim/pcs_label_3.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=4,
                dim=dim,
                file_path="datasets/data/phate_multi_dim/pcs_label_4.npz",
            ),
        ]
