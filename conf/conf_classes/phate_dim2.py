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
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "phate_dim2_01"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 64
        self.n_distributions = 5

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.warmup_epoch = 0
        self.first_coupling = "ind"
        self.sigma = 0.5
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 10000
        self.nb_outer_iterations = 100
        self.eps = 1e-3

        # Warmup epoch

        self.warmup = False
        self.warmup_nb_inner_opt_steps = 100000

        # ───── Optimization
        self.lr = 1e-3
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 128, ]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 128]
        self.net_bwd_time_dim = 64

        # ----- Inference

        self.sigma_inference = 0.5

        # ───── Visualisation
        self.fps = 20

        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 1000
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

        self.debug = True



class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 1000):
        self.dim = dim


        self.distributions_train = [
            PhateFromTrajectoryConfig(
                time=0,
                dim=2,
                file_path="datasets/data/phate_dim2/pcs_label_0_dim_2.npz",
            ),

                        PhateFromTrajectoryConfig(
                time=1,
                dim=2,
                file_path="datasets/data/phate_dim2/pcs_label_1_dim_2.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=2,
                dim=2,
                file_path="datasets/data/phate_dim2/pcs_label_2_dim_2.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=3,
                dim=2,
                file_path="datasets/data/phate_dim2/pcs_label_3_dim_2.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=4,
                dim=2,
                file_path="datasets/data/phate_dim2/pcs_label_4_dim_2.npz",
            ),
        ]
