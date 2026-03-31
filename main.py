# main.py

import os
import argparse
from bridge.runners.train_runner import trainer_bridges
from bridge.runners.inference_runner import inference_bridges
from utils.tracking_logger import WandbLogger
from conf.conf_loader import load_config, export_config_dict
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Launch Schrödinger Bridge training")
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration module in conf/conf_classes/ (e.g. 'experiment1')",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="Path to existing experiment directory for resume_train or inference (e.g. 'experiments/exp1')",
    )
    parser.add_argument(
        "--resume_train",
        action="store_true",
        help="Flag to resume training from the last checkpoint",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Flag to run inference instead of training",
    )
    parser.add_argument(
        "--weight_epoch",
        type=int,
        help="Epoch number to load for inference (e.g. 25 for 0025_forward.pth)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of inference runs to compute metrics with uncertainty (default: 1)",
    )
    # CLI overrides for sweeps
    parser.add_argument("--sigma", type=float, help="Override sigma value")
    parser.add_argument(
        "--leave_out",
        type=int,
        nargs="+",
        help="Override leave_out_list (e.g. --leave_out 1 3)",
    )
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument(
        "--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES (e.g. --gpu 0)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override the experiment name from the config",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Safety check
    if args.inference:
        if not args.experiment_path:
            raise ValueError(
                "When using --inference, you must provide --experiment_path"
            )
        if not args.weight_epoch:
            raise ValueError("When using --inference, you must provide --weight_epoch")
        config_classes = load_config(
            experiment_path=args.experiment_path, inference=True
        )
    elif args.resume_train:
        if not args.experiment_path:
            raise ValueError(
                "When using --resume_train, you must provide --experiment_path"
            )
        config_classes = load_config(
            experiment_path=args.experiment_path, resume_train=True
        )
    else:
        if not args.config:
            raise ValueError(
                "You must provide --config when not using --resume_train or --inference"
            )

        # Build CLI overrides dict to pass to load_config
        cli_overrides = {}
        if args.sigma is not None:
            cli_overrides["sigma"] = args.sigma
            cli_overrides["sigma_inference"] = args.sigma
        if args.leave_out is not None:
            cli_overrides["leave_out_list"] = args.leave_out
        if args.seed is not None:
            cli_overrides["seed"] = args.seed
        if args.experiment_name is not None:
            cli_overrides["experiment_name"] = args.experiment_name

        config_classes = load_config(
            config_name=args.config,
            resume_train=False,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

    # Setup logger
    logger = setup_logger(config_classes)
    logger.info("Logger initialized successfully.")
    logger.info(f"Experiment: {config_classes.experiment_name}")

    tracking_logger = WandbLogger(
        config=export_config_dict(config_classes=config_classes),
        project=config_classes.project_name,
        run_name=config_classes.experiment_name,
    )

    if args.inference:
        # Run inference
        logger.info("Starting inference mode...")
        inference_bridges(
            config_classes=config_classes,
            tracking_logger=tracking_logger,
            logger=logger,
            experiment_path=args.experiment_path,
            weight_epoch=args.weight_epoch,
            n_runs=args.n_runs,
        )
    else:
        # Run training
        logger.info("Starting training mode...")
        trainer_bridges(
            config_classes=config_classes,
            tracking_logger=tracking_logger,
            logger=logger,
            resume_train=args.resume_train,
        )


if __name__ == "__main__":
    main()
