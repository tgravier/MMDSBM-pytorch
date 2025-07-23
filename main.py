# main.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from bridge.runners.train_runner import trainer_bridges
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
        help="Path to existing experiment directory for resume_train (e.g. 'experiments/exp1')",
    )
    parser.add_argument(
        "--resume_train",
        action="store_true",
        help="Flag to resume training from the last checkpoint",
    )
    args = parser.parse_args()

    # Safety check
    if args.resume_train:
        if not args.experiment_path:
            raise ValueError("When using --resume_train, you must provide --experiment_path")
        config_classes = load_config(experiment_path=args.experiment_path, resume_train=True)
    else:
        if not args.config:
            raise ValueError("You must provide --config when not using --resume_train")
        config_classes = load_config(config_name=args.config, resume_train=False)

    # Setup logger
    logger = setup_logger(config_classes)
    logger.info("Logger initialized successfully.")
    logger.info(f"Experiment: {config_classes.experiment_name}")

    tracking_logger = WandbLogger(
        config=export_config_dict(config_classes=config_classes),
        project=config_classes.project_name,
        run_name=config_classes.experiment_dir + "/" + config_classes.experiment_name,
    )

    trainer = trainer_bridges(
        config_classes=config_classes,
        tracking_logger=tracking_logger,
        logger=logger,
        resume_train=args.resume_train
    )
    trainer.train()



if __name__ == "__main__":
    main()
