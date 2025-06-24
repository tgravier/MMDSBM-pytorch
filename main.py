# main.py

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
        required=True,
        help="Name of the configuration module in conf/conf_classes/ (e.g. 'experiment1')",
    )
    args = parser.parse_args()

    config_classes = load_config(args.config)

    # Initialize the logger using the config
    logger = setup_logger(config_classes)

    logger.info("Logger initialized successfully.")
    logger.info(f"Experiment: {config_classes.experiment_name}")

    # Initialize the tracking logger
    tracking_logger = WandbLogger(
        config=export_config_dict(config_classes=config_classes),
        project=config_classes.project_name,
        run_name=config_classes.experiment_dir + "/" + config_classes.experiment_name,
    )

    # Initialize and run trainer
    trainer = trainer_bridges(
        config_classes=config_classes, tracking_logger=tracking_logger, logger=logger
    )
    trainer.train()


if __name__ == "__main__":
    main()
