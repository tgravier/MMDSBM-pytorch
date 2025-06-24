import logging
import sys
import os
from typing import Optional


def setup_logger(experiment_cfg) -> logging.Logger:
    """
    Create and configure a logger using experiment config.
    Logs are stored in experiments_dir/experiment_name/log/.
    """

    # Build full log path from config
    base_log_dir = os.path.join(
        experiment_cfg.experiment_dir, experiment_cfg.experiment_name, "log"
    )
    os.makedirs(base_log_dir, exist_ok=True)

    logger = logging.getLogger("project")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # General log file
        general_log_path = os.path.join(base_log_dir, "project.log")
        general_file_handler = logging.FileHandler(general_log_path)
        general_file_handler.setFormatter(formatter)
        logger.addHandler(general_file_handler)

        # Run-specific log file
        run_log_path = os.path.join(
            base_log_dir, f"run_{experiment_cfg.experiment_name}.log"
        )
        run_file_handler = logging.FileHandler(run_log_path)
        run_file_handler.setFormatter(formatter)
        logger.addHandler(run_file_handler)

    return logger
