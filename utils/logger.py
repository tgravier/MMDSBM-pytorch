import logging
import sys
import os
from typing import Optional

"""

This file is use to setup the general usage for the different logger

"""

def setup_logger(run_id: Optional[str] = None, base_log_dir: str = "logs") -> logging.Logger:


    """
    
    Create a logger plug and play , with run_id to create a specific log files

    """

    # Create the log folder if he don't exist
    os.makedirs(base_log_dir, exist_ok = True)

    logger = logging.getLogger("project")
    logger.setLevel(logging.DEBUG)

    # if no handler is activate for the next message:

    if not logger.handlers:

        # Log Message Format

        formatter = logging.Formatter(
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"

        )

        # Handler console

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler general file

        general_log_path = os.path.join(base_log_dir, "project.log")
        general_file_handler = logging.FileHandler(general_log_path)
        general_file_handler.setFormatter(formatter)
        logger.addHandler(general_file_handler)

        # Handler specific run log file

        if run_id is not None:

            run_log_path = os.path.join(base_log_dir, f"run_{run_id}.log")
            run_file_handler = logging.FIleHandler(run_log_path)
            run_file_handler.setFormatter(formatter)
            logger.addHandler(run_file_handler)

        return logger


