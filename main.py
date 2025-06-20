# main.py

import argparse
from bridge.runners.train_runner import trainer_bridges


def main():
    parser = argparse.ArgumentParser(description="Launch Schrödinger Bridge training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the configuration module in conf/conf_classes/ (e.g. 'experiment1')",
    )
    args = parser.parse_args()

    # Initialize and run trainer
    trainer = trainer_bridges(conf_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
