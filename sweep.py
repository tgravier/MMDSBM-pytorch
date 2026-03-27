#!/usr/bin/env python
"""
Minimal sweep script for hyperparameter grid search.

Usage:
    python sweep.py --config multi_dim100 --sigmas 0.7 0.8 0.9 1.0 --n-seeds 3 --dry-run
    python sweep.py --config multi_dim100 --sigmas 0.7 0.8 0.9 1.0 --n-seeds 3
    python sweep.py --config multi_dim100 --sigmas 0.7 0.8 0.9 1.0 --n-seeds 3 --parallel 4
    python sweep.py --config phate_dim100_loo --sigmas 0.1 0.2 0.5 1.0 --n-seeds 3 --no-leave-out --dry-run
"""

import argparse
import random
import subprocess
import time
from itertools import product
from pathlib import Path

LOG_DIR = Path("sweep_logs")


def run_cmd(cmd_with_idx):
    """Run a single command and log output to file. At module level for pickling."""
    idx, cmd = cmd_with_idx
    # Parse cmd to create unique log filename
    parts = cmd.split()
    config = sigma = leave_out = seed = "unknown"
    for i, p in enumerate(parts):
        if p == "--config" and i + 1 < len(parts):
            config = parts[i + 1]
        elif p == "--sigma" and i + 1 < len(parts):
            sigma = parts[i + 1]
        elif p == "--leave_out" and i + 1 < len(parts):
            leave_out = parts[i + 1]
        elif p == "--seed" and i + 1 < len(parts):
            seed = parts[i + 1]

    log_file = LOG_DIR / f"{config}_sigma{sigma}_loo{leave_out}_seed{seed}.log"

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    return idx, cmd, result.returncode, str(log_file)


def generate_commands(config: str, sigmas: list, n_seeds: int, leave_outs=None) -> list:
    """Generate all sweep commands."""
    commands = []
    if leave_outs:
        for sigma, leave_out in product(sigmas, leave_outs):
            for _ in range(n_seeds):
                seed = random.randint(0, 99999)
                cmd = f"python main.py --config {config} --sigma {sigma} --leave_out {leave_out} --seed {seed}"
                commands.append(cmd)
    else:
        # No leave_out variation (leave_out_list is fixed in the config)
        for sigma in sigmas:
            for _ in range(n_seeds):
                seed = random.randint(0, 99999)
                cmd = f"python main.py --config {config} --sigma {sigma} --seed {seed}"
                commands.append(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(description="Sweep over hyperparameters")
    parser.add_argument("--config", type=str, required=True, help="Base config name")
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        dest="dry_run",
        help="Print commands without running",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        required=True,
        help="Sigma values to sweep (e.g. --sigmas 0.1 0.2 0.5 1.0)",
    )
    parser.add_argument(
        "--no-leave-out",
        action="store_true",
        dest="no_leave_out",
        help="Skip --leave_out in commands (use leave_out_list from config)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=True,
        help="Number of random seeds per sigma combo",
    )
    parser.add_argument(
        "--leave-outs",
        type=int,
        nargs="+",
        help="Leave-out indices to sweep over (e.g. --leave-outs 1 2). Omit to use config's leave_out_list.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="+",
        default=None,
        help="GPU(s) to use, round-robin across jobs (e.g. --gpu 0 1 2)",
    )
    args = parser.parse_args()

    leave_outs = (
        args.leave_outs if args.leave_outs else ([] if args.no_leave_out else None)
    )
    if leave_outs is None:
        parser.error("Provide --leave-outs or --no-leave-out")
    commands = generate_commands(
        args.config, sigmas=args.sigmas, n_seeds=args.n_seeds, leave_outs=leave_outs
    )

    # Assign GPUs round-robin
    gpus = args.gpu
    if gpus:
        commands = [
            f"{cmd} --gpu {gpus[i % len(gpus)]}" for i, cmd in enumerate(commands)
        ]
    print(f"Generated {len(commands)} runs")
    print(f"  Sigmas: {args.sigmas}")
    if leave_outs:
        print(f"  Leave-outs: {leave_outs}")
    else:
        print("  Leave-outs: from config (--no-leave-out)")
    print(f"  Seeds per combo: {args.n_seeds}")
    print()

    if args.dry_run:
        print("Commands to run:")
        for i, cmd in enumerate(commands, 1):
            print(f"  [{i:3d}] {cmd}")
        return

    # Run commands
    if args.parallel == 1:
        # Sequential
        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] {cmd}")
            subprocess.run(cmd, shell=True)
    else:
        # Parallel with pool of size N
        # Output goes to log files, not terminal
        from concurrent.futures import ProcessPoolExecutor, as_completed

        LOG_DIR.mkdir(exist_ok=True)

        print(f"Running {len(commands)} jobs with pool size {args.parallel}...")
        print(f"Logs in: {LOG_DIR.absolute()}")

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for i, cmd in enumerate(commands, 1):
                futures[executor.submit(run_cmd, (i, cmd))] = cmd
                print(f"  [{i}/{len(commands)}] Submitted '{cmd}'")
                time.sleep(30)  # avoids all jobs hitting the same things at once

            failed = []
            for future in as_completed(futures):
                idx, cmd, returncode, log_file = future.result()
                if returncode != 0:
                    print(
                        f"[{idx}/{len(commands)}] FAILED (exit {returncode}) - see {log_file}"
                    )
                    failed.append((idx, cmd, log_file))
                else:
                    print(f"[{idx}/{len(commands)}] OK")

        print(f"\nAll {len(commands)} jobs completed. {len(failed)} failed.")
        if failed:
            print("Failed jobs:")
            for idx, cmd, log_file in failed:
                print(f"  [{idx}] {log_file}")


if __name__ == "__main__":
    main()
