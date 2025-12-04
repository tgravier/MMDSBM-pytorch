#!/usr/bin/env python
"""
Minimal sweep script for hyperparameter grid search.

Usage:
    python sweep.py --config multi_dim100 --dry_run  # Preview commands
    python sweep.py --config multi_dim100            # Run sequentially
    python sweep.py --config multi_dim100 --parallel 4  # Run 4 in parallel
"""

import argparse
import random
import subprocess
import time
from itertools import product
from pathlib import Path

# ===== SWEEP CONFIGURATION =====
SIGMAS = [round(0.1 * i, 1) for i in range(7, 11)]
LEAVE_OUTS = [1, 2]
N_SEEDS = 3
LOG_DIR = Path("sweep_logs")
# ===============================


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


def generate_commands(config: str) -> list:
    """Generate all sweep commands."""
    commands = []
    for sigma, leave_out in product(SIGMAS, LEAVE_OUTS):
        for _ in range(N_SEEDS):
            seed = random.randint(0, 99999)
            cmd = f"python main.py --config {config} --sigma {sigma} --leave_out {leave_out} --seed {seed}"
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
    args = parser.parse_args()

    commands = generate_commands(args.config)
    print(f"Generated {len(commands)} runs")
    print(f"  Sigmas: {SIGMAS}")
    print(f"  Leave-outs: {LEAVE_OUTS}")
    print(f"  Seeds per combo: {N_SEEDS}")
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
