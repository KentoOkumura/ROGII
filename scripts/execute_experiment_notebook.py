from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from config_utils import ROOT

NOTEBOOK_KINDS = ("train", "inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute an experiment notebook locally for explicit smoke debugging only."
    )
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. expXXX_model")
    parser.add_argument("--notebook", choices=NOTEBOOK_KINDS, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Opt in to local notebook execution. Kaggle execution is authoritative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_local:
        raise SystemExit(
            "Local notebook execution is disabled by default. "
            "Prepare and run the Kaggle kernel instead, or pass --allow-local "
            "for an explicit smoke debug run."
        )

    experiment_dir = ROOT / "experiments" / args.experiment
    notebook_path = experiment_dir / f"{args.experiment}_{args.notebook}.ipynb"
    if not notebook_path.exists():
        raise FileNotFoundError(f"notebook not found: {notebook_path.relative_to(ROOT)}")

    env = os.environ.copy()
    env["EXPERIMENT_ALLOW_LOCAL"] = "1"
    env["EXPERIMENT_DEBUG"] = "1" if args.debug else "0"
    if args.max_wells is not None:
        env["EXPERIMENT_MAX_WELLS"] = str(args.max_wells)

    with tempfile.TemporaryDirectory(prefix=f"{args.experiment}_{args.notebook}_") as output_dir:
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook_path),
            "--output",
            notebook_path.name,
            "--output-dir",
            output_dir,
            "--ExecutePreprocessor.timeout=-1",
        ]
        subprocess.run(command, cwd=experiment_dir, env=env, check=True)


if __name__ == "__main__":
    main()
