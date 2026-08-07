from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP_NAME = "exp159_spatial_prior_confidence_features_on_exp092"
EXP = ROOT / "experiments" / EXP_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete previous exp159 outputs.",
    )
    args = parser.parse_args()
    targets = [
        EXP / "artifacts",
        EXP / "colab_runs",
    ]
    for target in targets:
        print("target", target, "exists", target.exists())
    if not args.yes:
        raise SystemExit("Pass --yes to delete these exp159 outputs.")
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print("deleted", target)
    (EXP / "colab_runs").mkdir(parents=True, exist_ok=True)
    print("clean_done", EXP)


if __name__ == "__main__":
    main()
