from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
ARCHIVE = Path("/content/exp159_spatial_prior_confidence_features_on_exp092.tar.gz")


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    target = ROOT / "experiments"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(target)
    exp = target / "exp159_spatial_prior_confidence_features_on_exp092"
    print("extracted", exp, exp.exists(), flush=True)
    print("code", (exp / "spatial_prior_confidence_features_on_exp092.py").exists(), flush=True)
    print("prepare", (exp / "colab_prepare_inputs.py").exists(), flush=True)
    print("start", (exp / "colab_start_full_train.py").exists(), flush=True)


if __name__ == "__main__":
    main()
