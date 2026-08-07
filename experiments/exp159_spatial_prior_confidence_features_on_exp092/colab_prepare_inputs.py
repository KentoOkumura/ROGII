from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from lightgbm import LGBMRegressor

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP_NAME = "exp159_spatial_prior_confidence_features_on_exp092"
EXP = ROOT / "experiments" / EXP_NAME
CACHE_SRC = (
    ROOT
    / "experiments"
    / "exp072_exp063_full_replay_feature_cache"
    / "artifacts"
    / "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
SPATIAL_SRC = (
    ROOT
    / "experiments"
    / "exp114_spatial_neighbor_prior_signal_audit"
    / "kaggle"
    / "output"
    / "train_v1"
    / "artifacts"
    / "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
)
LOCAL_DIR = Path("/content/rogii_cache/exp159_inputs")
LOCAL_CACHE = LOCAL_DIR / CACHE_SRC.name
LOCAL_SPATIAL = LOCAL_DIR / SPATIAL_SRC.name


def copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print("copy_skip", dst, dst.stat().st_size, flush=True)
        return
    print("copy_start", src, "->", dst, src.stat().st_size, flush=True)
    shutil.copy2(src, dst)
    print("copy_done", dst, dst.stat().st_size, flush=True)


def main() -> None:
    print("ram_gb", round(psutil.virtual_memory().total / 1024**3, 2), flush=True)
    print("cuda", torch.cuda.is_available(), flush=True)
    print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
    print("root_exists", ROOT.exists(), flush=True)
    print("project_yml", (ROOT / "project.yml").exists(), flush=True)
    print("exp_exists", EXP.exists(), flush=True)
    print("cache_src", CACHE_SRC.exists(), CACHE_SRC.stat().st_size if CACHE_SRC.exists() else None)
    print(
        "spatial_src",
        SPATIAL_SRC.exists(),
        SPATIAL_SRC.stat().st_size if SPATIAL_SRC.exists() else None,
    )
    copy_if_needed(CACHE_SRC, LOCAL_CACHE)
    copy_if_needed(SPATIAL_SRC, LOCAL_SPATIAL)
    print("cache_preview", pd.read_csv(LOCAL_CACHE, nrows=3).shape, flush=True)
    print("spatial_preview", pd.read_csv(LOCAL_SPATIAL, nrows=3).shape, flush=True)

    x = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    y = x[:, 0] * 0.5 + np.random.default_rng(1).normal(size=200).astype(np.float32) * 0.01
    model = LGBMRegressor(
        objective="regression",
        n_estimators=20,
        learning_rate=0.1,
        device_type="gpu",
        gpu_use_dp=True,
        verbose=-1,
        seed=123,
    )
    model.fit(x, y)
    print("lightgbm_gpu_smoke_ok", int(model.best_iteration_ or 20), flush=True)


if __name__ == "__main__":
    main()
