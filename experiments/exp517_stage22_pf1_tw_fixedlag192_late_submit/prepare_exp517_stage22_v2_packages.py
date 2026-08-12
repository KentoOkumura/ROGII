from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_NAME = "exp517_stage22_pf1_tw_fixedlag192_late_submit"
EXP_DIR = ROOT / "experiments" / EXP_NAME
TRAIN_SOURCE = EXP_DIR / f"{EXP_NAME}_stage22_v2_compact_selfcontained_train.ipynb"
INFERENCE_SOURCE = EXP_DIR / f"{EXP_NAME}_stage22_v2_compact_selfcontained_inference.ipynb"
CONFIG = EXP_DIR / "config.yaml"
TRAIN_DIR = EXP_DIR / "kaggle" / "train_v2"
INFERENCE_DIR = EXP_DIR / "kaggle" / "inference_v2"
TRAIN_NOTEBOOK = TRAIN_DIR / f"{EXP_NAME}_stage22_v2_train.ipynb"
INFERENCE_NOTEBOOK = INFERENCE_DIR / f"{EXP_NAME}_stage22_v2_inference.ipynb"


PROJECT = '''competition:
  name: ROGII - Wellbore Geology Prediction
  platform: kaggle
  slug: rogii-wellbore-geology-prediction
  is_code_competition: true

paths:
  data_dir: data
  experiments_dir: experiments
  docs_dir: docs
  submissions_file: SUBMISSIONS.md

data:
  raw_dir: data/raw
  train_dir: data/raw/train
  test_dir: data/raw/test
  processed_dir: data/processed
  target_column: TVT
  group_column: well
  score_rows: TVT_input_isna

defaults:
  seed: 42
  metric: rmse
  primary_validation: GroupKFold by well; score only rows where TVT_input is NaN
  n_folds: 5

submission:
  sample_file: data/raw/sample_submission.csv
  output_file: submission.csv
  id_column: id
  target_columns:
    - tvt
  allow_extra_columns: false

metadata:
  owner: kentookumura
  notes: "exp517 corrected stage2-2 five-PF fixed-lag-192 public tabular reconstruction"

runtime:
  kaggle:
    enable_gpu: true
    enable_internet: false
    time_limit_hours: 12
'''


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for source in [TRAIN_SOURCE, INFERENCE_SOURCE, CONFIG]:
        if not source.is_file():
            raise FileNotFoundError(source)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_NOTEBOOK.write_bytes(TRAIN_SOURCE.read_bytes())
    INFERENCE_NOTEBOOK.write_bytes(INFERENCE_SOURCE.read_bytes())
    for directory in [TRAIN_DIR, INFERENCE_DIR]:
        (directory / "config.yaml").write_bytes(CONFIG.read_bytes())
        (directory / "project.yml").write_text(PROJECT, encoding="utf-8")

    train_metadata = {
        "id": "kentookumura/exp517-stage22-5pf-fl192-tab-train",
        "title": "exp517 stage22 5pf fl192 tab train",
        "code_file": TRAIN_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "run_on_push": True,
        "dataset_sources": ["ravaghi/wellbore-geology-prediction-artifacts"],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    inference_metadata = {
        "id": "kentookumura/exp517-stage22-5pf-fl192-tab-infer",
        "title": "exp517 stage22 5pf fl192 tab infer",
        "code_file": INFERENCE_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "run_on_push": True,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": ["kentookumura/exp517-stage22-5pf-fl192-tab-train"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TRAIN_DIR / "kernel-metadata.json").write_text(
        json.dumps(train_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (INFERENCE_DIR / "kernel-metadata.json").write_text(
        json.dumps(inference_metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": EXP_NAME,
        "implementation_version": "stage22_corrected_v2",
        "train_notebook_sha256": sha256_path(TRAIN_NOTEBOOK),
        "inference_notebook_sha256": sha256_path(INFERENCE_NOTEBOOK),
        "config_sha256": sha256_path(CONFIG),
        "execution_count": {
            "scientific_variants": 1,
            "pf_banks": 5,
            "representations": 1,
            "lightgbm_configs": 3,
            "catboost_configs": 2,
            "folds": 5,
            "base_models": 25,
            "ridge_models": 5,
            "control_reruns": 0,
        },
    }
    for directory in [TRAIN_DIR, INFERENCE_DIR]:
        (directory / "component_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
