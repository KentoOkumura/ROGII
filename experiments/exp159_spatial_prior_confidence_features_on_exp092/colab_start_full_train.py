# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP_NAME = "exp159_spatial_prior_confidence_features_on_exp092"
EXP = ROOT / "experiments" / EXP_NAME
RUN_DIR = EXP / "colab_runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)

for filename in ["latest_done_summary.json", "latest_failed.txt"]:
    path = RUN_DIR / filename
    if path.exists():
        path.unlink()

run_id = time.strftime("run_%Y%m%d_%H%M%S_l4_highmem_local_cache")
run_py = RUN_DIR / f"{run_id}_{EXP_NAME}_full_train.py"
log_path = RUN_DIR / f"{run_id}_{EXP_NAME}_full_train.log"
pid_path = RUN_DIR / f"{run_id}_pid.txt"
latest_path = RUN_DIR / "latest_run.json"

run_py.write_text(
    r'''
from pathlib import Path
import json
import os
import sys
import time
import traceback

ROOT = Path("/content/drive/MyDrive/Kaggle/ROGII")
EXP_NAME = "exp159_spatial_prior_confidence_features_on_exp092"
EXP = ROOT / "experiments" / EXP_NAME
LOCAL_CACHE = Path("/content/rogii_cache/exp159_inputs/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz")
LOCAL_SPATIAL = Path("/content/rogii_cache/exp159_inputs/exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz")

os.chdir(ROOT)
sys.path.insert(0, str(EXP))

from settings import ExperimentPaths, get_nested, load_config
from spatial_prior_confidence_features_on_exp092 import (
    run_spatial_prior_confidence_features_on_exp092,
)


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


try:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    print("START exp159 spatial prior confidence", flush=True)
    print("cwd=", Path.cwd(), flush=True)
    print(
        "local_cache_exists=",
        LOCAL_CACHE.exists(),
        "size=",
        LOCAL_CACHE.stat().st_size if LOCAL_CACHE.exists() else None,
        flush=True,
    )
    print(
        "local_spatial_exists=",
        LOCAL_SPATIAL.exists(),
        "size=",
        LOCAL_SPATIAL.stat().st_size if LOCAL_SPATIAL.exists() else None,
        flush=True,
    )
    print("active_modes=", cfg_get(config, "model.training.active_modes"), flush=True)
    print(
        "active_variants=",
        [
            v["name"]
            for v in cfg_get(config, "model.feature_ablation.active_variants", [])
            if v.get("enabled", True)
        ],
        flush=True,
    )
    spatial_config = dict(cfg_get(config, "model.spatial_prior_confidence", {}))
    spatial_config["spatial_oof_path"] = str(LOCAL_SPATIAL)

    t0 = time.time()
    summary = run_spatial_prior_confidence_features_on_exp092(
        output_dir=paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=LOCAL_CACHE,
        projection_config=cfg_get(config, "model.u_projection", {}),
        spatial_config=spatial_config,
        variants=cfg_get(config, "model.feature_ablation.active_variants", []),
        modes=cfg_get(config, "model.training.modes", {}),
        active_modes=cfg_get(config, "model.training.active_modes", []),
        n_splits=int(cfg_get(config, "validation.n_folds", 5)),
        fast=bool(cfg_get(config, "audit.fast", False)),
        early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
        max_rows=cfg_get(config, "model.training.max_rows"),
        max_train_rows=cfg_get(config, "model.training.max_train_rows"),
        save_models=bool(cfg_get(config, "model.training.save_models", True)),
        save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
        top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 60)),
    )
    summary["colab_elapsed_seconds_outer"] = round(time.time() - t0, 3)
    run_dir = EXP / "colab_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latest_done_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE exp159 spatial prior confidence", flush=True)
    print(json.dumps(summary, indent=2)[:12000], flush=True)
except Exception:
    (EXP / "colab_runs/latest_failed.txt").write_text(traceback.format_exc())
    print("FAILED exp159 spatial prior confidence", flush=True)
    traceback.print_exc()
    raise
'''
)

cmd = f"python {run_py} > {log_path} 2>&1"
proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(ROOT), start_new_session=True)
pid_path.write_text(str(proc.pid))

latest = {
    "run_id": run_id,
    "pid": proc.pid,
    "run_py": str(run_py),
    "log_path": str(log_path),
    "pid_path": str(pid_path),
    "completion_condition": str(RUN_DIR / "latest_done_summary.json"),
    "failure_condition": str(RUN_DIR / "latest_failed.txt"),
    "local_cache": "/content/rogii_cache/exp159_inputs/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
    "local_spatial": "/content/rogii_cache/exp159_inputs/exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz",
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
latest_path.write_text(json.dumps(latest, indent=2))
print(json.dumps(latest, indent=2))
