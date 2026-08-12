#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def make_notebook(
    *,
    experiment: str,
    drive_root: str,
    cache_source: str,
    local_cache_dir: str,
) -> dict[str, Any]:
    cache_name = Path(cache_source).name
    local_cache = f"{local_cache_dir.rstrip('/')}/{cache_name}"
    return {
        "cells": [
            markdown(
                f"""# {experiment} Colab Train

Colab-first runner for `{experiment}`.

This notebook mounts Google Drive, validates the ROGII layout, copies large cache
artifacts to `/content`, runs a LightGBM GPU smoke test, and starts the full train
in the background with Drive-backed logs.
"""
            ),
            markdown("## 1. Mount Drive and Check Runtime\n"),
            code(
                f"""from google.colab import drive
drive.mount("/content/drive", force_remount=True)

from pathlib import Path
import psutil

try:
    import torch
except Exception:
    torch = None

DRIVE_ROOT = Path("{drive_root}")
EXP_NAME = "{experiment}"
EXP_DIR = DRIVE_ROOT / "experiments" / EXP_NAME
CACHE_SOURCE = DRIVE_ROOT / "{cache_source}"
LOCAL_CACHE_DIR = Path("{local_cache_dir}")
LOCAL_CACHE = LOCAL_CACHE_DIR / "{cache_name}"

print("drive_root:", DRIVE_ROOT, DRIVE_ROOT.exists())
print("experiment_dir:", EXP_DIR, EXP_DIR.exists())
print("RAM GB:", psutil.virtual_memory().total / 1024**3)
if torch is not None:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print("CUDA:", torch.cuda.is_available(), device_name)
"""
            ),
            markdown("## 2. Install Dependencies\n"),
            code("!pip install -q numpy pandas pyyaml scikit-learn matplotlib lightgbm psutil\n"),
            markdown("## 3. Validate Drive Layout\n"),
            code(
                """from pathlib import Path

required = [
    DRIVE_ROOT / "project.yml",
    DRIVE_ROOT / "data/raw/train",
    EXP_DIR / "config.yaml",
    EXP_DIR / "settings.py",
    EXP_DIR / "u_projection_correction_disagreement_fullrun.py",
    CACHE_SOURCE,
]

for p in required:
    print(p, p.exists(), p.stat().st_size if p.exists() and p.is_file() else "")

if not all(p.exists() for p in required):
    missing = [str(p) for p in required if not p.exists()]
    raise FileNotFoundError("Missing required Colab inputs:\\n" + "\\n".join(missing))

print("train_files:", len(list((DRIVE_ROOT / "data/raw/train").glob("*"))))
"""
            ),
            markdown("## 4. Copy Large Cache to /content\n"),
            code(
                """import shutil
import time
import pandas as pd

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not LOCAL_CACHE.exists() or LOCAL_CACHE.stat().st_size != CACHE_SOURCE.stat().st_size:
    t0 = time.time()
    shutil.copy2(CACHE_SOURCE, LOCAL_CACHE)
    print("copied cache seconds:", round(time.time() - t0, 2))
else:
    print("local cache already present")

print("local_cache:", LOCAL_CACHE, LOCAL_CACHE.exists(), LOCAL_CACHE.stat().st_size)
preview = pd.read_csv(LOCAL_CACHE, nrows=3, dtype={"id": str, "well": str})
print("preview shape:", preview.shape)
display(preview[[c for c in ["id", "well", "target", "last_known_tvt"] if c in preview.columns]])
"""
            ),
            markdown("## 5. LightGBM GPU Smoke Test\n"),
            code(
                """import numpy as np
from lightgbm import LGBMRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(5000, 16)).astype(np.float32)
y = (X[:, 0] * 0.7 - X[:, 1] * 0.2 + rng.normal(scale=0.1, size=5000)).astype(np.float32)
model = LGBMRegressor(
    objective="regression",
    n_estimators=20,
    num_leaves=31,
    learning_rate=0.1,
    device_type="gpu",
    gpu_use_dp=True,
    verbose=-1,
)
model.fit(X, y)
print("lightgbm_gpu_smoke_ok", model.booster_.current_iteration())
"""
            ),
            markdown("## 6. Start Background Full Train\n"),
            code(
                f"""from pathlib import Path
import os
import subprocess
import textwrap
import json
import time

RUN_DIR = EXP_DIR / "colab_runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)

run_id = time.strftime("run_%Y%m%d_%H%M%S_highmem_local_cache")
run_py = RUN_DIR / f"{{run_id}}_{{EXP_NAME}}_full_train.py"
log_path = RUN_DIR / f"{{run_id}}_{{EXP_NAME}}_full_train.log"
pid_path = RUN_DIR / f"{{run_id}}_pid.txt"
latest_path = RUN_DIR / "latest_run.json"

run_py.write_text(f'''
from pathlib import Path
import os
import sys
import json
import traceback
import time

ROOT = Path("{drive_root}")
EXP_NAME = "{experiment}"
EXP = ROOT / "experiments" / EXP_NAME
LOCAL_CACHE = Path("{local_cache}")
os.chdir(ROOT)
sys.path.insert(0, str(EXP))

from settings import ExperimentPaths, load_config, get_nested
from u_projection_correction_disagreement_fullrun import (
    run_u_projection_correction_disagreement_fullrun,
)

def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value

try:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    print("START exp092 highmem local_cache", flush=True)
    print("cwd=", Path.cwd(), flush=True)
    print(
        "local_cache_exists=",
        LOCAL_CACHE.exists(),
        "size=",
        LOCAL_CACHE.stat().st_size if LOCAL_CACHE.exists() else None,
        flush=True,
    )
    print("active_modes=", cfg_get(config, "model.training.active_modes"), flush=True)
    active_variants = cfg_get(config, "model.feature_ablation.active_variants", [])
    print("active_variants=", [v["name"] for v in active_variants], flush=True)

    t0 = time.time()
    summary = run_u_projection_correction_disagreement_fullrun(
        output_dir=paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=LOCAL_CACHE,
        projection_config=cfg_get(config, "model.u_projection", {{}}),
        variants=cfg_get(config, "model.feature_ablation.active_variants", []),
        modes=cfg_get(config, "model.training.modes", {{}}),
        active_modes=cfg_get(config, "model.training.active_modes", []),
        n_splits=int(cfg_get(config, "validation.n_folds", 5)),
        fast=bool(cfg_get(config, "audit.fast", False)),
        early_stopping_rounds=int(cfg_get(config, "model.training.early_stopping_rounds", 250)),
        max_rows=cfg_get(config, "model.training.max_rows"),
        max_train_rows=cfg_get(config, "model.training.max_train_rows"),
        save_models=bool(cfg_get(config, "model.training.save_models", True)),
        save_predictions=bool(cfg_get(config, "model.training.save_predictions", True)),
        top_n_importance=int(cfg_get(config, "model.training.top_n_importance", 40)),
    )
    summary["colab_elapsed_seconds_outer"] = round(time.time() - t0, 3)
    (EXP / "colab_runs/latest_done_summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE exp092 highmem local_cache", flush=True)
    print(json.dumps(summary, indent=2)[:12000], flush=True)
except Exception:
    (EXP / "colab_runs/latest_failed.txt").write_text(traceback.format_exc())
    print("FAILED exp092 highmem local_cache", flush=True)
    traceback.print_exc()
    raise
''')

cmd = f"python {{run_py}} > {{log_path}} 2>&1"
proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(DRIVE_ROOT), start_new_session=True)
pid_path.write_text(str(proc.pid))

latest = {{
    "run_id": run_id,
    "pid": proc.pid,
    "run_py": str(run_py),
    "log_path": str(log_path),
    "pid_path": str(pid_path),
    "local_cache": str(LOCAL_CACHE),
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}}
latest_path.write_text(json.dumps(latest, indent=2))
print(json.dumps(latest, indent=2))
"""
            ),
            markdown("## 7. Check Status and Logs\n"),
            code(
                """from pathlib import Path
import json
import subprocess

RUN_DIR = EXP_DIR / "colab_runs"
latest = json.loads((RUN_DIR / "latest_run.json").read_text())
print(json.dumps(latest, indent=2))

pid = str(latest["pid"])
process = subprocess.run(
    ["ps", "-p", pid, "-o", "pid,ppid,stat,etime,time,%cpu,%mem,rss,cmd"],
    capture_output=True,
    text=True,
)
print(process.stdout)

log = Path(latest["log_path"])
print("log:", log, log.exists(), log.stat().st_size if log.exists() else None)
if log.exists():
    print("\\n".join(log.read_text(errors="replace").splitlines()[-120:]))

failed = RUN_DIR / "latest_failed.txt"
done = RUN_DIR / "latest_done_summary.json"
print("failed:", failed.exists(), failed.stat().st_size if failed.exists() else None)
print("done:", done.exists(), done.stat().st_size if done.exists() else None)
"""
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Colab-first Kaggle train notebook.")
    parser.add_argument("--repo-root", default=".", help="Repository root used to resolve output.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/.",
    )
    parser.add_argument("--output", required=True, help="Output .ipynb path.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/Kaggle/ROGII")
    parser.add_argument("--cache-source", required=True, help="Cache path relative to drive root.")
    parser.add_argument("--local-cache-dir", default="/content/rogii_cache/exp072_artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = make_notebook(
        experiment=args.experiment,
        drive_root=args.drive_root,
        cache_source=args.cache_source,
        local_cache_dir=args.local_cache_dir,
    )
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
