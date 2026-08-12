---
name: colab-notebook-runner
description: Create Colab-first notebooks for running Kaggle experiments from this repository when Kaggle GPU quota is limited. Use when Codex needs to generate or adapt a notebook for Google Colab/Colab CLI execution, mount Google Drive, validate Drive layout, copy large Kaggle artifacts from Drive to /content, run an experiment module such as exp092 on Colab GPU/high-RAM, write logs and status files to Drive, or troubleshoot Colab-specific path, RAM, runtime, and session issues.
---

# Colab Notebook Runner

## Core Workflow

Use this skill when moving an existing `experiments/expXXX_name/*_train.ipynb` workflow to Colab.

1. Keep the original experiment ID. Do not create a new Kaggle experiment only because the runtime changes from Kaggle Notebook to Colab.
2. Prefer a Colab-specific runner notebook instead of mutating the canonical Kaggle train notebook. Name it `<exp>_colab_train.ipynb` unless the user requests otherwise.
3. Store long-running logs and run metadata under `<exp>/colab_runs/` on Google Drive.
4. For large inputs, never rely on DriveFS direct reads for heavy training. Copy large `.csv.gz`, model, or feature-cache artifacts from Drive to `/content/...` first, then pass the `/content` path to the experiment code.
5. Use foreground execution only for short smoke tests. Use Drive-backed log files for long runs so the result survives CLI or browser disconnects.

## Drive Layout

Assume the Colab Drive root is:

```text
/content/drive/MyDrive/Kaggle/ROGII
```

Require this repo-like layout:

```text
ROGII/
  project.yml
  data/raw/
  experiments/<exp>/
  experiments/<cache-exp>/artifacts/
```

For `exp092_u_projection_correction_disagreement_fullrun`, require:

```text
experiments/exp072_exp063_full_replay_feature_cache/artifacts/
  exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz
  exp063_full_replay_feature_cache_feature_schema.csv
  exp063_full_replay_feature_cache_summary.json
```

## Notebook Generation

When creating or substantially adapting a Colab notebook for an experiment, follow the repository
Jupytext notebook policy:

- Prefer writing a Jupytext percent `.py` first, with `# %%` / `# %% [markdown]` cells, then
  convert it to `.ipynb`.
- Keep the cell table of contents reproducible from markdown cells in the `.py` source. The section
  list is flexible: add, remove, split, or merge sections to match the Colab task.
- For notebooks that contain experiment logic, use the compact self-contained pattern: include only
  the functions and constants actually needed for the Colab train/inference path, and avoid
  importing sibling experiment helper `.py` files unless the user explicitly allows an import-based
  runner.
- When copying runtime helpers from `settings.py` or experiment helper modules into a notebook,
  make the code notebook-safe. Colab and Kaggle notebook cells do not define `__file__`; use
  `PACKAGE_DIR = Path.cwd()` and replace `Path(__file__).resolve()`,
  `Path(__file__).with_name(...)`, and `Path(__file__).resolve().parents[...]` with notebook-safe
  path logic before running or uploading.
- For Colab runner notebooks whose job is only to mount Drive, validate layout, copy caches to
  `/content`, and launch an existing repo module, keep the runner compact and make the orchestration
  cells explicit; do not inline the full experiment code unless the user asks for a self-contained
  Colab notebook.
- Do not overwrite canonical Kaggle `*_train.ipynb` / `*_inference.ipynb` when making a Colab
  variant. Use a Colab-specific or trial name such as `<exp>_colab_train.py` /
  `<exp>_colab_train.ipynb` or `<exp>_compact_selfcontained_colab_train.py`.

Suggested flexible Colab cell structure:

```python
# %% [markdown]
# # expXXX Colab train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Drive mount and runtime check
# 3. Configuration
# 4. Input and artifact validation
# 5. Cache copy to /content
# 6. Smoke test
# 7. Train or inference execution
# 8. Status, logs, and artifacts

# %%
# imports
```

After editing a Jupytext Colab source, convert and test the pairing:

```bash
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/<exp>/<exp>_colab_train.py
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/<exp>/<exp>_colab_train.py
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev ruff check experiments/<exp>/<exp>_colab_train.py --select F821
rg -n "__file__|Path\\(__file__\\)" experiments/<exp>/<exp>_colab_train.py
```

Use `scripts/create_colab_train_notebook.py` to create a Colab runner notebook:

```bash
uv run python .agents/skills/colab-notebook-runner/scripts/create_colab_train_notebook.py \
  --repo-root . \
  --experiment exp092_u_projection_correction_disagreement_fullrun \
  --output experiments/exp092_u_projection_correction_disagreement_fullrun/exp092_u_projection_correction_disagreement_fullrun_colab_train.ipynb \
  --drive-root /content/drive/MyDrive/Kaggle/ROGII \
  --cache-source experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz
```

The script creates cells for:

- Drive mount and RAM/GPU check.
- Dependency installation.
- Drive layout and artifact validation.
- Copying large cache artifacts to `/content/rogii_cache/...`.
- LightGBM GPU smoke test.
- Background full train with logs, PID, `latest_run.json`, `latest_done_summary.json`, and `latest_failed.txt`.
- Status/log/artifact inspection.

## Colab CLI Guidance

Use Colab CLI for session creation, URL retrieval, and small checks:

```bash
colab new -s exp092-train --gpu T4
colab url -s exp092-train
colab status -s exp092-train
```

### Completion Semantics

When the user asks to "run it", "create an execution record", or similar wording for an
experiment, a Colab CLI smoke test is not enough. Treat the run as successful only when the
experiment reaches its real completion condition, for example:

- `latest_done_summary.json` exists under `<exp>/colab_runs/`.
- The summary/metrics file contains the expected experiment name, active variants, active modes,
  fold/config coverage, and final CV metrics.
- `latest_failed.txt` is absent, or any failure has been explicitly handled and recorded.

Small checks such as `print("ping")`, CUDA detection, cache preview, or LightGBM GPU smoke tests
are prerequisites only. Report them as preflight results, not as an execution record for the
experiment.

### URL and Drive Auth Handling

Distinguish these two URLs explicitly:

- Colab notebook URL: output from `colab url -s ...`; opens the notebook/runtime frontend.
- Google Drive authorization URL: the `https://accounts.google.com/...` URL printed by
  `colab drivemount`; opens the consent screen.

If `colab drivemount` prints an authorization request, show the user the exact
`accounts.google.com` URL and label it as the Drive authorization URL. Do not tell the user to use
the Colab notebook URL for Drive authorization. If both URLs are useful, present them in separate
lines with unambiguous labels.

If the wrong URL was shown or the auth flow becomes stale, interrupt the waiting `drivemount`
command and restart the mount flow before continuing.

If the user creates or changes runtime from the GUI, Colab CLI may show it as `[?]` and may not reconnect by name. In that case, provide notebook cells for GUI execution instead of forcing CLI execution.

`colab exec -f FILE` reads a local file, not a remote `/content` file. For remote scripts, pipe a local wrapper through stdin.

## Reproducible Colab CLI Experiment Run

Use this sequence when the user explicitly wants a Kaggle experiment run on Colab CLI. The goal is
not merely to start a process; it is to reach the experiment completion marker and retrieve the
minimal output needed for local review.

1. Create a named session and prove CLI execution works.

```bash
colab new -s <session-name> --gpu L4
printf 'print("colab_cli_exec_ok")\n' | colab exec -s <session-name> --timeout 60
printf 'import torch\nprint(torch.cuda.is_available())\nprint(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")\n' | colab exec -s <session-name> --timeout 60
```

2. Mount Drive with `tty: true`.
   - If `colab drivemount` prints an auth request, show only the `accounts.google.com` URL as the
     Drive authorization URL.
   - After the user authorizes, send Enter to the waiting command.
   - If mount fails after credentials propagate, retry `colab drivemount` once before changing
     strategy.

```bash
colab drivemount -s <session-name>
```

3. Verify Drive layout and runtime before heavy work.

```python
from pathlib import Path
import psutil, torch
root = Path("/content/drive/MyDrive/Kaggle/ROGII")
cache = root / "experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
print("ram_gb", round(psutil.virtual_memory().total / 1024**3, 2))
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("root_exists", root.exists())
print("project_yml", (root / "project.yml").exists())
print("cache", cache.exists(), cache.stat().st_size if cache.exists() else None)
```

4. Copy large caches to `/content` and smoke-test the actual stack.
   - Do not train directly from DriveFS for large `.csv.gz` feature caches.
   - A successful cache preview and LightGBM GPU smoke test are prerequisites, not experiment
     completion.

5. Start the full experiment as a background process with Drive-backed files.
   - Write a run script under `<exp>/colab_runs/`.
   - Redirect stdout/stderr to a Drive log.
   - Write `latest_run.json` with `run_id`, `pid`, `run_py`, `log_path`, `pid_path`,
     `completion_condition`, and `failure_condition`.
   - The run script must write `latest_done_summary.json` only after the experiment function
     returns a final summary, and `latest_failed.txt` only on exception.

6. Monitor by markers and logs.
   - Fold-level LightGBM logs can be silent for many minutes; check process/GPU use if the session
     is still reachable.
   - `colab exec` may occasionally lose the WebSocket during heavy training. Retry a lightweight
     `print("ping")` before deciding the session is lost.
   - A Colab CLI session disappearing, becoming `[?]`, or returning `404/401` is not proof that the
     experiment failed. Always check Drive-backed `latest_done_summary.json` and
     `latest_failed.txt` from a fresh check session before reporting failure.

7. After completion, retrieve minimal output locally.
   - Download the lightweight evidence needed for review: `latest_run.json`,
     `latest_done_summary.json`, final log, summary JSON, metrics CSV, by-well CSV, bucket metrics,
     feature summaries, feature schema, feature-importance summary/plot, and model manifest.
   - Skip heavy `predictions.csv.gz` and model `.txt` files unless inference, submission, or a later
     experiment explicitly needs them.
   - Store Colab output under a Kaggle-like local folder, for example
     `experiments/<exp>/kaggle/output/colab_run_<run_id>/`, and add a
     `minimal_output_manifest.json`.
   - Validate downloaded JSON locally and hand the evidence to `kaggle-review-exp` for recording under the source-of-truth split in `AGENTS.md`. Update `KAGGLE_DIRECTION.md` only when the experiment lifecycle requires it; route idea-backlog changes through `kaggle-strategy`.

## Validation Before Full Run

Before starting a full run, verify:

- `psutil.virtual_memory().total / 1024**3` is high enough. For exp092-scale full run, ordinary ~12GB T4 RAM is not enough; use high-memory runtime.
- `torch.cuda.is_available()` is true and the GPU name matches the requested runtime.
- Feature cache exists and is non-empty.
- A small `pd.read_csv(cache, nrows=3)` succeeds.
- LightGBM GPU smoke test succeeds.

## Failure Handling

If logs stop after the initial `START` block and no `latest_failed.txt` appears, treat it as likely OOM or VM kill, not a Python exception.

If Drive mount works in GUI but not through Colab CLI, continue in GUI and read Drive logs from a new Colab notebook.

If full run is too large for Colab high-memory, stop and report that Kaggle Notebook or a larger runtime is required; do not silently reduce `max_rows` unless the user asked for a smoke test.
