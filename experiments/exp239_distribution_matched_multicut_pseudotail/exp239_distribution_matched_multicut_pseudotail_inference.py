# %% [markdown]
# # exp239 distribution-matched multicut pseudo-tail inference
#
# Submission inference for the completed v11 pseudo-tail augmentation models.
# It regenerates the validated exp218 current-test feature surface and replaces
# only the saved LightGBM boosters with the 15 exp239 v11 boosters.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input contract
# 3. v11 model and feature-schema verification
# 4. exp218 current-test replay with v11 boosters
# 5. Submission metrics and reproducibility summary

# %%
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
import gr_wavelet_rotation_confidence_features_on_exp148 as exp218


V11_KERNEL = "kentookumura/exp239-pseudotail-dual-cache-streaming-train"
V11_PREFIX = "exp239_distribution_matched_multicut_pseudotail_augmentation"
V11_VARIANT = "distribution_matched_multicut_weight050"
INFERENCE_MODE = "v11_saved_boosters_exp218_rawtest_replay"


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value = get_nested(config, dotted_key)
    return default if value is None else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_unique_input_file(filename: str) -> Path:
    roots = [Path("/kaggle/input"), Path.cwd()]
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(path for path in root.rglob(filename) if path.is_file())
    matches = sorted(set(matches), key=str)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename}, found {len(matches)}: {matches[:20]}"
        )
    return matches[0]


def find_exp218_config() -> Path:
    preferred = [Path.cwd() / "inputs" / "exp218_config.yaml"]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob("exp218_config.yaml"), key=str)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one exp218_config.yaml, found {matches}")
    return matches[0]


# %% [markdown]
# ## 2. Configuration and input contract

# %%
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
exp218_config_path = find_exp218_config()
exp218_config = yaml.safe_load(exp218_config_path.read_text()) or {}

print(f"experiment={EXPERIMENT_NAME}")
print(f"route={cfg_get(config, 'experiment.route')}")
print(f"inference_mode={INFERENCE_MODE}")
print(f"v11_train_kernel={V11_KERNEL}")
print(f"exp218_config={exp218_config_path}")

if not bool(cfg_get(config, "model.replay_contract.inference_generation_enabled", False)):
    raise RuntimeError("v11 submission inference must be explicitly enabled in config")
if cfg_get(config, "inference.selected_variant") != V11_VARIANT:
    raise RuntimeError("config inference.selected_variant differs from the v11 model contract")


# %% [markdown]
# ## 3. v11 model and feature-schema verification
#
# All 15 model files are checked against the SHA values emitted by the v11
# training run. The ordered 380-feature schema must exactly match the feature
# order reconstructed from the saved exp218 manifest.

# %%
v11_metrics_path = find_unique_input_file(f"{V11_PREFIX}_metrics.csv")
v11_schema_path = find_unique_input_file(f"{V11_PREFIX}_feature_schema.csv")
v11_metrics = pd.read_csv(v11_metrics_path)
v11_fold_metrics = v11_metrics[pd.to_numeric(v11_metrics["fold"], errors="coerce").notna()].copy()
v11_fold_metrics["fold"] = v11_fold_metrics["fold"].astype(int)
if len(v11_fold_metrics) != 15:
    raise ValueError(f"expected 15 v11 fold models, got {len(v11_fold_metrics)}")

v11_schema = pd.read_csv(v11_schema_path).sort_values("feature_index")
v11_features = v11_schema["feature"].astype(str).tolist()
if len(v11_features) != 380 or len(set(v11_features)) != 380:
    raise ValueError("v11 feature schema must contain 380 unique ordered features")

base_manifest_path = exp218.find_model_manifest()
base_manifest = json.loads(base_manifest_path.read_text())
selected_variant = "gr_wavelet_rotation_confidence_addonly"
selected_mode = "gpu_repro_guard_dp_threads8"
variant_configs = {
    str(item["name"]): dict(item)
    for item in base_manifest.get("variants", [])
    if item.get("enabled", True)
}
if selected_variant not in variant_configs:
    raise ValueError(f"{selected_variant} missing from exp218 train manifest")
feature_groups = {
    **dict(base_manifest.get("projection_feature_groups") or {}),
    **dict(base_manifest.get("learned_feature_groups") or {}),
    **dict(base_manifest.get("grwr_feature_groups") or {}),
}
expected_features = exp218.feature_columns_for_variant(
    [str(value) for value in base_manifest["feature_source"]["feature_columns"]],
    feature_groups,
    variant_configs[selected_variant],
)
if expected_features != v11_features:
    mismatch = next(
        (
            (index, expected, actual)
            for index, (expected, actual) in enumerate(zip(expected_features, v11_features))
            if expected != actual
        ),
        None,
    )
    raise ValueError(
        "v11 feature order differs from exp218 raw-test inference contract: "
        f"expected={len(expected_features)} actual={len(v11_features)} first={mismatch}"
    )

model_rows: list[dict[str, Any]] = []
for row in v11_fold_metrics.sort_values(["model", "fold"]).itertuples(index=False):
    filename = f"{row.model}_fold{int(row.fold)}.txt"
    model_path = v11_metrics_path.parent / f"{V11_PREFIX}_models" / filename
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty v11 model: {model_path}")
    actual_sha = sha256_file(model_path)
    expected_sha = str(row.model_sha256)
    if actual_sha != expected_sha:
        raise ValueError(f"model SHA mismatch for {filename}: {actual_sha} != {expected_sha}")
    model_rows.append(
        {
            "variant": selected_variant,
            "mode": selected_mode,
            "model": str(row.model),
            "model_index": int(str(row.model).replace("lgb", "")),
            "fold": int(row.fold),
            "best_iteration": int(row.best_iteration),
            "file": str(model_path),
            "sha256": actual_sha,
        }
    )

inference_manifest = copy.deepcopy(base_manifest)
inference_manifest["experiment"] = EXPERIMENT_NAME
inference_manifest["augmentation_variant"] = V11_VARIANT
inference_manifest["source_train_kernel"] = V11_KERNEL
inference_manifest["models"] = model_rows
inference_manifest["model_count"] = len(model_rows)
manifest_dir = paths.artifacts_dir / f"{V11_PREFIX}_inference_models"
manifest_dir.mkdir(parents=True, exist_ok=True)
manifest_path = manifest_dir / "manifest.json"
manifest_path.write_text(json.dumps(inference_manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "verified_models": len(model_rows),
            "verified_features": len(v11_features),
            "v11_schema_sha256": sha256_file(v11_schema_path),
            "inference_manifest": str(manifest_path),
        },
        indent=2,
    )
)


# %% [markdown]
# ## 4. exp218 current-test replay with v11 boosters

# %%
summary = exp218.run_saved_model_inference(
    output_dir=paths.artifacts_dir,
    submission_path=paths.submission_path,
    sample_submission_path=paths.sample_submission_path,
    data_dir=paths.raw_data_dir,
    test_dir=paths.test_data_dir,
    model_manifest_path=manifest_path,
    learned_feature_path=None,
    learned_schema_path=cfg_get(config, "data.learned_likelihood_rawtest_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_rawtest_summary_local"),
    projection_config=cfg_get(exp218_config, "model.u_projection", {}),
    learned_feature_config=cfg_get(exp218_config, "model.learned_likelihood_features", {}),
    grwr_feature_config=cfg_get(
        exp218_config,
        "model.gr_wavelet_rotation_confidence_features",
        {},
    ),
    variant_name=selected_variant,
    mode_name=selected_mode,
    model_name="lgb_mean",
    submission_target_column=cfg_get(config, "data.submission_target_column", "tvt"),
    n_jobs=int(cfg_get(exp218_config, "generator.rawtest_replay.n_jobs", 8)),
    pf_seeds=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_seeds", 128)),
    pf_particles=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_particles", 500)),
    fast=False,
)


# %% [markdown]
# ## 5. Submission metrics and reproducibility summary

# %%
exp239_summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "v11_submission_inference_completed",
    "source_train_kernel": V11_KERNEL,
    "augmentation_variant": V11_VARIANT,
    "official_oof_rmse": float(
        cfg_get(config, "model.exp218_augmentation.final_official_oof_rmse", 8.697380065917969)
    ),
    "model_count": len(model_rows),
    "feature_count": len(v11_features),
    "model_manifest_sha256": sha256_file(manifest_path),
    "submission": str(paths.submission_path),
    "submission_sha256": sha256_file(paths.submission_path),
    "inference_metrics": summary["metrics"],
}
summary_path = paths.artifacts_dir / f"{V11_PREFIX}_inference_summary.json"
summary_path.write_text(json.dumps(exp239_summary, indent=2) + "\n")
paths.metrics_path.write_text(json.dumps(exp239_summary, indent=2) + "\n")
print(json.dumps(exp239_summary, indent=2))
print(f"submission={paths.submission_path}")
