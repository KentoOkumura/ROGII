# %% [markdown]
# # exp245 parity-safe saved-selector inference (CPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Saved-model and guard contracts
# 3. Current-test candidate surface
# 4. Fold-specific saved-selector inference
# 5. Metrics and generated artifacts

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp245_selector_context_parity_on_exp238")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else PACKAGE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(name: str, candidates: list[Path]):
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_one(filename: str) -> Path:
    local = PACKAGE_DIR / filename
    if local.exists():
        return local
    matches = list(Path("/kaggle/input").rglob(filename)) if Path("/kaggle/input").exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {matches}")
    return matches[0]


engine = import_file(
    "exp245_engine",
    [PACKAGE_DIR / "selector_context_parity_on_exp238.py"],
)
exp237_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp237_source/settings.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/settings.py"),
    ],
)
exp237 = import_file(
    "hmm_exp226_candidate_selector_on_exp183",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/hmm_exp226_candidate_selector_on_exp183.py"),
    ],
)
rawtest = import_file(
    "exp237_rawtest_inference",
    [
        PACKAGE_DIR / "exp237_source/rawtest_inference.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/rawtest_inference.py"),
    ],
)
replay = import_file(
    "exp218_public_notebook_replay_audit",
    [
        PACKAGE_DIR / "exp218_source/public_notebook_replay_audit.py",
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/public_notebook_replay_audit.py"),
    ],
)
exp226 = import_file(
    "exp226_connortynan_k16_reproduction",
    [
        PACKAGE_DIR / "exp226_source/connortynan_k16_reproduction.py",
        Path("experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/connortynan_k16_reproduction.py"),
    ],
)


def cfg_get(config: dict, dotted_key: str, default=None):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def build_current_test_exp226_surface(paths) -> tuple[pd.DataFrame, dict]:
    """Run exp226 once and expose the candidate plus its four train-time diagnostics."""
    config_candidates = [
        PACKAGE_DIR / "exp226_source/config.yaml",
        Path("experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/config.yaml"),
    ]
    config_path = next((path for path in config_candidates if path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"cannot resolve exp226 config: {config_candidates}")
    exp226_config = yaml.safe_load(config_path.read_text())
    params = exp226.params_from_config(exp226_config)
    max_train_wells = cfg_get(exp226_config, "inference.max_train_wells")
    max_test_wells = cfg_get(exp226_config, "inference.max_test_wells")
    max_train_wells = int(max_train_wells) if max_train_wells is not None else None
    max_test_wells = int(max_test_wells) if max_test_wells is not None else None
    if max_train_wells is not None or max_test_wells is not None:
        raise ValueError("exp226 debug well limits are forbidden for current-test inference")

    train_wells = exp226.load_train_wells(paths.train_data_dir, params)
    test_wells = exp226.load_test_wells(paths.test_data_dir, params)
    if not train_wells or not test_wells:
        raise FileNotFoundError(
            f"exp226 wells missing: train={len(train_wells)}, test={len(test_wells)}"
        )
    fields = exp226.build_fields(train_wells, params)
    kappa = exp226.fit_kappa(train_wells, fields, params)
    predictions = {
        well.wid: exp226.predict_well(well, fields, kappa, params)
        for well in test_wells
    }
    starts = {well.wid: int(well.s + 1) for well in test_wells}

    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    parts = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("sample_submission id format must be '<well>_<row_idx>'")
    wells = parts[0].astype(str).to_numpy()
    row_indices = parts[1].astype(np.int64).to_numpy()
    values = {
        "exp226_v6_k16_geometry_gr_u_projection": np.empty(len(sample), dtype=np.float32),
        "exp226_geop_tvt": np.empty(len(sample), dtype=np.float32),
        "exp226_gr_delta": np.empty(len(sample), dtype=np.float32),
    }
    for row, (well_id, row_index) in enumerate(zip(wells, row_indices, strict=False)):
        result = predictions.get(str(well_id))
        if result is None:
            raise KeyError(f"exp226 has no current-test prediction for well {well_id}")
        offset = int(row_index) - starts[str(well_id)]
        if offset < 0 or offset >= len(result.pred):
            raise IndexError(
                f"exp226 row offset out of range: well={well_id}, row={row_index}, offset={offset}"
            )
        values["exp226_v6_k16_geometry_gr_u_projection"][row] = result.pred[offset]
        values["exp226_geop_tvt"][row] = result.geop[offset]
        values["exp226_gr_delta"][row] = result.delta[offset]

    surface = pd.DataFrame({"id": sample["id"].astype(str), "well": wells, **values})
    surface["exp226_geop_minus_pred"] = (
        surface["exp226_geop_tvt"].to_numpy(np.float32)
        - surface["exp226_v6_k16_geometry_gr_u_projection"].to_numpy(np.float32)
    ).astype(np.float32)
    surface["exp226_geop_minus_pred_abs"] = np.abs(
        surface["exp226_geop_minus_pred"].to_numpy(np.float32)
    ).astype(np.float32)
    diagnostic_columns = [
        "exp226_v6_k16_geometry_gr_u_projection",
        "exp226_geop_tvt",
        "exp226_gr_delta",
        "exp226_geop_minus_pred",
        "exp226_geop_minus_pred_abs",
    ]
    if not np.isfinite(surface[diagnostic_columns].to_numpy(np.float32)).all():
        raise ValueError("current-test exp226 candidate/diagnostics contain non-finite values")
    return surface, {
        "mode": "full_train_fit_current_test_predict_with_diagnostics",
        "config": str(config_path),
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "rows": len(surface),
        "diagnostic_columns": diagnostic_columns[1:],
        "kappa": np.asarray(kappa, dtype=float).tolist(),
    }

# %% [markdown]
# ## 2. Saved-model and guard contracts

# %%
parent_config_candidates = [
    PACKAGE_DIR / "exp237_source/config.yaml",
    Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/config.yaml"),
]
parent_config_path = next(
    (path for path in parent_config_candidates if path.exists()), None
)
if parent_config_path is None:
    raise FileNotFoundError(f"cannot resolve exp237 config: {parent_config_candidates}")
parent_config = yaml.safe_load(parent_config_path.read_text())
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = True
cluster_settings = parent_config.setdefault("ranker", {}).setdefault(
    "cluster_prior_features", {}
)
cluster_settings["enabled"] = False
cluster_settings["base_feature_columns"] = []
paths = exp237_settings.ExperimentPaths()
if Path("/kaggle/input").exists():
    paths.require_kaggle_runtime()
paths.ensure_output_dirs()

selector_summary_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_summary.json")
selector_manifest_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_model_manifest.csv")
selector_summary = json.loads(selector_summary_path.read_text())
selector_manifest = pd.read_csv(selector_manifest_path)
if not bool(selector_summary.get("decision", {}).get("guard_pass", False)):
    raise RuntimeError("exp245 selector safety guard did not pass; inference is forbidden")
if not bool(selector_summary.get("context_parity_contract", {}).get("pass", False)):
    raise RuntimeError("exp245 train summary does not certify context parity")
outer_fold_count = int(CONFIG["validation"]["outer_folds"])
inner_fold_count = int(CONFIG["validation"]["inner_folds"])
expected_model_count = outer_fold_count * inner_fold_count
if len(selector_manifest) != expected_model_count:
    raise ValueError(f"expected {expected_model_count} saved selectors, got {len(selector_manifest)}")
if int(selector_summary.get("selector_model_count", -1)) != expected_model_count:
    raise ValueError("selector summary does not certify 20 saved models")
candidate_columns = [str(value) for value in selector_summary["candidate_columns"]]
context_columns = [str(value) for value in selector_summary["context_columns"]]
expected_context_count = int(CONFIG["model"]["selector"]["expected_context_features"])
if len(context_columns) != expected_context_count:
    raise ValueError(
        f"saved context count mismatch: expected {expected_context_count}, got {len(context_columns)}"
    )
excluded_prefixes = tuple(
    str(value)
    for value in CONFIG["model"]["selector"]["excluded_train_only_prefixes"]
)
forbidden_context = [
    column for column in context_columns if column.startswith(excluded_prefixes)
]
if forbidden_context:
    raise ValueError(f"saved selector still requires train-only context: {forbidden_context}")
expected_feature_names = [
    *context_columns,
    "candidate_code",
    "candidate_minus_anchor",
    "candidate_abs_minus_anchor",
]

resolved_models: dict[int, list[tuple[dict, Path]]] = {
    outer_fold: [] for outer_fold in range(outer_fold_count)
}
loaded_model_audit: list[dict] = []
for raw_item in selector_manifest.to_dict(orient="records"):
    item = dict(raw_item)
    outer_fold = int(item["outer_fold"])
    inner_fold = int(item["inner_fold"])
    filename = Path(str(item["file"])).name
    candidates = list(selector_manifest_path.parent.rglob(filename))
    if not candidates and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one saved selector {filename}, found {candidates}")
    model_path = candidates[0]
    if engine._sha(model_path) != str(item["sha256"]):
        raise ValueError(f"saved selector SHA mismatch: {filename}")
    manifest_feature_names = json.loads(str(item["feature_names_json"]))
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != expected_feature_names or manifest_feature_names != expected_feature_names:
        raise ValueError(f"selector feature schema mismatch: {filename}")
    del booster
    resolved_models[outer_fold].append((item, model_path))
    loaded_model_audit.append({
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "file": filename,
        "sha256": str(item["sha256"]),
        "best_iteration": int(item["best_iteration"]),
    })
for outer_fold, items in resolved_models.items():
    inner_folds = sorted(int(item[0]["inner_fold"]) for item in items)
    if inner_folds != list(range(inner_fold_count)):
        raise ValueError(f"outer fold {outer_fold} saved-selector coverage mismatch: {inner_folds}")
print(json.dumps({
    "selector_train_kernel_status": selector_summary["status"],
    "saved_selector_models": len(loaded_model_audit),
    "outer_folds": outer_fold_count,
    "inner_models_per_outer": inner_fold_count,
    "selector_training_in_this_notebook": False,
    "submission_requested": bool(CONFIG["inference"]["submission_requested"]),
}, indent=2))

# %% [markdown]
# ## 3. Current-test candidate surface

# %%
candidates = exp237.candidate_specs_from_config(parent_config)
if [item.column for item in candidates] != candidate_columns:
    raise ValueError("saved selector candidate schema differs from current-test config")

exp218_config_candidates = [
    PACKAGE_DIR / "exp218_source/config.yaml",
    Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/config.yaml"),
]
exp218_config_path = next(
    (path for path in exp218_config_candidates if path.exists()), None
)
if exp218_config_path is None:
    raise FileNotFoundError(f"cannot resolve exp218 config: {exp218_config_candidates}")
exp218_config = yaml.safe_load(exp218_config_path.read_text())
replay.configure_public_runtime(
    data_dir=paths.raw_data_dir,
    output_dir=OUTPUT_DIR,
    n_jobs=int(cfg_get(exp218_config, "runtime.num_workers", 8)),
    pf_seeds=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_seeds", 128)),
    pf_particles=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_particles", 500)),
    fast=False,
    use_gpu="auto",
)
base_test, base_meta = replay.build_replay_test_frame()
base_test["id"] = base_test["id"].astype(str)
base_test["well"] = base_test["well"].astype(str)
test_frame = rawtest._base_candidates(base_test.copy())
test_frame, hmm_meta = rawtest._attach_hmm_candidates(test_frame, parent_config, paths)
exp226_surface, exp226_meta = build_current_test_exp226_surface(paths)
test_frame = rawtest._merge_required(
    test_frame,
    exp226_surface,
    name="dynamic exp226 K16 current-test candidate and diagnostics",
)
test_frame, multiobs_meta = rawtest._attach_multiobs(test_frame, parent_config, paths)
test_frame, _, test_enrichment_meta = exp237.add_feature_enrichment(
    test_frame, parent_config, max_rows=None
)
test_frame, test_candidate_values, _ = rawtest._test_candidate_features(test_frame, candidates)
if not np.isfinite(test_candidate_values).all():
    raise ValueError("current-test candidate values contain non-finite values")
if not test_frame[["id", "well"]].equals(base_test[["id", "well"]]):
    raise ValueError("current-test selector surface changed replay row order")

missing_context = [column for column in context_columns if column not in test_frame.columns]
if missing_context:
    raise ValueError(f"current-test selector context is missing: {missing_context}")
nonfinite_context_counts: dict[str, int] = {}
for column in context_columns:
    values = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(np.float32)
    bad_count = int((~np.isfinite(values)).sum())
    if bad_count:
        nonfinite_context_counts[column] = bad_count
if nonfinite_context_counts:
    raise ValueError(
        f"current-test selector context contains non-finite values: {nonfinite_context_counts}"
    )
test_frame["target"] = np.float32(0.0)
print({
    "test_rows": len(test_frame),
    "test_wells": test_frame.well.nunique(),
    "available_context_features": len(context_columns),
    "missing_context_features": 0,
    "nonfinite_context_values": 0,
})

# %% [markdown]
# ## 4. Fold-specific saved-selector inference

# %%
test_rows = np.arange(len(test_frame), dtype=np.int64)
score_columns = [f"pred_error__{name}" for name in candidate_columns]
score_artifacts: list[dict] = []
score_paths: list[Path] = []
for outer_fold in range(outer_fold_count):
    score_sum = np.zeros((len(test_frame), len(candidate_columns)), dtype=np.float32)
    for item, model_path in resolved_models[outer_fold]:
        booster = lgb.Booster(model_file=str(model_path))
        score_sum += engine.predict_candidate_errors(
            booster,
            test_frame,
            test_rows,
            candidate_columns,
            context_columns,
            chunk_rows=int(CONFIG["model"]["selector"]["predict_chunk_rows"]),
        ) / np.float32(inner_fold_count)
        del booster
        gc.collect()
    if not np.isfinite(score_sum).all():
        raise ValueError(f"outer fold {outer_fold} current-test selector scores contain non-finite values")
    artifact = test_frame[[*engine.KEYS, "last_known_tvt", *candidate_columns]].copy()
    for index, name in enumerate(candidate_columns):
        artifact[f"pred_error__{name}"] = score_sum[:, index]
    score_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_current_test_selector_scores_outer{outer_fold}.csv.gz"
    artifact.to_csv(score_path, index=False, compression="gzip")
    score_paths.append(score_path)
    score_artifacts.append({
        "outer_fold": outer_fold,
        "models": inner_fold_count,
        "file": score_path.name,
        "rows": int(len(artifact)),
        "sha256_decompressed": engine._sha(score_path, decompressed=True),
        "score_min": float(artifact[score_columns].to_numpy(np.float32).min()),
        "score_max": float(artifact[score_columns].to_numpy(np.float32).max()),
    })
    display(artifact.head(3))
    del artifact, score_sum
    gc.collect()

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
loaded_manifest_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_current_test_selector_model_manifest.json"
summary_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_current_test_selector_summary.json"
loaded_manifest_path.write_text(json.dumps(loaded_model_audit, indent=2) + "\n")
summary = {
    "status": "current_test_parity_selector_inference_completed_not_submitted",
    "rows": int(len(test_frame)),
    "wells": int(test_frame.well.nunique()),
    "candidate_columns": candidate_columns,
    "context_feature_count": len(context_columns),
    "selector_model_count": len(loaded_model_audit),
    "outer_fold_count": outer_fold_count,
    "inner_models_per_outer": inner_fold_count,
    "selector_training_executed": False,
    "missing_context_features": missing_context,
    "nonfinite_context_counts": nonfinite_context_counts,
    "context_parity_contract": {
        "expected_context_feature_count": expected_context_count,
        "actual_context_feature_count": len(context_columns),
        "excluded_train_only_prefixes": list(excluded_prefixes),
        "required_exp226_diagnostic_columns": CONFIG["model"]["selector"]["required_exp226_diagnostic_columns"],
        "missing_context_features": missing_context,
        "nonfinite_context_counts": nonfinite_context_counts,
        "pass": True,
    },
    "score_artifacts": score_artifacts,
    "sources": {
        "saved_selector_summary": str(selector_summary_path),
        "saved_selector_manifest": str(selector_manifest_path),
        "selector_parent_config": str(parent_config_path),
        "base_test_replay": base_meta,
        "hmm": hmm_meta,
        "exp226": exp226_meta,
        "multiobs": multiobs_meta,
        "test_enrichment": test_enrichment_meta,
    },
    "sha256": {
        "loaded_model_manifest": engine._sha(loaded_manifest_path),
        "train_selector_manifest": engine._sha(selector_manifest_path),
        "train_selector_summary": engine._sha(selector_summary_path),
    },
    "notes": [
        "No selector is fitted in this notebook; all 20 nested selector models are loaded from selector train.",
        "Each outer fold gets its own four-model averaged current-test score surface.",
        "All selector context columns are regenerated on current test; missing/non-finite fallback is forbidden.",
        "The four exp226 diagnostics are generated from the same current-test prediction result as the exp226 candidate.",
        "No Kaggle competition submission is made by this notebook.",
    ],
}
summary_path.write_text(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True) + "\n")
print(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True))
