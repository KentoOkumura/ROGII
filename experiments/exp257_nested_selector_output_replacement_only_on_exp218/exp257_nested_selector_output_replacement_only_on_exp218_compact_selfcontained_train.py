# %% [markdown]
# # exp257 nested selector output replacement-only train

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Configuration and immutable selector artifact contract
# 3. Candidate and outer-fold reconstruction
# 4. exp218 380-feature surface
# 5. Replacement-only schema audit
# 6. GPU LightGBM training
# 7. Metrics and artifacts

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp257_nested_selector_output_replacement_only_on_exp218"
    )
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(
    name: str,
    candidates: list[Path],
    *,
    reset_settings: bool = False,
):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    if reset_settings:
        sys.modules.pop("settings", None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = import_file(
    "exp257_engine",
    [PACKAGE_DIR / "nested_selector_output_replacement_only.py"],
)
exp237 = import_file(
    "exp237_source",
    [
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
        Path(
            "/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "hmm_exp226_candidate_selector_on_exp183.py",
    ],
    reset_settings=True,
)
exp218 = import_file(
    "exp218_source",
    [
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
        Path(
            "/kaggle/input/exp218-gr-wavelet-rotation-exp148-train/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        PACKAGE_DIR / "gr_wavelet_rotation_confidence_features_on_exp148.py",
    ],
    reset_settings=True,
)
exp218_settings = import_file(
    "exp218_settings_source",
    [
        PACKAGE_DIR / "exp218_source/settings.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "settings.py"
        ),
    ],
)


def cfg_get(config: dict, dotted_key: str):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


print(
    json.dumps(
        {
            "experiment": CONFIG["experiment"],
            "route": CONFIG["experiment"]["route"],
            "active_variants": CONFIG["model"]["active_variants"],
            "lightgbm_configs": CONFIG["model"]["final_configs"],
            "folds": CONFIG["model"]["folds"],
            "planned_boosters": CONFIG["model"]["planned_boosters"],
            "parent_control_retraining": CONFIG["model"]["parent_control_retraining"],
            "expected_feature_count": CONFIG["model"]["expected_feature_count"],
            "selector_replaced_features": CONFIG["model"][
                "replaced_selector_output_count"
            ],
            "selector_added_features": CONFIG["model"]["added_selector_feature_count"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Configuration and immutable selector artifact contract

# %%
selector_directories = [
    Path(CONFIG["data"]["selector_artifact_dir_local"]),
    Path("/kaggle/input/exp238-nested-selector-train/artifacts"),
    Path("/kaggle/input/exp238-nested-selector-train"),
    Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train/artifacts"),
    Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train"),
]
SELECTOR_DIR = next(
    (
        directory
        for directory in selector_directories
        if (
            directory
            / f"{engine.SELECTOR_OUTPUT_PREFIX}_selector_summary.json"
        ).exists()
    ),
    None,
)
if SELECTOR_DIR is None and Path("/kaggle/input").exists():
    matches = list(
        Path("/kaggle/input").rglob(
            f"{engine.SELECTOR_OUTPUT_PREFIX}_selector_summary.json"
        )
    )
    SELECTOR_DIR = matches[0].parent if matches else None
if SELECTOR_DIR is None:
    raise FileNotFoundError(f"selector artifact directory not found: {selector_directories}")

selector_summary_path = (
    SELECTOR_DIR / f"{engine.SELECTOR_OUTPUT_PREFIX}_selector_summary.json"
)
selector_summary = json.loads(selector_summary_path.read_text())
expected_score_sha = {
    int(key): str(value)
    for key, value in CONFIG["reproducibility"][
        "selector_v3_nested_score_sha256_decompressed"
    ].items()
}
actual_score_sha = engine.validate_selector_summary_sha(
    selector_summary, expected_score_sha
)
if int(selector_summary["selector_model_count"]) != 20:
    raise ValueError("exp238 selector model count must remain outer5 x inner4 = 20")
print(
    json.dumps(
        {
            "selector_status": selector_summary["status"],
            "selector_guard_pass": selector_summary["decision"]["guard_pass"],
            "selector_model_count": selector_summary["selector_model_count"],
            "selector_score_sha_contract": actual_score_sha,
            "selector_refit_in_this_experiment": False,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 3. Candidate and outer-fold reconstruction

# %%
parent_config = exp237.load_config()
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(parent_config)
required_columns = exp237.build_required_columns(parent_config, candidates)
selector_frame, _ = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=required_columns,
    max_rows=None,
)
selector_frame, _, _ = exp237.add_feature_enrichment(
    selector_frame, parent_config, max_rows=None
)
selector_frame, _, _ = exp237.add_cluster_prior_confidence_features(
    selector_frame, parent_config, max_rows=None
)
selector_frame, _, _ = exp237.add_hmm_exp226_candidate_sources(
    selector_frame, parent_config
)
candidate_columns = [candidate.column for candidate in candidates]
if candidate_columns != selector_summary["candidate_columns"]:
    raise ValueError(
        {
            "message": "candidate order differs from saved exp238 selector scores",
            "runtime": candidate_columns,
            "saved": selector_summary["candidate_columns"],
        }
    )
if len(candidate_columns) != 11:
    raise ValueError(f"expected 11 selector candidates, got {candidate_columns}")

outer = engine.load_nested_fold_contracts(
    SELECTOR_DIR,
    len(selector_frame),
    int(CONFIG["validation"]["outer_folds"]),
)
groups = selector_frame["well"].astype(str).to_numpy()
fold_audit = []
for outer_fold, (train_rows, valid_rows) in enumerate(outer):
    train_wells = set(groups[train_rows])
    valid_wells = set(groups[valid_rows])
    overlap = train_wells & valid_wells
    if overlap:
        raise ValueError(f"outer {outer_fold}: train/valid well overlap: {sorted(overlap)[:5]}")
    fold_audit.append(
        {
            "outer_fold": outer_fold,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_wells": len(train_wells),
            "valid_wells": len(valid_wells),
            "well_overlap": 0,
        }
    )
display(pd.DataFrame(fold_audit))

# %% [markdown]
# ## 4. exp218 380-feature surface

# %%
exp218_config = yaml.safe_load(Path(exp218.__file__).with_name("config.yaml").read_text())
base_frame, base_feature_columns, _ = exp218.load_exp072_full_replay_cache_frame(
    cfg_get(exp218_config, "data.exp072_train_feature_cache_local"),
    max_rows=None,
)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(
        f"resolved competition train directory does not exist: {resolved_train_dir}"
    )
print(
    {
        "resolved_train_dir": str(resolved_train_dir),
        "horizontal_files": len(list(resolved_train_dir.glob("*__horizontal_well.csv"))),
    }
)
base_frame, _ = exp218.add_anchor_columns(base_frame, resolved_train_dir)
projection_config = cfg_get(exp218_config, "model.u_projection") or {}
projection, projection_groups, _ = exp218.build_u_projection_features(
    base_frame,
    source_specs=dict(projection_config.get("sources") or {}),
    degree=int(projection_config.get("degree", 3)),
    robust_iters=int(projection_config.get("robust_iters", 3)),
    clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
)
projection_columns = [column for column in projection if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, projection.reset_index(drop=True), projection_columns
)

learned_source, _ = exp218.load_learned_likelihood_ml_features(
    cfg_get(exp218_config, "data.learned_likelihood_train_features_local"),
    schema_path=cfg_get(
        exp218_config, "data.learned_likelihood_train_feature_schema_local"
    ),
    summary_path=cfg_get(exp218_config, "data.learned_likelihood_train_summary_local"),
)
learned, learned_groups, _ = exp218.build_learned_likelihood_features(
    learned_source,
    base_frame,
    cfg_get(exp218_config, "model.learned_likelihood_features") or {},
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, learned.reset_index(drop=True), learned_columns
)

grwr, grwr_groups, _, _ = exp218.build_gr_wavelet_rotation_confidence_features(
    base_frame,
    train_dir=resolved_train_dir,
    config=cfg_get(exp218_config, "model.gr_wavelet_rotation_confidence_features") or {},
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, grwr.reset_index(drop=True), grwr_columns
)
del projection, learned_source, learned, grwr
gc.collect()

feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(
    variant
    for variant in cfg_get(exp218_config, "model.feature_ablation.active_variants")
    if variant.get("name") == "gr_wavelet_rotation_confidence_addonly"
)
exp218_features = exp218.feature_columns_for_variant(
    base_feature_columns, feature_groups, parent_variant
)
print(
    {
        "rows": len(base_frame),
        "wells": base_frame.well.nunique(),
        "exp218_features": len(exp218_features),
    }
)

# %% [markdown]
# ## 5. Replacement-only schema audit

# %%
configured_replace = CONFIG["model"]["selector_output_adapter"]["replace_columns"]
configured_preserve = CONFIG["model"]["selector_output_adapter"]["preserve_columns"]
if configured_replace != engine.REPLACEMENT_COLUMNS:
    raise ValueError("config and engine replacement column order differ")
if configured_preserve != engine.PRESERVED_SELECTOR_INPUT_COLUMNS:
    raise ValueError("config and engine preserved input column order differ")

feature_schema = engine.validate_feature_contract(
    exp218_features,
    expected_feature_count=int(CONFIG["model"]["expected_feature_count"]),
)
replacement_contract = engine.replacement_contract_frame()
feature_schema_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_feature_schema.csv"
replacement_contract_path = (
    OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_replacement_contract.csv"
)
feature_schema.to_csv(feature_schema_path, index=False)
replacement_contract.to_csv(replacement_contract_path, index=False)
display(replacement_contract.groupby("action").size().rename("columns"))
print(
    {
        "feature_count": len(exp218_features),
        "selector_output_replaced": int(
            feature_schema["is_selector_output_replaced"].sum()
        ),
        "selector_input_diagnostics_preserved": int(
            feature_schema["is_selector_input_diagnostic_preserved"].sum()
        ),
        "nsel_columns": sum(column.startswith("nsel_") for column in exp218_features),
    }
)

# The hundreds of selector context columns are no longer needed after candidate
# and fold reconstruction. Keeping only keys, anchor, and 11 candidates prevents
# them from competing with the 3M x 380 LightGBM matrix for RAM.
selector_frame = selector_frame[
    [*engine.KEYS, "last_known_tvt", *candidate_columns]
].copy()
gc.collect()

# %% [markdown]
# ## 6. GPU LightGBM training

# %%
mode = cfg_get(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8")
final_params = exp218.apply_mode_overrides(
    exp218.exp063_lgb_config_family(fast=False), mode
)
if len(final_params) != int(CONFIG["model"]["final_configs"]):
    raise ValueError("LightGBM config count changed")

metrics, predictions, importance, model_manifest, replacement_audit = (
    engine.fit_final_nested_replacement_only(
        base_frame,
        selector_frame,
        exp218_features,
        outer,
        SELECTOR_DIR,
        candidate_columns,
        final_params,
        OUTPUT_DIR,
        error_floor=float(
            CONFIG["model"]["selector_output_adapter"]["error_floor"]
        ),
        early_stopping_rounds=int(
            CONFIG["model"]["training"]["early_stopping_rounds"]
        ),
    )
)
display(metrics)
display(importance.groupby("model").head(30))
display(pd.DataFrame(replacement_audit))

# %% [markdown]
# ## 7. Metrics and artifacts

# %%
reference_filename = (
    "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_"
    "final_oof_predictions.csv.gz"
)
reference_candidates = [
    Path(CONFIG["data"]["exp238_final_oof_predictions_local"]),
    Path("/kaggle/input/exp238-nested-rank-slot-exp218-train/artifacts")
    / reference_filename,
    Path("/kaggle/input/exp238-nested-rank-slot-exp218-train")
    / reference_filename,
]
reference_path = next(
    (candidate for candidate in reference_candidates if candidate.exists()), None
)
if reference_path is None and Path("/kaggle/input").exists():
    matches = list(Path("/kaggle/input").rglob(reference_filename))
    reference_path = matches[0] if matches else None
if reference_path is None:
    raise FileNotFoundError(
        f"same-fold exp238 OOF reference not found: {reference_candidates}"
    )

reference = pd.read_csv(reference_path, dtype={"id": str, "well": str})
required_reference_columns = [*engine.KEYS, "lgb_mean_pred_tvt"]
missing_reference = sorted(set(required_reference_columns) - set(reference.columns))
if missing_reference:
    raise ValueError(f"exp238 OOF reference missing columns: {missing_reference}")
if len(reference) != len(predictions):
    raise ValueError("exp238 and exp257 OOF row counts differ")
if not reference[engine.KEYS].reset_index(drop=True).equals(
    predictions[engine.KEYS].astype(str).reset_index(drop=True)
):
    raise ValueError("exp238 and exp257 OOF id/well order differs")

comparison = predictions[
    [*engine.KEYS, "outer_fold", "last_known_tvt", "target", "lgb_mean_pred_tvt"]
].copy()
comparison = comparison.rename(columns={"lgb_mean_pred_tvt": "exp257_pred_tvt"})
comparison["exp238_pred_tvt"] = reference["lgb_mean_pred_tvt"].to_numpy("float32")
comparison["target_tvt"] = (
    comparison["last_known_tvt"].to_numpy("float32")
    + comparison["target"].to_numpy("float32")
)
comparison["exp257_sq_error"] = (
    comparison["exp257_pred_tvt"] - comparison["target_tvt"]
) ** 2
comparison["exp238_sq_error"] = (
    comparison["exp238_pred_tvt"] - comparison["target_tvt"]
) ** 2
comparison["md_since"] = base_frame["md_since"].to_numpy("float32")
comparison["distance_bucket"] = pd.cut(
    comparison["md_since"],
    bins=[-float("inf"), 50.0, 100.0, 250.0, 500.0, 1000.0, float("inf")],
    labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    include_lowest=True,
)


def rmse_from_squared(values: pd.Series) -> float:
    return float(values.mean() ** 0.5)


by_well = (
    comparison.groupby("well", as_index=False)
    .agg(
        rows=("id", "size"),
        outer_fold=("outer_fold", "first"),
        exp257_rmse=("exp257_sq_error", rmse_from_squared),
        exp238_rmse=("exp238_sq_error", rmse_from_squared),
    )
    .sort_values("exp257_rmse", ascending=False)
)
by_well["delta_rmse"] = by_well["exp257_rmse"] - by_well["exp238_rmse"]

bucket_rows = []
bucket_specs = {
    "global": pd.Series("global", index=comparison.index),
    "distance": comparison["distance_bucket"],
    "outer_fold": comparison["outer_fold"].astype(str),
}
for family, labels in bucket_specs.items():
    grouped = comparison.assign(_bucket=labels).groupby("_bucket", observed=True)
    for bucket, group in grouped:
        exp257_rmse = float(group["exp257_sq_error"].mean() ** 0.5)
        exp238_rmse = float(group["exp238_sq_error"].mean() ** 0.5)
        bucket_rows.append(
            {
                "family": family,
                "bucket": str(bucket),
                "rows": len(group),
                "exp257_rmse": exp257_rmse,
                "exp238_rmse": exp238_rmse,
                "delta_rmse": exp257_rmse - exp238_rmse,
            }
        )
bucket_metrics = pd.DataFrame(bucket_rows)


def bucket_delta(family: str, bucket: str) -> float:
    selected = bucket_metrics.loc[
        bucket_metrics["family"].eq(family) & bucket_metrics["bucket"].eq(bucket),
        "delta_rmse",
    ]
    if len(selected) != 1:
        raise ValueError(f"missing unique bucket metric: {family}/{bucket}")
    return float(selected.iloc[0])


fold_deltas = bucket_metrics.loc[
    bucket_metrics["family"].eq("outer_fold"), "delta_rmse"
]
guard = {
    "global_delta_rmse": bucket_delta("global", "global"),
    "near_000_050_delta_rmse": bucket_delta("distance", "000_050"),
    "longtail_1000_plus_delta_rmse": bucket_delta("distance", "1000_plus"),
    "improved_outer_folds": int((fold_deltas < 0).sum()),
    "worst_well_max_regression": float(by_well["delta_rmse"].max()),
    "max_allowed_worst_well_regression": 0.25,
}
guard["pass"] = bool(
    guard["global_delta_rmse"] <= 0
    and guard["near_000_050_delta_rmse"] <= 0
    and guard["longtail_1000_plus_delta_rmse"] <= 0
    and guard["improved_outer_folds"] >= 3
    and guard["worst_well_max_regression"] <= guard["max_allowed_worst_well_regression"]
)
display(bucket_metrics)
display(by_well.head(30))
print(json.dumps(guard, indent=2))

metrics_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_metrics.csv"
predictions_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_oof_predictions.csv.gz"
importance_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_feature_importance_mean.csv"
manifest_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_model_manifest.json"
replacement_audit_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_replacement_audit.csv"
by_well_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_by_well.csv"
bucket_metrics_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_bucket_metrics.csv"
guard_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_guard.json"
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False, compression="gzip")
importance.to_csv(importance_path, index=False)
manifest_path.write_text(json.dumps(model_manifest, indent=2))
pd.DataFrame(replacement_audit).to_csv(replacement_audit_path, index=False)
by_well.to_csv(by_well_path, index=False)
bucket_metrics.to_csv(bucket_metrics_path, index=False)
guard_path.write_text(json.dumps(guard, indent=2))

summary = {
    "status": "nested_selector_output_replacement_only_train_completed",
    "route": CONFIG["experiment"]["route"],
    "selector_source": {
        "kernel": "kentookumura/exp238-nested-selector-train",
        "version": 4,
        "model_count": int(selector_summary["selector_model_count"]),
        "score_sha_contract": actual_score_sha,
        "selector_refit": False,
    },
    "rows": len(base_frame),
    "wells": int(base_frame.well.nunique()),
    "candidate_columns": candidate_columns,
    "candidate_count": len(candidate_columns),
    "feature_count": len(exp218_features),
    "exp218_schema_preserved": True,
    "selector_output_replaced_features": len(engine.REPLACEMENT_COLUMNS),
    "selector_input_diagnostics_preserved": len(
        engine.PRESERVED_SELECTOR_INPUT_COLUMNS
    ),
    "selector_added_features": 0,
    "nsel_feature_count": 0,
    "active_variants": 1,
    "lightgbm_configs": len(final_params),
    "folds": len(outer),
    "boosters": len(model_manifest),
    "parent_control_retraining": False,
    "metrics": metrics.to_dict(orient="records"),
    "references": CONFIG["validation"]["reference_metrics"],
    "same_fold_exp238_guard": guard,
    "inference_allowed": bool(guard["pass"]),
    "sha256": {
        "metrics": engine.sha256_file(metrics_path),
        "predictions_decompressed": engine.sha256_file(
            predictions_path, decompressed=True
        ),
        "feature_schema": engine.sha256_file(feature_schema_path),
        "replacement_contract": engine.sha256_file(replacement_contract_path),
        "model_manifest": engine.sha256_file(manifest_path),
        "replacement_audit": engine.sha256_file(replacement_audit_path),
        "by_well": engine.sha256_file(by_well_path),
        "bucket_metrics": engine.sha256_file(bucket_metrics_path),
        "guard": engine.sha256_file(guard_path),
        "exp238_oof_reference_decompressed": engine.sha256_file(
            reference_path, decompressed=True
        ),
    },
}
summary_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
