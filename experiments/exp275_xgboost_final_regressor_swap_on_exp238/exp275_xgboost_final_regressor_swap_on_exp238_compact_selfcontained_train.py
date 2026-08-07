# %% [markdown]
# # exp275 XGBoost final regressor swap on exp238 train
#
# Rebuild the exp238 380-base + 35 nested-selector feature surface, preserve
# its saved outer-fold roles and residual target, and replace only the final
# LightGBM family with the public `cdeotte/xgb-starter-cv-15` version-3
# XGBoost configuration. The frozen exp238 LightGBM OOF is the read-only
# comparison; neither the parent regressor nor the selector is retrained.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration, public-source audit, and GPU-cost gate
# 3. Frozen exp238 input contracts
# 4. Candidate and exp218 feature reconstruction
# 5. Fold-specific nested rank-slot features
# 6. Public-parameter XGBoost training
# 7. Stress, diversity, and adoption readouts
# 8. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import ast
import gc
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from xgboost import XGBRegressor
from xgboost import __version__ as xgboost_version

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp275_xgboost_final_regressor_swap_on_exp238")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = str(CONFIG["audit"]["output_prefix"])
STARTED = time.time()


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


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
    module.NOTEBOOK_SOURCE_PATH = path
    return module


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    hasher = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_artifact(
    filename: str,
    *,
    explicit: str | Path | None = None,
    expected_sha256: str | None = None,
    decompressed: bool = False,
) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(PACKAGE_DIR.rglob(filename))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.rglob(filename))
    checked: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            continue
        if expected_sha256 is not None:
            actual = sha256_path(path, decompressed=decompressed)
            checked.append(f"{path}: {actual}")
            if actual != expected_sha256:
                continue
        return path
    raise FileNotFoundError(
        f"Cannot resolve {filename} with the required SHA. Checked: {checked[:30]}"
    )


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.mean(np.abs(error)))


def within10(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.mean(np.abs(error) <= 10.0))


def feature_content_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    hasher = hashlib.sha256()
    chunk_rows = 100_000
    ids = frame["id"].astype(str)
    for start in range(0, len(frame), chunk_rows):
        payload = "\n".join(ids.iloc[start : start + chunk_rows].tolist()).encode()
        hasher.update(payload)
        hasher.update(b"\n")
    for column in columns:
        hasher.update(column.encode())
        hasher.update(b"\0")
        values = frame[column].to_numpy(dtype=np.float32, copy=False)
        for start in range(0, len(values), chunk_rows):
            hasher.update(values[start : start + chunk_rows].tobytes())
    return hasher.hexdigest()


def matrix_content_sha256(
    ids: pd.Series,
    values: np.ndarray,
    columns: list[str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(columns, separators=(",", ":")).encode())
    chunk_rows = 10_000
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        hasher.update("\n".join(ids.iloc[start:stop].astype(str)).encode())
        hasher.update(b"\n")
        hasher.update(np.ascontiguousarray(values[start:stop]).tobytes())
    return hasher.hexdigest()


def eval_public_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unsupported public-source name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.IfExp):
        condition = bool(eval_public_ast(node.test, env))
        return eval_public_ast(node.body if condition else node.orelse, env)
    if isinstance(node, ast.Dict):
        return {
            eval_public_ast(key, env): eval_public_ast(value, env)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_public_ast(node.operand, env)
    raise ValueError(f"Unsupported public-source AST node: {ast.dump(node)}")


def extract_public_xgb_params(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text())
    env: dict[str, Any] = {"FAST_DEBUG": False, "RANDOM_STATE": 42}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "XGB_PARAMS" not in source:
            continue
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "FAST_DEBUG",
                        "RANDOM_STATE",
                    }:
                        env[target.id] = eval_public_ast(statement.value, env)
                    if isinstance(target, ast.Name) and target.id == "XGB_PARAMS":
                        return dict(eval_public_ast(statement.value, env))
    raise ValueError("XGB_PARAMS assignment was not found in the public notebook")


exp238_engine = import_file(
    "exp238_engine_source",
    [
        PACKAGE_DIR / "exp238_source/nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        Path(
            "experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218/"
            "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"
        ),
    ],
)
exp237 = import_file(
    "exp237_source",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
        Path(
            "/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
    ],
    reset_settings=True,
)
exp218 = import_file(
    "exp218_source",
    [
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
        Path(
            "/kaggle/input/exp218-gr-wavelet-rotation-exp148-train/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
    ],
    reset_settings=True,
)
exp218_settings = import_file(
    "exp218_settings_source",
    [
        PACKAGE_DIR / "exp218_source/settings.py",
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py"),
    ],
)

# %% [markdown]
# ## 2. Configuration, public-source audit, and GPU-cost gate

# %%
public_notebook_path = find_artifact(
    "xgb-starter-cv-15.ipynb",
    explicit=nested(CONFIG, "data.public_notebook_local"),
    expected_sha256=nested(CONFIG, "model.public_source.sha256"),
)
public_params = extract_public_xgb_params(public_notebook_path)
configured_params = dict(nested(CONFIG, "model.xgboost.source_exact_params", {}))
if public_params != configured_params:
    raise ValueError(
        "Configured XGBoost parameters differ from the selected public notebook: "
        f"source={public_params}, config={configured_params}"
    )

active_variants = list(nested(CONFIG, "model.active_variants", []))
active_variant_count = int(nested(CONFIG, "model.active_variant_count"))
config_count = int(nested(CONFIG, "model.xgboost_config_count"))
n_folds = int(nested(CONFIG, "model.folds"))
planned_boosters = int(nested(CONFIG, "model.total_new_boosters"))
if active_variant_count != len(active_variants) or active_variant_count != 1:
    raise ValueError("Exactly one active XGBoost variant is required")
if config_count != 1 or n_folds != 5 or planned_boosters != config_count * n_folds:
    raise ValueError("Expected 1 config x 5 folds = 5 new boosters")
if bool(nested(CONFIG, "model.parent_control_retraining")):
    raise ValueError("Parent/control LightGBM retraining must remain disabled")
if bool(nested(CONFIG, "model.selector_retraining")):
    raise ValueError("Selector retraining must remain disabled")

parameter_audit = {
    "experiment": nested(CONFIG, "experiment.name"),
    "public_notebook": nested(CONFIG, "model.public_source.notebook"),
    "public_notebook_path": str(public_notebook_path),
    "public_notebook_sha256": sha256_path(public_notebook_path),
    "public_notebook_kernel_id_no": nested(CONFIG, "model.public_source.kernel_id_no"),
    "public_fast_debug": False,
    "source_exact_params": public_params,
    "source_exact_fit_params": nested(CONFIG, "model.xgboost.source_exact_fit_params"),
    "active_variants": active_variants,
    "xgboost_configs": config_count,
    "folds": n_folds,
    "new_boosters": planned_boosters,
    "parent_control_retraining": False,
    "selector_retraining": False,
    "sample_weight": nested(CONFIG, "model.sample_weight"),
}
parameter_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_parameter_audit.json"
parameter_audit_path.write_text(json.dumps(parameter_audit, indent=2))
print(json.dumps(parameter_audit, indent=2))

if not bool(nested(CONFIG, "model.run_approved", False)):
    raise RuntimeError(
        "Kaggle train is fail-closed until the user separately approves the recorded "
        "1 variant / 1 public XGBoost config / 5 folds / 5 boosters plan."
    )
if nested(CONFIG, "model.approval.status") != "approved":
    raise RuntimeError("Approval status must be 'approved' before Kaggle training")
if not nested(CONFIG, "model.approval.approved_at"):
    raise RuntimeError("Approval timestamp must be recorded before Kaggle training")

# %% [markdown]
# ## 3. Frozen exp238 input contracts

# %%
parent_oof_filename = (
    "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_final_oof_predictions.csv.gz"
)
parent_oof_path = find_artifact(
    parent_oof_filename,
    explicit=nested(CONFIG, "data.parent_final_oof_local"),
    expected_sha256=nested(CONFIG, "frozen_parent.expected_oof_decompressed_sha256"),
    decompressed=True,
)
hidden_assignment_path = find_artifact(
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
    explicit=nested(CONFIG, "data.hidden_like_assignment_local"),
    expected_sha256=nested(CONFIG, "validation.expected_hidden_assignment_sha256"),
)

score_paths: list[Path] = []
for outer_fold in range(n_folds):
    filename = f"{exp238_engine.OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
    score_path = find_artifact(
        filename,
        expected_sha256=str(
            nested(
                CONFIG,
                f"frozen_parent.selector_score_sha256_decompressed.{outer_fold}",
            )
        ),
        decompressed=True,
    )
    score_paths.append(score_path)
selector_dirs = {path.parent.resolve() for path in score_paths}
if len(selector_dirs) != 1:
    raise ValueError(f"Nested score files resolved to different directories: {selector_dirs}")
SELECTOR_DIR = next(iter(selector_dirs))

parent_oof = pd.read_csv(parent_oof_path, dtype={"id": str, "well": str})
required_parent_columns = {
    "id",
    "well",
    "last_known_tvt",
    "target",
    "lgb_mean_pred_tvt",
}
if not required_parent_columns.issubset(parent_oof.columns):
    raise ValueError(
        f"Frozen exp238 OOF columns changed: missing {required_parent_columns - set(parent_oof)}"
    )
if len(parent_oof) != int(nested(CONFIG, "frozen_parent.expected_rows")):
    raise ValueError("Frozen exp238 OOF row count changed")
if parent_oof["well"].nunique() != int(nested(CONFIG, "frozen_parent.expected_wells")):
    raise ValueError("Frozen exp238 OOF well count changed")
parent_truth = parent_oof["last_known_tvt"].to_numpy(np.float32) + parent_oof["target"].to_numpy(
    np.float32
)
parent_prediction = parent_oof["lgb_mean_pred_tvt"].to_numpy(np.float32)
parent_rmse = rmse(parent_truth, parent_prediction)
if abs(parent_rmse - float(nested(CONFIG, "frozen_parent.expected_lgb_mean_rmse"))) > float(
    nested(CONFIG, "frozen_parent.oof_rmse_tolerance")
):
    raise ValueError(f"Frozen exp238 OOF RMSE changed: {parent_rmse}")
hidden_assignment = pd.read_csv(hidden_assignment_path, dtype={"well_id": str})
print(
    json.dumps(
        {
            "parent_oof": str(parent_oof_path),
            "parent_rows": len(parent_oof),
            "parent_wells": parent_oof["well"].nunique(),
            "parent_lgb_mean_rmse": parent_rmse,
            "parent_oof_decompressed_sha256": sha256_path(parent_oof_path, decompressed=True),
            "selector_dir": str(SELECTOR_DIR),
            "selector_score_decompressed_sha256": {
                str(index): sha256_path(path, decompressed=True)
                for index, path in enumerate(score_paths)
            },
            "hidden_assignment": str(hidden_assignment_path),
        },
        indent=2,
    )
)
display(parent_oof.head())
display(hidden_assignment.head())

# %% [markdown]
# ## 4. Candidate and exp218 feature reconstruction

# %%
selector_config = exp237.load_config()
selector_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(selector_config)
required_columns = exp237.build_required_columns(selector_config, candidates)
selector_frame, selector_cache_meta = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(selector_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(selector_config, "data.exp099_train_feature_schema_local"),
    required_columns=required_columns,
    max_rows=None,
)
selector_frame, enrichment_summary, _ = exp237.add_feature_enrichment(
    selector_frame, selector_config, max_rows=None
)
selector_frame, cluster_summary, _ = exp237.add_cluster_prior_confidence_features(
    selector_frame, selector_config, max_rows=None
)
selector_frame, hmm_summary, _ = exp237.add_hmm_exp226_candidate_sources(
    selector_frame, selector_config
)
candidate_columns = [item.column for item in candidates]
outer_roles = exp238_engine.load_nested_fold_contracts(
    SELECTOR_DIR,
    len(selector_frame),
    n_folds,
)

selector_min = selector_frame[[*exp238_engine.KEYS, "last_known_tvt", *candidate_columns]].copy()
del selector_frame
selector_frame = selector_min
gc.collect()

exp218_config_path = Path(exp218.NOTEBOOK_SOURCE_PATH).with_name("config.yaml")
if not exp218_config_path.exists():
    raise FileNotFoundError(f"Missing bootstrapped exp218 config: {exp218_config_path}")
exp218_config = yaml.safe_load(exp218_config_path.read_text())
base_frame, base_feature_columns, base_feature_meta = exp218.load_exp072_full_replay_cache_frame(
    nested(exp218_config, "data.exp072_train_feature_cache_local"),
    max_rows=None,
)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(
        f"Resolved competition train directory does not exist: {resolved_train_dir}"
    )
base_frame, anchor_meta = exp218.add_anchor_columns(base_frame, resolved_train_dir)

projection_config = nested(exp218_config, "model.u_projection", {})
projection, projection_groups, projection_summary = exp218.build_u_projection_features(
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

learned_source, learned_source_meta = exp218.load_learned_likelihood_ml_features(
    nested(exp218_config, "data.learned_likelihood_train_features_local"),
    schema_path=nested(exp218_config, "data.learned_likelihood_train_feature_schema_local"),
    summary_path=nested(exp218_config, "data.learned_likelihood_train_summary_local"),
)
learned, learned_groups, learned_summary = exp218.build_learned_likelihood_features(
    learned_source,
    base_frame,
    nested(exp218_config, "model.learned_likelihood_features", {}),
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
if not base_frame["id"].equals(learned["id"]) or not base_frame["well"].equals(learned["well"]):
    raise ValueError("exp145 learned-likelihood feature row order changed")
exp218._assign_aligned_float32_columns(base_frame, learned.reset_index(drop=True), learned_columns)

grwr, grwr_groups, grwr_summary, grwr_meta = exp218.build_gr_wavelet_rotation_confidence_features(
    base_frame,
    train_dir=resolved_train_dir,
    config=nested(exp218_config, "model.gr_wavelet_rotation_confidence_features", {}),
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
if not base_frame["id"].equals(grwr["id"]) or not base_frame["well"].equals(grwr["well"]):
    raise ValueError("GRWR feature row order changed")
exp218._assign_aligned_float32_columns(base_frame, grwr.reset_index(drop=True), grwr_columns)
del projection, learned_source, learned, grwr
gc.collect()

feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(
    variant
    for variant in nested(exp218_config, "model.feature_ablation.active_variants", [])
    if variant.get("name") == "gr_wavelet_rotation_confidence_addonly"
)
base_features = exp218.feature_columns_for_variant(
    base_feature_columns,
    feature_groups,
    parent_variant,
)
if len(base_features) != int(nested(CONFIG, "model.expected_base_feature_count")):
    raise ValueError(f"Expected 380 exp218 features, got {len(base_features)}")
if len(base_frame) != int(nested(CONFIG, "validation.expected_rows")):
    raise ValueError("Rebuilt exp218 surface row count changed")
if base_frame["well"].nunique() != int(nested(CONFIG, "validation.expected_wells")):
    raise ValueError("Rebuilt exp218 surface well count changed")
if (
    not base_frame[exp238_engine.KEYS]
    .astype(str)
    .reset_index(drop=True)
    .equals(selector_frame[exp238_engine.KEYS].astype(str).reset_index(drop=True))
):
    raise ValueError("exp218 and selector frames are not id/well row aligned")
if (
    not base_frame["id"]
    .astype(str)
    .reset_index(drop=True)
    .equals(parent_oof["id"].astype(str).reset_index(drop=True))
):
    raise ValueError("Rebuilt exp218 surface ID order differs from frozen exp238 OOF")
if (
    not base_frame["well"]
    .astype(str)
    .reset_index(drop=True)
    .equals(parent_oof["well"].astype(str).reset_index(drop=True))
):
    raise ValueError("Rebuilt exp218 surface well order differs from frozen exp238 OOF")
if not np.allclose(
    base_frame["target"].to_numpy(np.float32),
    parent_oof["target"].to_numpy(np.float32),
    atol=1e-6,
    rtol=0.0,
):
    raise ValueError("Rebuilt target differs from frozen exp238 OOF")
if not np.allclose(
    base_frame["last_known_tvt"].to_numpy(np.float32),
    parent_oof["last_known_tvt"].to_numpy(np.float32),
    atol=1e-6,
    rtol=0.0,
):
    raise ValueError("Rebuilt last-known TVT differs from frozen exp238 OOF")

for fold, (train_rows, valid_rows) in enumerate(outer_roles):
    train_wells = set(base_frame.iloc[train_rows]["well"].astype(str))
    valid_wells = set(base_frame.iloc[valid_rows]["well"].astype(str))
    if train_wells & valid_wells:
        raise ValueError(f"outer fold {fold}: well leakage")

probe_scores = np.zeros((1, len(candidate_columns)), dtype=np.float32)
probe_extra = exp238_engine.rank_slot_features(
    selector_frame,
    np.array([0], dtype=np.int64),
    probe_scores,
    candidate_columns,
).reset_index(drop=True)
selector_features = list(probe_extra.columns)
all_features = [*base_features, *selector_features]
if len(selector_features) != int(nested(CONFIG, "model.expected_selector_feature_count")):
    raise ValueError(f"Expected 35 selector features, got {len(selector_features)}")
if len(all_features) != int(nested(CONFIG, "model.expected_final_feature_count")):
    raise ValueError(f"Expected 415 final features, got {len(all_features)}")

feature_schema = pd.DataFrame(
    {
        "feature_index": np.arange(len(all_features), dtype=np.int32),
        "feature": all_features,
        "family": ["exp218_base"] * len(base_features)
        + ["exp238_nested_selector_rank_slot"] * len(selector_features),
    }
)
feature_schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_schema.csv"
feature_schema.to_csv(feature_schema_path, index=False)
base_feature_content_sha = feature_content_sha256(base_frame, base_features)
print(
    json.dumps(
        {
            "rows": len(base_frame),
            "wells": base_frame["well"].nunique(),
            "candidate_count": len(candidate_columns),
            "base_features": len(base_features),
            "selector_features": len(selector_features),
            "final_features": len(all_features),
            "base_feature_content_sha256": base_feature_content_sha,
            "feature_schema_sha256": sha256_path(feature_schema_path),
            "selector_cache_meta": selector_cache_meta,
            "base_feature_meta": base_feature_meta,
            "anchor_meta": anchor_meta,
            "learned_source_meta": learned_source_meta,
            "grwr_meta": grwr_meta,
        },
        indent=2,
        default=str,
    )
)
display(feature_schema)
display(enrichment_summary)
display(cluster_summary)
display(hmm_summary)
display(projection_summary.head())
display(learned_summary.head())
display(grwr_summary.head())
del parent_oof, probe_extra
gc.collect()

# %% [markdown]
# ## 5. Fold-specific nested rank-slot features

# %%
fold_assignment = np.full(len(base_frame), -1, dtype=np.int8)
for outer_fold, (_, valid_rows) in enumerate(outer_roles):
    fold_assignment[valid_rows] = outer_fold
if (fold_assignment < 0).any():
    raise ValueError("Outer-fold roles do not cover every row")

y_residual = base_frame["target"].to_numpy(np.float32)
last_known_tvt = base_frame["last_known_tvt"].to_numpy(np.float32)
target_tvt = last_known_tvt + y_residual
xgb_oof_residual = np.full(len(base_frame), np.nan, dtype=np.float32)
importance_rows: list[dict[str, Any]] = []
model_rows: list[dict[str, Any]] = []
fold_metric_rows: list[dict[str, Any]] = []
model_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_xgb_models"
model_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 6. Public-parameter XGBoost training

# %%
for outer_fold, (train_rows, valid_rows) in enumerate(outer_roles):
    score_item = exp238_engine.load_nested_score_artifact(
        SELECTOR_DIR,
        selector_frame,
        outer_fold,
        (train_rows, valid_rows),
        candidate_columns,
    )
    train_extra = exp238_engine.rank_slot_features(
        selector_frame,
        train_rows,
        score_item["train_scores"],
        candidate_columns,
    ).reset_index(drop=True)
    valid_extra = exp238_engine.rank_slot_features(
        selector_frame,
        valid_rows,
        score_item["valid_scores"],
        candidate_columns,
    ).reset_index(drop=True)
    if (
        list(train_extra.columns) != selector_features
        or list(valid_extra.columns) != selector_features
    ):
        raise ValueError(f"outer fold {outer_fold}: selector feature schema changed")

    x_train_values = np.empty((len(train_rows), len(all_features)), dtype=np.float32)
    x_valid_values = np.empty((len(valid_rows), len(all_features)), dtype=np.float32)
    chunk_columns = 32
    for start in range(0, len(base_features), chunk_columns):
        stop = min(start + chunk_columns, len(base_features))
        columns = base_features[start:stop]
        x_train_values[:, start:stop] = base_frame.iloc[train_rows][columns].to_numpy(
            dtype=np.float32, copy=True
        )
        x_valid_values[:, start:stop] = base_frame.iloc[valid_rows][columns].to_numpy(
            dtype=np.float32, copy=True
        )
    x_train_values[:, len(base_features) :] = train_extra.to_numpy(dtype=np.float32, copy=False)
    x_valid_values[:, len(base_features) :] = valid_extra.to_numpy(dtype=np.float32, copy=False)
    if np.isinf(x_train_values).any() or np.isinf(x_valid_values).any():
        raise ValueError(
            f"outer fold {outer_fold}: infinite values are not part of the parent contract"
        )

    train_matrix_sha = matrix_content_sha256(
        base_frame.iloc[train_rows]["id"].reset_index(drop=True),
        x_train_values,
        all_features,
    )
    valid_matrix_sha = matrix_content_sha256(
        base_frame.iloc[valid_rows]["id"].reset_index(drop=True),
        x_valid_values,
        all_features,
    )
    x_train = pd.DataFrame(x_train_values, columns=all_features, copy=False)
    x_valid = pd.DataFrame(x_valid_values, columns=all_features, copy=False)
    del train_extra, valid_extra, score_item
    gc.collect()

    model = XGBRegressor(**public_params)
    model.fit(
        x_train,
        y_residual[train_rows],
        eval_set=[(x_valid, y_residual[valid_rows])],
        verbose=int(nested(CONFIG, "model.xgboost.source_exact_fit_params.verbose")),
    )
    prediction_residual = model.predict(x_valid).astype(np.float32)
    xgb_oof_residual[valid_rows] = prediction_residual
    prediction_tvt = last_known_tvt[valid_rows] + prediction_residual
    parent_fold_prediction = parent_prediction[valid_rows]

    model_path = model_dir / f"xgb_public_cdeotte_v3__outer{outer_fold}.json"
    model.save_model(model_path)
    booster = model.get_booster()
    if int(booster.num_boosted_rounds()) != int(public_params["n_estimators"]):
        raise ValueError(f"outer fold {outer_fold}: public fixed tree count was not preserved")
    model_rows.append(
        {
            "model": "xgb_public_cdeotte_v3",
            "outer_fold": outer_fold,
            "file": str(model_path),
            "sha256": sha256_path(model_path),
            "trees": int(booster.num_boosted_rounds()),
            "features": len(all_features),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_wells": int(base_frame.iloc[train_rows]["well"].nunique()),
            "valid_wells": int(base_frame.iloc[valid_rows]["well"].nunique()),
            "train_matrix_sha256": train_matrix_sha,
            "valid_matrix_sha256": valid_matrix_sha,
        }
    )
    fold_metric_rows.append(
        {
            "fold": outer_fold,
            "valid_rows": len(valid_rows),
            "valid_wells": int(base_frame.iloc[valid_rows]["well"].nunique()),
            "parent_lgb_mean_rmse": rmse(target_tvt[valid_rows], parent_fold_prediction),
            "xgboost_rmse": rmse(target_tvt[valid_rows], prediction_tvt),
        }
    )
    fold_metric_rows[-1]["delta_rmse"] = (
        fold_metric_rows[-1]["xgboost_rmse"] - fold_metric_rows[-1]["parent_lgb_mean_rmse"]
    )
    importance_rows.extend(
        {
            "model": "xgb_public_cdeotte_v3",
            "outer_fold": outer_fold,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(all_features, model.feature_importances_, strict=True)
    )
    print(json.dumps(fold_metric_rows[-1], sort_keys=True), flush=True)

    del (
        x_train,
        x_valid,
        x_train_values,
        x_valid_values,
        prediction_residual,
        prediction_tvt,
        model,
        booster,
    )
    gc.collect()

if not np.isfinite(xgb_oof_residual).all():
    raise ValueError("XGBoost OOF coverage is incomplete")
xgb_prediction = last_known_tvt + xgb_oof_residual
fold_metrics = pd.DataFrame(fold_metric_rows)
display(fold_metrics)

# %% [markdown]
# ## 7. Stress, diversity, and adoption readouts

# %%
assignment_by_well = hidden_assignment.set_index("well_id")
role_records = base_frame["well"].astype(str).map(assignment_by_well.to_dict("index"))
spatial_valid = role_records.map(
    lambda value: isinstance(value, dict) and value.get("verification_like_spatial_role") == "valid"
).to_numpy(bool)
typewell_valid = role_records.map(
    lambda value: (
        isinstance(value, dict) and value.get("verification_like_typewell_purged_role") == "valid"
    )
).to_numpy(bool)
md_since = base_frame["md_since"].to_numpy(np.float32)
distance_bucket = pd.cut(
    md_since,
    bins=list(nested(CONFIG, "validation.distance_bins")),
    labels=list(nested(CONFIG, "validation.distance_labels")),
    right=False,
).astype(str)

surface_masks: dict[str, np.ndarray] = {
    "overall": np.ones(len(base_frame), dtype=bool),
    "near_000_050": md_since < 50,
    "1000_plus": md_since >= 1000,
    "hidden_like_spatial": spatial_valid,
    "hidden_like_typewell_purged": typewell_valid,
}
for bucket in list(nested(CONFIG, "validation.distance_labels")):
    surface_masks[f"distance_{bucket}"] = distance_bucket == bucket

stress_rows: list[dict[str, Any]] = []
for surface, mask in surface_masks.items():
    if not mask.any():
        continue
    parent_value = rmse(target_tvt[mask], parent_prediction[mask])
    xgb_value = rmse(target_tvt[mask], xgb_prediction[mask])
    stress_rows.append(
        {
            "surface": surface,
            "rows": int(mask.sum()),
            "wells": int(base_frame.loc[mask, "well"].nunique()),
            "parent_lgb_mean_rmse": parent_value,
            "xgboost_rmse": xgb_value,
            "delta_rmse": xgb_value - parent_value,
        }
    )
stress_metrics = pd.DataFrame(stress_rows)
bucket_metrics = stress_metrics[stress_metrics["surface"].str.startswith("distance_")].copy()
hidden_like_metrics = stress_metrics[
    stress_metrics["surface"].str.startswith("hidden_like_")
].copy()

blend_weight = float(nested(CONFIG, "model.blend_readout.xgboost_weight"))
blend_prediction = (
    (1.0 - blend_weight) * parent_prediction + blend_weight * xgb_prediction
).astype(np.float32)
blend_rows: list[dict[str, Any]] = []
for surface, mask in surface_masks.items():
    if not mask.any():
        continue
    parent_value = rmse(target_tvt[mask], parent_prediction[mask])
    blend_value = rmse(target_tvt[mask], blend_prediction[mask])
    blend_rows.append(
        {
            "xgboost_weight": blend_weight,
            "surface": surface,
            "rows": int(mask.sum()),
            "parent_lgb_mean_rmse": parent_value,
            "blend_rmse": blend_value,
            "delta_rmse": blend_value - parent_value,
        }
    )
blend_readout = pd.DataFrame(blend_rows)

metric_rows: list[dict[str, Any]] = []
for name, prediction in {
    "parent_exp238_lgb_mean": parent_prediction,
    "xgboost_public_cdeotte_v3": xgb_prediction,
    "fixed_blend_xgb_w0p25": blend_prediction,
}.items():
    metric_rows.append(
        {
            "model": name,
            "rmse": rmse(target_tvt, prediction),
            "mae": mae(target_tvt, prediction),
            "within10": within10(target_tvt, prediction),
            "prediction_correlation_with_parent": float(
                np.corrcoef(parent_prediction, prediction)[0, 1]
            ),
            "rows": len(prediction),
        }
    )
metrics = pd.DataFrame(metric_rows)

by_well_base = pd.DataFrame(
    {
        "well": base_frame["well"].astype(str),
        "target_tvt": target_tvt,
        "parent_sqerr": np.square(target_tvt - parent_prediction),
        "xgboost_sqerr": np.square(target_tvt - xgb_prediction),
        "blend_sqerr": np.square(target_tvt - blend_prediction),
    }
)
by_well = by_well_base.groupby("well", as_index=False).agg(
    rows=("target_tvt", "size"),
    parent_mse=("parent_sqerr", "mean"),
    xgboost_mse=("xgboost_sqerr", "mean"),
    blend_mse=("blend_sqerr", "mean"),
)
by_well["parent_lgb_mean_rmse"] = np.sqrt(by_well.pop("parent_mse"))
by_well["xgboost_rmse"] = np.sqrt(by_well.pop("xgboost_mse"))
by_well["blend_rmse"] = np.sqrt(by_well.pop("blend_mse"))
by_well["delta_rmse"] = by_well["xgboost_rmse"] - by_well["parent_lgb_mean_rmse"]
by_well = by_well.sort_values("delta_rmse", ascending=False).reset_index(drop=True)

stress_lookup = stress_metrics.set_index("surface")
guard = {
    "raw_overall_improvement": bool(stress_lookup.at["overall", "delta_rmse"] < 0.0),
    "rows_1000_plus_non_worse": bool(stress_lookup.at["1000_plus", "delta_rmse"] <= 0.0),
    "hidden_like_spatial_non_worse": bool(
        stress_lookup.at["hidden_like_spatial", "delta_rmse"] <= 0.0
    ),
    "hidden_like_typewell_purged_non_worse": bool(
        stress_lookup.at["hidden_like_typewell_purged", "delta_rmse"] <= 0.0
    ),
    "worst_well_regression_within_limit": bool(
        by_well["delta_rmse"].max()
        <= float(nested(CONFIG, "model.adoption_guards.max_worst_well_regression"))
    ),
    "improved_folds_at_least_minimum": bool(
        int((fold_metrics["delta_rmse"] < 0.0).sum())
        >= int(nested(CONFIG, "model.adoption_guards.min_improved_folds"))
    ),
    "worst_well_regression": float(by_well["delta_rmse"].max()),
    "improved_folds": int((fold_metrics["delta_rmse"] < 0.0).sum()),
}
guard["adoption_supported"] = bool(
    all(value for value in guard.values() if isinstance(value, bool))
)
display(metrics)
display(stress_metrics)
display(blend_readout)
display(by_well.head(30))
print(json.dumps(guard, indent=2))

# %% [markdown]
# ## 8. Metrics and generated artifacts

# %%
importance = pd.DataFrame(importance_rows)
importance_mean = (
    importance.groupby(["model", "feature"], as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        fold_records=("importance", "size"),
    )
    .sort_values(["model", "mean_importance"], ascending=[True, False])
)
plot_frame = importance_mean.nlargest(
    int(nested(CONFIG, "model.training.top_n_importance")), "mean_importance"
).sort_values("mean_importance")
fig, ax = plt.subplots(figsize=(12, max(6, 0.24 * len(plot_frame))))
ax.barh(plot_frame["feature"], plot_frame["mean_importance"], color="#2f6f8f")
ax.set_title(f"{OUTPUT_PREFIX}: mean XGBoost feature importance")
ax.set_xlabel("mean feature_importances_")
fig.tight_layout()
importance_plot_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png"
fig.savefig(importance_plot_path, dpi=160)
plt.close(fig)

oof_output = pd.DataFrame(
    {
        "id": base_frame["id"].astype(str),
        "well": base_frame["well"].astype(str),
        "fold": fold_assignment,
        "last_known_tvt": last_known_tvt,
        "target_tvt": target_tvt,
        "parent_lgb_mean_pred_tvt": parent_prediction,
        "xgboost_pred_tvt": xgb_prediction,
        "fixed_blend_xgb_w0p25_pred_tvt": blend_prediction,
    }
)

metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.csv"
fold_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fold_metrics.csv"
bucket_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
hidden_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv"
by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_by_well.csv"
blend_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_blend_readout.csv"
importance_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
oof_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
guard_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_guard.json"
manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"

metrics.to_csv(metrics_path, index=False)
fold_metrics.to_csv(fold_metrics_path, index=False)
bucket_metrics.to_csv(bucket_metrics_path, index=False)
hidden_like_metrics.to_csv(hidden_metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
blend_readout.to_csv(blend_path, index=False)
importance_mean.to_csv(importance_path, index=False)
oof_output.to_csv(oof_path, index=False, compression="gzip")
guard_path.write_text(json.dumps(guard, indent=2))

model_manifest = {
    "experiment": nested(CONFIG, "experiment.name"),
    "parent": nested(CONFIG, "lineage.parent"),
    "variant": active_variants[0],
    "public_notebook": nested(CONFIG, "model.public_source.notebook"),
    "public_notebook_sha256": sha256_path(public_notebook_path),
    "xgboost_version": xgboost_version,
    "params": public_params,
    "folds": n_folds,
    "model_count": len(model_rows),
    "control_retraining": False,
    "selector_retraining": False,
    "feature_count": len(all_features),
    "base_feature_content_sha256": base_feature_content_sha,
    "models": model_rows,
}
manifest_path.write_text(json.dumps(model_manifest, indent=2))

artifact_sha = {
    "parameter_audit": sha256_path(parameter_audit_path),
    "feature_schema": sha256_path(feature_schema_path),
    "metrics": sha256_path(metrics_path),
    "fold_metrics": sha256_path(fold_metrics_path),
    "bucket_metrics": sha256_path(bucket_metrics_path),
    "hidden_like_metrics": sha256_path(hidden_metrics_path),
    "by_well": sha256_path(by_well_path),
    "blend_readout": sha256_path(blend_path),
    "feature_importance_mean": sha256_path(importance_path),
    "oof_decompressed": sha256_path(oof_path, decompressed=True),
    "guard": sha256_path(guard_path),
    "model_manifest": sha256_path(manifest_path),
    "frozen_parent_oof_decompressed": sha256_path(parent_oof_path, decompressed=True),
}
summary = {
    "experiment": nested(CONFIG, "experiment.name"),
    "status": "train_completed_guard_pass"
    if guard["adoption_supported"]
    else "train_completed_guard_failed",
    "route": nested(CONFIG, "experiment.route"),
    "parent": nested(CONFIG, "lineage.parent"),
    "active_variants": 1,
    "xgboost_configs": 1,
    "folds": n_folds,
    "boosters": len(model_rows),
    "control_retraining": False,
    "selector_retraining": False,
    "feature_count": len(all_features),
    "base_feature_count": len(base_features),
    "selector_feature_count": len(selector_features),
    "primary_parent_rmse": parent_rmse,
    "primary_xgboost_rmse": rmse(target_tvt, xgb_prediction),
    "primary_delta_rmse": rmse(target_tvt, xgb_prediction) - parent_rmse,
    "metrics": metrics.to_dict("records"),
    "guard": guard,
    "input_sha256": {
        "public_notebook": sha256_path(public_notebook_path),
        "frozen_parent_oof_decompressed": sha256_path(parent_oof_path, decompressed=True),
        "hidden_like_assignment": sha256_path(hidden_assignment_path),
        "selector_scores_decompressed": {
            str(index): sha256_path(path, decompressed=True)
            for index, path in enumerate(score_paths)
        },
    },
    "artifact_sha256": artifact_sha,
    "artifacts": {
        "parameter_audit": parameter_audit_path.name,
        "metrics": metrics_path.name,
        "fold_metrics": fold_metrics_path.name,
        "bucket_metrics": bucket_metrics_path.name,
        "hidden_like_metrics": hidden_metrics_path.name,
        "by_well": by_well_path.name,
        "blend_readout": blend_path.name,
        "feature_importance_mean": importance_path.name,
        "feature_importance_plot": importance_plot_path.name,
        "oof_predictions": oof_path.name,
        "feature_schema": feature_schema_path.name,
        "model_manifest": manifest_path.name,
        "guard": guard_path.name,
    },
    "elapsed_seconds": round(time.time() - STARTED, 3),
}
summary_path.write_text(json.dumps(summary, indent=2))
summary_for_display = {**summary, "summary_sha256": sha256_path(summary_path)}
print(json.dumps(summary_for_display, indent=2), flush=True)
display(importance_mean.head(60))
