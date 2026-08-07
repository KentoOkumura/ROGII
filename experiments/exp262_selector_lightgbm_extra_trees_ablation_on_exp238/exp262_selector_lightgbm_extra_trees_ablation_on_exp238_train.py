# %% [markdown]
# # exp262 selector LightGBM extra-trees ablation on exp238 train
#
# This CPU notebook changes only `extra_trees=True` in the exp238 nested
# candidate-error selector. Historical exp238 nested scores and model metadata
# are frozen controls. The downstream exp218 TVT LightGBM is not trained here.

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Configuration, cost, and single-parameter contract
# 3. Frozen exp238 candidate and feature surface
# 4. Nested fold and historical-control contracts
# 5. Extra-trees nested selector training
# 6. Candidate-score and fixed-selection readouts
# 7. Guard, metrics, and generated artifacts

# %%
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.model_selection import GroupKFold

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp262_selector_lightgbm_extra_trees_ablation_on_exp238"
    )
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_PREFIX = str(CONFIG["audit"]["output_prefix"])
HISTORICAL_PREFIX = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = __import__("gzip").open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame, columns: list[str], *, chunk_rows: int = 50_000
) -> str:
    digest = hashlib.sha256()
    schema = [(column, str(frame[column].dtype)) for column in columns]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        hashed = pd.util.hash_pandas_object(
            frame.iloc[start:stop][columns], index=False, categorize=True
        ).to_numpy(np.uint64)
        digest.update(hashed.tobytes())
    return digest.hexdigest()


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
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


exp237 = import_file(
    "exp237_source",
    [
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        Path("/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "hmm_exp226_candidate_selector_on_exp183.py",
    ],
    reset_settings=True,
)

# %% [markdown]
# ## 2. Configuration, cost, and single-parameter contract

# %%
model_cfg = CONFIG["model"]
selector_cfg = model_cfg["selector"]
expected_scope = str(model_cfg["required_approval_scope"])
actual_scope = model_cfg.get("approved_scope")
cost_contract = {
    "experiment": CONFIG["experiment"]["name"],
    "route": CONFIG["experiment"]["route"],
    "runtime": "CPU",
    "active_variants": len(model_cfg["active_variants"]),
    "selector_configs": int(model_cfg["active_selector_configs"]),
    "outer_folds": int(model_cfg["outer_folds"]),
    "inner_folds": int(model_cfg["inner_folds"]),
    "selector_boosters": int(model_cfg["selector_boosters"]),
    "historical_control_retraining": bool(model_cfg["parent_control_retraining"]),
    "downstream_retraining": bool(model_cfg["downstream_retraining"]),
    "run_approved": bool(model_cfg["run_approved"]),
    "approved_scope": actual_scope,
    "required_approval_scope": expected_scope,
}
if cost_contract["active_variants"] != 1:
    raise ValueError("exactly one active selector variant is required")
if cost_contract["selector_configs"] != 1:
    raise ValueError("exactly one selector config is required")
if cost_contract["selector_boosters"] != 20:
    raise ValueError("the initial probe must contain exactly 20 CPU boosters")
if cost_contract["historical_control_retraining"] or cost_contract["downstream_retraining"]:
    raise ValueError("control and downstream retraining are forbidden in this notebook")
print(json.dumps(cost_contract, indent=2))

exp238_config_path = next(
    (
        path
        for path in [
            Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
            / "config.yaml",
            PACKAGE_DIR / "exp238_source/config.yaml",
        ]
        if path.exists()
    ),
    None,
)
if exp238_config_path is None:
    raise FileNotFoundError("canonical exp238 config is required for parameter parity")
exp238_config = yaml.safe_load(exp238_config_path.read_text())
historical_control_params = dict(exp238_config["model"]["selector"]["params"])
control_params = dict(selector_cfg["params"])
if control_params != historical_control_params:
    raise ValueError(
        {
            "message": "configured control params differ from canonical exp238",
            "exp238": historical_control_params,
            "exp262_control": control_params,
        }
    )
historical_contract = {
    "seed": int(exp238_config["validation"]["seed"]),
    "outer_folds": int(exp238_config["validation"]["outer_folds"]),
    "inner_folds": int(exp238_config["validation"]["inner_folds"]),
    "selector_configs": int(exp238_config["model"]["selector_configs"]),
    "selector_boosters": int(exp238_config["model"]["selector_boosters"]),
    "max_train_long_rows_per_model": int(
        exp238_config["model"]["selector"]["max_train_long_rows_per_model"]
    ),
    "max_valid_long_rows_per_model": int(
        exp238_config["model"]["selector"]["max_valid_long_rows_per_model"]
    ),
    "predict_chunk_rows": int(
        exp238_config["model"]["selector"]["predict_chunk_rows"]
    ),
}
current_contract = {
    "seed": int(CONFIG["validation"]["seed"]),
    "outer_folds": int(CONFIG["validation"]["outer_folds"]),
    "inner_folds": int(CONFIG["validation"]["inner_folds"]),
    "selector_configs": int(model_cfg["active_selector_configs"]),
    "selector_boosters": int(model_cfg["selector_boosters"]),
    "max_train_long_rows_per_model": int(
        selector_cfg["max_train_long_rows_per_model"]
    ),
    "max_valid_long_rows_per_model": int(
        selector_cfg["max_valid_long_rows_per_model"]
    ),
    "predict_chunk_rows": int(selector_cfg["predict_chunk_rows"]),
}
if current_contract != historical_contract:
    raise ValueError(
        {
            "message": "fold/sampling contract differs from canonical exp238",
            "exp238": historical_contract,
            "exp262": current_contract,
        }
    )
variant_params = {
    **historical_control_params,
    **dict(selector_cfg["changed_parameter"]),
}
all_parameter_keys = sorted(set(historical_control_params) | set(variant_params))
parameter_audit = pd.DataFrame(
    [
        {
            "parameter": key,
            "historical_exp238_value": json.dumps(
                historical_control_params.get(key), sort_keys=True
            ),
            "extra_trees_value": json.dumps(variant_params.get(key), sort_keys=True),
            "changed": historical_control_params.get(key) != variant_params.get(key),
        }
        for key in all_parameter_keys
    ]
)
changed_keys = parameter_audit.loc[parameter_audit["changed"], "parameter"].tolist()
if changed_keys != ["extra_trees"] or variant_params["extra_trees"] is not True:
    raise ValueError(f"only extra_trees=True may differ, got {changed_keys}")
if selector_cfg["objective"] != "regression_l1_candidate_absolute_error":
    raise ValueError("selector objective changed from exp238")
display(parameter_audit)
print(
    json.dumps(
        {
            "exp238_config": str(exp238_config_path),
            "exp238_config_sha256": sha256_path(exp238_config_path),
            "fold_and_sampling_contract": historical_contract,
        },
        indent=2,
    )
)

if not bool(model_cfg["run_approved"]) or actual_scope != expected_scope:
    raise RuntimeError(
        "Kaggle CPU train is not approved. Confirm the 20-booster scope and set "
        "model.run_approved=true plus model.approved_scope to the required value."
    )

# %% [markdown]
# ## 3. Frozen exp238 candidate and feature surface

# %%
parent_config = exp237.load_config()
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(parent_config)
required_columns = exp237.build_required_columns(parent_config, candidates)
frame, source_meta = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=required_columns,
    max_rows=None,
)
frame, enrichment_columns, enrichment_meta = exp237.add_feature_enrichment(
    frame, parent_config, max_rows=None
)
frame, cluster_columns, cluster_meta = exp237.add_cluster_prior_confidence_features(
    frame, parent_config, max_rows=None
)
frame, external_columns, external_meta = exp237.add_hmm_exp226_candidate_sources(
    frame, parent_config
)
frame, engineered_columns, _, _ = exp237.add_candidate_labels_and_features(
    frame, candidates, include_candidate_values=False
)
context_columns = exp237.select_numeric_feature_columns(
    frame,
    parent_config,
    [*engineered_columns, *enrichment_columns, *cluster_columns, *external_columns],
)
candidate_columns = [item.column for item in candidates]
candidate_names = [item.name for item in candidates]
expected_candidates = [str(value) for value in selector_cfg["candidates"]]
validation_cfg = CONFIG["validation"]
if candidate_names != expected_candidates:
    raise ValueError(
        {"message": "candidate bank differs from exp238", "actual": candidate_names}
    )
if len(context_columns) != int(validation_cfg["expected_context_feature_count"]):
    raise ValueError(
        f"context feature count changed: {len(context_columns)} != "
        f"{validation_cfg['expected_context_feature_count']}"
    )
if len(frame) != int(validation_cfg["expected_rows"]):
    raise ValueError(f"row count changed: {len(frame)}")
if int(frame["well"].nunique()) != int(validation_cfg["expected_wells"]):
    raise ValueError(f"well count changed: {frame['well'].nunique()}")

candidate_long_extra_columns = [
    "candidate_code",
    "candidate_minus_anchor",
    "candidate_abs_minus_anchor",
]
selector_feature_columns = [*context_columns, *candidate_long_extra_columns]
if len(candidate_long_extra_columns) != int(
    validation_cfg["expected_candidate_long_extra_feature_count"]
):
    raise AssertionError("candidate-long adapter changed")

context_schema = pd.DataFrame(
    {
        "position": np.arange(len(selector_feature_columns), dtype=np.int32),
        "feature": selector_feature_columns,
        "source": ["context"] * len(context_columns) + ["candidate_adapter"] * 3,
    }
)
content_columns = list(
    dict.fromkeys(
        [
            "id",
            "well",
            "last_known_tvt",
            "target",
            *candidate_columns,
            *context_columns,
        ]
    )
)
selector_feature_content_sha = dataframe_content_sha256(frame, content_columns)
print(
    json.dumps(
        {
            "rows": len(frame),
            "wells": int(frame.well.nunique()),
            "candidate_names": candidate_names,
            "candidate_columns": candidate_columns,
            "context_feature_count": len(context_columns),
            "selector_feature_count": len(selector_feature_columns),
            "selector_feature_content_sha256": selector_feature_content_sha,
            "source": source_meta,
            "enrichment": enrichment_meta,
            "cluster": cluster_meta,
            "external": external_meta,
        },
        indent=2,
        default=str,
    )
)

# %% [markdown]
# ## 4. Nested fold and historical-control contracts

# %%
def deterministic_outer_inner_splits(
    data: pd.DataFrame, outer_folds: int, inner_folds: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    groups = data["well"].astype(str).to_numpy()
    outer = list(GroupKFold(outer_folds).split(data, groups=groups))
    nested: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for outer_fold, (outer_train, outer_valid) in enumerate(outer):
        train_wells = set(groups[outer_train])
        valid_wells = set(groups[outer_valid])
        if train_wells & valid_wells:
            raise AssertionError(f"outer fold {outer_fold}: well overlap")
        local_splits = list(
            GroupKFold(inner_folds).split(outer_train, groups=groups[outer_train])
        )
        fold_rows = []
        covered = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(local_splits):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            if set(groups[inner_train]) & set(groups[inner_valid]):
                raise AssertionError(
                    f"outer {outer_fold} inner {inner_fold}: well overlap"
                )
            if set(groups[inner_train]) & valid_wells:
                raise AssertionError("outer-valid well leaked into selector train")
            fold_rows.append((inner_train, inner_valid))
            covered.append(inner_valid)
        if not np.array_equal(
            np.sort(np.concatenate(covered)), np.sort(outer_train)
        ):
            raise AssertionError(f"outer fold {outer_fold}: inner OOF coverage mismatch")
        nested.append(fold_rows)
    return outer, nested


def well_set_sha(data: pd.DataFrame, rows: np.ndarray) -> str:
    wells = sorted(data.iloc[rows]["well"].astype(str).unique())
    return hashlib.sha256(json.dumps(wells, separators=(",", ":")).encode()).hexdigest()


outer_splits, inner_splits = deterministic_outer_inner_splits(
    frame,
    int(validation_cfg["outer_folds"]),
    int(validation_cfg["inner_folds"]),
)
fold_records = []
for outer_fold, ((outer_train, outer_valid), local_splits) in enumerate(
    zip(outer_splits, inner_splits, strict=True)
):
    for inner_fold, (inner_train, inner_valid) in enumerate(local_splits):
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "outer_train_rows": len(outer_train),
                "outer_valid_rows": len(outer_valid),
                "inner_train_rows": len(inner_train),
                "inner_valid_rows": len(inner_valid),
                "outer_train_wells": int(frame.iloc[outer_train].well.nunique()),
                "outer_valid_wells": int(frame.iloc[outer_valid].well.nunique()),
                "inner_train_wells": int(frame.iloc[inner_train].well.nunique()),
                "inner_valid_wells": int(frame.iloc[inner_valid].well.nunique()),
                "outer_train_well_sha256": well_set_sha(frame, outer_train),
                "outer_valid_well_sha256": well_set_sha(frame, outer_valid),
                "inner_train_well_sha256": well_set_sha(frame, inner_train),
                "inner_valid_well_sha256": well_set_sha(frame, inner_valid),
            }
        )
fold_manifest = pd.DataFrame(fold_records)
display(fold_manifest)


def find_historical_selector_dir() -> Path:
    configured = Path(CONFIG["data"]["historical_selector_artifact_dir_local"])
    slug = str(CONFIG["data"]["historical_selector_kernel_slug"])
    candidates_dir = [
        configured,
        Path("/kaggle/input") / slug / "artifacts",
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/notebooks/kentookumura") / slug / "artifacts",
        Path("/kaggle/input/notebooks/kentookumura") / slug,
    ]
    score_name = f"{HISTORICAL_PREFIX}_nested_scores_outer0.csv.gz"
    resolved = next(
        (path for path in candidates_dir if (path / score_name).exists()), None
    )
    if resolved is None and Path("/kaggle/input").exists():
        matches = list(Path("/kaggle/input").rglob(score_name))
        if len(matches) > 1:
            raise ValueError(f"ambiguous historical selector inputs: {matches}")
        resolved = matches[0].parent if matches else None
    if resolved is None:
        raise FileNotFoundError(
            "saved exp238 nested scores are required; add exp238-nested-selector-train "
            "as a Kaggle kernel source"
        )
    return resolved


def find_historical_file(directory: Path, filename: str) -> Path:
    direct = directory / filename
    if direct.exists():
        return direct
    matches = list(directory.parent.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, got {matches}")
    return matches[0]


historical_dir = find_historical_selector_dir()
historical_summary_path = find_historical_file(
    historical_dir, f"{HISTORICAL_PREFIX}_selector_summary.json"
)
historical_model_manifest_path = find_historical_file(
    historical_dir, f"{HISTORICAL_PREFIX}_selector_model_manifest.csv"
)
historical_summary = json.loads(historical_summary_path.read_text())
historical_model_manifest = pd.read_csv(historical_model_manifest_path)
if int(historical_summary.get("selector_model_count", -1)) != 20:
    raise ValueError("historical summary does not describe 20 selector models")
if len(historical_model_manifest) != 20:
    raise ValueError("historical model manifest must contain 20 rows")
if historical_summary.get("candidate_columns") != candidate_columns:
    raise ValueError("historical candidate columns differ")
if historical_summary.get("context_columns") != context_columns:
    raise ValueError("historical context feature order differs")
if historical_model_manifest["feature_count"].astype(int).nunique() != 1:
    raise ValueError("historical selector feature counts are inconsistent")
if int(historical_model_manifest["feature_count"].iloc[0]) != len(
    selector_feature_columns
):
    raise ValueError("historical selector feature count differs")
historical_feature_names = json.loads(
    historical_model_manifest["feature_names_json"].iloc[0]
)
if historical_feature_names != selector_feature_columns:
    raise ValueError("historical selector feature-name order differs")


def load_historical_scores(
    directory: Path,
) -> tuple[list[dict[str, np.ndarray]], pd.DataFrame]:
    score_columns = [f"pred_error__{column}" for column in candidate_columns]
    items = []
    manifest = []
    for outer_fold, (expected_train, expected_valid) in enumerate(outer_splits):
        path = directory / f"{HISTORICAL_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
        artifact = pd.read_csv(
            path,
            usecols=["row_index", "role", "id", "well", *score_columns],
            dtype={"row_index": np.int32, "role": "category", "id": str, "well": str},
        )
        if len(artifact) != len(frame) or artifact["row_index"].duplicated().any():
            raise ValueError(f"historical fold {outer_fold}: row contract failed")
        aligned = artifact.sort_values("row_index")
        if not aligned[["id", "well"]].reset_index(drop=True).equals(
            frame[["id", "well"]].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"historical fold {outer_fold}: id/well alignment failed")
        train = artifact.loc[artifact.role.eq("train")].sort_values("row_index")
        valid = artifact.loc[artifact.role.eq("valid")].sort_values("row_index")
        train_rows = train.row_index.to_numpy(np.int64)
        valid_rows = valid.row_index.to_numpy(np.int64)
        if not np.array_equal(train_rows, np.sort(expected_train)):
            raise ValueError(f"historical fold {outer_fold}: train role differs")
        if not np.array_equal(valid_rows, np.sort(expected_valid)):
            raise ValueError(f"historical fold {outer_fold}: valid role differs")
        valid_scores = valid[score_columns].to_numpy(np.float32)
        if not np.isfinite(valid_scores).all():
            raise ValueError(f"historical fold {outer_fold}: non-finite score")
        items.append({"outer_valid": valid_rows, "valid_scores": valid_scores})
        manifest.append(
            {
                "kind": "nested_score",
                "outer_fold": outer_fold,
                "file": path.name,
                "rows": len(artifact),
                "valid_rows": len(valid),
                "sha256": sha256_path(path),
                "sha256_decompressed": sha256_path(path, decompressed=True),
            }
        )
        del aligned, train, valid, artifact
        gc.collect()
    return items, pd.DataFrame(manifest)


historical_nested, historical_input_manifest = load_historical_scores(historical_dir)
historical_input_manifest = pd.concat(
    [
        pd.DataFrame(
            [
                {
                    "kind": "selector_summary",
                    "outer_fold": pd.NA,
                    "file": historical_summary_path.name,
                    "rows": 1,
                    "valid_rows": pd.NA,
                    "sha256": sha256_path(historical_summary_path),
                    "sha256_decompressed": pd.NA,
                },
                {
                    "kind": "model_manifest",
                    "outer_fold": pd.NA,
                    "file": historical_model_manifest_path.name,
                    "rows": len(historical_model_manifest),
                    "valid_rows": pd.NA,
                    "sha256": sha256_path(historical_model_manifest_path),
                    "sha256_decompressed": pd.NA,
                },
            ]
        ),
        historical_input_manifest,
    ],
    ignore_index=True,
)
display(historical_input_manifest)

# %% [markdown]
# ## 5. Extra-trees nested selector training

# %%
def candidate_long(
    data: pd.DataFrame,
    rows: np.ndarray,
    *,
    with_target: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    blocks: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    true_tvt = (
        data["last_known_tvt"].to_numpy(np.float32)
        + data["target"].to_numpy(np.float32)
    )
    anchor = data["last_known_tvt"].to_numpy(np.float32)
    for code, column in enumerate(candidate_columns):
        values = data[column].to_numpy(np.float32)[rows]
        block = data.iloc[rows][context_columns].reset_index(drop=True).copy()
        block["candidate_code"] = np.float32(code)
        block["candidate_minus_anchor"] = values - anchor[rows]
        block["candidate_abs_minus_anchor"] = np.abs(values - anchor[rows])
        blocks.append(block)
        if with_target:
            labels.append(np.abs(values - true_tvt[rows]).astype(np.float32))
    long = pd.concat(blocks, ignore_index=True)
    target = np.concatenate(labels) if labels else None
    if long.columns.tolist() != selector_feature_columns:
        raise ValueError("candidate-long feature order differs from historical exp238")
    return long, target


def bounded_base_rows(
    rows: np.ndarray, max_long_rows: int | None, seed: int
) -> np.ndarray:
    if max_long_rows is None:
        return rows
    max_base_rows = max(1, int(max_long_rows) // len(candidate_columns))
    if len(rows) <= max_base_rows:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=max_base_rows, replace=False))


def predict_candidate_errors(model: lgb.LGBMRegressor, rows: np.ndarray) -> np.ndarray:
    parts = []
    chunk_rows = int(selector_cfg["predict_chunk_rows"])
    for start in range(0, len(rows), chunk_rows):
        chunk = rows[start : start + chunk_rows]
        long, _ = candidate_long(frame, chunk, with_target=False)
        pred = model.predict(long, num_iteration=model.best_iteration_)
        parts.append(np.asarray(pred, dtype=np.float32).reshape(len(candidate_columns), -1).T)
        del long, pred
        gc.collect()
    return np.concatenate(parts, axis=0)


def fit_extra_trees_nested_selector() -> tuple[
    list[dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame
]:
    outputs = []
    manifest_rows = []
    importance_parts = []
    seed = int(CONFIG["reproducibility"]["seed"])
    model_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold, ((outer_train, outer_valid), local_splits) in enumerate(
        zip(outer_splits, inner_splits, strict=True)
    ):
        train_scores = np.full(
            (len(frame), len(candidate_columns)), np.nan, dtype=np.float32
        )
        valid_models: list[lgb.LGBMRegressor] = []
        for inner_fold, (train_rows, valid_rows) in enumerate(local_splits):
            fit_train_rows = bounded_base_rows(
                train_rows,
                int(selector_cfg["max_train_long_rows_per_model"]),
                seed + 10_000 * outer_fold + 100 * inner_fold,
            )
            fit_valid_rows = bounded_base_rows(
                valid_rows,
                int(selector_cfg["max_valid_long_rows_per_model"]),
                seed + 20_000 * outer_fold + 100 * inner_fold,
            )
            x_train, y_train = candidate_long(frame, fit_train_rows, with_target=True)
            x_valid, y_valid = candidate_long(frame, fit_valid_rows, with_target=True)
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                random_state=seed + 100 * outer_fold + inner_fold,
                **variant_params,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(int(selector_cfg["early_stopping_rounds"])),
                    lgb.log_evaluation(int(selector_cfg["log_evaluation_period"])),
                ],
            )
            model_path = model_dir / f"selector_outer{outer_fold}_inner{inner_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
            train_scores[valid_rows] = predict_candidate_errors(model, valid_rows)
            feature_names = model.booster_.feature_name()
            if feature_names != selector_feature_columns:
                raise ValueError("trained model feature order differs from exp238")
            split_importance = model.booster_.feature_importance(
                importance_type="split", iteration=model.best_iteration_
            )
            gain_importance = model.booster_.feature_importance(
                importance_type="gain", iteration=model.best_iteration_
            )
            importance_parts.append(
                pd.DataFrame(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "feature": feature_names,
                        "split_importance": split_importance,
                        "gain_importance": gain_importance,
                    }
                )
            )
            manifest_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_rows": len(train_rows),
                    "valid_rows": len(valid_rows),
                    "fit_train_base_rows": len(fit_train_rows),
                    "fit_valid_base_rows": len(fit_valid_rows),
                    "fit_train_long_rows": len(x_train),
                    "fit_valid_long_rows": len(x_valid),
                    "train_wells": int(frame.iloc[train_rows].well.nunique()),
                    "valid_wells": int(frame.iloc[valid_rows].well.nunique()),
                    "random_state": seed + 100 * outer_fold + inner_fold,
                    "best_iteration": int(model.best_iteration_),
                    "extra_trees": True,
                    "feature_count": len(feature_names),
                    "feature_names_json": json.dumps(feature_names),
                    "file": str(model_path),
                    "sha256": sha256_path(model_path),
                }
            )
            valid_models.append(model)
            del x_train, x_valid, y_train, y_valid
            gc.collect()
        if not np.isfinite(train_scores[outer_train]).all():
            raise AssertionError(f"outer fold {outer_fold}: incomplete inner OOF scores")
        valid_scores = np.mean(
            [predict_candidate_errors(model, outer_valid) for model in valid_models],
            axis=0,
        ).astype(np.float32)
        if not np.isfinite(valid_scores).all():
            raise ValueError(f"outer fold {outer_fold}: non-finite outer-valid scores")
        outputs.append(
            {
                "outer_train": outer_train,
                "outer_valid": outer_valid,
                "train_scores": train_scores[outer_train].copy(),
                "valid_scores": valid_scores,
            }
        )
        del train_scores, valid_models, model
        gc.collect()
    importance = pd.concat(importance_parts, ignore_index=True)
    return outputs, pd.DataFrame(manifest_rows), importance


current_nested, model_manifest, fold_importance = fit_extra_trees_nested_selector()
if len(model_manifest) != 20:
    raise ValueError(f"expected 20 new selector models, got {len(model_manifest)}")

# %% [markdown]
# ## 6. Candidate-score and fixed-selection readouts

# %%
def pool_outer_valid(items: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(
        (len(frame), len(candidate_columns)), np.nan, dtype=np.float32
    )
    folds = np.full(len(frame), -1, dtype=np.int8)
    for outer_fold, item in enumerate(items):
        rows = item["outer_valid"]
        if np.any(folds[rows] != -1):
            raise ValueError("outer-valid rows overlap")
        scores[rows] = item["valid_scores"]
        folds[rows] = outer_fold
    if not np.isfinite(scores).all() or np.any(folds < 0):
        raise ValueError("pooled outer-valid score coverage is incomplete")
    return scores, folds


historical_scores, historical_outer_fold = pool_outer_valid(historical_nested)
current_scores, current_outer_fold = pool_outer_valid(current_nested)
if not np.array_equal(historical_outer_fold, current_outer_fold):
    raise ValueError("historical and current outer-fold assignments differ")

candidate_values = frame[candidate_columns].to_numpy(np.float32)
true_tvt = (
    frame["last_known_tvt"].to_numpy(np.float32)
    + frame["target"].to_numpy(np.float32)
)
fallback = frame[candidate_columns[candidate_names.index("likpf_mean")]].to_numpy(
    np.float32
)


def candidate_score_metrics(source: str, scores: np.ndarray) -> dict[str, Any]:
    total_abs_error = 0.0
    total_logloss = 0.0
    total_top1_accuracy = 0
    total_top1_regret = 0.0
    total_top1_true_rank = 0.0
    total_margin = 0.0
    pair_correct = 0
    pair_total = 0
    temperature = float(
        CONFIG["model"]["evaluation"]["oracle_candidate_logloss_temperature_ft"]
    )
    chunk_rows = 100_000
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        score = scores[start:stop].astype(np.float64)
        actual = np.abs(
            candidate_values[start:stop].astype(np.float64)
            - true_tvt[start:stop, None].astype(np.float64)
        )
        oracle = np.argmin(actual, axis=1)
        selected = np.argmin(score, axis=1)
        total_abs_error += float(np.abs(score - actual).sum())
        logits = -score / temperature
        max_logits = logits.max(axis=1, keepdims=True)
        log_denom = max_logits[:, 0] + np.log(
            np.exp(logits - max_logits).sum(axis=1)
        )
        total_logloss += float(
            (-logits[np.arange(len(logits)), oracle] + log_denom).sum()
        )
        selected_error = actual[np.arange(len(actual)), selected]
        oracle_error = actual[np.arange(len(actual)), oracle]
        total_top1_accuracy += int(np.sum(selected == oracle))
        total_top1_regret += float((selected_error - oracle_error).sum())
        total_top1_true_rank += float(
            np.sum(np.sum(actual < selected_error[:, None], axis=1))
        )
        ordered = np.partition(score, kth=1, axis=1)[:, :2]
        total_margin += float((ordered.max(axis=1) - ordered.min(axis=1)).sum())
        for left in range(len(candidate_columns)):
            for right in range(left + 1, len(candidate_columns)):
                actual_delta = actual[:, left] - actual[:, right]
                valid = actual_delta != 0.0
                pair_correct += int(
                    np.sum(
                        ((score[:, left] - score[:, right]) < 0.0)[valid]
                        == (actual_delta < 0.0)[valid]
                    )
                )
                pair_total += int(valid.sum())
    rows = len(frame)
    candidate_rows = rows * len(candidate_columns)
    return {
        "source": source,
        "rows": rows,
        "candidate_rows": candidate_rows,
        "candidate_error_mae": total_abs_error / candidate_rows,
        "oracle_candidate_logloss": total_logloss / rows,
        "pairwise_rank_accuracy": pair_correct / pair_total,
        "oracle_top1_accuracy": total_top1_accuracy / rows,
        "top1_regret_mean": total_top1_regret / rows,
        "top1_true_rank_mean": total_top1_true_rank
        / (rows * (len(candidate_columns) - 1)),
        "predicted_error_margin_mean": total_margin / rows,
        "logloss_temperature_ft": temperature,
    }


candidate_metrics = pd.DataFrame(
    [
        candidate_score_metrics("historical_exp238", historical_scores),
        candidate_score_metrics("extra_trees_true", current_scores),
    ]
)


def score_surface_comparison(
    historical: np.ndarray, current: np.ndarray
) -> dict[str, Any]:
    n_candidates = len(candidate_columns)
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    abs_difference_sum = 0.0
    abs_difference_max = 0.0
    spearman_sum = 0.0
    top1_agreement = 0
    chunk_rows = 100_000
    for start in range(0, len(historical), chunk_rows):
        stop = min(start + chunk_rows, len(historical))
        x = historical[start:stop].astype(np.float64)
        y = current[start:stop].astype(np.float64)
        count += x.size
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xx += float(np.square(x).sum())
        sum_yy += float(np.square(y).sum())
        sum_xy += float((x * y).sum())
        difference = np.abs(y - x)
        abs_difference_sum += float(difference.sum())
        abs_difference_max = max(abs_difference_max, float(difference.max()))
        hist_rank = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.int16)
        curr_rank = np.argsort(np.argsort(y, axis=1), axis=1).astype(np.int16)
        rank_delta_sq = (hist_rank - curr_rank) ** 2
        spearman_sum += float(
            np.sum(
                1.0
                - 6.0 * rank_delta_sq.sum(axis=1)
                / (n_candidates * (n_candidates * n_candidates - 1))
            )
        )
        top1_agreement += int(
            np.sum(np.argmin(x, axis=1) == np.argmin(y, axis=1))
        )
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_xx - sum_x * sum_x / count
    variance_y = sum_yy - sum_y * sum_y / count
    pearson = float(covariance / np.sqrt(variance_x * variance_y))
    return {
        "comparison": "extra_trees_true_vs_historical_exp238",
        "flat_pearson": pearson,
        "mean_rowwise_spearman": spearman_sum / len(historical),
        "top1_agreement": top1_agreement / len(historical),
        "mean_abs_score_difference": abs_difference_sum / count,
        "max_abs_score_difference": abs_difference_max,
    }


score_comparison = pd.DataFrame(
    [score_surface_comparison(historical_scores, current_scores)]
)

viterbi_specs = exp237.variant_specs_from_config(parent_config)
if len(viterbi_specs) != 1:
    raise ValueError(f"expected one fixed exp237 Viterbi rule, got {len(viterbi_specs)}")
viterbi_spec = viterbi_specs[0]
fixed_cfg = validation_cfg["fixed_viterbi"]
for key in [
    "switch_penalty",
    "nondefault_bias",
    "jump_penalty_weight",
    "jump_free_ft",
    "jump_scale_ft",
    "max_abs_delta_vs_default",
    "max_pf_ancc_std",
    "min_md_since",
    "min_segment_len",
]:
    actual = getattr(viterbi_spec, key)
    expected = fixed_cfg[key]
    if float(actual) != float(expected):
        raise ValueError(f"fixed Viterbi parameter changed: {key}={actual} != {expected}")
default_idx = candidate_names.index(str(fixed_cfg["default_candidate"]))
allowed_switch_idx = np.asarray(
    [
        candidate_names.index(str(name))
        for name in fixed_cfg["allowed_switch_candidates"]
    ],
    dtype=np.int16,
)
historical_top1 = np.argmin(historical_scores, axis=1).astype(np.int16)
current_top1 = np.argmin(current_scores, axis=1).astype(np.int16)
historical_viterbi = exp237.viterbi_select(
    frame=frame,
    predicted_error=historical_scores,
    candidate_values=candidate_values,
    candidate_names=candidate_names,
    default_idx=default_idx,
    allowed_switch_idx=allowed_switch_idx,
    spec=viterbi_spec,
)
current_viterbi = exp237.viterbi_select(
    frame=frame,
    predicted_error=current_scores,
    candidate_values=candidate_values,
    candidate_names=candidate_names,
    default_idx=default_idx,
    allowed_switch_idx=allowed_switch_idx,
    spec=viterbi_spec,
)

subgroup_context, subgroup_meta = exp237.subgroup_context(frame, parent_config)
if not subgroup_context[["id", "well"]].astype(str).reset_index(drop=True).equals(
    frame[["id", "well"]].astype(str).reset_index(drop=True)
):
    raise ValueError("exp115 subgroup context is not row-aligned")
required_roles = [
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
]
missing_roles = [column for column in required_roles if column not in subgroup_context]
if missing_roles:
    raise ValueError(f"hidden-like role columns are required: {missing_roles}")

mask_defs: dict[str, np.ndarray] = {
    "global": np.ones(len(frame), dtype=bool),
    "000_050": frame["md_since"].to_numpy(np.float32) <= 50.0,
    "1000_plus": frame["md_since"].to_numpy(np.float32) >= 1000.0,
    "exp115_spatial_valid": subgroup_context[
        "verification_like_spatial_role"
    ].eq("valid").to_numpy(),
    "exp115_typewell_purged_valid": subgroup_context[
        "verification_like_typewell_purged_role"
    ].eq("valid").to_numpy(),
}
for outer_fold in range(int(validation_cfg["outer_folds"])):
    mask_defs[f"outer_fold_{outer_fold}"] = current_outer_fold == outer_fold


def selected_values(selected_idx: np.ndarray) -> np.ndarray:
    return candidate_values[np.arange(len(frame)), selected_idx].astype(np.float32)


selection_index = {
    ("historical_exp238", "fixed_top1"): historical_top1,
    ("historical_exp238", "fixed_viterbi"): historical_viterbi,
    ("extra_trees_true", "fixed_top1"): current_top1,
    ("extra_trees_true", "fixed_viterbi"): current_viterbi,
}
selection_rows = []
by_well_parts = []
truth = true_tvt.astype(np.float64)
fallback_error_sq = (fallback.astype(np.float64) - truth) ** 2
for (source, mode), selected_idx in selection_index.items():
    prediction = selected_values(selected_idx).astype(np.float64)
    error = prediction - truth
    for slice_name, mask in mask_defs.items():
        if not mask.any():
            raise ValueError(f"empty required evaluation slice: {slice_name}")
        selection_rows.append(
            {
                "source": source,
                "mode": mode,
                "slice": slice_name,
                "rows": int(mask.sum()),
                "wells": int(frame.loc[mask, "well"].nunique()),
                "rmse_tvt": float(np.sqrt(np.mean(error[mask] ** 2))),
                "mae_tvt": float(np.mean(np.abs(error[mask]))),
                "within_10ft": float(np.mean(np.abs(error[mask]) <= 10.0)),
                "fallback_rmse": float(np.sqrt(np.mean(fallback_error_sq[mask]))),
            }
        )
    detail = pd.DataFrame(
        {
            "well": frame["well"].astype(str).to_numpy(),
            "error_sq": error**2,
            "fallback_error_sq": fallback_error_sq,
        }
    )
    by_well = detail.groupby("well", sort=True).agg(
        rows=("error_sq", "size"),
        selected_mse=("error_sq", "mean"),
        fallback_mse=("fallback_error_sq", "mean"),
    )
    by_well["rmse_tvt"] = np.sqrt(by_well.pop("selected_mse"))
    by_well["fallback_rmse"] = np.sqrt(by_well.pop("fallback_mse"))
    by_well["delta_rmse_vs_likpf"] = by_well["rmse_tvt"] - by_well["fallback_rmse"]
    by_well = by_well.reset_index()
    by_well.insert(0, "mode", mode)
    by_well.insert(0, "source", source)
    by_well_parts.append(by_well)

selection_metrics = pd.DataFrame(selection_rows)
by_well = pd.concat(by_well_parts, ignore_index=True)
historical_well = by_well.loc[
    by_well.source.eq("historical_exp238"), ["mode", "well", "rmse_tvt"]
].rename(columns={"rmse_tvt": "historical_exp238_rmse"})
by_well = by_well.merge(
    historical_well, on=["mode", "well"], how="left", validate="many_to_one"
)
by_well["delta_rmse_vs_historical"] = (
    by_well["rmse_tvt"] - by_well["historical_exp238_rmse"]
)

fixed_predictions = pd.DataFrame(
    {
        "id": frame["id"].astype(str),
        "well": frame["well"].astype(str),
        "outer_fold": current_outer_fold,
        "true_tvt": true_tvt,
        "likpf_mean": fallback,
        "historical_top1_candidate_index": historical_top1,
        "historical_top1_tvt": selected_values(historical_top1),
        "extra_trees_top1_candidate_index": current_top1,
        "extra_trees_top1_tvt": selected_values(current_top1),
        "historical_viterbi_candidate_index": historical_viterbi,
        "historical_viterbi_tvt": selected_values(historical_viterbi),
        "extra_trees_viterbi_candidate_index": current_viterbi,
        "extra_trees_viterbi_tvt": selected_values(current_viterbi),
    }
)
display(candidate_metrics)
display(score_comparison)
display(selection_metrics)

# %% [markdown]
# ## 7. Guard, metrics, and generated artifacts

# %%
def metric(source: str, mode: str, slice_name: str) -> float:
    selected = selection_metrics.loc[
        selection_metrics.source.eq(source)
        & selection_metrics["mode"].eq(mode)
        & selection_metrics["slice"].eq(slice_name),
        "rmse_tvt",
    ]
    if len(selected) != 1:
        raise ValueError(f"missing unique metric for {source}/{mode}/{slice_name}")
    return float(selected.iloc[0])


historical_candidate = candidate_metrics.set_index("source").loc["historical_exp238"]
current_candidate = candidate_metrics.set_index("source").loc["extra_trees_true"]
guard_cfg = validation_cfg["selector_guard"]
fold_deltas = {
    str(outer_fold): metric(
        "extra_trees_true", "fixed_viterbi", f"outer_fold_{outer_fold}"
    )
    - metric("historical_exp238", "fixed_viterbi", f"outer_fold_{outer_fold}")
    for outer_fold in range(int(validation_cfg["outer_folds"]))
}
nonworse_folds = sum(delta <= 0.0 for delta in fold_deltas.values())
current_comparison_worst = float(
    by_well.loc[
        by_well.source.eq("extra_trees_true")
        & by_well["mode"].isin(["fixed_top1", "fixed_viterbi"]),
        "delta_rmse_vs_historical",
    ].max()
)
historical_top1_worst_vs_fallback = float(
    by_well.loc[
        by_well.source.eq("historical_exp238")
        & by_well["mode"].eq("fixed_top1"),
        "delta_rmse_vs_likpf",
    ].max()
)
current_top1_worst_vs_fallback = float(
    by_well.loc[
        by_well.source.eq("extra_trees_true")
        & by_well["mode"].eq("fixed_top1"),
        "delta_rmse_vs_likpf",
    ].max()
)
historical_reference_worst = float(
    validation_cfg["historical_reference"]["worst_well_max_regression_vs_likpf"]
)
if abs(historical_top1_worst_vs_fallback - historical_reference_worst) > 1.0e-3:
    raise ValueError(
        "historical exp238 worst-well reference changed: "
        f"{historical_top1_worst_vs_fallback} != {historical_reference_worst}"
    )

hidden_slices = ["exp115_spatial_valid", "exp115_typewell_purged_valid"]
checks = {
    "score_surface_changed": float(score_comparison.mean_abs_score_difference.iloc[0])
    >= float(guard_cfg["min_mean_abs_score_difference"]),
    "candidate_error_mae_nonworse": float(current_candidate.candidate_error_mae)
    <= float(historical_candidate.candidate_error_mae)
    + float(guard_cfg["max_candidate_error_mae_delta"]),
    "oracle_candidate_logloss_nonworse": float(
        current_candidate.oracle_candidate_logloss
    )
    <= float(historical_candidate.oracle_candidate_logloss)
    + float(guard_cfg["max_oracle_candidate_logloss_delta"]),
    "pairwise_rank_accuracy_nonworse": float(
        current_candidate.pairwise_rank_accuracy
    )
    >= float(historical_candidate.pairwise_rank_accuracy)
    - float(guard_cfg["max_pairwise_rank_accuracy_drop"]),
    "top1_global_nonworse": metric("extra_trees_true", "fixed_top1", "global")
    <= metric("historical_exp238", "fixed_top1", "global")
    + float(guard_cfg["max_top1_global_rmse_delta"]),
    "viterbi_global_nonworse": metric(
        "extra_trees_true", "fixed_viterbi", "global"
    )
    <= metric("historical_exp238", "fixed_viterbi", "global")
    + float(guard_cfg["max_viterbi_global_rmse_delta"]),
    "viterbi_near_nonworse": metric(
        "extra_trees_true", "fixed_viterbi", "000_050"
    )
    <= metric("historical_exp238", "fixed_viterbi", "000_050")
    + float(guard_cfg["max_viterbi_near_rmse_delta"]),
    "viterbi_longtail_nonworse": metric(
        "extra_trees_true", "fixed_viterbi", "1000_plus"
    )
    <= metric("historical_exp238", "fixed_viterbi", "1000_plus")
    + float(guard_cfg["max_viterbi_longtail_rmse_delta"]),
    "viterbi_hidden_like_nonworse": all(
        metric("extra_trees_true", "fixed_viterbi", slice_name)
        <= metric("historical_exp238", "fixed_viterbi", slice_name)
        + float(guard_cfg["max_viterbi_hidden_like_rmse_delta"])
        for slice_name in hidden_slices
    ),
    "at_least_three_outer_folds_nonworse": nonworse_folds
    >= int(guard_cfg["min_nonworse_outer_folds"]),
    "worst_well_regression_vs_historical_bounded": current_comparison_worst
    <= float(guard_cfg["max_worst_well_regression_vs_historical"]),
    "known_worst_well_risk_not_expanded": current_top1_worst_vs_fallback
    <= historical_top1_worst_vs_fallback
    + float(guard_cfg["max_worst_well_vs_likpf_delta_from_historical"]),
}
decision = {
    "guard_pass": bool(all(checks.values())),
    "checks": checks,
    "fold_delta_rmse": fold_deltas,
    "nonworse_outer_folds": nonworse_folds,
    "current_worst_well_regression_vs_historical": current_comparison_worst,
    "historical_top1_worst_well_delta_vs_likpf": historical_top1_worst_vs_fallback,
    "current_top1_worst_well_delta_vs_likpf": current_top1_worst_vs_fallback,
    "downstream_retraining_allowed": False,
    "rawtest_inference_allowed": False,
    "submission_allowed": False,
    "next_step": (
        "review_selector_guard_then_request_downstream_scope_approval"
        if all(checks.values())
        else "close_extra_trees_selector_branch_no_downstream_retraining"
    ),
}
print(json.dumps(decision, indent=2))


def save_nested_score_artifacts(
    items: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    records = []
    for outer_fold, item in enumerate(items):
        rows = np.concatenate([item["outer_train"], item["outer_valid"]])
        scores = np.vstack([item["train_scores"], item["valid_scores"]])
        roles = np.concatenate(
            [
                np.repeat("train", len(item["outer_train"])),
                np.repeat("valid", len(item["outer_valid"])),
            ]
        )
        artifact = frame.iloc[rows][["id", "well"]].astype(str).reset_index(drop=True)
        artifact.insert(0, "row_index", rows)
        artifact.insert(1, "role", roles)
        for index, column in enumerate(candidate_columns):
            artifact[f"pred_error__{column}"] = scores[:, index]
        path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
        artifact.to_csv(path, index=False, compression="gzip")
        records.append(
            {
                "outer_fold": outer_fold,
                "file": path.name,
                "rows": len(artifact),
                "train_rows": int(np.sum(roles == "train")),
                "valid_rows": int(np.sum(roles == "valid")),
                "sha256": sha256_path(path),
                "sha256_decompressed": sha256_path(path, decompressed=True),
            }
        )
        del artifact, scores
        gc.collect()
    return records


parameter_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_parameter_audit.csv"
fold_manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fold_manifest.csv"
context_schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_context_feature_schema.csv"
historical_manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_historical_input_manifest.csv"
model_manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_model_manifest.csv"
importance_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
importance_plot_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png"
candidate_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
score_comparison_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_score_comparison.csv"
selection_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selection_metrics.csv"
by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_by_well.csv"
fixed_predictions_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fixed_selection_predictions.csv.gz"
guard_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_guard.json"

parameter_audit.to_csv(parameter_audit_path, index=False)
fold_manifest.to_csv(fold_manifest_path, index=False)
context_schema.to_csv(context_schema_path, index=False)
historical_input_manifest.to_csv(historical_manifest_path, index=False)
model_manifest.to_csv(model_manifest_path, index=False)
candidate_metrics.to_csv(candidate_metrics_path, index=False)
score_comparison.to_csv(score_comparison_path, index=False)
selection_metrics.to_csv(selection_metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
fixed_predictions.to_csv(fixed_predictions_path, index=False, compression="gzip")
guard_path.write_text(json.dumps(decision, indent=2))

importance = (
    fold_importance.groupby("feature", as_index=False)
    .agg(
        split_importance_mean=("split_importance", "mean"),
        split_importance_std=("split_importance", "std"),
        gain_importance_mean=("gain_importance", "mean"),
        gain_importance_std=("gain_importance", "std"),
    )
    .sort_values("split_importance_mean", ascending=False)
)
importance.to_csv(importance_path, index=False)
top_n = int(CONFIG["model"]["evaluation"]["top_n_feature_importance"])
plot_data = importance.head(top_n).sort_values("split_importance_mean")
plt.figure(figsize=(11, max(7, top_n * 0.18)))
plt.barh(plot_data.feature, plot_data.split_importance_mean)
plt.title("exp262 mean selector feature importance across 20 models")
plt.xlabel("mean split count")
plt.tight_layout()
plt.savefig(importance_plot_path, dpi=140)
plt.show()

score_manifest = save_nested_score_artifacts(current_nested)
summary = {
    "status": (
        "selector_guard_passed_downstream_still_requires_user_approval"
        if decision["guard_pass"]
        else "selector_guard_failed_no_downstream_no_inference_no_submit"
    ),
    "experiment": OUTPUT_PREFIX,
    "route": CONFIG["experiment"]["route"],
    "rows": len(frame),
    "wells": int(frame.well.nunique()),
    "candidate_names": candidate_names,
    "candidate_columns": candidate_columns,
    "context_feature_count": len(context_columns),
    "selector_feature_count": len(selector_feature_columns),
    "selector_model_count": len(model_manifest),
    "control_retraining": False,
    "downstream_retraining": False,
    "changed_parameter": {"extra_trees": True},
    "cost_contract": cost_contract,
    "decision": decision,
    "candidate_metrics": candidate_metrics.to_dict(orient="records"),
    "score_comparison": score_comparison.iloc[0].to_dict(),
    "score_artifacts": score_manifest,
    "subgroup_meta": subgroup_meta,
    "reproducibility": {
        "deterministic_anchor": False,
        "rerun_sha_match_checked": False,
        "selector_feature_content_sha256": selector_feature_content_sha,
    },
    "sha256": {
        "parameter_audit": sha256_path(parameter_audit_path),
        "fold_manifest": sha256_path(fold_manifest_path),
        "context_feature_schema": sha256_path(context_schema_path),
        "historical_input_manifest": sha256_path(historical_manifest_path),
        "selector_model_manifest": sha256_path(model_manifest_path),
        "feature_importance_mean": sha256_path(importance_path),
        "candidate_metrics": sha256_path(candidate_metrics_path),
        "score_comparison": sha256_path(score_comparison_path),
        "selection_metrics": sha256_path(selection_metrics_path),
        "selector_by_well": sha256_path(by_well_path),
        "fixed_selection_predictions_decompressed": sha256_path(
            fixed_predictions_path, decompressed=True
        ),
        "guard": sha256_path(guard_path),
        "historical_selector_summary": sha256_path(historical_summary_path),
        "historical_model_manifest": sha256_path(historical_model_manifest_path),
        "canonical_exp238_config": sha256_path(exp238_config_path),
    },
}
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))
display(importance.head(60))
display(by_well.sort_values("delta_rmse_vs_historical", ascending=False).head(30))
print(json.dumps({"summary": str(summary_path), "status": summary["status"]}, indent=2))
