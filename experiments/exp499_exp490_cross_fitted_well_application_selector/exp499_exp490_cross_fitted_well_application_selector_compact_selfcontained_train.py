# %% [markdown]
# # exp499 exp490 cross-fitted well application selector — train
#
# This saved-OOF diagnostic asks whether exp490 should be applied to a well or
# whether its saved exp357 parent should be retained.  Candidate generation is
# fixed.  Phase A reads target-free prediction columns only and freezes one
# well feature table.  Phase B then attaches fold and outcome metrics and runs
# an outer-5 / inner-4 selector.  No HMM, PF, Beam, inference, or submission is
# generated here.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Paths, SHA helpers, and output helpers
# 3. Resolve and validate SHA-pinned inputs
# 4. Phase A: target-free well feature aggregation
# 5. Freeze feature contract and content SHA
# 6. Phase B: attach outcomes after freeze
# 7. Strict-nested selector
# 8. Policy, predictability, tail, and univariate readouts
# 9. Artifacts, gates, metrics, and guarded execution

# %% [markdown]
# ## 1. Imports and immutable contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_NAME = "exp499_exp490_cross_fitted_well_application_selector"
PARENT_EXPERIMENT = "exp490_geometry_centered_mean_reverting_offset_hmm"
FALLBACK_EXPERIMENT = "exp357_exp226_huber_emission_independent_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

PREDICTION_SAFE_COLUMNS = (
    "well",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "geometry_mean_reverting_hmm",
    "geometry_mean_reverting_delta_mean",
    "geometry_mean_reverting_hmm_std",
    "exp357_parent_prediction",
    "exp226_pred",
)
PREDICTION_FORBIDDEN_BEFORE_FREEZE = frozenset(
    {
        "fold",
        "true_tvt_readout_only",
        "candidate_error",
        "parent_error",
        "exp226_error",
    }
)
EXP498_FEATURE_COLUMNS = (
    "rows",
    "visible_prefix_rows",
    "suffix_horizon_md",
    "k16_median_segment_span_ft",
    "prefix_gr_sigma",
    "prefix_gr_information_ratio",
    "geometry_disagreement_median_ft",
    "early_abs_offset_ft",
    "state_uncertainty_median_ft",
)
DERIVED_FEATURE_COLUMNS = (
    "parent_exp226_abs_mean",
    "parent_exp226_abs_std",
    "parent_exp226_abs_q90",
    "parent_geometry_abs_mean",
    "parent_geometry_abs_q90",
    "candidate_parent_abs_mean",
    "candidate_parent_abs_std",
    "candidate_parent_abs_q90",
    "candidate_exp226_abs_mean",
    "candidate_exp226_abs_q90",
    "candidate_geometry_abs_mean",
    "candidate_geometry_abs_q90",
    "delta_abs_mean",
    "delta_abs_q90",
    "posterior_std_mean",
    "posterior_std_q90",
    "early128_parent_exp226_abs_mean",
    "early128_candidate_parent_abs_mean",
    "early128_delta_abs_mean",
    "early128_posterior_std_mean",
    "parent_prediction_step_abs_mean",
    "candidate_prediction_step_abs_mean",
    "exp226_prediction_step_abs_mean",
)
FEATURE_COLUMNS = (*EXP498_FEATURE_COLUMNS, *DERIVED_FEATURE_COLUMNS)
LEARNED_MODEL_NAMES = ("weighted_ridge", "weighted_hist_gradient_boosting")
POLICY_PRIORITY = (
    "always_exp490",
    "weighted_ridge",
    "weighted_hist_gradient_boosting",
)


def get_nested(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def validate_immutable_config(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp499 route must remain ensemble")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp499 parent changed")
    if get_nested(config, "lineage.fallback") != FALLBACK_EXPERIMENT:
        raise ValueError("exp499 fallback changed")
    if bool(get_nested(config, "implementation.inference_enabled")):
        raise ValueError("inference is out of scope before selector gate")
    if bool(get_nested(config, "implementation.submission_enabled")):
        raise ValueError("submission is out of scope")
    safe = tuple(get_nested(config, "data.inputs.predictions.phase_a_safe_columns"))
    if safe != PREDICTION_SAFE_COLUMNS:
        raise ValueError("Phase A prediction allowlist changed")
    forbidden = set(safe).intersection(PREDICTION_FORBIDDEN_BEFORE_FREEZE)
    if forbidden:
        raise ValueError(f"Phase A allowlist leaks {sorted(forbidden)}")
    if int(get_nested(config, "features.expected_count")) != len(FEATURE_COLUMNS):
        raise ValueError("feature count contract changed")
    execution = get_nested(config, "execution_contract")
    zero_counts = (
        "lightgbm_configs",
        "lightgbm_boosters",
        "parent_control_retraining",
        "new_hmm_well_runs",
        "new_candidate_predictions",
        "pf_runs",
        "beam_runs",
        "gpu_runs",
    )
    if any(int(execution[name]) != 0 for name in zero_counts):
        raise ValueError("execution contract contains forbidden work")
    if int(execution["learned_model_configs"]) != 2:
        raise ValueError("exp499 must compare exactly two learned model configs")
    if int(execution["maximum_total_cpu_model_fits"]) != 45:
        raise ValueError("exp499 maximum model-fit count changed")


# %% [markdown]
# ## 2. Paths, SHA helpers, and output helpers

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").is_dir():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml" if PACKAGE_DIR.name == EXPERIMENT_NAME else Path("/nonexistent"),
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp499 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or config_path()
    with selected.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("exp499 config must be a mapping")
    validate_immutable_config(value)
    return value


def artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_gzip_content(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def resolve_artifact(candidates: Sequence[str], filename: str) -> Path:
    checked: list[str] = []
    for value in candidates:
        base = Path(value)
        direct = base / filename
        checked.append(str(direct))
        if direct.is_file():
            return direct
        if base.is_dir():
            matches = sorted(base.rglob(filename))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(f"ambiguous artifact {filename}: {matches}")
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous Kaggle artifact {filename}: {matches}")
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def assert_sha(path: Path, expected: str, *, gzip_content: bool = False) -> str:
    actual = sha256_gzip_content(path) if gzip_content else sha256_file(path)
    if actual != expected:
        mode = "decompressed" if gzip_content else "raw"
        raise ValueError(f"{path.name} {mode} SHA mismatch: {actual} != {expected}")
    return actual


# %% [markdown]
# ## 3. Resolve and validate SHA-pinned inputs

# %%
def resolve_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    merge_candidates = list(get_nested(config, "data.exp490_merge_source.candidates"))
    feature_candidates = list(get_nested(config, "data.exp498_feature_source.candidates"))
    paths = {
        "predictions": resolve_artifact(
            merge_candidates,
            str(get_nested(config, "data.inputs.predictions.filename")),
        ),
        "by_well": resolve_artifact(
            merge_candidates,
            str(get_nested(config, "data.inputs.exp490_by_well_metrics.filename")),
        ),
        "exp498_features": resolve_artifact(
            feature_candidates,
            str(get_nested(config, "data.inputs.exp498_features.filename")),
        ),
        "exp498_contract": resolve_artifact(
            feature_candidates,
            str(get_nested(config, "data.inputs.exp498_features.contract_filename")),
        ),
    }
    assert_sha(
        paths["predictions"],
        str(get_nested(config, "data.inputs.predictions.raw_gzip_sha256")),
    )
    assert_sha(
        paths["predictions"],
        str(get_nested(config, "data.inputs.predictions.decompressed_sha256")),
        gzip_content=True,
    )
    assert_sha(
        paths["exp498_features"],
        str(get_nested(config, "data.inputs.exp498_features.sha256")),
    )
    assert_sha(
        paths["exp498_contract"],
        str(get_nested(config, "data.inputs.exp498_features.contract_file_sha256")),
    )
    exp498_contract = json.loads(paths["exp498_contract"].read_text(encoding="utf-8"))
    embedded = str(get_nested(config, "data.inputs.exp498_features.embedded_contract_sha256"))
    if exp498_contract.get("feature_contract_sha256") != embedded:
        raise ValueError("exp498 embedded feature-contract SHA mismatch")
    return paths


# %% [markdown]
# ## 4. Phase A: target-free well feature aggregation

# %%
def _measure_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"well": predictions["well"].astype("string")})
    result["parent_exp226_abs"] = np.abs(
        predictions["exp357_parent_prediction"] - predictions["exp226_pred"]
    )
    result["parent_geometry_abs"] = np.abs(
        predictions["exp357_parent_prediction"] - predictions["tvt_geop"]
    )
    result["candidate_parent_abs"] = np.abs(
        predictions["geometry_mean_reverting_hmm"]
        - predictions["exp357_parent_prediction"]
    )
    result["candidate_exp226_abs"] = np.abs(
        predictions["geometry_mean_reverting_hmm"] - predictions["exp226_pred"]
    )
    result["candidate_geometry_abs"] = np.abs(
        predictions["geometry_mean_reverting_hmm"] - predictions["tvt_geop"]
    )
    result["delta_abs"] = np.abs(predictions["geometry_mean_reverting_delta_mean"])
    result["posterior_std"] = predictions["geometry_mean_reverting_hmm_std"]
    return result


def build_target_free_feature_table(
    predictions: pd.DataFrame,
    exp498_features: pd.DataFrame,
) -> pd.DataFrame:
    missing_prediction = set(PREDICTION_SAFE_COLUMNS).difference(predictions.columns)
    if missing_prediction:
        raise ValueError(f"prediction input missing {sorted(missing_prediction)}")
    if set(predictions.columns).intersection(PREDICTION_FORBIDDEN_BEFORE_FREEZE):
        raise ValueError("Phase A prediction frame contains forbidden outcome columns")
    predictions = predictions.loc[:, PREDICTION_SAFE_COLUMNS].copy()
    predictions["well"] = predictions["well"].astype("string")
    predictions = predictions.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if predictions[["well", "row_idx"]].duplicated().any():
        raise ValueError("duplicate well,row_idx keys")

    measures = _measure_frame(predictions)
    measure_names = [column for column in measures.columns if column != "well"]
    grouped = measures.groupby("well", sort=True, observed=True)
    means = grouped[measure_names].mean().add_suffix("_mean")
    stds = grouped[["parent_exp226_abs", "candidate_parent_abs"]].std(ddof=0).add_suffix("_std")
    q90 = grouped[measure_names].quantile(0.90).add_suffix("_q90")
    aggregate = means.join(stds, how="left").join(q90, how="left")

    early_mask = predictions["suffix_offset"].between(0, 127, inclusive="both")
    early = measures.loc[early_mask].groupby("well", sort=True, observed=True).mean()
    early = early.loc[
        :,
        ["parent_exp226_abs", "candidate_parent_abs", "delta_abs", "posterior_std"],
    ].add_prefix("early128_").add_suffix("_mean")
    aggregate = aggregate.join(early, how="left")

    roughness = pd.DataFrame({"well": predictions["well"]})
    group_key = predictions["well"]
    roughness["parent_prediction_step_abs"] = predictions.groupby(group_key, sort=False)[
        "exp357_parent_prediction"
    ].diff().abs()
    roughness["candidate_prediction_step_abs"] = predictions.groupby(group_key, sort=False)[
        "geometry_mean_reverting_hmm"
    ].diff().abs()
    roughness["exp226_prediction_step_abs"] = predictions.groupby(group_key, sort=False)[
        "exp226_pred"
    ].diff().abs()
    roughness_mean = roughness.groupby("well", sort=True, observed=True).mean().add_suffix("_mean")
    aggregate = aggregate.join(roughness_mean, how="left").reset_index()

    required_exp498 = {"well", *EXP498_FEATURE_COLUMNS}
    missing_exp498 = required_exp498.difference(exp498_features.columns)
    if missing_exp498:
        raise ValueError(f"exp498 feature input missing {sorted(missing_exp498)}")
    physics = exp498_features.loc[:, ["well", *EXP498_FEATURE_COLUMNS]].copy()
    physics["well"] = physics["well"].astype("string")
    if physics["well"].duplicated().any():
        raise ValueError("duplicate exp498 well")
    result = physics.merge(aggregate, on="well", how="inner", validate="one_to_one")
    result = result.loc[:, ["well", *FEATURE_COLUMNS]].sort_values("well", kind="stable")
    result = result.reset_index(drop=True)
    if list(result.columns[1:]) != list(FEATURE_COLUMNS):
        raise AssertionError("feature order drifted")
    values = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise ValueError(f"nonfinite target-free feature values at {bad}")
    if (values < 0).any():
        bad = np.argwhere(values < 0)[:10].tolist()
        raise ValueError(f"negative value violates log1p contract at {bad}")
    return result


def read_phase_a_features(config: Mapping[str, Any], paths: Mapping[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_header = pd.read_csv(paths["predictions"], nrows=0).columns.tolist()
    required = set(PREDICTION_SAFE_COLUMNS)
    missing = required.difference(prediction_header)
    if missing:
        raise ValueError(f"upstream predictions missing {sorted(missing)}")
    predictions = pd.read_csv(
        paths["predictions"],
        usecols=list(PREDICTION_SAFE_COLUMNS),
        dtype={"well": "string"},
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if len(predictions) != expected_rows:
        raise ValueError(f"prediction rows {len(predictions)} != {expected_rows}")
    exp498 = pd.read_csv(paths["exp498_features"], dtype={"well": "string"})
    features = build_target_free_feature_table(predictions, exp498)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(features) != expected_wells:
        raise ValueError(f"feature wells {len(features)} != {expected_wells}")
    row_counts = predictions.groupby("well", sort=True).size().rename("actual_rows")
    count_check = features[["well", "rows"]].merge(
        row_counts.reset_index(), on="well", how="left", validate="one_to_one"
    )
    if not np.array_equal(count_check["rows"].to_numpy(), count_check["actual_rows"].to_numpy()):
        raise ValueError("exp498 rows disagree with exp490 prediction rows")
    ledger = {
        "phase": "phase_a_target_free_feature_freeze",
        "prediction_header": prediction_header,
        "loaded_prediction_columns": list(PREDICTION_SAFE_COLUMNS),
        "forbidden_loaded_columns": sorted(
            set(PREDICTION_SAFE_COLUMNS).intersection(PREDICTION_FORBIDDEN_BEFORE_FREEZE)
        ),
        "outcome_files_read_before_feature_freeze": 0,
        "prediction_rows": int(len(predictions)),
        "wells": int(len(features)),
    }
    del predictions, exp498
    return features, ledger


# %% [markdown]
# ## 5. Freeze feature contract and content SHA

# %%
def freeze_features(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    features: pd.DataFrame,
    ledger: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    names = get_nested(config, "artifacts.files")
    feature_path = output / str(names["feature_table"])
    contract_path = output / str(names["feature_contract"])
    write_csv(feature_path, features)
    feature_sha = sha256_file(feature_path)
    schema = [{"column": c, "dtype": str(features[c].dtype)} for c in features.columns]
    schema_sha = sha256_json(schema)
    contract = {
        "experiment": EXPERIMENT_NAME,
        "phase": "phase_a_target_free_frozen_before_fold_or_outcome",
        "rows": int(len(features)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema": schema,
        "feature_schema_sha256": schema_sha,
        "feature_content_sha256": feature_sha,
        "transform": "log1p_nonnegative_then_fold_local_median_impute",
        "ridge_scaler": "fold_local_standard_scaler",
        "prediction_safe_columns": list(PREDICTION_SAFE_COLUMNS),
        "prediction_forbidden_before_freeze": sorted(PREDICTION_FORBIDDEN_BEFORE_FREEZE),
        "leakage_ledger": dict(ledger),
        "input_sha256": {
            "exp490_predictions_raw_gzip": sha256_file(paths["predictions"]),
            "exp490_predictions_decompressed": sha256_gzip_content(paths["predictions"]),
            "exp498_features": sha256_file(paths["exp498_features"]),
            "exp498_contract": sha256_file(paths["exp498_contract"]),
        },
        "new_hmm_runs": 0,
        "new_candidate_predictions": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    contract["feature_contract_sha256"] = sha256_json(contract)
    write_json(contract_path, contract)
    return contract, {"feature_table": feature_path, "feature_contract": contract_path}


# %% [markdown]
# ## 6. Phase B: attach outcomes after freeze

# %%
def attach_outcomes_after_freeze(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    features: pd.DataFrame,
) -> pd.DataFrame:
    assert_sha(
        paths["by_well"],
        str(get_nested(config, "data.inputs.exp490_by_well_metrics.sha256")),
    )
    by_well = pd.read_csv(paths["by_well"], dtype={"well": "string"})
    expected_columns = {
        "well",
        "fold",
        "rows",
        "candidate_rmse_ft",
        "exp357_parent_rmse_ft",
        "candidate_minus_parent_rmse_ft",
    }
    if not expected_columns.issubset(by_well.columns):
        raise ValueError(f"by-well metrics missing {sorted(expected_columns.difference(by_well.columns))}")
    if by_well["well"].duplicated().any():
        raise ValueError("duplicate by-well outcome")
    merged = features.merge(
        by_well.loc[:, sorted(expected_columns)],
        on="well",
        how="inner",
        suffixes=("", "_outcome"),
        validate="one_to_one",
    )
    if "rows_outcome" not in merged.columns:
        raise AssertionError("row-count outcome suffix missing")
    if not np.array_equal(merged["rows"].to_numpy(), merged["rows_outcome"].to_numpy()):
        raise ValueError("feature and outcome row counts disagree")
    merged = merged.drop(columns="rows_outcome")
    merged["actual_benefit_mse"] = (
        merged["exp357_parent_rmse_ft"].pow(2) - merged["candidate_rmse_ft"].pow(2)
    )
    merged["beneficial_well"] = merged["actual_benefit_mse"].gt(0)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_folds = set(get_nested(config, "validation.expected_folds"))
    if len(merged) != expected_wells or int(merged["rows"].sum()) != expected_rows:
        raise ValueError("outcome identity count mismatch")
    if set(merged["fold"].astype(int).unique()) != expected_folds:
        raise ValueError("fold identity mismatch")
    return merged.sort_values("well", kind="stable").reset_index(drop=True)


# %% [markdown]
# ## 7. Strict-nested selector

# %%
def transformed_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if (values < 0).any():
        raise ValueError("log1p feature matrix contains negative values")
    return np.log1p(values)


def build_model(name: str, config: Mapping[str, Any]) -> Pipeline:
    if name == "weighted_ridge":
        params = dict(get_nested(config, "model.configs.weighted_ridge"))
        estimator = Ridge(**params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if name == "weighted_hist_gradient_boosting":
        params = dict(get_nested(config, "model.configs.weighted_hist_gradient_boosting"))
        estimator = HistGradientBoostingRegressor(**params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    raise KeyError(name)


def fit_model(
    name: str,
    config: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    rows_train: np.ndarray,
) -> Pipeline:
    model = build_model(name, config)
    sample_weight = rows_train.astype(np.float64) / float(np.mean(rows_train))
    model.fit(x_train, y_train, model__sample_weight=sample_weight)
    return model


def policy_rmse(frame: pd.DataFrame, apply_exp490: np.ndarray) -> float:
    apply = np.asarray(apply_exp490, dtype=bool)
    if len(apply) != len(frame):
        raise ValueError("policy length mismatch")
    candidate_mse = frame["candidate_rmse_ft"].to_numpy(dtype=np.float64) ** 2
    parent_mse = frame["exp357_parent_rmse_ft"].to_numpy(dtype=np.float64) ** 2
    selected_mse = np.where(apply, candidate_mse, parent_mse)
    weights = frame["rows"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.average(selected_mse, weights=weights)))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2 or not np.isfinite(scores).all():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_spearman(actual: np.ndarray, scores: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.nanstd(actual) == 0 or np.nanstd(scores) == 0:
        return float("nan")
    return float(spearmanr(actual, scores, nan_policy="raise").statistic)


def _choose_learned_model(scores: Mapping[str, float], tolerance: float) -> str:
    minimum = min(scores.values())
    eligible = {name for name, score in scores.items() if score <= minimum + tolerance}
    for name in LEARNED_MODEL_NAMES:
        if name in eligible:
            return name
    raise AssertionError("no learned model selected")


def run_strict_nested_selector(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x = transformed_matrix(frame)
    y = frame["actual_benefit_mse"].to_numpy(dtype=np.float64)
    rows = frame["rows"].to_numpy(dtype=np.float64)
    folds = frame["fold"].to_numpy(dtype=np.int64)
    expected_folds = sorted(int(v) for v in get_nested(config, "validation.expected_folds"))
    tolerance = float(get_nested(config, "validation.tie_tolerance_ft"))
    outer_predictions: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
    model_fits = 0
    manifests: list[dict[str, Any]] = []

    for outer_fold in expected_folds:
        outer_valid = folds == outer_fold
        outer_train = ~outer_valid
        train_folds = [fold for fold in expected_folds if fold != outer_fold]
        if outer_train.sum() == 0 or outer_valid.sum() == 0:
            raise ValueError(f"empty outer split {outer_fold}")
        learned_predictions: dict[str, np.ndarray] = {}
        learned_scores: dict[str, float] = {}

        for model_name in LEARNED_MODEL_NAMES:
            inner_prediction = np.full(len(frame), np.nan, dtype=np.float64)
            for inner_valid_fold in train_folds:
                inner_valid = outer_train & (folds == inner_valid_fold)
                inner_train = outer_train & (folds != inner_valid_fold)
                if np.any(inner_train & inner_valid) or not inner_valid.any():
                    raise ValueError("invalid inner split")
                model = fit_model(
                    model_name,
                    config,
                    x[inner_train],
                    y[inner_train],
                    rows[inner_train],
                )
                model_fits += 1
                inner_prediction[inner_valid] = model.predict(x[inner_valid])
            if not np.isfinite(inner_prediction[outer_train]).all():
                raise ValueError(f"incomplete inner OOF prediction for outer {outer_fold}")
            learned_predictions[model_name] = inner_prediction
            learned_scores[model_name] = policy_rmse(
                frame.loc[outer_train],
                inner_prediction[outer_train] > 0,
            )
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "policy": model_name,
                    "inner_oof_rmse_ft": learned_scores[model_name],
                    "inner_apply_fraction": float(np.mean(inner_prediction[outer_train] > 0)),
                    "inner_auc": safe_auc(
                        frame.loc[outer_train, "beneficial_well"].to_numpy(),
                        inner_prediction[outer_train],
                    ),
                    "inner_spearman": safe_spearman(y[outer_train], inner_prediction[outer_train]),
                }
            )

        always_rmse = policy_rmse(frame.loc[outer_train], np.ones(outer_train.sum(), dtype=bool))
        inner_rows.append(
            {
                "outer_fold": outer_fold,
                "policy": "always_exp490",
                "inner_oof_rmse_ft": always_rmse,
                "inner_apply_fraction": 1.0,
                "inner_auc": 0.5,
                "inner_spearman": 0.0,
            }
        )
        learned_choice = _choose_learned_model(learned_scores, tolerance)
        learned_rmse = learned_scores[learned_choice]
        policy_choice = learned_choice if learned_rmse + tolerance < always_rmse else "always_exp490"

        outer_model = fit_model(
            learned_choice,
            config,
            x[outer_train],
            y[outer_train],
            rows[outer_train],
        )
        model_fits += 1
        score = outer_model.predict(x[outer_valid])
        apply = np.ones(outer_valid.sum(), dtype=bool) if policy_choice == "always_exp490" else score > 0
        valid_frame = frame.loc[outer_valid].copy()
        valid_frame["predicted_benefit_mse"] = score
        valid_frame["learned_model"] = learned_choice
        valid_frame["policy"] = policy_choice
        valid_frame["apply_exp490"] = apply
        outer_predictions.append(valid_frame)
        manifests.append(
            {
                "outer_fold": outer_fold,
                "outer_train_wells": int(outer_train.sum()),
                "outer_valid_wells": int(outer_valid.sum()),
                "outer_train_well_sha256": sha256_json(sorted(frame.loc[outer_train, "well"].tolist())),
                "outer_valid_well_sha256": sha256_json(sorted(frame.loc[outer_valid, "well"].tolist())),
                "learned_model_choice": learned_choice,
                "policy_choice": policy_choice,
                "inner_always_rmse_ft": always_rmse,
                "inner_learned_rmse_ft": learned_rmse,
                "inner_model_rmse_ft": learned_scores,
                "outer_score_min": float(np.min(score)),
                "outer_score_max": float(np.max(score)),
                "outer_apply_fraction": float(np.mean(apply)),
            }
        )

    selector_oof = pd.concat(outer_predictions, ignore_index=True)
    selector_oof = selector_oof.sort_values("well", kind="stable").reset_index(drop=True)
    if selector_oof["well"].duplicated().any() or len(selector_oof) != len(frame):
        raise ValueError("outer selector coverage is not exactly one prediction per well")
    if model_fits != int(get_nested(config, "execution_contract.maximum_total_cpu_model_fits")):
        raise ValueError(f"trained {model_fits} models; expected exactly 45")
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "model_configs": {
            name: get_nested(config, f"model.configs.{name}") for name in LEARNED_MODEL_NAMES
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_transform": "log1p_nonnegative",
        "target": str(get_nested(config, "validation.signed_target")),
        "sample_weight": str(get_nested(config, "model.sample_weight")),
        "outer_folds": expected_folds,
        "inner_folds_per_outer": 4,
        "trained_cpu_model_fits": model_fits,
        "lightgbm_boosters": 0,
        "pf_runs": 0,
        "gpu_runs": 0,
        "fold_models": manifests,
    }
    return selector_oof, pd.DataFrame(inner_rows), manifest


# %% [markdown]
# ## 8. Policy, predictability, tail, and univariate readouts

# %%
def evaluate_selector(
    config: Mapping[str, Any],
    selector_oof: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected = selector_oof.copy()
    selected["selected_rmse_ft"] = np.where(
        selected["apply_exp490"],
        selected["candidate_rmse_ft"],
        selected["exp357_parent_rmse_ft"],
    )
    selected["selected_minus_parent_rmse_ft"] = (
        selected["selected_rmse_ft"] - selected["exp357_parent_rmse_ft"]
    )
    selected["selected_squared_error_sum"] = selected["rows"] * selected["selected_rmse_ft"].pow(2)
    selected["candidate_squared_error_sum"] = selected["rows"] * selected["candidate_rmse_ft"].pow(2)
    selected["parent_squared_error_sum"] = selected["rows"] * selected["exp357_parent_rmse_ft"].pow(2)
    selected["oracle_squared_error_sum"] = np.minimum(
        selected["candidate_squared_error_sum"], selected["parent_squared_error_sum"]
    )

    def summarize(scope: pd.DataFrame, fold: str | int) -> dict[str, Any]:
        total_rows = float(scope["rows"].sum())
        parent_rmse = math.sqrt(float(scope["parent_squared_error_sum"].sum()) / total_rows)
        candidate_rmse = math.sqrt(float(scope["candidate_squared_error_sum"].sum()) / total_rows)
        selected_rmse = math.sqrt(float(scope["selected_squared_error_sum"].sum()) / total_rows)
        oracle_rmse = math.sqrt(float(scope["oracle_squared_error_sum"].sum()) / total_rows)
        applied = scope["apply_exp490"].to_numpy(dtype=bool)
        beneficial = scope["beneficial_well"].to_numpy(dtype=bool)
        precision = float(np.mean(beneficial[applied])) if applied.any() else float("nan")
        return {
            "fold": fold,
            "wells": int(len(scope)),
            "rows": int(total_rows),
            "parent_rmse_ft": parent_rmse,
            "always_exp490_rmse_ft": candidate_rmse,
            "selected_rmse_ft": selected_rmse,
            "oracle_rmse_ft": oracle_rmse,
            "selected_gain_vs_always_exp490_ft": candidate_rmse - selected_rmse,
            "apply_fraction": float(np.mean(applied)),
            "applied_wells": int(applied.sum()),
            "applied_beneficial_precision": precision,
            "auc": safe_auc(beneficial, scope["predicted_benefit_mse"].to_numpy()),
            "spearman": safe_spearman(
                scope["actual_benefit_mse"].to_numpy(),
                scope["predicted_benefit_mse"].to_numpy(),
            ),
            "harmful_applied_wells": int(np.sum(applied & ~beneficial)),
            "catastrophic_applied_wells": int(
                np.sum(applied & scope["candidate_minus_parent_rmse_ft"].gt(5).to_numpy())
            ),
        }

    fold_rows = [summarize(fold, int(fold_id)) for fold_id, fold in selected.groupby("fold", sort=True)]
    pooled = summarize(selected, "all")
    fold_metrics = pd.DataFrame([pooled, *fold_rows])
    fold_only = fold_metrics[fold_metrics["fold"].ne("all")].copy()
    gates = get_nested(config, "gates")
    predict_cfg = gates["predictability_requires_all"]
    safe_cfg = gates["safe_router_requires_all"]
    predict_checks = {
        "pooled_auc": bool(pooled["auc"] >= float(predict_cfg["pooled_auc_minimum"])),
        "fold_auc_stability": bool(
            int((fold_only["auc"] >= 0.55).sum())
            >= int(predict_cfg["folds_auc_ge_0_55_minimum"])
        ),
        "fold_spearman_direction": bool(
            int((fold_only["spearman"] > 0).sum())
            >= int(predict_cfg["folds_positive_spearman_minimum"])
        ),
    }
    bywell_p95 = float(selected["selected_minus_parent_rmse_ft"].quantile(0.95))
    bywell_worst = float(selected["selected_minus_parent_rmse_ft"].max())
    safe_checks = {
        "gain_vs_always_exp490": bool(
            pooled["selected_gain_vs_always_exp490_ft"]
            >= float(safe_cfg["gain_vs_always_exp490_minimum_ft"])
        ),
        "fold_nonworse": bool(
            int(
                (
                    fold_only["selected_rmse_ft"]
                    <= fold_only["always_exp490_rmse_ft"]
                    + float(safe_cfg["fold_nonworse_tolerance_ft"])
                ).sum()
            )
            >= int(safe_cfg["nonworse_folds_minimum"])
        ),
        "apply_fraction_minimum": bool(
            pooled["apply_fraction"] >= float(safe_cfg["apply_fraction_minimum"])
        ),
        "apply_fraction_maximum": bool(
            pooled["apply_fraction"] <= float(safe_cfg["apply_fraction_maximum"])
        ),
        "by_well_p95": bool(
            bywell_p95 <= float(safe_cfg["selected_minus_parent_by_well_rmse_p95_max_ft"])
        ),
        "by_well_worst": bool(
            bywell_worst <= float(safe_cfg["selected_minus_parent_by_well_rmse_worst_max_ft"])
        ),
    }
    metrics = {
        "pooled": pooled,
        "folds_auc_ge_0_55": int((fold_only["auc"] >= 0.55).sum()),
        "folds_positive_spearman": int((fold_only["spearman"] > 0).sum()),
        "folds_selected_nonworse_than_always": int(
            (fold_only["selected_rmse_ft"] <= fold_only["always_exp490_rmse_ft"]).sum()
        ),
        "selected_minus_parent_by_well_rmse_p95_ft": bywell_p95,
        "selected_minus_parent_by_well_rmse_worst_ft": bywell_worst,
        "predictability_checks": predict_checks,
        "predictability_supported": bool(all(predict_checks.values())),
        "safe_router_checks": safe_checks,
        "safe_router_supported": bool(all(safe_checks.values())),
        "improved_wells": int(selected["beneficial_well"].sum()),
        "worsened_wells": int((~selected["beneficial_well"]).sum()),
    }
    return metrics, fold_metrics, selected


def univariate_readout(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        values = frame[feature].to_numpy(dtype=np.float64)
        benefit = frame["actual_benefit_mse"].to_numpy(dtype=np.float64)
        labels = frame["beneficial_well"].to_numpy(dtype=bool)
        raw_auc = safe_auc(labels, values)
        fold_auc: list[float] = []
        fold_spearman: list[float] = []
        for _, scope in frame.groupby("fold", sort=True):
            fold_auc.append(safe_auc(scope["beneficial_well"].to_numpy(), scope[feature].to_numpy()))
            fold_spearman.append(
                safe_spearman(scope["actual_benefit_mse"].to_numpy(), scope[feature].to_numpy())
            )
        rows.append(
            {
                "feature": feature,
                "pooled_spearman": safe_spearman(benefit, values),
                "pooled_auc_raw_direction": raw_auc,
                "pooled_auc_oriented_descriptive": max(raw_auc, 1.0 - raw_auc),
                "fold_auc_raw_mean": float(np.nanmean(fold_auc)),
                "fold_auc_raw_min": float(np.nanmin(fold_auc)),
                "fold_auc_raw_max": float(np.nanmax(fold_auc)),
                "fold_spearman_mean": float(np.nanmean(fold_spearman)),
                "fold_spearman_positive_count": int(np.sum(np.asarray(fold_spearman) > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pooled_auc_oriented_descriptive", ascending=False, kind="stable"
    )


def save_plot(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for fold, scope in frame.groupby("fold", sort=True):
        ax.scatter(
            scope["predicted_benefit_mse"],
            scope["actual_benefit_mse"],
            s=16,
            alpha=0.65,
            label=f"fold {int(fold)}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("cross-fitted predicted signed MSE benefit (ft²/row)")
    ax.set_ylabel("actual signed MSE benefit (symlog)")
    ax.set_title("exp499 target-free score vs exp490 benefit")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %% [markdown]
# ## 9. Artifacts, gates, metrics, and guarded execution

# %%
def run_experiment() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    output = artifacts_dir()
    paths = resolve_inputs(config)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(config, "experiment.route"),
                "parent": PARENT_EXPERIMENT,
                "fallback": FALLBACK_EXPERIMENT,
                "execution_contract": get_nested(config, "execution_contract"),
                "inference_enabled": False,
                "submission_enabled": False,
            },
            indent=2,
        )
    )

    # Phase A: no fold or outcomes have been read at this point.
    features, leakage_ledger = read_phase_a_features(config, paths)
    feature_contract, feature_paths = freeze_features(
        config, paths, features, leakage_ledger, output
    )
    print(
        f"FEATURE_FREEZE wells={len(features)} count={len(FEATURE_COLUMNS)} "
        f"sha={feature_contract['feature_content_sha256']}"
    )

    # Phase B begins only after the feature table and contract exist on disk.
    analysis = attach_outcomes_after_freeze(config, paths, features)
    selector_oof, inner_scores, model_manifest = run_strict_nested_selector(config, analysis)
    evaluation, fold_metrics, well_metrics = evaluate_selector(config, selector_oof)
    univariate = univariate_readout(analysis)

    names = get_nested(config, "artifacts.files")
    artifact_paths: dict[str, Path] = dict(feature_paths)
    for key, frame in (
        ("selector_oof", selector_oof),
        ("inner_scores", inner_scores),
        ("fold_metrics", fold_metrics),
        ("well_metrics", well_metrics),
        ("univariate_metrics", univariate),
    ):
        path = output / str(names[key])
        write_csv(path, frame)
        artifact_paths[key] = path
    model_manifest["feature_content_sha256"] = feature_contract["feature_content_sha256"]
    model_manifest_path = output / str(names["model_manifest"])
    write_json(model_manifest_path, model_manifest)
    artifact_paths["model_manifest"] = model_manifest_path
    plot_path = output / str(names["plot"])
    save_plot(selector_oof, plot_path)
    artifact_paths["plot"] = plot_path

    technical_checks = {
        "input_sha_match": True,
        "expected_prediction_rows": int(leakage_ledger["prediction_rows"])
        == int(get_nested(config, "validation.expected_rows")),
        "expected_wells": len(features) == int(get_nested(config, "validation.expected_wells")),
        "expected_folds": int(analysis["fold"].nunique())
        == int(get_nested(config, "validation.n_folds")),
        "feature_count": len(FEATURE_COLUMNS) == int(get_nested(config, "features.expected_count")),
        "finite_feature_coverage": bool(
            np.isfinite(features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all()
        ),
        "feature_outcome_phase_violations": leakage_ledger[
            "outcome_files_read_before_feature_freeze"
        ]
        == 0,
        "outer_prediction_coverage": len(selector_oof) == len(analysis)
        and not selector_oof["well"].duplicated().any(),
        "trained_model_fit_count": int(model_manifest["trained_cpu_model_fits"])
        == int(get_nested(config, "execution_contract.maximum_total_cpu_model_fits")),
    }
    technical_passed = bool(all(technical_checks.values()))
    deployment_eligible = bool(
        technical_passed
        and evaluation["predictability_supported"]
        and evaluation["safe_router_supported"]
    )
    artifact_sha = {key: sha256_file(path) for key, path in artifact_paths.items()}
    artifact_sha["exp490_by_well_input"] = sha256_file(paths["by_well"])
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_pass" if deployment_eligible else "completed_fail_closed",
        "route": "ensemble",
        "kernel_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2,
        },
        "input_paths": {key: str(value) for key, value in paths.items()},
        "feature_contract": feature_contract,
        "technical_checks": technical_checks,
        "technical_passed": technical_passed,
        "evaluation": evaluation,
        "deployment_eligible": deployment_eligible,
        "execution_actual": {
            "variants": 1,
            "learned_model_configs": 2,
            "outer_folds": 5,
            "inner_folds_per_outer": 4,
            "trained_cpu_model_fits": int(model_manifest["trained_cpu_model_fits"]),
            "lightgbm_boosters": 0,
            "parent_control_retraining": 0,
            "new_hmm_well_runs": 0,
            "new_candidate_predictions": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "artifact_sha256": artifact_sha,
        "inference_enabled": False,
        "submission_enabled": False,
        "notes": (
            "A deployment pass only permits a separately approved inference-port design; "
            "a fail closes this fixed selector without same-OOF threshold or feature rescue."
        ),
    }
    summary_path = output / str(names["summary"])
    write_json(summary_path, summary)
    summary["artifact_paths"]["summary"] = str(summary_path)
    summary["artifact_sha256"]["summary"] = sha256_file(summary_path)

    root_metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "metric": "strict_nested_well_router_rmse_and_predictability",
        "cv": evaluation["pooled"]["selected_rmse_ft"],
        "public_lb": None,
        "private_lb": None,
        "technical_passed": technical_passed,
        "predictability_supported": evaluation["predictability_supported"],
        "safe_router_supported": evaluation["safe_router_supported"],
        "deployment_eligible": deployment_eligible,
        "evaluation": evaluation,
        "execution_actual": summary["execution_actual"],
        "artifact_paths": summary["artifact_paths"],
        "artifact_sha256": summary["artifact_sha256"],
        "notes": summary["notes"],
    }
    write_json(metrics_path(), root_metrics)
    print("EXP499_SUMMARY " + json.dumps(to_jsonable(summary), sort_keys=True))
    return summary


# %%
if __name__ == "__main__":
    RUN_EXP499 = True
    if RUN_EXP499:
        RUN_SUMMARY = run_experiment()
        print(
            "EXP499_VERDICT",
            {
                "technical": RUN_SUMMARY["technical_passed"],
                "predictability": RUN_SUMMARY["evaluation"]["predictability_supported"],
                "safe_router": RUN_SUMMARY["evaluation"]["safe_router_supported"],
                "deployment_eligible": RUN_SUMMARY["deployment_eligible"],
            },
        )

