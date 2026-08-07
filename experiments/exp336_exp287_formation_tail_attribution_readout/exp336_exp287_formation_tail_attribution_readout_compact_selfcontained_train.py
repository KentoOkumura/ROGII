# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp336 exp287 formation-tail attribution readout — train
#
# exp287 の outer-valid formation cache から、事前登録した6つの target-free
# formation risk family を well 単位へ集約する。Stage A で属性・四分位・SHAを
# 凍結した後だけ、Stage B で exp287 / corrected exp264 OOF を開き、well 等重みの
# RMSE delta を診断する。モデル学習、予測補正、推論、submission は行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Scientific, leakage, and execution contract
# 3. Frozen input and path helpers
# 4. Stage A formation-cache audit and target-free aggregation
# 5. Stage A raw target-free context and freeze barrier
# 6. Stage B OOF alignment and well-level endpoint
# 7. Quartile, fold, hidden-like, and decision metrics
# 8. Generated artifacts and reproducibility manifest
# 9. Execution orchestration

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp336_exp287_formation_tail_attribution_readout"
PARENT_EXPERIMENT = "exp287_fold_safe_formation_74_addonly_on_exp264"
COMPARISON_EXPERIMENT = "exp264_exp263_candidate_confidence_dual_selector"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP336_IMPORT_ONLY", "0") == "1"

IDENTITY_COLUMNS = ["id", "well"]
STAGE_A_EXACT_FORBIDDEN_COLUMNS = {
    "TVT",
    "target",
    "actual",
    "actual_tvt",
    "truth",
    "prediction",
    "pred_tvt",
    "error",
    "abs_error",
    "squared_error",
    "worst_well",
    "worst_well_id",
}
PRIMARY_FAMILY_ORDER = [
    "plane_reference_distance",
    "dense_reference_distance",
    "dense_neighbor_uncertainty",
    "plane_dense_disagreement",
    "formation_consensus_spread",
    "known_prefix_formation_calibration_error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def find_config_path() -> Path:
    direct = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"exp336 config resolution is ambiguous: {matches}")
    return matches[0].resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def canonical_csv_bytes(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> bytes:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    text = selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="NA",
    )
    return text.encode("utf-8")


def write_canonical_csv(
    path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_csv_bytes(frame, columns)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.is_dir() and KAGGLE_WORKING_ROOT.is_dir()


# %% [markdown]
# ## 2. Scientific, leakage, and execution contract
#
# 実装完了状態では全 run flag を false に保つ。full readout は Kaggle CPU 上で、
# `active_stage=full_attribution_readout`、push 承認、Stage A/B の両 flag がそろう
# 場合だけ開始する。Stage A の関数引数には OOF path を持たせない。


# %%
def canonical_formation_feature_names() -> list[str]:
    formations = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    names = ["sig_std", "sig_mean_d"]
    for formation in formations:
        names.extend(
            [
                f"tvtF_{formation}",
                f"tvtFw_{formation}",
                f"tvtF50_{formation}",
                f"bw_{formation}",
                f"bww_{formation}",
                f"bw50_{formation}",
                f"bw_early_{formation}",
                f"bw_mid_{formation}",
            ]
        )
    names.extend(f"frm_rmse_{formation}" for formation in formations)
    names.extend(
        [
            "form_mean_d",
            "form_std_d",
            "form_rng_d",
            "spatial_ancc_d",
            "spatial_knn_dist",
            "dense_ancc",
            "dense_std",
            "dense_dist",
            "tvt_dense_d",
            "tvt_densew_d",
            "tvt_dense50_d",
            "dense_rmse",
            "dense_bias",
            "dense_nb_std",
            "pf_vs_spatial",
            "pf_vs_dense",
            "spatial_vs_dense",
            "beam_vs_spatial",
        ]
    )
    if len(names) != 74 or len(set(names)) != 74:
        raise AssertionError("formation feature contract must contain 74 unique columns")
    return names


def is_stage_a_forbidden_column(name: str) -> bool:
    raw = str(name)
    lower = raw.lower()
    exact = {value.lower() for value in STAGE_A_EXACT_FORBIDDEN_COLUMNS}
    return (
        lower in exact
        or "__pred_tvt" in lower
        or lower.endswith("_prediction")
        or lower.endswith("_abs_error")
        or lower.endswith("_squared_error")
    )


def expected_family_contract() -> list[dict[str, Any]]:
    return [
        {
            "name": "plane_reference_distance",
            "source_columns": ["spatial_knn_dist"],
            "well_scalar": "row_quantile_0p90",
            "high_is_risky": True,
        },
        {
            "name": "dense_reference_distance",
            "source_columns": ["dense_dist"],
            "well_scalar": "row_quantile_0p90",
            "high_is_risky": True,
        },
        {
            "name": "dense_neighbor_uncertainty",
            "source_columns": ["dense_std"],
            "well_scalar": "row_quantile_0p90",
            "high_is_risky": True,
        },
        {
            "name": "plane_dense_disagreement",
            "source_columns": ["spatial_vs_dense"],
            "transform": "absolute_value",
            "well_scalar": "row_quantile_0p90",
            "high_is_risky": True,
        },
        {
            "name": "formation_consensus_spread",
            "source_columns": ["form_rng_d"],
            "well_scalar": "row_quantile_0p90",
            "high_is_risky": True,
        },
        {
            "name": "known_prefix_formation_calibration_error",
            "source_columns": [
                "frm_rmse_ANCC",
                "frm_rmse_ASTNU",
                "frm_rmse_ASTNL",
                "frm_rmse_EGFDU",
                "frm_rmse_EGFDL",
                "frm_rmse_BUDA",
                "dense_rmse",
            ],
            "well_scalar": "maximum_of_per_well_constant_values",
            "high_is_risky": True,
        },
    ]


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_execution: bool
) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp336 route must remain ml_model")
    if not bool(nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp336 implementation is not approved")
    if list(nested(config, "audit.primary_risk_families")) != expected_family_contract():
        raise ValueError("the six preregistered risk-family definitions changed")
    cost = {
        "active_variants": int(nested(config, "model.active_variants")),
        "lightgbm_configs": int(nested(config, "model.lightgbm_config_count")),
        "trained_folds": int(nested(config, "model.trained_fold_count")),
        "boosters": int(nested(config, "model.booster_count")),
        "control_retraining": bool(nested(config, "model.parent_control_retraining")),
    }
    expected_cost = {
        "active_variants": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "control_retraining": False,
    }
    if cost != expected_cost:
        raise ValueError(f"zero-booster cost contract changed: {cost}")
    if bool(nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("exp336 must remain CPU-only")
    if bool(nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("Kaggle internet must remain disabled")
    if int(nested(config, "runtime.num_workers")) != 1:
        raise ValueError("exp336 must remain single-worker")
    if int(nested(config, "runtime.blas_threads")) != 1:
        raise ValueError("exp336 BLAS thread count must remain one")
    for flag in [
        "execution.run_model_training",
        "execution.run_inference",
        "execution.create_submission",
        "execution.submit_to_kaggle",
    ]:
        if bool(nested(config, flag)):
            raise ValueError(f"forbidden execution flag is enabled: {flag}")
    allowed = list(nested(config, "execution.allowed_stages"))
    expected_allowed = ["implementation_complete_no_run", "full_attribution_readout"]
    if allowed != expected_allowed:
        raise ValueError(f"unexpected allowed stages: {allowed}")
    stage = str(nested(config, "execution.active_stage"))
    if stage not in allowed:
        raise ValueError(f"active stage is outside the fixed contract: {stage}")
    if require_execution:
        if not is_kaggle_runtime():
            raise RuntimeError("full exp336 readout is Kaggle-runtime only")
        if stage != "full_attribution_readout":
            raise RuntimeError("full readout requires active_stage=full_attribution_readout")
        if not bool(nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("Kaggle CPU package/push/run requires separate approval")
        if not bool(nested(config, "execution.run_stage_a_freeze")):
            raise RuntimeError("full readout requires run_stage_a_freeze=true")
        if not bool(nested(config, "execution.run_stage_b_attribution")):
            raise RuntimeError("full readout requires run_stage_b_attribution=true")
    return {"stage": stage, **cost, "primary_families": len(PRIMARY_FAMILY_ORDER)}


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "route": nested(config, "experiment.route"),
        "parent": nested(config, "lineage.parent"),
        "comparison": nested(config, "lineage.comparison"),
        "endpoint": nested(config, "validation.primary_endpoint"),
        "expected_rows": int(nested(config, "validation.expected_rows")),
        "expected_wells": int(nested(config, "validation.expected_wells")),
        "folds": int(nested(config, "validation.n_folds")),
        "stage_a_forbidden_columns": sorted(STAGE_A_EXACT_FORBIDDEN_COLUMNS),
        "primary_risk_families": expected_family_contract(),
        "decision_gate": dict(nested(config, "audit.decision_gate")),
        "model_training_count": 0,
        "prediction_or_submission_generated": False,
    }


# %% [markdown]
# ## 3. Frozen input and path helpers


# %%
def _existing_pattern_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in patterns:
        candidate = Path(str(raw))
        if candidate.exists():
            paths.append(candidate.resolve())
        if not candidate.is_absolute():
            local = PACKAGE_DIR / candidate
            if local.exists():
                paths.append(local.resolve())
    return list(dict.fromkeys(paths))


def resolve_exp287_artifact_root(config: Mapping[str, Any]) -> Path:
    expected_manifest_sha = str(
        nested(config, "data.expected_exp287_formation_fold_manifest_sha256")
    )
    candidates = _existing_pattern_paths(nested(config, "data.exp287_artifact_root_patterns"))
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent.resolve()
            for path in KAGGLE_INPUT_ROOT.rglob("formation_fold_manifest.json")
        )
    matches: list[Path] = []
    for candidate in dict.fromkeys(candidates):
        manifest = candidate / "formation_fold_manifest.json"
        if manifest.is_file() and sha256_file(manifest) == expected_manifest_sha:
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(f"SHA-matched exp287 artifact root was not unique: {matches}")
    return matches[0]


def resolve_sha_matched_file(
    patterns: Sequence[str],
    *,
    expected_sha256: str,
    fallback_name: str,
    label: str,
) -> Path:
    candidates = _existing_pattern_paths(patterns)
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(path.resolve() for path in KAGGLE_INPUT_ROOT.rglob(fallback_name))
    return select_first_sha_matched_file(
        candidates,
        expected_sha256=expected_sha256,
        label=label,
    )


def select_first_sha_matched_file(
    candidates: Sequence[Path], *, expected_sha256: str, label: str
) -> Path:
    matches = [
        path
        for path in dict.fromkeys(Path(value).resolve() for value in candidates)
        if path.is_file() and sha256_file(path) == str(expected_sha256)
    ]
    if not matches:
        raise FileNotFoundError(f"no SHA-matched {label} was found")
    selected = matches[0]
    if len(matches) > 1:
        print(
            json.dumps(
                {
                    "label": label,
                    "sha_equivalent_match_count": len(matches),
                    "selected_path": str(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return selected


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    candidates = _existing_pattern_paths(nested(config, "data.raw_train_dir_patterns"))
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.rglob("train")
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None
        )
    matches = [
        path
        for path in dict.fromkeys(candidates)
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"raw train directory was not unique: {matches}")
    return matches[0]


def safe_manifest_partition_path(root: Path, relative: str) -> Path:
    root_resolved = Path(root).resolve()
    path = (root_resolved / str(relative)).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"formation partition escapes artifact root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_stage_a_source_columns(columns: Sequence[str]) -> None:
    names = [str(value) for value in columns]
    duplicated = pd.Index(names)[pd.Index(names).duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Stage A source has duplicate columns: {duplicated}")
    forbidden = sorted(name for name in names if is_stage_a_forbidden_column(name))
    if forbidden:
        raise ValueError(f"Stage A source exposes forbidden error columns: {forbidden}")


def load_valid_partition_contracts(
    artifact_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = artifact_root / "formation_fold_manifest.json"
    verify_file_sha(
        manifest_path,
        nested(config, "data.expected_exp287_formation_fold_manifest_sha256"),
        "exp287 formation fold manifest",
    )
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("partition_count", -1)) != 10:
        raise ValueError("formation manifest must contain ten train/valid partitions")
    if int(manifest.get("feature_count", -1)) != 74:
        raise ValueError("formation manifest feature count must be 74")
    expected_schema = str(nested(config, "data.expected_exp287_formation_feature_schema_sha256"))
    if str(manifest.get("feature_schema_sha256")) != expected_schema:
        raise ValueError("formation manifest schema SHA mismatch")
    partitions = list(manifest.get("partitions") or [])
    valid = sorted(
        (dict(row) for row in partitions if str(row.get("role")) == "valid"),
        key=lambda row: int(row["downstream_outer_fold"]),
    )
    if [int(row["downstream_outer_fold"]) for row in valid] != [0, 1, 2, 3, 4]:
        raise ValueError("formation valid partitions must cover outer folds 0..4 exactly once")
    for row in valid:
        if bool(row.get("target_formation_columns_read")):
            raise ValueError("outer-valid target formation columns were read")
        if str(row.get("feature_schema_sha256")) != expected_schema:
            raise ValueError("formation partition schema SHA mismatch")
        if int(row.get("target_wells_inside_reference", -1)) != 0:
            raise ValueError("outer-valid target wells appear inside the reference set")
    return valid, manifest


# %% [markdown]
# ## 4. Stage A formation-cache audit and target-free aggregation


# %%
def numpy_linear_quantile(values: np.ndarray | pd.Series, probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile input must be non-empty and finite")
    return float(np.quantile(array, probability, method="linear"))


def aggregate_partition_target_free(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    families: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    validate_stage_a_source_columns(frame.columns)
    required = {"id", "well", *canonical_formation_feature_names()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"formation partition is missing columns: {missing}")
    if frame["id"].isna().any() or frame["well"].isna().any():
        raise ValueError("formation identity contains missing values")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("formation partition contains duplicate ids")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    numeric = frame[canonical_formation_feature_names()].to_numpy(np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("formation partition contains nonfinite feature values")
    grouped = frame.groupby("well", sort=True, observed=True)
    result = pd.DataFrame({"well": sorted(frame["well"].unique())})
    result["outer_fold"] = int(outer_fold)
    result = result.set_index("well")
    for family in families:
        name = str(family["name"])
        sources = [str(value) for value in family["source_columns"]]
        if family["well_scalar"] == "row_quantile_0p90":
            if len(sources) != 1:
                raise ValueError(f"{name} row-quantile family must have one source")
            source = sources[0]
            values = (
                frame[source].abs()
                if family.get("transform") == "absolute_value"
                else frame[source]
            )
            quantiles = values.groupby(frame["well"], sort=True).apply(
                lambda part: numpy_linear_quantile(part.to_numpy(), 0.90),
                include_groups=False,
            )
            result[name] = quantiles
        elif family["well_scalar"] == "maximum_of_per_well_constant_values":
            records: dict[str, float] = {}
            for well, part in grouped:
                matrix = part[sources].to_numpy(np.float64, copy=False)
                spread = np.ptp(matrix, axis=0)
                if np.any(spread > 1.0e-6):
                    raise ValueError(f"known-prefix calibration columns are not constant: {well}")
                records[str(well)] = float(np.max(matrix[0]))
            result[name] = pd.Series(records)
        else:
            raise ValueError(f"unsupported family scalar: {family['well_scalar']}")
    result["signal_disagreement_sig_std_p90"] = grouped["sig_std"].apply(
        lambda part: numpy_linear_quantile(part.to_numpy(), 0.90),
        include_groups=False,
    )
    dense_context: dict[str, float] = {}
    for well, part in grouped["dense_nb_std"]:
        values = part.to_numpy(np.float64)
        if np.ptp(values) > 1.0e-6:
            raise ValueError(f"dense_nb_std is not constant for well={well}")
        dense_context[str(well)] = float(values[0])
    result["dense_known_neighbor_std_dense_nb_std"] = pd.Series(dense_context)
    result = result.reset_index()
    numeric_result = result.drop(columns=["well"]).to_numpy(np.float64)
    if not np.isfinite(numeric_result).all():
        raise ValueError("target-free well aggregation produced nonfinite values")
    return result


def audit_and_aggregate_valid_partitions(
    artifact_root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    valid_contracts, source_manifest = load_valid_partition_contracts(artifact_root, config)
    expected_columns = ["id", "well", *canonical_formation_feature_names()]
    families = list(nested(config, "audit.primary_risk_families"))
    parts: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    seen_ids: set[str] = set()
    total_rows = 0
    for contract in valid_contracts:
        fold = int(contract["downstream_outer_fold"])
        path = safe_manifest_partition_path(artifact_root, str(contract["path"]))
        file_sha = verify_file_sha(path, contract["file_sha256"], f"formation valid fold {fold}")
        parquet = pq.ParquetFile(path)
        schema_columns = list(parquet.schema_arrow.names)
        validate_stage_a_source_columns(schema_columns)
        if schema_columns != expected_columns:
            raise ValueError(f"formation fold {fold} schema/order changed: {schema_columns[:8]}...")
        frame = parquet.read(columns=expected_columns).to_pandas()
        if len(frame) != int(contract["rows"]):
            raise ValueError(f"formation fold {fold} row count mismatch")
        wells = set(frame["well"].astype(str).unique())
        if len(wells) != int(contract["wells"]):
            raise ValueError(f"formation fold {fold} well count mismatch")
        ids = set(frame["id"].astype(str))
        duplicate_ids = seen_ids.intersection(ids)
        if duplicate_ids:
            raise ValueError(
                f"ids occur in multiple outer-valid partitions: {sorted(duplicate_ids)[:5]}"
            )
        seen_ids.update(ids)
        overlap = seen_wells.intersection(wells)
        if overlap:
            raise ValueError(
                f"wells occur in multiple outer-valid partitions: {sorted(overlap)[:5]}"
            )
        seen_wells.update(wells)
        aggregated = aggregate_partition_target_free(
            frame,
            outer_fold=fold,
            families=families,
        )
        parts.append(aggregated)
        total_rows += len(frame)
        evidence.append(
            {
                "outer_fold": fold,
                "role": "valid",
                "path": str(contract["path"]),
                "file_sha256": file_sha,
                "logical_content_sha256": str(contract["logical_content_sha256"]),
                "rows": len(frame),
                "wells": len(wells),
                "plane_reference_available_wells": int(
                    frame.groupby("well")["spatial_knn_dist"]
                    .apply(lambda x: np.isfinite(x).all())
                    .sum()
                ),
                "dense_reference_available_wells": int(
                    frame.groupby("well")["dense_dist"].apply(lambda x: np.isfinite(x).all()).sum()
                ),
                "formation_nonfinite_values": 0,
            }
        )
        del frame
    attributes = pd.concat(parts, ignore_index=True).sort_values("well").reset_index(drop=True)
    expected_rows = int(nested(config, "validation.expected_rows"))
    expected_wells = int(nested(config, "validation.expected_wells"))
    if total_rows != expected_rows or len(attributes) != expected_wells:
        raise ValueError(f"Stage A coverage mismatch: rows={total_rows}, wells={len(attributes)}")
    if len(seen_ids) != expected_rows:
        raise ValueError("Stage A did not observe every OOF id exactly once")
    if attributes["well"].duplicated().any():
        raise ValueError("Stage A well attributes are duplicated")
    return attributes, evidence, source_manifest


# %% [markdown]
# ## 5. Stage A raw target-free context and freeze barrier
#
# raw horizontal CSV は `MD/X/Y/Z/TVT_input` だけを `usecols` で読む。prefix span は
# known-row MD range、suffix span は last-known MD から max evaluation MD、prefix XY
# span は known-row bounding-box diagonal と固定する。これら context は primary gate
# を通せない。


# %%
def compute_raw_context_for_well(
    frame: pd.DataFrame, *, denominator_floor: float
) -> dict[str, float | int]:
    required = ["MD", "X", "Y", "Z", "TVT_input"]
    if list(frame.columns) != required:
        raise ValueError(f"raw context opened unexpected columns: {frame.columns.tolist()}")
    if not np.isfinite(frame[["MD", "X", "Y", "Z"]].to_numpy(np.float64)).all():
        raise ValueError("raw target-free geometry contains nonfinite values")
    known_mask = frame["TVT_input"].notna().to_numpy()
    evaluation_mask = ~known_mask
    known = frame.loc[known_mask]
    evaluation = frame.loc[evaluation_mask]
    if known.empty or evaluation.empty:
        raise ValueError("raw well must contain both known-prefix and evaluation rows")
    if not np.isfinite(known["TVT_input"].to_numpy(np.float64)).all():
        raise ValueError("known TVT_input contains nonfinite values")
    last_known_index = known["MD"].astype(float).idxmax()
    last_known_md = float(known.loc[last_known_index, "MD"])
    if float(evaluation["MD"].min()) + 1.0e-6 < last_known_md:
        raise ValueError("TVT_input missing rows are not a suffix after the known prefix")
    prefix_md_span = float(known["MD"].max() - known["MD"].min())
    suffix_md_span = float(evaluation["MD"].max() - last_known_md)
    prefix_xy_span = float(
        np.hypot(known["X"].max() - known["X"].min(), known["Y"].max() - known["Y"].min())
    )
    last_xy = known.loc[last_known_index, ["X", "Y"]].to_numpy(np.float64)
    evaluation_xy = evaluation[["X", "Y"]].to_numpy(np.float64)
    distance = np.sqrt(np.sum(np.square(evaluation_xy - last_xy), axis=1))
    return {
        "known_prefix_row_count": int(len(known)),
        "evaluation_row_count": int(len(evaluation)),
        "evaluation_to_known_row_count_ratio": float(len(evaluation) / len(known)),
        "suffix_to_prefix_md_span_ratio": float(
            suffix_md_span / max(prefix_md_span, denominator_floor)
        ),
        "evaluation_xy_distance_from_last_known_p90_div_prefix_xy_span": float(
            numpy_linear_quantile(distance, 0.90) / max(prefix_xy_span, denominator_floor)
        ),
        "prefix_md_span_denominator_floored": int(prefix_md_span < denominator_floor),
        "prefix_xy_span_denominator_floored": int(prefix_xy_span < denominator_floor),
    }


def load_raw_target_free_context(
    raw_train_dir: Path,
    *,
    expected_wells: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed = [str(value) for value in nested(config, "data.raw_context_allowed_columns")]
    if allowed != ["MD", "X", "Y", "Z", "TVT_input"]:
        raise ValueError("raw Stage A allowed-column contract changed")
    floor = float(nested(config, "audit.context_definitions.denominator_floor"))
    files = {
        path.name.split("__horizontal_well.csv")[0]: path
        for path in sorted(Path(raw_train_dir).glob("*__horizontal_well.csv"))
    }
    expected = {str(well) for well in expected_wells}
    if set(files) != expected:
        missing = sorted(expected - set(files))
        extra = sorted(set(files) - expected)
        raise ValueError(f"raw/formation well mismatch: missing={missing[:5]}, extra={extra[:5]}")
    rows: list[dict[str, Any]] = []
    headers_with_forbidden_columns = 0
    forbidden_raw = {str(value) for value in nested(config, "data.raw_context_forbidden_columns")}
    for well in sorted(expected):
        path = files[well]
        header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        if not set(allowed).issubset(header):
            raise ValueError(f"raw context columns missing for {well}")
        headers_with_forbidden_columns += int(bool(set(header).intersection(forbidden_raw)))
        frame = pd.read_csv(path, usecols=allowed)[allowed]
        context = compute_raw_context_for_well(frame, denominator_floor=floor)
        rows.append({"well": well, **context})
    result = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    numeric = result.drop(columns=["well"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("raw target-free context produced nonfinite values")
    audit = {
        "raw_train_dir": str(raw_train_dir),
        "wells": len(result),
        "opened_columns": allowed,
        "forbidden_value_columns_opened": [],
        "headers_with_forbidden_but_unopened_columns": headers_with_forbidden_columns,
        "context_can_pass_primary_gate": False,
    }
    return result, audit


def assign_frozen_quartiles(
    values: pd.Series | np.ndarray,
    edges: Sequence[float],
    *,
    high_is_risky: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.float64)
    if edge_array.shape != (3,) or not np.all(np.diff(edge_array) > 0):
        raise ValueError("quartile edges must be three strictly increasing values")
    if not np.isfinite(array).all():
        raise ValueError("quartile values must be finite")
    quartile = np.searchsorted(edge_array, array, side="left") + 1
    if not high_is_risky:
        quartile = 5 - quartile
    return quartile.astype(np.int8)


def freeze_target_free_attributes(
    attributes: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    partition_evidence: Sequence[Mapping[str, Any]],
    raw_context_audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    attributes = attributes.sort_values("well").reset_index(drop=True).copy()
    if len(attributes) != int(nested(config, "validation.expected_wells")):
        raise ValueError("cannot freeze incomplete well attributes")
    family_rows: list[dict[str, Any]] = []
    for family in nested(config, "audit.primary_risk_families"):
        name = str(family["name"])
        values = attributes[name].to_numpy(np.float64)
        edges = np.quantile(values, [0.25, 0.50, 0.75], method="linear").astype(float)
        eligible = bool(np.isfinite(edges).all() and np.all(np.diff(edges) > 0))
        quartile_column = f"{name}__risk_quartile"
        if eligible:
            attributes[quartile_column] = assign_frozen_quartiles(
                values,
                edges,
                high_is_risky=bool(family["high_is_risky"]),
            )
            counts = attributes[quartile_column].value_counts().sort_index().to_dict()
        else:
            attributes[quartile_column] = np.int8(0)
            counts = {0: len(attributes)}
        family_rows.append(
            {
                **dict(family),
                "q25": float(edges[0]),
                "q50": float(edges[1]),
                "q75": float(edges[2]),
                "eligible": eligible,
                "quartile_counts": {str(key): int(value) for key, value in counts.items()},
                "boundary_source": "target_free_773_well_population_numpy_linear",
                "error_dependent": False,
            }
        )
    ordered_columns = ["well", "outer_fold"]
    for name in PRIMARY_FAMILY_ORDER:
        ordered_columns.extend([name, f"{name}__risk_quartile"])
    ordered_columns.extend(
        [
            "signal_disagreement_sig_std_p90",
            "dense_known_neighbor_std_dense_nb_std",
            "known_prefix_row_count",
            "evaluation_row_count",
            "evaluation_to_known_row_count_ratio",
            "suffix_to_prefix_md_span_ratio",
            "evaluation_xy_distance_from_last_known_p90_div_prefix_xy_span",
            "prefix_md_span_denominator_floored",
            "prefix_xy_span_denominator_floored",
        ]
    )
    if set(attributes.columns) != set(ordered_columns):
        raise ValueError(
            "target-free attribute schema mismatch: "
            f"missing={sorted(set(ordered_columns) - set(attributes.columns))}, "
            f"extra={sorted(set(attributes.columns) - set(ordered_columns))}"
        )
    attributes = attributes[ordered_columns]
    attribute_path = Path(output_dir) / "target_free_well_attributes.csv"
    attribute_sha = write_canonical_csv(attribute_path, attributes)
    freeze_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_a_target_free_attributes_frozen_before_error_join",
        "experiment": EXPERIMENT_NAME,
        "rows": len(attributes),
        "columns": ordered_columns,
        "target_free_well_attributes_sha256": attribute_sha,
        "quantile_method": "numpy_linear",
        "quartile_probabilities": [0.25, 0.50, 0.75],
        "families": family_rows,
        "partition_evidence": [dict(row) for row in partition_evidence],
        "raw_context_audit": dict(raw_context_audit),
        "error_surface_opened": False,
        "forbidden_columns_opened": [],
        "canonical_order": {"rows": ["well"], "columns": ordered_columns},
    }
    freeze_path = Path(output_dir) / "target_free_attribute_freeze_manifest.json"
    write_json(freeze_path, freeze_manifest)
    return attributes, freeze_manifest, sha256_file(freeze_path)


def build_technical_context_readout(
    attributes: pd.DataFrame,
    partition_evidence: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evidence in partition_evidence:
        fold = int(evidence["outer_fold"])
        for key in [
            "plane_reference_available_wells",
            "dense_reference_available_wells",
            "formation_nonfinite_values",
        ]:
            rows.append(
                {
                    "scope": "outer_fold",
                    "outer_fold": fold,
                    "metric": key,
                    "value": float(evidence[key]),
                }
            )
    context_columns = [
        "known_prefix_row_count",
        "evaluation_row_count",
        "evaluation_to_known_row_count_ratio",
        "suffix_to_prefix_md_span_ratio",
        "evaluation_xy_distance_from_last_known_p90_div_prefix_xy_span",
        "signal_disagreement_sig_std_p90",
        "dense_known_neighbor_std_dense_nb_std",
        "prefix_md_span_denominator_floored",
        "prefix_xy_span_denominator_floored",
    ]
    for column in context_columns:
        values = attributes[column].to_numpy(np.float64)
        for statistic, value in [
            ("mean", float(np.mean(values))),
            ("median", float(np.median(values))),
            ("p90", numpy_linear_quantile(values, 0.90)),
            ("max", float(np.max(values))),
        ]:
            rows.append(
                {
                    "scope": "global_context_report_only",
                    "outer_fold": -1,
                    "metric": f"{column}__{statistic}",
                    "value": value,
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "outer_fold", "metric"]).reset_index(drop=True)


def run_stage_a(
    *,
    artifact_root: Path,
    raw_train_dir: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str, dict[str, Any]]:
    formation_attributes, partition_evidence, source_manifest = (
        audit_and_aggregate_valid_partitions(artifact_root, config)
    )
    raw_context, raw_audit = load_raw_target_free_context(
        raw_train_dir,
        expected_wells=formation_attributes["well"].astype(str).tolist(),
        config=config,
    )
    attributes = formation_attributes.merge(
        raw_context, on="well", how="inner", validate="one_to_one"
    )
    if len(attributes) != len(formation_attributes):
        raise ValueError("raw context join changed Stage A well coverage")
    frozen, freeze_manifest, freeze_sha = freeze_target_free_attributes(
        attributes,
        config=config,
        output_dir=output_dir,
        partition_evidence=partition_evidence,
        raw_context_audit=raw_audit,
    )
    technical = build_technical_context_readout(frozen, partition_evidence)
    write_canonical_csv(Path(output_dir) / "technical_context_readout.csv", technical)
    evidence = {
        "exp287_formation_manifest_sha256": sha256_file(
            artifact_root / "formation_fold_manifest.json"
        ),
        "exp287_formation_feature_schema_sha256": str(source_manifest["feature_schema_sha256"]),
        "valid_partitions": partition_evidence,
        "raw_context": raw_audit,
        "target_free_well_attributes_sha256": freeze_manifest["target_free_well_attributes_sha256"],
        "target_free_attribute_freeze_manifest_sha256": freeze_sha,
    }
    return frozen, freeze_manifest, freeze_sha, evidence


# %% [markdown]
# ## 6. Stage B OOF alignment and well-level endpoint


# %%
def load_and_validate_freeze(
    output_dir: Path, *, expected_freeze_manifest_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    freeze_path = Path(output_dir) / "target_free_attribute_freeze_manifest.json"
    verify_file_sha(freeze_path, expected_freeze_manifest_sha256, "Stage A freeze manifest barrier")
    freeze = json.loads(freeze_path.read_text())
    if bool(freeze.get("error_surface_opened")):
        raise ValueError("Stage A freeze manifest says an error surface was opened")
    if list(freeze.get("forbidden_columns_opened") or []):
        raise ValueError("Stage A freeze manifest reports forbidden columns")
    attribute_path = Path(output_dir) / "target_free_well_attributes.csv"
    verify_file_sha(
        attribute_path,
        str(freeze["target_free_well_attributes_sha256"]),
        "frozen target-free attributes",
    )
    attributes = pd.read_csv(attribute_path, dtype={"well": str})
    if attributes.columns.tolist() != list(freeze["columns"]):
        raise ValueError("frozen attribute schema changed before Stage B")
    return attributes, freeze


def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    label: str,
    expected_rows: int,
    expected_wells: int,
) -> pd.DataFrame:
    required = ["id", "well", "outer_fold", "actual_tvt", prediction_column]
    if frame.columns.tolist() != required:
        raise ValueError(f"{label} OOF schema/order mismatch: {frame.columns.tolist()}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError(f"{label} OOF row/well coverage mismatch")
    if frame["id"].duplicated().any() or frame["id"].isna().any():
        raise ValueError(f"{label} OOF identity is not unique and complete")
    numeric = frame[["outer_fold", "actual_tvt", prediction_column]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} OOF contains nonfinite values")
    folds_per_well = frame.groupby("well")["outer_fold"].nunique()
    if not folds_per_well.eq(1).all() or sorted(frame["outer_fold"].unique().tolist()) != [
        0,
        1,
        2,
        3,
        4,
    ]:
        raise ValueError(f"{label} well/fold assignment is invalid")
    return frame


def build_well_oof_delta_metrics(
    exp287: pd.DataFrame,
    exp264: pd.DataFrame,
    *,
    exp287_prediction_column: str,
    exp264_prediction_column: str,
    actual_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = exp287.rename(
        columns={
            "well": "well_exp287",
            "outer_fold": "outer_fold_exp287",
            "actual_tvt": "actual_tvt_exp287",
            exp287_prediction_column: "prediction_exp287",
        }
    )
    right = exp264.rename(
        columns={
            "well": "well_exp264",
            "outer_fold": "outer_fold_exp264",
            "actual_tvt": "actual_tvt_exp264",
            exp264_prediction_column: "prediction_exp264",
        }
    )
    joined = left.merge(right, on="id", how="inner", validate="one_to_one", sort=False)
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("exp287/exp264 OOF identity sets differ")
    if not joined["well_exp287"].equals(joined["well_exp264"]):
        raise ValueError("exp287/exp264 OOF well alignment differs")
    if not joined["outer_fold_exp287"].equals(joined["outer_fold_exp264"]):
        raise ValueError("exp287/exp264 OOF fold alignment differs")
    actual_diff = np.abs(
        joined["actual_tvt_exp287"].to_numpy(np.float64)
        - joined["actual_tvt_exp264"].to_numpy(np.float64)
    )
    if float(np.max(actual_diff)) > float(actual_tolerance):
        raise ValueError("exp287/exp264 actual TVT differs beyond tolerance")
    joined["sq_error_exp287"] = np.square(
        joined["prediction_exp287"].to_numpy(np.float64)
        - joined["actual_tvt_exp287"].to_numpy(np.float64)
    )
    joined["sq_error_exp264"] = np.square(
        joined["prediction_exp264"].to_numpy(np.float64)
        - joined["actual_tvt_exp287"].to_numpy(np.float64)
    )
    grouped = joined.groupby("well_exp287", sort=True, observed=True)
    well = grouped.agg(
        rows=("id", "size"),
        outer_fold=("outer_fold_exp287", "first"),
        exp287_sum_squared_error=("sq_error_exp287", "sum"),
        exp264_sum_squared_error=("sq_error_exp264", "sum"),
    ).reset_index(names="well")
    well["exp287_rmse"] = np.sqrt(well["exp287_sum_squared_error"] / well["rows"])
    well["exp264_rmse"] = np.sqrt(well["exp264_sum_squared_error"] / well["rows"])
    well["delta_rmse_exp287_minus_exp264"] = well["exp287_rmse"] - well["exp264_rmse"]
    audit = {
        "rows": len(joined),
        "wells": len(well),
        "max_actual_tvt_absolute_difference": float(np.max(actual_diff)),
        "id_well_fold_alignment": True,
    }
    return well, audit


def load_hidden_like_assignments(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    verify_file_sha(
        path,
        nested(config, "data.expected_hidden_like_assignment_sha256"),
        "hidden-like assignment",
    )
    required = [
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]
    frame = pd.read_csv(path, usecols=required, dtype={"well_id": str})[required]
    frame = frame.rename(columns={"well_id": "well"})
    if frame["well"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    return frame


# %% [markdown]
# ## 7. Quartile, fold, hidden-like, and decision metrics


# %%
def quartile_metrics_for_family(
    frame: pd.DataFrame, *, family: str, quartile_column: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for quartile in [1, 2, 3, 4]:
        part = frame.loc[frame[quartile_column] == quartile]
        if part.empty:
            rows.append({"family": family, "quartile": quartile, "wells": 0})
            continue
        exp287_row_rmse = float(
            np.sqrt(part["exp287_sum_squared_error"].sum() / part["rows"].sum())
        )
        exp264_row_rmse = float(
            np.sqrt(part["exp264_sum_squared_error"].sum() / part["rows"].sum())
        )
        delta = part["delta_rmse_exp287_minus_exp264"].to_numpy(np.float64)
        rows.append(
            {
                "family": family,
                "quartile": quartile,
                "wells": len(part),
                "rows": int(part["rows"].sum()),
                "risk_min": float(part[family].min()),
                "risk_max": float(part[family].max()),
                "mean_well_delta_rmse": float(np.mean(delta)),
                "median_well_delta_rmse": float(np.median(delta)),
                "exp287_row_weighted_rmse": exp287_row_rmse,
                "exp264_row_weighted_rmse": exp264_row_rmse,
                "row_weighted_rmse_delta": exp287_row_rmse - exp264_row_rmse,
                "worsened_plus_1ft_rate": float(np.mean(delta >= 1.0)),
                "worsened_plus_3ft_rate": float(np.mean(delta >= 3.0)),
                "worsened_plus_5ft_rate": float(np.mean(delta >= 5.0)),
            }
        )
    return pd.DataFrame(rows)


def direction_metrics(
    frame: pd.DataFrame,
    *,
    family: str,
    quartile_column: str,
    scope_column: str,
    scope_values: Sequence[Any],
    minimum_endpoint_wells: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_value in scope_values:
        part = frame.loc[frame[scope_column] == scope_value]
        q1 = part.loc[part[quartile_column] == 1, "delta_rmse_exp287_minus_exp264"]
        q4 = part.loc[part[quartile_column] == 4, "delta_rmse_exp287_minus_exp264"]
        difference = float(q4.mean() - q1.mean()) if len(q1) and len(q4) else float("nan")
        rows.append(
            {
                "family": family,
                "scope": str(scope_value),
                "q1_wells": len(q1),
                "q4_wells": len(q4),
                "q1_mean_well_delta_rmse": float(q1.mean()) if len(q1) else float("nan"),
                "q4_mean_well_delta_rmse": float(q4.mean()) if len(q4) else float("nan"),
                "q4_minus_q1_mean_well_delta_rmse": difference,
                "coverage_pass": bool(
                    len(q1) >= minimum_endpoint_wells and len(q4) >= minimum_endpoint_wells
                ),
                "direction_positive": bool(np.isfinite(difference) and difference > 0.0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_families(
    well: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gate = dict(nested(config, "audit.decision_gate"))
    global_min = int(gate["minimum_global_wells_per_endpoint_quartile"])
    fold_min = int(gate["minimum_wells_per_endpoint_quartile_per_fold"])
    hidden_min = int(gate["minimum_wells_per_endpoint_quartile_per_hidden_like_scope"])
    quartile_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    hidden_parts: list[pd.DataFrame] = []
    family_decisions: list[dict[str, Any]] = []
    family_manifest = {str(row["name"]): dict(row) for row in freeze_manifest["families"]}
    hidden_specs = [
        ("hidden_like_spatial", "verification_like_spatial_role"),
        ("hidden_like_typewell_purged", "verification_like_typewell_purged_role"),
    ]

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    for family in PRIMARY_FAMILY_ORDER:
        quartile_column = f"{family}__risk_quartile"
        quartile = quartile_metrics_for_family(well, family=family, quartile_column=quartile_column)
        quartile_parts.append(quartile)
        q1 = well.loc[well[quartile_column] == 1, "delta_rmse_exp287_minus_exp264"]
        q4 = well.loc[well[quartile_column] == 4, "delta_rmse_exp287_minus_exp264"]
        global_mean = float(q4.mean() - q1.mean()) if len(q1) and len(q4) else float("nan")
        global_median = float(q4.median() - q1.median()) if len(q1) and len(q4) else float("nan")
        fold = direction_metrics(
            well,
            family=family,
            quartile_column=quartile_column,
            scope_column="outer_fold",
            scope_values=[0, 1, 2, 3, 4],
            minimum_endpoint_wells=fold_min,
        )
        fold_parts.append(fold.rename(columns={"scope": "outer_fold"}))
        hidden_rows: list[pd.DataFrame] = []
        for scope_name, role_column in hidden_specs:
            valid = well.loc[well[role_column] == "valid"].copy()
            valid["hidden_like_scope"] = scope_name
            hidden_rows.append(
                direction_metrics(
                    valid,
                    family=family,
                    quartile_column=quartile_column,
                    scope_column="hidden_like_scope",
                    scope_values=[scope_name],
                    minimum_endpoint_wells=hidden_min,
                ).rename(columns={"scope": "hidden_like_scope"})
            )
        hidden = pd.concat(hidden_rows, ignore_index=True)
        hidden_parts.append(hidden)
        eligible = bool(family_manifest[family]["eligible"])
        global_coverage = bool(len(q1) >= global_min and len(q4) >= global_min)
        fold_coverage = bool(fold["coverage_pass"].all())
        hidden_coverage = bool(hidden["coverage_pass"].all())
        positive_folds = int(fold["direction_positive"].sum())
        hidden_positive = bool(hidden["direction_positive"].all())
        effect_pass = bool(
            np.isfinite(global_mean)
            and global_mean >= float(gate["global_q4_minus_q1_mean_delta_rmse_minimum_ft"])
        )
        median_pass = bool(np.isfinite(global_median) and global_median > 0.0)
        fold_direction_pass = positive_folds >= int(gate["minimum_positive_direction_folds"])
        error_independent = not bool(family_manifest[family].get("error_dependent", True))
        checks = {
            "eligible_strict_edges": eligible,
            "global_effect_pass": effect_pass,
            "global_median_pass": median_pass,
            "fold_direction_pass": fold_direction_pass,
            "hidden_like_direction_pass": hidden_positive,
            "global_coverage_pass": global_coverage,
            "fold_coverage_pass": fold_coverage,
            "hidden_like_coverage_pass": hidden_coverage,
            "error_independent_fixed_boundary_pass": error_independent,
        }
        largest_index = well["delta_rmse_exp287_minus_exp264"].abs().idxmax()
        sensitivity = well.drop(index=largest_index)
        sensitivity_q1 = sensitivity.loc[
            sensitivity[quartile_column] == 1, "delta_rmse_exp287_minus_exp264"
        ]
        sensitivity_q4 = sensitivity.loc[
            sensitivity[quartile_column] == 4, "delta_rmse_exp287_minus_exp264"
        ]
        delta_series = well["delta_rmse_exp287_minus_exp264"]
        spearman = (
            float(well[family].corr(delta_series, method="spearman"))
            if well[family].nunique() > 1 and delta_series.nunique() > 1
            else float("nan")
        )
        family_decisions.append(
            {
                "family": family,
                "q1_wells": len(q1),
                "q4_wells": len(q4),
                "global_q4_minus_q1_mean_well_delta_rmse": finite_or_none(global_mean),
                "global_q4_minus_q1_median_well_delta_rmse": finite_or_none(global_median),
                "positive_direction_folds": positive_folds,
                "hidden_like_positive_scopes": int(hidden["direction_positive"].sum()),
                "spearman_risk_vs_well_delta_report_only": finite_or_none(spearman),
                (
                    "single_largest_absolute_delta_well_excluded_q4_minus_q1_mean_report_only"
                ): finite_or_none(float(sensitivity_q4.mean() - sensitivity_q1.mean())),
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )
    family_quartile = (
        pd.concat(quartile_parts, ignore_index=True)
        .sort_values(["family", "quartile"])
        .reset_index(drop=True)
    )
    fold_metrics = (
        pd.concat(fold_parts, ignore_index=True)
        .sort_values(["family", "outer_fold"])
        .reset_index(drop=True)
    )
    fold_metrics["outer_fold"] = fold_metrics["outer_fold"].astype(int)
    hidden_metrics = (
        pd.concat(hidden_parts, ignore_index=True)
        .sort_values(["family", "hidden_like_scope"])
        .reset_index(drop=True)
    )
    passed = [row["family"] for row in family_decisions if row["passed"]]
    decision = {
        "schema_version": "1.0.0",
        "status": "ATTRIBUTION_SUPPORTED" if passed else "NO_STABLE_FORMATION_ATTRIBUTION_CLOSE",
        "primary_family_count": len(PRIMARY_FAMILY_ORDER),
        "passed_family_count": len(passed),
        "passed_families": passed,
        "families": family_decisions,
        "pass_semantics": "separate_single_change_intervention_experiment_is_eligible_only",
        "promotes_exp287_or_exp334": False,
        "authorizes_prediction_inference_or_submission": False,
    }
    return family_quartile, fold_metrics, hidden_metrics, decision


# %% [markdown]
# ## 8. Generated artifacts and reproducibility manifest


# %%
def verify_exp287_stage_b_artifacts(
    artifact_root: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    fixed = {
        "model_manifest": ("model_manifest.json", "expected_exp287_model_manifest_sha256"),
        "metrics": ("metrics.json", "expected_exp287_metrics_sha256"),
        "fold_metrics": ("fold_metrics.csv", "expected_exp287_fold_metrics_sha256"),
        "by_well": ("by_well_metrics.csv", "expected_exp287_by_well_sha256"),
        "formation_relationship": (
            "formation_feature_relationship_audit.csv",
            "expected_exp287_formation_relationship_audit_sha256",
        ),
        "raw_schema_audit": (
            "raw_train_current_test_schema_audit.csv",
            "expected_exp287_raw_schema_audit_sha256",
        ),
    }
    result: dict[str, str] = {}
    for label, (filename, config_key) in fixed.items():
        result[label] = verify_file_sha(
            artifact_root / filename,
            nested(config, f"data.{config_key}"),
            f"exp287 {label}",
        )
    model_manifest = json.loads((artifact_root / "model_manifest.json").read_text())
    if str(model_manifest.get("feature_schema_sha256")) != str(
        nested(config, "data.expected_exp287_model_feature_schema_sha256")
    ):
        raise ValueError("exp287 model feature schema SHA mismatch")
    return result


def final_reproducibility_manifest(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    freeze_manifest_sha256: str,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    expected_outputs = [str(value) for value in nested(config, "audit.expected_outputs")]
    artifact_sha: dict[str, str] = {}
    for name in expected_outputs:
        path = Path(output_dir) / name
        if name == "reproducibility_manifest.json":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"expected exp336 output is missing: {path}")
        artifact_sha[name] = sha256_file(path)
    return {
        "schema_version": "1.0.0",
        "status": str(decision["status"]),
        "experiment": EXPERIMENT_NAME,
        "runtime": {
            "kaggle_cpu": True,
            "gpu": False,
            "internet": False,
            "num_workers": int(nested(config, "runtime.num_workers")),
            "blas_threads": int(nested(config, "runtime.blas_threads")),
        },
        "seed_policy": nested(config, "reproducibility.seed_policy"),
        "stochastic_components": [],
        "canonical_order": "well/family/scope/fold/quartile and sorted JSON keys",
        "stage_a_freeze_manifest_sha256_recorded_before_stage_b": freeze_manifest_sha256,
        "input_artifacts": dict(input_manifest),
        "artifact_sha256": artifact_sha,
        "model_prediction_submission_sha256": None,
        "deterministic_submission_anchor": False,
    }


def run_stage_b(
    *,
    artifact_root: Path,
    exp264_oof_path: Path,
    hidden_like_path: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    expected_freeze_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes, freeze = load_and_validate_freeze(
        output_dir,
        expected_freeze_manifest_sha256=expected_freeze_manifest_sha256,
    )
    parent_evidence = verify_exp287_stage_b_artifacts(artifact_root, config)
    exp287_oof_path = artifact_root / str(nested(config, "data.exp287_oof_filename"))
    exp287_sha = verify_file_sha(
        exp287_oof_path,
        nested(config, "data.expected_exp287_oof_sha256"),
        "exp287 OOF",
    )
    exp264_sha = verify_file_sha(
        exp264_oof_path,
        nested(config, "data.expected_exp264_oof_sha256"),
        "corrected exp264 OOF",
    )
    expected_rows = int(nested(config, "validation.expected_rows"))
    expected_wells = int(nested(config, "validation.expected_wells"))
    exp287_prediction = str(nested(config, "data.exp287_prediction_column"))
    exp264_prediction = str(nested(config, "data.exp264_prediction_column"))
    exp287 = validate_oof_frame(
        pd.read_parquet(
            exp287_oof_path,
            columns=["id", "well", "outer_fold", "actual_tvt", exp287_prediction],
        ),
        prediction_column=exp287_prediction,
        label="exp287",
        expected_rows=expected_rows,
        expected_wells=expected_wells,
    )
    exp264 = validate_oof_frame(
        pd.read_parquet(
            exp264_oof_path,
            columns=["id", "well", "outer_fold", "actual_tvt", exp264_prediction],
        ),
        prediction_column=exp264_prediction,
        label="corrected exp264",
        expected_rows=expected_rows,
        expected_wells=expected_wells,
    )
    well, alignment_audit = build_well_oof_delta_metrics(
        exp287,
        exp264,
        exp287_prediction_column=exp287_prediction,
        exp264_prediction_column=exp264_prediction,
        actual_tolerance=float(
            nested(config, "audit.stage_b_error_join.actual_tvt_absolute_tolerance")
        ),
    )
    well = well.merge(attributes, on=["well", "outer_fold"], how="inner", validate="one_to_one")
    if len(well) != expected_wells:
        raise ValueError("Stage B frozen-attribute join changed well coverage")
    hidden = load_hidden_like_assignments(hidden_like_path, config)
    well = well.merge(hidden, on="well", how="left", validate="one_to_one")
    if (
        well[["verification_like_spatial_role", "verification_like_typewell_purged_role"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Stage B hidden-like assignment is incomplete")
    family_quartile, fold_metrics, hidden_metrics, decision = evaluate_frozen_families(
        well, freeze, config
    )
    well = well.sort_values("well").reset_index(drop=True)
    write_canonical_csv(Path(output_dir) / "well_oof_delta_metrics.csv", well)
    write_canonical_csv(Path(output_dir) / "family_quartile_metrics.csv", family_quartile)
    write_canonical_csv(Path(output_dir) / "fold_direction_metrics.csv", fold_metrics)
    write_canonical_csv(Path(output_dir) / "hidden_like_direction_metrics.csv", hidden_metrics)
    pooled_exp287 = rmse(exp287["actual_tvt"], exp287[exp287_prediction])
    pooled_exp264 = rmse(exp264["actual_tvt"], exp264[exp264_prediction])
    decision["pooled_oof_report_only"] = {
        "exp287_rmse": pooled_exp287,
        "exp264_rmse": pooled_exp264,
        "delta_rmse": pooled_exp287 - pooled_exp264,
        "well_delta_mean": float(well["delta_rmse_exp287_minus_exp264"].mean()),
        "well_delta_median": float(well["delta_rmse_exp287_minus_exp264"].median()),
        "worsened_plus_1ft_wells": int((well["delta_rmse_exp287_minus_exp264"] >= 1.0).sum()),
        "worsened_plus_3ft_wells": int((well["delta_rmse_exp287_minus_exp264"] >= 3.0).sum()),
        "worsened_plus_5ft_wells": int((well["delta_rmse_exp287_minus_exp264"] >= 5.0).sum()),
        "largest_absolute_delta_well_report_only": str(
            well.loc[well["delta_rmse_exp287_minus_exp264"].abs().idxmax(), "well"]
        ),
    }
    write_json(Path(output_dir) / "attribution_decision.json", decision)
    evidence = {
        "freeze_manifest_sha256_verified_before_oof_open": expected_freeze_manifest_sha256,
        "exp287_oof_sha256": exp287_sha,
        "exp264_oof_sha256": exp264_sha,
        "hidden_like_assignment_sha256": sha256_file(hidden_like_path),
        "exp287_support_artifact_sha256": parent_evidence,
        "alignment_audit": alignment_audit,
    }
    return decision, evidence


# %% [markdown]
# ## 9. Execution orchestration


# %%
def run_full_attribution_readout(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_scientific_contract(config, require_execution=True)
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    scientific_contract = build_scientific_contract(config)
    write_json(output_dir / "scientific_contract.json", scientific_contract)
    artifact_root = resolve_exp287_artifact_root(config)
    raw_train_dir = resolve_raw_train_dir(config)
    _, _, freeze_sha, stage_a_evidence = run_stage_a(
        artifact_root=artifact_root,
        raw_train_dir=raw_train_dir,
        config=config,
        output_dir=output_dir,
    )
    # Freeze SHA is materialized in the input manifest before either OOF is opened.
    input_manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "stage_a_frozen_before_stage_b",
        "scientific_contract_sha256": sha256_file(output_dir / "scientific_contract.json"),
        "stage_a": stage_a_evidence,
        "stage_a_freeze_manifest_sha256": freeze_sha,
        "oof_surfaces_opened": False,
    }
    write_json(output_dir / "input_artifact_manifest.json", input_manifest)
    exp264_path = resolve_sha_matched_file(
        nested(config, "data.exp264_oof_patterns"),
        expected_sha256=str(nested(config, "data.expected_exp264_oof_sha256")),
        fallback_name="stage_d_oof_predictions.parquet",
        label="corrected exp264 OOF",
    )
    hidden_path = resolve_sha_matched_file(
        nested(config, "data.hidden_like_assignment_patterns"),
        expected_sha256=str(nested(config, "data.expected_hidden_like_assignment_sha256")),
        fallback_name="exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
        label="hidden-like assignment",
    )
    decision, stage_b_evidence = run_stage_b(
        artifact_root=artifact_root,
        exp264_oof_path=exp264_path,
        hidden_like_path=hidden_path,
        config=config,
        output_dir=output_dir,
        expected_freeze_manifest_sha256=freeze_sha,
    )
    input_manifest.update(
        {
            "stage": "stage_b_complete",
            "oof_surfaces_opened": True,
            "stage_b": stage_b_evidence,
        }
    )
    write_json(output_dir / "input_artifact_manifest.json", input_manifest)
    reproducibility = final_reproducibility_manifest(
        output_dir,
        config=config,
        input_manifest=input_manifest,
        freeze_manifest_sha256=freeze_sha,
        decision=decision,
    )
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "execution_contract": contract,
        "decision": decision["status"],
        "passed_families": decision["passed_families"],
        "stage_a_freeze_manifest_sha256": freeze_sha,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "model_prediction_submission_generated": False,
    }
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    display(pd.read_csv(output_dir / "family_quartile_metrics.csv"))
    display(pd.read_csv(output_dir / "fold_direction_metrics.csv"))
    display(pd.read_csv(output_dir / "hidden_like_direction_metrics.csv"))
    return summary


CONFIG_PATH = find_config_path()
CONFIG = read_yaml(CONFIG_PATH)
IMPLEMENTATION_CONTRACT = validate_scientific_contract(CONFIG, require_execution=False)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "parent": PARENT_EXPERIMENT,
            "comparison": COMPARISON_EXPERIMENT,
            "route": CONFIG["experiment"]["route"],
            "status": CONFIG["experiment"]["status"],
            "config_path": str(CONFIG_PATH),
            "implementation_contract": IMPLEMENTATION_CONTRACT,
            "run_stage_a": CONFIG["execution"]["run_stage_a_freeze"],
            "run_stage_b": CONFIG["execution"]["run_stage_b_attribution"],
            "kaggle_push_approved": CONFIG["execution"]["kaggle_push_approved"],
        },
        sort_keys=True,
        indent=2,
    )
)

if not IMPORT_ONLY:
    if CONFIG["execution"]["active_stage"] == "full_attribution_readout":
        RUN_SUMMARY = run_full_attribution_readout(CONFIG)
    else:
        print(
            "Implementation-only state: no parent artifact, OOF, model, inference, or "
            "submission execution was started."
        )
