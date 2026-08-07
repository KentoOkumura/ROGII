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
# # exp256 segment-local corridor near-bucket signal attribution readout
#
# exp250 Stage 1 の保存済み生成物だけを読み、0--100 ft の pooled AUC 約 0.82 を
# distance / candidate family / well の base error と risk=1 飽和へ分解する。
# corridor、candidate、model、control は再計算せず、予測・feature・submission を作らない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime, configuration, and SHA helpers
# 3. Fixed-input and paired-contract helpers
# 4. Distance-conditioned AUC helpers
# 5. Family-by-well attribution helpers
# 6. Risk saturation, plot, and output helpers
# 7. Setup and fixed contract
# 8. Load and freeze exp250 inputs
# 9. Run attribution readouts
# 10. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

# %% [markdown]
# ## 2. Runtime, configuration, and SHA helpers

# %%
EXPERIMENT_NAME = "exp256_segment_local_corridor_near_bucket_signal_attribution_readout"
PARENT_EXPERIMENT = "exp250_segment_local_negative_space_gr_corridor_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def get_nested(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"config.yaml was not found; checked={candidates}")


CONFIG_PATH = find_config_path()
with CONFIG_PATH.open() as fp:
    config = yaml.safe_load(fp) or {}
if not isinstance(config, dict):
    raise ValueError("config.yaml must contain a mapping")


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_authorized_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "This readout must run first on Kaggle. Set EXPERIMENT_ALLOW_LOCAL=1 only "
        "for an explicitly authorized local smoke run."
    )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    text = "\n".join(f"{column}:{frame[column].dtype}" for column in frame.columns)
    return hashlib.sha256(text.encode()).hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve_input(spec: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    checked: list[str] = []
    for raw_value in spec.get("paths") or []:
        path = Path(str(raw_value))
        if not path.is_absolute():
            path = ROOT / path
        checked.append(str(path))
        if not path.exists() or path.stat().st_size == 0:
            continue
        sha_kind = str(spec.get("sha_kind") or "raw")
        expected = str(spec.get("expected_sha256") or "")
        actual = sha256_path(path, decompressed=sha_kind == "decompressed")
        if actual != expected:
            raise RuntimeError(
                f"{label} SHA mismatch: actual={actual} expected={expected} path={path}"
            )
        record = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha_kind": sha_kind,
            "sha256": actual,
            "raw_sha256": sha256_path(path),
        }
        if sha_kind == "decompressed":
            record["decompressed_content_sha256"] = actual
        return path, record
    raise FileNotFoundError(f"{label} was not found; checked={checked}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_close_arrays(
    left: pd.Series,
    right: pd.Series,
    label: str,
    tolerance: float,
) -> None:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
    same_nan = np.isnan(left_values) == np.isnan(right_values)
    finite_equal = np.isclose(
        np.nan_to_num(left_values),
        np.nan_to_num(right_values),
        atol=tolerance,
        rtol=0.0,
    )
    if not np.all(same_nan & finite_equal):
        count = int((~(same_nan & finite_equal)).sum())
        raise AssertionError(f"{label} differs for {count} paired rows")


# %% [markdown]
# ## 3. Fixed-input and paired-contract helpers


# %%
def safe_weighted_auc(
    scores: np.ndarray,
    positive_weight: np.ndarray,
    negative_weight: np.ndarray,
) -> float:
    finite = np.isfinite(scores) & np.isfinite(positive_weight) & np.isfinite(negative_weight)
    scores = scores[finite]
    positive_weight = positive_weight[finite]
    negative_weight = negative_weight[finite]
    total_positive = float(positive_weight.sum())
    total_negative = float(negative_weight.sum())
    if not len(scores) or total_positive <= 0 or total_negative <= 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    positive_weight = positive_weight[order]
    negative_weight = negative_weight[order]
    contribution = 0.0
    cumulative_negative = 0.0
    index = 0
    tolerance = 1.0e-12
    while index < len(scores):
        stop = index + 1
        while stop < len(scores) and abs(scores[stop] - scores[index]) <= tolerance:
            stop += 1
        positive_tie = float(positive_weight[index:stop].sum())
        negative_tie = float(negative_weight[index:stop].sum())
        contribution += positive_tie * (cumulative_negative + 0.5 * negative_tie)
        cumulative_negative += negative_tie
        index = stop
    return contribution / (total_positive * total_negative)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if not len(values):
        return math.nan
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cutoff = quantile * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def load_candidate_pairs(
    path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = [
        "well",
        "segment_id",
        "variant",
        "candidate",
        "start_md",
        "end_md",
        "risk",
        "bad_weight",
        "good_weight",
        "real_minus_shuffled_risk",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    require_columns(frame, usecols, "candidate_segment")
    expected_rows = int(get_nested(config, "data.expected_candidate_segment_rows"))
    expected_wells = int(get_nested(config, "data.expected_wells"))
    expected_candidates = set(get_nested(config, "data.candidates"))
    expected_variants = set(get_nested(config, "data.variants"))
    if len(frame) != expected_rows:
        raise ValueError(f"candidate rows={len(frame)} expected={expected_rows}")
    if frame["well"].astype(str).nunique() != expected_wells:
        raise ValueError("candidate well count mismatch")
    if set(frame["candidate"].astype(str)) != expected_candidates:
        raise ValueError("candidate family set mismatch")
    if set(frame["variant"].astype(str)) != expected_variants:
        raise ValueError("candidate variant set mismatch")

    key = ["well", "segment_id", "candidate"]
    if frame[key + ["variant"]].duplicated().any():
        raise ValueError("candidate_segment has duplicate paired keys")
    real = frame.loc[frame["variant"] == "real_gr"].copy()
    shuffled = frame.loc[frame["variant"] == "shuffled_typewell_gr"].copy()
    value_columns = [
        "start_md",
        "end_md",
        "risk",
        "bad_weight",
        "good_weight",
        "real_minus_shuffled_risk",
    ]
    real = real[key + value_columns].rename(
        columns={column: f"real_{column}" for column in value_columns}
    )
    shuffled = shuffled[key + value_columns].rename(
        columns={column: f"shuffled_{column}" for column in value_columns}
    )
    paired = real.merge(shuffled, on=key, how="inner", validate="one_to_one")
    expected_pairs = int(get_nested(config, "data.expected_candidate_pairs"))
    if len(paired) != expected_pairs:
        raise ValueError(f"candidate pairs={len(paired)} expected={expected_pairs}")
    tolerance = float(get_nested(config, "audit.equality_tolerance", 1.0e-12))
    for column in ["start_md", "end_md", "bad_weight", "good_weight"]:
        assert_close_arrays(
            paired[f"real_{column}"],
            paired[f"shuffled_{column}"],
            f"paired {column}",
            tolerance,
        )
    expected_delta = paired["real_risk"] - paired["shuffled_risk"]
    assert_close_arrays(
        expected_delta,
        paired["real_real_minus_shuffled_risk"],
        "real paired risk delta",
        tolerance,
    )
    assert_close_arrays(
        expected_delta,
        paired["shuffled_real_minus_shuffled_risk"],
        "shuffled paired risk delta",
        tolerance,
    )
    paired = paired.assign(
        start_md=paired["real_start_md"],
        end_md=paired["real_end_md"],
        bad_weight=paired["real_bad_weight"].astype(float),
        good_weight=paired["real_good_weight"].astype(float),
        real_minus_shuffled_risk=expected_delta,
    )
    paired["evaluated_weight"] = paired["bad_weight"] + paired["good_weight"]
    paired = paired.sort_values(key, kind="mergesort").reset_index(drop=True)
    contract = {
        "rows": int(len(frame)),
        "pairs": int(len(paired)),
        "wells": int(paired["well"].nunique()),
        "candidates": sorted(paired["candidate"].unique().tolist()),
        "paired_weight_identity": True,
        "paired_segment_identity": True,
        "paired_risk_delta_identity": True,
    }
    return paired, contract


def validate_parent_summary(summary: dict[str, Any]) -> None:
    if summary.get("experiment") != PARENT_EXPERIMENT:
        raise ValueError("parent summary experiment mismatch")
    if summary.get("status") != "stage1_complete":
        raise ValueError("parent Stage 1 is not complete")
    if summary.get("decision") != "fail_close_segment_local_hard_use_and_grid_search":
        raise ValueError("parent fail-close decision mismatch")


# %% [markdown]
# ## 4. Distance-conditioned AUC helpers

# %%
DISTANCE_METRICS = [
    "sample_segment_count",
    "evaluated_row_weight",
    "bad_row_weight",
    "good_row_weight",
    "auc",
    "q90_risk_threshold",
    "q90_bad_rate",
    "overall_bad_rate",
    "q90_bad_rate_lift",
    "q90_good_false_alert_rate",
]


def build_distance_pairs(
    group_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = [
        "entity",
        "variant",
        "candidate",
        "group_type",
        "group_name",
        *DISTANCE_METRICS,
    ]
    require_columns(group_metrics, required, "group_metrics")
    distance = group_metrics.loc[
        (group_metrics["entity"] == "candidate") & (group_metrics["group_type"] == "distance")
    ].copy()
    key = ["candidate", "group_name"]
    real = distance.loc[distance["variant"] == "real_gr", key + DISTANCE_METRICS]
    shuffled = distance.loc[distance["variant"] == "shuffled_typewell_gr", key + DISTANCE_METRICS]
    real = real.rename(columns={column: f"real_{column}" for column in DISTANCE_METRICS})
    shuffled = shuffled.rename(
        columns={column: f"shuffled_{column}" for column in DISTANCE_METRICS}
    )
    paired = real.merge(shuffled, on=key, how="inner", validate="one_to_one")
    tolerance = float(get_nested(config, "audit.equality_tolerance", 1.0e-12))
    identity_columns = [
        "sample_segment_count",
        "evaluated_row_weight",
        "bad_row_weight",
        "good_row_weight",
        "overall_bad_rate",
    ]
    for column in identity_columns:
        assert_close_arrays(
            paired[f"real_{column}"],
            paired[f"shuffled_{column}"],
            f"distance paired {column}",
            tolerance,
        )
        paired[column] = paired[f"real_{column}"]
    labels = list(get_nested(config, "audit.distance_labels"))
    if set(paired["group_name"]) != set(labels):
        raise ValueError("distance labels mismatch")
    paired["distance_order"] = paired["group_name"].map(
        {label: index for index, label in enumerate(labels)}
    )
    paired["pair_mass"] = paired["bad_row_weight"] * paired["good_row_weight"]
    paired["auc_delta"] = paired["real_auc"] - paired["shuffled_auc"]
    paired["near_bucket"] = paired["group_name"].isin(get_nested(config, "audit.near_labels"))
    totals = paired.groupby("candidate", observed=False)["evaluated_row_weight"].transform("sum")
    paired["evaluated_weight_share_within_candidate"] = paired["evaluated_row_weight"] / totals
    return paired.sort_values(["distance_order", "candidate"], kind="mergesort").reset_index(
        drop=True
    )


def stratified_auc(frame: pd.DataFrame, column: str) -> tuple[float, float, int]:
    valid = (
        np.isfinite(frame[column].to_numpy(float))
        & np.isfinite(frame["pair_mass"].to_numpy(float))
        & (frame["pair_mass"].to_numpy(float) > 0)
    )
    if not valid.any():
        return math.nan, 0.0, 0
    weights = frame.loc[valid, "pair_mass"].to_numpy(float)
    values = frame.loc[valid, column].to_numpy(float)
    return float(np.average(values, weights=weights)), float(weights.sum()), int(valid.sum())


def build_distance_conditional_summary(
    distance_pairs: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for distance in get_nested(config, "audit.distance_labels"):
        frame = distance_pairs.loc[distance_pairs["group_name"] == distance]
        pooled = frame.loc[frame["candidate"] == "__pooled__"]
        if len(pooled) != 1:
            raise ValueError(f"missing pooled distance row: {distance}")
        pooled_row = pooled.iloc[0]
        families = frame.loc[frame["candidate"] != "__pooled__"].copy()
        real_conditional, pair_mass, estimable = stratified_auc(families, "real_auc")
        shuffled_conditional, shuffled_mass, shuffled_estimable = stratified_auc(
            families, "shuffled_auc"
        )
        if pair_mass != shuffled_mass or estimable != shuffled_estimable:
            raise AssertionError("real/shuffled conditional support mismatch")
        finite = families["auc_delta"].notna() & (families["pair_mass"] > 0)
        rows.append(
            {
                "distance_bucket": distance,
                "near_bucket": bool(pooled_row["near_bucket"]),
                "evaluated_row_weight": float(pooled_row["evaluated_row_weight"]),
                "bad_row_weight": float(pooled_row["bad_row_weight"]),
                "good_row_weight": float(pooled_row["good_row_weight"]),
                "bad_rate": float(pooled_row["overall_bad_rate"]),
                "pooled_real_auc": float(pooled_row["real_auc"]),
                "pooled_shuffled_auc": float(pooled_row["shuffled_auc"]),
                "pooled_auc_delta": float(pooled_row["auc_delta"]),
                "within_family_real_auc": real_conditional,
                "within_family_shuffled_auc": shuffled_conditional,
                "within_family_auc_delta": real_conditional - shuffled_conditional,
                "pooled_minus_within_family_real_auc": (
                    float(pooled_row["real_auc"]) - real_conditional
                ),
                "estimable_family_count": estimable,
                "positive_auc_delta_family_count": int(
                    (families.loc[finite, "auc_delta"] > 0).sum()
                ),
                "family_count": int(len(families)),
                "within_family_pair_mass": pair_mass,
                "real_q90_risk_threshold": float(pooled_row["real_q90_risk_threshold"]),
                "shuffled_q90_risk_threshold": float(pooled_row["shuffled_q90_risk_threshold"]),
            }
        )
    return pd.DataFrame(rows)


def build_scope_summary(
    distance_pairs: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    labels = list(get_nested(config, "audit.distance_labels"))
    near_labels = set(get_nested(config, "audit.near_labels"))
    scopes = {
        "near_000_100": [label for label in labels if label in near_labels],
        "far_100_plus": [label for label in labels if label not in near_labels],
        "all_distance_buckets": labels,
    }
    pooled_all = distance_pairs.loc[distance_pairs["candidate"] == "__pooled__"]
    all_weight = float(pooled_all["evaluated_row_weight"].sum())
    rows: list[dict[str, Any]] = []
    for scope_name, scope_labels in scopes.items():
        scope = distance_pairs.loc[distance_pairs["group_name"].isin(scope_labels)]
        pooled = scope.loc[scope["candidate"] == "__pooled__"].copy()
        families = scope.loc[scope["candidate"] != "__pooled__"].copy()
        scope_weight = float(pooled["evaluated_row_weight"].sum())
        bad_weight = float(pooled["bad_row_weight"].sum())
        real_conditional, pair_mass, estimable = stratified_auc(families, "real_auc")
        shuffled_conditional, shuffled_mass, shuffled_estimable = stratified_auc(
            families, "shuffled_auc"
        )
        if pair_mass != shuffled_mass or estimable != shuffled_estimable:
            raise AssertionError("scope real/shuffled support mismatch")
        pooled_real_mean = float(
            np.average(pooled["real_auc"], weights=pooled["evaluated_row_weight"])
        )
        pooled_shuffled_mean = float(
            np.average(pooled["shuffled_auc"], weights=pooled["evaluated_row_weight"])
        )
        finite = families["auc_delta"].notna() & (families["pair_mass"] > 0)
        rows.append(
            {
                "scope": scope_name,
                "distance_buckets": ",".join(scope_labels),
                "evaluated_row_weight": scope_weight,
                "evaluated_weight_share": scope_weight / all_weight,
                "bad_row_weight": bad_weight,
                "good_row_weight": scope_weight - bad_weight,
                "bad_rate": bad_weight / scope_weight,
                "pooled_auc_weighted_bucket_mean_real": pooled_real_mean,
                "pooled_auc_weighted_bucket_mean_shuffled": pooled_shuffled_mean,
                "pooled_auc_weighted_bucket_mean_delta": (pooled_real_mean - pooled_shuffled_mean),
                "distance_family_conditional_real_auc": real_conditional,
                "distance_family_conditional_shuffled_auc": shuffled_conditional,
                "distance_family_conditional_auc_delta": (real_conditional - shuffled_conditional),
                "pooled_minus_conditional_real_auc": (pooled_real_mean - real_conditional),
                "candidate_family_bucket_strata": int(len(families)),
                "estimable_candidate_family_bucket_strata": estimable,
                "positive_auc_delta_strata": int((families.loc[finite, "auc_delta"] > 0).sum()),
                "unique_estimable_candidate_families": int(
                    families.loc[finite, "candidate"].nunique()
                ),
                "conditional_pair_mass": pair_mass,
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 5. Family-by-well attribution helpers


# %%
def risk_one_fraction(
    scores: np.ndarray,
    weights: np.ndarray,
    saturation_value: float,
    tolerance: float,
) -> tuple[float, float]:
    finite = np.isfinite(scores)
    if not finite.any():
        return math.nan, math.nan
    saturated = finite & np.isclose(scores, saturation_value, atol=tolerance, rtol=0.0)
    sample_fraction = float(saturated.sum() / finite.sum())
    finite_weight = float(weights[finite].sum())
    weight_fraction = (
        float(weights[saturated].sum() / finite_weight) if finite_weight > 0 else math.nan
    )
    return sample_fraction, weight_fraction


def build_family_well_attribution(
    pairs: pd.DataFrame,
    by_well: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    saturation_value = float(get_nested(config, "audit.risk_saturation_value"))
    tolerance = float(get_nested(config, "audit.equality_tolerance", 1.0e-12))
    rows: list[dict[str, Any]] = []
    for (well, candidate), frame in pairs.groupby(["well", "candidate"], sort=True, observed=False):
        bad = frame["bad_weight"].to_numpy(float)
        good = frame["good_weight"].to_numpy(float)
        evaluated = bad + good
        real_scores = frame["real_risk"].to_numpy(float)
        shuffled_scores = frame["shuffled_risk"].to_numpy(float)
        real_auc = safe_weighted_auc(real_scores, bad, good)
        shuffled_auc = safe_weighted_auc(shuffled_scores, bad, good)
        real_sample_sat, real_weight_sat = risk_one_fraction(
            real_scores, evaluated, saturation_value, tolerance
        )
        shuffled_sample_sat, shuffled_weight_sat = risk_one_fraction(
            shuffled_scores, evaluated, saturation_value, tolerance
        )
        bad_total = float(bad.sum())
        good_total = float(good.sum())
        rows.append(
            {
                "well": str(well),
                "candidate": str(candidate),
                "segment_count": int(len(frame)),
                "evaluated_row_weight": bad_total + good_total,
                "bad_row_weight": bad_total,
                "good_row_weight": good_total,
                "bad_rate": (
                    bad_total / (bad_total + good_total) if bad_total + good_total > 0 else math.nan
                ),
                "pair_mass": bad_total * good_total,
                "real_auc": real_auc,
                "shuffled_auc": shuffled_auc,
                "auc_delta": real_auc - shuffled_auc,
                "real_risk_one_sample_fraction": real_sample_sat,
                "real_risk_one_evaluated_weight_fraction": real_weight_sat,
                "shuffled_risk_one_sample_fraction": shuffled_sample_sat,
                "shuffled_risk_one_evaluated_weight_fraction": shuffled_weight_sat,
            }
        )
    attribution = pd.DataFrame(rows)
    estimable = (
        attribution["real_auc"].notna()
        & attribution["shuffled_auc"].notna()
        & (attribution["pair_mass"] > 0)
    )
    total_pair_mass = float(attribution.loc[estimable, "pair_mass"].sum())
    attribution["delta_contribution_to_stratified_auc"] = np.where(
        estimable,
        attribution["pair_mass"] * attribution["auc_delta"] / total_pair_mass,
        np.nan,
    )
    attribution["estimable"] = estimable

    parent_columns = [
        "well",
        "pooled_real_auc",
        "good_candidate_false_alert_rate",
        "bad_candidate_high_risk_recall",
        "truth_corridor_coverage_real_gr",
        "truth_corridor_coverage_shuffled_typewell_gr",
    ]
    require_columns(by_well, parent_columns, "by_well")
    if by_well["well"].duplicated().any():
        raise ValueError("by_well has duplicate wells")
    parent = by_well[parent_columns].rename(
        columns={column: f"parent_{column}" for column in parent_columns if column != "well"}
    )
    attribution = attribution.merge(parent, on="well", how="left", validate="many_to_one")
    if attribution["parent_pooled_real_auc"].isna().all():
        raise ValueError("by_well join failed")

    family_rows: list[dict[str, Any]] = []
    for candidate, frame in attribution.groupby("candidate", sort=True):
        valid = frame.loc[frame["estimable"]].copy()
        pair_mass = float(valid["pair_mass"].sum())
        family_rows.append(
            {
                "candidate": candidate,
                "well_count": int(frame["well"].nunique()),
                "estimable_well_count": int(len(valid)),
                "positive_auc_delta_well_count": int((valid["auc_delta"] > 0).sum()),
                "negative_auc_delta_well_count": int((valid["auc_delta"] < 0).sum()),
                "zero_auc_delta_well_count": int((valid["auc_delta"] == 0).sum()),
                "estimable_pair_mass": pair_mass,
                "positive_auc_delta_pair_mass_share": (
                    float(valid.loc[valid["auc_delta"] > 0, "pair_mass"].sum() / pair_mass)
                    if pair_mass > 0
                    else math.nan
                ),
                "family_well_conditional_real_auc": (
                    float(np.average(valid["real_auc"], weights=valid["pair_mass"]))
                    if pair_mass > 0
                    else math.nan
                ),
                "family_well_conditional_shuffled_auc": (
                    float(np.average(valid["shuffled_auc"], weights=valid["pair_mass"]))
                    if pair_mass > 0
                    else math.nan
                ),
                "family_well_conditional_auc_delta": (
                    float(np.average(valid["auc_delta"], weights=valid["pair_mass"]))
                    if pair_mass > 0
                    else math.nan
                ),
                "delta_contribution_to_global_stratified_auc": float(
                    valid["delta_contribution_to_stratified_auc"].sum()
                ),
                "median_well_auc_delta": (
                    float(valid["auc_delta"].median()) if len(valid) else math.nan
                ),
            }
        )
    family_summary = pd.DataFrame(family_rows)

    well_rows: list[dict[str, Any]] = []
    for well, frame in attribution.groupby("well", sort=True):
        valid = frame.loc[frame["estimable"]].copy()
        pair_mass = float(valid["pair_mass"].sum())
        row = {
            "well": well,
            "candidate_count": int(len(frame)),
            "estimable_candidate_count": int(len(valid)),
            "positive_auc_delta_candidate_count": int((valid["auc_delta"] > 0).sum()),
            "estimable_pair_mass": pair_mass,
            "well_conditional_real_auc": (
                float(np.average(valid["real_auc"], weights=valid["pair_mass"]))
                if pair_mass > 0
                else math.nan
            ),
            "well_conditional_shuffled_auc": (
                float(np.average(valid["shuffled_auc"], weights=valid["pair_mass"]))
                if pair_mass > 0
                else math.nan
            ),
            "well_conditional_auc_delta": (
                float(np.average(valid["auc_delta"], weights=valid["pair_mass"]))
                if pair_mass > 0
                else math.nan
            ),
            "delta_contribution_to_global_stratified_auc": float(
                valid["delta_contribution_to_stratified_auc"].sum()
            ),
        }
        for column in parent.columns:
            if column == "well":
                continue
            row[column] = frame[column].iloc[0]
        well_rows.append(row)
    well_summary = pd.DataFrame(well_rows)

    valid = attribution.loc[attribution["estimable"]]
    real_global = float(np.average(valid["real_auc"], weights=valid["pair_mass"]))
    shuffled_global = float(np.average(valid["shuffled_auc"], weights=valid["pair_mass"]))
    global_summary = {
        "real_auc": real_global,
        "shuffled_auc": shuffled_global,
        "auc_delta": real_global - shuffled_global,
        "estimable_family_well_strata": int(len(valid)),
        "total_family_well_strata": int(len(attribution)),
        "positive_auc_delta_strata": int((valid["auc_delta"] > 0).sum()),
        "positive_auc_delta_pair_mass_share": float(
            valid.loc[valid["auc_delta"] > 0, "pair_mass"].sum() / valid["pair_mass"].sum()
        ),
        "contribution_sum": float(valid["delta_contribution_to_stratified_auc"].sum()),
    }
    if not math.isclose(
        global_summary["auc_delta"],
        global_summary["contribution_sum"],
        abs_tol=1.0e-12,
        rel_tol=0.0,
    ):
        raise AssertionError("family-well AUC contributions do not sum to delta")
    return attribution, family_summary, well_summary, global_summary


# %% [markdown]
# ## 6. Risk saturation, plot, and output helpers


# %%
def saturation_row(
    frame: pd.DataFrame,
    variant: str,
    candidate: str,
    saturation_value: float,
    tolerance: float,
) -> dict[str, Any]:
    scores = frame[f"{variant}_risk"].to_numpy(float)
    bad = frame["bad_weight"].to_numpy(float)
    good = frame["good_weight"].to_numpy(float)
    total = bad + good
    finite = np.isfinite(scores)
    saturated = finite & np.isclose(scores, saturation_value, atol=tolerance, rtol=0.0)
    finite_weight = float(total[finite].sum())
    saturated_weight = float(total[saturated].sum())
    saturated_bad = float(bad[saturated].sum())
    saturated_good = float(good[saturated].sum())
    return {
        "variant": variant,
        "candidate": candidate,
        "segment_sample_count": int(len(frame)),
        "finite_risk_sample_count": int(finite.sum()),
        "risk_one_sample_count": int(saturated.sum()),
        "risk_one_sample_fraction": (
            float(saturated.sum() / finite.sum()) if finite.any() else math.nan
        ),
        "evaluated_row_weight": finite_weight,
        "risk_one_evaluated_row_weight": saturated_weight,
        "risk_one_evaluated_weight_fraction": (
            saturated_weight / finite_weight if finite_weight > 0 else math.nan
        ),
        "overall_bad_rate": (
            float(bad[finite].sum() / finite_weight) if finite_weight > 0 else math.nan
        ),
        "risk_one_bad_rate": (
            saturated_bad / (saturated_bad + saturated_good)
            if saturated_bad + saturated_good > 0
            else math.nan
        ),
        "q90_risk_threshold": weighted_quantile(scores, total, 0.90),
        "q90_is_saturated_at_one": bool(
            math.isclose(
                weighted_quantile(scores, total, 0.90),
                saturation_value,
                abs_tol=tolerance,
                rel_tol=0.0,
            )
        ),
    }


def build_risk_saturation(
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    saturation_value = float(get_nested(config, "audit.risk_saturation_value"))
    tolerance = float(get_nested(config, "audit.equality_tolerance", 1.0e-12))
    rows: list[dict[str, Any]] = []
    for variant in ["real", "shuffled"]:
        rows.append(saturation_row(pairs, variant, "__pooled__", saturation_value, tolerance))
        for candidate, frame in pairs.groupby("candidate", sort=True):
            rows.append(saturation_row(frame, variant, str(candidate), saturation_value, tolerance))
    return pd.DataFrame(rows).sort_values(["candidate", "variant"], kind="mergesort")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": int(path.stat().st_size),
        "raw_sha256": sha256_path(path),
    }
    if path.name.endswith(".gz"):
        record["decompressed_content_sha256"] = sha256_path(path, decompressed=True)
    return record


def write_plots(
    out_dir: Path,
    distance_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    saturation: pd.DataFrame,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    distance_path = out_dir / f"{EXPERIMENT_NAME}_distance_paired_auc.png"
    x = np.arange(len(distance_summary))
    plt.figure(figsize=(11, 5.5))
    plt.plot(x, distance_summary["pooled_real_auc"], marker="o", label="pooled real")
    plt.plot(
        x,
        distance_summary["pooled_shuffled_auc"],
        marker="o",
        label="pooled shuffled",
    )
    plt.plot(
        x,
        distance_summary["within_family_real_auc"],
        marker="s",
        label="within-family real",
    )
    plt.plot(
        x,
        distance_summary["within_family_shuffled_auc"],
        marker="s",
        label="within-family shuffled",
    )
    plt.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    plt.xticks(x, distance_summary["distance_bucket"], rotation=30, ha="right")
    plt.ylabel("Weighted AUC")
    plt.xlabel("Distance bucket (ft)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(distance_path, dpi=160)
    plt.close()
    paths["distance_paired_auc_plot"] = distance_path.name

    family_path = out_dir / f"{EXPERIMENT_NAME}_family_well_contribution.png"
    ordered = family_summary.sort_values("delta_contribution_to_global_stratified_auc")
    plt.figure(figsize=(9, 5))
    colors = np.where(
        ordered["delta_contribution_to_global_stratified_auc"] >= 0,
        "#4c78a8",
        "#e45756",
    )
    plt.barh(
        ordered["candidate"],
        ordered["delta_contribution_to_global_stratified_auc"],
        color=colors,
    )
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.xlabel("Contribution to family x well conditional AUC delta")
    plt.tight_layout()
    plt.savefig(family_path, dpi=160)
    plt.close()
    paths["family_well_contribution_plot"] = family_path.name

    saturation_path = out_dir / f"{EXPERIMENT_NAME}_risk_one_saturation.png"
    sat = saturation.loc[saturation["candidate"] != "__pooled__"].copy()
    pivot = sat.pivot(
        index="candidate",
        columns="variant",
        values="risk_one_evaluated_weight_fraction",
    ).sort_index()
    pivot.plot(kind="bar", figsize=(10, 5), color=["#4c78a8", "#f58518"])
    plt.ylabel("Evaluated weight fraction with risk = 1.0")
    plt.xlabel("Candidate family")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(saturation_path, dpi=160)
    plt.close()
    paths["risk_one_saturation_plot"] = saturation_path.name
    return paths


# %% [markdown]
# ## 7. Setup and fixed contract

# %%
started = time.time()
require_authorized_runtime()
assert get_nested(config, "experiment.name") == EXPERIMENT_NAME
assert get_nested(config, "experiment.route") == "pf_beam"
assert get_nested(config, "lineage.parent") == PARENT_EXPERIMENT
assert get_nested(config, "model.model_config_count") == 0
assert get_nested(config, "model.fold_count") == 0
assert get_nested(config, "model.booster_count") == 0
assert get_nested(config, "model.pf_beam_regeneration_count") == 0
assert get_nested(config, "model.corridor_regeneration_count") == 0
assert get_nested(config, "model.parent_control_retraining") is False
assert get_nested(config, "inference.enabled") is False
assert get_nested(config, "inference.create_submission") is False

if is_kaggle_runtime():
    output_root = KAGGLE_WORKING_ROOT
    out_dir = output_root / "artifacts"
else:
    output_root = ROOT / "experiments" / EXPERIMENT_NAME
    out_dir = output_root / "artifacts"
out_dir.mkdir(parents=True, exist_ok=True)

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Parent:", PARENT_EXPERIMENT)
print("Mode:", get_nested(config, "audit.mode"))
print("Runtime:", "kaggle" if is_kaggle_runtime() else "authorized-local-smoke")
print("Model config / folds / boosters: 0 / 0 / 0")
print("PF/Beam and corridor regeneration: 0 / 0")
print("Inference / submission: disabled / disabled")

# %% [markdown]
# ## 8. Load and freeze exp250 inputs

# %%
input_specs = get_nested(config, "data.inputs")
input_paths: dict[str, Path] = {}
input_manifest: dict[str, Any] = {}
for input_name, spec in input_specs.items():
    path, record = resolve_input(spec, input_name)
    input_paths[input_name] = path
    input_manifest[input_name] = record

pairs, pair_contract = load_candidate_pairs(input_paths["candidate_segment"], config)
group_metrics = pd.read_csv(input_paths["group_metrics"])
by_well = pd.read_csv(input_paths["by_well"])
parent_summary = json.loads(input_paths["parent_summary"].read_text())
validate_parent_summary(parent_summary)

if by_well["well"].astype(str).nunique() != int(get_nested(config, "data.expected_wells")):
    raise ValueError("by_well well count mismatch")
input_manifest["candidate_segment"]["schema_sha256"] = schema_sha256(pairs)
input_manifest["candidate_segment"]["paired_contract"] = pair_contract
input_manifest["group_metrics"]["schema_sha256"] = schema_sha256(group_metrics)
input_manifest["group_metrics"]["rows"] = int(len(group_metrics))
input_manifest["by_well"]["schema_sha256"] = schema_sha256(by_well)
input_manifest["by_well"]["rows"] = int(len(by_well))
input_manifest["parent_summary"]["parent_decision"] = parent_summary["decision"]
input_manifest["config"] = {
    "path": str(CONFIG_PATH),
    "bytes": int(CONFIG_PATH.stat().st_size),
    "raw_sha256": sha256_path(CONFIG_PATH),
}

print("Input manifest:")
display(pd.DataFrame(input_manifest).T)
print("Paired candidate preview:")
display(pairs.head(10))

# %% [markdown]
# ## 9. Run attribution readouts

# %%
distance_pairs = build_distance_pairs(group_metrics, config)
distance_summary = build_distance_conditional_summary(distance_pairs, config)
scope_summary = build_scope_summary(distance_pairs, config)
(
    family_well_attribution,
    family_summary,
    well_summary,
    family_well_global,
) = build_family_well_attribution(pairs, by_well, config)
risk_saturation = build_risk_saturation(pairs, config)

print("Distance paired AUC:")
display(distance_summary)
print("Near / far scope summary:")
display(scope_summary)
print("Family attribution summary:")
display(family_summary)
print("Risk=1 saturation:")
display(risk_saturation)

# %% [markdown]
# ## 10. Metrics, diagnostics, and generated artifacts

# %%
output_names = get_nested(config, "audit.outputs")
output_frames: list[tuple[pd.DataFrame, Path, bool]] = [
    (
        distance_pairs,
        out_dir / output_names["distance_paired_auc_filename"],
        False,
    ),
    (
        distance_summary,
        out_dir / output_names["distance_conditional_summary_filename"],
        False,
    ),
    (scope_summary, out_dir / output_names["scope_summary_filename"], False),
    (
        family_well_attribution,
        out_dir / output_names["family_well_attribution_filename"],
        True,
    ),
    (family_summary, out_dir / output_names["family_summary_filename"], False),
    (well_summary, out_dir / output_names["well_summary_filename"], False),
    (
        risk_saturation,
        out_dir / output_names["risk_saturation_filename"],
        False,
    ),
]
generated_paths: list[Path] = []
for frame, path, compressed in output_frames:
    if compressed:
        write_gzip_csv(path, frame)
    else:
        write_csv(path, frame)
    generated_paths.append(path)

input_manifest_path = out_dir / output_names["input_manifest_filename"]
write_json(input_manifest_path, input_manifest)
generated_paths.append(input_manifest_path)

plot_names = write_plots(out_dir, distance_summary, family_summary, risk_saturation)
generated_paths.extend(out_dir / name for name in plot_names.values())

near_row = scope_summary.loc[scope_summary["scope"] == "near_000_100"].iloc[0]
pooled_saturation = risk_saturation.loc[risk_saturation["candidate"] == "__pooled__"].set_index(
    "variant"
)
output_sha = {path.name: artifact_record(path) for path in generated_paths}
summary = {
    "experiment": EXPERIMENT_NAME,
    "parent": PARENT_EXPERIMENT,
    "route": "pf_beam",
    "status": "readout_complete",
    "decision": "diagnostic_only_no_exp250_route_or_use_change",
    "scope_constraints": {
        "stage1_replay": False,
        "new_corridor_or_candidate_generation": False,
        "model_training": False,
        "parameter_grid": False,
        "feature_generation": False,
        "raw_test_inference": False,
        "submission": False,
    },
    "input_manifest": input_manifest,
    "paired_contract": pair_contract,
    "primary_diagnostics": {
        "near_evaluated_row_weight": near_row["evaluated_row_weight"],
        "near_evaluated_weight_share": near_row["evaluated_weight_share"],
        "near_bad_rate": near_row["bad_rate"],
        "near_pooled_auc_weighted_bucket_mean_real": near_row[
            "pooled_auc_weighted_bucket_mean_real"
        ],
        "near_pooled_auc_weighted_bucket_mean_shuffled": near_row[
            "pooled_auc_weighted_bucket_mean_shuffled"
        ],
        "near_pooled_auc_weighted_bucket_mean_delta": near_row[
            "pooled_auc_weighted_bucket_mean_delta"
        ],
        "near_distance_family_conditional_real_auc": near_row[
            "distance_family_conditional_real_auc"
        ],
        "near_distance_family_conditional_shuffled_auc": near_row[
            "distance_family_conditional_shuffled_auc"
        ],
        "near_distance_family_conditional_auc_delta": near_row[
            "distance_family_conditional_auc_delta"
        ],
        "near_pooled_minus_conditional_real_auc": near_row["pooled_minus_conditional_real_auc"],
        "near_estimable_candidate_family_bucket_strata": near_row[
            "estimable_candidate_family_bucket_strata"
        ],
        "near_positive_auc_delta_strata": near_row["positive_auc_delta_strata"],
        "near_unique_estimable_candidate_families": near_row["unique_estimable_candidate_families"],
        "family_well_conditional": family_well_global,
        "real_pooled_risk_one_evaluated_weight_fraction": pooled_saturation.loc[
            "real", "risk_one_evaluated_weight_fraction"
        ],
        "shuffled_pooled_risk_one_evaluated_weight_fraction": pooled_saturation.loc[
            "shuffled", "risk_one_evaluated_weight_fraction"
        ],
        "real_pooled_q90_risk_threshold": pooled_saturation.loc["real", "q90_risk_threshold"],
        "shuffled_pooled_q90_risk_threshold": pooled_saturation.loc[
            "shuffled", "q90_risk_threshold"
        ],
    },
    "interpretation_contract": {
        "pooled_auc_is_not_family_or_well_conditioned": True,
        "weighted_bucket_mean_is_descriptive_not_a_recomputed_pooled_auc": True,
        "segment_overlap_weights_are_not_unique_row_counts": True,
        "readout_cannot_authorize_hard_use_feature_use_or_submission": True,
    },
    "rows": {
        "candidate_pairs": int(len(pairs)),
        "distance_pair_rows": int(len(distance_pairs)),
        "family_well_rows": int(len(family_well_attribution)),
        "family_rows": int(len(family_summary)),
        "well_rows": int(len(well_summary)),
    },
    "plots": plot_names,
    "output_sha": output_sha,
    "runtime": {
        "platform": "kaggle" if is_kaggle_runtime() else "authorized_local_smoke",
        "cpu_only": True,
        "internet_disabled": True,
        "single_process": True,
        "elapsed_seconds": time.time() - started,
    },
    "reproducibility": {
        "new_rng": False,
        "fixed_input_diagnostic_deterministic": True,
        "prediction_or_submission_anchor": False,
        "gzip_mtime": 0,
        "upstream_stochastic_provenance": ["exp250", "exp072 PF/Beam cache"],
    },
}
summary_path = out_dir / output_names["summary_filename"]
write_json(summary_path, summary)
summary_sha = sha256_path(summary_path)

metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "readout_complete",
    "route": "pf_beam",
    "metric": "diagnostic_paired_auc_attribution",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "decision": summary["decision"],
    "primary_diagnostics": summary["primary_diagnostics"],
    "summary_path": str(summary_path),
    "summary_sha256": summary_sha,
    "runtime_seconds": summary["runtime"]["elapsed_seconds"],
}
metrics_path = output_root / "metrics.json"
write_json(metrics_path, metrics)

print("Final summary:")
display(pd.DataFrame([clean_json(summary["primary_diagnostics"])]).T)
print("Summary path:", summary_path)
print("Summary SHA256:", summary_sha)
print("Metrics path:", metrics_path)
