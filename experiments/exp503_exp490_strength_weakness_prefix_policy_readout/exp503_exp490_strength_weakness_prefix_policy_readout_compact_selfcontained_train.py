# %% [markdown]
# # exp503 exp490 strength / weakness and prefix policy readout
#
# This notebook uses saved exp490 OOF predictions and truth to explain where the
# mean-reverting HMM succeeds or fails. It separately evaluates deployable
# outer-fold-safe fade/alpha policies and explicitly labels early-truth routing
# as an optimistic, nondeployable diagnostic. It never reruns HMM/PF/Beam.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, paths, SHA, and metric helpers
# 3. Well-level error-shape and depth helpers
# 4. Feature-association and archetype helpers
# 5. Public-notebook fade and outer-fold policy helpers
# 6. Plot and artifact helpers
# 7. Setup and SHA-pinned input checks
# 8. Truth-aware strength / weakness readout
# 9. Fold-safe policy and optimistic prefix-transfer audits
# 10. Metrics, gates, and generated artifacts

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
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Static/contract-only local environments may omit plotting.
    plt = None

EXPERIMENT_NAME = "exp503_exp490_strength_weakness_prefix_policy_readout"
PARENT_EXPERIMENT = "exp490_geometry_centered_mean_reverting_offset_hmm"
FEATURE_EXPERIMENT = "exp499_exp490_cross_fitted_well_application_selector"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

PREDICTION_COLUMNS = (
    "well",
    "row_idx",
    "suffix_offset",
    "md_since",
    "fold",
    "true_tvt_readout_only",
    "geometry_mean_reverting_hmm",
    "geometry_mean_reverting_delta_mean",
    "geometry_mean_reverting_hmm_std",
    "exp357_parent_prediction",
    "exp226_pred",
    "tvt_geop",
)


def get_nested(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def fade_profiles(config: Mapping[str, Any]) -> list[dict[str, float | str]]:
    profiles: list[dict[str, float | str]] = [
        {"profile": "parent", "alpha": 0.0, "tau_ft": 0.0}
    ]
    for alpha in get_nested(config, "policies.fade_grid.alphas"):
        for tau in get_nested(config, "policies.fade_grid.taus_ft"):
            profiles.append(
                {
                    "profile": f"alpha{float(alpha):g}_tau{float(tau):g}",
                    "alpha": float(alpha),
                    "tau_ft": float(tau),
                }
            )
    return profiles


def validate_immutable_config(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp503 route must remain ensemble")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp503 parent changed")
    if get_nested(config, "lineage.feature_source") != FEATURE_EXPERIMENT:
        raise ValueError("exp503 feature source changed")
    if tuple(get_nested(config, "data.inputs.predictions.columns")) != PREDICTION_COLUMNS:
        raise ValueError("prediction column contract changed")
    if bool(get_nested(config, "implementation.inference_enabled")):
        raise ValueError("inference is out of scope")
    if bool(get_nested(config, "implementation.submission_enabled")):
        raise ValueError("submission is out of scope")
    execution = get_nested(config, "execution_contract")
    forbidden = (
        "lightgbm_configs",
        "lightgbm_boosters",
        "parent_control_retraining",
        "new_hmm_well_runs",
        "new_candidate_predictions",
        "pf_runs",
        "beam_runs",
        "gpu_runs",
    )
    if any(int(execution[key]) != 0 for key in forbidden):
        raise ValueError("execution contract contains forbidden work")
    profiles = fade_profiles(config)
    if len(profiles) != int(get_nested(config, "policies.fade_grid.expected_profiles")):
        raise ValueError("fade profile count changed")
    if len(profiles) != int(execution["fixed_fade_profiles"]):
        raise ValueError("profile count and execution contract disagree")
    if int(execution["maximum_cpu_tree_fits"]) != 10:
        raise ValueError("tree fit budget changed")


# %% [markdown]
# ## 2. Runtime, paths, SHA, and metric helpers

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
    raise FileNotFoundError("exp503 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or config_path()).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    validate_immutable_config(config)
    return config


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


def rmse_from_sse(sse: float | np.ndarray, rows: int | np.ndarray) -> float | np.ndarray:
    return np.sqrt(np.asarray(sse, dtype=float) / np.maximum(np.asarray(rows, dtype=float), 1.0))


def bincount(codes: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(codes, weights=np.asarray(weights, dtype=float), minlength=size)


def make_row_arrays(predictions: pd.DataFrame) -> dict[str, Any]:
    wells = np.asarray(sorted(predictions["well"].astype(str).unique()))
    categories = pd.Categorical(predictions["well"].astype(str), categories=wells)
    well_codes = categories.codes.astype(np.int32)
    if np.any(well_codes < 0):
        raise ValueError("failed to encode well IDs")
    truth = predictions["true_tvt_readout_only"].to_numpy(dtype=float)
    parent = predictions["exp357_parent_prediction"].to_numpy(dtype=float)
    candidate = predictions["geometry_mean_reverting_hmm"].to_numpy(dtype=float)
    parent_error = parent - truth
    delta = candidate - parent
    return {
        "wells": wells,
        "well_codes": well_codes,
        "n_wells": len(wells),
        "fold": predictions["fold"].to_numpy(dtype=np.int8),
        "row_idx": predictions["row_idx"].to_numpy(dtype=np.int32),
        "suffix_offset": predictions["suffix_offset"].to_numpy(dtype=float),
        "md_since": predictions["md_since"].to_numpy(dtype=float),
        "truth": truth,
        "parent": parent,
        "candidate": candidate,
        "exp226": predictions["exp226_pred"].to_numpy(dtype=float),
        "geometry": predictions["tvt_geop"].to_numpy(dtype=float),
        "posterior_std": predictions["geometry_mean_reverting_hmm_std"].to_numpy(dtype=float),
        "delta_mean": predictions["geometry_mean_reverting_delta_mean"].to_numpy(dtype=float),
        "parent_error": parent_error,
        "candidate_error": parent_error + delta,
        "delta": delta,
    }


# %% [markdown]
# ## 3. Well-level error-shape and depth helpers

# %%
def group_slope(
    codes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    size: int,
) -> np.ndarray:
    n = np.bincount(codes, minlength=size).astype(float)
    sx = bincount(codes, x, size)
    sy = bincount(codes, y, size)
    sxx = bincount(codes, x * x, size)
    sxy = bincount(codes, x * y, size)
    denominator = n * sxx - sx * sx
    numerator = n * sxy - sx * sy
    return np.divide(numerator, denominator, out=np.zeros(size), where=np.abs(denominator) > 1e-12)


def longest_true_run(values: np.ndarray) -> int:
    if not np.any(values):
        return 0
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return int(np.max(changes[1::2] - changes[::2]))


def build_well_readout(arrays: Mapping[str, Any], features: pd.DataFrame, window: int) -> pd.DataFrame:
    codes = arrays["well_codes"]
    size = int(arrays["n_wells"])
    rows = np.bincount(codes, minlength=size).astype(int)
    parent_error = arrays["parent_error"]
    candidate_error = arrays["candidate_error"]
    exp226_error = arrays["exp226"] - arrays["truth"]
    geometry_error = arrays["geometry"] - arrays["truth"]
    delta = arrays["delta"]
    gain = parent_error**2 - candidate_error**2
    parent_sse = bincount(codes, parent_error**2, size)
    candidate_sse = bincount(codes, candidate_error**2, size)
    fold_min = np.full(size, 99, dtype=int)
    fold_max = np.full(size, -1, dtype=int)
    np.minimum.at(fold_min, codes, arrays["fold"])
    np.maximum.at(fold_max, codes, arrays["fold"])
    if not np.array_equal(fold_min, fold_max):
        raise ValueError("a well appears in multiple folds")

    required_move = arrays["truth"] - arrays["parent"]
    correction_dot = bincount(codes, delta * required_move, size)
    correction_norm = bincount(codes, delta**2, size)
    optimal_alpha = np.divide(
        correction_dot,
        correction_norm,
        out=np.zeros(size),
        where=correction_norm > 1e-12,
    )

    result = pd.DataFrame(
        {
            "well": arrays["wells"],
            "fold": fold_min,
            "rows": rows,
            "candidate_rmse_ft": rmse_from_sse(candidate_sse, rows),
            "parent_rmse_ft": rmse_from_sse(parent_sse, rows),
            "candidate_minus_parent_rmse_ft": rmse_from_sse(candidate_sse, rows)
            - rmse_from_sse(parent_sse, rows),
            "actual_benefit_mse_ft2": (parent_sse - candidate_sse) / rows,
            "beneficial_well": candidate_sse < parent_sse,
            "candidate_mae_ft": bincount(codes, np.abs(candidate_error), size) / rows,
            "parent_mae_ft": bincount(codes, np.abs(parent_error), size) / rows,
            "candidate_bias_ft": bincount(codes, candidate_error, size) / rows,
            "parent_bias_ft": bincount(codes, parent_error, size) / rows,
            "candidate_error_std_ft": np.sqrt(
                np.maximum(
                    candidate_sse / rows
                    - (bincount(codes, candidate_error, size) / rows) ** 2,
                    0.0,
                )
            ),
            "parent_error_std_ft": np.sqrt(
                np.maximum(
                    parent_sse / rows - (bincount(codes, parent_error, size) / rows) ** 2,
                    0.0,
                )
            ),
            "candidate_error_drift_ft_per_1k_rows": group_slope(
                codes, arrays["suffix_offset"], candidate_error, size
            )
            * 1000.0,
            "parent_error_drift_ft_per_1k_rows": group_slope(
                codes, arrays["suffix_offset"], parent_error, size
            )
            * 1000.0,
            "correction_required_alignment": np.divide(
                correction_dot,
                np.sqrt(
                    np.maximum(correction_norm, 1e-12)
                    * np.maximum(bincount(codes, required_move**2, size), 1e-12)
                ),
            ),
            "truth_optimal_direct_alpha": optimal_alpha,
            "truth_optimal_direct_alpha_clipped_0_1": np.clip(optimal_alpha, 0.0, 1.0),
            "exp226_rmse_ft": rmse_from_sse(bincount(codes, exp226_error**2, size), rows),
            "geometry_rmse_ft": rmse_from_sse(bincount(codes, geometry_error**2, size), rows),
        }
    )

    for horizon in (128, 256, 512, 1024):
        early = arrays["suffix_offset"] < horizon
        late = ~early
        early_rows = np.bincount(codes[early], minlength=size)
        late_rows = np.bincount(codes[late], minlength=size)
        result[f"first{horizon}_benefit_mse_ft2"] = np.divide(
            bincount(codes[early], gain[early], size),
            early_rows,
            out=np.full(size, np.nan),
            where=early_rows > 0,
        )
        result[f"after{horizon}_benefit_mse_ft2"] = np.divide(
            bincount(codes[late], gain[late], size),
            late_rows,
            out=np.full(size, np.nan),
            where=late_rows > 0,
        )

    longest_runs = np.zeros(size, dtype=int)
    worst_windows = np.full(size, np.nan)
    cumulative_positive_fraction = np.zeros(size, dtype=float)
    final_recovery_row = np.full(size, -1, dtype=int)
    order = np.lexsort((arrays["suffix_offset"], codes))
    ordered_codes = codes[order]
    ordered_gain = gain[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_codes)) + 1, len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        code = int(ordered_codes[start])
        values = ordered_gain[start:stop]
        longest_runs[code] = longest_true_run(values < 0)
        cumulative = np.cumsum(values)
        cumulative_positive_fraction[code] = float(np.mean(cumulative > 0))
        nonpositive = np.flatnonzero(cumulative <= 0)
        final_recovery_row[code] = int(nonpositive[-1] + 1) if len(nonpositive) else 0
        if len(values) >= window:
            rolling = (np.cumsum(np.r_[0.0, values])[window:] - np.cumsum(np.r_[0.0, values])[:-window]) / window
            worst_windows[code] = float(np.min(rolling))
        else:
            worst_windows[code] = float(np.mean(values))
    result["longest_consecutive_harm_rows"] = longest_runs
    result[f"worst_{window}_row_benefit_mse_ft2"] = worst_windows
    result["cumulative_gain_positive_fraction"] = cumulative_positive_fraction
    result["last_nonpositive_cumulative_gain_row"] = final_recovery_row

    feature_rows = features.set_index("well").loc[result["well"], "rows"].to_numpy(dtype=int)
    if not np.array_equal(feature_rows, result["rows"].to_numpy(dtype=int)):
        raise ValueError("computed and frozen-feature well row counts disagree")
    join_features = features.drop(columns="rows")
    merged = result.merge(join_features, on="well", how="left", validate="one_to_one")
    joined_feature_columns = [column for column in join_features.columns if column != "well"]
    if merged[joined_feature_columns].isna().any().any():
        raise ValueError("target-free feature join produced missing values")
    return merged


def slice_metric_rows(
    arrays: Mapping[str, Any],
    bucket_codes: np.ndarray,
    bucket_labels: Sequence[str],
    axis: str,
    include_folds: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [("pooled", np.ones(len(bucket_codes), dtype=bool))]
    if include_folds:
        scopes.extend(
            (f"fold_{fold}", arrays["fold"] == fold) for fold in sorted(np.unique(arrays["fold"]))
        )
    for scope, scope_mask in scopes:
        for code, label in enumerate(bucket_labels):
            mask = scope_mask & (bucket_codes == code)
            rows = int(np.sum(mask))
            if rows == 0:
                continue
            candidate_sse = float(np.sum(arrays["candidate_error"][mask] ** 2))
            parent_sse = float(np.sum(arrays["parent_error"][mask] ** 2))
            output.append(
                {
                    "axis": axis,
                    "bucket": label,
                    "scope": scope,
                    "rows": rows,
                    "wells": int(np.unique(arrays["well_codes"][mask]).size),
                    "candidate_rmse_ft": float(rmse_from_sse(candidate_sse, rows)),
                    "parent_rmse_ft": float(rmse_from_sse(parent_sse, rows)),
                    "candidate_minus_parent_rmse_ft": float(rmse_from_sse(candidate_sse, rows))
                    - float(rmse_from_sse(parent_sse, rows)),
                    "benefit_mse_ft2": (parent_sse - candidate_sse) / rows,
                }
            )
    return output


def build_depth_metrics(arrays: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    absolute_edges = np.asarray(get_nested(config, "readout.absolute_depth_edges"), dtype=float)
    absolute_labels = [
        f"[{absolute_edges[index]:g},{absolute_edges[index + 1]:g})"
        for index in range(len(absolute_edges) - 1)
    ]
    absolute_codes = np.clip(
        np.digitize(arrays["suffix_offset"], absolute_edges[1:-1], right=False),
        0,
        len(absolute_labels) - 1,
    )
    rows.extend(slice_metric_rows(arrays, absolute_codes, absolute_labels, "suffix_absolute_rows", True))

    well_rows = np.bincount(arrays["well_codes"], minlength=arrays["n_wells"])
    relative = arrays["suffix_offset"] / np.maximum(well_rows[arrays["well_codes"]] - 1, 1)
    relative_edges = np.asarray(get_nested(config, "readout.relative_depth_edges"), dtype=float)
    relative_labels = [
        f"q{index + 1}_{relative_edges[index]:.2f}_{relative_edges[index + 1]:.2f}"
        for index in range(len(relative_edges) - 1)
    ]
    relative_codes = np.clip(
        np.digitize(relative, relative_edges[1:-1], right=False),
        0,
        len(relative_labels) - 1,
    )
    rows.extend(slice_metric_rows(arrays, relative_codes, relative_labels, "suffix_relative", True))

    for axis, values in (
        ("posterior_std_quantile", arrays["posterior_std"]),
        ("correction_abs_quantile", np.abs(arrays["delta"])),
        ("parent_abs_error_truth_quantile", np.abs(arrays["parent_error"])),
    ):
        edges = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0.0, 1.0, 6)))
        codes = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
        labels = [f"q{index + 1}" for index in range(len(edges) - 1)]
        rows.extend(slice_metric_rows(arrays, codes, labels, axis, False))
    return pd.DataFrame(rows)


# %% [markdown]
# ## 4. Feature-association and archetype helpers

# %%
def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if np.sum(mask) < 3 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def build_feature_readouts(
    well_readout: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    association_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    target = well_readout["actual_benefit_mse_ft2"].to_numpy(dtype=float)
    beneficial = well_readout["beneficial_well"].to_numpy(dtype=bool)
    for feature in feature_columns:
        values = well_readout[feature].to_numpy(dtype=float)
        raw_auc = safe_auc(beneficial, values)
        fold_spearman = []
        fold_auc = []
        for fold in sorted(well_readout["fold"].unique()):
            mask = well_readout["fold"].to_numpy() == fold
            fold_spearman.append(safe_spearman(values[mask], target[mask]))
            fold_auc.append(safe_auc(beneficial[mask], values[mask]))
        association_rows.append(
            {
                "feature": feature,
                "spearman_with_benefit_mse": safe_spearman(values, target),
                "raw_beneficial_auc": raw_auc,
                "orientation_free_auc": max(raw_auc, 1.0 - raw_auc) if np.isfinite(raw_auc) else np.nan,
                "fold_spearman_min": float(np.nanmin(fold_spearman)),
                "fold_spearman_max": float(np.nanmax(fold_spearman)),
                "fold_auc_min": float(np.nanmin(fold_auc)),
                "fold_auc_max": float(np.nanmax(fold_auc)),
                "positive_spearman_folds": int(np.sum(np.asarray(fold_spearman) > 0)),
            }
        )
        try:
            buckets = pd.qcut(values, q=5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for bucket in sorted(pd.Series(buckets).dropna().unique()):
            mask = np.asarray(buckets == bucket)
            rows = int(well_readout.loc[mask, "rows"].sum())
            candidate_sse = float(
                np.sum(well_readout.loc[mask, "candidate_rmse_ft"] ** 2 * well_readout.loc[mask, "rows"])
            )
            parent_sse = float(
                np.sum(well_readout.loc[mask, "parent_rmse_ft"] ** 2 * well_readout.loc[mask, "rows"])
            )
            quantile_rows.append(
                {
                    "feature": feature,
                    "quantile": int(bucket) + 1,
                    "wells": int(np.sum(mask)),
                    "rows": rows,
                    "feature_min": float(np.nanmin(values[mask])),
                    "feature_median": float(np.nanmedian(values[mask])),
                    "feature_max": float(np.nanmax(values[mask])),
                    "candidate_rmse_ft": float(rmse_from_sse(candidate_sse, rows)),
                    "parent_rmse_ft": float(rmse_from_sse(parent_sse, rows)),
                    "candidate_minus_parent_rmse_ft": float(rmse_from_sse(candidate_sse, rows))
                    - float(rmse_from_sse(parent_sse, rows)),
                    "beneficial_well_fraction": float(np.mean(beneficial[mask])),
                }
            )
    associations = pd.DataFrame(association_rows).sort_values(
        "orientation_free_auc", ascending=False
    )
    return associations, pd.DataFrame(quantile_rows)


def build_archetypes(
    arrays: Mapping[str, Any],
    well_readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = int(arrays["n_wells"])
    well_rows = np.bincount(arrays["well_codes"], minlength=size)
    relative = arrays["suffix_offset"] / np.maximum(well_rows[arrays["well_codes"]] - 1, 1)
    edges = np.asarray(get_nested(config, "readout.relative_depth_edges"), dtype=float)
    codes = np.clip(np.digitize(relative, edges[1:-1]), 0, len(edges) - 2)
    gain = arrays["parent_error"] ** 2 - arrays["candidate_error"] ** 2
    matrix = np.full((size, len(edges) - 1), np.nan)
    for bucket in range(len(edges) - 1):
        mask = codes == bucket
        counts = np.bincount(arrays["well_codes"][mask], minlength=size)
        totals = bincount(arrays["well_codes"][mask], gain[mask], size)
        matrix[:, bucket] = np.divide(
            totals,
            counts,
            out=np.full(size, np.nan),
            where=counts > 0,
        )
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanquantile(matrix, 0.25, axis=0)
    q75 = np.nanquantile(matrix, 0.75, axis=0)
    scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    normalized = (np.where(np.isfinite(matrix), matrix, median) - median) / scale
    kmeans = KMeans(
        n_clusters=int(get_nested(config, "readout.archetypes.n_clusters")),
        n_init=int(get_nested(config, "readout.archetypes.n_init")),
        random_state=int(get_nested(config, "readout.archetypes.random_state")),
    )
    raw_labels = kmeans.fit_predict(normalized)
    label_strength = {
        label: float(np.average(
            well_readout.loc[raw_labels == label, "actual_benefit_mse_ft2"],
            weights=well_readout.loc[raw_labels == label, "rows"],
        ))
        for label in np.unique(raw_labels)
    }
    ordered = sorted(label_strength, key=label_strength.get, reverse=True)
    relabel = {old: rank + 1 for rank, old in enumerate(ordered)}
    labels = np.asarray([relabel[value] for value in raw_labels], dtype=int)
    assignment = pd.DataFrame({"well": arrays["wells"], "archetype": labels})
    for bucket in range(matrix.shape[1]):
        assignment[f"relative_q{bucket + 1}_benefit_mse_ft2"] = matrix[:, bucket]
    enriched = well_readout.merge(assignment, on="well", validate="one_to_one")
    summary_rows: list[dict[str, Any]] = []
    for label, group in enriched.groupby("archetype", sort=True):
        rows = int(group["rows"].sum())
        candidate_sse = float(np.sum(group["candidate_rmse_ft"] ** 2 * group["rows"]))
        parent_sse = float(np.sum(group["parent_rmse_ft"] ** 2 * group["rows"]))
        row: dict[str, Any] = {
            "archetype": int(label),
            "wells": int(len(group)),
            "rows": rows,
            "beneficial_well_fraction": float(group["beneficial_well"].mean()),
            "candidate_rmse_ft": float(rmse_from_sse(candidate_sse, rows)),
            "parent_rmse_ft": float(rmse_from_sse(parent_sse, rows)),
            "candidate_minus_parent_rmse_ft": float(rmse_from_sse(candidate_sse, rows))
            - float(rmse_from_sse(parent_sse, rows)),
            "median_candidate_bias_ft": float(group["candidate_bias_ft"].median()),
            "median_candidate_drift_ft_per_1k_rows": float(
                group["candidate_error_drift_ft_per_1k_rows"].median()
            ),
        }
        for bucket in range(matrix.shape[1]):
            row[f"relative_q{bucket + 1}_benefit_mse_ft2"] = float(
                group[f"relative_q{bucket + 1}_benefit_mse_ft2"].median()
            )
        summary_rows.append(row)
    return enriched, pd.DataFrame(summary_rows)


# %% [markdown]
# ## 5. Public-notebook fade and outer-fold policy helpers

# %%
def fade_ramp(md_since: np.ndarray, tau_ft: float) -> np.ndarray:
    if tau_ft <= 0:
        return np.ones(len(md_since), dtype=float)
    return 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau_ft)


def profile_weight(md_since: np.ndarray, profile: Mapping[str, Any]) -> np.ndarray:
    return float(profile["alpha"]) * fade_ramp(md_since, float(profile["tau_ft"]))


def profile_metric_tables(
    arrays: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    folds = np.asarray(sorted(np.unique(arrays["fold"])), dtype=int)
    fold_sse = np.zeros((len(profiles), len(folds)), dtype=float)
    pooled_rows = len(arrays["fold"])
    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        weights = profile_weight(arrays["md_since"], profile)
        error = arrays["parent_error"] + weights * arrays["delta"]
        for fold_index, fold in enumerate(folds):
            mask = arrays["fold"] == fold
            fold_sse[profile_index, fold_index] = float(np.sum(error[mask] ** 2))
            rows.append(
                {
                    **profile,
                    "scope": f"fold_{fold}",
                    "rows": int(np.sum(mask)),
                    "rmse_ft": float(rmse_from_sse(fold_sse[profile_index, fold_index], np.sum(mask))),
                }
            )
        rows.append(
            {
                **profile,
                "scope": "pooled",
                "rows": pooled_rows,
                "rmse_ft": float(rmse_from_sse(np.sum(fold_sse[profile_index]), pooled_rows)),
            }
        )
    return pd.DataFrame(rows), fold_sse, folds


def aggregate_policy(
    arrays: Mapping[str, Any],
    weights: np.ndarray,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    error = arrays["parent_error"] + weights * arrays["delta"]
    size = int(arrays["n_wells"])
    well_rows = np.bincount(arrays["well_codes"], minlength=size)
    policy_sse = bincount(arrays["well_codes"], error**2, size)
    candidate_sse = bincount(arrays["well_codes"], arrays["candidate_error"] ** 2, size)
    parent_sse = bincount(arrays["well_codes"], arrays["parent_error"] ** 2, size)
    well_alpha = bincount(arrays["well_codes"], weights, size) / well_rows
    well = pd.DataFrame(
        {
            "well": arrays["wells"],
            "policy": policy,
            "rows": well_rows,
            "mean_effective_alpha": well_alpha,
            "policy_rmse_ft": rmse_from_sse(policy_sse, well_rows),
            "exp490_rmse_ft": rmse_from_sse(candidate_sse, well_rows),
            "parent_rmse_ft": rmse_from_sse(parent_sse, well_rows),
        }
    )
    well["policy_minus_exp490_rmse_ft"] = well["policy_rmse_ft"] - well["exp490_rmse_ft"]
    well["policy_minus_parent_rmse_ft"] = well["policy_rmse_ft"] - well["parent_rmse_ft"]
    fold_rows: list[dict[str, Any]] = []
    for scope_fold in [None, *sorted(np.unique(arrays["fold"]))]:
        mask = np.ones(len(error), dtype=bool) if scope_fold is None else arrays["fold"] == scope_fold
        rows = int(np.sum(mask))
        policy_rmse = float(rmse_from_sse(np.sum(error[mask] ** 2), rows))
        candidate_rmse = float(rmse_from_sse(np.sum(arrays["candidate_error"][mask] ** 2), rows))
        parent_rmse = float(rmse_from_sse(np.sum(arrays["parent_error"][mask] ** 2), rows))
        fold_rows.append(
            {
                "policy": policy,
                "scope": "pooled" if scope_fold is None else f"fold_{scope_fold}",
                "rows": rows,
                "rmse_ft": policy_rmse,
                "exp490_rmse_ft": candidate_rmse,
                "parent_rmse_ft": parent_rmse,
                "gain_vs_exp490_ft": candidate_rmse - policy_rmse,
                "gain_vs_parent_ft": parent_rmse - policy_rmse,
            }
        )
    return well, pd.DataFrame(fold_rows)


def outer_global_fade_policy(
    arrays: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    fold_sse: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(len(arrays["fold"]), dtype=float)
    selections: list[dict[str, Any]] = []
    total_sse = np.sum(fold_sse, axis=1)
    fold_rows = np.asarray([np.sum(arrays["fold"] == fold) for fold in folds], dtype=int)
    for fold_index, fold in enumerate(folds):
        train_rows = int(np.sum(fold_rows) - fold_rows[fold_index])
        train_rmse = rmse_from_sse(total_sse - fold_sse[:, fold_index], train_rows)
        selected_index = int(np.argmin(train_rmse))
        selected = profiles[selected_index]
        mask = arrays["fold"] == fold
        weights[mask] = profile_weight(arrays["md_since"][mask], selected)
        selections.append(
            {
                "outer_fold": int(fold),
                "selected_profile": selected["profile"],
                "alpha": float(selected["alpha"]),
                "tau_ft": float(selected["tau_ft"]),
                "outer_train_rmse_ft": float(train_rmse[selected_index]),
            }
        )
    return weights, selections


def tree_transform_fit(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = train[list(feature_columns)].to_numpy(dtype=float)
    valid_x = valid[list(feature_columns)].to_numpy(dtype=float)
    medians = np.nanmedian(train_x, axis=0)
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    valid_x = np.where(np.isfinite(valid_x), valid_x, medians)
    train_x = np.log1p(np.maximum(train_x, 0.0))
    valid_x = np.log1p(np.maximum(valid_x, 0.0))
    return train_x, valid_x, medians


def outer_alpha_tree_policy(
    arrays: Mapping[str, Any],
    well_readout: pd.DataFrame,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    policy_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tau = float(get_nested(config, "policies.alpha_tree.fixed_tau_ft"))
    ramp = fade_ramp(arrays["md_since"], tau)
    x = ramp * arrays["delta"]
    size = int(arrays["n_wells"])
    numerator = -bincount(arrays["well_codes"], arrays["parent_error"] * x, size)
    denominator = bincount(arrays["well_codes"], x**2, size)
    optimal_alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros(size),
        where=denominator > 1e-12,
    )
    optimal_alpha = np.clip(optimal_alpha, 0.0, 1.0)
    lookup = well_readout.set_index("well").loc[arrays["wells"]].reset_index()
    lookup["truth_optimal_faded_alpha"] = optimal_alpha
    predicted_alpha = np.zeros(size, dtype=float)
    manifest: list[dict[str, Any]] = []
    for fold in sorted(lookup["fold"].unique()):
        train_mask = lookup["fold"].to_numpy() != fold
        valid_mask = ~train_mask
        train_x, valid_x, medians = tree_transform_fit(
            lookup.loc[train_mask], lookup.loc[valid_mask], feature_columns
        )
        model = DecisionTreeRegressor(
            max_depth=int(get_nested(config, "policies.alpha_tree.max_depth")),
            min_samples_leaf=int(get_nested(config, "policies.alpha_tree.min_samples_leaf")),
            splitter=str(get_nested(config, "policies.alpha_tree.splitter")),
            random_state=int(get_nested(config, "policies.alpha_tree.random_state")),
        )
        model.fit(
            train_x,
            lookup.loc[train_mask, "truth_optimal_faded_alpha"].to_numpy(dtype=float),
            sample_weight=lookup.loc[train_mask, "rows"].to_numpy(dtype=float),
        )
        predicted_alpha[valid_mask] = np.clip(model.predict(valid_x), 0.0, 1.0)
        manifest.append(
            {
                "policy": policy_name,
                "outer_fold": int(fold),
                "feature_columns": list(feature_columns),
                "feature_medians": medians.tolist(),
                "tree_text": export_text(model, feature_names=list(feature_columns)),
                "predicted_alpha_min": float(np.min(predicted_alpha[valid_mask])),
                "predicted_alpha_max": float(np.max(predicted_alpha[valid_mask])),
                "predicted_alpha_mean": float(np.mean(predicted_alpha[valid_mask])),
            }
        )
    row_weights = predicted_alpha[arrays["well_codes"]] * ramp
    return row_weights, manifest


def early_truth_transfer_audit(
    arrays: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
) -> pd.DataFrame:
    size = int(arrays["n_wells"])
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        early = arrays["suffix_offset"] < int(horizon)
        late = ~early
        early_counts = np.bincount(arrays["well_codes"][early], minlength=size)
        late_counts = np.bincount(arrays["well_codes"][late], minlength=size)
        eligible = (early_counts > 0) & (late_counts > 0)
        early_sse = np.full((len(profiles), size), np.inf)
        late_sse = np.full((len(profiles), size), np.inf)
        for profile_index, profile in enumerate(profiles):
            weight = profile_weight(arrays["md_since"], profile)
            error = arrays["parent_error"] + weight * arrays["delta"]
            early_sse[profile_index] = bincount(
                arrays["well_codes"][early], error[early] ** 2, size
            )
            late_sse[profile_index] = bincount(
                arrays["well_codes"][late], error[late] ** 2, size
            )
        choice = np.argmin(early_sse, axis=0)
        selected_late_sse = late_sse[choice, np.arange(size)]
        oracle_late_sse = np.min(late_sse, axis=0)
        candidate_index = next(
            index
            for index, profile in enumerate(profiles)
            if float(profile["alpha"]) == 1.0 and float(profile["tau_ft"]) == 0.0
        )
        parent_index = 0
        total_late_rows = int(np.sum(late_counts[eligible]))
        selected_rmse = float(
            rmse_from_sse(np.sum(selected_late_sse[eligible]), total_late_rows)
        )
        candidate_rmse = float(
            rmse_from_sse(np.sum(late_sse[candidate_index, eligible]), total_late_rows)
        )
        parent_rmse = float(
            rmse_from_sse(np.sum(late_sse[parent_index, eligible]), total_late_rows)
        )
        oracle_rmse = float(
            rmse_from_sse(np.sum(oracle_late_sse[eligible]), total_late_rows)
        )
        early_gain = (
            early_sse[parent_index, eligible] - early_sse[candidate_index, eligible]
        ) / early_counts[eligible]
        late_gain = (
            late_sse[parent_index, eligible] - late_sse[candidate_index, eligible]
        ) / late_counts[eligible]
        chosen_names = np.asarray([profiles[index]["profile"] for index in choice[eligible]])
        rows.append(
            {
                "early_truth_rows": int(horizon),
                "eligible_wells": int(np.sum(eligible)),
                "evaluated_late_rows": total_late_rows,
                "early_late_direct_benefit_spearman": safe_spearman(early_gain, late_gain),
                "early_late_direct_benefit_sign_accuracy": float(np.mean((early_gain > 0) == (late_gain > 0))),
                "early_selected_profile_late_rmse_ft": selected_rmse,
                "always_exp490_late_rmse_ft": candidate_rmse,
                "always_parent_late_rmse_ft": parent_rmse,
                "late_profile_oracle_rmse_ft": oracle_rmse,
                "optimistic_gain_vs_exp490_ft": candidate_rmse - selected_rmse,
                "selected_direct_exp490_fraction": float(np.mean(chosen_names == "alpha1_tau0")),
                "selected_parent_fraction": float(np.mean(chosen_names == "parent")),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 6. Plot and artifact helpers

# %%
def save_depth_plot(depth_metrics: pd.DataFrame, path: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for the full Kaggle readout")
    selected = depth_metrics[
        (depth_metrics["axis"].isin(["suffix_absolute_rows", "suffix_relative"]))
        & (depth_metrics["scope"] == "pooled")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, group_name in zip(axes, ["suffix_absolute_rows", "suffix_relative"]):
        group = selected[selected["axis"] == group_name]
        axis.plot(group["bucket"], group["parent_rmse_ft"], marker="o", label="exp357 parent")
        axis.plot(group["bucket"], group["candidate_rmse_ft"], marker="o", label="exp490")
        axis.set_title(group_name)
        axis.set_ylabel("RMSE ft")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_plot(associations: pd.DataFrame, path: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for the full Kaggle readout")
    selected = associations.assign(
        abs_spearman=associations["spearman_with_benefit_mse"].abs()
    ).nlargest(14, "abs_spearman").sort_values("spearman_with_benefit_mse")
    fig, axis = plt.subplots(figsize=(9, 6))
    colors = np.where(selected["spearman_with_benefit_mse"] >= 0, "#2a9d8f", "#e76f51")
    axis.barh(selected["feature"], selected["spearman_with_benefit_mse"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Spearman with exp490 MSE benefit (higher = stronger)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_representative_plot(
    arrays: Mapping[str, Any],
    well_readout: pd.DataFrame,
    path: Path,
    each_side: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for the full Kaggle readout")
    chosen = pd.concat(
        [
            well_readout.nsmallest(each_side, "candidate_minus_parent_rmse_ft"),
            well_readout.nlargest(each_side, "candidate_minus_parent_rmse_ft"),
        ]
    ).drop_duplicates("well")
    ncols = 3
    nrows = int(math.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), squeeze=False)
    for axis, (_, record) in zip(axes.ravel(), chosen.iterrows()):
        code = int(np.flatnonzero(arrays["wells"] == record["well"])[0])
        mask = arrays["well_codes"] == code
        order = np.argsort(arrays["suffix_offset"][mask])
        x = arrays["suffix_offset"][mask][order]
        parent_error = arrays["parent_error"][mask][order]
        candidate_error = arrays["candidate_error"][mask][order]
        stride = max(1, int(math.ceil(len(x) / 1200)))
        axis.plot(x[::stride], parent_error[::stride], linewidth=1.0, label="exp357 error")
        axis.plot(x[::stride], candidate_error[::stride], linewidth=1.0, label="exp490 error")
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(
            f"{record['well']}  ΔRMSE={record['candidate_minus_parent_rmse_ft']:+.2f}"
        )
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(chosen) :]:
        axis.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Representative strongest and weakest exp490 wells", y=1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def artifact_hashes(output: Path, filenames: Sequence[str]) -> dict[str, str]:
    return {filename: sha256_file(output / filename) for filename in filenames if (output / filename).is_file()}


# %% [markdown]
# ## 7--10. Setup, readout, policy audit, metrics, and generated artifacts

# %%
def run_readout() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    output = artifacts_dir()
    predictions_path = resolve_artifact(
        get_nested(config, "data.exp490_merge_source.candidates"),
        get_nested(config, "data.inputs.predictions.filename"),
    )
    features_path = resolve_artifact(
        get_nested(config, "data.exp499_feature_source.candidates"),
        get_nested(config, "data.inputs.target_free_features.filename"),
    )
    input_shas = {
        "predictions_raw_gzip_sha256": assert_sha(
            predictions_path,
            get_nested(config, "data.inputs.predictions.raw_gzip_sha256"),
        ),
        "predictions_decompressed_sha256": assert_sha(
            predictions_path,
            get_nested(config, "data.inputs.predictions.decompressed_sha256"),
            gzip_content=True,
        ),
        "target_free_features_sha256": assert_sha(
            features_path,
            get_nested(config, "data.inputs.target_free_features.sha256"),
        ),
    }
    print("Inputs SHA-verified", json.dumps(input_shas, sort_keys=True), flush=True)

    predictions = pd.read_csv(
        predictions_path,
        usecols=list(PREDICTION_COLUMNS),
        dtype={"well": "string", "fold": "int8"},
    )
    features = pd.read_csv(features_path, dtype={"well": "string"})
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(predictions) != expected_rows:
        raise ValueError(f"prediction row count changed: {len(predictions)}")
    if predictions["well"].nunique() != expected_wells:
        raise ValueError("prediction well count changed")
    if sorted(predictions["fold"].unique().tolist()) != get_nested(config, "validation.expected_folds"):
        raise ValueError("fold contract changed")
    if len(features) != expected_wells or features["well"].nunique() != expected_wells:
        raise ValueError("feature well count changed")
    feature_columns = [column for column in features.columns if column != "well"]
    if len(feature_columns) != int(get_nested(config, "data.inputs.target_free_features.expected_feature_count")):
        raise ValueError("feature count changed")
    arrays = make_row_arrays(predictions)
    print(
        f"Loaded {len(predictions):,} rows / {len(arrays['wells'])} wells / "
        f"{len(feature_columns)} target-free features",
        flush=True,
    )

    well_readout = build_well_readout(
        arrays,
        features,
        int(get_nested(config, "readout.adverse_window_rows")),
    )
    depth_metrics = build_depth_metrics(arrays, config)
    associations, feature_quantiles = build_feature_readouts(well_readout, feature_columns)
    well_readout, archetype_metrics = build_archetypes(arrays, well_readout, config)

    profiles = fade_profiles(config)
    fade_metrics, fold_sse, folds = profile_metric_tables(arrays, profiles)
    global_weights, global_selections = outer_global_fade_policy(arrays, profiles, fold_sse, folds)
    prefix_weights, prefix_manifest = outer_alpha_tree_policy(
        arrays,
        well_readout,
        get_nested(config, "policies.alpha_tree.prefix_feature_columns"),
        config,
        "outer_prefix_alpha_tree",
    )
    context_weights, context_manifest = outer_alpha_tree_policy(
        arrays,
        well_readout,
        get_nested(config, "policies.alpha_tree.context_feature_columns"),
        config,
        "outer_context_alpha_tree",
    )
    profiles_by_name = {str(profile["profile"]): profile for profile in profiles}
    policy_specs = {
        "always_parent": np.zeros(len(predictions), dtype=float),
        "always_exp490": np.ones(len(predictions), dtype=float),
        "fixed_public_tau85": profile_weight(
            arrays["md_since"], profiles_by_name["alpha1_tau85"]
        ),
        "fixed_robust_tau500": profile_weight(
            arrays["md_since"], profiles_by_name["alpha1_tau500"]
        ),
        "outer_selected_global_fade": global_weights,
        "outer_prefix_alpha_tree": prefix_weights,
        "outer_context_alpha_tree": context_weights,
    }
    policy_well_frames = []
    policy_fold_frames = []
    for policy, weights in policy_specs.items():
        policy_well, policy_fold = aggregate_policy(arrays, weights, policy)
        policy_well_frames.append(policy_well)
        policy_fold_frames.append(policy_fold)
    policy_oof = pd.concat(policy_well_frames, ignore_index=True)
    fold_metrics = pd.concat(policy_fold_frames, ignore_index=True)
    early_transfer = early_truth_transfer_audit(
        arrays,
        profiles,
        get_nested(config, "readout.early_transfer_horizons"),
    )

    names = get_nested(config, "artifacts.files")
    write_csv(output / names["well_readout"], well_readout.sort_values("candidate_minus_parent_rmse_ft"))
    write_csv(output / names["depth_metrics"], depth_metrics)
    write_csv(output / names["feature_associations"], associations)
    write_csv(output / names["feature_quantiles"], feature_quantiles)
    write_csv(output / names["archetype_metrics"], archetype_metrics)
    write_csv(output / names["fade_profile_metrics"], fade_metrics)
    write_csv(output / names["fold_metrics"], fold_metrics)
    write_csv(output / names["policy_oof"], policy_oof)
    write_csv(output / names["early_transfer"], early_transfer)
    model_manifest = {
        "global_fade_outer_selections": global_selections,
        "prefix_tree_models": prefix_manifest,
        "context_tree_models": context_manifest,
        "truth_use_policy": {
            "descriptive": ["well_readout", "depth_metrics", "feature_associations", "archetypes"],
            "outer_fold_safe": ["outer_selected_global_fade", "outer_prefix_alpha_tree", "outer_context_alpha_tree"],
            "nondeployable_optimistic": ["early_truth_transfer"],
        },
    }
    write_json(output / names["model_manifest"], model_manifest)
    save_depth_plot(depth_metrics, output / names["depth_plot"])
    save_feature_plot(associations, output / names["feature_plot"])
    save_representative_plot(
        arrays,
        well_readout,
        output / names["trajectory_plot"],
        int(get_nested(config, "readout.representative_wells_each_side")),
    )

    pooled = fold_metrics[fold_metrics["scope"] == "pooled"].set_index("policy")
    baseline_rmse = float(pooled.loc["always_exp490", "rmse_ft"])
    global_rmse = float(pooled.loc["outer_selected_global_fade", "rmse_ft"])
    fixed_public_rmse = float(pooled.loc["fixed_public_tau85", "rmse_ft"])
    fixed_robust_rmse = float(pooled.loc["fixed_robust_tau500", "rmse_ft"])
    prefix_rmse = float(pooled.loc["outer_prefix_alpha_tree", "rmse_ft"])
    context_rmse = float(pooled.loc["outer_context_alpha_tree", "rmse_ft"])
    global_folds = fold_metrics[
        (fold_metrics["policy"] == "outer_selected_global_fade")
        & (fold_metrics["scope"] != "pooled")
    ]
    global_well = policy_oof[policy_oof["policy"] == "outer_selected_global_fade"]
    prefix_folds = fold_metrics[
        (fold_metrics["policy"] == "outer_prefix_alpha_tree")
        & (fold_metrics["scope"] != "pooled")
    ]
    useful_fade_gate = {
        "gain_vs_exp490": baseline_rmse - global_rmse
        >= float(get_nested(config, "gates.useful_fade_requires_all.cross_fitted_gain_vs_exp490_minimum_ft")),
        "nonworse_folds": int(np.sum(global_folds["gain_vs_exp490_ft"] >= 0))
        >= int(get_nested(config, "gates.useful_fade_requires_all.nonworse_folds_minimum")),
        "well_p95_tail": float(global_well["policy_minus_exp490_rmse_ft"].quantile(0.95))
        <= float(get_nested(config, "gates.useful_fade_requires_all.by_well_delta_vs_exp490_p95_max_ft")),
        "well_worst_tail": float(global_well["policy_minus_exp490_rmse_ft"].max())
        <= float(get_nested(config, "gates.useful_fade_requires_all.by_well_delta_vs_exp490_worst_max_ft")),
    }
    robust_folds = fold_metrics[
        (fold_metrics["policy"] == "fixed_robust_tau500")
        & (fold_metrics["scope"] != "pooled")
    ]
    robust_well = policy_oof[policy_oof["policy"] == "fixed_robust_tau500"]
    safe_fixed_fade_gate = {
        "gain_vs_exp490": baseline_rmse - fixed_robust_rmse
        >= float(get_nested(config, "gates.safe_fixed_fade_requires_all.gain_vs_exp490_minimum_ft")),
        "nonworse_folds": int(np.sum(robust_folds["gain_vs_exp490_ft"] >= 0))
        >= int(get_nested(config, "gates.safe_fixed_fade_requires_all.nonworse_folds_minimum")),
        "well_p95_tail": float(robust_well["policy_minus_exp490_rmse_ft"].quantile(0.95))
        <= float(get_nested(config, "gates.safe_fixed_fade_requires_all.by_well_delta_vs_exp490_p95_max_ft")),
        "well_worst_tail": float(robust_well["policy_minus_exp490_rmse_ft"].max())
        <= float(get_nested(config, "gates.safe_fixed_fade_requires_all.by_well_delta_vs_exp490_worst_max_ft")),
    }
    prefix_alpha_values = policy_oof.loc[
        policy_oof["policy"] == "outer_prefix_alpha_tree", "mean_effective_alpha"
    ]
    prefix_tree_gate = {
        "gain_vs_exp490": baseline_rmse - prefix_rmse
        >= float(get_nested(config, "gates.prefix_tree_requires_all.gain_vs_exp490_minimum_ft")),
        "nonworse_folds": int(np.sum(prefix_folds["gain_vs_exp490_ft"] >= 0))
        >= int(get_nested(config, "gates.prefix_tree_requires_all.nonworse_folds_minimum")),
        "alpha_variation": float(prefix_alpha_values.std())
        >= float(get_nested(config, "gates.prefix_tree_requires_all.predicted_alpha_std_minimum")),
    }
    best_early = early_transfer.loc[early_transfer["optimistic_gain_vs_exp490_ft"].idxmax()]
    replay_gate = {
        "optimistic_gain": float(best_early["optimistic_gain_vs_exp490_ft"])
        >= float(get_nested(config, "gates.masked_prefix_replay_trigger_requires_all.optimistic_early_truth_gain_minimum_ft")),
        "early_late_transfer": float(best_early["early_late_direct_benefit_spearman"])
        >= float(get_nested(config, "gates.masked_prefix_replay_trigger_requires_all.early_late_benefit_spearman_minimum")),
    }
    technical_gate = {
        "input_sha_match": True,
        "expected_rows": len(predictions) == expected_rows,
        "expected_wells": len(arrays["wells"]) == expected_wells,
        "expected_folds": len(folds) == 5,
        "profile_count": len(profiles) == int(get_nested(config, "policies.fade_grid.expected_profiles")),
        "policy_prediction_coverage": len(policy_oof) == expected_wells * len(policy_specs),
    }

    generated_filenames = [
        names[key]
        for key in (
            "well_readout",
            "depth_metrics",
            "feature_associations",
            "feature_quantiles",
            "archetype_metrics",
            "fade_profile_metrics",
            "fold_metrics",
            "policy_oof",
            "early_transfer",
            "model_manifest",
            "depth_plot",
            "feature_plot",
            "trajectory_plot",
        )
    ]
    hashes = artifact_hashes(output, generated_filenames)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "accelerator": "cpu",
            "elapsed_sec": time.perf_counter() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "inputs": {
            "predictions_path": str(predictions_path),
            "features_path": str(features_path),
            "rows": len(predictions),
            "wells": len(arrays["wells"]),
            "folds": folds.tolist(),
            "shas": input_shas,
        },
        "headline": {
            "exp490_rmse_ft": baseline_rmse,
            "parent_rmse_ft": float(pooled.loc["always_parent", "rmse_ft"]),
            "outer_global_fade_rmse_ft": global_rmse,
            "fixed_public_tau85_rmse_ft": fixed_public_rmse,
            "fixed_robust_tau500_rmse_ft": fixed_robust_rmse,
            "outer_prefix_alpha_tree_rmse_ft": prefix_rmse,
            "outer_context_alpha_tree_rmse_ft": context_rmse,
            "beneficial_wells": int(well_readout["beneficial_well"].sum()),
            "harmful_wells": int((~well_readout["beneficial_well"]).sum()),
            "best_early_truth_horizon_rows": int(best_early["early_truth_rows"]),
            "best_optimistic_early_truth_gain_ft": float(best_early["optimistic_gain_vs_exp490_ft"]),
            "best_early_late_benefit_spearman": float(best_early["early_late_direct_benefit_spearman"]),
        },
        "top_strength_features": associations.head(12).to_dict(orient="records"),
        "strongest_wells": well_readout.nsmallest(10, "candidate_minus_parent_rmse_ft")[
            ["well", "fold", "rows", "candidate_rmse_ft", "parent_rmse_ft", "candidate_minus_parent_rmse_ft"]
        ].to_dict(orient="records"),
        "weakest_wells": well_readout.nlargest(10, "candidate_minus_parent_rmse_ft")[
            ["well", "fold", "rows", "candidate_rmse_ft", "parent_rmse_ft", "candidate_minus_parent_rmse_ft"]
        ].to_dict(orient="records"),
        "global_fade_outer_selections": global_selections,
        "gates": {
            "technical": technical_gate,
            "technical_passed": all(technical_gate.values()),
            "useful_fade": useful_fade_gate,
            "useful_fade_passed": all(useful_fade_gate.values()),
            "safe_fixed_fade": safe_fixed_fade_gate,
            "safe_fixed_fade_passed": all(safe_fixed_fade_gate.values()),
            "prefix_tree": prefix_tree_gate,
            "prefix_tree_passed": all(prefix_tree_gate.values()),
            "masked_prefix_replay_trigger": replay_gate,
            "masked_prefix_replay_trigger_passed": all(replay_gate.values()),
        },
        "execution_actual": {
            "fixed_fade_profiles": len(profiles),
            "alpha_tree_fits": len(prefix_manifest) + len(context_manifest),
            "kmeans_fits": 1,
            "lightgbm_boosters": 0,
            "hmm_well_runs": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
            "control_retraining": 0,
        },
        "artifact_paths": {key: str(output / value) for key, value in names.items()},
        "artifact_sha256": hashes,
    }
    write_json(output / names["summary"], summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed" if summary["gates"]["technical_passed"] else "technical_failed",
        "diagnostic_oof": summary["headline"],
        "gates": summary["gates"],
        "input_shas": input_shas,
        "artifact_sha256": {**hashes, names["summary"]: sha256_file(output / names["summary"])},
        "kaggle": {"kernel_id": None, "version": None, "url": None},
        "submission": None,
    }
    write_json(metrics_path(), metrics)
    print("\nHeadline", json.dumps(to_jsonable(summary["headline"]), sort_keys=True), flush=True)
    print("\nPolicy fold metrics\n", fold_metrics.to_string(index=False), flush=True)
    print("\nEarly-truth transfer\n", early_transfer.to_string(index=False), flush=True)
    print("\nTop feature associations\n", associations.head(15).to_string(index=False), flush=True)
    print("\nStrongest wells\n", summary["strongest_wells"], flush=True)
    print("\nWeakest wells\n", summary["weakest_wells"], flush=True)
    print("\nGates", json.dumps(to_jsonable(summary["gates"]), sort_keys=True), flush=True)
    print("Artifacts", output, flush=True)
    return summary


# %%
if __name__ == "__main__":
    RUN_SUMMARY = run_readout()
