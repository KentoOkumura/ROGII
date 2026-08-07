from __future__ import annotations

import gzip
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
EXP145_TRAIN_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "train_v2"
    / "artifacts"
)
EXP145_INFERENCE_ARTIFACTS = (
    Path("experiments")
    / "exp145_learned_likelihood_rawtest_feature_generator_parity"
    / "kaggle"
    / "output"
    / "inference_v3"
    / "artifacts"
)
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
EXP145_TRAIN_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz"
)
EXP145_RAWTEST_ML_FEATURES = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_rawtest_ml_features.csv.gz"
)
EXP145_FEATURE_SCHEMA = (
    "exp145_learned_likelihood_rawtest_feature_generator_parity_feature_schema.csv"
)
EXP145_SUMMARY = "exp145_learned_likelihood_rawtest_feature_generator_parity_summary.json"
EXP191_CONTINUITY_ARTIFACTS = (
    Path("experiments")
    / "exp191_typewell_late_range_continuity_selector_on_exp176"
    / "kaggle"
    / "output"
    / "train_v1"
    / "artifacts"
)
EXP191_CONTINUITY_OOF_PREDICTIONS = (
    "exp191_typewell_late_range_continuity_selector_on_exp176_oof_predictions.csv.gz"
)
EXP191_CONTINUITY_SUMMARY = (
    "exp191_typewell_late_range_continuity_selector_on_exp176_summary.json"
)
EXP148_OOF_PREDICTIONS = "exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz"
OUTPUT_PREFIX = "exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148"
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196

from typewell_late_range_continuity_selector_on_exp176 import (
    add_candidate_labels_and_features as exp191_add_candidate_labels_and_features,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    add_feature_enrichment as exp191_add_feature_enrichment,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    add_typewell_late_range_prior as exp191_add_typewell_late_range_prior,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    candidate_specs_from_config as exp191_candidate_specs_from_config,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    configured_raw_columns as exp191_configured_raw_columns,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    load_exp176_feature_columns as exp191_load_exp176_feature_columns,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    load_feature_cache as exp191_load_feature_cache,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    load_manifest as exp191_load_manifest,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    reconstruct_exp176_scores as exp191_reconstruct_exp176_scores,
)
from typewell_late_range_continuity_selector_on_exp176 import (
    second_margin_low as exp191_second_margin_low,
)
from settings import get_nested, load_config


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def _assert_finite_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for col in columns:
        values = frame[col].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            bad_count = int((~np.isfinite(values)).sum())
            raise ValueError(f"{label} contains non-finite values in {col}: {bad_count}")


def _assign_aligned_float32_columns(
    target: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
) -> None:
    for col in columns:
        target[col] = source[col].to_numpy(dtype=np.float32, copy=False)


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def find_optional_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path | None:
    try:
        return find_artifact(filename, explicit_path, local_artifacts=local_artifacts)
    except FileNotFoundError:
        return None


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    suffix = ids.astype(str).str.rsplit("_", n=1).str[-1]
    return pd.to_numeric(suffix, errors="coerce").fillna(-1).to_numpy(np.int32)


def exp063_lgb_config_family(*, fast: bool = False) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "max_bin": 255,
    }
    n_estimators = 600 if fast else 5000
    return [
        {
            **base,
            "num_leaves": 255,
            "min_child_samples": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "learning_rate": 0.03,
            "n_estimators": n_estimators,
            "seed": 123,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 0,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 29,
        },
    ]


def apply_mode_overrides(
    configs: list[dict[str, Any]],
    mode_config: dict[str, Any],
) -> list[dict[str, Any]]:
    use_gpu = bool(mode_config.get("use_gpu", False))
    common = dict(mode_config.get("common_overrides") or {})
    updated: list[dict[str, Any]] = []
    for params in configs:
        merged = dict(params)
        if use_gpu:
            merged["device_type"] = "gpu"
        else:
            merged.pop("device_type", None)
            merged.pop("gpu_use_dp", None)
        merged.update(common)
        if use_gpu and "gpu_use_dp" not in merged:
            merged["gpu_use_dp"] = False
        updated.append(merged)
    return updated


def load_exp072_full_replay_cache_frame(
    cache_path: str | Path | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
    frame = pd.read_csv(source, nrows=max_rows, dtype={"id": str, "well": str})
    required = {"id", "well", "target", "last_known_tvt", "z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    feature_columns = [col for col in frame.columns if col not in META_COLUMNS]
    if len(feature_columns) != EXPECTED_FULL_REPLAY_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_REPLAY_FEATURE_COUNT} full replay features, "
            f"got {len(feature_columns)} from {source}"
        )
    for col in ["target", *feature_columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    _assert_finite_columns(
        frame,
        ["target", *feature_columns],
        "exp072 full replay cache",
    )

    schema_path: Path | None = None
    summary_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    try:
        summary_path = find_artifact(FULL_REPLAY_CACHE_SUMMARY)
    except FileNotFoundError:
        summary_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_experiment": "exp072_exp063_full_replay_feature_cache",
        "source_kind": "exp063_full_public_replay_train_feature_cache",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "summary": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
    }
    return frame, feature_columns, metadata


def load_known_prefix_anchors(train_dir: str | Path, wells: list[str] | pd.Series) -> pd.DataFrame:
    train_dir = Path(train_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "anchor_tvt_true": float(anchor["TVT"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_anchor_columns(
    frame: pd.DataFrame,
    train_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_known_prefix_anchors(train_dir, frame["well"])
    anchor_lookup = anchors.set_index("well")
    for col in ["anchor_md", "anchor_z0", "anchor_t0", "anchor_tvt_true"]:
        frame[col] = frame["well"].map(anchor_lookup[col]).astype(np.float32)
    frame["known_prefix_rows"] = frame["well"].map(anchor_lookup["known_prefix_rows"]).astype(
        np.int32
    )
    if frame[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Anchor merge produced missing prefix anchor values")
    t0_delta = frame["last_known_tvt"].to_numpy(np.float32) - frame["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw prefix T0 does not match feature cache last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return frame, meta


def load_inference_prefix_anchors(
    test_dir: str | Path,
    wells: list[str] | pd.Series,
) -> pd.DataFrame:
    test_dir = Path(test_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = test_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw test well file not found for anchor recovery: {path}")
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for test well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_inference_anchor_columns(
    frame: pd.DataFrame,
    test_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_inference_prefix_anchors(test_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Inference anchor merge produced missing prefix anchor values")
    t0_delta = merged["last_known_tvt"].to_numpy(np.float32) - merged["anchor_t0"].to_numpy(
        np.float32
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw test prefix T0 does not match feature last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def find_model_manifest(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path if path.name == "manifest.json" else path / "manifest.json")
    candidates.extend(
        [
            Path.cwd() / "artifacts" / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path.cwd() / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path("experiments")
            / "exp092_u_projection_correction_disagreement_fullrun"
            / "kaggle"
            / "output"
            / "train"
            / "artifacts"
            / f"{OUTPUT_PREFIX}_lgb_models"
            / "manifest.json",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{OUTPUT_PREFIX}_lgb_models/manifest.json"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"model manifest not found. Checked:\n{checked}")


def _tail_rank(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(-1).to_numpy(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _tail_rank(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _source_tvt(frame: pd.DataFrame, name: str, spec: dict[str, Any]) -> np.ndarray:
    if spec.get("enabled", True) is False:
        raise ValueError(f"Projection source is disabled: {name}")
    value_column = spec.get("value_column")
    delta_column = spec.get("delta_column")
    if value_column:
        if value_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing value_column={value_column}")
        return frame[value_column].to_numpy(np.float32)
    if delta_column:
        if delta_column not in frame.columns:
            raise ValueError(f"Projection source {name} missing delta_column={delta_column}")
        return frame["last_known_tvt"].to_numpy(np.float32) + frame[delta_column].to_numpy(
            np.float32
        )
    raise ValueError(f"Projection source {name} needs value_column or delta_column")


def _weighted_polyfit_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    robust_iters: int,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        pred = np.full(len(y), fill, dtype=np.float32)
        zeros = np.zeros(len(y), dtype=np.float32)
        return pred, zeros, zeros, 0

    x_fit = x[finite]
    y_fit = y[finite]
    x_center = float(np.median(x_fit))
    x_scale = float(np.nanpercentile(x_fit, 95) - np.nanpercentile(x_fit, 5))
    if not np.isfinite(x_scale) or x_scale <= 1e-6:
        x_scale = max(float(np.max(x_fit) - np.min(x_fit)), 1.0)
    x_norm = (x - x_center) / x_scale
    x_fit_norm = x_norm[finite]
    fit_degree = int(min(max(degree, 0), max(finite.sum() - 1, 0)))
    if np.unique(np.round(x_fit_norm, 8)).size <= fit_degree:
        fit_degree = max(int(np.unique(np.round(x_fit_norm, 8)).size) - 1, 0)

    weights = np.ones(len(y_fit), dtype=np.float64)
    coef = np.array([float(np.mean(y_fit))])
    for _ in range(max(int(robust_iters), 1)):
        coef = np.polyfit(x_fit_norm, y_fit, deg=fit_degree, w=weights)
        residual = y_fit - np.polyval(coef, x_fit_norm)
        mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
        if not np.isfinite(mad) or mad <= 1e-6:
            break
        weights = np.minimum(1.0, (float(clip_sigma) * mad) / (np.abs(residual) + 1e-6))

    poly = np.poly1d(coef)
    pred = poly(x_norm)
    deriv1 = np.polyder(poly, 1)(x_norm) / x_scale
    if fit_degree >= 2:
        deriv2 = np.polyder(poly, 2)(x_norm) / (x_scale * x_scale)
    else:
        deriv2 = np.zeros_like(x_norm)
    return (
        pred.astype(np.float32),
        deriv1.astype(np.float32),
        deriv2.astype(np.float32),
        fit_degree,
    )


def build_u_projection_features(
    frame: pd.DataFrame,
    *,
    source_specs: dict[str, dict[str, Any]],
    degree: int = 3,
    robust_iters: int = 3,
    clip_sigma: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    enabled_specs = {
        str(name): dict(spec)
        for name, spec in source_specs.items()
        if dict(spec).get("enabled", True)
    }
    if len(enabled_specs) < 2:
        raise ValueError("At least two enabled projection sources are required")

    result = pd.DataFrame({"id": frame["id"].to_numpy(), "well": frame["well"].to_numpy()})
    group_columns: dict[str, list[str]] = {
        "projection_correction": [],
        "projection_shape": [],
        "u_disagreement": [],
    }
    summary_rows: list[dict[str, Any]] = []
    z = frame["z"].to_numpy(np.float32)
    u0 = frame["anchor_t0"].to_numpy(np.float32) + frame["anchor_z0"].to_numpy(np.float32)
    md_since = frame.get("md_since")
    if md_since is None:
        x_all = np.maximum(
            frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False).astype(float).to_numpy(),
            0.0,
        )
    else:
        x_all = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float32)

    source_u_columns: list[str] = []
    source_corr_columns: list[str] = []
    for source_name, spec in enabled_specs.items():
        prefix = f"uproj_{source_name}"
        tvt_source = _source_tvt(frame, source_name, spec)
        source_u = (tvt_source + z - u0).astype(np.float32)
        result[f"{prefix}_u"] = source_u
        source_u_columns.append(f"{prefix}_u")

        poly = np.zeros(len(frame), dtype=np.float32)
        slope = np.zeros(len(frame), dtype=np.float32)
        curvature = np.zeros(len(frame), dtype=np.float32)
        fit_degree = np.zeros(len(frame), dtype=np.int16)
        for _, idx in frame.groupby("well", sort=False).indices.items():
            idx_array = np.asarray(idx, dtype=np.int64)
            pred, deriv1, deriv2, used_degree = _weighted_polyfit_predict(
                x_all[idx_array],
                source_u[idx_array],
                degree=int(spec.get("degree", degree)),
                robust_iters=int(spec.get("robust_iters", robust_iters)),
                clip_sigma=float(spec.get("clip_sigma", clip_sigma)),
            )
            poly[idx_array] = pred
            slope[idx_array] = deriv1
            curvature[idx_array] = deriv2
            fit_degree[idx_array] = int(used_degree)

        resid = (source_u - poly).astype(np.float32)
        corr = (poly - source_u).astype(np.float32)
        abs_resid = np.abs(resid).astype(np.float32)
        mad_by_well = (
            pd.DataFrame({"well": frame["well"], "abs_resid": abs_resid})
            .groupby("well")["abs_resid"]
            .transform("median")
            .to_numpy(np.float32)
        )
        result[f"{prefix}_poly"] = poly
        result[f"{prefix}_resid"] = resid
        result[f"{prefix}_corr"] = corr
        result[f"{prefix}_abs_resid"] = abs_resid
        result[f"{prefix}_resid_mad"] = mad_by_well
        result[f"{prefix}_slope"] = slope
        result[f"{prefix}_curvature"] = curvature
        result[f"{prefix}_fit_degree"] = fit_degree.astype(np.float32)

        group_columns["projection_correction"].extend(
            [
                f"{prefix}_corr",
                f"{prefix}_resid",
                f"{prefix}_abs_resid",
                f"{prefix}_resid_mad",
            ]
        )
        group_columns["projection_shape"].extend(
            [
                f"{prefix}_poly",
                f"{prefix}_slope",
                f"{prefix}_curvature",
                f"{prefix}_fit_degree",
            ]
        )
        source_corr_columns.append(f"{prefix}_corr")
        summary_rows.append(
            {
                "source": source_name,
                "rows": int(len(source_u)),
                "u_mean": float(np.mean(source_u)),
                "u_std": float(np.std(source_u)),
                "abs_resid_mean": float(np.mean(abs_resid)),
                "abs_resid_p95": float(np.quantile(abs_resid, 0.95)),
                "resid_mad_mean": float(np.mean(mad_by_well)),
            }
        )

    source_names = list(enabled_specs)
    for left_i, left in enumerate(source_names):
        for right in source_names[left_i + 1 :]:
            left_col = f"uproj_{left}_u"
            right_col = f"uproj_{right}_u"
            diff_col = f"uproj_diff_{left}_minus_{right}"
            abs_col = f"uproj_absdiff_{left}_{right}"
            result[diff_col] = result[left_col].to_numpy(np.float32) - result[right_col].to_numpy(
                np.float32
            )
            result[abs_col] = np.abs(result[diff_col].to_numpy(np.float32))
            group_columns["u_disagreement"].extend([diff_col, abs_col])

    source_u_matrix = result[source_u_columns].to_numpy(np.float32)
    result["uproj_source_u_std"] = np.std(source_u_matrix, axis=1).astype(np.float32)
    result["uproj_source_u_range"] = (
        np.max(source_u_matrix, axis=1) - np.min(source_u_matrix, axis=1)
    ).astype(np.float32)
    group_columns["u_disagreement"].extend(["uproj_source_u_std", "uproj_source_u_range"])
    if len(source_corr_columns) >= 2:
        corr_matrix = result[source_corr_columns].to_numpy(np.float32)
        result["uproj_corr_std"] = np.std(corr_matrix, axis=1).astype(np.float32)
        result["uproj_corr_range"] = (
            np.max(corr_matrix, axis=1) - np.min(corr_matrix, axis=1)
        ).astype(np.float32)
        group_columns["u_disagreement"].extend(["uproj_corr_std", "uproj_corr_range"])

    numeric_cols = [col for col in result.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    _assert_finite_columns(result, numeric_cols, "U-projection feature frame")
    return result, group_columns, pd.DataFrame(summary_rows)


def load_learned_likelihood_ml_features(
    feature_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    *,
    feature_filename: str = EXP145_TRAIN_ML_FEATURES,
    local_artifacts: Path = EXP145_TRAIN_ARTIFACTS,
    source_experiment: str = "exp145_learned_likelihood_rawtest_feature_generator_parity",
    source_kind: str = "target_free_full_train_learned_likelihood_ml_features",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(feature_filename, feature_path, local_artifacts=local_artifacts)
    frame = pd.read_csv(source, dtype={"id": str, "well": str})
    required = {"id", "well", "fold", "md_since"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing learned likelihood feature columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame.duplicated(["id", "well"]).any():
        duplicated = int(frame.duplicated(["id", "well"]).sum())
        raise ValueError(
            f"learned likelihood ML feature cache has duplicated id/well rows: {duplicated}"
        )
    numeric_cols = [col for col in frame.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    _assert_finite_columns(frame, numeric_cols, "learned likelihood ML feature cache")

    resolved_schema: Path | None
    resolved_summary: Path | None
    try:
        resolved_schema = find_artifact(
            EXP145_FEATURE_SCHEMA,
            schema_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_schema = None
    try:
        resolved_summary = find_artifact(
            EXP145_SUMMARY,
            summary_path,
            local_artifacts=local_artifacts,
        )
    except FileNotFoundError:
        resolved_summary = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "source_experiment": source_experiment,
        "source_kind": source_kind,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": int(len(frame.columns)),
        "numeric_columns": numeric_cols,
        "schema": str(resolved_schema) if resolved_schema else None,
        "schema_sha256": sha256_file(resolved_schema) if resolved_schema else None,
        "summary": str(resolved_summary) if resolved_summary else None,
        "summary_sha256": sha256_file(resolved_summary) if resolved_summary else None,
    }
    return frame, metadata


def generate_current_test_learned_likelihood_ml_features(
    *,
    test_frame: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from learned_likelihood_rawtest_feature_generator_parity import (
        DEFAULT_EXP111_MANIFEST,
        DEFAULT_EXP111_SCHEMA,
        candidate_specs_from_config,
        ensure_multiobs_columns,
        exp111_model_feature_columns,
        generate_ml_features_from_frame,
        load_exp111_models,
        load_feature_schema,
        sha256_path,
        source_required_columns,
        write_ml_features,
    )
    from learned_likelihood_rawtest_feature_generator_parity import (
        find_artifact as find_generator_artifact,
    )
    from settings import get_nested, load_config

    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    exp111_artifacts = Path(str(get_nested(config, "data.exp111_artifact_dir_local") or ""))
    schema_path = find_generator_artifact(
        DEFAULT_EXP111_SCHEMA,
        get_nested(config, "data.exp111_feature_schema"),
        local_dirs=[exp111_artifacts],
    )
    manifest_path = find_generator_artifact(
        DEFAULT_EXP111_MANIFEST,
        get_nested(config, "data.exp111_model_manifest"),
        local_dirs=[exp111_artifacts],
    )
    row_feature_columns = load_feature_schema(schema_path)
    model_feature_columns = exp111_model_feature_columns(row_feature_columns)
    classifier, error_model, model_meta = load_exp111_models(manifest_path=manifest_path)

    source_frame, multiobs_meta = ensure_multiobs_columns(test_frame, candidates, config=config)
    required_columns = source_required_columns(config, candidates)
    missing = [column for column in required_columns if column not in source_frame.columns]
    if missing:
        raise ValueError(f"current test frame missing learned likelihood source columns: {missing}")
    features, long_likelihood = generate_ml_features_from_frame(
        source_frame[required_columns],
        candidates=candidates,
        row_feature_columns=row_feature_columns,
        model_feature_columns=model_feature_columns,
        classifier=classifier,
        error_model=error_model,
        config=config,
    )
    feature_path = (
        output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_ml_features.csv.gz"
    )
    long_path = output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_long.csv.gz"
    write_ml_features(feature_path, features)
    long_likelihood.to_csv(long_path, index=False, compression="gzip")
    return features, {
        "source": str(feature_path),
        "source_sha256": sha256_path(feature_path),
        "source_decompressed_sha256": sha256_path(feature_path, decompressed=True),
        "source_experiment": OUTPUT_PREFIX,
        "source_kind": "target_free_current_test_generated_learned_likelihood_ml_features",
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "columns": int(len(features.columns)),
        "exp111_schema": str(schema_path),
        "exp111_manifest": str(manifest_path),
        "exp111_model_meta": _jsonable(model_meta),
        "multiobs_generation": _jsonable(multiobs_meta),
        "long_likelihood": {
            "path": str(long_path),
            "rows": int(len(long_likelihood)),
            "sha256": sha256_path(long_path),
            "decompressed_sha256": sha256_path(long_path, decompressed=True),
        },
    }


def learned_feature_keys_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_keys = left[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    right_keys = (
        right[["id", "well"]].astype(str).sort_values(["id", "well"]).reset_index(drop=True)
    )
    return left_keys.equals(right_keys)


def build_learned_likelihood_features(
    learned_source: pd.DataFrame,
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "ll_")
    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {"learned_likelihood_confidence": []}

    direct_columns = [str(col) for col in config.get("direct_columns") or []]
    weighted_tvt_columns = [str(col) for col in config.get("weighted_tvt_columns") or []]
    candidate_tvt_columns = [str(col) for col in config.get("candidate_tvt_columns") or []]
    requested = direct_columns + weighted_tvt_columns + candidate_tvt_columns
    missing = [col for col in requested if col not in learned_source.columns]
    if missing:
        raise ValueError(
            f"learned likelihood ML feature cache missing configured columns: {missing}"
        )

    if learned_source["id"].equals(base_frame["id"]) and learned_source["well"].equals(
        base_frame["well"]
    ):
        value_source = learned_source
        features = learned_source[key_cols].copy()
        last_known_tvt = base_frame["last_known_tvt"].to_numpy(np.float32)
        likpf_mean_tvt = (
            last_known_tvt + base_frame["likpf_mean_d"].to_numpy(np.float32)
        ).astype(np.float32)
    else:
        base_lookup = base_frame[key_cols + ["last_known_tvt", "likpf_mean_d"]].copy()
        base_lookup["likpf_mean_tvt"] = (
            base_lookup["last_known_tvt"].to_numpy(np.float32)
            + base_lookup["likpf_mean_d"].to_numpy(np.float32)
        ).astype(np.float32)
        joined = learned_source[key_cols + requested].merge(
            base_lookup,
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError(
                "No shared rows between learned likelihood ML features and base feature frame"
            )
        value_source = joined
        features = joined[key_cols].copy()
        last_known_tvt = joined["last_known_tvt"].to_numpy(np.float32)
        likpf_mean_tvt = joined["likpf_mean_tvt"].to_numpy(np.float32)

    for col in direct_columns:
        out = f"{prefix}{col}"
        features[out] = value_source[col].to_numpy(np.float32)
        group_columns["learned_likelihood_confidence"].append(out)

    for col in weighted_tvt_columns + candidate_tvt_columns:
        raw = value_source[col].to_numpy(np.float32)
        minus_last = (raw - last_known_tvt).astype(np.float32)
        minus_likpf = (raw - likpf_mean_tvt).astype(np.float32)
        out_last = f"{prefix}{col}_minus_last_known_tvt"
        out_likpf = f"{prefix}{col}_minus_likpf_mean_tvt"
        features[out_last] = minus_last
        features[out_likpf] = minus_likpf
        group_columns["learned_likelihood_confidence"].extend([out_last, out_likpf])

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    _assert_finite_columns(features, feature_cols, "learned likelihood feature frame")

    summary = pd.DataFrame(
        [
            {
                "feature_group": "learned_likelihood_confidence",
                "configured_direct_columns": len(direct_columns),
                "configured_weighted_tvt_columns": len(weighted_tvt_columns),
                "configured_candidate_tvt_columns": len(candidate_tvt_columns),
                "generated_features": len(feature_cols),
                "rows": int(len(features)),
                "wells": int(features["well"].nunique()),
            }
        ]
    )
    return features, group_columns, summary


def _numeric_array(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(np.float32)
    return np.full(len(frame), np.float32(default), dtype=np.float32)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    denom = np.asarray(denominator, dtype=np.float32)
    return (numerator / np.maximum(np.abs(denom), np.float32(1.0))).astype(np.float32)


def _candidate_tvt(frame: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    column = str(spec["column"])
    kind = str(spec.get("kind") or ("delta" if column.endswith("_d") else "absolute"))
    raw = _numeric_array(frame, column)
    if kind == "delta":
        return (frame["last_known_tvt"].to_numpy(np.float32) + raw).astype(np.float32)
    if kind == "absolute":
        return raw.astype(np.float32)
    raise ValueError(f"Unsupported candidate kind={kind!r} for {column}")


def _rowwise_softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(centered, -50.0, 50.0))
    return (exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-6)).astype(np.float32)


def load_exp191_continuity_oof_predictions(
    prediction_path: str | Path | None,
    *,
    variant: str,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP191_CONTINUITY_OOF_PREDICTIONS,
        prediction_path,
        local_artifacts=EXP191_CONTINUITY_ARTIFACTS,
    )
    usecols = [
        "id",
        "well",
        "variant",
        "mode",
        "selected_candidate",
        "selected_candidate_index",
        "selected_tvt",
    ]
    frame = pd.read_csv(source, usecols=usecols, dtype={"id": str, "well": str})
    selected = frame[frame["variant"].eq(variant) & frame["mode"].eq(mode)].copy()
    if selected.empty:
        available = frame[["variant", "mode"]].drop_duplicates().head(40).to_dict("records")
        raise ValueError(
            f"exp191 continuity selector OOF variant={variant!r} mode={mode!r} not found; "
            f"available examples={available}"
        )
    duplicated = selected.duplicated(["id", "well"]).sum()
    if duplicated:
        raise ValueError(f"exp191 continuity OOF has duplicated id/well rows: {duplicated}")
    unused_columns = {
        "variant",
        "mode",
        "true_tvt",
        "abs_error",
        "oracle_candidate",
        "oracle_label",
    }
    selected = selected.drop(columns=[col for col in unused_columns if col in selected.columns])
    selected["selected_tvt"] = pd.to_numeric(selected["selected_tvt"], errors="coerce").astype(
        np.float32
    )
    selected["selected_candidate_index"] = pd.to_numeric(
        selected["selected_candidate_index"], errors="coerce"
    ).astype(np.int16)
    return selected, {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "source_kind": "exp191_oof_best_viterbi_selected_path",
        "variant": variant,
        "mode": mode,
        "rows": int(len(selected)),
        "wells": int(selected["well"].nunique()),
    }


def read_typewell_pct_context(
    train_dir: str | Path,
    *,
    min_typewell_span: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_dir = Path(train_dir)
    for path in sorted(train_dir.glob("*__typewell.csv")):
        well = path.name.replace("__typewell.csv", "")
        values = pd.to_numeric(pd.read_csv(path, usecols=["TVT"])["TVT"], errors="coerce")
        tvt = values[np.isfinite(values)].to_numpy(np.float32)
        if tvt.size:
            typewell_min = float(np.min(tvt))
            typewell_max = float(np.max(tvt))
            typewell_span = float(typewell_max - typewell_min)
        else:
            typewell_min = np.nan
            typewell_max = np.nan
            typewell_span = np.nan
        rows.append(
            {
                "well": well,
                "typewell_min": typewell_min,
                "typewell_max": typewell_max,
                "typewell_span": typewell_span,
                "valid_typewell_span": bool(
                    np.isfinite(typewell_span) and typewell_span >= float(min_typewell_span)
                ),
                "typewell_rows": int(tvt.size),
                "typewell_sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"No typewell files found under {train_dir}")
    context = pd.DataFrame(rows)
    valid = context[context["valid_typewell_span"].astype(bool)]
    if valid.empty:
        raise ValueError("No typewell context has a valid span")
    return context, {
        "train_dir": str(train_dir),
        "context_rows": int(len(context)),
        "valid_context_rows": int(len(valid)),
        "min_typewell_span": float(min_typewell_span),
        "typewell_span_min": float(valid["typewell_span"].min()),
        "typewell_span_median": float(valid["typewell_span"].median()),
        "typewell_span_max": float(valid["typewell_span"].max()),
    }


def load_optional_exp148_oof_predictions(
    prediction_path: str | Path | None,
    *,
    variant: str,
    mode: str,
    model: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    source = find_optional_artifact(EXP148_OOF_PREDICTIONS, prediction_path)
    if source is None:
        return None, {"available": False, "reason": "exp148 OOF prediction artifact not found"}
    frame = pd.read_csv(source, dtype={"id": str, "well": str})
    for column, value in [("variant", variant), ("mode", mode), ("model", model)]:
        if column in frame.columns:
            frame = frame[frame[column].astype(str).eq(str(value))]
    value_col = next(
        (col for col in ["pred_tvt", "prediction", "pred", "tvt"] if col in frame.columns),
        None,
    )
    if value_col is None:
        return None, {
            "available": False,
            "source": str(source),
            "reason": "prediction value column not found",
            "columns": list(frame.columns),
        }
    frame = frame[["id", "well", value_col]].rename(columns={value_col: "exp148_oof_tvt"})
    duplicated = frame.duplicated(["id", "well"]).sum()
    if duplicated:
        raise ValueError(f"exp148 OOF prediction artifact has duplicated id/well rows: {duplicated}")
    frame["exp148_oof_tvt"] = pd.to_numeric(frame["exp148_oof_tvt"], errors="coerce").astype(
        np.float32
    )
    return frame, {
        "available": True,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "variant": variant,
        "mode": mode,
        "model": model,
        "value_column": value_col,
    }


def _segment_features(
    ids: pd.Series,
    wells: pd.Series,
    selected_idx: np.ndarray,
    selected_tvt: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = len(ids)
    row_index = _row_indices_from_ids(ids)
    well_codes, _ = pd.factorize(wells.astype(str), sort=True)
    order = np.lexsort((row_index, well_codes.astype(np.int32)))
    local_switch = np.zeros(n_rows, dtype=np.float32)
    path_jump_abs = np.zeros(n_rows, dtype=np.float32)
    segment_len = np.ones(n_rows, dtype=np.float32)
    distance_to_boundary = np.zeros(n_rows, dtype=np.float32)

    start = 0
    while start < n_rows:
        well_code = well_codes[order[start]]
        end = start + 1
        while end < n_rows and well_codes[order[end]] == well_code:
            end += 1
        pos = order[start:end]
        if len(pos) > 1:
            prev = pos[:-1]
            cur = pos[1:]
            jumps = np.abs(selected_tvt[cur] - selected_tvt[prev]).astype(np.float32)
            switches = (selected_idx[cur] != selected_idx[prev]).astype(np.float32)
            path_jump_abs[cur] = jumps
            local_switch[cur] = switches
        run_start = 0
        ordered_selected = selected_idx[pos]
        while run_start < len(pos):
            run_end = run_start + 1
            while run_end < len(pos) and ordered_selected[run_end] == ordered_selected[run_start]:
                run_end += 1
            run_positions = pos[run_start:run_end]
            run_len = run_end - run_start
            segment_len[run_positions] = float(run_len)
            offsets = np.arange(run_len, dtype=np.float32)
            distance_to_boundary[run_positions] = np.minimum(offsets, offsets[::-1])
            run_start = run_end
        start = end
    return {
        "local_switch_flag": local_switch,
        "path_jump_abs": path_jump_abs,
        "segment_len": segment_len,
        "distance_to_segment_boundary": distance_to_boundary,
    }


def reconstruct_exp191_parent_score_surface(
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    config = config or load_config()
    candidates = exp191_candidate_specs_from_config(config)
    candidate_names = [spec.name for spec in candidates]
    raw_columns = exp191_configured_raw_columns(config, candidates)
    max_rows = len(base_frame)
    score_frame, source_meta = exp191_load_feature_cache(
        config=config,
        required_columns=raw_columns,
        max_rows=max_rows,
        cache_path=get_nested(config, "data.exp099_train_feature_cache_local"),
        schema_path=get_nested(config, "data.exp099_train_feature_schema_local"),
    )
    score_frame, _enrichment_columns, enrichment_meta = exp191_add_feature_enrichment(
        score_frame,
        config,
        max_rows=max_rows,
    )
    score_frame, late_range_columns, late_range_meta = exp191_add_typewell_late_range_prior(
        score_frame,
        config,
        candidates,
    )
    score_frame, candidate_values, oracle_labels = exp191_add_candidate_labels_and_features(
        score_frame,
        candidates,
    )
    if not score_frame["id"].equals(base_frame["id"]) or not score_frame["well"].equals(
        base_frame["well"]
    ):
        raise ValueError("exp191 parent score surface is not row-order aligned with exp148 frame")
    feature_columns, exp176_schema_meta = exp191_load_exp176_feature_columns(config)
    missing_features = [column for column in feature_columns if column not in score_frame.columns]
    if missing_features:
        raise ValueError(
            f"exp176 feature schema columns missing after exp191 reconstruction: {missing_features}"
        )
    manifest_path, manifest = exp191_load_manifest(config)
    scores, model_manifest = exp191_reconstruct_exp176_scores(
        frame=score_frame,
        candidates=candidates,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        feature_columns=feature_columns,
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    meta = {
        "source_meta": source_meta,
        "feature_enrichment": enrichment_meta,
        "typewell_late_range_prior": late_range_meta,
        "typewell_late_range_feature_count": int(len(late_range_columns)),
        "exp176_feature_schema": exp176_schema_meta,
        "exp176_model_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "resolved_models": int(len(model_manifest)),
        },
        "candidate_names": candidate_names,
    }
    return score_frame, candidate_names, candidate_values, scores, meta


def _rank_matrix_low(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row_idx = np.arange(values.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, values.shape[1] + 1, dtype=np.float32)
    return ranks


def build_exp191_continuity_selector_confidence_features(
    base_frame: pd.DataFrame,
    train_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    config = config or {}
    root_config = load_config()
    prefix = str(config.get("prefix") or "tlr191_")
    key_cols = ["id", "well"]
    group_name = "exp191_continuity_selector_confidence"
    group_columns: dict[str, list[str]] = {group_name: []}
    feature_data: dict[str, np.ndarray] = {}

    def add_feature(name: str, values: np.ndarray) -> None:
        feature_data[name] = np.asarray(values, dtype=np.float32)
        group_columns[group_name].append(name)

    selector_frame, selector_meta = load_exp191_continuity_oof_predictions(
        config.get("oof_predictions_path"),
        variant=str(config.get("selected_variant") or ""),
        mode=str(config.get("selected_mode") or "viterbi"),
    )
    if selector_frame["id"].equals(base_frame["id"]) and selector_frame["well"].equals(
        base_frame["well"]
    ):
        joined = selector_frame
    else:
        joined = base_frame[key_cols].merge(
            selector_frame,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    missing_rate = float(joined["selected_tvt"].isna().mean())
    max_missing_rate = float(config.get("max_missing_rate", 0.0))
    if missing_rate > max_missing_rate:
        raise ValueError(
            f"exp191 continuity feature join missing_rate={missing_rate:.6f} "
            f"exceeds max_missing_rate={max_missing_rate:.6f}"
        )

    last_tvt = base_frame["last_known_tvt"].to_numpy(np.float32)
    md_since = np.maximum(_numeric_array(base_frame, "md_since"), np.float32(0.0))
    selected_tvt = pd.to_numeric(joined["selected_tvt"], errors="coerce").to_numpy(np.float32)
    selected_idx = pd.to_numeric(
        joined["selected_candidate_index"], errors="coerce"
    ).fillna(-1).to_numpy(np.int16)
    selected_name = joined["selected_candidate"].fillna("missing").astype(str)

    score_frame, candidate_names, candidate_values, scores, score_meta = (
        reconstruct_exp191_parent_score_surface(base_frame, root_config)
    )
    candidate_index = {name: idx for idx, name in enumerate(candidate_names)}
    default_candidate = str(config.get("default_candidate") or "likpf_mean")
    default_idx = candidate_index.get(default_candidate, 0)
    default_tvt = candidate_values[:, default_idx].astype(np.float32)
    name_mapped_idx = selected_name.map(candidate_index).fillna(-1).to_numpy(np.int16)
    selected_idx = np.where(selected_idx >= 0, selected_idx, name_mapped_idx).astype(np.int16)
    missing = ~np.isfinite(selected_tvt)
    selected_tvt = np.where(missing, default_tvt, selected_tvt).astype(np.float32)
    selected_idx = np.where(missing | (selected_idx < 0), default_idx, selected_idx).astype(
        np.int16
    )

    dense_names = {str(value) for value in config.get("dense_candidates", [])}
    if not dense_names:
        dense_names = {"tvt_dense", "tvt_densew", "tvt_dense50"}
    pf_ancc_name = str(config.get("pf_ancc_candidate") or "pf_ancc")
    beam_names = {str(value) for value in config.get("beam_candidates", [])}
    if not beam_names:
        beam_names = {"beam_mean"}
    selected_is_default = selected_name.eq(default_candidate).to_numpy(np.float32)
    selected_is_pf_ancc = selected_name.eq(pf_ancc_name).to_numpy(np.float32)
    selected_is_dense = selected_name.isin(dense_names).to_numpy(np.float32)
    selected_is_beam = selected_name.isin(beam_names).to_numpy(np.float32)
    selected_family_code = np.full(len(base_frame), 3.0, dtype=np.float32)
    selected_family_code = np.where(selected_is_default > 0.5, 0.0, selected_family_code)
    selected_family_code = np.where(selected_is_pf_ancc > 0.5, 1.0, selected_family_code)
    selected_family_code = np.where(selected_is_dense > 0.5, 2.0, selected_family_code)
    selected_family_code = np.where(selected_name.eq("missing"), 0.0, selected_family_code)

    add_feature(f"{prefix}selected_candidate_code", selected_idx.astype(np.float32))
    add_feature(f"{prefix}selected_family_code", selected_family_code)
    add_feature(f"{prefix}is_likpf_default", selected_is_default)
    add_feature(f"{prefix}is_pf_ancc", selected_is_pf_ancc)
    add_feature(f"{prefix}is_dense_family", selected_is_dense)
    add_feature(f"{prefix}is_beam_family", selected_is_beam)
    add_feature(f"{prefix}selected_abs_delta_from_last", np.abs(selected_tvt - last_tvt))
    add_feature(
        f"{prefix}selected_abs_delta_from_last_norm_md",
        _safe_divide(np.abs(selected_tvt - last_tvt), md_since + 1.0),
    )

    predicted_error = np.maximum(scores["predicted_error"].astype(np.float32), 0.0)
    error_selected_idx, error_margin, error_top1 = exp191_second_margin_low(predicted_error)
    error_rank = _rank_matrix_low(predicted_error)
    row_idx = np.arange(len(base_frame))
    selected_pred_error = predicted_error[row_idx, selected_idx]
    selected_error_rank = error_rank[row_idx, selected_idx]
    for cand_idx, name in enumerate(candidate_names):
        add_feature(f"{prefix}pred_error_{name}", predicted_error[:, cand_idx])
        add_feature(f"{prefix}pred_error_rank_{name}", error_rank[:, cand_idx])
    add_feature(f"{prefix}pred_error_selected", selected_pred_error)
    add_feature(f"{prefix}pred_error_likpf", predicted_error[:, default_idx])
    add_feature(f"{prefix}error_margin_top2_top1", error_margin)
    add_feature(f"{prefix}error_top1_value", error_top1)
    add_feature(f"{prefix}selected_error_rank", selected_error_rank)
    add_feature(
        f"{prefix}selected_is_error_ranker_top1",
        (selected_idx == error_selected_idx).astype(np.float32),
    )
    add_feature(
        f"{prefix}pred_error_selected_minus_top1",
        (selected_pred_error - error_top1).astype(np.float32),
    )
    add_feature(
        f"{prefix}pred_error_selected_minus_likpf",
        (selected_pred_error - predicted_error[:, default_idx]).astype(np.float32),
    )

    typewell_context, typewell_meta = read_typewell_pct_context(
        train_dir,
        min_typewell_span=float(config.get("min_typewell_span", 1.0)),
    )
    joined_context = base_frame[["well"]].merge(
        typewell_context,
        on="well",
        how="left",
        validate="many_to_one",
    )
    context_missing_rate = float(joined_context["typewell_span"].isna().mean())
    if context_missing_rate > max_missing_rate:
        raise ValueError(
            f"exp191 typewell pct context missing_rate={context_missing_rate:.6f} "
            f"exceeds max_missing_rate={max_missing_rate:.6f}"
        )
    invalid_span = ~joined_context["valid_typewell_span"].fillna(False).astype(bool)
    if invalid_span.any():
        examples = joined_context.loc[invalid_span, "well"].head(5).tolist()
        raise ValueError(f"invalid typewell span for wells: {examples}")
    typewell_min = joined_context["typewell_min"].to_numpy(np.float32)
    typewell_span = joined_context["typewell_span"].to_numpy(np.float32)
    known_last_pct = ((last_tvt - typewell_min) / np.maximum(typewell_span, 1e-6)).astype(
        np.float32
    )
    selected_candidate_pct = (
        (selected_tvt - typewell_min) / np.maximum(typewell_span, 1e-6)
    ).astype(np.float32)
    pct_delta = (selected_candidate_pct - known_last_pct).astype(np.float32)
    add_feature(f"{prefix}known_last_pct", known_last_pct)
    add_feature(f"{prefix}selected_candidate_pct", selected_candidate_pct)
    add_feature(f"{prefix}selected_pct_minus_known_last_pct", pct_delta)

    risk_terms: list[np.ndarray] = []
    for lower in [float(value) for value in config.get("candidate_pct_lower_bounds", [0.5, 0.6, 0.7])]:
        label = str(lower).replace(".", "p")
        below = selected_candidate_pct < np.float32(lower)
        gap = np.maximum(np.float32(lower) - selected_candidate_pct, 0.0).astype(np.float32)
        add_feature(f"{prefix}selected_candidate_pct_below_{label}", below.astype(np.float32))
        add_feature(f"{prefix}selected_candidate_pct_gap_to_{label}", gap)
        risk_terms.append(gap)
    for margin in [float(value) for value in config.get("known_last_margins", [0.05, 0.10])]:
        label = str(margin).replace(".", "p")
        dynamic_lower = known_last_pct - np.float32(margin)
        below = selected_candidate_pct < dynamic_lower
        gap = np.maximum(dynamic_lower - selected_candidate_pct, 0.0).astype(np.float32)
        add_feature(
            f"{prefix}selected_candidate_pct_below_known_last_m{label}",
            below.astype(np.float32),
        )
        add_feature(f"{prefix}selected_candidate_pct_gap_to_known_last_m{label}", gap)
        risk_terms.append(gap)
    late_range_risk = (
        np.maximum.reduce(risk_terms).astype(np.float32)
        if risk_terms
        else np.zeros(len(base_frame), dtype=np.float32)
    )
    add_feature(f"{prefix}late_range_risk_score", late_range_risk)

    segments = _segment_features(
        base_frame["id"],
        base_frame["well"],
        selected_idx,
        selected_tvt,
    )
    for name, values in segments.items():
        add_feature(f"{prefix}{name}", values)
    segment_boundary_risk = _safe_divide(
        np.float32(1.0),
        feature_data[f"{prefix}distance_to_segment_boundary"] + np.float32(1.0),
    )
    add_feature(f"{prefix}segment_boundary_risk", segment_boundary_risk)
    add_feature(
        f"{prefix}late_risk_x_md_since_norm",
        late_range_risk * np.clip(md_since / np.float32(1000.0), 0.0, 5.0),
    )
    add_feature(f"{prefix}late_risk_x_dense", late_range_risk * selected_is_dense)
    add_feature(
        f"{prefix}pred_error_selected_x_late_risk",
        selected_pred_error * (np.float32(1.0) + late_range_risk),
    )
    add_feature(
        f"{prefix}error_margin_x_segment_boundary_risk",
        error_margin * segment_boundary_risk,
    )

    feature_cols = list(feature_data)
    feature_matrix = pd.DataFrame(feature_data, copy=False).fillna(0.0).astype(np.float32, copy=False)
    if feature_cols:
        _assert_finite_columns(
            feature_matrix,
            feature_cols,
            "exp191 continuity selector confidence feature frame",
        )
    result = pd.concat([base_frame[key_cols].reset_index(drop=True), feature_matrix], axis=1)
    summary = pd.DataFrame(
        [
            {
                "feature_group": group_name,
                "source_rows": selector_meta["rows"],
                "source_wells": selector_meta["wells"],
                "joined_rows": int(len(result)),
                "joined_wells": int(result["well"].nunique()),
                "missing_rate": missing_rate,
                "candidate_count": int(len(candidate_names)),
                "context_missing_rate": context_missing_rate,
                "generated_features": len(feature_cols),
            }
        ]
    )
    meta = {
        "continuity_selector_source": selector_meta,
        "score_surface": score_meta,
        "typewell_pct_context": typewell_meta,
        "generated_feature_count": int(len(feature_cols)),
        "generated_feature_columns": feature_cols,
    }
    del score_frame, candidate_values
    gc.collect()
    return result, group_columns, summary, meta


def build_sp45_bimodal_selector_features(
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    features, groups, summary, _meta = build_exp191_continuity_selector_confidence_features(
        base_frame,
        Path("data/raw/train"),
        config,
    )
    return features, groups, summary


def feature_columns_for_variant(
    base_feature_columns: list[str],
    feature_group_columns: dict[str, list[str]],
    variant: dict[str, Any],
) -> list[str]:
    columns = list(base_feature_columns)
    groups = list(variant.get("feature_groups") or [])
    extra: list[str] = []
    for group in groups:
        if group not in feature_group_columns:
            raise ValueError(f"Unknown feature group for variant {variant}: {group}")
        extra.extend(feature_group_columns[group])
    for col in variant.get("extra_columns") or []:
        extra.append(str(col))
    seen = set(columns)
    for col in extra:
        if col not in seen:
            columns.append(col)
            seen.add(col)
    return columns


def _by_well_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    return (
        frame.groupby(["variant", "mode", "model", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
        .sort_values(["variant", "mode", "model", "rmse_tvt"], ascending=[True, True, True, False])
    )


def _bucket_metrics(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[["id", "variant", "mode", "model", "target_tvt", "pred_tvt"]].copy()
    context = source_frame[["id"]].copy()
    distance_source = source_frame.get("md_since", pd.Series(np.nan, index=source_frame.index))
    context["distance_bucket"] = _distance_bucket(distance_source)
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    frame = frame.merge(context, on="id", how="left", validate="many_to_one")
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    rows: list[pd.DataFrame] = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        grouped = (
            frame.groupby(["variant", "mode", "model", bucket_col], observed=True)
            .agg(
                rows=("id", "size"),
                rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
                error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            )
            .reset_index()
            .rename(columns={bucket_col: "bucket"})
        )
        grouped.insert(3, "bucket_family", bucket_col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _fit_one_variant_mode(
    *,
    variant: dict[str, Any],
    mode_name: str,
    mode_config: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_splits: int,
    fast: bool,
    early_stopping_rounds: int,
    max_train_rows: int | None,
    save_models: bool,
    selected_lgb_config_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    row_index = np.arange(len(frame))
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    indexed_configs = list(enumerate(configs))
    if selected_lgb_config_indices is not None:
        selected = {int(index) for index in selected_lgb_config_indices}
        indexed_configs = [(index, params) for index, params in indexed_configs if index in selected]
        if not indexed_configs:
            raise ValueError(
                f"selected_lgb_config_indices={sorted(selected)} selected no configs "
                f"from {len(configs)} available configs"
            )
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models" / variant_name / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "variant": variant_name,
                "mode": mode_name,
                "rows": int(len(frame)),
                "features": int(len(feature_columns)),
                "configs": int(len(indexed_configs)),
                "selected_lgb_config_indices": [int(index) for index, _ in indexed_configs],
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in indexed_configs:
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(row_index, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            x_train = frame.iloc[train_idx][feature_columns].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            x_valid = frame.iloc[valid_idx][feature_columns].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_valid, num_iteration=best_iter).astype(np.float32)
            oof[valid_idx] = pred
            pred_tvt = base[valid_idx] + pred
            feature_importances = model.feature_importances_
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__lgb{model_index}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "feature_groups": ",".join(variant.get("feature_groups") or []),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(target_tvt[valid_idx], pred_tvt),
                    "rmse_target": rmse(y[valid_idx], pred),
                    "prediction_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred_tvt,
                        label=f"{variant_name}/{mode_name}/lgb{model_index}/fold{fold}/tvt",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            for feature, importance in zip(
                feature_columns,
                feature_importances,
                strict=False,
            ):
                importance_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
            if save_models:
                model_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": f"{variant_name}/{mode_name}/{model_file}",
                        "sha256": model_sha,
                    }
                )
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, x_train, x_valid, y_train, y_valid
            gc.collect()

        oof_by_model.append(oof)
        pred_tvt = base + oof
        metric_rows.append(
            {
                "variant": variant_name,
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "feature_groups": ",".join(variant.get("feature_groups") or []),
                "best_iteration": None,
                "rmse_tvt": rmse(target_tvt, pred_tvt),
                "rmse_target": rmse(y, oof),
                "prediction_sha256": prediction_sha256(
                    frame["id"],
                    pred_tvt,
                    label=f"{variant_name}/{mode_name}/lgb{model_index}/pooled/tvt",
                ),
                "model_file": None,
                "model_sha256": None,
            }
        )
    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_tvt = base + ensemble
    ensemble_sha = prediction_sha256(
        frame["id"],
        ensemble_tvt,
        label=f"{variant_name}/{mode_name}/lgb_mean/pooled/tvt",
    )
    metric_rows.append(
        {
            "variant": variant_name,
            "mode": mode_name,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "feature_groups": ",".join(variant.get("feature_groups") or []),
            "best_iteration": None,
            "rmse_tvt": rmse(target_tvt, ensemble_tvt),
            "rmse_target": rmse(y, ensemble),
            "prediction_sha256": ensemble_sha,
            "model_file": None,
            "model_sha256": None,
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well"].to_numpy(),
                "variant": variant_name,
                "mode": mode_name,
                "model": "lgb_mean",
                "last_known_tvt": base,
                "target": y,
                "target_tvt": target_tvt,
                "pred_target": ensemble,
                "pred_tvt": ensemble_tvt,
            }
        )
    )
    mode_summary = {
        "variant": variant_name,
        "mode": mode_name,
        "description": mode_config.get("description"),
        "feature_count": int(len(feature_columns)),
        "feature_groups": list(variant.get("feature_groups") or []),
        "use_gpu": bool(mode_config.get("use_gpu", False)),
        "common_overrides": mode_config.get("common_overrides") or {},
        "lgb_configs": [params for _, params in indexed_configs],
        "selected_lgb_config_indices": [int(index) for index, _ in indexed_configs],
        "lgb_mean_prediction_sha256": ensemble_sha,
        "model_count": int(len(model_rows)),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(importance_rows),
        model_rows,
        mode_summary,
    )


def _plot_mean_importance(mean_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    variants = mean_importance["variant"].drop_duplicates().tolist()
    if not variants:
        return
    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(12, max(4, 0.28 * int(top_n) * len(variants))),
        squeeze=False,
    )
    for ax, variant in zip(axes.ravel(), variants, strict=False):
        subset = mean_importance[mean_importance["variant"].eq(variant)].nlargest(
            top_n,
            "mean_importance",
        )
        subset = subset.sort_values("mean_importance", ascending=True)
        ax.barh(subset["feature"], subset["mean_importance"], color="#2f6f8f")
        ax.set_title(str(variant))
        ax.set_xlabel("mean feature_importances_")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_typewell_continuity_selector_confidence_replacement_only_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    typewell_continuity_feature_config: dict[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
    modes: dict[str, dict[str, Any]] | None = None,
    active_modes: list[str] | tuple[str, ...] | None = None,
    n_splits: int = 5,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_rows: int | None = None,
    max_train_rows: int | None = None,
    save_models: bool = True,
    save_predictions: bool = True,
    top_n_importance: int = 40,
    selected_lgb_config_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    print(
        json.dumps(
            {
                "stage": "loaded_exp072_full_replay_cache",
                "rows": int(len(frame)),
                "features": int(len(base_feature_columns)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    print(
        json.dumps(
            {
                "stage": "added_anchor_columns",
                "rows": int(len(frame)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
    )
    print(
        json.dumps(
            {
                "stage": "loaded_exp145_learned_features",
                "rows": int(learned_source_meta["rows"]),
                "columns": int(learned_source_meta["columns"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    projection_config = projection_config or {}
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError(
            "LGB OOF U-projection features require nested fold generation. "
            "This first ablation keeps them disabled to avoid leakage."
        )
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    full_frame = frame
    if not isinstance(full_frame.index, pd.RangeIndex) or not full_frame.index.equals(
        pd.RangeIndex(len(full_frame))
    ):
        full_frame.reset_index(drop=True, inplace=True)
    projection_features = projection_features.reset_index(drop=True)
    _assign_aligned_float32_columns(
        full_frame,
        projection_features,
        projection_feature_columns,
    )
    del frame, projection_features
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "added_projection_features",
                "rows": int(len(full_frame)),
                "projection_features": int(len(projection_feature_columns)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    enabled_feature_groups = {
        str(group)
        for variant in variants or []
        if variant.get("enabled", True)
        for group in variant.get("feature_groups") or []
    }
    learned_features_required = any(
        group in enabled_feature_groups for group in learned_group_columns
    )
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
    if learned_features_required:
        if full_frame["id"].equals(learned_features["id"]) and full_frame["well"].equals(
            learned_features["well"]
        ):
            _assign_aligned_float32_columns(
                full_frame,
                learned_features,
                learned_feature_columns,
            )
        else:
            full_frame = full_frame.merge(
                learned_features,
                on=["id", "well"],
                how="inner",
                validate="one_to_one",
            )
            if full_frame.empty:
                raise ValueError(
                    "No shared rows between exp072/exp092 feature surface and learned likelihood features"
                )
    else:
        if len(learned_features) != len(full_frame) or not learned_feature_keys_match(
            full_frame, learned_features
        ):
            raise ValueError(
                "Replacement-only audit requires complete exp145 learned-likelihood inventory "
                "even though the active variant excludes those columns"
            )
    coverage_meta = {
        "base_rows_before_feature_join": int(before_rows),
        "base_wells_before_feature_join": int(before_wells),
        "learned_feature_rows": int(learned_source_meta["rows"]),
        "learned_feature_wells": int(learned_source_meta["wells"]),
        "joined_rows": int(len(full_frame)),
        "joined_wells": int(full_frame["well"].nunique()),
        "dropped_base_rows": int(before_rows - len(full_frame)),
        "dropped_base_wells": int(before_wells - full_frame["well"].nunique()),
        "learned_features_attached_to_frame": bool(learned_features_required),
        "full_train_coverage_pass": bool(
            before_rows == len(full_frame) and before_wells == full_frame["well"].nunique()
        ),
    }
    del learned_features_source, learned_features
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "added_learned_likelihood_features",
                "rows": int(len(full_frame)),
                "learned_features": int(len(learned_feature_columns)),
                "attached_to_frame": bool(learned_features_required),
                "dropped_base_rows": int(coverage_meta["dropped_base_rows"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    continuity_features, continuity_group_columns, continuity_summary, continuity_meta = (
        build_exp191_continuity_selector_confidence_features(
            full_frame,
            train_dir,
            typewell_continuity_feature_config or {},
        )
    )
    continuity_feature_columns = [
        col for col in continuity_features.columns if col not in {"id", "well"}
    ]
    if not full_frame["id"].equals(continuity_features["id"]) or not full_frame["well"].equals(
        continuity_features["well"]
    ):
        raise ValueError(
            "exp191 continuity selector features are not row-order aligned with train feature frame"
        )
    _assign_aligned_float32_columns(
        full_frame,
        continuity_features,
        continuity_feature_columns,
    )
    del continuity_features
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "added_exp191_continuity_selector_features",
                "rows": int(len(full_frame)),
                "exp191_continuity_selector_features": int(len(continuity_feature_columns)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **continuity_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )
    continuity_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_exp191_continuity_feature_summary.csv",
        index=False,
    )

    selected_variants = list(variants or [])
    if not selected_variants:
        raise ValueError("No feature ablation variants configured")
    variant_names = [str(variant.get("name")) for variant in selected_variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"Duplicate variant names: {variant_names}")
    enabled_variant_names = [
        str(variant["name"]) for variant in selected_variants if variant.get("enabled", True)
    ]
    if not enabled_variant_names:
        raise ValueError("No enabled feature ablation variants configured")
    mode_map = modes or {}
    selected_modes = list(active_modes or mode_map)
    if not selected_modes:
        raise ValueError("No active LightGBM modes configured")

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    feature_schema_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        if not variant.get("enabled", True):
            continue
        variant_name = str(variant["name"])
        feature_columns = feature_columns_for_variant(
            base_feature_columns,
            feature_group_columns,
            variant,
        )
        for index, feature in enumerate(feature_columns):
            feature_schema_rows.append(
                {
                    "variant": variant_name,
                    "feature_index": int(index),
                    "feature": feature,
                    "is_projection_feature": bool(feature in projection_feature_columns),
                    "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                    "is_exp191_continuity_selector_feature": bool(
                        feature in continuity_feature_columns
                    ),
                }
            )
        for mode_name in selected_modes:
            if mode_name not in mode_map:
                raise ValueError(
                    f"active mode is not defined under model.training.modes: {mode_name}"
                )
            metrics, predictions, importance, models, mode_summary = _fit_one_variant_mode(
                variant=variant,
                mode_name=mode_name,
                mode_config=mode_map[mode_name],
                frame=full_frame,
                feature_columns=feature_columns,
                output_dir=output_dir,
                n_splits=n_splits,
                fast=fast,
                early_stopping_rounds=early_stopping_rounds,
                max_train_rows=max_train_rows,
                save_models=save_models,
                selected_lgb_config_indices=selected_lgb_config_indices,
            )
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            importance_frames.append(importance)
            model_rows.extend(models)
            mode_summaries.append(mode_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    mean_importance = (
        importance.groupby(["variant", "mode", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            fold_model_records=("importance", "size"),
        )
        .sort_values(["variant", "mode", "mean_importance"], ascending=[True, True, False])
    )
    by_well = _by_well_metrics(predictions)
    bucket_metrics = _bucket_metrics(predictions, full_frame)

    metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    importance.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv", index=False)
    mean_importance.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
        index=False,
    )
    _plot_mean_importance(
        mean_importance,
        output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
        int(top_n_importance),
    )
    if save_predictions:
        predictions.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(feature_schema_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        index=False,
    )

    model_root = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": OUTPUT_PREFIX,
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "continuity_selector_parent": "exp191_typewell_late_range_continuity_selector_on_exp176",
        "candidate_ranker_parent": "exp176_typewell_late_range_pfbeam_candidate_prior",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "typewell_continuity_selector_confidence_replacement_only_on_exp148_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "exp191_continuity_feature_source": continuity_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "typewell_continuity_feature_config": typewell_continuity_feature_config or {},
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "exp191_continuity_feature_groups": continuity_group_columns,
        "n_splits": int(n_splits),
        "selected_lgb_config_indices": (
            [int(index) for index in selected_lgb_config_indices]
            if selected_lgb_config_indices is not None
            else None
        ),
        "variants": selected_variants,
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    lgb_mean = pooled[pooled["model"].eq("lgb_mean")].sort_values("rmse_tvt")
    best = lgb_mean.iloc[0].to_dict() if not lgb_mean.empty else None
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "typewell_continuity_selector_confidence_replacement_only_on_exp148_full_train_rows",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "continuity_selector_parent": "exp191_typewell_late_range_continuity_selector_on_exp176",
        "candidate_ranker_parent": "exp176_typewell_late_range_pfbeam_candidate_prior",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "exp191_continuity_feature_source": continuity_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "selected_lgb_config_indices": (
            [int(index) for index in selected_lgb_config_indices]
            if selected_lgb_config_indices is not None
            else None
        ),
        "active_variants": enabled_variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "learned_feature_summary": f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
            "exp191_continuity_feature_summary": (
                f"{OUTPUT_PREFIX}_exp191_continuity_feature_summary.csv"
            ),
            "feature_importance": f"{OUTPUT_PREFIX}_feature_importance.csv",
            "feature_importance_mean": f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
            "feature_importance_plot": f"{OUTPUT_PREFIX}_feature_importance_mean_top.png",
            "predictions": f"{OUTPUT_PREFIX}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_saved_model_inference(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    data_dir: str | Path,
    test_dir: str | Path,
    model_manifest_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    sp45_feature_config: dict[str, Any] | None = None,
    variant_name: str = "sp45_bimodal_selector_confidence_replacement_only",
    mode_name: str = "gpu_repro_guard_dp_threads8",
    model_name: str = "lgb1",
    submission_target_column: str = "tvt",
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> dict[str, Any]:
    raise NotImplementedError(
        "exp191 continuity replacement-only is train-side only for the initial "
        "implementation. Port current-test continuity features in this same "
        "experiment only after split CV supports doing so."
    )
    import lightgbm as lgb

    try:
        from public_notebook_replay_audit import (
            build_replay_test_frame,
            configure_public_runtime,
        )
    except ModuleNotFoundError:
        from src.public_notebook_replay_audit import (  # type: ignore[import-not-found]
            build_replay_test_frame,
            configure_public_runtime,
        )

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    data_dir = Path(data_dir)
    test_dir = Path(test_dir)
    manifest_path = find_model_manifest(model_manifest_path)
    model_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    projection_config = projection_config or dict(manifest.get("projection_config") or {})
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("LGB OOF U-projection features are disabled for exp092 inference")

    print(f"loading saved LightGBM boosters from {model_root}", flush=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
    )
    base_test_frame, test_meta = build_replay_test_frame()
    base_test_frame["id"] = base_test_frame["id"].astype(str)
    base_test_frame["well"] = base_test_frame["well"].astype(str)
    base_feature_columns = [str(col) for col in manifest["feature_source"]["feature_columns"]]
    missing_base = sorted(set(base_feature_columns) - set(base_test_frame.columns))
    if missing_base:
        raise ValueError(f"raw-test replay frame is missing base features: {missing_base[:40]}")
    anchored_frame, anchor_meta = add_inference_anchor_columns(base_test_frame, test_dir)
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        anchored_frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    configured_groups = manifest.get("projection_feature_groups") or {}
    if configured_groups and {
        key: list(value) for key, value in projection_group_columns.items()
    } != {key: list(value) for key, value in configured_groups.items()}:
        raise ValueError("Projection feature groups differ from train manifest")

    variant_configs = {
        str(item["name"]): dict(item)
        for item in manifest.get("variants", [])
        if item.get("enabled", True)
    }
    if variant_name not in variant_configs:
        raise ValueError(f"variant={variant_name} not found in train manifest")
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    test_frame = pd.concat(
        [
            anchored_frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    try:
        rawtest_learned_features, rawtest_learned_meta = load_learned_likelihood_ml_features(
            learned_feature_path,
            schema_path=learned_schema_path,
            summary_path=learned_summary_path,
            feature_filename=EXP145_RAWTEST_ML_FEATURES,
            local_artifacts=EXP145_INFERENCE_ARTIFACTS,
            source_kind="target_free_rawtest_learned_likelihood_ml_features",
        )
    except FileNotFoundError:
        rawtest_learned_features, rawtest_learned_meta = (
            generate_current_test_learned_likelihood_ml_features(
                test_frame=anchored_frame,
                output_dir=output_dir,
            )
        )
    else:
        if not learned_feature_keys_match(rawtest_learned_features, anchored_frame):
            rawtest_learned_features, rawtest_learned_meta = (
                generate_current_test_learned_likelihood_ml_features(
                    test_frame=anchored_frame,
                    output_dir=output_dir,
                )
            )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        rawtest_learned_features,
        test_frame,
        learned_feature_config or dict(manifest.get("learned_feature_config") or {}),
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_join_rows = len(test_frame)
    test_frame = test_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test learned likelihood features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
    }
    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from train manifest")

    sp45_features, sp45_group_columns, sp45_summary = build_sp45_bimodal_selector_features(
        test_frame,
        sp45_feature_config or dict(manifest.get("sp45_feature_config") or {}),
    )
    sp45_feature_columns = [col for col in sp45_features.columns if col not in {"id", "well"}]
    if not test_frame[["id", "well"]].reset_index(drop=True).equals(
        sp45_features[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError(
            "Raw-test exp183 selector features are not row-order aligned with test feature frame"
        )
    test_frame = pd.concat(
        [
            test_frame.reset_index(drop=True),
            sp45_features[sp45_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    configured_sp45_groups = manifest.get("sp45_bimodal_feature_groups") or {}
    if configured_sp45_groups and {
        key: list(value) for key, value in sp45_group_columns.items()
    } != {key: list(value) for key, value in configured_sp45_groups.items()}:
        raise ValueError("exp183 selector feature groups differ from train manifest")
    feature_group_columns = {
        **feature_group_columns,
        **sp45_group_columns,
    }
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        feature_group_columns,
        variant_configs[variant_name],
    )

    missing_model = sorted(set(feature_columns) - set(test_frame.columns))
    if missing_model:
        raise ValueError(f"test frame is missing model features: {missing_model[:40]}")
    for col in feature_columns:
        test_frame[col] = pd.to_numeric(test_frame[col], errors="raise").astype(np.float32)
    if not np.isfinite(test_frame[feature_columns].to_numpy(np.float32)).all():
        raise ValueError("test feature matrix contains non-finite values")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("variant")) == variant_name
        and str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(
            f"No saved models for variant={variant_name} mode={mode_name} model={model_name}"
        )

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "variant": item.get("variant"),
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
                "rows": int(len(pred)),
            }
        )

    base = test_frame["last_known_tvt"].to_numpy(np.float32)
    pred_tvt = (base + pred_delta).astype(np.float32)
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "last_known_tvt": base,
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
        }
    )

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = dict(zip(predictions["id"].astype(str), predictions["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()

    predictions_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    projection_summary_path = (
        output_dir / f"{OUTPUT_PREFIX}_inference_projection_feature_summary.csv"
    )
    feature_schema_path = output_dir / f"{OUTPUT_PREFIX}_inference_feature_schema.csv"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    projection_summary.to_csv(projection_summary_path, index=False)
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
        index=False,
    )
    sp45_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_sp45_bimodal_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                "is_sp45_bimodal_feature": bool(feature in sp45_feature_columns),
            }
            for index, feature in enumerate(feature_columns)
        ]
    ).to_csv(feature_schema_path, index=False)

    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    submission_sha = sha256_file(submission_path)
    prediction_sha = prediction_sha256(
        predictions["id"],
        pred_delta,
        label=f"{variant_name}/{mode_name}/{model_name}/test",
    )
    metrics = {
        "variant": variant_name,
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "feature_count": int(len(feature_columns)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "prediction_sha256": prediction_sha,
        "submission_sha256": submission_sha,
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv",
        index=False,
    )
    summary = {
        "experiment": "exp194_exp183_selector_confidence_replacement_only_on_exp148",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "learned_feature_groups": learned_group_columns,
        "sp45_bimodal_feature_groups": sp45_group_columns,
        "selected": {
            "variant": variant_name,
            "mode": mode_name,
            "model": model_name,
            "model_count": int(len(model_rows)),
        },
        "metrics": metrics,
        "loaded_models": loaded_rows,
        "artifacts": {
            "predictions": predictions_path.name,
            "projection_feature_summary": projection_summary_path.name,
            "learned_feature_summary": f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
            "sp45_bimodal_feature_summary": (
                f"{OUTPUT_PREFIX}_inference_sp45_bimodal_feature_summary.csv"
            ),
            "feature_schema": feature_schema_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
            "submission": str(submission_path),
        },
        "known_followup_risk": "OOF worst-well degradation risk remains unresolved.",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
