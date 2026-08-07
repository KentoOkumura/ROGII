from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
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
EXP099_ARTIFACTS = (
    Path("experiments")
    / "exp099_pf_multi_observation_likelihood_probe"
    / "kaggle"
    / "output"
    / "train_v2"
    / "artifacts"
)
EXP065_ARTIFACTS = Path("experiments") / "exp065_typewell_supertype_cluster_cv_audit" / "artifacts"
EXP099_TRAIN_FEATURES = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXPERIMENT_NAME = "exp165_coordinate_frame_normalization_features_on_exp148"
OUTPUT_PREFIX = EXPERIMENT_NAME
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196


@dataclass(frozen=True)
class LearnedRankCandidateSpec:
    name: str
    source_column: str
    transform: str
    family: str
    probability_column: str
    error_column: str
    candidate_tvt_column: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class TypewellGroupMethod:
    name: str
    method: str
    threshold: str


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
    if not np.isfinite(frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("exp072 full replay cache contains non-finite numeric values")

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
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0", "anchor_md"]].isna().any().any():
        raise ValueError("Anchor merge produced missing prefix anchor values")
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
            "Recovered raw prefix T0 does not match feature cache last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


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


def find_model_manifest(
    explicit_path: str | Path | None = None,
    *,
    output_prefix: str = OUTPUT_PREFIX,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path if path.name == "manifest.json" else path / "manifest.json")
    candidates.extend(
        [
            Path.cwd() / "artifacts" / f"{output_prefix}_lgb_models" / "manifest.json",
            Path.cwd() / f"{output_prefix}_lgb_models" / "manifest.json",
            Path("experiments")
            / "exp092_u_projection_correction_disagreement_fullrun"
            / "kaggle"
            / "output"
            / "train"
            / "artifacts"
            / f"{output_prefix}_lgb_models"
            / "manifest.json",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{output_prefix}_lgb_models/manifest.json"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"model manifest not found. Checked:\n{checked}")


def find_model_manifests(
    explicit_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    output_prefixes: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    if explicit_paths:
        return [find_model_manifest(path) for path in explicit_paths]
    prefixes = list(output_prefixes or [OUTPUT_PREFIX])
    return [find_model_manifest(output_prefix=str(prefix)) for prefix in prefixes]


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
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("U-projection feature frame contains non-finite values")
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
    if not np.isfinite(frame[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood ML feature cache contains non-finite numeric values")

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
        find_artifact as find_generator_artifact,
        generate_ml_features_from_frame,
        load_exp111_models,
        load_feature_schema,
        sha256_path,
        source_required_columns,
        write_ml_features,
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
    features = joined[key_cols].copy()

    for col in direct_columns:
        out = f"{prefix}{col}"
        features[out] = joined[col].to_numpy(np.float32)
        group_columns["learned_likelihood_confidence"].append(out)

    for col in weighted_tvt_columns + candidate_tvt_columns:
        raw = joined[col].to_numpy(np.float32)
        minus_last = (raw - joined["last_known_tvt"].to_numpy(np.float32)).astype(np.float32)
        minus_likpf = (raw - joined["likpf_mean_tvt"].to_numpy(np.float32)).astype(np.float32)
        out_last = f"{prefix}{col}_minus_last_known_tvt"
        out_likpf = f"{prefix}{col}_minus_likpf_mean_tvt"
        features[out_last] = minus_last
        features[out_likpf] = minus_likpf
        group_columns["learned_likelihood_confidence"].extend([out_last, out_likpf])

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("learned likelihood feature frame contains non-finite values")

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


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    row_index = pd.to_numeric(
        ids.astype(str).str.extract(r"_(\d+)$", expand=False),
        errors="coerce",
    )
    if row_index.isna().any():
        bad = ids[row_index.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Could not parse raw row indices from ids: {bad}")
    return row_index.astype(np.int64).to_numpy()


def _safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    *,
    default: float = 0.0,
) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    return np.divide(
        num,
        den,
        out=np.full(np.broadcast_shapes(num.shape, den.shape), default, dtype=np.float32),
        where=np.abs(den) > 1e-6,
    ).astype(np.float32)


def _gradient_by_md(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    md = np.asarray(md, dtype=np.float64)
    if len(values) < 2 or np.unique(md).size < 2:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(md)
    sorted_md = md[order]
    sorted_values = values[order]
    grad_sorted = np.gradient(sorted_values, sorted_md, edge_order=1)
    grad = np.zeros(len(values), dtype=np.float64)
    grad[order] = grad_sorted
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _rolling_roughness(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float32))
    return (
        series.rolling(int(max(window, 3)), min_periods=2, center=True)
        .std()
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _raw_coordinate_frame_for_well(raw_dir: Path, well: str) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(f"raw horizontal well file not found: {path}")
    frame = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT_input"])
    for column in ["MD", "X", "Y", "Z", "TVT_input"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "X", "Y", "Z"]].isna().any().any():
        raise ValueError(f"raw coordinate frame has missing MD/X/Y/Z values: {path}")
    return frame


def build_coordinate_frame_features(
    base_frame: pd.DataFrame,
    raw_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "cfn_")
    tail_window_rows = int(config.get("prefix_tail_window_rows", 200))
    roughness_window_rows = int(config.get("roughness_window_rows", 31))
    raw_dir = Path(raw_dir)

    result_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {
        "coordinate_frame_geometry": [],
        "coordinate_frame_direction": [],
        "coordinate_frame_derivative": [],
        "coordinate_frame_interaction": [],
    }

    for well, group in base_frame.groupby("well", sort=False):
        raw = _raw_coordinate_frame_for_well(raw_dir, str(well))
        row_idx = _row_indices_from_ids(group["id"])
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(raw):
            raise ValueError(
                f"row index out of bounds for well={well}: "
                f"min={int(row_idx.min())}, max={int(row_idx.max())}, raw_rows={len(raw)}"
            )
        known_idx = np.flatnonzero(raw["TVT_input"].notna().to_numpy())
        if len(known_idx) == 0:
            raise ValueError(f"No known TVT_input prefix rows for well={well}")
        anchor_idx = int(known_idx[-1])
        anchor = raw.iloc[anchor_idx]
        tail_start = int(max(0, anchor_idx - tail_window_rows))
        tail_dx = float(raw.iloc[anchor_idx]["X"] - raw.iloc[tail_start]["X"])
        tail_dy = float(raw.iloc[anchor_idx]["Y"] - raw.iloc[tail_start]["Y"])
        tail_norm = float(np.hypot(tail_dx, tail_dy))
        if tail_norm <= 1e-6 and len(known_idx) >= 2:
            first_known = int(known_idx[0])
            tail_dx = float(raw.iloc[anchor_idx]["X"] - raw.iloc[first_known]["X"])
            tail_dy = float(raw.iloc[anchor_idx]["Y"] - raw.iloc[first_known]["Y"])
            tail_norm = float(np.hypot(tail_dx, tail_dy))
        if tail_norm <= 1e-6:
            tail_dx, tail_dy, tail_norm = 1.0, 0.0, 1.0
        ux = np.float32(tail_dx / tail_norm)
        uy = np.float32(tail_dy / tail_norm)

        md_all = raw["MD"].to_numpy(np.float32)
        x_all = raw["X"].to_numpy(np.float32)
        y_all = raw["Y"].to_numpy(np.float32)
        z_all = raw["Z"].to_numpy(np.float32)
        dx_dmd_all = _gradient_by_md(x_all, md_all)
        dy_dmd_all = _gradient_by_md(y_all, md_all)
        dz_dmd_all = _gradient_by_md(z_all, md_all)
        d2x_dmd2_all = _gradient_by_md(dx_dmd_all, md_all)
        d2y_dmd2_all = _gradient_by_md(dy_dmd_all, md_all)
        d2z_dmd2_all = _gradient_by_md(dz_dmd_all, md_all)
        xy_slope_all = np.hypot(dx_dmd_all, dy_dmd_all).astype(np.float32)
        xyz_slope_all = np.sqrt(
            np.square(dx_dmd_all) + np.square(dy_dmd_all) + np.square(dz_dmd_all)
        ).astype(np.float32)
        roughness_all = _rolling_roughness(xyz_slope_all, roughness_window_rows)

        selected = raw.iloc[row_idx].reset_index(drop=True)
        dx = selected["X"].to_numpy(np.float32) - np.float32(anchor["X"])
        dy = selected["Y"].to_numpy(np.float32) - np.float32(anchor["Y"])
        dz = selected["Z"].to_numpy(np.float32) - np.float32(anchor["Z"])
        md_since = selected["MD"].to_numpy(np.float32) - np.float32(anchor["MD"])
        xy_dist = np.hypot(dx, dy).astype(np.float32)
        xyz_dist = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz)).astype(np.float32)
        along = (dx * ux + dy * uy).astype(np.float32)
        cross = (-dx * uy + dy * ux).astype(np.float32)
        abs_cross = np.abs(cross).astype(np.float32)

        md_scale = float(np.nanpercentile(np.abs(md_since), 95))
        xy_scale = float(np.nanpercentile(xy_dist, 95))
        z_scale = float(np.nanpercentile(np.abs(dz), 95))
        md_scale = max(md_scale, 1.0)
        xy_scale = max(xy_scale, 1.0)
        z_scale = max(z_scale, 1.0)

        local_cos = _safe_divide(dx, xy_dist)
        local_sin = _safe_divide(dy, xy_dist)
        azimuth_cross = (local_cos * uy - local_sin * ux).astype(np.float32)
        azimuth_dot = (local_cos * ux + local_sin * uy).astype(np.float32)

        out = group[key_cols].copy().reset_index(drop=True)
        geometry_values = {
            f"{prefix}dx_anchor": dx,
            f"{prefix}dy_anchor": dy,
            f"{prefix}dz_anchor": dz,
            f"{prefix}md_since_raw": md_since,
            f"{prefix}dx_norm": dx / xy_scale,
            f"{prefix}dy_norm": dy / xy_scale,
            f"{prefix}dz_norm": dz / z_scale,
            f"{prefix}md_since_norm": md_since / md_scale,
            f"{prefix}xy_dist_norm": xy_dist / xy_scale,
            f"{prefix}xyz_dist_norm": xyz_dist / max(float(np.nanpercentile(xyz_dist, 95)), 1.0),
        }
        direction_values = {
            f"{prefix}along_track_norm": along / xy_scale,
            f"{prefix}cross_track_norm": cross / xy_scale,
            f"{prefix}cross_track_abs_norm": abs_cross / xy_scale,
            f"{prefix}prefix_tail_azimuth_cos": np.full(len(group), ux, dtype=np.float32),
            f"{prefix}prefix_tail_azimuth_sin": np.full(len(group), uy, dtype=np.float32),
            f"{prefix}row_azimuth_cos": local_cos,
            f"{prefix}row_azimuth_sin": local_sin,
            f"{prefix}prefix_tail_azimuth_cross": azimuth_cross,
            f"{prefix}prefix_tail_azimuth_dot": azimuth_dot,
            f"{prefix}straightness_xy_per_md": _safe_divide(xy_dist, np.abs(md_since)),
        }
        derivative_values = {
            f"{prefix}dX_dMD": dx_dmd_all[row_idx],
            f"{prefix}dY_dMD": dy_dmd_all[row_idx],
            f"{prefix}dZ_dMD": dz_dmd_all[row_idx],
            f"{prefix}d2X_dMD2": d2x_dmd2_all[row_idx],
            f"{prefix}d2Y_dMD2": d2y_dmd2_all[row_idx],
            f"{prefix}d2Z_dMD2": d2z_dmd2_all[row_idx],
            f"{prefix}xy_slope": xy_slope_all[row_idx],
            f"{prefix}xyz_slope": xyz_slope_all[row_idx],
            f"{prefix}trajectory_roughness": roughness_all[row_idx],
        }

        last_known = numeric_array(group, "last_known_tvt", default=0.0)
        pf_ancc = numeric_array(group, "pf_ancc", default=np.nan)
        beam_tvt = last_known + numeric_array(group, "beam_mean_d", default=0.0)
        likpf_tvt = last_known + numeric_array(group, "likpf_mean_d", default=0.0)
        pf_beam_absdiff = np.nan_to_num(np.abs(pf_ancc - beam_tvt), nan=0.0).astype(np.float32)
        likpf_beam_absdiff = np.abs(likpf_tvt - beam_tvt).astype(np.float32)
        pf_likpf_absdiff = np.nan_to_num(np.abs(pf_ancc - likpf_tvt), nan=0.0).astype(np.float32)
        near_row = (np.abs(md_since) <= 100.0).astype(np.float32)
        longtail = (np.abs(md_since) >= 1000.0).astype(np.float32)
        interaction_values = {
            f"{prefix}cross_abs_norm_x_pf_beam_absdiff": (abs_cross / xy_scale)
            * np.clip(pf_beam_absdiff / 100.0, 0.0, 5.0),
            f"{prefix}cross_abs_norm_x_likpf_beam_absdiff": (abs_cross / xy_scale)
            * np.clip(likpf_beam_absdiff / 100.0, 0.0, 5.0),
            f"{prefix}azimuth_mismatch_x_pf_likpf_absdiff": np.abs(azimuth_cross)
            * np.clip(pf_likpf_absdiff / 100.0, 0.0, 5.0),
            f"{prefix}near_row_x_xyz_dist_norm": near_row * (xyz_dist / max(md_scale, 1.0)),
            f"{prefix}near_row_x_cross_abs_norm": near_row * (abs_cross / xy_scale),
            f"{prefix}longtail_x_cross_abs_norm": longtail * (abs_cross / xy_scale),
            f"{prefix}longtail_x_roughness": longtail * roughness_all[row_idx],
            f"{prefix}md_norm_x_azimuth_mismatch_abs": (md_since / md_scale)
            * np.abs(azimuth_cross),
        }

        for group_name, values in [
            ("coordinate_frame_geometry", geometry_values),
            ("coordinate_frame_direction", direction_values),
            ("coordinate_frame_derivative", derivative_values),
            ("coordinate_frame_interaction", interaction_values),
        ]:
            for column, value in values.items():
                out[column] = np.asarray(value, dtype=np.float32)
                if column not in group_columns[group_name]:
                    group_columns[group_name].append(column)

        result_parts.append(out)
        summary_rows.append(
            {
                "well": str(well),
                "rows": int(len(group)),
                "raw_rows": int(len(raw)),
                "anchor_idx": int(anchor_idx),
                "anchor_md": float(anchor["MD"]),
                "tail_window_rows": int(tail_window_rows),
                "md_scale": float(md_scale),
                "xy_scale": float(xy_scale),
                "z_scale": float(z_scale),
                "prefix_tail_azimuth_cos": float(ux),
                "prefix_tail_azimuth_sin": float(uy),
            }
        )

    features = pd.concat(result_parts, ignore_index=True)
    numeric_cols = [col for col in features.columns if col not in key_cols]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("coordinate frame feature frame contains non-finite values")
    summary = pd.DataFrame(summary_rows)
    return features, group_columns, summary


def numeric_array(frame: pd.DataFrame, column: str, *, default: float | None = None) -> np.ndarray:
    if column not in frame.columns:
        if default is None:
            raise ValueError(f"required column is missing: {column}")
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def parse_typewell_group_methods(config: dict[str, Any]) -> list[TypewellGroupMethod]:
    methods = [
        TypewellGroupMethod(
            name=str(item["name"]),
            method=str(item["method"]),
            threshold=str(item["threshold"]),
        )
        for item in config.get("group_methods") or []
    ]
    if not methods:
        methods = [
            TypewellGroupMethod(
                name="native_overlap_0p999",
                method="native_overlap",
                threshold="0.999",
            )
        ]
    return methods


def load_typewell_prior_source_frame(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = config.get("exp099_train_feature_cache_local")
    source = find_artifact(
        EXP099_TRAIN_FEATURES,
        explicit,
        local_artifacts=EXP099_ARTIFACTS,
    )
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "likpf_mean",
        "md_since",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["true_tvt"] = frame["last_known_tvt"] + frame["target"]
    frame["true_delta_from_anchor"] = frame["true_tvt"] - frame["last_known_tvt"]
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
    }
    return frame, metadata


def load_typewell_cluster_assignments(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = config.get("exp065_cluster_assignments_local")
    source = find_artifact(
        EXP065_CLUSTER_ASSIGNMENTS,
        explicit,
        local_artifacts=EXP065_ARTIFACTS,
    )
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["cluster_size"] = (
        pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, metadata


def groupkfold_wells(wells: np.ndarray, n_folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    wells = np.array(sorted(map(str, wells)))
    rng = np.random.default_rng(seed)
    shuffled = wells.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    all_wells = set(wells.tolist())
    return [(all_wells.difference(set(valid.tolist())), set(valid.tolist())) for valid in folds]


def build_well_delta_arrays(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False):
        order = np.argsort(numeric_array(group, "md_since"))
        arrays[str(well)] = {
            "index": group.index.to_numpy(np.int64)[order],
            "md_since": numeric_array(group, "md_since")[order],
            "true_delta": numeric_array(group, "true_delta_from_anchor")[order],
        }
    return arrays


def make_typewell_group_lookup(
    assignments: pd.DataFrame,
    method: TypewellGroupMethod,
    *,
    min_cluster_size: int,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    subset = assignments[
        (assignments["method"].astype(str) == method.method)
        & (assignments["threshold"].astype(str) == method.threshold)
        & (assignments["cluster_size"] >= min_cluster_size)
    ].copy()
    well_to_cluster = dict(zip(subset["well_id"], subset["cluster_id"], strict=False))
    cluster_to_wells = {
        str(cluster): sorted(group["well_id"].astype(str).tolist())
        for cluster, group in subset.groupby("cluster_id", sort=False)
    }
    cluster_sizes = {cluster: len(wells) for cluster, wells in cluster_to_wells.items()}
    return well_to_cluster, cluster_to_wells, cluster_sizes


def interp_neighbor_delta(
    query_md: np.ndarray,
    neighbor_md: np.ndarray,
    neighbor_delta: np.ndarray,
    *,
    require_in_range: bool,
) -> np.ndarray:
    finite = np.isfinite(neighbor_md) & np.isfinite(neighbor_delta)
    if finite.sum() < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    x = neighbor_md[finite].astype(np.float64)
    y = neighbor_delta[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    left = np.nan if require_in_range else float(y[0])
    right = np.nan if require_in_range else float(y[-1])
    return np.interp(query_md.astype(np.float64), x, y, left=left, right=right).astype(np.float32)


def generate_fold_safe_typewell_prior(
    source_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    method: TypewellGroupMethod,
    config: dict[str, Any],
) -> pd.DataFrame:
    min_cluster_size = int(config.get("min_cluster_size", 2))
    min_neighbor_wells = int(config.get("min_neighbor_wells", 1))
    min_row_neighbor_values = int(config.get("min_row_neighbor_values", 1))
    require_in_range = bool((config.get("interpolation") or {}).get("require_in_range", True))
    n_folds = int(config.get("n_folds", 5))
    seed = int(config.get("seed", 42))

    well_to_cluster, cluster_to_wells, cluster_sizes = make_typewell_group_lookup(
        assignments,
        method,
        min_cluster_size=min_cluster_size,
    )
    well_arrays = build_well_delta_arrays(source_frame)
    prior_delta = np.full(len(source_frame), np.nan, dtype=np.float32)
    prior_std = np.full(len(source_frame), np.nan, dtype=np.float32)
    prior_count = np.zeros(len(source_frame), dtype=np.int16)
    prior_neighbor_wells = np.zeros(len(source_frame), dtype=np.int16)
    prior_cluster_size = np.zeros(len(source_frame), dtype=np.int16)

    splits = groupkfold_wells(source_frame["well"].unique(), n_folds, seed)
    for train_wells, valid_wells in splits:
        for well in sorted(valid_wells):
            cluster = well_to_cluster.get(well)
            if cluster is None or well not in well_arrays:
                continue
            candidate_neighbors = [
                neighbor
                for neighbor in cluster_to_wells.get(cluster, [])
                if neighbor in train_wells and neighbor in well_arrays and neighbor != well
            ]
            if len(candidate_neighbors) < min_neighbor_wells:
                continue
            query = well_arrays[well]
            row_idx = query["index"]
            query_md = query["md_since"]
            neighbor_values: list[np.ndarray] = []
            for neighbor in candidate_neighbors:
                neighbor_data = well_arrays[neighbor]
                values = interp_neighbor_delta(
                    query_md,
                    neighbor_data["md_since"],
                    neighbor_data["true_delta"],
                    require_in_range=require_in_range,
                )
                if np.isfinite(values).any():
                    neighbor_values.append(values)
            if not neighbor_values:
                continue
            stacked = np.vstack(neighbor_values)
            counts = np.isfinite(stacked).sum(axis=0)
            valid_rows = counts >= min_row_neighbor_values
            if not valid_rows.any():
                continue
            prior_delta[row_idx[valid_rows]] = np.nanmedian(stacked[:, valid_rows], axis=0).astype(
                np.float32
            )
            prior_std[row_idx[valid_rows]] = np.nanstd(stacked[:, valid_rows], axis=0).astype(
                np.float32
            )
            prior_count[row_idx] = counts.astype(np.int16)
            prior_neighbor_wells[row_idx] = len(candidate_neighbors)
            prior_cluster_size[row_idx] = cluster_sizes.get(cluster, 0)

    return pd.DataFrame(
        {
            "id": source_frame["id"].to_numpy(),
            "well": source_frame["well"].to_numpy(),
            f"{method.name}_prior_delta_raw": prior_delta,
            f"{method.name}_prior_std_raw": prior_std,
            f"{method.name}_prior_count": prior_count,
            f"{method.name}_neighbor_wells": prior_neighbor_wells,
            f"{method.name}_cluster_size": prior_cluster_size,
        }
    )


def build_typewell_neighbor_prior_features(
    learned_source: pd.DataFrame,
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    del learned_source
    config = config or {}
    prefix = str(config.get("prefix") or "twnp_")
    methods = parse_typewell_group_methods(config)
    source_frame, source_meta = load_typewell_prior_source_frame(config)
    assignments, assignment_meta = load_typewell_cluster_assignments(config)
    prior_frames = [
        generate_fold_safe_typewell_prior(source_frame, assignments, method, config)
        for method in methods
    ]
    key_cols = ["id", "well"]
    joined = base_frame[
        [
            "id",
            "well",
            "last_known_tvt",
            "md_since",
            "likpf_mean_d",
            "pf_ancc",
            "beam_mean_d",
        ]
    ].copy()
    for prior in prior_frames:
        prior_cols = [col for col in prior.columns if col not in key_cols]
        joined = joined.merge(prior[key_cols + prior_cols], on=key_cols, how="left")
    if len(joined) != len(base_frame):
        raise ValueError("typewell prior join changed row count")

    result = joined[key_cols].copy()
    group_columns: dict[str, list[str]] = {
        "typewell_neighbor_prior_value": [],
        "typewell_neighbor_prior_quality": [],
        "typewell_neighbor_prior_interaction": [],
        "typewell_neighbor_prior_correction_proxy": [],
    }
    last_known = joined["last_known_tvt"].to_numpy(np.float32)
    md_since = numeric_array(joined, "md_since", default=0.0)
    likpf_tvt = last_known + numeric_array(joined, "likpf_mean_d", default=0.0)
    pf_ancc = numeric_array(joined, "pf_ancc", default=np.nan)
    beam_tvt = last_known + numeric_array(joined, "beam_mean_d", default=0.0)
    pf_beam_absdiff = np.nan_to_num(np.abs(pf_ancc - beam_tvt), nan=0.0).astype(np.float32)

    summary_rows: list[dict[str, Any]] = [
        {
            "source": "exp099_feature_cache",
            "rows": source_meta["rows"],
            "wells": source_meta["wells"],
            "sha256": source_meta["source_decompressed_sha256"],
        },
        {
            "source": "exp065_cluster_assignments",
            "rows": assignment_meta["rows"],
            "wells": assignment_meta["wells"],
            "sha256": assignment_meta["source_sha256"],
        },
    ]

    for method in methods:
        raw_delta = joined[f"{method.name}_prior_delta_raw"].to_numpy(np.float32)
        raw_std = joined[f"{method.name}_prior_std_raw"].to_numpy(np.float32)
        count = joined[f"{method.name}_prior_count"].to_numpy(np.float32)
        neighbor_wells = joined[f"{method.name}_neighbor_wells"].to_numpy(np.float32)
        cluster_size = joined[f"{method.name}_cluster_size"].to_numpy(np.float32)
        valid = np.isfinite(raw_delta)
        prior_delta = np.nan_to_num(raw_delta, nan=0.0).astype(np.float32)
        prior_std = np.nan_to_num(raw_std, nan=999.0, posinf=999.0, neginf=999.0).astype(
            np.float32
        )
        prior_tvt = last_known + prior_delta
        prior_minus_likpf = np.where(valid, prior_tvt - likpf_tvt, 0.0).astype(np.float32)
        clipped = np.clip(prior_minus_likpf, -40.0, 40.0).astype(np.float32)
        gate = (valid & (prior_std <= 10.0) & (count >= 3)).astype(np.float32)
        method_prefix = f"{prefix}{method.name}_"

        value_cols = [
            f"{method_prefix}prior_valid",
            f"{method_prefix}prior_delta",
            f"{method_prefix}prior_abs_delta",
            f"{method_prefix}prior_minus_likpf_mean",
            f"{method_prefix}prior_abs_minus_likpf_mean",
        ]
        result[value_cols[0]] = valid.astype(np.float32)
        result[value_cols[1]] = prior_delta
        result[value_cols[2]] = np.abs(prior_delta).astype(np.float32)
        result[value_cols[3]] = prior_minus_likpf
        result[value_cols[4]] = np.abs(prior_minus_likpf).astype(np.float32)
        group_columns["typewell_neighbor_prior_value"].extend(value_cols)

        quality_cols = [
            f"{method_prefix}prior_std",
            f"{method_prefix}prior_count",
            f"{method_prefix}neighbor_wells",
            f"{method_prefix}cluster_size",
            f"{method_prefix}prior_count_log1p",
            f"{method_prefix}neighbor_wells_log1p",
            f"{method_prefix}std_x_valid",
        ]
        result[quality_cols[0]] = prior_std
        result[quality_cols[1]] = count
        result[quality_cols[2]] = neighbor_wells
        result[quality_cols[3]] = cluster_size
        result[quality_cols[4]] = np.log1p(count).astype(np.float32)
        result[quality_cols[5]] = np.log1p(neighbor_wells).astype(np.float32)
        result[quality_cols[6]] = (prior_std * valid.astype(np.float32)).astype(np.float32)
        group_columns["typewell_neighbor_prior_quality"].extend(quality_cols)

        md_scaled = np.clip(md_since / 1000.0, 0.0, 5.0).astype(np.float32)
        longtail = (md_since >= 1000.0).astype(np.float32)
        interaction_cols = [
            f"{method_prefix}valid_x_longtail",
            f"{method_prefix}delta_x_md_since_kft",
            f"{method_prefix}minus_likpf_x_md_since_kft",
            f"{method_prefix}valid_x_pf_beam_absdiff",
            f"{method_prefix}minus_likpf_x_pf_beam_absdiff",
        ]
        result[interaction_cols[0]] = valid.astype(np.float32) * longtail
        result[interaction_cols[1]] = prior_delta * md_scaled
        result[interaction_cols[2]] = prior_minus_likpf * md_scaled
        result[interaction_cols[3]] = valid.astype(np.float32) * pf_beam_absdiff
        result[interaction_cols[4]] = prior_minus_likpf * np.clip(pf_beam_absdiff / 100.0, 0.0, 5.0)
        group_columns["typewell_neighbor_prior_interaction"].extend(interaction_cols)

        correction_cols = [
            f"{method_prefix}clipped_correction_a0p2_c40",
            f"{method_prefix}clipped_correction_abs_a0p2_c40",
            f"{method_prefix}gate_std10_n3",
            f"{method_prefix}gated_clipped_correction_a0p2_c40",
        ]
        result[correction_cols[0]] = (0.2 * clipped).astype(np.float32)
        result[correction_cols[1]] = np.abs(result[correction_cols[0]].to_numpy(np.float32))
        result[correction_cols[2]] = gate
        result[correction_cols[3]] = (gate * result[correction_cols[0]].to_numpy(np.float32)).astype(
            np.float32
        )
        group_columns["typewell_neighbor_prior_correction_proxy"].extend(correction_cols)

        summary_rows.append(
            {
                "method": method.name,
                "valid_rows": int(valid.sum()),
                "valid_rate": float(valid.mean()),
                "mean_prior_count": float(np.mean(count)),
                "mean_neighbor_wells": float(np.mean(neighbor_wells)),
                "mean_cluster_size": float(np.mean(cluster_size)),
                "mean_prior_std_valid": float(np.mean(raw_std[np.isfinite(raw_std)]))
                if np.isfinite(raw_std).any()
                else None,
            }
        )

    numeric_cols = [col for col in result.columns if col not in key_cols]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("typewell neighbor prior feature frame contains non-finite values")
    return result, group_columns, pd.DataFrame(summary_rows)


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
    selected_lgb_models: list[str] | tuple[str, ...] | None = None,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    base_configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    model_config_pairs = [
        (f"lgb{model_index}", model_index, params)
        for model_index, params in enumerate(base_configs)
    ]
    if selected_lgb_models:
        requested = [str(model_name) for model_name in selected_lgb_models]
        known = {label for label, _, _ in model_config_pairs}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"Unknown selected_lgb_models: {unknown}")
        requested_set = set(requested)
        model_config_pairs = [pair for pair in model_config_pairs if pair[0] in requested_set]
    if not model_config_pairs:
        raise ValueError("No LightGBM configs selected")
    configs = [params for _, _, params in model_config_pairs]
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{output_prefix}_lgb_models" / variant_name / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "variant": variant_name,
                "mode": mode_name,
                "rows": int(len(frame)),
                "features": int(len(feature_columns)),
                "configs": int(len(configs)),
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_label, model_index, params in model_config_pairs:
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y[train_idx],
                eval_set=[(x_matrix[valid_idx], y[valid_idx])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
            oof[valid_idx] = pred
            pred_tvt = base[valid_idx] + pred
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__{model_label}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": model_label,
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
                        label=f"{variant_name}/{mode_name}/{model_label}/fold{fold}/tvt",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=False,
            ):
                importance_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": model_label,
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
                        "model": model_label,
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
                        "model": model_label,
                        "fold": int(fold),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        oof_by_model.append(oof)
        pred_tvt = base + oof
        metric_rows.append(
            {
                "variant": variant_name,
                "mode": mode_name,
                "model": model_label,
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
                    label=f"{variant_name}/{mode_name}/{model_label}/pooled/tvt",
                ),
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
                    "model": model_label,
                    "last_known_tvt": base,
                    "target": y,
                    "target_tvt": target_tvt,
                    "pred_target": oof,
                    "pred_tvt": pred_tvt,
                }
            )
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
        "lgb_configs": configs,
        "selected_lgb_models": [label for label, _, _ in model_config_pairs],
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


def run_coordinate_frame_normalization_features_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    coordinate_frame_config: dict[str, Any] | None = None,
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
    selected_lgb_models: list[str] | tuple[str, ...] | None = None,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    t0 = time.time()
    output_prefix = str(output_prefix)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
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
    full_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
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
    coverage_meta = {
        "base_rows_before_feature_join": int(before_rows),
        "base_wells_before_feature_join": int(before_wells),
        "learned_feature_rows": int(learned_source_meta["rows"]),
        "learned_feature_wells": int(learned_source_meta["wells"]),
        "joined_rows": int(len(full_frame)),
        "joined_wells": int(full_frame["well"].nunique()),
        "dropped_base_rows": int(before_rows - len(full_frame)),
        "dropped_base_wells": int(before_wells - full_frame["well"].nunique()),
        "full_train_coverage_pass": bool(
            before_rows == len(full_frame) and before_wells == full_frame["well"].nunique()
        ),
    }
    coordinate_frame_config = coordinate_frame_config or {}
    coordinate_features, coordinate_group_columns, coordinate_summary = (
        build_coordinate_frame_features(
            full_frame,
            train_dir,
            coordinate_frame_config,
        )
    )
    coordinate_feature_columns = [
        col for col in coordinate_features.columns if col not in {"id", "well"}
    ]
    full_frame = full_frame.merge(
        coordinate_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(full_frame) != coverage_meta["joined_rows"]:
        raise ValueError(
            "Coordinate frame features changed row coverage: "
            f"{len(full_frame)} of {coverage_meta['joined_rows']}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **coordinate_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{output_prefix}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{output_prefix}_learned_feature_summary.csv",
        index=False,
    )
    coordinate_summary.to_csv(
        output_dir / f"{output_prefix}_coordinate_frame_feature_summary.csv",
        index=False,
    )

    selected_variants = list(variants or [])
    if not selected_variants:
        raise ValueError("No feature ablation variants configured")
    variant_names = [str(variant.get("name")) for variant in selected_variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"Duplicate variant names: {variant_names}")
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
                    "is_coordinate_frame_feature": bool(feature in coordinate_feature_columns),
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
                selected_lgb_models=selected_lgb_models,
                output_prefix=output_prefix,
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

    metrics.to_csv(output_dir / f"{output_prefix}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{output_prefix}_by_well.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{output_prefix}_bucket_metrics.csv", index=False)
    importance.to_csv(output_dir / f"{output_prefix}_feature_importance.csv", index=False)
    mean_importance.to_csv(
        output_dir / f"{output_prefix}_feature_importance_mean.csv",
        index=False,
    )
    _plot_mean_importance(
        mean_importance,
        output_dir / f"{output_prefix}_feature_importance_mean_top.png",
        int(top_n_importance),
    )
    if save_predictions:
        predictions.to_csv(
            output_dir / f"{output_prefix}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(feature_schema_rows).to_csv(
        output_dir / f"{output_prefix}_feature_schema.csv",
        index=False,
    )

    model_root = output_dir / f"{output_prefix}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "coordinate_frame_normalization_features_on_exp148_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "coordinate_frame_config": coordinate_frame_config,
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "coordinate_frame_feature_groups": coordinate_group_columns,
        "n_splits": int(n_splits),
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
        "experiment": EXPERIMENT_NAME,
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "coordinate_frame_normalization_features_on_exp148_full_train_rows",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "selected_lgb_models": list(selected_lgb_models or ["lgb0", "lgb1", "lgb2"]),
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{output_prefix}_metrics.csv",
            "by_well": f"{output_prefix}_by_well.csv",
            "bucket_metrics": f"{output_prefix}_bucket_metrics.csv",
            "projection_feature_summary": f"{output_prefix}_projection_feature_summary.csv",
            "learned_feature_summary": f"{output_prefix}_learned_feature_summary.csv",
            "coordinate_frame_feature_summary": (
                f"{output_prefix}_coordinate_frame_feature_summary.csv"
            ),
            "feature_importance": f"{output_prefix}_feature_importance.csv",
            "feature_importance_mean": f"{output_prefix}_feature_importance_mean.csv",
            "feature_importance_plot": f"{output_prefix}_feature_importance_mean_top.png",
            "predictions": f"{output_prefix}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{output_prefix}_feature_schema.csv",
            "model_manifest": f"{output_prefix}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
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
    model_manifest_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    model_manifest_prefixes: list[str] | tuple[str, ...] | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    learned_typewell_prior_config: dict[str, Any] | None = None,
    variant_name: str = "typewell_neighbor_prior_addonly",
    mode_name: str = "cpu_deterministic_threads8",
    model_name: str = "lgb1",
    submission_target_column: str = "tvt",
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> dict[str, Any]:
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
    if model_manifest_path is not None and model_manifest_paths:
        raise ValueError("Use either model_manifest_path or model_manifest_paths, not both")
    explicit_manifest_paths = (
        [model_manifest_path] if model_manifest_path is not None else model_manifest_paths
    )
    manifest_paths = find_model_manifests(
        explicit_manifest_paths,
        output_prefixes=model_manifest_prefixes,
    )
    manifest_records = [
        {
            "path": manifest_path,
            "model_root": manifest_path.parent,
            "manifest": json.loads(manifest_path.read_text()),
        }
        for manifest_path in manifest_paths
    ]
    manifest = manifest_records[0]["manifest"]
    projection_config = projection_config or dict(manifest.get("projection_config") or {})
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("LGB OOF U-projection features are disabled for exp092 inference")

    print(
        "loading saved LightGBM boosters from "
        + ", ".join(str(record["model_root"]) for record in manifest_records),
        flush=True,
    )
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
    for record in manifest_records[1:]:
        other_columns = [
            str(col) for col in record["manifest"]["feature_source"]["feature_columns"]
        ]
        if other_columns != base_feature_columns:
            raise ValueError(
                "Feature columns differ across train manifests: "
                f"{manifest_paths[0]} vs {record['path']}"
            )
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
    learned_typewell_prior_config = learned_typewell_prior_config or dict(
        manifest.get("learned_typewell_prior_config") or {}
    )
    typewell_prior_features, typewell_prior_group_columns, typewell_prior_summary = (
        build_typewell_neighbor_prior_features(
            rawtest_learned_features,
            test_frame,
            learned_typewell_prior_config,
        )
    )
    typewell_prior_feature_columns = [
        col for col in typewell_prior_features.columns if col not in {"id", "well"}
    ]
    test_frame = test_frame.merge(
        typewell_prior_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test typewell neighbor prior features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **typewell_prior_group_columns,
    }
    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from train manifest")
    configured_typewell_prior_groups = manifest.get("learned_typewell_prior_feature_groups") or {}
    if configured_typewell_prior_groups and {
        key: list(value) for key, value in typewell_prior_group_columns.items()
    } != {key: list(value) for key, value in configured_typewell_prior_groups.items()}:
        raise ValueError("Learned likelihood typewell-prior feature groups differ from train manifest")
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

    model_rows: list[dict[str, Any]] = []
    for record in manifest_records:
        source_manifest = record["manifest"]
        for item in source_manifest.get("models", []):
            if (
                str(item.get("variant")) == variant_name
                and str(item.get("mode")) == mode_name
                and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
            ):
                model_rows.append(
                    {
                        **item,
                        "_model_root": record["model_root"],
                        "_manifest_path": record["path"],
                    }
                )
    if not model_rows:
        raise ValueError(
            f"No saved models for variant={variant_name} mode={mode_name} model={model_name}"
        )

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = Path(item["_model_root"]) / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "variant": item.get("variant"),
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "manifest": str(item.get("_manifest_path")),
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
    typewell_prior_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_typewell_prior_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                "is_typewell_neighbor_prior_feature": bool(
                    feature in typewell_prior_feature_columns
                ),
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
        "experiment": EXPERIMENT_NAME,
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifests": [str(path) for path in manifest_paths],
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "learned_feature_groups": learned_group_columns,
        "learned_typewell_prior_feature_groups": typewell_prior_group_columns,
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
            "learned_typewell_prior_feature_summary": (
                f"{OUTPUT_PREFIX}_inference_learned_typewell_prior_feature_summary.csv"
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
