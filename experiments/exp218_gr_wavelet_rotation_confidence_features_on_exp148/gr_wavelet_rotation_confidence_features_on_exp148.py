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
EXP148_OOF_PREDICTIONS = "exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz"
OUTPUT_PREFIX = "exp218_gr_wavelet_rotation_confidence_features_on_exp148"
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196


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


def _fill_numeric(values: pd.Series | np.ndarray, fallback: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def _savgol_or_rolling_mean(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if len(values) < window or window <= int(polyorder):
        effective = max(3, min(len(values), window))
        return _rolling_mean(values, effective), {
            "effective_kind": "rolling_mean_short_series",
            "window": int(effective),
        }
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(
                values,
                window_length=window,
                polyorder=int(polyorder),
                mode="interp",
            ).astype(np.float32),
            {"effective_kind": "savgol", "window": window, "polyorder": int(polyorder)},
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle image packages.
        return _rolling_mean(values, window), {
            "effective_kind": "rolling_mean_fallback",
            "window": window,
            "polyorder": int(polyorder),
            "fallback_reason": type(exc).__name__,
        }


def _build_gr_filters(
    values: np.ndarray,
    filters_config: list[dict[str, Any]],
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    filters: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for spec in filters_config:
        name = str(spec["name"])
        kind = str(spec.get("kind", "raw"))
        if kind == "raw":
            filtered = values.astype(np.float32)
            metadata = {"effective_kind": "raw"}
        elif kind == "rolling_median":
            window = int(spec.get("window", 11))
            filtered = _rolling_median(values, window)
            metadata = {"effective_kind": "rolling_median", "window": window}
        elif kind == "rolling_mean":
            window = int(spec.get("window", 21))
            filtered = _rolling_mean(values, window)
            metadata = {"effective_kind": "rolling_mean", "window": window}
        elif kind == "savgol":
            filtered, metadata = _savgol_or_rolling_mean(
                values,
                window=int(spec.get("window", 31)),
                polyorder=int(spec.get("polyorder", 2)),
            )
        else:
            raise ValueError(f"Unknown GR filter kind: {kind}")
        filters.append((name, filtered.astype(np.float32), metadata))
    return filters


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _rolling_absmean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.abs(values))
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    left_s = pd.Series(left, dtype="float64")
    right_s = pd.Series(right, dtype="float64")
    corr = left_s.rolling(int(window), center=True, min_periods=3).corr(right_s)
    return corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def _wavelet_approximation(
    values: np.ndarray,
    *,
    wavelet_name: str,
    level: int,
    fallback_window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values64 = np.asarray(values, dtype=np.float64)
    try:
        import pywt

        wavelet = pywt.Wavelet(str(wavelet_name))
        max_level = int(pywt.dwt_max_level(len(values64), wavelet.dec_len))
        used_level = int(min(max(int(level), 1), max_level))
        if used_level < 1:
            raise ValueError("series too short for configured wavelet")
        coeffs = pywt.wavedec(values64, wavelet, mode="symmetric", level=used_level)
        approx_coeffs = [coeffs[0], *[np.zeros_like(item) for item in coeffs[1:]]]
        approx = pywt.waverec(approx_coeffs, wavelet, mode="symmetric")[: len(values64)]
        return approx.astype(np.float32), {
            "effective_kind": "pywt_dwt_approximation",
            "wavelet": str(wavelet_name),
            "level": used_level,
            "max_level": max_level,
        }
    except Exception as exc:  # pragma: no cover - depends on Kaggle image packages.
        window = int(fallback_window)
        return _rolling_mean(values, window), {
            "effective_kind": "rolling_mean_wavelet_fallback",
            "wavelet": str(wavelet_name),
            "level": int(level),
            "window": window,
            "fallback_reason": type(exc).__name__,
        }


def _fft_rotation_summary(
    values: np.ndarray,
    md: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    md64 = np.asarray(md, dtype=np.float64)
    if len(values64) < 8:
        return {
            "fft_dominant_frequency_norm": 0.0,
            "fft_dominant_energy_ratio": 0.0,
            "fft_rotation_energy_ratio": 0.0,
            "fft_high_frequency_ratio": 0.0,
            "fft_notch_residual_energy_ratio": 0.0,
        }
    x = np.arange(len(values64), dtype=np.float64)
    centered = values64 - float(np.nanmean(values64))
    if str(config.get("detrend", "linear")) == "linear" and float(np.nanstd(centered)) > 1e-9:
        slope, intercept = np.polyfit(x, values64, deg=1)
        centered = values64 - (slope * x + intercept)
    spacing = float(np.nanmedian(np.diff(md64))) if len(md64) > 1 else 1.0
    if not np.isfinite(spacing) or abs(spacing) <= 1e-9:
        spacing = 1.0
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=abs(spacing))
    if len(power) <= 1:
        return {
            "fft_dominant_frequency_norm": 0.0,
            "fft_dominant_energy_ratio": 0.0,
            "fft_rotation_energy_ratio": 0.0,
            "fft_high_frequency_ratio": 0.0,
            "fft_notch_residual_energy_ratio": 0.0,
        }
    valid_power = power[1:]
    total = float(np.sum(valid_power))
    if not np.isfinite(total) or total <= 1e-12:
        total = 1e-12
    valid_freqs = freqs[1:]
    nyquist = float(np.max(valid_freqs)) if len(valid_freqs) else 1.0
    if not np.isfinite(nyquist) or nyquist <= 1e-12:
        nyquist = 1.0
    norm_freqs = valid_freqs / nyquist
    dominant_pos = int(np.argmax(valid_power))
    band = config.get("rotation_band_norm", [0.06, 0.35])
    lo, hi = float(band[0]), float(band[1])
    rotation_mask = (norm_freqs >= lo) & (norm_freqs <= hi)
    high_threshold = float(config.get("high_frequency_norm", 0.35))
    high_mask = norm_freqs >= high_threshold
    top_k = max(int(config.get("top_k_notch", 3)), 1)
    top_energy = float(np.sum(np.sort(valid_power)[-top_k:]))
    return {
        "fft_dominant_frequency_norm": float(norm_freqs[dominant_pos]),
        "fft_dominant_energy_ratio": float(valid_power[dominant_pos] / total),
        "fft_rotation_energy_ratio": float(np.sum(valid_power[rotation_mask]) / total),
        "fft_high_frequency_ratio": float(np.sum(valid_power[high_mask]) / total),
        "fft_notch_residual_energy_ratio": float(max(total - top_energy, 0.0) / total),
    }


def _candidate_match_scores(
    *,
    row_idx: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    slope: float,
    offsets: np.ndarray,
    ncc_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_idx = np.asarray(row_idx, dtype=np.int32)
    if row_idx.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    eval_gr = _gather_horizontal(horizontal_gr, row_idx, offsets)
    local_rows = np.clip(row_idx[:, None] + offsets[None, :], 0, len(md) - 1)
    local_md = md[local_rows].astype(np.float32)
    center_md = md[row_idx].astype(np.float32)[:, None]
    local_tvt = candidate_tvt[:, None].astype(np.float32) + float(slope) * (local_md - center_md)
    candidate_gr = _interpolate_typewell(type_tvt, type_gr, local_tvt)
    mae = np.mean(np.abs(candidate_gr - eval_gr), axis=1).astype(np.float32)
    ncc = np.mean(_standardize_rows(candidate_gr) * _standardize_rows(eval_gr), axis=1).astype(
        np.float32
    )
    cost = (mae - float(ncc_weight) * ncc).astype(np.float32)
    return mae, ncc, cost


def _cost_entropy(cost_matrix: np.ndarray) -> np.ndarray:
    if cost_matrix.size == 0 or cost_matrix.shape[1] <= 1:
        return np.zeros(cost_matrix.shape[0], dtype=np.float32)
    scale = np.nanmedian(np.abs(cost_matrix - np.nanmedian(cost_matrix, axis=1, keepdims=True)))
    temperature = float(max(scale, 1.0))
    logits = -cost_matrix.astype(np.float64) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(np.clip(logits, -80.0, 80.0))
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / np.log(cost_matrix.shape[1])
    return entropy.astype(np.float32)


def _rank_of_column(cost_matrix: np.ndarray, column_index: int) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.zeros(cost_matrix.shape[0], dtype=np.float32)
    order = np.argsort(cost_matrix, axis=1)
    ranks = np.empty(cost_matrix.shape[0], dtype=np.float32)
    for row in range(cost_matrix.shape[0]):
        ranks[row] = float(np.flatnonzero(order[row] == int(column_index))[0])
    return (ranks / max(cost_matrix.shape[1] - 1, 1)).astype(np.float32)


def _prefix_slope_prior(
    *,
    md: np.ndarray,
    tvt_input: np.ndarray,
    known_end: int,
    slope_window_rows: int,
    slope_clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    if known_end <= 1:
        raise ValueError("known_end must include at least two prefix rows")
    fit_start = max(0, int(known_end) - int(slope_window_rows))
    fit_md = md[fit_start:known_end].astype(np.float64)
    fit_tvt = tvt_input[fit_start:known_end].astype(np.float64)
    finite = np.isfinite(fit_md) & np.isfinite(fit_tvt)
    if finite.sum() >= 2 and float(np.nanstd(fit_md[finite])) > 1e-6:
        slope, intercept = np.polyfit(fit_md[finite], fit_tvt[finite], deg=1)
    else:
        slope = 1.0
        intercept = float(tvt_input[known_end - 1] - md[known_end - 1])
    unclipped_slope = float(slope)
    lo, hi = slope_clip
    slope = float(np.clip(slope, lo, hi))
    last_md = float(md[known_end - 1])
    last_tvt = float(tvt_input[known_end - 1])
    prior = last_tvt + slope * (md.astype(np.float64) - last_md)
    return prior.astype(np.float32), {
        "known_end": int(known_end),
        "fit_start": int(fit_start),
        "fit_rows": int(finite.sum()),
        "unclipped_slope": unclipped_slope,
        "slope": slope,
        "intercept": float(intercept),
        "last_md": last_md,
        "last_tvt": last_tvt,
    }


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1e-6
    return centered / scale


def _gather_horizontal(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.clip(centers[:, None] + offsets[None, :], 0, len(series) - 1)
    return series[idx].astype(np.float32)


def _interpolate_typewell(
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    candidate_tvt: np.ndarray,
) -> np.ndarray:
    flat = np.interp(
        candidate_tvt.reshape(-1),
        type_tvt.astype(np.float64),
        type_gr.astype(np.float64),
        left=float(type_gr[0]),
        right=float(type_gr[-1]),
    )
    return flat.reshape(candidate_tvt.shape).astype(np.float32)


def _local_minima_positions(cost: np.ndarray) -> list[int]:
    if len(cost) == 0:
        return []
    if len(cost) == 1:
        return [0]
    positions = [0] if cost[0] <= cost[1] else []
    for i in range(1, len(cost) - 1):
        if cost[i] <= cost[i - 1] and cost[i] <= cost[i + 1]:
            positions.append(i)
    if cost[-1] <= cost[-2]:
        positions.append(len(cost) - 1)
    return positions


def _choose_top2_modes(
    cost: np.ndarray,
    shifts: np.ndarray,
    *,
    min_separation_ft: float,
    max_separation_ft: float,
) -> tuple[int, int, bool]:
    order = np.argsort(cost)
    top1 = int(order[0])
    local_positions = _local_minima_positions(cost)
    local_sorted = sorted(local_positions, key=lambda pos: float(cost[pos]))
    for pos in local_sorted:
        separation = abs(float(shifts[pos] - shifts[top1]))
        if min_separation_ft <= separation <= max_separation_ft:
            return top1, int(pos), True
    for pos in order[1:]:
        separation = abs(float(shifts[pos] - shifts[top1]))
        if min_separation_ft <= separation <= max_separation_ft:
            return top1, int(pos), False
    second = int(order[1]) if len(order) > 1 else top1
    return top1, second, False


def _temp_suffix(value: float) -> str:
    return f"t{float(value):g}".replace(".", "p")


def _deterministic_sample_indices(rows: np.ndarray, max_rows: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int32)
    if rows.size <= int(max_rows):
        return rows
    positions = np.linspace(0, rows.size - 1, int(max_rows))
    return rows[np.unique(np.rint(positions).astype(np.int32))].astype(np.int32)


def _scan_matching_surface(
    *,
    row_idx: np.ndarray,
    prior_tvt_all: np.ndarray,
    horizontal_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    shifts: np.ndarray,
    local_offsets: np.ndarray,
    ncc_weight: float,
    posterior_temperatures: list[float],
    min_mode_separation_ft: float,
    max_mode_separation_ft: float,
) -> dict[str, np.ndarray]:
    row_idx = np.asarray(row_idx, dtype=np.int32)
    if row_idx.size == 0:
        return {}
    eval_gr = _gather_horizontal(horizontal_gr, row_idx, local_offsets)
    local_rows = np.clip(row_idx[:, None] + local_offsets[None, :], 0, len(prior_tvt_all) - 1)
    local_prior = prior_tvt_all[local_rows]
    candidate_tvt = local_prior[:, None, :] + shifts[None, :, None]
    candidate_gr = _interpolate_typewell(type_tvt, type_gr, candidate_tvt)
    mae = np.mean(np.abs(candidate_gr - eval_gr[:, None, :]), axis=2)
    eval_norm = _standardize_rows(eval_gr)
    cand_norm = _standardize_rows(candidate_gr)
    ncc = np.mean(cand_norm * eval_norm[:, None, :], axis=2)
    cost = mae - float(ncc_weight) * ncc

    n_rows = row_idx.size
    out: dict[str, np.ndarray] = {
        "top1_shift_ft": np.zeros(n_rows, dtype=np.float32),
        "top2_shift_ft": np.zeros(n_rows, dtype=np.float32),
        "top1_cost": np.zeros(n_rows, dtype=np.float32),
        "top2_cost": np.zeros(n_rows, dtype=np.float32),
        "top2_minus_top1_cost": np.zeros(n_rows, dtype=np.float32),
        "mode_separation_ft": np.zeros(n_rows, dtype=np.float32),
        "top2_is_local_min": np.zeros(n_rows, dtype=np.float32),
        "bimodal_flag": np.zeros(n_rows, dtype=np.float32),
        "commit_top1_tvt": np.zeros(n_rows, dtype=np.float32),
        "commit_top2_tvt": np.zeros(n_rows, dtype=np.float32),
    }
    for temperature in posterior_temperatures:
        suffix = _temp_suffix(float(temperature))
        out[f"posterior_mean_{suffix}_tvt"] = np.zeros(n_rows, dtype=np.float32)
        out[f"posterior_p_top1_{suffix}"] = np.zeros(n_rows, dtype=np.float32)
        out[f"posterior_entropy_{suffix}"] = np.zeros(n_rows, dtype=np.float32)

    prior_center = prior_tvt_all[row_idx].astype(np.float32)
    all_temperature = max(float(posterior_temperatures[0]) if posterior_temperatures else 8.0, 1e-6)
    logits_all = -cost.astype(np.float64) / all_temperature
    logits_all = logits_all - np.max(logits_all, axis=1, keepdims=True)
    probs_all = np.exp(np.clip(logits_all, -80.0, 80.0))
    probs_all = probs_all / np.maximum(probs_all.sum(axis=1, keepdims=True), 1e-12)
    out["shift_entropy_norm"] = (
        -np.sum(probs_all * np.log(probs_all + 1e-12), axis=1) / np.log(max(len(shifts), 2))
    ).astype(np.float32)

    for i in range(n_rows):
        top1_pos, top2_pos, top2_is_local_min = _choose_top2_modes(
            cost[i],
            shifts,
            min_separation_ft=float(min_mode_separation_ft),
            max_separation_ft=float(max_mode_separation_ft),
        )
        top1_shift = float(shifts[top1_pos])
        top2_shift = float(shifts[top2_pos])
        top1_tvt = float(prior_center[i] + top1_shift)
        top2_tvt = float(prior_center[i] + top2_shift)
        separation = abs(top2_tvt - top1_tvt)
        out["top1_shift_ft"][i] = top1_shift
        out["top2_shift_ft"][i] = top2_shift
        out["top1_cost"][i] = float(cost[i, top1_pos])
        out["top2_cost"][i] = float(cost[i, top2_pos])
        out["top2_minus_top1_cost"][i] = float(cost[i, top2_pos] - cost[i, top1_pos])
        out["mode_separation_ft"][i] = float(separation)
        out["top2_is_local_min"][i] = float(top2_is_local_min)
        out["bimodal_flag"][i] = float(
            bool(top2_is_local_min)
            and float(min_mode_separation_ft) <= separation <= float(max_mode_separation_ft)
        )
        out["commit_top1_tvt"][i] = top1_tvt
        out["commit_top2_tvt"][i] = top2_tvt
        for temperature in posterior_temperatures:
            suffix = _temp_suffix(float(temperature))
            logits = -np.asarray([cost[i, top1_pos], cost[i, top2_pos]], dtype=np.float64) / max(
                float(temperature), 1e-6
            )
            logits = logits - float(np.max(logits))
            probs = np.exp(np.clip(logits, -80.0, 80.0))
            probs = probs / (float(probs.sum()) + 1e-12)
            out[f"posterior_mean_{suffix}_tvt"][i] = float(probs[0] * top1_tvt + probs[1] * top2_tvt)
            out[f"posterior_p_top1_{suffix}"][i] = float(probs[0])
            out[f"posterior_entropy_{suffix}"][i] = -float(
                np.sum(probs * np.log(probs + 1e-12)) / np.log(2.0)
            )
    return out


def _interp_to_rows(
    query_rows: np.ndarray,
    scan_rows: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if scan_rows.size == 0:
        return np.zeros(query_rows.size, dtype=np.float32)
    if scan_rows.size == 1:
        return np.full(query_rows.size, float(values[0]), dtype=np.float32)
    order = np.argsort(scan_rows)
    return np.interp(
        query_rows.astype(np.float64),
        scan_rows[order].astype(np.float64),
        values[order].astype(np.float64),
    ).astype(np.float32)


def _build_prefix_backtest_summary(
    *,
    known_end: int,
    md: np.ndarray,
    tvt_input: np.ndarray,
    filters: list[tuple[str, np.ndarray, dict[str, Any]]],
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    config: dict[str, Any],
    shifts: np.ndarray,
    local_offsets: np.ndarray,
    slope_clip: tuple[float, float],
    posterior_temperatures: list[float],
) -> dict[str, float]:
    backtest_config = dict(config.get("prefix_backtest") or {})
    if not bool(backtest_config.get("enabled", True)):
        return {}
    tail_rows = int(backtest_config.get("tail_rows", 256))
    max_rows = int(backtest_config.get("max_rows_per_well", 128))
    min_prefix = int(config.get("prefix_slope_window_rows", 80))
    backtest_start = max(min_prefix, int(known_end) - tail_rows)
    if backtest_start >= known_end - 1:
        return {}
    rows = _deterministic_sample_indices(np.arange(backtest_start, known_end, dtype=np.int32), max_rows)
    prior, _meta = _prefix_slope_prior(
        md=md,
        tvt_input=tvt_input,
        known_end=backtest_start,
        slope_window_rows=int(config.get("prefix_slope_window_rows", 80)),
        slope_clip=slope_clip,
    )
    truth = tvt_input[rows].astype(np.float32)
    summary: dict[str, float] = {}
    for filter_name, filtered_gr, _metadata in filters:
        scan = _scan_matching_surface(
            row_idx=rows,
            prior_tvt_all=prior,
            horizontal_gr=filtered_gr,
            type_tvt=type_tvt,
            type_gr=type_gr,
            shifts=shifts,
            local_offsets=local_offsets,
            ncc_weight=float(config.get("ncc_weight", 8.0)),
            posterior_temperatures=posterior_temperatures,
            min_mode_separation_ft=float(config.get("min_mode_separation_ft", 6.0)),
            max_mode_separation_ft=float(config.get("max_mode_separation_ft", 30.0)),
        )
        if not scan:
            continue
        top1_error = scan["commit_top1_tvt"] - truth
        base = f"{filter_name}_prefix_backtest"
        summary[f"{base}_top1_mae"] = float(np.mean(np.abs(top1_error)))
        summary[f"{base}_top1_within10"] = float(np.mean(np.abs(top1_error) <= 10.0))
        summary[f"{base}_gap_mean"] = float(np.mean(scan["top2_minus_top1_cost"]))
        summary[f"{base}_shift_entropy_mean"] = float(np.mean(scan["shift_entropy_norm"]))
        for temperature in posterior_temperatures:
            suffix = _temp_suffix(float(temperature))
            error = scan[f"posterior_mean_{suffix}_tvt"] - truth
            summary[f"{base}_posterior_{suffix}_mae"] = float(np.mean(np.abs(error)))
    return summary


def build_gr_wavelet_rotation_confidence_features(
    base_frame: pd.DataFrame,
    train_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    config = config or {}
    prefix = str(config.get("prefix") or "grwr_")
    group_name = str(config.get("feature_group") or "gr_wavelet_rotation_confidence")
    key_cols = ["id", "well"]
    row_indices = _row_indices_from_ids(base_frame["id"])
    result = base_frame[key_cols].copy().reset_index(drop=True)
    group_columns: dict[str, list[str]] = {group_name: []}

    candidate_specs = [dict(item) for item in config.get("candidate_tvt_specs") or []]
    available_candidates: list[tuple[str, np.ndarray]] = []
    for spec in candidate_specs:
        if str(spec.get("column")) in base_frame.columns:
            available_candidates.append((str(spec["name"]), _candidate_tvt(base_frame, spec)))
    candidate_names = [name for name, _values in available_candidates]
    last_tvt = base_frame["last_known_tvt"].to_numpy(np.float32)
    md_since = np.maximum(_numeric_array(base_frame, "md_since"), np.float32(0.0))

    train_dir = Path(train_dir)
    slope_clip_config = config.get("slope_clip", [-3.0, 3.0])
    slope_clip = (float(slope_clip_config[0]), float(slope_clip_config[1]))
    local_windows = [int(value) for value in config.get("local_windows_rows", [33, 65, 129])]
    local_offsets = np.asarray(
        [int(value) for value in config.get("candidate_match_offsets_rows", [-24, -12, 0, 12, 24])],
        dtype=np.int32,
    )
    denoise_config = dict(config.get("denoise_filters") or {})
    wavelet_config = dict(config.get("wavelet") or {})
    fft_config = dict(config.get("fft") or {})
    match_names = {str(name) for name in config.get("candidate_match_names") or []}
    match_candidates = [
        (name, values)
        for name, values in available_candidates
        if not match_names or name in match_names
    ]
    default_candidate = str(config.get("default_candidate") or "likpf_mean")
    if default_candidate not in {name for name, _values in match_candidates} and match_candidates:
        default_candidate = match_candidates[0][0]

    feature_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for well, positions_raw in base_frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        sub_rows = row_indices[positions].astype(np.int32)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        typewell_path = train_dir / f"{well}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"missing raw train files for {well}")
        horizontal = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        required_horizontal = {"MD", "GR", "TVT_input"}
        required_typewell = {"TVT", "GR"}
        missing_h = required_horizontal - set(horizontal.columns)
        missing_t = required_typewell - set(typewell.columns)
        if missing_h or missing_t:
            raise ValueError(f"{well} missing columns: horizontal={missing_h}, typewell={missing_t}")

        md = _fill_numeric(horizontal["MD"])
        gr_missing_rate = float(pd.isna(horizontal["GR"]).mean())
        gr = _fill_numeric(horizontal["GR"])
        tvt_input_raw = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known = tvt_input_raw.notna().to_numpy()
        if not known.any():
            raise ValueError(f"No finite TVT_input prefix for {well}")
        known_end = int(np.flatnonzero(known)[-1] + 1)
        if known_end <= 1:
            raise ValueError(f"Too few finite TVT_input prefix rows for {well}: {known_end}")
        tvt_input = _fill_numeric(tvt_input_raw)
        _prior, prior_meta = _prefix_slope_prior(
            md=md,
            tvt_input=tvt_input,
            known_end=known_end,
            slope_window_rows=int(config.get("prefix_slope_window_rows", 80)),
            slope_clip=slope_clip,
        )
        type_sorted = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
        type_tvt = pd.to_numeric(type_sorted["TVT"], errors="coerce").to_numpy(np.float32)
        type_gr = _fill_numeric(type_sorted["GR"])
        type_gr = _rolling_mean(type_gr, int(config.get("typewell_smooth_window", 5)))
        if len(type_tvt) < 4:
            raise ValueError(f"Typewell GR too short for {well}")

        query_rows = np.clip(sub_rows, 0, len(horizontal) - 1).astype(np.int32)
        rolling_median = _rolling_median(
            gr,
            int(denoise_config.get("rolling_median_11", {}).get("window", 11)),
        )
        savgol_spec = denoise_config.get("savgol_31_p2", {})
        savgol, savgol_meta = _savgol_or_rolling_mean(
            gr,
            window=int(savgol_spec.get("window", 31)),
            polyorder=int(savgol_spec.get("polyorder", 2)),
        )
        dwt_approx, wavelet_meta = _wavelet_approximation(
            gr,
            wavelet_name=str(wavelet_config.get("name", "db4")),
            level=int(wavelet_config.get("level", 3)),
            fallback_window=int(wavelet_config.get("fallback_window", 65)),
        )
        dwt_detail = (gr - dwt_approx).astype(np.float32)
        fft_meta = _fft_rotation_summary(gr, md, fft_config)
        denoised = {
            "raw": gr,
            "rolling_median_11": rolling_median,
            "savgol_31_p2": savgol,
            "dwt_approx": dwt_approx,
        }

        columns: dict[str, Any] = {
            "id": base_frame["id"].iloc[positions].to_numpy(),
            "well": np.full(len(positions), str(well), dtype=object),
            f"{prefix}gr_missing_rate": np.full(len(positions), gr_missing_rate, dtype=np.float32),
            f"{prefix}typewell_gr_missing_rate": np.full(
                len(positions),
                float(pd.isna(typewell["GR"]).mean()),
                dtype=np.float32,
            ),
            f"{prefix}known_prefix_rows_log1p": np.full(
                len(positions),
                np.log1p(float(known_end)),
                dtype=np.float32,
            ),
            f"{prefix}known_prefix_fraction": np.full(
                len(positions),
                float(known_end) / max(float(len(horizontal)), 1.0),
                dtype=np.float32,
            ),
        }
        for name, value in fft_meta.items():
            columns[f"{prefix}{name}"] = np.full(len(positions), float(value), dtype=np.float32)

        window_feature_cache: dict[str, np.ndarray] = {}
        for window in local_windows:
            suffix = f"w{int(window):03d}"
            raw_std = _rolling_std(gr, window)
            detail_energy = _rolling_mean(np.square(dwt_detail).astype(np.float32), window)
            raw_energy = _rolling_mean(np.square(gr - _rolling_mean(gr, window)).astype(np.float32), window)
            dwt_ratio = _safe_divide(detail_energy, raw_energy + detail_energy)
            dwt_abs = _rolling_absmean(dwt_detail, window)
            raw_roll_abs = _rolling_absmean(gr - rolling_median, window)
            raw_savgol_abs = _rolling_absmean(gr - savgol, window)
            raw_dwt_abs = _rolling_absmean(gr - dwt_approx, window)
            raw_roll_corr = _rolling_corr(gr, rolling_median, window)
            raw_savgol_corr = _rolling_corr(gr, savgol, window)
            raw_dwt_corr = _rolling_corr(gr, dwt_approx, window)
            columns[f"{prefix}raw_std_{suffix}"] = raw_std[query_rows]
            columns[f"{prefix}dwt_detail_energy_{suffix}"] = detail_energy[query_rows]
            columns[f"{prefix}dwt_detail_absmean_{suffix}"] = dwt_abs[query_rows]
            columns[f"{prefix}dwt_detail_energy_ratio_{suffix}"] = dwt_ratio[query_rows]
            columns[f"{prefix}raw_minus_rolling_absmean_{suffix}"] = raw_roll_abs[query_rows]
            columns[f"{prefix}raw_minus_savgol_absmean_{suffix}"] = raw_savgol_abs[query_rows]
            columns[f"{prefix}raw_minus_dwt_absmean_{suffix}"] = raw_dwt_abs[query_rows]
            columns[f"{prefix}raw_rolling_corr_{suffix}"] = raw_roll_corr[query_rows]
            columns[f"{prefix}raw_savgol_corr_{suffix}"] = raw_savgol_corr[query_rows]
            columns[f"{prefix}raw_dwt_corr_{suffix}"] = raw_dwt_corr[query_rows]
            window_feature_cache[f"dwt_ratio_{suffix}"] = dwt_ratio[query_rows]
            window_feature_cache[f"raw_std_{suffix}"] = raw_std[query_rows]

        candidate_stack = np.empty((0, len(positions)), dtype=np.float32)
        candidate_std = np.zeros(len(positions), dtype=np.float32)
        candidate_range = np.zeros(len(positions), dtype=np.float32)
        if available_candidates:
            candidate_stack = np.vstack([values[positions] for _name, values in available_candidates])
            candidate_std = np.std(candidate_stack, axis=0).astype(np.float32)
            candidate_range = (
                np.max(candidate_stack, axis=0) - np.min(candidate_stack, axis=0)
            ).astype(np.float32)
            columns[f"{prefix}candidate_tvt_std"] = candidate_std
            columns[f"{prefix}candidate_tvt_range"] = candidate_range

        filter_cost_matrices: dict[str, np.ndarray] = {}
        filter_default_cost: dict[str, np.ndarray] = {}
        filter_default_ncc: dict[str, np.ndarray] = {}
        if match_candidates:
            slope = float(prior_meta.get("slope", 1.0))
            default_index = next(
                (idx for idx, (name, _values) in enumerate(match_candidates) if name == default_candidate),
                0,
            )
            zero_index = len(match_candidates)
            for filter_name in ["raw", "rolling_median_11", "savgol_31_p2", "dwt_approx"]:
                costs: list[np.ndarray] = []
                nccs: list[np.ndarray] = []
                for _candidate_name, candidate_values in match_candidates:
                    _mae, ncc, cost = _candidate_match_scores(
                        row_idx=query_rows,
                        md=md,
                        horizontal_gr=denoised[filter_name],
                        type_tvt=type_tvt,
                        type_gr=type_gr,
                        candidate_tvt=candidate_values[positions],
                        slope=slope,
                        offsets=local_offsets,
                        ncc_weight=float(config.get("ncc_weight", 8.0)),
                    )
                    costs.append(cost)
                    nccs.append(ncc)
                _zero_mae, zero_ncc, zero_cost = _candidate_match_scores(
                    row_idx=query_rows,
                    md=md,
                    horizontal_gr=denoised[filter_name],
                    type_tvt=type_tvt,
                    type_gr=type_gr,
                    candidate_tvt=last_tvt[positions],
                    slope=slope,
                    offsets=local_offsets,
                    ncc_weight=float(config.get("ncc_weight", 8.0)),
                )
                cost_matrix = np.vstack([*costs, zero_cost]).T.astype(np.float32)
                ncc_matrix = np.vstack([*nccs, zero_ncc]).T.astype(np.float32)
                filter_cost_matrices[filter_name] = cost_matrix
                filter_default_cost[filter_name] = cost_matrix[:, default_index]
                filter_default_ncc[filter_name] = ncc_matrix[:, default_index]
                columns[f"{prefix}{filter_name}_default_candidate_cost"] = cost_matrix[
                    :,
                    default_index,
                ]
                columns[f"{prefix}{filter_name}_default_candidate_ncc"] = ncc_matrix[
                    :,
                    default_index,
                ]
                columns[f"{prefix}{filter_name}_candidate_cost_entropy"] = _cost_entropy(
                    cost_matrix[:, : len(match_candidates)]
                )
                best_cost = np.min(cost_matrix[:, : len(match_candidates)], axis=1)
                columns[f"{prefix}{filter_name}_best_minus_default_cost"] = (
                    best_cost - cost_matrix[:, default_index]
                ).astype(np.float32)
                columns[f"{prefix}{filter_name}_best_is_default_candidate"] = (
                    np.argmin(cost_matrix[:, : len(match_candidates)], axis=1) == default_index
                ).astype(np.float32)
                columns[f"{prefix}{filter_name}_candidate_cost_std"] = np.std(
                    cost_matrix[:, : len(match_candidates)],
                    axis=1,
                ).astype(np.float32)
                columns[f"{prefix}{filter_name}_zero_rank_norm"] = _rank_of_column(
                    cost_matrix,
                    zero_index,
                )
                columns[f"{prefix}{filter_name}_zero_minus_best_cost"] = (
                    cost_matrix[:, zero_index] - np.min(cost_matrix, axis=1)
                ).astype(np.float32)

            raw_default_ncc = filter_default_ncc.get("raw")
            raw_default_cost = filter_default_cost.get("raw")
            if raw_default_ncc is not None and raw_default_cost is not None:
                for filter_name in ["rolling_median_11", "savgol_31_p2", "dwt_approx"]:
                    if filter_name in filter_default_ncc:
                        columns[f"{prefix}{filter_name}_minus_raw_default_candidate_ncc"] = (
                            filter_default_ncc[filter_name] - raw_default_ncc
                        ).astype(np.float32)
                        columns[f"{prefix}{filter_name}_minus_raw_default_candidate_cost"] = (
                            filter_default_cost[filter_name] - raw_default_cost
                        ).astype(np.float32)

        distance = md_since[positions]
        log_distance = np.log1p(distance).astype(np.float32)
        ratio_065 = window_feature_cache.get(
            "dwt_ratio_w065",
            next(iter(window_feature_cache.values()), np.zeros(len(positions), dtype=np.float32)),
        )
        raw_std_065 = window_feature_cache.get(
            "raw_std_w065",
            np.zeros(len(positions), dtype=np.float32),
        )
        columns[f"{prefix}dwt_energy_ratio_w065_x_candidate_std"] = (
            ratio_065 * candidate_std
        ).astype(np.float32)
        columns[f"{prefix}raw_std_w065_x_log1p_md_since"] = (
            raw_std_065 * log_distance
        ).astype(np.float32)
        columns[f"{prefix}fft_rotation_ratio_x_log1p_md_since"] = (
            float(fft_meta["fft_rotation_energy_ratio"]) * log_distance
        ).astype(np.float32)
        columns[f"{prefix}fft_rotation_ratio_x_candidate_range"] = (
            float(fft_meta["fft_rotation_energy_ratio"]) * candidate_range
        ).astype(np.float32)
        if "dwt_approx" in filter_default_ncc and "raw" in filter_default_ncc:
            dwt_ncc_gap = (
                filter_default_ncc["dwt_approx"] - filter_default_ncc["raw"]
            ).astype(np.float32)
            columns[f"{prefix}dwt_minus_raw_ncc_gap_x_candidate_range"] = (
                dwt_ncc_gap * candidate_range
            ).astype(np.float32)
            columns[f"{prefix}dwt_minus_raw_ncc_gap_x_dwt_energy_ratio_w065"] = (
                dwt_ncc_gap * ratio_065
            ).astype(np.float32)
        if "ll_learned_prob_entropy" in base_frame.columns:
            ll_entropy = base_frame["ll_learned_prob_entropy"].iloc[positions].to_numpy(
                np.float32,
            )
            columns[f"{prefix}ll_entropy_x_dwt_energy_ratio_w065"] = (
                ll_entropy * ratio_065
            ).astype(np.float32)

        well_features = pd.DataFrame(columns)
        summary_rows.append(
            {
                "well": str(well),
                "rows": int(len(positions)),
                "known_prefix_rows": int(known_end),
                "horizontal_rows": int(len(horizontal)),
                "typewell_rows": int(len(typewell)),
                "gr_missing_rate": gr_missing_rate,
                "typewell_gr_missing_rate": float(pd.isna(typewell["GR"]).mean()),
                "wavelet_metadata": json.dumps(_jsonable(wavelet_meta), sort_keys=True),
                "savgol_metadata": json.dumps(_jsonable(savgol_meta), sort_keys=True),
                **fft_meta,
                **{f"prior_{key}": value for key, value in prior_meta.items()},
            }
        )
        feature_frames.append(well_features)

    features = pd.concat(feature_frames, ignore_index=True)
    feature_cols = [col for col in features.columns if col not in {"id", "well"}]
    for col in feature_cols:
        features[col] = (
            pd.to_numeric(features[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    _assert_finite_columns(features, feature_cols, "GR wavelet rotation feature frame")
    if not result[key_cols].equals(features[key_cols]):
        features = result[key_cols].merge(features, on=key_cols, how="left", validate="one_to_one")
        feature_cols = [col for col in features.columns if col not in {"id", "well"}]
        for col in feature_cols:
            features[col] = (
                pd.to_numeric(features[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float32)
            )
    group_columns[group_name] = feature_cols
    summary = pd.DataFrame(summary_rows)
    summary_overview = pd.DataFrame(
        [
            {
                "feature_group": group_name,
                "rows": int(len(features)),
                "wells": int(features["well"].nunique()),
                "generated_features": int(len(feature_cols)),
                "local_windows_rows": ",".join(str(value) for value in local_windows),
                "candidate_match_offsets_rows": ",".join(str(value) for value in local_offsets),
                "available_candidate_tvt_specs": int(len(available_candidates)),
                "matched_candidate_tvt_specs": int(len(match_candidates)),
                "default_candidate": default_candidate,
                "wavelet": str(wavelet_config.get("name", "db4")),
                "wavelet_level": int(wavelet_config.get("level", 3)),
            }
        ]
    )
    summary = pd.concat([summary_overview, summary], ignore_index=True, sort=False)
    meta = {
        "source_kind": "target_free_gr_wavelet_rotation_confidence_features",
        "train_dir": str(train_dir),
        "generated_feature_count": int(len(feature_cols)),
        "generated_feature_columns": feature_cols,
        "local_windows_rows": local_windows,
        "candidate_match_offsets_rows": [int(value) for value in local_offsets],
        "wavelet_config": wavelet_config,
        "fft_config": fft_config,
        "denoise_filters": denoise_config,
        "candidate_names": candidate_names,
        "matched_candidate_names": [name for name, _values in match_candidates],
        "default_candidate": default_candidate,
    }
    return features, group_columns, summary, meta


def build_sp45_bimodal_selector_features(
    base_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    raise NotImplementedError(
        "exp218 does not implement the legacy sp45 inference feature wrapper. "
        "Train-side GRWR features require raw train_dir and are generated by "
        "build_gr_wavelet_rotation_confidence_features()."
    )


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    row_index = np.arange(len(frame))
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
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
                "configs": int(len(configs)),
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in enumerate(configs):
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
        "lgb_configs": configs,
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


def run_gr_wavelet_rotation_confidence_features_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    grwr_feature_config: dict[str, Any] | None = None,
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
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
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
    del learned_features_source, learned_features
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "added_learned_likelihood_features",
                "rows": int(len(full_frame)),
                "learned_features": int(len(learned_feature_columns)),
                "dropped_base_rows": int(coverage_meta["dropped_base_rows"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    grwr_features, grwr_group_columns, grwr_summary, grwr_meta = (
        build_gr_wavelet_rotation_confidence_features(
            full_frame,
            train_dir=train_dir,
            config=grwr_feature_config or {},
        )
    )
    grwr_feature_columns = [col for col in grwr_features.columns if col not in {"id", "well"}]
    if not full_frame["id"].equals(grwr_features["id"]) or not full_frame["well"].equals(
        grwr_features["well"]
    ):
        raise ValueError(
            "GRWR features are not row-order aligned with train feature frame"
        )
    _assign_aligned_float32_columns(
        full_frame,
        grwr_features,
        grwr_feature_columns,
    )
    del grwr_features
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "added_grwr_features",
                "rows": int(len(full_frame)),
                "grwr_features": int(len(grwr_feature_columns)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **grwr_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )
    grwr_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_grwr_feature_summary.csv",
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
                    "is_grwr_feature": bool(feature in grwr_feature_columns),
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
        "experiment": "exp218_gr_wavelet_rotation_confidence_features_on_exp148",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "gr_signal_audit_parents": [
            "exp167_fft_denoised_gr_matching_audit",
            "exp189_denoised_gr_pfbeam_generation_audit",
            "exp216_affine_shift_landscape_ruler_readout",
            "exp214_public_raw_gr_residual_scale_control",
        ],
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "gr_wavelet_rotation_confidence_features_on_exp148_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "grwr_feature_source": grwr_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "grwr_feature_config": grwr_feature_config or {},
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "grwr_feature_groups": grwr_group_columns,
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
        "experiment": "exp218_gr_wavelet_rotation_confidence_features_on_exp148",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "gr_wavelet_rotation_confidence_features_on_exp148_full_train_rows",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "gr_signal_audit_parents": [
            "exp167_fft_denoised_gr_matching_audit",
            "exp189_denoised_gr_pfbeam_generation_audit",
            "exp216_affine_shift_landscape_ruler_readout",
            "exp214_public_raw_gr_residual_scale_control",
        ],
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "grwr_feature_source": grwr_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": enabled_variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "learned_feature_summary": f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
            "grwr_feature_summary": f"{OUTPUT_PREFIX}_grwr_feature_summary.csv",
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
    grwr_feature_config: dict[str, Any] | None = None,
    variant_name: str = "gr_wavelet_rotation_confidence_addonly",
    mode_name: str = "gpu_repro_guard_dp_threads8",
    model_name: str = "lgb_mean",
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

    grwr_features, grwr_group_columns, grwr_summary, grwr_meta = (
        build_gr_wavelet_rotation_confidence_features(
            test_frame,
            train_dir=test_dir,
            config=grwr_feature_config or dict(manifest.get("grwr_feature_config") or {}),
        )
    )
    grwr_feature_columns = [col for col in grwr_features.columns if col not in {"id", "well"}]
    if not test_frame[["id", "well"]].reset_index(drop=True).equals(
        grwr_features[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("Raw-test GRWR features are not row-order aligned with test feature frame")
    test_frame = pd.concat(
        [
            test_frame.reset_index(drop=True),
            grwr_features[grwr_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    configured_grwr_groups = manifest.get("grwr_feature_groups") or {}
    if configured_grwr_groups and {
        key: list(value) for key, value in grwr_group_columns.items()
    } != {key: list(value) for key, value in configured_grwr_groups.items()}:
        raise ValueError("GRWR feature groups differ from train manifest")
    feature_group_columns = {
        **feature_group_columns,
        **grwr_group_columns,
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
    grwr_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_grwr_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                "is_grwr_feature": bool(feature in grwr_feature_columns),
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
        "experiment": "exp218_gr_wavelet_rotation_confidence_features_on_exp148",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "grwr_feature_source": _jsonable(grwr_meta),
        "learned_feature_groups": learned_group_columns,
        "grwr_feature_groups": grwr_group_columns,
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
            "grwr_feature_summary": f"{OUTPUT_PREFIX}_inference_grwr_feature_summary.csv",
            "feature_schema": feature_schema_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
            "submission": str(submission_path),
        },
        "known_followup_risk": "Train-side CV was positive but mid-bucket and some worst-well regressions remain unresolved.",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
