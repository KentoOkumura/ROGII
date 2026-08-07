from __future__ import annotations

import gzip
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
EXP114_ARTIFACTS = (
    Path("experiments")
    / "exp114_spatial_neighbor_prior_signal_audit"
    / "kaggle"
    / "output"
    / "train_v1"
    / "artifacts"
)
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
EXP114_SPATIAL_OOF = "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
EXP114_SPATIAL_SUMMARY = "exp114_spatial_neighbor_prior_signal_audit_summary.json"
OUTPUT_PREFIX = "exp164_spatial_prior_confidence_features_on_exp092_kaggle"
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
        prefix_tail = known.tail(min(40, len(known))).copy()
        prefix_u0 = float(anchor["TVT_input"]) + float(anchor["Z"])
        prefix_u = (
            pd.to_numeric(prefix_tail["TVT_input"], errors="coerce").to_numpy(np.float64)
            + pd.to_numeric(prefix_tail["Z"], errors="coerce").to_numpy(np.float64)
            - prefix_u0
        )
        prefix_md = pd.to_numeric(prefix_tail["MD"], errors="coerce").to_numpy(np.float64)
        prefix_md = prefix_md - float(anchor["MD"])
        finite = np.isfinite(prefix_md) & np.isfinite(prefix_u)
        prefix_u_slope = 0.0
        prefix_u_roughness = 0.0
        if finite.sum() >= 2 and np.ptp(prefix_md[finite]) > 1e-6:
            coef = np.polyfit(prefix_md[finite], prefix_u[finite], deg=1)
            prefix_u_slope = float(coef[0])
            fitted = np.polyval(coef, prefix_md[finite])
            prefix_u_roughness = float(np.median(np.abs(prefix_u[finite] - fitted)))
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "anchor_tvt_true": float(anchor["TVT"]),
                "known_prefix_rows": int(len(known)),
                "prefix_u_slope_per_md": prefix_u_slope,
                "prefix_u_roughness": prefix_u_roughness,
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
                "prefix_u_slope_per_md": 0.0,
                "prefix_u_roughness": 0.0,
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
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("U-projection feature frame contains non-finite values")
    return result, group_columns, pd.DataFrame(summary_rows)


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    ranks = series.rank(method="average", pct=True).to_numpy(np.float32)
    return np.nan_to_num(ranks, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32) / max(float(temperature), 1e-6)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(np.clip(x, -60.0, 60.0)).astype(np.float32)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-6)


def _candidate_shape_metrics(
    frame: pd.DataFrame,
    values: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = np.zeros(len(frame), dtype=np.float32)
    curvature = np.zeros(len(frame), dtype=np.float32)
    roughness = np.zeros(len(frame), dtype=np.float32)
    for _, idx in frame.groupby("well", sort=False).indices.items():
        idx_array = np.asarray(idx, dtype=np.int64)
        order = np.argsort(s[idx_array], kind="mergesort")
        ordered_idx = idx_array[order]
        x = s[ordered_idx].astype(np.float64)
        y = values[ordered_idx].astype(np.float64)
        if len(ordered_idx) < 2 or np.ptp(x) <= 1e-8:
            continue
        x_grad = x
        if np.any(np.diff(x_grad) <= 1e-8):
            x_grad = np.linspace(0.0, 1.0, num=len(ordered_idx), dtype=np.float64)
        local_slope = np.gradient(y, x_grad, edge_order=1).astype(np.float32)
        if len(ordered_idx) >= 3:
            local_curvature = np.gradient(
                local_slope.astype(np.float64),
                x_grad,
                edge_order=1,
            ).astype(np.float32)
        else:
            local_curvature = np.zeros(len(ordered_idx), dtype=np.float32)
        smooth = (
            pd.Series(y)
            .rolling(window=9, center=True, min_periods=3)
            .median()
            .bfill()
            .ffill()
            .to_numpy(np.float32)
        )
        local_roughness = np.abs(y.astype(np.float32) - smooth).astype(np.float32)
        slope[ordered_idx] = local_slope
        curvature[ordered_idx] = local_curvature
        roughness[ordered_idx] = local_roughness
    return slope, curvature, roughness


def _candidate_polynomial_shape_metrics(
    frame: pd.DataFrame,
    values: np.ndarray,
    s: np.ndarray,
    *,
    degree: int,
    robust_iters: int,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poly = np.zeros(len(frame), dtype=np.float32)
    residual = np.zeros(len(frame), dtype=np.float32)
    abs_residual = np.zeros(len(frame), dtype=np.float32)
    slope = np.zeros(len(frame), dtype=np.float32)
    curvature = np.zeros(len(frame), dtype=np.float32)
    for _, idx in frame.groupby("well", sort=False).indices.items():
        idx_array = np.asarray(idx, dtype=np.int64)
        pred, deriv1, deriv2, _ = _weighted_polyfit_predict(
            s[idx_array],
            values[idx_array],
            degree=degree,
            robust_iters=robust_iters,
            clip_sigma=clip_sigma,
        )
        poly[idx_array] = pred
        residual[idx_array] = (values[idx_array] - pred).astype(np.float32)
        abs_residual[idx_array] = np.abs(residual[idx_array]).astype(np.float32)
        slope[idx_array] = deriv1
        curvature[idx_array] = deriv2
    return poly, residual, abs_residual, slope, curvature


def read_spatial_prior_oof(
    spatial_oof_path: str | Path | None = None,
    *,
    variants: list[str] | tuple[str, ...],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP114_SPATIAL_OOF,
        spatial_oof_path,
        local_artifacts=EXP114_ARTIFACTS,
    )
    usecols = ["id", "well"]
    for variant in variants:
        usecols.extend(
            [
                f"{variant}_prior_delta",
                f"{variant}_prior_tvt",
                f"{variant}_prior_std",
                f"{variant}_prior_count",
                f"{variant}_neighbor_wells",
                f"{variant}_distance_mean",
                f"{variant}_same_typewell_share",
                f"{variant}_azimuth_mismatch",
                f"{variant}_dz_dmd_mismatch",
            ]
        )
    usecols = list(dict.fromkeys(usecols))
    frame = pd.read_csv(source, usecols=usecols, nrows=max_rows, dtype={"id": str, "well": str})
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame[["id", "well"]].duplicated().any():
        raise ValueError("exp114 spatial prior OOF contains duplicate id/well rows")

    summary_path: Path | None = None
    try:
        summary_path = find_artifact(
            EXP114_SPATIAL_SUMMARY,
            local_artifacts=EXP114_ARTIFACTS,
        )
    except FileNotFoundError:
        summary_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_gzip_decompressed(source),
        "source_experiment": "exp114_spatial_neighbor_prior_signal_audit",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "variants": list(variants),
        "columns": usecols,
        "summary": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
    }
    return frame, metadata


def build_spatial_prior_confidence_features(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "spc_")
    variants = [str(value) for value in config.get("prior_variants") or []]
    if not variants:
        variants = ["xy_only_k8", "xy_plus_trajectory_shape_k8"]
    reference_specs = {
        str(name): dict(spec)
        for name, spec in (config.get("reference_candidates") or {}).items()
        if dict(spec).get("enabled", True)
    }
    if not reference_specs:
        raise ValueError("spatial prior confidence features require reference candidates")

    min_md_scale = float(config.get("min_md_scale", 25.0))
    min_prior_scale = float(config.get("min_prior_scale", 10.0))
    near_md_threshold = float(config.get("near_md_threshold", 50.0))
    longtail_rank_threshold = int(config.get("longtail_rank_threshold", 1000))
    primary_variant = str(config.get("primary_prior_variant") or variants[0])
    if primary_variant not in variants:
        raise ValueError(f"primary prior variant is not configured: {primary_variant}")

    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {
        "spatial_prior_geometry": [],
        "spatial_prior_value": [],
        "spatial_prior_quality": [],
        "spatial_prior_disagreement": [],
        "spatial_prior_interaction": [],
    }
    features = frame[key_cols].copy()
    spatial, spatial_meta = read_spatial_prior_oof(
        config.get("spatial_oof_path"),
        variants=variants,
    )
    work = frame.merge(spatial, on=["id", "well"], how="inner", validate="one_to_one")
    if len(work) != len(frame):
        raise ValueError(
            f"spatial prior OOF merge lost rows: base={len(frame)} merged={len(work)}"
        )

    md_since_source = frame.get("md_since")
    if md_since_source is None:
        md_since = np.maximum(_tail_rank(frame["id"]).astype(np.float32), 0.0)
    else:
        md_since = pd.to_numeric(md_since_source, errors="coerce").to_numpy(np.float32)
        md_since = np.nan_to_num(md_since, nan=0.0, posinf=0.0, neginf=0.0)
        md_since = np.maximum(md_since, 0.0)
    tail_md_scale = (
        pd.DataFrame({"well": frame["well"], "md_since": md_since})
        .groupby("well")["md_since"]
        .transform("max")
        .to_numpy(np.float32)
    )
    tail_md_scale = np.maximum(tail_md_scale, min_md_scale).astype(np.float32)
    s_norm = (md_since / tail_md_scale).astype(np.float32)
    tail_rank = _tail_rank(frame["id"]).astype(np.float32)
    tail_rank_scale = (
        pd.DataFrame({"well": frame["well"], "tail_rank": tail_rank})
        .groupby("well")["tail_rank"]
        .transform("max")
        .to_numpy(np.float32)
    )
    tail_rank_scale = np.maximum(tail_rank_scale, float(longtail_rank_threshold)).astype(
        np.float32
    )
    tail_rank_norm = (tail_rank / tail_rank_scale).astype(np.float32)

    last_known = work["last_known_tvt"].to_numpy(np.float32)
    reference_tvt: dict[str, np.ndarray] = {}
    for name, spec in reference_specs.items():
        tvt = _source_tvt(work, name, spec)
        reference_tvt[name] = tvt.astype(np.float32)

    reference_matrix = np.column_stack(list(reference_tvt.values())).astype(np.float32)
    prior_values: dict[str, np.ndarray] = {}
    prior_delta_norm: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    scale_inputs = [
        np.full(len(work), min_prior_scale, dtype=np.float32),
        np.max(np.abs(reference_matrix - last_known[:, None]), axis=1).astype(np.float32),
        np.max(reference_matrix, axis=1) - np.min(reference_matrix, axis=1),
    ]
    for variant in variants:
        prior = work[f"{variant}_prior_tvt"].to_numpy(np.float32)
        prior_values[variant] = prior
        valid_masks[variant] = np.isfinite(prior)
        scale_inputs.append(
            np.nan_to_num(np.abs(prior - last_known), nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
        )
        scale_inputs.append(
            np.nan_to_num(work[f"{variant}_prior_std"].to_numpy(np.float32), nan=0.0)
        )
    scale_source = np.maximum.reduce(scale_inputs)
    prior_scale = (
        pd.DataFrame({"well": work["well"], "scale": scale_source})
        .groupby("well")["scale"]
        .transform(lambda value: float(np.nanquantile(value, 0.90)))
        .to_numpy(np.float32)
    )
    prior_scale = np.maximum(prior_scale, min_prior_scale).astype(np.float32)

    geometry_features = {
        f"{prefix}md_since_norm": s_norm,
        f"{prefix}tail_rank_norm": tail_rank_norm,
        f"{prefix}near_050_flag": (md_since <= near_md_threshold).astype(np.float32),
        f"{prefix}longtail_1000_flag": (tail_rank >= longtail_rank_threshold).astype(np.float32),
    }
    for col, values in geometry_features.items():
        features[col] = np.asarray(values, dtype=np.float32)
    group_columns["spatial_prior_geometry"].extend(geometry_features)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        prior = prior_values[variant]
        valid = valid_masks[variant]
        prior_delta = np.where(valid, prior - last_known, 0.0).astype(np.float32)
        prior_delta_norm[variant] = (prior_delta / prior_scale).astype(np.float32)
        std_norm = (
            np.nan_to_num(work[f"{variant}_prior_std"].to_numpy(np.float32), nan=0.0)
            / prior_scale
        ).astype(np.float32)
        count = np.nan_to_num(work[f"{variant}_prior_count"].to_numpy(np.float32), nan=0.0)
        neighbor_wells = np.nan_to_num(
            work[f"{variant}_neighbor_wells"].to_numpy(np.float32),
            nan=0.0,
        )
        distance_mean = np.nan_to_num(
            work[f"{variant}_distance_mean"].to_numpy(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        value_features = {
            f"{prefix}{variant}_prior_delta_norm": prior_delta_norm[variant],
            f"{prefix}{variant}_prior_abs_delta_norm": np.abs(
                prior_delta_norm[variant]
            ).astype(np.float32),
        }
        if "likpf_mean" in reference_tvt:
            value_features[f"{prefix}{variant}_prior_minus_likpf_mean_norm"] = np.nan_to_num(
                (prior - reference_tvt["likpf_mean"]) / prior_scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        if "beam_mean" in reference_tvt:
            value_features[f"{prefix}{variant}_prior_minus_beam_mean_norm"] = np.nan_to_num(
                (prior - reference_tvt["beam_mean"]) / prior_scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        if "pf_ancc" in reference_tvt:
            value_features[f"{prefix}{variant}_prior_minus_pf_ancc_norm"] = np.nan_to_num(
                (prior - reference_tvt["pf_ancc"]) / prior_scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        quality_features = {
            f"{prefix}{variant}_prior_valid_flag": valid.astype(np.float32),
            f"{prefix}{variant}_prior_std_norm": std_norm,
            f"{prefix}{variant}_prior_count_norm": (count / np.maximum(count.max(), 1.0)).astype(
                np.float32
            ),
            f"{prefix}{variant}_neighbor_wells_norm": (
                neighbor_wells / np.maximum(neighbor_wells.max(), 1.0)
            ).astype(np.float32),
            f"{prefix}{variant}_distance_mean_rank": _rank01(distance_mean),
            f"{prefix}{variant}_same_typewell_share": np.nan_to_num(
                work[f"{variant}_same_typewell_share"].to_numpy(np.float32),
                nan=0.0,
            ).astype(np.float32),
            f"{prefix}{variant}_azimuth_mismatch_rank": _rank01(
                np.nan_to_num(work[f"{variant}_azimuth_mismatch"].to_numpy(np.float32), nan=0.0)
            ),
            f"{prefix}{variant}_dz_dmd_mismatch_rank": _rank01(
                np.nan_to_num(work[f"{variant}_dz_dmd_mismatch"].to_numpy(np.float32), nan=0.0)
            ),
        }
        for col, values in value_features.items():
            features[col] = np.asarray(values, dtype=np.float32)
        for col, values in quality_features.items():
            features[col] = np.asarray(values, dtype=np.float32)
        group_columns["spatial_prior_value"].extend(value_features)
        group_columns["spatial_prior_quality"].extend(quality_features)
        summary_rows.append(
            {
                "candidate": variant,
                "kind": "spatial_prior",
                "valid_rate": float(valid.mean()),
                "abs_delta_norm_mean": float(np.mean(np.abs(prior_delta_norm[variant]))),
                "std_norm_mean": float(np.mean(std_norm)),
                "neighbor_wells_mean": float(np.mean(neighbor_wells)),
                "distance_mean_mean": float(np.mean(distance_mean)),
            }
        )

    prior_norm_matrix = np.column_stack([prior_delta_norm[name] for name in variants]).astype(
        np.float32
    )
    disagreement_features: dict[str, np.ndarray] = {
        f"{prefix}spatial_prior_delta_std_norm": np.std(prior_norm_matrix, axis=1).astype(
            np.float32
        ),
        f"{prefix}spatial_prior_delta_range_norm": (
            np.max(prior_norm_matrix, axis=1) - np.min(prior_norm_matrix, axis=1)
        ).astype(np.float32),
    }
    if len(variants) >= 2:
        left, right = variants[0], variants[1]
        diff = np.nan_to_num(
            (prior_values[left] - prior_values[right]) / prior_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        disagreement_features[f"{prefix}abs_{left}_minus_{right}_norm"] = np.abs(diff).astype(
            np.float32
        )
    primary = prior_values[primary_variant]
    primary_minus_likpf = (
        np.nan_to_num(
            (primary - reference_tvt["likpf_mean"]) / prior_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if "likpf_mean" in reference_tvt
        else np.zeros(len(work), dtype=np.float32)
    ).astype(np.float32)
    primary_std_norm = features[f"{prefix}{primary_variant}_prior_std_norm"].to_numpy(np.float32)
    primary_distance_rank = features[f"{prefix}{primary_variant}_distance_mean_rank"].to_numpy(
        np.float32
    )
    high_disagreement_proxy = (
        0.35 * _rank01(np.abs(primary_minus_likpf))
        + 0.25 * _rank01(primary_std_norm)
        + 0.20 * primary_distance_rank
        + 0.20 * np.clip(s_norm, 0.0, 1.0)
    ).astype(np.float32)
    disagreement_features[f"{prefix}primary_abs_minus_likpf_mean_norm"] = np.abs(
        primary_minus_likpf
    ).astype(np.float32)
    disagreement_features[f"{prefix}high_disagreement_proxy"] = high_disagreement_proxy
    for col, values in disagreement_features.items():
        features[col] = np.asarray(values, dtype=np.float32)
    group_columns["spatial_prior_disagreement"].extend(disagreement_features)

    gate_cfg = dict(config.get("exp118_best_gate") or {})
    gate_variant = str(gate_cfg.get("variant") or primary_variant)
    if gate_variant in variants:
        std = np.nan_to_num(work[f"{gate_variant}_prior_std"].to_numpy(np.float32), nan=np.inf)
        distance = np.nan_to_num(
            work[f"{gate_variant}_distance_mean"].to_numpy(np.float32),
            nan=np.inf,
        )
        finite_std = std[np.isfinite(std)]
        finite_distance = distance[np.isfinite(distance)]
        std_threshold = (
            float(np.quantile(finite_std, float(gate_cfg.get("std_quantile", 0.50))))
            if len(finite_std)
            else np.inf
        )
        distance_threshold = (
            float(np.quantile(finite_distance, float(gate_cfg.get("distance_quantile", 0.50))))
            if len(finite_distance)
            else np.inf
        )
        gate_flag = ((std <= std_threshold) & (distance <= distance_threshold)).astype(np.float32)
        alpha = float(gate_cfg.get("alpha", 0.05))
        clip_ft = float(gate_cfg.get("clip_ft", 5.0))
        gate_delta = (
            prior_values[gate_variant]
            - (reference_tvt["likpf_mean"] if "likpf_mean" in reference_tvt else last_known)
        ).astype(np.float32)
        correction_proxy = np.nan_to_num(
            gate_flag * alpha * np.clip(gate_delta, -clip_ft, clip_ft),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        interaction_features = {
            f"{prefix}exp118_gate_flag": gate_flag,
            f"{prefix}exp118_correction_proxy_ft": correction_proxy,
            f"{prefix}exp118_abs_correction_proxy_norm": (
                np.abs(correction_proxy) / prior_scale
            ).astype(np.float32),
            f"{prefix}high_disagreement_x_longtail": (
                high_disagreement_proxy * geometry_features[f"{prefix}longtail_1000_flag"]
            ).astype(np.float32),
            f"{prefix}primary_minus_likpf_x_md_since_norm": (
                primary_minus_likpf * s_norm
            ).astype(np.float32),
            f"{prefix}primary_minus_likpf_x_high_disagreement": (
                primary_minus_likpf * high_disagreement_proxy
            ).astype(np.float32),
        }
        for col, values in interaction_features.items():
            features[col] = np.asarray(values, dtype=np.float32)
        group_columns["spatial_prior_interaction"].extend(interaction_features)
        summary_rows.append(
            {
                "candidate": "__exp118_gate_proxy__",
                "kind": "gate_proxy",
                "gate_variant": gate_variant,
                "gate_rate": float(gate_flag.mean()),
                "std_threshold": std_threshold,
                "distance_threshold": distance_threshold,
                "correction_proxy_abs_max": float(np.max(np.abs(correction_proxy))),
            }
        )

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[feature_cols].to_numpy(np.float32)).all():
        bad = [col for col in feature_cols if not np.isfinite(features[col].to_numpy()).all()]
        raise ValueError(
            "spatial prior confidence feature frame contains non-finite values: "
            f"{bad}"
        )

    summary_rows.append(
        {
            "candidate": "__aggregate__",
            "kind": "aggregate",
            "generated_features": int(len(feature_cols)),
            "rows": int(len(features)),
            "wells": int(features["well"].nunique()),
            "spatial_prior_delta_std_norm_mean": float(
                features[f"{prefix}spatial_prior_delta_std_norm"].mean()
            ),
            "high_disagreement_proxy_mean": float(
                features[f"{prefix}high_disagreement_proxy"].mean()
            ),
            "spatial_oof_source": spatial_meta["source"],
            "spatial_oof_decompressed_sha256": spatial_meta["source_decompressed_sha256"],
        }
    )
    return features, group_columns, pd.DataFrame(summary_rows)


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
    selected_lgb_configs: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
    all_configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    selected_config_names = list(selected_lgb_configs or [])
    if selected_config_names:
        valid_config_names = {f"lgb{index}" for index in range(len(all_configs))}
        invalid_config_names = sorted(set(selected_config_names) - valid_config_names)
        if invalid_config_names:
            raise ValueError(
                f"Unknown LightGBM config names: {invalid_config_names}; "
                f"valid values are {sorted(valid_config_names)}"
            )
        config_items = [
            (index, params)
            for index, params in enumerate(all_configs)
            if f"lgb{index}" in selected_config_names
        ]
    else:
        config_items = list(enumerate(all_configs))
    configs = [params for _, params in config_items]
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
                "selected_lgb_configs": [f"lgb{index}" for index, _ in config_items],
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in config_items:
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            model_name = f"lgb{model_index}"
            model_file = f"{mode_name}__{model_name}__fold{fold}.txt"
            model_path = model_dir / model_file
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": model_name,
                        "fold": int(fold),
                        "event": "start_fold",
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
            model_sha = None
            if save_models:
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_row = {
                "variant": variant_name,
                "mode": mode_name,
                "model": model_name,
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
                    label=f"{variant_name}/{mode_name}/{model_name}/fold{fold}/tvt",
                ),
                "model_file": model_file if save_models else None,
                "model_sha256": model_sha,
            }
            metric_rows.append(metric_row)
            fold_importance_rows: list[dict[str, Any]] = []
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=False,
            ):
                fold_importance_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": model_name,
                        "fold": int(fold),
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
            importance_rows.extend(fold_importance_rows)
            if save_models:
                model_rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode_name,
                        "model": model_name,
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
                        "model": model_name,
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
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well"].to_numpy(),
                    "variant": variant_name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
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
        "selected_lgb_configs": [f"lgb{index}" for index, _ in config_items],
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


def run_spatial_prior_confidence_features_on_exp092(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    spatial_config: dict[str, Any] | None = None,
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
    selected_lgb_configs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)

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
    spatial_features, spatial_group_columns, spatial_summary = (
        build_spatial_prior_confidence_features(
            full_frame,
            spatial_config or {},
        )
    )
    spatial_feature_columns = [
        col for col in spatial_features.columns if col not in {"id", "well"}
    ]
    full_frame = full_frame.merge(
        spatial_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if full_frame.empty:
        raise ValueError("No rows after merging spatial prior confidence features")
    spatial_meta = {
        "rows": int(len(spatial_features)),
        "wells": int(spatial_features["well"].nunique()),
        "generated_features": int(len(spatial_feature_columns)),
        "feature_columns": spatial_feature_columns,
    }
    feature_group_columns = {
        **projection_group_columns,
        **spatial_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    spatial_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_spatial_feature_summary.csv",
        index=False,
    )

    selected_variants = [
        dict(variant) for variant in variants or [] if variant.get("enabled", True)
    ]
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
                    "is_spatial_prior_confidence_feature": bool(
                        feature in spatial_feature_columns
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
                selected_lgb_configs=selected_lgb_configs,
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
        "experiment": "exp164_spatial_prior_confidence_features_on_exp092_kaggle",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "spatial_prior_confidence_features_on_exp092_surface",
        "feature_source": feature_meta,
        "spatial_feature_source": spatial_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "spatial_config": spatial_config or {},
        "projection_feature_groups": projection_group_columns,
        "spatial_feature_groups": spatial_group_columns,
        "n_splits": int(n_splits),
        "variants": selected_variants,
        "selected_lgb_configs": list(selected_lgb_configs or []),
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    lgb_mean = pooled[pooled["model"].eq("lgb_mean")].sort_values("rmse_tvt")
    best = lgb_mean.iloc[0].to_dict() if not lgb_mean.empty else None
    summary = {
        "experiment": "exp164_spatial_prior_confidence_features_on_exp092_kaggle",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "spatial_prior_confidence_features_on_exp092_surface",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "spatial_feature_source": spatial_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "selected_lgb_configs": list(selected_lgb_configs or []),
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "spatial_feature_summary": f"{OUTPUT_PREFIX}_spatial_feature_summary.csv",
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
    projection_config: dict[str, Any] | None = None,
    spatial_config: dict[str, Any] | None = None,
    variant_name: str = "u_projection_correction_plus_disagreement",
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
        "exp164 is a train-side spatial prior confidence feature audit only. "
        "Inference requires separate raw-test/full-train spatial prior feature parity work."
    )
