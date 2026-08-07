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
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
OUTPUT_PREFIX = "exp151_tvt_dense_addonly_confidence_features_on_exp092"
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


def build_tvt_dense_confidence_features(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    config = config or {}
    prefix = str(config.get("prefix") or "tdc_")
    dense_specs = {
        str(name): dict(spec)
        for name, spec in (config.get("dense_candidates") or config.get("candidates") or {}).items()
        if dict(spec).get("enabled", True)
    }
    reference_specs = {
        str(name): dict(spec)
        for name, spec in (config.get("reference_candidates") or {}).items()
        if dict(spec).get("enabled", True)
    }
    if len(dense_specs) < 2:
        raise ValueError("tvt_dense confidence features require at least two dense candidates")
    if not reference_specs:
        raise ValueError("tvt_dense confidence features require reference candidates")

    min_md_scale = float(config.get("min_md_scale", 25.0))
    min_dense_scale = float(config.get("min_dense_scale", 10.0))
    near_md_threshold = float(config.get("near_md_threshold", 50.0))
    longtail_rank_threshold = int(config.get("longtail_rank_threshold", 1000))
    primary_name = str(config.get("primary_dense_candidate") or next(iter(dense_specs)))
    if primary_name not in dense_specs:
        raise ValueError(f"primary dense candidate is not configured: {primary_name}")

    key_cols = ["id", "well"]
    group_columns: dict[str, list[str]] = {
        "dense_confidence_geometry": [],
        "dense_candidate_path": [],
        "dense_candidate_disagreement": [],
    }
    features = frame[key_cols].copy()

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

    last_known = frame["last_known_tvt"].to_numpy(np.float32)
    dense_tvt: dict[str, np.ndarray] = {}
    reference_tvt: dict[str, np.ndarray] = {}
    for name, spec in dense_specs.items():
        tvt = _source_tvt(frame, name, spec)
        dense_tvt[name] = tvt.astype(np.float32)
    for name, spec in reference_specs.items():
        tvt = _source_tvt(frame, name, spec)
        reference_tvt[name] = tvt.astype(np.float32)

    dense_matrix = np.column_stack([dense_tvt[name] for name in dense_specs]).astype(np.float32)
    dense_delta_matrix = (dense_matrix - last_known[:, None]).astype(np.float32)
    scale_inputs = [
        np.max(np.abs(dense_delta_matrix), axis=1),
        np.max(dense_matrix, axis=1) - np.min(dense_matrix, axis=1),
        np.full(len(frame), min_dense_scale, dtype=np.float32),
    ]
    for values in reference_tvt.values():
        scale_inputs.append(np.abs(values - dense_tvt[primary_name]).astype(np.float32))
    for optional_col in ["dense_std", "dense_dist", "pf_vs_dense"]:
        if optional_col in frame.columns:
            scale_inputs.append(np.abs(frame[optional_col].to_numpy(np.float32)))
    scale_source = np.maximum.reduce(scale_inputs)
    dense_scale = (
        pd.DataFrame({"well": frame["well"], "scale": scale_source})
        .groupby("well")["scale"]
        .transform(lambda value: float(np.nanquantile(value, 0.90)))
        .to_numpy(np.float32)
    )
    dense_scale = np.maximum(dense_scale, min_dense_scale).astype(np.float32)

    geometry_features = {
        f"{prefix}md_since_norm": s_norm,
        f"{prefix}tail_rank_norm": tail_rank_norm,
        f"{prefix}near_050_flag": (md_since <= near_md_threshold).astype(np.float32),
        f"{prefix}longtail_1000_flag": (tail_rank >= longtail_rank_threshold).astype(np.float32),
    }
    for col, values in geometry_features.items():
        features[col] = np.asarray(values, dtype=np.float32)
    group_columns["dense_confidence_geometry"].extend(geometry_features)

    dense_delta_norm: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    for name in dense_specs:
        drift_norm = ((dense_tvt[name] - last_known) / dense_scale).astype(np.float32)
        dense_delta_norm[name] = drift_norm
        slope, _curvature, roughness = _candidate_shape_metrics(frame, drift_norm, s_norm)
        candidate_features = {
            f"{prefix}{name}_drift_norm": drift_norm,
            f"{prefix}{name}_abs_drift_norm": np.abs(drift_norm).astype(np.float32),
            f"{prefix}{name}_slope_norm": slope.astype(np.float32),
            f"{prefix}{name}_roughness_norm": roughness.astype(np.float32),
        }
        for col, values in candidate_features.items():
            features[col] = np.asarray(values, dtype=np.float32)
        group_columns["dense_candidate_path"].extend(candidate_features)
        summary_rows.append(
            {
                "candidate": name,
                "kind": "dense",
                "drift_abs_norm_mean": float(np.mean(np.abs(drift_norm))),
                "slope_abs_norm_mean": float(np.mean(np.abs(slope))),
                "roughness_norm_mean": float(np.mean(np.abs(roughness))),
            }
        )

    dense_norm_matrix = np.column_stack([dense_delta_norm[name] for name in dense_specs]).astype(
        np.float32
    )
    disagreement_features: dict[str, np.ndarray] = {
        f"{prefix}dense_family_std_norm": np.std(dense_norm_matrix, axis=1).astype(np.float32),
        f"{prefix}dense_family_range_norm": (
            np.max(dense_norm_matrix, axis=1) - np.min(dense_norm_matrix, axis=1)
        ).astype(np.float32),
    }

    primary = dense_tvt[primary_name]
    for name, values in reference_tvt.items():
        diff = ((values - primary) / dense_scale).astype(np.float32)
        disagreement_features[f"{prefix}{name}_minus_{primary_name}_norm"] = diff
        disagreement_features[f"{prefix}abs_{name}_minus_{primary_name}_norm"] = np.abs(
            diff
        ).astype(np.float32)
        summary_rows.append(
            {
                "candidate": name,
                "kind": "reference",
                "diff_vs_primary_abs_norm_mean": float(np.mean(np.abs(diff))),
            }
        )

    pair_specs = [("tvt_densew", "tvt_dense"), ("tvt_dense50", "tvt_densew")]
    for left, right in pair_specs:
        if left in dense_tvt and right in dense_tvt:
            diff = ((dense_tvt[left] - dense_tvt[right]) / dense_scale).astype(np.float32)
            disagreement_features[f"{prefix}abs_{left}_minus_{right}_norm"] = np.abs(
                diff
            ).astype(np.float32)

    dense_std_norm = (
        np.abs(frame["dense_std"].to_numpy(np.float32)) / dense_scale
        if "dense_std" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    pf_vs_dense_abs_norm = (
        np.abs(frame["pf_vs_dense"].to_numpy(np.float32)) / dense_scale
        if "pf_vs_dense" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    high_disagreement_proxy = (
        0.35 * _rank01(np.abs(disagreement_features[f"{prefix}dense_family_range_norm"]))
        + 0.35 * _rank01(pf_vs_dense_abs_norm)
        + 0.20 * _rank01(dense_std_norm)
        + 0.10 * np.clip(s_norm, 0.0, 1.0)
    ).astype(np.float32)
    disagreement_features[f"{prefix}pf_vs_dense_abs_norm"] = pf_vs_dense_abs_norm.astype(
        np.float32
    )
    disagreement_features[f"{prefix}dense_std_norm"] = dense_std_norm.astype(np.float32)
    disagreement_features[f"{prefix}high_disagreement_proxy"] = high_disagreement_proxy
    disagreement_features[f"{prefix}high_disagreement_x_longtail"] = (
        high_disagreement_proxy * geometry_features[f"{prefix}longtail_1000_flag"]
    ).astype(np.float32)

    for col, values in disagreement_features.items():
        features[col] = np.asarray(values, dtype=np.float32)
    group_columns["dense_candidate_disagreement"].extend(disagreement_features)

    feature_cols = [col for col in features.columns if col not in key_cols]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("tvt_dense confidence feature frame contains non-finite values")

    summary_rows.append(
        {
            "candidate": "__aggregate__",
            "kind": "aggregate",
            "generated_features": int(len(feature_cols)),
            "rows": int(len(features)),
            "wells": int(features["well"].nunique()),
            "dense_family_std_norm_mean": float(
                features[f"{prefix}dense_family_std_norm"].mean()
            ),
            "dense_family_range_norm_mean": float(
                features[f"{prefix}dense_family_range_norm"].mean()
            ),
            "high_disagreement_proxy_mean": float(
                features[f"{prefix}high_disagreement_proxy"].mean()
            ),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
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
                model.feature_importances_,
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


def run_tvt_dense_addonly_confidence_features_on_exp092(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    dense_config: dict[str, Any] | None = None,
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
    dense_features, dense_group_columns, dense_summary = (
        build_tvt_dense_confidence_features(
            full_frame,
            dense_config or {},
        )
    )
    dense_feature_columns = [
        col for col in dense_features.columns if col not in {"id", "well"}
    ]
    full_frame = full_frame.merge(
        dense_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if full_frame.empty:
        raise ValueError("No rows after merging tvt_dense confidence features")
    dense_meta = {
        "rows": int(len(dense_features)),
        "wells": int(dense_features["well"].nunique()),
        "generated_features": int(len(dense_feature_columns)),
        "feature_columns": dense_feature_columns,
    }
    feature_group_columns = {
        **projection_group_columns,
        **dense_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    dense_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_dense_feature_summary.csv",
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
                    "is_tvt_dense_confidence_feature": bool(feature in dense_feature_columns),
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
        "experiment": "exp151_tvt_dense_addonly_confidence_features_on_exp092",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "tvt_dense_addonly_confidence_features_on_exp092_surface",
        "feature_source": feature_meta,
        "dense_feature_source": dense_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "dense_config": dense_config or {},
        "projection_feature_groups": projection_group_columns,
        "dense_feature_groups": dense_group_columns,
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
        "experiment": "exp151_tvt_dense_addonly_confidence_features_on_exp092",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "tvt_dense_addonly_confidence_features_on_exp092_surface",
        "parent": "exp092_u_projection_correction_disagreement_fullrun",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "dense_feature_source": dense_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "dense_feature_summary": f"{OUTPUT_PREFIX}_dense_feature_summary.csv",
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
    dense_config: dict[str, Any] | None = None,
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
    import lightgbm as lgb
    from public_notebook_replay_audit import (
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
    dense_config = dense_config or dict(manifest.get("dense_config") or {})
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
    dense_features, dense_group_columns, dense_summary = build_tvt_dense_confidence_features(
        pd.concat(
            [
                anchored_frame.reset_index(drop=True),
                projection_features[
                    [col for col in projection_features.columns if col not in {"id", "well"}]
                ].reset_index(drop=True),
            ],
            axis=1,
        ),
        dense_config,
    )
    configured_dense_groups = manifest.get("dense_feature_groups") or {}
    if configured_dense_groups and {
        key: list(value) for key, value in dense_group_columns.items()
    } != {key: list(value) for key, value in configured_dense_groups.items()}:
        raise ValueError("TVT dense feature groups differ from train manifest")

    variant_configs = {
        str(item["name"]): dict(item)
        for item in manifest.get("variants", [])
        if item.get("enabled", True)
    }
    if variant_name not in variant_configs:
        raise ValueError(f"variant={variant_name} not found in train manifest")
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        {**projection_group_columns, **dense_group_columns},
        variant_configs[variant_name],
    )
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    test_frame = pd.concat(
        [
            anchored_frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    ).merge(
        dense_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
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
    dense_summary_path = output_dir / f"{OUTPUT_PREFIX}_inference_dense_feature_summary.csv"
    feature_schema_path = output_dir / f"{OUTPUT_PREFIX}_inference_feature_schema.csv"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    projection_summary.to_csv(projection_summary_path, index=False)
    dense_summary.to_csv(dense_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_tvt_dense_confidence_feature": bool(
                    feature in [col for col in dense_features.columns if col not in {"id", "well"}]
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
        "experiment": "exp151_tvt_dense_addonly_confidence_features_on_exp092",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "anchor_source": anchor_meta,
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
            "dense_feature_summary": dense_summary_path.name,
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
