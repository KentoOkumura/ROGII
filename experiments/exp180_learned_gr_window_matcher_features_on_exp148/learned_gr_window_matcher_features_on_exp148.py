from __future__ import annotations

import gzip
import hashlib
import json
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
OUTPUT_PREFIX = "exp180_learned_gr_window_matcher_features_on_exp148"
GR_MATCHER_TRAIN_FEATURES = (
    "exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_train_features.csv.gz"
)
GR_MATCHER_FEATURE_SCHEMA = (
    "exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_feature_schema.csv"
)
GR_MATCHER_FEATURE_SUMMARY = (
    "exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_feature_summary.csv"
)
GR_MATCHER_FEATURE_MANIFEST = (
    "exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_feature_manifest.json"
)
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
GR_MATCH_FEATURE_NAMES = [
    "raw_abs",
    "window_mae",
    "window_rmse",
    "window_ncc",
    "z_mae",
    "derivative_mae",
    "energy_abs",
    "missing_mean",
    "combo_score",
]
GR_MATCH_CONTEXT_FEATURES = [
    "candidate_minus_last_known_tvt",
    "candidate_abs_minus_last_known_tvt",
    "candidate_tvt_pct",
    "candidate_outside_typewell_ft",
    "md_to_last_known",
    "abs_md_to_last_known",
    "z_to_last_known",
    "abs_z_to_last_known",
    "prefix_fraction",
    "known_prefix_rows",
    "row_gr_missing_window",
    "typewell_gr_missing_window",
]


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


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


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


def _csv_dtype_float32_numeric(path: str | Path, *, string_columns: set[str]) -> dict[str, Any]:
    columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    return {column: (str if column in string_columns else np.float32) for column in columns}


def _finite_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            return False
    return True


def _keys_aligned(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        len(left) == len(right)
        and left["id"].astype(str).reset_index(drop=True).equals(
            right["id"].astype(str).reset_index(drop=True)
        )
        and left["well"].astype(str).reset_index(drop=True).equals(
            right["well"].astype(str).reset_index(drop=True)
        )
    )


def _append_float32_columns(
    base: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    for column in columns:
        base[column] = features[column].to_numpy(dtype=np.float32, copy=False)
    return base


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
    frame = pd.read_csv(
        source,
        nrows=max_rows,
        dtype=_csv_dtype_float32_numeric(source, string_columns={"id", "well"}),
    )
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
    if not _finite_numeric_columns(frame, ["target", *feature_columns]):
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
            / "exp180_learned_gr_window_matcher_features_on_exp148"
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


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or float(np.std(x[mask])) < 1e-6:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _affine_calibration(kgr: np.ndarray, tw_at_k: np.ndarray, *, min_pts: int = 20) -> tuple[float, float]:
    kgr = np.asarray(kgr, dtype=np.float64)
    tw_at_k = np.asarray(tw_at_k, dtype=np.float64)
    mask = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if mask.sum() < min_pts or float(np.std(tw_at_k[mask])) < 1e-6:
        if mask.any():
            return 1.0, float(np.nanmean(kgr[mask]) - np.nanmean(tw_at_k[mask]))
        return 1.0, 0.0
    a, b = np.polyfit(tw_at_k[mask], kgr[mask], 1)
    return float(a), float(b)


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True) + 1e-6
    return centered / scale


def _multi_scale_ncc(
    kgr: np.ndarray,
    ktvt: np.ndarray,
    hgr: np.ndarray,
    *,
    half_windows: tuple[int, ...] = (8, 15, 25),
    stride: int = 3,
    max_starts: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for half_window in half_windows:
        window = 2 * int(half_window) + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < window + 1 or nh == 0:
            out.append(
                (
                    np.full(nh, float(ktvt[-1]), dtype=np.float32),
                    np.zeros(nh, dtype=np.float32),
                )
            )
            continue
        kg = (
            pd.Series(kgr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        hg = (
            pd.Series(hgr)
            .rolling(5, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        starts = np.arange(0, nk - window + 1, int(stride), dtype=np.int32)
        if len(starts) == 0:
            out.append(
                (
                    np.full(nh, float(ktvt[-1]), dtype=np.float32),
                    np.zeros(nh, dtype=np.float32),
                )
            )
            continue
        if max_starts is not None and len(starts) > int(max_starts):
            sampled = np.linspace(0, len(starts) - 1, int(max_starts), dtype=np.int32)
            starts = starts[np.unique(sampled)]
        offsets = np.arange(window, dtype=np.int32)
        candidates = kg[starts[:, None] + offsets[None, :]].astype(np.float32)
        candidates = _standardize_rows(candidates)
        padded = np.pad(hg, int(half_window), mode="edge")
        observed = padded[np.arange(nh)[:, None] + offsets[None, :]].astype(np.float32)
        observed = _standardize_rows(observed)
        ncc = observed @ candidates.T / float(window)
        best = ncc.argmax(axis=1)
        score = ncc.max(axis=1).astype(np.float32)
        out.append((ktvt[np.clip(starts[best] + int(half_window), 0, nk - 1)], score))
    tvts = np.stack([item[0] for item in out], axis=1)
    scores = np.stack([item[1] for item in out], axis=1)
    weights = np.exp(3.0 * scores)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-9
    return out, (tvts * weights).sum(axis=1).astype(np.float32)


def _nearest_prefix_indices(prefix_tvt: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, candidate_tvt, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - candidate_tvt) < np.abs(
        sorted_tvt[left] - candidate_tvt
    )
    nearest_sorted = np.where(choose_right, right, left)
    return order[nearest_sorted].astype(np.int32)


def _candidate_multi_obs_scores_for_well(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    observation_offsets: np.ndarray,
    gr_scale: float,
    out_of_range_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_candidates = candidate_values.shape
    fill = float(prefix_tvt[-1])
    candidate_values = np.nan_to_num(candidate_values, nan=fill)
    nearest_prefix = _nearest_prefix_indices(prefix_tvt, candidate_values.reshape(-1)).reshape(
        n_rows, n_candidates
    )
    eval_vectors = []
    candidate_vectors = []
    for offset in observation_offsets:
        eval_indices = np.clip(row_idx + int(offset), 0, len(full_gr) - 1)
        prefix_indices = np.clip(nearest_prefix + int(offset), 0, len(full_gr) - 1)
        eval_vectors.append(full_gr[eval_indices])
        candidate_vectors.append(full_gr[prefix_indices])
    eval_matrix = np.stack(eval_vectors, axis=1).astype(np.float32)
    candidate_tensor = np.stack(candidate_vectors, axis=2).astype(np.float32)
    diff_mae = np.mean(np.abs(candidate_tensor - eval_matrix[:, None, :]), axis=2)
    eval_norm = _standardize_rows(eval_matrix)
    flat_candidate = candidate_tensor.reshape(n_rows * n_candidates, len(observation_offsets))
    candidate_norm = _standardize_rows(flat_candidate).reshape(
        n_rows,
        n_candidates,
        len(observation_offsets),
    )
    ncc = np.mean(candidate_norm * eval_norm[:, None, :], axis=2)
    ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)
    low = float(np.nanmin(prefix_tvt))
    high = float(np.nanmax(prefix_tvt))
    below = np.maximum(0.0, low - candidate_values)
    above = np.maximum(0.0, candidate_values - high)
    range_penalty = np.exp(-((below + above) / max(out_of_range_scale, 1e-6)))
    mae_score = np.exp(-(diff_mae / max(gr_scale, 1e-6)))
    score = np.clip(mae_score * (0.25 + 0.75 * ncc_score) * range_penalty, 0.0, 1.0)
    return score.astype(np.float32), diff_mae.astype(np.float32), ncc.astype(np.float32)


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
    frame = pd.read_csv(
        source,
        dtype=_csv_dtype_float32_numeric(source, string_columns={"id", "well"}),
    )
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
    if not _finite_numeric_columns(frame, numeric_cols):
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
    feature_path = output_dir / f"{OUTPUT_PREFIX}_current_test_learned_likelihood_ml_features.csv.gz"
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


def _candidate_values_from_frame(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[np.ndarray] = []
    last_known = frame["last_known_tvt"].to_numpy(np.float32)
    for spec in specs:
        if spec.get("enabled", True) is False:
            continue
        name = str(spec["name"])
        column = str(spec["column"])
        kind = str(spec.get("kind", "delta"))
        if column not in frame.columns:
            raise ValueError(f"learned GR matcher candidate column is missing: {column}")
        raw = frame[column].to_numpy(np.float32)
        if kind == "absolute":
            candidate = raw
        elif kind == "delta":
            candidate = last_known + raw
        else:
            raise ValueError(f"Unsupported learned GR matcher candidate kind={kind} for {name}")
        names.append(name)
        values.append(candidate.astype(np.float32))
    if not names:
        raise ValueError("No enabled learned GR matcher candidates configured")
    return names, np.stack(values, axis=1).astype(np.float32)


def _window_mask(
    *,
    prefix_md: np.ndarray,
    anchor_md: float,
    config: dict[str, Any],
) -> np.ndarray:
    kind = str(config.get("kind", "md_tail"))
    if kind == "md_tail":
        mask = prefix_md >= float(anchor_md) - float(config.get("md_back", 1000.0))
    elif kind == "last_n":
        rows = int(config.get("rows", 50))
        mask = np.zeros(len(prefix_md), dtype=bool)
        mask[max(0, len(prefix_md) - rows) :] = True
    else:
        raise ValueError(f"Unsupported learned GR matcher window kind={kind}")
    if int(mask.sum()) < int(config.get("min_rows", 8)):
        mask = np.ones(len(prefix_md), dtype=bool)
    return mask


def _stable_int(*parts: str, modulo: int | None = None) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value % int(modulo) if modulo else value


def _fill_numeric_array(series: pd.Series, fallback: float) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(float(fallback))
        .to_numpy(np.float32)
    )


def _rolling_mean_array(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def _nearest_indices(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, values, side="left")
    left = np.clip(positions - 1, 0, len(sorted_values) - 1)
    right = np.clip(positions, 0, len(sorted_values) - 1)
    choose_right = np.abs(sorted_values[right] - values) < np.abs(sorted_values[left] - values)
    return np.where(choose_right, right, left).astype(np.int32)


def _gather_windows(values: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.clip(centers[:, None] + offsets.astype(np.int32), 0, len(values) - 1)
    return values[idx].astype(np.float32)


def _read_matcher_well_arrays(
    well: str,
    raw_dir: str | Path,
    *,
    smoothing_window: int,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists():
        raise FileNotFoundError(f"raw horizontal well file not found: {horizontal_path}")
    if not typewell_path.exists():
        raise FileNotFoundError(f"raw typewell file not found: {typewell_path}")

    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        raise ValueError(f"No finite TVT_input prefix rows for well={well}")
    prefix_end = int(known[-1] + 1)
    last_idx = int(known[-1])

    horizontal_gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce")
    typewell_gr_raw = pd.to_numeric(typewell["GR"], errors="coerce")
    horizontal_fallback = float(horizontal_gr_raw.iloc[:prefix_end].mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = float(horizontal_gr_raw.mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = float(typewell_gr_raw.mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = 0.0
    typewell_fallback = float(typewell_gr_raw.mean())
    if not np.isfinite(typewell_fallback):
        typewell_fallback = horizontal_fallback

    horizontal_gr = _rolling_mean_array(
        _fill_numeric_array(horizontal_gr_raw, horizontal_fallback),
        int(smoothing_window),
    )
    typewell_gr = _rolling_mean_array(
        _fill_numeric_array(typewell_gr_raw, typewell_fallback),
        int(smoothing_window),
    )
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    finite_typewell = np.isfinite(typewell_tvt)
    if finite_typewell.sum() < 4:
        raise ValueError(f"Too few finite typewell TVT rows for well={well}")
    if not finite_typewell.all():
        keep = np.flatnonzero(finite_typewell)
        typewell_tvt = typewell_tvt[keep]
        typewell_gr = typewell_gr[keep]
        typewell = typewell.iloc[keep].reset_index(drop=True)

    return {
        "well": str(well),
        "horizontal_md": pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float32),
        "horizontal_z": pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float32),
        "horizontal_gr": horizontal_gr.astype(np.float32),
        "horizontal_missing": horizontal_gr_raw.isna().to_numpy(),
        "typewell_tvt": typewell_tvt.astype(np.float32),
        "typewell_gr": typewell_gr.astype(np.float32),
        "typewell_missing": pd.to_numeric(typewell["GR"], errors="coerce").isna().to_numpy(),
        "prefix_end": int(prefix_end),
        "last_known_tvt": float(tvt_input[last_idx]),
        "last_known_md": float(pd.to_numeric(horizontal["MD"], errors="coerce").iloc[last_idx]),
        "last_known_z": float(pd.to_numeric(horizontal["Z"], errors="coerce").iloc[last_idx]),
        "tvt_input": tvt_input,
    }


def _selected_known_prefix_rows(
    arrays: dict[str, Any],
    *,
    rows_per_well: int,
    row_margin: int,
) -> np.ndarray:
    tvt_input = np.asarray(arrays["tvt_input"], dtype=np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
    eligible = known[
        (known >= int(row_margin))
        & (known < int(arrays["prefix_end"]) - int(row_margin))
    ]
    if len(eligible) == 0:
        eligible = known
    if len(eligible) > int(rows_per_well):
        positions = np.linspace(0, len(eligible) - 1, int(rows_per_well)).round().astype(np.int64)
        eligible = eligible[positions]
    return eligible.astype(np.int32)


def _gr_window_feature_frame(
    *,
    horizontal_gr: np.ndarray,
    horizontal_missing: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_missing: np.ndarray,
    row_idx: np.ndarray,
    candidate_idx: np.ndarray,
    window_offsets: np.ndarray,
    derivative_step: int,
    prefix: str,
) -> pd.DataFrame:
    h_window = _gather_windows(horizontal_gr, row_idx, window_offsets)
    t_window = _gather_windows(typewell_gr, candidate_idx, window_offsets)
    h_center = horizontal_gr[np.clip(row_idx, 0, len(horizontal_gr) - 1)]
    t_center = typewell_gr[np.clip(candidate_idx, 0, len(typewell_gr) - 1)]
    h_norm = _standardize_rows(h_window)
    t_norm = _standardize_rows(t_window)
    ncc = np.mean(h_norm * t_norm, axis=1)

    h_derivative = np.gradient(horizontal_gr).astype(np.float32)
    t_derivative = np.gradient(typewell_gr).astype(np.float32)
    h_d = _gather_windows(h_derivative, row_idx, window_offsets)
    t_d = _gather_windows(t_derivative, candidate_idx, window_offsets)
    h_missing = _gather_windows(horizontal_missing.astype(np.float32), row_idx, window_offsets)
    t_missing = _gather_windows(typewell_missing.astype(np.float32), candidate_idx, window_offsets)

    raw_abs = np.abs(t_center - h_center)
    window_mae = np.mean(np.abs(t_window - h_window), axis=1)
    window_rmse = np.sqrt(np.mean(np.square(t_window - h_window), axis=1))
    z_mae = np.mean(np.abs(t_norm - h_norm), axis=1)
    derivative_mae = np.mean(np.abs(t_d - h_d), axis=1)
    energy_abs = np.abs(
        np.sqrt(np.mean(np.square(t_d), axis=1)) - np.sqrt(np.mean(np.square(h_d), axis=1))
    )
    missing_mean = 0.5 * h_missing.mean(axis=1) + 0.5 * t_missing.mean(axis=1)
    combo_cost = (
        0.20 * np.clip(raw_abs / 18.0, 0.0, 5.0)
        + 0.25 * np.clip(window_mae / 18.0, 0.0, 5.0)
        + 0.20 * np.clip(z_mae, 0.0, 5.0)
        + 0.20 * np.clip(derivative_mae / max(float(derivative_step), 1.0), 0.0, 5.0)
        + 0.10 * np.clip(energy_abs / max(float(derivative_step), 1.0), 0.0, 5.0)
        + 0.05 * np.clip(missing_mean, 0.0, 1.0)
    )
    return pd.DataFrame(
        {
            f"{prefix}_raw_abs": raw_abs.astype(np.float32),
            f"{prefix}_window_mae": window_mae.astype(np.float32),
            f"{prefix}_window_rmse": window_rmse.astype(np.float32),
            f"{prefix}_window_ncc": ncc.astype(np.float32),
            f"{prefix}_z_mae": z_mae.astype(np.float32),
            f"{prefix}_derivative_mae": derivative_mae.astype(np.float32),
            f"{prefix}_energy_abs": energy_abs.astype(np.float32),
            f"{prefix}_missing_mean": missing_mean.astype(np.float32),
            f"{prefix}_combo_score": np.exp(-combo_cost).astype(np.float32),
        }
    )


def _candidate_long_feature_frame(
    *,
    arrays: dict[str, Any],
    row_idx: np.ndarray,
    candidate_tvt: np.ndarray,
    candidate_names: list[str],
    window_offsets: np.ndarray,
    derivative_step: int,
) -> pd.DataFrame:
    n_rows, n_candidates = candidate_tvt.shape
    flat_candidate = candidate_tvt.reshape(-1).astype(np.float32)
    candidate_idx = _nearest_indices(arrays["typewell_tvt"], flat_candidate)
    row_repeat = np.repeat(row_idx.astype(np.int32), n_candidates)

    real_features = _gr_window_feature_frame(
        horizontal_gr=arrays["horizontal_gr"],
        horizontal_missing=arrays["horizontal_missing"],
        typewell_gr=arrays["typewell_gr"],
        typewell_missing=arrays["typewell_missing"],
        row_idx=row_repeat,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
        prefix="real",
    )
    roll = _stable_int(
        "exp180_learned_gr_window_matcher_shuffled_typewell",
        str(arrays["well"]),
        modulo=max(len(arrays["typewell_gr"]) - 1, 1),
    ) + 1
    shuffled_features = _gr_window_feature_frame(
        horizontal_gr=arrays["horizontal_gr"],
        horizontal_missing=arrays["horizontal_missing"],
        typewell_gr=np.roll(arrays["typewell_gr"], int(roll)),
        typewell_missing=np.roll(arrays["typewell_missing"], int(roll)),
        row_idx=row_repeat,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
        prefix="shuf",
    )

    typewell_min = float(np.nanmin(arrays["typewell_tvt"]))
    typewell_max = float(np.nanmax(arrays["typewell_tvt"]))
    typewell_span = max(typewell_max - typewell_min, 1.0)
    row_md = arrays["horizontal_md"][np.clip(row_idx, 0, len(arrays["horizontal_md"]) - 1)]
    row_z = arrays["horizontal_z"][np.clip(row_idx, 0, len(arrays["horizontal_z"]) - 1)]
    row_gr_missing = _gather_windows(
        arrays["horizontal_missing"].astype(np.float32),
        row_idx,
        window_offsets,
    ).mean(axis=1)
    typewell_gr_missing = _gather_windows(
        arrays["typewell_missing"].astype(np.float32),
        candidate_idx,
        window_offsets,
    ).mean(axis=1)
    outside = np.maximum(0.0, typewell_min - flat_candidate) + np.maximum(
        0.0,
        flat_candidate - typewell_max,
    )

    context = pd.DataFrame(
        {
            "candidate_name": np.tile(candidate_names, n_rows),
            "candidate_tvt": flat_candidate,
            "candidate_minus_last_known_tvt": (
                flat_candidate - float(arrays["last_known_tvt"])
            ).astype(np.float32),
            "candidate_abs_minus_last_known_tvt": np.abs(
                flat_candidate - float(arrays["last_known_tvt"])
            ).astype(np.float32),
            "candidate_tvt_pct": ((flat_candidate - typewell_min) / typewell_span).astype(
                np.float32
            ),
            "candidate_outside_typewell_ft": outside.astype(np.float32),
            "md_to_last_known": (
                np.repeat(row_md, n_candidates).astype(np.float32)
                - float(arrays["last_known_md"])
            ).astype(np.float32),
            "abs_md_to_last_known": np.abs(
                np.repeat(row_md, n_candidates).astype(np.float32)
                - float(arrays["last_known_md"])
            ).astype(np.float32),
            "z_to_last_known": (
                np.repeat(row_z, n_candidates).astype(np.float32)
                - float(arrays["last_known_z"])
            ).astype(np.float32),
            "abs_z_to_last_known": np.abs(
                np.repeat(row_z, n_candidates).astype(np.float32)
                - float(arrays["last_known_z"])
            ).astype(np.float32),
            "prefix_fraction": (
                np.repeat(row_idx, n_candidates) / max(int(arrays["prefix_end"]) - 1, 1)
            ).astype(np.float32),
            "known_prefix_rows": np.full(
                n_rows * n_candidates,
                int(arrays["prefix_end"]),
                dtype=np.float32,
            ),
            "row_gr_missing_window": np.repeat(row_gr_missing, n_candidates).astype(np.float32),
            "typewell_gr_missing_window": typewell_gr_missing.astype(np.float32),
        }
    )
    return pd.concat([context, real_features, shuffled_features], axis=1)


def _build_gr_matcher_pairs_for_well(
    well: str,
    raw_dir: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    scorer_cfg = dict(config.get("scorer_training") or {})
    smoothing_window = int(config.get("gr_smoothing_window", scorer_cfg.get("gr_smoothing_window", 5)))
    rows_per_well = int(scorer_cfg.get("rows_per_well", 64))
    row_margin = int(scorer_cfg.get("row_margin", 32))
    derivative_step = int(config.get("derivative_step", scorer_cfg.get("derivative_step", 3)))
    window_offsets = np.asarray(
        scorer_cfg.get(
            "window_offsets",
            config.get("window_offsets", [-24, -18, -12, -6, 0, 6, 12, 18, 24]),
        ),
        dtype=np.int32,
    )
    negative_offsets = np.asarray(
        scorer_cfg.get("negative_offsets_ft", [-100, -50, -25, -15, 15, 25, 50, 100]),
        dtype=np.float32,
    )

    try:
        arrays = _read_matcher_well_arrays(
            well,
            raw_dir,
            smoothing_window=smoothing_window,
        )
    except (FileNotFoundError, ValueError) as exc:
        return None, {"well": str(well), "status": "invalid_raw", "message": str(exc)}
    row_idx = _selected_known_prefix_rows(
        arrays,
        rows_per_well=rows_per_well,
        row_margin=row_margin,
    )
    if len(row_idx) == 0:
        return None, {"well": str(well), "status": "no_selected_prefix_rows"}

    true_tvt = arrays["tvt_input"][row_idx].astype(np.float32)
    candidate_offsets = [np.zeros(len(row_idx), dtype=np.float32)]
    candidate_names = ["positive"]
    labels = [np.ones(len(row_idx), dtype=np.int8)]
    for offset in negative_offsets:
        candidate_offsets.append(np.full(len(row_idx), float(offset), dtype=np.float32))
        candidate_names.append(f"decoy_{float(offset):+.0f}ft".replace("+", "p").replace("-", "m"))
        labels.append(np.zeros(len(row_idx), dtype=np.int8))
    offset_matrix = np.column_stack(candidate_offsets).astype(np.float32)
    label_matrix = np.column_stack(labels).astype(np.int8)
    candidate_tvt = (true_tvt[:, None] + offset_matrix).astype(np.float32)
    features = _candidate_long_feature_frame(
        arrays=arrays,
        row_idx=row_idx,
        candidate_tvt=candidate_tvt,
        candidate_names=candidate_names,
        window_offsets=window_offsets,
        derivative_step=derivative_step,
    )
    n_rows, n_candidates = candidate_tvt.shape
    features.insert(0, "anchor_id", [f"{well}_{int(idx)}" for idx in np.repeat(row_idx, n_candidates)])
    features.insert(
        1,
        "pair_id",
        [f"{well}_{int(idx)}_{name}" for idx in row_idx for name in candidate_names],
    )
    features.insert(2, "well", str(well))
    features.insert(3, "row_idx", np.repeat(row_idx, n_candidates).astype(np.int32))
    features.insert(4, "label_within_10ft", label_matrix.reshape(-1).astype(np.int8))
    features.insert(5, "abs_tvt_offset", np.abs(offset_matrix.reshape(-1)).astype(np.float32))

    return features, {
        "well": str(well),
        "status": "ok",
        "prefix_rows": int(arrays["prefix_end"]),
        "selected_prefix_rows": int(len(row_idx)),
        "pairs": int(len(features)),
        "typewell_rows": int(len(arrays["typewell_tvt"])),
    }


def _cap_gr_matcher_pairs(pair_frame: pd.DataFrame, max_pairs: int | None) -> pd.DataFrame:
    if max_pairs is None or len(pair_frame) <= int(max_pairs):
        return pair_frame.reset_index(drop=True)
    anchors = pair_frame[["anchor_id", "well", "row_idx"]].drop_duplicates().sort_values(
        ["well", "row_idx"]
    )
    pairs_per_anchor = int(pair_frame.groupby("anchor_id", sort=False).size().median())
    keep_anchor_count = max(1, int(max_pairs) // max(pairs_per_anchor, 1))
    keep_positions = (
        np.linspace(0, len(anchors) - 1, min(keep_anchor_count, len(anchors)))
        .round()
        .astype(np.int64)
    )
    keep_anchors = set(anchors.iloc[keep_positions]["anchor_id"].astype(str))
    return pair_frame[pair_frame["anchor_id"].isin(keep_anchors)].reset_index(drop=True)


def _build_gr_matcher_pair_dataset(
    *,
    wells: list[str],
    raw_dir: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer_cfg = dict(config.get("scorer_training") or {})
    max_wells = scorer_cfg.get("max_wells")
    if max_wells not in {None, "null"}:
        wells = list(wells)[: int(max_wells)]
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for well in wells:
        frame, status = _build_gr_matcher_pairs_for_well(well, raw_dir, config)
        statuses.append(status)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No learned GR matcher pair rows were generated.")
    pair_frame = pd.concat(frames, ignore_index=True)
    max_pairs = scorer_cfg.get("max_pairs")
    pair_frame = _cap_gr_matcher_pairs(
        pair_frame,
        None if max_pairs in {None, "null"} else int(max_pairs),
    )
    return pair_frame, pd.DataFrame(statuses)


def _gr_matcher_feature_sets() -> dict[str, list[str]]:
    real = [f"real_{name}" for name in GR_MATCH_FEATURE_NAMES] + GR_MATCH_CONTEXT_FEATURES
    shuffled = [f"shuf_{name}" for name in GR_MATCH_FEATURE_NAMES] + GR_MATCH_CONTEXT_FEATURES
    return {
        "real_gr_logistic": real,
        "shuffled_gr_logistic": shuffled,
        "no_gr_logistic": list(GR_MATCH_CONTEXT_FEATURES),
        "real_gr_expected_error": real,
    }


def _make_gr_match_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _make_gr_match_regressor(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=0.01,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _fit_gr_matcher_scorers(
    pair_frame: pd.DataFrame,
    *,
    train_wells: set[str] | None,
    seed: int,
) -> dict[str, Pipeline]:
    train = pair_frame
    if train_wells is not None:
        train = pair_frame[pair_frame["well"].astype(str).isin(train_wells)]
    if train.empty:
        raise ValueError("No pair rows available to fit learned GR matcher scorers")
    y = train["label_within_10ft"].to_numpy(np.int8)
    if len(np.unique(y)) < 2:
        raise ValueError("Learned GR matcher training pairs need both positive and negative labels")
    columns = _gr_matcher_feature_sets()
    models: dict[str, Pipeline] = {}
    for name in ["real_gr_logistic", "shuffled_gr_logistic", "no_gr_logistic"]:
        model = _make_gr_match_classifier(seed)
        model.fit(train[columns[name]], y)
        models[name] = model
    regressor = _make_gr_match_regressor(seed)
    regressor.fit(
        train[columns["real_gr_expected_error"]],
        train["abs_tvt_offset"].to_numpy(np.float32),
    )
    models["real_gr_expected_error"] = regressor
    return models


def _score_candidate_long_frame(
    candidate_frame: pd.DataFrame,
    *,
    models: dict[str, Pipeline],
) -> dict[str, np.ndarray]:
    columns = _gr_matcher_feature_sets()
    real_prob = models["real_gr_logistic"].predict_proba(
        candidate_frame[columns["real_gr_logistic"]]
    )[:, 1]
    shuffled_prob = models["shuffled_gr_logistic"].predict_proba(
        candidate_frame[columns["shuffled_gr_logistic"]]
    )[:, 1]
    no_gr_prob = models["no_gr_logistic"].predict_proba(
        candidate_frame[columns["no_gr_logistic"]]
    )[:, 1]
    expected_error = np.maximum(
        0.0,
        models["real_gr_expected_error"].predict(
            candidate_frame[columns["real_gr_expected_error"]]
        ),
    )
    return {
        "real_prob": np.clip(real_prob.astype(np.float32), 1e-6, 1.0 - 1e-6),
        "shuffled_prob": np.clip(shuffled_prob.astype(np.float32), 1e-6, 1.0 - 1e-6),
        "no_gr_prob": np.clip(no_gr_prob.astype(np.float32), 1e-6, 1.0 - 1e-6),
        "expected_error": expected_error.astype(np.float32),
        "combo_score": candidate_frame["real_combo_score"].to_numpy(np.float32),
    }


def _score_gr_matcher_frame_with_models(
    frame: pd.DataFrame,
    *,
    raw_dir: str | Path,
    config: dict[str, Any],
    models: dict[str, Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = str(config.get("prefix") or "grm_")
    candidate_specs = [
        dict(item)
        for item in config.get("candidates")
        or [
            {"name": "pf_ancc", "column": "pf_ancc", "kind": "absolute"},
            {"name": "beam_mean", "column": "beam_mean_d", "kind": "delta"},
            {"name": "likpf_mean", "column": "likpf_mean_d", "kind": "delta"},
            {"name": "sc_ens", "column": "sc_ens_d", "kind": "delta"},
            {"name": "hyb", "column": "hyb_d", "kind": "delta"},
        ]
    ]
    smoothing_window = int(config.get("gr_smoothing_window", 5))
    derivative_step = int(config.get("derivative_step", 3))
    window_offsets = np.asarray(
        config.get("window_offsets", [-24, -18, -12, -6, 0, 6, 12, 18, 24]),
        dtype=np.int32,
    )

    parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for well, index in frame.groupby("well", sort=False).groups.items():
        part = frame.loc[list(index)].reset_index(drop=True)
        arrays = _read_matcher_well_arrays(
            str(well),
            raw_dir,
            smoothing_window=smoothing_window,
        )
        row_idx = _row_indices_from_ids(part["id"])
        candidate_names, candidate_values = _candidate_values_from_frame(part, candidate_specs)
        long_frame = _candidate_long_feature_frame(
            arrays=arrays,
            row_idx=row_idx,
            candidate_tvt=candidate_values,
            candidate_names=candidate_names,
            window_offsets=window_offsets,
            derivative_step=derivative_step,
        )
        scores = _score_candidate_long_frame(long_frame, models=models)
        n_rows, n_candidates = candidate_values.shape
        real_prob = scores["real_prob"].reshape(n_rows, n_candidates)
        shuffled_prob = scores["shuffled_prob"].reshape(n_rows, n_candidates)
        no_gr_prob = scores["no_gr_prob"].reshape(n_rows, n_candidates)
        expected_error = scores["expected_error"].reshape(n_rows, n_candidates)
        combo_score = scores["combo_score"].reshape(n_rows, n_candidates)

        sorted_prob = np.sort(real_prob, axis=1)
        top1 = real_prob.argmax(axis=1)
        top2_prob = sorted_prob[:, -2] if n_candidates > 1 else sorted_prob[:, -1]
        prob_sum = real_prob.sum(axis=1, keepdims=True) + 1e-9
        prob_norm = real_prob / prob_sum
        entropy = -np.sum(prob_norm * np.log(np.clip(prob_norm, 1e-9, 1.0)), axis=1)
        min_error_idx = expected_error.argmin(axis=1)
        sorted_error = np.sort(expected_error, axis=1)
        md_since = (
            part["md_since"].to_numpy(np.float32)
            if "md_since" in part.columns
            else np.zeros(n_rows, dtype=np.float32)
        )

        out: dict[str, Any] = {
            "id": part["id"].to_numpy(),
            "well": part["well"].to_numpy(),
            f"{prefix}top1_candidate_index": top1.astype(np.float32),
            f"{prefix}top1_prob": real_prob[np.arange(n_rows), top1].astype(np.float32),
            f"{prefix}top2_prob": top2_prob.astype(np.float32),
            f"{prefix}prob_margin_top1_top2": (
                real_prob[np.arange(n_rows), top1] - top2_prob
            ).astype(np.float32),
            f"{prefix}prob_entropy": entropy.astype(np.float32),
            f"{prefix}prob_mean": real_prob.mean(axis=1).astype(np.float32),
            f"{prefix}prob_std": real_prob.std(axis=1).astype(np.float32),
            f"{prefix}expected_error_top1_prob_candidate": expected_error[
                np.arange(n_rows), top1
            ].astype(np.float32),
            f"{prefix}expected_error_min": expected_error[
                np.arange(n_rows), min_error_idx
            ].astype(np.float32),
            f"{prefix}expected_error_margin_top2_top1": (
                sorted_error[:, 1] - sorted_error[:, 0] if n_candidates > 1 else sorted_error[:, 0]
            ).astype(np.float32),
            f"{prefix}expected_error_mean": expected_error.mean(axis=1).astype(np.float32),
            f"{prefix}real_vs_shuffled_gap_top1": (
                real_prob[np.arange(n_rows), top1] - shuffled_prob[np.arange(n_rows), top1]
            ).astype(np.float32),
            f"{prefix}real_vs_shuffled_gap_mean": (
                real_prob - shuffled_prob
            ).mean(axis=1).astype(np.float32),
            f"{prefix}real_vs_no_gr_gap_top1": (
                real_prob[np.arange(n_rows), top1] - no_gr_prob[np.arange(n_rows), top1]
            ).astype(np.float32),
            f"{prefix}prob_margin_x_md_since": (
                (real_prob[np.arange(n_rows), top1] - top2_prob) * md_since
            ).astype(np.float32),
            f"{prefix}expected_error_min_x_md_since": (
                expected_error[np.arange(n_rows), min_error_idx] * md_since
            ).astype(np.float32),
        }
        for cand_i, cand_name in enumerate(candidate_names):
            safe_name = str(cand_name)
            out[f"{prefix}prob_{safe_name}"] = real_prob[:, cand_i].astype(np.float32)
            out[f"{prefix}shuffled_prob_{safe_name}"] = shuffled_prob[:, cand_i].astype(np.float32)
            out[f"{prefix}no_gr_prob_{safe_name}"] = no_gr_prob[:, cand_i].astype(np.float32)
            out[f"{prefix}prob_minus_shuffled_{safe_name}"] = (
                real_prob[:, cand_i] - shuffled_prob[:, cand_i]
            ).astype(np.float32)
            out[f"{prefix}prob_minus_no_gr_{safe_name}"] = (
                real_prob[:, cand_i] - no_gr_prob[:, cand_i]
            ).astype(np.float32)
            out[f"{prefix}expected_error_{safe_name}"] = expected_error[:, cand_i].astype(np.float32)
            out[f"{prefix}combo_score_{safe_name}"] = combo_score[:, cand_i].astype(np.float32)
            out[f"{prefix}top1_is_{safe_name}"] = (top1 == cand_i).astype(np.float32)
            out[f"{prefix}min_error_is_{safe_name}"] = (min_error_idx == cand_i).astype(np.float32)
        scored = pd.DataFrame(out)
        feature_cols = [col for col in scored.columns if col not in {"id", "well"}]
        for col in feature_cols:
            scored[col] = pd.to_numeric(scored[col], errors="coerce").astype(np.float32)
        if not np.isfinite(scored[feature_cols].to_numpy(np.float32)).all():
            raise ValueError(f"Non-finite learned GR matcher scores for well={well}")
        parts.append(scored)
        summary_rows.append(
            {
                "well": str(well),
                "rows": int(len(scored)),
                "prefix_rows": int(arrays["prefix_end"]),
                "candidates": int(n_candidates),
                "top1_prob_mean": float(scored[f"{prefix}top1_prob"].mean()),
                "expected_error_min_mean": float(scored[f"{prefix}expected_error_min"].mean()),
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summary_rows)


def build_gr_matcher_window_features(
    base_frame: pd.DataFrame,
    learned_source: pd.DataFrame,
    *,
    raw_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    del learned_source
    config = config or {}
    group_name = str(config.get("feature_group") or "learned_gr_window_matcher")
    scorer_cfg = dict(config.get("scorer_training") or {})
    seed = int(scorer_cfg.get("seed", config.get("seed", 42)))
    n_folds = int(scorer_cfg.get("n_folds", 5))
    fold_safe = bool(scorer_cfg.get("fold_safe_by_well", True))
    raw_dir = Path(raw_dir)

    required = {"id", "well", "last_known_tvt", "anchor_md", "md_since", "z"}
    missing = sorted(required - set(base_frame.columns))
    if missing:
        raise ValueError(f"base frame missing learned GR matcher required columns: {missing}")
    wells = sorted(base_frame["well"].astype(str).unique().tolist())
    t_build = time.time()
    _log_event(
        "gr_matcher_build_start",
        rows=int(len(base_frame)),
        wells=int(len(wells)),
        fold_safe=fold_safe,
        scorer_n_folds=int(n_folds),
    )

    pair_frame, pair_status = _build_gr_matcher_pair_dataset(
        wells=wells,
        raw_dir=raw_dir,
        config=config,
    )
    _log_event(
        "gr_matcher_pair_dataset_done",
        pairs=int(len(pair_frame)),
        anchors=int(pair_frame["anchor_id"].nunique()),
        wells=int(pair_frame["well"].nunique()),
        elapsed_seconds=round(time.time() - t_build, 3),
    )

    scored_parts: list[pd.DataFrame] = []
    score_summaries: list[pd.DataFrame] = []
    if fold_safe and base_frame["well"].nunique() >= 2 and pair_frame["well"].nunique() >= 2:
        split_count = min(int(n_folds), int(base_frame["well"].nunique()))
        well_frame = pd.DataFrame({"well": wells})
        cv = GroupKFold(n_splits=split_count)
        for fold, (_, valid_idx) in enumerate(cv.split(well_frame, groups=well_frame["well"])):
            valid_wells = set(well_frame.iloc[valid_idx]["well"].astype(str))
            train_wells = set(pair_frame["well"].astype(str).unique()) - valid_wells
            models = _fit_gr_matcher_scorers(pair_frame, train_wells=train_wells, seed=seed + fold)
            fold_frame = base_frame[base_frame["well"].astype(str).isin(valid_wells)].copy()
            scored, score_summary = _score_gr_matcher_frame_with_models(
                fold_frame,
                raw_dir=raw_dir,
                config=config,
                models=models,
            )
            score_summary.insert(0, "scorer_fold", int(fold))
            score_summary["scorer_train_wells"] = int(len(train_wells))
            score_summary["scorer_train_pairs"] = int(
                pair_frame[pair_frame["well"].astype(str).isin(train_wells)].shape[0]
            )
            scored_parts.append(scored)
            score_summaries.append(score_summary)
            _log_event(
                "gr_matcher_fold_scored",
                fold=int(fold),
                rows=int(len(scored)),
                valid_wells=int(len(valid_wells)),
                train_wells=int(len(train_wells)),
                elapsed_seconds=round(time.time() - t_build, 3),
            )
    else:
        models = _fit_gr_matcher_scorers(pair_frame, train_wells=None, seed=seed)
        scored, score_summary = _score_gr_matcher_frame_with_models(
            base_frame,
            raw_dir=raw_dir,
            config=config,
            models=models,
        )
        score_summary.insert(0, "scorer_fold", -1)
        score_summary["scorer_train_wells"] = int(pair_frame["well"].nunique())
        score_summary["scorer_train_pairs"] = int(len(pair_frame))
        scored_parts.append(scored)
        score_summaries.append(score_summary)

    features = pd.concat(scored_parts, ignore_index=True)
    order = pd.MultiIndex.from_frame(base_frame[["id", "well"]].astype(str))
    features = features.set_index(["id", "well"]).loc[order].reset_index()
    if features.duplicated(["id", "well"]).any():
        raise ValueError("learned GR matcher feature frame contains duplicated id/well rows")

    group_columns = {group_name: [col for col in features.columns if col not in {"id", "well"}]}
    summary = pd.concat(score_summaries, ignore_index=True)
    summary.insert(0, "feature_group", group_name)
    summary["pair_rows"] = int(len(pair_frame))
    summary["pair_wells"] = int(pair_frame["well"].nunique())
    summary["pair_anchors"] = int(pair_frame["anchor_id"].nunique())
    summary["generated_features"] = len(group_columns[group_name])
    summary["pair_status_counts"] = json.dumps(
        pair_status["status"].value_counts().to_dict(),
        sort_keys=True,
    )
    _log_event(
        "gr_matcher_build_done",
        rows=int(len(features)),
        wells=int(features["well"].nunique()),
        generated_features=int(len(group_columns[group_name])),
        elapsed_seconds=round(time.time() - t_build, 3),
    )
    return features, group_columns, summary


def _write_gr_matcher_feature_cache(
    *,
    features: pd.DataFrame,
    group_columns: dict[str, list[str]],
    summary: pd.DataFrame,
    output_dir: Path,
    source_meta: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / GR_MATCHER_TRAIN_FEATURES
    schema_path = output_dir / GR_MATCHER_FEATURE_SCHEMA
    summary_path = output_dir / GR_MATCHER_FEATURE_SUMMARY
    manifest_path = output_dir / GR_MATCHER_FEATURE_MANIFEST

    feature_columns = [col for col in features.columns if col not in {"id", "well"}]
    for col in feature_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)
    features.to_csv(feature_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "dtype": str(features[feature].dtype),
                "feature_group": next(
                    (
                        group
                        for group, columns in group_columns.items()
                        if feature in set(columns)
                    ),
                    "learned_gr_window_matcher",
                ),
            }
            for index, feature in enumerate(feature_columns)
        ]
    ).to_csv(schema_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "experiment": "exp180_learned_gr_window_matcher_features_on_exp148",
        "kind": "gr_matcher_train_feature_cache",
        "feature_file": feature_path.name,
        "feature_sha256": sha256_file(feature_path),
        "feature_decompressed_sha256": sha256_gzip_decompressed(feature_path),
        "schema_file": schema_path.name,
        "schema_sha256": sha256_file(schema_path),
        "summary_file": summary_path.name,
        "summary_sha256": sha256_file(summary_path),
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_groups": {key: list(value) for key, value in group_columns.items()},
        "source_meta": _jsonable(source_meta),
        "gr_matcher_config": _jsonable(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log_event(
        "gr_matcher_cache_written",
        path=str(feature_path),
        rows=int(manifest["rows"]),
        wells=int(manifest["wells"]),
        features=int(manifest["feature_count"]),
        sha256=str(manifest["feature_sha256"]),
    )
    return manifest


def load_gr_matcher_feature_cache(
    feature_path: str | Path | None = None,
    *,
    schema_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    resolved_feature = find_artifact(GR_MATCHER_TRAIN_FEATURES, feature_path)
    features = pd.read_csv(
        resolved_feature,
        dtype=_csv_dtype_float32_numeric(resolved_feature, string_columns={"id", "well"}),
    )
    required = {"id", "well"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"learned GR matcher cache missing required columns: {missing}")
    feature_columns = [col for col in features.columns if col not in {"id", "well"}]
    if not _finite_numeric_columns(features, feature_columns):
        raise ValueError("learned GR matcher feature cache contains non-finite values")

    resolved_schema = None
    try:
        resolved_schema = find_artifact(GR_MATCHER_FEATURE_SCHEMA, schema_path)
        schema = pd.read_csv(resolved_schema)
        if "feature_group" in schema.columns:
            group_columns = {
                str(group): group_frame["feature"].astype(str).tolist()
                for group, group_frame in schema.groupby("feature_group", sort=False)
            }
        else:
            group_columns = {"learned_gr_window_matcher": feature_columns}
    except FileNotFoundError:
        schema = pd.DataFrame()
        group_columns = {"learned_gr_window_matcher": feature_columns}

    resolved_summary = None
    try:
        resolved_summary = find_artifact(GR_MATCHER_FEATURE_SUMMARY, summary_path)
        summary = pd.read_csv(resolved_summary)
    except FileNotFoundError:
        summary = pd.DataFrame(
            [
                {
                    "feature_group": "learned_gr_window_matcher",
                    "rows": int(len(features)),
                    "wells": int(features["well"].nunique()),
                    "generated_features": int(len(feature_columns)),
                }
            ]
        )

    meta = {
        "source": str(resolved_feature),
        "source_sha256": sha256_file(resolved_feature),
        "source_decompressed_sha256": sha256_gzip_decompressed(resolved_feature),
        "schema": str(resolved_schema) if resolved_schema else None,
        "schema_sha256": sha256_file(resolved_schema) if resolved_schema else None,
        "summary": str(resolved_summary) if resolved_summary else None,
        "summary_sha256": sha256_file(resolved_summary) if resolved_summary else None,
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "feature_count": int(len(feature_columns)),
    }
    _log_event(
        "gr_matcher_cache_loaded",
        path=str(resolved_feature),
        rows=int(meta["rows"]),
        wells=int(meta["wells"]),
        features=int(meta["feature_count"]),
    )
    return features, group_columns, summary, meta


def generate_gr_matcher_window_feature_cache_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    gr_matcher_config: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    frame, _, feature_meta = load_exp072_full_replay_cache_frame(cache_path, max_rows=max_rows)
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
    )

    projection_config = projection_config or {}
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("LGB OOF U-projection features are disabled for this cache")
    projection_features, _, _ = build_u_projection_features(
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
        copy=False,
    )
    learned_features, _, _ = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    before_rows = len(full_frame)
    full_frame = full_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(full_frame) != before_rows:
        raise ValueError(
            "learned GR matcher cache generation requires full learned feature coverage: "
            f"{len(full_frame)} of {before_rows}"
        )
    features, group_columns, summary = build_gr_matcher_window_features(
        full_frame,
        learned_features_source,
        raw_dir=train_dir,
        config=gr_matcher_config or {},
    )
    manifest = _write_gr_matcher_feature_cache(
        features=features,
        group_columns=group_columns,
        summary=summary,
        output_dir=output_dir,
        source_meta={
            "feature_source": feature_meta,
            "learned_likelihood_feature_source": learned_source_meta,
            "anchor_source": anchor_meta,
        },
        config=gr_matcher_config or {},
    )
    manifest["elapsed_seconds"] = round(time.time() - t0, 3)
    (output_dir / GR_MATCHER_FEATURE_MANIFEST).write_text(json.dumps(manifest, indent=2))
    _log_event(
        "gr_matcher_cache_generation_done",
        elapsed_seconds=float(manifest["elapsed_seconds"]),
        rows=int(manifest["rows"]),
        features=int(manifest["feature_count"]),
    )
    return manifest


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
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = frame["well"].to_numpy()
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
        splits = cv.split(x_matrix, y, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            t_fold = time.time()
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            _log_event(
                "lgb_fold_start",
                variant=variant_name,
                mode=mode_name,
                model=f"lgb{model_index}",
                fold=int(fold),
                train_rows=int(len(train_idx)),
                valid_rows=int(len(valid_idx)),
                features=int(len(feature_columns)),
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
                        "elapsed_seconds": round(time.time() - t_fold, 3),
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


def run_learned_gr_window_matcher_features_on_exp148(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    learned_feature_path: str | Path | None = None,
    learned_schema_path: str | Path | None = None,
    learned_summary_path: str | Path | None = None,
    projection_config: dict[str, Any] | None = None,
    learned_feature_config: dict[str, Any] | None = None,
    gr_matcher_config: dict[str, Any] | None = None,
    gr_matcher_feature_path: str | Path | None = None,
    gr_matcher_schema_path: str | Path | None = None,
    gr_matcher_summary_path: str | Path | None = None,
    require_gr_matcher_cache: bool = False,
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
    full_frame = frame.reset_index(drop=True)
    if not _keys_aligned(full_frame, projection_features):
        raise ValueError("U-projection features are not in base frame row order")
    full_frame = _append_float32_columns(
        full_frame,
        projection_features.reset_index(drop=True),
        projection_feature_columns,
    )
    del projection_features
    gc.collect()
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config or {},
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
    if _keys_aligned(full_frame, learned_features):
        full_frame = _append_float32_columns(
            full_frame,
            learned_features.reset_index(drop=True),
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
    del learned_features
    if require_gr_matcher_cache or gr_matcher_feature_path is not None:
        del learned_features_source
    gc.collect()
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
    gr_matcher_cache_meta = None
    if require_gr_matcher_cache or gr_matcher_feature_path is not None:
        gr_matcher_features, gr_matcher_group_columns, gr_matcher_summary, gr_matcher_cache_meta = (
            load_gr_matcher_feature_cache(
                gr_matcher_feature_path,
                schema_path=gr_matcher_schema_path,
                summary_path=gr_matcher_summary_path,
            )
        )
    else:
        gr_matcher_features, gr_matcher_group_columns, gr_matcher_summary = (
            build_gr_matcher_window_features(
                full_frame,
                learned_features_source,
                raw_dir=train_dir,
                config=gr_matcher_config or {},
            )
        )
    if gr_matcher_cache_meta is None:
        gr_matcher_cache_meta = {
            "source": "generated_in_train_notebook",
            "rows": int(len(gr_matcher_features)),
            "wells": int(gr_matcher_features["well"].nunique()),
            "feature_count": int(len(gr_matcher_features.columns) - 2),
        }
    gr_matcher_feature_columns = [
        col for col in gr_matcher_features.columns if col not in {"id", "well"}
    ]
    before_prefix_rows = len(full_frame)
    prefix_keys_match = (
        len(gr_matcher_features) == len(full_frame)
        and gr_matcher_features["id"].astype(str).reset_index(drop=True).equals(
            full_frame["id"].astype(str).reset_index(drop=True)
        )
        and gr_matcher_features["well"].astype(str).reset_index(drop=True).equals(
            full_frame["well"].astype(str).reset_index(drop=True)
        )
    )
    if not prefix_keys_match:
        raise ValueError(
            "learned GR matcher features are not in full_frame row order; refusing memory-heavy merge "
            f"on {before_prefix_rows} rows"
        )
    _log_event(
        "gr_matcher_join_start",
        rows=int(before_prefix_rows),
        prefix_features=int(len(gr_matcher_feature_columns)),
    )
    full_frame = _append_float32_columns(
        full_frame.reset_index(drop=True),
        gr_matcher_features.reset_index(drop=True),
        gr_matcher_feature_columns,
    )
    del gr_matcher_features
    gc.collect()
    _log_event(
        "gr_matcher_join_done",
        rows=int(len(full_frame)),
        prefix_features=int(len(gr_matcher_feature_columns)),
    )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **gr_matcher_group_columns,
    }
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )
    gr_matcher_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_gr_matcher_feature_summary.csv",
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
    required_model_columns: set[str] = {"id", "well", "target", "last_known_tvt", "md_since"}
    for variant in selected_variants:
        if not variant.get("enabled", True):
            continue
        required_model_columns.update(
            feature_columns_for_variant(
                base_feature_columns,
                feature_group_columns,
                variant,
            )
        )
    droppable_columns = [
        column for column in full_frame.columns if column not in required_model_columns
    ]
    if droppable_columns:
        full_frame.drop(columns=droppable_columns, inplace=True)
        gc.collect()
        _log_event(
            "train_frame_pruned",
            rows=int(len(full_frame)),
            kept_columns=int(len(full_frame.columns)),
            dropped_columns=int(len(droppable_columns)),
        )

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
                    "is_gr_matcher_feature": bool(feature in gr_matcher_feature_columns),
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
        "experiment": "exp180_learned_gr_window_matcher_features_on_exp148",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "learned_gr_window_matcher_features_on_exp148_full_train_rows",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config or {},
        "gr_matcher_config": gr_matcher_config or {},
        "gr_matcher_feature_source": gr_matcher_cache_meta,
        "projection_feature_groups": projection_group_columns,
        "learned_feature_groups": learned_group_columns,
        "gr_matcher_feature_groups": gr_matcher_group_columns,
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
        "experiment": "exp180_learned_gr_window_matcher_features_on_exp148",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "learned_gr_window_matcher_features_on_exp148_full_train_rows",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "base_surface_parent": "exp092_u_projection_correction_disagreement_fullrun",
        "learned_likelihood_parent": "exp145_learned_likelihood_rawtest_feature_generator_parity",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "feature_join_coverage": coverage_meta,
        "anchor_source": anchor_meta,
        "gr_matcher_feature_source": gr_matcher_cache_meta,
        "active_modes": selected_modes,
        "selected_lgb_config_indices": (
            [int(index) for index in selected_lgb_config_indices]
            if selected_lgb_config_indices is not None
            else None
        ),
        "active_variants": [
            str(variant.get("name")) for variant in selected_variants if variant.get("enabled", True)
        ],
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "projection_feature_summary": f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
            "learned_feature_summary": f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
            "gr_matcher_feature_summary": f"{OUTPUT_PREFIX}_gr_matcher_feature_summary.csv",
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
    gr_matcher_config: dict[str, Any] | None = None,
    variant_name: str = "learned_gr_window_matcher_addonly",
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
    gr_matcher_features, gr_matcher_group_columns, gr_matcher_summary = (
        build_gr_matcher_window_features(
            test_frame,
            rawtest_learned_features,
            raw_dir=test_dir,
            config=gr_matcher_config or dict(manifest.get("gr_matcher_config") or {}),
        )
    )
    gr_matcher_feature_columns = [
        col for col in gr_matcher_features.columns if col not in {"id", "well"}
    ]
    before_prefix_rows = len(test_frame)
    test_frame = test_frame.merge(
        gr_matcher_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_prefix_rows:
        raise ValueError(
            "Raw-test learned GR matcher features do not cover every replay test row: "
            f"{len(test_frame)} of {before_prefix_rows}"
        )
    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
        **gr_matcher_group_columns,
    }
    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from train manifest")
    configured_gr_matcher_groups = manifest.get("gr_matcher_feature_groups") or {}
    if configured_gr_matcher_groups and {
        key: list(value) for key, value in gr_matcher_group_columns.items()
    } != {key: list(value) for key, value in configured_gr_matcher_groups.items()}:
        raise ValueError("Learned GR matcher feature groups differ from train manifest")
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
    gr_matcher_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_gr_matcher_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
                "is_gr_matcher_feature": bool(feature in gr_matcher_feature_columns),
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
        "experiment": "exp180_learned_gr_window_matcher_features_on_exp148",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_raw_test_feature_replay",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "anchor_source": anchor_meta,
        "learned_feature_groups": learned_group_columns,
        "gr_matcher_feature_groups": gr_matcher_group_columns,
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
            "gr_matcher_feature_summary": (
                f"{OUTPUT_PREFIX}_inference_gr_matcher_feature_summary.csv"
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
