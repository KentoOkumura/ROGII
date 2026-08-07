from __future__ import annotations

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
OUTPUT_PREFIX = "exp090_lateral_self_gr_match_pseudotail_probe"
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


def _column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required source column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _finite_ncc_and_l2(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    query_norm = (query - query.mean(axis=1, keepdims=True)) / (
        query.std(axis=1, keepdims=True) + 1e-6
    )
    candidates_norm = (candidates - candidates.mean(axis=1, keepdims=True)) / (
        candidates.std(axis=1, keepdims=True) + 1e-6
    )
    ncc = query_norm @ candidates_norm.T / float(query.shape[1])
    l2 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * ncc))
    return ncc.astype(np.float32), l2.astype(np.float32)


def _multi_scale_self_gr_match(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    eval_indices: np.ndarray,
    half_windows: tuple[int, ...],
    stride: int,
    prefix_tail_rows: int,
) -> dict[str, np.ndarray]:
    n_eval = len(eval_indices)
    prefix_len = len(prefix_tvt)
    smoothed_gr = (
        pd.Series(full_gr, dtype="float32")
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    scale_outputs: dict[int, dict[str, np.ndarray]] = {}
    for half_window in half_windows:
        window = 2 * int(half_window) + 1
        if prefix_len < window + 1 or n_eval == 0:
            scale_outputs[int(half_window)] = {
                "score": np.zeros(n_eval, dtype=np.float32),
                "second_score": np.zeros(n_eval, dtype=np.float32),
                "gap": np.zeros(n_eval, dtype=np.float32),
                "l2": np.full(n_eval, 10.0, dtype=np.float32),
                "matched_tvt": np.full(n_eval, float(prefix_tvt[-1]), dtype=np.float32),
                "matched_idx": np.full(n_eval, prefix_len - 1, dtype=np.float32),
            }
            continue
        start_min = max(0, prefix_len - int(prefix_tail_rows))
        starts = np.arange(start_min, prefix_len - window + 1, int(stride), dtype=np.int32)
        if len(starts) == 0:
            starts = np.array([max(0, prefix_len - window)], dtype=np.int32)
        offsets = np.arange(window, dtype=np.int32)
        candidates = smoothed_gr[starts[:, None] + offsets[None, :]].astype(np.float32)
        padded = np.pad(smoothed_gr, int(half_window), mode="edge")
        query_starts = eval_indices.astype(np.int32)
        query = padded[query_starts[:, None] + offsets[None, :]].astype(np.float32)
        ncc, l2 = _finite_ncc_and_l2(query, candidates)
        best = ncc.argmax(axis=1)
        best_score = ncc[np.arange(n_eval), best].astype(np.float32)
        if ncc.shape[1] > 1:
            second = np.partition(ncc, -2, axis=1)[:, -2].astype(np.float32)
        else:
            second = np.zeros(n_eval, dtype=np.float32)
        best_l2 = l2[np.arange(n_eval), best].astype(np.float32)
        matched_idx = np.clip(starts[best] + int(half_window), 0, prefix_len - 1).astype(np.float32)
        matched_tvt = prefix_tvt[matched_idx.astype(np.int32)].astype(np.float32)
        scale_outputs[int(half_window)] = {
            "score": best_score,
            "second_score": second,
            "gap": (best_score - second).astype(np.float32),
            "l2": best_l2,
            "matched_tvt": matched_tvt,
            "matched_idx": matched_idx,
        }

    score_matrix = np.stack(
        [scale_outputs[int(scale)]["score"] for scale in half_windows],
        axis=1,
    )
    delta_matrix = np.stack(
        [
            scale_outputs[int(scale)]["matched_tvt"] - float(prefix_tvt[-1])
            for scale in half_windows
        ],
        axis=1,
    )
    l2_matrix = np.stack([scale_outputs[int(scale)]["l2"] for scale in half_windows], axis=1)
    best_scale_pos = score_matrix.argmax(axis=1)
    score_weights = np.exp(3.0 * score_matrix)
    score_weights /= score_weights.sum(axis=1, keepdims=True) + 1e-9

    out: dict[str, np.ndarray] = {
        "self_gr_score_mean": score_matrix.mean(axis=1).astype(np.float32),
        "self_gr_score_std": score_matrix.std(axis=1).astype(np.float32),
        "self_gr_score_min": score_matrix.min(axis=1).astype(np.float32),
        "self_gr_score_max": score_matrix.max(axis=1).astype(np.float32),
        "self_gr_score_gap_best_second": np.max(
            np.stack([scale_outputs[int(scale)]["gap"] for scale in half_windows], axis=1),
            axis=1,
        ).astype(np.float32),
        "self_gr_best_scale": np.asarray(half_windows, dtype=np.float32)[best_scale_pos],
        "self_gr_delta_tvt_ens": (delta_matrix * score_weights).sum(axis=1).astype(np.float32),
        "self_gr_delta_tvt_range": (delta_matrix.max(axis=1) - delta_matrix.min(axis=1)).astype(
            np.float32
        ),
        "self_gr_delta_tvt_std": delta_matrix.std(axis=1).astype(np.float32),
        "self_gr_best_l2": l2_matrix.min(axis=1).astype(np.float32),
    }
    for scale in half_windows:
        key = int(scale)
        matched_idx = scale_outputs[key]["matched_idx"]
        out[f"self_gr_sc{key}_score"] = scale_outputs[key]["score"]
        out[f"self_gr_sc{key}_delta_tvt"] = (
            scale_outputs[key]["matched_tvt"] - float(prefix_tvt[-1])
        ).astype(np.float32)
        out[f"self_gr_sc{key}_l2"] = scale_outputs[key]["l2"]
        out[f"self_gr_sc{key}_lag_rows"] = (matched_idx - float(prefix_len - 1)).astype(np.float32)
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float32))
    if series.notna().sum() <= 1:
        return np.zeros(len(series), dtype=np.float32)
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)


def build_self_gr_match_features(
    frame: pd.DataFrame,
    *,
    train_dir: str | Path,
    self_gr_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    self_gr_config = self_gr_config or {}
    train_dir = Path(train_dir)
    half_windows = tuple(int(value) for value in self_gr_config.get("half_windows", [8, 15, 25]))
    stride = int(self_gr_config.get("stride", 3))
    prefix_tail_rows = int(self_gr_config.get("prefix_tail_rows", 1024))
    if not half_windows:
        raise ValueError("self_gr half_windows must not be empty")
    if stride <= 0:
        raise ValueError(f"self_gr stride must be positive, got {stride}")

    result = pd.DataFrame({"id": frame["id"].astype(str), "well": frame["well"].astype(str)})
    result_row_indices = _row_indices_from_ids(result["id"])
    result["_row_idx"] = result_row_indices
    feature_frames: list[pd.DataFrame] = []
    well_summaries: list[dict[str, Any]] = []
    for well, positions in result.groupby("well", sort=False).groups.items():
        well = str(well)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(f"raw train horizontal well file not found: {horizontal_path}")
        horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known_mask = tvt_input.notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No known TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = tvt_input.iloc[:prefix_len].to_numpy(np.float32)
        gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both").fillna(fallback).to_numpy(np.float32)
        )
        row_idx = result.loc[list(positions), "_row_idx"].to_numpy(np.int32)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        outputs = _multi_scale_self_gr_match(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            eval_indices=row_idx,
            half_windows=half_windows,
            stride=stride,
            prefix_tail_rows=prefix_tail_rows,
        )
        well_features = pd.DataFrame(
            {
                "id": result.loc[list(positions), "id"].to_numpy(),
                "well": well,
                "self_gr_available": np.ones(len(row_idx), dtype=np.float32),
                "self_gr_known_prefix_rows": np.full(len(row_idx), prefix_len, dtype=np.float32),
                "self_gr_eval_len": np.full(
                    len(row_idx),
                    max(0, len(horizontal) - prefix_len),
                    dtype=np.float32,
                ),
                "self_gr_prefix_missing_rate": np.full(
                    len(row_idx),
                    float(pd.isna(horizontal["GR"].iloc[:prefix_len]).mean()),
                    dtype=np.float32,
                ),
                "self_gr_eval_missing_rate": np.full(
                    len(row_idx),
                    float(pd.isna(horizontal["GR"].iloc[prefix_len:]).mean()),
                    dtype=np.float32,
                ),
                "self_gr_md_rank_from_anchor": (row_idx - (prefix_len - 1)).astype(np.float32),
            }
        )
        for column, values in outputs.items():
            well_features[column] = np.asarray(values, dtype=np.float32)
        feature_frames.append(well_features)
        well_summaries.append(
            {
                "well": well,
                "rows": int(len(row_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "prefix_missing_rate": float(pd.isna(horizontal["GR"].iloc[:prefix_len]).mean()),
                "eval_missing_rate": float(pd.isna(horizontal["GR"].iloc[prefix_len:]).mean()),
            }
        )

    result = result.drop(columns=["_row_idx"]).merge(
        pd.concat(feature_frames, ignore_index=True),
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    if result["self_gr_available"].isna().any():
        raise ValueError("self-GR feature merge produced missing rows")

    group_columns = {
        str(group): [str(column) for column in columns]
        for group, columns in (self_gr_config.get("feature_groups") or {}).items()
    }
    if not group_columns:
        multiscale_columns: list[str] = []
        for scale in half_windows:
            scale_key = int(scale)
            multiscale_columns.extend(
                [
                    f"self_gr_sc{scale_key}_score",
                    f"self_gr_sc{scale_key}_delta_tvt",
                    f"self_gr_sc{scale_key}_l2",
                ]
            )
        group_columns = {
            "self_gr_core": [
                "self_gr_score_max",
                "self_gr_score_gap_best_second",
                "self_gr_delta_tvt_ens",
                "self_gr_best_l2",
                "self_gr_best_scale",
            ],
            "self_gr_multiscale": multiscale_columns,
            "self_gr_context": [
                "self_gr_known_prefix_rows",
                "self_gr_eval_len",
                "self_gr_prefix_missing_rate",
                "self_gr_eval_missing_rate",
                "self_gr_md_rank_from_anchor",
                "self_gr_delta_tvt_range",
                "self_gr_delta_tvt_std",
            ],
        }
    for group, columns in group_columns.items():
        missing_group = [column for column in columns if column not in result.columns]
        if missing_group:
            raise ValueError(
                f"self-GR feature group {group!r} has missing columns: {missing_group}"
            )

    numeric_cols = [col for col in result.columns if col not in {"id", "well"}]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    if not np.isfinite(result[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("self-GR feature frame contains non-finite values")

    summary = pd.DataFrame(
        [
            {
                "feature": column,
                "mean": float(result[column].mean()),
                "std": float(result[column].std()),
                "p50": float(result[column].quantile(0.50)),
                "p90": float(result[column].quantile(0.90)),
                "p95": float(result[column].quantile(0.95)),
                "max": float(result[column].max()),
            }
            for column in numeric_cols
        ]
    )
    if well_summaries:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "feature": "_well_summary",
                            "mean": float(len(well_summaries)),
                            "std": float(np.mean([row["rows"] for row in well_summaries])),
                            "p50": float(
                                np.median([row["known_prefix_rows"] for row in well_summaries])
                            ),
                            "p90": float(
                                np.quantile([row["eval_len"] for row in well_summaries], 0.90)
                            ),
                            "p95": float(
                                np.quantile(
                                    [row["eval_missing_rate"] for row in well_summaries], 0.95
                                )
                            ),
                            "max": float(max(row["eval_len"] for row in well_summaries)),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return result, group_columns, summary


def build_sample_weight(
    frame: pd.DataFrame,
    *,
    policy_name: str | None,
    policy_config: dict[str, Any] | None,
) -> np.ndarray | None:
    if not policy_name:
        return None
    policy_config = policy_config or {}
    score_column = str(policy_config.get("score_column") or "self_gr_score_max")
    if score_column not in frame.columns:
        raise ValueError(
            f"sample weight policy {policy_name!r} missing score column: {score_column}"
        )
    score = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.5).to_numpy(np.float32)
    low_weight = float(policy_config.get("low_weight", 0.65))
    high_weight = float(policy_config.get("high_weight", 1.10))
    score = np.clip(score, 0.0, 1.0)
    weights = high_weight - (high_weight - low_weight) * score
    weights = np.clip(weights, min(low_weight, high_weight), max(low_weight, high_weight))
    if bool(policy_config.get("normalize_mean", True)):
        mean_weight = float(np.mean(weights))
        if np.isfinite(mean_weight) and mean_weight > 0:
            weights = weights / mean_weight
    return weights.astype(np.float32)


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
            raise ValueError(f"Unknown self-GR feature group for variant {variant}: {group}")
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
    sample_weight: np.ndarray | None,
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
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if len(sample_weight) != len(frame):
            raise ValueError(f"sample_weight length mismatch: {len(sample_weight)} != {len(frame)}")
        if not np.isfinite(sample_weight).all() or float(np.min(sample_weight)) <= 0:
            raise ValueError("sample_weight must be finite and strictly positive")
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
            train_sample_weight = sample_weight[train_idx] if sample_weight is not None else None
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y[train_idx],
                sample_weight=train_sample_weight,
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
                    "self_gr_feature_groups": ",".join(variant.get("feature_groups") or []),
                    "sample_weight_policy": variant.get("sample_weight_policy"),
                    "sample_weight_mean": (
                        float(np.mean(train_sample_weight))
                        if train_sample_weight is not None
                        else 1.0
                    ),
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
                "self_gr_feature_groups": ",".join(variant.get("feature_groups") or []),
                "sample_weight_policy": variant.get("sample_weight_policy"),
                "sample_weight_mean": (
                    float(np.mean(sample_weight)) if sample_weight is not None else 1.0
                ),
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
            "self_gr_feature_groups": ",".join(variant.get("feature_groups") or []),
            "sample_weight_policy": variant.get("sample_weight_policy"),
            "sample_weight_mean": (
                float(np.mean(sample_weight)) if sample_weight is not None else 1.0
            ),
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
        "self_gr_feature_groups": list(variant.get("feature_groups") or []),
        "sample_weight_policy": variant.get("sample_weight_policy"),
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


def run_lateral_self_gr_match_pseudotail_probe(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    self_gr_config: dict[str, Any] | None = None,
    sample_weight_config: dict[str, Any] | None = None,
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
    self_gr_config = self_gr_config or {}
    sample_weight_config = sample_weight_config or {}
    self_gr_features, self_gr_group_columns, self_gr_summary = build_self_gr_match_features(
        frame,
        train_dir=train_dir,
        self_gr_config=self_gr_config,
    )
    self_gr_feature_columns = [col for col in self_gr_features.columns if col not in {"id", "well"}]
    full_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            self_gr_features[self_gr_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    self_gr_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_self_gr_feature_summary.csv",
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
    sample_weight_summary_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        if not variant.get("enabled", True):
            continue
        variant_name = str(variant["name"])
        feature_columns = feature_columns_for_variant(
            base_feature_columns,
            self_gr_group_columns,
            variant,
        )
        policy_name = variant.get("sample_weight_policy")
        policies = sample_weight_config.get("policies") or {}
        policy_config = policies.get(policy_name) if policy_name else None
        sample_weight = build_sample_weight(
            full_frame,
            policy_name=policy_name,
            policy_config=policy_config,
        )
        if sample_weight is not None:
            sample_weight_summary_rows.append(
                {
                    "variant": variant_name,
                    "sample_weight_policy": policy_name,
                    "rows": int(len(sample_weight)),
                    "mean": float(np.mean(sample_weight)),
                    "std": float(np.std(sample_weight)),
                    "min": float(np.min(sample_weight)),
                    "p05": float(np.quantile(sample_weight, 0.05)),
                    "p50": float(np.quantile(sample_weight, 0.50)),
                    "p95": float(np.quantile(sample_weight, 0.95)),
                    "max": float(np.max(sample_weight)),
                }
            )
        else:
            sample_weight_summary_rows.append(
                {
                    "variant": variant_name,
                    "sample_weight_policy": None,
                    "rows": int(len(full_frame)),
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "p05": 1.0,
                    "p50": 1.0,
                    "p95": 1.0,
                    "max": 1.0,
                }
            )
        for index, feature in enumerate(feature_columns):
            feature_schema_rows.append(
                {
                    "variant": variant_name,
                    "feature_index": int(index),
                    "feature": feature,
                    "is_self_gr_feature": bool(feature in self_gr_feature_columns),
                    "sample_weight_policy": policy_name,
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
                sample_weight=sample_weight,
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
    pd.DataFrame(sample_weight_summary_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_sample_weight_summary.csv",
        index=False,
    )
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
        "experiment": "exp090_lateral_self_gr_match_pseudotail_probe",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "lateral_self_gr_match_pseudotail_probe_from_exp072_cache",
        "feature_source": feature_meta,
        "train_dir": str(train_dir),
        "self_gr_config": self_gr_config,
        "self_gr_feature_groups": self_gr_group_columns,
        "sample_weight_config": sample_weight_config,
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
        "experiment": "exp090_lateral_self_gr_match_pseudotail_probe",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "lateral_self_gr_match_pseudotail_probe_from_exp072_cache",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "active_modes": selected_modes,
        "active_variants": variant_names,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "self_gr_feature_summary": f"{OUTPUT_PREFIX}_self_gr_feature_summary.csv",
            "sample_weight_summary": f"{OUTPUT_PREFIX}_sample_weight_summary.csv",
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
