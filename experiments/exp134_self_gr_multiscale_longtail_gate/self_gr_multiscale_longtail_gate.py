from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp134_self_gr_multiscale_longtail_gate"
EXP072_FEATURE_CACHE = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    if isinstance(value, list | tuple):
        return [Path(item) for item in value if item]
    return []


def find_input_file(
    filename: str,
    configured: Any = None,
    *,
    local_roots: list[Path] | None = None,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    candidates.extend(as_path_list(configured))
    for root in local_roots or []:
        candidates.extend([root / filename, root / "artifacts" / filename])
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.glob(f"**/{filename}")))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if required:
        checked = "\n".join(str(path) for path in candidates[:100])
        raise FileNotFoundError(f"input file not found or empty: {filename}. Checked:\n{checked}")
    return None


def resolve_train_dir(config: dict[str, Any], paths: Any) -> Path:
    train_dir = getattr(paths, "train_data_dir", None)
    if train_dir is not None and Path(train_dir).exists():
        return Path(train_dir)
    configured = get_nested(config, "data.train_dir", "data/raw/train")
    path = Path(str(configured))
    if path.exists():
        return path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in sorted(input_root.glob("**/train")):
            if candidate.is_dir() and list(candidate.glob("*__horizontal_well.csv")):
                return candidate
    raise FileNotFoundError(f"Could not resolve raw train directory from {configured}")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame[columns].itertuples(index=False, name=None):
        digest.update(("\t".join(map(str, row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def rmse_from_error(error: pd.Series | np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def mae_from_error(error: pd.Series | np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.abs(values[finite])))


def parse_tail_rank(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce")
    if values.isna().any():
        bad = ids[values.isna()].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype("int64")


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def tail_rank_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def safe_qcut(values: pd.Series | np.ndarray, q: int, *, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.nunique() < 2:
        return pd.Series([f"{prefix}_all"] * len(numeric), index=numeric.index, dtype="object")
    ranks = numeric.rank(method="first")
    try:
        cut = pd.qcut(ranks, q=q, labels=[f"{prefix}_q{i + 1}" for i in range(q)])
    except ValueError:
        return pd.Series([f"{prefix}_all"] * len(numeric), index=numeric.index, dtype="object")
    return cut.astype("object").where(numeric.notna(), f"{prefix}_missing")


def rank01(values: pd.Series | np.ndarray, *, ascending: bool = True) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    finite = series[np.isfinite(series)]
    if finite.nunique() <= 1:
        return np.full(len(series), 0.5, dtype=np.float32)
    ranked = series.rank(method="average", pct=True, ascending=ascending).fillna(0.5)
    return ranked.to_numpy(np.float32)


def finite_ncc_and_l2(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def multi_scale_self_gr_match(
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
        query = padded[eval_indices.astype(np.int32)[:, None] + offsets[None, :]].astype(np.float32)
        ncc, l2 = finite_ncc_and_l2(query, candidates)
        best = ncc.argmax(axis=1)
        matched_idx = np.clip(starts[best] + int(half_window), 0, prefix_len - 1).astype(np.float32)
        scale_outputs[int(half_window)] = {
            "score": ncc[np.arange(n_eval), best].astype(np.float32),
            "l2": l2[np.arange(n_eval), best].astype(np.float32),
            "matched_tvt": prefix_tvt[matched_idx.astype(np.int32)].astype(np.float32),
            "matched_idx": matched_idx,
        }

    score_matrix = np.stack([scale_outputs[int(scale)]["score"] for scale in half_windows], axis=1)
    delta_matrix = np.stack(
        [
            scale_outputs[int(scale)]["matched_tvt"] - float(prefix_tvt[-1])
            for scale in half_windows
        ],
        axis=1,
    )
    l2_matrix = np.stack([scale_outputs[int(scale)]["l2"] for scale in half_windows], axis=1)
    out: dict[str, np.ndarray] = {
        "self_gr_score_max": score_matrix.max(axis=1).astype(np.float32),
        "self_gr_score_mean": score_matrix.mean(axis=1).astype(np.float32),
        "self_gr_score_std": score_matrix.std(axis=1).astype(np.float32),
        "self_gr_delta_tvt_std": delta_matrix.std(axis=1).astype(np.float32),
        "self_gr_delta_tvt_range": (delta_matrix.max(axis=1) - delta_matrix.min(axis=1)).astype(
            np.float32
        ),
        "self_gr_best_l2": l2_matrix.min(axis=1).astype(np.float32),
    }
    for scale in half_windows:
        key = int(scale)
        out[f"self_gr_sc{key}_score"] = scale_outputs[key]["score"]
        out[f"self_gr_sc{key}_delta_tvt"] = (
            scale_outputs[key]["matched_tvt"] - float(prefix_tvt[-1])
        ).astype(np.float32)
        out[f"self_gr_sc{key}_l2"] = scale_outputs[key]["l2"]
        out[f"self_gr_sc{key}_lag_rows"] = (
            scale_outputs[key]["matched_idx"] - float(prefix_len - 1)
        ).astype(np.float32)
    return out


def load_feature_cache(
    config: dict[str, Any], *, max_rows: int | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_input_file(
        EXP072_FEATURE_CACHE,
        get_nested(config, "data.exp072_feature_cache"),
        local_roots=[
            Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts"),
            Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1"),
        ],
    )
    assert source is not None
    header = pd.read_csv(source, nrows=0).columns.tolist()
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "pf_ancc",
        "beam_mean_d",
        "likpf_mean_d",
    ]
    optional = [
        "pf_ancc_std",
        "beam_std_d",
        "sc_ens_d",
        "hyb_d",
        "tvt_dense_d",
        "tvt_dense50_d",
        "tvt_densew_d",
        "pf_vs_dense",
        "dense_dist",
        "md_since",
    ]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    usecols = required + [col for col in optional if col in header]
    frame = pd.read_csv(
        source,
        usecols=usecols,
        dtype={"id": "string", "well": "string"},
        nrows=max_rows,
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for col in frame.columns:
        if col not in {"id", "well"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    schema_path = find_input_file(
        EXP072_SCHEMA,
        get_nested(config, "data.exp072_feature_schema"),
        local_roots=[Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts")],
        required=False,
    )
    return frame, {
        "path": str(source),
        "raw_file_sha256": sha256_path(source),
        "decompressed_content_sha256": sha256_path(source, decompressed=True),
        "schema_path": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": usecols,
    }


def add_raw_context(
    frame: pd.DataFrame,
    *,
    train_dir: Path,
    self_gr_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    half_windows = tuple(int(value) for value in self_gr_config.get("half_windows", [8, 15, 25]))
    stride = int(self_gr_config.get("stride", 3))
    prefix_tail_rows = int(self_gr_config.get("prefix_tail_rows", 1024))
    rows: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    tail_rank = parse_tail_rank(frame["id"])
    base = frame[["id", "well"]].copy()
    base["tail_rank"] = tail_rank.to_numpy(np.int64)
    for well, group in base.groupby("well", sort=False):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train horizontal well not found: {path}")
        raw = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
        for col in ["MD", "Z", "GR", "TVT_input"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        known_mask = raw["TVT_input"].notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = raw["TVT_input"].iloc[:prefix_len].to_numpy(np.float32)
        gr_series = raw["GR"]
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both").fillna(fallback).to_numpy(np.float32)
        )
        idx = group["tail_rank"].to_numpy(np.int64)
        if idx.min(initial=0) < 0 or idx.max(initial=0) >= len(raw):
            raise ValueError(f"row index out of range for well {well}")
        outputs = multi_scale_self_gr_match(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            eval_indices=idx.astype(np.int32),
            half_windows=half_windows,
            stride=stride,
            prefix_tail_rows=prefix_tail_rows,
        )
        selected = raw.iloc[idx].copy()
        part = pd.DataFrame(
            {
                "id": group["id"].to_numpy(),
                "well": str(well),
                "tail_rank": idx.astype(np.int64),
                "raw_md": selected["MD"].to_numpy(np.float32),
                "raw_z": selected["Z"].to_numpy(np.float32),
                "raw_gr": selected["GR"].to_numpy(np.float32),
                "md_since_raw": (
                    selected["MD"].to_numpy(np.float32) - np.float32(raw.loc[prefix_len - 1, "MD"])
                ),
                "self_gr_known_prefix_rows": np.full(len(idx), prefix_len, dtype=np.float32),
                "self_gr_eval_len": np.full(len(idx), len(raw) - prefix_len, dtype=np.float32),
                "self_gr_prefix_missing_rate": np.full(
                    len(idx), float(raw["GR"].iloc[:prefix_len].isna().mean()), dtype=np.float32
                ),
                "self_gr_eval_missing_rate": np.full(
                    len(idx), float(raw["GR"].iloc[prefix_len:].isna().mean()), dtype=np.float32
                ),
                "self_gr_md_rank_from_anchor": (idx - (prefix_len - 1)).astype(np.float32),
            }
        )
        for column, values in outputs.items():
            part[column] = np.asarray(values, dtype=np.float32)
        rows.append(part)
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(idx)),
                "prefix_length": int(prefix_len),
                "tail_length": int(len(raw) - prefix_len),
                "prefix_gr_missing_rate": float(raw["GR"].iloc[:prefix_len].isna().mean()),
                "eval_gr_missing_rate": float(raw["GR"].iloc[prefix_len:].isna().mean()),
            }
        )
    context = pd.concat(rows, ignore_index=True)
    well_summary = pd.DataFrame(well_rows)
    numeric_cols = [col for col in context.columns if col not in {"id", "well"}]
    signal_summary = pd.DataFrame(
        [
            {
                "feature": column,
                "mean": float(context[column].mean()),
                "std": float(context[column].std()),
                "p50": float(context[column].quantile(0.50)),
                "p90": float(context[column].quantile(0.90)),
                "p95": float(context[column].quantile(0.95)),
                "max": float(context[column].max()),
            }
            for column in numeric_cols
        ]
    )
    return context, well_summary, signal_summary


def load_optional_prediction(
    config: dict[str, Any],
    *,
    key: str,
    filename: str,
    selector_col: str,
    selector_value: str,
    pred_col: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = find_input_file(filename, get_nested(config, f"data.{key}"), required=False)
    if path is None:
        return None, {"available": False, "key": key, "filename": filename}
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["id", "well", selector_col, "pred_tvt"]
    missing = [col for col in required if col not in header]
    if missing:
        return None, {
            "available": False,
            "key": key,
            "path": str(path),
            "reason": f"missing columns: {missing}",
        }
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=required,
        dtype={"id": str, "well": str, selector_col: str},
        chunksize=500_000,
        low_memory=False,
    ):
        part = chunk[chunk[selector_col].astype(str).eq(selector_value)].copy()
        if not part.empty:
            chunks.append(part)
        if max_rows is not None and sum(len(item) for item in chunks) >= max_rows:
            break
    if not chunks:
        return None, {
            "available": False,
            "key": key,
            "path": str(path),
            "reason": f"selector {selector_col}={selector_value} not found",
        }
    frame = pd.concat(chunks, ignore_index=True)
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    frame = frame.rename(columns={"pred_tvt": pred_col})
    frame[pred_col] = pd.to_numeric(frame[pred_col], errors="coerce").astype("float32")
    return frame[["id", "well", pred_col]], {
        "available": True,
        "key": key,
        "path": str(path),
        "raw_file_sha256": sha256_path(path),
        "decompressed_content_sha256": sha256_path(path, decompressed=True),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "selector_col": selector_col,
        "selector_value": selector_value,
        "prediction_column": pred_col,
    }


def add_candidate_predictions(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = frame.copy()
    out["target_tvt"] = out["last_known_tvt"] + out["target"]
    out["pred_last_anchor_tvt"] = out["last_known_tvt"]
    out["pred_likpf_mean"] = out["last_known_tvt"] + out["likpf_mean_d"]
    out["pred_pf_ancc"] = out["pf_ancc"]
    out["pred_beam_mean"] = out["last_known_tvt"] + out["beam_mean_d"]
    if "tvt_dense_d" in out.columns:
        out["pred_tvt_dense"] = out["last_known_tvt"] + out["tvt_dense_d"]
    elif "pf_vs_dense" in out.columns:
        out["pred_tvt_dense"] = out["pf_ancc"] - out["pf_vs_dense"]
    else:
        out["pred_tvt_dense"] = np.nan

    metas: dict[str, Any] = {}
    exp073, metas["exp073_predictions"] = load_optional_prediction(
        config,
        key="exp073_predictions",
        filename="exp063_full_replay_repro_guard_predictions.csv.gz",
        selector_col="model",
        selector_value="lgb_mean",
        pred_col="pred_exp073_lgb_mean",
        max_rows=max_rows,
    )
    exp092, metas["exp092_predictions"] = load_optional_prediction(
        config,
        key="exp092_predictions",
        filename="exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz",
        selector_col="model",
        selector_value="lgb1",
        pred_col="pred_exp092_lgb1",
        max_rows=max_rows,
    )
    for optional in [exp073, exp092]:
        if optional is not None:
            out = out.merge(optional, on=["id", "well"], how="left", validate="one_to_one")

    candidate_cols = ["pred_last_anchor_tvt", "pred_likpf_mean", "pred_pf_ancc", "pred_beam_mean"]
    if out["pred_tvt_dense"].notna().any():
        candidate_cols.append("pred_tvt_dense")
    for col in ["pred_exp073_lgb_mean", "pred_exp092_lgb1"]:
        if col in out.columns and out[col].notna().any():
            candidate_cols.append(col)
    for col in candidate_cols:
        out[f"{col}_error"] = out[col] - out["target_tvt"]
        out[f"{col}_abs_error"] = out[f"{col}_error"].abs()
    return out, candidate_cols, metas


def add_gate_signals(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    focus = int(get_nested(config, "self_gr.focus_half_window", 25))
    score_col = f"self_gr_sc{focus}_score"
    l2_col = f"self_gr_sc{focus}_l2"
    for col in [score_col, l2_col, "self_gr_delta_tvt_std"]:
        if col not in out.columns:
            raise ValueError(f"missing self-GR signal column: {col}")
    weights = get_nested(config, "self_gr.quality_score", {}) or {}
    score_w = float(weights.get("score_rank_weight", 0.45))
    l2_w = float(weights.get("l2_inverse_rank_weight", 0.35))
    scale_w = float(weights.get("scale_agreement_weight", 0.20))
    out["self_gr_score_rank"] = rank01(out[score_col], ascending=True)
    out["self_gr_l2_inverse_rank"] = rank01(out[l2_col], ascending=False)
    out["self_gr_scale_agreement_rank"] = rank01(
        out["self_gr_delta_tvt_std"].abs(), ascending=False
    )
    total_w = score_w + l2_w + scale_w
    out["self_gr_quality_score"] = (
        score_w * out["self_gr_score_rank"]
        + l2_w * out["self_gr_l2_inverse_rank"]
        + scale_w * out["self_gr_scale_agreement_rank"]
    ) / max(total_w, 1e-9)
    out["md_since"] = (
        pd.to_numeric(out["md_since"], errors="coerce")
        if "md_since" in out.columns
        else pd.to_numeric(out["md_since_raw"], errors="coerce")
    )
    out["distance_bucket"] = distance_bucket(out["md_since"])
    out["tail_rank_bucket"] = tail_rank_bucket(out["tail_rank"])
    out["gr_missing_bucket"] = safe_qcut(out["self_gr_eval_missing_rate"], 4, prefix="gr_missing")
    out["self_gr_quality_bucket"] = safe_qcut(
        out["self_gr_quality_score"], 4, prefix="self_gr_quality"
    )
    out["pf_beam_abs_diff"] = (out["pred_pf_ancc"] - out["pred_beam_mean"]).abs()
    if out["pred_tvt_dense"].notna().any():
        out["pf_dense_abs_diff"] = (out["pred_likpf_mean"] - out["pred_tvt_dense"]).abs()
    else:
        out["pf_dense_abs_diff"] = np.nan
    out["pf_beam_disagreement_bucket"] = safe_qcut(
        out["pf_beam_abs_diff"], 4, prefix="pf_beam_diff"
    )
    out["pf_dense_disagreement_bucket"] = safe_qcut(
        out["pf_dense_abs_diff"], 4, prefix="pf_dense_diff"
    )
    return out


def apply_gate_variants(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    variants = get_nested(config, "audit.gate_variants", []) or []
    gate_prediction_cols: list[str] = []
    pf_dense_values = pd.to_numeric(out["pf_dense_abs_diff"], errors="coerce")
    for spec in variants:
        name = str(spec["name"])
        if not out["pred_tvt_dense"].notna().any():
            out[f"gate_flag_{name}"] = False
            out[f"pred_{name}"] = out["pred_likpf_mean"]
            gate_prediction_cols.append(f"pred_{name}")
            continue
        min_md_since = float(spec.get("min_md_since", 1000.0))
        pf_dense_quantile = float(spec.get("pf_dense_quantile", 0.75))
        self_quantile = spec.get("self_gr_quality_quantile")
        pf_dense_threshold = float(pf_dense_values.quantile(pf_dense_quantile))
        mask = out["md_since"].ge(min_md_since) & pf_dense_values.ge(pf_dense_threshold)
        if self_quantile is not None:
            self_threshold = float(out["self_gr_quality_score"].quantile(float(self_quantile)))
            mask = mask & out["self_gr_quality_score"].ge(self_threshold)
        out[f"gate_flag_{name}"] = mask.fillna(False).astype(bool)
        out[f"pred_{name}"] = np.where(
            out[f"gate_flag_{name}"], out["pred_tvt_dense"], out["pred_likpf_mean"]
        ).astype(np.float32)
        out[f"pred_{name}_error"] = out[f"pred_{name}"] - out["target_tvt"]
        out[f"pred_{name}_abs_error"] = out[f"pred_{name}_error"].abs()
        gate_prediction_cols.append(f"pred_{name}")
    return out, gate_prediction_cols


def summarize_predictions(frame: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_error = frame["pred_likpf_mean_error"]
    for col in prediction_cols:
        error_col = f"{col}_error"
        abs_error_col = f"{col}_abs_error"
        if error_col not in frame.columns:
            frame[error_col] = frame[col] - frame["target_tvt"]
            frame[abs_error_col] = frame[error_col].abs()
        gate_flag_col = col.replace("pred_", "gate_flag_", 1)
        gate_rate = float(frame[gate_flag_col].mean()) if gate_flag_col in frame.columns else 1.0
        rows.append(
            {
                "prediction": col,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "rmse_tvt": rmse_from_error(frame[error_col]),
                "mae_tvt": mae_from_error(frame[error_col]),
                "bias_tvt": float(frame[error_col].mean()),
                "within_5ft": float(frame[abs_error_col].le(5.0).mean()),
                "within_10ft": float(frame[abs_error_col].le(10.0).mean()),
                "gate_rate": gate_rate,
                "delta_rmse_vs_likpf": rmse_from_error(frame[error_col])
                - rmse_from_error(baseline_error),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_tvt").reset_index(drop=True)


def summarize_by_well(frame: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_rmse = frame.groupby("well")["pred_likpf_mean_error"].apply(rmse_from_error)
    for col in prediction_cols:
        error_col = f"{col}_error"
        per_well = frame.groupby("well")[error_col].apply(rmse_from_error)
        for well, value in per_well.items():
            rows.append(
                {
                    "well": well,
                    "prediction": col,
                    "rows": int((frame["well"] == well).sum()),
                    "rmse_tvt": float(value),
                    "delta_rmse_vs_likpf": float(value - baseline_rmse.loc[well]),
                }
            )
    return pd.DataFrame(rows)


def summarize_buckets(
    frame: pd.DataFrame,
    prediction_cols: list[str],
    group_specs: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, cols in group_specs:
        grouped = frame.groupby(cols, observed=True, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, Any] = {
                "group_family": family,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
            }
            for col, key in zip(cols, keys, strict=False):
                record[col] = key
            for pred in prediction_cols:
                error_col = f"{pred}_error"
                record[f"{pred}_rmse"] = rmse_from_error(group[error_col])
                record[f"{pred}_mae"] = mae_from_error(group[error_col])
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_common_worst(
    frame: pd.DataFrame,
    prediction_cols: list[str],
    top_ns: list[int],
) -> pd.DataFrame:
    by_well = frame.groupby("well")["pred_likpf_mean_error"].apply(rmse_from_error)
    rows: list[dict[str, Any]] = []
    for top_n in top_ns:
        worst = set(by_well.sort_values(ascending=False).head(int(top_n)).index.astype(str))
        subset = frame[frame["well"].isin(worst)]
        for pred in prediction_cols:
            error_col = f"{pred}_error"
            rows.append(
                {
                    "scope": f"likpf_worst_top{int(top_n)}_wells",
                    "prediction": pred,
                    "rows": int(len(subset)),
                    "wells": int(subset["well"].nunique()),
                    "rmse_tvt": rmse_from_error(subset[error_col]),
                    "mae_tvt": mae_from_error(subset[error_col]),
                    "delta_rmse_vs_likpf": rmse_from_error(subset[error_col])
                    - rmse_from_error(subset["pred_likpf_mean_error"]),
                }
            )
    return pd.DataFrame(rows)


def write_readme(summary: dict[str, Any], output_dir: Path) -> None:
    decision = summary["decision"]
    lines = [
        "# exp134_self_gr_multiscale_longtail_gate",
        "",
        "## 状態",
        "",
        f"- status: `{summary['status']}`",
        "- LightGBM 学習なし、submission なしの train-side posthoc audit。",
        "",
        "## 仮説",
        "",
        "exp090 の self-GR multiscale signal は単独では弱いが、",
        "longtail / high PF-dense disagreement regime で dense candidate を",
        "信用してよいかの補助 confidence になる可能性がある。",
        "",
        "## 検証方針",
        "",
        "- exp072 feature cache と raw train horizontal GR から",
        "  self-GR multiscale signal を再生成する。",
        "- `likpf_mean` と `tvt_dense` の low-frequency gate を posthoc に比較する。",
        "- overall RMSE だけでなく、distance / tail / PF-dense disagreement /",
        "  self-GR quality / common-worst / by-well regression を見る。",
        "",
        "## 所見",
        "",
        f"- recommendation: `{decision['recommendation']}`",
        f"- reason: {decision['reason']}",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def run_audit(config: dict[str, Any], paths: Any, *, max_rows: int | None = None) -> dict[str, Any]:
    started = time.time()
    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rows = max_rows if max_rows is not None else get_nested(config, "audit.max_rows")
    max_rows = None if max_rows is None else int(max_rows)

    cache, cache_meta = load_feature_cache(config, max_rows=max_rows)
    train_dir = resolve_train_dir(config, paths)
    self_context, well_summary, signal_summary = add_raw_context(
        cache,
        train_dir=train_dir,
        self_gr_config=get_nested(config, "self_gr", {}) or {},
    )
    frame = cache.merge(self_context, on=["id", "well"], how="inner", validate="one_to_one")
    if "md_since" not in frame.columns:
        frame["md_since"] = frame["md_since_raw"]
    frame, candidate_cols, prediction_meta = add_candidate_predictions(
        frame, config, max_rows=max_rows
    )
    frame = add_gate_signals(frame, config)
    frame, gate_prediction_cols = apply_gate_variants(frame, config)
    prediction_cols = candidate_cols + gate_prediction_cols

    metrics = summarize_predictions(frame, prediction_cols)
    by_well = summarize_by_well(frame, prediction_cols)
    common_worst = summarize_common_worst(
        frame,
        prediction_cols,
        [int(value) for value in get_nested(config, "audit.common_worst_top_n", [26, 50])],
    )
    bucket_metrics = summarize_buckets(
        frame,
        prediction_cols,
        [
            ("distance", ["distance_bucket"]),
            ("tail_rank", ["tail_rank_bucket"]),
            ("self_gr_quality", ["self_gr_quality_bucket"]),
            ("pf_dense_disagreement", ["pf_dense_disagreement_bucket"]),
            ("gr_missing", ["gr_missing_bucket"]),
            ("distance_x_pf_dense", ["distance_bucket", "pf_dense_disagreement_bucket"]),
            ("distance_x_self_gr", ["distance_bucket", "self_gr_quality_bucket"]),
        ],
    )

    gate_cols = [col for col in frame.columns if col.startswith("gate_flag_")]
    prediction_sample_cols = [
        "id",
        "well",
        "target_tvt",
        "md_since",
        "distance_bucket",
        "tail_rank_bucket",
        "pf_dense_abs_diff",
        "self_gr_quality_score",
        "self_gr_sc25_score",
        "self_gr_sc25_delta_tvt",
        "self_gr_sc25_l2",
        "pred_likpf_mean",
        "pred_tvt_dense",
        *gate_cols,
        *gate_prediction_cols,
    ]
    prediction_sample = frame[prediction_sample_cols].copy()

    artifact_paths = {
        "metrics": output_dir / f"{OUTPUT_PREFIX}_metrics.csv",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "bucket_metrics": output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
        "signal_metrics": output_dir / f"{OUTPUT_PREFIX}_signal_metrics.csv",
        "common_worst_metrics": output_dir / f"{OUTPUT_PREFIX}_common_worst_metrics.csv",
        "gate_predictions": output_dir / f"{OUTPUT_PREFIX}_gate_predictions.csv.gz",
        "feature_schema": output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    metrics.to_csv(artifact_paths["metrics"], index=False)
    by_well.to_csv(artifact_paths["by_well"], index=False)
    bucket_metrics.to_csv(artifact_paths["bucket_metrics"], index=False)
    signal_summary.to_csv(artifact_paths["signal_metrics"], index=False)
    common_worst.to_csv(artifact_paths["common_worst_metrics"], index=False)
    prediction_sample.to_csv(artifact_paths["gate_predictions"], index=False, compression="gzip")
    pd.DataFrame(
        {
            "feature": [
                "self_gr_sc25_delta_tvt",
                "self_gr_sc25_score",
                "self_gr_sc25_l2",
                "self_gr_quality_score",
                "pf_dense_abs_diff",
                "md_since",
                "self_gr_eval_missing_rate",
            ],
            "role": [
                "self_gr_signal",
                "self_gr_signal",
                "self_gr_signal",
                "derived_gate_score",
                "disagreement_context",
                "distance_context",
                "gr_missingness_context",
            ],
        }
    ).to_csv(artifact_paths["feature_schema"], index=False)

    likpf_rmse = float(metrics.loc[metrics["prediction"].eq("pred_likpf_mean"), "rmse_tvt"].iloc[0])
    best = metrics.iloc[0].to_dict()
    gate_metrics = metrics[metrics["prediction"].isin(gate_prediction_cols)].copy()
    if gate_metrics.empty:
        best_gate: dict[str, Any] | None = None
        best_gate_delta = float("inf")
    else:
        best_gate = gate_metrics.iloc[0].to_dict()
        best_gate_delta = float(best_gate["rmse_tvt"] - likpf_rmse)
    if best_gate is None or best_gate_delta >= 0.0:
        recommendation = "reject_self_gr_gate"
        reason = "No configured self-GR auxiliary gate beat likpf_mean on overall RMSE."
    elif best_gate_delta < -0.01:
        recommendation = "consider_addonly_or_ranker_ablation"
        reason = (
            "A self-GR gated diagnostic improved likpf_mean enough to justify "
            "feature/ranker follow-up, but not submission."
        )
    else:
        recommendation = "diagnostic_only"
        reason = (
            "Best diagnostic improvement is too small for inference; "
            "keep as confidence readout only."
        )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_posthoc_audit",
        "runtime_seconds": time.time() - started,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "input_cache": cache_meta,
        "optional_predictions": prediction_meta,
        "raw_train_dir": str(train_dir),
        "raw_well_summary": {
            "wells": int(len(well_summary)),
            "tail_length_min": int(well_summary["tail_length"].min()),
            "tail_length_max": int(well_summary["tail_length"].max()),
            "eval_gr_missing_rate_p95": float(well_summary["eval_gr_missing_rate"].quantile(0.95)),
        },
        "candidate_predictions": prediction_cols,
        "gate_predictions_sha256": sha256_path(
            artifact_paths["gate_predictions"], decompressed=True
        ),
        "gate_prediction_content_sha256": dataframe_content_sha(
            prediction_sample.head(min(len(prediction_sample), 200000)),
            ["id", "well", "self_gr_quality_score", "pred_likpf_mean", "pred_tvt_dense"],
        ),
        "best_prediction": to_jsonable(best),
        "best_gate_prediction": to_jsonable(best_gate),
        "decision": {"recommendation": recommendation, "reason": reason},
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }
    artifact_paths["summary"].write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    write_readme(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp134 self-GR multiscale gate audit.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    import yaml
    from settings import ExperimentPaths

    with Path(args.config).open() as fp:
        config = yaml.safe_load(fp) or {}
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    summary = run_audit(config, paths, max_rows=args.max_rows)
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(to_jsonable(summary["decision"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
