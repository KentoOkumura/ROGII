from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp112_learned_pf_likelihood_weight_or_feature_followup"
DEFAULT_EXP111_LONG = "exp111_learned_pf_observation_likelihood_probe_oof_likelihood_long.csv.gz"
DEFAULT_EXP111_SUMMARY = "exp111_learned_pf_observation_likelihood_probe_summary.json"
DEFAULT_EXP099_WIDE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(frame: pd.DataFrame, *, value_col: str) -> str:
    digest = hashlib.sha256()
    for row in frame[["id", "variant", value_col]].itertuples(index=False):
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.variant).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row[2]).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit = Path(explicit_path)
        candidates.append(explicit if explicit.name == filename else explicit / filename)
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("experiments")
            / "exp111_learned_pf_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def rmse_from_error(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error.astype(np.float64)))))


def mae_from_error(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error.astype(np.float64))))


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _row_indices_from_ids(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def row_zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return ((values - mean) / std).astype(np.float32)


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def load_exp111_long(
    *, artifact_dir: str | Path | None, long_path: str | Path | None, candidates: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(DEFAULT_EXP111_LONG, long_path or artifact_dir)
    required = [
        "id",
        "well",
        "candidate_name",
        "candidate_index",
        "candidate_tvt",
        "abs_error",
        "within_5ft",
        "within_10ft",
        "fold",
        "pred_within10_prob",
        "pred_abs_error",
        "baseline_multiobs_score",
        "baseline_multiobs_mae",
        "baseline_multiobs_ncc",
        "md_since",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required,
        dtype={"id": str, "well": str, "candidate_name": str},
        low_memory=False,
    )
    frame = frame[frame["candidate_name"].isin(candidates)].copy()
    frame["candidate_name"] = pd.Categorical(
        frame["candidate_name"], categories=candidates, ordered=True
    )
    frame = frame.sort_values(["id", "candidate_name"]).reset_index(drop=True)
    counts = frame.groupby("id", observed=True)["candidate_name"].nunique()
    bad = counts[counts.ne(len(candidates))]
    if not bad.empty:
        raise ValueError(f"exp111 long cache has incomplete candidate rows, examples={bad.head()}")
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "groups": int(counts.size),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
    }
    return frame, meta


def load_true_tvt(
    *,
    wide_path: str | Path | None,
    ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(DEFAULT_EXP099_WIDE, wide_path)
    required = ["id", "last_known_tvt", "target"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    wanted = set(ids.astype(str).unique().tolist())
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=required,
        dtype={"id": str},
        chunksize=500_000,
        low_memory=False,
    ):
        part = chunk[chunk["id"].isin(wanted)].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError(f"No exp099 wide rows matched exp111 ids in {source}")
    frame = pd.concat(chunks, ignore_index=True)
    frame["true_tvt"] = (
        pd.to_numeric(frame["last_known_tvt"], errors="coerce")
        + pd.to_numeric(frame["target"], errors="coerce")
    ).astype(np.float32)
    frame = frame[["id", "true_tvt"]].drop_duplicates("id")
    meta = {
        "path": str(source),
        "matched_rows": int(len(frame)),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
    }
    return frame, meta


def maybe_limit_groups(frame: pd.DataFrame, max_groups: int | None) -> pd.DataFrame:
    if max_groups is None:
        return frame
    keep = frame["id"].drop_duplicates().head(int(max_groups))
    return frame[frame["id"].isin(set(keep))].copy()


def pivot_matrix(frame: pd.DataFrame, value: str, candidates: list[str]) -> np.ndarray:
    wide = frame.pivot(index="id", columns="candidate_name", values=value).reindex(
        columns=candidates
    )
    values = wide.to_numpy(np.float32)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:5].tolist()
        raise ValueError(f"{value} matrix contains non-finite values, examples={bad}")
    return values


def build_eval_context(
    long_frame: pd.DataFrame, true_tvt_frame: pd.DataFrame, candidates: list[str]
) -> dict[str, Any]:
    ids = long_frame["id"].drop_duplicates().reset_index(drop=True)
    base = (
        long_frame.groupby("id", observed=True)
        .agg(well=("well", "first"), fold=("fold", "first"), md_since=("md_since", "first"))
        .reindex(ids)
        .reset_index()
    )
    base = base.merge(true_tvt_frame, on="id", how="left", validate="one_to_one")
    if base["true_tvt"].isna().any():
        bad = base[base["true_tvt"].isna()]["id"].head(5).tolist()
        raise ValueError(f"Missing true_tvt for exp111 ids: {bad}")
    candidate_tvt = pivot_matrix(long_frame, "candidate_tvt", candidates)
    probability = pivot_matrix(long_frame, "pred_within10_prob", candidates)
    pred_error = pivot_matrix(long_frame, "pred_abs_error", candidates)
    multiobs_score = pivot_matrix(long_frame, "baseline_multiobs_score", candidates)
    multiobs_mae = pivot_matrix(long_frame, "baseline_multiobs_mae", candidates)
    multiobs_ncc = pivot_matrix(long_frame, "baseline_multiobs_ncc", candidates)
    true_tvt = base["true_tvt"].to_numpy(np.float32)
    observed_error = np.abs(candidate_tvt - true_tvt[:, None]).astype(np.float32)
    return {
        "base": base,
        "candidate_tvt": candidate_tvt,
        "probability": probability,
        "pred_error": pred_error,
        "multiobs_score": multiobs_score,
        "multiobs_mae": multiobs_mae,
        "multiobs_ncc": multiobs_ncc,
        "observed_error": observed_error,
        "true_tvt": true_tvt,
    }


def evaluate_selected(
    *,
    name: str,
    mode: str,
    selected: np.ndarray,
    context: dict[str, Any],
    candidates: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = context["base"]
    error = context["observed_error"][np.arange(len(base)), selected]
    likpf_idx = candidates.index("likpf_mean")
    changed = selected != likpf_idx
    row = {
        "variant": name,
        "mode": mode,
        "rows": int(len(base)),
        "rmse_tvt": rmse_from_error(error),
        "mae_tvt": mae_from_error(error),
        "within_5ft": float(np.mean(error <= 5.0)),
        "within_10ft": float(np.mean(error <= 10.0)),
        "switch_rate_vs_likpf": float(np.mean(changed)),
        "mean_abs_delta_vs_likpf": float(
            np.mean(
                np.abs(
                    context["candidate_tvt"][np.arange(len(base)), selected]
                    - context["candidate_tvt"][:, likpf_idx]
                )
            )
        ),
    }
    for candidate_idx, candidate in enumerate(candidates):
        row[f"select_rate_{candidate}"] = float(np.mean(selected == candidate_idx))
    if extra:
        row.update(extra)
    return row


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def score_policy_metrics(
    context: dict[str, Any], candidates: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    n_rows = len(context["base"])
    likpf_idx = candidates.index(
        str(get_nested(config, "followup.default_candidate") or "likpf_mean")
    )
    probability = context["probability"]
    pred_error = context["pred_error"]
    multiobs_score = context["multiobs_score"]

    rows: list[dict[str, Any]] = []
    rows.append(
        evaluate_selected(
            name="likpf_mean_single",
            mode="baseline",
            selected=np.full(n_rows, likpf_idx, dtype=np.int16),
            context=context,
            candidates=candidates,
        )
    )
    rows.append(
        evaluate_selected(
            name="learned_prob_top1",
            mode="diagnostic_top1",
            selected=np.argmax(probability, axis=1).astype(np.int16),
            context=context,
            candidates=candidates,
        )
    )
    rows.append(
        evaluate_selected(
            name="learned_error_top1",
            mode="diagnostic_top1",
            selected=np.argmin(pred_error, axis=1).astype(np.int16),
            context=context,
            candidates=candidates,
        )
    )
    rows.append(
        evaluate_selected(
            name="multiobs_score_top1",
            mode="baseline",
            selected=np.argmax(multiobs_score, axis=1).astype(np.int16),
            context=context,
            candidates=candidates,
        )
    )
    rows.append(
        evaluate_selected(
            name="oracle_candidate",
            mode="oracle",
            selected=np.argmin(context["observed_error"], axis=1).astype(np.int16),
            context=context,
            candidates=candidates,
        )
    )
    base_score = row_zscore(multiobs_score)
    prob_signal = row_zscore(logit(probability))
    error_signal = -row_zscore(np.log1p(np.maximum(pred_error, 0.0)))
    for alpha in get_nested(config, "followup.pf_weight_alphas") or []:
        alpha_value = float(alpha)
        suffix = str(alpha_value).replace(".", "p")
        for signal_name, signal in [
            ("prob", prob_signal),
            ("expected_error", error_signal),
        ]:
            selected = np.argmax(base_score + alpha_value * signal, axis=1).astype(np.int16)
            rows.append(
                evaluate_selected(
                    name=f"pf_weight_{signal_name}_alpha_{suffix}",
                    mode="pf_weight_alpha",
                    selected=selected,
                    context=context,
                    candidates=candidates,
                    extra={"alpha": alpha_value, "score_signal": signal_name},
                )
            )
    rows.extend(verifier_gate_metrics(context, candidates, config))
    return pd.DataFrame(rows)


def verifier_gate_metrics(
    context: dict[str, Any], candidates: list[str], config: dict[str, Any]
) -> list[dict[str, Any]]:
    n_rows = len(context["base"])
    probability = context["probability"]
    pred_error = context["pred_error"]
    candidate_tvt = context["candidate_tvt"]
    likpf_idx = candidates.index(
        str(get_nested(config, "followup.default_candidate") or "likpf_mean")
    )
    default = np.full(n_rows, likpf_idx, dtype=np.int16)
    prob_order = np.argsort(-probability, axis=1)
    err_order = np.argsort(pred_error, axis=1)
    top_prob = prob_order[:, 0]
    top_prob_value = probability[np.arange(n_rows), top_prob]
    prob_margin = top_prob_value - probability[np.arange(n_rows), prob_order[:, 1]]
    top_err = err_order[:, 0]
    err_margin = (
        pred_error[np.arange(n_rows), err_order[:, 1]] - pred_error[np.arange(n_rows), top_err]
    )

    rows: list[dict[str, Any]] = []
    prob_mins = [float(v) for v in get_nested(config, "followup.verifier.probability_min") or []]
    prob_margin_mins = [
        float(v) for v in get_nested(config, "followup.verifier.probability_margin_min") or []
    ]
    err_margin_mins = [
        float(v) for v in get_nested(config, "followup.verifier.error_margin_min") or []
    ]
    delta_caps = [
        float(v) for v in get_nested(config, "followup.verifier.max_abs_delta_vs_likpf") or []
    ]
    for prob_min in prob_mins:
        for margin_min in prob_margin_mins:
            for delta_cap in delta_caps:
                abs_delta = np.abs(
                    candidate_tvt[np.arange(n_rows), top_prob] - candidate_tvt[:, likpf_idx]
                )
                switch = (
                    (top_prob != likpf_idx)
                    & (top_prob_value >= prob_min)
                    & (prob_margin >= margin_min)
                    & (abs_delta <= delta_cap)
                )
                selected = default.copy()
                selected[switch] = top_prob[switch]
                rows.append(
                    evaluate_selected(
                        name=(
                            "gate_prob"
                            f"_p{str(prob_min).replace('.', 'p')}"
                            f"_m{str(margin_min).replace('.', 'p')}"
                            f"_d{str(delta_cap).replace('.', 'p')}"
                        ),
                        mode="verifier_gate",
                        selected=selected,
                        context=context,
                        candidates=candidates,
                        extra={
                            "probability_min": prob_min,
                            "probability_margin_min": margin_min,
                            "max_abs_delta_vs_likpf": delta_cap,
                        },
                    )
                )
    for margin_min in err_margin_mins:
        for delta_cap in delta_caps:
            abs_delta = np.abs(
                candidate_tvt[np.arange(n_rows), top_err] - candidate_tvt[:, likpf_idx]
            )
            switch = (top_err != likpf_idx) & (err_margin >= margin_min) & (abs_delta <= delta_cap)
            selected = default.copy()
            selected[switch] = top_err[switch]
            rows.append(
                evaluate_selected(
                    name=(
                        "gate_expected_error"
                        f"_m{str(margin_min).replace('.', 'p')}"
                        f"_d{str(delta_cap).replace('.', 'p')}"
                    ),
                    mode="verifier_gate",
                    selected=selected,
                    context=context,
                    candidates=candidates,
                    extra={
                        "error_margin_min": margin_min,
                        "max_abs_delta_vs_likpf": delta_cap,
                    },
                )
            )
    return rows


def build_oof_predictions(
    metrics: pd.DataFrame, context: dict[str, Any], candidates: list[str]
) -> pd.DataFrame:
    base = context["base"][["id", "well", "fold", "md_since", "true_tvt"]].reset_index(drop=True)
    candidate_tvt = context["candidate_tvt"]
    pred_by_name: dict[str, np.ndarray] = {}
    probability = context["probability"]
    pred_error = context["pred_error"]
    multiobs_score = context["multiobs_score"]
    for _, row in metrics.iterrows():
        name = str(row["variant"])
        mode = str(row["mode"])
        if mode in {"baseline", "diagnostic_top1", "oracle", "pf_weight_alpha", "verifier_gate"}:
            if name == "likpf_mean_single":
                selected = np.full(len(base), candidates.index("likpf_mean"), dtype=np.int16)
            elif name == "learned_prob_top1":
                selected = np.argmax(probability, axis=1).astype(np.int16)
            elif name == "learned_error_top1":
                selected = np.argmin(pred_error, axis=1).astype(np.int16)
            elif name == "multiobs_score_top1":
                selected = np.argmax(multiobs_score, axis=1).astype(np.int16)
            elif name == "oracle_candidate":
                selected = np.argmin(context["observed_error"], axis=1).astype(np.int16)
            elif name.startswith("pf_weight_"):
                alpha = float(row["alpha"])
                signal = str(row["score_signal"])
                base_score = row_zscore(multiobs_score)
                signal_matrix = (
                    row_zscore(logit(probability))
                    if signal == "prob"
                    else -row_zscore(np.log1p(np.maximum(pred_error, 0.0)))
                )
                selected = np.argmax(base_score + alpha * signal_matrix, axis=1).astype(np.int16)
            else:
                selected = reconstruct_gate_selection(name, context, candidates)
            pred = candidate_tvt[np.arange(len(base)), selected]
            pred_by_name[name] = pred.astype(np.float32)
    chunks = []
    for variant, pred in pred_by_name.items():
        part = base.copy()
        part["variant"] = variant
        part["prediction"] = pred
        part["abs_error"] = np.abs(pred - part["true_tvt"].to_numpy(np.float32)).astype(np.float32)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def reconstruct_gate_selection(
    name: str, context: dict[str, Any], candidates: list[str]
) -> np.ndarray:
    n_rows = len(context["base"])
    probability = context["probability"]
    pred_error = context["pred_error"]
    candidate_tvt = context["candidate_tvt"]
    likpf_idx = candidates.index("likpf_mean")
    selected = np.full(n_rows, likpf_idx, dtype=np.int16)
    parts = name.split("_")
    if name.startswith("gate_prob"):
        prob_min = float(parts[2][1:].replace("p", "."))
        margin_min = float(parts[3][1:].replace("p", "."))
        delta_cap = float(parts[4][1:].replace("p", "."))
        order = np.argsort(-probability, axis=1)
        top = order[:, 0]
        margin = probability[np.arange(n_rows), top] - probability[np.arange(n_rows), order[:, 1]]
        abs_delta = np.abs(candidate_tvt[np.arange(n_rows), top] - candidate_tvt[:, likpf_idx])
        switch = (
            (top != likpf_idx)
            & (probability[np.arange(n_rows), top] >= prob_min)
            & (margin >= margin_min)
            & (abs_delta <= delta_cap)
        )
        selected[switch] = top[switch]
    elif name.startswith("gate_expected_error"):
        margin_min = float(parts[3][1:].replace("p", "."))
        delta_cap = float(parts[4][1:].replace("p", "."))
        order = np.argsort(pred_error, axis=1)
        top = order[:, 0]
        margin = pred_error[np.arange(n_rows), order[:, 1]] - pred_error[np.arange(n_rows), top]
        abs_delta = np.abs(candidate_tvt[np.arange(n_rows), top] - candidate_tvt[:, likpf_idx])
        switch = (top != likpf_idx) & (margin >= margin_min) & (abs_delta <= delta_cap)
        selected[switch] = top[switch]
    return selected


def build_by_well(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        oof_predictions.groupby(["variant", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("abs_error", lambda value: rmse_from_error(value.to_numpy(np.float32))),
            mae_tvt=("abs_error", "mean"),
            within_10ft=(
                "abs_error",
                lambda value: float(np.mean(value.to_numpy(np.float32) <= 10.0)),
            ),
        )
        .sort_values(["variant", "rmse_tvt"], ascending=[True, False])
    )


def build_bucket_metrics(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = oof_predictions.copy()
    frame["distance_bucket"] = distance_bucket(frame["md_since"])
    frame["tail_rank_bucket"] = tail_rank_bucket(frame["id"])
    rows = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        for (variant, bucket), group in frame.groupby(["variant", bucket_col], observed=True):
            error = group["abs_error"].to_numpy(np.float32)
            rows.append(
                {
                    "variant": variant,
                    "bucket_family": bucket_col,
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "rmse_tvt": rmse_from_error(error),
                    "mae_tvt": mae_from_error(error),
                    "within_10ft": float(np.mean(error <= 10.0)),
                }
            )
    return pd.DataFrame(rows)


def build_selection_distribution(metrics: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        for candidate in candidates:
            rows.append(
                {
                    "variant": row["variant"],
                    "mode": row["mode"],
                    "candidate": candidate,
                    "selection_rate": row.get(f"select_rate_{candidate}"),
                }
            )
    return pd.DataFrame(rows)


def select_prediction_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "likpf_mean_single",
        "learned_prob_top1",
        "learned_error_top1",
        "multiobs_score_top1",
        "oracle_candidate",
    }
    rows = [metrics[metrics["variant"].isin(keep)]]
    for mode in ["pf_weight_alpha", "verifier_gate"]:
        subset = metrics[metrics["mode"].eq(mode)].sort_values(
            ["rmse_tvt", "within_10ft"], ascending=[True, False]
        )
        if not subset.empty:
            rows.append(subset.head(1))
    return pd.concat(rows, ignore_index=True).drop_duplicates("variant")


def build_ml_features(
    context: dict[str, Any], candidates: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    base = context["base"][["id", "well", "fold", "md_since"]].reset_index(drop=True).copy()
    probability = context["probability"]
    pred_error = context["pred_error"]
    candidate_tvt = context["candidate_tvt"]
    multiobs_score = context["multiobs_score"]
    multiobs_mae = context["multiobs_mae"]
    multiobs_ncc = context["multiobs_ncc"]
    likpf_idx = candidates.index("likpf_mean")

    prob_order = np.argsort(-probability, axis=1)
    err_order = np.argsort(pred_error, axis=1)
    prob_sorted = np.take_along_axis(probability, prob_order, axis=1)
    err_sorted = np.take_along_axis(pred_error, err_order, axis=1)
    entropy = -np.sum(
        np.clip(probability, 1e-6, 1.0) * np.log(np.clip(probability, 1e-6, 1.0)), axis=1
    )

    out = base
    out["learned_prob_top1_index"] = prob_order[:, 0].astype(np.int16)
    out["learned_error_top1_index"] = err_order[:, 0].astype(np.int16)
    out["learned_prob_top1_value"] = prob_sorted[:, 0].astype(np.float32)
    out["learned_prob_top2_value"] = prob_sorted[:, 1].astype(np.float32)
    out["learned_prob_margin_top1_top2"] = (prob_sorted[:, 0] - prob_sorted[:, 1]).astype(
        np.float32
    )
    out["learned_prob_entropy"] = entropy.astype(np.float32)
    out["learned_error_top1_value"] = err_sorted[:, 0].astype(np.float32)
    out["learned_error_top2_value"] = err_sorted[:, 1].astype(np.float32)
    out["learned_error_margin_top2_top1"] = (err_sorted[:, 1] - err_sorted[:, 0]).astype(np.float32)
    out["learned_prob_likpf_rank"] = rank_desc(probability)[:, likpf_idx].astype(np.int16)
    out["learned_error_likpf_rank"] = rank_asc(pred_error)[:, likpf_idx].astype(np.int16)
    out["learned_prob_top3_contains_likpf"] = (rank_desc(probability)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["learned_error_top3_contains_likpf"] = (rank_asc(pred_error)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["candidate_tvt_std"] = candidate_tvt.std(axis=1).astype(np.float32)
    out["candidate_tvt_range"] = (candidate_tvt.max(axis=1) - candidate_tvt.min(axis=1)).astype(
        np.float32
    )
    prob_sum = probability.sum(axis=1)
    prob_sum = np.where(prob_sum > 1e-6, prob_sum, 1.0)
    out["learned_prob_weighted_tvt"] = (
        np.sum(candidate_tvt * probability, axis=1) / prob_sum
    ).astype(np.float32)
    inv_error_weight = 1.0 / np.maximum(pred_error, 1e-3)
    inv_error_sum = inv_error_weight.sum(axis=1)
    out["learned_error_weighted_tvt"] = (
        np.sum(candidate_tvt * inv_error_weight, axis=1) / inv_error_sum
    ).astype(np.float32)

    include_candidate_tvt = bool(get_nested(config, "followup.feature_cache.include_candidate_tvt"))
    include_multiobs = bool(get_nested(config, "followup.feature_cache.include_multiobs_scores"))
    for idx, candidate in enumerate(candidates):
        out[f"learned_prob_{candidate}"] = probability[:, idx].astype(np.float32)
        out[f"learned_pred_abs_error_{candidate}"] = pred_error[:, idx].astype(np.float32)
        if include_candidate_tvt:
            out[f"candidate_tvt_{candidate}"] = candidate_tvt[:, idx].astype(np.float32)
        if include_multiobs:
            out[f"multiobs_score_{candidate}"] = multiobs_score[:, idx].astype(np.float32)
            out[f"multiobs_mae_{candidate}"] = multiobs_mae[:, idx].astype(np.float32)
            out[f"multiobs_ncc_{candidate}"] = multiobs_ncc[:, idx].astype(np.float32)
    return out


def summarize_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    likpf = metrics[metrics["variant"].eq("likpf_mean_single")].iloc[0]
    candidates = metrics[
        metrics["mode"].isin(["pf_weight_alpha", "verifier_gate", "diagnostic_top1"])
        & ~metrics["variant"].eq("learned_prob_top1")
    ].copy()
    candidates["delta_rmse_vs_likpf"] = candidates["rmse_tvt"] - float(likpf["rmse_tvt"])
    candidates["delta_within10_vs_likpf"] = candidates["within_10ft"] - float(likpf["within_10ft"])
    best_rmse = candidates.sort_values(["delta_rmse_vs_likpf", "delta_within10_vs_likpf"]).head(1)
    recommendation = "ml_feature_followup_supported_only"
    if not best_rmse.empty:
        row = best_rmse.iloc[0]
        if (
            float(row["delta_rmse_vs_likpf"]) < -0.02
            and float(row["delta_within10_vs_likpf"]) >= -0.001
        ):
            recommendation = "pf_weight_or_gate_supported_train_side_needs_rawtest_parity"
        elif float(row["delta_rmse_vs_likpf"]) < 0.0:
            recommendation = "weak_posthoc_signal_keep_as_ml_feature_not_direct_policy"
    return {
        "recommendation": recommendation,
        "likpf_reference": to_jsonable(likpf.to_dict()),
        "best_non_oracle": to_jsonable(best_rmse.iloc[0].to_dict())
        if not best_rmse.empty
        else None,
        "direct_submission_candidate": "not_selected",
    }


def run_followup(
    *,
    output_dir: str | Path,
    exp111_artifact_dir: str | Path | None,
    exp111_long_path: str | Path | None,
    exp099_wide_path: str | Path | None,
    max_groups: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [str(value) for value in get_nested(config, "followup.candidates")]
    if "likpf_mean" not in candidates:
        raise ValueError("followup.candidates must include likpf_mean")

    long_frame, exp111_meta = load_exp111_long(
        artifact_dir=exp111_artifact_dir,
        long_path=exp111_long_path,
        candidates=candidates,
    )
    long_frame = maybe_limit_groups(long_frame, max_groups)
    true_tvt, exp099_meta = load_true_tvt(
        wide_path=exp099_wide_path,
        ids=long_frame["id"],
    )
    context = build_eval_context(long_frame, true_tvt, candidates)
    metrics = score_policy_metrics(context, candidates, config)
    prediction_metrics = select_prediction_metric_rows(metrics)
    oof_predictions = build_oof_predictions(prediction_metrics, context, candidates)
    by_well = build_by_well(oof_predictions)
    bucket_metrics = build_bucket_metrics(oof_predictions)
    selection_distribution = build_selection_distribution(metrics, candidates)
    ml_features = build_ml_features(context, candidates, config)
    feature_schema = pd.DataFrame(
        {"feature_index": range(len(ml_features.columns)), "feature": ml_features.columns}
    )
    decision = summarize_decision(metrics)

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    selection_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    oof_path = output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    feature_path = output_dir / f"{OUTPUT_PREFIX}_ml_features.csv.gz"
    schema_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    metrics.to_csv(metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    selection_distribution.to_csv(selection_path, index=False)
    oof_predictions.to_csv(oof_path, index=False, compression="gzip")
    ml_features.to_csv(feature_path, index=False, compression="gzip")
    feature_schema.to_csv(schema_path, index=False)

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_debug_completed"
        if max_groups is not None
        else "completed_train_side_posthoc_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "rows": int(len(context["base"])),
        "candidate_rows": int(len(long_frame)),
        "wells": int(context["base"]["well"].nunique()),
        "candidates": candidates,
        "source": {
            "exp111_oof_likelihood_long": exp111_meta,
            "exp099_wide_cache": exp099_meta,
        },
        "decision": to_jsonable(decision),
        "artifacts": {
            "metrics": metrics_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": bucket_path.name,
            "selection_distribution": selection_path.name,
            "oof_predictions": oof_path.name,
            "ml_features": feature_path.name,
            "feature_schema": schema_path.name,
        },
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "by_well": sha256_path(by_well_path),
            "bucket_metrics": sha256_path(bucket_path),
            "selection_distribution": sha256_path(selection_path),
            "oof_predictions": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "ml_features": sha256_path(feature_path),
            "ml_features_decompressed": sha256_path(feature_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
            "prediction": prediction_sha256(oof_predictions, value_col="prediction"),
        },
        "top_metrics": to_jsonable(
            metrics.sort_values(["mode", "rmse_tvt"]).groupby("mode").head(3).to_dict("records")
        ),
        "prediction_variants": prediction_metrics["variant"].tolist(),
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exp111-artifact-dir", type=Path, default=None)
    parser.add_argument("--exp111-long-path", type=Path, default=None)
    parser.add_argument("--exp099-wide-path", type=Path, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args(argv)

    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not (Path("/kaggle/working").exists())
        else Path("/kaggle/working") / "artifacts"
    )
    max_groups = args.max_groups
    configured_max = get_nested(config, "followup.max_groups")
    if max_groups is None and configured_max is not None:
        max_groups = int(configured_max)
    return run_followup(
        output_dir=output_dir,
        exp111_artifact_dir=args.exp111_artifact_dir
        or get_nested(config, "data.exp111_artifact_dir_local"),
        exp111_long_path=args.exp111_long_path
        or get_nested(config, "data.exp111_oof_likelihood_long"),
        exp099_wide_path=args.exp099_wide_path
        or get_nested(config, "data.exp099_train_feature_cache_local"),
        max_groups=max_groups,
    )


if __name__ == "__main__":
    main()
