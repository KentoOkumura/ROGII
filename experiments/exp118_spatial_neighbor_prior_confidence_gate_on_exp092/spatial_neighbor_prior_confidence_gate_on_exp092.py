from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp118_spatial_neighbor_prior_confidence_gate_on_exp092"
EXP114_OOF = "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
EXP114_SUMMARY = "exp114_spatial_neighbor_prior_signal_audit_summary.json"
EXP092_PREDICTIONS = "exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz"


@dataclass(frozen=True)
class GateSpec:
    name: str
    std_quantile: float | None = None
    distance_quantile: float | None = None
    min_neighbor_wells: int | None = None
    max_azimuth_mismatch: float | None = None
    max_abs_delta: float | None = None


@dataclass(frozen=True)
class PolicySpec:
    policy: str
    model: str
    variant: str
    alpha: float
    clip: float
    gate: GateSpec


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


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\n")
    for row_id, value in zip(ids.astype(str), values.astype(np.float64), strict=False):
        digest.update(row_id.encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(value).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp114_spatial_neighbor_prior_signal_audit"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp092_u_projection_correction_disagreement_fullrun"
            / "kaggle"
            / "output"
            / "train"
            / "artifacts"
            / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:100])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def float_tag(value: float) -> str:
    text = f"{float(value):.5g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    pred_values = pred.astype(np.float64)
    true_values = true.astype(np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(true_values)
    if not mask.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    error = pred_values[mask] - true_values[mask]
    return {
        "rows": int(mask.sum()),
        "coverage": float(mask.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
        "bias": float(np.mean(error)),
    }


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def parse_gate_specs(config: dict[str, Any]) -> list[GateSpec]:
    raw_gates = get_nested(config, "gate.gates") or []
    gates: list[GateSpec] = []
    for raw in raw_gates:
        gates.append(
            GateSpec(
                name=str(raw["name"]),
                std_quantile=(
                    None if raw.get("std_quantile") is None else float(raw["std_quantile"])
                ),
                distance_quantile=(
                    None
                    if raw.get("distance_quantile") is None
                    else float(raw["distance_quantile"])
                ),
                min_neighbor_wells=(
                    None
                    if raw.get("min_neighbor_wells") is None
                    else int(raw["min_neighbor_wells"])
                ),
                max_azimuth_mismatch=(
                    None
                    if raw.get("max_azimuth_mismatch") is None
                    else float(raw["max_azimuth_mismatch"])
                ),
                max_abs_delta=(
                    None if raw.get("max_abs_delta") is None else float(raw["max_abs_delta"])
                ),
            )
        )
    if not gates:
        gates.append(GateSpec(name="valid_prior"))
    return gates


def read_spatial_oof(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(EXP114_OOF, get_nested(config, "data.exp114_oof_predictions_local"))
    header = pd.read_csv(source, nrows=0).columns.tolist()
    variants = [str(value) for value in get_nested(config, "gate.spatial_variants") or []]
    required = [
        "id",
        "well",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        "eval_len",
        "likpf_mean",
    ]
    for variant in variants:
        required.extend(
            [
                f"{variant}_prior_tvt",
                f"{variant}_prior_delta",
                f"{variant}_prior_std",
                f"{variant}_prior_count",
                f"{variant}_neighbor_wells",
                f"{variant}_distance_mean",
                f"{variant}_azimuth_mismatch",
            ]
        )
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=required,
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    try:
        summary_path = find_artifact(
            EXP114_SUMMARY,
            get_nested(config, "data.exp114_summary_local"),
        )
        metadata["summary"] = str(summary_path)
        metadata["summary_sha256"] = sha256_path(summary_path)
    except FileNotFoundError:
        metadata["summary"] = None
        metadata["summary_sha256"] = None
    return frame, metadata


def read_exp092_predictions(
    config: dict[str, Any],
    *,
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP092_PREDICTIONS,
        get_nested(config, "data.exp092_predictions_local"),
    )
    usecols = ["id", "well", "variant", "mode", "model", "target_tvt", "pred_tvt"]
    selected_variant = str(get_nested(config, "gate.exp092_variant") or "")
    selected_mode = str(get_nested(config, "gate.exp092_mode") or "")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        dtype={"id": str, "well": str, "variant": str, "mode": str, "model": str},
        chunksize=int(get_nested(config, "runtime.read_chunksize") or 1_000_000),
        low_memory=False,
    ):
        mask = chunk["model"].isin(models)
        if selected_variant:
            mask &= chunk["variant"].astype(str) == selected_variant
        if selected_mode:
            mask &= chunk["mode"].astype(str) == selected_mode
        part = chunk.loc[mask].copy()
        if not part.empty:
            part["target_tvt"] = pd.to_numeric(part["target_tvt"], errors="coerce").astype(
                np.float32
            )
            part["pred_tvt"] = pd.to_numeric(part["pred_tvt"], errors="coerce").astype(
                np.float32
            )
            chunks.append(part)
    if not chunks:
        raise ValueError(f"no exp092 prediction rows matched models={models} in {source}")
    frame = pd.concat(chunks, ignore_index=True, sort=False)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "models": sorted(frame["model"].unique().tolist()),
        "variant": selected_variant,
        "mode": selected_mode,
    }
    return frame, metadata


def gate_mask(
    frame: pd.DataFrame,
    *,
    variant: str,
    gate: GateSpec,
    delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = numeric_array(frame, f"{variant}_prior_tvt")
    std = numeric_array(frame, f"{variant}_prior_std")
    neighbor_wells = numeric_array(frame, f"{variant}_neighbor_wells")
    distance_mean = numeric_array(frame, f"{variant}_distance_mean")
    azimuth_mismatch = numeric_array(frame, f"{variant}_azimuth_mismatch")

    mask = np.isfinite(prior) & np.isfinite(delta)
    thresholds: dict[str, Any] = {}
    if gate.std_quantile is not None:
        values = std[np.isfinite(std)]
        threshold = float(np.quantile(values, gate.std_quantile)) if len(values) else np.nan
        thresholds["std_threshold"] = threshold
        mask &= np.isfinite(std) & (std <= threshold)
    if gate.distance_quantile is not None:
        values = distance_mean[np.isfinite(distance_mean)]
        threshold = (
            float(np.quantile(values, gate.distance_quantile)) if len(values) else np.nan
        )
        thresholds["distance_mean_threshold"] = threshold
        mask &= np.isfinite(distance_mean) & (distance_mean <= threshold)
    if gate.min_neighbor_wells is not None:
        thresholds["min_neighbor_wells"] = int(gate.min_neighbor_wells)
        mask &= np.isfinite(neighbor_wells) & (neighbor_wells >= gate.min_neighbor_wells)
    if gate.max_azimuth_mismatch is not None:
        thresholds["max_azimuth_mismatch"] = float(gate.max_azimuth_mismatch)
        mask &= np.isfinite(azimuth_mismatch) & (
            azimuth_mismatch <= gate.max_azimuth_mismatch
        )
    if gate.max_abs_delta is not None:
        thresholds["max_abs_delta"] = float(gate.max_abs_delta)
        mask &= np.abs(delta) <= gate.max_abs_delta
    return mask, thresholds


def make_prediction(
    frame: pd.DataFrame,
    spec: PolicySpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base = numeric_array(frame, "exp092_pred_tvt")
    prior = numeric_array(frame, f"{spec.variant}_prior_tvt")
    delta = prior.astype(np.float32) - base.astype(np.float32)
    mask, thresholds = gate_mask(frame, variant=spec.variant, gate=spec.gate, delta=delta)
    correction = np.zeros(len(frame), dtype=np.float32)
    correction[mask] = spec.alpha * np.clip(delta[mask], -spec.clip, spec.clip)
    pred = base.copy()
    pred[mask] = pred[mask] + correction[mask]
    return pred.astype(np.float32), correction, thresholds


def compute_overall_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    variants: list[str],
    gates: list[GateSpec],
    alphas: list[float],
    clips: list[float],
) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    base = numeric_array(frame, "exp092_pred_tvt")
    base_score = score_prediction(base, true)
    rows: list[dict[str, Any]] = [
        {
            "policy": f"{model}__baseline_exp092",
            "model": model,
            "variant": "exp092",
            "gate": "baseline",
            "alpha": 0.0,
            "clip": 0.0,
            "gate_rate": 0.0,
            "correction_abs_mean": 0.0,
            "correction_abs_p95": 0.0,
            "correction_abs_max": 0.0,
            "prediction_sha256": prediction_sha256(
                frame["id"],
                base,
                label=f"{OUTPUT_PREFIX}/{model}/baseline",
            ),
            "delta_rmse_vs_exp092": 0.0,
            "delta_mae_vs_exp092": 0.0,
            "delta_within10_vs_exp092": 0.0,
            **base_score,
        }
    ]
    for variant in variants:
        for gate in gates:
            for alpha in alphas:
                for clip in clips:
                    policy = (
                        f"{model}__{variant}__{gate.name}"
                        f"__a{float_tag(alpha)}__c{float_tag(clip)}"
                    )
                    spec = PolicySpec(
                        policy=policy,
                        model=model,
                        variant=variant,
                        alpha=alpha,
                        clip=clip,
                        gate=gate,
                    )
                    pred, correction, thresholds = make_prediction(frame, spec)
                    score = score_prediction(pred, true)
                    active = np.abs(correction) > 0.0
                    active_abs = np.abs(correction[active])
                    rows.append(
                        {
                            "policy": policy,
                            "model": model,
                            "variant": variant,
                            "gate": gate.name,
                            "alpha": alpha,
                            "clip": clip,
                            "gate_rate": float(active.mean()),
                            "correction_abs_mean": (
                                float(np.mean(active_abs)) if len(active_abs) else 0.0
                            ),
                            "correction_abs_p95": (
                                float(np.quantile(active_abs, 0.95)) if len(active_abs) else 0.0
                            ),
                            "correction_abs_max": (
                                float(np.max(active_abs)) if len(active_abs) else 0.0
                            ),
                            "prediction_sha256": prediction_sha256(
                                frame["id"],
                                pred,
                                label=f"{OUTPUT_PREFIX}/{policy}",
                            ),
                            "delta_rmse_vs_exp092": (
                                None
                                if score["rmse"] is None
                                else float(score["rmse"] - base_score["rmse"])
                            ),
                            "delta_mae_vs_exp092": (
                                None
                                if score["mae"] is None
                                else float(score["mae"] - base_score["mae"])
                            ),
                            "delta_within10_vs_exp092": (
                                None
                                if score["within10"] is None
                                else float(score["within10"] - base_score["within10"])
                            ),
                            **thresholds,
                            **score,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["rmse", "policy"], na_position="last")


def parse_policy(row: pd.Series, gates: dict[str, GateSpec]) -> PolicySpec:
    return PolicySpec(
        policy=str(row["policy"]),
        model=str(row["model"]),
        variant=str(row["variant"]),
        alpha=float(row["alpha"]),
        clip=float(row["clip"]),
        gate=gates[str(row["gate"])],
    )


def detailed_metrics(
    frame: pd.DataFrame,
    policies: list[PolicySpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true = numeric_array(frame, "true_tvt")
    by_well_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    prediction_frame = frame[["id", "well", "true_tvt", "md_since", "exp092_pred_tvt"]].copy()
    prediction_frame = prediction_frame.rename(columns={"exp092_pred_tvt": "baseline_exp092"})
    prediction_limit = max(len(policies), 1)

    for index, spec in enumerate(policies):
        if spec.variant == "exp092":
            pred = numeric_array(frame, "exp092_pred_tvt")
            correction = np.zeros(len(frame), dtype=np.float32)
        else:
            pred, correction, _ = make_prediction(frame, spec)
        if index < prediction_limit:
            prediction_frame[spec.policy] = pred

        for well, group_idx in frame.groupby("well", sort=False).groups.items():
            positions = np.asarray(list(group_idx), dtype=np.int64)
            score = score_prediction(pred[positions], true[positions])
            if score["rows"] == 0:
                continue
            by_well_rows.append({"policy": spec.policy, "well": str(well), **score})
            order = np.argsort(numeric_array(frame.iloc[positions], "md_since"))
            ordered_pred = pred[positions][order]
            ordered_corr = correction[positions][order]
            pred_step = np.abs(np.diff(ordered_pred.astype(np.float64)))
            corr_step = np.abs(np.diff(ordered_corr.astype(np.float64)))
            path_rows.append(
                {
                    "policy": spec.policy,
                    "well": str(well),
                    "rows": int(len(positions)),
                    "pred_step_abs_p95": (
                        float(np.quantile(pred_step, 0.95)) if len(pred_step) else 0.0
                    ),
                    "pred_step_abs_max": float(np.max(pred_step)) if len(pred_step) else 0.0,
                    "pred_step_abs_ge10": int(np.sum(pred_step >= 10.0)),
                    "pred_step_abs_ge25": int(np.sum(pred_step >= 25.0)),
                    "correction_step_abs_p95": (
                        float(np.quantile(corr_step, 0.95)) if len(corr_step) else 0.0
                    ),
                    "correction_step_abs_max": (
                        float(np.max(corr_step)) if len(corr_step) else 0.0
                    ),
                    "correction_step_abs_ge5": int(np.sum(corr_step >= 5.0)),
                }
            )

        work = pd.DataFrame(
            {
                "bucket": distance_bucket(frame["md_since"]),
                "true": true,
                "pred": pred,
            }
        )
        for bucket, positions in work.groupby("bucket", observed=False).groups.items():
            idx = np.asarray(list(positions), dtype=np.int64)
            score = score_prediction(pred[idx], true[idx])
            if score["rows"] == 0:
                continue
            bucket_rows.append({"policy": spec.policy, "distance_bucket": str(bucket), **score})

    return (
        pd.DataFrame(by_well_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(path_rows),
        prediction_frame,
    )


def summarize_by_well_delta(by_well: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    baseline = by_well[by_well["policy"] == baseline_policy][["well", "rmse"]].rename(
        columns={"rmse": "baseline_rmse"}
    )
    merged = by_well.merge(baseline, on="well", how="left")
    merged["delta_rmse_vs_exp092"] = merged["rmse"] - merged["baseline_rmse"]
    rows = []
    for policy, group in merged.groupby("policy", sort=False):
        if policy == baseline_policy:
            continue
        delta = pd.to_numeric(group["delta_rmse_vs_exp092"], errors="coerce")
        rows.append(
            {
                "policy": policy,
                "wells": int(delta.notna().sum()),
                "improved_wells": int((delta < 0.0).sum()),
                "worse_wells": int((delta > 0.0).sum()),
                "same_wells": int((delta == 0.0).sum()),
                "max_regression_rmse": float(delta.max()),
                "max_improvement_rmse": float(delta.min()),
                "mean_delta_rmse": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_regression_rmse", "mean_delta_rmse"])


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": OUTPUT_PREFIX,
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    ).to_csv(path, index=False)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    models = [str(value) for value in get_nested(config, "gate.exp092_models") or ["lgb1"]]
    variants = [str(value) for value in get_nested(config, "gate.spatial_variants") or []]
    alphas = [float(value) for value in get_nested(config, "gate.correction_alphas") or [0.1]]
    clips = [float(value) for value in get_nested(config, "gate.correction_clip_ft") or [20.0]]
    top_n = int(get_nested(config, "audit.top_n_detailed_policies") or 12)
    top_prediction_n = int(get_nested(config, "audit.top_n_prediction_policies") or 6)
    gates = parse_gate_specs(config)
    gate_by_name = {gate.name: gate for gate in gates}

    spatial, spatial_meta = read_spatial_oof(config)
    exp092, exp092_meta = read_exp092_predictions(config, models=models)
    frame = exp092.merge(spatial, on=["id", "well"], how="inner", validate="many_to_one")
    frame = frame.rename(
        columns={"pred_tvt": "exp092_pred_tvt", "target_tvt": "exp092_target_tvt"}
    )
    target_diff = np.abs(
        numeric_array(frame, "exp092_target_tvt") - numeric_array(frame, "true_tvt")
    )
    target_diff_max = float(np.nanmax(target_diff)) if len(target_diff) else None
    if target_diff_max is not None and target_diff_max > 1.0e-3:
        raise ValueError(f"exp092 target_tvt and exp114 true_tvt differ: max={target_diff_max}")

    all_metrics: list[pd.DataFrame] = []
    detailed_policy_rows: list[pd.Series] = []
    for model, model_frame in frame.groupby("model", sort=False):
        metrics = compute_overall_metrics(
            model_frame.reset_index(drop=True),
            model=str(model),
            variants=variants,
            gates=gates,
            alphas=alphas,
            clips=clips,
        )
        all_metrics.append(metrics)
        detailed_policy_rows.extend(metrics.head(top_n).itertuples(index=False, name=None))

    gate_metrics = pd.concat(all_metrics, ignore_index=True, sort=False).sort_values(
        ["rmse", "policy"],
        na_position="last",
    )

    policies: list[PolicySpec] = []
    for _, row in gate_metrics.head(top_n).iterrows():
        if str(row["variant"]) == "exp092":
            policies.append(
                PolicySpec(
                    policy=str(row["policy"]),
                    model=str(row["model"]),
                    variant="exp092",
                    alpha=0.0,
                    clip=0.0,
                    gate=GateSpec(name="baseline"),
                )
            )
        else:
            policies.append(parse_policy(row, gate_by_name))

    best_model = str(gate_metrics.iloc[0]["model"])
    detail_frame = frame[frame["model"] == best_model].reset_index(drop=True)
    detail_policies = [policy for policy in policies if policy.model == best_model][
        : max(top_n, top_prediction_n)
    ]
    baseline_policy = f"{best_model}__baseline_exp092"
    if not any(policy.policy == baseline_policy for policy in detail_policies):
        detail_policies.insert(
            0,
            PolicySpec(
                policy=baseline_policy,
                model=best_model,
                variant="exp092",
                alpha=0.0,
                clip=0.0,
                gate=GateSpec(name="baseline"),
            ),
        )

    by_well, bucket_metrics, path_continuity, prediction_frame = detailed_metrics(
        detail_frame,
        detail_policies[:top_n],
    )
    by_well_delta = summarize_by_well_delta(by_well, baseline_policy)
    if top_prediction_n + 5 < len(prediction_frame.columns):
        fixed = ["id", "well", "true_tvt", "md_since", "baseline_exp092"]
        prediction_cols = fixed + [
            col for col in prediction_frame.columns if col not in fixed
        ][:top_prediction_n]
        prediction_frame = prediction_frame[prediction_cols]

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_gate_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    by_well_delta_path = artifacts / f"{OUTPUT_PREFIX}_by_well_delta.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    path_path = artifacts / f"{OUTPUT_PREFIX}_path_continuity.csv"
    predictions_path = artifacts / f"{OUTPUT_PREFIX}_top_gated_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    gate_metrics.to_csv(metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    by_well_delta.to_csv(by_well_delta_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    path_continuity.to_csv(path_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False, compression="gzip")
    feature_columns = [
        column
        for column in spatial.columns
        if column not in {"id", "well", "true_tvt", "last_known_tvt", "md_since", "eval_len"}
    ]
    write_feature_schema(schema_path, feature_columns)

    best = gate_metrics.iloc[0].to_dict() if len(gate_metrics) else {}
    baseline = gate_metrics[gate_metrics["policy"] == baseline_policy]
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) else {}
    best_by_well = (
        by_well_delta[by_well_delta["policy"] == best.get("policy")].iloc[0].to_dict()
        if len(by_well_delta) and best
        and (by_well_delta["policy"] == best.get("policy")).any()
        else {}
    )
    decision = "confidence_gate_not_supported"
    if best and float(best.get("delta_rmse_vs_exp092") or 0.0) < 0.0:
        max_regression = float(best_by_well.get("max_regression_rmse", np.inf))
        decision = (
            "confidence_gate_supported_for_review"
            if max_regression <= float(get_nested(config, "audit.max_regression_warn_rmse") or 0.25)
            else "global_gain_but_worst_well_warning"
        )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "models": models,
        "spatial_oof": spatial_meta,
        "exp092_predictions": exp092_meta,
        "target_tvt_max_abs_diff": target_diff_max,
        "variants": variants,
        "gates": [gate.__dict__ for gate in gates],
        "best_policy": to_jsonable(best),
        "baseline_policy": to_jsonable(baseline_row),
        "best_by_well_delta": to_jsonable(best_by_well),
        "decision": decision,
        "artifacts": {
            "gate_metrics": str(metrics_path),
            "by_well": str(by_well_path),
            "by_well_delta": str(by_well_delta_path),
            "bucket_metrics": str(bucket_path),
            "path_continuity": str(path_path),
            "top_gated_predictions": str(predictions_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "gate_metrics": sha256_path(metrics_path),
            "by_well": sha256_path(by_well_path),
            "by_well_delta": sha256_path(by_well_delta_path),
            "bucket_metrics": sha256_path(bucket_path),
            "path_continuity": sha256_path(path_path),
            "top_gated_predictions_raw": sha256_path(predictions_path),
            "top_gated_predictions_decompressed": sha256_path(
                predictions_path,
                decompressed=True,
            ),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_pending_kaggle_train",
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "best_policy": to_jsonable(best),
        "baseline_policy": to_jsonable(baseline_row),
        "best_by_well_delta": to_jsonable(best_by_well),
        "decision": decision,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": (
            "Implemented train-side exp092 confidence gate audit. "
            "Awaiting Kaggle train execution."
        ),
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_policy"]), indent=2, sort_keys=True))
