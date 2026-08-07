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

OUTPUT_PREFIX = "exp177_beam_topk_bimodal_gate_posthoc_audit"
EXP173_PREFIX = "exp173_beam_topk_path_posterior_audit"
EXP173_DIR = Path("experiments") / "exp173_beam_topk_path_posterior_audit"


@dataclass(frozen=True)
class PolicySpec:
    variant: str
    gate_name: str
    replacement: str
    policy: str


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


def artifact_meta(path: Path) -> dict[str, Any]:
    meta = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        meta["decompressed_sha256"] = sha256_path(path, decompressed=True)
    return meta


def configured_artifact(config: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = get_nested(config, "data.exp173_artifacts") or {}
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"data.exp173_artifacts.{name} must be configured")
    return value


def find_artifact(config: dict[str, Any], name: str) -> Path:
    artifact = configured_artifact(config, name)
    filename = str(artifact.get("filename") or "")
    if not filename:
        raise ValueError(f"data.exp173_artifacts.{name}.filename is required")

    candidates: list[Path] = []
    for raw in artifact.get("path_candidates") or []:
        path = Path(str(raw))
        candidates.extend([path, Path.cwd() / path])
    candidates.extend(
        [
            Path(filename),
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            EXP173_DIR / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"Artifact not found: {filename}\nChecked:\n{checked}")


def read_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def qtag(value: float) -> str:
    return f"q{int(round(float(value) * 100)):02d}"


def replacement_columns(
    config: dict[str, Any],
    variants: list[str],
    header: list[str],
) -> list[str]:
    suffixes = [str(value) for value in (get_nested(config, "gate.replacement_suffixes") or [])]
    if not suffixes:
        raise ValueError("gate.replacement_suffixes must define replacement candidate suffixes")
    header_set = set(header)
    columns: list[str] = []
    missing: list[str] = []
    for variant in variants:
        for suffix in suffixes:
            column = f"{variant}_{suffix}"
            if column in header_set:
                columns.append(column)
            else:
                missing.append(column)
    if missing:
        raise ValueError(f"candidate_wide is missing replacement columns: {missing[:20]}")
    return columns


def diagnostic_columns(config: dict[str, Any], variants: list[str], header: list[str]) -> list[str]:
    suffixes = [
        "top1_top2_sep",
        "top2_cost_gap_per_row",
        "topk_entropy",
        "topk_spread",
    ]
    header_set = set(header)
    required = ["id", "well", "row_idx"]
    columns = list(required)
    missing: list[str] = []
    for variant in variants:
        for suffix in suffixes:
            column = f"{variant}_{suffix}"
            if column in header_set:
                columns.append(column)
            else:
                missing.append(column)
    if missing:
        raise ValueError(f"topk_diagnostics is missing gate columns: {missing}")
    return columns


def read_parent_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants = [str(value) for value in (get_nested(config, "gate.beam_variants") or [])]
    if not variants:
        raise ValueError("gate.beam_variants must define exp173 Beam variants")
    max_rows = get_nested(config, "audit.max_rows")
    nrows = None if max_rows in {None, "null"} else int(max_rows)

    candidate_wide_path = find_artifact(config, "candidate_wide")
    diagnostics_path = find_artifact(config, "topk_diagnostics")
    topk_paths_path = find_artifact(config, "topk_paths")
    candidate_metrics_path = find_artifact(config, "candidate_metrics")

    wide_header = read_csv_header(candidate_wide_path)
    diagnostics_header = read_csv_header(diagnostics_path)
    candidate_metrics = pd.read_csv(candidate_metrics_path)

    primary_baseline = str(get_nested(config, "audit.primary_baseline") or "likpf_mean")
    comparison_baseline = str(get_nested(config, "audit.beam_baseline") or "beam_mean")
    required_wide = [
        "id",
        "well",
        "row_idx",
        "target",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        primary_baseline,
        comparison_baseline,
    ]
    replacements = replacement_columns(config, variants, wide_header)
    missing_wide = [column for column in required_wide if column not in wide_header]
    if missing_wide:
        raise ValueError(f"candidate_wide is missing required columns: {missing_wide}")

    wide = pd.read_csv(
        candidate_wide_path,
        usecols=required_wide + replacements,
        nrows=nrows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    diagnostics = pd.read_csv(
        diagnostics_path,
        usecols=diagnostic_columns(config, variants, diagnostics_header),
        nrows=nrows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame = wide.merge(
        diagnostics,
        on=["id", "well", "row_idx"],
        how="left",
        validate="1:1",
    )
    max_wells = get_nested(config, "audit.max_wells")
    if max_wells not in {None, "null"}:
        keep = sorted(frame["well"].astype(str).unique().tolist())[: int(max_wells)]
        frame = frame[frame["well"].isin(keep)].reset_index(drop=True)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    topk_paths_header = read_csv_header(topk_paths_path)
    metadata = {
        "variants": variants,
        "replacement_columns": replacements,
        "parent_candidate_metrics_rows": int(len(candidate_metrics)),
        "parent_primary_baseline": candidate_metrics[
            candidate_metrics["candidate"].astype(str).eq(primary_baseline)
        ].head(1).to_dict(orient="records"),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "topk_paths_columns": topk_paths_header,
        "artifacts": {
            "candidate_wide": artifact_meta(candidate_wide_path),
            "topk_diagnostics": artifact_meta(diagnostics_path),
            "topk_paths": artifact_meta(topk_paths_path),
            "candidate_metrics": artifact_meta(candidate_metrics_path),
        },
    }
    return frame, metadata


def score_error(err: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    finite = np.isfinite(err)
    if mask is not None:
        finite &= mask
    if not finite.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    selected = err[finite].astype(np.float64)
    abs_err = np.abs(selected)
    return {
        "rows": int(len(selected)),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(selected * selected))),
        "mae": float(np.mean(abs_err)),
        "within10": float(np.mean(abs_err <= 10.0)),
        "bias": float(np.mean(selected)),
    }


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.cut(
        pd.to_numeric(pd.Series(values), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanquantile(finite, float(q)))


def build_gate_masks(
    frame: pd.DataFrame,
    variant: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    quantiles = get_nested(config, "gate.quantiles") or {}
    feature_map = {
        "sep": (f"{variant}_top1_top2_sep", ">="),
        "cost": (f"{variant}_top2_cost_gap_per_row", "<="),
        "entropy": (f"{variant}_topk_entropy", ">="),
        "spread": (f"{variant}_topk_spread", ">="),
    }
    masks: dict[str, np.ndarray] = {}
    threshold_rows: list[dict[str, Any]] = []
    for short_name, (column, direction) in feature_map.items():
        values = numeric_array(frame, column)
        for raw_q in quantiles.get(short_name, []):
            q = float(raw_q)
            threshold = quantile(values, q)
            if direction == ">=":
                mask = values >= threshold
                label = f"{short_name}_ge_{qtag(q)}"
            else:
                mask = values <= threshold
                label = f"{short_name}_le_{qtag(q)}"
            mask &= np.isfinite(values)
            masks[label] = mask
            threshold_rows.append(
                {
                    "variant": variant,
                    "condition": label,
                    "column": column,
                    "direction": direction,
                    "quantile": q,
                    "threshold": threshold,
                    "rows": int(mask.sum()),
                    "rate": float(mask.mean()),
                }
            )

    for raw_policy in get_nested(config, "gate.and_policies") or []:
        parts = [str(part) for part in raw_policy]
        if not parts:
            continue
        missing = [part for part in parts if part not in masks]
        if missing:
            raise ValueError(f"gate.and_policies references missing conditions: {missing}")
        mask = np.ones(len(frame), dtype=bool)
        for part in parts:
            mask &= masks[part]
        label = "and_" + "__".join(parts)
        masks[label] = mask
        threshold_rows.append(
            {
                "variant": variant,
                "condition": label,
                "column": " AND ".join(parts),
                "direction": "and",
                "quantile": None,
                "threshold": None,
                "rows": int(mask.sum()),
                "rate": float(mask.mean()),
            }
        )
    return masks, threshold_rows


def policy_name(variant: str, gate_name: str, replacement: str) -> str:
    suffix = replacement.removeprefix(f"{variant}_")
    return f"{variant}__{gate_name}__replace_{suffix}"


def precompute_groups(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    true: np.ndarray,
    beam_baseline: str,
) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
    }
    if beam_baseline in frame.columns:
        beam_gap = np.abs(numeric_array(frame, beam_baseline) - baseline)
        threshold = quantile(beam_gap, 0.75)
        groups["beam_likpf_gap_top_quartile"] = np.isfinite(beam_gap) & (beam_gap >= threshold)
    return groups


def evaluate_policy(
    frame: pd.DataFrame,
    spec: PolicySpec,
    baseline: np.ndarray,
    true: np.ndarray,
    baseline_err: np.ndarray,
    baseline_score: dict[str, Any],
    baseline_group_scores: dict[str, dict[str, Any]],
    groups: dict[str, np.ndarray],
    well_codes: np.ndarray,
    well_counts: np.ndarray,
    baseline_well_rmse: np.ndarray,
) -> dict[str, Any]:
    replacement = numeric_array(frame, spec.replacement)
    gate = frame[f"__gate__{spec.variant}__{spec.gate_name}"].to_numpy(bool)
    active = gate & np.isfinite(replacement) & np.isfinite(baseline)
    changed = active & (np.abs(replacement - baseline) > 1e-6)

    err = baseline_err.copy()
    err[active] = replacement[active] - true[active]
    score = score_error(err)
    subset_candidate = score_error(replacement - true, active)
    subset_baseline = score_error(baseline_err, active)

    err_sq = np.where(np.isfinite(err), err * err, 0.0)
    well_sse = np.bincount(well_codes, weights=err_sq, minlength=len(well_counts))
    well_rmse = np.sqrt(well_sse / np.maximum(well_counts, 1))
    well_delta = well_rmse - baseline_well_rmse
    max_well_regression = float(np.nanmax(well_delta)) if len(well_delta) else None
    improved_wells = int(np.sum(well_delta < -1e-9))
    regressed_wells = int(np.sum(well_delta > 1e-9))

    row: dict[str, Any] = {
        "policy": spec.policy,
        "variant": spec.variant,
        "gate": spec.gate_name,
        "replacement": spec.replacement,
        "rows": score["rows"],
        "rmse": score["rmse"],
        "mae": score["mae"],
        "within10": score["within10"],
        "bias": score["bias"],
        "delta_rmse_vs_baseline": (
            None
            if score["rmse"] is None or baseline_score["rmse"] is None
            else float(score["rmse"] - baseline_score["rmse"])
        ),
        "gate_rows": int(active.sum()),
        "gate_rate": float(active.mean()),
        "changed_rows": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "changed_wells": int(frame.loc[changed, "well"].nunique()) if changed.any() else 0,
        "changed_subset_baseline_rmse": subset_baseline["rmse"],
        "changed_subset_candidate_rmse": subset_candidate["rmse"],
        "changed_subset_delta_rmse": (
            None
            if subset_baseline["rmse"] is None or subset_candidate["rmse"] is None
            else float(subset_candidate["rmse"] - subset_baseline["rmse"])
        ),
        "max_well_regression_vs_baseline": max_well_regression,
        "improved_wells": improved_wells,
        "regressed_wells": regressed_wells,
    }
    for group_name, mask in groups.items():
        group_score = score_error(err, mask)
        group_baseline = baseline_group_scores[group_name]
        row[f"{group_name}_rmse"] = group_score["rmse"]
        row[f"{group_name}_delta_rmse_vs_baseline"] = (
            None
            if group_score["rmse"] is None or group_baseline["rmse"] is None
            else float(group_score["rmse"] - group_baseline["rmse"])
        )
    sep = numeric_array(frame, f"{spec.variant}_top1_top2_sep")
    sep_q75 = quantile(sep, 0.75)
    sep_q90 = quantile(sep, 0.90)
    for name, mask in {
        "mode_sep_q75": np.isfinite(sep) & (sep >= sep_q75),
        "mode_sep_q90": np.isfinite(sep) & (sep >= sep_q90),
    }.items():
        group_score = score_error(err, mask)
        group_baseline = score_error(baseline_err, mask)
        row[f"{name}_rmse"] = group_score["rmse"]
        row[f"{name}_delta_rmse_vs_baseline"] = (
            None
            if group_score["rmse"] is None or group_baseline["rmse"] is None
            else float(group_score["rmse"] - group_baseline["rmse"])
        )
    return row


def build_policy_error(
    frame: pd.DataFrame,
    spec: PolicySpec,
    baseline: np.ndarray,
    true: np.ndarray,
    baseline_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    replacement = numeric_array(frame, spec.replacement)
    gate = frame[f"__gate__{spec.variant}__{spec.gate_name}"].to_numpy(bool)
    active = gate & np.isfinite(replacement) & np.isfinite(baseline)
    err = baseline_err.copy()
    err[active] = replacement[active] - true[active]
    return err, active


def detailed_group_metrics(
    frame: pd.DataFrame,
    policies: list[PolicySpec],
    baseline: np.ndarray,
    true: np.ndarray,
    baseline_err: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    md_buckets = distance_bucket(frame["md_since"])
    bucket_masks = {
        f"md_since_{str(bucket)}": (md_buckets.astype(str).to_numpy() == str(bucket))
        for bucket in md_buckets.cat.categories
    }
    for spec in policies:
        err, active = build_policy_error(frame, spec, baseline, true, baseline_err)
        groups = {
            "all": np.ones(len(frame), dtype=bool),
            "changed_rows": active,
            "near_000_050": numeric_array(frame, "md_since") <= 50.0,
            "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
            **bucket_masks,
        }
        for group_name, mask in groups.items():
            policy_score = score_error(err, mask)
            baseline_score = score_error(baseline_err, mask)
            rows.append(
                {
                    "policy": spec.policy,
                    "variant": spec.variant,
                    "gate": spec.gate_name,
                    "replacement": spec.replacement,
                    "group": group_name,
                    **policy_score,
                    "baseline_rmse": baseline_score["rmse"],
                    "delta_rmse_vs_baseline": (
                        None
                        if policy_score["rmse"] is None or baseline_score["rmse"] is None
                        else float(policy_score["rmse"] - baseline_score["rmse"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def detailed_by_well(
    frame: pd.DataFrame,
    policies: list[PolicySpec],
    baseline: np.ndarray,
    true: np.ndarray,
    baseline_err: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in policies:
        err, active = build_policy_error(frame, spec, baseline, true, baseline_err)
        for well, idx in frame.groupby("well", sort=False).indices.items():
            positions = np.asarray(idx, dtype=np.int64)
            policy_score = score_error(err[positions])
            baseline_score = score_error(baseline_err[positions])
            changed = active[positions]
            rows.append(
                {
                    "well": str(well),
                    "policy": spec.policy,
                    "variant": spec.variant,
                    "gate": spec.gate_name,
                    "replacement": spec.replacement,
                    **policy_score,
                    "baseline_rmse": baseline_score["rmse"],
                    "delta_rmse_vs_baseline": (
                        None
                        if policy_score["rmse"] is None or baseline_score["rmse"] is None
                        else float(policy_score["rmse"] - baseline_score["rmse"])
                    ),
                    "changed_rows": int(changed.sum()),
                }
            )
    return pd.DataFrame(rows)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    frame, input_meta = read_parent_inputs(config)
    primary_baseline = str(get_nested(config, "audit.primary_baseline") or "likpf_mean")
    beam_baseline = str(get_nested(config, "audit.beam_baseline") or "beam_mean")
    variants = [str(value) for value in (get_nested(config, "gate.beam_variants") or [])]

    true = numeric_array(frame, "true_tvt")
    baseline = numeric_array(frame, primary_baseline)
    baseline_err = baseline - true
    baseline_score = score_error(baseline_err)
    well_codes, well_names = pd.factorize(frame["well"].astype(str), sort=False)
    well_counts = np.bincount(well_codes, minlength=len(well_names))
    baseline_well_sse = np.bincount(
        well_codes,
        weights=np.where(np.isfinite(baseline_err), baseline_err * baseline_err, 0.0),
        minlength=len(well_names),
    )
    baseline_well_rmse = np.sqrt(baseline_well_sse / np.maximum(well_counts, 1))
    groups = precompute_groups(frame, baseline, true, beam_baseline)
    baseline_group_scores = {
        name: score_error(baseline_err, mask) for name, mask in groups.items()
    }

    threshold_rows: list[dict[str, Any]] = []
    policy_specs: list[PolicySpec] = []
    for variant in variants:
        masks, thresholds = build_gate_masks(frame, variant, config)
        threshold_rows.extend(thresholds)
        for gate_name, mask in masks.items():
            frame[f"__gate__{variant}__{gate_name}"] = mask
        for replacement in input_meta["replacement_columns"]:
            if not replacement.startswith(f"{variant}_"):
                continue
            for gate_name in masks:
                policy_specs.append(
                    PolicySpec(
                        variant=variant,
                        gate_name=gate_name,
                        replacement=replacement,
                        policy=policy_name(variant, gate_name, replacement),
                    )
                )

    metric_rows = [
        {
            "policy": primary_baseline,
            "variant": "baseline",
            "gate": "none",
            "replacement": primary_baseline,
            "rows": baseline_score["rows"],
            "rmse": baseline_score["rmse"],
            "mae": baseline_score["mae"],
            "within10": baseline_score["within10"],
            "bias": baseline_score["bias"],
            "delta_rmse_vs_baseline": 0.0,
            "gate_rows": 0,
            "gate_rate": 0.0,
            "changed_rows": 0,
            "changed_rate": 0.0,
            "changed_wells": 0,
            "changed_subset_baseline_rmse": None,
            "changed_subset_candidate_rmse": None,
            "changed_subset_delta_rmse": None,
            "max_well_regression_vs_baseline": 0.0,
            "improved_wells": 0,
            "regressed_wells": 0,
        }
    ]
    for spec in policy_specs:
        metric_rows.append(
            evaluate_policy(
                frame=frame,
                spec=spec,
                baseline=baseline,
                true=true,
                baseline_err=baseline_err,
                baseline_score=baseline_score,
                baseline_group_scores=baseline_group_scores,
                groups=groups,
                well_codes=well_codes,
                well_counts=well_counts,
                baseline_well_rmse=baseline_well_rmse,
            )
        )

    policy_metrics = (
        pd.DataFrame(metric_rows)
        .sort_values(
            ["rmse", "changed_rows", "policy"],
            ascending=[True, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    threshold_frame = pd.DataFrame(threshold_rows)
    top_n = int(get_nested(config, "audit.top_n_detailed_policies") or 16)
    detailed_names = [
        name
        for name in policy_metrics["policy"].astype(str).tolist()
        if name != primary_baseline
    ][:top_n]
    detailed_specs = [spec for spec in policy_specs if spec.policy in set(detailed_names)]
    group_frame = detailed_group_metrics(frame, detailed_specs, baseline, true, baseline_err)
    by_well = detailed_by_well(frame, detailed_specs, baseline, true, baseline_err)

    artifacts = paths.artifacts_dir
    policy_metrics_path = artifacts / f"{OUTPUT_PREFIX}_policy_metrics.csv"
    threshold_path = artifacts / f"{OUTPUT_PREFIX}_gate_thresholds.csv"
    group_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    policy_metrics.to_csv(policy_metrics_path, index=False)
    threshold_frame.to_csv(threshold_path, index=False)
    group_frame.to_csv(group_path, index=False)
    by_well.to_csv(by_well_path, index=False)

    best_policy = (
        policy_metrics[~policy_metrics["policy"].eq(primary_baseline)].iloc[0].to_dict()
        if len(policy_metrics) > 1
        else None
    )
    max_regression_guard = float(get_nested(config, "audit.max_regression_guard_rmse") or 0.25)
    best_delta = None if best_policy is None else best_policy.get("delta_rmse_vs_baseline")
    best_subset_delta = (
        None if best_policy is None else best_policy.get("changed_subset_delta_rmse")
    )
    best_max_regression = (
        None if best_policy is None else best_policy.get("max_well_regression_vs_baseline")
    )
    supported = (
        best_delta is not None
        and float(best_delta) < 0.0
        and best_subset_delta is not None
        and float(best_subset_delta) < 0.0
        and best_max_regression is not None
        and float(best_max_regression) <= max_regression_guard
    )
    decision = {
        "recommendation": (
            "review_positive_gate_before_any_feature_or_inference_followup"
            if supported
            else "diagnostic_only_not_submit_candidate"
        ),
        "supported": bool(supported),
        "max_regression_guard_rmse": max_regression_guard,
        "reason": (
            "A gate must improve global RMSE, improve the changed subset, and keep "
            "worst-well regression within the configured guard before any downstream use."
        ),
    }
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - start),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "primary_baseline": {"candidate": primary_baseline, **baseline_score},
        "beam_baseline": beam_baseline,
        "input_artifacts": input_meta,
        "policy_count": int(len(policy_specs)),
        "gate_threshold_count": int(len(threshold_frame)),
        "best_policy": to_jsonable(best_policy),
        "decision": decision,
        "artifacts": {
            "policy_metrics": str(policy_metrics_path),
            "gate_thresholds": str(threshold_path),
            "group_metrics": str(group_path),
            "by_well": str(by_well_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "policy_metrics": sha256_path(policy_metrics_path),
            "gate_thresholds": sha256_path(threshold_path),
            "group_metrics": sha256_path(group_path),
            "by_well": sha256_path(by_well_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    summary["artifact_sha256"]["summary"] = sha256_path(summary_path)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")

    paths.metrics_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    "experiment": OUTPUT_PREFIX,
                    "status": (
                        "completed_train_side_supported_review_required"
                        if supported
                        else "completed_train_side_rejected_no_submit"
                    ),
                    "route": "pf_beam",
                    "parent": "exp173_beam_topk_path_posterior_audit",
                    "cv": None if best_policy is None else best_policy.get("rmse"),
                    "public_lb": None,
                    "private_lb": None,
                    "metric": "rmse",
                    "primary_baseline": {"candidate": primary_baseline, **baseline_score},
                    "best_policy": best_policy,
                    "decision": decision,
                    "summary_path": str(summary_path),
                    "updated_at": datetime.now(UTC).date().isoformat(),
                    "key_idea": (
                        "No-training posthoc gate over exp173 Beam top-K posterior outputs "
                        "using target-free bimodal diagnostics."
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_policy"]), indent=2, sort_keys=True))
