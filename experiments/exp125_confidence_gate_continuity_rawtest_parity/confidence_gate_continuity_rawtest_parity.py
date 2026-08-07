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

OUTPUT_PREFIX = "exp125_confidence_gate_continuity_rawtest_parity"


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


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    cols = ["surface", "variant", "id", "prediction"]
    for row in frame[cols].itertuples(index=False):
        digest.update(str(row.surface).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.variant).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row.prediction).tobytes())
        digest.update(b"\n")
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


def row_index_from_id(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce")
    if values.isna().any():
        bad = ids[values.isna()].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype("int64")


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def tail_rank_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


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


def load_oof_predictions(
    *,
    source_name: str,
    path: Path,
    variants: list[str],
    value_col: str,
    true_col: str,
    selected_col: str | None,
    md_since_col: str | None,
    max_rows_per_variant: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["id", "well", "variant", value_col, true_col]
    if selected_col:
        required.append(selected_col)
    if md_since_col:
        required.append(md_since_col)
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    usecols = required.copy()
    if "mode" in header:
        usecols.append("mode")
    if "fold" in header:
        usecols.append("fold")

    chunks: list[pd.DataFrame] = []
    counts = {variant: 0 for variant in variants}
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"id": str, "well": str, "variant": str},
        chunksize=500_000,
        low_memory=False,
    ):
        part = chunk[chunk["variant"].isin(variants)].copy()
        if part.empty:
            continue
        if max_rows_per_variant is not None:
            limited_parts: list[pd.DataFrame] = []
            for variant, group in part.groupby("variant", sort=False):
                remaining = int(max_rows_per_variant) - counts.get(str(variant), 0)
                if remaining <= 0:
                    continue
                take = group.head(remaining).copy()
                counts[str(variant)] = counts.get(str(variant), 0) + len(take)
                limited_parts.append(take)
            if not limited_parts:
                continue
            part = pd.concat(limited_parts, ignore_index=True)
        chunks.append(part)

    if not chunks:
        raise ValueError(f"No requested variants found in {path}: {variants}")
    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.rename(columns={value_col: "prediction", true_col: "true_tvt"})
    frame["surface"] = source_name
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce").astype(np.float64)
    frame["true_tvt"] = pd.to_numeric(frame["true_tvt"], errors="coerce").astype(np.float64)
    frame["row_index"] = row_index_from_id(frame["id"])
    if selected_col:
        frame = frame.rename(columns={selected_col: "selected_candidate"})
        frame["selected_candidate"] = frame["selected_candidate"].astype(str)
    else:
        frame["selected_candidate"] = pd.NA
    if md_since_col:
        frame = frame.rename(columns={md_since_col: "md_since"})
        frame["md_since"] = pd.to_numeric(frame["md_since"], errors="coerce")
    else:
        frame["md_since"] = np.nan
    frame["error"] = frame["prediction"] - frame["true_tvt"]
    frame["abs_error"] = frame["error"].abs()
    frame["within_10ft"] = frame["abs_error"].le(10.0)
    frame["tail_rank_bucket"] = tail_rank_bucket(frame["row_index"]).astype(str)
    if frame["md_since"].notna().any():
        frame["distance_bucket"] = distance_bucket(frame["md_since"]).astype(str)
    else:
        frame["distance_bucket"] = "missing"

    present_variants = sorted(frame["variant"].unique().tolist())
    meta = {
        "path": str(path),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "variants": present_variants,
        "raw_file_sha256": sha256_path(path),
        "decompressed_content_sha256": sha256_path(path, decompressed=path.suffix == ".gz"),
    }
    missing_variants = sorted(set(variants) - set(present_variants))
    if missing_variants:
        meta["missing_variants"] = missing_variants
    return frame, meta


def load_dense_gate_if_available(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filename = str(
        get_nested(
            config,
            "data.optional_dense_gate_predictions_filename",
            "exp124_projection_confidence_error_map_gate_predictions.csv.gz",
        )
    )
    source = find_input_file(
        filename,
        get_nested(config, "data.optional_dense_gate_predictions"),
        local_roots=[Path("experiments/exp124_projection_confidence_error_map/kaggle/output/train_v1")],
        required=False,
    )
    if source is None:
        return pd.DataFrame(), {
            "available": False,
            "reason": "optional dense/high-drift gate predictions were not found",
            "filename": filename,
        }
    header = pd.read_csv(source, nrows=0).columns.tolist()
    required = ["id", "well", "variant", "prediction", "true_tvt"]
    missing = [col for col in required if col not in header]
    if missing:
        return pd.DataFrame(), {
            "available": False,
            "reason": f"optional dense gate file exists but is missing columns: {missing}",
            "path": str(source),
        }
    frame = pd.read_csv(source, usecols=required, dtype={"id": str, "well": str, "variant": str})
    frame["surface"] = "dense_high_drift_optional"
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["true_tvt"] = pd.to_numeric(frame["true_tvt"], errors="coerce")
    frame["selected_candidate"] = pd.NA
    frame["md_since"] = np.nan
    frame["row_index"] = row_index_from_id(frame["id"])
    frame["error"] = frame["prediction"] - frame["true_tvt"]
    frame["abs_error"] = frame["error"].abs()
    frame["within_10ft"] = frame["abs_error"].le(10.0)
    frame["tail_rank_bucket"] = tail_rank_bucket(frame["row_index"]).astype(str)
    frame["distance_bucket"] = "missing"
    return frame, {
        "available": True,
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "raw_file_sha256": sha256_path(source),
        "decompressed_content_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
    }


def metric_rows(frame: pd.DataFrame, *, surface_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (surface, variant), group in frame.groupby(["surface", "variant"], sort=False):
        error = group["prediction"].to_numpy(np.float64) - group["true_tvt"].to_numpy(np.float64)
        rows.append(
            {
                "surface_scope": surface_label,
                "surface": surface,
                "variant": variant,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
                "rmse_tvt": rmse_from_error(error),
                "mae_tvt": mae_from_error(error),
                "within_10ft": float(np.mean(np.abs(error) <= 10.0)),
                "prediction_mean": float(group["prediction"].mean()),
                "prediction_std": float(group["prediction"].std()),
                "true_tvt_mean": float(group["true_tvt"].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_baseline_deltas(
    metrics: pd.DataFrame, baseline_map: dict[str, str], *, suffix: str = ""
) -> pd.DataFrame:
    result = metrics.copy()
    baseline_rows = {}
    for surface, baseline_variant in baseline_map.items():
        match = result[(result["surface"].eq(surface)) & (result["variant"].eq(baseline_variant))]
        if not match.empty:
            baseline_rows[surface] = match.iloc[0].to_dict()
    result[f"delta_rmse_vs_surface_baseline{suffix}"] = np.nan
    result[f"delta_within10_vs_surface_baseline{suffix}"] = np.nan
    for idx, row in result.iterrows():
        base = baseline_rows.get(str(row["surface"]))
        if not base:
            continue
        result.loc[idx, f"delta_rmse_vs_surface_baseline{suffix}"] = (
            float(row["rmse_tvt"]) - float(base["rmse_tvt"])
        )
        result.loc[idx, f"delta_within10_vs_surface_baseline{suffix}"] = (
            float(row["within_10ft"]) - float(base["within_10ft"])
        )
    return result


def by_well_metrics(frame: pd.DataFrame, baseline_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (surface, variant, well), group in frame.groupby(["surface", "variant", "well"], sort=False):
        error = group["prediction"].to_numpy(np.float64) - group["true_tvt"].to_numpy(np.float64)
        rows.append(
            {
                "surface": surface,
                "variant": variant,
                "well": well,
                "rows": int(len(group)),
                "rmse_tvt": rmse_from_error(error),
                "mae_tvt": mae_from_error(error),
                "within_10ft": float(np.mean(np.abs(error) <= 10.0)),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[
        result.apply(lambda row: row["variant"] == baseline_map.get(str(row["surface"])), axis=1)
    ][["surface", "well", "rmse_tvt", "within_10ft"]].rename(
        columns={
            "rmse_tvt": "baseline_rmse_tvt",
            "within_10ft": "baseline_within_10ft",
        }
    )
    result = result.merge(baseline, on=["surface", "well"], how="left")
    result["delta_rmse_vs_surface_baseline"] = result["rmse_tvt"] - result["baseline_rmse_tvt"]
    result["delta_within10_vs_surface_baseline"] = (
        result["within_10ft"] - result["baseline_within_10ft"]
    )
    return result.sort_values(["surface", "variant", "delta_rmse_vs_surface_baseline"], ascending=[True, True, False])


def bucket_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (surface, variant, bucket), group in frame.groupby(
        ["surface", "variant", column], dropna=False, sort=False
    ):
        error = group["prediction"].to_numpy(np.float64) - group["true_tvt"].to_numpy(np.float64)
        rows.append(
            {
                "bucket_type": column,
                "bucket": str(bucket),
                "surface": surface,
                "variant": variant,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
                "rmse_tvt": rmse_from_error(error),
                "mae_tvt": mae_from_error(error),
                "within_10ft": float(np.mean(np.abs(error) <= 10.0)),
            }
        )
    return pd.DataFrame(rows)


def segment_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            lengths.append(idx - start)
            start = None
    if start is not None:
        lengths.append(len(mask) - start)
    return lengths


def continuity_metrics(frame: pd.DataFrame, baseline_map: dict[str, str]) -> pd.DataFrame:
    baseline_predictions = {}
    for surface, baseline_variant in baseline_map.items():
        base = frame[(frame["surface"].eq(surface)) & (frame["variant"].eq(baseline_variant))]
        if base.empty:
            continue
        baseline_predictions[surface] = base.set_index("id")["prediction"]

    rows: list[dict[str, Any]] = []
    for (surface, variant, well), group in frame.groupby(["surface", "variant", "well"], sort=False):
        ordered = group.sort_values("row_index").copy()
        pred = ordered["prediction"].to_numpy(np.float64)
        steps = np.abs(np.diff(pred)) if len(pred) > 1 else np.asarray([], dtype=np.float64)
        selected = ordered["selected_candidate"].astype("string")
        candidate_switches = np.nan
        if selected.notna().any():
            candidate_switches = int((selected != selected.shift()).iloc[1:].sum())
        base_series = baseline_predictions.get(str(surface))
        if base_series is not None:
            base_pred = ordered["id"].map(base_series).to_numpy(np.float64)
            changed = np.abs(pred - base_pred) > 1e-6
        else:
            changed = np.zeros(len(ordered), dtype=bool)
        lengths = segment_lengths(changed)
        change_transitions = int(np.count_nonzero(changed[1:] != changed[:-1])) if len(changed) > 1 else 0
        rows.append(
            {
                "surface": surface,
                "variant": variant,
                "well": well,
                "rows": int(len(ordered)),
                "changed_rows_vs_surface_baseline": int(np.count_nonzero(changed)),
                "changed_rate_vs_surface_baseline": float(np.mean(changed)) if len(changed) else 0.0,
                "change_transition_count": change_transitions,
                "changed_segment_count": int(len(lengths)),
                "changed_segment_min_length": int(min(lengths)) if lengths else 0,
                "changed_segment_median_length": float(np.median(lengths)) if lengths else 0.0,
                "candidate_switch_count": candidate_switches,
                "prediction_step_p95": float(np.percentile(steps, 95)) if len(steps) else 0.0,
                "prediction_step_max": float(np.max(steps)) if len(steps) else 0.0,
                "prediction_step_ge10": int(np.count_nonzero(steps >= 10.0)),
                "prediction_step_ge25": int(np.count_nonzero(steps >= 25.0)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_continuity(by_well: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (surface, variant), group in by_well.groupby(["surface", "variant"], sort=False):
        rows.append(
            {
                "surface": surface,
                "variant": variant,
                "wells": int(len(group)),
                "rows": int(group["rows"].sum()),
                "changed_rows_vs_surface_baseline": int(group["changed_rows_vs_surface_baseline"].sum()),
                "changed_rate_vs_surface_baseline": float(
                    group["changed_rows_vs_surface_baseline"].sum() / max(1, group["rows"].sum())
                ),
                "change_transition_count": int(group["change_transition_count"].sum()),
                "changed_segment_count": int(group["changed_segment_count"].sum()),
                "min_changed_segment_length": int(group["changed_segment_min_length"].min()),
                "median_changed_segment_length": float(group["changed_segment_median_length"].median()),
                "candidate_switch_count": (
                    float(group["candidate_switch_count"].sum())
                    if group["candidate_switch_count"].notna().any()
                    else np.nan
                ),
                "prediction_step_p95_median_by_well": float(group["prediction_step_p95"].median()),
                "prediction_step_max": float(group["prediction_step_max"].max()),
                "prediction_step_ge10": int(group["prediction_step_ge10"].sum()),
                "prediction_step_ge25": int(group["prediction_step_ge25"].sum()),
            }
        )
    return pd.DataFrame(rows)


def common_worst_metrics(
    by_well: pd.DataFrame, baseline_surface: str, baseline_variant: str, top_ns: list[int]
) -> pd.DataFrame:
    baseline = by_well[
        by_well["surface"].eq(baseline_surface) & by_well["variant"].eq(baseline_variant)
    ].sort_values("rmse_tvt", ascending=False)
    rows: list[dict[str, Any]] = []
    for top_n in top_ns:
        wells = set(baseline.head(int(top_n))["well"].astype(str))
        subset = by_well[by_well["well"].astype(str).isin(wells)]
        for (surface, variant), group in subset.groupby(["surface", "variant"], sort=False):
            rows.append(
                {
                    "worst_reference": f"{baseline_surface}:{baseline_variant}",
                    "top_n": int(top_n),
                    "surface": surface,
                    "variant": variant,
                    "wells": int(group["well"].nunique()),
                    "mean_rmse_tvt": float(group["rmse_tvt"].mean()),
                    "median_rmse_tvt": float(group["rmse_tvt"].median()),
                    "mean_delta_rmse_vs_surface_baseline": float(
                        group["delta_rmse_vs_surface_baseline"].mean()
                    ),
                    "max_delta_rmse_vs_surface_baseline": float(
                        group["delta_rmse_vs_surface_baseline"].max()
                    ),
                    "regression_wells_gt_1ft": int(
                        group["delta_rmse_vs_surface_baseline"].gt(1.0).sum()
                    ),
                    "regression_wells_gt_2ft": int(
                        group["delta_rmse_vs_surface_baseline"].gt(2.0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def rawtest_parity_check(config: dict[str, Any], source_meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checks = get_nested(config, "audit.rawtest_parity_checks", []) or []
    for item in checks:
        name = str(item.get("name", "unnamed_check"))
        filename = str(item.get("filename", ""))
        required = bool(item.get("required", True))
        configured = item.get("paths")
        path = find_input_file(filename, configured, required=False) if filename else None
        status = "pass" if path else ("missing_required" if required else "missing_optional")
        rows.append(
            {
                "check": name,
                "filename": filename,
                "required": required,
                "status": status,
                "path": str(path) if path else "",
                "raw_file_sha256": sha256_path(path) if path else "",
                "decompressed_content_sha256": (
                    sha256_path(path, decompressed=path.suffix == ".gz") if path else ""
                ),
            }
        )
    for source, meta in source_meta.items():
        rows.append(
            {
                "check": f"{source}_oof_prediction_input",
                "filename": Path(str(meta.get("path", ""))).name,
                "required": True,
                "status": "pass",
                "path": str(meta.get("path", "")),
                "raw_file_sha256": str(meta.get("raw_file_sha256", "")),
                "decompressed_content_sha256": str(meta.get("decompressed_content_sha256", "")),
            }
        )
    return pd.DataFrame(rows)


def build_fair_shared_surface(frame: pd.DataFrame, required_variants: list[dict[str, str]]) -> pd.DataFrame:
    if not required_variants:
        return frame.copy()
    labels = [f"{item['surface']}::{item['variant']}" for item in required_variants]
    temp = frame.copy()
    temp["_label"] = temp["surface"].astype(str) + "::" + temp["variant"].astype(str)
    counts = temp[temp["_label"].isin(labels)].groupby("id")["_label"].nunique()
    keep_ids = set(counts[counts.eq(len(labels))].index.astype(str))
    fair = temp[temp["id"].isin(keep_ids)].drop(columns=["_label"]).copy()
    return fair


def write_artifact_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# exp125 confidence gate continuity raw-test parity artifacts",
        "",
        "This directory contains a train-side posthoc comparison of confidence gate candidates.",
        "",
        "## Files",
        "",
        f"- `{OUTPUT_PREFIX}_metrics.csv`: overall metrics on full available and fair shared surfaces.",
        f"- `{OUTPUT_PREFIX}_by_well.csv`: per-well RMSE and regression deltas.",
        f"- `{OUTPUT_PREFIX}_bucket_metrics.csv`: tail-rank and distance bucket metrics.",
        f"- `{OUTPUT_PREFIX}_continuity_by_well.csv`: per-well path continuity diagnostics.",
        f"- `{OUTPUT_PREFIX}_continuity_summary.csv`: aggregate switch and jump diagnostics.",
        f"- `{OUTPUT_PREFIX}_common_worst_metrics.csv`: deltas on baseline worst-well subsets.",
        f"- `{OUTPUT_PREFIX}_rawtest_parity_checklist.csv`: input and manifest availability checks.",
        f"- `{OUTPUT_PREFIX}_summary.json`: machine-readable summary.",
        "",
        "## Decision Context",
        "",
        f"- Dense/high-drift optional input available: {summary['optional_dense_gate'].get('available')}",
        f"- Fair shared rows: {summary['fair_shared_surface']['rows']}",
        f"- Recommendation: {summary['decision']['recommendation']}",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def run_audit(config: dict[str, Any], paths: Any, *, max_rows_per_variant: int | None = None) -> dict[str, Any]:
    start = time.time()
    artifacts_dir = Path(paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    exp102_path = find_input_file(
        "exp102_confidence_gated_likpf_fallback_on_exp101_oof_predictions.csv.gz",
        get_nested(config, "data.exp102_oof_predictions"),
        local_roots=[
            Path("experiments/exp102_confidence_gated_likpf_fallback_on_exp101/kaggle/output/train_v2"),
        ],
    )
    exp112_path = find_input_file(
        "exp112_learned_pf_likelihood_weight_or_feature_followup_oof_predictions.csv.gz",
        get_nested(config, "data.exp112_oof_predictions"),
        local_roots=[
            Path("experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/kaggle/output/train_v1"),
        ],
    )
    assert exp102_path is not None
    assert exp112_path is not None

    exp102_variants = list(get_nested(config, "audit.exp102_variants", []))
    exp112_variants = list(get_nested(config, "audit.exp112_variants", []))
    if not exp102_variants or not exp112_variants:
        raise ValueError("audit.exp102_variants and audit.exp112_variants must be configured")

    exp102, exp102_meta = load_oof_predictions(
        source_name="exp102",
        path=exp102_path,
        variants=exp102_variants,
        value_col="selected_tvt",
        true_col="true_tvt",
        selected_col="selected_candidate",
        md_since_col=None,
        max_rows_per_variant=max_rows_per_variant,
    )
    exp112, exp112_meta = load_oof_predictions(
        source_name="exp112",
        path=exp112_path,
        variants=exp112_variants,
        value_col="prediction",
        true_col="true_tvt",
        selected_col=None,
        md_since_col="md_since",
        max_rows_per_variant=max_rows_per_variant,
    )
    dense, dense_meta = load_dense_gate_if_available(config)
    frames = [exp102, exp112]
    if not dense.empty:
        frames.append(dense)
    combined = pd.concat(frames, ignore_index=True)

    # Borrow exp112 md_since for matching ids so exp102 shared rows can be bucketed by distance.
    id_md = exp112.drop_duplicates("id").set_index("id")["md_since"]
    missing_md = combined["md_since"].isna()
    combined.loc[missing_md, "md_since"] = combined.loc[missing_md, "id"].map(id_md)
    combined["distance_bucket"] = distance_bucket(combined["md_since"]).astype(str)

    required_shared = get_nested(config, "audit.required_shared_variants", []) or []
    fair = build_fair_shared_surface(combined, required_shared)
    if fair.empty:
        raise ValueError("fair shared surface is empty; check required_shared_variants")

    baseline_map = {
        str(key): str(value)
        for key, value in (get_nested(config, "audit.surface_baseline_variants", {}) or {}).items()
    }
    full_metrics = add_baseline_deltas(metric_rows(combined, surface_label="available"), baseline_map)
    fair_metrics = add_baseline_deltas(metric_rows(fair, surface_label="fair_shared"), baseline_map)
    metrics = pd.concat([full_metrics, fair_metrics], ignore_index=True)

    fair_by_well = by_well_metrics(fair, baseline_map)
    fair_bucket = pd.concat(
        [bucket_metrics(fair, "distance_bucket"), bucket_metrics(fair, "tail_rank_bucket")],
        ignore_index=True,
    )
    continuity_by_well = continuity_metrics(fair, baseline_map)
    continuity_summary = aggregate_continuity(continuity_by_well)
    common_worst = common_worst_metrics(
        fair_by_well,
        baseline_surface=str(get_nested(config, "audit.common_worst_reference.surface", "exp102")),
        baseline_variant=str(
            get_nested(config, "audit.common_worst_reference.variant", "likpf_mean_single")
        ),
        top_ns=[int(value) for value in get_nested(config, "audit.common_worst_top_n", [26, 50])],
    )

    source_meta = {
        "exp102": exp102_meta,
        "exp112": exp112_meta,
    }
    if dense_meta.get("available"):
        source_meta["dense_high_drift_optional"] = dense_meta
    parity = rawtest_parity_check(config, source_meta)

    metrics_path = artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    by_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    bucket_path = artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    continuity_by_well_path = artifacts_dir / f"{OUTPUT_PREFIX}_continuity_by_well.csv"
    continuity_summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_continuity_summary.csv"
    common_worst_path = artifacts_dir / f"{OUTPUT_PREFIX}_common_worst_metrics.csv"
    parity_path = artifacts_dir / f"{OUTPUT_PREFIX}_rawtest_parity_checklist.csv"
    prediction_sample_path = artifacts_dir / f"{OUTPUT_PREFIX}_prediction_sample.csv.gz"

    metrics.to_csv(metrics_path, index=False)
    fair_by_well.to_csv(by_well_path, index=False)
    fair_bucket.to_csv(bucket_path, index=False)
    continuity_by_well.to_csv(continuity_by_well_path, index=False)
    continuity_summary.to_csv(continuity_summary_path, index=False)
    common_worst.to_csv(common_worst_path, index=False)
    parity.to_csv(parity_path, index=False)
    fair.head(int(get_nested(config, "audit.prediction_sample_rows", 200_000))).to_csv(
        prediction_sample_path, index=False, compression="gzip"
    )

    fair_metrics_only = metrics[metrics["surface_scope"].eq("fair_shared")].copy()
    best_row = fair_metrics_only.sort_values("rmse_tvt").head(1).iloc[0].to_dict()
    required_parity_missing = parity[
        parity["required"].astype(bool) & ~parity["status"].eq("pass")
    ]
    continuity_fail = continuity_summary[
        continuity_summary["prediction_step_ge25"].gt(
            int(get_nested(config, "audit.guardrails.max_prediction_step_ge25", 0))
        )
    ]
    by_well_fail = fair_by_well[
        fair_by_well["delta_rmse_vs_surface_baseline"].gt(
            float(get_nested(config, "audit.guardrails.max_well_regression_ft", 2.0))
        )
    ]
    direct_candidate_supported = (
        required_parity_missing.empty
        and continuity_fail.empty
        and by_well_fail.empty
        and bool(dense_meta.get("available"))
    )
    recommendation = (
        "direct_gate_candidate_supported_for_inference_design"
        if direct_candidate_supported
        else "keep_as_train_side_diagnostic_or_ml_feature_input"
    )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_train_side_posthoc_audit",
        "runtime_seconds": time.time() - start,
        "source_meta": source_meta,
        "optional_dense_gate": dense_meta,
        "available_surface": {
            "rows": int(len(combined)),
            "wells": int(combined["well"].nunique()),
            "variant_count": int(combined[["surface", "variant"]].drop_duplicates().shape[0]),
        },
        "fair_shared_surface": {
            "rows": int(len(fair)),
            "wells": int(fair["well"].nunique()),
            "variant_count": int(fair[["surface", "variant"]].drop_duplicates().shape[0]),
        },
        "best_fair_shared_variant": best_row,
        "guardrails": {
            "required_parity_missing_count": int(len(required_parity_missing)),
            "continuity_fail_variant_count": int(len(continuity_fail)),
            "well_regression_fail_rows": int(len(by_well_fail)),
            "max_well_regression_ft": float(
                fair_by_well["delta_rmse_vs_surface_baseline"].max()
            ),
        },
        "decision": {
            "direct_submission_candidate": bool(direct_candidate_supported),
            "recommendation": recommendation,
            "reason": (
                "Dense/high-drift gate predictions and raw-test regeneration evidence are required "
                "before any inference port. This run compares saved OOF gates on the fair shared surface."
            ),
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "by_well": str(by_well_path),
            "bucket_metrics": str(bucket_path),
            "continuity_by_well": str(continuity_by_well_path),
            "continuity_summary": str(continuity_summary_path),
            "common_worst_metrics": str(common_worst_path),
            "rawtest_parity_checklist": str(parity_path),
            "prediction_sample": str(prediction_sample_path),
        },
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "by_well": sha256_path(by_well_path),
            "bucket_metrics": sha256_path(bucket_path),
            "continuity_by_well": sha256_path(continuity_by_well_path),
            "continuity_summary": sha256_path(continuity_summary_path),
            "common_worst_metrics": sha256_path(common_worst_path),
            "rawtest_parity_checklist": sha256_path(parity_path),
            "prediction_sample_decompressed": sha256_path(prediction_sample_path, decompressed=True),
            "fair_prediction_content": prediction_sha256(fair),
        },
    }
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    summary["artifacts"]["summary"] = str(summary_path)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_artifact_readme(artifacts_dir, summary)
    return to_jsonable(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp125 confidence gate parity audit.")
    parser.add_argument("--max-rows-per-variant", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    from settings import ExperimentPaths, load_config

    args = parse_args()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    summary = run_audit(load_config(), paths, max_rows_per_variant=args.max_rows_per_variant)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
