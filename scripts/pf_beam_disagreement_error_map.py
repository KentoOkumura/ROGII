from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config_utils import ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package in tests.
    from scripts.config_utils import ROOT


DEFAULT_WELL_SUMMARY = (
    "experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/"
    "pf_beam_true_tvt_2d_well_eda_clean_all_well_summary.csv"
)
DEFAULT_PLOT_MANIFEST = (
    "experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/"
    "pf_beam_true_tvt_2d_well_eda_clean_all_plot_manifest.csv"
)
DEFAULT_OUTPUT_DIR = "studies/pf_beam_disagreement_error_map"

TAIL_LENGTH_BINS = [-np.inf, 1000, 2500, 5000, np.inf]
TAIL_LENGTH_LABELS = ["0000-1000", "1000-2500", "2500-5000", "5000+"]
PREFIX_LENGTH_BINS = [-np.inf, 500, 1000, 1500, 2500, np.inf]
PREFIX_LENGTH_LABELS = ["0000-0500", "0500-1000", "1000-1500", "1500-2500", "2500+"]
ERROR_GAP_BINS = [-np.inf, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, np.inf]
ERROR_GAP_LABELS = [
    "<-10",
    "-10--5",
    "-5--2",
    "-2-0",
    "0-2",
    "2-5",
    "5-10",
    "10+",
]


@dataclass(frozen=True)
class ErrorMapOutputs:
    overall: pd.DataFrame
    well_map: pd.DataFrame
    bucket_metrics: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build well-level PF/Beam disagreement error maps from exp083 well summaries. "
            "Optionally join exp073 by-well OOF metrics to compare PF/Beam against the "
            "current ML anchor."
        )
    )
    parser.add_argument("--well-summary", default=DEFAULT_WELL_SUMMARY)
    parser.add_argument("--plot-manifest", default=DEFAULT_PLOT_MANIFEST)
    parser.add_argument(
        "--ml-well-metrics",
        default="",
        help=(
            "Optional by-well model metrics CSV, for example exp073 "
            "exp063_full_replay_repro_guard_by_well.csv."
        ),
    )
    parser.add_argument(
        "--ml-model",
        default="lgb_mean",
        help="Model name to select from --ml-well-metrics when a model column is present.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_required(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{display_path(path)} is missing required columns: {missing}")
    return frame


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def to_numeric_column(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def read_plot_manifest(path: Path) -> pd.DataFrame:
    manifest = read_csv_required(path, {"well_id", "plot_path"})
    keep_columns = ["well_id", "plot_path"]
    if "reason" in manifest.columns:
        keep_columns.append("reason")
    manifest = manifest[keep_columns].copy()
    manifest["well_id"] = manifest["well_id"].astype(str)
    return manifest.drop_duplicates("well_id", keep="first")


def read_ml_well_metrics(path: Path, model: str) -> pd.DataFrame:
    frame = read_csv_required(path, {"well", "rows", "rmse_tvt"})
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == str(model)].copy()
        if frame.empty:
            raise ValueError(f"{display_path(path)} has no rows for model={model!r}")
    columns = ["well", "rows", "rmse_tvt"]
    optional = [
        column for column in ["mode", "model", "error_mean", "error_abs_mean"] if column in frame
    ]
    frame = frame[columns + optional].copy()
    rename = {
        "well": "well_id",
        "rows": "ml_rows",
        "rmse_tvt": "ml_rmse",
        "error_mean": "ml_error_mean",
        "error_abs_mean": "ml_error_abs_mean",
        "mode": "ml_mode",
        "model": "ml_model",
    }
    frame = frame.rename(columns=rename)
    frame["well_id"] = frame["well_id"].astype(str)
    return frame.drop_duplicates("well_id", keep="first")


def qbucket(values: pd.Series, prefix: str, quantiles: int = 4) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        return pd.Series("missing", index=values.index, dtype=object)
    bins = min(int(quantiles), int(valid.nunique()))
    try:
        bucket = pd.qcut(numeric, q=bins, duplicates="drop")
    except ValueError:
        return pd.Series("missing", index=values.index, dtype=object)
    labels: list[str] = []
    for interval in bucket.cat.categories:
        labels.append(f"{prefix}_q{len(labels) + 1}_{interval.left:.3g}_{interval.right:.3g}")
    return bucket.cat.rename_categories(labels).astype(object).fillna("missing")


def fixed_bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(numeric, bins=bins, labels=labels, right=False).astype(object).fillna("missing")


def prepare_well_map(
    well_summary: pd.DataFrame,
    plot_manifest: pd.DataFrame | None = None,
    ml_well_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = well_summary.copy()
    if "well_id" not in frame.columns:
        raise ValueError("well_summary must contain well_id")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["rows"] = to_numeric_column(frame, "rows").astype("Int64")

    pf_col = first_existing(frame, ["primary_pf_rmse", "pf_ancc_rmse"])
    beam_col = first_existing(frame, ["primary_beam_rmse", "beam_mean_rmse"])
    last_anchor_col = first_existing(frame, ["anchor_rmse", "last_anchor_tvt_rmse"])
    disagreement_col = first_existing(
        frame, ["pf_beam_abs_diff_mean", "source_pf_beam_abs_diff_mean"]
    )
    if pf_col is None or beam_col is None or disagreement_col is None:
        raise ValueError(
            "well_summary must include PF RMSE, Beam RMSE, and PF/Beam disagreement columns"
        )

    frame["pf_rmse"] = to_numeric_column(frame, pf_col)
    frame["beam_rmse"] = to_numeric_column(frame, beam_col)
    frame["last_anchor_rmse"] = (
        to_numeric_column(frame, last_anchor_col) if last_anchor_col else np.nan
    )
    frame["pf_beam_abs_diff_mean"] = to_numeric_column(frame, disagreement_col)
    frame["pf_beam_abs_diff_p95"] = to_numeric_column(frame, "pf_beam_abs_diff_p95")
    frame["pf_minus_beam_rmse"] = frame["pf_rmse"] - frame["beam_rmse"]
    frame["beam_minus_pf_rmse"] = frame["beam_rmse"] - frame["pf_rmse"]
    frame["last_anchor_minus_pf_rmse"] = frame["last_anchor_rmse"] - frame["pf_rmse"]

    frame["pf_std_mean"] = to_numeric_column(frame, "pf_ancc_std_mean")
    frame["beam_spread_mean"] = to_numeric_column(frame, "beam_std_d_mean")
    frame["prefix_length"] = to_numeric_column(frame, "known_len_mean")
    frame["eval_length"] = to_numeric_column(frame, "eval_len_mean").fillna(
        frame["rows"].astype(float)
    )
    frame["pfx_rmse_mean"] = to_numeric_column(frame, "pfx_rmse_mean")
    frame["likpf_abs_delta_mean"] = to_numeric_column(frame, "likpf_mean_d_mean").abs()
    frame["eval_to_prefix_ratio"] = frame["eval_length"] / frame["prefix_length"].replace(0, np.nan)

    if plot_manifest is not None:
        frame = frame.merge(plot_manifest, on="well_id", how="left")
    if ml_well_metrics is not None:
        frame = frame.merge(ml_well_metrics, on="well_id", how="left")
    else:
        frame["ml_rmse"] = np.nan
        frame["ml_error_mean"] = np.nan
        frame["ml_error_abs_mean"] = np.nan
        frame["ml_model"] = ""
        frame["ml_mode"] = ""

    frame["pf_minus_ml_rmse"] = frame["pf_rmse"] - frame["ml_rmse"]
    frame["beam_minus_ml_rmse"] = frame["beam_rmse"] - frame["ml_rmse"]

    engine_columns = {
        "pf": "pf_rmse",
        "beam": "beam_rmse",
        "ml": "ml_rmse",
        "last_anchor": "last_anchor_rmse",
    }
    best_engines: list[str] = []
    for _, row in frame.iterrows():
        candidates = {
            name: float(row[column])
            for name, column in engine_columns.items()
            if column in frame.columns and pd.notna(row[column]) and np.isfinite(float(row[column]))
        }
        best_engines.append(min(candidates, key=candidates.get) if candidates else "missing")
    frame["best_engine"] = best_engines
    frame["pf_beats_ml"] = frame["pf_rmse"] < frame["ml_rmse"]
    frame["beam_beats_ml"] = frame["beam_rmse"] < frame["ml_rmse"]
    frame["pf_beats_beam"] = frame["pf_rmse"] < frame["beam_rmse"]

    frame["tail_length_bucket"] = fixed_bucket(
        frame["eval_length"], TAIL_LENGTH_BINS, TAIL_LENGTH_LABELS
    )
    frame["prefix_length_bucket"] = fixed_bucket(
        frame["prefix_length"], PREFIX_LENGTH_BINS, PREFIX_LENGTH_LABELS
    )
    frame["pf_minus_beam_rmse_bucket"] = fixed_bucket(
        frame["pf_minus_beam_rmse"], ERROR_GAP_BINS, ERROR_GAP_LABELS
    )
    frame["pf_minus_ml_rmse_bucket"] = fixed_bucket(
        frame["pf_minus_ml_rmse"], ERROR_GAP_BINS, ERROR_GAP_LABELS
    )
    frame["pf_beam_disagreement_bucket"] = qbucket(
        frame["pf_beam_abs_diff_mean"], "pf_beam_disagreement"
    )
    frame["pf_std_bucket"] = qbucket(frame["pf_std_mean"], "pf_std")
    frame["beam_spread_bucket"] = qbucket(frame["beam_spread_mean"], "beam_spread")
    frame["pfx_rmse_bucket"] = qbucket(frame["pfx_rmse_mean"], "pfx_rmse")
    frame["likpf_abs_delta_bucket"] = qbucket(frame["likpf_abs_delta_mean"], "likpf_abs_delta")

    return frame


def pooled_rmse(frame: pd.DataFrame, rmse_column: str) -> float:
    if rmse_column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[rmse_column], errors="coerce")
    weights = pd.to_numeric(frame["rows"], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(
        math.sqrt(
            np.average(np.square(values[valid].to_numpy(dtype=float)), weights=weights[valid])
        )
    )


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["rows"], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def aggregate_segment(
    frame: pd.DataFrame, segment_type: str, segment: str
) -> dict[str, float | int | str]:
    rows = int(pd.to_numeric(frame["rows"], errors="coerce").fillna(0).sum())
    wells = int(frame["well_id"].nunique())
    pf_rmse = pooled_rmse(frame, "pf_rmse")
    beam_rmse = pooled_rmse(frame, "beam_rmse")
    ml_rmse = pooled_rmse(frame, "ml_rmse")
    last_anchor_rmse = pooled_rmse(frame, "last_anchor_rmse")
    best_counts = frame["best_engine"].value_counts()
    return {
        "segment_type": segment_type,
        "segment": segment,
        "wells": wells,
        "rows": rows,
        "pf_pooled_rmse": pf_rmse,
        "beam_pooled_rmse": beam_rmse,
        "ml_pooled_rmse": ml_rmse,
        "last_anchor_pooled_rmse": last_anchor_rmse,
        "pf_minus_beam_rmse": pf_rmse - beam_rmse,
        "pf_minus_ml_rmse": pf_rmse - ml_rmse,
        "beam_minus_ml_rmse": beam_rmse - ml_rmse,
        "pf_beats_ml_wells": int(frame["pf_beats_ml"].fillna(False).sum()),
        "beam_beats_ml_wells": int(frame["beam_beats_ml"].fillna(False).sum()),
        "pf_beats_beam_wells": int(frame["pf_beats_beam"].fillna(False).sum()),
        "best_pf_wells": int(best_counts.get("pf", 0)),
        "best_beam_wells": int(best_counts.get("beam", 0)),
        "best_ml_wells": int(best_counts.get("ml", 0)),
        "best_last_anchor_wells": int(best_counts.get("last_anchor", 0)),
        "mean_pf_beam_abs_diff": weighted_mean(frame, "pf_beam_abs_diff_mean"),
        "mean_pf_std": weighted_mean(frame, "pf_std_mean"),
        "mean_beam_spread": weighted_mean(frame, "beam_spread_mean"),
        "mean_prefix_length": weighted_mean(frame, "prefix_length"),
        "mean_eval_length": weighted_mean(frame, "eval_length"),
        "mean_pfx_rmse": weighted_mean(frame, "pfx_rmse_mean"),
        "mean_likpf_abs_delta": weighted_mean(frame, "likpf_abs_delta_mean"),
    }


def build_error_maps(
    *,
    well_summary_path: Path,
    plot_manifest_path: Path | None = None,
    ml_well_metrics_path: Path | None = None,
    ml_model: str = "lgb_mean",
) -> ErrorMapOutputs:
    well_summary = read_csv_required(well_summary_path, {"well_id", "rows"})
    plot_manifest = read_plot_manifest(plot_manifest_path) if plot_manifest_path else None
    ml_metrics = (
        read_ml_well_metrics(ml_well_metrics_path, ml_model) if ml_well_metrics_path else None
    )
    well_map = prepare_well_map(well_summary, plot_manifest, ml_metrics)

    overall = pd.DataFrame([aggregate_segment(well_map, "overall", "overall")])
    segment_columns = [
        "pf_beam_disagreement_bucket",
        "pf_minus_beam_rmse_bucket",
        "pf_minus_ml_rmse_bucket",
        "tail_length_bucket",
        "prefix_length_bucket",
        "pf_std_bucket",
        "beam_spread_bucket",
        "pfx_rmse_bucket",
        "likpf_abs_delta_bucket",
        "best_engine",
    ]
    rows: list[dict[str, float | int | str]] = []
    for column in segment_columns:
        if column not in well_map.columns:
            continue
        for value, group in well_map.groupby(column, sort=True, observed=False):
            if group.empty:
                continue
            rows.append(aggregate_segment(group, column, str(value)))
    bucket_metrics = pd.DataFrame(rows)

    sort_columns = ["ml_rmse", "pf_beam_abs_diff_mean"]
    existing_sort_columns = [column for column in sort_columns if column in well_map.columns]
    if existing_sort_columns:
        well_map = well_map.sort_values(
            existing_sort_columns, ascending=[False] * len(existing_sort_columns)
        )
    return ErrorMapOutputs(overall=overall, well_map=well_map, bucket_metrics=bucket_metrics)


def write_readme(
    output_dir: Path, outputs: ErrorMapOutputs, source_paths: dict[str, Path | None]
) -> Path:
    path = output_dir / "README.md"
    overall = outputs.overall.iloc[0]
    lines = [
        "# pf_beam_disagreement_error_map",
        "",
        "PF/Beam disagreement, confidence, and tail-length buckets from exp083 well summaries.",
        "",
        "## Inputs",
    ]
    for name, source_path in source_paths.items():
        if source_path is not None:
            lines.append(f"- {name}: `{display_path(source_path)}`")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| wells | {int(overall['wells'])} |",
            f"| rows | {int(overall['rows'])} |",
            f"| PF pooled RMSE | {overall['pf_pooled_rmse']:.6f} |",
            f"| Beam pooled RMSE | {overall['beam_pooled_rmse']:.6f} |",
            f"| ML pooled RMSE | {overall['ml_pooled_rmse']:.6f} |",
            f"| PF minus ML RMSE | {overall['pf_minus_ml_rmse']:.6f} |",
            f"| PF minus Beam RMSE | {overall['pf_minus_beam_rmse']:.6f} |",
            "",
            "## Outputs",
            "",
            "- `pf_beam_disagreement_overall_metrics.csv`",
            "- `pf_beam_disagreement_well_map.csv`",
            "- `pf_beam_disagreement_bucket_metrics.csv`",
            "",
            "This is diagnostic only; it does not define a hard router or selector.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_outputs(
    output_dir: Path,
    outputs: ErrorMapOutputs,
    source_paths: dict[str, Path | None],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = output_dir / "pf_beam_disagreement_overall_metrics.csv"
    well_path = output_dir / "pf_beam_disagreement_well_map.csv"
    bucket_path = output_dir / "pf_beam_disagreement_bucket_metrics.csv"

    outputs.overall.to_csv(overall_path, index=False)
    outputs.well_map.to_csv(well_path, index=False)
    outputs.bucket_metrics.sort_values(["segment_type", "segment"]).to_csv(bucket_path, index=False)
    readme_path = write_readme(output_dir, outputs, source_paths)
    return [overall_path, well_path, bucket_path, readme_path]


def main() -> None:
    args = parse_args()
    well_summary_path = resolve_path(args.well_summary)
    plot_manifest_path = resolve_path(args.plot_manifest) if args.plot_manifest else None
    ml_well_metrics_path = resolve_path(args.ml_well_metrics) if args.ml_well_metrics else None
    output_dir = resolve_path(args.output_dir)

    outputs = build_error_maps(
        well_summary_path=well_summary_path,
        plot_manifest_path=plot_manifest_path,
        ml_well_metrics_path=ml_well_metrics_path,
        ml_model=args.ml_model,
    )
    written = write_outputs(
        output_dir,
        outputs,
        {
            "well_summary": well_summary_path,
            "plot_manifest": plot_manifest_path,
            "ml_well_metrics": ml_well_metrics_path,
        },
    )

    print("pf_beam_disagreement_error_map complete")
    print(outputs.overall.to_string(index=False))
    for path in written:
        print(f"wrote: {display_path(path)}")


if __name__ == "__main__":
    main()
