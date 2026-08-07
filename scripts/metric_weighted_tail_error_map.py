from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config_utils import ROOT, get_nested, is_todo, load_project_config
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package in tests.
    from scripts.config_utils import ROOT, get_nested, is_todo, load_project_config


TARGET_COLUMN = "TVT"
SUBMISSION_TARGET_COLUMN = "tvt"
ID_COLUMN = "id"

DISTANCE_BINS = [-np.inf, 100.0, 250.0, 500.0, 1000.0, np.inf]
DISTANCE_LABELS = ["000-100", "100-250", "250-500", "500-1000", "1000+"]
TAIL_LENGTH_BINS = [-np.inf, 1000, 2500, 5000, np.inf]
TAIL_LENGTH_LABELS = ["0000-1000", "1000-2500", "2500-5000", "5000+"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a row-weighted tail error map for visible/public ROGII submissions. "
            "The target rows are taken from sample_submission ids and the ground truth "
            "is read from matching train horizontal wells."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate submission CSV as name=path. May be repeated.",
    )
    parser.add_argument(
        "--anchor",
        default="",
        help=(
            "Optional anchor submission CSV. If omitted, the forward-filled last-known "
            "TVT_input baseline is used."
        ),
    )
    parser.add_argument("--anchor-name", default="anchor", help="Label for --anchor.")
    parser.add_argument("--sample", default="", help="Override sample_submission.csv path.")
    parser.add_argument("--train-dir", default="", help="Override train horizontal well directory.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/metric_weighted_tail_error_map",
        help="Directory where metric CSV files will be written.",
    )
    parser.add_argument(
        "--write-row-map",
        action="store_true",
        help="Also write row-level prediction/error details. This can be large.",
    )
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


def parse_candidate_spec(value: str) -> CandidateSpec:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"invalid candidate spec: {value!r}")
        return CandidateSpec(name=name, path=resolve_path(raw_path))
    path = resolve_path(value)
    return CandidateSpec(name=path.stem, path=path)


def parse_submission_id(value: str) -> tuple[str, int]:
    try:
        well_id, row_index = value.rsplit("_", 1)
        return well_id, int(row_index)
    except ValueError as exc:
        raise ValueError(f"submission id must be '<well_id>_<row_index>': {value!r}") from exc


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return float("nan")
    total_weight = float(weights[valid].sum())
    if total_weight <= 0:
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / total_weight)


def rmse_from_sse(sse: float, weight_sum: float) -> float:
    if weight_sum <= 0:
        return float("nan")
    return float(math.sqrt(sse / weight_sum))


def read_sample(sample_path: Path, id_column: str = ID_COLUMN) -> pd.DataFrame:
    if not sample_path.exists():
        raise FileNotFoundError(f"sample submission not found: {sample_path}")
    sample = pd.read_csv(sample_path)
    if id_column not in sample.columns:
        raise ValueError(f"sample submission is missing id column {id_column!r}")
    if sample[id_column].duplicated().any():
        duplicated = sample.loc[sample[id_column].duplicated(), id_column].head(5).tolist()
        raise ValueError(f"sample submission contains duplicated ids: {duplicated}")
    parsed = sample[id_column].map(parse_submission_id)
    sample = sample[[id_column]].copy()
    sample["well_id"] = [well_id for well_id, _ in parsed]
    sample["row_index"] = [row_index for _, row_index in parsed]
    return sample


def build_truth_frame(sample: pd.DataFrame, train_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well_id, group in sample.groupby("well_id", sort=True):
        path = train_dir / f"{well_id}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"train horizontal well not found for {well_id}: {path}")
        well = pd.read_csv(path)
        required = {TARGET_COLUMN, "TVT_input", "MD"}
        missing = sorted(required - set(well.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        row_indices = group["row_index"].to_numpy(dtype=int)
        if row_indices.min() < 0 or row_indices.max() >= len(well):
            raise IndexError(
                f"sample ids for {well_id} reference rows outside {display_path(path)} "
                f"(rows={len(well)}, min={row_indices.min()}, max={row_indices.max()})"
            )

        truth = well.iloc[row_indices].copy()
        truth.insert(0, ID_COLUMN, group[ID_COLUMN].to_numpy())
        truth.insert(1, "well_id", well_id)
        truth.insert(2, "row_index", row_indices)
        truth["y_true"] = truth[TARGET_COLUMN].astype(float)
        truth["last_known_tvt"] = well["TVT_input"].ffill().iloc[row_indices].to_numpy(dtype=float)
        truth["tail_start_row"] = int(row_indices.min())
        truth["row_from_tail_start"] = truth["row_index"] - int(row_indices.min())
        tail_start_md = float(well["MD"].iloc[int(row_indices.min())])
        truth["md_since_tail_start"] = truth["MD"].astype(float) - tail_start_md
        truth["tail_rows"] = len(group)
        truth["row_weight"] = 1.0
        rows.append(
            truth[
                [
                    ID_COLUMN,
                    "well_id",
                    "row_index",
                    "MD",
                    "y_true",
                    "last_known_tvt",
                    "tail_start_row",
                    "row_from_tail_start",
                    "md_since_tail_start",
                    "tail_rows",
                    "row_weight",
                ]
            ]
        )

    truth_frame = pd.concat(rows, ignore_index=True)
    truth_frame["distance_bucket"] = pd.cut(
        truth_frame["md_since_tail_start"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        right=False,
    ).astype(str)
    truth_frame["tail_length_bucket"] = pd.cut(
        truth_frame["tail_rows"],
        bins=TAIL_LENGTH_BINS,
        labels=TAIL_LENGTH_LABELS,
        right=False,
    ).astype(str)
    return truth_frame


def read_prediction(
    spec: CandidateSpec,
    ids: pd.Series,
    id_column: str = ID_COLUMN,
    target_column: str = SUBMISSION_TARGET_COLUMN,
) -> pd.Series:
    if not spec.path.exists():
        raise FileNotFoundError(f"candidate not found: {spec.path}")
    frame = pd.read_csv(spec.path)
    missing = [column for column in [id_column, target_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"{display_path(spec.path)} is missing columns: {missing}")
    if frame[id_column].duplicated().any():
        duplicated = frame.loc[frame[id_column].duplicated(), id_column].head(5).tolist()
        raise ValueError(f"{display_path(spec.path)} contains duplicated ids: {duplicated}")

    aligned = pd.DataFrame({id_column: ids.to_numpy()}).merge(
        frame[[id_column, target_column]],
        on=id_column,
        how="left",
    )
    if aligned[target_column].isna().any():
        missing_ids = aligned.loc[aligned[target_column].isna(), id_column].head(5).tolist()
        raise ValueError(f"{display_path(spec.path)} is missing predictions for ids: {missing_ids}")
    return aligned[target_column].astype(float)


def add_step_columns(frame: pd.DataFrame, pred_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(["well_id", "row_index"]).copy()
    same_well = ordered["well_id"].eq(ordered["well_id"].shift())
    true_step = ordered["y_true"].diff().where(same_well)
    pred_step = ordered[pred_column].diff().where(same_well)
    ordered["true_step"] = true_step
    ordered["pred_step"] = pred_step
    ordered["abs_step_error"] = (pred_step - true_step).abs()
    ordered["abs_pred_step"] = pred_step.abs()
    ordered["abs_true_step"] = true_step.abs()
    return ordered


def aggregate_group(
    frame: pd.DataFrame,
    candidate: str,
    anchor_name: str,
    segment_type: str,
    segment: str,
    candidate_total_sse: float,
) -> dict[str, float | int | str | bool]:
    weight_sum = float(frame["row_weight"].sum())
    sse = float(frame["weighted_sse"].sum())
    anchor_sse = float(frame["anchor_weighted_sse"].sum())
    rmse = rmse_from_sse(sse, weight_sum)
    anchor_rmse = rmse_from_sse(anchor_sse, weight_sum)
    return {
        "candidate": candidate,
        "anchor": anchor_name,
        "segment_type": segment_type,
        "segment": segment,
        "rows": int(len(frame)),
        "weight_sum": weight_sum,
        "sse": sse,
        "anchor_sse": anchor_sse,
        "weighted_sse_share": sse / candidate_total_sse if candidate_total_sse > 0 else np.nan,
        "rmse": rmse,
        "anchor_rmse": anchor_rmse,
        "delta_sse_vs_anchor": sse - anchor_sse,
        "delta_rmse_vs_anchor": rmse - anchor_rmse,
        "mae": weighted_mean(frame["abs_error"], frame["row_weight"]),
        "anchor_mae": weighted_mean(frame["anchor_abs_error"], frame["row_weight"]),
        "bias": weighted_mean(frame["error"], frame["row_weight"]),
        "anchor_bias": weighted_mean(frame["anchor_error"], frame["row_weight"]),
        "mean_abs_step_error": float(frame["abs_step_error"].mean()),
        "mean_abs_pred_step": float(frame["abs_pred_step"].mean()),
        "mean_abs_true_step": float(frame["abs_true_step"].mean()),
        "worse_than_anchor": bool(sse > anchor_sse),
    }


def make_candidate_maps(
    truth: pd.DataFrame,
    candidate: str,
    prediction: pd.Series,
    anchor_name: str,
    anchor_prediction: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = truth.copy()
    frame["prediction"] = prediction.to_numpy(dtype=float)
    frame["anchor_prediction"] = anchor_prediction.to_numpy(dtype=float)
    frame["error"] = frame["prediction"] - frame["y_true"]
    frame["anchor_error"] = frame["anchor_prediction"] - frame["y_true"]
    frame["abs_error"] = frame["error"].abs()
    frame["anchor_abs_error"] = frame["anchor_error"].abs()
    frame["weighted_sse"] = frame["error"].pow(2) * frame["row_weight"]
    frame["anchor_weighted_sse"] = frame["anchor_error"].pow(2) * frame["row_weight"]
    frame = add_step_columns(frame, "prediction")
    candidate_total_sse = float(frame["weighted_sse"].sum())

    overall = pd.DataFrame(
        [
            aggregate_group(
                frame=frame,
                candidate=candidate,
                anchor_name=anchor_name,
                segment_type="overall",
                segment="overall",
                candidate_total_sse=candidate_total_sse,
            )
        ]
    )

    well_rows: list[dict[str, float | int | str | bool]] = []
    for well_id, group in frame.groupby("well_id", sort=True):
        row = aggregate_group(
            frame=group,
            candidate=candidate,
            anchor_name=anchor_name,
            segment_type="well",
            segment=str(well_id),
            candidate_total_sse=candidate_total_sse,
        )
        row.update(
            {
                "well_id": str(well_id),
                "tail_rows": int(group["tail_rows"].iloc[0]),
                "tail_length_bucket": str(group["tail_length_bucket"].iloc[0]),
                "tail_start_row": int(group["tail_start_row"].iloc[0]),
                "tail_end_row": int(group["row_index"].max()),
                "tail_start_md": float(group["MD"].min()),
                "tail_end_md": float(group["MD"].max()),
                "max_abs_error": float(group["abs_error"].max()),
                "anchor_max_abs_error": float(group["anchor_abs_error"].max()),
            }
        )
        well_rows.append(row)
    well_map = pd.DataFrame(well_rows).sort_values(
        ["weighted_sse_share", "delta_sse_vs_anchor"], ascending=[False, False]
    )

    bucket_rows: list[dict[str, float | int | str | bool]] = []
    for column in ["distance_bucket", "tail_length_bucket"]:
        for value, group in frame.groupby(column, sort=True, observed=False):
            if group.empty:
                continue
            bucket_rows.append(
                aggregate_group(
                    frame=group,
                    candidate=candidate,
                    anchor_name=anchor_name,
                    segment_type=column,
                    segment=str(value),
                    candidate_total_sse=candidate_total_sse,
                )
            )
    bucket_map = pd.DataFrame(bucket_rows)

    row_map = frame[
        [
            ID_COLUMN,
            "well_id",
            "row_index",
            "MD",
            "row_from_tail_start",
            "md_since_tail_start",
            "distance_bucket",
            "tail_rows",
            "tail_length_bucket",
            "y_true",
            "prediction",
            "anchor_prediction",
            "error",
            "anchor_error",
            "abs_error",
            "anchor_abs_error",
            "weighted_sse",
            "anchor_weighted_sse",
            "abs_step_error",
            "abs_pred_step",
            "abs_true_step",
        ]
    ].copy()
    row_map.insert(0, "candidate", candidate)
    row_map.insert(1, "anchor", anchor_name)
    row_map["weighted_sse_share"] = (
        row_map["weighted_sse"] / candidate_total_sse if candidate_total_sse > 0 else np.nan
    )

    return overall, well_map, bucket_map, row_map


def build_error_maps(
    candidate_specs: Iterable[CandidateSpec],
    sample_path: Path,
    train_dir: Path,
    anchor_spec: CandidateSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = read_sample(sample_path)
    truth = build_truth_frame(sample, train_dir)
    if anchor_spec is None:
        anchor_name = "last_known_tvt"
        anchor_prediction = truth["last_known_tvt"].astype(float)
    else:
        anchor_name = anchor_spec.name
        anchor_prediction = read_prediction(anchor_spec, truth[ID_COLUMN])

    overall_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    row_frames: list[pd.DataFrame] = []
    for spec in candidate_specs:
        prediction = read_prediction(spec, truth[ID_COLUMN])
        overall, well_map, bucket_map, row_map = make_candidate_maps(
            truth=truth,
            candidate=spec.name,
            prediction=prediction,
            anchor_name=anchor_name,
            anchor_prediction=anchor_prediction,
        )
        overall_frames.append(overall)
        well_frames.append(well_map)
        bucket_frames.append(bucket_map)
        row_frames.append(row_map)

    return (
        pd.concat(overall_frames, ignore_index=True),
        pd.concat(well_frames, ignore_index=True),
        pd.concat(bucket_frames, ignore_index=True),
        pd.concat(row_frames, ignore_index=True),
    )


def write_outputs(
    output_dir: Path,
    overall: pd.DataFrame,
    well_map: pd.DataFrame,
    bucket_map: pd.DataFrame,
    row_map: pd.DataFrame,
    write_row_map: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "weighted_tail_overall_metrics.csv",
        output_dir / "weighted_tail_well_error_map.csv",
        output_dir / "weighted_tail_bucket_metrics.csv",
    ]
    overall.sort_values(["rmse", "candidate"]).to_csv(outputs[0], index=False)
    well_map.sort_values(["candidate", "weighted_sse_share"], ascending=[True, False]).to_csv(
        outputs[1], index=False
    )
    bucket_map.sort_values(["candidate", "segment_type", "segment"]).to_csv(outputs[2], index=False)
    if write_row_map:
        row_path = output_dir / "weighted_tail_row_error_map.csv"
        row_map.sort_values(["candidate", "well_id", "row_index"]).to_csv(row_path, index=False)
        outputs.append(row_path)
    return outputs


def config_path(config: dict, cli_value: str, dotted_key: str) -> Path:
    if cli_value:
        return resolve_path(cli_value)
    value = get_nested(config, dotted_key)
    if is_todo(value):
        raise ValueError(f"{dotted_key} is TODO in project.yml")
    return resolve_path(str(value))


def main() -> None:
    args = parse_args()
    config = load_project_config()
    sample_path = config_path(config, args.sample, "submission.sample_file")
    train_dir = config_path(config, args.train_dir, "data.train_dir")
    output_dir = resolve_path(args.output_dir)
    candidate_specs = [parse_candidate_spec(value) for value in args.candidate]
    anchor_spec = (
        CandidateSpec(args.anchor_name, resolve_path(args.anchor)) if args.anchor else None
    )

    overall, well_map, bucket_map, row_map = build_error_maps(
        candidate_specs=candidate_specs,
        sample_path=sample_path,
        train_dir=train_dir,
        anchor_spec=anchor_spec,
    )
    outputs = write_outputs(
        output_dir=output_dir,
        overall=overall,
        well_map=well_map,
        bucket_map=bucket_map,
        row_map=row_map,
        write_row_map=args.write_row_map,
    )

    print("metric_weighted_tail_error_map complete")
    summary_columns = ["candidate", "rows", "rmse", "anchor_rmse", "delta_rmse_vs_anchor"]
    print(overall[summary_columns].to_string(index=False))
    for path in outputs:
        print(f"wrote: {display_path(path)}")


if __name__ == "__main__":
    main()
