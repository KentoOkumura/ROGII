"""Create a Kaggle-format viewer CSV from corrected exp264 Stage D v3 OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


EXPERIMENT_NAME = "exp264_exp263_candidate_confidence_dual_selector"
INPUT_FILENAME = "stage_d_oof_predictions.parquet"
PREDICTION_COLUMN = "selector_compact_addonly__lgb_mean__pred_tvt"
EXPECTED_INPUT_SHA256 = (
    "b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2"
)
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_RMSE = 8.460811237612477
DEFAULT_OUTPUT_NAME = f"{EXPERIMENT_NAME}_stage_d_v3_oof_viewer.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment_summary.md").is_file() and (
            candidate / "experiments"
        ).is_dir():
            return candidate
    raise FileNotFoundError("repository root was not found")


def resolve_input(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        exp_dir = repo_root / "experiments" / EXPERIMENT_NAME
        candidates = [
            exp_dir / "artifacts" / "stage_d_v3_corrected" / INPUT_FILENAME,
            exp_dir
            / "kaggle"
            / "output"
            / "stage_d_v3_corrected"
            / "artifacts"
            / INPUT_FILENAME,
            Path("/tmp/exp264-stage-d-v3-oof/artifacts") / INPUT_FILENAME,
        ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    raise FileNotFoundError(f"corrected Stage D v3 OOF not found: {candidates}")


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(Path.cwd())
    exp_dir = repo_root / "experiments" / EXPERIMENT_NAME
    input_path = resolve_input(repo_root, args.input)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else exp_dir / "artifacts" / DEFAULT_OUTPUT_NAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_sha = sha256_path(input_path)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError(f"input SHA {input_sha} != {EXPECTED_INPUT_SHA256}")

    parquet_file = pq.ParquetFile(input_path)
    required = {"id", "well", "actual_tvt", PREDICTION_COLUMN}
    missing = sorted(required.difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"corrected Stage D OOF missing columns: {missing}")
    if parquet_file.metadata.num_rows != EXPECTED_ROWS:
        raise ValueError(
            f"input rows {parquet_file.metadata.num_rows} != {EXPECTED_ROWS}"
        )

    first_chunk = True
    row_count = 0
    squared_error_sum = 0.0
    well_ids: set[str] = set()
    viewer_schema = pa.schema([pa.field("id", pa.string()), pa.field("tvt", pa.float32())])
    with pacsv.CSVWriter(
        output_path,
        viewer_schema,
        write_options=pacsv.WriteOptions(include_header=True),
    ) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=250_000,
            columns=["id", "well", "actual_tvt", PREDICTION_COLUMN],
        ):
            ids = batch.column("id")
            wells = batch.column("well")
            truth_array = batch.column("actual_tvt")
            prediction_array = batch.column(PREDICTION_COLUMN)
            if any(array.null_count for array in [ids, wells, truth_array, prediction_array]):
                raise ValueError("input contains null IDs, wells, truth, or prediction")
            truth = truth_array.to_numpy(zero_copy_only=False).astype(np.float64)
            prediction = prediction_array.to_numpy(zero_copy_only=False).astype(np.float64)
            if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
                raise ValueError("input truth or prediction contains NaN/Inf")
            well_ids.update(str(value) for value in pc.unique(wells).to_pylist())
            squared_error_sum += float(np.square(prediction - truth).sum())
            writer.write_batch(
                pa.RecordBatch.from_arrays(
                    [ids, prediction_array],
                    schema=viewer_schema,
                )
            )
            row_count += batch.num_rows
            first_chunk = False

    if row_count != EXPECTED_ROWS or len(well_ids) != EXPECTED_WELLS:
        raise ValueError(
            f"viewer coverage rows={row_count}, wells={len(well_ids)} changed"
        )
    rmse = float(np.sqrt(squared_error_sum / row_count))
    if not np.isclose(rmse, EXPECTED_RMSE, atol=1e-9):
        raise ValueError(f"viewer OOF RMSE {rmse} != {EXPECTED_RMSE}")

    verified_output_rows = 0
    output_reader = pacsv.open_csv(
        output_path,
        read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(
            column_types={"id": pa.string(), "tvt": pa.float64()}
        ),
    )
    for output_batch in output_reader:
        if output_batch.column("id").null_count or output_batch.column("tvt").null_count:
            raise ValueError("written viewer CSV contains null values")
        output_prediction = output_batch.column("tvt").to_numpy(zero_copy_only=False)
        if not np.isfinite(output_prediction).all():
            raise ValueError("written viewer CSV contains NaN/Inf")
        verified_output_rows += output_batch.num_rows
    if verified_output_rows != EXPECTED_ROWS:
        raise ValueError(
            f"written viewer rows {verified_output_rows} != {EXPECTED_ROWS}"
        )

    output_sha = sha256_path(output_path)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": EXPERIMENT_NAME,
        "status": "viewer_oof_csv_created",
        "source": str(input_path),
        "source_sha256": input_sha,
        "source_column": PREDICTION_COLUMN,
        "output": str(output_path),
        "output_sha256": output_sha,
        "columns": ["id", "tvt"],
        "rows": row_count,
        "wells": len(well_ids),
        "id_unique": True,
        "id_unique_evidence": "fixed source SHA was audited as 3,783,989 unique IDs; CSV is a one-to-one ordered projection",
        "prediction_finite": True,
        "rmse": rmse,
        "invalid_stage_d_v2_used": False,
        "stage_d_worst_well_guard_pass": False,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
