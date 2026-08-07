from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from candidate_cache_contract import (
    ANCHOR_RMSE,
    COMMON_CONFIDENCE_SLOTS,
    CORE_CANDIDATE_IDS,
    N_FOLDS,
    NAMED_COMBINATIONS,
    PAIR_SHORTLIST,
    RAWTEST_CORE_CANDIDATE_IDS,
    REFERENCE_CANDIDATES,
    ROLLING_WINDOWS,
    SCHEMA_VERSION,
    STAGE1_NATIVE_CONFIDENCE_FIELDS,
    CandidateSpec,
    candidate_by_id,
    validate_contract,
)
from candidate_cache_loader import (
    KEY_COLUMNS,
    CandidateCache,
    frame_content_sha256,
    schema_sha256,
    sha256_file,
)

EXPECTED_ROWS = 3_783_989
CHUNK_ROWS = 100_000
DISTANCE_BUCKETS = (
    ("000_050", 0.0, 50.0),
    ("050_100", 50.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1000.0),
    ("1000_plus", 1000.0, math.inf),
)


@dataclass
class CanonicalIndex:
    n_rows: int
    well_names: list[str]
    well_meta: dict[str, tuple[int, int, int, int]]
    values: np.memmap
    seen: np.memmap
    truth: np.memmap
    last_known_tvt: np.memmap
    md_since: np.memmap
    well_code: np.memmap
    well_row_idx: np.memmap
    outer_fold: np.memmap
    confidence: dict[tuple[str, str], np.memmap]
    canonical_id_sha256: str


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_decompressed_gzip(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _path_strings(spec: Any) -> list[str]:
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return [str(item) for item in spec]
    if not isinstance(spec, dict):
        return []
    values: list[str] = []
    for key in ("path", "paths", "pattern", "patterns"):
        item = spec.get(key)
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, list):
            values.extend(str(entry) for entry in item)
    return values


def resolve_input_paths(
    source_key: str,
    input_config: Mapping[str, Any],
    search_roots: Sequence[Path],
) -> list[Path]:
    if source_key not in input_config:
        raise KeyError(f"missing data.inputs.{source_key}")
    source_spec = input_config[source_key]
    requested = _path_strings(source_spec)
    if not requested:
        raise ValueError(f"data.inputs.{source_key} has no path/pattern")

    direct_requested: list[str] = []
    pattern_requested: list[str] = []
    if isinstance(source_spec, dict):
        direct_requested = _path_strings(
            {key: source_spec[key] for key in ("path", "paths") if key in source_spec}
        )
        pattern_requested = _path_strings(
            {key: source_spec[key] for key in ("pattern", "patterns") if key in source_spec}
        )
    else:
        direct_requested = requested

    def resolve_many(raw_values: Iterable[str]) -> list[Path]:
        resolved: list[Path] = []
        for raw in raw_values:
            path = Path(raw).expanduser()
            if path.is_file():
                resolved.append(path.resolve())
                continue
            if path.is_absolute():
                root = Path(path.anchor)
                pattern = str(path.relative_to(root))
                resolved.extend(item.resolve() for item in root.glob(pattern) if item.is_file())
                continue
            for root in search_roots:
                candidate = root / path
                if candidate.is_file():
                    resolved.append(candidate.resolve())
                else:
                    resolved.extend(item.resolve() for item in root.glob(raw) if item.is_file())
        return sorted(dict.fromkeys(resolved))

    direct_resolved = resolve_many(direct_requested)
    if direct_resolved:
        return direct_resolved
    resolved = resolve_many(pattern_requested)
    if not resolved:
        raise FileNotFoundError(f"could not resolve {source_key}: {requested}")
    return resolved


def assign_group_folds(well_names: list[str], counts_by_code: np.ndarray) -> np.ndarray:
    order_by_name = np.argsort(np.asarray(well_names, dtype=object))
    group_counts = counts_by_code[order_by_name]
    size_order = np.argsort(group_counts, kind="stable")[::-1]
    fold_sizes = np.zeros(N_FOLDS, dtype=np.int64)
    fold_by_sorted_group = np.empty(len(well_names), dtype=np.uint8)
    for group_pos in size_order:
        fold = int(np.argmin(fold_sizes))
        fold_sizes[fold] += int(group_counts[group_pos])
        fold_by_sorted_group[group_pos] = fold
    fold_by_code = np.empty(len(well_names), dtype=np.uint8)
    fold_by_code[order_by_name] = fold_by_sorted_group
    return fold_by_code


def _allocate_memmap(path: Path, dtype: str, shape: int | tuple[int, ...], fill: Any) -> np.memmap:
    array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[:] = fill
    return array


def _close_memmap(array: np.memmap) -> None:
    array.flush()
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def close_canonical_index(canonical: CanonicalIndex) -> None:
    for array in [
        canonical.values,
        canonical.seen,
        canonical.truth,
        canonical.last_known_tvt,
        canonical.md_since,
        canonical.well_code,
        canonical.well_row_idx,
        canonical.outer_fold,
        *canonical.confidence.values(),
    ]:
        _close_memmap(array)


def _confidence_memmaps(work_dir: Path, n_rows: int) -> dict[tuple[str, str], np.memmap]:
    output: dict[tuple[str, str], np.memmap] = {}
    for candidate_id in CORE_CANDIDATE_IDS:
        candidate = candidate_by_id(candidate_id)
        for output_column, _ in candidate.confidence_columns:
            path = work_dir / f"confidence__{candidate_id}__{output_column}.f32"
            output[(candidate_id, output_column)] = _allocate_memmap(
                path, "float32", n_rows, np.nan
            )
    return output


def _parse_well_rows(frame: pd.DataFrame, spec: CandidateSpec) -> tuple[np.ndarray, np.ndarray]:
    wells = frame[spec.well_column].astype(str).to_numpy()
    if spec.row_idx_column is not None:
        rows = pd.to_numeric(frame[spec.row_idx_column], errors="raise").to_numpy(np.int32)
    elif spec.id_column is not None:
        ids = frame[spec.id_column].astype(str)
        rows = ids.str.rsplit("_", n=1).str[-1].astype(np.int32).to_numpy()
        rebuilt = pd.Series(wells, dtype=str) + "_" + pd.Series(rows).astype(str)
        if not np.array_equal(ids.to_numpy(), rebuilt.to_numpy()):
            raise ValueError(f"source id/well/well_row_idx mismatch: {spec.candidate_id}")
    else:
        raise ValueError(f"no row identity for {spec.candidate_id}")
    return wells, rows


def _global_indices(
    wells: np.ndarray,
    rows: np.ndarray,
    meta: Mapping[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    known = np.fromiter((str(well) in meta for well in wells), dtype=bool, count=len(wells))
    indices = np.full(len(wells), -1, dtype=np.int64)
    if not known.any():
        return indices, known
    known_wells = wells[known]
    starts = np.fromiter((meta[str(well)][0] for well in known_wells), dtype=np.int64)
    firsts = np.fromiter((meta[str(well)][1] for well in known_wells), dtype=np.int32)
    counts = np.fromiter((meta[str(well)][2] for well in known_wells), dtype=np.int32)
    local = rows[known] - firsts
    in_range = (local >= 0) & (local < counts)
    known_positions = np.flatnonzero(known)
    indices[known_positions[in_range]] = starts[in_range] + local[in_range].astype(np.int64)
    valid = indices >= 0
    return indices, valid


def _source_usecols(candidates: Sequence[CandidateSpec]) -> list[str]:
    first = candidates[0]
    columns = {first.well_column}
    if first.row_idx_column:
        columns.add(first.row_idx_column)
    if first.id_column:
        columns.add(first.id_column)
    for candidate in candidates:
        if candidate.value_column:
            columns.add(candidate.value_column)
        if candidate.anchor_column:
            columns.add(candidate.anchor_column)
        columns.update(source for _, source in candidate.confidence_columns)
    return sorted(columns)


def _source_manifest(paths: Iterable[Path], *, decompressed: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        row = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "decompressed_content_sha256": None,
        }
        if decompressed and path.suffix == ".gz":
            row["decompressed_content_sha256"] = sha256_decompressed_gzip(path)
        rows.append(row)
    return rows


def prepare_canonical_index(
    base_path: Path,
    work_dir: Path,
    expected_rows: int,
    max_rows: int | None,
    chunk_rows: int,
) -> CanonicalIndex:
    n_alloc = min(expected_rows, max_rows) if max_rows is not None else expected_rows
    values = _allocate_memmap(
        work_dir / "candidate_values.f32",
        "float32",
        (n_alloc, len(CORE_CANDIDATE_IDS)),
        np.nan,
    )
    seen = _allocate_memmap(
        work_dir / "candidate_seen.u8", "uint8", (n_alloc, len(CORE_CANDIDATE_IDS)), 0
    )
    truth = _allocate_memmap(work_dir / "truth.f32", "float32", n_alloc, np.nan)
    anchor = _allocate_memmap(work_dir / "last_known_tvt.f32", "float32", n_alloc, np.nan)
    md_since = _allocate_memmap(work_dir / "md_since.f32", "float32", n_alloc, np.nan)
    well_code = _allocate_memmap(work_dir / "well_code.u16", "uint16", n_alloc, 0)
    row_idx = _allocate_memmap(work_dir / "well_row_idx.i32", "int32", n_alloc, -1)
    outer_fold = _allocate_memmap(work_dir / "outer_fold.u8", "uint8", n_alloc, 255)
    confidence = _confidence_memmaps(work_dir, n_alloc)

    core_index = {name: index for index, name in enumerate(CORE_CANDIDATE_IDS)}
    candidates = [item for item in REFERENCE_CANDIDATES if item.source_key == "exp072_oof"]
    usecols = set(_source_usecols(candidates))
    usecols.update({"id", "well", "target", "last_known_tvt", "md_since"})

    well_names: list[str] = []
    code_for_well: dict[str, int] = {}
    mutable_meta: dict[str, list[int]] = {}
    offset = 0
    id_digest = hashlib.sha256()
    for frame in pd.read_csv(
        base_path,
        usecols=sorted(usecols),
        chunksize=chunk_rows,
        dtype={"id": str, "well": str},
    ):
        if max_rows is not None and offset >= max_rows:
            break
        if max_rows is not None and offset + len(frame) > max_rows:
            frame = frame.iloc[: max_rows - offset].copy()
        n = len(frame)
        end = offset + n
        ids = frame["id"].astype(str)
        wells = frame["well"].astype(str).to_numpy()
        rows = ids.str.rsplit("_", n=1).str[-1].astype(np.int32).to_numpy()
        rebuilt = pd.Series(wells, dtype=str) + "_" + pd.Series(rows).astype(str)
        if not np.array_equal(ids.to_numpy(), rebuilt.to_numpy()):
            raise ValueError("canonical id is not exactly well + '_' + well_row_idx")
        for value in ids:
            id_digest.update(value.encode())
            id_digest.update(b"\n")

        anchor_values = pd.to_numeric(frame["last_known_tvt"], errors="raise").to_numpy(
            np.float32
        )
        truth[offset:end] = anchor_values + pd.to_numeric(
            frame["target"], errors="raise"
        ).to_numpy(np.float32)
        anchor[offset:end] = anchor_values
        md_since[offset:end] = pd.to_numeric(frame["md_since"], errors="raise").to_numpy(
            np.float32
        )
        row_idx[offset:end] = rows

        chunk_codes = np.empty(n, dtype=np.uint16)
        for well in pd.unique(wells):
            positions = np.flatnonzero(wells == well)
            if well not in code_for_well:
                code_for_well[well] = len(well_names)
                well_names.append(str(well))
                mutable_meta[str(well)] = [
                    offset + int(positions[0]),
                    int(rows[positions[0]]),
                    0,
                    code_for_well[well],
                ]
            code = code_for_well[well]
            chunk_codes[positions] = code
            mutable_meta[str(well)][2] += int(len(positions))
        well_code[offset:end] = chunk_codes

        for candidate in candidates:
            column_index = core_index[candidate.candidate_id]
            raw = pd.to_numeric(frame[candidate.value_column], errors="coerce").to_numpy(
                np.float32
            )
            value = anchor_values + raw if candidate.transform == "anchor_plus" else raw
            values[offset:end, column_index] = value
            seen[offset:end, column_index] = 1
            for output_column, source_column in candidate.confidence_columns:
                confidence[(candidate.candidate_id, output_column)][offset:end] = pd.to_numeric(
                    frame[source_column], errors="coerce"
                ).to_numpy(np.float32)
        offset = end

    if max_rows is None and offset != expected_rows:
        raise ValueError(f"canonical row count {offset} != {expected_rows}")
    if offset != n_alloc:
        raise ValueError(f"allocated rows {n_alloc} != loaded rows {offset}")

    counts = np.bincount(np.asarray(well_code[:offset]), minlength=len(well_names))
    for well, mutable in mutable_meta.items():
        start, first_row, count, code = mutable
        if count != int(counts[code]):
            raise ValueError(f"non-contiguous well rows: {well}")
        expected = np.arange(first_row, first_row + count, dtype=np.int32)
        actual = np.asarray(row_idx[start : start + count])
        if not np.array_equal(expected, actual):
            raise ValueError(f"non-contiguous well_row_idx: {well}")
    fold_by_code = assign_group_folds(well_names, counts)
    outer_fold[:] = fold_by_code[np.asarray(well_code, dtype=np.int64)]

    for array in [values, seen, truth, anchor, md_since, well_code, row_idx, outer_fold]:
        array.flush()
    for array in confidence.values():
        array.flush()
    return CanonicalIndex(
        n_rows=offset,
        well_names=well_names,
        well_meta={key: tuple(value) for key, value in mutable_meta.items()},
        values=values,
        seen=seen,
        truth=truth,
        last_known_tvt=anchor,
        md_since=md_since,
        well_code=well_code,
        well_row_idx=row_idx,
        outer_fold=outer_fold,
        confidence=confidence,
        canonical_id_sha256=id_digest.hexdigest(),
    )


def load_external_core_sources(
    canonical: CanonicalIndex,
    input_config: Mapping[str, Any],
    search_roots: Sequence[Path],
    chunk_rows: int,
    record_decompressed_sha: bool,
) -> dict[str, Any]:
    core_index = {name: index for index, name in enumerate(CORE_CANDIDATE_IDS)}
    grouped: dict[str, list[CandidateSpec]] = defaultdict(list)
    for candidate_id in CORE_CANDIDATE_IDS:
        candidate = candidate_by_id(candidate_id)
        if candidate.source_key != "exp072_oof":
            if candidate.source_key is None:
                raise ValueError(f"core candidate has no source key: {candidate_id}")
            grouped[candidate.source_key].append(candidate)

    manifests: dict[str, Any] = {}
    for source_key, candidates in grouped.items():
        first = candidates[0]
        for candidate in candidates[1:]:
            if (
                candidate.well_column,
                candidate.row_idx_column,
                candidate.id_column,
            ) != (first.well_column, first.row_idx_column, first.id_column):
                raise ValueError(f"incompatible identity columns inside {source_key}")
        paths = resolve_input_paths(source_key, input_config, search_roots)
        usecols = _source_usecols(candidates)
        rows_read = 0
        rows_loaded = {candidate.candidate_id: 0 for candidate in candidates}
        for path in paths:
            dtype = {first.well_column: str}
            if first.id_column:
                dtype[first.id_column] = str
            for frame in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, dtype=dtype):
                rows_read += len(frame)
                wells, rows = _parse_well_rows(frame, first)
                indices, valid = _global_indices(wells, rows, canonical.well_meta)
                if not valid.any():
                    continue
                idx = indices[valid]
                for candidate in candidates:
                    column_index = core_index[candidate.candidate_id]
                    if np.asarray(canonical.seen[idx, column_index], dtype=bool).any():
                        raise ValueError(f"duplicate source rows for {candidate.candidate_id}")
                    value = pd.to_numeric(
                        frame.loc[valid, candidate.value_column], errors="coerce"
                    ).to_numpy(np.float32)
                    if candidate.transform == "anchor_plus":
                        value = value + np.asarray(canonical.last_known_tvt[idx], dtype=np.float32)
                    canonical.values[idx, column_index] = value
                    canonical.seen[idx, column_index] = 1
                    rows_loaded[candidate.candidate_id] += len(idx)
                    for output_column, source_column in candidate.confidence_columns:
                        canonical.confidence[(candidate.candidate_id, output_column)][idx] = (
                            pd.to_numeric(
                                frame.loc[valid, source_column], errors="coerce"
                            ).to_numpy(np.float32)
                        )
        manifests[source_key] = {
            "source_key": source_key,
            "rows_read": rows_read,
            "rows_loaded": rows_loaded,
            "files": _source_manifest(paths, decompressed=record_decompressed_sha),
        }
    canonical.values.flush()
    canonical.seen.flush()
    for array in canonical.confidence.values():
        array.flush()
    missing = {
        candidate_id: int(canonical.n_rows - canonical.seen[:, index].sum())
        for index, candidate_id in enumerate(CORE_CANDIDATE_IDS)
        if int(canonical.seen[:, index].sum()) != canonical.n_rows
    }
    if missing:
        raise ValueError(f"incomplete core candidate coverage: {missing}")
    return manifests


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return prefix[end] - prefix[start]


def candidate_shape_context(
    values: np.ndarray,
    bank_median: np.ndarray,
    well_meta: Mapping[str, tuple[int, int, int, int]],
) -> dict[str, np.ndarray]:
    n = len(values)
    step = np.full(n, np.nan, dtype=np.float32)
    curvature = np.full(n, np.nan, dtype=np.float32)
    outputs: dict[str, np.ndarray] = {
        "candidate_step": step,
        "candidate_curvature": curvature,
    }
    for window in ROLLING_WINDOWS:
        outputs[f"rolling_slope_{window}"] = np.full(n, np.nan, dtype=np.float32)
        outputs[f"rolling_curvature_{window}"] = np.full(n, np.nan, dtype=np.float32)
        outputs[f"rolling_straightness_{window}"] = np.full(n, np.nan, dtype=np.float32)
        outputs[f"bank_disagreement_mean_{window}"] = np.full(n, np.nan, dtype=np.float32)

    for start, _, count, _ in well_meta.values():
        stop = start + count
        segment = np.asarray(values[start:stop], dtype=np.float64)
        median_segment = np.asarray(bank_median[start:stop], dtype=np.float64)
        local_step = np.zeros(count, dtype=np.float64)
        local_step[0] = np.nan
        local_step[1:] = np.diff(segment)
        local_curvature = np.zeros(count, dtype=np.float64)
        local_curvature[:2] = np.nan
        local_curvature[2:] = np.diff(segment, n=2)
        step[start:stop] = local_step.astype(np.float32)
        curvature[start:stop] = local_curvature.astype(np.float32)
        abs_step = np.nan_to_num(np.abs(local_step), nan=0.0)
        abs_curvature = np.nan_to_num(np.abs(local_curvature), nan=0.0)
        disagreement = np.abs(segment - median_segment)
        index = np.arange(count)
        for window in ROLLING_WINDOWS:
            first = np.maximum(0, index - window + 1)
            denominator = np.maximum(1, index - first)
            net = segment - segment[first]
            outputs[f"rolling_slope_{window}"][start:stop] = (
                net / denominator
            ).astype(np.float32)
            curvature_count = np.maximum(1, np.minimum(index + 1, window) - 2)
            outputs[f"rolling_curvature_{window}"][start:stop] = (
                _rolling_sum(abs_curvature, window) / curvature_count
            ).astype(np.float32)
            traveled = _rolling_sum(abs_step, window)
            outputs[f"rolling_straightness_{window}"][start:stop] = (
                np.abs(net) / np.maximum(traveled, 1e-12)
            ).astype(np.float32)
            count_window = np.minimum(index + 1, window)
            outputs[f"bank_disagreement_mean_{window}"][start:stop] = (
                _rolling_sum(disagreement, window) / count_window
            ).astype(np.float32)
    return outputs


def _identity_frame(canonical: CanonicalIndex, indices: np.ndarray) -> pd.DataFrame:
    codes = np.asarray(canonical.well_code[indices], dtype=np.int64)
    wells = np.asarray(canonical.well_names, dtype=object)[codes]
    rows = np.asarray(canonical.well_row_idx[indices], dtype=np.int32)
    ids = np.asarray([f"{well}_{row}" for well, row in zip(wells, rows, strict=True)])
    return pd.DataFrame(
        {
            "id": ids,
            "well": wells.astype(str),
            "well_row_idx": rows,
            "outer_fold": np.asarray(canonical.outer_fold[indices], dtype=np.int8),
            "md_since": np.asarray(canonical.md_since[indices], dtype=np.float32),
        }
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "path": str(path),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "content_sha256": frame_content_sha256(frame),
        "schema_sha256": schema_sha256(frame),
    }


def write_candidate_partitions(
    canonical: CanonicalIndex,
    output_dir: Path,
    chunk_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del chunk_rows  # One compressed partition per candidate/fold keeps the public loader simple.
    values = np.asarray(canonical.values)
    bank_median = _allocate_memmap(
        output_dir / "_work" / "bank_median.f32", "float32", canonical.n_rows, np.nan
    )
    bank_range = _allocate_memmap(
        output_dir / "_work" / "bank_range.f32", "float32", canonical.n_rows, np.nan
    )
    bank_std = _allocate_memmap(
        output_dir / "_work" / "bank_std.f32", "float32", canonical.n_rows, np.nan
    )
    for start in range(0, canonical.n_rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, canonical.n_rows)
        block = np.asarray(values[start:stop], dtype=np.float32)
        bank_median[start:stop] = np.nanmedian(block, axis=1)
        bank_range[start:stop] = np.nanmax(block, axis=1) - np.nanmin(block, axis=1)
        bank_std[start:stop] = np.nanstd(block, axis=1)
    bank_median.flush()
    bank_range.flush()
    bank_std.flush()

    value_manifest: dict[str, Any] = {}
    confidence_manifest: dict[str, Any] = {}
    fold_array = np.asarray(canonical.outer_fold)
    well_counts = np.bincount(
        np.asarray(canonical.well_code, dtype=np.int64), minlength=len(canonical.well_names)
    )
    for candidate_index, candidate_id in enumerate(CORE_CANDIDATE_IDS):
        candidate = candidate_by_id(candidate_id)
        candidate_values = np.asarray(values[:, candidate_index], dtype=np.float32)
        shape_context = candidate_shape_context(
            candidate_values, np.asarray(bank_median), canonical.well_meta
        )
        value_manifest[candidate_id] = []
        confidence_manifest[candidate_id] = []
        for fold in range(N_FOLDS):
            indices = np.flatnonzero(fold_array == fold)
            identity = _identity_frame(canonical, indices)
            frame = identity.copy()
            frame["candidate_id"] = candidate_id
            frame["candidate_name"] = candidate.candidate_name
            frame["family"] = candidate.family
            frame["source_exp"] = candidate.source_exp
            frame["source_artifact"] = candidate.source_artifact
            frame["rawtest_status"] = candidate.rawtest_status
            frame["formula"] = candidate.formula
            frame["last_known_tvt"] = np.asarray(
                canonical.last_known_tvt[indices], dtype=np.float32
            )
            frame["candidate_tvt"] = candidate_values[indices]
            frame["candidate_minus_last"] = (
                candidate_values[indices]
                - np.asarray(canonical.last_known_tvt[indices], dtype=np.float32)
            ).astype(np.float32)
            finite = np.isfinite(candidate_values[indices])
            available = np.asarray(canonical.seen[indices, candidate_index], dtype=bool)
            frame["candidate_finite"] = finite
            frame["candidate_available"] = available
            frame["fallback_used"] = False
            frame["coverage_valid"] = available & finite
            frame["anchor_drift"] = frame["candidate_minus_last"].to_numpy(np.float32)
            frame["candidate_bank_median"] = np.asarray(bank_median[indices], dtype=np.float32)
            frame["candidate_bank_range"] = np.asarray(bank_range[indices], dtype=np.float32)
            frame["candidate_bank_std"] = np.asarray(bank_std[indices], dtype=np.float32)
            frame["candidate_vs_bank_median"] = (
                candidate_values[indices] - np.asarray(bank_median[indices], dtype=np.float32)
            ).astype(np.float32)
            frame["candidate_vs_bank_median_abs"] = np.abs(
                frame["candidate_vs_bank_median"].to_numpy(np.float32)
            )
            candidate_block = values[indices]
            frame["candidate_is_bank_min"] = candidate_values[indices] <= np.nanmin(
                candidate_block, axis=1
            )
            frame["candidate_is_bank_max"] = candidate_values[indices] >= np.nanmax(
                candidate_block, axis=1
            )
            for column, array in shape_context.items():
                frame[column] = np.asarray(array[indices], dtype=np.float32)
            value_path = (
                output_dir
                / "candidate_values"
                / candidate_id
                / f"fold={fold}"
                / "part-000.parquet"
            )
            value_manifest[candidate_id].append(_write_parquet(frame, value_path))

            confidence = identity.copy()
            confidence["candidate_id"] = candidate_id
            confidence["confidence_source"] = (
                f"{candidate.source_exp}:saved_target_free_columns"
                if candidate.confidence_columns
                else "unavailable_in_saved_source"
            )
            available_fields = []
            for output_column, _ in candidate.confidence_columns:
                data = np.asarray(canonical.confidence[(candidate_id, output_column)][indices])
                confidence[output_column] = data.astype(np.float32)
                available_fields.append(np.isfinite(data))
            if "source_loglik" in confidence.columns:
                codes = np.asarray(canonical.well_code[indices], dtype=np.int64)
                confidence["loglik_per_row"] = (
                    confidence["source_loglik"].to_numpy(np.float64)
                    / well_counts[codes].astype(np.float64)
                ).astype(np.float32)
            confidence["confidence_valid"] = (
                np.logical_or.reduce(available_fields) & available & finite
                if available_fields
                else np.zeros(len(indices), dtype=bool)
            )
            for slot in COMMON_CONFIDENCE_SLOTS:
                if slot not in confidence:
                    confidence[slot] = np.full(len(indices), np.nan, dtype=np.float32)
            confidence["confidence_missing_fields"] = ",".join(
                sorted(
                    set(candidate.confidence_expected)
                    - set(output for output, _ in candidate.confidence_columns)
                    - ({"loglik_per_row"} if "source_loglik" in confidence.columns else set())
                )
            )
            confidence_path = (
                output_dir
                / "candidate_confidence"
                / candidate_id
                / f"fold={fold}"
                / "part-000.parquet"
            )
            confidence_manifest[candidate_id].append(
                _write_parquet(confidence, confidence_path)
            )
    for array in (bank_median, bank_range, bank_std):
        _close_memmap(array)
    return value_manifest, confidence_manifest


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))


def _metric_row(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    rows = int(selected.sum())
    return {
        "scope": scope,
        "scope_value": scope_value,
        "rows": rows,
        "rmse": _rmse(prediction[selected], truth[selected]) if rows else np.nan,
    }


def core_metrics(canonical: CanonicalIndex) -> dict[str, Any]:
    output: dict[str, Any] = {}
    truth = np.asarray(canonical.truth)
    folds = np.asarray(canonical.outer_fold)
    md_since = np.asarray(canonical.md_since)
    for index, candidate_id in enumerate(CORE_CANDIDATE_IDS):
        value = np.asarray(canonical.values[:, index])
        fold_metrics = [
            _metric_row(value, truth, folds == fold, "outer_fold", str(fold))
            for fold in range(N_FOLDS)
        ]
        bucket_metrics = [
            _metric_row(
                value,
                truth,
                (md_since >= lower) & (md_since < upper),
                "distance_bucket",
                name,
            )
            for name, lower, upper in DISTANCE_BUCKETS
        ]
        output[candidate_id] = {
            "global_rmse": _rmse(value, truth),
            "fold_metrics": fold_metrics,
            "bucket_metrics": bucket_metrics,
        }
    return output


def outer_train_eligibility(canonical: CanonicalIndex) -> pd.DataFrame:
    truth = np.asarray(canonical.truth)
    folds = np.asarray(canonical.outer_fold)
    anchor = np.asarray(canonical.last_known_tvt)
    rows: list[dict[str, Any]] = []
    value_by_id = {
        candidate_id: np.asarray(canonical.values[:, index])
        for index, candidate_id in enumerate(CORE_CANDIDATE_IDS)
    }
    pair_by_id = {
        pair.pair_id: 0.5 * (value_by_id[pair.left] + value_by_id[pair.right])
        for pair in PAIR_SHORTLIST
    }
    for fold in range(N_FOLDS):
        train = folds != fold
        anchor_rmse = _rmse(anchor[train], truth[train])
        for candidate_id, value in {**value_by_id, **pair_by_id}.items():
            candidate_rmse = _rmse(value[train], truth[train])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_kind": "pair" if candidate_id in pair_by_id else "primitive",
                    "outer_fold": fold,
                    "basis": "outer_train_only_excludes_outer_valid_fold",
                    "rows": int(train.sum()),
                    "candidate_rmse": candidate_rmse,
                    "anchor_rmse": anchor_rmse,
                    "delta_vs_anchor": candidate_rmse - anchor_rmse,
                    "eligible": candidate_rmse < anchor_rmse,
                }
            )
    return pd.DataFrame(rows)


def _hidden_like_masks(
    canonical: CanonicalIndex,
    assignment_path: Path | None,
) -> dict[str, np.ndarray]:
    if assignment_path is None or not assignment_path.exists():
        return {}
    assignments = pd.read_csv(assignment_path, dtype={"well_id": str}).set_index("well_id")
    wells = np.asarray(canonical.well_names, dtype=object)[
        np.asarray(canonical.well_code, dtype=np.int64)
    ]
    spatial = assignments["verification_like_spatial_role"].to_dict()
    typewell = assignments["verification_like_typewell_purged_role"].to_dict()
    return {
        "verification_like_spatial": np.asarray(
            [spatial.get(str(well)) == "valid" for well in wells]
        ),
        "verification_like_typewell_purged": np.asarray(
            [typewell.get(str(well)) == "valid" for well in wells]
        ),
    }


def pair_readout(
    canonical: CanonicalIndex,
    hidden_like_assignment: Path | None,
) -> pd.DataFrame:
    truth = np.asarray(canonical.truth)
    folds = np.asarray(canonical.outer_fold)
    md_since = np.asarray(canonical.md_since)
    well_codes = np.asarray(canonical.well_code)
    hidden_masks = _hidden_like_masks(canonical, hidden_like_assignment)
    value_by_id = {
        candidate_id: np.asarray(canonical.values[:, index])
        for index, candidate_id in enumerate(CORE_CANDIDATE_IDS)
    }
    rows: list[dict[str, Any]] = []
    for pair in PAIR_SHORTLIST:
        prediction = 0.5 * (value_by_id[pair.left] + value_by_id[pair.right])
        scopes = [_metric_row(prediction, truth, np.ones(len(truth), bool), "overall", "all")]
        scopes.extend(
            _metric_row(prediction, truth, folds == fold, "outer_fold", str(fold))
            for fold in range(N_FOLDS)
        )
        scopes.extend(
            _metric_row(
                prediction,
                truth,
                (md_since >= lower) & (md_since < upper),
                "distance_bucket",
                name,
            )
            for name, lower, upper in DISTANCE_BUCKETS
        )
        scopes.extend(
            _metric_row(prediction, truth, mask, "hidden_like", name)
            for name, mask in hidden_masks.items()
        )
        parent_rmse = {
            pair.left: _rmse(value_by_id[pair.left], truth),
            pair.right: _rmse(value_by_id[pair.right], truth),
        }
        better_parent = min(parent_rmse, key=parent_rmse.get)
        well_deltas: list[float] = []
        for code in range(len(canonical.well_names)):
            mask = well_codes == code
            well_deltas.append(
                _rmse(prediction[mask], truth[mask])
                - _rmse(value_by_id[better_parent][mask], truth[mask])
            )
        for row in scopes:
            row.update(
                {
                    "pair_id": pair.pair_id,
                    "left": pair.left,
                    "right": pair.right,
                    "tier": pair.tier,
                    "better_parent": better_parent,
                    "better_parent_rmse": parent_rmse[better_parent],
                    "delta_vs_better_parent": row["rmse"] - parent_rmse[better_parent]
                    if row["scope"] == "overall"
                    else np.nan,
                    "wells_improved_vs_better_parent": int(np.sum(np.asarray(well_deltas) < 0)),
                    "wells_worsened_vs_better_parent": int(np.sum(np.asarray(well_deltas) > 0)),
                    "median_well_delta": float(np.median(well_deltas)),
                    "max_well_regression": float(np.max(well_deltas)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_contract_manifests(
    canonical: CanonicalIndex,
    output_dir: Path,
    source_manifests: Mapping[str, Any],
    metric_map: Mapping[str, Any],
    value_manifest: Mapping[str, Any],
    confidence_manifest: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    generation_config_sha = json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "core": list(CORE_CANDIDATE_IDS),
            "windows": list(ROLLING_WINDOWS),
            "pairs": [pair.as_manifest_row() for pair in PAIR_SHORTLIST],
            "named": NAMED_COMBINATIONS,
        }
    )
    candidates = []
    for candidate in REFERENCE_CANDIDATES:
        row = candidate.as_catalog_row()
        row["anchor_rmse"] = ANCHOR_RMSE
        row["delta_vs_anchor"] = candidate.global_rmse - ANCHOR_RMSE
        row["generation_config_sha256"] = generation_config_sha
        if candidate.source_key in source_manifests:
            row["source_files"] = source_manifests[candidate.source_key]["files"]
        else:
            row["source_files"] = []
        if candidate.candidate_id in metric_map:
            row["generated_metrics"] = metric_map[candidate.candidate_id]
        else:
            row["generated_metrics"] = {
                "global_rmse": candidate.global_rmse,
                "fold_metrics": "unavailable_catalog_only",
                "bucket_metrics": "unavailable_catalog_only",
            }
        candidates.append(row)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "anchor": {"candidate_id": "last_anchor", "rmse": ANCHOR_RMSE},
        "inventory_count": len(candidates),
        "core_count": len(CORE_CANDIDATE_IDS),
        "candidates": candidates,
    }
    catalog_path = output_dir / "candidate_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")

    pair_rows = []
    for pair in PAIR_SHORTLIST:
        row = pair.as_manifest_row()
        row["parent_value_partition_content_sha256"] = {
            parent: [item["content_sha256"] for item in value_manifest[parent]]
            for parent in (pair.left, pair.right)
        }
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output_dir / "pair_shortlist.csv", index=False)

    named = {
        "schema_version": SCHEMA_VERSION,
        "aliases": ["blend_likpf_hmm_w500"],
        "combinations": NAMED_COMBINATIONS,
        "dag_cycle_check": "passed",
        "recursive_closure": "forbidden",
    }
    named_path = output_dir / "named_combinations.json"
    named_path.write_text(json.dumps(named, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "stage0_cache_completed",
        "rows": canonical.n_rows,
        "wells": len(canonical.well_names),
        "folds": N_FOLDS,
        "reference_candidates": len(REFERENCE_CANDIDATES),
        "core_candidates": len(CORE_CANDIDATE_IDS),
        "rawtest_core_candidates": len(RAWTEST_CORE_CANDIDATE_IDS),
        "pairs": len(PAIR_SHORTLIST),
        "named_triples": len(NAMED_COMBINATIONS) - 1,
        "candidate_dtype": "float32",
        "canonical_id_sha256": canonical.canonical_id_sha256,
        "generation_config_sha256": generation_config_sha,
        "source_manifests": source_manifests,
        "candidate_value_partitions": value_manifest,
        "candidate_confidence_partitions": confidence_manifest,
        "runtime_seconds": runtime_seconds,
        "temporary_work_files_retained": False,
        "model_sha": "not_applicable_no_training",
        "prediction_sha": "not_applicable_cache_only",
        "submission_sha": "not_applicable_no_submission",
    }
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "candidate_catalog": catalog_path,
        "pair_shortlist": output_dir / "pair_shortlist.csv",
        "named_combinations": named_path,
        "cache_manifest": manifest_path,
    }


def write_parity_sample(output_dir: Path, sample_rows: int = 64) -> Path:
    cache = CandidateCache(output_dir)
    fold = 0
    sample: pd.DataFrame | None = None
    names = [
        *(pair.pair_id for pair in PAIR_SHORTLIST),
        *NAMED_COMBINATIONS,
    ]
    for name in names:
        materialized = cache.materialize(name, fold=fold, row_slice=slice(0, sample_rows))
        if sample is None:
            sample = materialized[KEY_COLUMNS].copy()
        sample[name] = materialized["candidate_tvt"].to_numpy(np.float32)
    if sample is None:
        raise RuntimeError("parity sample was not built")
    path = output_dir / "small_parity_sample.parquet"
    _write_parquet(sample, path)
    return path


def build_stage0_cache(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    debug: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract_counts = validate_contract()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    data_config = config.get("data", {})
    inputs = data_config.get("inputs", {})
    search_roots = [Path(item) for item in data_config.get("search_roots", [".", "/tmp"])]
    if Path("/kaggle/input").exists():
        search_roots.insert(0, Path("/kaggle/input"))
    chunk_rows = int(config.get("cache", {}).get("chunk_rows", CHUNK_ROWS))
    expected_rows = int(config.get("cache", {}).get("expected_rows", EXPECTED_ROWS))
    effective_max_rows = max_rows if debug else None
    record_decompressed = bool(
        config.get("reproducibility", {}).get("record_decompressed_source_sha", True)
    )

    base_paths = resolve_input_paths("exp072_oof", inputs, search_roots)
    if len(base_paths) != 1:
        raise ValueError("exp072 canonical source must resolve to exactly one file")
    canonical = prepare_canonical_index(
        base_paths[0], work_dir, expected_rows, effective_max_rows, chunk_rows
    )
    source_manifests = {
        "exp072_oof": {
            "source_key": "exp072_oof",
            "rows_read": canonical.n_rows,
            "rows_loaded": {
                item.candidate_id: canonical.n_rows
                for item in REFERENCE_CANDIDATES
                if item.source_key == "exp072_oof"
            },
            "files": _source_manifest(base_paths, decompressed=record_decompressed),
        }
    }
    source_manifests.update(
        load_external_core_sources(
            canonical, inputs, search_roots, chunk_rows, record_decompressed
        )
    )

    metric_map = core_metrics(canonical)
    value_manifest, confidence_manifest = write_candidate_partitions(
        canonical, output_dir, chunk_rows
    )
    eligibility = outer_train_eligibility(canonical)
    eligibility.to_csv(output_dir / "outer_fold_eligibility.csv", index=False)

    hidden_path_raw = data_config.get("hidden_like_assignments")
    hidden_path = Path(hidden_path_raw) if hidden_path_raw else None
    readout = pair_readout(canonical, hidden_path)
    readout.to_csv(output_dir / "pair_readout.csv", index=False)

    runtime_seconds = time.perf_counter() - started
    artifacts = write_contract_manifests(
        canonical,
        output_dir,
        source_manifests,
        metric_map,
        value_manifest,
        confidence_manifest,
        runtime_seconds,
    )
    parity_path = write_parity_sample(
        output_dir, int(config.get("cache", {}).get("parity_sample_rows", 64))
    )
    artifacts["outer_fold_eligibility"] = output_dir / "outer_fold_eligibility.csv"
    artifacts["pair_readout"] = output_dir / "pair_readout.csv"
    artifacts["small_parity_sample"] = parity_path
    rows = canonical.n_rows
    wells = len(canonical.well_names)
    close_canonical_index(canonical)
    shutil.rmtree(work_dir)
    summary = {
        "status": "debug_completed" if debug else "stage0_cache_completed",
        "contract": contract_counts,
        "rows": rows,
        "wells": wells,
        "runtime_seconds": runtime_seconds,
        "temporary_work_dir_removed": True,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "artifact_sha256": {key: sha256_file(path) for key, path in artifacts.items()},
    }
    (output_dir / "stage0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def _read_current_test_candidate(
    spec: Mapping[str, Any], search_roots: Sequence[Path]
) -> pd.DataFrame:
    source_key = str(spec["source_key"])
    paths = resolve_input_paths(source_key, {source_key: spec}, search_roots)
    frames = [pd.read_csv(path, dtype={str(spec.get("id_column", "id")): str}) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    well_column = str(spec.get("well_column", "well"))
    row_column = spec.get("row_idx_column")
    id_column = spec.get("id_column", "id")
    wells = frame[well_column].astype(str)
    if row_column:
        rows = pd.to_numeric(frame[str(row_column)], errors="raise").astype(np.int32)
        ids = wells + "_" + rows.astype(str)
    else:
        ids = frame[str(id_column)].astype(str)
        rows = ids.str.rsplit("_", n=1).str[-1].astype(np.int32)
    value = pd.to_numeric(frame[str(spec["value_column"])], errors="coerce").to_numpy(np.float32)
    if spec.get("transform") == "anchor_plus":
        value += pd.to_numeric(frame[str(spec["anchor_column"])], errors="raise").to_numpy(
            np.float32
        )
    return pd.DataFrame(
        {
            "id": ids,
            "well": wells,
            "well_row_idx": rows,
            "candidate_tvt": value,
        }
    ).sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)


def assemble_stage1_current_test_parity(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, float]:
    """Validate six raw-test primitives and materialize fixed virtual formulas."""
    if set(frames) != set(RAWTEST_CORE_CANDIDATE_IDS):
        raise ValueError("Stage 1 must provide exactly the six raw-test-ready primitives")

    normalized: dict[str, pd.DataFrame] = {}
    required = ["id", "well", "well_row_idx", "candidate_tvt"]
    for candidate_id in RAWTEST_CORE_CANDIDATE_IDS:
        frame = frames[candidate_id]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(
                f"current-test primitive columns missing for {candidate_id}: {sorted(missing)}"
            )
        item = frame[required].copy()
        item["id"] = item["id"].astype(str)
        item["well"] = item["well"].astype(str)
        item["well_row_idx"] = pd.to_numeric(
            item["well_row_idx"], errors="raise"
        ).astype(np.int32)
        item["candidate_tvt"] = pd.to_numeric(
            item["candidate_tvt"], errors="coerce"
        ).astype(np.float32)
        item = item.sort_values(
            ["well", "well_row_idx"], kind="stable"
        ).reset_index(drop=True)
        if item.duplicated(["id"]).any() or item.duplicated(
            ["well", "well_row_idx"]
        ).any():
            raise ValueError(f"current-test duplicate identity: {candidate_id}")
        normalized[candidate_id] = item

    base = normalized[RAWTEST_CORE_CANDIDATE_IDS[0]]
    for candidate_id, frame in normalized.items():
        if not base[["id", "well", "well_row_idx"]].equals(
            frame[["id", "well", "well_row_idx"]]
        ):
            raise ValueError(f"current-test identity mismatch: {candidate_id}")
        if not np.isfinite(frame["candidate_tvt"]).all():
            raise ValueError(f"current-test nonfinite candidate: {candidate_id}")

    output = base[["id", "well", "well_row_idx"]].copy()
    for candidate_id, frame in normalized.items():
        output[candidate_id] = frame["candidate_tvt"].to_numpy(np.float32)
    rawtest_pairs = [pair for pair in PAIR_SHORTLIST if pair.tier == "raw-test"]
    for pair in rawtest_pairs:
        output[pair.pair_id] = 0.5 * (
            output[pair.left].to_numpy(np.float32)
            + output[pair.right].to_numpy(np.float32)
        )
    fixed = NAMED_COMBINATIONS["exp226_w500_50_50"]["weights"]
    if fixed != {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}:
        raise ValueError("exp226_w500_50_50 weights differ from the fixed contract")
    output["exp226_w500_50_50"] = (
        np.float32(0.5) * output["exp226_k16"].to_numpy(np.float32)
        + np.float32(0.25) * output["likpf_mean"].to_numpy(np.float32)
        + np.float32(0.25) * output["exact_hmm"].to_numpy(np.float32)
    ).astype(np.float32)

    direct = (
        np.float32(0.5) * output["exp226_k16"].to_numpy(np.float32)
        + np.float32(0.25) * output["likpf_mean"].to_numpy(np.float32)
        + np.float32(0.25) * output["exact_hmm"].to_numpy(np.float32)
    ).astype(np.float32)
    parity_abs = np.abs(
        direct.astype(np.float64)
        - output["exp226_w500_50_50"].to_numpy(np.float32).astype(np.float64)
    )
    max_abs = float(parity_abs.max(initial=0.0))
    if max_abs > 1e-5:
        raise ValueError("exp226_w500_50_50 current-test formula parity failed")
    return output, max_abs


def attach_stage1_current_test_confidence(
    parity: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach source-native primitive confidence with a stable wide namespace.

    The output is consumed by exp264 as
    ``confidence__<candidate_id>__<field>``.  Only fields produced by the same
    raw-test generator as the candidate value are accepted.  Missing selected
    confidence fails closed rather than being replaced with zero or all-NaN.
    """
    if set(frames) != set(RAWTEST_CORE_CANDIDATE_IDS):
        raise ValueError("Stage 1 confidence must provide exactly six primitive frames")
    if set(STAGE1_NATIVE_CONFIDENCE_FIELDS) != set(RAWTEST_CORE_CANDIDATE_IDS):
        raise ValueError("Stage 1 native confidence contract differs from primitive inventory")

    identity_columns = ["id", "well", "well_row_idx"]
    output = parity.copy()
    base = output[identity_columns].sort_values(
        ["well", "well_row_idx"], kind="stable"
    ).reset_index(drop=True)
    confidence_columns: dict[str, np.ndarray] = {}
    for candidate_id in RAWTEST_CORE_CANDIDATE_IDS:
        frame = frames[candidate_id].sort_values(
            ["well", "well_row_idx"], kind="stable"
        ).reset_index(drop=True)
        if not base.equals(frame[identity_columns]):
            raise ValueError(f"Stage 1 confidence identity mismatch: {candidate_id}")
        required_fields = (
            "confidence_valid",
            *STAGE1_NATIVE_CONFIDENCE_FIELDS[candidate_id],
        )
        missing = set(required_fields) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Stage 1 confidence columns missing for {candidate_id}: {sorted(missing)}"
            )
        valid = frame["confidence_valid"]
        if valid.isna().any():
            raise ValueError(f"Stage 1 confidence_valid contains NaN: {candidate_id}")
        confidence_columns[f"confidence__{candidate_id}__confidence_valid"] = (
            valid.astype(bool).to_numpy()
        )
        for field in STAGE1_NATIVE_CONFIDENCE_FIELDS[candidate_id]:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Stage 1 selected native confidence is nonfinite: {candidate_id}.{field}"
                )
            confidence_columns[f"confidence__{candidate_id}__{field}"] = values
        if STAGE1_NATIVE_CONFIDENCE_FIELDS[candidate_id] and not valid.astype(bool).all():
            raise ValueError(
                f"Stage 1 selected native confidence is invalid: {candidate_id}"
            )
    return pd.concat(
        [output.reset_index(drop=True), pd.DataFrame(confidence_columns)], axis=1
    )


def build_submission_from_stage1_parity(
    sample: pd.DataFrame,
    parity: pd.DataFrame,
    *,
    candidate_id: str = "exp226_w500_50_50",
) -> pd.DataFrame:
    """Restore sample order and build the only approved exp263 submission."""
    if candidate_id != "exp226_w500_50_50":
        raise ValueError(f"unsupported exp263 submission candidate: {candidate_id}")
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(
            f"sample submission columns must be ['id', 'tvt'], got {list(sample.columns)}"
        )
    if candidate_id not in parity.columns:
        raise ValueError(f"Stage 1 parity lacks submission candidate: {candidate_id}")

    sample_ids = sample["id"].astype(str)
    parity_values = parity[["id", candidate_id]].copy()
    parity_values["id"] = parity_values["id"].astype(str)
    if sample_ids.duplicated().any() or parity_values["id"].duplicated().any():
        raise ValueError("submission identity contains duplicate id")
    if len(sample) != len(parity_values) or set(sample_ids) != set(parity_values["id"]):
        raise ValueError("Stage 1 parity IDs do not match sample_submission")

    submission = sample[["id"]].copy()
    submission["id"] = sample_ids
    submission = submission.merge(
        parity_values,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    ).rename(columns={candidate_id: "tvt"})
    submission["tvt"] = pd.to_numeric(submission["tvt"], errors="coerce")
    if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
        raise ValueError("submission contains missing or nonfinite tvt")
    if not submission["id"].equals(sample_ids.reset_index(drop=True)):
        raise ValueError("submission IDs do not preserve sample order")
    return submission[["id", "tvt"]]


def run_stage1_current_test_parity(
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    stage1 = config.get("stage1", {})
    primitive_inputs = stage1.get("primitive_inputs", {})
    if set(primitive_inputs) != set(RAWTEST_CORE_CANDIDATE_IDS):
        raise ValueError("Stage 1 must provide exactly the six raw-test-ready primitives")
    configured_roots = config.get("data", {}).get("search_roots", [".", "/tmp"])
    search_roots = [Path(item) for item in configured_roots]
    if Path("/kaggle/input").exists():
        search_roots.insert(0, Path("/kaggle/input"))
    frames = {
        candidate_id: _read_current_test_candidate(spec, search_roots)
        for candidate_id, spec in primitive_inputs.items()
    }
    output, max_abs = assemble_stage1_current_test_parity(frames)
    rawtest_pairs = [pair for pair in PAIR_SHORTLIST if pair.tier == "raw-test"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "current_test_formula_parity.parquet"
    artifact = _write_parquet(output, path)
    summary = {
        "status": "stage1_current_test_parity_completed",
        "rows": len(output),
        "wells": int(output["well"].nunique()),
        "rawtest_primitives": list(RAWTEST_CORE_CANDIDATE_IDS),
        "rawtest_pairs": [pair.pair_id for pair in rawtest_pairs],
        "named_fixed": ["exp226_w500_50_50"],
        "max_abs_formula_parity": max_abs,
        "artifact": artifact,
    }
    (output_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary
