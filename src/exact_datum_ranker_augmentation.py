from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.coordinate_path_augmentation import stable_choice, stable_uint64


def select_stable_wells(
    wells: Sequence[str],
    *,
    fraction: float,
    seed: int,
    namespace: str,
) -> list[str]:
    """Select an exact-sized, order-independent well subset using SHA256 ranks."""
    unique = sorted({str(well) for well in wells})
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    count = int(round(len(unique) * fraction))
    ranked = sorted(unique, key=lambda well: (stable_uint64(seed, namespace, well), well))
    return sorted(ranked[:count])


def assign_stable_tvt_shifts(
    wells: Sequence[str],
    *,
    shift_grid_ft: Sequence[float],
    seed: int,
    namespace: str,
) -> dict[str, float]:
    grid = [float(value) for value in shift_grid_ft]
    if not grid:
        raise ValueError("shift_grid_ft must not be empty")
    if any(not np.isfinite(value) for value in grid):
        raise ValueError("shift_grid_ft must contain only finite values")
    return {
        str(well): float(stable_choice(grid, seed, namespace, str(well), "shift_ft"))
        for well in sorted({str(value) for value in wells})
    }


def build_exact_tvt_datum_long_view(
    clean_long: pd.DataFrame,
    clean_error: np.ndarray,
    clean_binary: np.ndarray,
    *,
    shift_by_well: Mapping[str, float],
    absolute_tvt_feature_columns: Sequence[str],
    selected_feature_columns: Sequence[str],
    tolerance: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Duplicate selected wells and apply a coherent exact TVT datum shift.

    Candidate errors, within-10 labels, and every selected relative feature remain
    unchanged. Only the explicitly enumerated absolute TVT features are translated.
    """
    if "well" not in clean_long.columns:
        raise ValueError("clean_long must contain a well column")
    if len(clean_long) != len(clean_error) or len(clean_long) != len(clean_binary):
        raise ValueError("clean feature and label lengths differ")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    selected = list(selected_feature_columns)
    absolute = list(absolute_tvt_feature_columns)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_feature_columns contains duplicates")
    if len(set(absolute)) != len(absolute):
        raise ValueError("absolute_tvt_feature_columns contains duplicates")
    missing_absolute = [column for column in absolute if column not in selected]
    if missing_absolute:
        raise ValueError(
            "absolute TVT features are outside the selected schema: "
            f"{missing_absolute}"
        )
    missing_frame = [column for column in selected if column not in clean_long.columns]
    if missing_frame:
        raise ValueError(f"selected features are missing from clean_long: {missing_frame[:20]}")

    normalized_shifts = {str(well): float(value) for well, value in shift_by_well.items()}
    if any(not np.isfinite(value) for value in normalized_shifts.values()):
        raise ValueError("shift_by_well contains non-finite values")
    well_values = clean_long["well"].astype(str)
    mask = well_values.isin(normalized_shifts).to_numpy()
    if not mask.any():
        raise ValueError("no clean_long rows matched shift_by_well")

    source = clean_long.loc[mask].copy().reset_index(drop=True)
    augmented = source.copy(deep=True)
    row_shifts = source["well"].astype(str).map(normalized_shifts).to_numpy(np.float64)
    if not np.isfinite(row_shifts).all():
        raise AssertionError("failed to map every augmented row to a finite shift")

    max_shift_error = 0.0
    for column in absolute:
        before = pd.to_numeric(source[column], errors="coerce").to_numpy(np.float64)
        after = before + row_shifts
        augmented[column] = after.astype(clean_long[column].dtype, copy=False)
        finite = np.isfinite(before)
        if finite.any():
            error = np.max(np.abs((after[finite] - before[finite]) - row_shifts[finite]))
            max_shift_error = max(max_shift_error, float(error))
        if not np.array_equal(np.isnan(before), np.isnan(after)):
            raise AssertionError(f"datum shift changed missingness for {column}")

    relative = [column for column in selected if column not in set(absolute)]
    for column in relative:
        left = source[column].to_numpy()
        right = augmented[column].to_numpy()
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"relative feature changed under datum shift: {column}")

    augmented_error = np.asarray(clean_error)[mask].copy()
    augmented_binary = np.asarray(clean_binary)[mask].copy()
    if not np.array_equal(augmented_error, np.asarray(clean_error)[mask], equal_nan=True):
        raise AssertionError("candidate error changed under datum shift")
    if not np.array_equal(augmented_binary, np.asarray(clean_binary)[mask]):
        raise AssertionError("within-10 label changed under datum shift")
    if max_shift_error > tolerance:
        raise AssertionError(
            f"exact TVT shift error {max_shift_error} exceeded tolerance {tolerance}"
        )

    guard = {
        "pass": True,
        "clean_long_rows": int(len(clean_long)),
        "augmented_long_rows": int(len(augmented)),
        "augmented_wells": int(source["well"].astype(str).nunique()),
        "absolute_tvt_feature_count": int(len(absolute)),
        "relative_feature_count": int(len(relative)),
        "candidate_error_invariant": True,
        "within10_label_invariant": True,
        "relative_features_invariant": True,
        "max_absolute_shift_error": float(max_shift_error),
        "tolerance": float(tolerance),
    }
    return augmented, augmented_error, augmented_binary, guard


__all__ = [
    "assign_stable_tvt_shifts",
    "build_exact_tvt_datum_long_view",
    "select_stable_wells",
]
