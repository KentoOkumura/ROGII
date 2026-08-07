from __future__ import annotations

import numpy as np
import pandas as pd

from src.exact_datum_ranker_augmentation import (
    assign_stable_tvt_shifts,
    build_exact_tvt_datum_long_view,
    select_stable_wells,
)


def test_stable_well_selection_and_shift_are_order_independent() -> None:
    wells = [f"well_{index:02d}" for index in range(20)]
    selected = select_stable_wells(wells, fraction=0.25, seed=42, namespace="exp259")
    reversed_selected = select_stable_wells(
        list(reversed(wells)), fraction=0.25, seed=42, namespace="exp259"
    )
    assert selected == reversed_selected
    assert len(selected) == 5

    shifts = assign_stable_tvt_shifts(
        selected,
        shift_grid_ft=[-40.0, -20.0, 20.0, 40.0],
        seed=42,
        namespace="exp259",
    )
    reversed_shifts = assign_stable_tvt_shifts(
        list(reversed(selected)),
        shift_grid_ft=[-40.0, -20.0, 20.0, 40.0],
        seed=42,
        namespace="exp259",
    )
    assert shifts == reversed_shifts
    assert set(shifts.values()).issubset({-40.0, -20.0, 20.0, 40.0})


def test_exact_datum_long_view_shifts_only_absolute_features() -> None:
    clean = pd.DataFrame(
        {
            "id": ["a0", "b0", "a1", "b1"],
            "well": ["a", "b", "a", "b"],
            "last_known_tvt": np.asarray([100.0, 200.0, 110.0, 210.0], dtype=np.float32),
            "candidate_tvt": np.asarray([105.0, 205.0, 115.0, 215.0], dtype=np.float32),
            "candidate_minus_last": np.asarray([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            "candidate_multiobs_score": np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        }
    )
    error = np.asarray([3.0, 7.0, 12.0, 1.0], dtype=np.float32)
    binary = (error <= 10.0).astype(np.int8)
    augmented, aug_error, aug_binary, guard = build_exact_tvt_datum_long_view(
        clean,
        error,
        binary,
        shift_by_well={"a": 40.0},
        absolute_tvt_feature_columns=["last_known_tvt", "candidate_tvt"],
        selected_feature_columns=[
            "last_known_tvt",
            "candidate_tvt",
            "candidate_minus_last",
            "candidate_multiobs_score",
        ],
        tolerance=1.0e-5,
    )

    np.testing.assert_allclose(augmented["last_known_tvt"], [140.0, 150.0])
    np.testing.assert_allclose(augmented["candidate_tvt"], [145.0, 155.0])
    np.testing.assert_allclose(augmented["candidate_minus_last"], [5.0, 5.0])
    np.testing.assert_allclose(augmented["candidate_multiobs_score"], [0.2, 0.4])
    np.testing.assert_array_equal(aug_error, [3.0, 12.0])
    np.testing.assert_array_equal(aug_binary, [1, 0])
    assert guard["pass"] is True
    assert guard["augmented_wells"] == 1
