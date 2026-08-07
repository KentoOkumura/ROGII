from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coordinate_path_augmentation import (
    ALL_TRANSFORMS,
    APPROXIMATE_TRANSFORMS,
    STRICT_TRANSFORMS,
    TransformSpec,
    apply_transform,
    choose_transform_spec,
    evaluate_distribution_guard,
    exact_inverse_error,
    fit_distribution_envelope,
    inverse_exact_transform,
    official_start_index,
    resample_typewell_gr,
    stable_choice,
    summarize_well,
)


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 16
    md = np.arange(rows, dtype=float) * 10.0
    x = 1000.0 + md
    y = 2000.0 + 0.12 * md + 0.002 * md**2
    z = 500.0 - 0.06 * md
    tvt = 60.0 + 0.42 * md + 0.0008 * md**2
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
    typewell_tvt = np.linspace(0.0, 180.0, 721)
    typewell_gr = 80.0 + 12.0 * np.sin(typewell_tvt / 13.0)
    expected_gr = np.interp(tvt, typewell_tvt, typewell_gr)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "TVT": tvt,
            "GR": expected_gr + np.linspace(-1.0, 1.0, rows),
            "TVT_input": tvt_input,
            "candidate_a": tvt + 5.0,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell


@pytest.mark.parametrize(
    ("kind", "parameters"),
    [
        ("heel_center_translation", {}),
        ("lateral_reflection", {}),
        ("yaw_rotation", {"angle_degrees": 90.0}),
        ("tvt_datum_shift", {"shift_ft": 40.0}),
    ],
)
def test_exact_transforms_roundtrip(kind: str, parameters: dict[str, float]) -> None:
    horizontal, typewell = synthetic_well()
    spec = TransformSpec(kind=kind, parameters=parameters, exact=True)
    result = apply_transform(horizontal, typewell, spec, candidate_columns=["candidate_a"])
    inverted = inverse_exact_transform(result, spec)
    errors = exact_inverse_error(horizontal, typewell, inverted, candidate_columns=["candidate_a"])

    assert errors["max_abs"] < 1e-9
    assert np.array_equal(
        horizontal["TVT_input"].isna().to_numpy(),
        inverted.horizontal["TVT_input"].isna().to_numpy(),
    )


@pytest.mark.parametrize(
    ("kind", "parameters"),
    [
        ("md_stretch", {"factor": 1.03}),
        ("tvt_shear", {"tail_delta_ft": 20.0}),
        ("xy_plane_tilt", {"slope_x": 0.005, "slope_y": -0.005}),
        (
            "low_frequency_spline_warp",
            {"amplitude_ft": 20.0, "middle_sign": -1.0},
        ),
        (
            "smooth_xyz_control_perturbation",
            {"amplitude_ft": 10.0, "sign_x": 1.0, "sign_y": -1.0, "sign_z": 1.0},
        ),
    ],
)
def test_approximate_transforms_preserve_prefix_and_resample_tail_gr(
    kind: str, parameters: dict[str, float]
) -> None:
    horizontal, typewell = synthetic_well()
    anchor = official_start_index(horizontal)
    spec = TransformSpec(kind=kind, parameters=parameters, exact=False)
    result = apply_transform(horizontal, typewell, spec, candidate_columns=["candidate_a"])

    prefix_columns = ["MD", "X", "Y", "Z", "TVT", "TVT_input", "GR"]
    pd.testing.assert_frame_equal(
        horizontal.loc[:anchor, prefix_columns].reset_index(drop=True),
        result.horizontal.loc[:anchor, prefix_columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert result.metadata["anchor_max_abs_delta"] == pytest.approx(0.0)
    assert np.all(np.diff(result.horizontal["MD"].to_numpy(float)) > 0.0)

    expected_gr, coverage = resample_typewell_gr(
        result.typewell, result.horizontal["TVT"].to_numpy(float)
    )
    np.testing.assert_allclose(
        result.horizontal["GR"].to_numpy(float)[anchor + 1 :],
        expected_gr[anchor + 1 :],
    )
    assert coverage[anchor + 1 :].all()


def test_tvt_warp_moves_candidate_equivariantly() -> None:
    horizontal, typewell = synthetic_well()
    original_error = horizontal["candidate_a"] - horizontal["TVT"]
    spec = TransformSpec(kind="tvt_shear", parameters={"tail_delta_ft": 20.0}, exact=False)
    result = apply_transform(horizontal, typewell, spec, candidate_columns=["candidate_a"])
    transformed_error = result.horizontal["candidate_a"] - result.horizontal["TVT"]
    np.testing.assert_allclose(original_error, transformed_error)


def test_stable_parameter_choice_is_order_independent() -> None:
    grid = {
        "slope_x": [-0.01, -0.005, 0.005, 0.01],
        "slope_y": [-0.01, -0.005, 0.005, 0.01],
    }
    left = choose_transform_spec(
        "xy_plane_tilt", seed=42, well="abc", view_slot=0, parameter_grid=grid
    )
    right = choose_transform_spec(
        "xy_plane_tilt",
        seed=42,
        well="abc",
        view_slot=0,
        parameter_grid=dict(reversed(list(grid.items()))),
    )
    assert left == right
    assert stable_choice([1, 2, 3], 42, "abc") == stable_choice([1, 2, 3], 42, "abc")


def test_distribution_guard_rejects_out_of_envelope_view() -> None:
    horizontal, typewell = synthetic_well()
    clean = summarize_well(horizontal, typewell)
    real = pd.DataFrame([{**clean, "well": f"w{index}"} for index in range(8)])
    envelope = fit_distribution_envelope(
        real,
        lower_quantile=0.005,
        upper_quantile=0.995,
        relative_margin=0.25,
        min_typewell_coverage=0.95,
    )
    bad = dict(clean)
    bad["xy_slope_q95"] = float(envelope["metrics"]["xy_slope_q95"]["upper"]) + 10.0
    accepted, reasons = evaluate_distribution_guard(
        bad,
        envelope,
        exact=False,
        inverse_max_abs=None,
        inverse_tolerance=1e-7,
        anchor_tolerance=1e-9,
    )
    assert not accepted
    assert "out_of_envelope:xy_slope_q95" in reasons


def test_transform_catalog_is_partitioned() -> None:
    assert set(STRICT_TRANSFORMS).isdisjoint(APPROXIMATE_TRANSFORMS)
    assert set(ALL_TRANSFORMS) == set(STRICT_TRANSFORMS) | set(APPROXIMATE_TRANSFORMS)
