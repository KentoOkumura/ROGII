from __future__ import annotations

import numpy as np
import pandas as pd

from src.gr_residual_noise_augmentation import (
    ResidualProfile,
    content_sha256,
    read_residual_profile,
    stable_uint64,
    synthesize_residual_view,
)


def _write_well(tmp_path, well: str, residual: np.ndarray, missing: np.ndarray):
    rows = len(residual)
    tvt = np.arange(rows, dtype=np.float64) + 10_000.0
    type_gr = 80.0 + 10.0 * np.sin(np.arange(rows, dtype=np.float64) / 17.0)
    observed = 1.25 * type_gr - 4.0 + residual
    observed = observed.astype(np.float64)
    observed[missing] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=np.float64),
            "TVT": tvt,
            "TVT_input": np.where(np.arange(rows) < rows // 2, tvt, np.nan),
            "GR": observed,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": type_gr})
    horizontal_path = tmp_path / f"{well}__horizontal_well.csv"
    typewell_path = tmp_path / f"{well}__typewell.csv"
    horizontal.to_csv(horizontal_path, index=False)
    typewell.to_csv(typewell_path, index=False)
    return horizontal_path, typewell_path


def _profile(tmp_path, well: str, phase: float = 0.0) -> ResidualProfile:
    rows = 320
    residual = 3.0 * np.sin(np.arange(rows, dtype=np.float64) / 9.0 + phase)
    residual[100] += 18.0
    missing = np.zeros(rows, dtype=bool)
    missing[150:160] = True
    horizontal, typewell = _write_well(tmp_path, well, residual, missing)
    return read_residual_profile(well, horizontal, typewell, fit_scope="full_true_tvt")


def test_affine_residual_profile_recovers_expected_signal(tmp_path):
    profile = _profile(tmp_path, "well_a")
    assert abs(profile.gain - 1.25) < 0.08
    assert abs(profile.bias + 4.0) < 7.0
    assert profile.fit_points >= 250
    assert profile.metadata["missing_run_max"] == 10
    assert profile.metadata["haar_dwt_detail_energy"] > 0.0
    finite = ~profile.missing_mask
    reconstructed = profile.clean_gr[finite] + profile.residual[finite]
    assert np.isfinite(reconstructed).all()


def test_block_transplant_is_stable_and_uses_only_supplied_donors(tmp_path):
    recipient = _profile(tmp_path, "recipient")
    donor_a = _profile(tmp_path, "donor_a", phase=0.7)
    donor_b = _profile(tmp_path, "donor_b", phase=1.4)
    first = synthesize_residual_view(
        recipient,
        [donor_b, donor_a],
        variant="real_residual_block",
        seed_parts=(42, 0, 1),
        block_lengths=(32, 64),
    )
    second = synthesize_residual_view(
        recipient,
        [donor_a, donor_b],
        variant="real_residual_block",
        seed_parts=(42, 0, 1),
        block_lengths=(32, 64),
    )
    assert content_sha256(first.imputed_gr) == content_sha256(second.imputed_gr)
    assert {item["donor_well"] for item in first.inventory} <= {"donor_a", "donor_b"}
    assert all(item["donor_well"] != "recipient" for item in first.inventory)


def test_negative_controls_keep_shape_and_are_distinct(tmp_path):
    recipient = _profile(tmp_path, "recipient")
    donor = _profile(tmp_path, "donor", phase=0.9)
    real = synthesize_residual_view(
        recipient,
        [donor],
        variant="real_residual_block",
        seed_parts=(42, "real"),
        block_lengths=(64,),
    )
    white = synthesize_residual_view(
        recipient,
        [donor],
        variant="white_noise",
        seed_parts=(42, "white"),
        block_lengths=(64,),
    )
    shuffled = synthesize_residual_view(
        recipient,
        [donor],
        variant="shuffled_residual",
        seed_parts=(42, "shuffled"),
        block_lengths=(64,),
    )
    clean = synthesize_residual_view(
        recipient,
        [donor],
        variant="clean_duplicate",
        seed_parts=(42, "clean"),
        block_lengths=(64,),
    )
    assert all(len(view.imputed_gr) == len(recipient.md) for view in (real, white, shuffled, clean))
    assert np.isfinite(real.imputed_gr).all()
    assert np.isfinite(white.imputed_gr).all()
    assert np.isfinite(shuffled.imputed_gr).all()
    assert not np.array_equal(real.imputed_gr, white.imputed_gr)
    assert not np.array_equal(real.imputed_gr, shuffled.imputed_gr)
    finite = ~recipient.missing_mask
    assert np.allclose(
        clean.transplanted_residual[finite], recipient.residual[finite], atol=1e-6
    )
    assert np.array_equal(clean.missing_mask, recipient.missing_mask)


def test_stable_seed_depends_on_immutable_key():
    assert stable_uint64(42, "well", 0) == stable_uint64(42, "well", 0)
    assert stable_uint64(42, "well", 0) != stable_uint64(42, "well", 1)
