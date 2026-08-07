from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT_DIR = Path(
    "experiments/exp376_exp226_formation_conditioned_k16_donor_kernel"
)
SOURCE_PATH = (
    EXPERIMENT_DIR
    / "exp376_exp226_formation_conditioned_k16_donor_kernel_compact_selfcontained_train.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RUN_CONFIG_PATH = EXPERIMENT_DIR / "kaggle" / "train" / "config.yaml"


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP376_IMPORT_ONLY")
    os.environ["EXP376_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(SOURCE_PATH))
    finally:
        if previous is None:
            os.environ.pop("EXP376_IMPORT_ONLY", None)
        else:
            os.environ["EXP376_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


def test_execution_contract_is_one_k16_variant_and_one_run_authorized(
    module: dict[str, object],
) -> None:
    run_config = module["read_yaml"](RUN_CONFIG_PATH)
    module["validate_execution_contract"](
        run_config, require_kaggle_authorization=True
    )
    assert module["EXPECTED_VARIANTS"] == {"formation_conditioned_k16": 16}
    config = module["read_yaml"](CONFIG_PATH)
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["kaggle_v2_execution_authorized"] is True
    assert config["execution"]["kaggle_target_version"] == 2
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False


def test_formation_plane_excludes_target_well(
    module: dict[str, object],
) -> None:
    wells = np.asarray([f"w{index:02d}" for index in range(12)], dtype=object)
    xy = np.column_stack(
        [np.arange(12.0), np.square(np.arange(12.0)) / 20.0]
    )
    formation = np.column_stack(
        [100.0 * column + 2.0 * np.arange(12.0) for column in range(6)]
    )
    formation[0] += 10_000.0
    plane = module["FormationPlaneKNN"](
        wells=wells,
        xy=xy,
        formation_medians=formation,
        k=10,
    )
    with_self, _ = plane.impute(xy[:1], target_well=None)
    without_self, _ = plane.impute(xy[:1], target_well="w00")
    assert float(with_self[0, 0]) > float(without_self[0, 0]) + 1_000.0


def test_soft_factor_formula_and_nonfinite_query_fallback(
    module: dict[str, object],
) -> None:
    rows = 50
    donor_signature = np.ones((rows, 11), dtype=float)
    field = np.column_stack(
        [
            np.arange(rows, dtype=float),
            np.zeros(rows, dtype=float),
            np.ones(rows, dtype=float),
            np.arange(rows) % 20,
            np.arange(rows) % 16,
            donor_signature,
        ]
    )
    query_mid = np.zeros((16, 2), dtype=float)
    query_signature = np.zeros((16, 11), dtype=float)
    params = module["K16Params"]()
    prediction, distance, support = module["local_linear"](
        field,
        -1,
        query_mid,
        query_signature,
        np.zeros(11),
        np.ones(11),
        params,
    )
    expected_factor = 0.5 + 0.5 * np.exp(-0.5)
    assert np.isfinite(prediction).all()
    assert np.isfinite(distance).all()
    assert np.allclose(support.formation_factor_min, expected_factor)
    assert np.allclose(support.formation_factor_max, expected_factor)
    assert not support.fallback.any()

    query_signature[0, 0] = np.nan
    _, _, fallback = module["local_linear"](
        field,
        -1,
        query_mid,
        query_signature,
        np.zeros(11),
        np.ones(11),
        params,
    )
    assert fallback.fallback[0]
    assert fallback.formation_factor_min[0] == 1.0
    assert fallback.formation_factor_max[0] == 1.0


def test_frame_content_hash_accepts_list_valued_manifest_cells(
    module: dict[str, object],
) -> None:
    frame = module["pd"].DataFrame(
        {
            "outer_fold": [0, 1],
            "unavailable_reference_wells": [[], ["well_b", "well_a"]],
        }
    )
    first = module["frame_content_sha256"](frame)
    second = module["frame_content_sha256"](frame.copy())
    assert first == second
    assert len(first) == 64
