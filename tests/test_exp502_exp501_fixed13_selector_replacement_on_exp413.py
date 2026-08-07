from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp502_exp501_fixed13_selector_replacement_on_exp413"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"


def load_source() -> ModuleType:
    previous = os.environ.get("EXP502_IMPORT_ONLY")
    os.environ["EXP502_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp502_train", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP502_IMPORT_ONLY", None)
        else:
            os.environ["EXP502_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_source()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_config_records_authorized_15_booster_train_contract(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_static_contract(config)
    assert config["experiment"]["route"] == "ml_model"
    assert config["experiment"]["status"] in {
        "kaggle_train_package_ready",
        "kaggle_train_running",
        "kaggle_train_complete_pass",
        "kaggle_train_complete_fail",
        "kaggle_train_error",
    }
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is True
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_train_approved"] is True
    assert config["authorization"]["inference_implementation_approved"] is False
    assert config["authorization"]["external_submission_approved"] is False
    assert config["implementation"]["jupytext_source_created"] is True
    assert config["implementation"]["tests_created"] is True
    assert config["implementation"]["canonical_notebooks_placeholder_only"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["train_executed"] is True
    assert config["execution"]["run_flags"] == {"train": False}
    assert contract["cost"] == {
        "treatment_variants": 1,
        "lightgbm_configs": 3,
        "outer_folds": 5,
        "planned_gpu_downstream_boosters": 15,
        "planned_total_new_boosters": 15,
        "exp413_control_retraining_boosters": 0,
        "exp501_selector_retraining_boosters": 0,
        "exp413_signed_selector_retraining_boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
    }


def test_train_requires_package_run_and_train_approval(
    train: ModuleType,
    config: dict,
) -> None:
    with pytest.raises(RuntimeError, match="train remains disabled"):
        train.require_train_authorization(config)
    approved = copy.deepcopy(config)
    approved["execution"]["run_approved"] = True
    approved["execution"]["run_flags"]["train"] = True
    train.require_train_authorization(approved)


def test_parent_configs_and_bootstrap_dependencies_are_pinned(config: dict) -> None:
    parent_paths = {
        "exp413": ROOT
        / "experiments/exp413_scale5_likpf_full_replacement_on_exp335/config.yaml",
        "exp501": ROOT
        / "experiments/exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264/config.yaml",
    }
    for name, path in parent_paths.items():
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == config["data"]["parent_configs"][name]["sha256"]
    contract = (
        ROOT
        / "experiments/exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264/candidate_contract.yaml"
    )
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == (
        config["data"]["exp501_candidate_contract"]["sha256"]
    )
    dependencies = config["runtime"]["kaggle"]["bootstrap_dependency_files"]
    assert len(dependencies) == 13
    for item in dependencies:
        assert (ROOT / item["source"]).is_file(), item


def test_feature_surface_is_replacement_only_and_final373(train: ModuleType) -> None:
    base = [f"base_{index}" for index in range(273)]
    old = [f"selector_{index}" for index in range(74)]
    replacement = [*old, "selector_74", "selector_75", "selector_76"]
    signed = [f"signed_{index}" for index in range(23)]
    surface = train.build_feature_surface_contract(
        base_features=base,
        replacement_features=replacement,
        signed_features=signed,
        removed_features=old,
    )
    assert surface["feature_count"] == 373
    assert surface["old_selector_block_instances_in_final"] == 0
    assert surface["new_selector_block_instances_in_final"] == 1
    assert surface["old_feature_name_overlap_with_replacement"] == 74
    assert [item["source"] for item in surface["blocks"]] == [
        "exp413",
        "exp501",
        "exp413",
    ]
    assert surface["features"] == [*base, *replacement, *signed]


def test_final373_matrix_uses_exp501_values_in_replacement_slot(
    train: ModuleType,
) -> None:
    rows = 4
    base_features = [f"base_{index}" for index in range(273)]
    compact_features = [f"selector_{index}" for index in range(77)]
    signed_features = [f"signed_{index}" for index in range(23)]
    base = pd.DataFrame(
        {
            feature: np.full(rows, index, dtype=np.float32)
            for index, feature in enumerate(base_features)
        }
    )
    compact = pd.DataFrame(
        {
            feature: np.full(rows, 1000 + index, dtype=np.float32)
            for index, feature in enumerate(compact_features)
        }
    )
    signed = pd.DataFrame(
        {
            feature: np.full(rows, 2000 + index, dtype=np.float32)
            for index, feature in enumerate(signed_features)
        }
    )
    positions = np.array([3, 1, 0], dtype=np.int64)
    matrix = train.assemble_matrix(
        base=base,
        positions=positions,
        compact=compact.iloc[:3].reset_index(drop=True),
        signed=signed.iloc[:3].reset_index(drop=True),
        base_features=base_features,
        compact_features=compact_features,
        signed_features=signed_features,
        chunk_columns=32,
    )
    assert matrix.shape == (3, 373)
    np.testing.assert_array_equal(matrix[:, 273], np.full(3, 1000, np.float32))
    np.testing.assert_array_equal(matrix[:, 349], np.full(3, 1076, np.float32))
    np.testing.assert_array_equal(matrix[:, 350], np.full(3, 2000, np.float32))
    assert train.matrix_content_sha256(
        matrix, [*base_features, *compact_features, *signed_features]
    ) == train.matrix_content_sha256(
        matrix.copy(), [*base_features, *compact_features, *signed_features]
    )


def _fold_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0"],
            "well": ["a", "a", "b"],
            "last_known_tvt": [100.0, 100.0, 200.0],
        }
    )
    compact = pd.DataFrame(
        {
            "id": ["b_0", "a_0"],
            "well": ["b", "a"],
            "well_row_idx": [0, 0],
            "outer_fold": [1, 0],
            "md_since": [1.0, 1.0],
            "last_known_tvt": [200.0, 100.0],
        }
    )
    return base, compact, compact.copy()


def test_fold_alignment_checks_full_keys_and_anchor(train: ModuleType) -> None:
    base, compact, signed = _fold_frames()
    positions, evidence = train.validate_fold_alignment(
        base=base,
        compact=compact,
        signed=signed,
        downstream_outer_fold=2,
        role="train",
    )
    np.testing.assert_array_equal(positions, np.array([2, 0]))
    assert evidence["missing_base_rows"] == 0
    assert evidence["duplicate_ids"] == 0
    broken = signed.copy()
    broken.loc[0, "outer_fold"] = 4
    with pytest.raises(ValueError, match="key mismatch"):
        train.validate_fold_alignment(
            base=base,
            compact=compact,
            signed=broken,
            downstream_outer_fold=2,
            role="train",
        )


def test_gate_keeps_tail_report_only(train: ModuleType, config: dict, tmp_path: Path) -> None:
    rows = 100
    md_pattern = np.asarray([10, 300, 1100, 20, 500], dtype=np.float32)
    base = pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(rows)],
            "well": [f"well_{index}" for index in range(rows)],
            "last_known_tvt": np.zeros(rows, dtype=np.float32),
            "target": np.zeros(rows, dtype=np.float32),
            "md_since": np.resize(md_pattern, rows),
        }
    )
    parent_column = config["data"]["exp413_source"]["stage_d_prediction_column"]
    saved = base[["id", "well"]].copy()
    saved[parent_column] = np.float32(1.0)
    prediction = np.zeros(rows, dtype=np.float32)
    prediction[0] = np.float32(2.1)
    assignment = pd.DataFrame(
        {
            "well_id": base["well"],
            "verification_like_spatial_role": ["valid"] * rows,
            "verification_like_typewell_purged_role": ["valid"] * rows,
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    gate, *_ = train.evaluate_exp502_gate(
        config=config,
        base=base,
        saved_control=saved,
        oof_fold=np.arange(rows, dtype=np.int8) % 5,
        prediction=prediction,
        hidden_like_assignment_path=assignment_path,
        technical_checks={"synthetic": True},
    )
    assert gate["tail_readout"]["worst_well_delta_rmse"] > 1.0
    assert gate["tail_readout"]["policy"] == "report_only_not_automatic_stop"
    assert gate["passed"] is True


def test_jupytext_candidate_is_full_and_canonical_train_notebook_is_adopted() -> None:
    source = SOURCE.read_text()
    for chapter in range(1, 10):
        assert f"# ## {chapter}." in source
    assert source.count("# %% [markdown]") >= 10
    assert "Path(__file__)" not in source
    assert "verify_saved_feature_sources(" in source
    assert "build_replacement_clean273_surface(" in source
    assert "load_stage_d_compact_fold(" in source
    assert "load_signed_compact_fold(" in source
    assert "saved_exp413_control_retraining" in source
    assert "planned_boosters\": 15" in source
    assert "Inference executed: False" in source
    assert "Submission generated or submitted: False" in source
    canonical = json.loads((EXP_DIR / f"{EXP}_train.ipynb").read_text())
    canonical_source = "\n".join(
        "".join(cell.get("source", [])) for cell in canonical.get("cells", [])
    )
    assert "Notebook-first training entrypoint." not in canonical_source
    assert "run_exp502_train" in canonical_source
    assert "Path(__file__)" not in canonical_source
