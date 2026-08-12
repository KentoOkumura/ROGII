from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp351_exp306_l1_full_convergence_audit"
TRAIN_SOURCE = EXP_DIR / (
    "exp351_exp306_l1_full_convergence_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp351_exp306_l1_full_convergence_audit_compact_selfcontained_inference.py"
)
PARENT_TRAIN_SOURCE = (
    ROOT
    / "experiments"
    / "exp306_robust_rts_l1_convergence_calibration_audit"
    / "exp306_robust_rts_l1_convergence_calibration_audit_compact_selfcontained_train.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP351_IMPORT_ONLY")
    os.environ["EXP351_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP351_IMPORT_ONLY", None)
        else:
            os.environ["EXP351_IMPORT_ONLY"] = previous


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_frames(rows: int = 96) -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinate = np.arange(rows, dtype=float)
    known_rows = min(24, rows)
    horizontal_gr = 50.0 + 0.2 * coordinate + np.sin(coordinate / 7.0)
    horizontal_gr[sorted({0, min(25, rows - 1), rows - 1})] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": coordinate,
            "GR": horizontal_gr,
            "TVT_input": np.r_[
                100.0 + 0.1 * coordinate[:known_rows],
                [np.nan] * (rows - known_rows),
            ],
        }
    )
    typewell_tvt = np.linspace(80.0, 180.0, rows + 20)
    typewell_gr = 45.0 + 0.18 * typewell_tvt + np.sin(typewell_tvt / 8.0)
    typewell_gr[[5, 6]] = np.nan
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell


def prepared_wells(module, config: dict, count: int = 2):
    horizontal, typewell = synthetic_frames()
    return {
        f"well_{index}": module.prepare_gr_inputs(horizontal, typewell, config)
        for index in range(count)
    }


def synthetic_full_run(module, config: dict, count: int = 2):
    prepared = prepared_wells(module, config, count=count)
    input_frame = module.build_input_frame(prepared)
    output, status, elapsed = module.run_l1_full(prepared, config)
    return prepared, input_frame, output, status, elapsed


def configure_synthetic_parity(
    module,
    config: dict,
    input_frame: pd.DataFrame,
    output: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    wells = sorted(status["well_id"].astype(str).unique())
    sample = module.stable_parent_sample(wells, sample_wells=len(wells))
    config["audit"]["cross_run_parity"]["parent_sample_wells"] = len(wells)
    config["audit"]["cross_run_parity"]["parent_parity_wells"] = 1
    config["parent_anchor"]["stage0_input_content_sha256"] = (
        module.dataframe_content_sha(input_frame)
    )
    config["parent_anchor"]["stage0_l1_output_content_sha256"] = (
        module.dataframe_content_sha(output)
    )
    config["parent_anchor"]["stage0_l1_status_content_sha256"] = (
        module.dataframe_content_sha(status)
    )
    parity_well = sample.sort_values("sample_rank").iloc[0]["well_id"]
    parity = module.parity_content_hashes(
        output.loc[output["well_id"] == parity_well].copy(),
        status.loc[status["well_id"] == parity_well].copy(),
    )
    config["parent_anchor"]["parity_output_content_sha256"] = parity[
        "output_content_sha256"
    ]
    config["parent_anchor"]["parity_status_content_sha256"] = parity[
        "status_content_sha256"
    ]
    config["parent_anchor"]["parity_iteration_content_sha256"] = parity[
        "iteration_content_sha256"
    ]
    return sample


def write_synthetic_parent_artifacts(
    module,
    tmp_path: Path,
    config: dict,
    input_frame: pd.DataFrame,
    output: pd.DataFrame,
    status: pd.DataFrame,
    sample: pd.DataFrame,
) -> Path:
    parent_dir = tmp_path / "parent" / "artifacts"
    parent_dir.mkdir(parents=True)
    filenames = module.parent_artifact_filenames()

    contract = {
        "experiment": module.PARENT_EXPERIMENT,
        "truth_or_scientific_score_loaded": False,
    }
    contract_sha = module.mapping_sha256(contract)
    contract["scientific_contract_sha256"] = contract_sha
    contract_path = parent_dir / filenames["scientific_contract"]
    module.write_json(contract_path, contract)

    sample = sample.copy()
    sample["horizontal_raw_sha256"] = [
        f"horizontal-{well_id}" for well_id in sample["well_id"]
    ]
    sample["typewell_raw_sha256"] = [
        f"typewell-{well_id}" for well_id in sample["well_id"]
    ]
    sample_path = parent_dir / filenames["sample_manifest"]
    sample.to_csv(sample_path, index=False)

    input_artifact = module.write_csv_gzip(
        input_frame, parent_dir / filenames["stage0_input"]
    )
    output_artifact = module.write_csv_gzip(
        output, parent_dir / filenames["stage0_output"]
    )
    status_artifact = module.write_csv_gzip(
        status, parent_dir / filenames["stage0_status"]
    )
    raw_identity_sha = "synthetic-raw-identity"
    gate = {
        "truth_or_scientific_score_loaded": False,
        "full_audit_executed": False,
        "raw_well_identity_content_sha256": raw_identity_sha,
        "input_artifact": input_artifact,
        "output_artifact": output_artifact,
        "solver_status_artifact": status_artifact,
        "branches": {
            module.BRANCH_L1: {
                "output_content_sha256": module.dataframe_content_sha(output),
                "status_content_sha256": module.dataframe_content_sha(status),
                "full_eligible": True,
            }
        },
    }
    gate_path = parent_dir / filenames["stage0_gate"]
    module.write_json(gate_path, gate)
    summary = {
        "scientific_score": None,
        "submission": None,
        "full_eligible_branches": [module.BRANCH_L1],
    }
    summary_path = parent_dir / filenames["stage0_summary"]
    module.write_json(summary_path, summary)

    parity_count = int(config["audit"]["cross_run_parity"]["parent_parity_wells"])
    parity_wells = sample.sort_values("sample_rank")["well_id"].head(parity_count)
    parity_hashes = module.parity_content_hashes(
        output.loc[output["well_id"].isin(parity_wells)].copy(),
        status.loc[status["well_id"].isin(parity_wells)].copy(),
    )
    parity_manifest = {
        "branches": [
            {
                "branch": module.BRANCH_L1,
                "main": parity_hashes,
                "rerun": parity_hashes,
                "exact_identity": True,
            }
        ]
    }
    module.write_json(parent_dir / filenames["parity_manifest"], parity_manifest)

    config["data"]["expected_raw_well_identity_sha256"] = raw_identity_sha
    config["parent_anchor"]["scientific_contract_file_sha256"] = module.sha256_path(
        contract_path
    )
    config["parent_anchor"]["scientific_contract_content_sha256"] = contract_sha
    config["parent_anchor"]["stage0_gate_file_sha256"] = module.sha256_path(gate_path)
    config["parent_anchor"]["stage0_summary_file_sha256"] = module.sha256_path(
        summary_path
    )
    config["parent_anchor"]["sample_manifest_raw_sha256"] = module.sha256_path(
        sample_path
    )
    config["parent_anchor"]["sample_manifest_content_sha256"] = (
        module.dataframe_content_sha(sample)
    )
    return parent_dir


def test_contract_records_approval_but_fail_closes_after_completed_run() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_contract")
    config = load_config()
    module.validate_technical_contract(config)
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["run_full_l1"] is False
    with pytest.raises(RuntimeError, match="full L1 run flag is not enabled"):
        module.validate_technical_contract(config, require_run_approval=True)

    not_approved = deepcopy(config)
    not_approved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        module.validate_technical_contract(not_approved, require_run_approval=True)

    approved_snapshot = deepcopy(config)
    approved_snapshot["execution"]["run_full_l1"] = True
    module.validate_technical_contract(approved_snapshot, require_run_approval=True)

    illegal = deepcopy(config)
    illegal["execution"]["run_scientific_score"] = True
    with pytest.raises(RuntimeError, match="fail-closed"):
        module.validate_technical_contract(illegal)


def test_horizontal_reader_uses_allowlist_and_guard_rejects_truth(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp351_schema")
    config = load_config()
    horizontal, _ = synthetic_frames(rows=12)
    raw = horizontal.assign(TVT=np.arange(12), error=1.0, formation="x")
    path = tmp_path / "well__horizontal_well.csv"
    raw.to_csv(path, index=False)
    safe = module.load_horizontal_target_free(path, config)
    assert list(safe.columns) == ["MD", "GR", "TVT_input"]
    assert not {"TVT", "error", "formation"}.intersection(safe.columns)
    with pytest.raises(ValueError, match="forbidden columns"):
        module.validate_horizontal_target_free_frame(raw, config)


def test_common_preparation_is_parent_compatible_and_deterministic() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_prepare")
    config = load_config()
    horizontal, typewell = synthetic_frames()
    first = module.prepare_gr_inputs(horizontal, typewell, config)
    second = module.prepare_gr_inputs(horizontal, typewell, config)
    for series_kind in module.SERIES_KINDS:
        np.testing.assert_array_equal(
            first[f"{series_kind}_coordinate"],
            second[f"{series_kind}_coordinate"],
        )
        np.testing.assert_array_equal(
            first[f"{series_kind}_gr"],
            second[f"{series_kind}_gr"],
        )
        assert np.isfinite(first[f"{series_kind}_gr"]).all()
        assert np.all(np.diff(first[f"{series_kind}_coordinate"]) >= 0.0)


def test_frozen_l1_solver_is_exactly_deterministic() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_solver")
    config = load_config()
    spec = module.l1_spec(config)
    coordinate = np.arange(128, dtype=float)
    observed = 20.0 + 0.1 * coordinate + np.sin(coordinate / 5.0)
    observed[60] += 12.0
    first, first_status = module.l1_trend_smooth(observed, spec)
    second, second_status = module.l1_trend_smooth(observed, spec)
    np.testing.assert_array_equal(first, second)
    assert first_status == second_status
    assert first_status["iterations"] <= 2000
    assert spec["rho"] == 1.0
    assert spec["absolute_tolerance"] == 1.0e-4
    assert spec["relative_tolerance"] == 1.0e-4
    assert not hasattr(module, "robust_rts_smooth")


def test_preparation_and_l1_output_are_exactly_parent_compatible() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_parent_compat")
    parent = load_module(PARENT_TRAIN_SOURCE, "exp306_parent_compat")
    config = load_config()
    parent_config = yaml.safe_load(
        (
            PARENT_TRAIN_SOURCE.parent / "config.yaml"
        ).read_text()
    )
    assert isinstance(parent_config, dict)
    horizontal, typewell = synthetic_frames(rows=128)
    child_prepared = module.prepare_gr_inputs(horizontal, typewell, config)
    parent_prepared = parent.prepare_gr_inputs(
        horizontal, typewell, parent_config
    )
    for key in child_prepared:
        np.testing.assert_array_equal(child_prepared[key], parent_prepared[key])

    child_output, child_status = module.l1_trend_smooth(
        child_prepared["horizontal_gr"], module.l1_spec(config)
    )
    parent_output, parent_status = parent.l1_trend_smooth(
        parent_prepared["horizontal_gr"],
        parent.branch_spec(parent_config, parent.BRANCH_L1),
    )
    np.testing.assert_array_equal(child_output, parent_output)
    assert child_status == parent_status


def test_parent_anchor_guard_accepts_exact_files_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp351_parent_anchor")
    config = load_config()
    _, input_frame, output, status, _ = synthetic_full_run(module, config)
    sample = configure_synthetic_parity(
        module, config, input_frame, output, status
    )
    parent_dir = write_synthetic_parent_artifacts(
        module,
        tmp_path,
        config,
        input_frame,
        output,
        status,
        sample,
    )
    manifest, loaded_sample = module.validate_parent_anchors(parent_dir, config)
    assert manifest["all_parent_anchors_match"]
    assert len(loaded_sample) == 2

    contract_path = (
        parent_dir / module.parent_artifact_filenames()["scientific_contract"]
    )
    contract_path.write_text(contract_path.read_text() + "\n")
    with pytest.raises(ValueError, match="file SHA mismatch"):
        module.validate_parent_anchors(parent_dir, config)


def test_parent_sample_reconstruction_detects_raw_identity_mutation() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_parent_sample")
    config = load_config()
    wells = ["well_a", "well_b", "well_c"]
    raw_identity = pd.DataFrame(
        {
            "well_id": wells,
            "horizontal_raw_sha256": [f"h-{well}" for well in wells],
            "typewell_raw_sha256": [f"t-{well}" for well in wells],
        }
    )
    config["audit"]["cross_run_parity"]["parent_sample_wells"] = 2
    parent_sample = module.stable_parent_sample(
        wells, sample_wells=2
    ).merge(raw_identity, on="well_id", validate="one_to_one")
    config["parent_anchor"]["sample_manifest_content_sha256"] = (
        module.dataframe_content_sha(parent_sample)
    )
    result = module.validate_parent_sample_against_raw(
        parent_sample, raw_identity, config
    )
    assert result["sample_manifest_exact_frame"]

    mutated = raw_identity.copy()
    mutated.loc[0, "horizontal_raw_sha256"] = "mutated"
    with pytest.raises(ValueError, match="does not reconstruct"):
        module.validate_parent_sample_against_raw(
            parent_sample, mutated, config
        )


def test_cross_run_parity_requires_exact_64_and_8_well_style_hashes() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_parity")
    config = load_config()
    _, input_frame, output, status, _ = synthetic_full_run(module, config)
    sample = configure_synthetic_parity(
        module, config, input_frame, output, status
    )
    parity = module.evaluate_cross_run_parity(
        input_frame, output, status, sample, config
    )
    assert parity["sample_exact_identity"]
    assert parity["parity_exact_identity"]
    assert parity["all_cross_run_parity_passed"]

    mutated = output.copy()
    parity_well = parity["parent_parity_wells"][0]
    row = mutated.index[mutated["well_id"] == parity_well][0]
    mutated.loc[row, "output_gr"] += 1.0e-6
    failed = module.evaluate_cross_run_parity(
        input_frame, mutated, status, sample, config
    )
    assert not failed["parity_exact_identity"]
    assert not failed["all_cross_run_parity_passed"]


def test_full_gate_checks_coverage_failure_runtime_and_parity() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_gate")
    config = load_config()
    _, input_frame, output, status, _ = synthetic_full_run(module, config)
    sample = configure_synthetic_parity(
        module, config, input_frame, output, status
    )
    config["audit"]["full"]["expected_wells"] = 2
    config["audit"]["full"]["expected_series"] = 4
    config["audit"]["full"]["runtime_limit_seconds"] = 10.0
    config["data"]["expected_raw_well_identity_sha256"] = "synthetic-raw"
    parity = module.evaluate_cross_run_parity(
        input_frame, output, status, sample, config
    )
    gate = module.evaluate_full_gate(
        input_frame,
        output,
        status,
        raw_identity_sha256="synthetic-raw",
        parent_anchors_match=True,
        cross_run_parity=parity,
        audit_elapsed_seconds=1.0,
        config=config,
    )
    assert gate["full_technical_pass"]
    assert gate["converged_series"] == 4

    broken_status = status.copy()
    broken_status.loc[0, ["converged", "technical_pass"]] = False
    broken_status.loc[0, "error"] = "synthetic solver failure"
    broken = module.evaluate_full_gate(
        input_frame,
        output,
        broken_status,
        raw_identity_sha256="synthetic-raw",
        parent_anchors_match=True,
        cross_run_parity=parity,
        audit_elapsed_seconds=1.0,
        config=config,
    )
    assert not broken["full_technical_pass"]
    assert broken["status"] == "full_technical_fail_closed"
    assert broken["error_count"] == 1

    slow = module.evaluate_full_gate(
        input_frame,
        output,
        status,
        raw_identity_sha256="synthetic-raw",
        parent_anchors_match=True,
        cross_run_parity=parity,
        audit_elapsed_seconds=11.0,
        config=config,
    )
    assert not slow["criteria"]["runtime_within_limit"]
    assert not slow["full_technical_pass"]


def test_generated_contract_contains_no_scientific_or_inference_path() -> None:
    module = load_module(TRAIN_SOURCE, "exp351_generated_contract")
    contract = module.build_technical_contract(load_config())
    assert contract["truth_or_scientific_score_loaded"] is False
    assert contract["prediction_created"] is False
    assert contract["submission_created"] is False
    assert contract["full_audit"]["series_runs"] == 1546
    assert "scientific_score" in contract["forbidden"]
    assert "RTS" in contract["forbidden"]
    assert "submission" in contract["forbidden"]


def test_inference_contract_remains_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp351_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["run_inference"]
    assert not contract["execution_create_submission"]
    config["execution"]["run_scientific_score"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)
