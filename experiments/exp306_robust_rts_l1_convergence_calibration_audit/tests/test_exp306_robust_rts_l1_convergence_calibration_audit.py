from __future__ import annotations

import hashlib
import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp306_robust_rts_l1_convergence_calibration_audit"
TRAIN_SOURCE = EXP_DIR / (
    "exp306_robust_rts_l1_convergence_calibration_audit_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp306_robust_rts_l1_convergence_calibration_audit_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP306_IMPORT_ONLY")
    os.environ["EXP306_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP306_IMPORT_ONLY", None)
        else:
            os.environ["EXP306_IMPORT_ONLY"] = previous


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


def test_contract_approves_only_stage0_execution_shape() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_contract")
    config = load_config()
    module.validate_technical_contract(config)
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["run_full_rts"] is False
    assert config["execution"]["run_full_l1"] is False
    approved = deepcopy(config)
    approved["execution"]["kaggle_push_approved"] = True
    approved["execution"]["run_stage0"] = True
    module.validate_technical_contract(approved, require_run_approval=True)

    not_approved = deepcopy(config)
    not_approved["execution"]["kaggle_push_approved"] = False
    not_approved["execution"]["run_stage0"] = True
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        module.validate_technical_contract(not_approved, require_run_approval=True)

    not_enabled = deepcopy(config)
    not_enabled["execution"]["kaggle_push_approved"] = True
    not_enabled["execution"]["run_stage0"] = False
    with pytest.raises(RuntimeError, match="Stage 0 run flag is not enabled"):
        module.validate_technical_contract(not_enabled, require_run_approval=True)

    illegal = deepcopy(config)
    illegal["execution"]["run_full_rts"] = True
    with pytest.raises(RuntimeError, match="fail-closed"):
        module.validate_technical_contract(illegal)


def test_horizontal_reader_loads_only_allowlist_and_frame_guard_rejects_truth(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp306_schema")
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


def test_common_preparation_is_finite_ordered_and_deterministic() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_prepare")
    config = load_config()
    horizontal, typewell = synthetic_frames()
    first = module.prepare_gr_inputs(horizontal, typewell, config)
    second = module.prepare_gr_inputs(horizontal, typewell, config)
    for series_kind in module.SERIES_KINDS:
        np.testing.assert_array_equal(
            first[f"{series_kind}_coordinate"], second[f"{series_kind}_coordinate"]
        )
        np.testing.assert_array_equal(
            first[f"{series_kind}_gr"], second[f"{series_kind}_gr"]
        )
        assert np.isfinite(first[f"{series_kind}_gr"]).all()
        assert np.all(np.diff(first[f"{series_kind}_coordinate"]) >= 0.0)


def test_stable_stage0_sample_uses_exact_salted_sha_order() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_sample")
    wells = ["well_c", "well_a", "well_b", "well_d"]
    first = module.stable_stage0_sample(
        wells, salt="exp306-stage0-v1", sample_wells=3
    )
    second = module.stable_stage0_sample(
        reversed(wells), salt="exp306-stage0-v1", sample_wells=3
    )
    pd.testing.assert_frame_equal(first, second)
    expected = sorted(
        (
            hashlib.sha256(f"exp306-stage0-v1|{well_id}".encode()).hexdigest(),
            well_id,
        )
        for well_id in wells
    )[:3]
    assert first[["sample_sha256", "well_id"]].apply(tuple, axis=1).tolist() == expected


def test_rts_and_l1_fixed_solvers_are_exactly_deterministic() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_solvers")
    config = load_config()
    coordinate = np.arange(128, dtype=float)
    clean = 20.0 + 0.1 * coordinate + np.sin(coordinate / 5.0)
    observed = clean.copy()
    observed[60] += 12.0

    rts_spec = module.branch_spec(config, module.BRANCH_RTS_A)
    first, first_variance, first_status = module.robust_rts_smooth(
        observed, coordinate, rts_spec
    )
    second, second_variance, second_status = module.robust_rts_smooth(
        observed, coordinate, rts_spec
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_variance, second_variance)
    assert first_status == second_status
    assert first_status["iterations"] <= 32
    assert np.isfinite(first).all() and np.isfinite(first_variance).all()

    l1_spec = module.branch_spec(config, module.BRANCH_L1)
    l1_first, l1_first_status = module.l1_trend_smooth(observed, l1_spec)
    l1_second, l1_second_status = module.l1_trend_smooth(observed, l1_spec)
    np.testing.assert_array_equal(l1_first, l1_second)
    assert l1_first_status == l1_second_status
    assert l1_first_status["iterations"] <= 2000
    assert np.isfinite(l1_first).all()


def test_rts_b_runs_only_after_any_rts_a_technical_failure(monkeypatch) -> None:
    module = load_module(TRAIN_SOURCE, "exp306_ladder")
    config = load_config()
    prepared = {"well_a": {}}
    calls: list[str] = []

    def fake_run_branch(_prepared, branch, _config):
        calls.append(branch)
        passed = branch != module.BRANCH_RTS_A
        status = pd.DataFrame(
            {
                "well_id": ["well_a", "well_a"],
                "series_kind": ["horizontal", "typewell"],
                "technical_pass": [passed, passed],
            }
        )
        return pd.DataFrame(), status, 0.1

    monkeypatch.setattr(module, "run_branch", fake_run_branch)
    results = module.run_stage0_core(prepared, config)
    assert calls == [module.BRANCH_RTS_A, module.BRANCH_L1, module.BRANCH_RTS_B]
    assert set(results) == {
        module.BRANCH_RTS_A,
        module.BRANCH_RTS_B,
        module.BRANCH_L1,
    }

    calls.clear()

    def all_pass(_prepared, branch, _config):
        calls.append(branch)
        status = pd.DataFrame(
            {
                "well_id": ["well_a", "well_a"],
                "series_kind": ["horizontal", "typewell"],
                "technical_pass": [True, True],
            }
        )
        return pd.DataFrame(), status, 0.1

    monkeypatch.setattr(module, "run_branch", all_pass)
    results = module.run_stage0_core(prepared, config)
    assert calls == [module.BRANCH_RTS_A, module.BRANCH_L1]
    assert module.BRANCH_RTS_B not in results


def test_branch_gate_checks_coverage_order_fallback_and_runtime() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_gate")
    config = load_config()
    config["audit"]["stage0"]["sample_wells"] = 2
    config["audit"]["stage0"]["expected_series_per_candidate"] = 4
    config["validation"]["expected_wells"] = 10
    config["audit"]["stage0"]["runtime_projection_limit_seconds"] = 6.0
    prepared = prepared_wells(module, config)
    input_frame = module.build_input_frame(prepared)
    output, status, _ = module.run_branch(prepared, module.BRANCH_L1, config)
    gate = module.evaluate_branch_gate(
        module.BRANCH_L1,
        input_frame,
        output,
        status,
        stage0_elapsed_seconds=1.0,
        config=config,
    )
    assert gate["projected_full_runtime_seconds"] == pytest.approx(5.0)
    assert gate["technical_passed"]
    assert gate["provisional_full_eligible"]

    broken = output.copy()
    broken.loc[0, "coordinate"] += 1.0
    broken_gate = module.evaluate_branch_gate(
        module.BRANCH_L1,
        input_frame,
        broken,
        status,
        stage0_elapsed_seconds=1.0,
        config=config,
    )
    assert not broken_gate["criteria"]["length_order_identity"]
    assert not broken_gate["technical_passed"]

    slow_gate = module.evaluate_branch_gate(
        module.BRANCH_L1,
        input_frame,
        output,
        status,
        stage0_elapsed_seconds=2.0,
        config=config,
    )
    assert slow_gate["technical_passed"]
    assert not slow_gate["provisional_full_eligible"]


def test_parity_requires_exact_output_status_and_iteration_hashes() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_parity")
    config = load_config()
    config["audit"]["stage0"]["deterministic_parity_wells"] = 1
    prepared = dict(reversed(list(prepared_wells(module, config).items())))
    output, status, _ = module.run_branch(prepared, module.BRANCH_L1, config)
    parity = module.run_parity_audit(
        module.BRANCH_L1, prepared, output, status, config
    )
    assert parity["series_runs"] == 2
    assert parity["exact_identity"]
    assert parity["wells"] == [list(prepared)[0]]
    assert set(parity["main"]) == {
        "output_content_sha256",
        "status_content_sha256",
        "iteration_content_sha256",
    }

    mutated = output.copy()
    parity_well = list(prepared)[0]
    row = mutated.index[mutated["well_id"] == parity_well][0]
    mutated.loc[row, "output_gr"] += 1.0e-6
    failed = module.run_parity_audit(
        module.BRANCH_L1, prepared, mutated, status, config
    )
    assert not failed["exact_identity"]


def test_generated_contract_contains_no_scientific_selection_path() -> None:
    module = load_module(TRAIN_SOURCE, "exp306_generated_contract")
    contract = module.build_technical_contract(load_config())
    assert contract["truth_or_scientific_score_loaded"] is False
    assert contract["branches"][module.BRANCH_RTS_B][
        "conditional_on_any_rts_a_technical_failure"
    ]
    assert "scientific_contract_sha256" in contract
    assert "truth" in contract["forbidden"]
    assert "submission" in contract["forbidden"]


def test_inference_contract_remains_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp306_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["run_inference"]
    assert not contract["execution_create_submission"]
    config["execution"]["run_scientific_score"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)
