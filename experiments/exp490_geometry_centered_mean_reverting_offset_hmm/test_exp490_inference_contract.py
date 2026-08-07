from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
SOURCE = EXP_DIR / (
    "exp490_geometry_centered_mean_reverting_offset_hmm_compact_selfcontained_inference.py"
)
CONFIG = EXP_DIR / "config.yaml"
EXPECTED_SCIENTIFIC_SHA = "6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35"


def load_module():
    spec = importlib.util.spec_from_file_location("exp490_inference_contract", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_inference_execution_contract_is_exact_and_submit_is_disabled() -> None:
    module = load_module()
    contract = module.validate_inference_execution(
        load_config(), require_run_approval=False
    )
    assert contract["candidate_hmm_well_runs"] == "runtime_sample_well_count"
    assert contract["exp226_test_geometry_well_runs"] == "runtime_sample_well_count"
    assert contract["model_configs"] == 0
    assert load_config()["execution"]["competition_submission_approved"] is False


def test_hidden_dynamic_v2_run_requires_separate_approval() -> None:
    module = load_module()
    config = load_config()
    config["execution"]["hidden_dynamic_v2_run_approved"] = True
    module.validate_inference_execution(config)
    config["execution"]["hidden_dynamic_v2_run_approved"] = False
    with np.testing.assert_raises_regex(
        ValueError, "hidden_dynamic_v2_run_approved"
    ):
        module.validate_inference_execution(config)


def test_oof_scientific_contract_sha_is_preserved() -> None:
    module = load_module()
    contract = module.build_frozen_oof_scientific_contract(load_config())
    observed = hashlib.sha256(module.stable_json_bytes(contract)).hexdigest()
    assert observed == EXPECTED_SCIENTIFIC_SHA


def test_sample_contract_and_exp226_source_pins() -> None:
    module = load_module()
    config = load_config()
    sample = pd.read_csv(ROOT / config["data"]["sample_submission"], dtype={"id": str})
    reference = config["data"]["public_test_reference"]
    assert len(sample) == reference["submission_rows"]
    wells = sample["id"].str.rsplit("_", n=1).str[0]
    assert wells.nunique() == reference["test_wells"]
    assert config["data"]["hidden_test_runtime_contract"]["sample_sha_policy"] == "record_only"
    _, runtime_wells, runtime_contract = module.validate_runtime_test_inventory(
        sample, ROOT / config["data"]["test_dir"]
    )
    assert len(runtime_wells) == reference["test_wells"]
    assert runtime_contract["rows"] == reference["submission_rows"]
    source = (
        ROOT
        / "experiments"
        / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
        / "connortynan_k16_reproduction.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["data"][
        "exp226_inference_source"
    ]["source_sha256"]


def test_runtime_inventory_accepts_variable_sample_size(tmp_path: Path) -> None:
    module = load_module()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    ids: list[str] = []
    for well, total_rows, unknown_start in (("well_a", 7, 3), ("well_b", 9, 5)):
        tvt_input = np.arange(total_rows, dtype=float)
        tvt_input[unknown_start:] = np.nan
        pd.DataFrame({"TVT_input": tvt_input}).to_csv(
            test_dir / f"{well}__horizontal_well.csv", index=False
        )
        pd.DataFrame({"TVT": [1.0], "GR": [2.0]}).to_csv(
            test_dir / f"{well}__typewell.csv", index=False
        )
        ids.extend(f"{well}_{row}" for row in range(unknown_start, total_rows))
    sample = pd.DataFrame({"id": ids, "tvt": np.zeros(len(ids))})
    identity, wells, contract = module.validate_runtime_test_inventory(sample, test_dir)
    assert len(identity) == 8
    assert wells == ["well_a", "well_b"]
    assert contract["rows"] == 8
    assert contract["wells"] == 2
    assert contract["sample_raw_identity_exact"] is True


def test_runtime_inventory_rejects_sample_raw_row_mismatch(tmp_path: Path) -> None:
    module = load_module()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    pd.DataFrame({"TVT_input": [1.0, np.nan, np.nan]}).to_csv(
        test_dir / "well_a__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": [1.0], "GR": [2.0]}).to_csv(
        test_dir / "well_a__typewell.csv", index=False
    )
    sample = pd.DataFrame({"id": ["well_a_1"], "tvt": [0.0]})
    with np.testing.assert_raises_regex(ValueError, "sample ids and raw unknown-suffix rows differ"):
        module.validate_runtime_test_inventory(sample, test_dir)


def test_k16_half_life_and_zero_state_identity() -> None:
    module = load_module()
    md = np.cumsum(np.linspace(0.4, 1.4, 160)) + 1000.0
    segment = module.k16_segment_half_life(md, last_known_md=999.5, segment_count=16)
    assert np.allclose(segment["segment_cumulative_rho"], 0.5, atol=1.0e-12)
    assert module.zero_state_geometry_identity(segment["dmd"], segment["rho"])["pass"]


def test_inference_source_is_not_placeholder_or_truth_reader() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "shutil.copyfile(sample_submission" not in text
    assert "PredictionResult.geop" in text
    assert "competition_submission_approved" in text
    assert "sample_submission SHA changed" not in text
    assert "sample row/well count changed" not in text
    assert "full_mounted_test_inventory_before_exp226_fit" in text
    assert "__file__" not in text
    assert "pd.read_csv(horizontal_path, usecols=lambda column: str(column) != \"TVT\")" in text
    assert json.loads(json.dumps({"source": SOURCE.name}))["source"] == SOURCE.name
