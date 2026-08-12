from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.strict_public_core import validate_stage_i_visible_parity


ROOT = Path(__file__).resolve().parents[3]
EXP = "exp509_exp413_strict_public_core_final_slot"
EXP_DIR = ROOT / "experiments" / EXP
CONFIG_PATH = EXP_DIR / "config.yaml"
CANDIDATE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
CANDIDATE_NOTEBOOK = EXP_DIR / f"{EXP}_compact_selfcontained_inference.ipynb"
CANONICAL_NOTEBOOK = EXP_DIR / f"{EXP}_inference.ipynb"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_freezes_reference_override_and_zero_training() -> None:
    config = load_config()
    implementation = config["implementation"]
    ensemble = config["ensemble"]
    execution = config["execution_contract"]
    deployment = config["inference"]["saved_model_deployment"]
    stage = config["stages"]["stage_i_saved_model_inference"]

    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["inference_enabled"] is False
    assert implementation["approved"] is True
    assert implementation["compact_inference_candidate_created"] is True
    assert implementation["canonical_inference_notebook_adopted"] is False
    assert ensemble["exp413_weight"] + ensemble["strict_public_core_weight"] == 1.0
    assert ensemble["strict_public_core_weight"] == pytest.approx(
        np.median(ensemble["meta_fold_public_core_weights"]), abs=1e-15
    )
    assert ensemble["weight_refit"] is False
    assert ensemble["row_gate"] is False
    assert ensemble["well_gate"] is False
    assert ensemble["conditional_router"] is False
    assert ensemble["final_postprocess"] == "none"
    assert execution["trained_models"] == 0
    assert execution["total_boosters"] == 0
    assert execution["parent_or_control_retraining"] == 0
    assert stage["fitted_boosters"] == 0
    assert stage["loaded_exp497_boosters"] == 40
    assert stage["loaded_exp497_ridge_models"] == 2
    assert stage["loaded_exp413_boosters"] == 75
    assert stage["weight_refit"] == 0
    assert deployment["external_submission"] is False
    assert deployment["public_test_exp413_sidecar_allowed"] is False


def test_known_exp497_v1_drift_passes_only_component_specific_contract() -> None:
    config = load_config()
    deployment = config["inference"]["saved_model_deployment"]
    observed = validate_stage_i_visible_parity(
        strict_public_core_max_abs=0.0012812500008294592,
        blend_max_abs=0.01419531250030559,
        strict_public_core_tolerance=deployment["visible_strict_public_core_max_abs"],
        blend_tolerance=deployment["visible_blend_max_abs"],
        exp413_max_abs=0.016,
    )
    assert observed["passed"] is True
    assert observed["exp413_max_abs"] <= deployment["visible_exp413_max_abs"]
    with pytest.raises(ValueError, match="visible saved-model parity failed"):
        validate_stage_i_visible_parity(
            strict_public_core_max_abs=0.0012812500008294592,
            blend_max_abs=0.01419531250030559,
            strict_public_core_tolerance=0.001,
            blend_tolerance=0.001,
            exp413_max_abs=0.016,
        )


def test_candidate_is_dynamic_saved_model_only_and_fail_closed() -> None:
    source = CANDIDATE_SOURCE.read_text()
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "generate_dynamic_exp413_prediction" in called_names
    assert "build_stage_i_test_feature_frame" in called_names
    assert "run_stage_i_saved_model_inference" in called_names
    assert "isolate_exp413_intermediate_submission" in called_names
    assert "prediction_float64_sha256" in called_names
    assert "fit" not in called_names
    assert "train" not in called_names
    assert "exp413_current_test_predictions.csv.gz" not in source
    assert "kaggle competitions submit" not in source.lower()
    assert "gold_visible_prefix" not in source.lower()
    assert "guarded_contact" not in source.lower()
    assert "same_well_train_test" not in source.lower()
    assert "__file__" not in source
    assert "np.random" not in source
    assert 'CORE_OUTPUT_DIR / "exp497_intermediate_submission.csv"' in source
    assert 'OUTPUT_DIR / "exp413_intermediate_submission.csv"' in source
    assert 'KAGGLE_WORKING / "submission.csv"' in source


def test_candidate_has_exact_float64_formula_and_required_outputs() -> None:
    source = CANDIDATE_SOURCE.read_text()
    config = load_config()
    weight = config["ensemble"]["strict_public_core_weight"]
    exp413_weight = config["ensemble"]["exp413_weight"]
    exp413 = np.array([10_000.125, 12_000.25, 15_000.5], dtype=np.float64)
    strict = np.array([10_010.5, 11_980.0, 15_100.75], dtype=np.float64)
    expected = exp413_weight * exp413 + weight * strict

    assert expected.dtype == np.float64
    assert np.isfinite(expected).all()
    assert (
        "final_values = EXP413_WEIGHT * exp413_values "
        "+ STRICT_PUBLIC_CORE_WEIGHT * strict_values"
    ) in source
    assert '"exp509_component_predictions.csv.gz"' in source
    assert '"exp509_prediction_difference_summary.json"' in source
    assert '"exp509_input_manifest.json"' in source
    assert '"exp509_reproducibility_manifest.json"' in source
    assert 'float_format="%.17g"' in source
    assert '"scientific_promotion": False' in source


def test_candidate_notebook_has_readable_eight_section_structure() -> None:
    notebook = json.loads(CANDIDATE_NOTEBOOK.read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    combined = "\n".join(sources)

    assert len(notebook["cells"]) >= 17
    for heading in (
        "## 1. Imports and runtime helpers",
        "## 2. Authorization and zero-training inventory",
        "## 3. Saved-model and dynamic input contracts",
        "## 4. Dynamic hidden-safe exp413 generation",
        "## 5. Dynamic strict public-core feature generation",
        "## 6. Saved-model component inference",
        "## 7. Fixed float64 final-slot blend",
        "## 8. Technical audit and reproducibility outputs",
    ):
        assert heading in combined
    assert "generate_dynamic_exp413_prediction()" in combined
    assert "run_stage_i_saved_model_inference(" in combined
    assert "final_values = EXP413_WEIGHT * exp413_values" in combined


def test_canonical_notebook_remains_unmodified_placeholder() -> None:
    notebook = json.loads(CANONICAL_NOTEBOOK.read_text())
    combined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Notebook-first inference entrypoint" in combined
    assert "shutil.copyfile(sample_submission, submission_path)" in combined
    assert "run_stage_i_saved_model_inference(" not in combined
