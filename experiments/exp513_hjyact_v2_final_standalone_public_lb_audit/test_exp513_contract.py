from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATE = (
    HERE
    / "exp513_hjyact_v2_final_standalone_public_lb_audit_"
    "compact_selfcontained_inference.py"
)
GENERATOR = ROOT / "scripts/prepare_exp513_hjyact_v2_standalone_candidate.py"
PARENT_GENERATOR = ROOT / "scripts/prepare_exp512_hjyact_v2_candidate.py"


def read_yaml(name: str) -> dict:
    value = yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def candidate_text() -> str:
    return CANDIDATE.read_text(encoding="utf-8")


def test_candidate_is_parseable_standalone_and_route_is_frozen() -> None:
    source = candidate_text()
    ast.parse(source)
    config = read_yaml("config.yaml")
    contract = read_yaml("standalone_contract.yaml")
    assert config["experiment"]["route"] == "ensemble"
    assert contract["component"]["name"] == "hjyact_v2_final"
    assert contract["final_output"]["operation"] == "identity"
    assert contract["final_output"]["downstream_blend"] is False
    assert "generate_dynamic_exp413_prediction" not in source
    assert "CANDIDATE_REUSE_TRACKER" not in source
    assert "fixed_blend" not in source
    assert "exp413_component_submission" not in source
    assert "Stand-alone" not in source
    assert '"standalone_component": "hjyact_v2_final"' in source


def test_source_identity_and_complete_final_boundary_are_pinned() -> None:
    source = candidate_text()
    assert (
        'SOURCE_PULL_NOTEBOOK_SHA256 = '
        '"4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c"'
        in source
    )
    assert (
        'SOURCE_CODE_CELL_SHA256 = '
        '"ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088"'
        in source
    )
    assert "PF seed-branch hedge:" in source
    assert "Final submission audit: verify the final file after all enabled correction layers" in source
    standalone_marker = "# ## 7. Standalone submission and reproducibility outputs"
    assert source.index("PF seed-branch hedge:") < source.index(standalone_marker)
    assert "_find_precomputed_learned_submission" not in source
    assert "inference-time training is forbidden" in source


def test_exp512_mount_failures_are_fixed_at_both_boundaries() -> None:
    source = candidate_text()
    assert "def resolve_competition_data_root()" in source
    assert 'Path("/kaggle/input/rogii-wellbore-geology-prediction")' in source
    assert (
        "COMPETITION_DATA_ROOT = '/kaggle/input/competitions/"
        "rogii-wellbore-geology-prediction'"
        not in source
    )
    assert (
        "RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/"
        "wellbore-geology-prediction-artifacts'"
        not in source
    )
    assert 'RIDGE_ARTIFACT_ROOT = HJYACT_INPUT_AUDIT["roots"]["ridge"]' in source
    assert 'RIDGE_ARTIFACT_ROOT = str(HJYACT_INPUT_AUDIT["roots"]["ridge"])' in source
    assert '_ridge_train_path = CFG.artifacts_path / "data" / "train.csv"' in source
    assert "input SHA mismatch:" in source
    assert source.index("HJYACT_INPUT_AUDIT = verify_hjyact_inputs()") < source.index(
        '_ridge_train_path = CFG.artifacts_path / "data" / "train.csv"'
    )


def test_dynamic_sample_precedes_posthoc_visible_sha_gate() -> None:
    source = candidate_text()
    dynamic_read = (
        'sample = pd.read_csv(Path(COMPETITION_DATA_ROOT) / '
        '"sample_submission.csv", dtype={"id": str})'
    )
    identity = '"sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256'
    parity_gate = 'if visible_reference_checks["sample_id_order_match"]:'
    assert dynamic_read in source
    assert identity in source
    assert parity_gate in source
    assert source.index(dynamic_read) < source.index(identity) < source.index(parity_gate)
    assert "14151" not in source
    assert "visible_well_ids" not in source
    assert "static visible prediction" in source
    assert "kaggle competitions submit" not in source


def test_model_inventory_excludes_parent_models_and_training() -> None:
    config = read_yaml("config.yaml")
    manifest = read_yaml("model_manifest.yaml")
    assert manifest["new_booster_training_count"] == 0
    assert manifest["parent_control_retraining_count"] == 0
    assert manifest["runtime_fit"]["total_fits"] == 5
    assert manifest["saved_model_files"]["total"] == 13
    assert manifest["saved_model_files"]["contained_estimators_total"] == 33
    assert len(manifest["saved_model_files"]["trainer_wrappers"]) == 5
    assert len(manifest["saved_model_files"]["learned_trajectory"]) == 3
    assert len(manifest["saved_model_files"]["model_package"]) == 5
    assert manifest["excluded_model_files"]["exp413"] == 75
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is False
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["output_archive_download_approved"] is True
    assert config["authorization"]["competition_submission_approved"] is False


def test_generator_and_candidate_hashes_are_frozen() -> None:
    config = read_yaml("config.yaml")
    generator_text = GENERATOR.read_text(encoding="utf-8")
    expected_parent = re.search(
        r'EXPECTED_PARENT_GENERATOR_SHA256 = \(\n\s+"([0-9a-f]{64})"',
        generator_text,
    )
    assert expected_parent is not None
    assert (
        expected_parent.group(1)
        == config["implementation"]["parent_generator_sha256"]
    )
    generator_sha = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    assert f'GENERATOR_SHA256 = "{generator_sha}"' in candidate_text()
    candidate_sha = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert config["implementation"]["generator_sha256"] == generator_sha
    assert config["implementation"]["candidate_source_sha256"] == candidate_sha
    assert config["implementation"]["candidate_source_lines"] == len(
        candidate_text().splitlines()
    )


def test_notebook_safe_paths_and_compact_size_guard() -> None:
    source = candidate_text()
    assert "Path(__file__)" not in source
    assert CANDIDATE.stat().st_size < 400_000
    assert source.count("# %% [markdown]") >= 9
    assert "# ## Contents" in source
    assert "# ## 7. Standalone submission and reproducibility outputs" in source
