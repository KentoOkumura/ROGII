from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_NAME = "exp252_pf_seed_medoid_selectability_audit"
EXP_DIR = ROOT / "experiments" / EXP_NAME


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_exp252_is_k8_only_no_training_no_pf_replay() -> None:
    config = load_config()
    assert config["experiment"]["route"] == "pf_beam"
    assert config["lineage"]["parent"] == "exp243_pf_seed_medoids"
    assert config["model"]["k_values"] == [8]
    assert config["model"]["lightgbm_config_count"] == 0
    assert config["model"]["fold_count"] == 0
    assert config["model"]["booster_count"] == 0
    assert config["model"]["pf_replay_count"] == 0
    assert config["model"]["parent_control_retraining"] is False
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False
    assert len(config["audit"]["base_candidates"]) == 8
    assert config["audit"]["medoid_candidates"] == [
        f"pf_seed_medoid_k8_m{slot}" for slot in range(8)
    ]
    assert config["validation"]["scopes"] == [
        "row",
        "block_128",
        "block_256",
        "block_512",
        "whole_well",
    ]


def test_exp252_uses_canonical_exp243_v3_content_sha() -> None:
    config = load_config()
    parent_metrics = json.loads(
        (ROOT / "experiments" / "exp243_pf_seed_medoids" / "metrics.json").read_text()
    )
    expected = parent_metrics["sha256_decompressed_for_gzip"]
    inputs = config["data"]["inputs"]
    assert inputs["row_candidates"]["sha_kind"] == "decompressed"
    assert inputs["cluster_manifest"]["sha_kind"] == "decompressed"
    assert inputs["row_candidates"]["expected_sha256"] == expected["row_candidates"]
    assert inputs["cluster_manifest"]["expected_sha256"] == expected["cluster_manifest"]
    assert inputs["cluster_summary"]["expected_sha256"] == expected["cluster_summary"]
    assert inputs["pf_diagnostics"]["expected_sha256"] == expected["pf_diagnostics"]


def test_exp252_target_free_score_freeze_does_not_read_true_tvt() -> None:
    source_path = EXP_DIR / f"{EXP_NAME}_train.py"
    source = source_path.read_text()
    module = ast.parse(source)
    freeze = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_target_free_scores"
    )
    attributes = {
        node.attr for node in ast.walk(freeze) if isinstance(node, ast.Attribute)
    }
    names = {node.id for node in ast.walk(freeze) if isinstance(node, ast.Name)}
    assert "true_tvt" not in attributes
    assert "target" not in names
    assert "__file__" not in source
    assert "from settings" not in source


def test_exp252_notebook_exposes_full_audit_structure() -> None:
    notebook = json.loads((EXP_DIR / f"{EXP_NAME}_train.ipynb").read_text())
    headings = [
        "".join(cell["source"]).splitlines()[0]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown" and cell.get("source")
    ]
    assert len(notebook["cells"]) >= 20
    assert "## 3. Fixed input preflight helpers" in headings
    assert "## 4. Target-free score construction helpers" in headings
    assert "## 5. Row, block, and whole-well scope helpers" in headings
    assert "## 6. AUC, coverage, shuffled control, and regret helpers" in headings
    assert "## 9. Join true TVT for labels and evaluate all scopes" in headings
    assert "## 10. Metrics, diagnostics, and generated artifacts" in headings
