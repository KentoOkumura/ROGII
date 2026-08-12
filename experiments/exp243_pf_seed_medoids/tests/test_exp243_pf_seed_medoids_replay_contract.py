from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "exp243_pf_seed_medoids" / "pf_seed_medoids.py"
CONFIG = ROOT / "experiments" / "exp243_pf_seed_medoids" / "config.yaml"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _called_names(function: ast.FunctionDef) -> list[str]:
    return [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_exp072_replay_inputs_never_round_trip_through_float32() -> None:
    function = _function("run_exp072_seed_trajectories")
    calls = _called_names(function)

    assert calls.count("numeric_array64") == 3
    assert "numeric_array" not in calls


def test_exp072_parity_control_uses_separate_sha_fixed_exp209_column() -> None:
    function = _function("read_exp072_eval_cache")
    calls = _called_names(function)
    config = CONFIG.read_text()

    assert "read_exp209_reconstructed_likpf_control" in calls
    assert "align_exp209_likpf_control" in calls
    assert "exp072_train_feature_cache_expected_sha256" in config
    assert "exp072_train_feature_cache_expected_decompressed_sha256" in config
    assert "exp072_feature_schema_expected_sha256" in config
    assert "exp209_enriched_likpf_control_expected_sha256" in config
    assert "exp209_enriched_likpf_control_expected_decompressed_sha256" in config

    align_source = ast.get_source_segment(
        SOURCE.read_text(), _function("align_exp209_likpf_control")
    )
    assert align_source is not None
    assert "likpf_mean_exp209_reconstructed" in align_source
    assert "refuse an ambiguous parity-control merge" in align_source
