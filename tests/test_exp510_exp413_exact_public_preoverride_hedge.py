from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp510_exp413_exact_public_preoverride_hedge"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
RUNTIME_SOURCE = EXP_DIR / "exp510_exp413_hidden_safe_runtime.py"
PARENT_EXP413_SOURCE = (
    ROOT
    / "experiments/exp413_scale5_likpf_full_replacement_on_exp335"
    / "exp413_scale5_likpf_full_replacement_on_exp335_current_test_inference.py"
)
PARENT_EXP413_SHA256 = (
    "0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388"
)
ARCHIVED_SOURCE = (
    ROOT
    / "docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260627"
    / "degnonguidi__public-score-rogii-lb-7-159/public-score-rogii-lb-7-159.ipynb"
)
ARCHIVED_SHA256 = "4d0712983788dc7d9b97fdb8e5dc7c30b6d3634a9c64597d84d21da28e9623eb"


def load_source() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp510_inference", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXP_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    return load_source()


def test_archived_source_sha_and_visible_boundary_are_exact() -> None:
    assert hashlib.sha256(ARCHIVED_SOURCE.read_bytes()).hexdigest() == ARCHIVED_SHA256
    source = ARCHIVED_SOURCE.read_text()
    blend_start = source.index("_sp45 = read_submission_frame")
    blend_end = source.index("final_blend.to_csv")
    excluded_stage = source.index("def run_guarded_contact_override")
    blend_cell = source[blend_start:excluded_stage].replace('\\"', '"')
    assert blend_start < blend_end < excluded_stage
    assert 'CFG.OUT / "sp45_projection_submission.csv"' in blend_cell
    assert 'CFG.OUT / "submission_B.csv"' in blend_cell
    assert "SP45_WEIGHT * _merged[\"tvt_sp45\"]" in blend_cell
    assert "(1 - SP45_WEIGHT) * _merged[\"tvt_fleongg\"]" in blend_cell
    assert "submission_A.csv" not in blend_cell


def test_candidate_ast_contains_no_excluded_stage_or_training_route() -> None:
    text = SOURCE.read_text()
    tree = ast.parse(text)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    loaded_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    forbidden_names = {
        "run_guarded_contact_override",
        "ENABLE_GOLD_OVERLAY",
        "CVTrainer",
        "GroupKFold",
        "LGBMRegressor",
        "CatBoostRegressor",
        "train_stack_B",
    }
    assert not (forbidden_names & defined)
    assert not (forbidden_names & loaded_names)
    assert "submission_A.csv" not in text
    assert "/kaggle/input/notebooks" not in text
    assert "np.random.seed" not in text
    assert "np.random.randn" not in text
    assert "np.random.uniform" not in text


def test_jupytext_markdown_cells_do_not_swallow_python_definitions() -> None:
    in_markdown = False
    violations: list[tuple[int, str]] = []
    for line_number, line in enumerate(SOURCE.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            in_markdown = "[markdown]" in stripped
            continue
        if in_markdown and stripped and not line.lstrip().startswith("#"):
            violations.append((line_number, line))
    assert violations == []


def test_stable_seed_depends_on_immutable_context(inference: ModuleType) -> None:
    first = inference.stable_seed("test", "public_likpf_bank", "well_a", 7)
    assert first == inference.stable_seed("test", "public_likpf_bank", "well_a", 7)
    assert 0 <= first < 2**32
    assert first != inference.stable_seed("test", "public_likpf_bank", "well_b", 7)
    assert first != inference.stable_seed("test", "other_family", "well_a", 7)
    assert len(
        {
            inference.stable_seed("test", "public_likpf_bank", "well_a", seed_index)
            for seed_index in range(48)
        }
    ) == 48


def synthetic_components() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = ["aa_0", "aa_1", "bb_0"]
    sp45 = pd.DataFrame({"id": ids, "tvt": [10.0, 20.0, 30.0]})
    pipeline_b = pd.DataFrame({"id": ids, "tvt": [14.0, 18.0, 34.0]})
    exp413 = pd.DataFrame({"id": ids, "tvt": [11.0, 21.0, 29.0]})
    return sp45, pipeline_b, exp413


def test_fixed_float64_formulas_are_exact(inference: ModuleType) -> None:
    sp45, pipeline_b, exp413 = synthetic_components()
    public = inference.exact_public_preoverride(sp45, pipeline_b)
    final = inference.exact_final_hedge(exp413, public)
    np.testing.assert_array_equal(
        public["tvt"].to_numpy(),
        np.float64(0.55) * sp45["tvt"].to_numpy(np.float64)
        + np.float64(0.45) * pipeline_b["tvt"].to_numpy(np.float64),
    )
    np.testing.assert_array_equal(
        final["tvt"].to_numpy(),
        np.float64(0.90) * exp413["tvt"].to_numpy(np.float64)
        + np.float64(0.10) * public["tvt"].to_numpy(np.float64),
    )
    assert final["tvt"].dtype == np.float64


def test_dynamic_exp413_uses_serialized_component_boundary(
    inference: ModuleType, tmp_path: Path
) -> None:
    in_memory = pd.DataFrame(
        {
            "id": ["aa_0", "aa_1"],
            "well": ["aa", "aa"],
            "last_known_tvt": np.asarray([10000.0, 10000.0], dtype=np.float32),
            "pred_tvt": np.asarray([11747.101, 11747.021], dtype=np.float32),
        }
    )
    artifact = tmp_path / "exp413_current_test_predictions.csv.gz"
    in_memory.to_csv(artifact, index=False)

    serialized, roundtrip_max_abs = inference.reload_dynamic_exp413_artifact(
        artifact, in_memory
    )

    assert serialized["pred_tvt"].dtype == np.float64
    assert 0.0 < roundtrip_max_abs < 1e-3
    assert serialized["pred_tvt"].tolist() == [11747.101, 11747.021]


def test_formula_rejects_order_mismatch(inference: ModuleType) -> None:
    sp45, pipeline_b, exp413 = synthetic_components()
    with pytest.raises(RuntimeError, match="identical ID order"):
        inference.exact_public_preoverride(sp45, pipeline_b.iloc[::-1].reset_index(drop=True))
    public = inference.exact_public_preoverride(sp45, pipeline_b)
    with pytest.raises(RuntimeError, match="identical ID order"):
        inference.exact_final_hedge(exp413.iloc[::-1].reset_index(drop=True), public)


def test_dynamic_sample_contract_is_fail_closed(inference: ModuleType) -> None:
    sample = inference.validate_sample(
        pd.DataFrame({"id": ["aa_0", "aa_1", "bb_0"], "tvt": [0.0, 0.0, 0.0]})
    )
    shuffled = pd.DataFrame(
        {"id": ["bb_0", "aa_0", "aa_1"], "value": [30.0, 10.0, 20.0]}
    )
    aligned = inference.validate_component(shuffled, sample, "synthetic", value_column="value")
    assert aligned["id"].tolist() == sample["id"].tolist()
    assert aligned["tvt"].tolist() == [10.0, 20.0, 30.0]

    with pytest.raises(RuntimeError, match="duplicate"):
        inference.validate_component(
            pd.concat([shuffled, shuffled.iloc[[0]]], ignore_index=True),
            sample,
            "duplicate",
            value_column="value",
        )
    with pytest.raises(RuntimeError, match="ID mismatch"):
        inference.validate_component(
            shuffled.iloc[:-1], sample, "missing", value_column="value"
        )
    nonfinite = shuffled.copy()
    nonfinite.loc[0, "value"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        inference.validate_component(
            nonfinite, sample, "nonfinite", value_column="value"
        )


def test_model_artifact_discovery_is_sha_and_cardinality_closed(
    inference: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    names = ["features.json", "lgb0.pkl", "lgb1.pkl", "lgb2.pkl"]
    expected = {}
    for index, name in enumerate(names):
        path = model_dir / name
        path.write_bytes(f"artifact-{index}".encode())
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(inference, "MODEL_SHA256", expected)
    assert inference.find_exact_model_dir([tmp_path]) == model_dir.resolve()

    (model_dir / "lgb1.pkl").write_bytes(b"mutation")
    with pytest.raises(RuntimeError, match="expected one"):
        inference.find_exact_model_dir([tmp_path])


def test_exp413_runtime_is_dynamic_and_parent_sha_guarded(
    inference: ModuleType,
) -> None:
    assert hashlib.sha256(PARENT_EXP413_SOURCE.read_bytes()).hexdigest() == (
        PARENT_EXP413_SHA256
    )
    assert inference.EXP413_PARENT_SOURCE_SHA256 == PARENT_EXP413_SHA256
    runtime_text = RUNTIME_SOURCE.read_text()
    runtime_tree = ast.parse(runtime_text)
    assert "def generate_dynamic_exp413_prediction():" in runtime_text
    assert "config[\"inference\"][\"run_enabled\"] = True" in runtime_text
    assert "from exp413_runtime.settings import" in runtime_text
    assert not any(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        for node in runtime_tree.body
    )

    candidate_text = SOURCE.read_text()
    assert "find_exp413_prediction" not in candidate_text
    assert "EXP413_RAW_GZIP_SHA256" not in candidate_text
    assert "generate_dynamic_exp413_prediction()" in candidate_text


def test_dynamic_exp413_component_uses_current_sample_ids(
    inference: ModuleType,
) -> None:
    sample = inference.validate_sample(
        pd.DataFrame({"id": ["hidden_a_0", "hidden_b_0"], "tvt": [0.0, 0.0]})
    )
    dynamic = pd.DataFrame(
        {
            "id": ["hidden_b_0", "hidden_a_0"],
            "well": ["hidden_b", "hidden_a"],
            "last_known_tvt": [20.0, 10.0],
            "pred_tvt": [22.0, 11.0],
        }
    )
    component, context = inference.load_exp413_component(dynamic, sample)
    assert component["id"].tolist() == sample["id"].tolist()
    assert component["tvt"].tolist() == [11.0, 22.0]
    assert context["well"].tolist() == ["hidden_a", "hidden_b"]

    with pytest.raises(RuntimeError, match="ID mismatch"):
        inference.load_exp413_component(dynamic.iloc[:1], sample)


def test_kaggle_inputs_use_exp413_generation_sources_not_public_sidecar() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    sources = config["runtime"]["kaggle"]["current_test_inference_kernel_sources"]
    assert "kentookumura/exp413-scale5-likpf-current-test-inference" not in sources
    assert len(sources) == 11
    assert {
        "kentookumura/exp073-full-replay-repro-guard-infer",
        "kentookumura/exp413-scale5-likpf-selector-train",
        "kentookumura/exp413-scale5-likpf-signed-train",
        "kentookumura/exp413-scale5-likpf-downstream-train",
    }.issubset(sources)

    destinations = {
        item["destination"]
        for item in config["runtime"]["kaggle"]["bootstrap_dependency_files"]
    }
    assert "exp413_runtime/settings.py" in destinations
    assert "exp413_runtime/config.yaml" in destinations
    assert "exp413_runtime/inputs/exp264_candidate_contract.yaml" in destinations


def test_expected_source_model_schema_gap_is_frozen(inference: ModuleType) -> None:
    assert len(inference.EXPECTED_ZERO_FILLED_FEATURES) == 39
    assert {"beam_vcons_d", "beam_vloose_d", "beam_stiff_d"}.issubset(
        inference.EXPECTED_ZERO_FILLED_FEATURES
    )
    assert "tda-80" in inference.EXPECTED_ZERO_FILLED_FEATURES
    assert "tdpf30" in inference.EXPECTED_ZERO_FILLED_FEATURES


def test_component_readout_is_truth_free(inference: ModuleType) -> None:
    sp45, pipeline_b, exp413 = synthetic_components()
    public = inference.exact_public_preoverride(sp45, pipeline_b)
    final = inference.exact_final_hedge(exp413, public)
    metrics = inference.difference_metrics(final["tvt"], exp413["tvt"])
    assert set(metrics) == {"rmse", "mae", "mean_signed", "p95_abs", "max_abs"}
    assert all(np.isfinite(list(metrics.values())))
