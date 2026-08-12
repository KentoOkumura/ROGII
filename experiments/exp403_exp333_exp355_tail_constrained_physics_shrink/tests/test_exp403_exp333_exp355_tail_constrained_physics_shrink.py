from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT / "experiments/exp403_exp333_exp355_tail_constrained_physics_shrink"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp403_exp333_exp355_tail_constrained_physics_shrink_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp403_exp333_exp355_tail_constrained_physics_shrink_compact_selfcontained_inference.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return load_module("exp403_train_test_module", TRAIN_SOURCE)


@pytest.fixture(scope="module")
def inference_module():
    return load_module("exp403_inference_test_module", INFERENCE_SOURCE)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_frozen_implementation_contract(train_module, config) -> None:
    preview = train_module.validate_contract(
        config,
        require_run_authorization=False,
    )
    assert all(preview["checks"].values())
    assert tuple(preview["lambda_candidates"]) == train_module.EXPECTED_LAMBDAS
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"].startswith("kaggle_train_")
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["kaggle_train_run_approved"] is False
    assert config["execution"]["run_train"] is False
    with pytest.raises(RuntimeError, match="fail-closed"):
        train_module.validate_contract(config, require_run_authorization=True)


def test_prefreeze_allowlists_exclude_truth_and_error(config) -> None:
    for key in ("exp333_oof", "exp355_oof"):
        spec = config["data"][key]
        allowed = set(spec["allowed_pre_freeze_columns"])
        forbidden = set(spec["forbidden_pre_freeze_columns"])
        assert not allowed & forbidden
        assert not any(
            token in column.lower()
            for column in allowed
            for token in ("truth", "tvt_true", "true_tvt", "error", "oracle")
        )


def test_truth_access_ledger_requires_complete_freeze(train_module) -> None:
    ledger = train_module.AccessLedger()
    with pytest.raises(RuntimeError, match="truth"):
        ledger.pre_freeze_read("bad", ["well_id", "tvt_true"], 1)

    ledger = train_module.AccessLedger()
    ledger.pre_freeze_read("saved_prediction", ["well_id", "candidate_tvt"], 4)
    evidence = {
        "rows": 4,
        "wells": 2,
        "source_schema_sha256": "a" * 64,
        "source_content_sha256": "b" * 64,
        "formula_sha256": "c" * 64,
    }
    ledger.freeze(evidence)
    ledger.truth_late("suffix_truth", 4)
    assert ledger.prediction_frozen is True
    assert ledger.truth_columns_read_before_freeze == 0
    assert ledger.late_truth_rows == 4


def _component_frame(
    candidate: str,
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["w0_10", "w0_11", "w1_20"],
            "well_id": ["w0", "w0", "w1"],
            "row_idx": np.array([10, 11, 20], dtype=np.int32),
            "exp263_generation_fold": np.array([0, 0, 0], dtype=np.int8),
            "md_since": [100.0, 500.0, 1500.0],
            candidate: values,
        }
    )


def test_source_partition_reconstructs_only_frozen_formulas(
    train_module,
    config,
) -> None:
    exp226 = np.array([12739.560546875, 20.0, 30.0], dtype=np.float32)
    likpf = np.array([9388.7841796875, 24.0, 34.0], dtype=np.float32)
    exact = np.array([13585.9794921875, 28.0, 38.0], dtype=np.float32)
    exp333 = np.array([12740.560546875, 21.0, 31.0], dtype=np.float32)
    exp355 = np.array([13584.9794921875, 27.0, 37.0], dtype=np.float32)
    well_codes = train_module._well_code_map(["w0", "w1"])
    keys = train_module.global_row_key(
        ["w0", "w0", "w1"],
        np.array([10, 11, 20], dtype=np.int32),
        well_codes,
    )
    branch_frame = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1"],
            "row_idx": np.array([10, 11, 20], dtype=np.int32),
            "reporting_fold": np.array([1, 1, 2], dtype=np.int8),
            "exp333_exp226_parity": exp226,
            "exp333_stage1": exp333,
            "exp355_stage1": exp355,
        },
        index=pd.Index(keys, name="global_row_key"),
    )
    branch = train_module.BranchPredictions(
        frame=branch_frame,
        well_codes=well_codes,
        source_reports=[],
        exp333_prediction_sha256="a" * 64,
        exp355_selected_content_sha256="b" * 64,
        used=np.zeros(3, dtype=bool),
    )
    source, parity = train_module.assemble_source_partition(
        {
            "exp226_k16": _component_frame("exp226_k16", exp226),
            "likpf_mean": _component_frame("likpf_mean", likpf),
            "exp209_exact_hmm": _component_frame(
                "exp209_exact_hmm",
                exact,
            ),
        },
        branch,
        config,
    )
    expected_control = (
        0.50 * np.asarray(exp226, dtype=np.float64)
        + 0.25 * np.asarray(likpf, dtype=np.float64)
        + 0.25 * np.asarray(exact, dtype=np.float64)
    )
    expected_full = (
        0.50 * np.asarray(exp333, dtype=np.float64)
        + 0.25 * np.asarray(likpf, dtype=np.float64)
        + 0.25 * np.asarray(exp355, dtype=np.float64)
    )
    np.testing.assert_allclose(source["exp263_control"], expected_control)
    np.testing.assert_allclose(source["full_replacement"], expected_full)
    np.testing.assert_allclose(
        source["correction"],
        expected_full - expected_control,
    )
    assert branch.used.all()
    assert parity["exp226_parity_max_abs_ft"] == 0.0
    assert parity["exp263_formula_parity_max_abs_ft"] > 1.0e-5
    assert parity["exp263_formula_parity_max_float32_ulps"] <= 1.0


def _calibration_frame(correction: float = 1.0) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for well_index in range(2):
            well = f"f{fold}w{well_index}"
            for row_index, distance in enumerate((100.0, 500.0, 1500.0)):
                rows.append(
                    {
                        "id": f"{well}_{row_index}",
                        "well_id": well,
                        "row_idx": row_index,
                        "reporting_fold": fold,
                        "exp263_generation_fold": (fold + 1) % 5,
                        "md_since": distance,
                        "exp263_control": 0.0,
                        "full_replacement": correction,
                        "correction": correction,
                        "true_tvt": 1.0,
                        "hidden_like_spatial": well_index == 0,
                        "hidden_like_typewell_purged": well_index == 1,
                    }
                )
    return pd.DataFrame(rows)


def test_lambda_selection_uses_largest_eligible_and_zero_fallback(
    train_module,
    config,
) -> None:
    metrics, selection = train_module.calibrate_lambdas(
        _calibration_frame(correction=1.0),
        config,
    )
    assert len(metrics) == 45
    assert selection["lambda_fold"].eq(1.0).all()
    assert selection["outer_valid_rows_used_for_selection"].eq(0).all()

    _, failed = train_module.calibrate_lambdas(
        _calibration_frame(correction=-1.0),
        config,
    )
    assert failed["lambda_fold"].eq(0.0).all()
    assert failed["fallback_to_zero"].all()


def test_heldout_fold_cannot_change_its_lambda(train_module, config) -> None:
    frame = _calibration_frame(correction=1.0)
    _, original = train_module.calibrate_lambdas(frame, config)
    changed = frame.copy()
    heldout = changed["reporting_fold"].eq(0)
    changed.loc[heldout, "true_tvt"] = 1000.0
    changed.loc[heldout, "correction"] = -1000.0
    _, modified = train_module.calibrate_lambdas(changed, config)
    lambda_original = original.set_index("outer_valid_fold").loc[0, "lambda_fold"]
    lambda_modified = modified.set_index("outer_valid_fold").loc[0, "lambda_fold"]
    assert lambda_original == lambda_modified


def test_persistent_episode_and_recovery_contract(train_module, config) -> None:
    rows = 140
    frame = pd.DataFrame(
        {
            "well_id": ["well"] * rows,
            "row_idx": np.arange(rows, dtype=np.int32),
            "reporting_fold": np.zeros(rows, dtype=np.int8),
            "true_tvt": np.zeros(rows),
            "exp263_control": np.full(rows, 12.0),
            "crossfit_shrink": np.r_[np.full(128, 12.0), np.zeros(12)],
        }
    )
    episodes, recovery = train_module.persistent_offset_episodes(frame, config)
    assert len(episodes) == 2
    summary = recovery.set_index("candidate")
    assert summary.loc["crossfit_shrink", "episodes"] == 1
    assert summary.loc["crossfit_shrink", "recovered_within_512_rate"] == 1.0
    assert summary.loc["exp263_control", "recovered_within_512_rate"] == 0.0


def test_promotion_gate_is_all_and(train_module, config) -> None:
    selection = pd.DataFrame(
        {
            "outer_valid_fold": range(5),
            "lambda_fold": [0.25] * 5,
        }
    )
    scopes = [
        "pooled",
        "near_0_250",
        "mid_250_1000",
        "1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    scope_metrics = pd.DataFrame(
        {
            "scope": scopes,
            "control_rmse": [1.0] * len(scopes),
            "candidate_rmse": [0.9] * len(scopes),
            "delta_rmse_ft": [-0.1] * len(scopes),
            "gain_ft": [0.1] * len(scopes),
        }
    )
    fold_metrics = pd.DataFrame({"improved": [True] * 5})
    by_well = pd.DataFrame(
        {
            "well_id": ["a", "b", "c"],
            "delta_rmse_ft": [-0.1, -0.2, -0.3],
        }
    )
    recovery = pd.DataFrame(
        {
            "candidate": ["crossfit_shrink", "exp263_control"],
            "episodes": [0, 1],
            "recovered_within_512_rate": [np.nan, 0.0],
        }
    )
    passed = train_module.evaluate_promotion_gate(
        selection,
        scope_metrics,
        fold_metrics,
        by_well,
        recovery,
        config,
    )
    assert passed["passed"] is True
    by_well.loc[0, "delta_rmse_ft"] = 1.0
    failed = train_module.evaluate_promotion_gate(
        selection,
        scope_metrics,
        fold_metrics,
        by_well,
        recovery,
        config,
    )
    assert failed["passed"] is False
    assert failed["checks"]["worst_well"] is False


def test_partition_hasher_is_chunk_boundary_independent(train_module) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    whole = train_module.PartitionContentHasher(frame.columns)
    whole.update(frame)
    chunked = train_module.PartitionContentHasher(frame.columns)
    chunked.update(frame.iloc[:1])
    chunked.update(frame.iloc[1:])
    assert whole.hexdigest() == chunked.hexdigest()
    assert whole.rows == chunked.rows == 3


def test_portable_prediction_surface_sha_is_dtype_independent(train_module) -> None:
    narrow = pd.DataFrame(
        {
            "well_id": ["b", "a"],
            "row_idx": np.array([2, 1], dtype=np.int32),
            "outer_fold": np.array([1, 0], dtype=np.int8),
            "tvt_pred_stage1": np.array([1.5, -2.25], dtype=np.float32),
        }
    )
    wide = narrow.astype(
        {
            "well_id": str,
            "row_idx": np.int64,
            "outer_fold": np.int64,
            "tvt_pred_stage1": np.float64,
        }
    )
    columns = ("well_id", "row_idx", "outer_fold", "tvt_pred_stage1")
    assert train_module.portable_prediction_surface_sha256(
        narrow, columns
    ) == train_module.portable_prediction_surface_sha256(wide, columns)


def test_raw_truth_resolver_skips_schema_incomplete_duplicate(
    train_module,
    tmp_path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    decoy = input_root / "notebooks" / "decoy"
    valid = input_root / "rogii-wellbore-geology-prediction" / "train"
    wells = ("w0", "w1", "w2")
    for directory, columns in (
        (decoy, "TVT_input\n1.0\n"),
        (valid, "TVT,TVT_input\n1.0,1.0\n"),
    ):
        directory.mkdir(parents=True)
        for well in wells:
            (directory / f"{well}__horizontal_well.csv").write_text(columns)
    monkeypatch.setattr(train_module, "KAGGLE_INPUT_ROOT", input_root)
    config = {"data": {"train_dir": "does/not/exist"}}
    assert train_module.resolve_raw_train_dir(config, wells) == valid


def test_inference_remains_fail_closed(inference_module, config) -> None:
    status = inference_module.validate_inference_is_disabled(config)
    assert status["train_readout_implemented"] is True
    assert status["canonical_notebook_adopted"] is True
    assert status["training_enabled"] is False
    assert status["run_train"] is False
    assert status["inference_enabled"] is False
    assert status["submission_enabled"] is False
    assert status["promotion_result"] is False
