from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp368_marginalized_reliability_pf"
TRAIN_SOURCE = EXP_DIR / "exp368_marginalized_reliability_pf_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp368_marginalized_reliability_pf_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_module(path: Path, name: str):
    old_value = os.environ.get("EXP368_IMPORT_ONLY")
    os.environ["EXP368_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_value is None:
            os.environ.pop("EXP368_IMPORT_ONLY", None)
        else:
            os.environ["EXP368_IMPORT_ONLY"] = old_value


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_frozen_config_and_zero_pf_stage0_contract() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_contract")
    config = load_config()
    train.validate_scientific_contract(config)

    assert config["experiment"]["status"] == "stage_0_failed_close_without_rescue"
    assert config["implementation"]["stage_1_implemented"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["results"]["technical_gate_passed"] is True
    assert config["results"]["scientific_gate_passed"] is False
    assert config["results"]["stage_1_eligible"] is False
    assert config["execution_contract"]["stage_0"] == {
        "diagnostic_variants": 1,
        "reporting_folds": 5,
        "pf_seed_well_runs": 0,
        "pf_control_replays": 0,
        "model_configs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert config["execution_contract"]["parent_control_retraining"] is False


def test_normalized_gaussian_forward_prefers_normal_then_persistent_weak() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_forward")
    transition = np.asarray(
        [[511.0 / 512.0, 1.0 / 512.0], [1.0 / 128.0, 127.0 / 128.0]]
    )
    initial = np.asarray([0.8, 0.2])
    multipliers = np.asarray([1.0, 4.0])

    neutral, neutral_log_density, _ = train.sticky_gaussian_forward_filter(
        np.zeros(8),
        10.0,
        transition,
        initial,
        multipliers,
        600.0,
    )
    assert np.all(np.diff(neutral) < 0.0)
    assert neutral[-1] < neutral[0] < 0.2
    assert np.isfinite(neutral_log_density).all()

    surprised, _, _ = train.sticky_gaussian_forward_filter(
        np.asarray([50.0, 50.0, 0.0]),
        10.0,
        transition,
        initial,
        multipliers,
        600.0,
    )
    assert surprised[0] > 0.99
    assert surprised[1] > surprised[0]
    assert surprised[2] > 0.90

    normal = train.gaussian_log_density(np.asarray([0.0]), 10.0, 1.0, 600.0)
    weak = train.gaussian_log_density(np.asarray([0.0]), 10.0, 4.0, 600.0)
    assert normal[0] - weak[0] == pytest.approx(np.log(4.0))


def test_final_contiguous_known_prefix_selection() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_prefix_selection")
    mask = np.asarray([True] * 200 + [False] + [True] * 192)
    selected = train.final_contiguous_true_indices(mask, 192)
    assert selected[0] == 201
    assert selected[-1] == 392
    assert len(selected) == 192
    assert len(train.final_contiguous_true_indices(mask, 193)) == 0


def test_known_prefix_readout_uses_history_then_heldout_without_truth(
    tmp_path: Path,
) -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_prefix")
    well = "synthetic"
    rows = 220
    tvt = np.arange(rows, dtype=float) + 1000.0
    residual = np.tile(np.asarray([-10.0, 10.0]), rows // 2)
    residual[-64:] = 50.0
    horizontal = pd.DataFrame(
        {
            "TVT_input": tvt,
            "GR": 100.0 + residual,
            "TVT": tvt + 999.0,
        }
    )
    typewell = pd.DataFrame({"TVT": tvt, "GR": 100.0})
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)

    readout = train.build_known_prefix_nll_rows_for_well(
        well, tmp_path, load_config()
    )
    assert len(readout) == 64
    assert readout["row_idx"].iloc[0] == rows - 64
    assert readout["nll_gain"].sum() > 0.0
    assert "TVT" not in readout.columns
    assert "true_tvt" not in readout.columns


def test_saved_exp072_projection_does_not_parse_target(tmp_path: Path) -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_saved_path")
    path = tmp_path / "cache.csv.gz"
    pd.DataFrame(
        {
            "id": ["well_a_10", "well_a_11"],
            "well": ["well_a", "well_a"],
            "target": ["forbidden", "forbidden"],
            "last_known_tvt": [100.0, 100.0],
            "likpf_mean_d": [1.5, 2.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    frame, report = train.load_saved_exp072_path(path, load_config())
    assert frame["path_tvt"].tolist() == [101.5, 102.0]
    assert frame["row_idx"].tolist() == [10, 11]
    assert "target" not in frame.columns
    assert report["forbidden_columns_read"] == []


def test_block_policy_keeps_short_tail_and_stride_overlap() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_blocks")
    config = load_config()
    rows = pd.DataFrame(
        {
            "id": [f"well0001_{index}" for index in range(700)],
            "well_id": "well0001",
            "row_idx": np.arange(700, dtype=np.int64),
            "suffix_offset": np.arange(700, dtype=np.int64),
            "raw_gr_observed": 1,
            "gr_sigma": 20.0,
            "gr_residual": np.linspace(0.0, 60.0, 700),
        }
    )
    ledger, posterior = train.build_well_block_features(rows, config)
    assert ledger["start_suffix_offset"].tolist() == [0, 256, 512]
    assert ledger["block_row_count"].tolist() == [512, 444, 188]
    assert posterior["block_id"].tolist() == [0, 1, 2]
    assert posterior["weak_posterior_mean"].between(0.0, 1.0).all()


def test_circular_offset_is_stable_and_nonzero_for_multi_block_well() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_circular")
    first = train.stable_circular_offset("well0001", 17, "exp368")
    second = train.stable_circular_offset("well0001", 17, "exp368")
    assert first == second
    assert 1 <= first <= 16
    assert train.stable_circular_offset("well0001", 1, "exp368") == 0


def test_auc_handles_ties_and_one_class_scope() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_auc")
    assert train.roc_auc_binary(
        np.asarray([False, False, True, True]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    ) == pytest.approx(1.0)
    assert (
        train.roc_auc_binary(
            np.asarray([True, True]), np.asarray([0.0, 1.0])
        )
        is None
    )


def passing_gate_inputs(train):
    records = []
    for fold in range(5):
        for local_well in range(2):
            well_id = f"well_{fold}_{local_well}"
            for block_id, (score, circular, rmse, bad, quartile) in enumerate(
                [(0.1, 0.9, 5.0, False, 1), (0.9, 0.1, 15.0, True, 4)]
            ):
                records.append(
                    {
                        "well_id": well_id,
                        "block_id": block_id,
                        "fold": fold,
                        "weak_posterior_mean": score,
                        "weak_posterior_sum": score * 512.0,
                        "circular_weak_score": circular,
                        "circular_offset_blocks": 1,
                        "block_row_count": 512,
                        "block_rmse": rmse,
                        "bad10": bad,
                        "weak_quartile": quartile,
                        "hidden_like_spatial": True,
                        "hidden_like_typewell_purged": True,
                    }
                )
    block_readout = pd.DataFrame(records)
    config = copy.deepcopy(load_config())
    config["validation"]["expected_wells"] = 10
    config["validation"]["expected_rows"] = 20
    scopes = train.build_scope_metrics(block_readout, config)
    prefix_nll = pd.DataFrame(
        {
            "well_id": np.repeat(
                [
                    f"well_{fold}_{local}"
                    for fold in range(5)
                    for local in range(2)
                ],
                64,
            ),
            "base_nll": 10.0,
            "marginal_nll": 9.0,
            "nll_gain": 1.0,
            "weak_posterior": 0.2,
        }
    )
    frozen = {
        "freeze": {
            "strict_quartile_boundaries": True,
            "truth_columns_read_before_freeze": 0,
        }
    }
    preflight = {"saved_path_frame": pd.DataFrame(index=np.arange(20))}
    return block_readout, prefix_nll, scopes, frozen, preflight, config


def test_stage0_gate_requires_prefix_and_suffix_readouts() -> None:
    train = load_module(TRAIN_SOURCE, "exp368_train_gate")
    block_readout, prefix_nll, scopes, frozen, preflight, config = passing_gate_inputs(
        train
    )
    gate = train.evaluate_stage_0_gate(
        block_readout, prefix_nll, scopes, frozen, preflight, config
    )
    assert gate["technical_pass"] is True
    assert gate["scientific_pass"] is True
    assert gate["passed"] is True
    assert gate["scientific"]["known_prefix_predictive_nll_gain_fraction"] == pytest.approx(
        0.1
    )
    assert gate["scientific"]["passing_folds_auc_gt_0p50"] == 5

    broken_prefix = prefix_nll.copy()
    broken_prefix["marginal_nll"] = 10.0
    gate = train.evaluate_stage_0_gate(
        block_readout, broken_prefix, scopes, frozen, preflight, config
    )
    assert gate["scientific_pass"] is False
    assert gate["stage_1_eligible"] is False


def test_inference_is_fail_closed_and_never_creates_submission() -> None:
    inference = load_module(INFERENCE_SOURCE, "exp368_inference_contract")
    config = load_config()
    contract = inference.validate_disabled_inference(config)
    assert contract["pf_seed_well_runs"] == 0
    with pytest.raises(RuntimeError, match="Stage 1 PF replay"):
        inference.stop_disabled_inference(config)
    source = INFERENCE_SOURCE.read_text()
    assert "shutil.copyfile" not in source
    assert "submission.csv" not in source


def test_source_is_not_a_thin_helper_entrypoint() -> None:
    source = TRAIN_SOURCE.read_text()
    headings = [
        "## 2. Runtime, configuration, path, and SHA helpers",
        "## 3. Frozen scientific and execution contract",
        "## 4. Input preflight and target-free exp072 path preparation",
        "## 5. Exact reliability recursion, known-prefix NLL, and suffix block freeze",
        "## 6. Late truth, fold, and hidden-like attachment",
        "## 7. AUC, quartile, fold, and promotion-gate readout",
        "## 8. Metrics, diagnostics, and generated artifacts",
        "## 9. Setup and configuration preview",
        "## 10. Run the approved Kaggle CPU Stage 0",
    ]
    for heading in headings:
        assert heading in source
    assert "from settings import" not in source
    assert "__file__" not in source
