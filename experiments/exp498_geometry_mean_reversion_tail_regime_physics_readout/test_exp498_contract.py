from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXP_DIR = Path("experiments/exp498_geometry_mean_reversion_tail_regime_physics_readout")
SOURCE_PATH = EXP_DIR / "exp498_geometry_mean_reversion_tail_regime_physics_readout_train.py"
CONFIG_PATH = EXP_DIR / "config.yaml"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("exp498_train_contract", SOURCE_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_config_preserves_zero_model_prediction_and_submission_contract(config: dict) -> None:
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == "completed_terminal_fail"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["kaggle_run_approved"] is True
    assert config["execution"]["run_readout"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["runtime"]["kaggle"]["run_on_push"] is False
    assert config["implementation"]["inference_enabled"] is False
    assert config["implementation"]["submission_enabled"] is False
    planned = config["execution_contract"]["if_separately_implemented_and_run"]
    assert planned["readouts"] == 1
    for key in (
        "new_hmm_well_runs",
        "new_predictions",
        "model_configs",
        "trained_folds",
        "boosters",
        "pf_runs",
        "beam_runs",
        "gpu_runs",
    ):
        assert planned[key] == 0


def test_phase_a_allowlists_exclude_outcomes_and_horizontal_truth(module, config: dict) -> None:
    safe = tuple(config["data"]["inputs"]["predictions"]["phase_a_safe_columns"])
    assert safe == module.PREDICTION_SAFE_COLUMNS
    assert not set(safe).intersection(module.PREDICTION_FORBIDDEN_BEFORE_FREEZE)
    assert "TVT" not in module.HORIZONTAL_READ_COLUMNS
    assert tuple(config["data"]["visible_prefix_columns"]) == module.HORIZONTAL_SAFE_COLUMNS


def test_raw_sha_decoder_manifests_are_pinned_and_cover_773_wells(config: dict) -> None:
    combined: set[str] = set()
    scientific = config["data"]["inputs"]["scientific_contract"]["sha256"]
    for shard in config["data"]["inputs"]["raw_sha_manifests"]["shards"]:
        local = Path(shard["candidates"][-1]) / shard["filename"]
        assert local.is_file()
        digest = __import__("hashlib").sha256(local.read_bytes()).hexdigest()
        assert digest == shard["sha256"]
        payload = json.loads(local.read_text(encoding="utf-8"))
        assert payload["scientific_contract_sha256"] == scientific
        wells = set(payload["raw_input_sha256_by_well"])
        assert not combined.intersection(wells)
        combined.update(wells)
    assert len(combined) == 773


def test_prefix_gr_features_match_exp490_std_and_information_contract(module) -> None:
    horizontal = pd.DataFrame(
        {
            "TVT_input": [0.0, 1.0, 2.0, 3.0, np.nan],
            "GR": [10.0, 20.0, 30.0, 40.0, 999.0],
        }
    )
    typewell = pd.DataFrame({"TVT": [0.0, 1.0, 2.0, 3.0], "GR": [8.0, 18.0, 28.0, 38.0]})
    sigma, information, rows = module.prefix_gr_features(horizontal, typewell)
    expected_at_known = np.array([8.0, 18.0, 28.0, 38.0])
    assert sigma == 10.0
    assert information == pytest.approx(
        (np.percentile(expected_at_known, 95.0) - np.percentile(expected_at_known, 5.0)) / sigma
    )
    assert rows == 4


def test_fixed_bucket_boundaries_and_primary_regime_are_pre_registered(
    module, config: dict
) -> None:
    frame = pd.DataFrame(
        {
            "well": [f"w{index}" for index in range(6)],
            "suffix_horizon_md": [0.0, 4000.0, 4000.1, 6000.0, 6000.1, 7000.0],
            "k16_median_segment_span_ft": [0.0, 240.0, 240.1, 360.0, 360.1, 500.0],
            "prefix_gr_sigma": [10.0, 19.999, 20.0, 39.999, 40.0, 60.0],
            "prefix_gr_information_ratio": [0.0, 0.999, 1.0, 1.999, 2.0, 3.0],
            "geometry_disagreement_median_ft": [0.0, 1.999, 2.0, 9.999, 10.0, 11.0],
            "early_abs_offset_ft": [0.0, 0.999, 1.0, 4.999, 5.0, 6.0],
            "state_uncertainty_median_ft": [0.0, 1.999, 2.0, 4.999, 5.0, 6.0],
        }
    )
    result = module.apply_fixed_buckets_and_regime(frame, config)
    assert result.loc[1, "suffix_horizon_bucket"] == "short_0_to_4000"
    assert result.loc[2, "suffix_horizon_bucket"] == "medium_over4000_to_6000"
    assert result.loc[3, "suffix_horizon_bucket"] == "medium_over4000_to_6000"
    assert result.loc[4, "suffix_horizon_bucket"] == "long_over6000"
    assert result.loc[5, "prefix_gr_sigma_bucket"] == "high_40_to_60"
    assert result.loc[4, module.PRIMARY_REGIME_COLUMN]
    assert result.loc[5, module.PRIMARY_REGIME_COLUMN]
    assert not result.loc[:3, module.PRIMARY_REGIME_COLUMN].any()


def test_truth_late_ledger_fails_closed_before_freeze(module) -> None:
    ledger = module.TruthLateLedger(expected_wells=1)
    with pytest.raises(RuntimeError, match="before feature freeze"):
        ledger.record_outcome("saved_delta", 1)
    clean = module.TruthLateLedger(expected_wells=1)
    features = pd.DataFrame({"well": ["well_a"], "feature": [1.0]})
    clean.freeze_features(features, "contract_sha")
    clean.record_outcome("saved_delta", 1)
    assert clean.outcome_reads_before_freeze == {}
    assert clean.post_freeze_reads == {"saved_delta": 1}


def test_saved_outcome_loader_requires_freeze_and_validates_schema(module, tmp_path: Path) -> None:
    by_well_path = tmp_path / "by_well.csv"
    episode_path = tmp_path / "episodes.csv"
    pd.DataFrame(
        {
            "well": ["well_a"],
            "fold": [0],
            "rows": [4],
            "candidate_rmse_ft": [2.0],
            "exp357_parent_rmse_ft": [1.0],
            "candidate_minus_parent_rmse_ft": [1.0],
        }
    ).to_csv(by_well_path, index=False)
    pd.DataFrame(
        {
            "episode_id": ["well_a:000"],
            "well": ["well_a"],
            "start_row_idx": [0],
            "end_row_idx_exclusive": [4],
            "rows": [4],
            "parent_sse": [4.0],
            "candidate_sse": [8.0],
            "parent_recovered_within_256": [False],
            "candidate_recovered_within_256": [False],
            "parent_recovered_within_512": [True],
            "candidate_recovered_within_512": [False],
        }
    ).to_csv(episode_path, index=False)
    blocked = module.TruthLateLedger(expected_wells=1)
    with pytest.raises(RuntimeError, match="before feature freeze"):
        module.load_saved_outcomes(
            by_well_path,
            episode_path,
            expected_wells=1,
            expected_episodes=1,
            ledger=blocked,
        )
    ledger = module.TruthLateLedger(expected_wells=1)
    ledger.freeze_features(pd.DataFrame({"well": ["well_a"]}), "contract_sha")
    by_well, episodes = module.load_saved_outcomes(
        by_well_path,
        episode_path,
        expected_wells=1,
        expected_episodes=1,
        ledger=ledger,
    )
    assert by_well["rows"].dtype == np.int64
    assert episodes["candidate_recovered_within_512"].dtype == bool


def _synthetic_gate_frame(module) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for index in range(4):
            rows.append(
                {
                    "well": f"r{fold}_{index}",
                    "fold": fold,
                    module.PRIMARY_REGIME_COLUMN: True,
                    "candidate_minus_parent_rmse_ft": 6.0,
                    "harmful_well": True,
                    "catastrophic_tail_well": True,
                }
            )
        for index in range(10):
            rows.append(
                {
                    "well": f"c{fold}_{index}",
                    "fold": fold,
                    module.PRIMARY_REGIME_COLUMN: False,
                    "candidate_minus_parent_rmse_ft": 0.0,
                    "harmful_well": False,
                    "catastrophic_tail_well": False,
                }
            )
    return pd.DataFrame(rows)


def test_truth_late_join_uses_declared_suffixes_and_checks_fold_parity(module, config: dict) -> None:
    features = pd.DataFrame(
        {
            "well": [f"well_{fold}" for fold in range(5)],
            "rows": [10] * 5,
            "prediction_rows": [10] * 5,
        }
    )
    folds = pd.DataFrame(
        {
            "well": features["well"],
            "fold": list(range(5)),
            "fold_rows": [10] * 5,
        }
    )
    by_well = pd.DataFrame(
        {
            "well": features["well"],
            "fold": list(range(5)),
            "rows": [10] * 5,
            "candidate_rmse_ft": [2.0] * 5,
            "exp357_parent_rmse_ft": [1.0] * 5,
            "candidate_minus_parent_rmse_ft": [1.0] * 5,
        }
    )
    joined = module.attach_truth_late_outcomes(features, folds, by_well, config)
    assert joined["fold"].tolist() == list(range(5))
    assert "fold_manifest" not in joined.columns
    assert "fold_outcome" not in joined.columns
    assert joined["harmful_well"].all()

    by_well.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="prediction fold and saved by-well fold disagree"):
        module.attach_truth_late_outcomes(features, folds, by_well, config)


def test_primary_all_and_gate_passes_only_the_single_fixed_regime(module, config: dict) -> None:
    frame = _synthetic_gate_frame(module)
    episodes = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(len(frame))],
            "well": frame["well"],
            "parent_sse": np.full(len(frame), 10.0),
            "candidate_sse": np.where(frame[module.PRIMARY_REGIME_COLUMN], 20.0, 5.0),
            "parent_recovered_within_256": False,
            "candidate_recovered_within_256": False,
            "parent_recovered_within_512": False,
            "candidate_recovered_within_512": False,
        }
    )
    by_fold, summary = module.evaluate_primary_regime(frame, episodes, config)
    assert len(by_fold) == 10
    assert summary["supported_folds"] == 5
    assert math.isinf(summary["harmful_rate_ratio_regime_over_complement"])
    assert summary["catastrophic_tail_capture_fraction"] == 1.0
    assert summary["regime_coverage_fraction"] == pytest.approx(20 / 70)
    assert all(summary["checks"].values())
    assert summary["passed"] is True


def test_primary_gate_does_not_rescue_sparse_regime(module, config: dict) -> None:
    frame = _synthetic_gate_frame(module)
    frame.loc[
        frame[module.PRIMARY_REGIME_COLUMN] & ~frame.index.isin(frame.index[:5]),
        module.PRIMARY_REGIME_COLUMN,
    ] = False
    episodes = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(len(frame))],
            "well": frame["well"],
            "parent_sse": 10.0,
            "candidate_sse": 10.0,
            "parent_recovered_within_256": False,
            "candidate_recovered_within_256": False,
            "parent_recovered_within_512": False,
            "candidate_recovered_within_512": False,
        }
    )
    _, summary = module.evaluate_primary_regime(frame, episodes, config)
    assert summary["checks"]["coverage_and_fold_support"] is False
    assert summary["passed"] is False
    assert summary["decision"] == config["readout"]["fail_action"]


def test_secondary_readout_keeps_empty_pre_registered_buckets(module, config: dict) -> None:
    frame = pd.DataFrame(
        {
            "well": ["only_well"],
            "suffix_horizon_md": [3000.0],
            "k16_median_segment_span_ft": [200.0],
            "prefix_gr_sigma": [15.0],
            "prefix_gr_information_ratio": [0.5],
            "geometry_disagreement_median_ft": [1.0],
            "early_abs_offset_ft": [0.5],
            "state_uncertainty_median_ft": [1.0],
        }
    )
    frame = module.apply_fixed_buckets_and_regime(frame, config)
    frame["fold"] = 0
    frame["candidate_minus_parent_rmse_ft"] = 0.0
    frame["harmful_well"] = False
    frame["catastrophic_tail_well"] = False
    summary = module.build_secondary_bucket_summary(frame, config)
    assert len(summary) == 21
    assert (summary.groupby("feature").size() == 3).all()
    assert (summary["wells"] == 0).any()


def test_chunked_prediction_aggregation_handles_well_split_across_chunks(
    module, tmp_path: Path
) -> None:
    rows = []
    for well, count, base in (("a", 4, 0.0), ("b", 3, 10.0)):
        for offset in range(count):
            rows.append(
                {
                    "well": well,
                    "row_idx": offset,
                    "suffix_offset": offset,
                    "tvt_geop": base + offset,
                    "geometry_mean_reverting_delta_mean": base + offset,
                    "geometry_mean_reverting_hmm_std": 1.0 + offset,
                    "dmd": 1.0,
                    "k16_segment_id": 0,
                    "k16_segment_span": 4.0,
                    "rho": 0.9,
                    "exp226_pred": base + offset + 2.0,
                    "md_since": offset + 1.0,
                }
            )
    path = tmp_path / "prediction.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    ledger = module.TruthLateLedger(expected_wells=2)
    result = module.stream_prediction_features(
        path,
        expected_rows=7,
        chunk_rows=3,
        ledger=ledger,
    )
    assert result["well"].tolist() == ["a", "b"]
    assert result["prediction_rows"].tolist() == [4, 3]
    assert result["geometry_disagreement_median_ft"].tolist() == [2.0, 2.0]
    assert ledger.safe_reads["prediction_safe_columns"] == 7


def test_train_source_is_not_a_thin_helper_wrapper() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "Path(__file__)" not in source
    assert "from settings import" not in source
    assert source.count("# %% [markdown]") >= 10
    assert "def stream_prediction_features" in source
    assert "def aggregate_raw_prefix_features" in source
    assert "def evaluate_primary_regime" in source
    assert "def run_readout" in source


def test_canonical_train_notebook_is_readable_and_unexecuted() -> None:
    notebook_path = (
        EXP_DIR / "exp498_geometry_mean_reversion_tail_regime_physics_readout_train.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert sum(cell["cell_type"] == "markdown" for cell in cells) == 11
    assert sum(cell["cell_type"] == "code" for cell in cells) == 10
    assert sum(len(cell.get("outputs", [])) for cell in cells) == 0
    assert any(
        "Truth-late" in "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
