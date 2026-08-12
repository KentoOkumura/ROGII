from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = ROOT / "experiments" / "exp400_all_well_1p3_sigma_gr_likelihood_pf"
TRAIN_SOURCE = (
    EXPERIMENT_DIR / "exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXPERIMENT_DIR / "exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp072_exp063_full_replay_feature_cache"
    / "public_notebook_replay_audit.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp400_train_contract")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp400_inference_contract")


@pytest.fixture(scope="module")
def config(train):
    return train.load_experiment_config(EXPERIMENT_DIR)


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(12, dtype=float) + 1.0,
            "Z": np.linspace(0.0, 1.1, 12),
            "GR": [
                50.0,
                np.nan,
                70.0,
                80.0,
                65.0,
                60.0,
                np.nan,
                55.0,
                58.0,
                np.nan,
                62.0,
                64.0,
            ],
            "TVT_input": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(90.0, 130.0, 81),
            "GR": np.linspace(45.0, 85.0, 81),
        }
    )
    return horizontal, typewell


def test_frozen_scientific_contract_and_execution_boundary(train, config) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["primary_candidate"] == "likpf_mean_x1p3"
    assert contract["secondary_candidates"] == [
        "likpf_scale_3_x1p3",
        "likpf_scale_5_x1p3",
        "likpf_scale_8_x1p3",
        "likpf_scale_12_x1p3",
    ]
    assert contract["execution_counts"]["candidate_pf_well_runs"] == 773
    assert contract["execution_counts"]["seed_well_trajectories"] == 98_944
    assert contract["execution_counts"]["particle_starts"] == 49_472_000
    assert len(contract["scientific_contract_sha256"]) == 64
    run_contract = train.validate_scientific_contract(config, require_run_approval=True)
    assert run_contract["scientific_contract_sha256"] == contract["scientific_contract_sha256"]


def test_inference_is_fail_closed(inference) -> None:
    config = inference.load_config(EXPERIMENT_DIR)
    status = inference.validate_inference_is_disabled(config)

    assert status["implementation_scope"] == "train_side_candidate_audit_only"
    assert status["inference_enabled"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False
    assert status["submit_to_kaggle"] is False


def test_stable_seed_matches_exp072_policy(train) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2_147_483_647 + 1

    assert train.stable_seed("likpf", "train", "well-a") == expected
    assert train.stable_seed("likpf", "train", "well-a") == train.stable_seed(
        "likpf", "train", "well-a"
    )
    assert train.stable_seed("likpf", "train", "well-b") != expected


def test_gr_scale_is_clipped_then_multiplied_once_without_reclip(train) -> None:
    horizontal, typewell = synthetic_well()
    prepared = train.prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=1.3,
        grid_step=0.2,
    )
    known = horizontal["TVT_input"].notna().to_numpy()
    typewell_at_known = np.interp(
        horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    residual = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64) - typewell_at_known
    expected_base = float(np.clip(np.nanstd(residual), 10.0, 60.0))

    assert prepared["scale_audit"]["base_scale"] == pytest.approx(expected_base)
    assert prepared["scale_audit"]["candidate_scale"] == pytest.approx(expected_base * 1.3)
    assert prepared["scale_audit"]["multiplier"] == 1.3
    assert prepared["scale_audit"]["post_multiplier_clip_applied"] is False
    assert prepared["scale_audit"]["post_multiplier_clip_count"] == 0
    assert prepared["scale_audit"]["known_gr_missing_rows"] == 1
    assert np.isfinite(prepared["eval_gr"]).all()


def test_candidate_x1p0_kernel_is_exact_parent_fixture_parity(train) -> None:
    if "numba" not in sys.modules:
        numba_stub = types.ModuleType("numba")

        def identity_njit(*args, **kwargs):
            del kwargs
            if args and callable(args[0]):
                return args[0]

            def decorator(function):
                return function

            return decorator

        numba_stub.njit = identity_njit
        sys.modules["numba"] = numba_stub
    parent = load_module(PARENT_SOURCE, "exp072_parent_for_exp400")
    md = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    z = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    gr = np.array([50.0, 52.0, 54.0, 53.0, 51.0], dtype=np.float64)
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        24,
        4,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )

    observed = train._pf_lik_allseeds(*args)
    expected_predictions, expected_likelihoods = parent._pf_lik_allseeds(*args)

    assert np.array_equal(observed[0], expected_predictions)
    assert np.array_equal(observed[1], expected_likelihoods)
    assert observed[2].shape == (4,)
    assert observed[3].shape == (4,)
    assert observed[4].shape == (4,)


def test_seed_aggregation_keeps_arithmetic_mean_as_primary(train) -> None:
    predictions = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float64,
    )
    likelihoods = np.array([-3.0, -2.0, -1.0], dtype=np.float64)

    outputs = train.aggregate_seed_predictions(predictions, likelihoods, [3.0, 5.0, 8.0, 12.0])

    np.testing.assert_allclose(outputs["pf_mean"], [3.0, 4.0])
    assert set(outputs) == {
        "pf_scale_3",
        "pf_scale_5",
        "pf_scale_8",
        "pf_scale_12",
        "pf_mean",
    }


def test_horizontal_loader_excludes_truth_and_ledger_requires_freeze(train, tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)

    frame = train.load_horizontal_without_truth("a", tmp_path)
    ledger = train.TruthAccessLedger()

    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    with pytest.raises(RuntimeError, match="requires a frozen prediction"):
        ledger.require_frozen()
    with pytest.raises(RuntimeError, match="frozen prediction"):
        train._require_frozen_prediction({})
    ledger.mark_frozen()
    ledger.require_frozen()


def test_metric_outputs_keep_scales_nonselective_when_x1p0_scales_absent(train) -> None:
    rows = 10
    truth = np.linspace(100.0, 109.0, rows)
    frame = pd.DataFrame(
        {
            "id": [f"w{i // 5}_{i}" for i in range(rows)],
            "well_id": ["w0"] * 5 + ["w1"] * 5,
            "true_tvt": truth,
            "fold": [0, 1] * 5,
            "raw_gr_observed": [True, False] * 5,
            "well_missing_fraction": [0.4] * 5 + [0.1] * 5,
            "md_since": [1100.0] + [100.0] * 9,
            "hidden_like_spatial": [True] + [False] * 9,
            "hidden_like_typewell_purged": [False, True] + [False] * 8,
            "saved_exp072_likpf_mean": truth + 1.0,
            "saved_exp209_hmm": truth + 0.5,
            "likpf_mean_x1p3": truth + 0.25,
            "likpf_scale_3_x1p3": truth + 0.3,
            "likpf_scale_5_x1p3": truth + 0.4,
            "likpf_scale_8_x1p3": truth + 0.5,
            "likpf_scale_12_x1p3": truth + 0.6,
        }
    )
    frame["candidate_hmm_50_50"] = 0.5 * (frame["likpf_mean_x1p3"] + frame["saved_exp209_hmm"])
    frame["parent_hmm_50_50"] = 0.5 * (frame["saved_exp072_likpf_mean"] + frame["saved_exp209_hmm"])

    primary, by_well, secondary, blend = train.build_metric_outputs(frame)

    overall = primary.loc[primary["scope"].eq("overall")].iloc[0]
    assert overall["improvement_ft"] == pytest.approx(0.75)
    assert len(by_well) == 2
    assert len(secondary) == 4
    assert not secondary["control_available"].any()
    assert secondary["comparison"].eq("candidate_only_saved_x1p0_scale_unavailable").all()
    assert blend.loc[blend["scope"].eq("overall"), "improvement_ft"].iloc[0] > 0.0


def test_notebook_sources_are_not_file_relative_or_submission_creating() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()

    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "shutil.copyfile" not in inference_source
    assert "sample_submission" not in inference_source
    assert "run_full_experiment(CONFIG)" in train_source
