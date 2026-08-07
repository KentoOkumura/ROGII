from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp429_self_gr_weak_boost_likelihood_pf_ablation"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
EXP223_SOURCE = (
    ROOT
    / "experiments"
    / "exp223_joint_typewell_self_gr_hmm_likelihood_probe"
    / "exact_hmm_smoother.py"
)
EXP400_SOURCE = (
    ROOT
    / "experiments"
    / "exp400_all_well_1p3_sigma_gr_likelihood_pf"
    / "exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_train.py"
)
PREFLIGHT_ASSET = EXP_DIR / "assets" / f"{EXP}_preflight_wells.csv"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP429_IMPORT_ONLY")
    os.environ["EXP429_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp429_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP429_IMPORT_ONLY", None)
        else:
            os.environ["EXP429_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp223() -> ModuleType:
    return load_module(EXP223_SOURCE, "exp223_surface_for_exp429")


@pytest.fixture(scope="module")
def exp400() -> ModuleType:
    return load_module(EXP400_SOURCE, "exp400_pf_for_exp429")


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    return load_module(INFERENCE_SOURCE, "exp429_inference_contract")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_horizontal(rows: int = 80, prefix_rows: int = 52) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    tvt = 100.0 + 0.35 * index
    tvt_input = tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    return pd.DataFrame(
        {
            "MD": index + 1.0,
            "Z": 0.2 * np.sin(index / 9.0),
            "GR": 60.0 + 12.0 * np.sin(index / 4.0) + 3.0 * np.cos(index / 11.0),
            "TVT_input": tvt_input,
        }
    )


def test_frozen_contract_and_execution_costs(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    counts = contract["execution_counts"]

    assert contract["primary_candidate"] == (
        "likpf_scale5_selfgr_boost_only_a070_c100"
    )
    assert contract["secondary_candidate"] == (
        "likpf_mean_selfgr_boost_only_a070_c100"
    )
    assert contract["self_gr_surface"]["alpha"] == 0.07
    assert contract["self_gr_surface"]["clip"] == 1.0
    assert contract["self_gr_surface"]["mode"] == "boost_only"
    assert counts["scientific_variants"] == 1
    assert counts["full_candidate_pf_well_runs"] == 773
    assert counts["full_seed_well_trajectories"] == 98_944
    assert counts["full_particle_starts"] == 49_472_000
    assert counts["parent_full_control_reruns"] == 0
    assert counts["lightgbm_configs"] == 0
    assert counts["boosters"] == 0
    assert counts["models"] == 0
    assert counts["gpu_runs"] == 0
    assert len(contract["scientific_contract_sha256"]) == 64
    stage = train.selected_stage(config)
    if stage is None:
        with pytest.raises(RuntimeError, match="no approved execution stage selected"):
            train.validate_scientific_contract(
                config,
                require_run_approval=True,
            )
    else:
        approved = train.validate_scientific_contract(
            config,
            require_run_approval=True,
        )
        assert approved["scientific_contract_sha256"] == (
            contract["scientific_contract_sha256"]
        )


def test_contract_rejects_any_scientific_grid(
    train: ModuleType,
    config: dict,
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["self_gr_surface"]["alpha"] = 0.15
    with pytest.raises(ValueError, match="model.self_gr_surface.alpha"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["pf"]["other_seed_weighting_scales_enabled"] = True
    with pytest.raises(
        ValueError,
        match="model.pf.other_seed_weighting_scales_enabled",
    ):
        train.validate_scientific_contract(broken)


def test_full_shard_requires_approved_preflight_before_raw_access(
    train: ModuleType,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train,
        "validate_scientific_contract",
        lambda *_args, **_kwargs: {},
    )

    def reject_missing_preflight(_config: dict) -> None:
        raise RuntimeError("approved preflight is required")

    monkeypatch.setattr(train, "load_preflight_summary", reject_missing_preflight)
    with pytest.raises(RuntimeError, match="approved preflight is required"):
        train.run_full_shard_stage(copy.deepcopy(config))


def test_exp223_surface_formula_is_exact(
    train: ModuleType,
    exp223: ModuleType,
    config: dict,
) -> None:
    horizontal = synthetic_horizontal()
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    grid = np.arange(70.0, 180.2, 0.2)
    peak = np.linspace(112.0, 124.0, len(eval_index))
    ours_config = config["model"]["self_gr_surface"]
    parent_config = {
        **ours_config,
        "gaussian_sigma_tvt": ours_config["gaussian_sigma_tvt_ft"],
        "typewell_agreement_sigma_tvt": (
            ours_config["typewell_agreement_sigma_tvt_ft"]
        ),
    }
    exp223.build_gr_window_descriptors = train.build_gr_window_descriptors

    observed = train.build_self_gr_likelihood_surface(
        horizontal,
        eval_index,
        grid,
        peak,
        ours_config,
    )
    expected = exp223.build_self_gr_likelihood_surface(
        horizontal,
        eval_index,
        grid,
        peak,
        parent_config,
    )

    np.testing.assert_array_equal(
        observed["centered_logl"],
        expected["centered_logl"],
    )
    np.testing.assert_array_equal(observed["quality"], expected["quality"])
    np.testing.assert_array_equal(observed["valid"], expected["valid"])
    np.testing.assert_array_equal(observed["peak_tvt"], expected["peak_tvt"])
    assert observed["prefix_anchor_count"] == expected["prefix_anchor_count"]


def test_alpha0_kernel_has_exact_exp400_rng_and_prediction_parity(
    train: ModuleType,
    exp400: ModuleType,
) -> None:
    md = np.arange(1.0, 9.0, dtype=np.float64)
    z = np.linspace(0.0, 0.7, len(md))
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0, 51.0])
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    parent_args = (
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
    expected = exp400._pf_lik_allseeds(*parent_args)
    observed = train._pf_selfgr_allseeds(
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        np.ones((len(md), len(grid_gr)), dtype=np.float32),
        np.ones(len(md), dtype=np.float32),
        90.0,
        0.2,
        0.0,
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

    for index in range(5):
        np.testing.assert_array_equal(observed[index], expected[index])
    assert int(observed[5].sum()) == 0
    assert float(observed[6].sum()) == 0.0


def test_positive_boost_changes_weights_without_changing_seed_policy(
    train: ModuleType,
) -> None:
    md = np.arange(1.0, 9.0, dtype=np.float64)
    z = np.zeros(len(md), dtype=np.float64)
    gr = np.linspace(48.0, 60.0, len(md))
    grid_gr = np.linspace(40.0, 70.0, 151)
    boost = np.tile(
        np.linspace(0.0, 1.0, len(grid_gr), dtype=np.float32),
        (len(md), 1),
    )
    args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        boost,
        np.ones(len(md), dtype=np.float32),
        90.0,
        0.2,
        0.07,
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
    first = train._pf_selfgr_allseeds(*args)
    second = train._pf_selfgr_allseeds(*args)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert int(first[5].sum()) > 0
    assert float(first[6].sum()) > 0.0
    assert np.isfinite(first[0]).all()


def test_fixed_preflight_asset_is_target_free_sha_ordered(
    train: ModuleType,
    config: dict,
) -> None:
    expected_sha = config["data"]["preflight_wells"]["expected_sha256"]
    assert hashlib.sha256(PREFLIGHT_ASSET.read_bytes()).hexdigest() == expected_sha
    asset = train.load_preflight_well_asset(config)

    assert len(asset) == 4
    assert asset["eligible_rows"].gt(0).all()
    assert "TVT" not in asset.columns
    assert asset["well_id"].tolist() == sorted(
        asset["well_id"],
        key=lambda well: hashlib.sha256(str(well).encode()).hexdigest(),
    )


def test_saved_float32_comparator_restores_artifact_semantics(
    train: ModuleType,
) -> None:
    serialized = pd.Series(["11183.766", "11022.869", "12161.080"])
    restored = train.restore_frozen_float32_column(
        serialized,
        label="synthetic saved float32",
    )
    expected = np.asarray([11183.766, 11022.869, 12161.080], dtype=np.float32)

    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored.to_numpy(), expected)
    assert np.array_equal(
        restored.to_numpy().view(np.uint32),
        expected.view(np.uint32),
    )


def test_lpt_sharding_is_deterministic_and_disjoint(train: ModuleType) -> None:
    manifest = pd.DataFrame(
        {
            "well_id": [f"w{index}" for index in range(12)],
            "rows": [100] * 12,
            "prefix_rows": [20] * 12,
            "suffix_rows": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8],
        }
    )
    first = train.assign_lpt_shards(manifest)
    second = train.assign_lpt_shards(manifest.sample(frac=1.0, random_state=7))

    left = first.set_index("well_id")["shard_index"].sort_index()
    right = second.set_index("well_id")["shard_index"].sort_index()
    pd.testing.assert_series_equal(left, right)
    assert sorted(first["shard_index"].unique().tolist()) == [0, 1, 2, 3]
    assert not first["well_id"].duplicated().any()


def test_shard_manifest_roundtrip_preserves_lpt_dtype(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame(
        {
            "well_id": ["000d7d20", "00bbac68"],
            "rows": [5278, 7559],
            "prefix_rows": [1442, 1545],
            "suffix_rows": [3836, 6014],
            "shard_index": np.asarray([0, 3], dtype=np.int8),
        }
    )
    path = tmp_path / "shard_manifest.csv"
    expected.to_csv(path, index=False)

    restored = train.read_shard_manifest(path)

    assert restored["shard_index"].dtype == np.dtype(np.int8)
    pd.testing.assert_frame_equal(restored, expected)


def test_metric_outputs_use_fixed_primary_and_secondary_controls(
    train: ModuleType,
) -> None:
    truth = np.arange(12, dtype=float) + 100.0
    frame = pd.DataFrame(
        {
            "id": [f"w{index // 6}_{index}" for index in range(12)],
            "well_id": ["w0"] * 6 + ["w1"] * 6,
            "true_tvt": truth,
            "fold": [0, 1, 2, 3, 4, 0] * 2,
            "raw_gr_observed": [True, False] * 6,
            "well_missing_fraction": [0.4] * 6 + [0.1] * 6,
            "md_since": [1100.0] + [100.0] * 11,
            "hidden_like_spatial": [True] + [False] * 11,
            "hidden_like_typewell_purged": [False, True] + [False] * 10,
            train.PRIMARY_CONTROL: truth + 1.0,
            train.SECONDARY_CONTROL: truth + 1.2,
            train.PRIMARY_CANDIDATE: truth + 0.5,
            train.SECONDARY_CANDIDATE: truth + 0.8,
            "saved_exp209_hmm": truth + 0.4,
        }
    )
    frame["candidate_hmm_50_50"] = 0.5 * (
        frame[train.PRIMARY_CANDIDATE] + frame["saved_exp209_hmm"]
    )
    frame["parent_hmm_50_50"] = 0.5 * (
        frame[train.PRIMARY_CONTROL] + frame["saved_exp209_hmm"]
    )

    primary, by_well, secondary, blend = train.build_metric_outputs(frame)

    assert train._scope_row(primary, "overall")["improvement_ft"] == pytest.approx(
        0.5
    )
    assert train._scope_row(secondary, "overall")[
        "improvement_ft"
    ] == pytest.approx(0.4)
    assert len(by_well) == 2
    assert train._scope_row(blend, "overall")["improvement_ft"] > 0.0


def test_truth_access_and_source_contracts_are_fail_closed(
    train: ModuleType,
    config: dict,
) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen prediction"):
        ledger.require_frozen()
    ledger.mark_frozen()
    ledger.require_frozen()

    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["preflight_run_approved"] is True
    active_stages = sum(
        bool(config["execution"][key])
        for key in ("run_preflight", "run_full", "run_merge")
    )
    assert active_stages <= 1
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    source = TRAIN_SOURCE.read_text()
    preflight_source = source.split("def run_preflight_stage", maxsplit=1)[1].split(
        "# %% [markdown]\n# ## 7.",
        maxsplit=1,
    )[0]
    assert "__file__" not in source
    assert "from exp429" not in source
    assert "alpha_or_clip_grid" in (EXP_DIR / "config.yaml").read_text()
    assert config["data"]["exp404_scale5_control"][
        "arithmetic_prediction_column"
    ] == "likpf_mean_x1p0"
    assert config["data"]["exp404_scale5_control"][
        "arithmetic_prediction_dtype"
    ] == "float32"
    assert config["guards"]["technical"][
        "require_preflight_alpha0_comparator"
    ] == "saved_exp404_likpf_mean_x1p0_bit_exact"
    assert config["guards"]["technical"][
        "require_preflight_alpha0_arithmetic_max_abs_parity_ft"
    ] == 0.00001
    assert config["guards"]["technical"][
        "require_preflight_alpha0_bit_exact"
    ] is True
    assert 'preflight["paths"]["exp404_scale5_control"]' in preflight_source
    assert 'preflight["paths"]["exp072_control"]' not in preflight_source
    assert "restore_frozen_float32_column" in preflight_source
    assert "run_selected_stage(CONFIG)" in source


def test_inference_is_explicitly_fail_closed(
    inference: ModuleType,
    config: dict,
) -> None:
    status = inference.validate_inference_is_disabled(config)

    assert status["implementation_enabled"] is True
    assert status["inference_enabled"] is False
    assert status["inference_approved"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False
