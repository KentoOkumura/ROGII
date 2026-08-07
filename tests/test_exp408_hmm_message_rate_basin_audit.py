from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if importlib.util.find_spec("numba") is None:
    numba_stub = types.ModuleType("numba")
    numba_stub.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)

    def _njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

    numba_stub.njit = _njit
    numba_stub.prange = range
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.get_num_threads = lambda: 1
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "exp408_hmm_message_rate_basin_audit"
MODULE_PATH = EXP / "exp408_hmm_message_rate_basin_audit_compact_selfcontained_train.py"
REFERENCE_PATH = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp408 = load_module(MODULE_PATH, "exp408_test_module")
exp209 = load_module(REFERENCE_PATH, "exp209_reference_module")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_and_execution_count_are_fixed() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    assert config["experiment"]["route"] == "pf_beam"
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    assert config["inference"]["enabled"] is False
    assert config["execution"]["run_stage"] == "none"
    assert config["execution"]["kaggle_execution_approved"] is False
    assert config["execution"]["rerun_enabled"] is False

    approved_config = yaml.safe_load((EXP / "config.yaml").read_text())
    approved_config["execution"]["kaggle_execution_approved"] = True
    assert exp408.validate_execution_contract(approved_config) == {
        "active_hmm_variants": 1,
        "hmm_well_runs": 450,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }


def test_fixed_assets_match_contract() -> None:
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    wells_path = EXP / "assets" / "target_wells.csv"
    episodes_path = EXP / "assets" / "persistent_offset_episodes.csv"
    wells = pd.read_csv(wells_path)
    episodes = pd.read_csv(episodes_path)
    assert len(wells) == wells["well"].nunique() == 450
    assert len(episodes) == 638
    assert int(episodes["rows"].sum()) == 807_710
    assert file_sha256(wells_path) == config["data"]["target_wells"]["expected_sha256"]
    assert (
        file_sha256(episodes_path)
        == config["data"]["persistent_episodes"]["expected_sha256"]
    )


def test_decoder_interface_cannot_receive_truth_or_episode() -> None:
    signature = inspect.signature(exp408._hmm2_message_sufficient_statistics)
    forbidden = {"truth", "tvt", "error", "episode", "fold", "hidden"}
    assert not forbidden.intersection(signature.parameters)
    source = inspect.getsource(exp408.run_current_hmm_messages)
    assert "truth" not in source.lower()
    assert "episode" not in source.lower()


def test_prepare_hmm_inputs_accepts_competition_raw_without_id() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(9, dtype=np.float64),
            "Z": np.linspace(0.0, 0.8, 9),
            "GR": np.linspace(50.0, 58.0, 9),
            "TVT_input": [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, np.nan, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(-50.0, 50.0, 101),
            "GR": np.linspace(40.0, 80.0, 101),
        }
    )
    config = yaml.safe_load((EXP / "config.yaml").read_text())
    prepared = exp408.prepare_hmm_inputs(
        horizontal,
        typewell,
        **exp408.fixed_hmm_kwargs(config),
    )
    assert prepared["eval_id"] is None
    assert len(prepared["eval_index"]) == 3


def test_message_kernel_matches_exp270_posterior_mean() -> None:
    rng = np.random.default_rng(123)
    row_count = 64
    emission = rng.normal(0.0, 0.3, size=(row_count, 13)).astype(np.float32)
    dm = 1.0 + (np.arange(row_count, dtype=np.float64) % 5) * 0.2
    dz = 0.25 * np.sin(np.arange(row_count, dtype=np.float64) / 7.0)
    rates = np.linspace(-0.06, 0.06, 9, dtype=np.float64)
    common = (
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        6.0,
        0.75,
        0.01,
        0.01,
        1.0,
        0.998,
    )
    reference_position, reference_loglik = exp209._hmm2_fb(emission, *common)
    result = exp408._hmm2_message_sufficient_statistics(emission, *common)
    observed_position = result[0]
    observed_joint = result[1]
    observed_loglik = result[2]
    np.testing.assert_allclose(
        observed_position,
        reference_position,
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        observed_position.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        observed_joint.sum(axis=(1, 2)),
        1.0,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert abs(float(observed_loglik) - float(reference_loglik)) <= 2.0e-6
    for index in (3, 7):
        np.testing.assert_allclose(
            result[index].sum(axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-6,
        )
    for index in (6, 10):
        np.testing.assert_allclose(
            result[index].sum(axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-6,
        )


def test_interval_sum_and_rate_neighborhood() -> None:
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    observed = exp408.interval_sum_by_row(
        matrix,
        grid,
        np.array([1.5, 2.5]),
        0.6,
    )
    np.testing.assert_allclose(observed, [0.5, 0.3], rtol=0.0, atol=1.0e-12)


def test_rate_neighborhood_mass_exact() -> None:
    rates = np.array([-0.1, 0.0, 0.1, 0.2])
    mass = np.array([[0.1, 0.2, 0.3, 0.4]])
    observed = exp408.rate_neighborhood_mass(
        mass,
        rates,
        np.array([0.1]),
        1,
    )
    np.testing.assert_allclose(observed, [0.9], rtol=0.0, atol=1.0e-12)


def test_local_full_run_is_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("EXP408_ALLOW_LOCAL", raising=False)
    monkeypatch.setattr(exp408, "KAGGLE_WORKING_ROOT", Path("/definitely/missing"))
    try:
        exp408.require_kaggle_runtime()
    except RuntimeError as exc:
        assert "Kaggle CPU" in str(exc)
    else:
        raise AssertionError("local full run was not blocked")
