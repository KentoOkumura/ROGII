from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp271_pf_ancc_small_seed_mean_candidate_audit"
TRAIN_PATH = EXP_DIR / "exp271_pf_ancc_small_seed_mean_candidate_audit_train.py"
PARENT_PATH = (
    ROOT
    / "experiments"
    / "exp266_pf_ancc_pf_z_multiseed_stability_audit"
    / "exp266_pf_ancc_pf_z_multiseed_stability_audit_train.py"
)


def identity_njit(*args, **kwargs):  # noqa: ARG001
    if args and callable(args[0]) and len(args) == 1:
        return args[0]

    def decorate(function):
        return function

    return decorate


def load_train_module(monkeypatch):
    numba = types.ModuleType("numba")
    numba.njit = identity_njit
    monkeypatch.setitem(sys.modules, "numba", numba)
    spec = importlib.util.spec_from_file_location("exp268_train_contract", TRAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parent_kernel_namespace() -> dict[str, object]:
    tree = ast.parse(PARENT_PATH.read_text())
    wanted = {"_interp1", "_resamp", "_pf_ancc", "_pf_ancc_seeded"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace: dict[str, object] = {"np": np, "njit": identity_njit}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARENT_PATH), "exec"), namespace)
    return namespace


def test_seed_vector_reuses_exact_exp266_namespace(monkeypatch):
    module = load_train_module(monkeypatch)
    well = "11d0f5ac"
    actual = module.seed_vector(well, 8)

    def stable(*parts: object) -> int:
        key = "::".join(str(part) for part in parts)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:16], 16) % 2_147_483_647 + 1

    expected = np.asarray(
        [
            stable("pf_ancc", well),
            *[
                stable(
                    "exp266_pf_ancc_pf_z_multiseed_stability_audit",
                    "train",
                    "pf_ancc",
                    well,
                    seed_index,
                )
                for seed_index in range(1, 8)
            ],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(actual)) == 8


def test_pf_ancc_kernel_is_exact_exp266_equation(monkeypatch):
    module = load_train_module(monkeypatch)
    parent = load_parent_kernel_namespace()
    rows = 24
    md = np.arange(rows, dtype=np.float64) + 1000.0
    z = np.linspace(5000.0, 5003.0, rows, dtype=np.float64)
    gr = np.linspace(35.0, 85.0, rows, dtype=np.float64)
    grid = np.linspace(30.0, 90.0, 151, dtype=np.float64)
    args = (
        md,
        z,
        gr,
        grid,
        0.0,
        0.2,
        30.0,
        5010.0,
        0.01,
        600,
        0.998,
        0.002,
        0.005,
        0.3,
        0.1,
        0.001,
        0.5,
    )
    actual_prediction, actual_std = module._pf_ancc_seeded(123456, *args)
    expected_prediction, expected_std = parent["_pf_ancc_seeded"](123456, *args)
    np.testing.assert_array_equal(actual_prediction, expected_prediction)
    np.testing.assert_array_equal(actual_std, expected_std)


def test_oracle_scope_reports_unique_new_candidate(monkeypatch):
    module = load_train_module(monkeypatch)
    target = np.asarray([0.0, 0.0, 10.0, 10.0])
    values = np.asarray(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [10.0, 14.0],
            [10.0, 14.0],
        ],
        dtype=np.float32,
    )
    wells = np.asarray(["a", "a", "b", "b"])
    prediction, summary = module.oracle_prediction(
        values,
        target,
        wells,
        [0, 1],
        ["core", "mean4"],
        scope="block_2",
        core_count=1,
        tie_tolerance=1e-6,
    )
    np.testing.assert_array_equal(prediction, np.asarray([0.0, 0.0, 10.0, 10.0]))
    assert summary["unit_count"] == 2
    assert summary["unique_best_units"]["mean4"] == 1
    assert summary["unique_best_rows"]["mean4"] == 2


def test_exp266_per_well_parity_is_fail_closed(monkeypatch):
    module = load_train_module(monkeypatch)
    generated = pd.DataFrame(
        {
            "well": ["a", "a", "b", "b"],
            "target_tvt": [0.0, 2.0, 10.0, 12.0],
            "pf_ancc_seed_mean_4": [1.0, 1.0, 9.0, 13.0],
            "pf_ancc_seed_mean_8": [0.0, 2.0, 8.0, 14.0],
        }
    )
    rows = []
    for count, column in [(4, "pf_ancc_seed_mean_4"), (8, "pf_ancc_seed_mean_8")]:
        for well, frame in generated.groupby("well"):
            error = frame[column].to_numpy() - frame["target_tvt"].to_numpy()
            rows.append(
                {
                    "well": well,
                    "algorithm": "pf_ancc",
                    "aggregation": "mean",
                    "seed_count": count,
                    "rows": len(frame),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
    expected = pd.DataFrame(rows)
    parity = module.exp266_parity_readout(generated, expected, 0.0)
    assert parity["passed"].all()
    expected.loc[0, "rmse"] += 0.1
    try:
        module.exp266_parity_readout(generated, expected, 1e-6)
    except RuntimeError as error:
        assert "RMSE parity failed" in str(error)
    else:
        raise AssertionError("changed exp266 RMSE must fail closed")


def test_raw_target_is_loaded_exactly_after_candidate_freeze(monkeypatch, tmp_path):
    module = load_train_module(monkeypatch)
    well = "000d7d20"
    raw_tvt = np.asarray([1000.123456789, 1001.234567891, 1002.345678912])
    pd.DataFrame(
        {
            "TVT": raw_tvt,
            "TVT_input": [1000.1234, np.nan, np.nan],
        }
    ).to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    targets = module.load_raw_targets_after_candidate_freeze(tmp_path, [well])
    assert targets["id"].tolist() == [f"{well}_1", f"{well}_2"]
    np.testing.assert_array_equal(targets["target_tvt"].to_numpy(), raw_tvt[1:])

    source = TRAIN_PATH.read_text()
    assert source.index("candidate_path = write_csv(") < source.index(
        "raw_targets = load_raw_targets_after_candidate_freeze(train_dir, wells)"
    )


def test_config_has_zero_boosters_and_disabled_inference(monkeypatch):
    module = load_train_module(monkeypatch)
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    module.validate_execution_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["parent_control_retraining"] is False
    assert config["inference"]["enabled"] is False
    assert config["execution"]["submission_enabled"] is False
