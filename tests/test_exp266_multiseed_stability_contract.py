from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp266_pf_ancc_pf_z_multiseed_stability_audit"
TRAIN_SOURCE = EXP_DIR / "exp266_pf_ancc_pf_z_multiseed_stability_audit_train.py"
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp072_exp063_full_replay_feature_cache"
    / "public_notebook_replay_audit.py"
)


def load_exp266_definition_namespace() -> dict[str, object]:
    added_numba_shim = False
    if "numba" not in sys.modules:
        shim = types.ModuleType("numba")

        def identity_njit(*args, **kwargs):  # noqa: ARG001
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

        shim.njit = identity_njit
        sys.modules["numba"] = shim
        added_numba_shim = True
    try:
        source = TRAIN_SOURCE.read_text()
        prefix = source.split("# ## 5. Per-well replay helpers", maxsplit=1)[0]
        tree = ast.parse(prefix)
        namespace: dict[str, object] = {}
        exec(compile(tree, str(TRAIN_SOURCE), "exec"), namespace)
        return namespace
    finally:
        if added_numba_shim:
            sys.modules.pop("numba", None)


def load_parent_module():
    added_numba_shim = False
    if "numba" not in sys.modules:
        shim = types.ModuleType("numba")

        def identity_njit(*args, **kwargs):  # noqa: ARG001
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

        shim.njit = identity_njit
        sys.modules["numba"] = shim
        added_numba_shim = True
    try:
        spec = importlib.util.spec_from_file_location("exp266_parent_exp072", PARENT_SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added_numba_shim:
            sys.modules.pop("numba", None)


def test_config_freezes_approved_cost_and_seed_contract() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["algorithms"] == ["pf_ancc", "pf_z"]
    assert config["model"]["pf"]["particles"] == 600
    assert config["model"]["pf"]["seed_count"] == 64
    assert config["model"]["pf"]["nested_seed_counts"] == [1, 4, 8, 16, 32, 64]
    assert config["execution"]["active_variant_count"] == 2
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["parent_control_retraining"] is False
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["inference"]["enabled"] is False


def test_train_source_is_self_contained_and_fail_closed() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "from settings import" not in source
    assert "__file__" not in source
    assert "original-seed exact parity failed; multiseed phase is blocked" in source
    assert 'stable_seed("pf_ancc", well)' in source
    assert 'stable_seed("pf_z", well)' in source
    assert 'stable_seed(EXPERIMENT_NAME, "train", algorithm, well, seed_index)' in source
    assert "true TVT" in source
    assert '"submission": False' in source
    assert "submission_enabled" not in source


def test_reference_loader_preserves_numeric_looking_well_ids(tmp_path: Path) -> None:
    exp266 = load_exp266_definition_namespace()
    well = "01234567"
    ids = [f"{well}_0", f"{well}_1"]
    pf_ancc_values = [101.123456789, 102.234567891]
    pf_z_values = [101.345678912, 102.456789123]

    base_path = tmp_path / "base.csv.gz"
    hmm_path = tmp_path / "hmm.csv.gz"
    exp226_path = tmp_path / "exp226.csv.gz"
    pd.DataFrame(
        {
            "id": ids,
            "well": [well, well],
            "target": [1.0, 2.0],
            "last_known_tvt": [100.0, 100.0],
            "md_since": [1.0, 2.0],
            "pf_ancc": pf_ancc_values,
            "pf_z": pf_z_values,
            "likpf_mean_d": [1.0, 2.0],
        }
    ).to_csv(base_path, index=False)
    pd.DataFrame({"id": ids, "hmm_mean_tvt": [101.0, 102.0]}).to_csv(
        hmm_path, index=False
    )
    pd.DataFrame(
        {"well_id": [well, well], "row_idx": [0, 1], "tvt_pred": [101.0, 102.0]}
    ).to_csv(exp226_path, index=False)

    paths = {
        "base.csv.gz": base_path,
        "hmm.csv.gz": hmm_path,
        "exp226.csv.gz": exp226_path,
    }
    exp266["resolve_artifact"] = lambda filename, _candidates: paths[filename]
    exp266["assert_file_sha"] = lambda *_args, **_kwargs: {}
    config = {
        "data": {
            "exp072": {
                "filename": "base.csv.gz",
                "expected_sha256": "unused",
                "expected_decompressed_sha256": "unused",
            },
            "exp209": {
                "filename": "hmm.csv.gz",
                "expected_sha256": "unused",
                "expected_decompressed_sha256": "unused",
            },
            "exp226": {
                "filename": "exp226.csv.gz",
                "expected_decompressed_sha256": "unused",
            },
        },
        "validation": {"expected_rows": 2, "expected_wells": 1, "strong_margin_ft": 2.0},
    }

    rows, by_well, _ = exp266["load_reference_surface"](config)

    assert rows["well"].tolist() == [well, well]
    assert rows["pf_ancc"].dtype == np.dtype(np.float32)
    assert rows["pf_z"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        rows["pf_ancc"].to_numpy(), np.asarray(pf_ancc_values, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        rows["pf_z"].to_numpy(), np.asarray(pf_z_values, dtype=np.float32)
    )
    assert by_well["well"].tolist() == [well]


def test_exact_seeded_kernels_match_exp072_on_synthetic_paths() -> None:
    exp266 = load_exp266_definition_namespace()
    parent = load_parent_module()

    n = 21
    md = np.linspace(100.0, 180.0, n, dtype=np.float64)
    z = np.linspace(-10.0, -6.0, n, dtype=np.float64)
    gr = np.linspace(40.0, 85.0, n, dtype=np.float64)
    gr_smooth = np.linspace(42.0, 82.0, n, dtype=np.float64)
    grid = np.linspace(35.0, 90.0, 301, dtype=np.float64)
    smooth_grid = np.linspace(36.0, 88.0, 301, dtype=np.float64)
    seed = 1234567

    parent_ancc = parent._pf_ancc_seeded(
        seed,
        md,
        z,
        gr,
        grid,
        11000.0,
        0.2,
        20.0,
        11500.0,
        0.05,
        64,
        parent.ANCC_ALPHA,
        parent.ANCC_RN,
        parent.ANCC_PN,
        parent.ANCC_IS,
        parent.ANCC_RP,
        parent.ANCC_RR,
        parent.PF_RESAMP,
    )
    exp266_ancc = exp266["_pf_ancc_seeded"](
        seed,
        md,
        z,
        gr,
        grid,
        11000.0,
        0.2,
        20.0,
        11500.0,
        0.05,
        64,
        parent.ANCC_ALPHA,
        parent.ANCC_RN,
        parent.ANCC_PN,
        parent.ANCC_IS,
        parent.ANCC_RP,
        parent.ANCC_RR,
        parent.PF_RESAMP,
    )
    assert np.array_equal(parent_ancc[0], exp266_ancc[0])
    assert np.array_equal(parent_ancc[1], exp266_ancc[1])

    parent_pf_z = parent._pf_z_seeded(
        seed,
        md,
        z,
        gr,
        gr_smooth,
        grid,
        smooth_grid,
        11000.0,
        0.2,
        20.0,
        11500.0,
        0.03,
        -1.0,
        0.0,
        0.1,
        64,
        parent.PF_MOM,
        parent.PF_VN,
        parent.PF_PN,
        parent.PF_GR_WT,
        parent.PF_ROUGH_P,
        parent.PF_ROUGH_V,
        parent.PF_RESAMP,
    )
    exp266_pf_z = exp266["_pf_z_seeded"](
        seed,
        md,
        z,
        gr,
        gr_smooth,
        grid,
        smooth_grid,
        11000.0,
        0.2,
        20.0,
        11500.0,
        0.03,
        -1.0,
        0.0,
        0.1,
        64,
        parent.PF_MOM,
        parent.PF_VN,
        parent.PF_PN,
        parent.PF_GR_WT,
        parent.PF_ROUGH_P,
        parent.PF_ROUGH_V,
        parent.PF_RESAMP,
    )
    assert np.array_equal(parent_pf_z[0], exp266_pf_z[0])
    assert np.array_equal(parent_pf_z[1], exp266_pf_z[1])
