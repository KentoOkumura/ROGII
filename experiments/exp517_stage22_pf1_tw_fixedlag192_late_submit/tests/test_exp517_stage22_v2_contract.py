from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parents[1]
TRAIN = EXP_DIR / (
    "exp517_stage22_pf1_tw_fixedlag192_late_submit_"
    "stage22_v2_compact_selfcontained_train.py"
)
INFERENCE = EXP_DIR / (
    "exp517_stage22_pf1_tw_fixedlag192_late_submit_"
    "stage22_v2_compact_selfcontained_inference.py"
)
V1 = EXP_DIR / "exp517_stage22_pf1_tw_fixedlag192_late_submit_compact_selfcontained_inference.py"
EXPECTED_BANKS = ["pf_1", "pf_2", "pf_3", "r0_seed32", "r1_seed32"]
PF_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function(path: Path, name: str):
    tree = ast.parse(_source(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    namespace = {
        "np": np,
        "pd": pd,
        "PF_BANKS": EXPECTED_BANKS,
        "PF_OFFSETS": PF_OFFSETS,
        "BEAMS": BEAMS,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_v1_failure_source_is_preserved() -> None:
    assert V1.is_file()
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == (
        "f01b011475a2c205658e06edac9df6ba435e9296590e4561b75ff40053113b29"
    )


def test_corrected_contract_has_five_banks_and_stage22_only_overrides() -> None:
    source = _source(TRAIN)
    assert f"PF_BANKS = {EXPECTED_BANKS!r}" in source
    required = {
        'p["smooth_mode"] = PF_SMOOTH_MODE',
        'p["smooth_lag"] = PF_SMOOTH_LAG',
        'p["use_anchor"] = False',
        'p["use_phys"] = False',
        'p["robust_nu"] = 0.0',
        'p["temper_beta"] = 1.0',
        'p["_physics"] = False',
        'p["_w_nn"] = 0.0',
        'p["_ps_combo_tau"] = 0.0',
    }
    assert all(line in source for line in required)
    assert 'PF_SMOOTH_MODE = "fixedlag"' in source
    assert "PF_SMOOTH_LAG = 192" in source
    assert "PF_N_SEEDS = 32" in source


def test_training_quantity_is_one_variant_25_base_models_and_no_control() -> None:
    source = _source(TRAIN)
    assert '"scientific_variants": 1' in source
    assert '"pf_banks": 5' in source
    assert '"lightgbm_configs": 3' in source
    assert '"catboost_configs": 2' in source
    assert '"folds": 5' in source
    assert '"base_models": 25' in source
    assert '"ridge_models": 5' in source
    assert '"control_reruns": 0' in source
    assert "for config_index, params in enumerate(lgb_params)" in source
    assert "for config_index, params in enumerate(cb_params)" in source


def test_five_pf_feature_augmentation_and_alias_contract() -> None:
    augment = _function(TRAIN, "augment_stage22_frame")
    n = 4
    last = np.full(n, 100.0, np.float32)
    base = pd.DataFrame(
        {
            "id": [f"well_{idx}" for idx in range(n)],
            "well": ["well"] * n,
            "target": np.arange(n, dtype=np.float32),
            "last_known_tvt": last,
            "tvtF_ANCC": last + 2.0,
            "tvt_dense_d": np.full(n, 3.0, np.float32),
            "pf_z": last + 4.0,
            "pf_ancc": last,
            "pf_ancc_std": np.ones(n, np.float32),
            "pf_ancc_delta": np.zeros(n, np.float32),
            "pf_vs_z": np.zeros(n, np.float32),
            "pf_vs_spatial": np.zeros(n, np.float32),
            "pf_vs_dense": np.zeros(n, np.float32),
            "sig_std": np.zeros(n, np.float32),
            "sig_mean_d": np.zeros(n, np.float32),
            "sc8_d": np.zeros(n, np.float32),
            "sc15_d": np.zeros(n, np.float32),
            "sc25_d": np.zeros(n, np.float32),
            "sc_ens_d": np.zeros(n, np.float32),
        }
    )
    for *_, tag in BEAMS:
        base[f"beam_{tag}_d"] = np.zeros(n, np.float32)
    for offset in PF_OFFSETS:
        base[f"tdpf{int(offset)}"] = np.zeros(n, np.float32)

    pf = pd.DataFrame({"id": base["id"], "well": base["well"]})
    for idx in range(1, 6):
        mean = last + np.float32(idx)
        pf[f"pf_ancc_{idx}"] = mean
        pf[f"pf_ancc_std_{idx}"] = np.full(n, idx / 10, np.float32)
        pf[f"pf_ancc_delta_{idx}"] = mean - last
        for offset in PF_OFFSETS:
            pf[f"tdpf{int(offset)}_{idx}"] = np.full(n, idx + offset / 100, np.float32)

    out, features = augment(base, pf)
    assert np.array_equal(out["pf_ancc"], out["pf_ancc_1"])
    assert np.array_equal(out["pf_ancc_std"], out["pf_ancc_std_1"])
    assert np.array_equal(out["pf_ancc_delta"], out["pf_ancc_delta_1"])
    for idx in range(1, 6):
        assert f"pf_ancc_{idx}" in features
        assert f"pf_ancc_std_{idx}" in features
        assert f"pf_ancc_delta_{idx}" in features
        assert f"pf_vs_z_{idx}" in features
        assert f"pf_vs_spatial_{idx}" in features
        assert f"pf_vs_dense_{idx}" in features
        assert f"tdpf0_{idx}" in features
    assert "target" not in features
    assert np.isfinite(out[features].to_numpy()).all()
    assert np.all(out["sig_std"].to_numpy() > 0)


def test_public_decode_formula_is_not_direct_pf() -> None:
    apply_pp = _function(TRAIN, "apply_public_postprocess")
    frame = pd.DataFrame({"md_since": [0.0, 85.0, 170.0]})
    model = np.array([10.0, 10.0, 10.0])
    pf = np.array([20.0, 20.0, 20.0])
    actual = apply_pp(frame, model, pf)
    expected = (0.91 * model + 0.09 * pf) * (1.0 - np.exp(-frame["md_since"].to_numpy() / 85.0))
    assert np.allclose(actual, expected)
    assert actual[0] == 0.0
    assert not np.allclose(actual, pf)


def test_inference_requires_saved_model_manifest_and_runtime_sample_alignment() -> None:
    source = _source(INFERENCE)
    assert 'expected_counts = {"base_models": 25, "ridge_models": 5, "control_reruns": 0}' in source
    assert 'if features != manifest["features"]' in source
    assert 'validate="one_to_one"' in source
    assert 'submission["id"].equals(sample["id"])' in source
    assert 'submission.to_csv(submission_path' in source
