from __future__ import annotations

import importlib.util
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp309_well_adaptive_transition_noise"
TRAIN_SOURCE = EXP_DIR / "exp309_well_adaptive_transition_noise_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp309_well_adaptive_transition_noise_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP309_IMPORT_ONLY")
    os.environ["EXP309_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP309_IMPORT_ONLY", None)
        else:
            os.environ["EXP309_IMPORT_ONLY"] = previous


def test_contract_is_implemented_but_execution_remains_parent_blocked() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_contract")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_variants") == [
        "robust_prefix_rate_diffusion"
    ]
    assert module.get_nested(config, "model.execution_counts") == {
        "variants": 1,
        "hmm_well_runs": 773,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.parent_dependency_frozen") is False
    assert module.get_nested(config, "execution.kaggle_push_approved") is False
    with pytest.raises(RuntimeError, match="parent exp308 dependency"):
        module.validate_scientific_contract(config, require_run_approval=True)


def test_transition_scale_matches_preregistered_formula() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_transition_formula")
    rows = 34
    md = np.arange(rows, dtype=float) * 4.0
    rates = 0.02 + 0.0004 * np.square(np.arange(rows - 1, dtype=float))
    u = np.r_[100.0, 100.0 + np.cumsum(4.0 * rates)]
    frame = pd.DataFrame({"MD": md, "Z": np.zeros(rows), "TVT_input": u})
    audit = module.compute_prefix_transition_scale_audit(frame)
    innovations = np.diff(rates) / 2.0
    raw = 1.4826 * np.median(np.abs(innovations - np.median(innovations)))
    alpha = len(innovations) / (len(innovations) + 100.0)
    unclip = math.exp(alpha * math.log(max(raw, 1.0e-6)) + (1.0 - alpha) * math.log(0.002))
    assert audit["valid_rate_pairs"] == rows - 1
    assert audit["valid_rate_innovations"] == rows - 2
    assert audit["raw_sig_r"] == pytest.approx(raw)
    assert audit["shrinkage_alpha"] == pytest.approx(alpha)
    assert audit["unclipped_sig_r"] == pytest.approx(unclip)
    assert audit["sig_r"] == pytest.approx(np.clip(unclip, 0.001, 0.004))
    assert audit["sig_r_fallback"] is False


def test_transition_scale_fallback_and_clip_are_fixed() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_transition_guards")
    short = pd.DataFrame(
        {
            "MD": np.arange(10, dtype=float),
            "Z": np.zeros(10),
            "TVT_input": np.arange(10, dtype=float),
        }
    )
    fallback = module.compute_prefix_transition_scale_audit(short)
    assert fallback["valid_rate_innovations"] == 8
    assert fallback["sig_r"] == 0.002
    assert fallback["sig_r_fallback"] is True

    rows = 40
    constant_rate = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "Z": np.zeros(rows),
            "TVT_input": np.arange(rows, dtype=float) * 0.03,
        }
    )
    clipped = module.compute_prefix_transition_scale_audit(constant_rate)
    assert clipped["valid_rate_innovations"] == rows - 2
    assert clipped["sig_r"] == 0.001
    assert clipped["sig_r_clip_low"] is True


def test_missing_gr_confidence_matches_exp308_contract() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_confidence")
    raw_gr = np.array([10.0, np.nan, np.nan, np.nan, 20.0, np.nan], dtype=float)
    weights, audit = module.compute_missing_gr_confidence(raw_gr)
    expected = np.array([1.0, 2 ** (-1 / 8), 2 ** (-2 / 8), 2 ** (-1 / 8), 1.0, 2 ** (-1 / 8)])
    assert weights == pytest.approx(expected)
    assert audit["raw_missing_gr_rows"] == 4
    assert audit["no_finite_gr"] is False
    no_finite, no_finite_audit = module.compute_missing_gr_confidence(
        np.full(5, np.nan)
    )
    assert no_finite == pytest.approx(np.full(5, 0.25))
    assert no_finite_audit["no_finite_gr"] is True


def test_prepare_and_exact_hmm_use_parent_observation_and_adaptive_sig_r() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_synthetic_hmm")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    rows = 34
    known_rows = 29
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.2,
            "GR": np.linspace(40.0, 60.0, rows),
            "TVT_input": np.r_[100.0 + np.arange(known_rows, dtype=float), [np.nan] * 5],
        }
    )
    horizontal.loc[[30, 31], "GR"] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 160.0, 161),
            "GR": np.linspace(35.0, 65.0, 161),
        }
    )
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    assert prepared["observation_weight"].shape == (5,)
    assert prepared["observation_weight"][1] < 1.0
    assert 0.001 <= prepared["transition_audit"]["sig_r"] <= 0.004
    result = module.run_exact_hmm_variant(
        prepared,
        float(prepared["scale_audit"]["finite_mad"]),
        float(prepared["transition_audit"]["sig_r"]),
        config,
    )
    assert result["mean"].shape == (5,)
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["posterior_row_sum_max_abs_error"] < 1.0e-6


def test_horizontal_loader_and_truth_attachment_remain_fail_closed(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp309_truth_free")
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = module.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})


def test_parent_dependency_preflight_checks_gate_and_prediction_sha(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp309_dependency")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    wells = ["a", "b"]
    identities = [(well, row_idx) for well in wells for row_idx in range(5)]
    parent_path = tmp_path / "parent.csv.gz"
    likpf_path = tmp_path / "likpf.csv.gz"
    fold_path = tmp_path / "fold.csv.gz"
    hidden_path = tmp_path / "hidden.csv"
    parent_column = config["data"]["saved_parent"]["prediction_column"]
    pd.DataFrame(
        {
            "id": [f"{well}_{row_idx}" for well, row_idx in identities],
            "well_id": [well for well, _ in identities],
            "row_idx": [row_idx for _, row_idx in identities],
            parent_column: np.ones(10),
        }
    ).to_csv(parent_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "id": [f"{well}_{row_idx}" for well, row_idx in identities],
            "well": [well for well, _ in identities],
            "md_since": np.arange(10, dtype=float),
            "likpf_mean": np.ones(10),
        }
    ).to_csv(likpf_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": [well for well, _ in identities],
            "row_idx": [row_idx for _, row_idx in identities],
            "suffix_offset": [row_idx for _, row_idx in identities],
            "fold": np.arange(10) % 5,
            "tvt_geop": np.zeros(10),
            "tvt_true": np.zeros(10),
        }
    ).to_csv(fold_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": wells,
            "verification_like_spatial_role": ["valid", "train"],
            "verification_like_typewell_purged_role": ["train", "valid"],
        }
    ).to_csv(hidden_path, index=False)
    parent_sha = module.inspect_gzip_csv(parent_path)["decompressed_sha256"]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "experiment": "exp308_imputed_gr_confidence_downweight",
                "status": "train_side_missing_weight_gate_passed_no_automatic_downstream",
                "promotion_gate": {"passed": True},
                "prediction_sha256": {"decompressed_sha256": parent_sha},
            }
        )
    )
    config["validation"].update({"expected_rows": 10, "expected_wells": 2})
    config["data"]["saved_parent"].update(
        {
            "prediction_filename": parent_path.name,
            "metrics_filename": metrics_path.name,
            "candidates": [str(tmp_path)],
            "expected_prediction_decompressed_sha256": parent_sha,
        }
    )
    config["references"]["parent_prediction_decompressed_sha256"] = parent_sha
    config["data"]["saved_likpf"].update(
        {
            "filename": likpf_path.name,
            "candidates": [str(tmp_path)],
            "expected_decompressed_sha256": module.inspect_gzip_csv(likpf_path)[
                "decompressed_sha256"
            ],
        }
    )
    config["data"]["fold_assignment"].update(
        {
            "filename": fold_path.name,
            "candidates": [str(tmp_path)],
            "expected_decompressed_sha256": module.inspect_gzip_csv(fold_path)[
                "decompressed_sha256"
            ],
        }
    )
    config["data"]["hidden_like_assignment"].update(
        {
            "filename": hidden_path.name,
            "candidates": [str(tmp_path)],
            "expected_sha256": module.sha256_path(hidden_path),
        }
    )
    report = module.preflight_controls_and_assignments(config)
    assert report["parent_metrics"]["promotion_gate_passed"] is True
    broken = deepcopy(config)
    broken["data"]["saved_parent"]["expected_prediction_decompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent prediction"):
        module.preflight_controls_and_assignments(broken)


def test_promotion_gate_enforces_technical_and_accuracy_guards() -> None:
    module = load_module(TRAIN_SOURCE, "exp309_gate")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    config["validation"].update({"expected_rows": 10, "expected_wells": 2})
    config["model"]["execution_counts"]["hmm_well_runs"] = 2
    config["references"].update(
        {"parent_rmse": 1.0, "likpf_rmse": 1.0, "parent_likpf_50_50_rmse": 1.0}
    )
    scopes = [
        "overall",
        *[f"fold_{fold}" for fold in range(5)],
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    paired = pd.DataFrame(
        [
            {
                "variant": module.VARIANT_ORDER[0],
                "comparison": comparison,
                "scope": scope,
                "candidate_rmse": 0.8 if comparison == "direct" else 0.9,
                "control_rmse": 1.0,
                "delta_rmse_candidate_minus_control": -0.2 if comparison == "direct" else -0.1,
                "improvement_ft": 0.2 if comparison == "direct" else 0.1,
            }
            for comparison in ("direct", "fixed_likpf_50_50")
            for scope in scopes
        ]
    )
    by_well = pd.DataFrame(
        [
            {
                "variant": module.VARIANT_ORDER[0],
                "comparison": comparison,
                "well_id": well,
                "candidate_rmse": 0.8 if comparison == "direct" else 0.9,
                "control_rmse": 1.0,
                "delta_rmse_candidate_minus_control": -0.2 if comparison == "direct" else -0.1,
            }
            for comparison in ("direct", "fixed_likpf_50_50")
            for well in ("a", "b")
        ]
    )
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 5 + ["b"] * 5,
            "true_tvt": np.zeros(10),
            "parent_hmm_tvt": np.ones(10),
            "likpf_mean": np.ones(10),
            f"{module.VARIANT_ORDER[0]}_hmm_tvt": np.full(10, 0.8),
            f"{module.VARIANT_ORDER[0]}_likpf_50_50": np.full(10, 0.9),
        }
    )
    runtime = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "variant": [module.VARIANT_ORDER[0]] * 2,
            "posterior_row_sum_max_abs_error": [0.0, 0.0],
        }
    )
    audit = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "gr_finite_mad": [20.0, 20.0],
            "sig_r": [0.0018, 0.0022],
            "confidence_confidence_min": [0.25, 0.25],
            "confidence_confidence_mean": [0.9, 0.9],
            "sig_r_fallback": [False, False],
            "sig_r_clip_low": [False, False],
            "sig_r_clip_high": [False, False],
        }
    )
    preflight = {
        "raw_train": {"content_sha256": "a" * 64},
        "controls": {
            "parent_metrics": {"promotion_gate_passed": True},
            "saved_parent": {"decompressed_sha256": "b" * 64},
        },
    }
    gate = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, audit, preflight, 1.0, config
    )
    assert gate["passed"] is True
    audit.loc[0, "sig_r_fallback"] = True
    failed = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, audit, preflight, 1.0, config
    )
    assert failed["passed"] is False
    assert failed["technical_gate"]["sig_r_fallback_fraction"] == 0.5


def test_inference_is_explicitly_disabled() -> None:
    module = load_module(INFERENCE_SOURCE, "exp309_inference")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    contract = module.validate_disabled_inference(config)
    assert contract["mode"] == "disabled_train_side_transition_audit_only"
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        module.stop_disabled_inference(config)
