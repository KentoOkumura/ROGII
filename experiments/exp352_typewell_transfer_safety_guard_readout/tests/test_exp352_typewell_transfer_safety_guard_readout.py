from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp352_typewell_transfer_safety_guard_readout"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp352_typewell_transfer_safety_guard_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp352_typewell_transfer_safety_guard_readout_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    payload = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def train_module():
    return load_module(TRAIN_SOURCE, "exp352_train_contract")


def test_config_locks_implementation_only_gr_unit_and_zero_model_contract(
    train_module,
) -> None:
    config = load_config()
    train_module.validate_scientific_contract(config)
    assert config["experiment"]["status"] == "completed_stage_0_guard_failed_closed"
    assert config["implementation"]["train_notebook_state"] == (
        "compact_selfcontained_canonical_adopted"
    )
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["validation"]["score_unit"] == "horizontal_gr_api"
    assert config["execution_contract"]["stage_0"] == {
        "diagnostic_variants": 1,
        "audit_surfaces": 3,
        "folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "decoder_runs": 0,
    }
    with pytest.raises(RuntimeError, match="disabled"):
        train_module.validate_run_approval(config)
    approved = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    approved["execution"]["run_stage_0"] = True
    approved["runtime"]["kaggle"]["train_run_on_push"] = True
    train_module.validate_run_approval(approved)
    changed = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    changed["validation"]["score_unit"] = "ft"
    with pytest.raises(ValueError, match="score_unit"):
        train_module.validate_scientific_contract(changed)


def synthetic_parent_tables():
    folds = pd.DataFrame(
        {
            "well_id": ["wa", "wb", "wc"],
            "fold": [0, 0, 0],
        }
    )
    membership = pd.DataFrame(
        {
            "well_id": ["wa", "wb", "wc"],
            "fold": [0, 0, 0],
            "group_scheme": ["native_overlap_1"] * 3,
            "group_id": ["ga", "gb", "gc"],
        }
    )
    prior_rows = []
    for surface in ["same_typewell_heldout_well", "spatial_typewell_purged"]:
        prior_rows.extend(
            [
                {
                    "fold": 0,
                    "group_scheme": "native_overlap_1",
                    "surface": surface,
                    "control": "real",
                    "group_id": "ga",
                    "source_wells": 2,
                    "support_rows": 64,
                    "available": True,
                    "slope": 1.1,
                    "intercept": 1.0,
                    "bias_at_gr50": 6.0,
                    "residual_sigma_mad": 2.0,
                    "fit_rmse": 3.0,
                },
                {
                    "fold": 0,
                    "group_scheme": "native_overlap_1",
                    "surface": surface,
                    "control": "real",
                    "group_id": "gb",
                    "source_wells": 3,
                    "support_rows": 96,
                    "available": True,
                    "slope": 0.9,
                    "intercept": -1.0,
                    "bias_at_gr50": -6.0,
                    "residual_sigma_mad": 4.0,
                    "fit_rmse": 5.0,
                },
                {
                    "fold": 0,
                    "group_scheme": "native_overlap_1",
                    "surface": surface,
                    "control": "real",
                    "group_id": "gc",
                    "source_wells": 1,
                    "support_rows": 63,
                    "available": True,
                    "slope": 1.5,
                    "intercept": 9.0,
                    "bias_at_gr50": 34.0,
                    "residual_sigma_mad": 8.0,
                    "fit_rmse": 9.0,
                },
            ]
        )
    priors = pd.DataFrame(prior_rows)
    score_index = pd.DataFrame(
        [
            {
                "surface": "same_typewell_heldout_well",
                "fold": 0,
                "well_id": "wa",
                "group_scheme": "native_overlap_1",
                "control": "real",
                "group_id": "ga",
            },
            {
                "surface": "same_typewell_heldout_well",
                "fold": 0,
                "well_id": "wc",
                "group_scheme": "native_overlap_1",
                "control": "real",
                "group_id": "gc",
            },
            {
                "surface": "leave_one_typewell_group_out",
                "fold": 0,
                "well_id": "wa",
                "group_scheme": "native_overlap_1",
                "control": "real",
                "group_id": "ga",
            },
            {
                "surface": "spatial_typewell_purged",
                "fold": 0,
                "well_id": "wa",
                "group_scheme": "native_overlap_1",
                "control": "real",
                "group_id": "ga",
            },
        ]
    )
    return folds, membership, priors, score_index


def test_manifest_is_deterministic_and_fixes_exact_global_identity_order(
    train_module,
) -> None:
    config = load_config()
    folds, membership, priors, score_index = synthetic_parent_tables()
    first, first_sha = train_module.build_availability_manifest(
        folds,
        membership,
        priors,
        score_index,
        config,
    )
    second, second_sha = train_module.build_availability_manifest(
        folds.iloc[::-1],
        membership.iloc[::-1],
        priors.iloc[::-1],
        score_index.iloc[::-1],
        config,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_sha == second_sha
    assert len(first_sha) == 64
    selected = first.set_index(["surface", "well_id"])["selected_source"]
    assert selected[("same_typewell_heldout_well", "wa")] == (
        "exact_native_overlap_group"
    )
    assert selected[("same_typewell_heldout_well", "wc")] == (
        "global_outer_train_prior"
    )
    assert selected[("leave_one_typewell_group_out", "wa")] == (
        "global_outer_train_prior"
    )
    leave = first[
        first["surface"].eq("leave_one_typewell_group_out")
        & first["well_id"].eq("wa")
    ].iloc[0]
    assert leave["slope"] == pytest.approx(0.9)
    assert leave["selected_source_groups"] == 1

    only_target_group = priors[~priors["group_id"].eq("gb")].copy()
    identity_manifest, _ = train_module.build_availability_manifest(
        folds,
        membership,
        only_target_group,
        score_index[score_index["surface"].eq("leave_one_typewell_group_out")],
        config,
    )
    assert identity_manifest.iloc[0]["selected_source"] == "identity_no_correction"
    assert identity_manifest.iloc[0]["slope"] == 1.0
    assert identity_manifest.iloc[0]["intercept"] == 0.0


def test_suffix_truth_requires_freeze_and_identity_replay_is_exact(
    train_module,
) -> None:
    manifest = pd.DataFrame(
        [
            {
                "surface": "same_typewell_heldout_well",
                "fold": 0,
                "well_id": "wa",
                "group_id": "ga",
                "selected_source": "identity_no_correction",
                "fallback_reason": "test",
                "exact_group_available": False,
                "slope": 1.0,
                "intercept": 0.0,
            }
        ]
    )
    freeze_sha = "a" * 64
    manifest["availability_manifest_freeze_sha256"] = freeze_sha
    pairs = pd.DataFrame(
        {
            "fold": [0, 0],
            "well_id": ["wa", "wa"],
            "row_idx": [10, 11],
            "typewell_gr": [10.0, 20.0],
            "horizontal_gr": [11.0, 19.0],
        }
    )
    with pytest.raises(ValueError, match="complete frozen"):
        train_module.score_guard_manifest(
            manifest,
            pairs,
            availability_manifest_freeze_sha256="short",
        )
    scored, parity = train_module.score_guard_manifest(
        manifest,
        pairs,
        availability_manifest_freeze_sha256=freeze_sha,
    )
    assert parity == 0.0
    assert scored.iloc[0]["guarded_delta_vs_identity_gr_api"] == 0.0


def test_surface_gate_passes_only_when_all_three_safety_surfaces_pass(
    train_module,
) -> None:
    config = load_config()
    score_rows = []
    manifest_rows = []
    for surface in train_module.PARENT_SURFACES:
        for fold in range(5):
            guarded = 0.8 if surface == "same_typewell_heldout_well" else 1.0
            score_rows.append(
                {
                    "surface": surface,
                    "fold": fold,
                    "well_id": f"{surface}_{fold}",
                    "suffix_rows": 32,
                    "selected_source": (
                        "exact_native_overlap_group"
                        if surface == "same_typewell_heldout_well"
                        else "global_outer_train_prior"
                    ),
                    "exact_group_available": True,
                    "identity_suffix_gr_rmse": 1.0,
                    "guarded_suffix_gr_rmse": guarded,
                    "guarded_gain_vs_identity_gr_api": 1.0 - guarded,
                    "guarded_delta_vs_identity_gr_api": guarded - 1.0,
                }
            )
            manifest_rows.append(
                {
                    "outer_valid_truth_rows_before_manifest_freeze": 0,
                }
            )
    scored = pd.DataFrame(score_rows)
    manifest = pd.DataFrame(manifest_rows)
    metrics = train_module.aggregate_surface_metrics(scored)
    gate = train_module.evaluate_stage_0_gate(
        manifest,
        scored,
        metrics,
        0.0,
        config,
    )
    assert gate["passed"]
    assert gate["same_group_folds_improved"] == 5
    assert gate["score_unit"] == "horizontal_gr_api"

    failed = scored.copy()
    spatial = failed["surface"].eq("spatial_typewell_purged")
    failed.loc[spatial, "guarded_suffix_gr_rmse"] = 1.1
    failed.loc[spatial, "guarded_gain_vs_identity_gr_api"] = -0.1
    failed.loc[spatial, "guarded_delta_vs_identity_gr_api"] = 0.1
    failed_metrics = train_module.aggregate_surface_metrics(failed)
    failed_gate = train_module.evaluate_stage_0_gate(
        manifest,
        failed,
        failed_metrics,
        0.0,
        config,
    )
    assert not failed_gate["passed"]
    assert not failed_gate["checks"]["spatial_typewell_purged_non_regression"]


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp352_inference_contract")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)
