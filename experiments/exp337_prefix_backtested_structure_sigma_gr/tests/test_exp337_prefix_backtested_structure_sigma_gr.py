from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp337_prefix_backtested_structure_sigma_gr"
TRAIN_SOURCE = EXP_DIR / (
    "exp337_prefix_backtested_structure_sigma_gr_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp337_prefix_backtested_structure_sigma_gr_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP337_IMPORT_ONLY")
    os.environ["EXP337_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP337_IMPORT_ONLY", None)
        else:
            os.environ["EXP337_IMPORT_ONLY"] = previous


def load_config(module):
    return module.read_yaml(EXP_DIR / "config.yaml")


def residual_frame(values: np.ndarray, missing: set[int] | None = None) -> pd.DataFrame:
    missing = missing or set()
    observed = np.asarray(values, dtype=np.float64).copy()
    for index in missing:
        observed[index] = np.nan
    finite = np.isfinite(observed)
    return pd.DataFrame(
        {
            "row_idx": np.arange(len(observed), dtype=np.int64),
            "tvt_input": np.arange(len(observed), dtype=np.float64),
            "horizontal_gr": observed,
            "typewell_gr_at_tvt_input": np.zeros(len(observed)),
            "residual": np.where(finite, observed, np.nan),
            "finite_pair": finite,
        }
    )


def test_stage0_only_contract_records_zero_hmm_and_training_cost() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_contract")
    config = load_config(module)
    module.validate_scientific_contract(config)
    counts = module.get_nested(config, "execution_contract.stage_0")
    assert counts == {
        "diagnostic_variants": 1,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is True
    assert module.get_nested(config, "execution.run_stage_0") is True
    assert (
        module.get_nested(config, "model.stage_1_exact_hmm.enabled_after_implementation") is False
    )
    module.validate_scientific_contract(config, require_run_approval=True)
    disabled = deepcopy(config)
    disabled["execution"]["run_stage_0"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        module.validate_scientific_contract(disabled, require_run_approval=True)
    broken = deepcopy(config)
    broken["model"]["scale_estimator"]["internal_fit_fraction"] = 0.50
    with pytest.raises(ValueError, match="internal_fit_fraction"):
        module.validate_scientific_contract(broken)


def test_structure_scale_uses_early_std_and_late_zero_center_mse() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_scale")
    config = load_config(module)
    values = np.r_[np.tile([-2.0, 2.0], 30), np.full(40, 10.0)]
    audit = module.compute_available_prefix_scales(residual_frame(values), 100, config)
    assert audit["finite_pair_count"] == 100
    assert audit["early_finite_pair_count"] == 60
    assert audit["late_finite_pair_count"] == 40
    assert audit["sigma_finite_early_raw"] == pytest.approx(2.0)
    assert audit["late_zero_center_mse"] == pytest.approx(100.0)
    assert audit["tau_structure_variance"] == pytest.approx(96.0)
    assert audit["tau_structure"] == pytest.approx(np.sqrt(96.0))
    assert audit["structure_sigma_raw"] == pytest.approx(10.0)
    assert audit["structure_sigma"] == pytest.approx(10.0)
    assert audit["structure_fallback"] is False
    assert audit["affine_a"] == 1.0 and audit["affine_b"] == 0.0


def test_insufficient_finite_pairs_fall_back_exactly_to_zero_fill_scale() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_fallback")
    config = load_config(module)
    values = np.linspace(-20.0, 20.0, 60)
    missing = set(range(49, 60))
    audit = module.compute_available_prefix_scales(residual_frame(values, missing), 60, config)
    assert audit["finite_pair_count"] == 49
    assert audit["structure_fallback"] is True
    assert "minimum_total_finite_pairs" in audit["structure_fallback_reason"]
    assert audit["structure_sigma_raw"] == pytest.approx(audit["exp209_zero_fill_sigma_raw"])
    assert audit["structure_sigma"] == pytest.approx(audit["exp209_zero_fill_sigma"])
    assert np.isnan(audit["tau_structure"])


def test_prefix_loader_and_residual_builder_never_read_suffix_truth(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp337_truth_free_loader")
    pd.DataFrame(
        {
            "GR": [10.0, np.nan, 12.0, 13.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0],
            "error": [0.0, 0.0, 99.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": [90.0, 110.0], "GR": [5.0, 15.0]}).to_csv(
        tmp_path / "a__typewell.csv", index=False
    )
    horizontal = module.load_horizontal_without_truth("a", tmp_path)
    assert list(horizontal.columns) == ["GR", "TVT_input"]
    assert "TVT" not in horizontal and "error" not in horizontal
    prefix = module.build_prefix_residual_frame(horizontal, module.load_typewell("a", tmp_path))
    assert len(prefix) == 2
    assert prefix["finite_pair"].tolist() == [True, False]

    noncontiguous = horizontal.copy()
    noncontiguous.loc[1, "TVT_input"] = np.nan
    noncontiguous.loc[2, "TVT_input"] = 102.0
    with pytest.raises(ValueError, match="contiguous known"):
        module.build_prefix_residual_frame(noncontiguous, module.load_typewell("a", tmp_path))


def test_rolling_origin_fit_does_not_use_forward_evaluation_block() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_rolling_origin")
    config = load_config(module)
    tvt_input = np.r_[np.arange(100, dtype=float), np.full(20, np.nan)]
    base_gr = np.r_[np.tile([-4.0, 4.0], 50), np.zeros(20)]
    horizontal = pd.DataFrame({"GR": base_gr, "TVT_input": tvt_input})
    typewell = pd.DataFrame({"TVT": np.arange(130, dtype=float), "GR": np.zeros(130)})
    first, _ = module.compute_well_stage0("well-a", 0, horizontal, typewell, config)
    changed = horizontal.copy()
    changed.loc[60:79, "GR"] += 50.0
    second, _ = module.compute_well_stage0("well-a", 0, changed, typewell, config)
    origin_first = next(row for row in first if row["origin"] == 0.60)
    origin_second = next(row for row in second if row["origin"] == 0.60)
    for key in (
        "finite_only_sigma",
        "exp209_zero_fill_sigma",
        "sigma_finite_early_raw",
        "tau_structure",
        "structure_sigma",
    ):
        assert origin_first[key] == pytest.approx(origin_second[key])
    assert origin_first["structure_added_nll_mean"] != pytest.approx(
        origin_second["structure_added_nll_mean"]
    )


def test_gaussian_nll_matches_frozen_formula() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_nll")
    residual = np.array([-2.0, 0.0, 4.0])
    observed = module.gaussian_nll_without_constant(residual, 5.0)
    expected = np.log(5.0) + 0.5 * (residual / 5.0) ** 2
    np.testing.assert_allclose(observed, expected)


def test_stage0_gate_requires_both_origins_and_all_final_checks() -> None:
    module = load_module(TRAIN_SOURCE, "exp337_gate")
    config = deepcopy(load_config(module))
    config["validation"]["expected_wells"] = 10
    rows = []
    for origin in (0.60, 0.80):
        for well_index in range(10):
            rows.append(
                {
                    "well_id": f"w{well_index}",
                    "fold": well_index % 5,
                    "origin": origin,
                    "evaluable": True,
                    "evaluation_finite_pair_count": 10,
                    "structure_fallback": False,
                    "finite_only_nll_sum": 50.0,
                    "exp209_zero_fill_nll_sum": 40.0,
                    "structure_added_nll_sum": 30.0,
                }
            )
    rolling = pd.DataFrame(rows)
    final = pd.DataFrame(
        {
            "well_id": [f"w{i}" for i in range(10)],
            "structure_fallback": np.zeros(10, dtype=bool),
            "tau_structure": np.full(10, 6.0),
            "structure_clip_low": np.zeros(10, dtype=bool),
        }
    )
    gate = module.evaluate_stage0_gate(rolling, final, config)
    assert gate["passed"] is True
    assert gate["origin_reports"]["0.60"]["folds_structure_beats_zero_fill"] == 5
    assert gate["stage1_enabled"] is False
    broken = rolling.copy()
    broken.loc[np.isclose(broken["origin"], 0.80), "structure_added_nll_sum"] = 45.0
    assert module.evaluate_stage0_gate(broken, final, config)["passed"] is False


def test_dependency_preflight_checks_parent_negative_evidence_and_fold_sha(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp337_preflight")
    config = deepcopy(load_config(module))
    wells = ["a", "b"]
    for well in wells:
        pd.DataFrame({"GR": [10.0, 11.0], "TVT_input": [100.0, np.nan]}).to_csv(
            tmp_path / f"{well}__horizontal_well.csv", index=False
        )
        pd.DataFrame({"TVT": [90.0, 110.0], "GR": [8.0, 12.0]}).to_csv(
            tmp_path / f"{well}__typewell.csv", index=False
        )
    identities = [(well, row) for well in wells for row in range(5)]
    saved_hmm = tmp_path / "saved_hmm.csv.gz"
    saved_exp072 = tmp_path / "saved_exp072.csv.gz"
    fold_path = tmp_path / "fold.csv.gz"
    hidden_path = tmp_path / "hidden.csv"
    scale_path = tmp_path / "scale.csv.gz"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "hmm_mean_tvt": np.ones(10),
        }
    ).to_csv(saved_hmm, index=False, compression="gzip")
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "md_since": np.arange(10),
            "last_known_tvt": np.ones(10),
            "likpf_mean_d": np.ones(10),
        }
    ).to_csv(saved_exp072, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": [well for well, _ in identities],
            "fold": [0 if well == "a" else 1 for well, _ in identities],
        }
    ).to_csv(fold_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": wells,
            "verification_like_spatial_role": ["valid", "train"],
            "verification_like_typewell_purged_role": ["train", "valid"],
        }
    ).to_csv(hidden_path, index=False)
    pd.DataFrame(
        {
            "well_id": wells,
            "current_zero_fill_std": [30.0, 31.0],
            "finite_std": [12.0, 13.0],
            "finite_mad": [10.0, 11.0],
        }
    ).to_csv(scale_path, index=False, compression="gzip")
    summary_path.write_text(
        '{"status":"train_side_finite_mad_gate_failed_closed",'
        '"promotion_gate":{"passed":false}}\n'
    )
    raw_rows = []
    for well in wells:
        raw_rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": module.sha256_path(
                    tmp_path / f"{well}__horizontal_well.csv"
                ),
                "typewell_raw_sha256": module.sha256_path(
                    tmp_path / f"{well}__typewell.csv"
                ),
            }
        )
    raw_frame = pd.DataFrame(raw_rows).sort_values("well_id").reset_index(drop=True)
    config["validation"].update(
        {"expected_rows": 10, "expected_wells": 2, "expected_folds": [0, 1]}
    )
    config["data"]["expected_raw_well_identity_sha256"] = module.dataframe_content_sha(
        raw_frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    control = config["data"]["saved_controls"]
    control.update(
        {
            "hmm_cache_filename": saved_hmm.name,
            "exp072_cache_filename": saved_exp072.name,
            "candidates": [str(tmp_path)],
            "expected_hmm_prediction_decompressed_sha256": module.inspect_gzip_csv(saved_hmm)[
                "decompressed_sha256"
            ],
            "expected_exp072_cache_decompressed_sha256": module.inspect_gzip_csv(saved_exp072)[
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
    config["data"]["negative_evidence"].update(
        {
            "scale_audit_filename": scale_path.name,
            "summary_filename": summary_path.name,
            "candidates": [str(tmp_path)],
            "expected_scale_decompressed_sha256": module.inspect_gzip_csv(scale_path)[
                "decompressed_sha256"
            ],
            "expected_summary_sha256": module.sha256_path(summary_path),
        }
    )
    report, fold_map = module.preflight_dependencies(config, tmp_path)
    assert fold_map == {"a": 0, "b": 1}
    assert report["raw_train"]["wells"] == 2
    assert report["exp307_negative_evidence"]["gate_passed"] is False
    broken = deepcopy(config)
    broken["data"]["negative_evidence"]["expected_summary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exp307 summary SHA mismatch"):
        module.preflight_dependencies(broken, tmp_path)


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp337_inference")
    config = load_config(module)
    contract = module.validate_disabled_inference(config)
    assert contract["stage1_enabled"] is False
    assert contract["inference_enabled"] is False
    assert contract["execution_create_submission"] is False
    with pytest.raises(RuntimeError, match="Stage 1 inference and submission are disabled"):
        module.stop_disabled_inference(config)


def test_sources_are_self_contained_notebook_safe_and_stage0_only() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def compute_available_prefix_scales" in train_text
    assert "def run_stage0_prefix_audit" in train_text
    assert "def evaluate_stage0_gate" in train_text
    assert "def _hmm2_fb" not in train_text
    assert "def run_exact_hmm" not in train_text
    assert "from settings import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
