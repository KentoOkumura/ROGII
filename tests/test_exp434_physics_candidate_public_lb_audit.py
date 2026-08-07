from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp434_physics_candidate_public_lb_audit"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    previous = os.environ.get("EXP434_IMPORT_ONLY")
    os.environ["EXP434_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp434_inference_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP434_IMPORT_ONLY", None)
        else:
            os.environ["EXP434_IMPORT_ONLY"] = previous


@pytest.fixture()
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def primitive_frames(inference: ModuleType) -> dict[str, pd.DataFrame]:
    values = {
        "exp226_k16": [10010.125, 10020.375, 10030.625],
        "selfgr_hmm_a070": [10011.25, 10021.50, 10031.75],
        "likpf_mean": [10012.375, 10022.625, 10032.875],
        "exact_hmm": [10014.50, 10024.75, 10035.00],
        "pf_ancc": [10016.625, 10026.875, 10037.125],
        "beam_mean": [10018.75, 10029.00, 10039.25],
    }
    output = {}
    for candidate_id in inference.PRIMITIVE_IDS:
        output[candidate_id] = pd.DataFrame(
            {
                "id": ["well_a_0", "well_a_1", "well_a_2"],
                "well": ["well_a"] * 3,
                "well_row_idx": np.arange(3, dtype=np.int32),
                "candidate_tvt": np.asarray(values[candidate_id], dtype=np.float32),
            }
        )
    return output


def test_contract_is_exactly_frozen_twelve_and_zero_booster(
    inference: ModuleType,
    config: dict,
) -> None:
    contract = inference.validate_candidate_contract(config)
    assert tuple(contract["candidate_ids"]) == inference.CANDIDATE_IDS
    assert len(contract["candidate_ids"]) == 12
    assert tuple(contract["batch_1"]) == inference.PAIR_IDS
    assert tuple(contract["batch_2"]) == (
        "selfgr_hmm_a070",
        "exact_hmm",
        "pf_ancc",
        "beam_mean",
    )
    assert contract["normal_submission_count"] == 9
    assert contract["maximum_submission_count"] == 11
    assert contract["model_configs"] == 0
    assert contract["trained_folds"] == 0
    assert contract["boosters"] == 0
    assert contract["parent_retraining"] == 0
    assert len(contract["sha256"]) == 64


def test_run_contract_requires_all_approvals_and_one_frozen_candidate(
    inference: ModuleType,
    config: dict,
) -> None:
    contract = inference.validate_candidate_contract(
        config,
        require_run_approval=True,
    )
    assert contract["candidate_ids"] == list(inference.CANDIDATE_IDS)

    unapproved = copy.deepcopy(config)
    unapproved["execution"]["kaggle_run_approved"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        inference.validate_candidate_contract(
            unapproved,
            require_run_approval=True,
        )

    fixed = copy.deepcopy(config)
    fixed["execution"]["selected_candidate"] = inference.FIXED_ID
    with pytest.raises(ValueError, match="not approved by the frozen plan"):
        inference.validate_candidate_contract(fixed)


def test_only_frozen_pairs_and_fixed_float32_formula_are_materialized(
    inference: ModuleType,
) -> None:
    frames = primitive_frames(inference)
    bank, parity = inference.build_candidate_bank(frames)

    assert tuple(bank.columns[3:]) == inference.CANDIDATE_IDS
    assert set(parity) == {*inference.PAIR_IDS, inference.FIXED_ID}
    assert max(parity.values()) <= 1.0e-5

    pair = (
        np.float32(0.5) * frames["exp226_k16"]["candidate_tvt"].to_numpy(np.float32)
        + np.float32(0.5)
        * frames["selfgr_hmm_a070"]["candidate_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    fixed = (
        np.float32(0.5) * frames["exp226_k16"]["candidate_tvt"].to_numpy(np.float32)
        + np.float32(0.25)
        * frames["likpf_mean"]["candidate_tvt"].to_numpy(np.float32)
        + np.float32(0.25)
        * frames["exact_hmm"]["candidate_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        bank["exp226_k16__selfgr_hmm_a070"].to_numpy(np.float32),
        pair,
    )
    np.testing.assert_array_equal(bank[inference.FIXED_ID].to_numpy(np.float32), fixed)


def test_submission_restores_sample_order_and_refuses_unknown_candidate(
    inference: ModuleType,
) -> None:
    bank, _ = inference.build_candidate_bank(primitive_frames(inference))
    sample = pd.DataFrame(
        {
            "id": ["well_a_2", "well_a_0", "well_a_1"],
            "tvt": [0.0, 0.0, 0.0],
        }
    )
    submission = inference.build_submission(
        sample,
        bank,
        "exp226_k16__exact_hmm",
    )
    expected = bank.set_index("id").loc[sample["id"], "exp226_k16__exact_hmm"]
    assert submission["id"].tolist() == sample["id"].tolist()
    np.testing.assert_array_equal(submission["tvt"], expected.to_numpy())

    with pytest.raises(ValueError, match="unknown frozen candidate"):
        inference.build_submission(sample, bank, "lb_chosen_rescue")


def test_existing_equivalence_gate_passes_or_requests_same_candidate_submit(
    inference: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    bank, _ = inference.build_candidate_bank(primitive_frames(inference))
    previous = inference.build_submission(
        pd.DataFrame(
            {
                "id": bank["id"],
                "tvt": np.zeros(len(bank)),
            }
        ),
        bank,
        "exp226_k16",
    )
    path = tmp_path / "submission.csv"
    previous.to_csv(path, index=False)

    local_config = copy.deepcopy(config)
    local_config["existing_lb"]["exp226_k16"]["submission_sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    passed = inference.existing_submission_gate(
        bank,
        "exp226_k16",
        path,
        local_config,
    )
    assert passed["status"] == "passed_reusable_existing_public_lb"
    assert passed["max_abs_ft"] == 0.0

    previous["tvt"] = previous["tvt"] + 0.01
    previous.to_csv(path, index=False)
    local_config["existing_lb"]["exp226_k16"]["submission_sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    failed = inference.existing_submission_gate(
        bank,
        "exp226_k16",
        path,
        local_config,
    )
    assert failed["status"] == "failed_submit_same_candidate_required"
    assert failed["max_abs_ft"] > 0.001


def test_parent_generator_sources_are_sha_pinned(config: dict) -> None:
    local_sources = {
        "exp209_exact_hmm": (
            ROOT
            / "experiments"
            / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
            / "exact_hmm_smoother.py"
        ),
        "exp223_selfgr_hmm": (
            ROOT
            / "experiments"
            / "exp223_joint_typewell_self_gr_hmm_likelihood_probe"
            / "exact_hmm_smoother.py"
        ),
        "exp226_k16": (
            ROOT
            / "experiments"
            / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
            / "connortynan_k16_reproduction.py"
        ),
    }
    expected = config["data"]["parent_source_sha256"]
    for source_id, path in local_sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[source_id]
    assert len(expected["exp073_pf_replay"]) == 64
    assert config["runtime"]["parent_raw_test_generation"]["pf_seed_policy"] == (
        "stable_sha256_per_well"
    )
    assert config["generator_contract"]["pf_replay"]["pf_seeds"] == 128
    assert config["generator_contract"]["pf_replay"]["pf_particles"] == 500


def test_trusted_source_is_copied_under_a_fixed_module_name(
    inference: ModuleType,
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "experiments"
        / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
        / "connortynan_k16_reproduction.py"
    )
    copied = inference.copy_trusted_source(
        source,
        tmp_path,
        "exp434_trusted_exp226_k16",
    )
    assert copied.name == "exp434_trusted_exp226_k16.py"
    assert copied.read_bytes() == source.read_bytes()


def test_compact_source_is_not_thin_and_canonical_notebook_is_adopted(
    inference: ModuleType,
) -> None:
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "importlib.util" not in source
    assert "from settings import" not in source
    assert "from candidate_cache_" not in source
    assert "## 8. Hidden-safe six-primitive regeneration" in source
    assert "## 9. Twelve-candidate formula bank and equivalence gates" in source
    assert "## 11. Metrics, SHA, manifests, and generated artifacts" in source
    assert len(source.splitlines()) > 1000

    canonical = json.loads((EXP_DIR / f"{EXP}_inference.ipynb").read_text())
    canonical_text = "\n".join(
        "".join(cell.get("source", [])) for cell in canonical["cells"]
    )
    assert "## 8. Hidden-safe six-primitive regeneration" in canonical_text
    assert "## 9. Twelve-candidate formula bank and equivalence gates" in canonical_text
    assert "Design-only placeholder" not in canonical_text
    assert inference.IMPORT_ONLY_ENV == "EXP434_IMPORT_ONLY"
