from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_NAME = "exp263_last_anchor_better_candidate_confidence_pair_cache"
EXP_DIR = ROOT / "experiments" / EXP_NAME
sys.path.insert(0, str(EXP_DIR))

from candidate_cache_builder import (  # noqa: E402
    assemble_stage1_current_test_parity,
    attach_stage1_current_test_confidence,
    build_stage0_cache,
    build_submission_from_stage1_parity,
)
from candidate_cache_contract import (  # noqa: E402
    CORE_CANDIDATE_IDS,
    FORBIDDEN_CANDIDATE_IDS,
    NAMED_COMBINATIONS,
    PAIR_SHORTLIST,
    RAWTEST_CORE_CANDIDATE_IDS,
    REFERENCE_CANDIDATES,
    STAGE1_NATIVE_CONFIDENCE_FIELDS,
    topological_formula_order,
    validate_contract,
    validate_selectable_names,
)
from candidate_cache_loader import (  # noqa: E402
    KEY_COLUMNS,
    CandidateCache,
    assert_key_alignment,
    frame_content_sha256,
    materialize_formula_frames,
    schema_sha256,
)


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def primitive_frame(candidate_id: str, values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "id": [f"well_a_{index}" for index in range(n)],
            "well": ["well_a"] * n,
            "well_row_idx": np.arange(n, dtype=np.int32),
            "outer_fold": np.zeros(n, dtype=np.int8),
            "md_since": np.arange(n, dtype=np.float32),
            "candidate_id": candidate_id,
            "candidate_name": candidate_id,
            "family": "test",
            "source_exp": "test",
            "rawtest_status": "test",
            "formula": candidate_id,
            "last_known_tvt": np.full(n, 100.0, dtype=np.float32),
            "candidate_tvt": np.asarray(values, dtype=np.float32),
            "candidate_minus_last": np.asarray(values, dtype=np.float32) - 100.0,
            "candidate_finite": True,
            "candidate_available": True,
            "fallback_used": False,
            "coverage_valid": True,
        }
    )


def test_logical_hash_is_stable_across_object_and_string_dtypes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series(["well_a_0", "well_a_1"], dtype=object),
            "value": np.asarray([1.0, 2.0], dtype=np.float32),
        }
    )
    expected_content = frame_content_sha256(frame)
    expected_schema = schema_sha256(frame)

    string_frame = frame.copy()
    string_frame["id"] = string_frame["id"].astype("str")
    assert frame_content_sha256(string_frame) == expected_content
    assert schema_sha256(string_frame) == expected_schema

    parquet_path = tmp_path / "logical_hash.parquet"
    frame.to_parquet(parquet_path, index=False)
    restored = pd.read_parquet(parquet_path)
    assert frame_content_sha256(restored) == expected_content
    assert schema_sha256(restored) == expected_schema


def test_inventory_is_exactly_33_reference_12_core_6_rawtest() -> None:
    counts = validate_contract()
    ids = {item.candidate_id for item in REFERENCE_CANDIDATES}

    assert counts == {
        "reference_candidates": 33,
        "core_candidates": 12,
        "rawtest_core_candidates": 6,
        "shortlisted_pairs": 8,
        "rawtest_pairs": 5,
        "named_triples": 3,
    }
    assert len(CORE_CANDIDATE_IDS) == len(set(CORE_CANDIDATE_IDS)) == 12
    assert set(RAWTEST_CORE_CANDIDATE_IDS) == {
        "exp226_k16",
        "selfgr_hmm_a070",
        "likpf_mean",
        "exact_hmm",
        "pf_ancc",
        "beam_mean",
    }
    assert not ids.intersection(FORBIDDEN_CANDIDATE_IDS)


def test_exp104_is_catalog_only_superseded_and_not_pair_swept() -> None:
    exp104 = [item for item in REFERENCE_CANDIDATES if item.source_exp == "exp104"]
    pair_components = {item.left for item in PAIR_SHORTLIST} | {
        item.right for item in PAIR_SHORTLIST
    }

    assert len(exp104) == 5
    assert all(item.cache_role == "superseded_reference" for item in exp104)
    assert all(item.source_key is None and item.value_column is None for item in exp104)
    assert not pair_components.intersection(item.candidate_id for item in exp104)


def test_pf_family_is_compressed_without_dropped_path_reentry() -> None:
    core = set(CORE_CANDIDATE_IDS)
    assert "pf_medoid_k8_m0" in core
    assert not core.intersection({f"pf_medoid_k8_m{slot}" for slot in range(1, 8)})
    assert "exp103_xy_likpf_scale_12" in core
    assert not core.intersection(
        {
            "exp103_xy_likpf_mean",
            "exp103_xy_likpf_scale_3",
            "exp103_xy_likpf_scale_5",
            "exp103_xy_likpf_scale_8",
        }
    )
    assert not core.intersection({"pf_z", "exp192_pf_z", "exp106_pf_z_ms_scale_12"})


def test_pair_and_named_formula_contract_is_fixed() -> None:
    assert len(PAIR_SHORTLIST) == 8
    assert sum(item.tier == "raw-test" for item in PAIR_SHORTLIST) == 5
    assert len(NAMED_COMBINATIONS) == 4
    fixed = NAMED_COMBINATIONS["exp226_w500_50_50"]
    assert fixed["weights"] == {
        "exp226_k16": 0.5,
        "likpf_mean": 0.25,
        "exact_hmm": 0.25,
    }
    assert len(NAMED_COMBINATIONS) - 1 == 3
    assert topological_formula_order()


def test_formula_cycle_guard_and_parent_child_selectable_guard() -> None:
    with pytest.raises(ValueError, match="cycle"):
        topological_formula_order({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="w500"):
        validate_selectable_names(["blend_likpf_hmm_w500", "exact_hmm"])
    validate_selectable_names(["exp226_k16", "likpf_mean", "exact_hmm"])


def test_virtual_formula_materializes_without_persisted_pair_tensor() -> None:
    frames = {
        "exp226_k16": primitive_frame("exp226_k16", [10.0, 20.0, 30.0]),
        "likpf_mean": primitive_frame("likpf_mean", [14.0, 24.0, 34.0]),
        "exact_hmm": primitive_frame("exact_hmm", [18.0, 28.0, 38.0]),
    }
    fixed = NAMED_COMBINATIONS["exp226_w500_50_50"]
    output = materialize_formula_frames(
        "exp226_w500_50_50",
        frames,
        fixed["components"],
        fixed["weights"],
        fixed["formula"],
    )
    expected = 0.5 * np.array([10.0, 20.0, 30.0]) + 0.25 * np.array(
        [14.0, 24.0, 34.0]
    ) + 0.25 * np.array([18.0, 28.0, 38.0])
    np.testing.assert_allclose(output["candidate_tvt"], expected, rtol=0, atol=1e-6)
    assert output[KEY_COLUMNS].equals(frames["exp226_k16"][KEY_COLUMNS])


def test_virtual_formula_refuses_misaligned_candidate_keys() -> None:
    left = primitive_frame("left", [1.0, 2.0])
    right = primitive_frame("right", [3.0, 4.0])
    right.loc[1, "id"] = "other_1"

    with pytest.raises(ValueError, match="id"):
        assert_key_alignment(left, right)
    with pytest.raises(ValueError, match="id"):
        materialize_formula_frames(
            "pair",
            {"left": left, "right": right},
            ["left", "right"],
            {"left": 0.5, "right": 0.5},
            "0.5*left + 0.5*right",
        )


def test_config_is_zero_booster_pf_beam_and_stage1_submission_is_fixed() -> None:
    config = load_config()
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["active_variants"] == []
    assert config["model"]["lightgbm_config_count"] == 0
    assert config["model"]["fold_training_count"] == 0
    assert config["model"]["booster_count"] == 0
    assert config["model"]["parent_control_retraining"] is False
    assert config["stage0"]["enabled"] is True
    assert config["stage1"]["enabled"] is True
    assert config["stage1"]["create_submission"] is True
    assert config["stage1"]["selected_submission_candidate"] == "exp226_w500_50_50"
    assert set(config["stage1"]["primitive_inputs"]) == set(RAWTEST_CORE_CANDIDATE_IDS)
    configured_confidence = config["stage1"]["confidence_output"][
        "required_fields_by_primitive"
    ]
    assert set(configured_confidence) == set(RAWTEST_CORE_CANDIDATE_IDS)
    for candidate_id, fields in STAGE1_NATIVE_CONFIDENCE_FIELDS.items():
        assert configured_confidence[candidate_id] == ["confidence_valid", *fields]
    assert config["cache"]["materialize_pair_tensor"] is False
    assert config["cache"]["materialize_named_combination_tensor"] is False


def test_stage1_assembles_six_primitives_and_fixed_submission() -> None:
    values = {
        "exp226_k16": [10010.123, 10020.456, 10030.789],
        "selfgr_hmm_a070": [10011.234, 10021.567, 10031.891],
        "likpf_mean": [10012.345, 10022.678, 10032.912],
        "exact_hmm": [10014.456, 10024.789, 10034.123],
        "pf_ancc": [10016.567, 10026.891, 10036.234],
        "beam_mean": [10018.678, 10028.912, 10038.345],
    }
    frames = {
        candidate_id: primitive_frame(candidate_id, candidate_values)[
            ["id", "well", "well_row_idx", "candidate_tvt"]
        ]
        for candidate_id, candidate_values in values.items()
    }
    parity, max_abs = assemble_stage1_current_test_parity(frames)
    expected = (
        np.float32(0.5) * np.asarray(values["exp226_k16"], dtype=np.float32)
        + np.float32(0.25) * np.asarray(values["likpf_mean"], dtype=np.float32)
        + np.float32(0.25) * np.asarray(values["exact_hmm"], dtype=np.float32)
    ).astype(np.float32)
    np.testing.assert_allclose(parity["exp226_w500_50_50"], expected, atol=1e-6)
    assert max_abs <= 1e-5

    sample = pd.DataFrame(
        {
            "id": ["well_a_2", "well_a_0", "well_a_1"],
            "tvt": [0.0, 0.0, 0.0],
        }
    )
    submission = build_submission_from_stage1_parity(sample, parity)
    assert submission["id"].tolist() == sample["id"].tolist()
    np.testing.assert_allclose(submission["tvt"], expected[[2, 0, 1]], atol=1e-6)


def test_stage1_attaches_exact_namespaced_native_confidence() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for position, candidate_id in enumerate(RAWTEST_CORE_CANDIDATE_IDS):
        frame = primitive_frame(candidate_id, [100.0 + position, 101.0 + position])[
            ["id", "well", "well_row_idx", "candidate_tvt"]
        ]
        fields = STAGE1_NATIVE_CONFIDENCE_FIELDS[candidate_id]
        frame["confidence_valid"] = bool(fields)
        for field_position, field in enumerate(fields):
            frame[field] = np.full(
                len(frame), np.float32(position + field_position / 10.0)
            )
        frames[candidate_id] = frame

    parity, _ = assemble_stage1_current_test_parity(frames)
    output = attach_stage1_current_test_confidence(parity, frames)
    confidence_columns = [
        column for column in output if column.startswith("confidence__")
    ]
    assert len(confidence_columns) == 21
    assert not output["confidence__likpf_mean__confidence_valid"].any()
    assert output["confidence__exact_hmm__confidence_valid"].all()
    np.testing.assert_allclose(
        output["confidence__beam_mean__beam_family_std"],
        frames["beam_mean"]["beam_family_std"],
    )

    invalid = dict(frames)
    invalid["exact_hmm"] = invalid["exact_hmm"].drop(columns="sigma_tvt")
    with pytest.raises(ValueError, match="sigma_tvt"):
        attach_stage1_current_test_confidence(parity, invalid)


def test_row_partition_writer_does_not_read_truth_or_target() -> None:
    source = (EXP_DIR / "candidate_cache_builder.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_candidate_partitions"
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None
    assert "canonical.truth" not in function_source
    assert '"target"' not in function_source
    assert '"true_tvt"' not in function_source


def test_notebooks_expose_cache_and_parity_structure() -> None:
    train = json.loads((EXP_DIR / f"{EXP_NAME}_train.ipynb").read_text())
    inference = json.loads((EXP_DIR / f"{EXP_NAME}_inference.ipynb").read_text())
    train_headings = {
        "".join(cell["source"]).splitlines()[0]
        for cell in train["cells"]
        if cell["cell_type"] == "markdown" and cell.get("source")
    }
    inference_headings = {
        "".join(cell["source"]).splitlines()[0]
        for cell in inference["cells"]
        if cell["cell_type"] == "markdown" and cell.get("source")
    }
    assert len(train["cells"]) >= 20
    assert "## 4. Input source resolution and schema preflight" in train_headings
    assert "## 6. Pair shortlist and formula DAG checks" in train_headings
    assert "## 7. Stage 0 cache generation orchestration" in train_headings
    assert "## 9. Virtual loader parity sample" in train_headings
    assert len(inference["cells"]) >= 16
    assert "## 4. Trusted upstream source resolution" in inference_headings
    assert "## 5. Raw-test six-primitive regeneration" in inference_headings
    assert "## 6. Five pair and fixed named-formula parity" in inference_headings
    assert "## 7. Current-test reference parity and submission generation" in inference_headings


def test_stage0_builder_and_virtual_loader_end_to_end_on_synthetic_sources(
    tmp_path: Path,
) -> None:
    wells = [f"w{index}" for index in range(5)]
    identity = [(well, row) for well in wells for row in range(3)]
    n = len(identity)
    well_values = [well for well, _ in identity]
    row_values = np.asarray([row for _, row in identity], dtype=np.int32)
    ids = [f"{well}_{row}" for well, row in identity]
    anchor = np.asarray([100.0 + 2.0 * int(well[1:]) for well in well_values])
    target = row_values.astype(float) + 1.0
    md_since = (row_values + 1).astype(float) * 25.0

    base = pd.DataFrame(
        {
            "id": ids,
            "well": well_values,
            "target": target,
            "last_known_tvt": anchor,
            "md_since": md_since,
            "likpf_mean_d": target + 0.5,
            "pf_ancc": anchor + target + 1.0,
            "pf_ancc_std": np.full(n, 2.0),
            "beam_mean_d": target + 1.5,
            "beam_std_d": np.full(n, 3.0),
        }
    )
    exp209 = base[["id", "well"]].assign(
        hmm_mean_tvt=anchor + target + 0.25,
        hmm_std=1.5,
        hmm_loglik=-10.0,
    )
    exp223 = base[["id", "well"]].assign(
        hmm_selfgr_boost_only_a070_c100_mean_tvt=anchor + target - 0.25,
        hmm_selfgr_boost_only_a070_c100_std=1.0,
        hmm_selfgr_boost_only_a070_c100_loglik=-8.0,
        hmm_selfgr_boost_only_a070_c100_finite=1.0,
        hmm_selfgr_boost_only_a150_c100_mean_tvt=anchor + target + 0.75,
        hmm_selfgr_boost_only_a150_c100_std=1.2,
        hmm_selfgr_boost_only_a150_c100_loglik=-9.0,
        hmm_selfgr_boost_only_a150_c100_finite=1.0,
        self_gr_quality=0.8,
        self_gr_peak_tvt=anchor + target,
        self_gr_peak_gap=0.3,
        self_gr_typewell_agreement=0.7,
        self_gr_valid=1.0,
    )
    exp225 = base[["id", "well"]].assign(
        hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_mean_tvt=anchor
        + target
        + 2.0,
        hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_std=2.0,
        hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_loglik=-12.0,
        hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_finite=1.0,
        self_gr_quality=0.5,
        self_gr_peak_gap=0.1,
        self_gr_valid=1.0,
        self_gr_state_valid_rate=0.9,
    )
    exp226 = pd.DataFrame(
        {
            "well_id": well_values,
            "row_idx": row_values,
            "tvt_pred": anchor + target - 0.5,
            "gr_delta": np.full(n, 0.2),
        }
    )
    exp231 = base[["id", "well"]].assign(
        hmm_peer_atlas_a025_mean_tvt=anchor + target + 0.4,
        hmm_peer_atlas_a025_std=1.1,
        hmm_peer_atlas_a025_loglik=-7.0,
        hmm_peer_atlas_a025_finite=1.0,
        hmm_prefix_sigma=0.6,
        peer_atlas_support=4.0,
        peer_atlas_match_confidence=0.8,
        peer_atlas_novelty=0.2,
        peer_atlas_uniqueness=0.7,
        peer_atlas_base_uncertainty=0.3,
        peer_atlas_innovation=0.4,
        peer_atlas_change_point=0.1,
        peer_atlas_confidence=0.75,
        peer_atlas_available=1.0,
    )
    exp192 = base[["id", "well", "last_known_tvt"]].assign(
        likpf_mean_d=target + 0.1
    )
    exp103 = pd.DataFrame(
        {
            "well": well_values,
            "row_idx": row_values,
            "xy_likpf_scale_12": anchor + target + 1.2,
            "xy_likpf_seed_std": np.full(n, 1.8),
        }
    )
    exp243 = pd.DataFrame(
        {
            "well": well_values,
            "row_idx": row_values,
            "pf_seed_medoid_k8_m0": anchor + target + 0.9,
            "pf_seed_std_diag": np.full(n, 2.2),
        }
    )
    frames = {
        "exp072_oof": base,
        "exp103_oof": exp103,
        "exp192_oof": exp192,
        "exp209_oof": exp209,
        "exp223_oof": exp223,
        "exp225_oof": exp225,
        "exp226_oof": exp226,
        "exp231_oof": exp231,
        "exp243_oof": exp243,
    }
    input_paths: dict[str, dict[str, str]] = {}
    for source_key, frame in frames.items():
        path = tmp_path / f"{source_key}.csv"
        frame.to_csv(path, index=False)
        input_paths[source_key] = {"path": str(path)}

    output_dir = tmp_path / "cache"
    config = {
        "data": {"search_roots": [str(tmp_path)], "inputs": input_paths},
        "cache": {
            "expected_rows": n,
            "chunk_rows": 4,
            "parity_sample_rows": 3,
        },
        "reproducibility": {"record_decompressed_source_sha": False},
    }
    summary = build_stage0_cache(
        config,
        output_dir,
        debug=True,
        max_rows=n,
    )

    assert summary["status"] == "debug_completed"
    assert summary["rows"] == n
    assert (output_dir / "candidate_catalog.json").exists()
    assert (output_dir / "cache_manifest.json").exists()
    assert not (output_dir / "_work").exists()
    assert len(list((output_dir / "candidate_values").glob("*/fold=*/*.parquet"))) == 60
    assert len(list((output_dir / "candidate_confidence").glob("*/fold=*/*.parquet"))) == 60

    cache = CandidateCache(output_dir)
    fixed = cache.materialize("exp226_w500_50_50", fold=0)
    expected = (
        0.5 * cache.load_primitive("exp226_k16", fold=0)["candidate_tvt"].to_numpy()
        + 0.25 * cache.load_primitive("likpf_mean", fold=0)["candidate_tvt"].to_numpy()
        + 0.25 * cache.load_primitive("exact_hmm", fold=0)["candidate_tvt"].to_numpy()
    )
    np.testing.assert_allclose(fixed["candidate_tvt"], expected, rtol=0, atol=1e-6)
    exact_confidence = cache.load_confidence("exact_hmm", fold=0)
    np.testing.assert_allclose(
        exact_confidence["loglik_per_row"],
        np.full(len(exact_confidence), -10.0 / 3.0),
        rtol=0,
        atol=1e-6,
    )
