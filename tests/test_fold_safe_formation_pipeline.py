from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.fold_safe_formation_pipeline import (
    DenseANCCImputer,
    FormationPlaneKNN,
    FormationReferenceCatalog,
    audit_feature_relationships,
    build_current_test_formation_surface,
    build_well_formation_features,
    canonical_formation_feature_names,
    formation_cost_contract,
    select_unique_columns,
)


def test_cost_contract_is_one_variant_three_configs_five_folds() -> None:
    config = {
        "model": {
            "formation_addonly_stage": {
                "active_variants": ["fold_safe_formation_74_addonly"],
                "lightgbm_config_indices": [0, 1, 2],
                "folds": 5,
                "planned_gpu_boosters": 15,
                "control_retraining": False,
                "clean_base_feature_count": 273,
                "nested_compact_feature_count": 74,
                "fold_safe_formation_feature_count": 74,
                "final_feature_count": 421,
            }
        }
    }
    contract = formation_cost_contract(config)
    assert contract["planned_gpu_boosters"] == 15
    assert contract["parent_control_retraining"] is False


def test_saved_control_pin_matches_corrected_exp264_stage_d_v3() -> None:
    exp287_config = yaml.safe_load(
        Path(
            "experiments/exp287_fold_safe_formation_74_addonly_on_exp264/config.yaml"
        ).read_text()
    )
    exp264_metrics = json.loads(
        Path(
            "experiments/exp264_exp263_candidate_confidence_dual_selector/metrics.json"
        ).read_text()
    )
    corrected = exp264_metrics["corrected_stage_d"]
    assert corrected["kernel_version"] == 3
    assert (
        exp287_config["data"]["saved_exp264_stage_d_oof_sha256"]
        == corrected["artifacts"]["oof_predictions_sha256"]
        == "b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2"
    )
    assert np.isclose(
        exp287_config["validation"]["parent_exp264_rmse"],
        corrected["pooled_metrics"]["selector_compact_addonly_lgb_mean_rmse"],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_clean_projection_selects_anchor_once_when_it_is_also_a_model_feature() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "well": ["w", "w"],
            "target": [1.0, 2.0],
            "last_known_tvt": [100.0, 100.0],
            "md_since": [10.0, 20.0],
            "base_a": [3.0, 4.0],
        }
    )
    selected = select_unique_columns(
        frame,
        [
            "id",
            "well",
            "target",
            "last_known_tvt",
            "md_since",
            "last_known_tvt",
            "base_a",
        ],
        context="test clean surface",
    )
    assert selected.columns.tolist() == [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "base_a",
    ]
    assert selected["last_known_tvt"].to_numpy().shape == (2,)


def test_clean_projection_rejects_duplicate_source_column_labels() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["last_known_tvt", "last_known_tvt"])
    with pytest.raises(ValueError, match="duplicate column labels"):
        select_unique_columns(
            frame,
            ["last_known_tvt"],
            context="test clean surface",
        )


def _plane_and_dense() -> tuple[FormationPlaneKNN, DenseANCCImputer]:
    wells = np.asarray([f"w{index}" for index in range(12)], dtype=object)
    xy = np.column_stack([np.linspace(0.0, 11.0, 12), np.square(np.linspace(0.0, 1.1, 12))])
    formation = np.column_stack([10.0 * column + np.arange(12, dtype=float) for column in range(6)])
    formation[0] = 10_000.0
    plane = FormationPlaneKNN(wells=wells, xy=xy, formation_medians=formation, k=3)
    dense_wells = np.repeat(wells, 3)
    dense_xy = np.repeat(xy, 3, axis=0) + np.tile(
        np.asarray([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]), (12, 1)
    )
    dense_ancc = np.repeat(formation[:, 0], 3).astype(np.float32)
    dense = DenseANCCImputer(
        wells=dense_wells,
        xy=dense_xy,
        ancc=dense_ancc,
        k=3,
        nfetch=len(dense_ancc),
    )
    return plane, dense


def test_target_well_is_excluded_from_both_reference_imputers() -> None:
    plane, dense = _plane_and_dense()
    query = np.asarray([[0.0, 0.0]])
    plane_with_self, _ = plane.impute(query, target_well=None)
    plane_without_self, _ = plane.impute(query, target_well="w0")
    dense_with_self, _, _ = dense.impute(query, target_well=None)
    dense_without_self, _, _ = dense.impute(query, target_well="w0")
    assert float(plane_with_self[0, 0]) > float(plane_without_self[0, 0]) + 100.0
    assert float(dense_with_self[0]) > float(dense_without_self[0]) + 100.0


def test_reference_catalog_skips_missing_plane_and_dense_sources_independently(
    tmp_path: Path,
) -> None:
    complete_wells = ["complete0", "complete1", "complete2"]
    dense_only_well = "dense_only"
    no_ancc_well = "no_ancc"
    formations = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    for well_index, well in enumerate([*complete_wells, dense_only_well, no_ancc_well]):
        rows = 8
        frame = pd.DataFrame(
            {
                "X": np.linspace(float(well_index), float(well_index) + 0.1, rows),
                "Y": np.linspace(0.0, 0.1, rows),
                **{
                    formation: np.linspace(
                        100.0 * formation_index + well_index,
                        100.0 * formation_index + well_index + 1.0,
                        rows,
                    )
                    for formation_index, formation in enumerate(formations)
                },
            }
        )
        if well == dense_only_well:
            frame["EGFDL"] = np.nan
        if well == no_ancc_well:
            frame["ANCC"] = np.nan
        frame.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)

    requested_wells = [*complete_wells, dense_only_well, no_ancc_well]
    catalog = FormationReferenceCatalog.from_raw(
        tmp_path,
        requested_wells,
        samples_per_well=4,
    )
    assert set(map(str, catalog.plane_wells)) == set(complete_wells)
    assert set(map(str, catalog.dense_wells)) == {*complete_wells, dense_only_well}

    plane, dense, evidence = catalog.fit(
        requested_wells,
        plane_k=2,
        dense_k=2,
        dense_nfetch=8,
        query_workers=1,
    )
    plane_prediction, _ = plane.impute(np.asarray([[0.5, 0.05]]), target_well=no_ancc_well)
    dense_prediction, _, _ = dense.impute(
        np.asarray([[0.5, 0.05]]),
        target_well=no_ancc_well,
    )
    assert np.isfinite(plane_prediction).all()
    assert np.isfinite(dense_prediction).all()
    assert evidence["reference_wells"] == 5
    assert evidence["plane_reference_wells"] == 3
    assert evidence["plane_missing_reference_wells"] == [dense_only_well, no_ancc_well]
    assert evidence["dense_reference_wells"] == 4
    assert evidence["dense_missing_reference_wells"] == [no_ancc_well]


def test_well_generator_never_requires_target_formation_columns(tmp_path: Path) -> None:
    plane, dense = _plane_and_dense()
    well = "target"
    rows = 24
    known_rows = 12
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) + 1000.0,
            "X": np.linspace(2.0, 3.0, rows),
            "Y": np.linspace(0.1, 0.2, rows),
            "Z": np.linspace(-1000.0, -1010.0, rows),
            "TVT_input": np.r_[
                np.linspace(900.0, 911.0, known_rows), np.full(rows - known_rows, np.nan)
            ],
        }
    )
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    pd.DataFrame(
        {
            "TVT": np.linspace(800.0, 1100.0, 50),
            "GR": np.linspace(20.0, 120.0, 50),
        }
    ).to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    evaluation_index = np.arange(known_rows, rows)
    last = 911.0
    base = pd.DataFrame(
        {
            "id": [f"{well}_{index}" for index in evaluation_index],
            "well": well,
            "last_known_tvt": last,
            "pf_ancc": np.linspace(915.0, 925.0, len(evaluation_index)),
            "beam_cons_d": np.linspace(1.0, 2.0, len(evaluation_index)),
            "beam_loose_d": np.linspace(2.0, 3.0, len(evaluation_index)),
            "beam_vcons_d": np.linspace(3.0, 4.0, len(evaluation_index)),
            "beam_sm5_d": np.linspace(4.0, 5.0, len(evaluation_index)),
            "beam_vloose_d": np.linspace(5.0, 6.0, len(evaluation_index)),
            "beam_mid_d": np.linspace(6.0, 7.0, len(evaluation_index)),
            "beam_stiff_d": np.linspace(7.0, 8.0, len(evaluation_index)),
            "sc8_d": np.linspace(1.5, 2.5, len(evaluation_index)),
            "sc15_d": np.linspace(2.5, 3.5, len(evaluation_index)),
            "sc25_d": np.linspace(3.5, 4.5, len(evaluation_index)),
            "sc_ens_d": np.linspace(4.5, 5.5, len(evaluation_index)),
        }
    )
    features = canonical_formation_feature_names()
    generated = build_well_formation_features(
        well=well,
        base_well=base,
        raw_dir=tmp_path,
        plane=plane,
        dense=dense,
        reference_wells={f"w{index}" for index in range(12)},
        feature_names=features,
    )
    assert generated.columns.tolist() == ["id", "well", *features]
    assert generated.shape == (rows - known_rows, 76)
    assert np.isfinite(generated[features].to_numpy()).all()


def test_current_test_surface_uses_train_references_and_test_target_schema(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    formations = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    reference_wells = [f"w{index}" for index in range(12)]
    for well_index, well in enumerate(reference_wells):
        rows = 12
        pd.DataFrame(
            {
                "X": np.linspace(float(well_index), float(well_index) + 0.1, rows),
                "Y": np.linspace(0.0, 0.2, rows),
                **{
                    formation: np.linspace(
                        10.0 * formation_index + well_index,
                        10.0 * formation_index + well_index + 1.0,
                        rows,
                    )
                    for formation_index, formation in enumerate(formations)
                },
            }
        ).to_csv(train_dir / f"{well}__horizontal_well.csv", index=False)

    target_well = "w0"
    rows = 24
    known_rows = 12
    pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) + 1000.0,
            "X": np.linspace(2.0, 3.0, rows),
            "Y": np.linspace(0.1, 0.2, rows),
            "Z": np.linspace(-1000.0, -1010.0, rows),
            "TVT_input": np.r_[
                np.linspace(900.0, 911.0, known_rows),
                np.full(rows - known_rows, np.nan),
            ],
        }
    ).to_csv(test_dir / f"{target_well}__horizontal_well.csv", index=False)
    pd.DataFrame(
        {
            "TVT": np.linspace(800.0, 1100.0, 50),
            "GR": np.linspace(20.0, 120.0, 50),
        }
    ).to_csv(test_dir / f"{target_well}__typewell.csv", index=False)
    evaluation_index = np.arange(known_rows, rows)
    base = pd.DataFrame(
        {
            "id": [f"{target_well}_{index}" for index in evaluation_index],
            "well": target_well,
            "last_known_tvt": 911.0,
            "pf_ancc": np.linspace(915.0, 925.0, len(evaluation_index)),
            **{
                name: np.linspace(offset, offset + 1.0, len(evaluation_index))
                for offset, name in enumerate(
                    [
                        "beam_cons_d",
                        "beam_loose_d",
                        "beam_vcons_d",
                        "beam_sm5_d",
                        "beam_vloose_d",
                        "beam_mid_d",
                        "beam_stiff_d",
                        "sc8_d",
                        "sc15_d",
                        "sc25_d",
                        "sc_ens_d",
                    ],
                    start=1,
                )
            },
        }
    )
    features = canonical_formation_feature_names()
    generated, evidence = build_current_test_formation_surface(
        base_frame=base,
        raw_train_dir=train_dir,
        raw_test_dir=test_dir,
        reference_wells=reference_wells,
        feature_names=features,
        generator_config={
            "dense_samples_per_well": 4,
            "plane_k": 3,
            "dense_k": 3,
            "dense_nfetch": 48,
            "query_workers": 1,
            "n_jobs": 1,
        },
    )
    assert generated[["id", "well"]].equals(base[["id", "well"]])
    assert np.isfinite(generated[features].to_numpy()).all()
    assert evidence["generation_role"] == "current_test_all_train_reference"
    assert evidence["target_formation_columns_read"] is False
    assert evidence["target_train_name_overlap_wells"] == [target_well]
    assert evidence["target_train_name_overlap_self_excluded"] == 1


def test_relationship_audit_reports_duplicate_without_pruning() -> None:
    rows = 20
    identifiers = [f"w_{index}" for index in range(rows)]
    existing = pd.DataFrame(
        {
            "id": identifiers,
            "well": "w",
            "base_a": np.arange(rows, dtype=np.float32),
            "base_b": np.square(np.arange(rows, dtype=np.float32)),
        }
    )
    formation = pd.DataFrame(
        {
            "id": identifiers,
            "well": "w",
            "formation_a": existing["base_a"].to_numpy(),
            "formation_b": np.linspace(5.0, 10.0, rows, dtype=np.float32),
        }
    )
    audit = audit_feature_relationships(
        existing=existing,
        formation=formation,
        existing_features=["base_a", "base_b"],
        formation_features=["formation_a", "formation_b"],
        correlation_sample_rows=20,
    )
    duplicate = audit.set_index("formation_feature").loc["formation_a"]
    assert duplicate["exact_duplicate_count"] == 1
    assert duplicate["exact_duplicate_features"] == "base_a"
    assert not audit["pruned"].any()
