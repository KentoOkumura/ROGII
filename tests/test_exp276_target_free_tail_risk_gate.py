from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp276_exp264_compact_tail_risk_target_free_gate_audit"
SOURCE = EXP_DIR / "exp276_exp264_compact_tail_risk_target_free_gate_audit_train.py"


def load_module():
    previous = os.environ.get("EXP276_IMPORT_ONLY")
    os.environ["EXP276_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp276_train", SOURCE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP276_IMPORT_ONLY", None)
        else:
            os.environ["EXP276_IMPORT_ONLY"] = previous


MODULE = load_module()
CONFIG = yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def compact_frame() -> pd.DataFrame:
    rows = []
    compact_columns = MODULE.selected_compact_columns(CONFIG)
    for well_index, well in enumerate(("well_a", "well_b")):
        for row_index in range(6):
            row = {
                "id": f"{well}_{row_index}",
                "well": well,
                "well_row_idx": row_index + 10,
                "outer_fold": well_index,
                "md_since": float(row_index + 1),
            }
            for column_index, column in enumerate(compact_columns):
                value = float(1 + row_index + column_index)
                if "top1_minus_anchor" in column and row_index % 2 == 0:
                    value *= -1.0
                if column == "selector__confidence_valid_count":
                    value = float(12 - row_index)
                if column == "selector__available_count":
                    value = 12.0
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def minimal_risk_frame(wells: list[str], offset: float = 0.0) -> pd.DataFrame:
    families = [
        "score_dispersion",
        "candidate_divergence",
        "top1_anchor_distance",
        "confidence_coverage",
        "geometry_context",
    ]
    frame = pd.DataFrame({"well": wells})
    for family_index, family in enumerate(families):
        frame[f"{family}__full__signal__mean"] = (
            np.arange(len(wells), dtype=np.float64) + offset + family_index
        )
    return frame


def test_compact_aggregation_applies_fixed_target_free_transforms():
    result = MODULE.aggregate_compact_partition(compact_frame(), CONFIG)
    assert result["well"].tolist() == ["well_a", "well_b"]
    assert len(result) == 2
    confidence_columns = [
        column
        for column in result
        if column.startswith("confidence_coverage__prefix128")
        and column.endswith("confidence_valid_count__mean")
    ]
    assert len(confidence_columns) == 1
    # candidate_count - confidence_valid_count = [0,1,2,3,4,5]
    assert result.loc[0, confidence_columns[0]] == pytest.approx(2.5)
    anchor_columns = [
        column
        for column in result
        if column.startswith("top1_anchor_distance__prefix128")
        and column.endswith("__mean")
    ]
    assert anchor_columns
    assert all(result[column].ge(0).all() for column in anchor_columns)


def test_single_parquet_reader_ignores_hive_parent_partition_column(tmp_path: Path):
    partition_dir = tmp_path / "downstream_outer_fold=0" / "role=train"
    partition_dir.mkdir(parents=True)
    path = partition_dir / "compact.parquet"
    expected = pd.DataFrame(
        {
            "id": ["a", "b"],
            "downstream_outer_fold": pd.Series([0, 0], dtype="int8"),
        }
    )
    expected.to_parquet(path, index=False)
    actual = MODULE.read_parquet_columns(path, ["id", "downstream_outer_fold"])
    pd.testing.assert_frame_equal(actual, expected)


def test_config_pins_corrected_exp264_stage_c_v6_and_stage_d_v3():
    assert CONFIG["experiment"]["route"] == "ml_model"
    assert CONFIG["data"]["stage_c_kernel_version"] == 6
    assert CONFIG["data"]["stage_c_expected_manifest_sha256"] == (
        "f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c"
    )
    assert CONFIG["data"]["stage_c_expected_partition_manifest_sha256"] == (
        "17930b7b50da7c783bffb8db8e34a0f69e5e583e028bde5b356d50a63bfacf66"
    )
    assert CONFIG["data"]["stage_d_kernel_version"] == 3
    assert CONFIG["data"]["stage_d_expected_oof_sha256"] == (
        "b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2"
    )
    assert CONFIG["guards"]["technical"]["expected_worsened_wells"] == 255
    assert CONFIG["guards"]["technical"]["expected_over_0p25_wells"] == 220


def test_corrected_stage_c_manifest_contract_resolves_locally():
    root, partitions, schema = MODULE.load_stage_c_contract(CONFIG)
    assert "stage_c_v6" in str(root)
    assert len(partitions) == 25
    assert int(partitions["rows"].sum()) == 18_919_945
    assert len(schema["features"]) == 74


def test_risk_fit_uses_outer_train_distribution_and_rejects_label_like_extras():
    train = minimal_risk_frame([f"train_{index}" for index in range(8)])
    valid = minimal_risk_frame(["valid_low", "valid_high"], offset=0.5)
    train_scores, valid_scores, preprocessor = MODULE.fit_target_free_risk(
        train, valid, [0.70, 0.80, 0.90]
    )
    assert len(train_scores) == 8
    assert len(valid_scores) == 2
    assert set(preprocessor["thresholds"]) == {"q70", "q80", "q90"}
    assert valid_scores["risk_score"].between(0.0, 1.0).all()
    train["actual_tvt"] = np.linspace(-1e9, 1e9, len(train))
    valid["delta_rmse_addonly_minus_control"] = [999.0, -999.0]
    with pytest.raises(ValueError, match="cannot enter target-free risk fit"):
        MODULE.fit_target_free_risk(train, valid, [0.70, 0.80, 0.90])


def test_risk_fit_rejects_outer_train_valid_well_overlap():
    train = minimal_risk_frame(["shared", "train_only"])
    valid = minimal_risk_frame(["shared", "valid_only"], offset=1.0)
    with pytest.raises(ValueError, match="overlap"):
        MODULE.fit_target_free_risk(train, valid, [0.70])


def test_outer_fold_risk_keeps_unsuffixed_partition_metadata():
    parts = []
    for fold in range(5):
        train = minimal_risk_frame([f"f{fold}_train_{index}" for index in range(4)])
        valid = minimal_risk_frame([f"f{fold}_valid_{index}" for index in range(2)], offset=0.5)
        for frame, role, source_fold in (
            (train, "train", (fold + 1) % 5),
            (valid, "valid", fold),
        ):
            frame["downstream_outer_fold"] = fold
            frame["role"] = role
            frame["source_outer_fold"] = source_fold
            parts.append(frame)
    config = copy.deepcopy(CONFIG)
    config["guards"]["technical"]["expected_wells"] = 10
    features, scores, preprocessors = MODULE.build_outer_fold_risk(
        pd.concat(parts, ignore_index=True), config
    )
    assert "downstream_outer_fold" in features
    assert "role" in features
    assert not any(column.endswith(("_x", "_y")) for column in features)
    assert len(scores) == 10
    assert len(preprocessors) == 5


def test_geometry_builder_does_not_read_tvt_values(tmp_path: Path):
    raw_dir = tmp_path / "train"
    raw_dir.mkdir()
    raw = pd.DataFrame(
        {
            "MD": np.arange(8, dtype=float),
            "X": np.arange(8, dtype=float) * 2,
            "Y": np.arange(8, dtype=float) * -1,
            "Z": np.arange(8, dtype=float) * 0.5,
            "GR": [10.0, 11.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0],
            "TVT": np.linspace(-1e6, 1e6, 8),
            "TVT_input": np.linspace(1e9, -1e9, 8),
        }
    )
    raw.to_csv(raw_dir / "well_a__horizontal_well.csv", index=False)
    keys = pd.DataFrame({"well": ["well_a"] * 4, "well_row_idx": [3, 4, 5, 6]})
    first, _ = MODULE.build_geometry_features(keys, raw_dir, CONFIG)
    raw["TVT"] *= -1000
    raw["TVT_input"] += 123456789
    raw.to_csv(raw_dir / "well_a__horizontal_well.csv", index=False)
    second, _ = MODULE.build_geometry_features(keys, raw_dir, CONFIG)
    pd.testing.assert_frame_equal(first, second)


def test_gate_readout_keeps_target_connection_after_risk_assignment():
    risk_rows = []
    oof_rows = []
    for fold in range(5):
        for well_index in range(4):
            well = f"f{fold}_w{well_index}"
            is_risk = well_index < 2
            risk_rows.append(
                {
                    "well": well,
                    "downstream_outer_fold": fold,
                    "risk_score": 0.9 if is_risk else 0.1,
                    "risk_family__candidate_divergence": 0.9 if is_risk else 0.1,
                    "risk_q70": is_risk,
                    "risk_q80": is_risk,
                    "risk_q90": is_risk,
                }
            )
            if is_risk:
                addonly_prediction = 4.0
            elif well_index == 2:
                addonly_prediction = 3.0
            else:
                addonly_prediction = 1.0
            oof_rows.append(
                {
                    "id": f"{well}_0",
                    "well": well,
                    "outer_fold": fold,
                    "actual_tvt": 0.0,
                    MODULE.CONTROL_COLUMN: 2.0,
                    MODULE.ADDONLY_COLUMN: addonly_prediction,
                }
            )
    config = copy.deepcopy(CONFIG)
    config["guards"]["technical"]["expected_worsened_wells"] = 15
    config["guards"]["technical"]["expected_over_0p25_wells"] = 15
    fold, pooled, by_well, gated_oof, guard = MODULE.evaluate_target_free_gates(
        pd.DataFrame(risk_rows), pd.DataFrame(oof_rows), config
    )
    assert len(fold) == 15
    assert len(pooled) == 3
    assert len(by_well) == 60
    assert all(f"gated_q{value}__pred_tvt" in gated_oof for value in (70, 80, 90))
    assert pooled["gt0__risk_bad_rate_lift_vs_safe"].gt(1.0).all()
    assert set(guard["guard_by_quantile"]) == {"q70", "q80", "q90"}
