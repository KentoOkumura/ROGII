from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from src.candidate_selector_pipeline import (
    build_stage_d_exp218_surface,
    load_stage_d_compact_fold,
    sha256_file,
    sha256_json,
    verify_stage_c_artifact_root,
    write_json,
)

FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
TARGET_HORIZONTAL_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")
SIGNAL_COMPONENT_COLUMNS = (
    "pf_ancc",
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
    "last_known_tvt",
)


def canonical_formation_feature_names() -> list[str]:
    names = ["sig_std", "sig_mean_d"]
    for formation in FORMATION_COLUMNS:
        names.extend(
            [
                f"tvtF_{formation}",
                f"tvtFw_{formation}",
                f"tvtF50_{formation}",
                f"bw_{formation}",
                f"bww_{formation}",
                f"bw50_{formation}",
                f"bw_early_{formation}",
                f"bw_mid_{formation}",
            ]
        )
    names.extend(f"frm_rmse_{formation}" for formation in FORMATION_COLUMNS)
    names.extend(
        [
            "form_mean_d",
            "form_std_d",
            "form_rng_d",
            "spatial_ancc_d",
            "spatial_knn_dist",
            "dense_ancc",
            "dense_std",
            "dense_dist",
            "tvt_dense_d",
            "tvt_densew_d",
            "tvt_dense50_d",
            "dense_rmse",
            "dense_bias",
            "dense_nb_std",
            "pf_vs_spatial",
            "pf_vs_dense",
            "spatial_vs_dense",
            "beam_vs_spatial",
        ]
    )
    if len(names) != 74 or len(set(names)) != 74:
        raise AssertionError("canonical formation feature schema must contain 74 unique names")
    return names


def select_unique_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Select requested columns once and reject ambiguous source schemas."""
    duplicated_source = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated_source:
        raise ValueError(
            f"{context} source has duplicate column labels: {sorted(set(duplicated_source))}"
        )
    ordered = list(dict.fromkeys(str(column) for column in columns))
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} columns missing: {missing}")
    selected = frame.loc[:, ordered].copy()
    if not selected.columns.is_unique:
        raise AssertionError(f"{context} projection produced duplicate column labels")
    return selected


def load_formation_feature_contract(
    audit_path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    path = Path(audit_path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            f"formation availability audit SHA mismatch: {actual_sha256} != {expected_sha256}"
        )
    audit = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "feature",
        "family",
        "current_test_generated",
        "fold_safe",
        "hidden_safe",
        "status",
        "dependency",
        "evidence",
        "action",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"formation availability audit columns missing: {sorted(missing)}")
    selected = audit[
        audit["status"].eq("fail")
        & audit["family"].eq("base_replay")
        & audit["dependency"].eq("full_train_formation_reference")
    ].copy()
    features = selected["feature"].astype(str).tolist()
    canonical = canonical_formation_feature_names()
    if features != canonical:
        raise ValueError("the fixed 74-feature audit selection differs from the canonical schema")
    if not selected["action"].eq("drop_or_rebuild_inside_each_outer_fold").all():
        raise ValueError("the selected formation audit rows have an unexpected action")
    return features, {
        "path": str(path),
        "sha256": actual_sha256,
        "source_rows": int(len(audit)),
        "selected_rows": int(len(selected)),
        "selection": "status=fail,family=base_replay,dependency=full_train_formation_reference",
        "feature_schema_sha256": sha256_json(features),
    }


def formation_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = dict(config["model"]["formation_addonly_stage"])
    variants = [str(value) for value in stage["active_variants"]]
    config_indices = [int(value) for value in stage["lightgbm_config_indices"]]
    folds = int(stage["folds"])
    planned = int(stage["planned_gpu_boosters"])
    calculated = len(variants) * len(config_indices) * folds
    if variants != ["fold_safe_formation_74_addonly"]:
        raise ValueError(f"unexpected formation variants: {variants}")
    if config_indices != [0, 1, 2]:
        raise ValueError(f"unexpected LightGBM config indices: {config_indices}")
    if folds != 5 or calculated != 15 or planned != 15:
        raise ValueError("cost contract must be 1 variant x 3 configs x 5 folds = 15 boosters")
    if bool(stage.get("control_retraining", True)):
        raise ValueError("saved exp264 corrected 347-feature control must not be retrained")
    expected = {
        "clean_base_feature_count": 273,
        "nested_compact_feature_count": 74,
        "fold_safe_formation_feature_count": 74,
        "final_feature_count": 421,
    }
    for key, value in expected.items():
        if int(stage.get(key, -1)) != value:
            raise ValueError(f"formation cost surface mismatch: {key}={stage.get(key)}")
    return {
        "active_variants": variants,
        "lightgbm_config_indices": config_indices,
        "folds": folds,
        "configs": len(config_indices),
        "planned_gpu_boosters": calculated,
        "parent_control_retraining": False,
        **expected,
    }


def _horizontal_path(data_dir: Path, well: str) -> Path:
    return Path(data_dir) / f"{well}__horizontal_well.csv"


def _typewell_path(data_dir: Path, well: str) -> Path:
    return Path(data_dir) / f"{well}__typewell.csv"


def audit_raw_schema(
    *,
    train_dir: Path,
    test_dir: Path,
    train_wells: Sequence[str],
    expected_test_wells_minimum: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_train_horizontal = set(TARGET_HORIZONTAL_COLUMNS) | set(FORMATION_COLUMNS)
    required_test_horizontal = set(TARGET_HORIZONTAL_COLUMNS)
    required_typewell = set(TYPEWELL_COLUMNS)
    for split, data_dir, wells, required_horizontal in [
        ("train", Path(train_dir), sorted(set(map(str, train_wells))), required_train_horizontal),
        (
            "current_test",
            Path(test_dir),
            sorted(
                path.name.removesuffix("__horizontal_well.csv")
                for path in Path(test_dir).glob("*__horizontal_well.csv")
            ),
            required_test_horizontal,
        ),
    ]:
        if split == "current_test" and len(wells) < int(expected_test_wells_minimum):
            raise ValueError(
                f"current-test horizontal well count {len(wells)} is below "
                f"{expected_test_wells_minimum}"
            )
        for well in wells:
            horizontal_path = _horizontal_path(data_dir, well)
            typewell_path = _typewell_path(data_dir, well)
            if not horizontal_path.exists() or not typewell_path.exists():
                raise FileNotFoundError(f"raw well pair missing for {split}/{well}")
            horizontal_columns = set(pd.read_csv(horizontal_path, nrows=0).columns)
            typewell_columns = set(pd.read_csv(typewell_path, nrows=0).columns)
            horizontal_missing = sorted(required_horizontal - horizontal_columns)
            typewell_missing = sorted(required_typewell - typewell_columns)
            rows.append(
                {
                    "split": split,
                    "well": well,
                    "horizontal_required_columns": ",".join(sorted(required_horizontal)),
                    "horizontal_missing_columns": ",".join(horizontal_missing),
                    "typewell_missing_columns": ",".join(typewell_missing),
                    "target_generator_reads_formation_columns": False,
                    "reference_fit_reads_formation_columns": split == "train",
                    "passed": not horizontal_missing and not typewell_missing,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or not result["passed"].all():
        failed = result[~result["passed"]]
        raise ValueError(f"raw formation schema audit failed: {failed.head(20).to_dict('records')}")
    return result


@dataclass(frozen=True)
class FormationReferenceCatalog:
    plane_wells: np.ndarray
    plane_xy: np.ndarray
    formation_medians: np.ndarray
    dense_wells: np.ndarray
    dense_xy: np.ndarray
    dense_ancc: np.ndarray
    samples_per_well: int

    @classmethod
    def from_raw(
        cls,
        train_dir: Path,
        wells: Sequence[str],
        *,
        samples_per_well: int,
    ) -> FormationReferenceCatalog:
        plane_wells: list[str] = []
        plane_xy: list[list[float]] = []
        formation_medians: list[np.ndarray] = []
        dense_wells: list[str] = []
        dense_xy: list[np.ndarray] = []
        dense_ancc: list[np.ndarray] = []
        for well in sorted(set(map(str, wells))):
            path = _horizontal_path(Path(train_dir), well)
            raw = pd.read_csv(path, usecols=["X", "Y", *FORMATION_COLUMNS])
            plane_frame = raw.dropna(subset=["X", "Y", *FORMATION_COLUMNS])
            dense_frame = raw.dropna(subset=["X", "Y", "ANCC"])
            if not plane_frame.empty:
                medians = plane_frame[list(FORMATION_COLUMNS)].median().to_numpy(np.float64)
                xy_median = plane_frame[["X", "Y"]].median().to_numpy(np.float64)
                if not np.isfinite(medians).all() or not np.isfinite(xy_median).all():
                    raise ValueError(f"reference well has nonfinite formation summary: {well}")
                plane_wells.append(well)
                plane_xy.append(xy_median.tolist())
                formation_medians.append(medians)
            if not dense_frame.empty:
                sample_indices = np.linspace(
                    0,
                    len(dense_frame) - 1,
                    min(int(samples_per_well), len(dense_frame)),
                    dtype=int,
                )
                sample = dense_frame.iloc[sample_indices]
                dense_wells.extend([well] * len(sample))
                dense_xy.append(sample[["X", "Y"]].to_numpy(np.float64))
                dense_ancc.append(sample["ANCC"].to_numpy(np.float32))
        if not plane_wells:
            raise ValueError("formation plane reference catalog is empty")
        if not dense_ancc:
            raise ValueError("dense ANCC reference catalog is empty")
        catalog = cls(
            plane_wells=np.asarray(plane_wells, dtype=object),
            plane_xy=np.asarray(plane_xy, dtype=np.float64),
            formation_medians=np.asarray(formation_medians, dtype=np.float64),
            dense_wells=np.asarray(dense_wells, dtype=object),
            dense_xy=np.concatenate(dense_xy, axis=0),
            dense_ancc=np.concatenate(dense_ancc).astype(np.float32),
            samples_per_well=int(samples_per_well),
        )
        for values, name in [
            (catalog.plane_xy, "plane_xy"),
            (catalog.formation_medians, "formation_medians"),
            (catalog.dense_xy, "dense_xy"),
            (catalog.dense_ancc, "dense_ancc"),
        ]:
            if not np.isfinite(values).all():
                raise ValueError(f"reference catalog contains nonfinite {name}")
        return catalog

    def fit(
        self,
        reference_wells: Sequence[str],
        *,
        plane_k: int,
        dense_k: int,
        dense_nfetch: int,
        query_workers: int,
    ) -> tuple[FormationPlaneKNN, DenseANCCImputer, dict[str, Any]]:
        reference = set(map(str, reference_wells))
        plane_mask = np.asarray([str(well) in reference for well in self.plane_wells])
        dense_mask = np.asarray([str(well) in reference for well in self.dense_wells])
        actual_plane = sorted(set(map(str, self.plane_wells[plane_mask])))
        actual_dense = sorted(set(map(str, self.dense_wells[dense_mask])))
        missing_plane = sorted(reference - set(actual_plane))
        missing_dense = sorted(reference - set(actual_dense))
        if len(actual_plane) < int(plane_k) + 1:
            raise ValueError(
                "formation plane catalog lacks enough available reference wells: "
                f"{len(actual_plane)} < {int(plane_k) + 1}"
            )
        if int(dense_mask.sum()) < int(dense_k) + 1:
            raise ValueError(
                "dense ANCC catalog lacks enough available reference rows: "
                f"{int(dense_mask.sum())} < {int(dense_k) + 1}"
            )
        plane = FormationPlaneKNN(
            wells=self.plane_wells[plane_mask],
            xy=self.plane_xy[plane_mask],
            formation_medians=self.formation_medians[plane_mask],
            k=int(plane_k),
            query_workers=int(query_workers),
        )
        dense = DenseANCCImputer(
            wells=self.dense_wells[dense_mask],
            xy=self.dense_xy[dense_mask],
            ancc=self.dense_ancc[dense_mask],
            k=int(dense_k),
            nfetch=int(dense_nfetch),
            query_workers=int(query_workers),
        )
        return (
            plane,
            dense,
            {
                "reference_wells": len(reference),
                "reference_well_sha256": sha256_json(sorted(reference)),
                "plane_reference_wells": len(actual_plane),
                "plane_reference_well_sha256": sha256_json(actual_plane),
                "plane_missing_reference_wells": missing_plane,
                "plane_missing_reference_well_sha256": sha256_json(missing_plane),
                "dense_reference_wells": len(actual_dense),
                "dense_reference_well_sha256": sha256_json(actual_dense),
                "dense_missing_reference_wells": missing_dense,
                "dense_missing_reference_well_sha256": sha256_json(missing_dense),
                "plane_rows": int(plane_mask.sum()),
                "dense_rows": int(dense_mask.sum()),
                "availability_policy": "skip_wells_without_complete_source_rows_per_imputer",
                "target_horizontal_columns": list(TARGET_HORIZONTAL_COLUMNS),
                "target_formation_columns_read": False,
            },
        )


def _query_matrix(
    tree: cKDTree, query: np.ndarray, k: int, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    distance, indices = tree.query(query, k=k, workers=workers)
    distance = np.asarray(distance, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distance.ndim == 1:
        if len(np.atleast_2d(query)) == 1 and k > 1:
            distance = distance[None, :]
            indices = indices[None, :]
        else:
            distance = distance[:, None]
            indices = indices[:, None]
    return distance, indices


@dataclass
class FormationPlaneKNN:
    wells: np.ndarray
    xy: np.ndarray
    formation_medians: np.ndarray
    k: int = 10
    query_workers: int = 1

    def __post_init__(self) -> None:
        if len(self.wells) < self.k + 1:
            raise ValueError("formation plane reference requires at least k+1 wells")
        self.wells = np.asarray(self.wells, dtype=object)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.formation_medians = np.asarray(self.formation_medians, dtype=np.float64)
        self.well_to_index = {str(well): index for index, well in enumerate(self.wells)}
        self.scale = np.where(self.xy.std(axis=0) < 1.0e-3, 1.0, self.xy.std(axis=0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(
        self, xy_query: np.ndarray, *, target_well: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=np.float64))
        nfetch = min(self.k + 5, len(self.wells))
        distance, indices = _query_matrix(
            self.tree,
            xy_query / self.scale,
            nfetch,
            self.query_workers,
        )
        target_index = self.well_to_index.get(str(target_well)) if target_well is not None else None
        if target_index is not None:
            distance = np.where(indices == target_index, np.inf, distance)
        take = min(self.k, distance.shape[1])
        order = np.argpartition(distance, take - 1, axis=1)[:, :take]
        selected_distance = np.take_along_axis(distance, order, axis=1)
        selected_index = np.take_along_axis(indices, order, axis=1)
        valid = np.isfinite(selected_distance)
        if (valid.sum(axis=1) < min(self.k, len(self.wells) - 1)).any():
            raise ValueError("formation plane query lacks enough non-self reference wells")
        weight = np.where(valid, 1.0 / (selected_distance + 1.0e-3), 0.0)
        xn = self.xy[selected_index, 0]
        yn = self.xy[selected_index, 1]
        fn = self.formation_medians[selected_index]
        wx = weight * xn
        wy = weight * yn
        matrix = np.zeros((len(xy_query), 3, 3), dtype=np.float64)
        matrix[:, 0, 0] = (wx * xn).sum(axis=1)
        matrix[:, 0, 1] = (wx * yn).sum(axis=1)
        matrix[:, 0, 2] = wx.sum(axis=1)
        matrix[:, 1, 0] = matrix[:, 0, 1]
        matrix[:, 1, 1] = (wy * yn).sum(axis=1)
        matrix[:, 1, 2] = wy.sum(axis=1)
        matrix[:, 2, 0] = matrix[:, 0, 2]
        matrix[:, 2, 1] = matrix[:, 1, 2]
        matrix[:, 2, 2] = weight.sum(axis=1)
        matrix[:, 0, 0] += 1.0e-9
        matrix[:, 1, 1] += 1.0e-9
        matrix[:, 2, 2] += 1.0e-9
        rhs = np.stack(
            [
                (wx[:, :, None] * fn).sum(axis=1),
                (wy[:, :, None] * fn).sum(axis=1),
                (weight[:, :, None] * fn).sum(axis=1),
            ],
            axis=1,
        )
        try:
            coefficients = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.stack(
                [
                    np.linalg.pinv(row_matrix) @ row_rhs
                    for row_matrix, row_rhs in zip(matrix, rhs, strict=True)
                ]
            )
        prediction = (
            xy_query[:, 0, None] * coefficients[:, 0]
            + xy_query[:, 1, None] * coefficients[:, 1]
            + coefficients[:, 2]
        ).astype(np.float32)
        minimum_distance = np.where(valid, selected_distance, np.inf).min(axis=1)
        return prediction, minimum_distance.astype(np.float32)


@dataclass
class DenseANCCImputer:
    wells: np.ndarray
    xy: np.ndarray
    ancc: np.ndarray
    k: int = 20
    nfetch: int = 5000
    query_workers: int = 1

    def __post_init__(self) -> None:
        self.wells = np.asarray(self.wells, dtype=object)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.ancc = np.asarray(self.ancc, dtype=np.float32)
        if len(self.ancc) < self.k + 1:
            raise ValueError("dense ANCC reference requires at least k+1 samples")
        self.scale = np.where(self.xy.std(axis=0) < 1.0e-3, 1.0, self.xy.std(axis=0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(
        self,
        xy_query: np.ndarray,
        *,
        target_well: str | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=np.float64))
        nfetch = min(self.nfetch, len(self.ancc))
        distance, indices = _query_matrix(
            self.tree,
            xy_query / self.scale,
            nfetch,
            self.query_workers,
        )
        if target_well is not None:
            distance = np.where(self.wells[indices] == str(target_well), np.inf, distance)
        take = min(self.k, distance.shape[1])
        order = np.argpartition(distance, take - 1, axis=1)[:, :take]
        selected_distance = np.take_along_axis(distance, order, axis=1)
        selected_index = np.take_along_axis(indices, order, axis=1)
        valid = np.isfinite(selected_distance)
        if (valid.sum(axis=1) < self.k).any():
            raise ValueError("dense ANCC query lacks enough non-self reference samples")
        weight = np.where(valid, 1.0 / (selected_distance + 1.0e-3), 0.0)
        weight_sum = weight.sum(axis=1)
        if (weight_sum <= 1.0e-9).any():
            raise ValueError("dense ANCC query has zero reference weight")
        values = self.ancc[selected_index]
        prediction = (values * weight).sum(axis=1) / weight_sum
        variance = ((values - prediction[:, None]) ** 2 * weight).sum(axis=1) / weight_sum
        minimum_distance = np.where(valid, selected_distance, np.inf).min(axis=1)
        return (
            prediction.astype(np.float32),
            np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
            minimum_distance.astype(np.float32),
        )


def _segment_bias(
    known_tvt: np.ndarray,
    known_z: np.ndarray,
    formation: np.ndarray,
) -> tuple[float, float, float, float, float]:
    bias = known_tvt + known_z - formation
    n_rows = len(bias)
    full = float(np.median(bias))
    late = float(np.median(bias[max(0, n_rows - 50) :])) if n_rows >= 5 else full
    first, second = n_rows // 3, 2 * n_rows // 3
    early = float(np.median(bias[: max(1, first)])) if first > 0 else full
    middle = float(np.median(bias[first : max(first + 1, second)])) if second > first else full
    weight = np.exp(0.02 * np.arange(n_rows, dtype=np.float64))
    weight /= weight.sum()
    weighted = float(np.dot(weight, bias))
    return full, early, middle, late, weighted


def _broadcast(value: float, rows: int) -> np.ndarray:
    return np.full(rows, np.float32(value), dtype=np.float32)


def build_well_formation_features(
    *,
    well: str,
    base_well: pd.DataFrame,
    raw_dir: Path,
    plane: FormationPlaneKNN,
    dense: DenseANCCImputer,
    reference_wells: set[str],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    horizontal = pd.read_csv(
        _horizontal_path(Path(raw_dir), well),
        usecols=list(TARGET_HORIZONTAL_COLUMNS),
    )
    typewell = pd.read_csv(
        _typewell_path(Path(raw_dir), well),
        usecols=list(TYPEWELL_COLUMNS),
    ).sort_values("TVT")
    known = horizontal[horizontal["TVT_input"].notna()]
    evaluation = horizontal[horizontal["TVT_input"].isna()]
    if len(known) < 10 or evaluation.empty:
        raise ValueError(f"well lacks the required known/evaluation rows: {well}")
    expected_ids = pd.Series([f"{well}_{index}" for index in evaluation.index], dtype=str)
    base_well = base_well.reset_index(drop=True)
    if not base_well["id"].astype(str).reset_index(drop=True).equals(expected_ids):
        raise ValueError(f"base surface is not aligned to raw evaluation rows for well={well}")
    missing_signals = set(SIGNAL_COMPONENT_COLUMNS) - set(base_well.columns)
    if missing_signals:
        raise ValueError(
            f"base surface lacks formation signal components: {sorted(missing_signals)}"
        )
    last_known_tvt = float(known["TVT_input"].iloc[-1])
    base_anchor = base_well["last_known_tvt"].to_numpy(np.float32)
    if (
        not np.isfinite(base_anchor).all()
        or float(np.max(np.abs(base_anchor - last_known_tvt))) > 1.0e-4
    ):
        raise ValueError(f"last-known TVT alignment mismatch for well={well}")
    known_tvt = known["TVT_input"].to_numpy(np.float32)
    known_z = known["Z"].to_numpy(np.float32)
    eval_z = evaluation["Z"].to_numpy(np.float32)
    target_well = well if well in reference_wells else None
    formation_eval, spatial_distance = plane.impute(
        evaluation[["X", "Y"]].to_numpy(np.float64),
        target_well=target_well,
    )
    formation_known, _ = plane.impute(
        known[["X", "Y"]].to_numpy(np.float64),
        target_well=target_well,
    )
    values: dict[str, np.ndarray] = {}
    formation_predictions: list[np.ndarray] = []
    formation_rmse: dict[str, float] = {}
    for index, formation_name in enumerate(FORMATION_COLUMNS):
        full, early, middle, late, weighted = _segment_bias(
            known_tvt,
            known_z,
            formation_known[:, index],
        )
        prediction = (-eval_z + formation_eval[:, index] + full).astype(np.float32)
        values[f"tvtF_{formation_name}"] = prediction
        values[f"tvtFw_{formation_name}"] = (-eval_z + formation_eval[:, index] + weighted).astype(
            np.float32
        )
        values[f"tvtF50_{formation_name}"] = (-eval_z + formation_eval[:, index] + late).astype(
            np.float32
        )
        values[f"bw_{formation_name}"] = _broadcast(full, len(evaluation))
        values[f"bww_{formation_name}"] = _broadcast(weighted, len(evaluation))
        values[f"bw50_{formation_name}"] = _broadcast(late, len(evaluation))
        values[f"bw_early_{formation_name}"] = _broadcast(early, len(evaluation))
        values[f"bw_mid_{formation_name}"] = _broadcast(middle, len(evaluation))
        known_prediction = -known_z + formation_known[:, index] + full
        formation_rmse[formation_name] = float(
            np.sqrt(np.mean(np.square(known_tvt - known_prediction, dtype=np.float64)))
        )
        formation_predictions.append(prediction)
    formation_matrix = np.stack(formation_predictions, axis=1)
    values["form_mean_d"] = (formation_matrix.mean(axis=1) - np.float32(last_known_tvt)).astype(
        np.float32
    )
    values["form_std_d"] = formation_matrix.std(axis=1).astype(np.float32)
    values["form_rng_d"] = (formation_matrix.max(axis=1) - formation_matrix.min(axis=1)).astype(
        np.float32
    )
    for name, rmse_value in formation_rmse.items():
        values[f"frm_rmse_{name}"] = _broadcast(rmse_value, len(evaluation))

    dense_eval, dense_std, dense_distance = dense.impute(
        evaluation[["X", "Y"]].to_numpy(np.float64),
        target_well=target_well,
    )
    dense_known, dense_known_std, _ = dense.impute(
        known[["X", "Y"]].to_numpy(np.float64),
        target_well=target_well,
    )
    _, _, _, dense_late, dense_weighted = _segment_bias(
        known_tvt,
        known_z,
        dense_known,
    )
    dense_full = float(np.median(known_tvt + known_z - dense_known))
    dense_prediction = (-eval_z + dense_eval + dense_full).astype(np.float32)
    dense_weighted_prediction = (-eval_z + dense_eval + dense_weighted).astype(np.float32)
    dense_late_prediction = (-eval_z + dense_eval + dense_late).astype(np.float32)
    dense_known_residual = known_tvt + known_z - dense_known
    type_tvt = typewell["TVT"].to_numpy(np.float64)
    type_gr = typewell["GR"].to_numpy(np.float64)
    if len(type_tvt) < 3 or not np.isfinite(type_tvt).all() or not np.isfinite(type_gr).all():
        raise ValueError(f"typewell is invalid for well={well}")
    spatial = values["tvtF_ANCC"]
    pf = base_well["pf_ancc"].to_numpy(np.float32)
    beam_names = [
        "beam_cons_d",
        "beam_loose_d",
        "beam_vcons_d",
        "beam_sm5_d",
        "beam_vloose_d",
        "beam_mid_d",
        "beam_stiff_d",
    ]
    beam_absolute = [
        base_well[name].to_numpy(np.float32) + np.float32(last_known_tvt) for name in beam_names
    ]
    scale_absolute = [
        base_well[name].to_numpy(np.float32) + np.float32(last_known_tvt)
        for name in ["sc8_d", "sc15_d", "sc25_d", "sc_ens_d"]
    ]
    signal_matrix = np.stack(
        [pf, *beam_absolute, *scale_absolute, spatial, dense_prediction],
        axis=1,
    )
    values["sig_std"] = signal_matrix.std(axis=1).astype(np.float32)
    values["sig_mean_d"] = (signal_matrix.mean(axis=1) - np.float32(last_known_tvt)).astype(
        np.float32
    )
    values["spatial_ancc_d"] = (
        formation_eval[:, 0] - np.float32(np.interp(last_known_tvt, type_tvt, type_gr))
    ).astype(np.float32)
    values["spatial_knn_dist"] = spatial_distance.astype(np.float32)
    values["dense_ancc"] = dense_eval.astype(np.float32)
    values["dense_std"] = dense_std.astype(np.float32)
    values["dense_dist"] = dense_distance.astype(np.float32)
    values["tvt_dense_d"] = (dense_prediction - np.float32(last_known_tvt)).astype(np.float32)
    values["tvt_densew_d"] = (dense_weighted_prediction - np.float32(last_known_tvt)).astype(
        np.float32
    )
    values["tvt_dense50_d"] = (dense_late_prediction - np.float32(last_known_tvt)).astype(
        np.float32
    )
    values["dense_rmse"] = _broadcast(
        float(np.sqrt(np.mean(np.square(dense_known_residual, dtype=np.float64)))),
        len(evaluation),
    )
    values["dense_bias"] = _broadcast(float(np.mean(dense_known_residual)), len(evaluation))
    values["dense_nb_std"] = _broadcast(float(np.mean(dense_known_std)), len(evaluation))
    values["pf_vs_spatial"] = (pf - spatial).astype(np.float32)
    values["pf_vs_dense"] = (pf - dense_prediction).astype(np.float32)
    values["spatial_vs_dense"] = (spatial - dense_prediction).astype(np.float32)
    values["beam_vs_spatial"] = (beam_absolute[0] - spatial).astype(np.float32)
    missing = set(feature_names) - set(values)
    extra = set(values) - set(feature_names)
    if missing or extra:
        raise ValueError(
            f"formation feature schema mismatch for {well}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = pd.DataFrame({"id": expected_ids, "well": well})
    for name in feature_names:
        result[name] = np.asarray(values[name], dtype=np.float32)
    numeric = result[list(feature_names)].to_numpy(np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(f"formation features contain nonfinite values for well={well}")
    return result


def build_fold_formation_surface(
    *,
    base_frame: pd.DataFrame,
    raw_train_dir: Path,
    catalog: FormationReferenceCatalog,
    reference_wells: Sequence[str],
    target_wells: Sequence[str],
    feature_names: Sequence[str],
    generator_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_set = set(map(str, reference_wells))
    target = sorted(set(map(str, target_wells)))
    plane, dense, reference_evidence = catalog.fit(
        sorted(reference_set),
        plane_k=int(generator_config["plane_k"]),
        dense_k=int(generator_config["dense_k"]),
        dense_nfetch=int(generator_config["dense_nfetch"]),
        query_workers=int(generator_config["query_workers"]),
    )
    base_by_well = {
        str(well): part.reset_index(drop=True)
        for well, part in base_frame[base_frame["well"].astype(str).isin(target)].groupby(
            "well", sort=True
        )
    }
    if set(base_by_well) != set(target):
        raise ValueError("target well set differs from base surface well set")
    parts = Parallel(
        n_jobs=int(generator_config["n_jobs"]),
        prefer="threads",
    )(
        delayed(build_well_formation_features)(
            well=well,
            base_well=base_by_well[well],
            raw_dir=raw_train_dir,
            plane=plane,
            dense=dense,
            reference_wells=reference_set,
            feature_names=feature_names,
        )
        for well in target
    )
    surface = pd.concat(parts, ignore_index=True)
    expected_ids = base_frame[base_frame["well"].astype(str).isin(target)]["id"].astype(str)
    surface_index = surface.set_index("id", drop=False)
    if not surface_index.index.is_unique:
        raise ValueError("fold-safe formation surface ids are not unique")
    missing_ids = set(expected_ids) - set(surface_index.index.astype(str))
    extra_ids = set(surface_index.index.astype(str)) - set(expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "fold-safe formation surface id mismatch: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )
    surface = surface_index.loc[expected_ids].reset_index(drop=True)
    expected_wells = (
        base_frame.loc[base_frame["well"].astype(str).isin(target), "well"]
        .astype(str)
        .reset_index(drop=True)
    )
    if not surface["well"].astype(str).equals(expected_wells):
        raise ValueError("fold-safe formation surface well alignment mismatch")
    return surface, {
        **reference_evidence,
        "target_wells": len(target),
        "target_well_sha256": sha256_json(target),
        "target_wells_inside_reference": int(sum(well in reference_set for well in target)),
        "target_wells_self_excluded_from_reference_query": int(
            sum(well in reference_set for well in target)
        ),
        "rows": int(len(surface)),
        "feature_count": len(feature_names),
        "feature_schema_sha256": sha256_json(list(feature_names)),
    }


def build_current_test_formation_surface(
    *,
    base_frame: pd.DataFrame,
    raw_train_dir: Path,
    raw_test_dir: Path,
    reference_wells: Sequence[str],
    feature_names: Sequence[str],
    generator_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the deployment formation surface from all train references.

    The target horizontal files are read from ``raw_test_dir`` and therefore need
    only MD/X/Y/Z/TVT_input.  If a public-test well name also exists in train, its
    train reference rows are self-excluded to preserve unseen-well semantics.
    """
    reference = sorted(set(map(str, reference_wells)))
    target = sorted(set(base_frame["well"].astype(str)))
    if not reference or not target:
        raise ValueError("current-test formation requires non-empty reference and target wells")
    catalog = FormationReferenceCatalog.from_raw(
        raw_train_dir,
        reference,
        samples_per_well=int(generator_config["dense_samples_per_well"]),
    )
    plane, dense, reference_evidence = catalog.fit(
        reference,
        plane_k=int(generator_config["plane_k"]),
        dense_k=int(generator_config["dense_k"]),
        dense_nfetch=int(generator_config["dense_nfetch"]),
        query_workers=int(generator_config["query_workers"]),
    )
    base_by_well = {
        str(well): part.reset_index(drop=True)
        for well, part in base_frame.groupby("well", sort=True)
    }
    if set(base_by_well) != set(target):
        raise ValueError("current-test target well set differs from base surface")
    reference_set = set(reference)
    parts = Parallel(
        n_jobs=int(generator_config["n_jobs"]),
        prefer="threads",
    )(
        delayed(build_well_formation_features)(
            well=well,
            base_well=base_by_well[well],
            raw_dir=raw_test_dir,
            plane=plane,
            dense=dense,
            reference_wells=reference_set,
            feature_names=feature_names,
        )
        for well in target
    )
    surface = pd.concat(parts, ignore_index=True)
    expected_ids = base_frame["id"].astype(str).reset_index(drop=True)
    surface_index = surface.set_index("id", drop=False)
    if not surface_index.index.is_unique:
        raise ValueError("current-test formation surface ids are not unique")
    missing_ids = set(expected_ids) - set(surface_index.index.astype(str))
    extra_ids = set(surface_index.index.astype(str)) - set(expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "current-test formation surface id mismatch: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )
    surface = surface_index.loc[expected_ids].reset_index(drop=True)
    expected_wells = base_frame["well"].astype(str).reset_index(drop=True)
    if not surface["well"].astype(str).equals(expected_wells):
        raise ValueError("current-test formation surface well alignment mismatch")
    values = surface[list(feature_names)].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("current-test formation surface contains nonfinite values")
    overlap = sorted(reference_set.intersection(target))
    return surface, {
        **reference_evidence,
        "generation_role": "current_test_all_train_reference",
        "target_raw_dir": str(raw_test_dir),
        "target_formation_columns_read": False,
        "target_wells": len(target),
        "target_well_sha256": sha256_json(target),
        "target_train_name_overlap_wells": overlap,
        "target_train_name_overlap_well_sha256": sha256_json(overlap),
        "target_train_name_overlap_self_excluded": len(overlap),
        "rows": int(len(surface)),
        "feature_count": len(feature_names),
        "feature_schema_sha256": sha256_json(list(feature_names)),
        "logical_content_sha256": logical_feature_content_sha256(surface, feature_names),
    }


def logical_feature_content_sha256(frame: pd.DataFrame, feature_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(feature_names), separators=(",", ":")).encode())
    for identifier in frame["id"].astype(str):
        encoded = identifier.encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    values = frame[list(feature_names)].to_numpy(np.float32, copy=False)
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def audit_feature_relationships(
    *,
    existing: pd.DataFrame,
    formation: pd.DataFrame,
    existing_features: Sequence[str],
    formation_features: Sequence[str],
    correlation_sample_rows: int,
) -> pd.DataFrame:
    existing_features = list(map(str, existing_features))
    formation_features = list(map(str, formation_features))
    if set(existing_features).intersection(formation_features):
        raise ValueError("formation feature names collide with the existing 347-feature surface")
    if (
        not existing[["id", "well"]]
        .reset_index(drop=True)
        .equals(formation[["id", "well"]].reset_index(drop=True))
    ):
        raise ValueError("existing and formation feature surfaces are not row-aligned")
    existing_values = existing[existing_features].to_numpy(np.float32, copy=False)
    formation_values = formation[formation_features].to_numpy(np.float32, copy=False)
    if not np.isfinite(existing_values).all() or not np.isfinite(formation_values).all():
        raise ValueError("feature relationship audit received nonfinite values")
    existing_hash: dict[str, list[int]] = {}
    for index, _name in enumerate(existing_features):
        column_hash = hashlib.sha256(
            existing_values[:, index].astype("<f4", copy=False).tobytes()
        ).hexdigest()
        existing_hash.setdefault(column_hash, []).append(index)
    sample_count = min(int(correlation_sample_rows), len(existing))
    sample_indices = np.unique(
        np.rint(np.linspace(0, len(existing) - 1, sample_count)).astype(np.int64)
    )
    existing_sample = existing_values[sample_indices].astype(np.float64)
    formation_sample = formation_values[sample_indices].astype(np.float64)

    def standardized(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(centered * centered, axis=0, keepdims=True))
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)

    existing_standardized = standardized(existing_sample)
    formation_standardized = standardized(formation_sample)
    pearson = formation_standardized.T @ existing_standardized
    existing_rank = np.column_stack(
        [
            rankdata(existing_sample[:, index]).astype(np.float64)
            for index in range(existing_sample.shape[1])
        ]
    )
    formation_rank = np.column_stack(
        [
            rankdata(formation_sample[:, index]).astype(np.float64)
            for index in range(formation_sample.shape[1])
        ]
    )
    spearman = standardized(formation_rank).T @ standardized(existing_rank)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(formation_features):
        column_hash = hashlib.sha256(
            formation_values[:, index].astype("<f4", copy=False).tobytes()
        ).hexdigest()
        duplicate_names = [
            existing_features[candidate]
            for candidate in existing_hash.get(column_hash, [])
            if np.array_equal(formation_values[:, index], existing_values[:, candidate])
        ]
        pearson_index = int(np.argmax(np.abs(pearson[index])))
        spearman_index = int(np.argmax(np.abs(spearman[index])))
        rows.append(
            {
                "formation_feature": name,
                "exact_duplicate_count": len(duplicate_names),
                "exact_duplicate_features": ",".join(duplicate_names),
                "max_abs_pearson_feature": existing_features[pearson_index],
                "max_abs_pearson": float(abs(pearson[index, pearson_index])),
                "max_abs_spearman_feature": existing_features[spearman_index],
                "max_abs_spearman": float(abs(spearman[index, spearman_index])),
                "correlation_sample_rows": int(len(sample_indices)),
                "pruned": False,
            }
        )
    return pd.DataFrame(rows)


def load_saved_exp264_control(
    *,
    path: Path,
    expected_sha256: str,
    base_frame: pd.DataFrame,
    expected_rmse: float,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not base_frame.columns.is_unique:
        duplicates = base_frame.columns[base_frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(
            "clean base surface has duplicate column labels before saved-control validation: "
            f"{sorted(set(duplicates))}"
        )
    path = Path(path)
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha256):
        raise ValueError(f"saved exp264 OOF SHA mismatch: {actual_sha} != {expected_sha256}")
    columns = [
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        "matched_control__lgb_mean__pred_tvt",
        "selector_compact_addonly__lgb_mean__pred_tvt",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("saved exp264 OOF ids are not unique")
    indexed = frame.set_index(frame["id"].astype(str), drop=False)
    base_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(base_ids):
        raise ValueError("saved exp264 OOF ids differ from the clean base surface")
    frame = indexed.loc[base_ids].reset_index(drop=True)
    if not frame["well"].astype(str).equals(base_frame["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved exp264 OOF well alignment mismatch")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    actual = frame["actual_tvt"].to_numpy(np.float32)
    if float(np.max(np.abs(truth - actual))) > 1.0e-4:
        raise ValueError("saved exp264 OOF truth differs from the clean base surface")
    parent = frame["selector_compact_addonly__lgb_mean__pred_tvt"].to_numpy(np.float32)
    rmse = _rmse(truth, parent)
    if abs(rmse - float(expected_rmse)) > float(tolerance):
        raise ValueError(f"saved exp264 parent RMSE mismatch: {rmse} != {expected_rmse}")
    return frame, {
        "path": str(path),
        "sha256": actual_sha,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "parent_prediction_column": "selector_compact_addonly__lgb_mean__pred_tvt",
        "clean_control_prediction_column": "matched_control__lgb_mean__pred_tvt",
        "parent_rmse": rmse,
    }


def _rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def run_preflight(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    clean_allowlist_path: Path,
    availability_audit_path: Path,
    saved_parent_oof_path: Path,
    raw_train_dir: Path,
    raw_test_dir: Path,
    hidden_like_assignment_path: Path,
    output_dir: Path,
    verify_stage_c_partition_sha256: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cost = formation_cost_contract(config)
    formation_features, formation_contract = load_formation_feature_contract(
        availability_audit_path,
        expected_sha256=str(config["data"]["formation_availability_audit_sha256"]),
    )
    stage_c = verify_stage_c_artifact_root(
        stage_c_root,
        config,
        verify_partition_sha256=verify_stage_c_partition_sha256,
        expected_compact_feature_count=cost["nested_compact_feature_count"],
    )
    base_frame, base_features, base_evidence, _, _ = build_stage_d_exp218_surface(
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        base_feature_allowlist_path=clean_allowlist_path,
        raw_train_dir=raw_train_dir,
        config=config,
    )
    retained = ["id", "well", "target", "last_known_tvt", "md_since", *base_features]
    base_frame = select_unique_columns(
        base_frame,
        retained,
        context="clean exp218 preflight surface",
    )
    if len(base_features) != cost["clean_base_feature_count"]:
        raise ValueError("clean base feature count mismatch")
    if set(formation_features).intersection(base_frame.columns):
        raise ValueError("old full-train formation columns survived the clean base projection")
    schema_audit = audit_raw_schema(
        train_dir=raw_train_dir,
        test_dir=raw_test_dir,
        train_wells=base_frame["well"].astype(str).unique(),
        expected_test_wells_minimum=int(
            config["validation"]["expected_current_test_wells_minimum"]
        ),
    )
    schema_audit.to_csv(output_dir / "raw_train_current_test_schema_audit.csv", index=False)
    saved_parent, saved_parent_evidence = load_saved_exp264_control(
        path=saved_parent_oof_path,
        expected_sha256=str(config["data"]["saved_exp264_stage_d_oof_sha256"]),
        base_frame=base_frame,
        expected_rmse=float(config["validation"]["parent_exp264_rmse"]),
        tolerance=float(config["validation"]["parent_metric_tolerance"]),
    )
    hidden_like_sha256 = sha256_file(hidden_like_assignment_path)
    expected_hidden_like_sha256 = str(config["data"]["hidden_like_assignment_sha256"])
    if hidden_like_sha256 != expected_hidden_like_sha256:
        raise ValueError(
            "hidden-like assignment SHA mismatch: "
            f"{hidden_like_sha256} != {expected_hidden_like_sha256}"
        )
    hidden = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    required_hidden = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    if required_hidden - set(hidden.columns):
        raise ValueError("hidden-like assignment schema is incomplete")
    manifest = {
        "schema_version": "1.0.0",
        "status": "preflight_passed_zero_boosters",
        "cost_contract": cost,
        "formation_contract": formation_contract,
        "current_test_generation_contract": {
            "reference": "all_train_wells",
            "target_horizontal_columns": list(TARGET_HORIZONTAL_COLUMNS),
            "target_formation_columns_read": False,
            "feature_schema_sha256": sha256_json(formation_features),
            "feature_count": len(formation_features),
            "prediction_or_submission_generated": False,
        },
        "stage_c": stage_c,
        "clean_base": base_evidence,
        "saved_parent": saved_parent_evidence,
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": hidden_like_sha256,
        },
        "raw_schema_audit": {
            "rows": int(len(schema_audit)),
            "train_wells": int(schema_audit["split"].eq("train").sum()),
            "current_test_wells": int(schema_audit["split"].eq("current_test").sum()),
            "passed": bool(schema_audit["passed"].all()),
            "sha256": sha256_file(output_dir / "raw_train_current_test_schema_audit.csv"),
        },
        "base_rows": int(len(base_frame)),
        "base_wells": int(base_frame["well"].nunique()),
        "saved_parent_rows": int(len(saved_parent)),
    }
    write_json(output_dir / "preflight_manifest.json", manifest)
    return manifest


def _prepare_fold_caches_and_audits(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    stage_c_evidence: Mapping[str, Any],
    base_frame: pd.DataFrame,
    base_features: Sequence[str],
    formation_features: Sequence[str],
    raw_train_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    generator = dict(config["formation_generator"])
    all_wells = sorted(base_frame["well"].astype(str).unique())
    catalog = FormationReferenceCatalog.from_raw(
        raw_train_dir,
        all_wells,
        samples_per_well=int(generator["dense_samples_per_well"]),
    )
    compact_features = [str(value) for value in stage_c_evidence["compact_features"]]
    manifest_rows: list[dict[str, Any]] = []
    relationship_parts: list[pd.DataFrame] = []
    for outer_fold in range(int(config["model"]["formation_addonly_stage"]["folds"])):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=outer_fold,
        )
        for role, compact in [("train", compact_train), ("valid", compact_valid)]:
            target_wells = sorted(compact["well"].astype(str).unique())
            reference_wells = sorted(compact_train["well"].astype(str).unique())
            formation, evidence = build_fold_formation_surface(
                base_frame=base_frame,
                raw_train_dir=raw_train_dir,
                catalog=catalog,
                reference_wells=reference_wells,
                target_wells=target_wells,
                feature_names=formation_features,
                generator_config=generator,
            )
            formation = (
                formation.set_index(formation["id"].astype(str), drop=False)
                .loc[compact["id"].astype(str)]
                .reset_index(drop=True)
            )
            if (
                not formation["well"]
                .astype(str)
                .equals(compact["well"].astype(str).reset_index(drop=True))
            ):
                raise ValueError("formation and compact well alignment mismatch before audit")
            base_index = pd.Index(base_frame["id"].astype(str))
            indices = base_index.get_indexer(compact["id"].astype(str))
            if np.any(indices < 0):
                raise ValueError("compact ids are absent from clean base during feature audit")
            existing = base_frame.iloc[indices][["id", "well", *base_features]].reset_index(
                drop=True
            )
            compact_aligned = compact.reset_index(drop=True)
            if not existing[["id", "well"]].equals(compact_aligned[["id", "well"]]):
                raise ValueError("base and compact surfaces are not aligned before audit")
            for name in compact_features:
                existing[name] = compact_aligned[name].to_numpy(np.float32, copy=False)
            relationships = audit_feature_relationships(
                existing=existing,
                formation=formation,
                existing_features=[*base_features, *compact_features],
                formation_features=formation_features,
                correlation_sample_rows=int(generator["correlation_sample_rows"]),
            )
            relationships.insert(0, "role", role)
            relationships.insert(0, "outer_fold", outer_fold)
            relationship_parts.append(relationships)
            cache_path = (
                output_dir
                / "fold_safe_formation"
                / f"downstream_outer_fold={outer_fold}"
                / f"role={role}"
                / "part-00000.parquet"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            formation.to_parquet(cache_path, index=False)
            manifest_rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "role": role,
                    "rows": len(formation),
                    "wells": int(formation["well"].nunique()),
                    "reference_wells": int(evidence["reference_wells"]),
                    "target_wells_inside_reference": int(evidence["target_wells_inside_reference"]),
                    "target_wells_self_excluded_from_reference_query": int(
                        evidence["target_wells_self_excluded_from_reference_query"]
                    ),
                    "reference_well_sha256": str(evidence["reference_well_sha256"]),
                    "target_well_sha256": str(evidence["target_well_sha256"]),
                    "path": str(cache_path.relative_to(output_dir)),
                    "file_sha256": sha256_file(cache_path),
                    "logical_content_sha256": logical_feature_content_sha256(
                        formation, formation_features
                    ),
                    "feature_schema_sha256": sha256_json(list(formation_features)),
                    "exact_duplicate_count": int(relationships["exact_duplicate_count"].sum()),
                    "correlation_pruned_count": 0,
                    "target_formation_columns_read": False,
                }
            )
            del formation, existing, compact_aligned, relationships
            gc.collect()
        del compact_train, compact_valid
        gc.collect()
    manifest = pd.DataFrame(manifest_rows)
    if len(manifest) != 10:
        raise AssertionError("formation cache manifest must contain train/valid for five folds")
    relationship = pd.concat(relationship_parts, ignore_index=True)
    manifest.to_csv(output_dir / "formation_fold_manifest.csv", index=False)
    relationship.to_csv(output_dir / "formation_feature_relationship_audit.csv", index=False)
    write_json(
        output_dir / "formation_fold_manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "five_fold_feature_generation_and_audit_passed_before_fit",
            "partitions": manifest.to_dict(orient="records"),
            "partition_count": len(manifest),
            "rows_across_fold_roles": int(manifest["rows"].sum()),
            "feature_count": len(formation_features),
            "feature_schema_sha256": sha256_json(list(formation_features)),
            "exact_duplicate_count": int(manifest["exact_duplicate_count"].sum()),
            "correlation_pruned_count": 0,
            "relationship_audit_sha256": sha256_file(
                output_dir / "formation_feature_relationship_audit.csv"
            ),
        },
    )
    return manifest_rows, relationship


def _load_formation_cache(
    *,
    output_dir: Path,
    outer_fold: int,
    role: str,
    compact: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    path = (
        Path(output_dir)
        / "fold_safe_formation"
        / f"downstream_outer_fold={outer_fold}"
        / f"role={role}"
        / "part-00000.parquet"
    )
    frame = pd.read_parquet(path, columns=["id", "well", *feature_names])
    if not frame[["id", "well"]].equals(compact[["id", "well"]].reset_index(drop=True)):
        raise ValueError(f"formation cache alignment mismatch for fold={outer_fold}, role={role}")
    values = frame[list(feature_names)].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("formation cache contains nonfinite values before fit")
    return frame


def _evaluate_guard(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    saved_parent: pd.DataFrame,
    oof_fold: np.ndarray,
    new_prediction: np.ndarray,
    hidden_like_assignment_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    parent = saved_parent["selector_compact_addonly__lgb_mean__pred_tvt"].to_numpy(np.float32)
    clean = saved_parent["matched_control__lgb_mean__pred_tvt"].to_numpy(np.float32)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mask = oof_fold == fold
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "parent_exp264_rmse": parent_rmse,
                "fold_safe_formation_rmse": new_rmse,
                "delta_rmse_new_minus_parent": new_rmse - parent_rmse,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base_frame["md_since"].to_numpy(np.float32)
    masks = {
        "all": np.ones(len(base_frame), dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        bucket_rows.append(
            {
                "bucket": name,
                "rows": int(mask.sum()),
                "parent_exp264_rmse": parent_rmse,
                "fold_safe_formation_rmse": new_rmse,
                "delta_rmse_new_minus_parent": new_rmse - parent_rmse,
            }
        )
    bucket_metrics = pd.DataFrame(bucket_rows)
    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str}).set_index(
        "well_id"
    )
    hidden_rows: list[dict[str, Any]] = []
    for column in [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]:
        mask = base_frame["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        hidden_rows.append(
            {
                "assignment": column,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "parent_exp264_rmse": parent_rmse,
                "fold_safe_formation_rmse": new_rmse,
                "delta_rmse_new_minus_parent": new_rmse - parent_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    well_frame = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "clean_273_control": clean,
            "parent_exp264_347": parent,
            "fold_safe_formation_421": new_prediction,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in well_frame.groupby("well", sort=True):
        clean_rmse = _rmse(group["actual_tvt"], group["clean_273_control"])
        parent_rmse = _rmse(group["actual_tvt"], group["parent_exp264_347"])
        new_rmse = _rmse(group["actual_tvt"], group["fold_safe_formation_421"])
        well_rows.append(
            {
                "well": well,
                "rows": len(group),
                "clean_273_control_rmse": clean_rmse,
                "parent_exp264_347_rmse": parent_rmse,
                "fold_safe_formation_421_rmse": new_rmse,
                "parent_minus_clean_delta": parent_rmse - clean_rmse,
                "new_minus_clean_delta": new_rmse - clean_rmse,
                "new_minus_parent_delta": new_rmse - parent_rmse,
            }
        )
    by_well = pd.DataFrame(well_rows)
    guard_config = dict(config["guards"]["promotion"])
    pooled_parent = _rmse(truth, parent)
    pooled_new = _rmse(truth, new_prediction)
    bucket_lookup = bucket_metrics.set_index("bucket")
    threshold_counts: dict[str, dict[str, int | bool]] = {}
    threshold_checks: list[bool] = []
    for threshold in [1.0, 3.0, 5.0]:
        parent_count = int((by_well["parent_minus_clean_delta"] > threshold).sum())
        new_count = int((by_well["new_minus_clean_delta"] > threshold).sum())
        passed = new_count <= parent_count
        threshold_counts[f"plus_{threshold:g}ft"] = {
            "parent_exp264_vs_clean_count": parent_count,
            "new_vs_clean_count": new_count,
            "nonincrease": passed,
        }
        threshold_checks.append(passed)
    checks = {
        "pooled_rmse_delta": (pooled_new - pooled_parent)
        <= float(guard_config["maximum_pooled_delta_rmse"]),
        "improved_folds": int((fold_metrics["delta_rmse_new_minus_parent"] < 0).sum())
        >= int(guard_config["minimum_improved_folds"]),
        "near_non_regression": float(bucket_lookup.loc["near_0_250", "delta_rmse_new_minus_parent"])
        <= float(guard_config["maximum_scope_delta_rmse"]),
        "mid_non_regression": float(
            bucket_lookup.loc["mid_250_1000", "delta_rmse_new_minus_parent"]
        )
        <= float(guard_config["maximum_scope_delta_rmse"]),
        "long_tail_non_regression": float(
            bucket_lookup.loc["1000_plus", "delta_rmse_new_minus_parent"]
        )
        <= float(guard_config["maximum_scope_delta_rmse"]),
        "hidden_like_non_regression": float(hidden_metrics["delta_rmse_new_minus_parent"].max())
        <= float(guard_config["maximum_scope_delta_rmse"]),
        "worst_well_regression": float(by_well["new_minus_parent_delta"].max())
        <= float(guard_config["maximum_worst_well_delta_rmse"]),
        "worsened_well_threshold_counts_nonincrease": all(threshold_checks),
    }
    guard = {
        "parent_exp264_rmse": pooled_parent,
        "fold_safe_formation_rmse": pooled_new,
        "delta_rmse_new_minus_parent": pooled_new - pooled_parent,
        "improved_folds": int((fold_metrics["delta_rmse_new_minus_parent"] < 0).sum()),
        "worst_well_delta_rmse_new_minus_parent": float(by_well["new_minus_parent_delta"].max()),
        "worsened_well_threshold_counts": threshold_counts,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    return guard, fold_metrics, bucket_metrics, hidden_metrics, by_well


def run_fold_safe_formation_train(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    clean_allowlist_path: Path,
    availability_audit_path: Path,
    saved_parent_oof_path: Path,
    raw_train_dir: Path,
    raw_test_dir: Path,
    hidden_like_assignment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(config["execution"].get("run_approved", False)):
        raise RuntimeError("15-booster Kaggle train requires execution.run_approved=true")
    cost = formation_cost_contract(config)
    formation_features, formation_contract = load_formation_feature_contract(
        availability_audit_path,
        expected_sha256=str(config["data"]["formation_availability_audit_sha256"]),
    )
    stage_c_evidence = verify_stage_c_artifact_root(stage_c_root, config)
    base_frame, base_features, base_evidence, exp218, exp218_config = build_stage_d_exp218_surface(
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        base_feature_allowlist_path=clean_allowlist_path,
        raw_train_dir=raw_train_dir,
        config=config,
    )
    retained = ["id", "well", "target", "last_known_tvt", "md_since", *base_features]
    base_frame = select_unique_columns(
        base_frame,
        retained,
        context="clean exp218 train surface",
    )
    if set(formation_features).intersection(base_frame.columns):
        raise ValueError("old full-train formation columns survived clean projection")
    schema_audit = audit_raw_schema(
        train_dir=raw_train_dir,
        test_dir=raw_test_dir,
        train_wells=base_frame["well"].astype(str).unique(),
        expected_test_wells_minimum=int(
            config["validation"]["expected_current_test_wells_minimum"]
        ),
    )
    schema_audit.to_csv(output_dir / "raw_train_current_test_schema_audit.csv", index=False)
    saved_parent, saved_parent_evidence = load_saved_exp264_control(
        path=saved_parent_oof_path,
        expected_sha256=str(config["data"]["saved_exp264_stage_d_oof_sha256"]),
        base_frame=base_frame,
        expected_rmse=float(config["validation"]["parent_exp264_rmse"]),
        tolerance=float(config["validation"]["parent_metric_tolerance"]),
    )
    hidden_like_sha256 = sha256_file(hidden_like_assignment_path)
    expected_hidden_like_sha256 = str(config["data"]["hidden_like_assignment_sha256"])
    if hidden_like_sha256 != expected_hidden_like_sha256:
        raise ValueError(
            "hidden-like assignment SHA mismatch: "
            f"{hidden_like_sha256} != {expected_hidden_like_sha256}"
        )
    manifest_rows, relationship = _prepare_fold_caches_and_audits(
        config=config,
        stage_c_root=stage_c_root,
        stage_c_evidence=stage_c_evidence,
        base_frame=base_frame,
        base_features=base_features,
        formation_features=formation_features,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
    )
    if len(manifest_rows) != 10 or int(relationship["pruned"].sum()) != 0:
        raise AssertionError("all five fold feature audits must pass without pruning before fit")
    compact_features = [str(value) for value in stage_c_evidence["compact_features"]]
    final_features = [*base_features, *compact_features, *formation_features]
    if len(final_features) != cost["final_feature_count"] or len(set(final_features)) != len(
        final_features
    ):
        raise ValueError("421-feature surface contract mismatch")
    mode_name = str(config["model"]["formation_addonly_stage"]["mode"])
    mode_config = dict(exp218_config["model"]["training"]["modes"].get(mode_name, {}))
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False),
        mode_config,
    )
    config_indices = cost["lightgbm_config_indices"]
    params_family = [params_family[index] for index in config_indices]
    base_index = pd.Index(base_frame["id"].astype(str))
    n_rows = len(base_frame)
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof_by_config = [np.full(n_rows, np.nan, dtype=np.float32) for _ in params_family]
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    stage = dict(config["model"]["formation_addonly_stage"])
    chunk_columns = int(stage["matrix_copy_chunk_columns"])
    for outer_fold in range(cost["folds"]):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=outer_fold,
        )
        formation_train = _load_formation_cache(
            output_dir=output_dir,
            outer_fold=outer_fold,
            role="train",
            compact=compact_train,
            feature_names=formation_features,
        )
        formation_valid = _load_formation_cache(
            output_dir=output_dir,
            outer_fold=outer_fold,
            role="valid",
            compact=compact_valid,
            feature_names=formation_features,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("Stage C ids are absent from clean base before fit")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("outer train and valid indices overlap")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("OOF valid rows were assigned twice")
        oof_fold[valid_indices] = np.int8(outer_fold)
        x_train_values = np.empty((len(train_indices), len(final_features)), dtype=np.float32)
        x_valid_values = np.empty((len(valid_indices), len(final_features)), dtype=np.float32)
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            source = base_frame[columns]
            x_train_values[:, start:stop] = source.iloc[train_indices].to_numpy(
                np.float32, copy=True
            )
            x_valid_values[:, start:stop] = source.iloc[valid_indices].to_numpy(
                np.float32, copy=True
            )
        compact_start = len(base_features)
        formation_start = compact_start + len(compact_features)
        x_train_values[:, compact_start:formation_start] = compact_train[compact_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, compact_start:formation_start] = compact_valid[compact_features].to_numpy(
            np.float32, copy=False
        )
        x_train_values[:, formation_start:] = formation_train[formation_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, formation_start:] = formation_valid[formation_features].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(x_train_values).all() or not np.isfinite(x_valid_values).all():
            raise ValueError("421-feature matrix contains nonfinite values before fit")
        x_train = pd.DataFrame(x_train_values, columns=final_features, copy=False)
        x_valid = pd.DataFrame(x_valid_values, columns=final_features, copy=False)
        fold_predictions: list[np.ndarray] = []
        for family_position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                target[train_indices],
                eval_set=[(x_valid, target[valid_indices])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(stage["early_stopping_rounds"]), verbose=False),
                    log_evaluation(int(stage["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            prediction = model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            oof_by_config[family_position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            rmse_value = _rmse(truth[valid_indices], anchor[valid_indices] + prediction)
            model_rows.append(
                {
                    "variant": "fold_safe_formation_74_addonly",
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": outer_fold,
                    "feature_count": len(final_features),
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                }
            )
            fold_model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_indices),
                    "rmse_tvt": rmse_value,
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ["gain", "split"]:
                importance = model.booster_.feature_importance(importance_type=importance_type)
                importance_rows.extend(
                    {
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "importance_type": importance_type,
                        "feature": feature,
                        "feature_group": (
                            "fold_safe_formation"
                            if feature in formation_features
                            else "nested_compact"
                            if feature in compact_features
                            else "clean_base"
                        ),
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, importance, strict=True)
                )
            print(
                json.dumps(
                    {
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": rmse_value,
                        "completed_boosters": len(model_rows),
                        "planned_boosters": 15,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, prediction
            gc.collect()
        mean_prediction = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_indices),
                "rmse_tvt": _rmse(truth[valid_indices], anchor[valid_indices] + mean_prediction),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            formation_train,
            formation_valid,
            x_train,
            x_valid,
            x_train_values,
            x_valid_values,
            fold_predictions,
            mean_prediction,
        )
        gc.collect()
    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("15-model OOF contract is incomplete")
    for prediction in oof_by_config:
        if not np.isfinite(prediction).all():
            raise AssertionError("OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    guard, fold_metrics, bucket_metrics, hidden_metrics, by_well = _evaluate_guard(
        config=config,
        base_frame=base_frame,
        saved_parent=saved_parent,
        oof_fold=oof_fold,
        new_prediction=mean_prediction,
        hidden_like_assignment_path=hidden_like_assignment_path,
    )
    prediction_frame = base_frame[["id", "well", "md_since", "last_known_tvt", "target"]].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    for config_index, residual in zip(config_indices, oof_by_config, strict=True):
        prediction_frame[f"fold_safe_formation_74_addonly__lgb{config_index}__pred_tvt"] = (
            anchor + residual
        ).astype(np.float32)
    prediction_frame["fold_safe_formation_74_addonly__lgb_mean__pred_tvt"] = mean_prediction
    paths = {
        "oof": output_dir / "fold_safe_formation_oof_predictions.parquet",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "bucket_metrics": output_dir / "bucket_metrics.csv",
        "hidden_metrics": output_dir / "hidden_like_metrics.csv",
        "by_well": output_dir / "by_well_metrics.csv",
        "importance": output_dir / "feature_importance.csv",
        "model_manifest": output_dir / "model_manifest.json",
        "metrics": output_dir / "metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(
        fold_metrics,
        on="outer_fold",
        how="left",
    ).to_csv(paths["fold_metrics"], index=False)
    bucket_metrics.to_csv(paths["bucket_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    pd.DataFrame(importance_rows).to_csv(paths["importance"], index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "fold_safe_formation_15_gpu_boosters_completed",
        "cost_contract": cost,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "feature_groups": {
            "clean_base": list(base_features),
            "nested_compact": compact_features,
            "fold_safe_formation": list(formation_features),
        },
        "parent_control_retrained": False,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": "train_complete_guard_passed"
        if guard["passed"]
        else "train_complete_guard_failed",
        "cost_contract": cost,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": {
            "clean_base": len(base_features),
            "nested_compact": len(compact_features),
            "fold_safe_formation": len(formation_features),
            "final": len(final_features),
        },
        "guard": guard,
        "model_count": len(model_rows),
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "cost_contract": cost,
        "formation_contract": formation_contract,
        "stage_c_input": stage_c_evidence,
        "clean_base_input": base_evidence,
        "saved_parent_input": saved_parent_evidence,
        "formation_fold_manifest_sha256": sha256_file(output_dir / "formation_fold_manifest.json"),
        "formation_relationship_audit_sha256": sha256_file(
            output_dir / "formation_feature_relationship_audit.csv"
        ),
        "raw_schema_audit_sha256": sha256_file(
            output_dir / "raw_train_current_test_schema_audit.csv"
        ),
        "artifact_sha256": artifact_sha,
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "oof_prediction_sha256": artifact_sha["oof"],
        "submission_generated": False,
        "guard": guard,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    return metrics


__all__ = [
    "DenseANCCImputer",
    "FORMATION_COLUMNS",
    "FormationPlaneKNN",
    "FormationReferenceCatalog",
    "TARGET_HORIZONTAL_COLUMNS",
    "audit_feature_relationships",
    "audit_raw_schema",
    "build_current_test_formation_surface",
    "build_fold_formation_surface",
    "build_well_formation_features",
    "canonical_formation_feature_names",
    "formation_cost_contract",
    "load_formation_feature_contract",
    "logical_feature_content_sha256",
    "run_fold_safe_formation_train",
    "run_preflight",
    "select_unique_columns",
]
