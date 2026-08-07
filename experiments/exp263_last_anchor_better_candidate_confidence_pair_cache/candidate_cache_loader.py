from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from candidate_cache_contract import (
    COMMON_CONFIDENCE_SLOTS,
    CORE_CANDIDATE_IDS,
    NAMED_COMBINATIONS,
    PAIR_SHORTLIST,
    candidate_by_id,
    topological_formula_order,
    validate_selectable_names,
)

KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
VALUE_COLUMNS = [
    *KEY_COLUMNS,
    "candidate_id",
    "candidate_name",
    "family",
    "source_exp",
    "rawtest_status",
    "formula",
    "last_known_tvt",
    "candidate_tvt",
    "candidate_minus_last",
    "candidate_finite",
    "candidate_available",
    "fallback_used",
    "coverage_valid",
]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep logical hashes stable across pandas object/string inference changes."""
    string_columns = [
        column
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.StringDtype)
    ]
    if not string_columns:
        return frame
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].astype(object)
    return normalized


def frame_content_sha256(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def assert_key_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError(f"row count mismatch: {len(left)} != {len(right)}")
    for column in KEY_COLUMNS:
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        if column == "md_since":
            equal = np.array_equal(a, b, equal_nan=True)
        else:
            equal = np.array_equal(a, b)
        if not equal:
            raise ValueError(f"candidate key mismatch in {column}")


def _weights_for_formula(spec: Mapping[str, Any], fold: int | None) -> dict[str, float]:
    if spec["kind"] == "outer_crossfit":
        if fold is None:
            raise ValueError("outer_crossfit formula requires one explicit outer fold")
        if fold < 0 or fold >= len(spec["fold_weights"]):
            raise ValueError(f"invalid outer fold: {fold}")
        weights = dict(spec["fold_weights"][fold])
    else:
        weights = dict(spec["weights"])
    if set(weights) != set(spec["components"]):
        raise ValueError("formula component/weight mismatch")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError("formula weights must sum to one")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("formula weights must be non-negative")
    return weights


def materialize_formula_frames(
    name: str,
    frames: Mapping[str, pd.DataFrame],
    components: Iterable[str],
    weights: Mapping[str, float],
    formula: str,
) -> pd.DataFrame:
    component_list = list(components)
    if set(component_list) != set(weights):
        raise ValueError("components and weights differ")
    if not component_list:
        raise ValueError("formula has no components")
    base = frames[component_list[0]].reset_index(drop=True)
    for component in component_list[1:]:
        assert_key_alignment(base, frames[component].reset_index(drop=True))

    values = np.zeros(len(base), dtype=np.float64)
    available = np.ones(len(base), dtype=bool)
    coverage = np.ones(len(base), dtype=bool)
    fallback = np.zeros(len(base), dtype=bool)
    component_values: dict[str, np.ndarray] = {}
    for component in component_list:
        frame = frames[component].reset_index(drop=True)
        value = pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(np.float64)
        component_values[component] = value
        values += float(weights[component]) * value
        available &= frame["candidate_available"].astype(bool).to_numpy()
        coverage &= frame["coverage_valid"].astype(bool).to_numpy()
        fallback |= frame["fallback_used"].astype(bool).to_numpy()

    out = base[KEY_COLUMNS].copy()
    out["candidate_id"] = name
    out["candidate_name"] = name
    out["family"] = "virtual_combination"
    out["source_exp"] = "exp263"
    out["rawtest_status"] = "formula_components_controlled_by_manifest"
    out["formula"] = formula
    out["last_known_tvt"] = base["last_known_tvt"].to_numpy(np.float32)
    out["candidate_tvt"] = values.astype(np.float32)
    out["candidate_minus_last"] = (
        out["candidate_tvt"].to_numpy(np.float32)
        - out["last_known_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    out["candidate_finite"] = np.isfinite(values)
    out["candidate_available"] = available
    out["fallback_used"] = fallback
    out["coverage_valid"] = coverage

    matrix = np.column_stack([component_values[item] for item in component_list])
    out["component_range"] = (np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)).astype(
        np.float32
    )
    out["component_std"] = np.nanstd(matrix, axis=1).astype(np.float32)
    if len(component_list) == 2:
        left, right = component_list
        signed = component_values[left] - component_values[right]
        midpoint = 0.5 * (component_values[left] + component_values[right])
        anchor = out["last_known_tvt"].to_numpy(np.float64)
        out["signed_disagreement"] = signed.astype(np.float32)
        out["absolute_disagreement"] = np.abs(signed).astype(np.float32)
        out["mean_50_tvt"] = midpoint.astype(np.float32)
        out["midpoint_minus_anchor"] = (midpoint - anchor).astype(np.float32)
        left_direction = np.sign(component_values[left] - anchor)
        right_direction = np.sign(component_values[right] - anchor)
        out["parent_to_anchor_direction_agreement"] = left_direction == right_direction
    return out


@dataclass
class CandidateCache:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._catalog = self._load_json("candidate_catalog.json", default=[])
        self._named = self._load_json(
            "named_combinations.json", default=NAMED_COMBINATIONS
        )
        self._pair_rows = {
            item.pair_id: item.as_manifest_row() for item in PAIR_SHORTLIST
        }
        self._named_rows = (
            dict(self._named.get("combinations", {}))
            if "combinations" in self._named
            else dict(self._named)
        )
        topological_formula_order()

    def _load_json(self, relative: str, default: Any) -> Any:
        path = self.root / relative
        if not path.exists():
            return default
        return json.loads(path.read_text())

    @property
    def catalog(self) -> list[dict[str, Any]]:
        if isinstance(self._catalog, dict):
            return list(self._catalog.get("candidates", []))
        return list(self._catalog)

    def list_available(self) -> dict[str, list[str]]:
        return {
            "primitive": list(CORE_CANDIDATE_IDS),
            "pair": sorted(self._pair_rows),
            "named": sorted(self._named_rows),
        }

    def validate_selectable(self, names: Iterable[str]) -> None:
        validate_selectable_names(names)
        known = set().union(*self.list_available().values())
        unknown = set(names) - known
        if unknown:
            raise KeyError(f"unknown selectable candidates: {sorted(unknown)}")

    def _partition_paths(self, kind: str, candidate_id: str, fold: int | None) -> list[Path]:
        base = self.root / kind / candidate_id
        if fold is not None:
            paths = sorted((base / f"fold={fold}").glob("*.parquet"))
        else:
            paths = sorted(base.glob("fold=*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no {kind} partition for {candidate_id}, fold={fold}")
        return paths

    @staticmethod
    def _read_partitions(
        paths: Iterable[Path],
        columns: Iterable[str] | None = None,
        row_slice: slice | None = None,
    ) -> pd.DataFrame:
        selected = list(columns) if columns is not None else None
        frames = [pd.read_parquet(path, columns=selected) for path in paths]
        frame = pd.concat(frames, ignore_index=True)
        if row_slice is not None:
            frame = frame.iloc[row_slice].reset_index(drop=True)
        return frame

    def load_primitive(
        self,
        candidate_id: str,
        *,
        fold: int | None,
        columns: Iterable[str] | None = None,
        row_slice: slice | None = None,
    ) -> pd.DataFrame:
        if candidate_id not in CORE_CANDIDATE_IDS:
            raise KeyError(f"not a core primitive: {candidate_id}")
        paths = self._partition_paths("candidate_values", candidate_id, fold)
        return self._read_partitions(paths, columns=columns, row_slice=row_slice)

    def load_confidence(
        self,
        candidate_id: str,
        *,
        fold: int | None,
        row_slice: slice | None = None,
    ) -> pd.DataFrame:
        paths = self._partition_paths("candidate_confidence", candidate_id, fold)
        return self._read_partitions(paths, row_slice=row_slice)

    def _formula_spec(self, name: str, fold: int | None) -> tuple[list[str], dict[str, float], str]:
        if name in self._pair_rows:
            spec = self._pair_rows[name]
            components = list(spec["components"])
            weights = dict(spec["weights"])
            return components, weights, str(spec["formula"])
        if name in self._named_rows:
            spec = self._named_rows[name]
            components = list(spec["components"])
            weights = _weights_for_formula(spec, fold)
            formula = str(spec.get("formula") or json.dumps(weights, sort_keys=True))
            return components, weights, formula
        raise KeyError(name)

    def materialize(
        self,
        name: str,
        *,
        fold: int | None,
        row_slice: slice | None = None,
        include_confidence: bool = False,
    ) -> pd.DataFrame:
        if name in CORE_CANDIDATE_IDS:
            return self.load_primitive(name, fold=fold, row_slice=row_slice)
        components, weights, formula = self._formula_spec(name, fold)
        if any(component not in CORE_CANDIDATE_IDS for component in components):
            raise ValueError("recursive formula closure is forbidden")
        frames = {
            component: self.load_primitive(component, fold=fold, row_slice=row_slice)
            for component in components
        }
        out = materialize_formula_frames(name, frames, components, weights, formula)
        if include_confidence:
            out = self._attach_formula_confidence(out, components, fold, row_slice)
        return out

    def _attach_formula_confidence(
        self,
        out: pd.DataFrame,
        components: list[str],
        fold: int | None,
        row_slice: slice | None,
    ) -> pd.DataFrame:
        confidence_frames: dict[str, pd.DataFrame] = {}
        for component in components:
            confidence = self.load_confidence(component, fold=fold, row_slice=row_slice)
            assert_key_alignment(out, confidence)
            confidence_frames[component] = confidence
            out[f"{component}__confidence_valid"] = confidence[
                "confidence_valid"
            ].astype(bool)
            for column in confidence.columns:
                if column in KEY_COLUMNS or column in {"candidate_id", "confidence_valid"}:
                    continue
                out[f"{component}__{column}"] = confidence[column].to_numpy()
        valid_columns = [f"{component}__confidence_valid" for component in components]
        out["all_parent_confidence_valid"] = out[valid_columns].all(axis=1)

        families = {candidate_by_id(component).family for component in components}
        if len(families) == 1:
            for slot in COMMON_CONFIDENCE_SLOTS:
                namespaced = [f"{component}__{slot}" for component in components]
                if not all(column in out.columns for column in namespaced):
                    continue
                values = out[namespaced].apply(pd.to_numeric, errors="coerce")
                out[f"shared_{slot}_min"] = values.min(axis=1)
                out[f"shared_{slot}_max"] = values.max(axis=1)
                out[f"shared_{slot}_mean"] = values.mean(axis=1)
                denominator = values.min(axis=1).abs().clip(lower=1e-12)
                out[f"shared_{slot}_range_ratio"] = (
                    values.max(axis=1) - values.min(axis=1)
                ) / denominator
        return out
