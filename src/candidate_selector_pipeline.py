from __future__ import annotations

import gc
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
COMMON_CONFIDENCE_SLOTS = [
    "sigma_tvt",
    "loglik_per_row",
    "entropy",
    "score_margin",
    "support_count",
    "ess_fraction",
    "fallback_rate",
]
LABEL_COLUMNS = {"true_tvt", "candidate_abs_error", "candidate_within10"}
FORBIDDEN_FEATURE_TOKENS = (
    "true_tvt",
    "target",
    "absolute_error",
    "oracle",
    "catalog_rmse",
    "pair_readout",
    "outer_eligibility",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["string"]).columns:
        normalized[column] = normalized[column].astype(object)
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode())
    digest.update("|".join(str(dtype) for dtype in normalized.dtypes).encode())
    hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def resolve_existing_path(patterns: Sequence[str], search_roots: Sequence[Path]) -> Path:
    direct: list[Path] = []
    for raw in patterns:
        path = Path(raw)
        if path.exists():
            direct.append(path)
    if direct:
        return sorted(set(direct))[0]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(root.glob(pattern))
    matches = sorted({path for path in matches if path.exists()})
    if not matches:
        raise FileNotFoundError(f"no input matches patterns={list(patterns)}")
    return matches[0]


def resolve_exp263_cache_root(config: Mapping[str, Any], search_roots: Sequence[Path]) -> Path:
    data = dict(config.get("data", {}))
    patterns = [str(item) for item in data.get("exp263_stage0_root_patterns", [])]
    for raw in patterns:
        candidate = Path(raw)
        if (candidate / "cache_manifest.json").exists():
            return candidate
    manifest_patterns = [
        str(item)
        for item in data.get("exp263_stage0_manifest_patterns", ["**/cache_manifest.json"])
    ]
    manifest = resolve_existing_path(manifest_patterns, search_roots)
    return manifest.parent


def verify_exp263_root(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(config.get("data", {}))
    manifest_path = root / "cache_manifest.json"
    catalog_path = root / "candidate_catalog.json"
    if not manifest_path.exists() or not catalog_path.exists():
        raise FileNotFoundError(f"exp263 cache contract files missing under {root}")
    manifest_sha = sha256_file(manifest_path)
    catalog_sha = sha256_file(catalog_path)
    expected_manifest = str(data.get("exp263_expected_stage0_manifest_sha256", ""))
    expected_catalog = str(data.get("exp263_expected_catalog_sha256", ""))
    if expected_manifest and manifest_sha != expected_manifest:
        raise ValueError(f"exp263 manifest SHA mismatch: {manifest_sha}")
    if expected_catalog and catalog_sha != expected_catalog:
        raise ValueError(f"exp263 catalog SHA mismatch: {catalog_sha}")
    manifest = json.loads(manifest_path.read_text())
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "catalog_sha256": catalog_sha,
        "rows": int(manifest.get("rows", -1)),
        "wells": int(manifest.get("wells", -1)),
        "folds": int(manifest.get("folds", -1)),
    }


def candidate_contract_sha(contract: Mapping[str, Any]) -> str:
    return sha256_json(contract)


def candidate_ids(contract: Mapping[str, Any]) -> list[str]:
    return [str(item["id"]) for item in contract["score_candidates"]]


def primitive_ids(contract: Mapping[str, Any]) -> list[str]:
    return [
        str(item["id"]) for item in contract["score_candidates"] if str(item["kind"]) == "primitive"
    ]


def contract_by_id(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in contract["score_candidates"]}


def validate_candidate_contract(contract: Mapping[str, Any]) -> None:
    names = candidate_ids(contract)
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("exp264 requires exactly 12 unique score candidates")
    if len(primitive_ids(contract)) != 6:
        raise ValueError("exp264 requires exactly six primitives")
    specs = contract_by_id(contract)
    for name, spec in specs.items():
        parents = [str(item) for item in spec.get("parents", [])]
        weights = [float(item) for item in spec.get("weights", [])]
        if parents:
            if len(parents) != len(weights) or not np.isclose(sum(weights), 1.0):
                raise ValueError(f"invalid formula weights: {name}")
            if any(parent not in primitive_ids(contract) for parent in parents):
                raise ValueError(f"recursive or unknown parent: {name}")
    domains = contract["legal_domains"]
    if len(domains["primitive_pair_bank"]["candidates"]) != 11:
        raise ValueError("primitive_pair_bank must have 11 candidates")
    if len(domains["primitive_fixed_bank"]["candidates"]) != 7:
        raise ValueError("primitive_fixed_bank must have 7 candidates")


def assert_key_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError(f"candidate row mismatch: {len(left)} != {len(right)}")
    for column in KEY_COLUMNS:
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        equal = (
            np.array_equal(a, b, equal_nan=True) if column == "md_since" else np.array_equal(a, b)
        )
        if not equal:
            raise ValueError(f"candidate key mismatch in {column}")


def _read_one_partition(root: Path, kind: str, candidate_id: str, fold: int) -> pd.DataFrame:
    paths = sorted((root / kind / candidate_id / f"fold={fold}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing {kind}/{candidate_id}/fold={fold}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


@dataclass
class FoldBundle:
    base: pd.DataFrame
    values: np.ndarray
    available: np.ndarray
    confidence: dict[str, pd.DataFrame]
    candidate_ids: list[str]
    specs: dict[str, dict[str, Any]]


class Exp263CandidateCache:
    def __init__(self, root: Path, contract: Mapping[str, Any]):
        self.root = Path(root)
        self.contract = dict(contract)
        validate_candidate_contract(contract)
        self.ids = candidate_ids(contract)
        self.primitive_ids = primitive_ids(contract)
        self.specs = contract_by_id(contract)

    def load_fold(self, fold: int) -> FoldBundle:
        primitive_frames: dict[str, pd.DataFrame] = {}
        confidence: dict[str, pd.DataFrame] = {}
        for name in self.primitive_ids:
            frame = _read_one_partition(self.root, "candidate_values", name, fold)
            frame = frame.sort_values(["well", "well_row_idx"], kind="stable").reset_index(
                drop=True
            )
            primitive_frames[name] = frame
            conf = _read_one_partition(self.root, "candidate_confidence", name, fold)
            conf = conf.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)
            assert_key_alignment(frame, conf)
            confidence[name] = conf
        base_frame = primitive_frames[self.primitive_ids[0]]
        for name in self.primitive_ids[1:]:
            assert_key_alignment(base_frame, primitive_frames[name])
        base = base_frame[KEY_COLUMNS + ["last_known_tvt"]].copy()
        values_by_id: dict[str, np.ndarray] = {
            name: pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(np.float32)
            for name, frame in primitive_frames.items()
        }
        available_by_id: dict[str, np.ndarray] = {
            name: frame["candidate_available"].astype(bool).to_numpy()
            & np.isfinite(values_by_id[name])
            for name, frame in primitive_frames.items()
        }
        for name in self.ids:
            if name in values_by_id:
                continue
            spec = self.specs[name]
            parents = [str(item) for item in spec["parents"]]
            weights = np.asarray(spec["weights"], dtype=np.float32)
            combined = np.zeros(len(base), dtype=np.float32)
            for parent, weight in zip(parents, weights, strict=True):
                combined = (
                    combined
                    + np.float32(weight) * values_by_id[parent].astype(np.float32, copy=False)
                ).astype(np.float32)
            values_by_id[name] = combined
            available_by_id[name] = np.logical_and.reduce(
                [available_by_id[parent] for parent in parents]
            ) & np.isfinite(values_by_id[name])
        values = np.column_stack([values_by_id[name] for name in self.ids]).astype(np.float32)
        available = np.column_stack([available_by_id[name] for name in self.ids]).astype(bool)
        return FoldBundle(base, values, available, confidence, self.ids, self.specs)


def _raw_horizontal_path(raw_dir: Path, well: str) -> Path:
    path = Path(raw_dir) / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _raw_typewell_path(raw_dir: Path, well: str) -> Path:
    path = Path(raw_dir) / f"{well}__typewell.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def audit_raw_context_availability(
    train_dir: Path,
    test_dir: Path,
    horizontal_allowlist: Sequence[str],
) -> pd.DataFrame:
    """Fail closed when a configured raw context column is not present in both splits."""

    allowlist = [str(column) for column in horizontal_allowlist]
    if not allowlist or len(allowlist) != len(set(allowlist)):
        raise ValueError("raw context allowlist must be non-empty and unique")

    split_headers: dict[str, list[set[str]]] = {}
    split_counts: dict[str, int] = {}
    for split, directory in (("train", Path(train_dir)), ("current_test", Path(test_dir))):
        paths = sorted(directory.glob("*__horizontal_well.csv"))
        if not paths:
            raise FileNotFoundError(f"No horizontal well files under {directory}")
        headers = [set(pd.read_csv(path, nrows=0).columns.astype(str)) for path in paths]
        split_headers[split] = headers
        split_counts[split] = len(paths)

    rows = []
    for column in allowlist:
        train_present = sum(column in header for header in split_headers["train"])
        test_present = sum(column in header for header in split_headers["current_test"])
        rows.append(
            {
                "column": column,
                "train_files": split_counts["train"],
                "train_present_files": train_present,
                "current_test_files": split_counts["current_test"],
                "current_test_present_files": test_present,
                "train_all_present": train_present == split_counts["train"],
                "current_test_all_present": test_present == split_counts["current_test"],
                "availability_pass": train_present == split_counts["train"]
                and test_present == split_counts["current_test"],
            }
        )
    audit = pd.DataFrame(rows)
    failed = audit.loc[~audit["availability_pass"], "column"].astype(str).tolist()
    if failed:
        raise ValueError(
            "raw context columns are not identically available on actual train/current-test: "
            f"{failed}"
        )
    return audit


def build_raw_context(
    base: pd.DataFrame,
    raw_dir: Path,
    feature_config: Mapping[str, Any],
    *,
    require_truth: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    raw_cfg = dict(feature_config.get("raw_context", {}))
    allowlist = [str(item) for item in raw_cfg.get("horizontal_numeric_allowlist", [])]
    forbidden = {str(item) for item in raw_cfg.get("forbidden_columns", [])}
    if forbidden.intersection(allowlist):
        raise ValueError("raw context allowlist contains a forbidden target column")
    context = pd.DataFrame(index=np.arange(len(base)))
    context["ctx__md_since"] = pd.to_numeric(base["md_since"], errors="coerce").to_numpy(np.float32)
    context["ctx__last_known_tvt"] = pd.to_numeric(
        base["last_known_tvt"], errors="coerce"
    ).to_numpy(np.float32)
    context["ctx__well_row_idx"] = pd.to_numeric(base["well_row_idx"], errors="coerce").to_numpy(
        np.float32
    )
    for column in allowlist:
        context[f"ctx__raw__{column.lower()}"] = np.nan
        if raw_cfg.get("add_delta_from_last_known_row", True):
            context[f"ctx__raw_delta_last__{column.lower()}"] = np.nan
    typewell_columns = [
        "ctx__typewell__rows",
        "ctx__typewell__tvt_min",
        "ctx__typewell__tvt_max",
        "ctx__typewell__gr_mean",
        "ctx__typewell__gr_std",
        "ctx__typewell__gr_min",
        "ctx__typewell__gr_max",
        "ctx__typewell__row_gr_z",
    ]
    if raw_cfg.get("add_typewell_summary", True):
        for column in typewell_columns:
            context[column] = np.nan
    truth = np.full(len(base), np.nan, dtype=np.float32) if require_truth else None
    eval_len = base.groupby("well", sort=False)["id"].transform("size").to_numpy(np.float32)
    eval_position = base.groupby("well", sort=False).cumcount().to_numpy(np.float32) + 1.0
    context["ctx__eval_len"] = eval_len
    context["ctx__evaluation_progress"] = eval_position / np.maximum(eval_len, 1.0)
    for well, positions in base.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        raw = pd.read_csv(_raw_horizontal_path(raw_dir, str(well)))
        row_index = pd.to_numeric(base.iloc[pos]["well_row_idx"], errors="raise").to_numpy(np.int64)
        if row_index.min(initial=0) < 0 or row_index.max(initial=-1) >= len(raw):
            raise ValueError(f"raw row index out of bounds for well={well}")
        selected = raw.iloc[row_index]
        if require_truth:
            if "TVT" not in raw.columns:
                raise ValueError(f"training TVT missing for well={well}")
            assert truth is not None
            truth[pos] = pd.to_numeric(selected["TVT"], errors="coerce").to_numpy(np.float32)
        known = pd.to_numeric(raw.get("TVT_input"), errors="coerce").notna().to_numpy()
        previous_known = np.flatnonzero(
            known & (np.arange(len(raw)) < row_index.min(initial=len(raw)))
        )
        last_known_idx = (
            int(previous_known[-1]) if len(previous_known) else max(int(row_index.min()) - 1, 0)
        )
        for column in allowlist:
            if column not in raw.columns:
                continue
            current = pd.to_numeric(selected[column], errors="coerce").to_numpy(np.float32)
            context.loc[pos, f"ctx__raw__{column.lower()}"] = current
            delta_column = f"ctx__raw_delta_last__{column.lower()}"
            if delta_column in context:
                anchor = float(pd.to_numeric(raw.iloc[last_known_idx][column], errors="coerce"))
                context.loc[pos, delta_column] = current - np.float32(anchor)
        if raw_cfg.get("add_typewell_summary", True):
            typewell = pd.read_csv(_raw_typewell_path(raw_dir, str(well)))
            tw_tvt = pd.to_numeric(typewell.get("TVT"), errors="coerce").to_numpy(np.float64)
            tw_gr = pd.to_numeric(typewell.get("GR"), errors="coerce").to_numpy(np.float64)
            finite_gr = tw_gr[np.isfinite(tw_gr)]
            finite_tvt = tw_tvt[np.isfinite(tw_tvt)]
            gr_mean = float(np.mean(finite_gr)) if len(finite_gr) else np.nan
            gr_std = float(np.std(finite_gr)) if len(finite_gr) else np.nan
            summaries = {
                "ctx__typewell__rows": float(len(typewell)),
                "ctx__typewell__tvt_min": float(np.min(finite_tvt)) if len(finite_tvt) else np.nan,
                "ctx__typewell__tvt_max": float(np.max(finite_tvt)) if len(finite_tvt) else np.nan,
                "ctx__typewell__gr_mean": gr_mean,
                "ctx__typewell__gr_std": gr_std,
                "ctx__typewell__gr_min": float(np.min(finite_gr)) if len(finite_gr) else np.nan,
                "ctx__typewell__gr_max": float(np.max(finite_gr)) if len(finite_gr) else np.nan,
            }
            for column, value in summaries.items():
                context.loc[pos, column] = value
            if "GR" in selected.columns:
                row_gr = pd.to_numeric(selected["GR"], errors="coerce").to_numpy(np.float64)
                context.loc[pos, "ctx__typewell__row_gr_z"] = (row_gr - gr_mean) / max(gr_std, 1e-6)
    if require_truth and (truth is None or not np.isfinite(truth).all()):
        raise ValueError("truth join produced missing values")
    return context.astype(np.float32), truth


@dataclass
class ShapeState:
    values: np.ndarray
    previous: np.ndarray
    group_start: np.ndarray
    cumulative_abs_step: np.ndarray

    @classmethod
    def from_bundle(cls, base: pd.DataFrame, values: np.ndarray) -> ShapeState:
        n_rows, _ = values.shape
        previous = np.arange(n_rows, dtype=np.int64) - 1
        group_start = np.zeros(n_rows, dtype=np.int64)
        cumulative = np.zeros_like(values, dtype=np.float32)
        for positions in base.groupby("well", sort=False).indices.values():
            pos = np.asarray(positions, dtype=np.int64)
            start = int(pos[0])
            group_start[pos] = start
            previous[start] = start
            steps = np.zeros((len(pos), values.shape[1]), dtype=np.float32)
            if len(pos) > 1:
                steps[1:] = np.abs(np.diff(values[pos], axis=0)).astype(np.float32)
            cumulative[pos] = np.cumsum(steps, axis=0, dtype=np.float32)
        return cls(values, previous, group_start, cumulative)

    def extract(self, indices: np.ndarray, windows: Sequence[int]) -> dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        prev = np.maximum(self.previous[idx], self.group_start[idx])
        step = self.values[idx] - self.values[prev]
        prev2 = np.maximum(self.previous[prev], self.group_start[idx])
        previous_step = self.values[prev] - self.values[prev2]
        out = {
            "cand__step": step.astype(np.float32),
            "cand__curvature": (step - previous_step).astype(np.float32),
        }
        for window in windows:
            lag = np.maximum(idx - int(window), self.group_start[idx])
            span = np.maximum(idx - lag, 1).astype(np.float32)[:, None]
            net = self.values[idx] - self.values[lag]
            slope = net / span
            prev_idx = np.maximum(idx - 1, self.group_start[idx])
            prev_lag = np.maximum(prev_idx - int(window), self.group_start[idx])
            prev_span = np.maximum(prev_idx - prev_lag, 1).astype(np.float32)[:, None]
            previous_slope = (self.values[prev_idx] - self.values[prev_lag]) / prev_span
            path = self.cumulative_abs_step[idx] - self.cumulative_abs_step[lag]
            straightness = np.abs(net) / np.maximum(path, 1e-6)
            out[f"cand__slope_{window}"] = slope.astype(np.float32)
            out[f"cand__curvature_{window}"] = (slope - previous_slope).astype(np.float32)
            out[f"cand__straightness_{window}"] = straightness.astype(np.float32)
        return out


def deterministic_sample_indices(
    base: pd.DataFrame, limit: int | None, *seed_parts: Any
) -> np.ndarray:
    if limit is None or limit >= len(base):
        return np.arange(len(base), dtype=np.int64)
    rng = np.random.default_rng(stable_seed(*seed_parts))
    sampled = rng.choice(len(base), size=int(limit), replace=False)
    return np.sort(sampled.astype(np.int64))


def _confidence_numeric_fields(confidence: Mapping[str, pd.DataFrame]) -> list[str]:
    excluded = set(KEY_COLUMNS) | {
        "candidate_id",
        "confidence_source",
        "confidence_valid",
        "confidence_missing_fields",
    }
    fields: set[str] = set()
    for frame in confidence.values():
        for column in frame.columns:
            if column in excluded:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
                frame[column]
            ):
                fields.add(str(column))
    return sorted(fields)


def build_candidate_long_features(
    bundle: FoldBundle,
    raw_context: pd.DataFrame,
    indices: np.ndarray,
    feature_config: Mapping[str, Any],
    *,
    shape_state: ShapeState | None = None,
    expected_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = bundle.candidate_ids
    specs = bundle.specs
    n_candidates = len(ids)
    idx = np.asarray(indices, dtype=np.int64)
    n_rows = len(idx)
    if shape_state is None:
        shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
    windows = [int(item) for item in feature_config.get("shape_windows", [32, 128, 512])]
    shapes = shape_state.extract(idx, windows)
    values = bundle.values[idx]
    available = bundle.available[idx]
    anchor = pd.to_numeric(bundle.base.iloc[idx]["last_known_tvt"], errors="coerce").to_numpy(
        np.float32
    )
    bank_median = np.nanmedian(values, axis=1).astype(np.float32)
    bank_range = (np.nanmax(values, axis=1) - np.nanmin(values, axis=1)).astype(np.float32)
    bank_std = np.nanstd(values, axis=1).astype(np.float32)
    primary_ids = [str(item) for item in feature_config.get("primary_domain", ids[:-1])]
    fixed_ids = [str(item) for item in feature_config.get("fixed_domain", ids[:6] + [ids[-1]])]
    id_to_index = {name: position for position, name in enumerate(ids)}
    primary_pos = [id_to_index[name] for name in primary_ids]
    fixed_pos = [id_to_index[name] for name in fixed_ids]
    primary_values = values[:, primary_pos]
    fixed_values = values[:, fixed_pos]
    context = raw_context.iloc[idx].reset_index(drop=True)
    long = pd.DataFrame(index=np.arange(n_rows * n_candidates))
    for column in context.columns:
        long[column] = np.repeat(context[column].to_numpy(np.float32), n_candidates)
    candidate_value = values.reshape(-1)
    anchor_long = np.repeat(anchor, n_candidates)
    long["cand__tvt"] = candidate_value
    long["cand__minus_last"] = candidate_value - anchor_long
    long["cand__available"] = available.reshape(-1).astype(np.int8)
    long["cand__finite"] = np.isfinite(candidate_value).astype(np.int8)
    for name, matrix in shapes.items():
        long[name] = matrix.reshape(-1)
    long["bank__median"] = np.repeat(bank_median, n_candidates)
    long["bank__range"] = np.repeat(bank_range, n_candidates)
    long["bank__std"] = np.repeat(bank_std, n_candidates)
    long["bank__candidate_minus_median"] = candidate_value - np.repeat(bank_median, n_candidates)
    long["bank__candidate_abs_minus_median"] = np.abs(long["bank__candidate_minus_median"])
    long["bank__candidate_rank_fraction"] = (
        np.argsort(np.argsort(values, axis=1), axis=1).astype(np.float32) / max(n_candidates - 1, 1)
    ).reshape(-1)
    long["bank__candidate_is_min"] = (
        (values == np.nanmin(values, axis=1)[:, None]).reshape(-1).astype(np.int8)
    )
    long["bank__candidate_is_max"] = (
        (values == np.nanmax(values, axis=1)[:, None]).reshape(-1).astype(np.int8)
    )
    long["bank__candidate_mean_abs_disagreement"] = (
        np.mean(np.abs(values[:, :, None] - values[:, None, :]), axis=2)
        .astype(np.float32)
        .reshape(-1)
    )
    for domain, matrix in (("primary", primary_values), ("fixed", fixed_values)):
        long[f"bank__{domain}_median"] = np.repeat(np.nanmedian(matrix, axis=1), n_candidates)
        long[f"bank__{domain}_range"] = np.repeat(
            np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1), n_candidates
        )
        long[f"bank__{domain}_std"] = np.repeat(np.nanstd(matrix, axis=1), n_candidates)

    row_candidate_ids = np.tile(np.asarray(ids, dtype=object), n_rows)
    kind_names = sorted({str(specs[name]["kind"]) for name in ids})
    family_names = sorted({str(specs[name]["family"]) for name in ids})
    for name in ids:
        long[f"id__candidate__{name}"] = (row_candidate_ids == name).astype(np.int8)
    for kind in kind_names:
        members = {name for name in ids if str(specs[name]["kind"]) == kind}
        long[f"id__kind__{kind}"] = np.isin(row_candidate_ids, list(members)).astype(np.int8)
    for family in family_names:
        members = {name for name in ids if str(specs[name]["family"]) == family}
        long[f"id__family__{family}"] = np.isin(row_candidate_ids, list(members)).astype(np.int8)

    primitive_names = [name for name in ids if str(specs[name]["kind"]) == "primitive"]
    native_fields = _confidence_numeric_fields(bundle.confidence)
    conf_valid = np.zeros((n_rows, n_candidates), dtype=np.int8)
    native_arrays: dict[str, np.ndarray] = {
        field: np.full((n_rows, n_candidates), np.nan, dtype=np.float32) for field in native_fields
    }
    parent_valid_by_primitive: dict[str, np.ndarray] = {}
    parent_slot_by_primitive: dict[tuple[str, str], np.ndarray] = {}
    for candidate_position, name in enumerate(ids):
        if name not in bundle.confidence:
            continue
        frame = bundle.confidence[name].iloc[idx]
        valid = frame["confidence_valid"].astype(bool).to_numpy()
        conf_valid[:, candidate_position] = valid.astype(np.int8)
        parent_valid_by_primitive[name] = valid
        for field in native_fields:
            if field in frame.columns:
                native_arrays[field][:, candidate_position] = pd.to_numeric(
                    frame[field], errors="coerce"
                ).to_numpy(np.float32)
        for slot in COMMON_CONFIDENCE_SLOTS:
            if slot in frame.columns:
                parent_slot_by_primitive[(name, slot)] = pd.to_numeric(
                    frame[slot], errors="coerce"
                ).to_numpy(np.float32)
    parent_valid_count = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    component_range = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    component_std = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    direction_agreement = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    formula_weight_max = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    formula_weight_entropy = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    for candidate_position, name in enumerate(ids):
        parents = [str(item) for item in specs[name].get("parents", [])]
        if not parents:
            continue
        parent_positions = [id_to_index[parent] for parent in parents]
        components = values[:, parent_positions]
        component_range[:, candidate_position] = np.ptp(components, axis=1)
        component_std[:, candidate_position] = np.std(components, axis=1)
        directions = np.sign(components - anchor[:, None])
        direction_agreement[:, candidate_position] = np.all(
            directions == directions[:, :1], axis=1
        ).astype(np.float32)
        valid_components = [
            parent_valid_by_primitive.get(parent, np.zeros(n_rows, dtype=bool))
            for parent in parents
        ]
        parent_valid_count[:, candidate_position] = np.sum(valid_components, axis=0)
        weights = np.asarray(specs[name]["weights"], dtype=np.float64)
        formula_weight_max[:, candidate_position] = float(weights.max())
        formula_weight_entropy[:, candidate_position] = float(
            -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)))
        )
        conf_valid[:, candidate_position] = np.logical_or.reduce(valid_components).astype(np.int8)
    formula_parent_columns: dict[str, np.ndarray] = {}
    for parent in primitive_names:
        membership = np.zeros(n_candidates, dtype=bool)
        for candidate_position, name in enumerate(ids):
            membership[candidate_position] = parent in specs[name].get("parents", [])
        member_matrix = np.broadcast_to(membership, (n_rows, n_candidates))
        valid = parent_valid_by_primitive.get(parent, np.zeros(n_rows, dtype=bool))[:, None]
        formula_parent_columns[f"formula__parent__{parent}__confidence_valid"] = (
            (member_matrix & valid).reshape(-1).astype(np.int8)
        )
        for slot in COMMON_CONFIDENCE_SLOTS:
            values_slot = parent_slot_by_primitive.get(
                (parent, slot), np.full(n_rows, np.nan, dtype=np.float32)
            )[:, None]
            matrix = np.where(member_matrix, values_slot, np.nan)
            formula_parent_columns[f"formula__parent__{parent}__{slot}"] = matrix.reshape(-1)
    confidence_formula_columns: dict[str, np.ndarray] = {
        "conf__native_valid": conf_valid.reshape(-1)
    }
    confidence_formula_columns.update(
        {
            f"conf__native__{field}": matrix.reshape(-1)
            for field, matrix in native_arrays.items()
        }
    )
    confidence_formula_columns.update(
        {
            "formula__parent_valid_count": parent_valid_count.reshape(-1),
            "formula__component_range": component_range.reshape(-1),
            "formula__component_std": component_std.reshape(-1),
            "formula__parent_direction_agreement": direction_agreement.reshape(-1),
            "formula__weight_max": formula_weight_max.reshape(-1),
            "formula__weight_entropy": formula_weight_entropy.reshape(-1),
        }
    )
    confidence_formula_columns.update(formula_parent_columns)
    long = pd.concat(
        [long, pd.DataFrame(confidence_formula_columns, index=long.index)], axis=1
    )

    metadata = pd.DataFrame(
        {
            "id": np.repeat(bundle.base.iloc[idx]["id"].astype(str).to_numpy(), n_candidates),
            "well": np.repeat(bundle.base.iloc[idx]["well"].astype(str).to_numpy(), n_candidates),
            "well_row_idx": np.repeat(
                bundle.base.iloc[idx]["well_row_idx"].to_numpy(np.int32), n_candidates
            ),
            "outer_fold": np.repeat(
                bundle.base.iloc[idx]["outer_fold"].to_numpy(np.int8), n_candidates
            ),
            "md_since": np.repeat(
                bundle.base.iloc[idx]["md_since"].to_numpy(np.float32), n_candidates
            ),
            "candidate_id": row_candidate_ids,
            "candidate_tvt": candidate_value,
            "candidate_available": available.reshape(-1),
            "confidence_valid": conf_valid.reshape(-1).astype(bool),
        }
    )
    if expected_features is not None:
        for feature in expected_features:
            if feature not in long:
                long[feature] = np.nan
        long = long[list(expected_features)]
    else:
        long = long.copy()
    long = long.replace([np.inf, -np.inf], np.nan)
    for column in long.columns:
        if not pd.api.types.is_numeric_dtype(long[column]):
            raise TypeError(f"non-numeric selector feature: {column}")
    forbidden = [
        column
        for column in long.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"target-derived selector features detected: {forbidden}")
    return long, metadata


def validate_inference_feature_missingness(
    frame: pd.DataFrame,
    training_missing_rate_by_feature: Mapping[str, float],
    *,
    structural_prefixes: Sequence[str] = ("conf__", "formula__"),
    context: str = "inference feature matrix",
) -> pd.DataFrame:
    """Validate inference missingness without erasing LightGBM's learned NaN semantics."""

    expected_features = [str(feature) for feature in training_missing_rate_by_feature]
    if list(frame.columns) != expected_features:
        raise ValueError(f"{context} feature order differs from the training missingness contract")
    if frame.empty:
        raise ValueError(f"{context} is empty")

    matrix = frame.to_numpy(np.float32, copy=False)
    infinite_count = np.isinf(matrix).sum(axis=0).astype(np.int64)
    if np.any(infinite_count):
        offenders = [
            f"{feature}={int(count)}"
            for feature, count in zip(expected_features, infinite_count, strict=True)
            if count
        ]
        raise ValueError(f"{context} contains +/-inf: {offenders[:20]}")

    training_rates = np.asarray(
        [float(training_missing_rate_by_feature[feature]) for feature in expected_features],
        dtype=np.float64,
    )
    if (
        not np.isfinite(training_rates).all()
        or np.any(training_rates < 0.0)
        or np.any(training_rates >= 1.0)
    ):
        raise ValueError(f"{context} training missing rates must be finite in [0, 1)")

    missing_count = np.isnan(matrix).sum(axis=0).astype(np.int64)
    current_rates = missing_count.astype(np.float64) / float(len(frame))
    unexpected_dense_missing = [
        feature
        for feature, training_rate, count in zip(
            expected_features, training_rates, missing_count, strict=True
        )
        if training_rate <= 1.0e-12 and count
    ]
    if unexpected_dense_missing:
        raise ValueError(
            f"{context} introduced NaN in training-dense features: "
            f"{unexpected_dense_missing[:20]}"
        )

    structural = np.asarray(
        [feature.startswith(tuple(structural_prefixes)) for feature in expected_features],
        dtype=bool,
    )
    tolerance = max(0.5 / float(len(frame)), 1.0e-7)
    structural_mismatch = [
        (
            feature,
            float(training_rate),
            float(current_rate),
        )
        for feature, is_structural, training_rate, current_rate in zip(
            expected_features,
            structural,
            training_rates,
            current_rates,
            strict=True,
        )
        if is_structural and abs(current_rate - training_rate) > tolerance
    ]
    if structural_mismatch:
        raise ValueError(
            f"{context} structural NaN rate differs from training: "
            f"{structural_mismatch[:20]}"
        )

    return pd.DataFrame(
        {
            "feature": expected_features,
            "training_missing_rate": training_rates,
            "current_missing_rate": current_rates,
            "missing_count": missing_count,
            "structural_missingness": structural,
        }
    )


def add_candidate_labels(
    metadata: pd.DataFrame, truth: np.ndarray, n_candidates: int
) -> pd.DataFrame:
    labels = metadata.copy()
    true_long = np.repeat(np.asarray(truth, dtype=np.float32), n_candidates)
    error = np.abs(labels["candidate_tvt"].to_numpy(np.float32) - true_long)
    labels["true_tvt"] = true_long
    labels["candidate_abs_error"] = error.astype(np.float32)
    labels["candidate_within10"] = (error <= 10.0).astype(np.int8)
    return labels


def _exact_duplicate_groups(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, str]:
    seen: dict[tuple[str, int], str] = {}
    duplicate_of: dict[str, str] = {}
    for column in columns:
        series = frame[column]
        hashes = pd.util.hash_pandas_object(series, index=False, categorize=True).to_numpy(
            np.uint64
        )
        key = (
            str(series.dtype),
            int(hashlib.sha256(hashes.astype("<u8").tobytes()).hexdigest()[:16], 16),
        )
        if key in seen and series.equals(frame[seen[key]]):
            duplicate_of[column] = seen[key]
        else:
            seen[key] = column
    return duplicate_of


def feature_group(feature: str) -> str:
    return feature.split("__", 1)[0]


def feature_description(feature: str) -> str:
    descriptions = {
        "ctx": "raw train/current-testの両方で生成する候補非依存の行・well・typewell context",
        "cand": "現在候補の値、anchor差、局所shape",
        "conf": "source-native confidenceと有効性。未提供はNaNとvalidityで表現",
        "bank": "exp263 deployable candidate bank内の位置、spread、disagreement",
        "formula": "固定formulaの親値・親confidence・weight・lineage",
        "id": "candidate/family/kindのone-hot。ordinal indexは不使用",
    }
    return descriptions.get(feature_group(feature), "selector feature") + f" ({feature})"


def feature_provenance(feature: str) -> str:
    group = feature_group(feature)
    return {
        "ctx": "competition raw horizontal/typewell + exp263 identity",
        "cand": "exp263 candidate values; formulaは固定weightsから決定再構成",
        "conf": "exp263 target-free candidate_confidence",
        "bank": "exp263 deployable 12-surfaceからtarget-free再計算",
        "formula": "candidate_contract.yaml fixed DAG",
        "id": "candidate_contract.yaml categorical identity",
    }.get(group, "exp264")


def audit_feature_frame(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features = list(frame.columns)
    all_missing = {column for column in features if frame[column].isna().all()}
    constant = {
        column
        for column in features
        if column not in all_missing and frame[column].nunique(dropna=False) <= 1
    }
    candidates = [column for column in features if column not in all_missing | constant]
    duplicate_of = _exact_duplicate_groups(frame, candidates)
    selected = [column for column in candidates if column not in duplicate_of]
    audit_cfg = dict(config.get("audit", {}))
    corr_limit = int(audit_cfg.get("correlation_long_rows", 50000))
    corr_indices = deterministic_sample_indices(frame, corr_limit, "exp264", "correlation")
    corr_frame = frame.iloc[corr_indices][selected]
    pearson = corr_frame.corr(method="pearson", min_periods=100)
    spearman = corr_frame.corr(method="spearman", min_periods=100)
    pearson_threshold = float(audit_cfg.get("pearson_abs_threshold", 0.999))
    spearman_threshold = float(audit_cfg.get("spearman_abs_threshold", 0.999))
    correlation_rows: list[dict[str, Any]] = []
    for right_pos in range(1, len(selected)):
        right = selected[right_pos]
        for left_pos in range(right_pos):
            left = selected[left_pos]
            p = float(pearson.iat[left_pos, right_pos])
            s = float(spearman.iat[left_pos, right_pos])
            if (math.isfinite(p) and abs(p) >= pearson_threshold) or (
                math.isfinite(s) and abs(s) >= spearman_threshold
            ):
                correlation_rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "relation": "near_duplicate_report_only",
                        "pearson": p,
                        "spearman": s,
                        "drop_action": "none",
                    }
                )
    for duplicate, source in duplicate_of.items():
        correlation_rows.append(
            {
                "feature_left": source,
                "feature_right": duplicate,
                "relation": "exact_duplicate",
                "pearson": 1.0,
                "spearman": 1.0,
                "drop_action": f"drop:{duplicate}",
            }
        )
    correlation = pd.DataFrame(
        correlation_rows,
        columns=[
            "feature_left",
            "feature_right",
            "relation",
            "pearson",
            "spearman",
            "drop_action",
        ],
    )
    pearson_group: dict[str, str] = {}
    spearman_group: dict[str, str] = {}
    for group_index, row in correlation.reset_index(drop=True).iterrows():
        if row["relation"] == "near_duplicate_report_only":
            if (
                math.isfinite(float(row["pearson"]))
                and abs(float(row["pearson"])) >= pearson_threshold
            ):
                pearson_group[str(row["feature_left"])] = f"pearson_{group_index}"
                pearson_group[str(row["feature_right"])] = f"pearson_{group_index}"
            if (
                math.isfinite(float(row["spearman"]))
                and abs(float(row["spearman"])) >= spearman_threshold
            ):
                spearman_group[str(row["feature_left"])] = f"spearman_{group_index}"
                spearman_group[str(row["feature_right"])] = f"spearman_{group_index}"
    catalog_rows = []
    for feature in features:
        catalog_rows.append(
            {
                "feature": feature,
                "group": feature_group(feature),
                "description": feature_description(feature),
                "provenance": feature_provenance(feature),
                "raw_test_status": (
                    "generated_from_raw_or_exp263_stage1; native confidence requires parity"
                ),
                "dtype": str(frame[feature].dtype),
                "missing_rate": float(frame[feature].isna().mean()),
                "all_missing": feature in all_missing,
                "constant": feature in constant,
                "exact_duplicate_of": duplicate_of.get(feature, ""),
                "pearson_near_duplicate_group": pearson_group.get(feature, ""),
                "spearman_near_duplicate_group": spearman_group.get(feature, ""),
                "selected": feature in selected,
                "objective": "",
                "fold": pd.NA,
                "gain_importance": np.nan,
                "split_importance": np.nan,
                "importance_mean": np.nan,
                "importance_std": np.nan,
                "importance_rank": np.nan,
            }
        )
    return pd.DataFrame(catalog_rows), selected, correlation


def classify_exp251_parent_schema(schema_path: Path) -> pd.DataFrame:
    schema = pd.read_csv(schema_path)
    if "feature" not in schema:
        raise ValueError("exp251 selected schema lacks feature column")
    old_tokens = (
        "sc_ens",
        "hyb",
        "tvt_dense",
        "blend_likpf_hmm_w500",
        "candidate_index",
        "candidate_name_code",
    )
    rows = []
    for feature in schema["feature"].astype(str):
        if feature in {"last_known_tvt", "eval_len", "md_since"}:
            action = "retain_as_ctx_renamed"
            replacement = f"ctx__{feature}"
        elif feature.startswith("copcf_"):
            action = "defer_until_rawtest_regenerator_is_attached"
            replacement = ""
        elif feature.startswith("multiobs_") or feature.startswith("view_"):
            action = "recompute_from_exp263_bank_if_equivalent"
            replacement = "bank__*"
        elif any(token in feature for token in old_tokens):
            action = "remove_old_candidate_specific"
            replacement = ""
        elif feature.startswith(
            ("candidate_", "pf_ancc_", "beam_mean_", "likpf_mean_", "hmm_", "self_gr_", "exp226_")
        ):
            action = "replace_with_exp263_cand_conf_bank"
            replacement = "cand__/conf__/bank__/formula__/id__"
        elif "_vs_" in feature or "_minus_last" in feature:
            action = "recompute_from_exp263_bank"
            replacement = "cand__/bank__"
        else:
            action = "retain_only_if_raw_context_builder_emits_equivalent"
            replacement = "ctx__*"
        rows.append(
            {"exp251_feature": feature, "action": action, "exp264_replacement": replacement}
        )
    return pd.DataFrame(rows)


def compact_feature_names(contract: Mapping[str, Any]) -> list[str]:
    ids = candidate_ids(contract)
    primary = [str(item) for item in contract["legal_domains"]["primitive_pair_bank"]["candidates"]]
    names = []
    for name in ids:
        names.extend([f"selector__pred_abs_error__{name}", f"selector__p_within10__{name}"])
    for domain in contract["legal_domains"]:
        for objective in ("pred_abs_error", "p_within10"):
            names.extend(
                [
                    f"selector__{domain}__{objective}__top1_value",
                    f"selector__{domain}__{objective}__top2_value",
                    f"selector__{domain}__{objective}__top1_score",
                    f"selector__{domain}__{objective}__top2_score",
                    f"selector__{domain}__{objective}__margin",
                    f"selector__{domain}__{objective}__top1_minus_anchor",
                ]
            )
        names.append(f"selector__{domain}__top1_objective_agreement")
    names.extend(
        [
            "selector__pred_abs_error_mean",
            "selector__pred_abs_error_std",
            "selector__p_within10_mean",
            "selector__p_within10_std",
            "selector__p_within10_candidate_entropy",
            "selector__candidate_value_range",
            "selector__candidate_value_std",
            "selector__available_count",
            "selector__confidence_valid_count",
            "selector__primary_top1_is_primitive",
            "selector__primary_top1_is_pair",
            "selector__fixed_top1_is_primitive",
            "selector__fixed_top1_is_fixed",
        ]
    )
    names.extend(f"selector__primary_error_top1__{name}" for name in primary)
    return names


def _top_two(
    score: np.ndarray, positions: list[int], maximize: bool
) -> tuple[np.ndarray, np.ndarray]:
    domain = score[:, positions]
    order = np.argsort(-domain if maximize else domain, axis=1, kind="stable")
    return order[:, 0], order[:, 1]


def build_compact_meta(
    base: pd.DataFrame,
    values: np.ndarray,
    pred_abs_error: np.ndarray,
    p_within10: np.ndarray,
    available: np.ndarray,
    confidence_valid: np.ndarray,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    ids = candidate_ids(contract)
    id_to_pos = {name: position for position, name in enumerate(ids)}
    specs = contract_by_id(contract)
    output = base[KEY_COLUMNS + ["last_known_tvt"]].copy()
    for position, name in enumerate(ids):
        output[f"selector__pred_abs_error__{name}"] = pred_abs_error[:, position]
        output[f"selector__p_within10__{name}"] = p_within10[:, position]
    anchor = pd.to_numeric(base["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    domain_top1: dict[tuple[str, str], np.ndarray] = {}
    for domain_name, domain_spec in contract["legal_domains"].items():
        domain_ids = [str(item) for item in domain_spec["candidates"]]
        positions = [id_to_pos[name] for name in domain_ids]
        for objective, score, maximize in (
            ("pred_abs_error", pred_abs_error, False),
            ("p_within10", p_within10, True),
        ):
            top1_local, top2_local = _top_two(score, positions, maximize)
            top1 = np.asarray(positions, dtype=np.int64)[top1_local]
            top2 = np.asarray(positions, dtype=np.int64)[top2_local]
            rows = np.arange(len(base))
            top1_score = score[rows, top1]
            top2_score = score[rows, top2]
            prefix = f"selector__{domain_name}__{objective}"
            output[f"{prefix}__top1_value"] = values[rows, top1]
            output[f"{prefix}__top2_value"] = values[rows, top2]
            output[f"{prefix}__top1_score"] = top1_score
            output[f"{prefix}__top2_score"] = top2_score
            output[f"{prefix}__margin"] = (
                top1_score - top2_score if maximize else top2_score - top1_score
            )
            output[f"{prefix}__top1_minus_anchor"] = values[rows, top1] - anchor
            domain_top1[(domain_name, objective)] = top1
        output[f"selector__{domain_name}__top1_objective_agreement"] = (
            domain_top1[(domain_name, "pred_abs_error")] == domain_top1[(domain_name, "p_within10")]
        ).astype(np.int8)
    output["selector__pred_abs_error_mean"] = np.mean(pred_abs_error, axis=1)
    output["selector__pred_abs_error_std"] = np.std(pred_abs_error, axis=1)
    output["selector__p_within10_mean"] = np.mean(p_within10, axis=1)
    output["selector__p_within10_std"] = np.std(p_within10, axis=1)
    normalized = p_within10 / np.maximum(p_within10.sum(axis=1, keepdims=True), 1e-12)
    output["selector__p_within10_candidate_entropy"] = -np.sum(
        normalized * np.log(np.clip(normalized, 1e-12, 1.0)), axis=1
    )
    output["selector__candidate_value_range"] = np.ptp(values, axis=1)
    output["selector__candidate_value_std"] = np.std(values, axis=1)
    output["selector__available_count"] = available.sum(axis=1).astype(np.int16)
    output["selector__confidence_valid_count"] = confidence_valid.sum(axis=1).astype(np.int16)
    primary_top1 = domain_top1[("primitive_pair_bank", "pred_abs_error")]
    primary_ids = [
        str(item) for item in contract["legal_domains"]["primitive_pair_bank"]["candidates"]
    ]
    top1_names = np.asarray(ids, dtype=object)[primary_top1]
    output["selector__primary_top1_is_primitive"] = np.asarray(
        [specs[str(name)]["kind"] == "primitive" for name in top1_names], dtype=np.int8
    )
    output["selector__primary_top1_is_pair"] = np.asarray(
        [str(specs[str(name)]["kind"]).startswith("pair") for name in top1_names], dtype=np.int8
    )
    fixed_top1 = domain_top1[("primitive_fixed_bank", "pred_abs_error")]
    fixed_top1_names = np.asarray(ids, dtype=object)[fixed_top1]
    output["selector__fixed_top1_is_primitive"] = np.asarray(
        [specs[str(name)]["kind"] == "primitive" for name in fixed_top1_names],
        dtype=np.int8,
    )
    output["selector__fixed_top1_is_fixed"] = np.asarray(
        [specs[str(name)]["kind"] == "named_fixed" for name in fixed_top1_names],
        dtype=np.int8,
    )
    for name in primary_ids:
        output[f"selector__primary_error_top1__{name}"] = (top1_names == name).astype(np.int8)
    expected = set(compact_feature_names(contract))
    actual = {column for column in output if column.startswith("selector__")}
    if actual != expected:
        raise ValueError(
            "compact schema mismatch "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return output


def confidence_coverage_rows(bundle: FoldBundle, fold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = bundle.specs
    for name in bundle.candidate_ids:
        if name in bundle.confidence:
            frame = bundle.confidence[name]
            valid = frame["confidence_valid"].astype(bool).to_numpy()
            rows.append(
                {
                    "candidate_id": name,
                    "outer_fold": fold,
                    "kind": specs[name]["kind"],
                    "field": "confidence_valid",
                    "rows": len(frame),
                    "coverage": float(np.mean(valid)),
                }
            )
            for field in _confidence_numeric_fields({name: frame}):
                values = pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float64)
                rows.append(
                    {
                        "candidate_id": name,
                        "outer_fold": fold,
                        "kind": specs[name]["kind"],
                        "field": field,
                        "rows": len(frame),
                        "coverage": float(np.mean(np.isfinite(values))),
                    }
                )
        else:
            parents = [str(item) for item in specs[name].get("parents", [])]
            parent_valid = [
                bundle.confidence[parent]["confidence_valid"].astype(bool).to_numpy()
                for parent in parents
            ]
            rows.append(
                {
                    "candidate_id": name,
                    "outer_fold": fold,
                    "kind": specs[name]["kind"],
                    "field": "any_parent_confidence_valid",
                    "rows": len(bundle.base),
                    "coverage": float(np.mean(np.logical_or.reduce(parent_valid))),
                }
            )
    return rows


def _feature_schema_payload(features: Sequence[str], contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0.0",
        "features": list(features),
        "feature_count": len(features),
        "candidate_order": candidate_ids(contract),
        "candidate_id_encoding": "one_hot",
        "ordinal_candidate_index": False,
    }
    payload["feature_schema_sha256"] = sha256_json(payload)
    return payload


def run_stage_a(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    raw_train_dir: Path,
    output_dir: Path,
    parent_schema_path: Path,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_meta = verify_exp263_root(cache_root, config)
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    audit_limit = int(config["features"]["audit"]["base_rows_per_fold"])
    audit_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    sample_sha: dict[str, str] = {}
    feature_cfg = dict(config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"]["candidates"]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"]["candidates"]
    for fold in range(int(config["validation"]["outer_folds"])):
        bundle = cache.load_fold(fold)
        context, _ = build_raw_context(bundle.base, raw_train_dir, feature_cfg, require_truth=False)
        indices = deterministic_sample_indices(bundle.base, audit_limit, "exp264", "stage_a", fold)
        long, _ = build_candidate_long_features(bundle, context, indices, feature_cfg)
        audit_parts.append(long)
        coverage_rows.extend(confidence_coverage_rows(bundle, fold))
        sample_sha[str(fold)] = logical_frame_sha256(
            bundle.base.iloc[indices][KEY_COLUMNS].reset_index(drop=True)
        )
    audit_frame = pd.concat(audit_parts, ignore_index=True)
    catalog, selected, correlation = audit_feature_frame(audit_frame, feature_cfg)
    schema = _feature_schema_payload(selected, contract)
    parent_mapping = classify_exp251_parent_schema(parent_schema_path)
    if len(parent_mapping) != int(config["data"]["exp251_expected_selected_feature_count"]):
        raise ValueError("exp251 v4 selected schema count mismatch")
    paths = {
        "feature_catalog": output_dir / "feature_catalog.csv",
        "feature_schema": output_dir / "feature_schema.json",
        "feature_duplicate_correlation_audit": output_dir
        / "feature_duplicate_correlation_audit.csv",
        "confidence_coverage": output_dir / "confidence_coverage_by_candidate_fold.csv",
        "parent_feature_mapping": output_dir / "exp251_v4_feature_mapping.csv",
        "compact_meta_schema": output_dir / "compact_meta_schema.json",
        "reproducibility_manifest": output_dir / "reproducibility_manifest.json",
        "stage_a_summary": output_dir / "stage_a_summary.json",
    }
    catalog.to_csv(paths["feature_catalog"], index=False)
    correlation.to_csv(paths["feature_duplicate_correlation_audit"], index=False)
    pd.DataFrame(coverage_rows).to_csv(paths["confidence_coverage"], index=False)
    parent_mapping.to_csv(paths["parent_feature_mapping"], index=False)
    write_json(paths["feature_schema"], schema)
    compact_schema = {
        "schema_version": "1.0.0",
        "features": compact_feature_names(contract),
    }
    compact_schema["compact_meta_schema_sha256"] = sha256_json(compact_schema)
    write_json(paths["compact_meta_schema"], compact_schema)
    reproducibility = {
        "status": "stage_a_feature_contract_frozen",
        "exp263": root_meta,
        "candidate_contract_sha256": candidate_contract_sha(contract),
        "parent_schema_sha256": sha256_file(parent_schema_path),
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "audit_sample_identity_sha256_by_fold": sample_sha,
        "audit_long_content_sha256": logical_frame_sha256(audit_frame[selected]),
    }
    write_json(paths["reproducibility_manifest"], reproducibility)
    summary = {
        "status": "stage_a_completed",
        "rows_audited": len(audit_frame),
        "feature_count_before_audit": audit_frame.shape[1],
        "selected_feature_count": len(selected),
        "dropped_all_missing": int(catalog["all_missing"].sum()),
        "dropped_constant": int(catalog["constant"].sum()),
        "dropped_exact_duplicate": int(catalog["exact_duplicate_of"].astype(bool).sum()),
        "near_duplicate_pairs": int(
            (correlation["relation"] == "near_duplicate_report_only").sum()
        ),
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "compact_meta_feature_count": len(compact_schema["features"]),
        "artifacts": {key: path.name for key, path in paths.items()},
    }
    write_json(paths["stage_a_summary"], summary)
    return summary


def load_feature_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(Path(path).read_text())
    expected = schema.get("feature_schema_sha256")
    payload = {key: value for key, value in schema.items() if key != "feature_schema_sha256"}
    if expected != sha256_json(payload):
        raise ValueError("feature schema SHA self-check failed")
    return schema


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    target = np.asarray(y, dtype=np.float64)
    return float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(np.square(np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)))
        )
    )


class IncrementalParquetWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.writer: Any = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise ValueError(f"no parquet rows written: {self.path}")
        self.writer.close()
        self.writer = None


def build_nested_inner_fold_maps(
    fold_well_counts: Mapping[int, pd.DataFrame],
    n_outer_folds: int,
    n_inner_folds: int,
) -> tuple[dict[int, dict[str, int]], pd.DataFrame]:
    """Build deterministic, well-disjoint inner folds inside each outer-train split.

    GroupKFold balances row counts by assigning large groups to the currently
    lightest fold.  This compact implementation applies the same principle to
    pre-aggregated well counts and uses the well string as a stable tie-breaker.
    """

    expected_outer = set(range(int(n_outer_folds)))
    if set(int(key) for key in fold_well_counts) != expected_outer:
        raise ValueError("nested fold counts must cover every outer fold")
    maps: dict[int, dict[str, int]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for downstream_outer_fold in range(int(n_outer_folds)):
        outer_valid_counts = fold_well_counts[downstream_outer_fold].copy()
        train_parts = [
            fold_well_counts[source_fold].copy()
            for source_fold in range(int(n_outer_folds))
            if source_fold != downstream_outer_fold
        ]
        train_counts = pd.concat(train_parts, ignore_index=True)
        if train_counts["well"].astype(str).duplicated().any():
            raise ValueError("well appears in more than one source outer fold")
        train_counts["well"] = train_counts["well"].astype(str)
        train_counts["rows"] = pd.to_numeric(train_counts["rows"], errors="raise").astype(
            np.int64
        )
        train_counts = train_counts.sort_values(
            ["rows", "well"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        fold_rows = np.zeros(int(n_inner_folds), dtype=np.int64)
        fold_wells = np.zeros(int(n_inner_folds), dtype=np.int64)
        assignment: dict[str, int] = {}
        for row in train_counts.itertuples(index=False):
            inner_fold = int(np.argmin(fold_rows))
            well = str(row.well)
            assignment[well] = inner_fold
            fold_rows[inner_fold] += int(row.rows)
            fold_wells[inner_fold] += 1
        if len(assignment) != len(train_counts):
            raise AssertionError("nested inner assignment lost wells")
        outer_valid_wells = set(outer_valid_counts["well"].astype(str))
        if outer_valid_wells.intersection(assignment):
            raise AssertionError("outer-valid well leaked into nested inner assignment")
        if np.any(fold_wells == 0):
            raise AssertionError("nested inner fold has no wells")
        maps[downstream_outer_fold] = assignment
        for inner_fold in range(int(n_inner_folds)):
            valid_wells = {
                well for well, assigned in assignment.items() if assigned == inner_fold
            }
            train_wells = set(assignment).difference(valid_wells)
            if train_wells.intersection(valid_wells):
                raise AssertionError("nested train/valid well overlap")
            manifest_rows.append(
                {
                    "downstream_outer_fold": downstream_outer_fold,
                    "inner_fold": inner_fold,
                    "outer_train_wells": len(assignment),
                    "outer_valid_wells": len(outer_valid_wells),
                    "inner_train_wells": len(train_wells),
                    "inner_valid_wells": len(valid_wells),
                    "inner_valid_rows": int(fold_rows[inner_fold]),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    expected_rows = int(n_outer_folds) * int(n_inner_folds)
    if len(manifest) != expected_rows:
        raise AssertionError("nested fold manifest row count mismatch")
    return maps, manifest


def run_stage_b(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    raw_train_dir: Path,
    output_dir: Path,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
    candidate_task_weight_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    if candidate_task_weight_config is not None:
        from src.candidate_task_weighting import validate_inverse_rmse_weight_config

        validate_inverse_rmse_weight_config(candidate_task_weight_config)

    schema = load_feature_schema(output_dir / "feature_schema.json")
    features = [str(item) for item in schema["features"]]
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    n_folds = int(config["validation"]["outer_folds"])
    n_candidates = len(cache.ids)
    feature_cfg = dict(config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"]["candidates"]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"]["candidates"]
    train_cfg = dict(config["model"]["training"])
    per_fold_train_limit = max(
        1, int(math.ceil(int(train_cfg["max_train_base_rows_per_outer_fold"]) / (n_folds - 1)))
    )
    valid_limit = int(train_cfg["max_valid_base_rows_for_early_stopping"])
    sampled: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    fold_label_summary: dict[int, dict[str, np.ndarray]] = {}
    for fold in range(n_folds):
        bundle = cache.load_fold(fold)
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        indices = deterministic_sample_indices(
            bundle.base, max(per_fold_train_limit, valid_limit), "exp264", "stage_b_sample", fold
        )
        long, metadata = build_candidate_long_features(
            bundle, context, indices, feature_cfg, expected_features=features
        )
        labels = add_candidate_labels(metadata, truth[indices], n_candidates)
        sampled[fold] = (long, labels)
        full_error = np.abs(bundle.values - truth[:, None])
        fold_label_summary[fold] = {
            "error_sum": full_error.sum(axis=0, dtype=np.float64),
            "within_sum": (full_error <= 10.0).sum(axis=0, dtype=np.float64),
            "count": np.full(n_candidates, len(bundle.base), dtype=np.float64),
        }

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    common = dict(config["model"]["lightgbm_common"])
    seed = int(config["validation"]["seed"])
    num_round = int(train_cfg["num_boost_round"])

    def model_callbacks() -> list[Any]:
        return [
            early_stopping(int(train_cfg["early_stopping_rounds"]), verbose=False),
            log_evaluation(int(train_cfg["log_evaluation_period"])),
        ]

    importance_rows: list[dict[str, Any]] = []
    manifest_models: list[dict[str, Any]] = []
    score_path = output_dir / "candidate_score_oof.parquet"
    compact_path = output_dir / "compact_meta_oof.parquet"
    score_writer = IncrementalParquetWriter(score_path)
    compact_writer = IncrementalParquetWriter(compact_path)
    metric_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    calibration_parts: list[pd.DataFrame] = []
    distance_metric_rows: list[dict[str, Any]] = []
    selection_counts = defaultdict(int)
    by_well_error: dict[str, list[np.ndarray]] = defaultdict(list)
    candidate_task_weight_results: list[Any] = []
    for outer_fold in range(n_folds):
        train_feature_parts = []
        train_label_parts = []
        for source_fold in range(n_folds):
            if source_fold == outer_fold:
                continue
            features_part, labels_part = sampled[source_fold]
            sampled_base_count = len(features_part) // n_candidates
            sampled_base = pd.DataFrame(index=np.arange(sampled_base_count))
            selected_base = deterministic_sample_indices(
                sampled_base,
                min(per_fold_train_limit, sampled_base_count),
                "exp264",
                "outer_train",
                outer_fold,
                source_fold,
            )
            selected_long = (
                selected_base[:, None] * n_candidates + np.arange(n_candidates)[None, :]
            ).reshape(-1)
            train_feature_parts.append(features_part.iloc[selected_long])
            train_label_parts.append(labels_part.iloc[selected_long])
        x_train = pd.concat(train_feature_parts, ignore_index=True).astype(np.float32)
        y_train = pd.concat(train_label_parts, ignore_index=True)
        x_valid_all, y_valid_all = sampled[outer_fold]
        valid_long_limit = min(len(x_valid_all), valid_limit * n_candidates)
        x_valid = x_valid_all.iloc[:valid_long_limit].astype(np.float32)
        y_valid = y_valid_all.iloc[:valid_long_limit]
        train_sample_weight: np.ndarray | None = None
        weight_result: Any = None
        if candidate_task_weight_config is not None:
            from src.candidate_task_weighting import (
                build_inverse_rmse_candidate_task_weights,
            )

            weight_result = build_inverse_rmse_candidate_task_weights(
                y_train,
                cache.ids,
                partition_id=outer_fold,
                config=candidate_task_weight_config,
            )
            train_wells = set(y_train["well"].astype(str))
            valid_wells = set(y_valid["well"].astype(str))
            fit_valid_well_overlap = len(train_wells.intersection(valid_wells))
            if fit_valid_well_overlap:
                raise ValueError(
                    "candidate task weight fit partition overlaps outer validation wells"
                )
            weight_result.audit["fit_valid_well_overlap"] = fit_valid_well_overlap
            weight_result.audit["fit_wells"] = len(train_wells)
            weight_result.audit["valid_wells"] = len(valid_wells)
            weight_result.audit["fit_feature_schema_sha256"] = schema[
                "feature_schema_sha256"
            ]
            weight_result.audit["fit_feature_content_sha256"] = logical_frame_sha256(
                x_train
            )
            train_sample_weight = weight_result.sample_weight
            candidate_task_weight_results.append(weight_result)
        classifier = LGBMClassifier(
            objective="binary",
            n_estimators=num_round,
            random_state=seed + outer_fold,
            **common,
        )
        regressor = LGBMRegressor(
            objective="regression_l1",
            n_estimators=num_round,
            random_state=seed + 100 + outer_fold,
            **common,
        )
        classifier.fit(
            x_train,
            y_train["candidate_within10"],
            eval_set=[(x_valid, y_valid["candidate_within10"])],
            eval_metric="binary_logloss",
            callbacks=model_callbacks(),
            **(
                {"sample_weight": train_sample_weight}
                if train_sample_weight is not None
                else {}
            ),
        )
        regressor.fit(
            x_train,
            y_train["candidate_abs_error"],
            eval_set=[(x_valid, y_valid["candidate_abs_error"])],
            eval_metric="l1",
            callbacks=model_callbacks(),
            **(
                {"sample_weight": train_sample_weight}
                if train_sample_weight is not None
                else {}
            ),
        )
        models = {
            "p_within10": classifier,
            "pred_abs_error": regressor,
        }
        model_sha_by_objective: dict[str, str] = {}
        for objective, model in models.items():
            model_path = model_dir / f"selector_{objective}_fold{outer_fold}.txt"
            model.booster_.save_model(str(model_path))
            model_sha_by_objective[objective] = sha256_file(model_path)
            model_record = {
                "outer_fold": outer_fold,
                "objective": objective,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": model_sha_by_objective[objective],
                "best_iteration": int(model.best_iteration_),
                "train_long_rows": len(x_train),
                "early_stop_long_rows": len(x_valid),
            }
            if weight_result is not None:
                model_record.update(
                    {
                        "training_sample_weight_applied": True,
                        "training_sample_weight_sha256": weight_result.audit[
                            "sample_weight_content_sha256"
                        ],
                        "validation_sample_weight_applied": False,
                    }
                )
            manifest_models.append(model_record)
            for importance_type in ("gain", "split"):
                values_importance = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                for feature, importance in zip(features, values_importance, strict=True):
                    importance_rows.append(
                        {
                            "feature": feature,
                            "objective": objective,
                            "fold": outer_fold,
                            "importance_type": importance_type,
                            "importance": float(importance),
                        }
                    )

        bundle = cache.load_fold(outer_fold)
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
        prior_error_sum = sum(
            fold_label_summary[fold]["error_sum"] for fold in range(n_folds) if fold != outer_fold
        )
        prior_within_sum = sum(
            fold_label_summary[fold]["within_sum"] for fold in range(n_folds) if fold != outer_fold
        )
        prior_count = sum(
            fold_label_summary[fold]["count"] for fold in range(n_folds) if fold != outer_fold
        )
        error_prior = prior_error_sum / prior_count
        within_prior = prior_within_sum / prior_count
        fold_actual_error: list[np.ndarray] = []
        fold_pred_error: list[np.ndarray] = []
        fold_actual_within: list[np.ndarray] = []
        fold_pred_within: list[np.ndarray] = []
        fold_prior_error: list[np.ndarray] = []
        fold_prior_within: list[np.ndarray] = []
        hard_truth: list[np.ndarray] = []
        hard_prediction: list[np.ndarray] = []
        fallback_prediction: list[np.ndarray] = []
        hard_md_since: list[np.ndarray] = []
        fold_actual_error_matrix: list[np.ndarray] = []
        fold_pred_error_matrix: list[np.ndarray] = []
        fold_actual_within_matrix: list[np.ndarray] = []
        fold_pred_within_matrix: list[np.ndarray] = []
        rank_regret_error: list[np.ndarray] = []
        rank_regret_probability: list[np.ndarray] = []
        top3_error_coverage: list[np.ndarray] = []
        top3_probability_coverage: list[np.ndarray] = []
        chunk_size = int(train_cfg["predict_base_row_chunk_size"])
        for start in range(0, len(bundle.base), chunk_size):
            stop = min(start + chunk_size, len(bundle.base))
            indices = np.arange(start, stop, dtype=np.int64)
            long, metadata = build_candidate_long_features(
                bundle,
                context,
                indices,
                feature_cfg,
                shape_state=shape_state,
                expected_features=features,
            )
            x = long.astype(np.float32)
            p = classifier.predict_proba(x, num_iteration=classifier.best_iteration_)[:, 1]
            e = regressor.predict(x, num_iteration=regressor.best_iteration_)
            p_matrix = p.reshape(len(indices), n_candidates).astype(np.float32)
            e_matrix = np.maximum(e.reshape(len(indices), n_candidates), 0.0).astype(np.float32)
            labels = add_candidate_labels(metadata, truth[indices], n_candidates)
            actual_error = (
                labels["candidate_abs_error"]
                .to_numpy(np.float32)
                .reshape(len(indices), n_candidates)
            )
            actual_within = (
                labels["candidate_within10"].to_numpy(np.int8).reshape(len(indices), n_candidates)
            )
            score = metadata.copy()
            score["actual_abs_error"] = actual_error.reshape(-1)
            score["actual_within10"] = actual_within.reshape(-1)
            score["pred_abs_error"] = e_matrix.reshape(-1)
            score["p_within10"] = p_matrix.reshape(-1)
            score["feature_schema_sha"] = schema["feature_schema_sha256"]
            score["candidate_contract_sha"] = candidate_contract_sha(contract)
            score["model_fold"] = outer_fold
            score["pred_abs_error_model_sha"] = model_sha_by_objective["pred_abs_error"]
            score["p_within10_model_sha"] = model_sha_by_objective["p_within10"]
            score_writer.write(score)
            confidence_valid = (
                metadata["confidence_valid"].to_numpy(bool).reshape(len(indices), n_candidates)
            )
            compact = build_compact_meta(
                bundle.base.iloc[indices].reset_index(drop=True),
                bundle.values[indices],
                e_matrix,
                p_matrix,
                bundle.available[indices],
                confidence_valid,
                contract,
            )
            compact_writer.write(compact)
            primary_ids = contract["legal_domains"]["primitive_pair_bank"]["candidates"]
            primary_pos = [cache.ids.index(str(name)) for name in primary_ids]
            selected_local = np.argmin(e_matrix[:, primary_pos], axis=1)
            selected_pos = np.asarray(primary_pos)[selected_local]
            selected_names = np.asarray(cache.ids, dtype=object)[selected_pos]
            for name, count in zip(*np.unique(selected_names, return_counts=True), strict=True):
                selection_counts[(outer_fold, "pred_abs_error", str(name))] += int(count)
            probability_local = np.argmax(p_matrix[:, primary_pos], axis=1)
            probability_pos = np.asarray(primary_pos)[probability_local]
            probability_names = np.asarray(cache.ids, dtype=object)[probability_pos]
            for name, count in zip(*np.unique(probability_names, return_counts=True), strict=True):
                selection_counts[(outer_fold, "p_within10", str(name))] += int(count)
            rows = np.arange(len(indices))
            hard_prediction.append(bundle.values[indices][rows, selected_pos])
            hard_truth.append(truth[indices])
            fallback_pos = cache.ids.index("exp226_w500_50_50")
            fallback_prediction.append(bundle.values[indices, fallback_pos])
            md_chunk = bundle.base.iloc[indices]["md_since"].to_numpy(np.float32)
            hard_md_since.append(md_chunk)
            fold_actual_error.append(actual_error.reshape(-1))
            fold_pred_error.append(e_matrix.reshape(-1))
            fold_actual_within.append(actual_within.reshape(-1))
            fold_pred_within.append(p_matrix.reshape(-1))
            fold_prior_error.append(np.tile(error_prior, len(indices)))
            fold_prior_within.append(np.tile(within_prior, len(indices)))
            fold_actual_error_matrix.append(actual_error)
            fold_pred_error_matrix.append(e_matrix)
            fold_actual_within_matrix.append(actual_within)
            fold_pred_within_matrix.append(p_matrix)
            primary_actual = actual_error[:, primary_pos]
            oracle_local = np.argmin(primary_actual, axis=1)
            oracle_error = primary_actual[rows, oracle_local]
            rank_regret_error.append(primary_actual[rows, selected_local] - oracle_error)
            rank_regret_probability.append(primary_actual[rows, probability_local] - oracle_error)
            error_order = np.argsort(e_matrix[:, primary_pos], axis=1, kind="stable")
            probability_order = np.argsort(-p_matrix[:, primary_pos], axis=1, kind="stable")
            top3_error_coverage.append(np.any(error_order[:, :3] == oracle_local[:, None], axis=1))
            top3_probability_coverage.append(
                np.any(probability_order[:, :3] == oracle_local[:, None], axis=1)
            )

            distance_bucket = pd.cut(
                md_chunk,
                bins=[-np.inf, 250.0, 500.0, 1000.0, np.inf],
                labels=["near_0_250", "250_500", "500_1000", "1000_plus"],
            ).astype(str)
            calibration_frame = pd.DataFrame(
                {
                    "outer_fold": outer_fold,
                    "candidate_id": np.tile(np.asarray(cache.ids, dtype=object), len(indices)),
                    "distance_bucket": np.repeat(distance_bucket, n_candidates),
                    "confidence_valid": confidence_valid.reshape(-1),
                    "probability_bin": np.minimum((p_matrix.reshape(-1) * 10).astype(np.int8), 9),
                    "observed_within10_sum": actual_within.reshape(-1).astype(np.float64),
                    "predicted_within10_sum": p_matrix.reshape(-1).astype(np.float64),
                    "actual_abs_error_sum": actual_error.reshape(-1).astype(np.float64),
                    "pred_abs_error_sum": e_matrix.reshape(-1).astype(np.float64),
                }
            )
            calibration_frame["rows"] = 1
            calibration_parts.append(
                calibration_frame.groupby(
                    [
                        "outer_fold",
                        "candidate_id",
                        "distance_bucket",
                        "confidence_valid",
                        "probability_bin",
                    ],
                    observed=True,
                    as_index=False,
                )[
                    [
                        "rows",
                        "observed_within10_sum",
                        "predicted_within10_sum",
                        "actual_abs_error_sum",
                        "pred_abs_error_sum",
                    ]
                ].sum()
            )
            for well, local_positions in (
                bundle.base.iloc[indices].groupby("well", sort=False).indices.items()
            ):
                local = np.asarray(local_positions, dtype=np.int64)
                by_well_error[str(well)].append(
                    np.column_stack(
                        [
                            truth[indices][local],
                            hard_prediction[-1][local],
                            fallback_prediction[-1][local],
                            md_chunk[local],
                        ]
                    )
                )
        actual_error_vector = np.concatenate(fold_actual_error)
        pred_error_vector = np.concatenate(fold_pred_error)
        actual_within_vector = np.concatenate(fold_actual_within)
        pred_within_vector = np.concatenate(fold_pred_within)
        prior_error_vector = np.concatenate(fold_prior_error)
        prior_within_vector = np.concatenate(fold_prior_within)
        hard_y = np.concatenate(hard_truth)
        hard_p = np.concatenate(hard_prediction)
        fallback_p = np.concatenate(fallback_prediction)
        hard_md = np.concatenate(hard_md_since)
        actual_error_matrix = np.concatenate(fold_actual_error_matrix, axis=0)
        pred_error_matrix = np.concatenate(fold_pred_error_matrix, axis=0)
        actual_within_matrix = np.concatenate(fold_actual_within_matrix, axis=0)
        pred_within_matrix = np.concatenate(fold_pred_within_matrix, axis=0)
        for candidate_position, candidate_id in enumerate(cache.ids):
            candidate_metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_id": candidate_id,
                    "rows": len(actual_error_matrix),
                    "expected_error_mae": float(
                        np.mean(
                            np.abs(
                                pred_error_matrix[:, candidate_position]
                                - actual_error_matrix[:, candidate_position]
                            )
                        )
                    ),
                    "within10_logloss": _binary_logloss(
                        actual_within_matrix[:, candidate_position],
                        pred_within_matrix[:, candidate_position],
                    ),
                    "within10_brier": float(
                        np.mean(
                            np.square(
                                pred_within_matrix[:, candidate_position]
                                - actual_within_matrix[:, candidate_position]
                            )
                        )
                    ),
                }
            )
        for bucket, lower, upper in (
            ("near_0_250", -np.inf, 250.0),
            ("250_500", 250.0, 500.0),
            ("500_1000", 500.0, 1000.0),
            ("1000_plus", 1000.0, np.inf),
        ):
            mask = (hard_md >= lower) & (hard_md < upper)
            distance_metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "distance_bucket": bucket,
                    "rows": int(mask.sum()),
                    "hard_primary_rmse": _rmse(hard_y[mask], hard_p[mask])
                    if mask.any()
                    else np.nan,
                    "fixed_fallback_rmse": _rmse(hard_y[mask], fallback_p[mask])
                    if mask.any()
                    else np.nan,
                }
            )
        metric_rows.append(
            {
                "scope": "outer_fold",
                "fold": outer_fold,
                "rows": len(actual_error_vector),
                "expected_error_mae": float(
                    np.mean(np.abs(pred_error_vector - actual_error_vector))
                ),
                "expected_error_prior_mae": float(
                    np.mean(np.abs(prior_error_vector - actual_error_vector))
                ),
                "within10_logloss": _binary_logloss(actual_within_vector, pred_within_vector),
                "within10_prior_logloss": _binary_logloss(
                    actual_within_vector, prior_within_vector
                ),
                "within10_brier": float(
                    np.mean(np.square(pred_within_vector - actual_within_vector))
                ),
                "within10_prior_brier": float(
                    np.mean(np.square(prior_within_vector - actual_within_vector))
                ),
                "hard_primary_rmse": _rmse(hard_y, hard_p),
                "fixed_fallback_rmse": _rmse(hard_y, fallback_p),
                "rank_regret_pred_abs_error": float(np.mean(np.concatenate(rank_regret_error))),
                "rank_regret_p_within10": float(np.mean(np.concatenate(rank_regret_probability))),
                "top3_oracle_coverage_pred_abs_error": float(
                    np.mean(np.concatenate(top3_error_coverage))
                ),
                "top3_oracle_coverage_p_within10": float(
                    np.mean(np.concatenate(top3_probability_coverage))
                ),
            }
        )

    candidate_score_oof_rows = int(score_writer.rows)
    compact_meta_oof_rows = int(compact_writer.rows)
    score_writer.close()
    compact_writer.close()
    candidate_task_weight_manifest = None
    if candidate_task_weight_results:
        from src.candidate_task_weighting import write_candidate_task_weight_artifacts

        candidate_task_weight_manifest = write_candidate_task_weight_artifacts(
            output_dir,
            candidate_task_weight_results,
        )
        if not bool(candidate_task_weight_manifest["all_checks_passed"]):
            raise RuntimeError("candidate task weight technical audit failed")
    importance = pd.DataFrame(importance_rows)
    pivot = (
        importance.pivot_table(
            index=["feature", "objective", "fold"],
            columns="importance_type",
            values="importance",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"gain": "gain_importance", "split": "split_importance"})
    )
    summary_importance = pivot.groupby(["feature", "objective"], as_index=False)[
        "gain_importance"
    ].agg(importance_mean="mean", importance_std="std")
    summary_importance["importance_rank"] = summary_importance.groupby("objective")[
        "importance_mean"
    ].rank(method="dense", ascending=False)
    pivot = pivot.merge(summary_importance, on=["feature", "objective"], how="left")
    base_catalog = pd.read_csv(output_dir / "feature_catalog.csv")
    base_catalog = base_catalog.drop(
        columns=[
            "objective",
            "fold",
            "gain_importance",
            "split_importance",
            "importance_mean",
            "importance_std",
            "importance_rank",
        ],
        errors="ignore",
    )
    trained_catalog = base_catalog[base_catalog["selected"].astype(bool)].merge(
        pivot, on="feature", how="left", validate="one_to_many"
    )
    dropped_catalog = base_catalog[~base_catalog["selected"].astype(bool)].copy()
    for column in [
        "objective",
        "fold",
        "gain_importance",
        "split_importance",
        "importance_mean",
        "importance_std",
        "importance_rank",
    ]:
        dropped_catalog[column] = np.nan
    pd.concat([trained_catalog, dropped_catalog], ignore_index=True).to_csv(
        output_dir / "feature_catalog.csv", index=False
    )
    pivot.to_csv(output_dir / "feature_importance_by_objective_fold.csv", index=False)
    selection = pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "objective": objective,
                "candidate_id": name,
                "selected_rows": count,
            }
            for (fold, objective, name), count in sorted(selection_counts.items())
        ]
    )
    selection.to_csv(output_dir / "selector_selection_rate.csv", index=False)
    by_well_rows = []
    for well, parts in by_well_error.items():
        values_pair = np.concatenate(parts, axis=0)
        by_well_rows.append(
            {
                "well": well,
                "rows": len(values_pair),
                "hard_primary_rmse": _rmse(values_pair[:, 0], values_pair[:, 1]),
                "fixed_fallback_rmse": _rmse(values_pair[:, 0], values_pair[:, 2]),
                "delta_rmse_hard_minus_fixed": _rmse(values_pair[:, 0], values_pair[:, 1])
                - _rmse(values_pair[:, 0], values_pair[:, 2]),
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    by_well.to_csv(output_dir / "selector_by_well.csv", index=False)
    pd.DataFrame(candidate_metric_rows).to_csv(
        output_dir / "selector_candidate_metrics.csv", index=False
    )
    pd.DataFrame(distance_metric_rows).to_csv(
        output_dir / "selector_distance_bucket_metrics.csv", index=False
    )
    calibration = (
        pd.concat(calibration_parts, ignore_index=True)
        .groupby(
            [
                "outer_fold",
                "candidate_id",
                "distance_bucket",
                "confidence_valid",
                "probability_bin",
            ],
            as_index=False,
        )[
            [
                "rows",
                "observed_within10_sum",
                "predicted_within10_sum",
                "actual_abs_error_sum",
                "pred_abs_error_sum",
            ]
        ]
        .sum()
    )
    for name, numerator in (
        ("observed_within10", "observed_within10_sum"),
        ("predicted_within10", "predicted_within10_sum"),
        ("actual_abs_error", "actual_abs_error_sum"),
        ("pred_abs_error", "pred_abs_error_sum"),
    ):
        calibration[name] = calibration[numerator] / calibration["rows"]
    calibration.to_csv(output_dir / "selector_calibration.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "selector_metrics.csv", index=False)
    weights = metrics["rows"].to_numpy(np.float64)
    pooled = {
        column: float(np.average(metrics[column], weights=weights))
        for column in [
            "expected_error_mae",
            "expected_error_prior_mae",
            "within10_logloss",
            "within10_prior_logloss",
            "within10_brier",
            "within10_prior_brier",
        ]
    }
    score_guard = {
        "expected_error_mae_improved_folds": int(
            (metrics["expected_error_mae"] < metrics["expected_error_prior_mae"]).sum()
        ),
        "within10_logloss_improved_folds": int(
            (metrics["within10_logloss"] < metrics["within10_prior_logloss"]).sum()
        ),
        "within10_brier_improved_folds": int(
            (metrics["within10_brier"] < metrics["within10_prior_brier"]).sum()
        ),
        "expected_error_mae_improved_pooled": bool(
            pooled["expected_error_mae"] < pooled["expected_error_prior_mae"]
        ),
        "within10_logloss_improved_pooled": bool(
            pooled["within10_logloss"] < pooled["within10_prior_logloss"]
        ),
        "within10_brier_improved_pooled": bool(
            pooled["within10_brier"] < pooled["within10_prior_brier"]
        ),
    }
    score_guard["passed"] = (
        score_guard["expected_error_mae_improved_folds"] >= 4
        and score_guard["within10_logloss_improved_folds"] >= 4
        and score_guard["within10_brier_improved_folds"] >= 4
        and score_guard["expected_error_mae_improved_pooled"]
        and score_guard["within10_logloss_improved_pooled"]
        and score_guard["within10_brier_improved_pooled"]
    )
    hard_rmse = float(np.sqrt(np.average(np.square(metrics["hard_primary_rmse"]), weights=weights)))
    fixed_rmse = float(
        np.sqrt(np.average(np.square(metrics["fixed_fallback_rmse"]), weights=weights))
    )
    bucket_metrics = pd.DataFrame(distance_metric_rows)
    hard_guard_cfg = config["guards"]["hard_readout_diagnostic"]
    bucket_delta = {}
    for bucket in ["near_0_250", "1000_plus"]:
        selected = bucket_metrics[bucket_metrics["distance_bucket"].eq(bucket)]
        selected_weights = selected["rows"].to_numpy(np.float64)
        hard_bucket = float(
            np.sqrt(np.average(np.square(selected["hard_primary_rmse"]), weights=selected_weights))
        )
        fixed_bucket = float(
            np.sqrt(
                np.average(np.square(selected["fixed_fallback_rmse"]), weights=selected_weights)
            )
        )
        bucket_delta[bucket] = hard_bucket - fixed_bucket
    hard_guard = {
        "hard_primary_oof_rmse": hard_rmse,
        "fixed_fallback_oof_rmse": fixed_rmse,
        "overall_improvement": fixed_rmse - hard_rmse,
        "improved_folds": int(
            (metrics["hard_primary_rmse"] < metrics["fixed_fallback_rmse"]).sum()
        ),
        "near_delta_rmse": bucket_delta["near_0_250"],
        "distance_1000_plus_delta_rmse": bucket_delta["1000_plus"],
        "worst_well_regression": float(by_well["delta_rmse_hard_minus_fixed"].max()),
        "hidden_like_status": "not_computed_without_independent_assignment_join",
    }
    hard_guard["passed_without_hidden_like"] = bool(
        hard_guard["overall_improvement"] >= float(hard_guard_cfg["min_overall_rmse_improvement"])
        and hard_guard["improved_folds"] >= int(hard_guard_cfg["min_improved_folds"])
        and hard_guard["near_delta_rmse"] <= float(hard_guard_cfg["max_near_delta_rmse"])
        and hard_guard["distance_1000_plus_delta_rmse"]
        <= float(hard_guard_cfg["max_1000_plus_delta_rmse"])
        and hard_guard["worst_well_regression"]
        <= float(hard_guard_cfg["max_worst_well_regression"])
    )
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "selector_outer_oof_completed",
        "candidate_order": cache.ids,
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "feature_count": len(features),
        "models": manifest_models,
        "model_count": len(manifest_models),
        "score_guard": score_guard,
        "hard_readout_guard": hard_guard,
    }
    if candidate_task_weight_manifest is not None:
        model_manifest["candidate_task_weight"] = candidate_task_weight_manifest
    write_json(output_dir / "selector_model_manifest.json", model_manifest)
    selector_metrics = {
        "status": "selector_outer_oof_completed",
        "fold_metrics": metric_rows,
        "pooled_score_metrics": pooled,
        "score_guard": score_guard,
        "hard_readout_guard": hard_guard,
        "hard_primary_oof_rmse": hard_rmse,
        "model_count": len(manifest_models),
        "candidate_score_oof_sha256": sha256_file(score_path),
        "compact_meta_oof_sha256": sha256_file(compact_path),
        "model_manifest_sha256": sha256_file(output_dir / "selector_model_manifest.json"),
    }
    if candidate_task_weight_manifest is not None:
        selector_metrics.update(
            {
                "candidate_score_oof_rows": candidate_score_oof_rows,
                "compact_meta_oof_rows": compact_meta_oof_rows,
                "candidate_task_weight": candidate_task_weight_manifest,
            }
        )
    write_json(output_dir / "selector_metrics.json", selector_metrics)
    reproducibility_path = output_dir / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility_update = {
        "status": "stage_b_selector_outer_oof_completed",
        "model_manifest_sha256": selector_metrics["model_manifest_sha256"],
        "candidate_score_oof_sha256": selector_metrics["candidate_score_oof_sha256"],
        "compact_meta_oof_sha256": selector_metrics["compact_meta_oof_sha256"],
    }
    if candidate_task_weight_manifest is not None:
        reproducibility_update["candidate_task_weight_manifest_sha256"] = (
            candidate_task_weight_manifest["manifest_sha256"]
        )
    reproducibility.update(reproducibility_update)
    write_json(reproducibility_path, reproducibility)
    return selector_metrics


def run_stage_c(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    raw_train_dir: Path,
    output_dir: Path,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
    hard_readout_enabled: bool = True,
) -> dict[str, Any]:
    """Fit leakage-free nested selectors and emit downstream-fold compact meta features."""

    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    schema = load_feature_schema(output_dir / "feature_schema.json")
    features = [str(item) for item in schema["features"]]
    compact_schema = json.loads((output_dir / "compact_meta_schema.json").read_text())
    expected_compact_features = [str(item) for item in compact_schema["features"]]
    if expected_compact_features != compact_feature_names(contract):
        raise ValueError("Stage C compact schema differs from frozen Stage A schema")
    root_meta = verify_exp263_root(cache_root, config)
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    n_outer_folds = int(config["validation"]["outer_folds"])
    n_inner_folds = int(config["validation"]["inner_folds"])
    n_candidates = len(cache.ids)
    stage_cfg = dict(config["model"]["nested_downstream_stage"])
    if not bool(stage_cfg.get("enabled", False)):
        raise RuntimeError("Stage C nested_downstream_stage.enabled must be true")
    if int(stage_cfg["planned_cpu_selector_boosters"]) != (
        n_outer_folds * n_inner_folds * 2
    ):
        raise ValueError("Stage C booster contract mismatch")
    feature_cfg = dict(config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"][
        "candidates"
    ]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"][
        "candidates"
    ]
    train_cfg = dict(config["model"]["training"])
    max_train_base_rows = int(train_cfg["max_train_base_rows_per_outer_fold"])
    max_valid_base_rows = int(train_cfg["max_valid_base_rows_for_early_stopping"])
    sample_base_rows_per_source = max(
        max_valid_base_rows,
        int(math.ceil(max_train_base_rows / max(n_inner_folds - 1, 1))),
    )

    sampled: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sampled_base_wells: dict[int, np.ndarray] = {}
    fold_label_summary: dict[int, dict[str, np.ndarray]] = {}
    fold_well_counts: dict[int, pd.DataFrame] = {}
    for source_fold in range(n_outer_folds):
        bundle = cache.load_fold(source_fold)
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        sample_indices = deterministic_sample_indices(
            bundle.base,
            sample_base_rows_per_source,
            "exp264",
            "stage_c_sample",
            source_fold,
        )
        long, metadata = build_candidate_long_features(
            bundle,
            context,
            sample_indices,
            feature_cfg,
            expected_features=features,
        )
        sampled[source_fold] = (
            long,
            add_candidate_labels(metadata, truth[sample_indices], n_candidates),
        )
        sampled_base_wells[source_fold] = (
            bundle.base.iloc[sample_indices]["well"].astype(str).to_numpy()
        )
        full_error = np.abs(bundle.values - truth[:, None])
        fold_label_summary[source_fold] = {
            "error_sum": full_error.sum(axis=0, dtype=np.float64),
            "within_sum": (full_error <= 10.0).sum(axis=0, dtype=np.float64),
            "count": np.full(n_candidates, len(bundle.base), dtype=np.float64),
        }
        fold_well_counts[source_fold] = (
            bundle.base.groupby("well", sort=True)
            .size()
            .rename("rows")
            .reset_index()
        )
        del bundle, context, truth, long, metadata, full_error
        gc.collect()

    inner_maps, fold_manifest = build_nested_inner_fold_maps(
        fold_well_counts,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
    )
    fold_manifest_path = output_dir / "nested_fold_manifest.csv"
    fold_manifest.to_csv(fold_manifest_path, index=False)

    def sampled_descriptor(downstream_outer_fold: int) -> pd.DataFrame:
        assignment = inner_maps[downstream_outer_fold]
        parts = []
        for source_fold in range(n_outer_folds):
            if source_fold == downstream_outer_fold:
                continue
            wells = sampled_base_wells[source_fold]
            inner = np.asarray([assignment.get(str(well), -1) for well in wells], dtype=np.int8)
            if np.any(inner < 0):
                raise AssertionError("sampled outer-train well lacks inner-fold assignment")
            parts.append(
                pd.DataFrame(
                    {
                        "source_fold": np.int8(source_fold),
                        "base_position": np.arange(len(wells), dtype=np.int32),
                        "well": wells,
                        "inner_fold": inner,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def bounded_descriptor(
        descriptor: pd.DataFrame, limit: int, *seed_parts: Any
    ) -> pd.DataFrame:
        chosen = deterministic_sample_indices(descriptor, min(limit, len(descriptor)), *seed_parts)
        return descriptor.iloc[chosen].sort_values(
            ["source_fold", "base_position"], kind="stable"
        )

    def gather_sampled_long(
        descriptor: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_parts: list[pd.DataFrame] = []
        label_parts: list[pd.DataFrame] = []
        for source_fold, group in descriptor.groupby("source_fold", sort=True):
            base_positions = group["base_position"].to_numpy(np.int64)
            long_positions = (
                base_positions[:, None] * n_candidates
                + np.arange(n_candidates, dtype=np.int64)[None, :]
            ).reshape(-1)
            source_features, source_labels = sampled[int(source_fold)]
            feature_parts.append(source_features.iloc[long_positions])
            label_parts.append(source_labels.iloc[long_positions])
        return (
            pd.concat(feature_parts, ignore_index=True).astype(np.float32),
            pd.concat(label_parts, ignore_index=True),
        )

    common = dict(config["model"]["lightgbm_common"])
    seed = int(config["validation"]["seed"])
    num_round = int(train_cfg["num_boost_round"])

    def model_callbacks() -> list[Any]:
        return [
            early_stopping(int(train_cfg["early_stopping_rounds"]), verbose=False),
            log_evaluation(int(train_cfg["log_evaluation_period"])),
        ]

    nested_model_dir = output_dir / "nested_models"
    nested_model_dir.mkdir(parents=True, exist_ok=True)
    compact_root = output_dir / "nested_compact_meta"
    compact_root.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "nested_outer_valid_candidate_score.parquet"
    score_writer = IncrementalParquetWriter(score_path)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    compact_partition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for downstream_outer_fold in range(n_outer_folds):
        assignment = inner_maps[downstream_outer_fold]
        descriptor = sampled_descriptor(downstream_outer_fold)
        outer_valid_wells = set(
            fold_well_counts[downstream_outer_fold]["well"].astype(str)
        )
        models_by_inner: dict[int, dict[str, Any]] = {}
        model_sha_by_inner: dict[int, dict[str, str]] = {}
        for inner_fold in range(n_inner_folds):
            train_pool = descriptor[descriptor["inner_fold"].ne(inner_fold)]
            valid_pool = descriptor[descriptor["inner_fold"].eq(inner_fold)]
            fit_train = bounded_descriptor(
                train_pool,
                max_train_base_rows,
                "exp264",
                "stage_c_train",
                downstream_outer_fold,
                inner_fold,
            )
            fit_valid = bounded_descriptor(
                valid_pool,
                max_valid_base_rows,
                "exp264",
                "stage_c_valid",
                downstream_outer_fold,
                inner_fold,
            )
            train_wells = set(fit_train["well"].astype(str))
            valid_wells = set(fit_valid["well"].astype(str))
            if train_wells.intersection(valid_wells):
                raise AssertionError("Stage C inner train/valid well overlap")
            if train_wells.intersection(outer_valid_wells):
                raise AssertionError("Stage C outer-valid well leaked into selector fit")
            x_train, y_train = gather_sampled_long(fit_train)
            x_valid, y_valid = gather_sampled_long(fit_valid)
            classifier = LGBMClassifier(
                objective="binary",
                n_estimators=num_round,
                random_state=seed + 10_000 * downstream_outer_fold + inner_fold,
                **common,
            )
            regressor = LGBMRegressor(
                objective="regression_l1",
                n_estimators=num_round,
                random_state=seed + 20_000 * downstream_outer_fold + inner_fold,
                **common,
            )
            classifier.fit(
                x_train,
                y_train["candidate_within10"],
                eval_set=[(x_valid, y_valid["candidate_within10"])],
                eval_metric="binary_logloss",
                callbacks=model_callbacks(),
            )
            regressor.fit(
                x_train,
                y_train["candidate_abs_error"],
                eval_set=[(x_valid, y_valid["candidate_abs_error"])],
                eval_metric="l1",
                callbacks=model_callbacks(),
            )
            models = {"p_within10": classifier, "pred_abs_error": regressor}
            models_by_inner[inner_fold] = models
            model_sha_by_inner[inner_fold] = {}
            for objective, model in models.items():
                model_path = nested_model_dir / (
                    f"selector_{objective}_outer{downstream_outer_fold}_inner{inner_fold}.txt"
                )
                model.booster_.save_model(str(model_path))
                model_sha = sha256_file(model_path)
                model_sha_by_inner[inner_fold][objective] = model_sha
                model_rows.append(
                    {
                        "downstream_outer_fold": downstream_outer_fold,
                        "inner_fold": inner_fold,
                        "objective": objective,
                        "path": str(model_path.relative_to(output_dir)),
                        "sha256": model_sha,
                        "best_iteration": int(model.best_iteration_),
                        "fit_train_base_rows": len(fit_train),
                        "fit_valid_base_rows": len(fit_valid),
                        "fit_train_long_rows": len(x_train),
                        "fit_valid_long_rows": len(x_valid),
                        "fit_train_wells": len(train_wells),
                        "fit_valid_wells": len(valid_wells),
                    }
                )
                for importance_type in ("gain", "split"):
                    importance = model.booster_.feature_importance(
                        importance_type=importance_type
                    )
                    for feature, value in zip(features, importance, strict=True):
                        importance_rows.append(
                            {
                                "downstream_outer_fold": downstream_outer_fold,
                                "inner_fold": inner_fold,
                                "objective": objective,
                                "feature": feature,
                                "importance_type": importance_type,
                                "importance": float(value),
                            }
                        )
            del x_train, x_valid, y_train, y_valid
            gc.collect()

        outer_model_set_sha = sha256_json(
            [
                {
                    "inner_fold": inner_fold,
                    **model_sha_by_inner[inner_fold],
                }
                for inner_fold in range(n_inner_folds)
            ]
        )
        prior_error_sum = sum(
            fold_label_summary[source_fold]["error_sum"]
            for source_fold in range(n_outer_folds)
            if source_fold != downstream_outer_fold
        )
        prior_within_sum = sum(
            fold_label_summary[source_fold]["within_sum"]
            for source_fold in range(n_outer_folds)
            if source_fold != downstream_outer_fold
        )
        prior_count = sum(
            fold_label_summary[source_fold]["count"]
            for source_fold in range(n_outer_folds)
            if source_fold != downstream_outer_fold
        )
        error_prior = prior_error_sum / prior_count
        within_prior = prior_within_sum / prior_count
        fold_actual_error: list[np.ndarray] = []
        fold_pred_error: list[np.ndarray] = []
        fold_actual_within: list[np.ndarray] = []
        fold_pred_within: list[np.ndarray] = []
        hard_truth: list[np.ndarray] = []
        hard_prediction: list[np.ndarray] = []
        fallback_prediction: list[np.ndarray] = []
        chunk_size = int(train_cfg["predict_base_row_chunk_size"])
        for source_fold in range(n_outer_folds):
            role = "valid" if source_fold == downstream_outer_fold else "train"
            bundle = cache.load_fold(source_fold)
            context, truth = build_raw_context(
                bundle.base, raw_train_dir, feature_cfg, require_truth=True
            )
            assert truth is not None
            shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
            partition_path = (
                compact_root
                / f"downstream_outer_fold={downstream_outer_fold}"
                / f"role={role}"
                / f"source_outer_fold={source_fold}"
                / "part-00000.parquet"
            )
            compact_writer = IncrementalParquetWriter(partition_path)
            source_inner = None
            if role == "train":
                source_inner = np.asarray(
                    [assignment.get(str(well), -1) for well in bundle.base["well"]],
                    dtype=np.int8,
                )
                if np.any(source_inner < 0):
                    raise AssertionError("outer-train row lacks nested inner-fold assignment")
            for start in range(0, len(bundle.base), chunk_size):
                stop = min(start + chunk_size, len(bundle.base))
                indices = np.arange(start, stop, dtype=np.int64)
                long, metadata = build_candidate_long_features(
                    bundle,
                    context,
                    indices,
                    feature_cfg,
                    shape_state=shape_state,
                    expected_features=features,
                )
                x = long.astype(np.float32)
                p_matrix = np.zeros((len(indices), n_candidates), dtype=np.float32)
                e_matrix = np.zeros((len(indices), n_candidates), dtype=np.float32)
                if role == "valid":
                    for inner_fold in range(n_inner_folds):
                        models = models_by_inner[inner_fold]
                        p = models["p_within10"].predict_proba(
                            x,
                            num_iteration=models["p_within10"].best_iteration_,
                        )[:, 1]
                        e = models["pred_abs_error"].predict(
                            x,
                            num_iteration=models["pred_abs_error"].best_iteration_,
                        )
                        p_matrix += p.reshape(len(indices), n_candidates).astype(np.float32)
                        e_matrix += np.maximum(
                            e.reshape(len(indices), n_candidates), 0.0
                        ).astype(np.float32)
                    p_matrix /= np.float32(n_inner_folds)
                    e_matrix /= np.float32(n_inner_folds)
                    selector_model_count = n_inner_folds
                else:
                    assert source_inner is not None
                    chunk_inner = source_inner[indices]
                    for inner_fold in np.unique(chunk_inner):
                        base_positions = np.flatnonzero(chunk_inner == inner_fold)
                        long_positions = (
                            base_positions[:, None] * n_candidates
                            + np.arange(n_candidates, dtype=np.int64)[None, :]
                        ).reshape(-1)
                        subset = x.iloc[long_positions]
                        models = models_by_inner[int(inner_fold)]
                        p = models["p_within10"].predict_proba(
                            subset,
                            num_iteration=models["p_within10"].best_iteration_,
                        )[:, 1]
                        e = models["pred_abs_error"].predict(
                            subset,
                            num_iteration=models["pred_abs_error"].best_iteration_,
                        )
                        p_matrix[base_positions] = p.reshape(
                            len(base_positions), n_candidates
                        ).astype(np.float32)
                        e_matrix[base_positions] = np.maximum(
                            e.reshape(len(base_positions), n_candidates), 0.0
                        ).astype(np.float32)
                    selector_model_count = 1
                if not np.isfinite(p_matrix).all() or not np.isfinite(e_matrix).all():
                    raise ValueError("Stage C selector produced non-finite score")
                confidence_valid = (
                    metadata["confidence_valid"]
                    .to_numpy(bool)
                    .reshape(len(indices), n_candidates)
                )
                compact = build_compact_meta(
                    bundle.base.iloc[indices].reset_index(drop=True),
                    bundle.values[indices],
                    e_matrix,
                    p_matrix,
                    bundle.available[indices],
                    confidence_valid,
                    contract,
                )
                compact["downstream_outer_fold"] = np.int8(downstream_outer_fold)
                compact["nested_role"] = role
                compact["selector_model_count"] = np.int8(selector_model_count)
                compact_writer.write(compact)
                if role == "valid":
                    labels = add_candidate_labels(metadata, truth[indices], n_candidates)
                    actual_error = (
                        labels["candidate_abs_error"]
                        .to_numpy(np.float32)
                        .reshape(len(indices), n_candidates)
                    )
                    actual_within = (
                        labels["candidate_within10"]
                        .to_numpy(np.int8)
                        .reshape(len(indices), n_candidates)
                    )
                    score = metadata.copy()
                    score["actual_abs_error"] = actual_error.reshape(-1)
                    score["actual_within10"] = actual_within.reshape(-1)
                    score["pred_abs_error"] = e_matrix.reshape(-1)
                    score["p_within10"] = p_matrix.reshape(-1)
                    score["downstream_outer_fold"] = np.int8(downstream_outer_fold)
                    score["nested_model_count"] = np.int8(n_inner_folds)
                    score["nested_model_set_sha"] = outer_model_set_sha
                    score["feature_schema_sha"] = schema["feature_schema_sha256"]
                    score["candidate_contract_sha"] = candidate_contract_sha(contract)
                    score_writer.write(score)
                    fold_actual_error.append(actual_error.reshape(-1))
                    fold_pred_error.append(e_matrix.reshape(-1))
                    fold_actual_within.append(actual_within.reshape(-1))
                    fold_pred_within.append(p_matrix.reshape(-1))
                    if hard_readout_enabled:
                        primary_ids = contract["legal_domains"]["primitive_pair_bank"][
                            "candidates"
                        ]
                        primary_pos = [cache.ids.index(str(name)) for name in primary_ids]
                        selected_local = np.argmin(e_matrix[:, primary_pos], axis=1)
                        selected_pos = np.asarray(primary_pos, dtype=np.int64)[
                            selected_local
                        ]
                        rows = np.arange(len(indices))
                        hard_prediction.append(
                            bundle.values[indices][rows, selected_pos]
                        )
                        hard_truth.append(truth[indices])
                        fallback_pos = cache.ids.index("exp226_w500_50_50")
                        fallback_prediction.append(bundle.values[indices, fallback_pos])
                del long, metadata, x, compact, p_matrix, e_matrix
            compact_writer.close()
            compact_partition_rows.append(
                {
                    "downstream_outer_fold": downstream_outer_fold,
                    "role": role,
                    "source_outer_fold": source_fold,
                    "rows": compact_writer.rows,
                    "wells": int(bundle.base["well"].nunique()),
                    "selector_model_count": n_inner_folds if role == "valid" else 1,
                    "path": str(partition_path.relative_to(output_dir)),
                    "sha256": sha256_file(partition_path),
                    "model_set_sha256": outer_model_set_sha,
                }
            )
            del bundle, context, truth, shape_state, compact_writer
            gc.collect()

        actual_error_vector = np.concatenate(fold_actual_error)
        pred_error_vector = np.concatenate(fold_pred_error)
        actual_within_vector = np.concatenate(fold_actual_within)
        pred_within_vector = np.concatenate(fold_pred_within)
        base_rows = len(actual_error_vector) // n_candidates
        metric_row = {
            "scope": "outer_valid_inner_ensemble",
            "fold": downstream_outer_fold,
            "base_rows": base_rows,
            "long_rows": len(actual_error_vector),
            "expected_error_mae": float(
                np.mean(np.abs(pred_error_vector - actual_error_vector))
            ),
            "expected_error_prior_mae": float(
                np.mean(
                    np.abs(np.tile(error_prior, base_rows) - actual_error_vector)
                )
            ),
            "within10_logloss": _binary_logloss(
                actual_within_vector, pred_within_vector
            ),
            "within10_prior_logloss": _binary_logloss(
                actual_within_vector, np.tile(within_prior, base_rows)
            ),
            "within10_brier": float(
                np.mean(np.square(pred_within_vector - actual_within_vector))
            ),
            "within10_prior_brier": float(
                np.mean(
                    np.square(
                        np.tile(within_prior, base_rows) - actual_within_vector
                    )
                )
            ),
        }
        if hard_readout_enabled:
            hard_y = np.concatenate(hard_truth)
            hard_p = np.concatenate(hard_prediction)
            fallback_p = np.concatenate(fallback_prediction)
            metric_row.update(
                {
                    "hard_primary_rmse": _rmse(hard_y, hard_p),
                    "fixed_fallback_rmse": _rmse(hard_y, fallback_p),
                }
            )
        metric_rows.append(metric_row)
        del models_by_inner, descriptor
        gc.collect()

    score_writer.close()
    model_count = len(model_rows)
    if model_count != n_outer_folds * n_inner_folds * 2:
        raise AssertionError("Stage C model count mismatch")
    partition_manifest = pd.DataFrame(compact_partition_rows)
    partition_manifest_path = output_dir / "nested_compact_partition_manifest.csv"
    partition_manifest.to_csv(partition_manifest_path, index=False)
    expected_partition_count = n_outer_folds * n_outer_folds
    expected_compact_rows = int(root_meta["rows"]) * n_outer_folds
    compact_rows = int(partition_manifest["rows"].sum())
    if len(partition_manifest) != expected_partition_count:
        raise AssertionError("Stage C compact partition count mismatch")
    if compact_rows != expected_compact_rows:
        raise AssertionError("Stage C compact row coverage mismatch")
    if score_writer.rows != int(root_meta["rows"]) * n_candidates:
        raise AssertionError("Stage C outer-valid score coverage mismatch")

    importance = pd.DataFrame(importance_rows)
    importance_path = output_dir / "nested_feature_importance_by_objective_outer_inner.csv"
    importance.to_csv(importance_path, index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "nested_selector_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    weights = metrics["long_rows"].to_numpy(np.float64)
    pooled = {
        column: float(np.average(metrics[column], weights=weights))
        for column in [
            "expected_error_mae",
            "expected_error_prior_mae",
            "within10_logloss",
            "within10_prior_logloss",
            "within10_brier",
            "within10_prior_brier",
        ]
    }
    score_guard = {
        "expected_error_mae_improved_folds": int(
            (metrics["expected_error_mae"] < metrics["expected_error_prior_mae"]).sum()
        ),
        "within10_logloss_improved_folds": int(
            (metrics["within10_logloss"] < metrics["within10_prior_logloss"]).sum()
        ),
        "within10_brier_improved_folds": int(
            (metrics["within10_brier"] < metrics["within10_prior_brier"]).sum()
        ),
        "expected_error_mae_improved_pooled": bool(
            pooled["expected_error_mae"] < pooled["expected_error_prior_mae"]
        ),
        "within10_logloss_improved_pooled": bool(
            pooled["within10_logloss"] < pooled["within10_prior_logloss"]
        ),
        "within10_brier_improved_pooled": bool(
            pooled["within10_brier"] < pooled["within10_prior_brier"]
        ),
    }
    score_guard["passed"] = bool(
        score_guard["expected_error_mae_improved_folds"] >= 4
        and score_guard["within10_logloss_improved_folds"] >= 4
        and score_guard["within10_brier_improved_folds"] >= 4
        and score_guard["expected_error_mae_improved_pooled"]
        and score_guard["within10_logloss_improved_pooled"]
        and score_guard["within10_brier_improved_pooled"]
    )
    hard_primary_rmse = None
    fixed_fallback_rmse = None
    if hard_readout_enabled:
        hard_weights = metrics["base_rows"].to_numpy(np.float64)
        hard_primary_rmse = float(
            np.sqrt(
                np.average(
                    np.square(metrics["hard_primary_rmse"]), weights=hard_weights
                )
            )
        )
        fixed_fallback_rmse = float(
            np.sqrt(
                np.average(
                    np.square(metrics["fixed_fallback_rmse"]), weights=hard_weights
                )
            )
        )
    leakage_audit = {
        "outer_valid_excluded_from_inner_assignments": True,
        "inner_train_valid_well_disjoint": True,
        "outer_train_compact_source": "inner_oof",
        "outer_valid_compact_source": "four_inner_model_ensemble",
        "model_count": model_count,
        "compact_partition_count": len(partition_manifest),
        "compact_rows": compact_rows,
        "outer_valid_score_long_rows": score_writer.rows,
        "passed": True,
    }
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "nested_compact_meta_completed",
        "candidate_order": cache.ids,
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "compact_meta_schema_sha256": compact_schema["compact_meta_schema_sha256"],
        "models": model_rows,
        "model_count": model_count,
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "leakage_audit": leakage_audit,
    }
    model_manifest_path = output_dir / "nested_selector_model_manifest.json"
    write_json(model_manifest_path, model_manifest)
    compact_manifest = {
        "schema_version": "1.0.0",
        "status": "nested_compact_meta_completed",
        "layout": "downstream_outer_fold/role/source_outer_fold",
        "compact_meta_schema_sha256": compact_schema["compact_meta_schema_sha256"],
        "partition_count": len(partition_manifest),
        "rows": compact_rows,
        "expected_rows": expected_compact_rows,
        "partitions": compact_partition_rows,
    }
    compact_manifest_path = output_dir / "nested_compact_manifest.json"
    write_json(compact_manifest_path, compact_manifest)
    summary = {
        "status": "nested_compact_meta_completed",
        "model_count": model_count,
        "compact_partition_count": len(partition_manifest),
        "compact_rows": compact_rows,
        "outer_valid_score_long_rows": score_writer.rows,
        "pooled_score_metrics": pooled,
        "score_guard": score_guard,
        "hard_readout_enabled": hard_readout_enabled,
        "hard_primary_oof_rmse": hard_primary_rmse,
        "fixed_fallback_oof_rmse": fixed_fallback_rmse,
        "leakage_audit": leakage_audit,
        "nested_selector_model_manifest_sha256": sha256_file(model_manifest_path),
        "nested_compact_manifest_sha256": sha256_file(compact_manifest_path),
        "nested_outer_valid_candidate_score_sha256": sha256_file(score_path),
    }
    summary_path = output_dir / "nested_selector_metrics.json"
    write_json(summary_path, summary)
    reproducibility_path = output_dir / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "status": "stage_c_nested_compact_meta_completed",
            "nested_selector_model_manifest_sha256": summary[
                "nested_selector_model_manifest_sha256"
            ],
            "nested_compact_manifest_sha256": summary[
                "nested_compact_manifest_sha256"
            ],
            "nested_outer_valid_candidate_score_sha256": summary[
                "nested_outer_valid_candidate_score_sha256"
            ],
        }
    )
    write_json(reproducibility_path, reproducibility)
    return summary


def validate_current_test_native_confidence(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on the namespaced native-confidence contract used at inference."""
    fields_by_primitive = (
        contract.get("confidence_contract", {}).get(
            "current_test_required_fields_by_primitive", {}
        )
    )
    primitives = primitive_ids(contract)
    if set(fields_by_primitive) != set(primitives):
        raise ValueError(
            "current-test confidence contract must enumerate every primitive candidate"
        )
    missing: list[str] = []
    required_columns: list[str] = []
    coverage: dict[str, Any] = {}
    for name in primitives:
        fields = [str(field) for field in fields_by_primitive[name]]
        if "confidence_valid" not in fields:
            raise ValueError(f"current-test confidence_valid is not required for {name}")
        candidate_columns = [f"confidence__{name}__{field}" for field in fields]
        required_columns.extend(candidate_columns)
        missing.extend(column for column in candidate_columns if column not in frame)
    if missing:
        raise ValueError(f"current-test native confidence columns missing: {sorted(missing)}")

    for name in primitives:
        fields = [str(field) for field in fields_by_primitive[name]]
        valid_column = f"confidence__{name}__confidence_valid"
        valid_raw = frame[valid_column]
        if valid_raw.isna().any():
            raise ValueError(f"current-test confidence_valid contains NaN: {name}")
        if pd.api.types.is_bool_dtype(valid_raw):
            valid = valid_raw.astype(bool).to_numpy()
        else:
            numeric_valid = pd.to_numeric(valid_raw, errors="coerce")
            if numeric_valid.isna().any() or not numeric_valid.isin([0, 1]).all():
                raise ValueError(f"current-test confidence_valid is not boolean: {name}")
            valid = numeric_valid.astype(bool).to_numpy()
        native_fields = [field for field in fields if field != "confidence_valid"]
        for field in native_fields:
            column = f"confidence__{name}__{field}"
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"current-test native confidence is nonfinite: {name}.{field}")
        if native_fields and not valid.all():
            raise ValueError(f"current-test native confidence is invalid: {name}")
        coverage[name] = {
            "fields": fields,
            "valid_rate": float(valid.mean()),
        }
    return {
        "required_column_count": len(required_columns),
        "coverage": coverage,
    }


def current_test_bundle_from_wide(frame: pd.DataFrame, contract: Mapping[str, Any]) -> FoldBundle:
    ids = candidate_ids(contract)
    required = {"id", "well", "well_row_idx", *ids}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"exp263 current-test frame missing: {sorted(missing)}")
    base = frame[["id", "well", "well_row_idx"]].copy()
    base["outer_fold"] = np.int8(-1)
    base["md_since"] = base.groupby("well", sort=False).cumcount().to_numpy(np.float32) + 1.0
    if "last_known_tvt" in frame:
        base["last_known_tvt"] = pd.to_numeric(frame["last_known_tvt"], errors="coerce")
    else:
        base["last_known_tvt"] = np.nan
    values = frame[ids].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    available = np.isfinite(values)
    if not available.all():
        raise ValueError("current-test candidate values contain nonfinite rows")
    confidence: dict[str, pd.DataFrame] = {}
    for name in primitive_ids(contract):
        prefix = f"confidence__{name}__"
        source_columns = [column for column in frame if column.startswith(prefix)]
        if not source_columns:
            continue
        confidence_frame = base[KEY_COLUMNS].copy()
        confidence_frame["candidate_id"] = name
        for source_column in source_columns:
            field = source_column.removeprefix(prefix)
            confidence_frame[field] = frame[source_column].to_numpy()
        if "confidence_valid" not in confidence_frame:
            numeric = [
                column
                for column in confidence_frame
                if column not in KEY_COLUMNS + ["candidate_id"]
            ]
            confidence_frame["confidence_valid"] = (
                confidence_frame[numeric].notna().any(axis=1) if numeric else False
            )
        confidence[name] = confidence_frame
    return FoldBundle(base, values, available, confidence, ids, contract_by_id(contract))


def fill_current_test_anchor(bundle: FoldBundle, raw_test_dir: Path) -> None:
    anchor = np.full(len(bundle.base), np.nan, dtype=np.float32)
    for well, positions in bundle.base.groupby("well", sort=False).indices.items():
        raw = pd.read_csv(_raw_horizontal_path(raw_test_dir, str(well)), usecols=["TVT_input"])
        known = pd.to_numeric(raw["TVT_input"], errors="coerce").dropna()
        if known.empty:
            raise ValueError(f"no known TVT_input prefix for test well={well}")
        anchor[np.asarray(positions, dtype=np.int64)] = np.float32(known.iloc[-1])
    bundle.base["last_known_tvt"] = anchor


def stage_d_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the explicitly approved Stage D GPU training scope."""

    downstream_config = config["model"].get("downstream_tvt_stage") or config["model"].get(
        "downstream_tvt"
    )
    if not isinstance(downstream_config, Mapping):
        raise ValueError("downstream TVT stage config is missing")
    stage_cfg = dict(downstream_config)
    expected_approval_scope = (
        "clean273_control15_compact347_addonly15_three_configs_five_folds_30_gpu_boosters"
    )
    variants = [str(item) for item in stage_cfg["variants"]]
    config_indices = [int(item) for item in stage_cfg["lightgbm_config_indices"]]
    folds = int(stage_cfg["folds"])
    planned = int(stage_cfg["planned_gpu_boosters"])
    calculated = len(variants) * len(config_indices) * folds
    if variants != ["matched_control", "selector_compact_addonly"]:
        raise ValueError(f"unexpected Stage D variants: {variants}")
    if config_indices != [0, 1, 2]:
        raise ValueError(f"unexpected Stage D LightGBM config indices: {config_indices}")
    if folds != 5 or calculated != 30 or planned != calculated:
        raise ValueError(
            "Stage D cost contract must be 2 variants x 3 configs x 5 folds = 30 boosters"
        )
    if not bool(stage_cfg.get("control_retraining", False)):
        raise ValueError("Stage D matched control retraining must be enabled")
    if not bool(stage_cfg.get("enabled", False)):
        raise ValueError("Stage D downstream_tvt_stage.enabled must be true")
    if not bool(stage_cfg.get("previous_approval_scope_invalidated", False)):
        raise ValueError("Stage D must retain the invalidation of the historical approval")
    if not stage_cfg.get("corrected_run_approval_received_at"):
        raise ValueError("corrected Stage D approval is missing")
    if str(stage_cfg.get("corrected_run_approval_scope", "")) != expected_approval_scope:
        raise ValueError("corrected Stage D approval scope mismatch")
    expected_surface = {
        "feature_surface": "exp218_clean_273_drop_107",
        "expected_source_base_feature_count": 380,
        "expected_base_feature_count": 273,
        "expected_compact_feature_count": 74,
        "matched_control_feature_count": 273,
        "selector_compact_addonly_feature_count": 347,
    }
    for key, expected in expected_surface.items():
        if stage_cfg.get(key) != expected:
            raise ValueError(
                f"corrected Stage D feature surface mismatch for {key}: "
                f"{stage_cfg.get(key)} != {expected}"
            )
    return {
        "variants": variants,
        "lightgbm_config_indices": config_indices,
        "folds": folds,
        "boosters_per_variant": len(config_indices) * folds,
        "total_gpu_boosters": calculated,
        "control_retraining": True,
        "approval_received_at": stage_cfg.get("corrected_run_approval_received_at"),
        "approval_scope": stage_cfg.get("corrected_run_approval_scope"),
    }


def resolve_stage_c_artifact_root(
    config: Mapping[str, Any], search_roots: Sequence[Path]
) -> Path:
    """Resolve a complete Stage C artifact root, rejecting metadata-only downloads."""

    data = dict(config.get("data", {}))
    patterns = [str(item) for item in data.get("stage_c_artifact_root_patterns", [])]
    candidates: list[Path] = []
    for raw in patterns:
        path = Path(raw)
        if path.exists():
            candidates.append(path)
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path for path in root.glob(raw) if path.is_dir())
        candidates.extend(path.parent for path in root.rglob("nested_compact_manifest.json"))
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        manifest_path = candidate / "nested_compact_manifest.json"
        metrics_path = candidate / "nested_selector_metrics.json"
        schema_path = candidate / "compact_meta_schema.json"
        model_manifest_path = candidate / "nested_selector_model_manifest.json"
        checked.append(str(candidate))
        if not all(
            path.exists()
            for path in [manifest_path, metrics_path, schema_path, model_manifest_path]
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        partition_paths = [
            candidate / str(item["path"]) for item in manifest.get("partitions", [])
        ]
        if partition_paths and all(
            path.exists() and path.stat().st_size > 0 for path in partition_paths
        ):
            return candidate
    raise FileNotFoundError(
        "complete Stage C artifact root not found; checked=" + json.dumps(checked[:80])
    )


def verify_stage_c_artifact_root(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_partition_sha256: bool = True,
    expected_compact_feature_count: int = 74,
    require_score_guard: bool = True,
) -> dict[str, Any]:
    """Verify frozen Stage C score/leakage/schema/partition evidence."""

    root = Path(root)
    data = dict(config.get("data", {}))
    files = {
        "nested_selector_metrics": root / "nested_selector_metrics.json",
        "nested_selector_model_manifest": root / "nested_selector_model_manifest.json",
        "nested_compact_manifest": root / "nested_compact_manifest.json",
        "compact_meta_schema": root / "compact_meta_schema.json",
    }
    expected_keys = {
        "nested_selector_metrics": "stage_c_expected_nested_selector_metrics_sha256",
        "nested_selector_model_manifest": (
            "stage_c_expected_nested_selector_model_manifest_sha256"
        ),
        "nested_compact_manifest": "stage_c_expected_nested_compact_manifest_sha256",
        "compact_meta_schema": "stage_c_expected_compact_meta_schema_file_sha256",
    }
    sha: dict[str, str] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Stage C contract file missing: {path}")
        sha[name] = sha256_file(path)
        expected = str(data.get(expected_keys[name], ""))
        if expected and sha[name] != expected:
            raise ValueError(f"Stage C {name} SHA mismatch: {sha[name]} != {expected}")

    metrics = json.loads(files["nested_selector_metrics"].read_text())
    model_manifest = json.loads(files["nested_selector_model_manifest"].read_text())
    compact_manifest = json.loads(files["nested_compact_manifest"].read_text())
    compact_schema = json.loads(files["compact_meta_schema"].read_text())
    if require_score_guard and not bool(
        metrics.get("score_guard", {}).get("passed", False)
    ):
        raise ValueError("Stage C selector score guard did not pass")
    if not bool(metrics.get("leakage_audit", {}).get("passed", False)):
        raise ValueError("Stage C leakage audit did not pass")
    if int(metrics.get("model_count", -1)) != 40 or int(
        model_manifest.get("model_count", -1)
    ) != 40:
        raise ValueError("Stage C must contain exactly 40 selector models")
    if int(compact_manifest.get("partition_count", -1)) != 25:
        raise ValueError("Stage C must contain exactly 25 compact partitions")
    if int(compact_manifest.get("rows", -1)) != 18_919_945:
        raise ValueError("Stage C compact row contract mismatch")
    compact_features = [str(item) for item in compact_schema.get("features", [])]
    if len(compact_features) != int(expected_compact_feature_count) or len(
        set(compact_features)
    ) != int(expected_compact_feature_count):
        raise ValueError(
            "Stage C compact schema feature count mismatch: "
            f"{len(compact_features)} != {expected_compact_feature_count}"
        )
    schema_sha = str(compact_schema.get("compact_meta_schema_sha256", ""))
    if schema_sha != str(compact_manifest.get("compact_meta_schema_sha256", "")):
        raise ValueError("Stage C compact schema logical SHA mismatch")
    expected_logical_schema_sha = str(
        data.get("stage_c_expected_compact_meta_schema_logical_sha256", "")
    )
    if expected_logical_schema_sha and schema_sha != expected_logical_schema_sha:
        raise ValueError(
            "Stage C compact schema logical SHA mismatch: "
            f"{schema_sha} != {expected_logical_schema_sha}"
        )

    partition_rows = 0
    partition_evidence: list[dict[str, Any]] = []
    for item in compact_manifest.get("partitions", []):
        path = root / str(item["path"])
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage C compact partition missing: {path}")
        actual_sha = sha256_file(path) if verify_partition_sha256 else None
        if actual_sha is not None and actual_sha != str(item["sha256"]):
            raise ValueError(f"Stage C compact partition SHA mismatch: {path}")
        rows = int(item["rows"])
        partition_rows += rows
        partition_evidence.append(
            {
                "downstream_outer_fold": int(item["downstream_outer_fold"]),
                "role": str(item["role"]),
                "source_outer_fold": int(item["source_outer_fold"]),
                "rows": rows,
                "path": str(path),
                "sha256": actual_sha or str(item["sha256"]),
            }
        )
    if len(partition_evidence) != 25 or partition_rows != 18_919_945:
        raise ValueError("Stage C compact partition inventory mismatch")
    return {
        "root": str(root),
        "sha256": sha,
        "model_count": 40,
        "partition_count": 25,
        "compact_rows": partition_rows,
        "compact_feature_count": len(compact_features),
        "compact_features": compact_features,
        "compact_meta_schema_sha256": schema_sha,
        "partitions": partition_evidence,
        "score_guard": metrics["score_guard"],
        "leakage_audit": metrics["leakage_audit"],
    }


def _load_python_module(module_path: Path, module_name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_get(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def apply_stage_d_base_feature_allowlist(
    parent_features: Sequence[str],
    *,
    allowlist_path: Path,
    expected_source_count: int,
    expected_selected_count: int,
    expected_allowlist_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    """Filter the historical exp218 surface to the audited fold-safe allowlist."""

    parent = [str(item) for item in parent_features]
    if len(parent) != expected_source_count or len(set(parent)) != expected_source_count:
        raise ValueError(
            "exp218 source feature contract mismatch: "
            f"{len(parent)} != {expected_source_count} unique features"
        )
    path = Path(allowlist_path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != str(expected_allowlist_sha256):
        raise ValueError(
            "exp218 clean allowlist SHA mismatch: "
            f"{actual_sha256} != {expected_allowlist_sha256}"
        )
    frame = pd.read_csv(path)
    if list(frame.columns) != ["feature", "family"]:
        raise ValueError("exp218 clean allowlist must contain exactly feature,family columns")
    allowlist = frame["feature"].astype(str).tolist()
    if len(allowlist) != expected_selected_count or len(set(allowlist)) != expected_selected_count:
        raise ValueError(
            "exp218 clean allowlist count mismatch: "
            f"{len(allowlist)} != {expected_selected_count} unique features"
        )
    allowlist_set = set(allowlist)
    unknown = sorted(allowlist_set - set(parent))
    if unknown:
        raise ValueError(f"exp218 clean allowlist contains unknown features: {unknown}")
    selected = [feature for feature in parent if feature in allowlist_set]
    if selected != allowlist:
        raise ValueError("exp218 clean allowlist order differs from the source feature order")
    evidence = {
        "path": str(path),
        "sha256": actual_sha256,
        "source_feature_count": len(parent),
        "selected_feature_count": len(selected),
        "dropped_feature_count": len(parent) - len(selected),
        "selected_feature_schema_sha256": sha256_json(selected),
    }
    return selected, evidence


def build_stage_d_exp218_surface(
    *,
    exp218_source_path: Path,
    exp218_config_path: Path,
    base_feature_allowlist_path: Path,
    raw_train_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any], Any, dict[str, Any]]:
    """Rebuild exp218 and retain only the audited fold-safe 273-feature surface."""

    exp218 = _load_python_module(Path(exp218_source_path), "exp264_stage_d_exp218")
    exp218_config = read_yaml(Path(exp218_config_path))
    base_frame, base_feature_columns, base_meta = exp218.load_exp072_full_replay_cache_frame(
        _nested_get(exp218_config, "data.exp072_train_feature_cache_local"),
        max_rows=None,
    )
    base_frame, anchor_meta = exp218.add_anchor_columns(base_frame, raw_train_dir)

    projection_cfg = dict(_nested_get(exp218_config, "model.u_projection", {}) or {})
    projection, projection_groups, projection_summary = exp218.build_u_projection_features(
        base_frame,
        source_specs=dict(projection_cfg.get("sources") or {}),
        degree=int(projection_cfg.get("degree", 3)),
        robust_iters=int(projection_cfg.get("robust_iters", 3)),
        clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
    )
    if not base_frame[["id", "well"]].reset_index(drop=True).equals(
        projection[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("exp218 projection features are not base-row aligned")
    projection_columns = [item for item in projection if item not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(base_frame, projection, projection_columns)

    learned_source, learned_meta = exp218.load_learned_likelihood_ml_features(
        _nested_get(exp218_config, "data.learned_likelihood_train_features_local"),
        schema_path=_nested_get(
            exp218_config, "data.learned_likelihood_train_feature_schema_local"
        ),
        summary_path=_nested_get(exp218_config, "data.learned_likelihood_train_summary_local"),
    )
    learned, learned_groups, learned_summary = exp218.build_learned_likelihood_features(
        learned_source,
        base_frame,
        dict(_nested_get(exp218_config, "model.learned_likelihood_features", {}) or {}),
    )
    if not base_frame[["id", "well"]].reset_index(drop=True).equals(
        learned[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("exp218 learned-likelihood features are not base-row aligned")
    learned_columns = [item for item in learned if item not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(base_frame, learned, learned_columns)

    grwr, grwr_groups, grwr_summary, grwr_meta = (
        exp218.build_gr_wavelet_rotation_confidence_features(
            base_frame,
            train_dir=raw_train_dir,
            config=dict(
                _nested_get(
                    exp218_config,
                    "model.gr_wavelet_rotation_confidence_features",
                    {},
                )
                or {}
            ),
        )
    )
    if not base_frame[["id", "well"]].reset_index(drop=True).equals(
        grwr[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("exp218 GRWR features are not base-row aligned")
    grwr_columns = [item for item in grwr if item not in {"id", "well"}]
    exp218._assign_aligned_float32_columns(base_frame, grwr, grwr_columns)
    feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
    active_variants = list(
        _nested_get(exp218_config, "model.feature_ablation.active_variants", []) or []
    )
    parent_variant = next(
        item
        for item in active_variants
        if str(item.get("name")) == "gr_wavelet_rotation_confidence_addonly"
    )
    source_features = exp218.feature_columns_for_variant(
        base_feature_columns, feature_groups, parent_variant
    )
    downstream_config = config["model"].get("downstream_tvt_stage") or config["model"].get(
        "downstream_tvt"
    )
    if not isinstance(downstream_config, Mapping):
        raise ValueError("downstream TVT stage config is missing")
    stage_cfg = dict(downstream_config)
    features, allowlist_evidence = apply_stage_d_base_feature_allowlist(
        source_features,
        allowlist_path=base_feature_allowlist_path,
        expected_source_count=int(stage_cfg["expected_source_base_feature_count"]),
        expected_selected_count=int(stage_cfg["expected_base_feature_count"]),
        expected_allowlist_sha256=str(stage_cfg["base_feature_allowlist_sha256"]),
    )
    required = {"id", "well", "target", "last_known_tvt", "md_since", *features}
    missing = required - set(base_frame.columns)
    if missing:
        raise ValueError(f"exp218 Stage D base surface missing columns: {sorted(missing)}")
    if base_frame["id"].astype(str).duplicated().any():
        raise ValueError("exp218 Stage D base ids are not unique")
    for start in range(0, len(features), 32):
        columns = features[start : start + 32]
        values = base_frame.loc[:, columns].to_numpy(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(
                f"exp218 Stage D base features contain non-finite values: {columns}"
            )
        del values
    del projection, learned_source, learned, grwr
    gc.collect()
    evidence = {
        "source": str(exp218_source_path),
        "source_sha256": sha256_file(exp218_source_path),
        "config": str(exp218_config_path),
        "config_sha256": sha256_file(exp218_config_path),
        "rows": int(len(base_frame)),
        "wells": int(base_frame["well"].nunique()),
        "feature_count": len(features),
        "feature_schema_sha256": sha256_json(features),
        "clean_base_feature_allowlist": allowlist_evidence,
        "base_cache": base_meta,
        "anchor": anchor_meta,
        "learned_likelihood": learned_meta,
        "projection_summary_rows": int(len(projection_summary)),
        "learned_summary_rows": int(len(learned_summary)),
        "grwr_summary_rows": int(len(grwr_summary)),
        "grwr": grwr_meta,
    }
    return base_frame, features, evidence, exp218, exp218_config


def load_stage_d_compact_fold(
    *,
    stage_c_root: Path,
    stage_c_evidence: Mapping[str, Any],
    downstream_outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one authoritative Stage C downstream fold in train/valid roles."""

    compact_features = [str(item) for item in stage_c_evidence["compact_features"]]
    metadata_columns = [
        *KEY_COLUMNS,
        "last_known_tvt",
        "downstream_outer_fold",
        "nested_role",
        "selector_model_count",
    ]
    read_columns = [*metadata_columns, *compact_features]
    by_role: dict[str, list[pd.DataFrame]] = {"train": [], "valid": []}
    selected = [
        item
        for item in stage_c_evidence["partitions"]
        if int(item["downstream_outer_fold"]) == int(downstream_outer_fold)
    ]
    for item in sorted(selected, key=lambda value: int(value["source_outer_fold"])):
        role = str(item["role"])
        if role not in by_role:
            raise ValueError(f"unexpected Stage C compact role: {role}")
        part = pd.read_parquet(Path(item["path"]), columns=read_columns)
        if len(part) != int(item["rows"]):
            raise ValueError(f"Stage C compact partition row mismatch: {item['path']}")
        if not part["outer_fold"].eq(int(item["source_outer_fold"])).all():
            raise ValueError(f"Stage C compact source fold mismatch: {item['path']}")
        expected_model_count = 4 if role == "valid" else 1
        if not part["selector_model_count"].eq(expected_model_count).all():
            raise ValueError(f"Stage C compact selector model count mismatch: {item['path']}")
        by_role[role].append(part)
    if len(by_role["train"]) != 4 or len(by_role["valid"]) != 1:
        raise ValueError("Stage D fold must contain four train and one valid compact partitions")
    train = pd.concat(by_role["train"], ignore_index=True)
    valid = pd.concat(by_role["valid"], ignore_index=True)
    for role, frame in [("train", train), ("valid", valid)]:
        if not frame["nested_role"].eq(role).all():
            raise ValueError(f"Stage C nested role mismatch for {role}")
        if not frame["downstream_outer_fold"].eq(downstream_outer_fold).all():
            raise ValueError(f"Stage C downstream fold mismatch for {role}")
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"Stage C compact ids are duplicated within {role}")
        numeric = frame[["last_known_tvt", *compact_features]].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"Stage C compact features are non-finite in {role}")
    if set(train["well"].astype(str)).intersection(set(valid["well"].astype(str))):
        raise ValueError("Stage D compact train/valid wells overlap")
    return train, valid


def stage_d_matched_guard(
    *,
    pooled_delta_rmse: float,
    fold_deltas: Sequence[float],
    near_delta_rmse: float,
    distance_1000_plus_delta_rmse: float,
    hidden_like_deltas: Sequence[float],
    worst_well_delta_rmse: float,
    guard_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen add-only-vs-matched-control decision rule."""

    improved_folds = int(sum(float(value) < 0.0 for value in fold_deltas))
    hidden_max = max((float(value) for value in hidden_like_deltas), default=float("inf"))
    checks = {
        "primary_overall_improvement": bool(pooled_delta_rmse < 0.0),
        "minimum_improved_folds": bool(
            improved_folds >= int(guard_config["min_improved_folds"])
        ),
        "near_non_regression": bool(
            near_delta_rmse <= float(guard_config["max_near_delta_rmse"])
        ),
        "distance_1000_plus_non_regression": bool(
            distance_1000_plus_delta_rmse
            <= float(guard_config["max_1000_plus_delta_rmse"])
        ),
        "hidden_like_non_regression": bool(
            hidden_max <= float(guard_config["max_hidden_like_delta_rmse"])
        ),
        "worst_well_regression": bool(
            worst_well_delta_rmse
            <= float(guard_config["max_worst_well_regression"])
        ),
    }
    return {
        "pooled_delta_rmse_addonly_minus_control": float(pooled_delta_rmse),
        "fold_deltas_addonly_minus_control": [float(item) for item in fold_deltas],
        "improved_folds": improved_folds,
        "near_delta_rmse": float(near_delta_rmse),
        "distance_1000_plus_delta_rmse": float(distance_1000_plus_delta_rmse),
        "hidden_like_max_delta_rmse": float(hidden_max),
        "worst_well_delta_rmse": float(worst_well_delta_rmse),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _rmse_arrays(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(prediction, dtype=np.float64)
                    - np.asarray(actual, dtype=np.float64)
                )
            )
        )
    )


def run_stage_d(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    base_feature_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the approved 30-booster matched Stage D downstream TVT ablation."""

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cost = stage_d_cost_contract(config)
    stage_c_evidence = verify_stage_c_artifact_root(stage_c_root, config)
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=base_feature_allowlist_path,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    compact_features = [str(item) for item in stage_c_evidence["compact_features"]]
    stage_cfg = dict(config["model"]["downstream_tvt_stage"])
    if len(compact_features) != int(stage_cfg["expected_compact_feature_count"]):
        raise ValueError("Stage D compact feature count differs from config")
    mode_name = str(stage_cfg["mode"])
    mode_config = dict(
        _nested_get(exp218_config, f"model.training.modes.{mode_name}", {}) or {}
    )
    if not bool(mode_config.get("use_gpu", False)):
        raise ValueError("Stage D approved mode must use GPU")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode_config
    )
    config_indices = [int(item) for item in cost["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    variants = [str(item) for item in cost["variants"]]
    expected_features = {
        "matched_control": int(stage_cfg["matched_control_feature_count"]),
        "selector_compact_addonly": int(stage_cfg["selector_compact_addonly_feature_count"]),
    }
    base_index = pd.Index(base_frame["id"].astype(str), name="id")
    if not base_index.is_unique:
        raise ValueError("Stage D base id index is not unique")
    n_rows = len(base_frame)
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof = {
        variant: [np.full(n_rows, np.nan, np.float32) for _ in params_family]
        for variant in variants
    }
    oof_fold = np.full(n_rows, -1, np.int8)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    model_dir = output_dir / "stage_d_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    chunk_columns = int(stage_cfg["matrix_copy_chunk_columns"])

    for outer_fold in range(int(cost["folds"])):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=outer_fold,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("Stage C compact ids are absent from the exp218 base surface")
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != n_rows:
            raise ValueError("Stage D train/valid compact rows do not cover base rows exactly once")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Stage D train/valid base indices overlap")
        for frame, indices, role in [
            (compact_train, train_indices, "train"),
            (compact_valid, valid_indices, "valid"),
        ]:
            base_wells = base_frame["well"].iloc[indices].astype(str).to_numpy()
            if not np.array_equal(base_wells, frame["well"].astype(str).to_numpy()):
                raise ValueError(f"Stage D {role} well alignment mismatch")
            base_anchor = anchor[indices]
            compact_anchor = frame["last_known_tvt"].to_numpy(np.float32)
            if float(np.max(np.abs(base_anchor - compact_anchor))) > 1.0e-4:
                raise ValueError(f"Stage D {role} anchor alignment mismatch")
            base_md_since = base_frame["md_since"].iloc[indices].to_numpy(np.float32)
            compact_md_since = frame["md_since"].to_numpy(np.float32)
            if float(np.max(np.abs(base_md_since - compact_md_since))) > 1.0e-4:
                raise ValueError(f"Stage D {role} md_since alignment mismatch")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("Stage D OOF valid rows were assigned more than once")
        oof_fold[valid_indices] = np.int8(outer_fold)

        for variant in variants:
            variant_features = (
                base_features
                if variant == "matched_control"
                else [*base_features, *compact_features]
            )
            if len(variant_features) != expected_features[variant]:
                raise ValueError(
                    f"Stage D {variant} feature count mismatch: {len(variant_features)}"
                )
            x_train_values = np.empty(
                (len(train_indices), len(variant_features)), dtype=np.float32
            )
            x_valid_values = np.empty(
                (len(valid_indices), len(variant_features)), dtype=np.float32
            )
            for start in range(0, len(base_features), chunk_columns):
                stop = min(start + chunk_columns, len(base_features))
                columns = base_features[start:stop]
                base_chunk = base_frame.loc[:, columns]
                x_train_values[:, start:stop] = base_chunk.iloc[train_indices].to_numpy(
                    np.float32, copy=True
                )
                x_valid_values[:, start:stop] = base_chunk.iloc[valid_indices].to_numpy(
                    np.float32, copy=True
                )
                del base_chunk
            if variant == "selector_compact_addonly":
                start = len(base_features)
                x_train_values[:, start:] = compact_train[compact_features].to_numpy(
                    np.float32, copy=False
                )
                x_valid_values[:, start:] = compact_valid[compact_features].to_numpy(
                    np.float32, copy=False
                )
            x_train = pd.DataFrame(
                x_train_values, columns=variant_features, copy=False
            )
            x_valid = pd.DataFrame(
                x_valid_values, columns=variant_features, copy=False
            )
            fold_predictions: list[np.ndarray] = []
            variant_model_dir = model_dir / variant
            variant_model_dir.mkdir(parents=True, exist_ok=True)
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
                        early_stopping(
                            int(stage_cfg["early_stopping_rounds"]), verbose=False
                        ),
                        log_evaluation(int(stage_cfg["log_evaluation_period"])),
                    ],
                )
                best_iteration = int(model.best_iteration_ or params["n_estimators"])
                prediction = model.predict(
                    x_valid, num_iteration=best_iteration
                ).astype(np.float32)
                oof[variant][family_position][valid_indices] = prediction
                fold_predictions.append(prediction)
                model_path = (
                    variant_model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
                )
                model.booster_.save_model(
                    str(model_path), num_iteration=best_iteration
                )
                model_sha = sha256_file(model_path)
                rmse_value = _rmse_arrays(
                    truth[valid_indices], anchor[valid_indices] + prediction
                )
                fold_metric_rows.append(
                    {
                        "variant": variant,
                        "model": f"lgb{config_index}",
                        "outer_fold": outer_fold,
                        "rows": len(valid_indices),
                        "train_rows": len(train_indices),
                        "features": len(variant_features),
                        "best_iteration": best_iteration,
                        "rmse_tvt": rmse_value,
                    }
                )
                model_rows.append(
                    {
                        "variant": variant,
                        "model": f"lgb{config_index}",
                        "config_index": config_index,
                        "outer_fold": outer_fold,
                        "feature_count": len(variant_features),
                        "best_iteration": best_iteration,
                        "path": str(model_path.relative_to(output_dir)),
                        "sha256": model_sha,
                        "params": params,
                    }
                )
                for importance_type in ["gain", "split"]:
                    importance = model.booster_.feature_importance(
                        importance_type=importance_type
                    )
                    importance_rows.extend(
                        {
                            "variant": variant,
                            "model": f"lgb{config_index}",
                            "outer_fold": outer_fold,
                            "importance_type": importance_type,
                            "feature": feature,
                            "importance": float(value),
                        }
                        for feature, value in zip(
                            variant_features, importance, strict=True
                        )
                    )
                print(
                    json.dumps(
                        {
                            "stage": "D",
                            "variant": variant,
                            "model": f"lgb{config_index}",
                            "outer_fold": outer_fold,
                            "rmse_tvt": rmse_value,
                            "best_iteration": best_iteration,
                            "completed_boosters": len(model_rows),
                            "planned_boosters": cost["total_gpu_boosters"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del model, prediction
                gc.collect()
            fold_mean = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
            fold_metric_rows.append(
                {
                    "variant": variant,
                    "model": "lgb_mean",
                    "outer_fold": outer_fold,
                    "rows": len(valid_indices),
                    "train_rows": len(train_indices),
                    "features": len(variant_features),
                    "best_iteration": None,
                    "rmse_tvt": _rmse_arrays(
                        truth[valid_indices], anchor[valid_indices] + fold_mean
                    ),
                }
            )
            del x_train, x_valid, x_train_values, x_valid_values, fold_predictions, fold_mean
            gc.collect()
        del compact_train, compact_valid, train_indices, valid_indices
        gc.collect()

    if len(model_rows) != int(cost["total_gpu_boosters"]):
        raise AssertionError(f"Stage D trained {len(model_rows)} models instead of 30")
    if np.any(oof_fold < 0):
        raise AssertionError("Stage D OOF fold assignment is incomplete")
    for variant in variants:
        for config_index, prediction in zip(config_indices, oof[variant], strict=True):
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Stage D OOF is incomplete: {variant}/lgb{config_index}")

    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    pooled_metric_rows: list[dict[str, Any]] = []
    variant_mean_tvt: dict[str, np.ndarray] = {}
    for variant in variants:
        for config_index, prediction in zip(config_indices, oof[variant], strict=True):
            pred_tvt = (anchor + prediction).astype(np.float32)
            prediction_frame[f"{variant}__lgb{config_index}__pred_tvt"] = pred_tvt
            pooled_metric_rows.append(
                {
                    "variant": variant,
                    "model": f"lgb{config_index}",
                    "rmse_tvt": _rmse_arrays(truth, pred_tvt),
                    "rows": n_rows,
                }
            )
        mean_residual = np.mean(np.vstack(oof[variant]), axis=0).astype(np.float32)
        mean_tvt = (anchor + mean_residual).astype(np.float32)
        variant_mean_tvt[variant] = mean_tvt
        prediction_frame[f"{variant}__lgb_mean__pred_tvt"] = mean_tvt
        pooled_metric_rows.append(
            {
                "variant": variant,
                "model": "lgb_mean",
                "rmse_tvt": _rmse_arrays(truth, mean_tvt),
                "rows": n_rows,
            }
        )

    control = variant_mean_tvt["matched_control"]
    addonly = variant_mean_tvt["selector_compact_addonly"]
    fold_metrics = pd.DataFrame(fold_metric_rows)
    pooled_metrics = pd.DataFrame(pooled_metric_rows)
    by_well_source = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "matched_control": control,
            "selector_compact_addonly": addonly,
        }
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in by_well_source.groupby("well", sort=True):
        control_rmse = _rmse_arrays(group["actual_tvt"], group["matched_control"])
        add_rmse = _rmse_arrays(
            group["actual_tvt"], group["selector_compact_addonly"]
        )
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "matched_control_rmse": control_rmse,
                "selector_compact_addonly_rmse": add_rmse,
                "delta_rmse_addonly_minus_control": add_rmse - control_rmse,
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    md_since = base_frame["md_since"].to_numpy(np.float32)
    bucket_masks = {
        "all": np.ones(n_rows, dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_rows: list[dict[str, Any]] = []
    for bucket, mask in bucket_masks.items():
        if not np.any(mask):
            continue
        control_rmse = _rmse_arrays(truth[mask], control[mask])
        add_rmse = _rmse_arrays(truth[mask], addonly[mask])
        bucket_rows.append(
            {
                "bucket": bucket,
                "rows": int(mask.sum()),
                "matched_control_rmse": control_rmse,
                "selector_compact_addonly_rmse": add_rmse,
                "delta_rmse_addonly_minus_control": add_rmse - control_rmse,
            }
        )
    bucket_metrics = pd.DataFrame(bucket_rows)

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    assignment_by_well = assignment.set_index("well_id")
    hidden_rows: list[dict[str, Any]] = []
    for column in [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]:
        role = base_frame["well"].astype(str).map(assignment_by_well[column])
        mask = role.eq("valid").to_numpy()
        if not np.any(mask):
            raise ValueError(f"hidden-like assignment has no valid rows for {column}")
        control_rmse = _rmse_arrays(truth[mask], control[mask])
        add_rmse = _rmse_arrays(truth[mask], addonly[mask])
        hidden_rows.append(
            {
                "assignment": column,
                "role": "valid",
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "matched_control_rmse": control_rmse,
                "selector_compact_addonly_rmse": add_rmse,
                "delta_rmse_addonly_minus_control": add_rmse - control_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)

    mean_folds = fold_metrics[fold_metrics["model"].eq("lgb_mean")]
    fold_deltas: list[float] = []
    for fold in range(int(cost["folds"])):
        control_rmse = float(
            mean_folds[
                mean_folds["variant"].eq("matched_control")
                & mean_folds["outer_fold"].eq(fold)
            ]["rmse_tvt"].iloc[0]
        )
        add_rmse = float(
            mean_folds[
                mean_folds["variant"].eq("selector_compact_addonly")
                & mean_folds["outer_fold"].eq(fold)
            ]["rmse_tvt"].iloc[0]
        )
        fold_deltas.append(add_rmse - control_rmse)
    pooled_control = _rmse_arrays(truth, control)
    pooled_add = _rmse_arrays(truth, addonly)
    bucket_lookup = bucket_metrics.set_index("bucket")
    guard = stage_d_matched_guard(
        pooled_delta_rmse=pooled_add - pooled_control,
        fold_deltas=fold_deltas,
        near_delta_rmse=float(
            bucket_lookup.loc["near_0_250", "delta_rmse_addonly_minus_control"]
        ),
        distance_1000_plus_delta_rmse=float(
            bucket_lookup.loc["1000_plus", "delta_rmse_addonly_minus_control"]
        ),
        hidden_like_deltas=hidden_metrics[
            "delta_rmse_addonly_minus_control"
        ].tolist(),
        worst_well_delta_rmse=float(
            by_well["delta_rmse_addonly_minus_control"].max()
        ),
        guard_config=config["guards"]["downstream_tvt_addonly"],
    )

    fold_path = output_dir / "stage_d_fold_metrics.csv"
    oof_path = output_dir / "stage_d_oof_predictions.parquet"
    importance_path = output_dir / "stage_d_feature_importance.csv"
    by_well_path = output_dir / "stage_d_by_well.csv"
    bucket_path = output_dir / "stage_d_bucket_metrics.csv"
    hidden_path = output_dir / "stage_d_hidden_like_metrics.csv"
    manifest_path = output_dir / "stage_d_model_manifest.json"
    metrics_path = output_dir / "stage_d_metrics.json"
    fold_metrics.to_csv(fold_path, index=False)
    prediction_frame.to_parquet(oof_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    hidden_metrics.to_csv(hidden_path, index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_d_30_gpu_boosters_completed",
        "cost_contract": cost,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_surfaces": {
            "matched_control": base_features,
            "selector_compact_addonly": [*base_features, *compact_features],
        },
        "feature_schema_sha256": {
            "matched_control": sha256_json(base_features),
            "selector_compact_addonly": sha256_json([*base_features, *compact_features]),
        },
        "stage_c_input": stage_c_evidence,
        "exp218_input": base_evidence,
    }
    write_json(manifest_path, model_manifest)
    metrics = {
        "status": "stage_d_30_gpu_boosters_completed",
        "cost_contract": cost,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": expected_features,
        "pooled_metrics": pooled_metrics.to_dict(orient="records"),
        "matched_control_lgb_mean_rmse": pooled_control,
        "selector_compact_addonly_lgb_mean_rmse": pooled_add,
        "delta_rmse_addonly_minus_control": pooled_add - pooled_control,
        "guard": guard,
        "model_count": len(model_rows),
    }
    write_json(metrics_path, metrics)
    artifact_sha = {
        path.name: sha256_file(path)
        for path in [
            metrics_path,
            fold_path,
            oof_path,
            manifest_path,
            importance_path,
            by_well_path,
            bucket_path,
            hidden_path,
        ]
    }
    reproducibility = {
        "schema_version": "1.0.0",
        "status": "stage_d_30_gpu_boosters_completed",
        "cost_contract": cost,
        "stage_c_input": stage_c_evidence,
        "exp218_input": base_evidence,
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": sha256_file(hidden_like_assignment_path),
        },
        "output_sha256": artifact_sha,
        "model_manifest_sha256": artifact_sha[manifest_path.name],
        "guard": guard,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    return metrics


__all__ = [
    "Exp263CandidateCache",
    "ShapeState",
    "add_candidate_labels",
    "apply_stage_d_base_feature_allowlist",
    "build_candidate_long_features",
    "build_compact_meta",
    "build_nested_inner_fold_maps",
    "build_raw_context",
    "candidate_contract_sha",
    "candidate_ids",
    "compact_feature_names",
    "current_test_bundle_from_wide",
    "deterministic_sample_indices",
    "fill_current_test_anchor",
    "logical_frame_sha256",
    "read_yaml",
    "resolve_existing_path",
    "resolve_exp263_cache_root",
    "run_stage_a",
    "run_stage_b",
    "run_stage_c",
    "run_stage_d",
    "sha256_file",
    "stage_d_cost_contract",
    "stage_d_matched_guard",
    "validate_current_test_native_confidence",
    "verify_exp263_root",
    "resolve_stage_c_artifact_root",
    "verify_stage_c_artifact_root",
    "write_json",
]
