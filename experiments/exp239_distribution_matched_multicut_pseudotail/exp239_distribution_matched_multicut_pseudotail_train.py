# %% [markdown]
# # exp239_distribution_matched_multicut_pseudotail train
#
# Deterministic CPU audit for distribution-matched early-start pseudo-tail
# cutoffs. This notebook creates manifests only: it does not train exp218,
# produce test predictions, or submit.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and hidden-like metadata checks
# 4. Official-start well metadata
# 5. Multi-source cutoff candidate generation
# 6. Distribution-matched deterministic selection
# 7. Fold-safe replay contract and leakage checks
# 8. Prefix materialization
# 9. Residual learnability probe
# 10. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp239_distribution_matched_multicut_pseudotail"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_PRIORITY = {
    "gr_missing_boundary": 0,
    "gr_change_point": 1,
    "trajectory_curvature": 2,
    "fixed_hidden_rows": 3,
    "prefix_eval_quantile": 4,
}


# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_allowed_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "This notebook must run on Kaggle first. Set EXPERIMENT_ALLOW_LOCAL=1 "
        "only for an explicitly approved local smoke run."
    )


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def find_named_file(name: str, preferred: Iterable[Path] = ()) -> Path:
    for path in preferred:
        if path.exists():
            return path
    roots = [Path.cwd()]
    if KAGGLE_INPUT_ROOT.exists():
        roots.append(KAGGLE_INPUT_ROOT)
    matches: list[Path] = []
    for root in roots:
        matches.extend(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name}")
    return sorted(set(matches), key=lambda path: (len(path.parts), str(path)))[0]


def find_train_dir(config: dict[str, Any]) -> Path:
    local = Path(str(nested(config, "data.train_dir", "data/raw/train")))
    if local.is_dir():
        return local
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    candidates: list[Path] = []
    if KAGGLE_INPUT_ROOT.exists():
        for path in KAGGLE_INPUT_ROOT.rglob("train"):
            if path.is_dir() and next(path.glob(pattern), None) is not None:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No raw train directory containing horizontal well CSVs was found")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def artifact_dir() -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = Path("experiments") / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    payload = "\x1f".join(map(str, parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def schema_sha(frame: pd.DataFrame) -> str:
    payload = "\n".join(f"{column}:{frame[column].dtype}" for column in frame.columns)
    return hashlib.sha256(payload.encode()).hexdigest()


# %% [markdown]
# ## 3. Input and hidden-like metadata checks


# %%
def load_hidden_like_roles(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filename = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
    configured = Path(str(nested(config, "data.hidden_like_fold_assignments_local", filename)))
    preferred = [configured, Path("inputs") / filename, Path.cwd() / "inputs" / filename]
    try:
        path = find_named_file(filename, preferred)
    except FileNotFoundError:
        empty = pd.DataFrame(
            columns=[
                "well_id",
                "verification_like_spatial_role",
                "verification_like_typewell_purged_role",
            ]
        )
        return empty, {"available": False, "path": None, "sha256": None}
    roles = pd.read_csv(path)
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    missing = sorted(required - set(roles.columns))
    if missing:
        raise ValueError(f"exp115 roles missing columns: {missing}")
    role_columns = [
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]
    roles = roles[role_columns].copy()
    roles["well_id"] = roles["well_id"].astype(str)
    if roles["well_id"].duplicated().any():
        raise ValueError("exp115 roles contain duplicate well_id")
    return roles, {"available": True, "path": str(path), "sha256": sha256_file(path)}


def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(values.shape, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    center = float(np.nanmedian(values[finite]))
    mad = float(np.nanmedian(np.abs(values[finite] - center)))
    scale = max(1.4826 * mad, 1e-12)
    result[finite] = np.abs(values[finite] - center) / scale
    return result


# %% [markdown]
# ## 4. Official-start well metadata


# %%
def official_metadata_for_well(
    path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    use_columns = [
        str(nested(config, "data.md_column", "MD")),
        *list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"])),
        str(nested(config, "data.gr_column", "GR")),
        str(nested(config, "data.target_column", "TVT")),
        str(nested(config, "data.input_target_column", "TVT_input")),
    ]
    frame = pd.read_csv(path, usecols=lambda column: column in set(use_columns))
    require_columns(frame, use_columns, path.name)
    tvt_input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    tvt_input = pd.to_numeric(frame[tvt_input_column], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if known.size == 0:
        raise ValueError(f"{path.name} has no known TVT_input prefix")
    official_cutoff = int(known[-1])
    expected = np.arange(official_cutoff + 1)
    if known.size != expected.size or not np.array_equal(known, expected):
        raise ValueError(f"{path.name} TVT_input known rows are not a contiguous prefix")
    n_rows = int(len(frame))
    eval_rows = n_rows - official_cutoff - 1
    if eval_rows <= 0:
        raise ValueError(f"{path.name} has no official evaluation tail")
    gr = pd.to_numeric(frame[gr_column], errors="coerce").to_numpy(float)
    coordinates = frame[list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"]))]
    coordinates = coordinates.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    delta = np.diff(coordinates, axis=0)
    path_length = float(np.nansum(np.linalg.norm(delta, axis=1)))
    chord = float(np.linalg.norm(coordinates[-1] - coordinates[0]))
    row = {
        "well_id": well_id_from_path(path),
        "source_path": str(path),
        "n_rows": n_rows,
        "official_cutoff_index": official_cutoff,
        "prefix_rows": official_cutoff + 1,
        "eval_rows": eval_rows,
        "prefix_fraction": (official_cutoff + 1) / n_rows,
        "trajectory_phase": official_cutoff / max(n_rows - 1, 1),
        "gr_missing_rate_prefix": float(np.mean(~np.isfinite(gr[: official_cutoff + 1]))),
        "gr_missing_rate_full": float(np.mean(~np.isfinite(gr))),
        "trajectory_tortuosity": path_length / max(chord, 1e-12),
    }
    return row, frame


def build_well_metadata(
    train_files: list[Path],
    roles: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(train_files, key=lambda item: well_id_from_path(item)):
        row, frame = official_metadata_for_well(path, config)
        rows.append(row)
        frames[str(row["well_id"])] = frame
    metadata = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    if not roles.empty:
        metadata = metadata.merge(roles, on="well_id", how="left", validate="1:1")
    for column in [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]:
        if column not in metadata:
            metadata[column] = "unknown"
        metadata[column] = metadata[column].fillna("unknown").astype(str)
    metadata["hidden_like_valid"] = metadata["verification_like_spatial_role"].eq(
        "valid"
    ) | metadata["verification_like_typewell_purged_role"].eq("valid")
    if metadata["well_id"].duplicated().any():
        raise ValueError("well metadata contains duplicate well_id")
    return metadata, frames


# %% [markdown]
# ## 5. Multi-source cutoff candidate generation


# %%
def separated_top_indices(
    scores: np.ndarray,
    feasible: np.ndarray,
    *,
    threshold: float,
    max_candidates: int,
    min_separation: int,
) -> list[int]:
    indices = np.flatnonzero(feasible & np.isfinite(scores) & (scores >= threshold))
    ordered = sorted(indices.tolist(), key=lambda index: (-float(scores[index]), int(index)))
    selected: list[int] = []
    for index in ordered:
        if all(abs(index - previous) >= min_separation for previous in selected):
            selected.append(index)
        if len(selected) >= max_candidates:
            break
    return sorted(selected)


def add_candidate(
    rows: list[dict[str, Any]],
    meta: pd.Series,
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    cutoff: int,
    source: str,
    source_value: float | int | str,
    event_score: float = 0.0,
) -> None:
    min_prefix = int(nested(config, "model.cutoff_generation.min_prefix_rows", 200))
    min_hidden = int(nested(config, "model.cutoff_generation.min_newly_hidden_rows", 50))
    official = int(meta["official_cutoff_index"])
    if cutoff < min_prefix - 1 or cutoff > official - min_hidden:
        return
    gr_column = str(nested(config, "data.gr_column", "GR"))
    gr = pd.to_numeric(frame[gr_column], errors="coerce").to_numpy(float)
    n_rows = int(meta["n_rows"])
    prefix = cutoff + 1
    max_rows = int(
        nested(config, "model.distribution_matching.max_estimated_rows_per_cutoff", 1000)
    )
    rows.append(
        {
            "well_id": str(meta["well_id"]),
            "cutoff_index": int(cutoff),
            "mask_start_index": int(cutoff + 1),
            "source": source,
            "source_value": source_value,
            "event_score": float(event_score),
            "official_cutoff_index": official,
            "newly_hidden_rows": official - cutoff,
            "prefix_rows": prefix,
            "eval_rows": n_rows - prefix,
            "prefix_fraction": prefix / n_rows,
            "trajectory_phase": cutoff / max(n_rows - 1, 1),
            "gr_missing_rate_prefix": float(np.mean(~np.isfinite(gr[:prefix]))),
            "estimated_augmented_rows": min(n_rows - prefix, max_rows),
            "hidden_like_valid": bool(meta["hidden_like_valid"]),
            "verification_like_spatial_role": str(meta["verification_like_spatial_role"]),
            "verification_like_typewell_purged_role": str(
                meta["verification_like_typewell_purged_role"]
            ),
        }
    )


def candidates_for_well(
    meta: pd.Series,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff_config = nested(config, "model.cutoff_generation", {})
    min_prefix = int(cutoff_config.get("min_prefix_rows", 200))
    min_hidden = int(cutoff_config.get("min_newly_hidden_rows", 50))
    official = int(meta["official_cutoff_index"])
    min_cutoff = min_prefix - 1
    max_cutoff = official - min_hidden
    if max_cutoff < min_cutoff:
        return rows

    for quantile in cutoff_config.get("prefix_eval_quantiles", []):
        cutoff = int(round(min_cutoff + float(quantile) * (max_cutoff - min_cutoff)))
        add_candidate(
            rows,
            meta,
            frame,
            config,
            cutoff=cutoff,
            source="prefix_eval_quantile",
            source_value=float(quantile),
        )
    for hidden_rows in cutoff_config.get("fixed_hidden_rows", []):
        add_candidate(
            rows,
            meta,
            frame,
            config,
            cutoff=official - int(hidden_rows),
            source="fixed_hidden_rows",
            source_value=int(hidden_rows),
        )

    feasible = np.zeros(len(frame), dtype=bool)
    feasible[min_cutoff : max_cutoff + 1] = True
    gr_column = str(nested(config, "data.gr_column", "GR"))
    gr = pd.to_numeric(frame[gr_column], errors="coerce").to_numpy(float)

    gr_change_config = cutoff_config.get("gr_change", {})
    if bool(gr_change_config.get("enabled", True)):
        differences = np.full(len(frame), np.nan)
        finite_pairs = np.isfinite(gr[1:]) & np.isfinite(gr[:-1])
        min_finite_fraction = float(gr_change_config.get("min_finite_pair_fraction", 0.25))
        if float(np.mean(finite_pairs)) >= min_finite_fraction:
            pair_indices = np.flatnonzero(finite_pairs) + 1
            differences[pair_indices] = np.abs(gr[pair_indices] - gr[pair_indices - 1])
            scores = robust_z(differences)
            event_feasible = np.roll(feasible, 1)
            event_feasible[0] = False
            for event_index in separated_top_indices(
                scores,
                event_feasible,
                threshold=float(gr_change_config.get("robust_z_threshold", 4.0)),
                max_candidates=int(gr_change_config.get("max_candidates_per_well", 4)),
                min_separation=int(gr_change_config.get("min_separation_rows", 32)),
            ):
                add_candidate(
                    rows,
                    meta,
                    frame,
                    config,
                    cutoff=event_index - 1,
                    source="gr_change_point",
                    source_value=event_index,
                    event_score=float(scores[event_index]),
                )

    missing_config = cutoff_config.get("gr_missing_blocks", {})
    if bool(missing_config.get("enabled", True)):
        missing = ~np.isfinite(gr)
        boundaries = np.flatnonzero(missing[1:] != missing[:-1]) + 1
        runs: list[tuple[int, int]] = []
        start = 0
        for boundary in [*boundaries.tolist(), len(missing)]:
            runs.append((start, int(boundary)))
            start = int(boundary)
        min_block = int(missing_config.get("min_block_rows", 8))
        valid_boundaries: list[int] = []
        for left, right in zip(runs[:-1], runs[1:], strict=True):
            boundary = left[1]
            if min(left[1] - left[0], right[1] - right[0]) >= min_block:
                valid_boundaries.append(boundary)
        selected_boundaries: list[int] = []
        min_separation = int(missing_config.get("min_separation_rows", 16))
        for boundary in valid_boundaries:
            cutoff = boundary - 1
            if not (min_cutoff <= cutoff <= max_cutoff):
                continue
            if all(abs(boundary - old) >= min_separation for old in selected_boundaries):
                selected_boundaries.append(boundary)
            if len(selected_boundaries) >= int(missing_config.get("max_candidates_per_well", 4)):
                break
        for boundary in selected_boundaries:
            add_candidate(
                rows,
                meta,
                frame,
                config,
                cutoff=boundary - 1,
                source="gr_missing_boundary",
                source_value=boundary,
            )

    curvature_config = cutoff_config.get("trajectory_curvature", {})
    if bool(curvature_config.get("enabled", True)):
        coordinate_columns = list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"]))
        xyz = frame[coordinate_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        steps = np.diff(xyz, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        units = np.divide(
            steps,
            norms[:, None],
            out=np.zeros_like(steps),
            where=np.isfinite(norms[:, None]) & (norms[:, None] > 1e-12),
        )
        curvature = np.full(len(frame), np.nan)
        if len(frame) >= 3:
            curvature[2:] = np.linalg.norm(np.diff(units, axis=0), axis=1)
        scores = robust_z(curvature)
        event_feasible = np.roll(feasible, 1)
        event_feasible[0] = False
        for event_index in separated_top_indices(
            scores,
            event_feasible,
            threshold=float(curvature_config.get("robust_z_threshold", 4.0)),
            max_candidates=int(curvature_config.get("max_candidates_per_well", 4)),
            min_separation=int(curvature_config.get("min_separation_rows", 32)),
        ):
            add_candidate(
                rows,
                meta,
                frame,
                config,
                cutoff=event_index - 1,
                source="trajectory_curvature",
                source_value=event_index,
                event_score=float(scores[event_index]),
            )

    return rows


def build_cutoff_candidates(
    metadata: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for meta in metadata.sort_values("well_id").itertuples(index=False):
        series = pd.Series(meta._asdict())
        rows.extend(candidates_for_well(series, frames[str(meta.well_id)], config))
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No feasible early-start cutoff candidates were generated")
    candidates["source_priority"] = candidates["source"].map(SOURCE_PRIORITY).fillna(99)
    candidates["tie_key"] = [
        stable_key(well, cutoff, source, value)
        for well, cutoff, source, value in zip(
            candidates["well_id"],
            candidates["cutoff_index"],
            candidates["source"],
            candidates["source_value"],
            strict=True,
        )
    ]
    candidates = candidates.sort_values(["well_id", "cutoff_index", "source_priority", "tie_key"])
    aliases = (
        candidates.groupby(["well_id", "cutoff_index"], sort=False)["source"]
        .agg(lambda values: "|".join(dict.fromkeys(map(str, values))))
        .rename("source_aliases")
    )
    candidates = candidates.drop_duplicates(["well_id", "cutoff_index"], keep="first")
    candidates = candidates.merge(
        aliases.reset_index(), on=["well_id", "cutoff_index"], how="left", validate="1:1"
    )
    fixed_quantiles = set(
        float(value) for value in nested(config, "model.historical_exp023.fixed_quantiles", [])
    )
    candidates["exp023_fixed_cutoff"] = candidates["source"].eq(
        "prefix_eval_quantile"
    ) & candidates["source_value"].astype(float).round(8).isin(
        {round(value, 8) for value in fixed_quantiles}
    )
    return candidates.sort_values(["well_id", "cutoff_index"]).reset_index(drop=True)


# %% [markdown]
# ## 6. Distribution-matched deterministic selection


# %%
def quantile_bin_edges(values: pd.Series, quantiles: list[float]) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    interior = np.unique(np.quantile(finite, quantiles))
    return np.concatenate(([-np.inf], interior, [np.inf]))


def apply_bins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    labels = np.searchsorted(edges[1:-1], numeric, side="right")
    labels[~np.isfinite(numeric)] = len(edges) - 1
    return labels.astype(int)


def prepare_matching_bins(
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]]]:
    target = metadata.copy()
    candidate_frame = candidates.copy()
    specs = nested(config, "model.distribution_matching.features", {})
    edges_record: dict[str, list[float]] = {}
    for feature, quantiles in specs.items():
        edges = quantile_bin_edges(target[feature], list(map(float, quantiles)))
        target[f"bin__{feature}"] = apply_bins(target[feature], edges)
        candidate_frame[f"bin__{feature}"] = apply_bins(candidate_frame[feature], edges)
        edges_record[feature] = [float(value) for value in edges]
    return target, candidate_frame, edges_record


def select_distribution_matched_cutoffs(
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[float]]]:
    target, pool, edges = prepare_matching_bins(metadata, candidates, config)
    matching = nested(config, "model.distribution_matching", {})
    features = list(matching.get("features", {}).keys())
    max_per_well = int(matching.get("max_cutoffs_per_well", 3))
    max_per_source_well = int(matching.get("max_cutoffs_per_source_per_well", 1))
    strategy = str(matching.get("strategy", "global_marginal_deficit_quota"))
    if strategy != "global_marginal_deficit_quota":
        raise ValueError(f"Unsupported distribution matching strategy: {strategy}")
    configured_target = int(matching.get("target_total_cutoffs", len(metadata) * 2))
    max_total = int(matching.get("max_total_cutoffs", configured_target))
    target_total = min(max_total, configured_target)
    max_ratio = float(matching.get("max_estimated_augmentation_ratio", 0.45))
    row_budget = int(float(metadata["eval_rows"].sum()) * max_ratio)
    source_share = {
        str(key): float(value) for key, value in matching.get("source_target_share", {}).items()
    }
    weights = matching.get("score_weights", {})
    marginal_weight = float(weights.get("marginal_deficit", 1.0))
    source_weight = float(weights.get("source_deficit", 0.35))
    hidden_bonus = float(weights.get("hidden_like_bonus", 0.05))
    coverage_bonus = float(weights.get("new_well_coverage_bonus", 0.15))

    desired_bins: dict[tuple[str, int], float] = {}
    selected_bins: Counter[tuple[str, int]] = Counter()
    for feature in features:
        column = f"bin__{feature}"
        proportions = target[column].value_counts(normalize=True)
        for bin_id, proportion in proportions.items():
            desired_bins[(feature, int(bin_id))] = float(proportion) * target_total
    desired_sources = {source: share * target_total for source, share in source_share.items()}
    selected_sources: Counter[str] = Counter()
    selected_wells: Counter[str] = Counter()
    selected_well_sources: Counter[tuple[str, str]] = Counter()
    selected_indices: list[int] = []
    selected_rows = 0

    pool = pool.sort_values("tie_key").reset_index(drop=True)
    well_names = sorted(pool["well_id"].astype(str).unique())
    source_names = sorted(pool["source"].astype(str).unique())
    well_to_index = {well: index for index, well in enumerate(well_names)}
    source_to_index = {source: index for index, source in enumerate(source_names)}
    well_index = pool["well_id"].astype(str).map(well_to_index).to_numpy(int)
    source_index = pool["source"].astype(str).map(source_to_index).to_numpy(int)
    well_source_index = well_index * len(source_names) + source_index
    estimated_rows_array = pool["estimated_augmented_rows"].to_numpy(int)
    hidden_like_array = pool["hidden_like_valid"].to_numpy(bool)
    feature_bins = {feature: pool[f"bin__{feature}"].to_numpy(int) for feature in features}
    active = np.ones(len(pool), dtype=bool)
    well_counts = np.zeros(len(well_names), dtype=int)
    well_source_counts = np.zeros(len(well_names) * len(source_names), dtype=int)
    source_counts_array = np.zeros(len(source_names), dtype=int)

    for _step in range(target_total):
        eligible = (
            active
            & (well_counts[well_index] < max_per_well)
            & (well_source_counts[well_source_index] < max_per_source_well)
            & (selected_rows + estimated_rows_array <= row_budget)
        )
        if not eligible.any():
            break
        scores = np.full(len(pool), -np.inf, dtype=float)
        scores[eligible] = 0.0
        for feature in features:
            bins = feature_bins[feature]
            unique_bins = np.unique(bins)
            bin_scores = np.zeros(int(unique_bins.max()) + 1, dtype=float)
            for bin_id in unique_bins:
                desired = desired_bins.get((feature, int(bin_id)), 0.0)
                if desired > 0:
                    deficit = max(desired - selected_bins[(feature, int(bin_id))], 0.0)
                    bin_scores[int(bin_id)] = marginal_weight * deficit / desired
            scores[eligible] += bin_scores[bins[eligible]]
        source_score_array = np.zeros(len(source_names), dtype=float)
        for source, source_id in source_to_index.items():
            desired = desired_sources.get(source, 0.0)
            if desired > 0:
                deficit = max(desired - source_counts_array[source_id], 0.0)
                source_score_array[source_id] = source_weight * deficit / desired
        scores[eligible] += source_score_array[source_index[eligible]]
        scores[eligible & hidden_like_array] += hidden_bonus
        scores[eligible & (well_counts[well_index] == 0)] += coverage_bonus

        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            break
        row = pool.iloc[index]
        well = str(row["well_id"])
        source = str(row["source"])
        active[index] = False
        selected_indices.append(index)
        selected_rows += int(row["estimated_augmented_rows"])
        selected_wells[well] += 1
        selected_well_sources[(well, source)] += 1
        selected_sources[source] += 1
        well_counts[well_index[index]] += 1
        well_source_counts[well_source_index[index]] += 1
        source_counts_array[source_index[index]] += 1
        for feature in features:
            selected_bins[(feature, int(row[f"bin__{feature}"]))] += 1

    selected = pool.iloc[selected_indices].copy().reset_index(drop=True)
    selected["selection_order"] = np.arange(len(selected), dtype=int)
    selected["estimated_augmentation_ratio"] = selected_rows / max(
        int(metadata["eval_rows"].sum()), 1
    )
    hidden_like_wells = int(metadata["hidden_like_valid"].sum())
    selected_hidden_like_wells = int(
        selected.loc[selected["hidden_like_valid"], "well_id"].nunique()
    )
    if selected.groupby("well_id").size().max() > max_per_well:
        raise AssertionError("max_cutoffs_per_well was violated")
    if selected.groupby(["well_id", "source"]).size().max() > max_per_source_well:
        raise AssertionError("max_cutoffs_per_source_per_well was violated")
    summary = {
        "target_total_cutoffs": target_total,
        "strategy": strategy,
        "selected_total_cutoffs": int(len(selected)),
        "selected_wells": int(selected["well_id"].nunique()),
        "estimated_augmented_rows": int(selected_rows),
        "row_budget": int(row_budget),
        "estimated_augmentation_ratio": selected_rows / max(int(metadata["eval_rows"].sum()), 1),
        "selected_well_fraction": selected["well_id"].nunique() / max(len(metadata), 1),
        "selected_hidden_like_wells": selected_hidden_like_wells,
        "selected_hidden_like_well_fraction": selected_hidden_like_wells
        / max(hidden_like_wells, 1),
        "source_counts": dict(sorted(selected_sources.items())),
        "desired_source_counts": desired_sources,
    }
    return selected, summary, edges


def evaluate_distribution_guard(
    report: pd.DataFrame,
    selection_summary: dict[str, Any],
    leakage: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    guard = nested(config, "model.distribution_matching.guard", {})
    max_delta = float(report["selected_minus_target_share"].abs().max())
    selected_well_fraction = float(selection_summary["selected_well_fraction"])
    augmentation_ratio = float(selection_summary["estimated_augmentation_ratio"])
    hidden_like_fraction = float(selection_summary["selected_hidden_like_well_fraction"])
    leakage_pass = all(bool(value) for value in leakage.values())
    checks = {
        "marginal_share_delta": max_delta <= float(guard.get("max_abs_marginal_share_delta", 0.05)),
        "selected_well_fraction": selected_well_fraction
        >= float(guard.get("min_selected_well_fraction", 0.65)),
        "hidden_like_selected_well_fraction": hidden_like_fraction
        >= float(guard.get("min_hidden_like_selected_well_fraction", 0.90)),
        "augmentation_ratio": augmentation_ratio
        <= float(guard.get("max_estimated_augmentation_ratio", 0.45)),
        "leakage": leakage_pass if bool(guard.get("require_all_leakage_checks", True)) else True,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "max_abs_marginal_share_delta": max_delta,
            "selected_well_fraction": selected_well_fraction,
            "selected_hidden_like_well_fraction": hidden_like_fraction,
            "estimated_augmentation_ratio": augmentation_ratio,
            "leakage_pass": leakage_pass,
        },
        "thresholds": guard,
    }


def distribution_report(
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    target, pool, edges = prepare_matching_bins(metadata, candidates, config)
    selected_binned = selected.copy()
    for feature, edge_values in edges.items():
        selected_binned[f"bin__{feature}"] = apply_bins(
            selected_binned[feature], np.asarray(edge_values, dtype=float)
        )
    rows: list[dict[str, Any]] = []
    for feature in nested(config, "model.distribution_matching.features", {}).keys():
        column = f"bin__{feature}"
        all_bins = sorted(
            set(target[column].unique())
            | set(pool[column].unique())
            | set(selected_binned[column].unique())
        )
        for bin_id in all_bins:
            record: dict[str, Any] = {"feature": feature, "bin": int(bin_id)}
            for name, frame in [
                ("official_target", target),
                ("candidate_pool", pool),
                ("selected", selected_binned),
            ]:
                count = int(frame[column].eq(bin_id).sum())
                record[f"{name}_count"] = count
                record[f"{name}_share"] = count / max(len(frame), 1)
            record["selected_minus_target_share"] = (
                record["selected_share"] - record["official_target_share"]
            )
            rows.append(record)
    return pd.DataFrame(rows)


# %% [markdown]
# ## 7. Fold-safe replay contract and leakage checks


# %%
def build_fold_manifest(metadata: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    wells = metadata[["well_id"]].sort_values("well_id").reset_index(drop=True)
    n_folds = int(nested(config, "validation.n_folds", 5))
    splitter = GroupKFold(n_splits=n_folds)
    fold = np.full(len(wells), -1, dtype=int)
    dummy = np.zeros((len(wells), 1), dtype=np.float32)
    groups = wells["well_id"].to_numpy()
    for fold_id, (_, valid_index) in enumerate(splitter.split(dummy, groups=groups)):
        fold[valid_index] = fold_id
    if np.any(fold < 0):
        raise AssertionError("Some wells were not assigned a validation fold")
    wells["fold"] = fold
    wells["fold_key"] = [
        stable_key("fold", well, value) for well, value in zip(wells.well_id, fold, strict=True)
    ]
    return wells


def build_replay_requests(
    selected: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    meta_columns = ["well_id", "source_path", "n_rows", "official_cutoff_index"]
    replay = selected.merge(
        metadata[meta_columns], on=["well_id", "official_cutoff_index"], validate="many_to_one"
    )
    replay = replay.merge(fold_manifest[["well_id", "fold"]], on="well_id", validate="many_to_one")
    contract = nested(config, "model.replay_contract", {})
    replay["request_id"] = [
        stable_key(EXPERIMENT_NAME, well, cutoff, source)
        for well, cutoff, source in zip(
            replay.well_id,
            replay.cutoff_index,
            replay.source,
            strict=True,
        )
    ]
    replay["source_well"] = replay["well_id"]
    replay["mask_columns_after_cutoff"] = "|".join(contract.get("mask_columns_after_cutoff", []))
    replay["target_only_columns_after_cutoff"] = "|".join(
        contract.get("target_only_columns_after_cutoff", [])
    )
    replay["regenerate_feature_groups"] = "|".join(contract.get("regenerate_feature_groups", []))
    replay["forbid_full_prefix_cache_slice"] = bool(
        contract.get("forbid_full_prefix_cache_slice", True)
    )
    replay["feature_generation_may_read_tail_tvt"] = False
    replay["target_usage"] = "outer_train_only"
    replay["validation_surface"] = "official_start_only"
    return replay.sort_values(["fold", "well_id", "cutoff_index"]).reset_index(drop=True)


def assert_leakage_contract(
    replay: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if replay["request_id"].duplicated().any():
        raise AssertionError("Duplicate replay request_id")
    if replay.groupby("source_well")["fold"].nunique().max() != 1:
        raise AssertionError("A source well was assigned to multiple folds")
    expected_fold = fold_manifest.set_index("well_id")["fold"]
    actual = replay.set_index("source_well")["fold"]
    aligned = expected_fold.reindex(actual.index).to_numpy()
    if not np.array_equal(aligned.astype(int), actual.to_numpy(dtype=int)):
        raise AssertionError("Replay fold does not match source-well fold")
    min_hidden = int(nested(config, "model.cutoff_generation.min_newly_hidden_rows", 50))
    if int(replay["newly_hidden_rows"].min()) < min_hidden:
        raise AssertionError("A selected cutoff violates min_newly_hidden_rows")
    if not replay["forbid_full_prefix_cache_slice"].all():
        raise AssertionError("Replay contract allows forbidden full-prefix cache slicing")
    if replay["feature_generation_may_read_tail_tvt"].any():
        raise AssertionError("Replay contract allows tail TVT feature leakage")
    return {
        "request_ids_unique": True,
        "one_fold_per_source_well": True,
        "source_fold_alignment": True,
        "min_newly_hidden_rows": int(replay["newly_hidden_rows"].min()),
        "full_prefix_cache_slice_forbidden": True,
        "tail_tvt_feature_read_forbidden": True,
    }


# %% [markdown]
# ## 8. Prefix materialization


# %%
def evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if count <= 0 or indices.size == 0:
        return np.empty(0, dtype=int)
    if indices.size <= count:
        return indices
    positions = np.linspace(0, indices.size - 1, num=count, dtype=int)
    return indices[np.unique(positions)]


def sampled_tail_rows(
    n_rows: int,
    cutoff_index: int,
    materialization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    tail = np.arange(cutoff_index + 1, n_rows, dtype=int)
    steps = tail - cutoff_index - 1
    selected: list[np.ndarray] = []
    labels: dict[int, str] = {}
    for bucket in materialization.get("distance_buckets", []):
        lower = int(bucket["min_step"])
        upper_value = bucket.get("max_step")
        upper = int(upper_value) if upper_value is not None else np.iinfo(np.int64).max
        eligible = tail[(steps >= lower) & (steps <= upper)]
        chosen = evenly_spaced_indices(eligible, int(bucket.get("quota", 0)))
        selected.append(chosen)
        labels.update({int(index): str(bucket["name"]) for index in chosen})
    chosen_all = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=int)
    cap = int(materialization.get("max_rows_per_request", 1000))
    if bool(materialization.get("fill_remaining", True)) and chosen_all.size < cap:
        remaining = np.setdiff1d(tail, chosen_all, assume_unique=True)
        fill = evenly_spaced_indices(remaining, cap - chosen_all.size)
        chosen_all = np.unique(np.concatenate([chosen_all, fill]))
        labels.update({int(index): "fill_remaining" for index in fill})
    chosen_all = chosen_all[:cap]
    return chosen_all, np.asarray([labels[int(index)] for index in chosen_all], dtype=object)


def finite_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "mad": np.nan, "last": np.nan}
    median = float(np.median(finite))
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": median,
        "mad": float(np.median(np.abs(finite - median))),
        "last": float(finite[-1]),
    }


def materialize_prefix_features(
    replay: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = nested(config, "model.materialization", {})
    target_source = str(nested(config, "data.target_column", "TVT"))
    target_output = str(spec.get("target_column", "target_tvt"))
    forbidden = set(map(str, spec.get("forbidden_feature_columns", ["TVT", target_output])))
    md_column = str(nested(config, "data.md_column", "MD"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    coordinates = list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"]))
    feature_parts: list[pd.DataFrame] = []
    request_rows: list[dict[str, Any]] = []
    for request in replay.sort_values("request_id").itertuples(index=False):
        frame = frames[str(request.source_well)]
        target = pd.to_numeric(frame[target_source], errors="coerce").to_numpy(float)
        feature_source = frame.drop(columns=[target_source])
        cutoff = int(request.cutoff_index)
        indices, bucket_names = sampled_tail_rows(len(frame), cutoff, spec)
        prefix = feature_source.iloc[: cutoff + 1]
        anchor = feature_source.iloc[cutoff]
        gr_prefix = pd.to_numeric(prefix[gr_column], errors="coerce").to_numpy(float)
        gr_stats = finite_summary(gr_prefix)
        constants: dict[str, Any] = {
            "request_id": request.request_id,
            "source_well": request.source_well,
            "fold": int(request.fold),
            "cutoff_index": cutoff,
            "cutoff_source": request.source,
            "prefix_rows": cutoff + 1,
            "prefix_fraction": (cutoff + 1) / len(frame),
            "newly_hidden_rows": int(request.newly_hidden_rows),
            "prefix_gr_missing_rate": float(np.mean(~np.isfinite(gr_prefix))),
            **{f"prefix_gr_{key}": value for key, value in gr_stats.items()},
        }
        for column in [
            md_column,
            *coordinates,
            gr_column,
            str(nested(config, "data.input_target_column", "TVT_input")),
        ]:
            constants[f"anchor_{column.lower()}"] = pd.to_numeric(
                pd.Series([anchor[column]]), errors="coerce"
            ).iloc[0]
        for window in map(int, spec.get("recent_windows", [32, 128])):
            recent = gr_prefix[-window:]
            stats = finite_summary(recent)
            constants.update({f"prefix_gr_w{window}_{key}": value for key, value in stats.items()})
            constants[f"prefix_gr_w{window}_missing_rate"] = float(np.mean(~np.isfinite(recent)))
        rows = feature_source.iloc[indices]
        out = pd.DataFrame(
            {key: np.repeat(value, len(indices)) for key, value in constants.items()}
        )
        out["row_index"] = indices
        out["eval_step"] = indices - cutoff - 1
        out["distance_bucket"] = bucket_names
        for column in [md_column, *coordinates, gr_column]:
            out[f"row_{column.lower()}"] = pd.to_numeric(rows[column], errors="coerce").to_numpy(
                float
            )
            out[f"delta_{column.lower()}"] = out[f"row_{column.lower()}"] - float(
                constants[f"anchor_{column.lower()}"]
            )
        out["row_gr_missing"] = ~np.isfinite(out["row_gr"])
        out["delta_xy"] = np.hypot(out["delta_x"], out["delta_y"])
        out["delta_xyz"] = np.sqrt(out["delta_xy"] ** 2 + out["delta_z"] ** 2)
        out["dz_dmd"] = out["delta_z"] / out["delta_md"].replace(0.0, np.nan)
        if forbidden.intersection(out.columns):
            raise AssertionError("A forbidden target column entered the feature builder")
        out[target_output] = target[indices]
        feature_parts.append(out)
        request_rows.append(
            {
                "request_id": request.request_id,
                "source_well": request.source_well,
                "fold": int(request.fold),
                "cutoff_index": cutoff,
                "available_tail_rows": len(frame) - cutoff - 1,
                "materialized_rows": len(indices),
            }
        )
    materialized = pd.concat(feature_parts, ignore_index=True)
    request_summary = pd.DataFrame(request_rows).sort_values("request_id").reset_index(drop=True)
    feature_columns = [column for column in materialized.columns if column != target_output]
    schema = pd.DataFrame(
        {
            "column": materialized.columns,
            "dtype": [str(materialized[column].dtype) for column in materialized.columns],
            "role": [
                "target" if column == target_output else "feature_or_metadata"
                for column in materialized.columns
            ],
        }
    )
    checks = {
        "all_requests_materialized": len(request_summary) == len(replay),
        "request_ids_unique": not request_summary["request_id"].duplicated().any(),
        "row_cap_respected": int(request_summary["materialized_rows"].max())
        <= int(spec.get("max_rows_per_request", 1000)),
        "fold_inheritance": materialized.groupby("source_well")["fold"].nunique().max() == 1,
        "target_excluded_from_features": not bool(forbidden.intersection(feature_columns)),
        "target_finite": bool(np.isfinite(materialized[target_output]).all()),
    }
    audit = {
        "enabled": True,
        "pass": bool(all(checks.values())),
        "checks": checks,
        "requests": int(len(request_summary)),
        "rows": int(len(materialized)),
        "max_rows_per_request": int(request_summary["materialized_rows"].max()),
        "memory_bytes": int(materialized.memory_usage(index=True, deep=True).sum()),
        "feature_schema_sha256": schema_sha(materialized[feature_columns]),
    }
    return materialized, request_summary, schema, audit


# %% [markdown]
# ## 9. Residual learnability probe


# %%
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def run_residual_probe(
    materialized: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import lightgbm as lgb

    spec = nested(config, "model.residual_probe", {})
    target_column = str(spec.get("target_column", "target_tvt"))
    anchor_column = str(spec.get("anchor_column", "anchor_tvt_input"))
    exclude = set(map(str, spec.get("exclude_columns", [])))
    numeric = materialized.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    features = [column for column in numeric if column not in exclude]
    if target_column in features or anchor_column not in features:
        raise AssertionError("Residual probe feature/target contract is invalid")
    target = materialized[target_column].to_numpy(float)
    anchor = materialized[anchor_column].to_numpy(float)
    residual = target - anchor
    oof_residual = np.full(len(materialized), np.nan, dtype=np.float64)
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    n_folds = int(spec.get("n_folds", 5))
    for fold in range(n_folds):
        valid = materialized["fold"].to_numpy(int) == fold
        train = ~valid
        if not valid.any() or not train.any():
            raise AssertionError(f"Residual probe fold {fold} is empty")
        train_set = lgb.Dataset(
            materialized.loc[train, features], label=residual[train], free_raw_data=True
        )
        valid_set = lgb.Dataset(
            materialized.loc[valid, features], label=residual[valid], reference=train_set
        )
        model = lgb.train(
            dict(spec.get("lightgbm", {})),
            train_set,
            num_boost_round=int(spec.get("num_boost_round", 500)),
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(int(spec.get("early_stopping_rounds", 50)), verbose=False)
            ],
        )
        prediction = model.predict(
            materialized.loc[valid, features], num_iteration=model.best_iteration
        )
        oof_residual[valid] = prediction
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "valid_wells": int(materialized.loc[valid, "source_well"].nunique()),
                "best_iteration": int(model.best_iteration),
                "rmse_tvt": rmse(target[valid], anchor[valid] + prediction),
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "feature": feature,
                "gain": float(gain),
            }
            for feature, gain in zip(
                features, model.feature_importance(importance_type="gain"), strict=True
            )
        )
    if not np.isfinite(oof_residual).all():
        raise AssertionError("Residual probe OOF coverage is incomplete")
    prediction = anchor + oof_residual
    baseline_anchor = anchor
    baseline_delta_z = anchor + materialized["delta_z"].to_numpy(float)
    oof = materialized[
        ["request_id", "source_well", "fold", "row_index", "eval_step", "distance_bucket"]
    ].copy()
    oof[target_column] = target
    oof["prediction_anchor_hold"] = baseline_anchor
    oof["prediction_anchor_plus_delta_z"] = baseline_delta_z
    oof["prediction_residual_probe"] = prediction
    oof["residual_target"] = residual
    oof["residual_oof_prediction"] = oof_residual
    metric_rows: list[dict[str, Any]] = []
    prediction_columns = {
        "anchor_hold": baseline_anchor,
        "anchor_plus_delta_z": baseline_delta_z,
        "residual_probe": prediction,
    }
    surfaces: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(oof), dtype=bool))]
    surfaces.extend(
        (f"distance__{bucket}", oof["distance_bucket"].eq(bucket).to_numpy())
        for bucket in sorted(oof["distance_bucket"].unique())
    )
    for surface, mask in surfaces:
        for model_name, values in prediction_columns.items():
            metric_rows.append(
                {
                    "surface": surface,
                    "model": model_name,
                    "rows": int(mask.sum()),
                    "rmse_tvt": rmse(target[mask], values[mask]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    by_well_rows: list[dict[str, Any]] = []
    for well, indices in oof.groupby("source_well", sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        record: dict[str, Any] = {"source_well": well, "rows": len(index)}
        for model_name, values in prediction_columns.items():
            record[f"rmse__{model_name}"] = rmse(target[index], values[index])
        record["delta_vs_best_baseline"] = record["rmse__residual_probe"] - min(
            record["rmse__anchor_hold"], record["rmse__anchor_plus_delta_z"]
        )
        by_well_rows.append(record)
    by_well = pd.DataFrame(by_well_rows)
    importance = (
        pd.DataFrame(importance_rows)
        .groupby("feature", as_index=False)
        .agg(mean_gain=("gain", "mean"), folds=("fold", "nunique"))
        .sort_values("mean_gain", ascending=False)
    )
    overall = metrics[metrics["surface"].eq("overall")].set_index("model")["rmse_tvt"]
    best_baseline = float(overall[["anchor_hold", "anchor_plus_delta_z"]].min())
    probe_rmse = float(overall["residual_probe"])
    guard_spec = spec.get("guard", {})
    checks = {
        "oof_coverage": bool(np.isfinite(oof_residual).all()),
        "one_fold_per_well": materialized.groupby("source_well")["fold"].nunique().max() == 1,
        "overall_improvement_vs_best_baseline": probe_rmse < best_baseline
        if bool(guard_spec.get("require_overall_improvement_vs_best_baseline", True))
        else True,
        "max_well_regression": float(by_well["delta_vs_best_baseline"].max())
        <= float(guard_spec.get("max_well_rmse_regression", 20.0)),
    }
    summary = {
        "enabled": True,
        "pass": bool(all(checks.values())),
        "checks": checks,
        "features": len(features),
        "feature_columns": features,
        "folds": fold_rows,
        "overall_rmse": {str(key): float(value) for key, value in overall.items()},
        "delta_vs_best_baseline": probe_rmse - best_baseline,
        "improved_wells": int((by_well["delta_vs_best_baseline"] < 0).sum()),
        "worsened_wells": int((by_well["delta_vs_best_baseline"] > 0).sum()),
        "max_well_regression": float(by_well["delta_vs_best_baseline"].max()),
    }
    return oof, metrics, by_well, importance, summary


# %% [markdown]
# ## 10. Metrics, diagnostics, and generated artifacts


# %%
def save_frame(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> dict[str, Any]:
    if gzip:
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        with gzip_module_open(path) as fp:
            decompressed = fp.read()
        content_sha = hashlib.sha256(decompressed).hexdigest()
    else:
        frame.to_csv(path, index=False, lineterminator="\n")
        content_sha = sha256_file(path)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "file_sha256": sha256_file(path),
        "content_sha256": content_sha,
        "schema_sha256": schema_sha(frame),
    }


def gzip_module_open(path: Path):
    return gzip.open(path, "rb")


def source_summary(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    candidate_counts = candidates.groupby("source").agg(
        candidate_rows=("cutoff_index", "size"),
        candidate_wells=("well_id", "nunique"),
    )
    selected_counts = selected.groupby("source").agg(
        selected_rows=("cutoff_index", "size"),
        selected_wells=("well_id", "nunique"),
        estimated_augmented_rows=("estimated_augmented_rows", "sum"),
    )
    return candidate_counts.join(selected_counts, how="outer").fillna(0).reset_index()


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    require_allowed_runtime()
    train_dir = find_train_dir(config)
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    train_files = sorted(train_dir.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"No {pattern} files found in {train_dir}")
    roles, roles_meta = load_hidden_like_roles(config)
    print(f"train_dir={train_dir}")
    print(f"horizontal wells={len(train_files)}")
    print(f"hidden-like roles available={roles_meta['available']}")

    metadata, frames = build_well_metadata(train_files, roles, config)
    candidates = build_cutoff_candidates(metadata, frames, config)
    fold_manifest = build_fold_manifest(metadata, config)
    selected, selection_summary, bin_edges = select_distribution_matched_cutoffs(
        metadata, candidates, config
    )
    report = distribution_report(metadata, candidates, selected, config)
    replay = build_replay_requests(selected, metadata, fold_manifest, config)
    leakage = assert_leakage_contract(replay, fold_manifest, config)
    distribution_guard = evaluate_distribution_guard(report, selection_summary, leakage, config)
    sources = source_summary(candidates, selected)

    materialization_audit: dict[str, Any] = {"enabled": False, "pass": True}
    materialized = request_summary = materialization_schema = None
    if bool(nested(config, "model.materialization.enabled", False)):
        if not distribution_guard["pass"]:
            raise RuntimeError("Prefix materialization requires a passing distribution guard")
        materialized, request_summary, materialization_schema, materialization_audit = (
            materialize_prefix_features(replay, frames, config)
        )
        if not materialization_audit["pass"]:
            raise AssertionError("Prefix materialization audit failed")

    residual_probe: dict[str, Any] = {"enabled": False, "pass": True}
    probe_oof = probe_metrics = probe_by_well = probe_importance = None
    if bool(nested(config, "model.residual_probe.enabled", False)):
        if materialized is None:
            raise RuntimeError("Residual probe requires materialized prefix features")
        probe_oof, probe_metrics, probe_by_well, probe_importance, residual_probe = (
            run_residual_probe(materialized, config)
        )

    augmentation_summary: dict[str, Any] = {"enabled": False}
    if bool(nested(config, "model.exp218_augmentation.enabled", False)):
        if materialized is None:
            raise RuntimeError("exp218 augmentation requires materialized pseudo-tail rows")
        from exp239_exp218_pseudotail_augmentation import run_full_augmentation_evaluation

        augmentation_summary = {
            "enabled": True,
            **run_full_augmentation_evaluation(
                replay=replay,
                materialized=materialized,
                frames=frames,
                raw_train_dir=train_dir,
                config=config,
                output_dir=artifact_dir(),
            ),
        }

    output = artifact_dir()
    artifacts = {
        "well_metadata": save_frame(metadata, output / f"{OUTPUT_PREFIX}_well_metadata.csv"),
        "fold_manifest": save_frame(fold_manifest, output / f"{OUTPUT_PREFIX}_fold_manifest.csv"),
        "cutoff_candidates": save_frame(
            candidates,
            output / f"{OUTPUT_PREFIX}_cutoff_candidates.csv.gz",
            gzip=True,
        ),
        "selected_cutoffs": save_frame(selected, output / f"{OUTPUT_PREFIX}_selected_cutoffs.csv"),
        "distribution_report": save_frame(
            report, output / f"{OUTPUT_PREFIX}_distribution_report.csv"
        ),
        "source_summary": save_frame(sources, output / f"{OUTPUT_PREFIX}_source_summary.csv"),
        "prefix_replay_requests": save_frame(
            replay, output / f"{OUTPUT_PREFIX}_prefix_replay_requests.csv"
        ),
    }
    if (
        materialized is not None
        and request_summary is not None
        and materialization_schema is not None
    ):
        artifacts["prefix_materialized_features"] = save_frame(
            materialized, output / f"{OUTPUT_PREFIX}_prefix_materialized_features.csv.gz", gzip=True
        )
        artifacts["prefix_materialization_request_summary"] = save_frame(
            request_summary, output / f"{OUTPUT_PREFIX}_prefix_materialization_request_summary.csv"
        )
        artifacts["prefix_materialization_schema"] = save_frame(
            materialization_schema, output / f"{OUTPUT_PREFIX}_prefix_materialization_schema.csv"
        )
    if (
        probe_oof is not None
        and probe_metrics is not None
        and probe_by_well is not None
        and probe_importance is not None
    ):
        artifacts["residual_probe_oof"] = save_frame(
            probe_oof, output / f"{OUTPUT_PREFIX}_residual_probe_oof.csv.gz", gzip=True
        )
        artifacts["residual_probe_metrics"] = save_frame(
            probe_metrics, output / f"{OUTPUT_PREFIX}_residual_probe_metrics.csv"
        )
        artifacts["residual_probe_by_well"] = save_frame(
            probe_by_well, output / f"{OUTPUT_PREFIX}_residual_probe_by_well.csv"
        )
        artifacts["residual_probe_feature_importance"] = save_frame(
            probe_importance, output / f"{OUTPUT_PREFIX}_residual_probe_feature_importance.csv"
        )
    max_abs_share_delta = float(report["selected_minus_target_share"].abs().max())
    mean_abs_share_delta = float(report["selected_minus_target_share"].abs().mean())
    if augmentation_summary.get("enabled"):
        status = (
            "exp218_augmentation_preflight_completed"
            if augmentation_summary.get("preflight")
            else "exp218_augmentation_completed"
        )
    else:
        status = (
            "residual_probe_completed_guard_pass"
            if distribution_guard["pass"] and residual_probe["pass"]
            else "residual_probe_completed_guard_failed"
        )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": nested(config, "experiment.route"),
        "parent": nested(config, "lineage.parent"),
        "train_dir": str(train_dir),
        "input_wells": int(len(metadata)),
        "official_eval_rows": int(metadata["eval_rows"].sum()),
        "candidate_cutoffs": int(len(candidates)),
        "candidate_wells": int(candidates["well_id"].nunique()),
        "selection": selection_summary,
        "distribution_match": {
            "max_abs_marginal_share_delta": max_abs_share_delta,
            "mean_abs_marginal_share_delta": mean_abs_share_delta,
            "bin_edges": bin_edges,
            "guard": distribution_guard,
        },
        "hidden_like": {
            **roles_meta,
            "metadata_valid_wells": int(metadata["hidden_like_valid"].sum()),
            "selected_valid_wells": int(
                selected.loc[selected["hidden_like_valid"], "well_id"].nunique()
            ),
        },
        "leakage_contract": leakage,
        "feature_regeneration_contract": nested(config, "model.replay_contract"),
        "prefix_materialization": materialization_audit,
        "residual_probe": residual_probe,
        "exp218_augmentation": augmentation_summary,
        "artifacts": artifacts,
        "model_training": {
            "active_variants": int(bool(augmentation_summary["enabled"])),
            "lightgbm_configs": int(augmentation_summary.get("lightgbm_configs", 0)),
            "folds_trained": int(augmentation_summary.get("folds", 0)),
            "boosters": int(augmentation_summary.get("boosters", 0)),
            "parent_control_retrained": False,
        },
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    summary["artifacts"]["summary"] = {
        "path": str(summary_path),
        "file_sha256": sha256_file(summary_path),
    }
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print("\nSelected cutoff sources")
    print(sources.to_string(index=False))
    print("\nLargest marginal distribution deltas")
    print(
        report.reindex(
            report["selected_minus_target_share"].abs().sort_values(ascending=False).index
        )
        .head(20)
        .to_string(index=False)
    )
    if probe_metrics is not None:
        print("\nResidual probe metrics")
        print(probe_metrics.to_string(index=False))
    return summary


# %% [markdown]
# ### Setup and configuration
#
# The notebook prints its route, parent, cutoff sources, caps, and historical
# references before touching raw data.

# %%
if os.environ.get("EXP239_IMPORT_ONLY", "0") != "1":
    CONFIG_PATH = find_named_file(
        "config.yaml",
        [
            Path("experiments") / EXPERIMENT_NAME / "config.yaml",
            Path.cwd() / "config.yaml",
        ],
    )
    CONFIG = read_yaml(CONFIG_PATH)
    print(f"config={CONFIG_PATH}")
    print(f"experiment={nested(CONFIG, 'experiment.name')}")
    print(f"route={nested(CONFIG, 'experiment.route')}")
    print(f"parent={nested(CONFIG, 'lineage.parent')}")
    print(f"historical_exp023={nested(CONFIG, 'model.historical_exp023')}")
    print(f"cutoff_generation={nested(CONFIG, 'model.cutoff_generation')}")
    print(f"distribution_caps={nested(CONFIG, 'model.distribution_matching')}")


# %% [markdown]
# ### Execute deterministic manifest audit

# %%
if os.environ.get("EXP239_IMPORT_ONLY", "0") != "1":
    SUMMARY = run_audit(CONFIG)
