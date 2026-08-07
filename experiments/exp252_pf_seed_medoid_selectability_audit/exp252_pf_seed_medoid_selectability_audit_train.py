# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp252 PF seed-medoid selectability audit
#
# exp243 v3で固定されたbase8 + K8 medoid候補を再生成せず、PF/cluster由来の
# target-free scoreがK8 trajectory-mode headroomを識別できるか監査する。
# true TVTはscore tableを固定した後のlabel / regret評価にだけ使用する。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime, configuration, and SHA helpers
# 3. Fixed input preflight helpers
# 4. Target-free score construction helpers
# 5. Row, block, and whole-well scope helpers
# 6. AUC, coverage, shuffled control, and regret helpers
# 7. Setup and fixed contract
# 8. Load exp243 inputs and freeze target-free scores
# 9. Join true TVT for labels and evaluate all scopes
# 10. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

# %% [markdown]
# ## 2. Runtime, configuration, and SHA helpers

# %%
EXPERIMENT_NAME = "exp252_pf_seed_medoid_selectability_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found; checked={candidates}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def output_experiment_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT
    return ROOT / "experiments" / EXPERIMENT_NAME


def require_authorized_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "Kaggle Notebook execution is authoritative. Local execution requires "
        "the explicit EXPERIMENT_ALLOW_LOCAL=1 debug opt-in."
    )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, base_seed: int) -> int:
    payload = "::".join([EXPERIMENT_NAME, str(base_seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


# %% [markdown]
# ## 3. Fixed input preflight helpers

# %%
def resolve_input(root: Path, spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    checked: list[str] = []
    for value in spec.get("paths") or []:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        checked.append(str(path))
        if path.exists():
            sha_kind = str(spec.get("sha_kind") or "raw")
            actual_sha = sha256_path(path, decompressed=sha_kind == "decompressed")
            expected_sha = str(spec.get("expected_sha256") or "")
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Input SHA mismatch for {path}: actual={actual_sha} expected={expected_sha}"
                )
            return path, {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha_kind": sha_kind,
                "sha256": actual_sha,
            }
    raise FileNotFoundError(
        f"Input {spec.get('filename')} was not found. Checked: {checked}"
    )


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        bad = int(np.size(values) - np.isfinite(values).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")


def aligned_frame(
    frame: pd.DataFrame,
    well_names: np.ndarray,
    *,
    name: str,
    slot_count: int | None = None,
) -> pd.DataFrame:
    work = frame.copy()
    work["well"] = work["well"].astype(str)
    if slot_count is None:
        if work["well"].duplicated().any():
            raise ValueError(f"{name} has duplicate well rows")
        aligned = work.set_index("well").reindex(well_names.astype(str))
    else:
        required = pd.MultiIndex.from_product(
            [well_names.astype(str), range(slot_count)], names=["well", "slot"]
        )
        if work[["well", "slot"]].duplicated().any():
            raise ValueError(f"{name} has duplicate well/slot rows")
        aligned = work.set_index(["well", "slot"]).reindex(required)
    if aligned.isna().all(axis=1).any():
        raise ValueError(f"{name} does not cover every required well/slot")
    return aligned


@dataclass
class InputBundle:
    well_names: np.ndarray
    well_codes: np.ndarray
    well_starts: np.ndarray
    well_ends: np.ndarray
    true_tvt: np.ndarray
    base_values: np.ndarray
    medoid_values: np.ndarray
    seed_std_by_row: np.ndarray
    cluster_manifest: pd.DataFrame
    cluster_summary: pd.DataFrame
    pf_diagnostics: pd.DataFrame
    input_meta: dict[str, Any]


def load_exp243_inputs(config: dict[str, Any], root: Path) -> InputBundle:
    input_specs = get_nested(config, "data.inputs") or {}
    resolved: dict[str, Path] = {}
    input_meta: dict[str, Any] = {}
    for key in ("row_candidates", "cluster_manifest", "cluster_summary", "pf_diagnostics"):
        resolved[key], input_meta[key] = resolve_input(root, input_specs[key])

    base_columns = list(get_nested(config, "audit.base_candidates") or [])
    medoid_columns = list(get_nested(config, "audit.medoid_candidates") or [])
    row_columns = [
        "well",
        "true_tvt",
        "pf_seed_std_diag",
        *base_columns,
        *medoid_columns,
    ]
    dtype = {
        column: np.float32
        for column in row_columns
        if column not in {"well"}
    }
    rows = pd.read_csv(resolved["row_candidates"], usecols=row_columns, dtype=dtype)
    if len(rows) != int(get_nested(config, "data.expected_rows")):
        raise ValueError(f"Unexpected row count: {len(rows)}")
    if rows["well"].nunique() != int(get_nested(config, "data.expected_wells")):
        raise ValueError(f"Unexpected well count: {rows['well'].nunique()}")
    rows["well"] = rows["well"].astype(str)
    well_codes, well_names = pd.factorize(rows["well"], sort=False)
    well_codes = np.asarray(well_codes, dtype=np.int32)
    well_names = np.asarray(well_names, dtype=str)
    starts = np.r_[0, np.flatnonzero(well_codes[1:] != well_codes[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(rows)].astype(np.int64)
    if len(starts) != len(well_names):
        raise ValueError("row candidates are not contiguous by well")
    if not np.array_equal(well_codes[starts], np.arange(len(well_names), dtype=np.int32)):
        raise ValueError("well order is not a single contiguous segment per well")

    true_tvt = rows.pop("true_tvt").to_numpy(np.float64)
    seed_std = rows.pop("pf_seed_std_diag").to_numpy(np.float32)
    base_values = rows[base_columns].to_numpy(np.float32)
    medoid_values = rows[medoid_columns].to_numpy(np.float32)
    require_finite("true_tvt", true_tvt)
    require_finite("seed_std_by_row", seed_std)
    require_finite("base_values", base_values)
    require_finite("medoid_values", medoid_values)

    manifest = pd.read_csv(resolved["cluster_manifest"])
    summary = pd.read_csv(resolved["cluster_summary"])
    diagnostics = pd.read_csv(resolved["pf_diagnostics"])
    require_columns(
        manifest,
        [
            "well",
            "k",
            "slot",
            "seed_mass",
            "likelihood_mass",
            "mean_within_distance",
            "nearest_other_medoid_distance",
            "medoid_log_likelihood",
        ],
        "cluster_manifest",
    )
    require_columns(
        summary,
        [
            "well",
            "k",
            "mean_assignment_distance",
            "normalized_entropy",
            "hhi",
            "effective_cluster_count",
            "max_pairwise_seed_distance",
            "ess_mean",
            "resampling_rate",
            "log_likelihood_std",
        ],
        "cluster_summary",
    )
    require_columns(
        diagnostics,
        ["well", "ess_mean", "resampling_rate", "log_likelihood_std"],
        "pf_diagnostics",
    )
    manifest = manifest.loc[pd.to_numeric(manifest["k"], errors="raise") == 8].copy()
    manifest["slot"] = pd.to_numeric(manifest["slot"], errors="raise").astype(int)
    summary = summary.loc[pd.to_numeric(summary["k"], errors="raise") == 8].copy()
    manifest = aligned_frame(
        manifest,
        well_names,
        name="K8 cluster manifest",
        slot_count=8,
    )
    summary = aligned_frame(summary, well_names, name="K8 cluster summary")
    diagnostics = aligned_frame(diagnostics, well_names, name="PF diagnostics")
    for column in ("ess_mean", "resampling_rate", "log_likelihood_std"):
        left = pd.to_numeric(summary[column], errors="raise").to_numpy(np.float64)
        right = pd.to_numeric(diagnostics[column], errors="raise").to_numpy(np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"cluster summary / PF diagnostics mismatch: {column}")

    input_meta["row_candidates"].update(
        {
            "rows": int(len(true_tvt)),
            "wells": int(len(well_names)),
            "base_candidates": base_columns,
            "medoid_candidates": medoid_columns,
        }
    )
    input_meta["cluster_manifest"].update({"k8_rows": int(len(manifest))})
    input_meta["cluster_summary"].update({"k8_rows": int(len(summary))})
    input_meta["pf_diagnostics"].update({"rows": int(len(diagnostics))})
    return InputBundle(
        well_names=well_names,
        well_codes=well_codes,
        well_starts=starts,
        well_ends=ends,
        true_tvt=true_tvt,
        base_values=base_values,
        medoid_values=medoid_values,
        seed_std_by_row=seed_std,
        cluster_manifest=manifest,
        cluster_summary=summary,
        pf_diagnostics=diagnostics,
        input_meta=input_meta,
    )


# %% [markdown]
# ## 4. Target-free score construction helpers

# %%
@dataclass
class TargetFreeScores:
    candidate_static: dict[str, np.ndarray]
    bank_static: dict[str, np.ndarray]
    candidate_nearest_base_by_row: np.ndarray
    bank_dynamic_by_row: dict[str, np.ndarray]
    contract: pd.DataFrame


def manifest_matrix(manifest: pd.DataFrame, column: str, well_count: int) -> np.ndarray:
    values = pd.to_numeric(manifest[column], errors="raise").to_numpy(np.float64)
    matrix = values.reshape(well_count, 8)
    require_finite(f"manifest.{column}", matrix)
    return matrix


def summary_vector(summary: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(summary[column], errors="raise").to_numpy(np.float64)
    require_finite(f"cluster_summary.{column}", values)
    return values


def likelihood_rank_score(log_likelihood: np.ndarray) -> np.ndarray:
    output = np.empty_like(log_likelihood, dtype=np.float64)
    denominator = max(log_likelihood.shape[1] - 1, 1)
    for index, row in enumerate(log_likelihood):
        ranks = pd.Series(row).rank(method="average", ascending=False).to_numpy(np.float64)
        output[index] = 1.0 - (ranks - 1.0) / denominator
    return output


def nearest_base_disagreement(
    base_values: np.ndarray,
    medoid_values: np.ndarray,
) -> np.ndarray:
    output = np.full(medoid_values.shape, np.inf, dtype=np.float32)
    for base_slot in range(base_values.shape[1]):
        distance = np.abs(medoid_values - base_values[:, base_slot, None])
        np.minimum(output, distance, out=output)
    require_finite("nearest_base8_disagreement", output)
    return output


def freeze_target_free_scores(
    bundle: InputBundle,
    config: dict[str, Any],
) -> TargetFreeScores:
    well_count = len(bundle.well_names)
    seed_mass = manifest_matrix(bundle.cluster_manifest, "seed_mass", well_count)
    likelihood_mass = manifest_matrix(bundle.cluster_manifest, "likelihood_mass", well_count)
    medoid_log_likelihood = manifest_matrix(
        bundle.cluster_manifest, "medoid_log_likelihood", well_count
    )
    mean_within = manifest_matrix(bundle.cluster_manifest, "mean_within_distance", well_count)
    separation = manifest_matrix(
        bundle.cluster_manifest, "nearest_other_medoid_distance", well_count
    )
    epsilon = float(get_nested(config, "audit.ratio_epsilon") or 1.0e-6)
    candidate_static = {
        "cluster_seed_mass": seed_mass,
        "cluster_likelihood_mass": likelihood_mass,
        "medoid_likelihood_rank_score": likelihood_rank_score(medoid_log_likelihood),
        "medoid_likelihood_gap_from_best": (
            medoid_log_likelihood - medoid_log_likelihood.max(axis=1, keepdims=True)
        ),
        "negative_mean_within_distance": -mean_within,
        "separation_to_within_ratio": separation / np.maximum(mean_within, epsilon),
    }
    summary = bundle.cluster_summary
    bank_static = {
        "cluster_normalized_entropy": summary_vector(summary, "normalized_entropy"),
        "negative_cluster_hhi": -summary_vector(summary, "hhi"),
        "effective_cluster_count": summary_vector(summary, "effective_cluster_count"),
        "mean_assignment_distance": summary_vector(summary, "mean_assignment_distance"),
        "max_pairwise_seed_distance": summary_vector(summary, "max_pairwise_seed_distance"),
        "negative_ess_mean": -summary_vector(summary, "ess_mean"),
        "resampling_rate": summary_vector(summary, "resampling_rate"),
        "log_likelihood_std": summary_vector(summary, "log_likelihood_std"),
    }
    disagreement = nearest_base_disagreement(bundle.base_values, bundle.medoid_values)
    bank_dynamic = {
        "seed_prediction_std": bundle.seed_std_by_row.astype(np.float32, copy=False),
        "k8_max_nearest_base_disagreement": disagreement.max(axis=1),
    }
    candidate_order = list(get_nested(config, "audit.candidate_scores") or [])
    bank_order = list(get_nested(config, "audit.bank_scores") or [])
    dynamic_candidate_names = ["nearest_base8_disagreement"]
    if candidate_order != [*candidate_static, *dynamic_candidate_names]:
        raise ValueError(
            f"Candidate score contract mismatch: config={candidate_order} "
            f"implementation={[*candidate_static, *dynamic_candidate_names]}"
        )
    if bank_order != [*bank_static, *bank_dynamic]:
        raise ValueError(
            f"Bank score contract mismatch: config={bank_order} "
            f"implementation={[*bank_static, *bank_dynamic]}"
        )
    contract_rows: list[dict[str, Any]] = []
    candidate_sources = {
        "cluster_seed_mass": "exp243 K8 cluster seed_mass; higher is more supported",
        "cluster_likelihood_mass": "exp243 K8 likelihood_mass; higher is more supported",
        "medoid_likelihood_rank_score": "within-well K8 medoid likelihood rank; best=1",
        "medoid_likelihood_gap_from_best": "medoid log likelihood minus K8 best; best=0",
        "negative_mean_within_distance": "negative cluster mean assignment distance",
        "separation_to_within_ratio": "nearest other medoid distance / mean within distance",
        "nearest_base8_disagreement": "scope RMS distance to nearest base8 path",
    }
    bank_sources = {
        "cluster_normalized_entropy": "exp243 K8 normalized cluster entropy",
        "negative_cluster_hhi": "negative exp243 K8 cluster HHI",
        "effective_cluster_count": "exp243 K8 effective cluster count",
        "mean_assignment_distance": "exp243 K8 mean seed-to-medoid assignment distance",
        "max_pairwise_seed_distance": "exp243 maximum pairwise seed trajectory distance",
        "negative_ess_mean": "negative exp243 PF mean effective sample size",
        "resampling_rate": "exp243 PF resampling rate",
        "log_likelihood_std": "exp243 seed log-likelihood standard deviation",
        "seed_prediction_std": "scope mean of exp243 row seed prediction std",
        "k8_max_nearest_base_disagreement": "max K8 medoid scope RMS distance to nearest base8",
    }
    for name in candidate_order:
        contract_rows.append(
            {
                "score_level": "candidate",
                "score": name,
                "dynamic_by_scope": name in dynamic_candidate_names,
                "higher_is_more_selectable": True,
                "source": candidate_sources[name],
            }
        )
    for name in bank_order:
        contract_rows.append(
            {
                "score_level": "bank",
                "score": name,
                "dynamic_by_scope": name in bank_dynamic,
                "higher_is_more_selectable": True,
                "source": bank_sources[name],
            }
        )
    return TargetFreeScores(
        candidate_static=candidate_static,
        bank_static=bank_static,
        candidate_nearest_base_by_row=disagreement,
        bank_dynamic_by_row=bank_dynamic,
        contract=pd.DataFrame(contract_rows),
    )


# %% [markdown]
# ## 5. Row, block, and whole-well scope helpers

# %%
@dataclass
class ScopeView:
    name: str
    well_codes: np.ndarray
    unit_row_counts: np.ndarray
    base_loss: np.ndarray
    medoid_loss: np.ndarray
    candidate_dynamic: dict[str, np.ndarray]
    bank_dynamic: dict[str, np.ndarray]


def prefix_sum(values: np.ndarray) -> np.ndarray:
    output = np.empty((len(values) + 1, *values.shape[1:]), dtype=np.float64)
    output[0] = 0.0
    np.cumsum(values, axis=0, dtype=np.float64, out=output[1:])
    return output


def scope_bounds(
    well_starts: np.ndarray,
    well_ends: np.ndarray,
    block_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[int] = []
    ends: list[int] = []
    wells: list[int] = []
    for well_code, (start, end) in enumerate(zip(well_starts, well_ends, strict=True)):
        if block_rows is None:
            starts.append(int(start))
            ends.append(int(end))
            wells.append(well_code)
            continue
        for block_start in range(int(start), int(end), block_rows):
            starts.append(block_start)
            ends.append(min(block_start + block_rows, int(end)))
            wells.append(well_code)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(wells, dtype=np.int32),
    )


def aggregate_from_prefix(
    prefix: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    *,
    root_mean: bool,
) -> np.ndarray:
    values = (prefix[ends] - prefix[starts]) / counts.reshape((-1,) + (1,) * (prefix.ndim - 1))
    if root_mean:
        values = np.sqrt(np.maximum(values, 0.0))
    return values.astype(np.float32)


def build_scope_views(
    bundle: InputBundle,
    scores: TargetFreeScores,
    config: dict[str, Any],
) -> list[ScopeView]:
    # This is the first function that receives true_tvt. Target-free score arrays
    # above are already frozen and cannot depend on these losses or labels.
    true_tvt = bundle.true_tvt
    base_row_loss = np.abs(bundle.base_values.astype(np.float64) - true_tvt[:, None]).astype(
        np.float32
    )
    medoid_row_loss = np.abs(
        bundle.medoid_values.astype(np.float64) - true_tvt[:, None]
    ).astype(np.float32)
    base_sq_prefix = prefix_sum(np.square(base_row_loss, dtype=np.float64))
    medoid_sq_prefix = prefix_sum(np.square(medoid_row_loss, dtype=np.float64))
    disagreement_sq_prefix = prefix_sum(
        np.square(scores.candidate_nearest_base_by_row, dtype=np.float64)
    )
    seed_std_prefix = prefix_sum(bundle.seed_std_by_row.astype(np.float64))

    views = [
        ScopeView(
            name="row",
            well_codes=bundle.well_codes,
            unit_row_counts=np.ones(len(true_tvt), dtype=np.int32),
            base_loss=base_row_loss,
            medoid_loss=medoid_row_loss,
            candidate_dynamic={
                "nearest_base8_disagreement": scores.candidate_nearest_base_by_row
            },
            bank_dynamic={
                "seed_prediction_std": scores.bank_dynamic_by_row["seed_prediction_std"],
                "k8_max_nearest_base_disagreement": scores.bank_dynamic_by_row[
                    "k8_max_nearest_base_disagreement"
                ],
            },
        )
    ]
    scope_specs = [
        (f"block_{block}", int(block))
        for block in (get_nested(config, "audit.block_rows") or [])
    ]
    scope_specs.append(("whole_well", None))
    for name, block_rows in scope_specs:
        starts, ends, well_codes = scope_bounds(
            bundle.well_starts, bundle.well_ends, block_rows
        )
        counts = (ends - starts).astype(np.int32)
        candidate_disagreement = aggregate_from_prefix(
            disagreement_sq_prefix,
            starts,
            ends,
            counts,
            root_mean=True,
        )
        views.append(
            ScopeView(
                name=name,
                well_codes=well_codes,
                unit_row_counts=counts,
                base_loss=aggregate_from_prefix(
                    base_sq_prefix, starts, ends, counts, root_mean=True
                ),
                medoid_loss=aggregate_from_prefix(
                    medoid_sq_prefix, starts, ends, counts, root_mean=True
                ),
                candidate_dynamic={
                    "nearest_base8_disagreement": candidate_disagreement
                },
                bank_dynamic={
                    "seed_prediction_std": aggregate_from_prefix(
                        seed_std_prefix, starts, ends, counts, root_mean=False
                    ),
                    "k8_max_nearest_base_disagreement": candidate_disagreement.max(axis=1),
                },
            )
        )
    return views


def scope_labels(
    view: ScopeView,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    best_base = view.base_loss.min(axis=1)
    best_medoid = view.medoid_loss.min(axis=1)
    useful_bank = best_medoid + tolerance < best_base
    primary_candidate = useful_bank[:, None] & (
        view.medoid_loss <= best_medoid[:, None] + tolerance
    )
    secondary_candidate = view.medoid_loss + tolerance < best_base[:, None]
    return useful_bank, {
        "medoid_is_union_best": primary_candidate,
        "medoid_beats_best_base8": secondary_candidate,
    }


# %% [markdown]
# ## 6. AUC, coverage, shuffled control, and regret helpers

# %%
def metrics_from_grouped_counts(
    score_values: np.ndarray,
    positive_counts: dict[str, np.ndarray],
    total_counts: np.ndarray,
    top_fraction: float,
) -> dict[str, dict[str, float | int]]:
    score = np.asarray(score_values, dtype=np.float64).ravel()
    total = np.broadcast_to(total_counts, score_values.shape).astype(np.float64).ravel()
    if not np.isfinite(score).all() or np.any(total < 0):
        raise ValueError("Static score/count arrays must be finite and non-negative")
    order = np.argsort(score, kind="quicksort")
    sorted_score = score[order]
    group_start = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    grouped_total = np.add.reduceat(total[order], group_start)
    group_score = sorted_score[group_start]
    required_top = max(float(grouped_total.sum()) * top_fraction, 1.0)
    reverse_cumulative = np.cumsum(grouped_total[::-1])
    crossing = min(
        int(np.searchsorted(reverse_cumulative, required_top, side="left")),
        len(group_score) - 1,
    )
    threshold = group_score[::-1][crossing]
    selected_groups = group_score >= threshold
    output: dict[str, dict[str, float | int]] = {}
    for label_name, counts in positive_counts.items():
        positive = np.asarray(counts, dtype=np.float64).ravel()[order]
        grouped_positive = np.add.reduceat(positive, group_start)
        grouped_negative = grouped_total - grouped_positive
        positives = float(grouped_positive.sum())
        negatives = float(grouped_negative.sum())
        negative_before = np.cumsum(grouped_negative) - grouped_negative
        auc = (
            float(
                np.sum(
                    grouped_positive
                    * (negative_before + 0.5 * grouped_negative)
                )
                / (positives * negatives)
            )
            if positives > 0 and negatives > 0
            else float("nan")
        )
        selected_total = float(grouped_total[selected_groups].sum())
        selected_positive = float(grouped_positive[selected_groups].sum())
        prevalence = positives / (positives + negatives) if positives + negatives else float("nan")
        precision = selected_positive / selected_total if selected_total else float("nan")
        coverage = selected_positive / positives if positives else float("nan")
        output[label_name] = {
            "records": int(positives + negatives),
            "positives": int(positives),
            "positive_rate": prevalence,
            "roc_auc": auc,
            "top_fraction_requested": top_fraction,
            "top_fraction_actual_with_ties": selected_total / (positives + negatives),
            "top_precision": precision,
            "positive_coverage": coverage,
            "top_precision_lift": precision / prevalence if prevalence > 0 else float("nan"),
        }
    return output


def metrics_from_dense_scores(
    score_values: np.ndarray,
    labels: dict[str, np.ndarray],
    top_fraction: float,
) -> dict[str, dict[str, float | int]]:
    score = np.asarray(score_values, dtype=np.float64).ravel()
    if not np.isfinite(score).all():
        raise ValueError("Dynamic score array contains non-finite values")
    order = np.argsort(score, kind="quicksort")
    sorted_score = score[order]
    group_start = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    grouped_total = np.diff(np.r_[group_start, len(score)]).astype(np.float64)
    group_score = sorted_score[group_start]
    required_top = max(len(score) * top_fraction, 1.0)
    reverse_cumulative = np.cumsum(grouped_total[::-1])
    crossing = min(
        int(np.searchsorted(reverse_cumulative, required_top, side="left")),
        len(group_score) - 1,
    )
    threshold = group_score[::-1][crossing]
    selected_groups = group_score >= threshold
    output: dict[str, dict[str, float | int]] = {}
    for label_name, label_values in labels.items():
        positive = np.asarray(label_values, dtype=np.uint8).ravel()[order]
        grouped_positive = np.add.reduceat(positive, group_start).astype(np.float64)
        grouped_negative = grouped_total - grouped_positive
        positives = float(grouped_positive.sum())
        negatives = float(grouped_negative.sum())
        negative_before = np.cumsum(grouped_negative) - grouped_negative
        auc = (
            float(
                np.sum(
                    grouped_positive
                    * (negative_before + 0.5 * grouped_negative)
                )
                / (positives * negatives)
            )
            if positives > 0 and negatives > 0
            else float("nan")
        )
        selected_total = float(grouped_total[selected_groups].sum())
        selected_positive = float(grouped_positive[selected_groups].sum())
        prevalence = positives / len(score) if len(score) else float("nan")
        precision = selected_positive / selected_total if selected_total else float("nan")
        output[label_name] = {
            "records": int(len(score)),
            "positives": int(positives),
            "positive_rate": prevalence,
            "roc_auc": auc,
            "top_fraction_requested": top_fraction,
            "top_fraction_actual_with_ties": selected_total / len(score),
            "top_precision": precision,
            "positive_coverage": selected_positive / positives if positives else float("nan"),
            "top_precision_lift": precision / prevalence if prevalence > 0 else float("nan"),
        }
    return output


def counts_by_well(
    labels: np.ndarray,
    well_codes: np.ndarray,
    well_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(well_codes, minlength=well_count).astype(np.int64)
    output_shape = (well_count, *labels.shape[1:])
    positives = np.zeros(output_shape, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(well_codes[1:] != well_codes[:-1]) + 1]
    ends = np.r_[starts[1:], len(well_codes)]
    if len(starts) != well_count:
        raise ValueError("Scope units are not contiguous by well")
    for start, end in zip(starts, ends, strict=True):
        well = int(well_codes[start])
        positives[well] = labels[start:end].sum(axis=0, dtype=np.int64)
    return positives, counts


def stable_permutation(length: int, *, scope: str, score: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed("shuffled_score", scope, score, base_seed=seed)
    )
    return rng.permutation(length)


def append_score_metrics(
    rows: list[dict[str, Any]],
    metrics: dict[str, dict[str, float | int]],
    *,
    scope: str,
    score_level: str,
    score: str,
    control: str,
) -> None:
    for label, values in metrics.items():
        rows.append(
            {
                "scope": scope,
                "score_level": score_level,
                "score": score,
                "control": control,
                "label": label,
                **values,
            }
        )


def top1_readout(
    view: ScopeView,
    selected_slot: np.ndarray,
    useful_bank: np.ndarray,
    candidate_labels: dict[str, np.ndarray],
    *,
    score: str,
    control: str,
    tolerance: float,
) -> dict[str, Any]:
    selected_loss = view.medoid_loss[
        np.arange(len(view.medoid_loss)), selected_slot.astype(np.int64)
    ].astype(np.float64)
    best_medoid = view.medoid_loss.min(axis=1).astype(np.float64)
    best_base = view.base_loss.min(axis=1).astype(np.float64)
    regret = np.maximum(selected_loss - best_medoid, 0.0)
    selected_beats_base = selected_loss + tolerance < best_base
    selected_is_union_best = candidate_labels["medoid_is_union_best"]
    selected_is_union_best = selected_is_union_best[
        np.arange(len(view.medoid_loss)), selected_slot.astype(np.int64)
    ]
    useful_regret = regret[useful_bank]
    return {
        "scope": view.name,
        "score": score,
        "control": control,
        "units": int(len(view.medoid_loss)),
        "useful_bank_units": int(useful_bank.sum()),
        "selected_beats_base_units": int(selected_beats_base.sum()),
        "useful_medoid_coverage": (
            float(selected_beats_base[useful_bank].mean())
            if useful_bank.any()
            else float("nan")
        ),
        "union_best_match_rate_on_useful": (
            float(selected_is_union_best[useful_bank].mean())
            if useful_bank.any()
            else float("nan")
        ),
        "top1_regret_mean": float(np.mean(regret)),
        "top1_regret_p90": float(np.quantile(regret, 0.90)),
        "top1_regret_max": float(np.max(regret)),
        "top1_regret_mean_on_useful": (
            float(np.mean(useful_regret)) if len(useful_regret) else float("nan")
        ),
        "selected_minus_best_base_loss_mean": float(np.mean(selected_loss - best_base)),
    }


def pooled_oracle_rmse(loss: np.ndarray, row_counts: np.ndarray) -> float:
    weights = row_counts.astype(np.float64)
    return float(np.sqrt(np.sum(np.square(loss, dtype=np.float64) * weights) / weights.sum()))


def scope_metric_row(
    view: ScopeView,
    useful_bank: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    best_base = view.base_loss.min(axis=1)
    best_medoid = view.medoid_loss.min(axis=1)
    best_union = np.minimum(best_base, best_medoid)
    base_rmse = pooled_oracle_rmse(best_base, view.unit_row_counts)
    medoid_rmse = pooled_oracle_rmse(best_medoid, view.unit_row_counts)
    union_rmse = pooled_oracle_rmse(best_union, view.unit_row_counts)
    return {
        "scope": view.name,
        "units": int(len(view.well_codes)),
        "rows": int(view.unit_row_counts.sum()),
        "useful_bank_units": int(useful_bank.sum()),
        "useful_bank_rate": float(useful_bank.mean()),
        "base8_oracle_pooled_rmse": base_rmse,
        "k8_oracle_pooled_rmse": medoid_rmse,
        "base8_plus_k8_oracle_pooled_rmse": union_rmse,
        "union_delta_rmse_vs_base8": union_rmse - base_rmse,
        "k8_delta_rmse_vs_base8": medoid_rmse - base_rmse,
        "ties_within_tolerance": int(np.sum(np.abs(best_medoid - best_base) <= tolerance)),
    }


def evaluate_scope(
    view: ScopeView,
    scores: TargetFreeScores,
    config: dict[str, Any],
    well_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tolerance = float(get_nested(config, "audit.tie_tolerance_ft") or 1.0e-6)
    top_fraction = float(get_nested(config, "audit.coverage_top_fraction") or 0.10)
    shuffle_seed = int(get_nested(config, "audit.shuffled_control.seed") or 42)
    useful_bank, candidate_labels = scope_labels(view, tolerance)
    score_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []

    bank_positive, unit_counts = counts_by_well(
        useful_bank.astype(np.uint8), view.well_codes, well_count
    )
    candidate_positive: dict[str, np.ndarray] = {}
    for label_name, label_values in candidate_labels.items():
        candidate_positive[label_name], candidate_unit_counts = counts_by_well(
            label_values.astype(np.uint8), view.well_codes, well_count
        )
        if not np.array_equal(candidate_unit_counts, unit_counts):
            raise ValueError("candidate and bank unit counts differ")

    for score_name, static_values in scores.bank_static.items():
        well_permutation = stable_permutation(
            well_count,
            scope=view.name,
            score=score_name,
            seed=shuffle_seed,
        )
        for control, values in (
            ("real", static_values),
            ("shuffled_score", static_values[well_permutation]),
        ):
            metrics = metrics_from_grouped_counts(
                values,
                {"union_best_source_is_k8": bank_positive},
                unit_counts,
                top_fraction,
            )
            append_score_metrics(
                score_rows,
                metrics,
                scope=view.name,
                score_level="bank",
                score=score_name,
                control=control,
            )

    for score_name, dynamic_values in view.bank_dynamic.items():
        permutation = stable_permutation(
            len(dynamic_values),
            scope=view.name,
            score=score_name,
            seed=shuffle_seed,
        )
        for control, values in (
            ("real", dynamic_values),
            ("shuffled_score", dynamic_values[permutation]),
        ):
            metrics = metrics_from_dense_scores(
                values,
                {"union_best_source_is_k8": useful_bank},
                top_fraction,
            )
            append_score_metrics(
                score_rows,
                metrics,
                scope=view.name,
                score_level="bank",
                score=score_name,
                control=control,
            )

    for score_name, static_values in scores.candidate_static.items():
        well_permutation = stable_permutation(
            well_count,
            scope=view.name,
            score=score_name,
            seed=shuffle_seed,
        )
        for control, values in (
            ("real", static_values),
            ("shuffled_score", static_values[well_permutation]),
        ):
            metrics = metrics_from_grouped_counts(
                values,
                candidate_positive,
                unit_counts[:, None],
                top_fraction,
            )
            append_score_metrics(
                score_rows,
                metrics,
                scope=view.name,
                score_level="candidate",
                score=score_name,
                control=control,
            )
            selected_by_well = np.argmax(values, axis=1).astype(np.int16)
            regret_rows.append(
                top1_readout(
                    view,
                    selected_by_well[view.well_codes],
                    useful_bank,
                    candidate_labels,
                    score=score_name,
                    control=control,
                    tolerance=tolerance,
                )
            )

    for score_name, dynamic_values in view.candidate_dynamic.items():
        permutation = stable_permutation(
            len(dynamic_values),
            scope=view.name,
            score=score_name,
            seed=shuffle_seed,
        )
        for control, values in (
            ("real", dynamic_values),
            ("shuffled_score", dynamic_values[permutation]),
        ):
            metrics = metrics_from_dense_scores(values, candidate_labels, top_fraction)
            append_score_metrics(
                score_rows,
                metrics,
                scope=view.name,
                score_level="candidate",
                score=score_name,
                control=control,
            )
            regret_rows.append(
                top1_readout(
                    view,
                    np.argmax(values, axis=1).astype(np.int16),
                    useful_bank,
                    candidate_labels,
                    score=score_name,
                    control=control,
                    tolerance=tolerance,
                )
            )
    return score_rows, regret_rows, scope_metric_row(view, useful_bank, tolerance)


# %% [markdown]
# ## 7. Setup and fixed contract

# %%
started = time.time()
require_authorized_runtime()
config_path = find_config_path()
config = read_yaml(config_path)
experiment_dir = output_experiment_dir()
artifacts_dir = experiment_dir / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

assert get_nested(config, "experiment.name") == EXPERIMENT_NAME
assert get_nested(config, "experiment.route") == "pf_beam"
assert get_nested(config, "model.k_values") == [8]
assert int(get_nested(config, "model.lightgbm_config_count")) == 0
assert int(get_nested(config, "model.fold_count")) == 0
assert int(get_nested(config, "model.booster_count")) == 0
assert int(get_nested(config, "model.pf_replay_count")) == 0
assert get_nested(config, "model.parent_control_retraining") is False
assert get_nested(config, "inference.enabled") is False
assert get_nested(config, "inference.create_submission") is False

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(config, "experiment.route"),
            "parent": get_nested(config, "lineage.parent"),
            "scopes": get_nested(config, "validation.scopes"),
            "K": get_nested(config, "model.k_values"),
            "bank_scores": get_nested(config, "audit.bank_scores"),
            "candidate_scores": get_nested(config, "audit.candidate_scores"),
            "target_usage": "loss_label_regret_coverage_only_after_score_freeze",
            "model_configs": 0,
            "folds": 0,
            "boosters": 0,
            "pf_replays": 0,
            "inference": False,
            "submission": False,
        },
        indent=2,
        ensure_ascii=False,
    )
)


# %% [markdown]
# ## 8. Load exp243 inputs and freeze target-free scores
#
# このセルのscore生成関数へtrue TVTは渡さない。候補値、cluster manifest / summary、
# PF diagnosticsだけで全scoreとscore contractを固定する。

# %%
bundle = load_exp243_inputs(config, ROOT)
target_free_scores = freeze_target_free_scores(bundle, config)

print("Input preflight")
print(json.dumps(to_jsonable(bundle.input_meta), indent=2, ensure_ascii=False))
print("Target-free score contract")
display(target_free_scores.contract)
print(
    {
        "rows": len(bundle.true_tvt),
        "wells": len(bundle.well_names),
        "K8_manifest_rows": len(bundle.cluster_manifest),
        "target_free_score_freeze": "complete_before_true_tvt_loss_join",
    }
)


# %% [markdown]
# ## 9. Join true TVT for labels and evaluate all scopes
#
# score freeze完了後に初めてtrue TVTを使い、row absolute error、block/well RMSE、
# best-source label、AUC、coverage、top1 regretを計算する。

# %%
scope_views = build_scope_views(bundle, target_free_scores, config)
expected_scopes = list(get_nested(config, "validation.scopes") or [])
if [view.name for view in scope_views] != expected_scopes:
    raise ValueError(
        f"Scope contract mismatch: actual={[view.name for view in scope_views]} "
        f"expected={expected_scopes}"
    )

score_metric_rows: list[dict[str, Any]] = []
top1_regret_rows: list[dict[str, Any]] = []
scope_metric_rows: list[dict[str, Any]] = []
for view in scope_views:
    scope_scores, scope_regret, scope_summary = evaluate_scope(
        view,
        target_free_scores,
        config,
        len(bundle.well_names),
    )
    score_metric_rows.extend(scope_scores)
    top1_regret_rows.extend(scope_regret)
    scope_metric_rows.append(scope_summary)
    print(
        f"[scope] {view.name}: units={len(view.well_codes):,} "
        f"union_delta_rmse={scope_summary['union_delta_rmse_vs_base8']:.9f}",
        flush=True,
    )

score_metrics = pd.DataFrame(score_metric_rows)
top1_regret = pd.DataFrame(top1_regret_rows)
scope_metrics = pd.DataFrame(scope_metric_rows)
shuffle_seed = int(get_nested(config, "audit.shuffled_control.seed") or 42)
score_metrics["shuffled_control_seed"] = [
    stable_seed(
        "shuffled_score",
        scope,
        score,
        base_seed=shuffle_seed,
    )
    for scope, score in zip(score_metrics["scope"], score_metrics["score"], strict=True)
]
top1_regret["shuffled_control_seed"] = [
    stable_seed(
        "shuffled_score",
        scope,
        score,
        base_seed=shuffle_seed,
    )
    for scope, score in zip(top1_regret["scope"], top1_regret["score"], strict=True)
]

whole_well = next(view for view in scope_views if view.name == "whole_well")
whole_useful, _ = scope_labels(
    whole_well, float(get_nested(config, "audit.tie_tolerance_ft") or 1.0e-6)
)
best_base_well = whole_well.base_loss.min(axis=1)
best_medoid_well = whole_well.medoid_loss.min(axis=1)
best_medoid_slot = np.argmin(whole_well.medoid_loss, axis=1)
by_well = pd.DataFrame(
    {
        "well": bundle.well_names,
        "rows": whole_well.unit_row_counts,
        "best_base8_rmse": best_base_well,
        "best_k8_rmse": best_medoid_well,
        "best_k8_slot": best_medoid_slot,
        "k8_minus_base8_rmse": best_medoid_well - best_base_well,
        "union_best_source_is_k8": whole_useful,
    }
)
for score_name, values in target_free_scores.bank_static.items():
    by_well[score_name] = values
for score_name, values in whole_well.bank_dynamic.items():
    by_well[score_name] = values

print("Scope oracle metrics")
display(scope_metrics)
print("Real-vs-shuffled AUC preview")
display(
    score_metrics.pivot_table(
        index=["scope", "score_level", "score", "label"],
        columns="control",
        values="roc_auc",
    ).reset_index()
)
print("Candidate top1 regret preview")
display(top1_regret.head(80))


# %% [markdown]
# ## 10. Metrics, diagnostics, and generated artifacts

# %%
outputs = get_nested(config, "audit.outputs") or {}
artifact_paths = {
    "score_contract": artifacts_dir / outputs["score_contract_filename"],
    "scope_metrics": artifacts_dir / outputs["scope_metrics_filename"],
    "score_metrics": artifacts_dir / outputs["score_metrics_filename"],
    "top1_regret": artifacts_dir / outputs["top1_regret_filename"],
    "by_well": artifacts_dir / outputs["by_well_filename"],
    "summary": artifacts_dir / outputs["summary_filename"],
}
target_free_scores.contract.to_csv(artifact_paths["score_contract"], index=False)
scope_metrics.to_csv(artifact_paths["scope_metrics"], index=False)
score_metrics.to_csv(artifact_paths["score_metrics"], index=False)
top1_regret.to_csv(artifact_paths["top1_regret"], index=False)
by_well.to_csv(artifact_paths["by_well"], index=False)

real_auc = score_metrics.loc[score_metrics["control"] == "real", "roc_auc"]
shuffled_auc = score_metrics.loc[
    score_metrics["control"] == "shuffled_score", "roc_auc"
]
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_selectability_audit",
    "route": "pf_beam",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "runtime_seconds": float(time.time() - started),
    "rows": int(len(bundle.true_tvt)),
    "wells": int(len(bundle.well_names)),
    "model_configs": 0,
    "folds": 0,
    "boosters": 0,
    "pf_replays": 0,
    "K": [8],
    "input_preflight": bundle.input_meta,
    "score_contract": target_free_scores.contract.to_dict("records"),
    "scope_metrics": scope_metrics.to_dict("records"),
    "score_metric_rows": int(len(score_metrics)),
    "top1_regret_rows": int(len(top1_regret)),
    "auc_range": {
        "real_min": float(real_auc.min()),
        "real_max": float(real_auc.max()),
        "shuffled_min": float(shuffled_auc.min()),
        "shuffled_max": float(shuffled_auc.max()),
    },
    "target_usage": "score_freeze_then_loss_label_auc_regret_coverage_only",
    "negative_control": get_nested(config, "audit.shuffled_control"),
    "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    "inference": False,
    "submission": False,
}
write_json(artifact_paths["summary"], summary)
artifact_sha = {
    key: sha256_path(path)
    for key, path in artifact_paths.items()
}
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_selectability_audit",
    "route": "pf_beam",
    "metric": "diagnostic_auc_regret_coverage",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "rows": summary["rows"],
    "wells": summary["wells"],
    "model_configs": 0,
    "folds": 0,
    "boosters": 0,
    "pf_replays": 0,
    "scope_metrics": summary["scope_metrics"],
    "input_preflight": summary["input_preflight"],
    "artifact_sha256": artifact_sha,
    "artifacts": summary["artifacts"],
    "notes": (
        "Fixed exp243 K8 target-free selectability audit only; no selector, "
        "PF replay, raw-test inference, or submission."
    ),
}
metrics_path = experiment_dir / "metrics.json"
write_json(metrics_path, metrics)

print("Generated artifacts")
print(json.dumps(to_jsonable(summary["artifacts"]), indent=2))
print("Artifact SHA256")
print(json.dumps(artifact_sha, indent=2))
print(
    json.dumps(
        {
            "status": summary["status"],
            "runtime_seconds": summary["runtime_seconds"],
            "rows": summary["rows"],
            "wells": summary["wells"],
            "score_metric_rows": summary["score_metric_rows"],
            "top1_regret_rows": summary["top1_regret_rows"],
            "metrics_path": str(metrics_path),
        },
        indent=2,
    )
)
