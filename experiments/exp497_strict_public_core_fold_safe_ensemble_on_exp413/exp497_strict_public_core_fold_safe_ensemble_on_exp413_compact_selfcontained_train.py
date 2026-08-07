# %% [markdown]
# # exp497 strict public core fold-safe ensemble on exp413 — Stage 0
#
# This compact self-contained candidate implements only the frozen zero-model
# preflight. It audits source identity, parent evidence, nested split boundaries,
# deterministic seeds, execution counts, and feature-memory contracts. It does
# not generate physical paths, fit models, replace the canonical notebook, or
# create an inference/submission path.

# %% [markdown]
# ## Contents
# 1. Imports and package resolution
# 2. Serialization, hashing, and source-audit helpers
# 3. Static feature and execution contracts
# 4. Parent OOF and nested-fold helpers
# 5. Target-free selector and prediction-freeze helpers
# 6. Constant convex meta-blend helpers
# 7. Setup and authorization guard
# 8. Stage 0 orchestration and generated evidence

# %%
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
EXECUTE_NOTEBOOK = os.environ.get("EXP497_IMPORT_ONLY", "0") != "1"
STAGE0_APPROVED_SCOPE = "stage0_source_parent_fold_execution_feature_preflight_no_route_run"


def locate_experiment_dir(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = [
        start / "experiments" / EXPERIMENT_NAME,
        start,
        Path("/kaggle/working") / "experiments" / EXPERIMENT_NAME,
        Path("/kaggle/working"),
    ]
    for parent in (start, *start.parents):
        candidates.append(parent / "experiments" / EXPERIMENT_NAME)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "config.yaml").is_file() and (
            candidate / "public_source_inventory.yaml"
        ).is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {EXPERIMENT_NAME}")


EXPERIMENT_DIR = locate_experiment_dir()


# %% [markdown]
# ## 2. Serialization, hashing, and source-audit helpers


# %%
def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_contract_bundle(
    experiment_dir: Path = EXPERIMENT_DIR,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        read_yaml(experiment_dir / "config.yaml"),
        read_yaml(experiment_dir / "public_core_contract.yaml"),
        read_yaml(experiment_dir / "ensemble_contract.yaml"),
        read_yaml(experiment_dir / "public_source_inventory.yaml"),
    )


def _definition_lines(source_text: str, name: str, kind: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(kind)}\s+{re.escape(name)}\b", re.MULTILINE)
    starts = [match.start() for match in pattern.finditer(source_text)]
    return [source_text.count("\n", 0, start) + 1 for start in starts]


def scan_reference_source(path: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    payload = path.read_bytes()
    source_text = payload.decode("utf-8")
    expected = inventory["source"]
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != str(expected["converted_source_sha256"]):
        raise ValueError(
            "Reference source SHA mismatch: "
            f"observed={observed_sha} expected={expected['converted_source_sha256']}"
        )
    if len(payload) != int(expected["converted_source_bytes"]):
        raise ValueError("Reference source byte count mismatch")
    if len(source_text.splitlines()) != int(expected["converted_source_lines"]):
        raise ValueError("Reference source line count mismatch")

    symbol_rows: list[dict[str, Any]] = []
    for item in inventory["required_symbols"]:
        kind = "def" if item["kind"] == "function" else "class"
        observed_lines = _definition_lines(source_text, str(item["name"]), kind)
        expected_lines = [int(value) for value in item["definition_lines"]]
        if observed_lines != expected_lines:
            raise ValueError(
                f"Reference symbol mismatch for {item['name']}: "
                f"observed={observed_lines} expected={expected_lines}"
            )
        symbol_rows.append(
            {
                "name": str(item["name"]),
                "kind": str(item["kind"]),
                "definition_lines": observed_lines,
            }
        )

    exclusion_counts = {
        label: len(re.findall(pattern, source_text))
        for label, pattern in inventory["reference_exclusion_patterns"].items()
    }
    return {
        "status": "pass",
        "path": str(path),
        "sha256": observed_sha,
        "bytes": len(payload),
        "lines": len(source_text.splitlines()),
        "required_symbols": symbol_rows,
        "reference_exclusion_marker_counts": exclusion_counts,
    }


def scan_decontaminated_executable(
    source_text: str,
    inventory: dict[str, Any],
) -> dict[str, int]:
    hits = {
        label: len(re.findall(pattern, source_text))
        for label, pattern in inventory["decontaminated_executable_forbidden_patterns"].items()
    }
    nonzero = {label: count for label, count in hits.items() if count}
    if nonzero:
        raise ValueError(f"Decontamination scan failed: {nonzero}")
    return hits


def resolve_reference_source(
    config: dict[str, Any],
    experiment_dir: Path = EXPERIMENT_DIR,
) -> Path:
    source_cfg = config["data"]["public_source_py"]
    expected_sha = str(source_cfg["expected_sha256"])
    patterns: list[str] = []
    explicit = os.environ.get("EXP497_PUBLIC_SOURCE_PATH")
    if explicit:
        patterns.append(explicit)
    patterns.extend(str(value) for value in source_cfg["patterns"])
    candidates: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if not path.is_absolute():
            path = experiment_dir.parents[1] / path
        candidates.extend(Path(value) for value in glob.glob(str(path), recursive=True))
    seen: set[str] = set()
    mismatches: dict[str, str] = {}
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        observed = sha256_file(candidate)
        if observed == expected_sha:
            return candidate
        mismatches[key] = observed
    raise FileNotFoundError(
        f"No SHA-qualified converted reference source was found; mismatches={mismatches}"
    )


# %% [markdown]
# ## 3. Static feature and execution contracts


# %%
def physical_execution_inventory(config: dict[str, Any]) -> dict[str, int]:
    contract = config["training_contract"]["physical_execution_inventory"]
    wells = int(contract["expected_wells"])
    seeds = int(contract["seeds_per_bank"])
    particles = int(contract["particles_per_seed"])
    selector_banks = wells
    learned_banks = wells
    total_banks = selector_banks + learned_banks
    total_seed_well_runs = total_banks * seeds
    result = {
        "expected_wells": wells,
        "selector_likelihood_pf_seed_banks": selector_banks,
        "learned_likelihood_pf_seed_banks": learned_banks,
        "likelihood_pf_total_seed_banks": total_banks,
        "seeds_per_bank": seeds,
        "particles_per_seed": particles,
        "likelihood_pf_total_seed_well_runs": total_seed_well_runs,
        "likelihood_pf_total_particle_starts": total_seed_well_runs * particles,
        "pf_ancc_well_runs": wells,
        "pf_ancc_particles_per_well": int(contract["pf_ancc_particles_per_well"]),
        "pf_z_well_runs": wells,
        "pf_z_particles_per_well": int(contract["pf_z_particles_per_well"]),
        "selector_beam_configs": int(contract["selector_beam_configs"]),
        "selector_beam_well_runs": wells * int(contract["selector_beam_configs"]),
        "learned_beam_configs": int(contract["learned_beam_configs"]),
        "learned_beam_well_runs": wells * int(contract["learned_beam_configs"]),
        "total_beam_well_runs": wells
        * (int(contract["selector_beam_configs"]) + int(contract["learned_beam_configs"])),
        "ncc_windows": int(contract["ncc_windows"]),
        "ncc_well_window_runs": wells * int(contract["ncc_windows"]),
        "formation_plane_pool_fits": int(contract["formation_plane_pool_fits"]),
        "dense_ancc_pool_fits": int(contract["dense_ancc_pool_fits"]),
        "fold_surface_well_queries_per_pool_family": wells
        * int(config["validation"]["outer_folds"]),
    }
    frozen = {key: int(value) for key, value in contract.items()}
    if result != frozen:
        raise ValueError(f"Physical execution inventory drift: {result} != {frozen}")
    return result


def feature_schema_plan(
    config: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    rows = int(config["validation"]["expected_rows"])
    source_counts = config["training_contract"]["source_feature_counts"]
    sp45 = int(source_counts["sp45_residual_base"])
    learned_base = int(source_counts["learned_base_before_likelihood_pf"])
    learned_added = int(source_counts["learned_likelihood_pf_added"])
    learned_total = int(source_counts["learned_total"])
    if learned_base + learned_added != learned_total:
        raise ValueError("Learned feature count arithmetic is inconsistent")
    source_inventory = inventory["feature_inventory"]
    if sp45 != int(source_inventory["sp45_residual"]["model_feature_columns"]):
        raise ValueError("SP45 source feature count drift")
    if learned_total != int(source_inventory["learned_trajectory"]["model_feature_columns"]):
        raise ValueError("Learned source feature count drift")
    bytes_per_float32 = 4
    return {
        "rows": rows,
        "sp45_feature_count": sp45,
        "learned_base_feature_count": learned_base,
        "learned_likelihood_feature_count": learned_added,
        "learned_total_feature_count": learned_total,
        "sp45_float32_bytes": rows * sp45 * bytes_per_float32,
        "learned_float32_bytes": rows * learned_total * bytes_per_float32,
        "both_surfaces_float32_bytes": rows * (sp45 + learned_total) * bytes_per_float32,
        "both_surfaces_kept_in_ram": False,
        "partition_policy": "outer_fold_then_inner_partition",
    }


def validate_static_contract(
    config: dict[str, Any],
    public_contract: dict[str, Any],
    ensemble_contract: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("Experiment name drift")
    if config["experiment"]["route"] != "ensemble":
        raise ValueError("Route must remain ensemble")
    if config["public_core"]["active_variants"] != ["strict_public_core"]:
        raise ValueError("Exactly one scientific variant is allowed")
    training = config["training_contract"]
    expected_cost = {
        "scientific_variants": 1,
        "ml_branches": 2,
        "configs_per_branch": 5,
        "lightgbm_configs_per_branch": 3,
        "catboost_configs_per_branch": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "planned_lightgbm_boosters": 120,
        "planned_catboost_boosters": 80,
        "planned_total_boosters": 200,
        "planned_ridge_models": 10,
        "exp413_parent_retraining": 0,
        "exp413_selector_retraining": 0,
        "exp413_signed_selector_retraining": 0,
        "exp413_tvt_model_retraining": 0,
    }
    observed_cost = {key: int(training[key]) for key in expected_cost}
    if observed_cost != expected_cost:
        raise ValueError(f"Model cost contract drift: {observed_cost}")
    if config["implementation"]["approval_scope"] not in {
        "stage0_compact_preflight_and_contract_tests_only",
        "stage0_plus_stage_p_m1_m2_e_kaggle_package_and_run",
        "stage0_plus_stage_p_m1_m2_e_and_stage_i_override_kaggle_run",
    }:
        raise ValueError("Implementation authorization scope drift")
    if config["stage0"]["stop_after_preflight"] is not True:
        raise ValueError("Stage 0 must stop before route execution")
    if config["data"]["parent_exp413"]["control_retraining_allowed"] is not False:
        raise ValueError("Parent retraining must remain disabled")
    if public_contract["contract"] != "strict_public_core_v1":
        raise ValueError("Public-core contract version drift")
    if ensemble_contract["contract"] != "exp413_strict_public_core_meta5_v1":
        raise ValueError("Ensemble contract version drift")
    if inventory["inventory"] != "strict_public_core_stage0_source_inventory_v1":
        raise ValueError("Source inventory version drift")
    post_stage_approval_required = inventory["implementation_boundary"][
        "stage_p_m1_m2_e_require_separate_approval"
    ]
    if post_stage_approval_required is not True:
        raise ValueError("Post-preflight approval boundary drift")
    execution = physical_execution_inventory(config)
    features = feature_schema_plan(config, inventory)
    return {
        "status": "pass",
        "model_cost": observed_cost,
        "physical_execution": execution,
        "feature_schema": features,
    }


def stable_seed(
    *,
    experiment: str,
    stage: str,
    split: str,
    outer_fold: int,
    inner_fold: int,
    family: str,
    well_id: str,
    seed_index: int,
    base_seed: int = 42,
) -> int:
    parts = (
        experiment,
        stage,
        split,
        str(int(outer_fold)),
        str(int(inner_fold)),
        family,
        str(well_id),
        str(int(seed_index)),
        str(int(base_seed)),
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


# %% [markdown]
# ## 4. Parent OOF and nested-fold helpers


# %%
def resolve_parent_root(
    config: dict[str, Any],
    experiment_dir: Path = EXPERIMENT_DIR,
) -> Path:
    parent_cfg = config["data"]["parent_exp413"]
    patterns: list[str] = []
    explicit = os.environ.get("EXP497_PARENT_STAGE_D_ROOT")
    if explicit:
        patterns.append(explicit)
    patterns.extend(str(value) for value in parent_cfg["root_patterns"])
    required = [str(value) for value in parent_cfg["files"].values()]
    for pattern in patterns:
        path = Path(pattern)
        if not path.is_absolute():
            path = experiment_dir.parents[1] / path
        for candidate_text in glob.glob(str(path), recursive=True):
            candidate = Path(candidate_text)
            if candidate.is_dir() and all((candidate / name).is_file() for name in required):
                return candidate
    raise FileNotFoundError("No complete exp413 Stage D evidence root was found")


def verify_parent_artifacts(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    parent_cfg = config["data"]["parent_exp413"]
    expected_by_label = {
        "oof": str(parent_cfg["expected_final_oof_sha256"]),
        "fold_metrics": str(parent_cfg["expected_fold_metrics_sha256"]),
        "scope_metrics": str(parent_cfg["expected_scope_metrics_sha256"]),
        "hidden_like_metrics": str(parent_cfg["expected_hidden_like_metrics_sha256"]),
        "by_well": str(parent_cfg["expected_by_well_sha256"]),
    }
    observed: dict[str, Any] = {}
    for label, filename in parent_cfg["files"].items():
        path = root / str(filename)
        digest = sha256_file(path)
        if digest != expected_by_label[label]:
            raise ValueError(
                f"Parent {label} SHA mismatch: observed={digest} "
                f"expected={expected_by_label[label]}"
            )
        observed[label] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    return observed


def canonicalize_parent_oof(
    frame: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    required = {
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        prediction_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Parent OOF is missing columns: {missing}")
    ids = frame["id"].astype(str)
    wells = frame["well"].astype(str)
    split = ids.str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("Parent id cannot be split into well and row index")
    if not split[0].eq(wells).all():
        raise ValueError("Parent id/well prefix mismatch")
    row_idx = pd.to_numeric(split[1], errors="raise").astype(np.int64)
    result = pd.DataFrame(
        {
            "well_id": wells,
            "row_idx": row_idx,
            "fold": pd.to_numeric(frame["outer_fold"], errors="raise").astype(np.int8),
            "actual_tvt": pd.to_numeric(frame["actual_tvt"], errors="raise").astype(np.float64),
            "exp413_pred_tvt": pd.to_numeric(frame[prediction_column], errors="raise").astype(
                np.float64
            ),
        }
    )
    if "md_since" in frame:
        result["md_since"] = pd.to_numeric(frame["md_since"], errors="raise").astype(np.float64)
    return result


def validate_parent_fold_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_wells: int | None = None,
    expected_folds: int = 5,
) -> dict[str, Any]:
    required = {"well_id", "row_idx", "fold", "actual_tvt", "exp413_pred_tvt"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical parent frame missing columns: {missing}")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("Canonical parent keys are duplicated")
    if not np.isfinite(frame[["actual_tvt", "exp413_pred_tvt"]].to_numpy()).all():
        raise ValueError("Parent truth or prediction contains non-finite values")
    per_well_folds = frame.groupby("well_id", sort=False)["fold"].nunique()
    if not per_well_folds.eq(1).all():
        raise ValueError("A well appears in more than one outer fold")
    folds = sorted(int(value) for value in frame["fold"].unique())
    if folds != list(range(expected_folds)):
        raise ValueError(f"Outer folds are not 0..{expected_folds - 1}: {folds}")
    rows = len(frame)
    wells = int(frame["well_id"].nunique())
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Parent row count mismatch: {rows} != {expected_rows}")
    if expected_wells is not None and wells != expected_wells:
        raise ValueError(f"Parent well count mismatch: {wells} != {expected_wells}")
    fold_rows = {
        str(int(key)): int(value) for key, value in frame.groupby("fold", sort=True).size().items()
    }
    fold_wells = {
        str(int(key)): int(value)
        for key, value in frame.groupby("fold", sort=True)["well_id"].nunique().items()
    }
    return {
        "rows": rows,
        "wells": wells,
        "folds": folds,
        "fold_rows": fold_rows,
        "fold_wells": fold_wells,
    }


def _groupkfold_assignments(
    well_counts: pd.Series,
    n_splits: int,
) -> dict[str, int]:
    if len(well_counts) < n_splits:
        raise ValueError("Not enough wells for inner GroupKFold")
    wells = well_counts.index.astype(str).to_numpy()
    counts = well_counts.to_numpy(np.int64)
    order = np.argsort(counts)[::-1]
    fold_load = np.zeros(n_splits, dtype=np.int64)
    assignment = np.full(len(wells), -1, dtype=np.int8)
    for original_index in order:
        inner_fold = int(np.argmin(fold_load))
        assignment[original_index] = inner_fold
        fold_load[inner_fold] += counts[original_index]
    return {str(well): int(inner_fold) for well, inner_fold in zip(wells, assignment, strict=True)}


def build_inner_fold_manifest(
    parent: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
) -> pd.DataFrame:
    per_well = (
        parent.groupby("well_id", sort=True)
        .agg(outer_fold=("fold", "first"), row_count=("row_idx", "size"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for outer_holdout in range(outer_folds):
        train = per_well.loc[per_well["outer_fold"].ne(outer_holdout)].copy()
        counts = train.set_index("well_id")["row_count"]
        assignment = _groupkfold_assignments(counts, inner_folds)
        for row in train.itertuples(index=False):
            rows.append(
                {
                    "outer_holdout_fold": outer_holdout,
                    "well_id": str(row.well_id),
                    "source_outer_fold": int(row.outer_fold),
                    "inner_fold": assignment[str(row.well_id)],
                    "row_count": int(row.row_count),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(["outer_holdout_fold", "inner_fold", "well_id"])
    manifest.reset_index(drop=True, inplace=True)
    return manifest


def validate_inner_fold_manifest(
    parent: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
) -> dict[str, Any]:
    well_outer = parent.groupby("well_id", sort=True)["fold"].first().astype(int)
    rows: list[dict[str, Any]] = []
    for outer_holdout in range(outer_folds):
        part = manifest.loc[manifest["outer_holdout_fold"].eq(outer_holdout)]
        expected_train = set(well_outer.index[well_outer.ne(outer_holdout)].astype(str))
        outer_valid = set(well_outer.index[well_outer.eq(outer_holdout)].astype(str))
        observed = set(part["well_id"].astype(str))
        if observed != expected_train:
            raise ValueError(f"Inner manifest coverage mismatch for outer {outer_holdout}")
        if observed & outer_valid:
            raise ValueError(f"Outer-valid well leaked into outer {outer_holdout} fit")
        if part["well_id"].duplicated().any():
            raise ValueError(f"Inner manifest duplicates wells for outer {outer_holdout}")
        observed_inner = sorted(int(value) for value in part["inner_fold"].unique())
        if observed_inner != list(range(inner_folds)):
            raise ValueError(f"Inner fold coverage mismatch for outer {outer_holdout}")
        rows.append(
            {
                "outer_holdout_fold": outer_holdout,
                "outer_valid_wells": len(outer_valid),
                "outer_train_wells": len(observed),
                "inner_folds": observed_inner,
                "outer_valid_overlap": 0,
            }
        )
    return {"status": "pass", "outer_folds": rows}


def build_spatial_pool_ledger(
    parent: pd.DataFrame,
    *,
    outer_folds: int = 5,
) -> pd.DataFrame:
    well_outer = parent.groupby("well_id", sort=True)["fold"].first().astype(int)
    rows: list[dict[str, Any]] = []
    for outer_holdout in range(outer_folds):
        pool = set(well_outer.index[well_outer.ne(outer_holdout)].astype(str))
        valid = set(well_outer.index[well_outer.eq(outer_holdout)].astype(str))
        overlap = pool & valid
        if overlap:
            raise ValueError(f"Spatial pool leakage in outer fold {outer_holdout}")
        rows.append(
            {
                "outer_holdout_fold": outer_holdout,
                "pool_wells": len(pool),
                "outer_valid_wells": len(valid),
                "pool_outer_valid_overlap": len(overlap),
                "query_wells": len(pool | valid),
                "self_well_exclusion_required": True,
                "formation_plane_pool_sha_input": sha256_json(sorted(pool)),
                "dense_ancc_pool_sha_input": sha256_json(sorted(pool)),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 5. Target-free selector and prediction-freeze helpers


# %%
def assign_well_shape_bins(
    metadata: pd.DataFrame,
    *,
    n_eval_threshold: float,
    z_span_thresholds: Sequence[float],
) -> np.ndarray:
    if len(z_span_thresholds) != 2:
        raise ValueError("Exactly two z-span thresholds are required")
    n_eval = pd.to_numeric(metadata["n_eval"], errors="raise").to_numpy(float)
    z_span = pd.to_numeric(metadata["z_span"], errors="raise").to_numpy(float)
    n_bin = (n_eval > float(n_eval_threshold)).astype(np.int8)
    z_bin = np.searchsorted(
        np.asarray(z_span_thresholds, dtype=float), z_span, side="right"
    ).astype(np.int8)
    return n_bin + 2 * z_bin


def _variant_winner(
    losses: pd.DataFrame,
    variant_order: Sequence[str],
) -> str:
    order = {str(name): index for index, name in enumerate(variant_order)}
    pooled = losses.groupby("variant", sort=False)["sse"].sum().reset_index()
    pooled["order"] = pooled["variant"].astype(str).map(order)
    if pooled["order"].isna().any():
        raise ValueError("Loss table contains a variant outside the frozen order")
    pooled.sort_values(["sse", "order"], inplace=True)
    return str(pooled.iloc[0]["variant"])


def fit_well_shape_selector(
    outer_train_metadata: pd.DataFrame,
    outer_train_losses: pd.DataFrame,
    *,
    variant_order: Sequence[str],
    minimum_bin_wells: int,
) -> dict[str, Any]:
    metadata = outer_train_metadata[["well_id", "n_eval", "z_span"]].copy()
    if metadata["well_id"].duplicated().any():
        raise ValueError("Well-shape metadata must have one row per outer-train well")
    if set(outer_train_losses["well_id"].astype(str)) != set(metadata["well_id"].astype(str)):
        raise ValueError("Selector loss and metadata well coverage differ")
    if set(outer_train_losses["variant"].astype(str)) != set(str(value) for value in variant_order):
        raise ValueError("Selector variant coverage differs from the frozen order")
    if outer_train_losses.duplicated(["well_id", "variant"]).any():
        raise ValueError("Selector loss contains duplicate well/variant rows")
    variants_per_well = outer_train_losses.groupby("well_id", sort=False)["variant"].nunique()
    if not variants_per_well.eq(len(variant_order)).all():
        raise ValueError("Selector loss does not contain every variant for every well")
    sse = pd.to_numeric(outer_train_losses["sse"], errors="raise").to_numpy(float)
    if not np.isfinite(sse).all() or np.any(sse < 0):
        raise ValueError("Selector SSE must be finite and non-negative")
    n_eval_threshold = float(metadata["n_eval"].median())
    z_span_thresholds = [float(value) for value in metadata["z_span"].quantile([1 / 3, 2 / 3])]
    metadata["shape_bin"] = assign_well_shape_bins(
        metadata,
        n_eval_threshold=n_eval_threshold,
        z_span_thresholds=z_span_thresholds,
    )
    joined = outer_train_losses.merge(
        metadata[["well_id", "shape_bin"]],
        on="well_id",
        how="inner",
        validate="many_to_one",
    )
    global_winner = _variant_winner(joined, variant_order)
    mapping: dict[str, str] = {}
    support: dict[str, int] = {}
    for shape_bin in range(6):
        part = joined.loc[joined["shape_bin"].eq(shape_bin)]
        wells = int(part["well_id"].nunique())
        support[str(shape_bin)] = wells
        mapping[str(shape_bin)] = (
            _variant_winner(part, variant_order)
            if wells >= int(minimum_bin_wells)
            else global_winner
        )
    return {
        "n_eval_threshold": n_eval_threshold,
        "z_span_thresholds": z_span_thresholds,
        "minimum_bin_wells": int(minimum_bin_wells),
        "global_winner": global_winner,
        "bin_support_wells": support,
        "bin_to_variant": mapping,
        "variant_order": [str(value) for value in variant_order],
        "outer_train_wells": int(metadata["well_id"].nunique()),
    }


def apply_well_shape_selector(
    metadata: pd.DataFrame,
    selector: dict[str, Any],
) -> pd.DataFrame:
    required = {"well_id", "n_eval", "z_span"}
    if not required.issubset(metadata.columns):
        raise ValueError("Target-free well-shape metadata is incomplete")
    result = metadata[["well_id", "n_eval", "z_span"]].copy()
    result["shape_bin"] = assign_well_shape_bins(
        result,
        n_eval_threshold=float(selector["n_eval_threshold"]),
        z_span_thresholds=selector["z_span_thresholds"],
    )
    result["selected_variant"] = result["shape_bin"].astype(str).map(selector["bin_to_variant"])
    if result["selected_variant"].isna().any():
        raise ValueError("Selector map did not cover every target-free shape bin")
    return result


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    row_chunk: int = 250_000,
) -> str:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Hash columns missing: {missing}")
    digest = hashlib.sha256()
    digest.update(sha256_json([str(value) for value in columns]).encode("ascii"))
    for start in range(0, len(frame), row_chunk):
        part = frame.iloc[start : start + row_chunk][list(columns)]
        hashed = pd.util.hash_pandas_object(part, index=False).to_numpy(np.uint64)
        digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def freeze_target_free_prediction(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    prediction_columns: Sequence[str],
    forbidden_truth_columns: Iterable[str] = (
        "actual_tvt",
        "target",
        "truth",
        "error",
        "squared_error",
    ),
) -> dict[str, Any]:
    forbidden = sorted(set(frame.columns) & set(forbidden_truth_columns))
    if forbidden:
        raise ValueError(f"Prediction freeze occurred after truth attach: {forbidden}")
    columns = [*keys, *prediction_columns]
    if frame.duplicated(list(keys)).any():
        raise ValueError("Prediction freeze keys are duplicated")
    if not np.isfinite(frame[list(prediction_columns)].to_numpy(float)).all():
        raise ValueError("Prediction freeze contains non-finite values")
    return {
        "status": "frozen_before_truth_attach",
        "rows": len(frame),
        "keys": list(keys),
        "prediction_columns": list(prediction_columns),
        "content_sha256": dataframe_content_sha256(frame, columns),
    }


def attach_truth_after_freeze(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    freeze_manifest: dict[str, Any],
    *,
    truth_columns: Sequence[str],
) -> pd.DataFrame:
    keys = [str(value) for value in freeze_manifest["keys"]]
    prediction_columns = [str(value) for value in freeze_manifest["prediction_columns"]]
    observed = dataframe_content_sha256(prediction, [*keys, *prediction_columns])
    if observed != str(freeze_manifest["content_sha256"]):
        raise ValueError("Frozen prediction changed before truth attach")
    merged = prediction.merge(
        truth[[*keys, *truth_columns]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Truth attach did not cover every frozen prediction row")
    merged.drop(columns="_merge", inplace=True)
    return merged


# %% [markdown]
# ## 6. Constant convex meta-blend helpers


# %%
def fit_constant_convex_weight(
    truth: Sequence[float],
    base: Sequence[float],
    auxiliary: Sequence[float],
    *,
    lower: float,
    upper: float,
) -> float:
    y = np.asarray(truth, dtype=np.float64)
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(auxiliary, dtype=np.float64)
    if not (len(y) == len(a) == len(b)) or len(y) == 0:
        raise ValueError("Constant weight inputs must be non-empty and aligned")
    if not np.isfinite(np.column_stack([y, a, b])).all():
        raise ValueError("Constant weight inputs contain non-finite values")
    delta = b - a
    denominator = float(np.dot(delta, delta))
    unconstrained = 0.0 if denominator <= 0 else float(np.dot(delta, y - a) / denominator)
    return float(np.clip(unconstrained, float(lower), float(upper)))


def crossfit_constant_blend(
    frame: pd.DataFrame,
    *,
    truth_column: str,
    base_column: str,
    auxiliary_column: str,
    fold_column: str,
    folds: int,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    observed_folds = sorted(int(value) for value in frame[fold_column].unique())
    if observed_folds != list(range(folds)):
        raise ValueError("Meta-fold coverage differs from the frozen contract")
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    weights: list[float] = []
    weight_rows: list[dict[str, Any]] = []
    for held_fold in range(folds):
        train_mask = frame[fold_column].ne(held_fold).to_numpy()
        valid_mask = ~train_mask
        weight = fit_constant_convex_weight(
            frame.loc[train_mask, truth_column],
            frame.loc[train_mask, base_column],
            frame.loc[train_mask, auxiliary_column],
            lower=lower,
            upper=upper,
        )
        base = frame.loc[valid_mask, base_column].to_numpy(np.float64)
        auxiliary = frame.loc[valid_mask, auxiliary_column].to_numpy(np.float64)
        prediction[valid_mask] = (1.0 - weight) * base + weight * auxiliary
        weights.append(weight)
        weight_rows.append(
            {
                "held_meta_fold": held_fold,
                "fit_rows": int(train_mask.sum()),
                "apply_rows": int(valid_mask.sum()),
                "public_core_weight": weight,
                "exp413_weight": 1.0 - weight,
            }
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Cross-fit prediction coverage is incomplete")
    deployment_weight = float(np.clip(np.median(weights), lower, upper))
    return {
        "crossfit_prediction": prediction,
        "meta_fold_weights": np.asarray(weights, dtype=np.float64),
        "weight_rows": weight_rows,
        "deployment_weight": deployment_weight,
        "full_oof_refit": False,
    }


# %% [markdown]
# ## 7. Setup and authorization guard

# %%
CONFIG, PUBLIC_CONTRACT, ENSEMBLE_CONTRACT, SOURCE_INVENTORY = load_contract_bundle()
STATIC_CONTRACT = validate_static_contract(
    CONFIG,
    PUBLIC_CONTRACT,
    ENSEMBLE_CONTRACT,
    SOURCE_INVENTORY,
)


def require_stage0_run_authorization(config: dict[str, Any]) -> None:
    stage0 = config["stage0"]
    if stage0["kaggle_run_approved"] is not True:
        raise RuntimeError(
            "The exp497 Stage 0 Kaggle run is not authorized. The current approval "
            "covers the compact implementation and contract tests only."
        )
    if stage0.get("approved_scope") != STAGE0_APPROVED_SCOPE:
        raise RuntimeError("Stage 0 approved scope does not match the frozen preflight")


print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "status": CONFIG["experiment"]["status"],
            "implementation_scope": CONFIG["implementation"]["approval_scope"],
            "stage0_run_approved": CONFIG["stage0"]["kaggle_run_approved"],
            "model_cost": STATIC_CONTRACT["model_cost"],
            "physical_execution": STATIC_CONTRACT["physical_execution"],
            "feature_schema": STATIC_CONTRACT["feature_schema"],
        },
        indent=2,
        sort_keys=True,
    )
)


# %% [markdown]
# ## 8. Stage 0 orchestration and generated evidence
#
# The run gate is checked before source discovery or parent input reads. Once a
# separate run approval is recorded, this section performs only the zero-model
# preflight and writes small manifests. It stops before every route execution.


# %%
def run_stage0(
    config: dict[str, Any] = CONFIG,
    public_contract: dict[str, Any] = PUBLIC_CONTRACT,
    ensemble_contract: dict[str, Any] = ENSEMBLE_CONTRACT,
    inventory: dict[str, Any] = SOURCE_INVENTORY,
    experiment_dir: Path = EXPERIMENT_DIR,
) -> dict[str, Any]:
    require_stage0_run_authorization(config)
    static = validate_static_contract(
        config,
        public_contract,
        ensemble_contract,
        inventory,
    )
    source_path = resolve_reference_source(config, experiment_dir)
    source_audit = scan_reference_source(source_path, inventory)

    parent_root = resolve_parent_root(config, experiment_dir)
    parent_artifacts = verify_parent_artifacts(parent_root, config)
    parent_cfg = config["data"]["parent_exp413"]
    parent_raw = pd.read_parquet(parent_root / str(parent_cfg["files"]["oof"]))
    parent = canonicalize_parent_oof(
        parent_raw,
        str(parent_cfg["prediction_column"]),
    )
    parent_manifest = validate_parent_fold_frame(
        parent,
        expected_rows=int(parent_cfg["expected_rows"]),
        expected_wells=int(parent_cfg["expected_wells"]),
        expected_folds=int(parent_cfg["expected_outer_folds"]),
    )
    inner_manifest = build_inner_fold_manifest(
        parent,
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
    )
    inner_audit = validate_inner_fold_manifest(
        parent,
        inner_manifest,
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
    )
    spatial_ledger = build_spatial_pool_ledger(
        parent,
        outer_folds=int(config["validation"]["outer_folds"]),
    )

    output_dir = Path("/kaggle/working") / "artifacts"
    if not Path("/kaggle/working").exists():
        output_dir = experiment_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "status": "pass",
        "parent_root": str(parent_root),
        "parent_artifacts": parent_artifacts,
        "parent_oof": parent_manifest,
        "parent_prediction_column": str(parent_cfg["prediction_column"]),
        "inner_fold_audit": inner_audit,
        "truth_attach_policy": "freeze_target_free_prediction_before_truth_attach",
    }
    execution_inventory = static["physical_execution"] | static["model_cost"]
    feature_plan = static["feature_schema"]
    preflight = {
        "status": "stage0_complete_stop_before_route_execution",
        "source_audit_sha256": sha256_json(source_audit),
        "input_manifest_sha256": sha256_json(input_manifest),
        "inner_fold_manifest_sha256": dataframe_content_sha256(
            inner_manifest,
            list(inner_manifest.columns),
        ),
        "spatial_pool_ledger_sha256": dataframe_content_sha256(
            spatial_ledger,
            list(spatial_ledger.columns),
        ),
        "execution_inventory_sha256": sha256_json(execution_inventory),
        "feature_schema_plan_sha256": sha256_json(feature_plan),
        "route_execution_started": False,
        "models_fitted": 0,
        "physical_paths_generated": 0,
        "parent_retraining": 0,
    }
    write_json(output_dir / "stage_0_source_audit.json", source_audit)
    write_json(output_dir / "stage_0_input_fold_row_manifest.json", input_manifest)
    inner_manifest.to_csv(output_dir / "stage_0_inner_fold_manifest.csv", index=False)
    spatial_ledger.to_csv(output_dir / "stage_0_spatial_pool_ledger.csv", index=False)
    write_json(output_dir / "stage_0_execution_inventory.json", execution_inventory)
    write_json(output_dir / "stage_0_feature_schema_plan.json", feature_plan)
    write_json(output_dir / "stage_0_preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    return preflight


if EXECUTE_NOTEBOOK:
    run_stage0()
